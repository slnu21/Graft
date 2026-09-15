"""v0.6 ``label-yolo-roi``(KNOWN-ISSUES #8 #4) — Qt 없이: 이웃 YOLO 라벨 찾기(``find_yolo_label``·``find_names_file``) ·
초안 파싱(``parse_yolo_draft``) · ``LabelSession.load_yolo_draft/apply_draft``(박스 추정·폴리곤 채움·되돌리기·mask_origin) ·
ROI PNG 저장(``save_roi_png``·``roi_png_path`` → ``roi_from_mask`` 가 읽는 형식)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anograft.core.roi import roi_from_mask
from anograft.gui.label.session import (
    LabelError,
    LabelSession,
    find_names_file,
    find_yolo_label,
    parse_yolo_draft,
    roi_png_path,
)
from anograft.io import imgio
from tests.fixtures import blob_image, fake_yolo_dataset

# --- 이웃 라벨 찾기 -------------------------------------------------------------


def test_find_yolo_label_prefers_images_to_labels_mirror(tmp_path: Path) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    assert find_yolo_label(d["images"] / "d0.png") == d["labels"] / "d0.txt"
    assert (
        find_yolo_label(d["images"] / "sub" / "d2.png") == d["labels"] / "sub" / "d2.txt"
    )  # 하위 폴더 유지
    assert find_yolo_label(d["images"] / "n1.png") is None  # 라벨 파일 없음
    assert (
        find_yolo_label(d["images"] / "n0.png") == d["labels"] / "n0.txt"
    )  # 빈 파일도 "있음"(항목 0)
    # 같은 폴더 <stem>.txt · <folder>/labels/<stem>.txt
    flat = tmp_path / "flat"
    imgio.write_image(flat / "x.png", blob_image(64))
    (flat / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert find_yolo_label(flat / "x.png") == flat / "x.txt"
    (flat / "x.txt").unlink()
    (flat / "labels").mkdir()
    (flat / "labels" / "x.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert find_yolo_label(flat / "x.png") == flat / "labels" / "x.txt"


def test_find_names_file_walks_up(tmp_path: Path) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    assert find_names_file(d["images"] / "sub" / "d2.png") == d["names"]
    assert find_names_file(tmp_path / "nowhere" / "x.png") is None


# --- 초안 파싱 -------------------------------------------------------------------


def test_parse_yolo_draft_boxes_polygons_names_and_warnings(tmp_path: Path) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    draft = parse_yolo_draft(d["labels"] / "d1.txt", (96, 96), d["names"])
    kinds = [(it.class_id, it.kind) for it in draft.items]
    assert kinds == [(0, "box"), (1, "box"), (1, "polygon")]
    assert (
        draft.class_ids == [0, 1]
        and draft.class_name(0) == "spot"
        and draft.class_name(7) == "class 7"
    )
    assert draft.items[0].box is not None and all(v >= 0 for v in draft.items[0].box)
    assert (
        "박스 2" in draft.summary()
        and "폴리곤 1" in draft.summary()
        and "spot 1" in draft.summary()
    )
    assert draft.names_path == d["names"] and not draft.warnings
    # 깨진 줄 · 이미지 밖 박스 · names 실패는 경고로
    bad = tmp_path / "bad.txt"
    bad.write_text("x y z\n0 1.6 1.6 0.1 0.1\n0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    draft = parse_yolo_draft(bad, (64, 64), tmp_path / "missing.yaml")
    assert len(draft.items) == 1 and len(draft.warnings) == 3
    assert draft.names is None and draft.names_path is None


# --- LabelSession 초안 적용 ----------------------------------------------------------


def _open(d: dict, stem: str) -> LabelSession:
    s = LabelSession()
    s.load_image(d["images"] / f"{stem}.png")
    return s


def test_load_and_apply_draft_fills_mask_per_class(tmp_path: Path) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    s = _open(d, "d1")
    assert s.draft is None
    draft = s.load_yolo_draft()
    assert (
        draft is not None
        and draft.label_path == d["labels"] / "d1.txt"
        and draft.names == ["spot", "crack"]
    )
    assert not s.mask.any() and not s.dirty  # 파싱만
    used = s.apply_draft(0, "rect")
    assert used == ["rect"] and s.mask.any() and s.dirty and s.can_undo
    x, y, w, h = draft.items[0].box
    assert s.mask[y : y + h, x : x + w].all() and s.mask.sum() // 255 == w * h  # rect = 박스 그대로
    assert s.mask_origin() == "yolo-box:rect"
    # 다른 클래스로 교체 — 박스 추정 + 폴리곤 채움, 이전 마스크는 되돌리기로
    used = s.apply_draft(1, "rect")
    assert (
        used == ["rect"] and s.mask_origin() == "yolo-box:rect"
    )  # 폴리곤이 섞여도 추정이 있으면 est
    assert s.mask[20:34, 20:34].any()  # 삼각형 폴리곤 자리
    assert not s.mask[y : y + h, x : x + w].all()
    s.undo()
    assert s.mask[y : y + h, x : x + w].all()
    # 손을 대면 manual:mixed
    s.redo()
    s.stroke([(5.0, 5.0)], 2)
    assert s.mask_origin() == "manual:mixed"


def test_apply_draft_polygon_only_and_errors(tmp_path: Path) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    s = _open(d, "d1")
    with pytest.raises(LabelError, match="초안이 없습니다"):
        s.apply_draft(0)
    lp = tmp_path / "poly.txt"
    lp.write_text("2 0.1 0.1 0.5 0.1 0.3 0.5\n", encoding="utf-8")
    draft = s.load_yolo_draft(lp, find_names=False)
    assert draft is not None and draft.names is None and draft.class_name(2) == "class 2"
    assert s.apply_draft(2) == [] and s.mask_origin() == "yolo-polygon"
    with pytest.raises(LabelError, match="항목이 없습니다"):
        s.apply_draft(0)
    with pytest.raises(LabelError, match="알 수 없는"):
        s.apply_draft(2, "magic")
    # 라벨 없는 이미지 → None, 새 이미지를 열면 초안 초기화
    n = _open(d, "n1")
    assert n.load_yolo_draft() is None and n.draft is None
    s.load_image(d["images"] / "n1.png")
    assert s.draft is None


def test_apply_draft_is_deterministic(tmp_path: Path) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    a, b = _open(d, "d0"), _open(d, "d0")
    a.load_yolo_draft()
    b.load_yolo_draft()
    a.apply_draft(0, "grabcut")
    b.apply_draft(0, "grabcut")
    assert np.array_equal(a.mask, b.mask)


# --- ROI 저장 -----------------------------------------------------------------------


def test_save_roi_png_full_size_and_readable_by_roi_from_mask(tmp_path: Path) -> None:
    img = tmp_path / "normals" / "n0.png"
    imgio.write_image(img, blob_image(80))
    s = LabelSession()
    s.load_image(img)
    with pytest.raises(LabelError, match="비어"):
        s.save_roi_png(tmp_path / "roi" / "n0.png")
    s.stroke([(10.0, 40.0), (70.0, 40.0)], 6)
    s.mask[20, 20] = 128  # 회색값이 섞여도 이진화해서 쓴다
    out = roi_png_path(tmp_path / "roi", img)
    assert out == tmp_path / "roi" / "n0.png"
    assert s.save_roi_png(out) == out and out.is_file() and not s.dirty
    m = imgio.read_mask(out)
    assert m.shape == (80, 80) and set(np.unique(m)) <= {0, 255} and m[20, 20] == 255
    roi = roi_from_mask(m, (80, 80))
    assert roi.shape == (80, 80) and roi.any()
    with pytest.raises(LabelError, match="PNG"):
        s.save_roi_png(tmp_path / "roi" / "n0.jpg")

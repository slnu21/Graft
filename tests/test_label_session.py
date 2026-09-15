"""``gui/label/session.py`` — Qt 없이 도는 라벨 편집 로직: 스트로크/지우개/폴리곤/자동 선택/팽창·침식 · 되돌리기 왕복 ·
통계 · 은행 저장(임포터와 같은 화폐 — 성분 분리·mask_origin·bank.yaml 병합·기존 은행에 이어 쓰기) · 오류(빈 마스크·빈 클래스)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anograft.bank import Bank
from anograft.gui.label.session import LabelError, LabelSession
from anograft.io import imgio
from tests.fixtures import blob_image, blob_mask, disk_image


def _session(size: int = 96) -> LabelSession:
    s = LabelSession()
    s.set_image(disk_image(size), path=Path("plate_01.png"))
    return s


def test_load_image_from_file_and_state(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    imgio.write_image(p, disk_image(64)[:, :, 0])  # 흑백 파일
    s = LabelSession()
    assert not s.loaded
    with pytest.raises(LabelError):
        s.stroke([(1, 1)], 2)
    s.load_image(p)
    assert s.loaded and s.gray and s.shape == (64, 64) and s.path == p
    assert s.mask is not None and not s.mask.any() and not s.dirty and not s.can_undo


def test_stroke_paints_round_line_and_eraser_removes() -> None:
    s = _session()
    s.push_undo()
    s.stroke([(20, 48), (70, 48)], 4)
    m = s.mask
    assert m[48, 45] == 255 and m[44, 45] == 255 and m[52, 45] == 255  # 두께 ≈ 2r
    assert m[40, 45] == 0 and m[48, 10] == 0
    assert m[48, 16] == 255 and m[48, 74] == 255  # 끝은 둥글게(반지름만큼 더)
    assert s.dirty and "brush" in s.tools_used
    s.stroke([(45, 48)], 6, erase=True)
    assert m[48, 45] == 0 and m[48, 30] == 255 and "eraser" in s.tools_used
    # 점 하나 = 원
    s.stroke([(80, 80)], 3)
    assert s.mask[80, 80] == 255 and s.mask[80, 83] == 255 and s.mask[80, 85] == 0


def test_fill_polygon_and_min_points() -> None:
    s = _session()
    with pytest.raises(LabelError):
        s.fill_polygon([(1, 1), (5, 5)])
    s.fill_polygon([(10, 10), (40, 10), (40, 40), (10, 40)])
    assert s.mask[25, 25] == 255 and s.mask[5, 5] == 0 and int((s.mask > 0).sum()) >= 30 * 30
    s.fill_polygon([(15, 15), (30, 15), (30, 30), (15, 30)], erase=True)
    assert s.mask[20, 20] == 0 and s.mask[35, 35] == 255
    assert "polygon" in s.tools_used


@pytest.mark.parametrize("method", ["grabcut", "otsu", "rect"])
def test_auto_select_adds_estimated_mask_and_reports_method(method: str) -> None:
    blobs = [(48, 48, 12)]
    s = LabelSession()
    s.set_image(blob_image(96, blobs), path=Path("blob.png"))
    truth = blob_mask(96, blobs) > 0
    used = s.auto_select((30, 30, 36, 36), method)
    assert used in ("grabcut", "otsu", "ellipse", "rect")
    got = s.mask > 0
    iou = (got & truth).sum() / float((got | truth).sum())
    assert iou > (0.3 if method == "rect" else 0.6), (method, used, iou)  # rect 는 박스 = 마스크
    assert "auto" in s.tools_used and used in s.auto_methods_used
    # 두 번째 박스는 더한다(합집합)
    before = int(got.sum())
    s.stroke([(5, 5)], 2)
    assert int((s.mask > 0).sum()) > before
    with pytest.raises(LabelError):
        s.auto_select((0, 0, 0, 5))
    with pytest.raises(LabelError):
        s.auto_select((10, 10, 5, 5), "nope")
    with pytest.raises(LabelError):
        s.auto_select((200, 200, 5, 5))


def test_auto_select_is_deterministic_for_same_box() -> None:
    blobs = [(48, 48, 12)]
    a, b = LabelSession(), LabelSession()
    for s in (a, b):
        s.set_image(blob_image(96, blobs), path=Path("blob.png"))
        s.auto_select((30, 30, 36, 36), "grabcut")
    assert np.array_equal(a.mask, b.mask)


def test_dilate_erode_and_clear() -> None:
    s = _session()
    s.stroke([(48, 48)], 5)
    n0 = int((s.mask > 0).sum())
    s.dilate(2)
    n1 = int((s.mask > 0).sum())
    s.erode(2)
    n2 = int((s.mask > 0).sum())
    assert n1 > n0 >= n2 and "morph" in s.tools_used
    s.dilate(0)
    assert int((s.mask > 0).sum()) == n2
    s.clear()
    assert not s.mask.any()


def test_undo_redo_roundtrip_and_depth() -> None:
    s = _session()
    s.push_undo()
    s.stroke([(10, 10)], 3)
    s.push_undo()
    s.stroke([(50, 50)], 3)
    assert s.can_undo and not s.can_redo
    assert s.undo() and s.mask[10, 10] == 255 and s.mask[50, 50] == 0
    assert s.undo() and not s.mask.any() and not s.can_undo
    assert not s.undo()
    assert s.redo() and s.mask[10, 10] == 255
    assert s.redo() and s.mask[50, 50] == 255 and not s.can_redo
    # 새 편집이 redo 를 지운다
    s.undo()
    s.push_undo()
    s.stroke([(70, 70)], 2)
    assert not s.can_redo
    for _ in range(60):
        s.push_undo()
    assert len(s._undo) == 40


def test_stats_values() -> None:
    s = _session(128)
    assert s.stats().area_px == 0 and s.stats().bbox is None
    s.stroke([(30, 64), (90, 64)], 3)  # 원판(200) 위 밝은? 아니 — 마스크만, 이미지는 원판
    st = s.stats(um_per_px=2.5)
    assert st.area_px > 0 and 0 < st.area_ratio < 0.1 and st.n_components == 1
    x, _y, w, h = st.bbox
    assert x <= 30 and x + w >= 90 and h <= 8
    assert 55 <= st.length_px <= 70 and st.length_um == pytest.approx(st.length_px * 2.5)
    assert st.contrast is not None and abs(st.contrast) < 5  # 원판 안 균일 → 대비 ≈ 0
    s.stroke([(10, 10)], 3)  # 배경(40) 위 조각 하나 더 — 성분 2, 대비 음수 쪽으로
    st2 = s.stats()
    assert st2.n_components == 2 and st2.contrast is not None


def test_mask_origin_reflects_tools() -> None:
    s = _session()
    assert s.mask_origin() == "manual"
    s.stroke([(5, 5)], 2)
    assert s.mask_origin() == "manual:brush"
    s.dilate(1)
    assert s.mask_origin() == "manual:brush"  # morph 는 세지 않는다
    s.fill_polygon([(20, 20), (30, 20), (30, 30)])
    assert s.mask_origin() == "manual:mixed"
    t = LabelSession()
    t.set_image(blob_image(96, [(48, 48, 12)]), path=Path("b.png"))
    used = t.auto_select((30, 30, 36, 36), "rect")
    assert t.mask_origin() == f"manual:auto:{used}"


def test_save_to_bank_splits_components_and_merges_classes(tmp_path: Path) -> None:
    s = _session(128)
    s.stroke([(30, 64), (60, 64)], 3)
    s.stroke([(90, 40), (100, 40)], 3)
    added, warns = s.save_to_bank(
        tmp_path / "bank", "scratch", tags=("manual", " gui "), um_per_px=3.0
    )
    assert [a.cls for a in added] == ["scratch", "scratch"] and warns == []
    assert sorted(a.source_id for a in added) == ["plate_01-0", "plate_01-1"]
    assert not s.dirty
    bank = Bank.load(tmp_path / "bank")
    assert bank.classes == ["scratch"] and len(bank) == 2
    src = bank.by_class("scratch")[0]
    assert (
        src.mask_origin == "manual:brush" and src.um_per_px == 3.0 and src.tags == ("manual", "gui")
    )
    assert src.origin == "plate_01.png"
    # 다른 클래스를 같은 은행에 이어 쓰기 — 클래스는 끝에 붙고 imports 이력이 쌓인다
    t = _session(128)
    t.fill_polygon([(40, 40), (70, 40), (70, 70)])
    added2, _ = t.save_to_bank(tmp_path / "bank", "dent")
    assert len(added2) == 1 and added2[0].source_id == "plate_01"
    bank = Bank.load(tmp_path / "bank")
    assert bank.classes == ["scratch", "dent"] and len(bank) == 3
    assert bank.imports[-1]["importer"] == "label" and bank.imports[-1]["class"] == "dent"
    assert bank.imports[-1]["mask_origin"] == "manual:polygon"
    # 같은 id 를 다시 저장하면 -dup 으로 (덮어쓰지 않음)
    added3, warns3 = t.save_to_bank(tmp_path / "bank", "dent")
    assert added3[0].source_id == "plate_01-dup1" and any("dup" in w for w in warns3)


def test_save_to_bank_keep_whole_and_min_area(tmp_path: Path) -> None:
    s = _session(128)
    s.stroke([(30, 64), (60, 64)], 3)
    s.stroke([(100, 100)], 1)  # 작은 조각(면적 < 16)
    added, _ = s.save_to_bank(tmp_path / "b1", "scratch")
    assert len(added) == 1  # 작은 성분은 버린다
    added, _ = s.save_to_bank(tmp_path / "b2", "scratch", keep_whole=True)
    assert len(added) == 1 and added[0].area_px == int((s.mask > 0).sum())


def test_save_to_bank_errors(tmp_path: Path) -> None:
    s = _session()
    with pytest.raises(LabelError, match="비어"):
        s.save_to_bank(tmp_path / "b", "scratch")
    s.stroke([(48, 48)], 4)
    with pytest.raises(LabelError, match="클래스"):
        s.save_to_bank(tmp_path / "b", "  ")
    with pytest.raises(LabelError, match="구분자"):
        s.save_to_bank(tmp_path / "b", "a/b")
    assert not (tmp_path / "b").exists()


def test_load_mask_png_requires_same_size(tmp_path: Path) -> None:
    s = _session(64)
    m = np.zeros((64, 64), dtype=np.uint8)
    m[10:20, 10:20] = 255
    imgio.write_image(tmp_path / "m.png", m)
    s.load_mask(tmp_path / "m.png")
    assert s.mask[15, 15] == 255 and s.can_undo and "png" in s.tools_used
    imgio.write_image(tmp_path / "bad.png", np.zeros((32, 32), dtype=np.uint8))
    with pytest.raises(LabelError, match="크기"):
        s.load_mask(tmp_path / "bad.png")


def test_auto_select_records_confidence_for_status(tmp_path: Path) -> None:
    """v0.7.x — 라벨 탭 자동 선택 뒤 상태줄에 보여 줄 타당성 점수."""
    s = LabelSession()
    s.set_image(blob_image(96, [(48, 48, 12)]), path=Path("b.png"))
    assert s.last_auto_confidence is None
    s.auto_select((30, 30, 36, 36), "grabcut")
    c = s.last_auto_confidence
    assert c is not None and c.score >= 0.9 and c.flags == ()
    s.auto_select((30, 30, 36, 36), "rect")  # 박스 그대로 → box-edge
    assert s.last_auto_confidence is not None and "box-edge" in s.last_auto_confidence.flags
    s.set_image(blob_image(64), path=Path("c.png"))
    assert s.last_auto_confidence is None

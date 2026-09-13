"""YOLO 임포터 (설계 §3.2·§11 test_yolo_import) — 박스·폴리곤·빈 라벨 구분, names 3형식, 하위 폴더 미러링, 클립 경고, list-normals."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from anograft.bank import Bank
from anograft.bank.importers import yolo as Y
from anograft.io import imgio
from tests.fixtures import fake_yolo_dataset


def test_parse_label_text_distinguishes_lines() -> None:
    text = "0 0.5 0.5 0.1 0.1\n\n1 0.1 0.1 0.2 0.1 0.2 0.2\n# comment\n2 0.5 0.5\n0 a b c d\n"
    parsed = Y.parse_label_text(text)
    kinds = [(lab.line_no, lab.class_id, lab.kind) for lab in parsed.labels]
    assert kinds == [(1, 0, "box"), (3, 1, "polygon")]
    assert len(parsed.warnings) == 2 and "5행" in parsed.warnings[0] and "6행" in parsed.warnings[1]
    assert Y.parse_label_text("").labels == [] and Y.parse_label_text("\n \n").labels == []


def test_denormalize_clip_and_polygon() -> None:
    box, clipped = Y.denormalize_box((0.5, 0.5, 0.2, 0.1), (100, 200))
    assert box == (80, 45, 40, 10) and not clipped
    box, clipped = Y.denormalize_box((0.0, 0.0, 0.2, 0.2), (100, 100))
    assert clipped and box == (0, 0, 10, 10)
    box, _ = Y.denormalize_box((1.5, 1.5, 0.1, 0.1), (100, 100))
    assert box is None
    m = Y.polygon_mask((0.1, 0.1, 0.5, 0.1, 0.3, 0.5), (100, 100))
    ref = np.zeros((100, 100), np.uint8)
    cv2.fillPoly(ref, [np.array([[10, 10], [50, 10], [30, 50]], np.int32)], 255)
    assert np.array_equal(m, ref)


def test_parse_names_three_ways(tmp_path: Path) -> None:
    y = tmp_path / "data.yaml"
    y.write_text("names:\n  0: a\n  1: b\n", encoding="utf-8")
    assert Y.parse_names(y) == ["a", "b"]
    y.write_text("names: [x, y, z]\n", encoding="utf-8")
    assert Y.parse_names(y) == ["x", "y", "z"]
    y.write_text("names:\n  0: a\n  2: c\n", encoding="utf-8")
    with pytest.raises(ValueError, match="연속"):
        Y.parse_names(y)
    y.write_text("nc: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="names"):
        Y.parse_names(y)
    c = tmp_path / "classes.txt"
    c.write_text("p\n\nq\n", encoding="utf-8")
    assert Y.parse_names(c) == ["p", "q"]
    assert Y.parse_names("a, b ,c") == ["a", "b", "c"]
    with pytest.raises(ValueError):
        Y.parse_names(" , ")


def test_find_pairs_mirrors_subfolders(tmp_path: Path) -> None:
    d = fake_yolo_dataset(tmp_path)
    pairs = Y.find_pairs(d["images"], d["labels"])
    rel = {img.relative_to(d["images"]).as_posix(): lab is not None for img, lab in pairs}
    assert rel == {
        "d0.png": True,
        "d1.png": True,
        "n0.png": True,
        "n1.png": False,
        "sub/d2.png": True,
    }
    rels = [img.relative_to(d["images"]).as_posix() for img, _ in pairs]
    assert rels == sorted(rels)  # 이름(상대경로) 정렬 — 재현성


def test_import_yolo_end_to_end(tmp_path: Path) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    normals = tmp_path / "lists" / "normals.txt"
    logs: list[str] = []
    res = Y.import_yolo(
        d["images"],
        d["labels"],
        d["names"],
        tmp_path / "bank",
        mask_from="otsu",
        list_normals=normals,
        log=logs.append,
    )
    assert res.classes == ["spot", "crack"] and res.n_images == 5
    assert sorted(p.name for p in res.normals) == ["n0.png", "n1.png"]
    assert res.stats.per_class() == {"spot": 2, "crack": 3}  # 박스 4 + 폴리곤 1
    assert set(res.mask_methods) <= {"otsu", "ellipse"} and sum(res.mask_methods.values()) == 4
    bank = Bank.load(tmp_path / "bank")
    assert bank.classes == ["spot", "crack"] and len(bank) == 5
    ids = sorted(s.id for s in bank.sources())
    assert ids == ["crack/d1-02", "crack/d1-03", "crack/d2-01", "spot/d0-01", "spot/d1-01"]
    poly = next(s for s in bank.sources() if s.id == "crack/d1-03")
    assert poly.mask_origin == "yolo-polygon"
    boxed = next(s for s in bank.sources() if s.id == "spot/d0-01")
    assert boxed.mask_origin.startswith("yolo-box:") and boxed.image.shape[:2] == boxed.mask.shape
    # 얼룩(r=8)을 찾았다: 마스크 면적이 원 면적 근처
    assert 0.6 * np.pi * 64 < np.count_nonzero(boxed.mask) < 1.5 * np.pi * 64
    # 크롭 = 마스크 bbox + margin(16) — 마스크가 크롭 경계에서 margin 이상 떨어져 있다
    ys, xs = np.where(boxed.mask > 0)
    assert ys.min() >= 16 and xs.min() >= 16
    # list-normals: 목록 파일 기준 상대경로, imgio.read_path_list로 되읽힌다
    got = imgio.read_path_list(normals)
    assert sorted(p.resolve() for p in got) == sorted(p.resolve() for p in d["normals"])
    assert bank.imports[0]["importer"] == "yolo" and bank.imports[0]["n_normals"] == 2


def test_import_records_box_and_origin(tmp_path: Path) -> None:
    import json

    d = fake_yolo_dataset(tmp_path / "ds")
    Y.import_yolo(d["images"], d["labels"], ["spot", "crack"], tmp_path / "bank", mask_from="rect")
    meta = json.loads((tmp_path / "bank" / "spot" / "d0-01.json").read_text(encoding="utf-8"))
    assert meta["mask_origin"] == "yolo-box:rect" and meta["origin"] == "d0.png"
    _cid, cx, cy, r = d["boxes"]["d0"][0]
    assert meta["box_in_origin"] == [cx - r - 1, cy - r - 1, 2 * r + 2, 2 * r + 2]
    sub = json.loads((tmp_path / "bank" / "crack" / "d2-01.json").read_text(encoding="utf-8"))
    assert sub["origin"] == "sub/d2.png"


def test_import_warns_on_bad_class_and_clipped_box(tmp_path: Path) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    (d["labels"] / "d0.txt").write_text("7 0.5 0.5 0.1 0.1\n0 0.02 0.5 0.1 0.1\n", encoding="utf-8")
    res = Y.import_yolo(d["images"], d["labels"], d["names"], tmp_path / "bank", mask_from="rect")
    assert any("범위 밖" in w for w in res.warnings) and any("잘라냄" in w for w in res.warnings)
    assert "spot/d0-02" in [f"{a.cls}/{a.source_id}" for a in res.stats.added]


def test_import_twice_accumulates_with_dup_suffix(tmp_path: Path) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    Y.import_yolo(d["images"], d["labels"], d["names"], tmp_path / "bank", mask_from="rect")
    res = Y.import_yolo(d["images"], d["labels"], d["names"], tmp_path / "bank", mask_from="rect")
    assert res.stats.duplicates == 5 and all(a.source_id.endswith("-dup1") for a in res.stats.added)
    bank = Bank.load(tmp_path / "bank")
    assert bank.classes == ["spot", "crack"] and len(bank) == 10  # 누적
    # names 순서가 다른 세트를 합치면: 이름 기준 병합(은행 순서 유지) + 순서 경고, id는 이름으로 다시 매핑
    res = Y.import_yolo(d["images"], d["labels"], "crack,spot", tmp_path / "bank", mask_from="rect")
    assert any("순서" in w for w in res.warnings)
    bank = Bank.load(tmp_path / "bank")
    assert bank.classes == ["spot", "crack"] and len(bank) == 15
    assert bank.counts() == {"spot": 2 + 2 + 3, "crack": 3 + 3 + 2}  # 0→crack, 1→spot 으로 뒤집힘


def test_import_rejects_bad_args(tmp_path: Path) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    with pytest.raises(ValueError, match="mask-from"):
        Y.import_yolo(d["images"], d["labels"], d["names"], tmp_path / "b", mask_from="nope")
    with pytest.raises(ValueError, match="margin"):
        Y.import_yolo(d["images"], d["labels"], d["names"], tmp_path / "b", margin=2)
    with pytest.raises(FileNotFoundError):
        Y.import_yolo(tmp_path / "missing", d["labels"], d["names"], tmp_path / "b")

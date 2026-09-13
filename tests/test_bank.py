"""은행 포맷 로드 (설계 §3.1·§11 test_bank) — 정렬·이진화·흑백 승격·클래스 순서·지문·fail-soft. 쓰기는 ``BankWriter``(§3.5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from anograft.bank import BANK_FILE, Bank
from anograft.bank.bank import BankError
from anograft.bank.importers import (
    MIN_MARGIN,
    BankWriter,
    ImportOptions,
    ImportRecord,
    check_margin,
)
from anograft.core.channels import promote_to_bgr
from anograft.io import imgio
from tests.fixtures import blob_image


def _record(
    stem: str, cls: str, blobs, *, gray: bool = False, mask_origin: str = "png"
) -> ImportRecord:
    img = blob_image(96, blobs, gray=gray)
    mask = np.zeros((96, 96), dtype=np.uint8)
    for cx, cy, r in blobs:
        yy, xx = np.ogrid[:96, :96]
        mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 255
    return ImportRecord(
        image=promote_to_bgr(img),
        gray=gray,
        mask=mask,
        cls=cls,
        origin=f"{stem}.png",
        id_hint=stem,
        mask_origin=mask_origin,
        box_in_origin=None if mask_origin == "png" else (1, 2, 3, 4),
    )


def _write_bank(root: Path) -> Path:
    w = BankWriter(root, name="t")
    w.ensure_classes(["spot", "crack"])
    opts = ImportOptions(margin=8)
    w.add(_record("b", "spot", [(40, 40, 8)]), opts)
    w.add(_record("a", "spot", [(30, 30, 6)], mask_origin="yolo-box:grabcut"), opts)
    w.add(_record("g", "crack", [(50, 50, 7)], gray=True), opts)
    w.finish({"importer": "test"})
    return root


def test_load_sorted_binarized_promoted(tmp_path: Path) -> None:
    root = _write_bank(tmp_path / "bank")
    bank = Bank.load(root)
    assert bank.name == "t" and bank.classes == ["spot", "crack"]
    assert bank.class_ids == {"spot": 0, "crack": 1}
    assert bank.counts() == {"spot": 2, "crack": 1} and len(bank) == 3
    ids = [s.id for s in bank.by_class("spot")]
    assert ids == ["spot/a", "spot/b"]  # 이름 정렬
    for s in bank.sources():
        assert s.image.ndim == 3 and s.image.shape[2] == 3 and s.image.dtype == np.uint8
        assert s.mask.shape == s.image.shape[:2] and set(np.unique(s.mask)) <= {0, 255}
        assert np.count_nonzero(s.mask) > 0
    g = bank.by_class("crack")[0]
    assert np.array_equal(g.image[:, :, 0], g.image[:, :, 1])  # 흑백 승격 = 세 채널 동일
    disk = imgio.read_image(root / "crack" / "g.png")
    assert disk[1] is True  # 디스크에는 1ch로 저장
    a = bank.by_class("spot")[0]
    assert a.mask_origin == "yolo-box:grabcut" and a.origin == "a.png"
    assert bank.imports and bank.imports[0]["importer"] == "test"
    assert bank.warnings == []


def test_mask_binarization_on_load(tmp_path: Path) -> None:
    root = _write_bank(tmp_path / "bank")
    p = root / "spot" / "a.mask.png"
    m = imgio.read_mask(p)
    m[m > 0] = 200  # 회색값 섞기
    m[0, 0] = 100  # 127 이하 → 0
    imgio.write_image(p, m)
    bank = Bank.load(root)
    a = bank.by_class("spot")[0]
    assert set(np.unique(a.mask)) == {0, 255} and a.mask[0, 0] == 0


def test_summary_counts_exact_vs_estimated(tmp_path: Path) -> None:
    bank = Bank.load(_write_bank(tmp_path / "bank"))
    rows = {r.cls: r for r in bank.summary()}
    assert rows["spot"].count == 2 and rows["spot"].exact == 1 and rows["spot"].estimated == 1
    assert rows["spot"].origins == {"png": 1, "yolo-box:grabcut": 1}
    assert rows["crack"].area_median > 0 and rows["crack"].class_id == 1


def test_fingerprint_changes_with_content(tmp_path: Path) -> None:
    root = _write_bank(tmp_path / "bank")
    fp1 = Bank.load(root).fingerprint()
    assert fp1 == Bank.load(root).fingerprint()
    w = BankWriter(root)
    w.add(_record("c", "spot", [(60, 60, 5)]), ImportOptions(margin=8))
    w.finish()
    assert Bank.load(root).fingerprint() != fp1


def test_missing_bank_and_broken_source_fail_soft(tmp_path: Path) -> None:
    with pytest.raises(BankError):
        Bank.load(tmp_path / "nope")
    root = _write_bank(tmp_path / "bank")
    (root / "spot" / "a.mask.png").write_bytes(b"not a png")
    warned: list[str] = []
    bank = Bank.load(root, warn=warned.append)
    assert bank.counts() == {"spot": 1, "crack": 1}
    assert warned and "spot/a" in warned[0] and bank.warnings == warned


def test_unlisted_class_folder_is_appended_with_warning(tmp_path: Path) -> None:
    root = _write_bank(tmp_path / "bank")
    meta = yaml.safe_load((root / BANK_FILE).read_text(encoding="utf-8"))
    meta["classes"] = ["spot"]  # crack 폴더는 있는데 목록에서 빠짐
    (root / BANK_FILE).write_text(yaml.safe_dump(meta), encoding="utf-8")
    bank = Bank.load(root)
    assert bank.classes == ["spot", "crack"] and any("crack" in w for w in bank.warnings)


def test_writer_components_min_area_dup_and_margin(tmp_path: Path) -> None:
    root = tmp_path / "bank"
    w = BankWriter(root)
    two = _record("two", "spot", [(25, 25, 6), (70, 70, 3)])  # 성분 2개, 작은 쪽 면적 ≈ 29
    added = w.add(two, ImportOptions(margin=8, min_area=16))
    assert [a.source_id for a in added] == ["two-0", "two-1"]
    w2 = BankWriter(root)
    added = w2.add(two, ImportOptions(margin=8, min_area=40))
    assert [a.source_id for a in added] == ["two-0-dup1"] and w2.stats.dropped_small == 1
    assert w2.stats.duplicates == 1
    whole = w2.add(two, ImportOptions(margin=8, keep_whole=True))
    assert len(whole) == 1 and whole[0].source_id == "two"  # "two.json"은 없었으므로 접미사 없음
    meta = json.loads((root / "spot" / "two-0.json").read_text(encoding="utf-8"))
    assert meta["margin_px"] == 8 and meta["bbox_in_origin"][2] >= 2 * 6 + 2 * 8 - 1
    with pytest.raises(ValueError, match="margin"):
        check_margin(MIN_MARGIN - 1)
    with pytest.raises(ValueError):
        w2.add(two, ImportOptions(margin=3))
    assert MIN_MARGIN == 6


def test_writer_merges_class_names_and_warns_on_order(tmp_path: Path) -> None:
    root = tmp_path / "bank"
    w = BankWriter(root)
    assert w.ensure_classes(["a", "b"]) == []
    w.finish()
    w2 = BankWriter(root)
    warnings = w2.ensure_classes(["b", "a", "c"])
    assert w2.classes == ["a", "b", "c"] and warnings and "순서" in warnings[0]


def test_from_sources_orders_by_id_and_keeps_classes() -> None:
    from tests.fixtures import line_defect

    srcs = [line_defect(10, 3, cls="z"), line_defect(12, 3, cls="a")]
    bank = Bank.from_sources(srcs, classes=["z", "a"])
    assert bank.classes == ["z", "a"] and bank.class_ids == {"z": 0, "a": 1}
    auto = Bank.from_sources(srcs)
    assert auto.classes == ["a", "z"]  # id 정렬 등장 순서

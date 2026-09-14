"""``import-pairs`` (설계 §3.3·§11 test_bank) — suffix/동일 stem/CSV 세 매칭 · 성분 분리 · min-area · 중복 id · 누적 ·
클래스 지정 방식(고정/폴더) · 마스크 크기 불일치 fail-soft · 흑백 보존."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anograft.bank import Bank
from anograft.bank.importers.pairs import (
    PairRecord,
    discover_pairs,
    find_mask,
    import_pair_records,
    import_pairs,
    read_pairs_csv,
)
from anograft.io import imgio
from tests.fixtures import blob_image, blob_mask, fake_pairs_dataset


def test_find_mask_prefers_suffix_then_same_stem(tmp_path: Path) -> None:
    masks = tmp_path / "m"
    (masks / "sub").mkdir(parents=True)
    imgio.write_image(masks / "sub" / "x_mask.png", np.zeros((4, 4), np.uint8))
    imgio.write_image(masks / "sub" / "x.png", np.zeros((4, 4), np.uint8))
    imgio.write_image(masks / "sub" / "y.bmp", np.zeros((4, 4), np.uint8))
    assert find_mask(masks, Path("sub/x.png"), "_mask") == masks / "sub" / "x_mask.png"
    assert find_mask(masks, Path("sub/y.png"), "_mask") == masks / "sub" / "y.bmp"
    assert find_mask(masks, Path("sub/z.png"), "_mask") is None
    assert find_mask(masks, Path("sub/x.png"), "") == masks / "sub" / "x.png"


def test_discover_pairs_class_modes_and_missing(tmp_path: Path) -> None:
    d = fake_pairs_dataset(tmp_path)
    warns: list[str] = []
    pairs, missing = discover_pairs(
        d["images"],
        d["masks"],
        cls=None,
        class_from_dir=True,
        mask_suffix="_mask",
        warn=warns.append,
    )
    assert missing == 1 and len(pairs) == 3
    assert [(p.cls, p.origin) for p in pairs] == [
        ("crack", "crack/c.png"),
        ("spot", "spot/a.png"),
        ("spot", "spot/b.png"),
    ]
    assert pairs[2].mask.name == "b.png"  # 동일 stem 매칭
    assert any("z.png" in w and "마스크 없음" in w for w in warns)
    fixed, _ = discover_pairs(
        d["images"],
        d["masks"],
        cls="dent",
        class_from_dir=False,
        mask_suffix="_mask",
        warn=warns.append,
    )
    assert {p.cls for p in fixed} == {"dent"}
    with pytest.raises(ValueError, match="--class"):
        discover_pairs(
            d["images"], d["masks"], cls=None, class_from_dir=False, mask_suffix="", warn=print
        )


def test_read_pairs_csv_resolves_relative_to_csv(tmp_path: Path) -> None:
    d = fake_pairs_dataset(tmp_path)
    rows = read_pairs_csv(d["csv"])
    assert [r.cls for r in rows] == ["spot", "crack"]
    assert rows[0].image == tmp_path / "images" / "spot" / "a.png" and rows[0].mask.is_file()
    bad = tmp_path / "bad.csv"
    bad.write_text("image,cls\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="image,mask,class"):
        read_pairs_csv(bad)
    bad.write_text("image,mask,class\nx,,c\n", encoding="utf-8")
    with pytest.raises(ValueError, match="비어"):
        read_pairs_csv(bad)


def test_import_pairs_components_ids_classes_and_meta(tmp_path: Path) -> None:
    d = fake_pairs_dataset(tmp_path)
    res = import_pairs(d["images"], d["masks"], tmp_path / "bank", class_from_dir=True)
    assert res.n_pairs == 3 and res.n_missing_mask == 1 and res.stats.dropped_small == 0
    ids = sorted((a.cls, a.source_id) for a in res.stats.added)
    assert ids == [("crack", "c"), ("spot", "a"), ("spot", "b-0"), ("spot", "b-1")]  # b는 성분 2개
    assert res.classes == ["crack", "spot"]  # 탐색 순서(정렬) = id 순서
    bank = Bank.load(tmp_path / "bank")
    assert bank.class_ids == {"crack": 0, "spot": 1} and len(bank) == 4
    meta = json.loads((tmp_path / "bank" / "spot" / "a.json").read_text(encoding="utf-8"))
    assert (
        meta["mask_origin"] == "png" and meta["origin"] == "spot/a.png" and meta["margin_px"] == 16
    )
    # 크롭 = 성분 bbox + margin, 마스크 면적 = 원판
    a = next(s for s in bank.sources() if s.id == "spot/a")
    assert a.mask.sum() // 255 == pytest.approx(np.pi * 8 * 8, rel=0.15)
    assert a.image.shape[:2] == a.mask.shape


def test_import_pairs_csv_mode_and_accumulation(tmp_path: Path) -> None:
    d = fake_pairs_dataset(tmp_path)
    out = tmp_path / "bank"
    res = import_pairs(None, None, out, csv_path=d["csv"], tags=("lot1",))
    assert res.n_pairs == 2 and sorted(a.source_id for a in res.stats.added) == ["a", "c"]
    # 같은 CSV를 다시 넣으면 누적 + -dup 경고
    res2 = import_pairs(None, None, out, csv_path=d["csv"])
    assert res2.stats.duplicates == 2 and sorted(a.source_id for a in res2.stats.added) == [
        "a-dup1",
        "c-dup1",
    ]
    bank = Bank.load(out)
    assert len(bank) == 4 and len(bank.imports) == 2 and bank.imports[0]["importer"] == "pairs"
    assert "lot1" in next(s for s in bank.sources() if s.id == "spot/a").tags


def test_import_pair_records_fail_soft_on_size_mismatch_and_min_area(tmp_path: Path) -> None:
    img = tmp_path / "i.png"
    imgio.write_image(img, blob_image(64, [(32, 32, 6)]))
    small = tmp_path / "m_small.png"
    imgio.write_image(small, blob_mask(32, [(16, 16, 3)]))  # 크기 불일치
    good = tmp_path / "m.png"
    imgio.write_image(good, blob_mask(64, [(32, 32, 6), (10, 10, 1)]))  # 두 번째 성분은 면적 5px
    gray = tmp_path / "g.png"
    imgio.write_image(gray, blob_image(64, [(32, 32, 6)], gray=True))
    res = import_pair_records(
        [
            PairRecord(img, small, "a", origin="i-small"),
            PairRecord(img, good, "a", origin="i-good"),
            PairRecord(gray, good, "a", origin="g"),
            PairRecord(tmp_path / "nope.png", good, "a"),
        ],
        tmp_path / "bank",
        min_area=16,
    )
    assert res.n_pairs == 2 and res.stats.dropped_small == 2
    assert any("크기" in w for w in res.warnings) and any("읽기 실패" in w for w in res.warnings)
    bank = Bank.load(tmp_path / "bank")
    # 성분 번호 k는 min_area 필터 전 인덱스 — 작은 성분(0)이 버려져도 남은 성분은 -1 (BankWriter 규약 유지)
    assert sorted(s.id for s in bank.sources()) == ["a/g-1", "a/i-1"]
    # 흑백 원본은 1ch PNG로 저장됐고 로드 시 3ch 승격
    raw, was_gray = imgio.read_image(tmp_path / "bank" / "a" / "g-1.png")
    assert was_gray and raw.shape[2] == 3


def test_import_pairs_keep_whole_and_missing_dirs(tmp_path: Path) -> None:
    d = fake_pairs_dataset(tmp_path)
    res = import_pairs(d["images"], d["masks"], tmp_path / "bank", cls="x", keep_whole=True)
    assert sorted(a.source_id for a in res.stats.added) == ["a", "b", "c"]
    with pytest.raises(FileNotFoundError):
        import_pairs(tmp_path / "nope", d["masks"], tmp_path / "b2", cls="x")
    with pytest.raises(FileNotFoundError):
        import_pairs(d["images"], tmp_path / "nope", tmp_path / "b2", cls="x")
    with pytest.raises(ValueError, match="margin"):
        import_pairs(d["images"], d["masks"], tmp_path / "b2", cls="x", margin=2)

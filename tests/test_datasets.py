"""표준 데이터셋 어댑터 (설계 §3.4·§11 test_datasets) — mvtec-ad 가 fake 트리에서 (image, mask, class)를 내고 good 을 제외;
``import-dataset`` 결과가 같은 쌍을 ``import-pairs``로 넣은 것과 **동일**(어댑터 = 변환기일 뿐); 레이아웃 오류 메시지; info 출력."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anograft.bank import Bank
from anograft.bank.importers.dataset import import_dataset
from anograft.bank.importers.pairs import import_pair_records
from anograft.datasets import (
    REGISTRY,
    DatasetError,
    adapter_names,
    get_adapter,
    info_lines,
    mvtec_ad,
)
from tests.fixtures import fake_mvtec_tree


def test_registry_has_mvtec_ad_with_license_and_categories() -> None:
    assert adapter_names() == ["mvtec-ad"] and "mvtec-ad" in REGISTRY
    a = get_adapter("mvtec-ad")
    assert a.info.license.startswith("CC BY-NC-SA") and len(a.info.categories) == 15
    assert "metal_nut" in a.info.categories and "zipper" in mvtec_ad.GRAY_CATEGORIES
    lines = info_lines(a.info)
    assert any("ground_truth" in ln for ln in lines) and any("재배포" in ln for ln in lines)
    with pytest.raises(DatasetError, match="mvtec-ad"):
        get_adapter("nope")


def test_mvtec_adapter_yields_pairs_excludes_good_and_warns_missing_mask(tmp_path: Path) -> None:
    cat = fake_mvtec_tree(tmp_path)
    a = get_adapter("mvtec-ad")
    warns: list[str] = []
    pairs = list(a.defects(cat, warn=warns.append))
    assert [(p.cls, p.image.name) for p in pairs] == [
        ("hole", "000.png"),
        ("scratch", "000.png"),
        ("scratch", "001.png"),
    ]
    assert all(p.mask.name == f"{p.image.stem}_mask.png" for p in pairs)
    assert pairs[0].id_hint == "metal_nut-hole-000" and pairs[0].tags == ("mvtec-ad", "metal_nut")
    assert len(warns) == 1 and "hole/001.png" in warns[0]
    assert [p.name for p in a.normals(cat)] == ["000.png", "001.png", "002.png", "003.png"]
    assert a.category(cat) == "metal_nut"


def test_mvtec_layout_errors_carry_help(tmp_path: Path) -> None:
    a = get_adapter("mvtec-ad")
    with pytest.raises(DatasetError, match="ground_truth"):
        list(a.defects(tmp_path / "empty"))
    with pytest.raises(DatasetError, match="train/good"):
        a.normals(tmp_path / "empty")


def test_import_dataset_equals_import_pairs_of_same_records(tmp_path: Path) -> None:
    """어댑터는 변환기일 뿐 — 은행 파일이 바이트 단위로 같다(bank.yaml의 imports 이력만 다르다)."""
    cat = fake_mvtec_tree(tmp_path / "mvtec")
    res = import_dataset("mvtec-ad", cat, tmp_path / "bank_ds")
    assert res.dataset == "mvtec-ad" and res.category == "metal_nut"
    assert res.pairs.n_pairs == 3 and len(res.normals) == 4
    assert any("hole/001.png" in w for w in res.pairs.warnings)
    pairs = list(get_adapter("mvtec-ad").defects(cat))
    import_pair_records(pairs, tmp_path / "bank_pairs")

    def tree(root: Path) -> dict[str, bytes]:
        return {
            p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "bank.yaml"
        }

    a, b = tree(tmp_path / "bank_ds"), tree(tmp_path / "bank_pairs")
    assert a and a == b
    bank = Bank.load(tmp_path / "bank_ds")
    assert bank.classes == ["hole", "scratch"] and len(bank) == 4  # hole/000 성분 2개
    assert bank.imports[0]["importer"] == "dataset" and bank.imports[0]["category"] == "metal_nut"
    meta = json.loads(
        (tmp_path / "bank_ds" / "scratch" / "metal_nut-scratch-000.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["tags"] == ["mvtec-ad", "metal_nut"] and meta["um_per_px"] is None
    assert meta["origin"] == "metal_nut/test/scratch/000.png"


def test_import_dataset_unknown_name_and_bad_root(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        import_dataset("visa", tmp_path, tmp_path / "b")
    with pytest.raises(DatasetError, match="카테고리 폴더"):
        import_dataset("mvtec-ad", tmp_path / "nope", tmp_path / "b")

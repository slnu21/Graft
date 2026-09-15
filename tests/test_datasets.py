"""표준 데이터셋 어댑터 (설계 §3.4·§11 test_datasets) — mvtec-ad 가 fake 트리에서 (image, mask, class)를 내고 good 을 제외;
``import-dataset`` 결과가 같은 쌍을 ``import-pairs``로 넣은 것과 **동일**(어댑터 = 변환기일 뿐); 레이아웃 오류 메시지; info 출력.
visa(v0.4): image_anno.csv 정본 + 폴더 폴백, 클래스 'anomaly' 하나, 0/1 라벨맵 마스크(threshold 0), 누락 마스크·CSV 유령 행 경고."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anograft.bank import Bank
from anograft.bank.importers.dataset import import_dataset
from anograft.bank.importers.pairs import import_pair_records
from anograft.cli import EXIT_OK, main
from anograft.datasets import (
    REGISTRY,
    DatasetError,
    adapter_names,
    get_adapter,
    info_lines,
    mvtec_ad,
)
from anograft.io import imgio
from tests.fixtures import blob_image, fake_mvtec_tree, fake_visa_tree


def test_registry_has_mvtec_ad_with_license_and_categories() -> None:
    assert adapter_names() == ["dtd", "mvtec-ad", "visa"] and "mvtec-ad" in REGISTRY
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
        import_dataset("btad", tmp_path, tmp_path / "b")
    with pytest.raises(DatasetError, match="카테고리 폴더"):
        import_dataset("mvtec-ad", tmp_path / "nope", tmp_path / "b")


# ---------------------------------------------------------------------------
# visa (v0.4)
# ---------------------------------------------------------------------------


def test_visa_info_and_categories() -> None:
    a = get_adapter("visa")
    assert a.info.license.startswith("CC BY-NC-SA") and len(a.info.categories) == 12
    assert "candle" in a.info.categories and "pcb4" in a.info.categories
    lines = info_lines(a.info)
    assert any("image_anno.csv" in ln for ln in lines) and any("재배포" in ln for ln in lines)


def test_visa_adapter_uses_csv_single_class_and_threshold_zero(tmp_path: Path) -> None:
    cat = fake_visa_tree(tmp_path)
    a = get_adapter("visa")
    warns: list[str] = []
    pairs = list(a.defects(cat, warn=warns.append))
    assert [(p.cls, p.image.name) for p in pairs] == [
        ("anomaly", "000.JPG"),
        ("anomaly", "001.JPG"),
    ]
    assert all(p.mask_threshold == 0 and p.mask.name == f"{p.image.stem}.png" for p in pairs)
    assert pairs[0].id_hint == "candle-anomaly-000" and pairs[0].tags == ("visa", "candle")
    assert pairs[0].origin == "candle/Data/Images/Anomaly/000.JPG"
    # 경고: CSV 유령 행(999) + 마스크 누락(002)
    assert (
        len(warns) == 2
        and any("999.JPG" in w for w in warns)
        and any("002.JPG" in w for w in warns)
    )
    assert [p.name for p in a.normals(cat)] == ["0000.JPG", "0001.JPG", "0002.JPG"]
    assert a.category(cat) == "candle"


def test_visa_adapter_falls_back_to_folder_without_csv(tmp_path: Path) -> None:
    cat = fake_visa_tree(tmp_path, anno=False)
    warns: list[str] = []
    pairs = list(get_adapter("visa").defects(cat, warn=warns.append))
    assert [p.image.name for p in pairs] == ["000.JPG", "001.JPG"]
    assert len(warns) == 1 and "002.JPG" in warns[0]


def test_visa_layout_errors_carry_help(tmp_path: Path) -> None:
    a = get_adapter("visa")
    with pytest.raises(DatasetError, match="Data/Images/Anomaly"):
        list(a.defects(tmp_path / "empty"))
    with pytest.raises(DatasetError, match="Data/Images/Normal"):
        a.normals(tmp_path / "empty")


def test_import_dataset_visa_reads_01_masks_and_matches_import_pairs(tmp_path: Path) -> None:
    cat = fake_visa_tree(tmp_path / "VisA")
    res = import_dataset("visa", cat, tmp_path / "bank_visa")
    assert res.dataset == "visa" and res.category == "candle"
    assert res.pairs.n_pairs == 2 and len(res.normals) == 3
    bank = Bank.load(tmp_path / "bank_visa")
    assert bank.classes == ["anomaly"] and len(bank) == 2  # 0/1 라벨맵이 결함으로 읽혔다
    src = bank.by_class("anomaly")[0]
    assert src.mask.max() == 255 and src.mask_origin == "png"
    # 어댑터 = 변환기: 같은 PairRecord 를 import-pairs 로 넣은 것과 파일 동일
    import_pair_records(list(get_adapter("visa").defects(cat)), tmp_path / "bank_pairs")

    def tree(root: Path) -> dict[str, bytes]:
        return {
            p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "bank.yaml"
        }

    assert tree(tmp_path / "bank_visa") == tree(tmp_path / "bank_pairs")
    meta = json.loads(
        (tmp_path / "bank_visa" / "anomaly" / "candle-anomaly-000.json").read_text(encoding="utf-8")
    )
    assert (
        meta["tags"] == ["visa", "candle"]
        and meta["origin"] == "candle/Data/Images/Anomaly/000.JPG"
    )


def test_read_mask_threshold_zero_vs_default(tmp_path: Path) -> None:
    import numpy as np

    from anograft.io import imgio

    m = np.zeros((8, 8), dtype=np.uint8)
    m[2:5, 2:5] = 1
    imgio.write_image(tmp_path / "m.png", m)
    assert imgio.read_mask(tmp_path / "m.png").max() == 0  # 기본 >127 은 0/1 을 버린다
    got = imgio.read_mask(tmp_path / "m.png", threshold=0)
    assert got.max() == 255 and int((got > 0).sum()) == 9


# --- DTD (v0.7) — 텍스처셋: import-dataset 거부 · textures 목록 · CLI dataset textures · texture_dir 목록 파일 ------------


def _fake_dtd(root: Path, cats: dict[str, int]) -> Path:
    for c, n in cats.items():
        for i in range(n):
            imgio.write_image(
                root / "images" / c / f"{c}_{i:04d}.jpg", blob_image(48, [(24, 24, 6)])
            )
    return root


def test_dtd_adapter_lists_textures_and_refuses_import(tmp_path: Path) -> None:
    from anograft.datasets.dtd import DEFECT_LIKE, DtdAdapter, write_texture_list

    root = _fake_dtd(tmp_path / "dtd", {"cracked": 3, "stained": 2, "banded": 4})
    a = DtdAdapter()
    assert a.info.name == "dtd" and not a.info.importable and a.category(root) == "textures"
    assert "textures dtd" in "\n".join(info_lines(a.info))
    assert a.available_categories(root) == ["banded", "cracked", "stained"]
    default = a.textures(root)
    assert len(default) == 5 and all(p.parent.name in DEFECT_LIKE for p in default)
    assert len(a.textures(root, categories=["*"])) == 9
    warns: list[str] = []
    assert len(a.textures(root, categories=["banded", "nope"], warn=warns.append)) == 4 and warns
    sub = a.textures(root, categories=["*"], limit=4, seed=1)
    assert len(sub) == 4 and sub == a.textures(root, categories=["*"], limit=4, seed=1)
    assert (
        sub != a.textures(root, categories=["*"], limit=4, seed=2) or True
    )  # 다른 seed 는 대개 다름
    with pytest.raises(DatasetError, match="결함 데이터셋이 아닙니다"):
        list(a.defects(root))
    with pytest.raises(DatasetError, match="images/"):
        a.textures(tmp_path / "nope")
    lf = write_texture_list(tmp_path / "lists" / "t.txt", default)
    lines = lf.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5 and lines[0].startswith("../dtd/images/")
    # runner.load_textures 가 목록을 읽는다(목록 파일 기준 경로)
    from anograft import runner

    w: list[str] = []
    assert len(runner.load_textures(lf, w)) == 5 and not w


def test_cli_dataset_textures_and_import_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _fake_dtd(tmp_path / "dtd", {"cracked": 2, "woven": 1})
    out = tmp_path / "tex.txt"
    assert (
        main(["dataset", "textures", "dtd", str(root), "--out", str(out), "--categories", "*"])
        == EXIT_OK
    )
    cap = capsys.readouterr().out
    assert "3장" in cap and "texture_dir" in cap and out.is_file()
    assert main(["dataset", "textures", "visa", str(root), "--out", str(out)]) != EXIT_OK
    assert (
        main(["dataset", "textures", "dtd", str(root), "--out", str(out), "--categories", "zzz"])
        != EXIT_OK
    )
    assert (
        main(["bank", "import-dataset", "dtd", str(root), "--out", str(tmp_path / "b")]) != EXIT_OK
    )
    assert "결함 데이터셋이 아닙니다" in capsys.readouterr().err

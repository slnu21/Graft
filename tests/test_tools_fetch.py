"""``tools/fetch_public_datasets.py`` — 네트워크 없는 순수 부분(접두 선택 · Magnetic Tile pairs.csv)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from anograft.io import imgio

_spec = importlib.util.spec_from_file_location(
    "fetch_public_datasets",
    Path(__file__).resolve().parents[1] / "tools" / "fetch_public_datasets.py",
)
fetch = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(fetch)


def test_select_prefixed() -> None:
    files = ["screw/test/good/0.png", "grid/x.png", "results/screw/y.png", "README.md"]
    assert fetch.select_prefixed(files, ("screw/",)) == ["screw/test/good/0.png"]
    assert fetch.select_prefixed(files, ("screw/", "grid/")) == files[:2]
    assert fetch.select_prefixed(files, ()) == files
    assert "metal_nut" in fetch.MVTEC_CATEGORIES and len(fetch.DTD_DEFECT_LIKE) == 15


def test_mt_pairs_csv(tmp_path: Path) -> None:
    for cls, n in (("MT_Blowhole", 2), ("MT_Free", 3)):
        d = tmp_path / cls / "Imgs"
        d.mkdir(parents=True)
        for i in range(n):
            imgio.write_image(d / f"exp_{i}.jpg", np.full((8, 8), 128, np.uint8))
            imgio.write_image(d / f"exp_{i}.png", np.zeros((8, 8), np.uint8))
    imgio.write_image(
        tmp_path / "MT_Blowhole" / "Imgs" / "lonely.jpg", np.zeros((8, 8), np.uint8)
    )  # 마스크 없음 → 제외
    pairs, normals = fetch.mt_pairs_csv(tmp_path)
    assert (pairs, normals) == (2, 3)
    lines = (tmp_path / "pairs.csv").read_text(encoding="utf-8").splitlines()
    assert (
        lines[0] == "image,mask,class"
        and lines[1] == "MT_Blowhole/Imgs/exp_0.jpg,MT_Blowhole/Imgs/exp_0.png,blowhole"
    )
    assert (tmp_path / "normals.txt").read_text(encoding="utf-8").splitlines() == [
        f"MT_Free/Imgs/exp_{i}.jpg" for i in range(3)
    ]

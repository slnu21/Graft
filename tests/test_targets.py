"""``io/targets.py`` — 대상 풀 나열 · 같은 stem 충돌 경고(마스크 PNG 가 이미지 옆에 있는 배치, 리허설 2026-09-16 MT)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anograft import runner
from anograft.core import recipe as R
from anograft.io import imgio
from anograft.io.targets import list_targets, stem_collisions, targets_warning
from tests.fixtures import fake_yolo_dataset


def test_stem_collisions_pure() -> None:
    d = Path("x")
    paths = [d / "a.jpg", d / "a.png", d / "b.jpg", Path("y") / "a.jpg", d / "c.png", d / "c.PNG"]
    assert stem_collisions(paths) == [(d / "a.jpg", d / "a.png"), (d / "c.png", d / "c.PNG")]
    assert stem_collisions([d / "a.jpg", d / "b.jpg"]) == []
    assert targets_warning([d / "a.jpg", d / "b.jpg"]) is None
    w = targets_warning(paths)
    assert w and w.startswith("targets:") and "2쌍" in w and "a.jpg + a.png" in w and ".txt" in w


def test_prepare_surfaces_targets_warning(tmp_path: Path) -> None:
    from anograft.bank.importers import yolo as Y

    d = fake_yolo_dataset(tmp_path / "ds")
    normals = tmp_path / "normals.txt"
    Y.import_yolo(
        d["images"],
        d["labels"],
        d["names"],
        tmp_path / "bank",
        mask_from="rect",
        list_normals=normals,
    )
    # 정상 폴더에 같은 stem 의 마스크 PNG 를 옆에 놓는다
    folder = tmp_path / "normals"
    folder.mkdir()
    for i, src in enumerate(imgio.read_path_list(normals)):
        img, _ = imgio.read_image(src)
        imgio.write_image(folder / f"n{i}.jpg", img)
        imgio.write_image(folder / f"n{i}.png", np.zeros(img.shape[:2], dtype=np.uint8))
    n = len(imgio.read_path_list(normals))
    assert len(list_targets(folder)) == 2 * n
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": "t",
            "seed": 1,
            "inputs": {"bank": (tmp_path / "bank").as_posix(), "targets": folder.as_posix()},
            "output": {"root": (tmp_path / "o").as_posix(), "count": 1},
            "pipeline": {
                "preset": "hard-paste",
                "source": {"method": "bank", "min_sources_warn": 1},
            },
        }
    )
    prep = runner.prepare(rec)
    hits = [w for w in prep.warnings if w.startswith("targets:")]
    assert len(hits) == 1 and f"{n}쌍" in hits[0]
    # reprepare(카드 편집) 뒤에도 유지 · .txt 목록이면 경고 없음
    assert any(w.startswith("targets:") for w in runner.reprepare(prep, rec).warnings)
    rec2 = R.Recipe.from_dict(
        {**rec.to_dict(), "inputs": {**rec.to_dict()["inputs"], "targets": normals.as_posix()}}
    )
    assert not any(w.startswith("targets:") for w in runner.prepare(rec2).warnings)

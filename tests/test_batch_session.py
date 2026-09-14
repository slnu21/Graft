"""``gui/batch/session.py`` + ``runner.run(should_stop=)`` — Qt 없이: 오버라이드 적용(출력·장수·시드·writer 형식 갈아 끼우기) ·
검증 오류 → BatchError · run_batch 가 CLI run 과 같은 결과 · 취소(should_stop)가 그때까지의 파일·manifest 를 남긴다."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from anograft.bank.importers import yolo as Y
from anograft.core import recipe as R
from anograft.gui.batch.session import BatchError, BatchSession, run_batch, summary_text
from anograft.io.manifest import read_manifest
from tests.fixtures import fake_yolo_dataset


@pytest.fixture
def recipe_file(tmp_path: Path) -> Path:
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
    data = {
        "version": 1,
        "name": "b",
        "seed": 3,
        "inputs": {"bank": (tmp_path / "bank").as_posix(), "targets": normals.as_posix()},
        "output": {"root": (tmp_path / "out").as_posix(), "count": 4, "include_normals": False},
        "pipeline": {
            "preset": "hard-paste",
            "source": {"method": "bank", "min_sources_warn": 1},
            "placement": {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 4},
        },
    }
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def test_set_recipe_fills_overrides_and_build_applies_them(
    recipe_file: Path, tmp_path: Path
) -> None:
    s = BatchSession()
    with pytest.raises(BatchError, match="레시피"):
        s.build_recipe()
    rec = s.load(recipe_file)
    assert (
        s.out == rec.output.root.as_posix() and s.count == 4 and s.seed == 3 and s.writer == "yolo"
    )
    s.out, s.count, s.seed, s.writer, s.mvtec_category = (
        (tmp_path / "o2").as_posix(),
        2,
        9,
        "mvtec",
        "plate",
    )
    built = s.build_recipe()
    assert built.output.root == tmp_path / "o2" and built.output.count == 2 and built.seed == 9
    assert built.output.writer.format == "mvtec" and built.output.writer.category == "plate"
    assert built.pipeline.blend.method == "paste"  # 나머지는 그대로
    assert "anograft run" in s.run_command() and "--count 2" in s.run_command()
    s.writer = "pairs"
    assert s.build_recipe().output.writer.format == "pairs"
    s.writer = "coco"
    with pytest.raises(BatchError, match="writer"):
        s.build_recipe()
    s.writer, s.count = "yolo", 0
    with pytest.raises(BatchError, match="장수"):
        s.build_recipe()
    s.count, s.out = 1, ""
    with pytest.raises(BatchError, match="출력"):
        s.build_recipe()
    with pytest.raises(BatchError, match="로드 실패"):
        s.load(tmp_path / "nope.yaml")


def test_run_batch_matches_cli_and_reports_progress(recipe_file: Path, tmp_path: Path) -> None:
    s = BatchSession()
    s.load(recipe_file)
    seen: list[int] = []
    summary = run_batch(s.build_recipe(), progress=lambda d, t, r: seen.append(d))
    assert seen == [1, 2, 3, 4] and not summary.cancelled and summary.done == 4
    assert summary.writer.n_ok + summary.writer.n_skipped == 4
    assert (tmp_path / "out" / "manifest.csv").is_file() and "완료" in summary_text(summary)
    # CLI 와 같은 결과
    from anograft.cli import main

    assert main(
        ["run", str(recipe_file), "--out", (tmp_path / "cli").as_posix(), "--workers", "0"]
    ) in (0, 2)
    a = sorted(p.name for p in (tmp_path / "out" / "images").glob("*.png"))
    b = sorted(p.name for p in (tmp_path / "cli" / "images").glob("*.png"))
    assert a == b
    for n in a:
        assert (tmp_path / "out" / "images" / n).read_bytes() == (
            tmp_path / "cli" / "images" / n
        ).read_bytes()


def test_run_batch_cancel_keeps_partial_output(recipe_file: Path, tmp_path: Path) -> None:
    s = BatchSession()
    s.load(recipe_file)
    s.out = (tmp_path / "part").as_posix()
    done: list[int] = []
    summary = run_batch(
        s.build_recipe(),
        progress=lambda d, t, r: done.append(d),
        should_stop=lambda: len(done) >= 2,
    )
    assert summary.cancelled and summary.done == 2 and done == [1, 2]
    rows = read_manifest(tmp_path / "part" / "manifest.csv")
    assert len(rows) == 2 and any("취소" in w for w in summary.warnings)
    assert summary_text(summary).startswith("취소됨 — 2/4")


def test_run_batch_prepare_error_is_batch_error(recipe_file: Path, tmp_path: Path) -> None:
    s = BatchSession()
    s.load(recipe_file)
    rec = s.build_recipe()
    d = rec.to_dict()
    d["inputs"]["bank"] = (tmp_path / "no-bank").as_posix()
    with pytest.raises(BatchError):
        run_batch(R.Recipe.from_dict(d))

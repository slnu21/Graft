"""``source.single_class_per_image``(v0.8.x) — 한 이미지의 결함은 첫 결함이 뽑은 클래스로(mvtec writer 이미지당 단일 클래스)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from anograft.bank.importers import yolo as Y
from anograft.cli import EXIT_OK, main
from anograft.core import recipe as R
from anograft.core.pipeline import Pipeline
from tests.fixtures import disk_target, fake_yolo_dataset, line_defect, memory_bank, pipeline_deps


def _recipe(single: bool, n: int = 3) -> R.Recipe:
    return R.Recipe.from_dict(
        {
            "version": 1,
            "name": "hard-paste",
            "seed": 5,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1, "defects_per_image": [n, n]},
            "pipeline": {
                "preset": "hard-paste",
                "source": {
                    "method": "bank",
                    "single_class_per_image": single,
                    "min_sources_warn": 0,
                },
                "placement": {"roi": {"method": "none"}, "margin_px": 4},
            },
        }
    )


def test_all_defects_share_first_class_and_default_is_off() -> None:
    assert R.BankSourceConfig().single_class_per_image is False
    bank = memory_bank([line_defect(18, 4), line_defect(10, 6, cls="dent")])
    target = disk_target(160)
    mixed_seen = False
    for i in range(12):
        on = Pipeline.from_recipe(_recipe(True), pipeline_deps(_recipe(True), bank)).run_one(
            target, i
        )
        classes = [
            d["source"]["class"]
            for d in on.sidecar["defects"]
            if "source" in d and "class" in d["source"]
        ]
        assert len(set(classes)) == 1, (i, classes)
        off = Pipeline.from_recipe(_recipe(False), pipeline_deps(_recipe(False), bank)).run_one(
            target, i
        )
        off_classes = [
            d["source"]["class"]
            for d in off.sidecar["defects"]
            if "source" in d and "class" in d["source"]
        ]
        mixed_seen |= len(set(off_classes)) > 1
        # 첫 결함은 두 설정이 같은 rng 를 쓰므로 같은 클래스·같은 소스
        assert (
            classes[0] == off_classes[0]
            and on.sidecar["defects"][0]["source"]["source_id"]
            == off.sidecar["defects"][0]["source"]["source_id"]
        )
    assert mixed_seen, "기본(off)에서는 클래스가 섞이는 이미지가 있어야 비교가 된다"


def test_mvtec_writer_no_mixed_with_single_class(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
        "name": "m",
        "seed": 3,
        "inputs": {"bank": (tmp_path / "bank").as_posix(), "targets": normals.as_posix()},
        "output": {
            "root": (tmp_path / "out").as_posix(),
            "count": 6,
            "defects_per_image": [2, 3],
            "writer": {"format": "mvtec", "category": "plate", "test_normal_ratio": 0.5},
        },
        "pipeline": {
            "preset": "hard-paste",
            "source": {"method": "bank", "min_sources_warn": 1, "single_class_per_image": True},
            "placement": {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 4},
        },
    }
    recipe = tmp_path / "r.yaml"
    recipe.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert main(["run", str(recipe), "--workers", "0"]) == EXIT_OK
    err = capsys.readouterr().err
    assert "가 섞임" not in err
    data["pipeline"]["source"]["single_class_per_image"] = False
    data["output"]["root"] = (tmp_path / "out2").as_posix()
    recipe.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert main(["run", str(recipe), "--workers", "0"]) == EXIT_OK
    err2 = capsys.readouterr().err
    if "가 섞임" in err2:
        assert "single_class_per_image" in err2  # 경고가 해법을 가리킨다

"""COCO writer (v0.7) — 정본 위에 ``annotations.json`` · images(정상 포함, id 순서) · categories(은행 classes, id 1-based) ·
annotations(폴리곤 segmentation 을 다시 칠하면 GT 마스크와 IoU ≥ 0.8, bbox = Instance.bbox, area = 픽셀 수) · 사이드카 writer ·
manifest label · 면적 0 경고 · 결정성 · CLI run e2e."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from anograft.cli import EXIT_OK, main
from anograft.core import recipe as R
from anograft.core.pipeline import Pipeline
from anograft.core.types import GraftResult, Instance
from anograft.io import imgio
from anograft.io.manifest import read_manifest
from anograft.io.writers import make_writer
from anograft.io.writers.coco import CocoWriter, annotations_for, instance_polygons
from tests.fixtures import disk_target, fake_yolo_dataset, line_defect, memory_bank, pipeline_deps


def _recipe(root: Path, *, count: int = 3, seed: int = 5, **writer: object) -> R.Recipe:
    return R.Recipe.from_dict(
        {
            "version": 1,
            "name": "c",
            "seed": seed,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {
                "root": str(root),
                "count": count,
                "defects_per_image": [2, 2],
                "writer": {"format": "coco", **writer},
            },
            "pipeline": {
                "preset": "hard-paste",
                "placement": {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 4},
            },
        }
    )


def _results(rec: R.Recipe, n: int) -> tuple[list[GraftResult], list[str]]:
    bank = memory_bank([line_defect(14, 3), line_defect(8, 5, cls="dent")])
    pipe = Pipeline.from_recipe(rec, pipeline_deps(rec, bank))
    return [pipe.run_one(disk_target(96), i) for i in range(n)], bank.classes


def _write(tmp_path: Path, *, n: int = 3, n_normals: int = 2, **writer: object):
    root = tmp_path / "out"
    rec = _recipe(root, count=n, **writer)
    results, classes = _results(rec, n)
    w = CocoWriter(rec.output.writer)  # type: ignore[arg-type]
    w.begin(root, rec, "hash", classes, bank_fingerprint="fp")
    for k in range(n_normals):
        p = tmp_path / "n" / f"plate{k}.png"
        imgio.write_image(p, disk_target(64).image)
        w.write_normal(p)
    for r in results:
        w.write_synthetic(r)
    return root, rec, results, w.finish(), w


def _poly_area(polys: list[list[float]]) -> float:
    return sum(cv2.contourArea(np.array(p, dtype=np.float32).reshape(-1, 1, 2)) for p in polys)


def _raster_iou(polys: list[list[float]], mask: np.ndarray) -> float:
    """학습 프레임워크처럼 폴리곤을 다시 칠해 GT 마스크와 IoU — 가는 결함은 윤곽 면적(픽셀 중심 통과)이 작게 나온다."""
    canvas = np.zeros(mask.shape[:2], dtype=np.uint8)
    for p in polys:
        pts = np.round(np.array(p, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
        cv2.fillPoly(canvas, [pts], 255)
    a, b = canvas > 0, mask > 0
    return float((a & b).sum()) / float((a | b).sum())


def test_instance_polygons_and_annotations_for() -> None:
    m = np.zeros((40, 40), dtype=np.uint8)
    cv2.circle(m, (20, 20), 10, 255, -1)
    polys = instance_polygons(m)
    assert len(polys) == 1 and len(polys[0]) >= 6 and len(polys[0]) % 2 == 0
    assert abs(_poly_area(polys) - np.count_nonzero(m)) / np.count_nonzero(m) < 0.15
    inst = Instance("a", 0, m, (10, 10, 21, 21), int(np.count_nonzero(m)), 0)
    empty = Instance("b", 1, np.zeros((40, 40), dtype=np.uint8), (0, 0, 1, 1), 0, 1)
    anns, warns = annotations_for([inst, empty], image_id=7, next_id=3)
    assert len(anns) == 1 and anns[0]["id"] == 3 and anns[0]["image_id"] == 7
    assert (
        anns[0]["category_id"] == 1
        and anns[0]["bbox"] == [10, 10, 21, 21]
        and anns[0]["iscrowd"] == 0
    )
    assert anns[0]["area"] == int(np.count_nonzero(m)) and len(warns) == 1 and "면적 0" in warns[0]
    assert instance_polygons(np.zeros((4, 4), dtype=np.uint8)) == []


def test_annotations_json_structure_matches_canonical(tmp_path: Path) -> None:
    root, _rec, results, summary, w = _write(tmp_path)
    assert summary.files["annotations"] == "annotations.json" and summary.n_ok == 3
    doc = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    assert set(doc) == {"info", "licenses", "images", "annotations", "categories"}
    assert [c["name"] for c in doc["categories"]] == ["dent", "scratch"] or [
        c["name"] for c in doc["categories"]
    ] == w.classes
    assert [c["id"] for c in doc["categories"]] == list(range(1, len(w.classes) + 1))
    # images: 정상 2 + 합성 3, id 연속, 파일이 실제로 있고 크기가 맞다
    assert [im["id"] for im in doc["images"]] == [1, 2, 3, 4, 5]
    for im in doc["images"]:
        img, _ = imgio.read_image(root / "images" / im["file_name"])
        assert img.shape[:2] == (im["height"], im["width"])
    normal_ids = {im["id"] for im in doc["images"] if im["file_name"].startswith("n_")}
    assert normal_ids == {1, 2} and not any(a["image_id"] in normal_ids for a in doc["annotations"])
    # annotations: 인스턴스 수와 같고, 폴리곤 면적 ≈ GT 면적, bbox = Instance.bbox, category = class_id + 1
    n_inst = sum(len(r.instances) for r in results)
    assert len(doc["annotations"]) == n_inst and [a["id"] for a in doc["annotations"]] == list(
        range(1, n_inst + 1)
    )
    rows = {r["index"]: r for r in read_manifest(root / "manifest.csv") if r["status"] == "ok"}
    for r in results:
        meta = json.loads((root / "meta" / f"{r.index:06d}.json").read_text(encoding="utf-8"))
        we = meta["writer"]
        assert (
            we["format"] == "coco"
            and rows[str(r.index)]["label"] == f"annotations.json#{we['image_id']}"
        )
        anns = [a for a in doc["annotations"] if a["image_id"] == we["image_id"]]
        assert [a["id"] for a in anns] == we["annotation_ids"] and len(anns) == len(r.instances)
        for a, inst in zip(anns, r.instances, strict=True):
            assert a["bbox"] == list(inst.bbox) and a["area"] == inst.area_px
            assert a["category_id"] == inst.class_id + 1
            assert _raster_iou(a["segmentation"], inst.mask) >= 0.8
    # 정상 이미지의 manifest label 도 annotations.json#id
    normals = [r for r in read_manifest(root / "manifest.csv") if r["status"] == "normal"]
    assert [r["label"] for r in normals] == ["annotations.json#1", "annotations.json#2"]


def test_coco_is_deterministic_and_config_validates(tmp_path: Path) -> None:
    a = _write(tmp_path / "a", description="x", supercategory="anomaly")
    b = _write(tmp_path / "b", description="x", supercategory="anomaly")
    ja = (a[0] / "annotations.json").read_text(encoding="utf-8")
    jb = (b[0] / "annotations.json").read_text(encoding="utf-8")
    assert ja == jb
    doc = json.loads(ja)
    assert doc["info"]["description"] == "x" and doc["categories"][0]["supercategory"] == "anomaly"
    assert "date" not in doc["info"]
    w, warn = make_writer(R.CocoWriterConfig())
    assert isinstance(w, CocoWriter) and warn is None
    with pytest.raises(ValidationError):
        R.CocoWriterConfig(format="coco", bogus=1)  # type: ignore[call-arg]


def test_cli_run_with_coco_writer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
    data = {
        "version": 1,
        "name": "coco",
        "seed": 2,
        "inputs": {"bank": (tmp_path / "bank").as_posix(), "targets": normals.as_posix()},
        "output": {"root": (tmp_path / "out").as_posix(), "count": 3, "writer": {"format": "coco"}},
        "pipeline": {
            "preset": "hard-paste",
            "source": {"method": "bank", "min_sources_warn": 1},
            "placement": {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 4},
        },
    }
    recipe = tmp_path / "r.yaml"
    recipe.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert main(["run", str(recipe), "--workers", "0"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "annotations: annotations.json" in out
    doc = json.loads((tmp_path / "out" / "annotations.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in doc["categories"]] == ["spot", "crack"] and doc["images"]

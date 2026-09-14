"""YOLO writer (설계 §8.2·§11 test_writer_yolo) — 인스턴스마다 한 줄 · 정규화 0..1 · class_id = data.yaml names 순서 ·
박스 = 인스턴스 마스크 bbox(역정규화 ±1px) · 2클래스 2줄 · seg 폴리곤(닫힘·bbox 안·조각마다 한 줄) · 정상 이미지 빈 txt ·
**왕복**(출력을 import-yolo --mask-from rect 로 읽으면 박스 == 사이드카 bbox) · 클래스 수 무관."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from anograft.bank import Bank
from anograft.bank.importers.yolo import import_yolo
from anograft.core import recipe as R
from anograft.core.pipeline import Pipeline
from anograft.core.types import GraftResult, Instance
from anograft.io import imgio
from anograft.io.manifest import read_manifest
from anograft.io.writers.yolo import YoloWriter, box_line, label_lines, polygon_lines
from tests.fixtures import disk_target, line_defect, memory_bank, pipeline_deps


def _recipe(root: Path, *, seg: bool = False, count: int = 3) -> R.Recipe:
    return R.Recipe.from_dict(
        {
            "version": 1,
            "name": "y",
            "seed": 5,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {
                "root": str(root),
                "count": count,
                "defects_per_image": [2, 2],
                "writer": {"format": "yolo", "seg": seg},
            },
            "pipeline": {
                "preset": "hard-paste",
                "placement": {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 4},
            },
        }
    )


def _results(rec: R.Recipe, n: int, *, gray: bool = False) -> tuple[list[GraftResult], list[str]]:
    bank = memory_bank([line_defect(14, 3), line_defect(8, 5, cls="dent")])
    pipe = Pipeline.from_recipe(rec, pipeline_deps(rec, bank))
    return [pipe.run_one(disk_target(96, gray=gray), i) for i in range(n)], bank.classes


def _write(tmp_path: Path, *, seg: bool = False, n: int = 3, gray: bool = False):
    root = tmp_path / "out"
    rec = _recipe(root, seg=seg, count=n)
    results, classes = _results(rec, n, gray=gray)
    w = YoloWriter(rec.output.writer)
    w.begin(root, rec, "hash", classes, bank_fingerprint="fp")
    normal = tmp_path / "n" / "plate.png"
    imgio.write_image(normal, disk_target(64).image)
    w.write_normal(normal)
    for r in results:
        w.write_synthetic(r)
    return root, results, classes, w.finish()


def test_box_line_normalization_and_precision() -> None:
    line = box_line(2, (10, 20, 30, 40), (100, 200, 3))
    assert line == "2 0.125000 0.400000 0.150000 0.400000"
    assert all(0.0 <= float(v) <= 1.0 for v in line.split()[1:])


def test_polygon_lines_closed_inside_bbox_and_multi_piece() -> None:
    m = np.zeros((50, 80), dtype=np.uint8)
    m[10:20, 10:30] = 255
    m[30:45, 50:70] = 255  # 두 조각
    lines = polygon_lines(1, m)
    assert len(lines) == 2
    for ln in lines:
        toks = ln.split()
        assert toks[0] == "1" and (len(toks) - 1) % 2 == 0 and (len(toks) - 1) // 2 >= 3
        xs = [float(t) * 80 for t in toks[1::2]]
        ys = [float(t) * 50 for t in toks[2::2]]
        assert min(xs) >= 10 and max(xs) <= 70 and min(ys) >= 10 and max(ys) <= 45
    assert polygon_lines(0, np.zeros((8, 8), np.uint8)) == []


def test_label_lines_skip_empty_instance_with_warning() -> None:
    m = np.zeros((16, 16), dtype=np.uint8)
    empty = Instance("a", 0, m, (0, 0, 0, 0), 0)
    m2 = m.copy()
    m2[4:8, 4:12] = 255
    ok = Instance("b", 1, m2, (4, 4, 8, 4), 32)
    lines, warns = label_lines([empty, ok], (16, 16, 3), seg=False)
    assert lines == ["1 0.500000 0.375000 0.500000 0.250000"] and len(warns) == 1
    seg_lines, seg_warns = label_lines([ok], (16, 16, 3), seg=True)
    assert len(seg_lines) == 1 and seg_lines[0].startswith("1 ") and not seg_warns


def test_layout_labels_data_yaml_manifest_and_sidecar(tmp_path: Path) -> None:
    root, results, classes, summary = _write(tmp_path)
    assert (root / "data.yaml").is_file() and (root / "labels").is_dir()
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == classes == ["dent", "scratch"] and data["nc"] == 2
    assert data["path"] == "." and data["train"] == "images"
    assert (
        summary.files["data.yaml"] == "data.yaml" and summary.n_ok == 3 and summary.n_normals == 1
    )
    # 정상 이미지 = 빈 라벨, manifest label 열에도 경로
    assert (root / "labels" / "n_plate.txt").read_text(encoding="utf-8") == ""
    rows = read_manifest(root / "manifest.csv")
    assert next(r for r in rows if r["status"] == "normal")["label"] == "labels/n_plate.txt"
    ok_rows = [r for r in rows if r["status"] == "ok"]
    assert all(r["label"] == f"labels/{r['index']:0>6}.txt" for r in ok_rows)
    for r, res in zip(ok_rows, results, strict=True):
        lines = (root / r["label"]).read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(res.instances) == 2
        h, w = res.image.shape[:2]
        for ln, inst in zip(lines, res.instances, strict=True):
            c, cx, cy, bw, bh = ln.split()
            assert int(c) == inst.class_id == classes.index(inst.cls)
            x, y, iw, ih = inst.bbox
            assert abs(float(cx) * w - (x + iw / 2)) < 1e-3 and abs(float(bw) * w - iw) < 1e-3
            assert abs(float(cy) * h - (y + ih / 2)) < 1e-3 and abs(float(bh) * h - ih) < 1e-3
            assert all(0.0 <= float(v) <= 1.0 for v in (cx, cy, bw, bh))
        # 한 이미지에 두 클래스 → 두 줄이 서로 다른 id
        sidecar = json.loads((root / r["sidecar"]).read_text(encoding="utf-8"))
        assert sidecar["writer"] == {
            "format": "yolo",
            "seg": False,
            "label": r["label"],
            "n_lines": 2,
        }


def test_seg_polygons_are_inside_instance_bbox(tmp_path: Path) -> None:
    root, results, _classes, _s = _write(tmp_path, seg=True, n=2)
    for res in results:
        h, w = res.image.shape[:2]
        lines = (root / "labels" / f"{res.index:06d}.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) >= len(res.instances)
        for ln in lines:
            toks = ln.split()
            cid = int(toks[0])
            xs = np.array([float(t) for t in toks[1::2]]) * w
            ys = np.array([float(t) for t in toks[2::2]]) * h

            # 같은 클래스 인스턴스가 둘일 수 있다 — 그중 하나의 bbox 안에 폴리곤이 들어가면 된다
            assert any(_inside(i.bbox, xs, ys) for i in res.instances if i.class_id == cid), ln


def _inside(bbox: tuple[int, int, int, int], xs: np.ndarray, ys: np.ndarray) -> bool:
    x, y, iw, ih = bbox
    return bool(
        xs.min() >= x - 1
        and xs.max() <= x + iw + 1
        and ys.min() >= y - 1
        and ys.max() <= y + ih + 1
    )


@pytest.mark.parametrize("gray", [False, True])
def test_round_trip_import_yolo_rect_boxes_match_sidecar(tmp_path: Path, gray: bool) -> None:
    """출력 images/+labels/를 import-yolo --mask-from rect 로 읽으면 박스 == 사이드카 bbox, 클래스도 그대로."""
    root, results, classes, _s = _write(tmp_path, gray=gray)
    res = import_yolo(
        root / "images",
        root / "labels",
        root / "data.yaml",
        tmp_path / "bank2",
        mask_from="rect",
        margin=6,
    )
    assert res.classes == classes and len(res.normals) == 1  # n_plate = 빈 라벨 → 정상
    bank = Bank.load(tmp_path / "bank2")
    by_origin: dict[str, list] = {}
    for s in bank.sources():
        by_origin.setdefault(s.origin, []).append(s)
    for r in results:
        srcs = by_origin[f"{r.index:06d}.png"]
        assert len(srcs) == 2
        got = sorted((s.cls, tuple(json.loads(_meta(bank, s))["box_in_origin"])) for s in srcs)
        want = sorted((i.cls, tuple(i.bbox)) for i in r.instances)
        assert got == want


def _meta(bank: Bank, s) -> str:
    cls, sid = s.id.split("/", 1)
    return (Path(bank.root) / cls / f"{sid}.json").read_text(encoding="utf-8")


def test_many_classes_same_behaviour(tmp_path: Path) -> None:
    root = tmp_path / "out"
    rec = _recipe(root, count=1)
    classes = [f"c{i:02d}" for i in range(30)]
    w = YoloWriter(rec.output.writer)
    w.begin(root, rec, "h", classes)
    m = np.zeros((32, 32), dtype=np.uint8)
    m[8:16, 8:24] = 255
    inst = Instance("c29", 29, m, (8, 8, 16, 8), 128)
    r = GraftResult(0, "ok", np.zeros((32, 32, 3), np.uint8), m, (inst,), {"defects": []})
    w.write_synthetic(r)
    s = w.finish()
    assert (root / "labels" / "000000.txt").read_text(encoding="utf-8").startswith("29 ")
    assert yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["nc"] == 30
    assert s.n_ok == 1

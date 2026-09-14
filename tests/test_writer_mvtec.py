"""MVTec 형식 writer (설계 §8.2b, v0.4) — 정본 위에 ``mvtec/<category>/{train/good, test/good, test/<class>, ground_truth/<class>}``
를 추가로 · 합성 이미지는 면적 최대 인스턴스 클래스 폴더(섞이면 ``mixed`` + 경고) · 정상 분할은 seed 로 결정적이고 비율을 따른다 ·
사본은 정본과 바이트 동일 · 사이드카 ``writer``·manifest ``label`` · 정상 없으면 경고 · 설정 검증 · CLI run e2e."""

from __future__ import annotations

import json
from pathlib import Path

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
from anograft.io.writers.mvtec import MvtecWriter, image_class
from tests.fixtures import disk_target, fake_yolo_dataset, line_defect, memory_bank, pipeline_deps


def _recipe(root: Path, *, count: int = 3, seed: int = 5, **writer: object) -> R.Recipe:
    return R.Recipe.from_dict(
        {
            "version": 1,
            "name": "m",
            "seed": seed,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {
                "root": str(root),
                "count": count,
                "defects_per_image": [2, 2],
                "writer": {"format": "mvtec", **writer},
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


def _write(tmp_path: Path, *, n: int = 3, n_normals: int = 10, seed: int = 5, **writer: object):
    root = tmp_path / "out"
    rec = _recipe(root, count=n, seed=seed, **writer)
    results, classes = _results(rec, n)
    w = MvtecWriter(rec.output.writer)  # type: ignore[arg-type]
    w.begin(root, rec, "hash", classes, bank_fingerprint="fp")
    for k in range(n_normals):
        p = tmp_path / "n" / f"plate{k}.png"
        imgio.write_image(p, disk_target(64).image)
        w.write_normal(p)
    for r in results:
        w.write_synthetic(r)
    return root, rec, results, w.finish(), w


def _inst(cls: str, cid: int, area: int, k: int) -> Instance:
    m = np.zeros((8, 8), dtype=np.uint8)
    m.flat[:area] = 255
    return Instance(cls=cls, class_id=cid, mask=m, bbox=(0, 0, 8, 8), area_px=area, defect_index=k)


def test_image_class_picks_largest_and_flags_mixed() -> None:
    assert image_class([]) == (None, False)
    assert image_class([_inst("a", 0, 5, 0)]) == ("a", False)
    assert image_class([_inst("a", 0, 5, 0), _inst("b", 1, 9, 1)]) == ("b", True)
    assert image_class([_inst("a", 0, 5, 0), _inst("a", 0, 9, 1)]) == ("a", False)
    # 동률이면 앞선 결함
    assert image_class([_inst("a", 0, 5, 0), _inst("b", 1, 5, 1)]) == ("a", True)


def test_layout_copies_are_byte_identical_and_sidecar_manifest_point_to_them(
    tmp_path: Path,
) -> None:
    root, _rec, results, summary, w = _write(tmp_path)
    cat = root / "mvtec" / "graft"
    assert (cat / "train" / "good").is_dir() and (cat / "test" / "good").is_dir()
    assert (
        summary.files["mvtec"] == "mvtec/graft/" and summary.n_ok == 3 and summary.n_normals == 10
    )
    rows = {r["index"]: r for r in read_manifest(root / "manifest.csv") if r["status"] == "ok"}
    for r in results:
        assert r.status == "ok"
        name = f"{r.index:06d}"
        meta = json.loads((root / "meta" / f"{name}.json").read_text(encoding="utf-8"))
        we = meta["writer"]
        assert we["format"] == "mvtec" and we["split"] == "test" and we["category"] == "graft"
        cls = we["class"]
        assert we["image"] == f"mvtec/graft/test/{cls}/{name}.png"
        assert we["mask"] == f"mvtec/graft/ground_truth/{cls}/{name}_mask.png"
        assert (root / we["image"]).read_bytes() == (root / "images" / f"{name}.png").read_bytes()
        assert (root / we["mask"]).read_bytes() == (root / "masks" / f"{name}.png").read_bytes()
        assert rows[str(r.index)]["label"] == we["image"]
        # 대표 클래스 = 면적 최대 인스턴스
        assert cls == max(r.instances, key=lambda i: i.area_px).cls
        if we["mixed"]:
            assert any("섞임" in x for x in meta["warnings"])
    # 정상: train/test 합이 10, 사본은 정본과 동일, manifest label 이 사본 경로
    train = sorted((cat / "train" / "good").glob("*.png"))
    test = sorted((cat / "test" / "good").glob("*.png"))
    assert (
        len(train) + len(test) == 10 and len(test) == w.n_test_good and len(train) == w.n_train_good
    )
    assert 1 <= len(test) <= 5  # ratio 0.2 · 10장 · 고정 시드 → 대략 2장(결정적)
    for p in train + test:
        assert p.read_bytes() == (root / "images" / p.name).read_bytes()
    normal_rows = [r for r in read_manifest(root / "manifest.csv") if r["status"] == "normal"]
    assert len(normal_rows) == 10
    assert all(
        r["label"].startswith("mvtec/graft/") and r["label"].endswith(".png") for r in normal_rows
    )
    assert all((root / r["label"]).is_file() for r in normal_rows)


def test_normal_split_is_deterministic_by_seed_and_follows_ratio(tmp_path: Path) -> None:
    _, _, _, _, a = _write(tmp_path / "a", n=1, n_normals=40, seed=7, test_normal_ratio=0.5)
    _, _, _, _, b = _write(tmp_path / "b", n=1, n_normals=40, seed=7, test_normal_ratio=0.5)
    assert (a.n_train_good, a.n_test_good) == (b.n_train_good, b.n_test_good)
    assert 10 <= a.n_test_good <= 30
    _write(tmp_path / "c", n=1, n_normals=40, seed=8, test_normal_ratio=0.5)
    names_a = sorted(
        p.name for p in (tmp_path / "a" / "out" / "mvtec" / "graft" / "test" / "good").iterdir()
    )
    names_c = sorted(
        p.name for p in (tmp_path / "c" / "out" / "mvtec" / "graft" / "test" / "good").iterdir()
    )
    assert names_a != names_c  # 시드가 다르면 분할이 다르다
    _, _, _, _, z = _write(tmp_path / "z", n=1, n_normals=10, test_normal_ratio=0.0)
    assert z.n_test_good == 0 and z.n_train_good == 10


def test_warns_when_no_normals_or_no_test_good(tmp_path: Path) -> None:
    _, _, _, summary, _ = _write(tmp_path / "a", n=1, n_normals=0)
    assert any("train/good 이 비었습니다" in w for w in summary.warnings)
    _, _, _, summary, _ = _write(tmp_path / "b", n=1, n_normals=3, test_normal_ratio=0.0)
    assert any("test/good 이 비었습니다" in w for w in summary.warnings)


def test_custom_category_and_layout_dir(tmp_path: Path) -> None:
    root, _, _results, summary, _ = _write(
        tmp_path, n=1, category="metal_nut", layout_dir="anomalib"
    )
    assert summary.files["mvtec"] == "anomalib/metal_nut/"
    meta = json.loads((root / "meta" / "000000.json").read_text(encoding="utf-8"))
    assert meta["writer"]["image"].startswith("anomalib/metal_nut/test/")


def test_config_validation_and_make_writer() -> None:
    cfg = R.MvtecWriterConfig()
    assert cfg.category == "graft" and cfg.test_normal_ratio == 0.2 and cfg.layout_dir == "mvtec"
    with pytest.raises(ValidationError):
        R.MvtecWriterConfig(category="bad name/with slash")
    with pytest.raises(ValidationError):
        R.MvtecWriterConfig(test_normal_ratio=1.5)
    with pytest.raises(ValidationError):
        R.MvtecWriterConfig(seg=True)  # yolo 의 키
    w, warn = make_writer(cfg)
    assert isinstance(w, MvtecWriter) and warn is None


def test_skipped_result_has_no_mvtec_files(tmp_path: Path) -> None:
    root = tmp_path / "out"
    rec = _recipe(root, count=1)
    w = MvtecWriter(rec.output.writer)  # type: ignore[arg-type]
    w.begin(root, rec, "h", ["scratch"])
    bad = GraftResult(
        index=0,
        status="skipped",
        reason="test",
        image=np.zeros((4, 4, 3), np.uint8),
        gt_mask=np.zeros((4, 4), np.uint8),
        instances=(),
        sidecar={"warnings": []},
        warnings=(),
    )
    w.write_synthetic(bad)
    s = w.finish()
    assert s.n_skipped == 1 and not list((root / "mvtec" / "graft" / "test").rglob("*.png"))


def test_cli_run_with_mvtec_writer_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    bank, normals, recipe = tmp_path / "bank", tmp_path / "normals.txt", tmp_path / "r.yaml"
    args = ["bank", "import-yolo", "--images", str(d["images"]), "--labels", str(d["labels"])]
    args += ["--names", str(d["names"]), "--out", str(bank), "--mask-from", "otsu"]
    args += ["--list-normals", str(normals)]
    assert main(args) == EXIT_OK
    args = [
        "recipe",
        "init",
        "--preset",
        "hard-paste",
        "--bank",
        str(bank),
        "--targets",
        str(normals),
    ]
    args += ["--out", str(tmp_path / "out"), "--count", "4", "--seed", "3", "--write", str(recipe)]
    assert main(args) == EXIT_OK
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    data["output"]["writer"] = {"format": "mvtec", "category": "plate", "test_normal_ratio": 0.5}
    data["output"]["defects_per_image"] = [1, 1]
    data["pipeline"]["placement"]["roi"]["erode_px"] = 2
    data["pipeline"]["placement"]["margin_px"] = 4
    data["pipeline"]["source"]["min_sources_warn"] = 1
    recipe.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    capsys.readouterr()
    assert main(["run", str(recipe), "--workers", "0"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "mvtec: mvtec/plate/" in out
    cat = tmp_path / "out" / "mvtec" / "plate"
    assert (cat / "train" / "good").is_dir() and (cat / "ground_truth").is_dir()
    test_imgs = sorted(p for p in (cat / "test").rglob("*.png") if p.parent.name != "good")
    assert test_imgs, "합성 결함이 test/<class>/ 에 있어야 한다"
    for p in test_imgs:
        gt = cat / "ground_truth" / p.parent.name / f"{p.stem}_mask.png"
        assert gt.is_file()
        m, _ = imgio.read_image(gt)
        assert m.max() == 255  # 결함 마스크가 비어 있지 않다
    for p in test_imgs:
        meta = json.loads(
            (tmp_path / "out" / "meta" / f"{p.stem}.json").read_text(encoding="utf-8")
        )
        assert meta["writer"]["mixed"] is False

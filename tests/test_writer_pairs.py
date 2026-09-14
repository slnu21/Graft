"""정본 writer (설계 §8.1·§11 test_writer_pairs) — 레이아웃·세 쌍 불변식·manifest·정상 이미지·skipped 행·사이드카 헤더."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anograft import __version__
from anograft.core import recipe as R
from anograft.core.pipeline import Pipeline
from anograft.core.types import GraftResult
from anograft.io import imgio
from anograft.io.manifest import COLUMNS, read_manifest, row_from_sidecar
from anograft.io.writers import make_writer
from anograft.io.writers.pairs import PairsWriter
from anograft.io.writers.yolo import YoloWriter
from tests.fixtures import disk_target, line_defect, memory_bank, pipeline_deps


def _recipe(root: Path, **output: object) -> R.Recipe:
    out = {
        "root": str(root),
        "count": 3,
        "writer": {"format": "pairs"},
        "defects_per_image": [1, 1],
    }
    out.update(output)
    return R.Recipe.from_dict(
        {
            "version": 1,
            "name": "w",
            "seed": 3,
            "inputs": {"bank": "b", "targets": "t"},
            "output": out,
            "pipeline": {
                "preset": "hard-paste",
                "placement": {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 4},
            },
        }
    )


def _results(rec: R.Recipe, n: int) -> list[GraftResult]:
    bank = memory_bank([line_defect(14, 3)])
    pipe = Pipeline.from_recipe(rec, pipeline_deps(rec, bank))
    return [pipe.run_one(disk_target(96), i) for i in range(n)]


def _normals(tmp_path: Path) -> list[Path]:
    d = tmp_path / "normals"
    d.mkdir()
    a, b = d / "a.png", d / "b.png"
    imgio.write_image(a, disk_target(64).image)
    imgio.write_image(b, disk_target(64, gray=True).image[:, :, 0])  # 흑백 1ch
    (d / "sub").mkdir()
    imgio.write_image(d / "sub" / "a.png", disk_target(48).image)  # stem 충돌
    return [a, b, d / "sub" / "a.png"]


def test_layout_three_files_per_ok_and_manifest(tmp_path: Path) -> None:
    root = tmp_path / "out"
    rec = _recipe(root)
    w = PairsWriter()
    w.begin(root, rec, "deadbeef00000000", ["scratch"], bank_fingerprint="fp")
    results = _results(rec, 3)
    for r in results:
        w.write_synthetic(r)
    s = w.finish()
    assert s.n_ok == 3 and s.n_skipped == 0 and s.per_class == {"scratch": 3}
    for i in range(3):
        name = f"{i:06d}"
        img, mask, meta = (
            root / "images" / f"{name}.png",
            root / "masks" / f"{name}.png",
            root / "meta" / f"{name}.json",
        )
        assert img.is_file() and mask.is_file() and meta.is_file()
        m = imgio.read_mask(mask)
        assert set(np.unique(m)) <= {0, 255} and np.array_equal(m, results[i].gt_mask)
        sc = json.loads(meta.read_text(encoding="utf-8"))
        assert list(sc)[:2] == ["anograft", "pipeline_hash"] and list(sc)[-2:] == [
            "writer",
            "warnings",
        ]
        assert sc["anograft"] == __version__ and sc["pipeline_hash"] == "deadbeef00000000"
        assert sc["writer"] == {"format": "pairs"} and sc["index"] == i
    rows = read_manifest(root / "manifest.csv")
    assert list(rows[0]) == list(COLUMNS) and len(rows) == 3
    assert rows[0] == {
        **rows[0],
        "index": "0",
        "status": "ok",
        "image": "images/000000.png",
        "mask": "masks/000000.png",
        "sidecar": "meta/000000.json",
        "classes": "scratch",
        "n_defects": "1",
        "blend": "paste",
        "fallback": "0",
    }
    assert rows[0]["source_ids"].startswith("scratch/") and int(rows[0]["area_px"]) > 0
    text = (root / "recipe.resolved.yaml").read_text(encoding="utf-8")
    assert text.startswith(
        f"# anograft {__version__} pipeline_hash=deadbeef00000000 bank_fingerprint=fp"
    )
    assert (
        R.Recipe.from_yaml(text).to_yaml() == rec.to_yaml()
    )  # 헤더 주석은 파서가 무시 → 라운드트립
    assert s.files == {"recipe": "recipe.resolved.yaml", "manifest": "manifest.csv"}


def test_normals_written_with_empty_mask_and_unique_names(tmp_path: Path) -> None:
    root = tmp_path / "out"
    rec = _recipe(root)
    w = PairsWriter()
    w.begin(root, rec, "h", ["scratch"])
    for p in _normals(tmp_path):
        w.write_normal(p)
    s = w.finish()
    assert s.n_normals == 3
    names = sorted(p.name for p in (root / "images").iterdir())
    assert names == ["n_a-2.png", "n_a.png", "n_b.png"]
    for n in names:
        stem = n[:-4]
        img, _gray = imgio.read_image(root / "images" / n)
        m = imgio.read_mask(root / "masks" / n)
        assert m.shape == img.shape[:2] and not m.any()
        sc = json.loads((root / "meta" / f"{stem}.json").read_text(encoding="utf-8"))
        assert sc["normal"] is True and sc["writer"]["format"] == "pairs"
    # 흑백 원본은 흑백으로 복사됐다
    assert imgio.read_image(root / "images" / "n_b.png")[1] is True
    rows = read_manifest(root / "manifest.csv")
    assert all(r["status"] == "normal" and r["index"] == "" and r["n_defects"] == "0" for r in rows)


def test_hardlink_mode_links_or_falls_back(tmp_path: Path) -> None:
    root = tmp_path / "out"
    rec = _recipe(root, copy_mode="hardlink")
    w = PairsWriter()
    w.begin(root, rec, "h", ["scratch"])
    src = _normals(tmp_path)[0]
    w.write_normal(src)
    sc = json.loads((root / "meta" / "n_a.json").read_text(encoding="utf-8"))
    assert sc["writer"]["copy_mode"] == "hardlink"
    out = root / "images" / "n_a.png"
    assert out.read_bytes() == src.read_bytes()
    if sc["writer"]["hardlink"]:
        assert out.stat().st_ino == src.stat().st_ino
    else:
        assert w.summary is not None and w.summary.warnings


def test_skipped_result_leaves_no_files_but_a_row(tmp_path: Path) -> None:
    root = tmp_path / "out"
    rec = _recipe(root)
    w = PairsWriter()
    w.begin(root, rec, "h", ["scratch"])
    ok = _results(rec, 1)[0]
    skipped = GraftResult(
        index=7,
        status="skipped",
        image=ok.image,
        gt_mask=ok.gt_mask,
        instances=(),
        sidecar={**ok.sidecar, "index": 7, "defects": []},
        reason="배치 실패",
    )
    w.write_synthetic(ok)
    w.write_synthetic(skipped)
    s = w.finish()
    assert s.n_ok == 1 and s.n_skipped == 1
    assert not (root / "images" / "000007.png").exists()
    rows = read_manifest(root / "manifest.csv")
    assert (
        rows[1]["status"] == "skipped"
        and rows[1]["reason"] == "배치 실패"
        and rows[1]["image"] == ""
    )


def test_row_from_sidecar_joins_multi_defects() -> None:
    sc = {
        "target": {"file": "t.png"},
        "gtmask": {"area_px_total": 30},
        "defects": [
            {
                "source": {"class": "a", "source_id": "a/1"},
                "blend": {"method": "poisson", "fallback": True},
                "gt": {},
            },
            {"source": {"class": "b", "source_id": "b/2"}, "blend": {"method": "alpha"}, "gt": {}},
            {"source": {"class": "c", "source_id": "c/3"}},  # gt 없음 = 실패한 결함 → 제외
        ],
    }
    row = row_from_sidecar(4, "ok", sc, image="i", mask="m", meta="j")
    assert row["classes"] == "a;b" and row["source_ids"] == "a/1;b/2" and row["n_defects"] == 2
    assert row["blend"] == "alpha;poisson" and row["fallback"] == 1 and row["area_px"] == 30


def test_make_writer_picks_format_without_warning() -> None:
    w, warn = make_writer(R.PairsWriterConfig())
    assert type(w) is PairsWriter and warn is None
    w, warn = make_writer(R.YoloWriterConfig(seg=True))
    assert isinstance(w, YoloWriter) and w.format == "yolo" and w.cfg.seg and warn is None


@pytest.mark.parametrize("gray", [False, True])
def test_gray_results_round_trip(tmp_path: Path, gray: bool) -> None:
    root = tmp_path / "out"
    rec = _recipe(root, count=1)
    bank = memory_bank([line_defect(14, 3)])
    pipe = Pipeline.from_recipe(rec, pipeline_deps(rec, bank))
    r = pipe.run_one(disk_target(96, gray=gray), 0)
    w = PairsWriter()
    w.begin(root, rec, "h", ["scratch"])
    w.write_synthetic(r)
    w.finish()
    img, is_gray = imgio.read_image(root / "images" / "000000.png")
    assert is_gray is gray
    assert np.array_equal(img[:, :, 0] if gray else img, r.image)

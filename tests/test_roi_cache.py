"""v0.7.1 ``RoiCache`` — 대상당 ROI 1회: 키(경로·크기·gray·ROI 설정) · LRU · 캐시 유무 결과 동일(rng 소비 0) · 같은 대상 두 번째부터
ROI 스테이지 호출 없음 · 설정이 바뀌면 다시 품 · runner.prepare/reprepare 가 켠다 · 스튜디오 변형 k 개가 공유."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anograft import runner
from anograft.core import recipe as R
from anograft.core.pipeline import Pipeline, RoiCache
from anograft.core.types import TargetImage
from anograft.studio.jobs import KIND_PREVIEW, PreviewJob, run_preview
from tests.fixtures import (
    disk_image,
    disk_target,
    fake_yolo_dataset,
    line_defect,
    memory_bank,
    pipeline_deps,
)


def _pipe(roi: dict | None = None) -> Pipeline:
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": "t",
            "seed": 3,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1, "defects_per_image": [2, 2]},
            "pipeline": {
                "preset": "hard-paste",
                "placement": {"roi": roi or {"method": "otsu", "erode_px": 2}, "margin_px": 6},
            },
        }
    )
    bank = memory_bank([line_defect(14, 3), line_defect(8, 5, cls="dent")])
    return Pipeline.from_recipe(rec, pipeline_deps(rec, bank))


def test_key_and_lru() -> None:
    t = disk_target(64, name="a.png")
    cfg = R.OtsuRoiConfig()
    k1 = RoiCache.key_for(t, cfg)
    assert k1 == RoiCache.key_for(disk_target(64, name="a.png"), R.OtsuRoiConfig())
    assert k1 != RoiCache.key_for(disk_target(64, name="b.png"), cfg)
    assert k1 != RoiCache.key_for(disk_target(96, name="a.png"), cfg)
    assert k1 != RoiCache.key_for(t, R.OtsuRoiConfig(erode_px=3))
    c = RoiCache(max_items=2)
    assert c.get(k1) is None and c.misses == 1
    c.put(k1, np.ones((2, 2), np.uint8), {"method": "otsu"}, ("roi: x",))
    hit = c.get(k1)
    assert (
        hit is not None and hit[1] == {"method": "otsu"} and hit[2] == ("roi: x",) and c.hits == 1
    )
    c.put(("k2",), None, {}, ())
    c.put(("k3",), None, {}, ())
    assert len(c) == 2 and c.get(k1) is None  # LRU 로 k1 이 밀려남
    c.clear()
    assert len(c) == 0


def test_cached_run_is_identical_and_skips_roi_stage(monkeypatch) -> None:
    plain, cached = _pipe(), _pipe()
    cached.roi_cache = RoiCache()
    calls = {"n": 0}
    real_apply = cached.stages["roi"].apply

    def counting(ctx):
        calls["n"] += 1
        return real_apply(ctx)

    monkeypatch.setattr(cached.stages["roi"], "apply", counting)
    t = disk_target(96, name="p.png")
    for k in range(3):
        a = plain.run_one(t, k)
        b = cached.run_one(t, k)
        assert a.status == b.status == "ok"
        assert np.array_equal(a.image, b.image) and np.array_equal(a.gt_mask, b.gt_mask)
        assert a.sidecar == b.sidecar  # roi 로그 포함, 캐시 표시 없음
    assert calls["n"] == 1 and cached.roi_cache.hits == 2 and cached.roi_cache.misses == 1
    # 다른 대상·같은 이름이라도 크기가 다르면 다시 푼다
    cached.run_one(disk_target(64, name="p.png"), 0)
    assert calls["n"] == 2
    # 경고도 재사용된다: mask_dir 로 실패하는 ROI
    bad = _pipe({"method": "mask_dir", "path": "nope-dir"})
    bad.roi_cache = RoiCache()
    r1 = bad.run_one(t, 0)
    r2 = bad.run_one(t, 1)
    assert r1.status == r2.status == "skipped" and bad.roi_cache.hits == 1
    assert any(w.startswith("roi:") for w in r2.warnings)


def test_prepare_enables_cache_and_studio_variants_share_it(tmp_path: Path) -> None:
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
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": "s",
            "seed": 1,
            "inputs": {"bank": (tmp_path / "bank").as_posix(), "targets": normals.as_posix()},
            "output": {"root": (tmp_path / "o").as_posix(), "count": 2},
            "pipeline": {
                "preset": "hard-paste",
                "source": {"method": "bank", "min_sources_warn": 1},
                "placement": {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 4},
            },
        }
    )
    prep = runner.prepare(rec)
    assert isinstance(prep.pipeline.roi_cache, RoiCache)
    target = prep.targets[0]
    results = [
        run_preview(prep, PreviewJob(KIND_PREVIEW, 1, target=target, index=k, long_side=64))
        for k in range(4)
    ]
    assert prep.pipeline.roi_cache.misses == 1 and prep.pipeline.roi_cache.hits == 3
    assert len({r.result.sidecar["roi"].get("method") for r in results}) == 1
    # reprepare(카드 편집)는 캐시를 이어 받는다 — blend 를 바꿔도 같은 대상의 ROI 는 다시 풀지 않는다
    prep2 = runner.reprepare(prep, rec.with_method("blend", "alpha"))
    assert prep2.pipeline.roi_cache is prep.pipeline.roi_cache
    run_preview(prep2, PreviewJob(KIND_PREVIEW, 2, target=target, index=0, long_side=64))
    assert prep2.pipeline.roi_cache.hits == 4 and prep2.pipeline.roi_cache.misses == 1
    # ROI 설정이 바뀌면 키가 달라 다시 푼다
    prep3 = runner.reprepare(prep2, rec.with_method("roi", "none"))
    run_preview(prep3, PreviewJob(KIND_PREVIEW, 3, target=target, index=0, long_side=64))
    assert prep3.pipeline.roi_cache.misses == 2
    # 정상 목록 파일과 무관하게 TargetImage 키는 경로 문자열 — 같은 파일을 다른 표기로 넘겨도 별개 키(보수적)
    t1 = TargetImage(path=Path("x/../a.png"), image=disk_image(32), gray=False)
    t2 = TargetImage(path=Path("a.png"), image=disk_image(32), gray=False)
    assert RoiCache.key_for(t1, R.OtsuRoiConfig()) != RoiCache.key_for(t2, R.OtsuRoiConfig())


def test_prepare_roi_cache_size_option(tmp_path: Path) -> None:
    """``prepare(roi_cache_size=)``(v0.8.x, CLI ``run --roi-cache``): 크기 지정 · 0 = 캐시 없음(결과 동일) · reprepare 는 None 도 이어 받음."""
    from anograft.bank.importers import yolo as Y
    from anograft.cli import EXIT_OK, main

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
        "name": "s",
        "seed": 1,
        "inputs": {"bank": (tmp_path / "bank").as_posix(), "targets": normals.as_posix()},
        "output": {"root": (tmp_path / "o").as_posix(), "count": 2},
        "pipeline": {
            "preset": "hard-paste",
            "source": {"method": "bank", "min_sources_warn": 1},
            "placement": {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 4},
        },
    }
    rec = R.Recipe.from_dict(data)
    small = runner.prepare(rec, roi_cache_size=2)
    assert (
        isinstance(small.pipeline.roi_cache, RoiCache) and small.pipeline.roi_cache.max_items == 2
    )
    off = runner.prepare(rec, roi_cache_size=0)
    assert off.pipeline.roi_cache is None
    assert runner.reprepare(off, rec.with_method("blend", "alpha")).pipeline.roi_cache is None
    a = runner.run_index(runner.prepare(rec), 0)
    b = runner.run_index(off, 0)
    assert (a.image == b.image).all() and (a.gt_mask == b.gt_mask).all()  # 캐시 유무와 결과 무관
    # CLI: --roi-cache 0 · 2 로 돌아가고 결과 파일이 있다
    import yaml

    recipe = tmp_path / "r.yaml"
    recipe.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert main(["run", str(recipe), "--workers", "0", "--roi-cache", "0"]) == EXIT_OK
    assert (
        main(
            [
                "run",
                str(recipe),
                "--workers",
                "2",
                "--roi-cache",
                "2",
                "--out",
                str(tmp_path / "o2"),
            ]
        )
        == EXIT_OK
    )
    assert (tmp_path / "o2" / "manifest.csv").is_file()

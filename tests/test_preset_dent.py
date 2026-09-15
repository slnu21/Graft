"""v0.6 ``preset-rotate``(KNOWN-ISSUES #5) — 조명 의존 결함용 프리셋 ``dent-graft``: 회전 ±15 · flip 끔 · structure-aware
uniform/along · poisson NORMAL · 조화 0.2. 기존 프리셋의 ±180 은 그대로(골든·게시 레시피 재현성)."""

from __future__ import annotations

import numpy as np

from anograft.core import recipe as R
from anograft.core.pipeline import Pipeline
from tests.fixtures import disk_target, line_defect, memory_bank, pipeline_deps


def _recipe(preset: str, seed: int = 5, **placement: object) -> R.Recipe:
    pl: dict = {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 6}
    pl.update(placement)
    return R.Recipe.from_dict(
        {
            "version": 1,
            "name": preset,
            "seed": seed,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1, "defects_per_image": [2, 2]},
            "pipeline": {"preset": preset, "placement": pl},
        }
    )


def _run(rec: R.Recipe, seed: int = 0):
    bank = memory_bank([line_defect(18, 4), line_defect(10, 6, cls="dent")])
    return Pipeline.from_recipe(rec, pipeline_deps(rec, bank)).run_one(disk_target(128), seed)


def test_dent_graft_preset_values_and_others_unchanged() -> None:
    assert "dent-graft" in R.preset_names()
    d = _recipe("dent-graft")
    g, p = d.pipeline.geometry, d.pipeline.placement
    assert g.rotate == (-15.0, 15.0) and g.flip is False and g.scale == (0.9, 1.1)
    assert p.method == "structure-aware" and p.prefer == "uniform" and p.align == "along"
    assert p.jitter_deg == 5 and p.max_align_deg == 30.0
    assert d.pipeline.blend.method == "poisson" and d.pipeline.blend.poisson_mode == "normal"
    assert d.pipeline.harmonize.strength == 0.2
    # 기존 프리셋은 ±180 · flip 유지 (재현성)
    for name in ("poisson-graft", "hard-paste", "structure-aware-graft", "annulus-graft"):
        g2 = _recipe(name).pipeline.geometry
        assert g2.rotate == (-180.0, 180.0) and g2.flip is True, name


def test_dent_graft_runs_and_rotation_stays_in_range() -> None:
    angles: list[float] = []
    for seed in range(6):
        r = _run(_recipe("dent-graft", align="none"), seed)  # 정렬을 끄면 회전 = geometry 만
        assert r.status == "ok", r.warnings
        for d in r.sidecar["defects"]:
            g = d["geometry"]
            assert g["flip"] == [False, False]
            angles.append(float(g["rotate"]))
    assert all(-15.0 <= a <= 15.0 for a in angles) and max(angles) - min(angles) > 1.0
    # 정렬을 켜면 정렬된 자리의 각도는 구조 방향 ± jitter_deg(5) — 사이드카에 aligned 가 기록된다
    r = _run(_recipe("dent-graft"), 1)
    assert r.status == "ok"
    for d in r.sidecar["defects"]:
        pl = d["placement"]
        assert "aligned" in pl and "angle_deg" in pl
        if pl["aligned"]:
            diff = abs(((pl["angle_deg"] - pl["orientation_deg"]) + 90) % 180 - 90)
            assert diff <= 5.0 + 1e-6
            assert abs(pl["angle_deg"]) <= 30.0 + 5.0 + 1e-6  # max_align_deg 30 + jitter 5
        else:
            assert pl["angle_deg"] == 0.0 and abs(pl.get("align_capped", 0.0)) <= 90.0


def test_dent_graft_is_deterministic() -> None:
    a = _run(_recipe("dent-graft"), 2)
    b = _run(_recipe("dent-graft"), 2)
    assert a.status == b.status == "ok" and np.array_equal(a.image, b.image)

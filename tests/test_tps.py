"""``core/tps.py`` — thin-plate spline 워프(numpy): 항등·평행이동·마스크 이진·rng 계약(jitter 0 = 0회)·geometry 스테이지 연결."""

from __future__ import annotations

import numpy as np

from anograft.core import recipe as R
from anograft.core import tps
from anograft.core.pipeline import Pipeline
from tests.fixtures import disk_target, line_defect, memory_bank, pipeline_deps


def test_solve_and_apply_interpolate_control_points() -> None:
    src = tps.control_grid((40, 60), 3)
    rng = np.random.default_rng(0)
    dst = src + rng.normal(0, 2.0, src.shape)
    coef = tps.tps_solve(src, dst)
    back = tps.tps_apply(coef, src, src)
    assert np.allclose(back, dst, atol=1e-6)  # 제어점은 정확히 지난다
    # 순수 평행이동은 어디서나 같은 이동
    coef_t = tps.tps_solve(src, src + np.array([3.0, -2.0]))
    pts = np.array([[10.0, 10.0], [33.3, 7.1]])
    assert np.allclose(tps.tps_apply(coef_t, src, pts), pts + np.array([3.0, -2.0]), atol=1e-6)


def test_maps_identity_and_translation() -> None:
    ctrl = tps.control_grid((32, 48), 3)
    mx, my = tps.tps_maps((32, 48), ctrl, ctrl)
    xs, ys = np.meshgrid(np.arange(48, dtype=np.float32), np.arange(32, dtype=np.float32))
    assert np.allclose(mx, xs, atol=1e-4) and np.allclose(my, ys, atol=1e-4)
    mx2, my2 = tps.tps_maps(
        (32, 48), ctrl, ctrl + np.array([2.0, 0.0])
    )  # 출력이 +2 로 밀림 → 입력은 −2
    assert np.allclose(mx2, xs - 2.0, atol=1e-4) and np.allclose(my2, ys, atol=1e-4)


def test_deform_keeps_mask_binary_and_area_and_is_off_by_default() -> None:
    src = line_defect(30, 8, margin=6)
    rng = np.random.default_rng(3)
    img, mask, ctrl, moved = tps.tps_deform(src.image, src.mask, points=3, jitter=0.08, rng=rng)
    assert set(np.unique(mask)) <= {0, 255} and img.shape[:2] == mask.shape
    a0, a1 = np.count_nonzero(src.mask), np.count_nonzero(mask)
    assert 0.6 * a0 <= a1 <= 1.6 * a0  # 휘어져도 면적은 같은 자릿수
    assert ctrl.shape == (9, 2) and moved.shape == (9, 2) and not np.allclose(ctrl, moved)
    assert tps.tps_pad(mask.shape[:2], 0.08) >= 1 and tps.tps_pad(mask.shape[:2], 0.0) == 0
    # jitter 0 → 원본 그대로 + rng 0회
    rng0 = np.random.default_rng(3)
    before = rng0.bit_generator.state
    img0, mask0, c0, _m0 = tps.tps_deform(src.image, src.mask, points=3, jitter=0.0, rng=rng0)
    assert (
        rng0.bit_generator.state == before
        and img0 is src.image
        and mask0 is src.mask
        and len(c0) == 0
    )
    # 결정성
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    a = tps.tps_deform(src.image, src.mask, points=4, jitter=0.05, rng=rng_a)
    b = tps.tps_deform(src.image, src.mask, points=4, jitter=0.05, rng=rng_b)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def _recipe(jitter: float) -> R.Recipe:
    return R.Recipe.from_dict(
        {
            "version": 1,
            "name": "hard-paste",
            "seed": 2,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1, "defects_per_image": [1, 1]},
            "pipeline": {
                "preset": "hard-paste",
                "geometry": {"tps": {"points": 3, "jitter": jitter}},
                "placement": {"roi": {"method": "none"}, "margin_px": 4},
            },
        }
    )


def test_geometry_stage_logs_tps_and_default_is_unchanged() -> None:
    assert R.TpsConfig().jitter == 0.0 and R.TpsConfig().points == 3
    bank = memory_bank([line_defect(18, 4), line_defect(10, 6, cls="dent")])
    target = disk_target(128)
    off = Pipeline.from_recipe(_recipe(0.0), pipeline_deps(_recipe(0.0), bank)).run_one(target, 0)
    on = Pipeline.from_recipe(_recipe(0.06), pipeline_deps(_recipe(0.06), bank)).run_one(target, 0)
    assert off.status == on.status == "ok"
    g_off, g_on = off.sidecar["defects"][0]["geometry"], on.sidecar["defects"][0]["geometry"]
    assert g_off["tps"] is None and g_on["tps"]["points"] == 3 and g_on["tps"]["max_shift_px"] > 0
    assert g_off["scale"] == g_on["scale"] and g_off["rotate"] == g_on["rotate"]  # 앞 단계 rng 동일
    assert not np.array_equal(off.gt_mask, on.gt_mask) or not np.array_equal(off.image, on.image)
    # 카드 폼(스키마 → UI): 필드가 스펙에 잡힌다
    from anograft.gui.studio.params import field_specs

    names = {f.name for f in field_specs(R.AffineGeometryConfig())}
    assert {"tps.points", "tps.jitter"} <= names

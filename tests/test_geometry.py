"""2단계 geometry(affine) — 설계 §11: 90° 회전 패턴 일치 · flip · 마스크 값 집합 ⊆ {0,255} · 회전 후 마스크가 캔버스 안에
온전히 · elastic alpha=0이면 항등. 추가로 물리 축척·결정성·면적 미달 skip·로그 키."""

from __future__ import annotations

import numpy as np
import pytest

from anograft.core.recipe import AffineGeometryConfig
from anograft.core.stages.geometry import (
    MIN_MASK_AREA,
    AffineGeometry,
    elastic_deform,
    elastic_pad,
    rotated_canvas_size,
    warp_affine,
)
from tests.fixtures import context, disk_target, line_defect


def _rand_patch(h: int = 7, w: int = 11, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[2:5, 3:8] = 255
    return img, mask


# ---------------------------------------------------------------------------
# 순수 함수
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(7, 11), (8, 12)])
@pytest.mark.parametrize(("angle", "k"), [(0, 0), (90, 1), (180, 2), (270, 3), (-90, -1)])
def test_right_angle_rotation_matches_rot90(shape: tuple[int, int], angle: int, k: int) -> None:
    img, mask = _rand_patch(*shape)
    patch, out = warp_affine(img, mask, 1.0, angle)
    assert np.array_equal(patch, np.rot90(img, k))
    assert np.array_equal(out, np.rot90(mask, k))


def test_flip_h_v() -> None:
    img, mask = _rand_patch()
    p, m = warp_affine(img, mask, 1.0, 0, flip_h=True)
    assert np.array_equal(p, img[:, ::-1]) and np.array_equal(m, mask[:, ::-1])
    p, m = warp_affine(img, mask, 1.0, 0, flip_v=True)
    assert np.array_equal(p, img[::-1]) and np.array_equal(m, mask[::-1])
    p, m = warp_affine(img, mask, 1.0, 0, flip_h=True, flip_v=True)
    assert np.array_equal(p, img[::-1, ::-1])


def test_scale_doubles_mask_area() -> None:
    img, mask = _rand_patch()
    p, m = warp_affine(img, mask, 2.0, 0)
    assert p.shape == (14, 22, 3)
    assert np.count_nonzero(m) == 4 * np.count_nonzero(mask)


@pytest.mark.parametrize("angle", [17.0, 33.3, 61.0, 120.0, -45.0])
def test_rotation_keeps_mask_binary_and_inside_canvas(angle: float) -> None:
    img, mask = _rand_patch()
    p, m = warp_affine(img, mask, 1.0, angle)
    assert m.dtype == np.uint8 and set(np.unique(m).tolist()) <= {0, 255}
    assert p.shape[:2] == m.shape == rotated_canvas_size(11, 7, 1.0, angle)[::-1]
    # 마스크는 원본 크롭 안쪽에 있으므로 회전 bbox 캔버스의 테두리에 닿지 않는다
    assert not (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
    # 면적은 대략 보존 (NEAREST 리샘플 오차 허용)
    assert abs(np.count_nonzero(m) - np.count_nonzero(mask)) <= 4


def test_rotated_canvas_size_no_float_noise() -> None:
    assert rotated_canvas_size(11, 7, 1.0, 90) == (7, 11)
    assert rotated_canvas_size(11, 7, 1.0, 0) == (11, 7)
    assert rotated_canvas_size(10, 10, 1.0, 45) == (15, 15)


def test_elastic_alpha_zero_is_identity() -> None:
    img, mask = _rand_patch()
    p, m = elastic_deform(img, mask, 0.0, 4.0, np.random.default_rng(1))
    assert np.array_equal(p, img) and np.array_equal(m, mask)
    assert elastic_pad(0.0, 4.0) == 0


def test_elastic_moves_pixels_but_keeps_mask_binary() -> None:
    img, mask = _rand_patch(20, 30)
    mask[:] = 0
    mask[6:14, 8:22] = 255
    rng = np.random.default_rng(3)
    p, m = elastic_deform(img, mask, 30.0, 4.0, rng)
    pad = elastic_pad(30.0, 4.0)
    assert p.shape[:2] == m.shape == (20 + 2 * pad, 30 + 2 * pad)
    assert set(np.unique(m).tolist()) <= {0, 255}
    assert not np.array_equal(m[pad:-pad, pad:-pad], mask)
    assert abs(np.count_nonzero(m) - np.count_nonzero(mask)) < 0.3 * np.count_nonzero(mask)


# ---------------------------------------------------------------------------
# 스테이지
# ---------------------------------------------------------------------------


def _stage(**kw) -> AffineGeometry:
    return AffineGeometry(AffineGeometryConfig(**kw), {})


def test_stage_fills_patch_and_logs() -> None:
    ctx = context(source=line_defect(), seed=7)
    out = _stage().apply(ctx)
    assert out.patch is not None and out.patch_mask is not None
    assert out.patch.shape[:2] == out.patch_mask.shape
    assert set(np.unique(out.patch_mask).tolist()) <= {0, 255}
    log = out.log["geometry"]
    assert log["method"] == "affine" and "failed" not in log
    assert 0.8 <= log["scale"] <= 1.25 and -180 <= log["rotate"] <= 180
    assert log["physical_applied"] is False and log["elastic"] is None
    assert log["mask_area_px"] == np.count_nonzero(out.patch_mask)
    # 입력 Context는 그대로
    assert ctx.patch is None


def test_stage_is_deterministic_and_seed_varies() -> None:
    a = _stage().apply(context(source=line_defect(), seed=1))
    b = _stage().apply(context(source=line_defect(), seed=1))
    c = _stage().apply(context(source=line_defect(), seed=2))
    assert np.array_equal(a.patch, b.patch) and a.log == b.log
    assert a.log["geometry"]["rotate"] != c.log["geometry"]["rotate"]


def test_stage_without_flip_and_fixed_ranges() -> None:
    st = _stage(scale=(1.0, 1.0), rotate=(0.0, 0.0), flip=False)
    src = line_defect()
    out = st.apply(context(source=src, seed=5))
    assert np.array_equal(out.patch, src.image) and np.array_equal(out.patch_mask, src.mask)
    assert out.log["geometry"]["flip"] == [False, False]


def test_stage_applies_physical_scale() -> None:
    st = _stage(scale=(1.0, 1.0), rotate=(0.0, 0.0), flip=False)
    src = line_defect(um_per_px=10.0)
    out = st.apply(context(disk_target(um_per_px=5.0), source=src))
    log = out.log["geometry"]
    assert log["physical_scale"] == 2.0 and log["physical_applied"] is True and log["scale"] == 2.0
    assert out.patch_mask.shape == (src.mask.shape[0] * 2, src.mask.shape[1] * 2)


def test_stage_elastic_logged_when_enabled() -> None:
    st = _stage(elastic={"alpha": 20.0, "sigma": 3.0})
    out = st.apply(context(source=line_defect(), seed=3))
    assert out.patch is not None
    assert out.log["geometry"]["elastic"] == {"alpha": 20.0, "sigma": 3.0}


def test_stage_skips_tiny_mask_with_warning_and_log() -> None:
    src = line_defect(length=2, width=1)  # 2px 마스크 → 어떤 축척에서도 4px 미만
    st = _stage(scale=(0.5, 0.5), rotate=(0.0, 0.0), flip=False)
    out = st.apply(context(source=src))
    assert out.patch is None and out.patch_mask is None
    assert (
        out.log["geometry"]["failed"] is True
        and f"{MIN_MASK_AREA}px" in out.log["geometry"]["reason"]
    )
    assert any("geometry" in w for w in out.warnings)


def test_stage_without_source_logs_and_passes_through() -> None:
    out = _stage().apply(context())
    assert out.patch is None and out.log["geometry"]["skipped"]

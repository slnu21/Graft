"""``core/perlin.py`` — 펄린 노이즈·DRAEM 마스크: 값 범위 · 결정성 · 해상도(칸 크기) · 임계 커버리지 · 회전 · rng 소비 순서."""

from __future__ import annotations

import numpy as np
import pytest

from anograft.core.perlin import perlin_mask, perlin_noise, rotate_keep


def test_noise_shape_range_and_determinism() -> None:
    a = perlin_noise(np.random.default_rng(1), 64, 96, 4, 8)
    b = perlin_noise(np.random.default_rng(1), 64, 96, 4, 8)
    assert a.shape == (64, 96) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert float(a.min()) >= -1.05 and float(a.max()) <= 1.05  # √2 정규화 후 ≈ [-1, 1]
    assert float(a.std()) > 0.05  # 상수가 아니다
    c = perlin_noise(np.random.default_rng(2), 64, 96, 4, 8)
    assert not np.array_equal(a, c)


def test_noise_is_zero_at_grid_corners_and_smooth() -> None:
    """격자 모서리에서는 거리 벡터가 0이라 값이 0. 칸 안은 연속(인접 픽셀 차이가 작다)."""
    res = 4
    n = perlin_noise(np.random.default_rng(3), 64, 64, res, res)
    cell = 64 // res
    corners = n[::cell, ::cell]
    assert np.abs(corners).max() < 1e-5
    dx = np.abs(np.diff(n, axis=1)).max()
    assert dx < 0.35  # 칸 16px 에서 한 픽셀 변화는 작다


def test_noise_handles_non_divisible_sizes_and_res_1() -> None:
    n = perlin_noise(np.random.default_rng(0), 50, 37, 3, 5)  # 50/3, 37/5 나누어떨어지지 않음
    assert n.shape == (50, 37) and np.isfinite(n).all()
    one = perlin_noise(np.random.default_rng(0), 16, 16, 1, 1)
    assert one.shape == (16, 16)
    with pytest.raises(ValueError):
        perlin_noise(np.random.default_rng(0), 0, 4, 1, 1)


def test_higher_resolution_means_finer_structure() -> None:
    """res 가 클수록 부호가 더 자주 바뀐다(더 잘게 쪼개진 마스크)."""
    lo = perlin_noise(np.random.default_rng(5), 128, 128, 1, 1)
    hi = perlin_noise(np.random.default_rng(5), 128, 128, 16, 16)
    flips_lo = int((np.diff(np.sign(lo), axis=1) != 0).sum())
    flips_hi = int((np.diff(np.sign(hi), axis=1) != 0).sum())
    assert flips_hi > flips_lo * 3


def test_mask_threshold_rotation_and_log() -> None:
    rng = np.random.default_rng(7)
    m, log = perlin_mask(rng, 96, 96, scale_range=(2, 2), threshold=0.5, rotate=(30.0, 30.0))
    assert m.shape == (96, 96) and m.dtype == np.uint8
    assert set(np.unique(m).tolist()) <= {0, 255}
    assert log["res_y"] == 4 and log["res_x"] == 4 and log["rotate"] == pytest.approx(30.0)
    assert log["area_px"] == int(np.count_nonzero(m)) and 0 < log["area_px"] < 96 * 96
    # 임계를 낮추면 면적이 늘고, 아주 높으면 0
    m_lo, _ = perlin_mask(
        np.random.default_rng(7), 96, 96, scale_range=(2, 2), threshold=0.0, rotate=(30.0, 30.0)
    )
    m_hi, _ = perlin_mask(
        np.random.default_rng(7), 96, 96, scale_range=(2, 2), threshold=0.99, rotate=(30.0, 30.0)
    )
    assert np.count_nonzero(m_lo) > log["area_px"] > np.count_nonzero(m_hi)


def test_mask_is_deterministic_and_rng_order_fixed() -> None:
    a, la = perlin_mask(np.random.default_rng(11), 48, 48)
    b, lb = perlin_mask(np.random.default_rng(11), 48, 48)
    assert np.array_equal(a, b) and la == lb
    # 소비 순서: integers(k_y) → integers(k_x) → random → uniform — 직접 재현하면 같은 마스크
    rng = np.random.default_rng(11)
    k_y, k_x = int(rng.integers(0, 6)), int(rng.integers(0, 6))
    noise = perlin_noise(rng, 48, 48, 2**k_y, 2**k_x)
    angle = float(rng.uniform(-90.0, 90.0))
    manual = rotate_keep((noise > 0.5).astype(np.uint8) * 255, angle)
    assert np.array_equal(manual, a)


def test_rotate_keep_preserves_shape_and_is_binary() -> None:
    m = np.zeros((40, 60), dtype=np.uint8)
    m[10:20, 5:50] = 255
    r = rotate_keep(m, 45.0)
    assert r.shape == m.shape and set(np.unique(r).tolist()) <= {0, 255}
    assert rotate_keep(m, 0.0) is m
    assert 0 < np.count_nonzero(r) <= np.count_nonzero(m)  # 캔버스 밖으로 잘린 만큼만 줄어든다

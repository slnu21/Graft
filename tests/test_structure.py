"""``core/structure.py`` — 그래디언트 크기 · 창 구조 텐서 방향/일관성 · 마스크 주축 · 정렬각 규약.

각도 규약은 **실제로 돌려서** 고정한다: 가로 막대를 ``warp_affine(A)``로 돌리면 주축이 ``-A``(y 아래 좌표계), 줄무늬 이미지의
``theta_g + 90``이 줄 방향, ``align_rotation``으로 돌린 막대의 주축이 줄 방향과 일치.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anograft.core import structure as S
from anograft.core.stages.geometry import warp_affine


def _bar(size: int = 41, length: int = 31, width: int = 5) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.uint8)
    y0, x0 = (size - width) // 2, (size - length) // 2
    m[y0 : y0 + width, x0 : x0 + length] = 255
    return m


def stripes(size: int = 200, angle_deg: float = -30.0, period: int = 12) -> np.ndarray:
    """``angle_deg`` 방향(+x 에서 +y 쪽, y 아래)으로 뻗는 줄무늬 float32 그레이."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    a = math.radians(angle_deg)
    n = -xs * math.sin(a) + ys * math.cos(a)  # 줄에 수직인 좌표
    return np.where((n // (period / 2)).astype(int) % 2 == 0, 200.0, 40.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 크기 맵 · 가중
# ---------------------------------------------------------------------------


def test_gradient_magnitude_peaks_at_edge_and_smoothing_spreads() -> None:
    g = np.zeros((32, 32), dtype=np.float32)
    g[:, 16:] = 200.0
    raw = S.gradient_magnitude(g, 0.0)
    assert raw[10, 15] > 0 and raw[10, 16] > 0 and raw[10, 5] == 0 and raw[10, 28] == 0
    sm = S.gradient_magnitude(g, 2.0)
    assert sm[10, 12] > 0 and sm[10, 12] < raw[10, 16]  # 평활은 번지고 낮아진다


def test_structure_weights_prefer_edges_vs_flat_and_uniform_none() -> None:
    g = np.zeros((32, 32), dtype=np.float32)
    g[:, 16:] = 200.0
    mag = S.gradient_magnitude(g, 0.0)
    allowed = np.ones((32, 32), dtype=bool)
    idx_edge = np.flatnonzero(allowed).tolist().index(10 * 32 + 16)
    idx_flat = np.flatnonzero(allowed).tolist().index(10 * 32 + 5)
    w_e = S.structure_weights(mag, allowed, "edges", 1.0)
    w_f = S.structure_weights(mag, allowed, "flat", 1.0)
    assert w_e is not None and w_f is not None
    assert w_e[idx_edge] > w_e[idx_flat] and w_f[idx_flat] > w_f[idx_edge]
    assert w_e.min() >= S.WEIGHT_EPS and w_f.min() >= S.WEIGHT_EPS  # 누적합 0 방지
    assert S.structure_weights(mag, allowed, "uniform", 1.0) is None
    # strength 0 → 사실상 균등
    w0 = S.structure_weights(mag, allowed, "edges", 0.0)
    assert w0 is not None and np.allclose(w0, w0[0])
    with pytest.raises(ValueError):
        S.structure_weights(mag, allowed, "nope", 1.0)  # type: ignore[arg-type]


def test_structure_weights_flat_image_and_empty_allowed() -> None:
    mag = np.zeros((8, 8), dtype=np.float32)
    w = S.structure_weights(mag, np.ones((8, 8), dtype=bool), "edges", 1.0)
    assert w is not None and np.allclose(w, S.WEIGHT_EPS)
    assert S.structure_weights(mag, np.zeros((8, 8), dtype=bool), "edges", 1.0) is None


# ---------------------------------------------------------------------------
# 방향
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [-30.0, 0.0, 45.0, 80.0])
def test_window_orientation_edge_direction_matches_stripes(angle: float) -> None:
    g = stripes(angle_deg=angle)
    theta_g, coh = S.window_orientation(g, 100, 100, 30)
    edge_dir = S.wrap_180(theta_g + 90.0)
    diff = abs(S.wrap_180(edge_dir - angle))
    assert diff < 3.0, (angle, edge_dir)
    assert coh > 0.8


def test_window_orientation_flat_and_isotropic_have_low_coherence() -> None:
    flat = np.full((64, 64), 100.0, dtype=np.float32)
    assert S.window_orientation(flat, 32, 32, 10) == (0.0, 0.0)
    rng = np.random.default_rng(0)
    noise = rng.random((64, 64)).astype(np.float32) * 200
    _, coh = S.window_orientation(noise, 32, 32, 20)
    assert coh < 0.2
    # 창이 이미지 밖으로 나가도 잘라 쓰고, 너무 작으면 (0, 0)
    assert S.window_orientation(noise, 0, 0, 5)[1] < 0.5
    assert S.window_orientation(noise, 0, 0, 0) == (0.0, 0.0)


def test_mask_principal_axis_and_rotation_convention() -> None:
    bar = _bar()
    phi, aniso = S.mask_principal_axis(bar)
    assert abs(phi) < 0.5 and aniso > 0.8
    img = np.zeros((*bar.shape, 3), dtype=np.uint8)
    for a in (30.0, -30.0, 60.0):
        _, rotated = warp_affine(img, bar, 1.0, a)
        phi_r, _ = S.mask_principal_axis(rotated)
        assert abs(S.wrap_180(phi_r + a)) < 1.5, (
            a,
            phi_r,
        )  # 화면 반시계 회전 = 이 좌표계에서 각도 감소
    # 등방(원판)·빈 마스크
    yy, xx = np.mgrid[0:41, 0:41]
    disk = (((yy - 20) ** 2 + (xx - 20) ** 2) <= 15**2).astype(np.uint8) * 255
    assert S.mask_principal_axis(disk)[1] < 0.02
    assert S.mask_principal_axis(np.zeros((5, 5), dtype=np.uint8)) == (0.0, 0.0)


def test_align_rotation_brings_bar_onto_stripe_direction() -> None:
    bar = _bar()
    img = np.zeros((*bar.shape, 3), dtype=np.uint8)
    for stripe_angle in (-30.0, 20.0, 75.0):
        g = stripes(angle_deg=stripe_angle)
        theta_g, _ = S.window_orientation(g, 100, 100, 30)
        target = S.wrap_180(theta_g + 90.0)
        phi, _ = S.mask_principal_axis(bar)
        _, rotated = warp_affine(img, bar, 1.0, S.align_rotation(phi, target))
        phi_r, _ = S.mask_principal_axis(rotated)
        assert abs(S.wrap_180(phi_r - stripe_angle)) < 3.0, (stripe_angle, phi_r)


def test_wrap_180() -> None:
    assert S.wrap_180(0.0) == 0.0
    assert S.wrap_180(90.0) == 90.0
    assert S.wrap_180(-90.0) == 90.0
    assert S.wrap_180(180.0) == 0.0
    assert S.wrap_180(200.0) == pytest.approx(20.0)
    assert S.wrap_180(-100.0) == pytest.approx(80.0)
    assert S.align_rotation(10.0, -80.0) == pytest.approx(90.0)

"""6단계 degrade — 설계 §11: σ=0이면 항등; 출력 dtype uint8·클립. 추가: 블러→노이즈→JPEG 각각의 효과와 로그 · 결정성 ·
흑백 대상은 세 채널 동일 유지 · rng 소비 순서(노이즈 σ → 블러 σ → q → 노이즈 필드)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from anograft.core import recipe as R
from anograft.core.stages.degrade import (
    CameraDegrade,
    NoneDegrade,
    add_noise,
    gaussian_blur,
    jpeg_roundtrip,
)
from anograft.core.types import TargetImage
from tests.fixtures import context, disk_image


def _ctx(gray: bool = False, seed: int = 0):
    img = disk_image(64)
    t = TargetImage(path=Path("t.png"), image=img, gray=gray)
    return context(t, seed=seed)


def _camera(**kw) -> CameraDegrade:
    return CameraDegrade(R.CameraDegradeConfig(**kw), {})


def test_none_is_identity() -> None:
    ctx = _ctx()
    out = NoneDegrade(R.NoneDegradeConfig(), {}).apply(ctx)
    assert out.composite is ctx.composite and out.log["degrade"] == {"method": "none"}


def test_camera_all_zero_is_identity_with_log() -> None:
    ctx = _ctx()
    out = _camera(noise_sigma=(0, 0), blur_sigma=(0, 0), jpeg_quality=None).apply(ctx)
    assert np.array_equal(out.composite, ctx.composite)
    assert out.log["degrade"] == {
        "method": "camera",
        "noise_sigma": 0.0,
        "blur_sigma": 0.0,
        "blur_applied": False,
        "jpeg_quality": None,
    }


def test_blur_below_threshold_is_skipped() -> None:
    img = disk_image(32)
    assert gaussian_blur(img, 0.05) is img
    assert not np.array_equal(gaussian_blur(img, 1.0), img)


def test_noise_changes_pixels_keeps_uint8_and_clips() -> None:
    img = np.full((16, 16, 3), 250, dtype=np.uint8)
    out = add_noise(img, 30.0, np.random.default_rng(0), gray=False)
    assert out.dtype == np.uint8 and out.max() <= 255 and out.min() < 250
    assert not np.array_equal(out, img)
    assert add_noise(img, 0.0, np.random.default_rng(0), gray=False) is img


def test_noise_on_gray_keeps_channels_equal() -> None:
    img = np.full((16, 16, 3), 120, dtype=np.uint8)
    out = add_noise(img, 5.0, np.random.default_rng(1), gray=True)
    assert np.array_equal(out[:, :, 0], out[:, :, 1]) and np.array_equal(out[:, :, 1], out[:, :, 2])
    color = add_noise(img, 5.0, np.random.default_rng(1), gray=False)
    assert not np.array_equal(color[:, :, 0], color[:, :, 1])


@pytest.mark.parametrize("gray", [False, True])
def test_jpeg_roundtrip_changes_but_stays_close(gray: bool) -> None:
    img = disk_image(64)
    out = jpeg_roundtrip(img, 40, gray)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 8
    if gray:
        assert np.array_equal(out[:, :, 0], out[:, :, 1])


def test_camera_stage_applies_all_and_logs_actual_values() -> None:
    ctx = _ctx()
    st = _camera(noise_sigma=(1.0, 3.0), blur_sigma=(0.5, 1.0), jpeg_quality=(60, 90))
    out = st.apply(ctx)
    log = out.log["degrade"]
    assert 1.0 <= log["noise_sigma"] <= 3.0 and 0.5 <= log["blur_sigma"] <= 1.0
    assert log["blur_applied"] is True and 60 <= log["jpeg_quality"] <= 90
    assert out.composite.dtype == np.uint8 and not np.array_equal(out.composite, ctx.composite)


def test_camera_noise_only_roughens_flat_region() -> None:
    out = _camera(noise_sigma=(2.0, 2.0), blur_sigma=(0, 0), jpeg_quality=None).apply(_ctx())
    assert out.composite[:8, :8].std() > 0  # 원판 밖(균일 40)에 노이즈가 얹혔다
    assert abs(out.composite[:8, :8].astype(float).mean() - 40) < 2


def test_camera_gray_target_keeps_channels_equal() -> None:
    ctx = _ctx(gray=True)
    out = _camera(noise_sigma=(2.0, 2.0), blur_sigma=(0.8, 0.8), jpeg_quality=(70, 70)).apply(ctx)
    c = out.composite
    assert np.array_equal(c[:, :, 0], c[:, :, 1]) and np.array_equal(c[:, :, 1], c[:, :, 2])


def test_camera_is_deterministic_and_seed_varies() -> None:
    st = _camera(noise_sigma=(1.0, 2.0))
    a = st.apply(_ctx(seed=3))
    b = st.apply(_ctx(seed=3))
    c = st.apply(_ctx(seed=4))
    assert np.array_equal(a.composite, b.composite) and a.log == b.log
    assert a.log["degrade"]["noise_sigma"] != c.log["degrade"]["noise_sigma"]


def test_rng_consumption_order() -> None:
    """σ_noise → σ_blur → q → 노이즈 필드. 같은 시드로 손으로 뽑은 값과 로그가 일치해야 한다."""
    ctx = _ctx(seed=9)
    st = _camera(noise_sigma=(0.5, 1.5), blur_sigma=(0.0, 0.05), jpeg_quality=(50, 60))
    out = st.apply(ctx)
    rng = np.random.default_rng(9)
    s_noise = float(rng.uniform(0.5, 1.5))
    s_blur = float(rng.uniform(0.0, 0.05))
    q = int(rng.integers(50, 61))
    assert out.log["degrade"]["noise_sigma"] == s_noise
    assert out.log["degrade"]["blur_sigma"] == s_blur
    assert out.log["degrade"]["jpeg_quality"] == q
    # 남은 스트림 = 노이즈 필드 한 장(컬러 → HxWx3)
    expected = add_noise(ctx.composite, s_noise, rng, gray=False)
    assert np.array_equal(jpeg_roundtrip(expected, q, False), out.composite)


def test_composite_not_target_is_degraded() -> None:
    """degrade는 composite(합성 결과)에 걸린다 — target.image는 그대로."""
    ctx = _ctx()
    comp = ctx.composite.copy()
    comp[10:20, 10:20] = 255
    ctx2 = replace(ctx, composite=comp)
    out = _camera(noise_sigma=(1.0, 1.0), blur_sigma=(0, 0)).apply(ctx2)
    assert np.array_equal(out.target.image, ctx.target.image)
    assert out.composite[10:20, 10:20].mean() > 240

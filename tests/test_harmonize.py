"""5단계 harmonize — 설계 §11: strength 0 = 항등, 마스크 밖 불변; stats: strength 1이면 내부 평균 ≈ 링 평균(L 채널).
추가: 흑백 대상은 세 채널 동일 유지 · 링/내부가 비면 skipped · 로그."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from anograft.core import recipe as R
from anograft.core.stages.harmonize import (
    NoneHarmonize,
    StatsHarmonize,
    dilate_mask,
    match_stats,
    ring_of,
)
from anograft.core.types import Context, Placement, TargetImage
from tests.fixtures import context

H, W = 48, 64


def _scene(gray: bool = False, seed: int = 0) -> tuple[Context, np.ndarray]:
    """대상 = 밝은(180±10) 텍스처, 마스크 안은 어두운(60±10) 텍스처 — 조화 전에는 평균이 크게 다르다."""
    rng = np.random.default_rng(seed)
    base = np.clip(rng.normal(180, 10, (H, W)), 0, 255).astype(np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[16:32, 20:44] = 255
    inside = np.clip(rng.normal(60, 10, (H, W)), 0, 255).astype(np.uint8)
    img = np.where(mask > 0, inside, base)
    if gray:
        image = np.repeat(img[:, :, None], 3, axis=2)
    else:
        image = np.stack([img, np.clip(img.astype(int) + 20, 0, 255), img], axis=2).astype(np.uint8)
    t = TargetImage(path=Path("t.png"), image=image, gray=gray)
    ctx = context(t)
    pl = Placement(center=(32, 24), bbox=(20, 16, 24, 16), offset=(20, 16), tries=1)
    return replace(ctx, placement=pl, placed_mask=mask), mask


def _lum(image: np.ndarray, gray: bool) -> np.ndarray:
    if gray:
        return image[:, :, 0].astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)


def test_ring_and_dilate() -> None:
    m = np.zeros((20, 20), dtype=np.uint8)
    m[8:12, 8:12] = 255
    d = dilate_mask(m, 2)
    assert d.sum() > m.sum() and (d[m > 0] == 255).all()
    ring = ring_of(m, 2)
    assert ring.any() and not ring[m > 0].any() and ring.sum() == (d > 0).sum() - (m > 0).sum()
    assert dilate_mask(m, 0) is m


def test_match_stats_moves_mean_and_std() -> None:
    v = np.array([0.0, 10.0, 20.0])
    r = np.array([100.0, 102.0, 104.0])
    out = match_stats(v, r)
    assert out.mean() == pytest.approx(r.mean()) and out.std() == pytest.approx(r.std())
    flat = match_stats(np.array([5.0, 5.0]), r)
    assert (flat == r.mean()).all()


def test_none_is_identity_with_log() -> None:
    ctx, _ = _scene()
    out = NoneHarmonize(R.NoneHarmonizeConfig(), {}).apply(ctx)
    assert out.composite is ctx.composite and out.log["harmonize"] == {"method": "none"}


@pytest.mark.parametrize("gray", [False, True])
def test_stats_strength_zero_is_exact_identity(gray: bool) -> None:
    ctx, _ = _scene(gray)
    out = StatsHarmonize(R.StatsHarmonizeConfig(strength=0.0), {}).apply(ctx)
    assert np.array_equal(out.composite, ctx.composite)
    assert out.log["harmonize"]["skipped"] == "strength 0"


@pytest.mark.parametrize("gray", [False, True])
def test_stats_strength_one_matches_ring_mean_and_keeps_outside(gray: bool) -> None:
    ctx, mask = _scene(gray)
    st = StatsHarmonize(R.StatsHarmonizeConfig(strength=1.0, ring_px=6), {})
    out = st.apply(ctx)
    inside, ring = mask > 0, ring_of(mask, 6)
    assert np.array_equal(out.composite[~inside], ctx.composite[~inside])
    lum = _lum(out.composite, gray)
    assert abs(lum[inside].mean() - lum[ring].mean()) < 2.0
    assert abs(lum[inside].std() - lum[ring].std()) < 2.0
    log = out.log["harmonize"]
    assert log["method"] == "stats" and log["mean_in"] < log["mean_ring"]
    assert abs(log["mean_after"] - log["mean_ring"]) < 1.0
    if gray:
        c = out.composite
        assert np.array_equal(c[:, :, 0], c[:, :, 1]) and np.array_equal(c[:, :, 1], c[:, :, 2])


def test_stats_half_strength_is_between() -> None:
    ctx, mask = _scene()
    full = StatsHarmonize(R.StatsHarmonizeConfig(strength=1.0), {}).apply(ctx)
    half = StatsHarmonize(R.StatsHarmonizeConfig(strength=0.5), {}).apply(ctx)
    inside = mask > 0
    m0, m1, mh = (
        _lum(ctx.composite, False)[inside].mean(),
        _lum(full.composite, False)[inside].mean(),
        _lum(half.composite, False)[inside].mean(),
    )
    assert m0 < mh < m1 and abs(mh - (m0 + m1) / 2) < 2.0


def test_stats_skips_when_ring_empty_or_no_placement() -> None:
    ctx, _ = _scene()
    full = np.full((H, W), 255, dtype=np.uint8)  # 마스크가 이미지 전체 → 링 없음
    out = StatsHarmonize(R.StatsHarmonizeConfig(strength=1.0), {}).apply(
        replace(ctx, placed_mask=full)
    )
    assert np.array_equal(out.composite, ctx.composite)
    assert "비어" in out.log["harmonize"]["skipped"]

    out = StatsHarmonize(R.StatsHarmonizeConfig(), {}).apply(context())
    assert out.log["harmonize"]["skipped"] == "placement 없음"


def test_stats_is_deterministic() -> None:
    ctx, _ = _scene()
    a = StatsHarmonize(R.StatsHarmonizeConfig(strength=0.7), {}).apply(ctx)
    b = StatsHarmonize(R.StatsHarmonizeConfig(strength=0.7), {}).apply(ctx)
    assert np.array_equal(a.composite, b.composite) and a.log == b.log

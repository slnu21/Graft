"""5단계 harmonize — 설계 §11: **4방법 공통** strength 0 = 항등, 마스크 밖 불변. stats/reinhard: strength 1이면 내부 평균 ≈ 링 평균
(reinhard는 a·b 채널까지); histmatch: 내부 CDF ≈ 링 CDF. 추가: 흑백 대상은 세 채널 동일 유지 · 링/내부가 비면 skipped · 로그 · 흑백이면 reinhard == stats."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from anograft.core import recipe as R
from anograft.core.stages.harmonize import (
    HistmatchHarmonize,
    NoneHarmonize,
    ReinhardHarmonize,
    RelativeHarmonize,
    StatsHarmonize,
    dilate_mask,
    match_hist,
    match_relative,
    match_stats,
    ring_of,
    source_ring_values,
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


# ---------------------------------------------------------------------------
# 4방법 공통 + reinhard · histmatch
# ---------------------------------------------------------------------------

_RING = {
    "stats": lambda **kw: StatsHarmonize(R.StatsHarmonizeConfig(**kw), {}),
    "reinhard": lambda **kw: ReinhardHarmonize(R.ReinhardHarmonizeConfig(**kw), {}),
    "histmatch": lambda **kw: HistmatchHarmonize(R.HistmatchHarmonizeConfig(**kw), {}),
}


def _lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)


def _color_scene(seed: int = 0) -> tuple[Context, np.ndarray]:
    """색조까지 다른 장면: 대상은 붉은 기(B<R), 마스크 안은 푸른 기(B>R) — reinhard/histmatch가 a·b도 옮겨야 한다."""
    rng = np.random.default_rng(seed)
    base = np.clip(rng.normal(170, 12, (H, W)), 0, 255).astype(np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[16:32, 20:44] = 255
    inner = np.clip(rng.normal(80, 12, (H, W)), 0, 255).astype(np.uint8)
    lum = np.where(mask > 0, inner, base).astype(int)
    b = np.where(mask > 0, lum + 40, lum - 30)
    r = np.where(mask > 0, lum - 30, lum + 40)
    image = np.clip(np.stack([b, lum, r], axis=2), 0, 255).astype(np.uint8)
    t = TargetImage(path=Path("t.png"), image=image, gray=False)
    pl = Placement(center=(32, 24), bbox=(20, 16, 24, 16), offset=(20, 16), tries=1)
    return replace(context(t), placement=pl, placed_mask=mask), mask


@pytest.mark.parametrize("method", ["stats", "reinhard", "histmatch"])
@pytest.mark.parametrize("gray", [False, True])
def test_ring_methods_strength_zero_is_exact_identity(method: str, gray: bool) -> None:
    ctx, _ = _scene(gray)
    out = _RING[method](strength=0.0).apply(ctx)
    assert np.array_equal(out.composite, ctx.composite)
    assert (
        out.log["harmonize"]["method"] == method and out.log["harmonize"]["skipped"] == "strength 0"
    )


@pytest.mark.parametrize("method", ["stats", "reinhard", "histmatch"])
@pytest.mark.parametrize("gray", [False, True])
def test_ring_methods_change_inside_only_and_keep_gray_invariant(method: str, gray: bool) -> None:
    ctx, mask = _scene(gray)
    out = _RING[method](strength=1.0, ring_px=6).apply(ctx)
    inside = mask > 0
    assert out.composite.dtype == np.uint8
    assert np.array_equal(out.composite[~inside], ctx.composite[~inside])
    assert not np.array_equal(out.composite[inside], ctx.composite[inside])
    assert "skipped" not in out.log["harmonize"]
    if gray:
        c = out.composite
        assert np.array_equal(c[:, :, 0], c[:, :, 1]) and np.array_equal(c[:, :, 1], c[:, :, 2])


@pytest.mark.parametrize("method", ["reinhard", "histmatch"])
def test_ring_methods_skip_and_log_like_stats(method: str) -> None:
    ctx, _ = _scene()
    full = np.full((H, W), 255, dtype=np.uint8)
    out = _RING[method](strength=1.0).apply(replace(ctx, placed_mask=full))
    assert (
        np.array_equal(out.composite, ctx.composite) and "비어" in out.log["harmonize"]["skipped"]
    )
    out = _RING[method]().apply(context())
    assert out.log["harmonize"]["skipped"] == "placement 없음"
    out = _RING[method](strength=1.0, ring_px=6).apply(ctx)
    log = out.log["harmonize"]
    assert {"mean_in", "mean_ring", "std_in", "std_ring", "mean_after", "inside_px"} <= set(log)
    assert log["mean_in"] < log["mean_ring"] and abs(log["mean_after"] - log["mean_ring"]) < 1.5


def test_reinhard_matches_lab_mean_and_std_on_all_three_channels() -> None:
    ctx, mask = _color_scene()
    out = ReinhardHarmonize(R.ReinhardHarmonizeConfig(strength=1.0, ring_px=6), {}).apply(ctx)
    inside, ring = mask > 0, ring_of(mask, 6)
    before, after = _lab(ctx.composite), _lab(out.composite)
    for c in range(3):
        assert abs(before[inside][:, c].mean() - before[ring][:, c].mean()) > 8  # 애초에 달랐다
        assert abs(after[inside][:, c].mean() - after[ring][:, c].mean()) < 2.5
        assert abs(after[inside][:, c].std() - after[ring][:, c].std()) < 2.5
    # stats는 L만 옮기므로 a·b는 그대로 다르다
    st = StatsHarmonize(R.StatsHarmonizeConfig(strength=1.0, ring_px=6), {}).apply(ctx)
    lab_st = _lab(st.composite)
    assert abs(lab_st[inside][:, 0].mean() - lab_st[ring][:, 0].mean()) < 2.5
    assert abs(lab_st[inside][:, 1].mean() - lab_st[ring][:, 1].mean()) > 8


def test_reinhard_equals_stats_on_gray() -> None:
    ctx, _ = _scene(gray=True)
    a = ReinhardHarmonize(R.ReinhardHarmonizeConfig(strength=0.6, ring_px=8), {}).apply(ctx)
    b = StatsHarmonize(R.StatsHarmonizeConfig(strength=0.6, ring_px=8), {}).apply(ctx)
    assert np.array_equal(a.composite, b.composite)
    assert {k: v for k, v in a.log["harmonize"].items() if k != "method"} == {
        k: v for k, v in b.log["harmonize"].items() if k != "method"
    }


def test_match_hist_maps_quantiles_and_keeps_rank() -> None:
    rng = np.random.default_rng(1)
    v = rng.normal(60, 10, 2000).astype(np.float32)
    r = rng.normal(180, 25, 3000).astype(np.float32)
    out = match_hist(v, r)
    assert out.dtype == v.dtype
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert abs(np.quantile(out, q) - np.quantile(r, q)) < 3.0
    order = np.argsort(v)
    assert (np.diff(out[order]) >= 0).all()  # 순위 보존
    assert (match_hist(v, np.full(5, 7.0, np.float32)) == 7.0).all()  # 링이 상수면 상수


@pytest.mark.parametrize("gray", [False, True])
def test_histmatch_strength_one_matches_ring_cdf(gray: bool) -> None:
    ctx, mask = _scene(gray)
    out = HistmatchHarmonize(R.HistmatchHarmonizeConfig(strength=1.0, ring_px=8), {}).apply(ctx)
    inside, ring = mask > 0, ring_of(mask, 8)
    lum = _lum(out.composite, gray)
    for q in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert abs(np.quantile(lum[inside], q) - np.quantile(lum[ring], q)) < 4.0
    # 내부 순위 보존: 조화 전에 어두웠던 픽셀은 조화 후에도 어둡다
    before = _lum(ctx.composite, gray)[inside]
    lo, hi = before < np.quantile(before, 0.2), before > np.quantile(before, 0.8)
    assert lum[inside][lo].mean() < lum[inside][hi].mean()


def test_histmatch_moves_color_channels_too() -> None:
    ctx, mask = _color_scene()
    out = HistmatchHarmonize(R.HistmatchHarmonizeConfig(strength=1.0, ring_px=6), {}).apply(ctx)
    inside, ring = mask > 0, ring_of(mask, 6)
    after = _lab(out.composite)
    for c in range(3):
        assert abs(np.median(after[inside][:, c]) - np.median(after[ring][:, c])) < 4.0


@pytest.mark.parametrize("method", ["reinhard", "histmatch"])
def test_ring_methods_half_strength_is_between(method: str) -> None:
    ctx, mask = _scene()
    full = _RING[method](strength=1.0).apply(ctx)
    half = _RING[method](strength=0.5).apply(ctx)
    inside = mask > 0
    m0 = _lum(ctx.composite, False)[inside].mean()
    m1 = _lum(full.composite, False)[inside].mean()
    mh = _lum(half.composite, False)[inside].mean()
    assert m0 < mh < m1 and abs(mh - (m0 + m1) / 2) < 2.0


@pytest.mark.parametrize("method", ["stats", "reinhard", "histmatch"])
def test_ring_is_clipped_to_roi_so_background_does_not_inflate_sigma(method: str) -> None:
    """마스크가 물체(200±3) 가장자리에서 ring_px 안: ROI 없으면 링이 배경(40)을 물어 σ_R 이 폭발 → 내부 대비가 5배로
    커진다. ROI를 주면 링 = 물체 픽셀만 → σ_R 작음 → 내부가 얌전하다(로그 ring_in_roi=True)."""
    rng = np.random.default_rng(3)
    img = np.full((H, W), 40, dtype=np.uint8)
    img[:, :40] = np.clip(rng.normal(200, 3, (H, 40)), 0, 255).astype(np.uint8)  # 왼쪽 40px = 물체
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[16:32, 20:34] = 255  # 물체 오른쪽 가장자리(x=40)에서 6px
    img[mask > 0] = np.clip(rng.normal(120, 3, (H, W)), 0, 255).astype(np.uint8)[mask > 0]
    image = np.repeat(img[:, :, None], 3, axis=2)
    t = TargetImage(path=Path("t.png"), image=image, gray=True)
    pl = Placement(center=(27, 24), bbox=(20, 16, 14, 16), offset=(20, 16), tries=1)
    roi = np.zeros((H, W), dtype=bool)
    roi[:, :40] = True
    base = replace(context(t), placement=pl, placed_mask=mask)
    st = _RING[method](strength=1.0, ring_px=12)

    no_roi = st.apply(base)
    with_roi = st.apply(replace(base, roi=roi))
    inside = mask > 0
    assert no_roi.log["harmonize"]["ring_in_roi"] is False
    assert with_roi.log["harmonize"]["ring_in_roi"] is True
    assert no_roi.log["harmonize"]["std_ring"] > 50 > 5 > with_roi.log["harmonize"]["std_ring"]
    sd_no, sd_roi = (
        no_roi.composite[inside, 0].astype(float).std(),
        with_roi.composite[inside, 0].astype(float).std(),
    )
    assert sd_no > 3 * sd_roi
    assert abs(with_roi.composite[inside, 0].astype(float).mean() - 200) < 4
    # ROI가 링을 전부 배제하면(자른 링이 빔) 자르지 않은 링으로 돌아간다
    none = st.apply(replace(base, roi=np.zeros((H, W), dtype=bool)))
    assert none.log["harmonize"]["ring_in_roi"] is False
    assert np.array_equal(none.composite, no_roi.composite)


# ---------------------------------------------------------------------------
# relative(v0.8.2) — 소스 링 → 대상 링 노출 보정, 내부 상대 대비 보존
# ---------------------------------------------------------------------------

PH, PW = 24, 32


def _relative_scene(gray: bool, seed: int = 1) -> tuple[Context, np.ndarray, float]:
    """_scene(대상 180 · 내부 60)에 소스 패치를 붙인다: 패치 링 120±8 · 패치 내부 = 대상 마스크 안 픽셀 그대로(paste 로 옮긴 셈).
    반환 (ctx, mask, 소스 상대 대비 = L(내부) − L(소스 링) ≈ −60)."""
    ctx, mask = _scene(gray, seed=seed)
    rng = np.random.default_rng(seed + 100)
    patch_mask = np.zeros((PH, PW), dtype=np.uint8)
    patch_mask[4:20, 4:28] = 255  # 마스크 16×24 = 대상 마스크와 같은 크기, 링 4px 여유
    plane = np.clip(rng.normal(120, 8, (PH, PW)), 0, 255).astype(np.uint8)
    if gray:
        patch = np.repeat(plane[:, :, None], 3, axis=2)
    else:
        patch = np.stack([plane, np.clip(plane.astype(int) + 20, 0, 255), plane], axis=2).astype(
            np.uint8
        )
    patch[patch_mask > 0] = ctx.composite[mask > 0]  # 같은 픽셀 수(384) — paste 결과와 일치
    ring_vals = source_ring_values(patch, patch_mask, 6, gray)
    src_contrast = float(_lum(patch, gray)[patch_mask > 0].mean() - ring_vals.mean())
    return replace(ctx, patch=patch, patch_mask=patch_mask), mask, src_contrast


def test_match_relative_offset_and_gain() -> None:
    src_ring = np.array([100.0, 110.0, 120.0])  # μ 110 · σ ≈ 8.2
    v = np.array([40.0, 50.0, 60.0])  # 소스 링보다 60 어두움
    ref = np.array([170.0, 180.0, 190.0])  # 대상 링 μ 180
    out = match_relative(src_ring, gain=False)(v, ref)
    assert np.allclose(out, v + 70)  # 오프셋 = 180 − 110, 내부 분산·상대 대비 불변
    assert out.mean() - ref.mean() == pytest.approx(v.mean() - src_ring.mean())
    ref2 = np.array([160.0, 180.0, 200.0])  # σ 두 배
    out2 = match_relative(src_ring, gain=True)(v, ref2)
    assert out2.mean() == pytest.approx(180 + (50 - 110) * 2) and out2.std() == pytest.approx(
        v.std() * 2
    )
    flat = match_relative(np.array([5.0, 5.0]), gain=True)(v, ref)  # σ_s 0 → 오프셋만
    assert np.allclose(flat, v + 175)


def test_source_ring_values_clips_to_patch() -> None:
    patch = np.full((PH, PW, 3), 90, dtype=np.uint8)
    pm = np.zeros((PH, PW), dtype=np.uint8)
    pm[:, :] = 255
    assert source_ring_values(patch, pm, 6, gray=True).size == 0  # 마스크가 패치 전체 → 링 없음
    pm[:] = 0
    pm[8:16, 8:24] = 255
    vals = source_ring_values(patch, pm, 6, gray=True)
    assert vals.size == int(ring_of(pm, 6).sum()) and (vals == 90).all()


@pytest.mark.parametrize("gray", [False, True])
def test_relative_keeps_contrast_and_fixes_exposure(gray: bool) -> None:
    ctx, mask, src_contrast = _relative_scene(gray)
    out = RelativeHarmonize(R.RelativeHarmonizeConfig(strength=1.0, ring_px=6), {}).apply(ctx)
    inside, ring = mask > 0, ring_of(mask, 6)
    assert np.array_equal(out.composite[~inside], ctx.composite[~inside])  # 마스크 밖 불변
    lum = _lum(out.composite, gray)
    # 보정 뒤 (내부 − 대상 링) ≈ 소스의 (내부 − 소스 링): 상대 대비 보존. stats 였다면 ≈ 0 이 됐을 것
    assert lum[inside].mean() - lum[ring].mean() == pytest.approx(src_contrast, abs=2.5)
    assert src_contrast < -50
    # 내부 분산은 오프셋이라 그대로(gain off)
    assert lum[inside].std() == pytest.approx(_lum(ctx.composite, gray)[inside].std(), abs=1.5)
    log = out.log["harmonize"]
    assert log["method"] == "relative" and log["gain"] is False and log["source_ring_px"] > 0
    assert log["mean_after"] > log["mean_in"]  # 어두운 소스(링 120) → 밝은 대상(180): 위로 옮김
    if gray:
        c = out.composite
        assert np.array_equal(c[:, :, 0], c[:, :, 1]) and np.array_equal(c[:, :, 1], c[:, :, 2])


def test_relative_stats_would_erase_contrast_but_relative_does_not() -> None:
    ctx, mask, _ = _relative_scene(False)
    inside, ring = mask > 0, ring_of(mask, 6)
    rel = RelativeHarmonize(R.RelativeHarmonizeConfig(strength=1.0, ring_px=6), {}).apply(ctx)
    st = StatsHarmonize(R.StatsHarmonizeConfig(strength=1.0, ring_px=6), {}).apply(ctx)
    lr, ls = _lum(rel.composite, False), _lum(st.composite, False)
    assert abs(ls[inside].mean() - ls[ring].mean()) < 2.0  # stats: 대비 0
    assert lr[inside].mean() - lr[ring].mean() < -50  # relative: 대비 유지


def test_relative_half_strength_and_gain() -> None:
    ctx, mask, _ = _relative_scene(False)
    inside = mask > 0
    full = RelativeHarmonize(R.RelativeHarmonizeConfig(strength=1.0, ring_px=6), {}).apply(ctx)
    half = RelativeHarmonize(R.RelativeHarmonizeConfig(strength=0.5, ring_px=6), {}).apply(ctx)
    m0 = _lum(ctx.composite, False)[inside].mean()
    m1 = _lum(full.composite, False)[inside].mean()
    mh = _lum(half.composite, False)[inside].mean()
    assert m0 < mh < m1 and abs(mh - (m0 + m1) / 2) < 2.0
    gain = RelativeHarmonize(R.RelativeHarmonizeConfig(ring_px=6, gain=True), {}).apply(ctx)
    log = gain.log["harmonize"]
    assert log["gain"] is True and log["std_source_ring"] > 0
    # σ_Rt(≈10) / σ_Rs(≈8) > 1 → 내부 분산이 커진다
    assert _lum(gain.composite, False)[inside].std() > _lum(full.composite, False)[inside].std()


def test_relative_skips_without_patch_or_source_ring() -> None:
    ctx, _mask, _ = _relative_scene(False)
    no_patch = RelativeHarmonize(R.RelativeHarmonizeConfig(), {}).apply(
        replace(ctx, patch=None, patch_mask=None)
    )
    assert np.array_equal(no_patch.composite, ctx.composite)
    assert "패치 없음" in no_patch.log["harmonize"]["skipped"]
    full_mask = np.full((PH, PW), 255, dtype=np.uint8)
    no_ring = RelativeHarmonize(R.RelativeHarmonizeConfig(), {}).apply(
        replace(ctx, patch_mask=full_mask)
    )
    assert np.array_equal(no_ring.composite, ctx.composite)
    assert "소스 링" in no_ring.log["harmonize"]["skipped"]
    zero = RelativeHarmonize(R.RelativeHarmonizeConfig(strength=0.0), {}).apply(ctx)
    assert np.array_equal(zero.composite, ctx.composite)
    assert zero.log["harmonize"]["skipped"] == "strength 0"
    assert (
        RelativeHarmonize(R.RelativeHarmonizeConfig(), {})
        .apply(context())
        .log["harmonize"]["skipped"]
        == "placement 없음"
    )


def test_relative_config_in_recipe_union() -> None:
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": "t",
            "seed": 1,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1},
            "pipeline": {"preset": "relative-paste"},
        }
    )
    h = rec.pipeline.harmonize
    assert h.method == "relative" and h.strength == 1.0 and h.gain is False and h.ring_px == 12
    assert rec.pipeline.blend.method == "paste" and rec.pipeline.gtmask.policy == "source"
    with pytest.raises(
        Exception, match=r"gain|extra"
    ):  # 다른 method 의 키는 거부(조용히 무시 금지)
        R.StatsHarmonizeConfig(gain=True)

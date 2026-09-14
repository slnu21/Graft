"""5단계 조화 — ``none`` · ``stats`` · ``reinhard`` · ``histmatch`` (설계 §6). 마스크 링의 대상 통계를 참조해 **마스크 내부만** 바꾼다.

- 링 ``R = dilate(placed_mask, ring_px) − placed_mask`` (대상 좌표 = ``ctx.placed_mask``).
- 세 방법 모두 같은 틀(``harmonize_channels``): 컬러는 Lab로 바꿔 채널별로, 흑백은 세 채널이 같으므로 채널 0만 맞추고
  세 채널에 같이 써서 승격 불변식을 지킨다. ``strength``로 원본과 보간, 마스크 밖은 바이트 단위 불변.
  - ``stats``: L채널 평균·표준편차 정합 ``x' = μ_R + (x − μ_in)·(σ_R/σ_in)``.
  - ``reinhard``: 같은 정합을 L·a·b 세 채널에(Reinhard 2001 색 전이) — 컬러 대상에서 색조까지 맞춘다. 흑백이면 stats와 같다.
  - ``histmatch``: 채널별 CDF 매칭(``np.interp``)으로 내부 히스토그램을 링 히스토그램에 — 순위를 보존하므로 텍스처 대비가 가장 잘 남는다.
- **링은 ROI 안으로 자른다**(``ctx.roi``가 있을 때). 결함이 물체 가장자리에서 ``ring_px`` 안에 놓이면 링이 배경을 물고 들어와
  σ_R이 폭발하고(물체 200 / 배경 40 → σ 65), σ 정합이 내부 대비를 5배로 키워 경계에 검은 테두리가 생긴다(2026-09-14 골든
  64px 디스크에서 발견). 자른 링이 비면 자르지 않은 링으로(로그 ``ring_in_roi``).
- ``strength == 0``이면 변환 없이 항등(Lab 왕복 손실도 없음). 링이나 내부가 비면 ``skipped``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, ClassVar

import cv2
import numpy as np

from anograft.core.recipe import (
    HistmatchHarmonizeConfig,
    NoneHarmonizeConfig,
    ReinhardHarmonizeConfig,
    StatsHarmonizeConfig,
)
from anograft.core.registry import register
from anograft.core.types import Context

_EPS = 1e-6
Matcher = Callable[[np.ndarray, np.ndarray], np.ndarray]


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """0/255 마스크를 ``px``만큼 팽창(타원 커널). 0이면 그대로."""
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def ring_of(mask: np.ndarray, ring_px: int) -> np.ndarray:
    """마스크 바깥 ``ring_px`` 폭 링 (bool)."""
    return (dilate_mask(mask, ring_px) > 0) & ~(mask > 0)


def ring_in_roi(mask: np.ndarray, ring_px: int, roi: np.ndarray | None) -> tuple[np.ndarray, bool]:
    """링을 ROI 안으로 자른다. ROI가 없거나 자른 결과가 비면 원래 링 + False."""
    ring = ring_of(mask, ring_px)
    if roi is None:
        return ring, False
    clipped = ring & (roi > 0)
    if not clipped.any():
        return ring, False
    return clipped, True


def match_stats(values: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``values``의 평균·표준편차를 ``ref``에 맞춘다. σ_in≈0이면 평균만 옮긴다."""
    mu_in, sd_in = float(values.mean()), float(values.std())
    mu_r, sd_r = float(ref.mean()), float(ref.std())
    if sd_in < _EPS:
        return np.full_like(values, mu_r)
    return mu_r + (values - mu_in) * (sd_r / sd_in)


def match_hist(values: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``values``의 히스토그램을 ``ref``에 맞춘다(CDF 매칭). 같은 값은 같은 값으로 가고 순위가 보존된다."""
    _v_uniq, v_inv, v_cnt = np.unique(values, return_inverse=True, return_counts=True)
    r_uniq, r_cnt = np.unique(ref, return_counts=True)
    v_cdf = np.cumsum(v_cnt) / float(values.size)
    r_cdf = np.cumsum(r_cnt) / float(ref.size)
    matched = np.interp(v_cdf, r_cdf, r_uniq.astype(np.float64))
    return matched[v_inv].astype(values.dtype)


def harmonize_channels(
    composite: np.ndarray,
    mask: np.ndarray,
    ring_px: int,
    strength: float,
    gray: bool,
    matcher: Matcher,
    channels: tuple[int, ...],
    roi: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """반환 ``(새 composite, 통계 로그)``. 컬러는 Lab ``channels``를, 흑백은 채널 0만 맞춘다. 마스크 밖은 불변."""
    inside = mask > 0
    ring, in_roi = ring_in_roi(mask, ring_px, roi)
    n_in, n_ring = int(inside.sum()), int(ring.sum())
    stats: dict[str, Any] = {
        "ring_px": ring_px,
        "inside_px": n_in,
        "ring_px_count": n_ring,
        "ring_in_roi": in_roi,
    }
    if n_in == 0 or n_ring == 0 or strength <= 0:
        stats["skipped"] = "strength 0" if strength <= 0 else "내부 또는 링이 비어 있음"
        return composite, stats

    work = composite[:, :, :1] if gray else cv2.cvtColor(composite, cv2.COLOR_BGR2LAB)
    work = work.copy()
    chans = (0,) if gray else channels
    for c in chans:
        plane = work[:, :, c].astype(np.float32)
        matched = matcher(plane[inside], plane[ring])
        mixed = plane[inside] * (1.0 - strength) + matched * strength
        if c == 0:  # 로그는 L(밝기) 채널 기준 — 방법 간 비교가 되게
            stats.update(
                mean_in=float(plane[inside].mean()),
                mean_ring=float(plane[ring].mean()),
                std_in=float(plane[inside].std()),
                std_ring=float(plane[ring].std()),
                mean_after=float(mixed.mean()),
            )
        work[:, :, c][inside] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    out = composite.copy()
    if gray:
        out[inside] = np.repeat(work[:, :, :1][inside], 3, axis=1)
    else:
        out[inside] = cv2.cvtColor(work, cv2.COLOR_LAB2BGR)[inside]
    return out, stats


def harmonize_stats(
    composite: np.ndarray,
    mask: np.ndarray,
    ring_px: int,
    strength: float,
    gray: bool,
    roi: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``stats``: L채널 평균·표준편차 정합."""
    return harmonize_channels(composite, mask, ring_px, strength, gray, match_stats, (0,), roi)


def harmonize_reinhard(
    composite: np.ndarray,
    mask: np.ndarray,
    ring_px: int,
    strength: float,
    gray: bool,
    roi: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``reinhard``: L·a·b 세 채널 평균·표준편차 정합."""
    return harmonize_channels(composite, mask, ring_px, strength, gray, match_stats, (0, 1, 2), roi)


def harmonize_histmatch(
    composite: np.ndarray,
    mask: np.ndarray,
    ring_px: int,
    strength: float,
    gray: bool,
    roi: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``histmatch``: L·a·b 채널별 CDF 매칭."""
    return harmonize_channels(composite, mask, ring_px, strength, gray, match_hist, (0, 1, 2), roi)


def _inputs(ctx: Context) -> np.ndarray | None:
    return ctx.placed_mask if ctx.placement is not None else None


@register
class NoneHarmonize:
    stage: ClassVar[str] = "harmonize"
    methods: ClassVar[tuple[str, ...]] = ("none",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: NoneHarmonizeConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def apply(self, ctx: Context) -> Context:
        return ctx.with_log("harmonize", {"method": "none"})


class _RingHarmonize:
    """stats · reinhard · histmatch 공통 골격 — 서브클래스는 ``method``·``_fn``만 고른다."""

    stage: ClassVar[str] = "harmonize"
    requires: ClassVar[tuple[str, ...]] = ()
    method: ClassVar[str]
    _fn: ClassVar[Callable[..., tuple[np.ndarray, dict[str, Any]]]]

    def __init__(
        self,
        cfg: StatsHarmonizeConfig | ReinhardHarmonizeConfig | HistmatchHarmonizeConfig,
        deps: Mapping[str, Any],
    ) -> None:
        self.cfg = cfg

    def apply(self, ctx: Context) -> Context:
        mask = _inputs(ctx)
        log: dict[str, Any] = {"method": self.method, "strength": self.cfg.strength}
        if mask is None:
            log["skipped"] = "placement 없음"
            return ctx.with_log("harmonize", log)
        out, stats = type(self)._fn(
            ctx.composite, mask, self.cfg.ring_px, self.cfg.strength, ctx.target.gray, ctx.roi
        )
        log.update(stats)
        return replace(ctx, composite=out).with_log("harmonize", log)


@register
class StatsHarmonize(_RingHarmonize):
    methods: ClassVar[tuple[str, ...]] = ("stats",)
    method = "stats"
    _fn = staticmethod(harmonize_stats)


@register
class ReinhardHarmonize(_RingHarmonize):
    methods: ClassVar[tuple[str, ...]] = ("reinhard",)
    method = "reinhard"
    _fn = staticmethod(harmonize_reinhard)


@register
class HistmatchHarmonize(_RingHarmonize):
    methods: ClassVar[tuple[str, ...]] = ("histmatch",)
    method = "histmatch"
    _fn = staticmethod(harmonize_histmatch)

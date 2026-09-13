"""5단계 조화 — ``none`` · ``stats`` (설계 §6). 마스크 링의 대상 통계를 참조해 **마스크 내부만** 바꾼다.

- 링 ``R = dilate(placed_mask, ring_px) − placed_mask`` (대상 좌표 = ``ctx.placed_mask``).
- ``stats``: Lab L채널 평균·표준편차 정합 ``x' = μ_R + (x − μ_in)·(σ_R/σ_in)``, ``strength``로 원본과 보간.
  흑백 대상은 세 채널이 같으므로 채널 0을 L로 쓰고 세 채널에 같이 써서 승격 불변식을 지킨다.
- ``strength == 0``이면 변환 없이 항등(Lab 왕복 손실도 없음). 링이나 내부가 비면 ``skipped``.
- reinhard(Lab 3ch) · histmatch(CDF)는 stages-more-methods에서 같은 ``_ring``·``_mix`` 위에 얹는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

import cv2
import numpy as np

from anograft.core.recipe import NoneHarmonizeConfig, StatsHarmonizeConfig
from anograft.core.registry import register
from anograft.core.types import Context

_EPS = 1e-6


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """0/255 마스크를 ``px``만큼 팽창(타원 커널). 0이면 그대로."""
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def ring_of(mask: np.ndarray, ring_px: int) -> np.ndarray:
    """마스크 바깥 ``ring_px`` 폭 링 (bool)."""
    return (dilate_mask(mask, ring_px) > 0) & ~(mask > 0)


def match_stats(values: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """``values``의 평균·표준편차를 ``ref``에 맞춘다. σ_in≈0이면 평균만 옮긴다."""
    mu_in, sd_in = float(values.mean()), float(values.std())
    mu_r, sd_r = float(ref.mean()), float(ref.std())
    if sd_in < _EPS:
        return np.full_like(values, mu_r)
    return mu_r + (values - mu_in) * (sd_r / sd_in)


def harmonize_stats(
    composite: np.ndarray, mask: np.ndarray, ring_px: int, strength: float, gray: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    """반환 ``(새 composite, 통계 로그)``. 마스크 밖 픽셀은 바이트 단위로 불변."""
    inside = mask > 0
    ring = ring_of(mask, ring_px)
    n_in, n_ring = int(inside.sum()), int(ring.sum())
    stats: dict[str, Any] = {"ring_px": ring_px, "inside_px": n_in, "ring_px_count": n_ring}
    if n_in == 0 or n_ring == 0 or strength <= 0:
        stats["skipped"] = "strength 0" if strength <= 0 else "내부 또는 링이 비어 있음"
        return composite, stats

    if gray:
        lum = composite[:, :, 0].astype(np.float32)
    else:
        lab = cv2.cvtColor(composite, cv2.COLOR_BGR2LAB)
        lum = lab[:, :, 0].astype(np.float32)
    matched = match_stats(lum[inside], lum[ring])
    mixed = lum[inside] * (1.0 - strength) + matched * strength
    stats.update(
        mean_in=float(lum[inside].mean()),
        mean_ring=float(lum[ring].mean()),
        std_in=float(lum[inside].std()),
        std_ring=float(lum[ring].std()),
        mean_after=float(mixed.mean()),
    )
    out = composite.copy()
    if gray:
        v = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
        out[inside] = v[:, None]
    else:
        lab2 = lab.copy()
        lab2[:, :, 0][inside] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        out[inside] = bgr[inside]
    return out, stats


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


@register
class StatsHarmonize:
    stage: ClassVar[str] = "harmonize"
    methods: ClassVar[tuple[str, ...]] = ("stats",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: StatsHarmonizeConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def apply(self, ctx: Context) -> Context:
        mask = _inputs(ctx)
        log: dict[str, Any] = {"method": "stats", "strength": self.cfg.strength}
        if mask is None:
            log["skipped"] = "placement 없음"
            return ctx.with_log("harmonize", log)
        out, stats = harmonize_stats(
            ctx.composite, mask, self.cfg.ring_px, self.cfg.strength, ctx.target.gray
        )
        log.update(stats)
        return replace(ctx, composite=out).with_log("harmonize", log)

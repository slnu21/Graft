"""3단계 배치 — ``sampled`` (설계 §6 placement): ROI 픽셀 가중 추첨 × 거절 샘플링, 실패 시 축소 재시도.

채택 조건(전부): 패치 마스크의 모든 픽셀이 ROI 안 ∧ 이미지 테두리에서 ``margin_px`` 이상 ∧ 기존 결함(``ctx.placed``)과
겹치지 않음. 후보 중심은 ``ROI ∧ 테두리 여유 ∧ 기존 결함 제외`` 픽셀에서 뽑는다:
``uniform`` = 균등 · ``edge`` = ``1/(1+d)`` · ``center`` = ``d`` (``d`` = ROI 경계까지 거리, 거리 변환 1회).

``max_tries``를 다 쓰면 패치를 ``factor``배 축소해 ``rounds``회 더 시도한다(축소된 패치가 ``ctx.patch``로 돌아간다).
최종 실패 → ``placement=None`` + 경고 + 로그. **예외를 던지지 않는다.**

무작위 소비: 시도마다 정확히 1회 — ``uniform``은 ``rng.integers(n)``, 가중은 ``rng.random()`` + 누적분포 이분탐색
(``rng.choice(p=…)``와 같은 소비량이지만 누적합을 시도마다 다시 계산하지 않아 4K 이미지에서도 빠르다).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

import cv2
import numpy as np

from anograft.core.channels import binarize
from anograft.core.recipe import SampledPlacementConfig
from anograft.core.registry import register
from anograft.core.roi import distance_to_edge
from anograft.core.types import Context, Placement

MIN_MASK_AREA = 4  # px — 축소 끝에 이보다 작아지면 포기


def shrink_patch(
    patch: np.ndarray, mask: np.ndarray, factor: float
) -> tuple[np.ndarray, np.ndarray]:
    """패치·마스크를 ``factor``배 축소. 이미지 ``INTER_AREA``, 마스크 ``INTER_NEAREST`` + 이진화."""
    h, w = mask.shape[:2]
    nw, nh = max(1, round(w * factor)), max(1, round(h * factor))
    out_patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
    out_mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return out_patch, binarize(out_mask)


def allowed_centers(roi: np.ndarray, margin_px: int, existing: np.ndarray | None) -> np.ndarray:
    """후보 중심 픽셀 = ROI ∧ 테두리 여유 ∧ 기존 결함 제외 (HxW bool)."""
    h, w = roi.shape[:2]
    allowed = np.zeros((h, w), dtype=bool)
    m = margin_px
    if h > 2 * m and w > 2 * m:
        allowed[m : h - m, m : w - m] = roi[m : h - m, m : w - m]
    if existing is not None:
        allowed &= ~existing
    return allowed


def candidate_cdf(allowed: np.ndarray, distribution: str) -> np.ndarray | None:
    """가중 분포의 누적합(허용 픽셀 순서 = row-major). ``uniform``은 None(균등 정수 추첨)."""
    if distribution == "uniform":
        return None
    d = distance_to_edge(allowed)[allowed].astype(np.float64)
    if distribution == "edge":
        weights = 1.0 / (1.0 + d)
    elif distribution == "center":
        weights = d
    else:
        raise ValueError(f"알 수 없는 distribution: {distribution!r}")
    return np.cumsum(weights)


def draw_index(rng: np.random.Generator, n: int, cdf: np.ndarray | None) -> int:
    """허용 픽셀 중 하나의 순번. 시도마다 rng를 정확히 1회 소비."""
    if cdf is None:
        return int(rng.integers(n))
    k = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
    return min(k, n - 1)


@register
class SampledPlacement:
    stage: ClassVar[str] = "placement"
    methods: ClassVar[tuple[str, ...]] = ("sampled",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: SampledPlacementConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def apply(self, ctx: Context) -> Context:
        cfg = self.cfg
        log: dict[str, Any] = {"method": "sampled", "distribution": cfg.distribution}
        if ctx.patch is None or ctx.patch_mask is None:
            log["skipped"] = "patch 없음"
            return ctx.with_log("placement", log)
        if ctx.roi is None:
            return self._fail(ctx, log, "ROI 없음", 0, 0)

        roi = ctx.roi
        h, w = roi.shape[:2]
        m = cfg.margin_px
        existing: np.ndarray | None = None
        if ctx.placed:
            existing = np.zeros((h, w), dtype=bool)
            for p in ctx.placed:
                existing |= p.mask > 0
        allowed = allowed_centers(roi, m, existing)
        flat = np.flatnonzero(allowed)
        n = int(flat.size)
        log["roi_area_px"] = int(np.count_nonzero(roi))
        log["candidates"] = n
        if n == 0:
            return self._fail(ctx, log, "후보 중심 없음 (ROI ∧ 테두리 여유 ∧ 기존 결함 제외)", 0, 0)
        cdf = candidate_cdf(allowed, cfg.distribution)

        patch, mask = ctx.patch, ctx.patch_mask
        rng = ctx.rng
        tries = 0
        reason = "max_tries 소진"
        for round_ in range(cfg.shrink_on_fail.rounds + 1):
            if round_ > 0:
                patch, mask = shrink_patch(patch, mask, cfg.shrink_on_fail.factor)
                if np.count_nonzero(mask) < MIN_MASK_AREA:
                    reason = f"축소 {round_}회 후 마스크 면적 < {MIN_MASK_AREA}px"
                    break
            mx, my, mw, mh = cv2.boundingRect(mask)
            if mw == 0 or mh == 0:
                reason = "마스크가 비어 있음"
                break
            if mw > w - 2 * m or mh > h - 2 * m:
                reason = f"마스크 bbox {mw}×{mh} 가 테두리 여유를 뺀 이미지보다 큼"
                continue  # 시도 없이 바로 축소
            crop = mask[my : my + mh, mx : mx + mw] > 0
            for _ in range(cfg.max_tries):
                tries += 1
                cy, cx = divmod(int(flat[draw_index(rng, n, cdf)]), w)
                x0, y0 = cx - mw // 2, cy - mh // 2
                if x0 < m or y0 < m or x0 + mw > w - m or y0 + mh > h - m:
                    continue
                if not roi[y0 : y0 + mh, x0 : x0 + mw][crop].all():
                    continue
                if existing is not None and existing[y0 : y0 + mh, x0 : x0 + mw][crop].any():
                    continue
                return self._accept(
                    ctx, log, patch, mask, (x0, y0, mw, mh), (mx, my), tries, round_
                )
        return self._fail(ctx, log, reason, tries, round_)

    # ------------------------------------------------------------------

    def _accept(
        self,
        ctx: Context,
        log: dict[str, Any],
        patch: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        mask_origin: tuple[int, int],
        tries: int,
        rounds: int,
    ) -> Context:
        x0, y0, mw, mh = bbox
        mx, my = mask_origin
        h, w = ctx.target.image.shape[:2]
        placed = np.zeros((h, w), dtype=np.uint8)
        placed[y0 : y0 + mh, x0 : x0 + mw] = mask[my : my + mh, mx : mx + mw]
        pl = Placement(
            center=(x0 + mw // 2, y0 + mh // 2),
            bbox=(x0, y0, mw, mh),
            offset=(x0 - mx, y0 - my),
            tries=tries,
            shrink_rounds=rounds,
        )
        log.update(
            {
                "center": list(pl.center),
                "bbox": list(pl.bbox),
                "offset": list(pl.offset),
                "tries": tries,
                "shrink_rounds": rounds,
                "shrink_scale": self.cfg.shrink_on_fail.factor**rounds,
                "patch_shape": [int(mask.shape[0]), int(mask.shape[1])],
                "mask_area_px": int(np.count_nonzero(mask)),
            }
        )
        return replace(
            ctx, patch=patch, patch_mask=mask, placement=pl, placed_mask=placed
        ).with_log("placement", log)

    def _fail(
        self, ctx: Context, log: dict[str, Any], reason: str, tries: int, rounds: int
    ) -> Context:
        log.update({"failed": True, "reason": reason, "tries": tries, "shrink_rounds": rounds})
        src_id = ctx.source.id if ctx.source is not None else "?"
        return (
            replace(ctx, placement=None, placed_mask=None)
            .warn(f"placement: {src_id} 배치 실패 — {reason} (시도 {tries}, 축소 {rounds})")
            .with_log("placement", log)
        )

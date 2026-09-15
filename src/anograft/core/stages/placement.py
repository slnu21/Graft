"""3단계 배치 (설계 §6 placement) — ``sampled``(v0.1) · ``structure-aware``(v0.4).

공통 채택 조건(전부): 패치 마스크의 모든 픽셀이 ROI 안 ∧ 이미지 테두리에서 ``margin_px`` 이상 ∧ 기존 결함(``ctx.placed``)과
겹치지 않음. 후보 중심은 ``ROI ∧ 테두리 여유 ∧ 기존 결함 제외`` 픽셀에서 뽑는다. ``max_tries``를 다 쓰면 패치를 ``factor``배
축소해 ``rounds``회 더 시도한다(축소된 패치가 ``ctx.patch``로 돌아간다). 최종 실패 → ``placement=None`` + 경고 + 로그.
**예외를 던지지 않는다.**

- ``sampled``: 가중 = ``uniform`` 균등 · ``edge`` = ``1/(1+d)`` · ``center`` = ``d`` (``d`` = ROI 경계까지 거리, 거리 변환 1회).
  무작위 소비: 시도마다 정확히 1회 — ``uniform``은 ``rng.integers(n)``, 가중은 ``rng.random()`` + 누적분포 이분탐색
  (``rng.choice(p=…)``와 같은 소비량이지만 누적합을 시도마다 다시 계산하지 않아 4K 이미지에서도 빠르다).
- ``structure-aware``: 가중 = 그래디언트 크기(``prefer`` edges/flat/uniform, ``core/structure.py``), 방향 = 후보 자리 창의
  구조 텐서 지배 방향에 패치 주축을 정렬(``align`` along/across, 일관성·이방성 문턱, ``jitter_deg``). 정렬 회전은
  ``warp_affine``(캔버스 확장)으로 시도마다 다시 만들고 채택될 때만 ``ctx.patch``로 돌아간다.
  무작위 소비: 시도마다 정확히 2회 — 자리(위와 같음) → ``rng.uniform(-jitter, jitter)``(정렬을 안 해도 소비해
  켜고 끄는 게 스트림을 안 밀게). 그래디언트 크기 맵·그레이는 **이미지당 1회** 계산해 스테이지 안에 캐시한다
  (같은 ``target.image`` 객체를 붙들고 ``is`` 비교 — id 재사용으로 다른 이미지에 옛 맵을 쓰는 일이 없다).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

import cv2
import numpy as np

from anograft.core import structure as S
from anograft.core.channels import binarize
from anograft.core.recipe import SampledPlacementConfig, StructureAwarePlacementConfig
from anograft.core.registry import register
from anograft.core.roi import distance_to_edge
from anograft.core.stages.geometry import warp_affine
from anograft.core.types import Context, Placement

MIN_MASK_AREA = 4  # px — 축소 끝에 이보다 작아지면 포기
MIN_WINDOW_HALF = 8  # structure-aware 방향 창 최소 반폭(px) — 작은 결함도 결을 읽을 만큼

# 채택 결과 = (patch, mask, bbox(x, y, w, h), mask_origin(mx, my))
Hit = tuple[np.ndarray, np.ndarray, tuple[int, int, int, int], tuple[int, int]]


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


def fits_at(
    roi: np.ndarray,
    existing: np.ndarray | None,
    crop: np.ndarray,
    x0: int,
    y0: int,
    margin: int,
) -> bool:
    """마스크 bbox 크롭(bool)을 ``(x0, y0)``에 놓았을 때 테두리 여유·ROI·기존 결함 조건을 전부 만족하는가."""
    h, w = roi.shape[:2]
    mh, mw = crop.shape[:2]
    if x0 < margin or y0 < margin or x0 + mw > w - margin or y0 + mh > h - margin:
        return False
    if not roi[y0 : y0 + mh, x0 : x0 + mw][crop].all():
        return False
    return not (existing is not None and existing[y0 : y0 + mh, x0 : x0 + mw][crop].any())


def fit_diagnostic(
    allowed: np.ndarray,
    original_mask: np.ndarray | None,
    last_mask: np.ndarray,
    log: dict[str, Any],
) -> str:
    """배치 실패 사유에 덧붙일 한 줄: 허용 영역의 최대 폭(내접원 지름)과 패치 bbox(원본 → 마지막 축소).
    짧은 변조차 폭을 넘으면 어디에도 들어갈 수 없다 — 그때는 조치(scale·shrink_on_fail·ROI)를 힌트로."""
    if not allowed.any():
        return ""
    width = float(distance_to_edge(allowed).max()) * 2.0
    _, _, ow, oh = cv2.boundingRect(original_mask) if original_mask is not None else (0, 0, 0, 0)
    _, _, lw, lh = cv2.boundingRect(last_mask)
    log["roi_max_width_px"] = round(width, 1)
    log["patch_bbox_px"] = [int(ow), int(oh)]
    log["patch_bbox_last_px"] = [int(lw), int(lh)]
    text = f" · ROI 최대 폭 {width:.0f}px vs 패치 {ow}×{oh}px"
    if (lw, lh) != (ow, oh):
        text += f"(축소 후 {lw}×{lh})"
    if min(lw, lh) > width:
        text += " — 패치가 ROI 폭보다 큽니다: geometry.scale 을 낮추거나 shrink_on_fail 을 늘리거나 ROI 를 넓히세요"
    return text


class _PlacementBase:
    """두 method 의 공통 뼈대 — 후보 준비·축소 루프·채택·실패 로그. 서브클래스는 ``_prepare``(결함당 1회 준비물)와
    ``_try_round``(한 축소 라운드의 ``max_tries`` 시도)만 구현한다."""

    stage: ClassVar[str] = "placement"
    requires: ClassVar[tuple[str, ...]] = ()
    method_name: ClassVar[str] = ""

    def __init__(
        self, cfg: SampledPlacementConfig | StructureAwarePlacementConfig, deps: Mapping[str, Any]
    ) -> None:
        self.cfg = cfg

    # --- 서브클래스 훅 ---------------------------------------------------

    def _describe(self, log: dict[str, Any]) -> None:
        """method 별 설정 요약을 로그 앞머리에."""

    def _prepare(self, ctx: Context, allowed: np.ndarray, log: dict[str, Any]) -> Any:
        """결함당 1회 준비물(누적합 등). 반환값은 ``_try_round``의 ``state``."""
        raise NotImplementedError

    def _try_round(
        self,
        ctx: Context,
        state: Any,
        patch: np.ndarray,
        mask: np.ndarray,
        flat: np.ndarray,
        roi: np.ndarray,
        existing: np.ndarray | None,
        tries: int,
        log: dict[str, Any],
    ) -> tuple[int, Hit | None]:
        """``max_tries``번 시도. 반환 ``(누적 tries, 채택 결과 | None)``."""
        raise NotImplementedError

    # --- 공통 흐름 -------------------------------------------------------

    def apply(self, ctx: Context) -> Context:
        cfg = self.cfg
        log: dict[str, Any] = {"method": self.method_name}
        self._describe(log)
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
        state = self._prepare(ctx, allowed, log)

        patch, mask = ctx.patch, ctx.patch_mask
        tries = 0
        reason = "max_tries 소진"
        for round_ in range(cfg.shrink_on_fail.rounds + 1):
            if round_ > 0:
                patch, mask = shrink_patch(patch, mask, cfg.shrink_on_fail.factor)
                if np.count_nonzero(mask) < MIN_MASK_AREA:
                    reason = f"축소 {round_}회 후 마스크 면적 < {MIN_MASK_AREA}px"
                    break
            _, _, mw, mh = cv2.boundingRect(mask)
            if mw == 0 or mh == 0:
                reason = "마스크가 비어 있음"
                break
            if mw > w - 2 * m or mh > h - 2 * m:
                reason = f"마스크 bbox {mw}×{mh} 가 테두리 여유를 뺀 이미지보다 큼"
                continue  # 시도 없이 바로 축소
            tries, hit = self._try_round(ctx, state, patch, mask, flat, roi, existing, tries, log)
            if hit is not None:
                p2, m2, bbox, origin = hit
                return self._accept(ctx, log, p2, m2, bbox, origin, tries, round_)
        # 실패 진단(KNOWN-ISSUES 부록): 좁은 ROI(링 등)에 패치가 안 들어가는 경우를 사용자가 알아볼 수 있게
        # "ROI 최대 폭(내접원 지름) vs 패치 bbox" 를 사유·로그에 붙인다. 실패 경로에서만 계산(distanceTransform 1회).
        reason += fit_diagnostic(allowed, ctx.patch_mask, mask, log)
        return self._fail(ctx, log, reason, tries, round_)

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


@register
class SampledPlacement(_PlacementBase):
    methods: ClassVar[tuple[str, ...]] = ("sampled",)
    method_name: ClassVar[str] = "sampled"
    cfg: SampledPlacementConfig

    def _describe(self, log: dict[str, Any]) -> None:
        log["distribution"] = self.cfg.distribution

    def _prepare(self, ctx: Context, allowed: np.ndarray, log: dict[str, Any]) -> Any:
        return candidate_cdf(allowed, self.cfg.distribution)

    def _try_round(
        self,
        ctx: Context,
        state: Any,
        patch: np.ndarray,
        mask: np.ndarray,
        flat: np.ndarray,
        roi: np.ndarray,
        existing: np.ndarray | None,
        tries: int,
        log: dict[str, Any],
    ) -> tuple[int, Hit | None]:
        cdf: np.ndarray | None = state
        n, w = int(flat.size), roi.shape[1]
        m = self.cfg.margin_px
        mx, my, mw, mh = cv2.boundingRect(mask)
        crop = mask[my : my + mh, mx : mx + mw] > 0
        for _ in range(self.cfg.max_tries):
            tries += 1
            cy, cx = divmod(int(flat[draw_index(ctx.rng, n, cdf)]), w)
            x0, y0 = cx - mw // 2, cy - mh // 2
            if fits_at(roi, existing, crop, x0, y0, m):
                return tries, (patch, mask, (x0, y0, mw, mh), (mx, my))
        return tries, None


class StructureField:
    """이미지당 1회 계산하는 구조 맵. ``image``를 붙들어 두어 ``is`` 비교가 안전하다."""

    __slots__ = ("gray", "image", "mag")

    def __init__(self, image: np.ndarray, smooth_px: float) -> None:
        self.image = image
        self.gray = S.to_gray_f32(image)
        self.mag = S.gradient_magnitude(self.gray, smooth_px)


@register
class StructureAwarePlacement(_PlacementBase):
    methods: ClassVar[tuple[str, ...]] = ("structure-aware",)
    method_name: ClassVar[str] = "structure-aware"
    cfg: StructureAwarePlacementConfig

    def __init__(self, cfg: StructureAwarePlacementConfig, deps: Mapping[str, Any]) -> None:
        super().__init__(cfg, deps)
        self._field: StructureField | None = None

    def field_for(self, image: np.ndarray) -> StructureField:
        f = self._field
        if f is None or f.image is not image:
            f = self._field = StructureField(image, self.cfg.smooth_px)
        return f

    def _describe(self, log: dict[str, Any]) -> None:
        log.update({"prefer": self.cfg.prefer, "align": self.cfg.align})

    def _prepare(self, ctx: Context, allowed: np.ndarray, log: dict[str, Any]) -> Any:
        field = self.field_for(ctx.target.image)
        weights = S.structure_weights(field.mag, allowed, self.cfg.prefer, self.cfg.strength)
        cdf = None if weights is None else np.cumsum(weights)
        return field, cdf

    def _try_round(
        self,
        ctx: Context,
        state: Any,
        patch: np.ndarray,
        mask: np.ndarray,
        flat: np.ndarray,
        roi: np.ndarray,
        existing: np.ndarray | None,
        tries: int,
        log: dict[str, Any],
    ) -> tuple[int, Hit | None]:
        cfg = self.cfg
        field, cdf = state
        n, w = int(flat.size), roi.shape[1]
        m = cfg.margin_px
        rng = ctx.rng
        phi, aniso = S.mask_principal_axis(mask)
        _, _, bw, bh = cv2.boundingRect(mask)
        half = max(MIN_WINDOW_HALF, max(bw, bh) // 2)
        can_align = cfg.align != "none" and aniso >= cfg.min_anisotropy
        log.update({"patch_axis_deg": round(phi, 2), "anisotropy": round(aniso, 3)})
        for _ in range(cfg.max_tries):
            tries += 1
            cy, cx = divmod(int(flat[draw_index(rng, n, cdf)]), w)
            jitter = float(
                rng.uniform(-cfg.jitter_deg, cfg.jitter_deg)
            )  # 정렬 여부와 무관하게 항상 소비
            theta_g, coh = S.window_orientation(field.gray, cx, cy, half)
            aligned = can_align and coh >= cfg.min_coherence
            if aligned:
                target = S.wrap_180(theta_g + 90.0) if cfg.align == "along" else theta_g
                angle = S.align_rotation(phi, target) + jitter
                p2, m2 = warp_affine(patch, mask, 1.0, angle)
            else:
                angle, p2, m2 = 0.0, patch, mask
            mx, my, mw, mh = cv2.boundingRect(m2)
            if mw == 0 or mh == 0:
                continue
            crop = m2[my : my + mh, mx : mx + mw] > 0
            x0, y0 = cx - mw // 2, cy - mh // 2
            if fits_at(roi, existing, crop, x0, y0, m):
                log.update(
                    {
                        "aligned": aligned,
                        "angle_deg": round(angle, 2),
                        "orientation_deg": round(theta_g, 2),
                        "coherence": round(coh, 3),
                    }
                )
                return tries, (p2, m2, (x0, y0, mw, mh), (mx, my))
        return tries, None

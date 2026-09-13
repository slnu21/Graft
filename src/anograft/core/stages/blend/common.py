"""blend 스테이지 공통 — 캔버스 창 클리핑 · 입력 검사 · 하드 붙이기 · 페더 알파.

모든 blend 방법의 입력은 ``(composite, patch, patch_mask, placement)``, 출력은 **마스크 bbox 밖이 불변**인 ``composite``.
``Placement.offset``이 패치 캔버스 → 대상 좌표 변환이고, 캔버스는 이미지 밖으로 걸쳐도 되므로 창을 잘라 쓴다:
``composite[y, x] ↔ patch[y - oy, x - ox]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from anograft.core.types import Context, Placement


@dataclass(frozen=True)
class Window:
    """대상과 패치 캔버스가 겹치는 창. ``ys/xs``는 대상 슬라이스, ``pys/pxs``는 패치 슬라이스(같은 크기)."""

    ys: slice
    xs: slice
    pys: slice
    pxs: slice

    @property
    def empty(self) -> bool:
        return self.ys.stop <= self.ys.start or self.xs.stop <= self.xs.start


def canvas_window(
    target_shape: tuple[int, ...], patch_shape: tuple[int, ...], offset: tuple[int, int]
) -> Window:
    h, w = target_shape[:2]
    ph, pw = patch_shape[:2]
    ox, oy = offset
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + pw, w), min(oy + ph, h)
    return Window(slice(y0, y1), slice(x0, x1), slice(y0 - oy, y1 - oy), slice(x0 - ox, x1 - ox))


@dataclass(frozen=True)
class BlendInputs:
    composite: np.ndarray
    patch: np.ndarray
    mask: np.ndarray  # 패치 크기, 0/255
    placement: Placement
    window: Window


def blend_inputs(ctx: Context) -> BlendInputs | None:
    """결함 루프 앞 단계가 실패했으면 None — 호출자는 ``log.blend.skipped``만 남기고 통과시킨다."""
    if ctx.patch is None or ctx.patch_mask is None or ctx.placement is None:
        return None
    win = canvas_window(ctx.composite.shape, ctx.patch_mask.shape, ctx.placement.offset)
    return BlendInputs(ctx.composite, ctx.patch, ctx.patch_mask, ctx.placement, win)


def skipped(ctx: Context, method: str) -> Context:
    return ctx.with_log("blend", {"method": method, "skipped": "patch/placement 없음"})


def paste(inp: BlendInputs) -> np.ndarray:
    """마스크 내부를 패치로 그대로 치환. CutPaste 기준선이자 alpha(feather 0)의 실체."""
    out = inp.composite.copy()
    w = inp.window
    if w.empty:
        return out
    m = inp.mask[w.pys, w.pxs] > 0
    region = out[w.ys, w.xs]
    region[m] = inp.patch[w.pys, w.pxs][m]
    return out


def feather_alpha(mask: np.ndarray, feather_px: int) -> np.ndarray:
    """마스크 **안쪽** 경계에서 ``feather_px`` 폭 선형 페더. 반환 float32 HxW ∈ [0,1], 마스크 밖 0, 깊은 안쪽 1.

    페더가 마스크 안쪽에 있으므로 마스크 bbox 밖은 절대 변하지 않는다. ``feather_px=0``이면 마스크 그대로(=paste).
    """
    if feather_px <= 0:
        return (mask > 0).astype(np.float32)
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    return np.clip(dist / float(feather_px), 0.0, 1.0).astype(np.float32)


def alpha_blend(inp: BlendInputs, feather_px: int) -> np.ndarray:
    out = inp.composite.copy()
    w = inp.window
    if w.empty:
        return out
    a = feather_alpha(inp.mask, feather_px)[w.pys, w.pxs][..., None]
    region = out[w.ys, w.xs].astype(np.float32)
    src = inp.patch[w.pys, w.pxs].astype(np.float32)
    mixed = region * (1.0 - a) + src * a
    out[w.ys, w.xs] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    return out


def bbox_inside(bbox: tuple[int, int, int, int], shape: tuple[int, ...], margin: int) -> bool:
    x, y, w, h = bbox
    hh, ww = shape[:2]
    return x >= margin and y >= margin and x + w <= ww - margin and y + h <= hh - margin


def touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def log_base(method: str, **extra: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"method": method}
    d.update(extra)
    return d

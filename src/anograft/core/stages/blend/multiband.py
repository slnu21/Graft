"""4단계 blend — ``multiband``: 라플라시안 피라미드 블렌딩(Burt & Adelson 1983). 자체 구현.

``cv2.detail_MultiBandBlender``는 스티칭용 API라 마스크 의미(가중 합)가 다르고 결과 영역을 통제할 수 없어 쓰지 않는다.

1. **레벨 수** ``levels_used = min(levels, floor(log2(두께)) − 2, floor(log2(창 짧은 변)) − 1)``, 최소 1. 두께 = 마스크 내접
   지름(거리 변환 최대 × 2). 코어스 레벨의 전이 폭(≈ 2^L px)이 결함 두께의 절반을 넘으면 결함 **안쪽 저주파(톤)까지 대상으로
   빨려** 결함이 옅어진다 — 실측(밝기 250 결함 / 100 대상): 4px 스크래치에 levels 4 → 대비 70%, 두께 캡 → 85~88%. 이 캡이
   ``levels``를 "최대"로 만들고, 실제 쓴 값은 ``log.blend.levels_used``·``thickness_px``.
2. **창** = 마스크 bbox를 ``2**levels_used``만큼 넓힌 사각형(대상 안으로 클립). 창이 패치 캔버스보다 크면 캔버스 밖은 대상
   자신으로 채운다 — 거기선 소스 == 대상이라 라플라시안 차가 0이고 마스크도 0이므로 블렌딩에 영향이 없다.
3. 창의 소스·대상 라플라시안 피라미드 + 마스크(float 0/1) 가우시안 피라미드 → 레벨마다 ``L = Ls·m + Ld·(1−m)`` → 복원.
   ``pyrDown``이 마스크를 레벨마다 흐리므로 저주파(색·밝기)는 넓게, 고주파(텍스처)는 좁게 전이한다 — 알파 페더(모든 주파수를
   같은 폭으로)와 Poisson(그래디언트 도메인)의 중간.
4. **결과는 마스크 내부에만 쓴다.** 전이 대역의 바깥 절반은 버린다 — "마스크 밖 불변"이 blend 공통 계약이고(GT가 마스크
   기준), 바깥까지 바꾸면 GT ``diff`` 창(``dilate_px+2``) 밖의 픽셀이 라벨 없이 바뀐다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import cv2
import numpy as np

from anograft.core.recipe import MultibandBlendConfig
from anograft.core.registry import register
from anograft.core.stages.blend import common as C
from anograft.core.types import Context


def mask_thickness(mask: np.ndarray) -> float:
    """마스크 내접 지름(px) = 거리 변환 최대 × 2. 4px 선 → 4.0, 32×32 정사각형 → 32.0. 빈 마스크 → 0."""
    if not mask.any():
        return 0.0
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return 2.0 * float(dist.max())


def _cap(px: float, minus: int) -> int:
    return math.floor(math.log2(max(px, 1.0))) - minus


def levels_for(
    levels: int, thickness_px: float, window_shape: tuple[int, ...] | None = None
) -> int:
    """실제 레벨 수: ``min(levels, floor(log2(두께)) − 2, floor(log2(min(h, w))) − 1)``, 최소 1."""
    used = min(levels, _cap(thickness_px, 2))
    if window_shape is not None:
        used = min(used, _cap(min(window_shape[:2]), 1))
    return max(1, used)


def gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    g = [img]
    for _ in range(levels):
        g.append(cv2.pyrDown(g[-1]))
    return g


def laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """``levels``개의 라플라시안 + 마지막 가우시안(잔여). float32."""
    g = gaussian_pyramid(img, levels)
    lp = [g[i] - cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0])) for i in range(levels)]
    lp.append(g[levels])
    return lp


def collapse(lp: list[np.ndarray]) -> np.ndarray:
    img = lp[-1]
    for lap in reversed(lp[:-1]):
        img = cv2.pyrUp(img, dstsize=(lap.shape[1], lap.shape[0])) + lap
    return img


def blend_pyramids(src: np.ndarray, dst: np.ndarray, mask01: np.ndarray, levels: int) -> np.ndarray:
    """``src``·``dst`` HxWx3, ``mask01`` HxW float32 ∈ {0,1}. 반환 float32 HxWx3 (클립 전)."""
    ls = laplacian_pyramid(src.astype(np.float32), levels)
    ld = laplacian_pyramid(dst.astype(np.float32), levels)
    gm = gaussian_pyramid(mask01, levels)
    mixed = [
        s * m[..., None] + d * (1.0 - m[..., None]) for s, d, m in zip(ls, ld, gm, strict=True)
    ]
    return collapse(mixed)


@dataclass(frozen=True)
class Crop:
    """창(대상 좌표) 안에 소스·마스크를 대상 위에 얹은 것. ``ys/xs``가 대상 슬라이스."""

    ys: slice
    xs: slice
    src: np.ndarray  # 창 크기 HxWx3 — 패치 캔버스 안은 패치, 밖은 대상
    dst: np.ndarray  # 창 크기 HxWx3 (대상 원본)
    mask: np.ndarray  # 창 크기 HxW 0/255


def crop_window(inp: C.BlendInputs, pad: int) -> Crop | None:
    """마스크 bbox ± ``pad``(대상 안 클립) 창. 마스크가 창 안에 없으면 None."""
    x, y, w, h = cv2.boundingRect(inp.mask)
    if w == 0 or h == 0:
        return None
    ox, oy = inp.placement.offset
    hh, ww = inp.composite.shape[:2]
    x0, y0 = max(ox + x - pad, 0), max(oy + y - pad, 0)
    x1, y1 = min(ox + x + w + pad, ww), min(oy + y + h + pad, hh)
    if x1 <= x0 or y1 <= y0:
        return None
    dst = inp.composite[y0:y1, x0:x1]
    src = dst.copy()
    mask = np.zeros(dst.shape[:2], dtype=np.uint8)
    # 창 안에서 패치 캔버스가 겹치는 부분만 덮는다(캔버스가 대상 밖으로 걸쳐도 됨)
    win = C.canvas_window(dst.shape, inp.mask.shape, (ox - x0, oy - y0))
    if win.empty:
        return None
    src[win.ys, win.xs] = inp.patch[win.pys, win.pxs]
    mask[win.ys, win.xs] = inp.mask[win.pys, win.pxs]
    if not mask.any():
        return None
    return Crop(slice(y0, y1), slice(x0, x1), src, dst, mask)


def multiband_blend(inp: C.BlendInputs, levels: int) -> tuple[np.ndarray, dict[str, Any]]:
    """반환 ``(새 composite, 로그 조각)``. 마스크 밖은 바이트 단위로 불변."""
    out = inp.composite.copy()
    thickness = mask_thickness(inp.mask)
    tentative = levels_for(levels, thickness)
    crop = crop_window(inp, 2**tentative)
    if crop is None:
        return out, {"levels_used": 0, "thickness_px": thickness, "skipped": "창 안에 마스크 없음"}
    used = levels_for(tentative, thickness, crop.mask.shape)
    mixed = blend_pyramids(crop.src, crop.dst, (crop.mask > 0).astype(np.float32), used)
    inside = crop.mask > 0
    region = out[crop.ys, crop.xs]
    region[inside] = np.clip(np.rint(mixed[inside]), 0, 255).astype(np.uint8)
    h, w = crop.mask.shape
    return out, {"levels_used": used, "thickness_px": thickness, "window_px": [w, h]}


@register
class MultibandBlend:
    stage: ClassVar[str] = "blend"
    methods: ClassVar[tuple[str, ...]] = ("multiband",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: MultibandBlendConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def apply(self, ctx: Context) -> Context:
        inp = C.blend_inputs(ctx)
        if inp is None:
            return C.skipped(ctx, "multiband")
        out, info = multiband_blend(inp, self.cfg.levels)
        log = C.log_base("multiband", levels=self.cfg.levels, **info)
        return replace(ctx, composite=out).with_log("blend", log)

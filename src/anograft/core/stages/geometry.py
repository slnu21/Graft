"""2단계 기하 변환 — ``affine`` (설계 §6): 물리 축척 × scale, rotate, flip, elastic(alpha>0).

- 이미지는 ``INTER_LINEAR``, 마스크는 ``INTER_NEAREST`` 후 ``>127`` 이진화 — LINEAR로 돌린 마스크는 회색값이 생겨 GT를 오염시킨다.
- 회전으로 잘리지 않게 출력 캔버스 = 회전 bbox 크기. 캔버스 바깥은 이미지는 반사(``BORDER_REFLECT_101`` — 마스크 밖
  픽셀도 블렌딩 페더·Poisson 그래디언트에 쓰이므로 검은 모서리를 만들지 않는다), 마스크는 0.
- 무작위 소비 순서(고정): ``uniform(scale)`` → ``uniform(rotate)`` → (flip이면) ``random()`` ×2 → (elastic이면) ``normal`` 필드 2장.
- 결과 마스크 면적 < ``MIN_MASK_AREA``이면 ``patch=None`` + 경고 → 파이프라인이 그 결함을 건너뛴다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

import cv2
import numpy as np

from anograft.core.channels import binarize
from anograft.core.recipe import AffineGeometryConfig
from anograft.core.registry import register
from anograft.core.scale import physical_scale
from anograft.core.types import Context

MIN_MASK_AREA = 4  # px — 이보다 작은 마스크는 결함으로 의미가 없다


def rotated_canvas_size(w: int, h: int, scale: float, angle_deg: float) -> tuple[int, int]:
    """``w×h``를 ``scale``·``angle``로 변환했을 때 잘리지 않는 캔버스 ``(nw, nh)``."""
    rad = math.radians(angle_deg)
    c, s = abs(math.cos(rad)) * scale, abs(math.sin(rad)) * scale
    # cos(90°) ≈ 6e-17 같은 부동소수 잡음이 ceil을 한 칸 올리지 않게 먼저 반올림
    nw = max(1, math.ceil(round(w * c + h * s, 6)))
    nh = max(1, math.ceil(round(w * s + h * c, 6)))
    return nw, nh


def warp_affine(
    image: np.ndarray,
    mask: np.ndarray,
    scale: float,
    angle_deg: float,
    flip_h: bool = False,
    flip_v: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """flip → 회전+축척. 반환 ``(patch HxWx3, mask HxW 0/255)``, 크기 = 회전 bbox 캔버스."""
    if flip_h:
        image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
    if flip_v:
        image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
    h, w = mask.shape[:2]
    nw, nh = rotated_canvas_size(w, h, scale, angle_deg)
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle_deg, scale)
    m[0, 2] += (nw - 1) / 2.0 - (w - 1) / 2.0
    m[1, 2] += (nh - 1) / 2.0 - (h - 1) / 2.0
    patch = cv2.warpAffine(
        image, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    out_mask = cv2.warpAffine(
        mask, m, (nw, nh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return patch, binarize(out_mask)


def elastic_maps(
    shape: tuple[int, int], alpha: float, sigma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """``rng.normal`` 변위 필드 2장 → ``GaussianBlur(σ)`` → ``×alpha``. 반환 ``(map_x, map_y)`` float32 (cv2.remap용)."""
    h, w = shape
    dx = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    dy = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return xs + dx, ys + dy


def elastic_deform(
    image: np.ndarray, mask: np.ndarray, alpha: float, sigma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """이미지·마스크에 **같은** 변위 맵을 적용. 변위로 마스크가 캔버스 밖으로 밀리지 않게 여유를 두른다."""
    pad = elastic_pad(alpha, sigma)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    map_x, map_y = elastic_maps(mask.shape[:2], alpha, sigma, rng)
    out_img = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    out_mask = cv2.remap(
        mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return out_img, binarize(out_mask)


def elastic_pad(alpha: float, sigma: float) -> int:
    """변위의 대략 3σ. 단위 정규 노이즈를 σ로 블러하면 표준편차 ≈ 1/(2σ√π)."""
    if alpha <= 0:
        return 0
    std = alpha / (2.0 * sigma * math.sqrt(math.pi))
    return math.ceil(3.0 * std) + 1


@register
class AffineGeometry:
    stage: ClassVar[str] = "geometry"
    methods: ClassVar[tuple[str, ...]] = ("affine",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: AffineGeometryConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def apply(self, ctx: Context) -> Context:
        src = ctx.source
        if src is None:
            return ctx.with_log("geometry", {"method": "affine", "skipped": "source 없음"})
        cfg = self.cfg
        rng = ctx.rng

        phys = physical_scale(src.um_per_px, ctx.target.um_per_px)
        scale = phys.factor * float(rng.uniform(cfg.scale[0], cfg.scale[1]))
        angle = float(rng.uniform(cfg.rotate[0], cfg.rotate[1]))
        flip_h = flip_v = False
        if cfg.flip:
            flip_h = bool(rng.random() < 0.5)
            flip_v = bool(rng.random() < 0.5)

        patch, mask = warp_affine(src.image, src.mask, scale, angle, flip_h, flip_v)
        elastic_log: dict[str, Any] | None = None
        if cfg.elastic.alpha > 0:
            patch, mask = elastic_deform(patch, mask, cfg.elastic.alpha, cfg.elastic.sigma, rng)
            elastic_log = {"alpha": cfg.elastic.alpha, "sigma": cfg.elastic.sigma}

        area = int(np.count_nonzero(mask))
        log: dict[str, Any] = {
            "method": "affine",
            "physical_scale": phys.factor,
            "physical_applied": phys.applied,
            "scale": scale,
            "rotate": angle,
            "flip": [flip_h, flip_v],
            "elastic": elastic_log,
            "patch_shape": [int(mask.shape[0]), int(mask.shape[1])],
            "mask_area_px": area,
        }
        if phys.reason:
            log["physical_reason"] = phys.reason
        if area < MIN_MASK_AREA:
            log["failed"] = True
            log["reason"] = f"변환 후 마스크 면적 {area}px < {MIN_MASK_AREA}px"
            return (
                replace(ctx, patch=None, patch_mask=None)
                .warn(f"geometry: {src.id} {log['reason']}")
                .with_log("geometry", log)
            )
        return replace(ctx, patch=patch, patch_mask=mask).with_log("geometry", log)

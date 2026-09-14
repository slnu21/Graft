"""1단계 소스 — ``self-cut``(CutPaste, Li et al. 2021). 대상 이미지 자신에서 패치를 잘라 소스로 쓴다. 은행 불필요.

- ``rect``: 면적비 ``area_ratio``(이미지 면적 대비, 균등) · 종횡비 ``aspect``(로그균등)의 축 정렬 사각형.
- ``scar``: 폭 ``scar_width_px`` × 길이 ``scar_length_px``의 세로 띠 — 회전은 geometry(affine rotate)가 맡는다
  (CutPaste-Scar의 −45~45° 회전은 프리셋 ``rotate``로).
- ``mixed``: 결함마다 50/50.
- 자를 자리는 **ROI 안**(``ctx.roi``가 있으면 사각형 전체가 ROI 안이어야 함)에서 ``max_tries``번 시도, 실패하면 아무 자리
  (로그 ``roi_fallback: true``). 크롭은 사각형 ± ``margin_px``(이미지 안으로 클립) — 마스크는 사각형만이라 poisson 팽창·페더가
  잘리지 않는다.
- 색 지터(CutPaste ColorJitter 0.1): 밝기 → 대비 → 채도 → 색상 순, 각 ``uniform``. 흑백(3채널 동일) 대상은 채도·색상이
  항등이라 3채널 동일 불변식이 유지된다.
- rng 소비 순서(고정): ``random``(mixed일 때만) → 크기(rect: ``uniform``×2 · scar: ``integers``×2) → 자리(``integers``×2 ×
  시도) → 지터(``uniform``×4, 켜져 있을 때만).
- 로그 ``source = {method, source_id, class, class_id, mask_origin: "self-cut:<shape>", shape, cut_box, roi_fallback, jitter}``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

import cv2
import numpy as np

from anograft.core.recipe import ColorJitterConfig, SelfCutSourceConfig
from anograft.core.registry import register
from anograft.core.types import Context, DefectSource


def color_jitter(
    image: np.ndarray, rng: np.random.Generator, cfg: ColorJitterConfig
) -> tuple[np.ndarray, dict[str, float]]:
    """밝기·대비·채도·색상 지터 (BGR uint8 → BGR uint8). rng 소비 ``uniform``×4(항상 4회 — 값이 0인 항목도 소비해
    설정을 바꿔도 스트림 위치가 밀리지 않게)."""
    b = float(rng.uniform(1.0 - cfg.brightness, 1.0 + cfg.brightness))
    c = float(rng.uniform(1.0 - cfg.contrast, 1.0 + cfg.contrast))
    s = float(rng.uniform(1.0 - cfg.saturation, 1.0 + cfg.saturation))
    h = float(rng.uniform(-cfg.hue, cfg.hue)) * 180.0  # OpenCV H는 0..180
    x = image.astype(np.float32)
    if cfg.brightness > 0:
        x = x * b
    if cfg.contrast > 0:
        mean = float(cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).mean())
        x = (x - mean) * c + mean
    x8 = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    if cfg.saturation > 0 or cfg.hue > 0:
        hsv = cv2.cvtColor(x8, cv2.COLOR_BGR2HSV).astype(np.float32)
        if cfg.saturation > 0:
            hsv[..., 1] = np.clip(hsv[..., 1] * s, 0, 255)
        if cfg.hue > 0:
            hsv[..., 0] = np.mod(hsv[..., 0] + h, 180.0)
        x8 = cv2.cvtColor(np.clip(np.rint(hsv), 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return x8, {"brightness": b, "contrast": c, "saturation": s, "hue_deg": h}


def _skip(ctx: Context, reason: str) -> Context:
    return (
        replace(ctx, source=None)
        .warn(f"source: {reason}")
        .with_log("source", {"method": "self-cut", "skipped": True, "reason": reason})
    )


@register
class SelfCutSource:
    stage: ClassVar[str] = "source"
    methods: ClassVar[tuple[str, ...]] = ("self-cut",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: SelfCutSourceConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg
        ids: Mapping[str, int] | None = deps.get("class_ids")
        self.class_id: int = int(ids.get(cfg.cls, 0)) if ids else 0

    # --- 크기 ---------------------------------------------------------------

    def _size(
        self, shape: str, rng: np.random.Generator, h_img: int, w_img: int
    ) -> tuple[int, int]:
        cfg = self.cfg
        if shape == "rect":
            area = float(rng.uniform(cfg.area_ratio[0], cfg.area_ratio[1])) * h_img * w_img
            aspect = math.exp(float(rng.uniform(math.log(cfg.aspect[0]), math.log(cfg.aspect[1]))))
            w = round(math.sqrt(area * aspect))
            h = round(math.sqrt(area / aspect))
        else:
            w = int(rng.integers(cfg.scar_width_px[0], cfg.scar_width_px[1] + 1))
            h = int(rng.integers(cfg.scar_length_px[0], cfg.scar_length_px[1] + 1))
        # 이미지보다 크면 줄인다 (테두리 1px 남김)
        return max(1, min(h, h_img - 2)), max(1, min(w, w_img - 2))

    # --- 자리 ---------------------------------------------------------------

    def _place(
        self,
        rng: np.random.Generator,
        roi: np.ndarray | None,
        h_img: int,
        w_img: int,
        h: int,
        w: int,
    ) -> tuple[int, int, bool]:
        """(y, x, roi_fallback). ROI가 있으면 사각형 전체가 ROI 안인 자리를 ``max_tries``번 찾는다."""
        y = x = 0
        for _ in range(self.cfg.max_tries):
            y = int(rng.integers(0, h_img - h + 1))
            x = int(rng.integers(0, w_img - w + 1))
            if roi is None or bool(roi[y : y + h, x : x + w].all()):
                return y, x, False
        return y, x, roi is not None

    # --- apply --------------------------------------------------------------

    def apply(self, ctx: Context) -> Context:
        img = ctx.target.image
        h_img, w_img = img.shape[:2]
        if h_img < 4 or w_img < 4:
            return _skip(ctx, f"대상이 너무 작습니다 ({w_img}x{h_img})")
        cfg = self.cfg
        rng = ctx.rng
        shape = cfg.shape
        if shape == "mixed":
            shape = "rect" if float(rng.random()) < 0.5 else "scar"
        h, w = self._size(shape, rng, h_img, w_img)
        y, x, fallback = self._place(rng, ctx.roi, h_img, w_img, h, w)

        m = cfg.margin_px
        y0, x0 = max(0, y - m), max(0, x - m)
        y1, x1 = min(h_img, y + h + m), min(w_img, x + w + m)
        patch = img[y0:y1, x0:x1].copy()
        mask = np.zeros(patch.shape[:2], dtype=np.uint8)
        mask[y - y0 : y - y0 + h, x - x0 : x - x0 + w] = 255

        jitter_log: dict[str, float] | None = None
        if cfg.jitter.enabled:
            patch, jitter_log = color_jitter(patch, rng, cfg.jitter)

        k = len(ctx.defect_logs)
        src = DefectSource(
            id=f"self-cut/{shape}-{k:02d}",
            cls=cfg.cls,
            image=patch,
            mask=mask,
            um_per_px=ctx.target.um_per_px,  # 대상 자신 → 물리 축척 정합은 항등
            origin=ctx.target.path.as_posix(),
            mask_origin=f"self-cut:{shape}",
        )
        log: dict[str, Any] = {
            "method": "self-cut",
            "source_id": src.id,
            "class": cfg.cls,
            "class_id": self.class_id,
            "mask_origin": src.mask_origin,
            "shape": shape,
            "cut_box": [x, y, w, h],
            "roi_fallback": fallback,
            "jitter": jitter_log,
        }
        out = replace(ctx, source=src).with_log("source", log)
        if fallback:
            out = out.warn(
                f"source: self-cut {w}x{h} 이 ROI 안에 {cfg.max_tries}번 안에 안 들어가 아무 자리에서 잘랐습니다"
            )
        return out

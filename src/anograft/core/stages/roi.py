"""3단계(앞부분) ROI 스테이지 — 이미지당 1회, 결함 루프 밖 (설계 §6 roi). 순수 계산은 ``core/roi.py``.

- ``otsu``: 그레이 Otsu → ``invert: auto``는 테두리 접촉 비율이 낮은 쪽을 전경으로 → ``erode_px`` 침식.
- ``none``: 전체 허용 (테두리 ``margin_px``는 placement가 뺀다).
- ``mask_dir``: ``<path>/<대상 stem>.png``. core는 파일을 읽지 않으므로 로더는 ``deps["read_mask"]``(경로 → HxW uint8)로
  주입한다. 로더가 없거나 읽기에 실패하면 ROI None + 경고 (fail-soft) → placement가 그 대상을 건너뛴다.
- ``grabcut``(v0.4): ``core/roi.roi_grabcut`` — rect/otsu 초기화, 작업 해상도 ``work_px``, 시드는 대상 **파일 이름**의
  ``stable_seed``(폴더를 옮겨도 같은 ROI). GrabCut 이 실패하면 Otsu 로 대체하고 ``fallback`` 을 로그·경고에.

ROI 면적 0이면 ``roi``는 전부 False인 배열이고 경고를 남긴다 — 예외를 던지지 않는다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from anograft.core import roi as R
from anograft.core.recipe import GrabCutRoiConfig, MaskDirRoiConfig, NoneRoiConfig, OtsuRoiConfig
from anograft.core.registry import register
from anograft.core.seeds import stable_seed
from anograft.core.types import Context

MaskReader = Callable[[Path], np.ndarray]


def _finish(ctx: Context, roi: np.ndarray, log: dict[str, Any]) -> Context:
    area = int(np.count_nonzero(roi))
    h, w = roi.shape[:2]
    log["area_px"] = area
    log["area_ratio"] = area / float(h * w) if h * w else 0.0
    out = replace(ctx, roi=roi).with_log("roi", log)
    if area == 0:
        out = out.warn(f"roi: 면적 0 ({log['method']}) — 이 대상에는 결함을 놓을 수 없습니다")
    return out


@register
class OtsuRoi:
    stage: ClassVar[str] = "roi"
    methods: ClassVar[tuple[str, ...]] = ("otsu",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: OtsuRoiConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def apply(self, ctx: Context) -> Context:
        res = R.roi_otsu(ctx.target.image, self.cfg.invert, self.cfg.erode_px)
        log: dict[str, Any] = {
            "method": "otsu",
            "threshold": res.threshold,
            "invert": self.cfg.invert,
            "inverted": res.inverted,
            "border_touch": {"bright": res.border_touch_bright, "dark": res.border_touch_dark},
            "erode_px": self.cfg.erode_px,
            "area_before_erode_px": res.area_before_erode,
        }
        return _finish(ctx, res.roi, log)


@register
class NoneRoi:
    stage: ClassVar[str] = "roi"
    methods: ClassVar[tuple[str, ...]] = ("none",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: NoneRoiConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def apply(self, ctx: Context) -> Context:
        h, w = ctx.target.image.shape[:2]
        return _finish(ctx, R.roi_none((h, w)), {"method": "none"})


@register
class MaskDirRoi:
    stage: ClassVar[str] = "roi"
    methods: ClassVar[tuple[str, ...]] = ("mask_dir",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: MaskDirRoiConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.read_mask: MaskReader | None = deps.get("read_mask")

    def mask_path(self, ctx: Context) -> Path:
        return Path(self.cfg.path) / f"{ctx.target.path.stem}.png"

    def apply(self, ctx: Context) -> Context:
        path = self.mask_path(ctx)
        log: dict[str, Any] = {"method": "mask_dir", "path": path.as_posix()}
        h, w = ctx.target.image.shape[:2]
        if self.read_mask is None:
            reason = "deps['read_mask'] 로더가 없습니다"
        else:
            try:
                mask = self.read_mask(path)
                roi = R.roi_from_mask(mask, (h, w))
            except Exception as e:  # 로더 실패는 그 대상 skip으로 끝나야 한다 (fail-soft)
                reason = f"{type(e).__name__}: {e}"
            else:
                if mask.shape[:2] != (h, w):  # 미리보기 축소 등 — 사이드카에 남긴다
                    log["resized_from"] = [int(mask.shape[0]), int(mask.shape[1])]
                return _finish(ctx, roi, log)
        log["failed"] = True
        log["reason"] = reason
        return (
            replace(ctx, roi=None)
            .warn(f"roi: mask_dir 로드 실패 {path.as_posix()} — {reason}")
            .with_log("roi", log)
        )


@register
class GrabCutRoi:
    stage: ClassVar[str] = "roi"
    methods: ClassVar[tuple[str, ...]] = ("grabcut",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: GrabCutRoiConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def apply(self, ctx: Context) -> Context:
        cfg = self.cfg
        seed = stable_seed(ctx.target.path.name)
        res = R.roi_grabcut(
            ctx.target.image,
            init=cfg.init,
            invert=cfg.invert,
            rect_margin=cfg.rect_margin,
            iters=cfg.iters,
            work_px=cfg.work_px,
            erode_px=cfg.erode_px,
            seed=seed,
        )
        log: dict[str, Any] = {
            "method": "grabcut",
            "init": cfg.init,
            "init_used": res.init_used,
            "inverted": res.inverted,
            "rect_margin": cfg.rect_margin,
            "iters": cfg.iters,
            "work_scale": round(res.work_scale, 4),
            "seed": seed,
            "erode_px": cfg.erode_px,
            "area_before_erode_px": res.area_before_erode,
        }
        if res.fallback is not None:
            log["fallback"] = res.fallback
        out = _finish(ctx, res.roi, log)
        if res.fallback is not None:
            out = out.warn(f"roi: grabcut 실패 → otsu 로 대체 — {res.fallback}")
        return out

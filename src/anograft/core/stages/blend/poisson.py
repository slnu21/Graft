"""4단계 blend — ``poisson``: ``cv2.seamlessClone``(NORMAL / MIXED). 안 되면 alpha 폴백, ``log.blend.fallback=true``.

**풀이 영역 = 소스 마스크를 ``mask_dilate_px``만큼 팽창한 것.** OpenCV ``seamlessClone``은 내부에서 마스크를 3×3 커널로 3회
침식한 뒤 그래디언트를 섞고, 마스크 테두리 1px도 0으로 깎는다. 마스크가 결함에 딱 맞으면 결함 가장자리(그래디언트가 있는 곳)가
경계 조건(대상값 고정)에 먹혀 **얇은 결함은 통째로 사라진다**(실측: 팽창 0~2 → 대비 ≈0, 3 → 부분, 5 → 100%).
GT는 여전히 소스 마스크 — 팽창 링에서 다시 풀린 픽셀은 gtmask ``diff``/``union`` 정책이 잡는다.

호출 전 검사(예외를 기다리지 않는다):
- 팽창 마스크의 bbox가 대상 안에 1px 이상 여유로 들어가야 한다 — OpenCV는 ``dst(roi_d)``가 밖이면 assert.
- 팽창 마스크는 패치 캔버스 테두리 1px를 비운다(OpenCV가 어차피 깎으므로 먼저 깎아 bbox를 정확히 맞춘다).
- 마스크 면적 > 0.
그래도 ``cv2.error``가 나면 alpha 폴백(fail-soft).
**mask 인자는 복사본으로 넘긴다** — OpenCV가 제자리에서 망가뜨린다(아래 ``seamless_clone``).

정렬: OpenCV는 ``roi_s = boundingRect(mask)``, ``roi_d = Rect(p.x - w//2, p.y - h//2, w, h)``. 그래서
``p = (offset + roi_s 좌상단) + (w//2, h//2)``가 마스크를 정확히 같은 자리에 놓는다 — ``test_blend``가 변경 픽셀 bbox로 고정한다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

import cv2
import numpy as np

from anograft.core.recipe import PoissonBlendConfig
from anograft.core.registry import register
from anograft.core.stages.blend import common as C
from anograft.core.stages.harmonize import dilate_mask
from anograft.core.types import Context

_MODES = {"normal": cv2.NORMAL_CLONE, "mixed": cv2.MIXED_CLONE}


def solve_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """Poisson 풀이 영역: 소스 마스크 팽창 후 캔버스 테두리 1px 비움. 항상 새 배열."""
    m = dilate_mask(mask, dilate_px).copy()
    m[0, :] = m[-1, :] = 0
    m[:, 0] = m[:, -1] = 0
    return m


def solve_bbox(mask: np.ndarray, offset: tuple[int, int]) -> tuple[int, int, int, int]:
    """풀이 마스크의 대상 좌표 bbox (x, y, w, h)."""
    x, y, w, h = cv2.boundingRect(mask)
    return (offset[0] + int(x), offset[1] + int(y), int(w), int(h))


def precheck(
    mask: np.ndarray, bbox: tuple[int, int, int, int], target_shape: tuple[int, ...]
) -> str | None:
    """seamlessClone 호출이 안전한지. 안전하면 None, 아니면 폴백 사유."""
    if not np.any(mask):
        return "마스크 면적 0"
    if not C.bbox_inside(bbox, target_shape, 1):
        return "마스크 bbox가 대상 테두리에 닿음"
    return None


def clone_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, w, h = bbox
    return (x0 + w // 2, y0 + h // 2)


def seamless_clone(inp: C.BlendInputs, mask: np.ndarray, mode: str) -> np.ndarray:
    # 마스크는 반드시 복사해서 넘긴다 — OpenCV 4.14의 seamlessClone은 mask 인자를 제자리에서 망가뜨린다
    # (내부 copyMakeBorder가 같은 버퍼를 덮어 내부가 지워짐, 198px → 138px 실측). ctx.patch_mask가 오염되면 trace·GUI가 틀린다.
    m = np.ascontiguousarray(mask.copy())
    center = clone_center(solve_bbox(m, inp.placement.offset))
    return cv2.seamlessClone(inp.patch, inp.composite, m, center, _MODES[mode])


@register
class PoissonBlend:
    stage: ClassVar[str] = "blend"
    methods: ClassVar[tuple[str, ...]] = ("poisson",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: PoissonBlendConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def apply(self, ctx: Context) -> Context:
        inp = C.blend_inputs(ctx)
        if inp is None:
            return C.skipped(ctx, "poisson")
        cfg = self.cfg
        log = C.log_base(
            "poisson", mode=cfg.poisson_mode, mask_dilate_px=cfg.mask_dilate_px, fallback=False
        )
        mask = solve_mask(inp.mask, cfg.mask_dilate_px)
        reason = precheck(mask, solve_bbox(mask, inp.placement.offset), inp.composite.shape)
        out: np.ndarray | None = None
        if reason is None:
            try:
                out = seamless_clone(inp, mask, cfg.poisson_mode)
            except cv2.error as e:  # 솔버 실패도 그 결함을 죽이지 않는다
                reason = "cv2.error: " + str(e).strip().splitlines()[-1][:120]
        if out is None:
            out = C.alpha_blend(inp, self.cfg.feather_px)
            log.update(fallback=True, fallback_reason=reason, feather_px=self.cfg.feather_px)
        return replace(ctx, composite=out).with_log("blend", log)

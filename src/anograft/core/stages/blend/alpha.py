"""4단계 blend — ``alpha``: 마스크 안쪽 ``feather_px`` 폭 선형 페더로 ``composite*(1-a) + patch*a``.

경계 색이 안 맞으므로 alpha-paste 프리셋은 harmonize reinhard로 색까지 맞춘다. poisson의 폴백이기도 하다.
``opacity``(DRAEM β)가 설정되면 결함마다 ``rng.uniform(lo, hi)`` 1회를 소비해 ``a``에 곱한다 — null이면 rng를 건드리지
않는다(v0.1 프리셋 스트림 불변). 로그 ``opacity``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

from anograft.core.recipe import AlphaBlendConfig
from anograft.core.registry import register
from anograft.core.stages.blend import common as C
from anograft.core.types import Context


@register
class AlphaBlend:
    stage: ClassVar[str] = "blend"
    methods: ClassVar[tuple[str, ...]] = ("alpha",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: AlphaBlendConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def apply(self, ctx: Context) -> Context:
        inp = C.blend_inputs(ctx)
        if inp is None:
            return C.skipped(ctx, "alpha")
        opacity = 1.0
        if self.cfg.opacity is not None:
            opacity = float(ctx.rng.uniform(self.cfg.opacity[0], self.cfg.opacity[1]))
        out = C.alpha_blend(inp, self.cfg.feather_px, opacity)
        log = C.log_base("alpha", feather_px=self.cfg.feather_px)
        if self.cfg.opacity is not None:
            log["opacity"] = opacity
        return replace(ctx, composite=out).with_log("blend", log)

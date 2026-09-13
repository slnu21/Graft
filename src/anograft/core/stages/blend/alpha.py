"""4단계 blend — ``alpha``: 마스크 안쪽 ``feather_px`` 폭 선형 페더로 ``composite*(1-a) + patch*a``.

경계 색이 안 맞으므로 alpha-paste 프리셋은 harmonize reinhard로 색까지 맞춘다. poisson의 폴백이기도 하다.
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
        out = C.alpha_blend(inp, self.cfg.feather_px)
        return replace(ctx, composite=out).with_log(
            "blend", C.log_base("alpha", feather_px=self.cfg.feather_px)
        )

"""4단계 blend — ``paste``: 마스크 내부를 패치로 치환. CutPaste 기준선(hard-paste 프리셋 = 학습 실험의 대조군)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

from anograft.core.recipe import PasteBlendConfig
from anograft.core.registry import register
from anograft.core.stages.blend import common as C
from anograft.core.types import Context


@register
class HardPaste:
    stage: ClassVar[str] = "blend"
    methods: ClassVar[tuple[str, ...]] = ("paste",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: PasteBlendConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def apply(self, ctx: Context) -> Context:
        inp = C.blend_inputs(ctx)
        if inp is None:
            return C.skipped(ctx, "paste")
        return replace(ctx, composite=C.paste(inp)).with_log("blend", C.log_base("paste"))

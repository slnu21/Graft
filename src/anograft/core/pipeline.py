"""파이프라인 조립·실행 (설계 §6 루프 구조 · §7).

::

    ctx = Context.initial(image_rng(seed, i), target)
    ctx = roi(ctx)                                  # 이미지당 1회
    n   = rng.integers(min, max+1)
    for k in range(n):                              # 결함 루프
        ctx = source → geometry → placement → blend → harmonize
    ctx = replace(ctx, pre_degrade=ctx.composite)
    ctx = degrade → gtmask                          # 이미지당 1회

스테이지가 결함 하나를 포기하면(``source``/``patch``/``placement``가 None) 그 결함만 건너뛴다. 예외를 던지지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from anograft.core import registry
from anograft.core.channels import demote_from_bgr
from anograft.core.recipe import STAGE_KEYS, Recipe
from anograft.core.seeds import image_rng
from anograft.core.types import Context, GraftResult, TargetImage

# 결함 루프 스테이지와, 각 스테이지가 성공했을 때 채워야 하는 Context 필드.
_DEFECT_LOOP: tuple[tuple[str, str], ...] = (
    ("source", "source"),
    ("geometry", "patch"),
    ("placement", "placement"),
    ("blend", "composite"),
    ("harmonize", "composite"),
)


@dataclass(frozen=True)
class TraceStep:
    stage: str
    defect_index: int | None  # 결함 루프 안이면 k, 이미지 단계면 None
    ctx: Context


class Pipeline:
    """레시피로 조립된 스테이지 묶음. ``run_one``은 순수 함수처럼 동작한다(입력 같으면 결과 같음)."""

    def __init__(self, recipe: Recipe, stages: Mapping[str, registry.Stage]) -> None:
        missing = [k for k in (*STAGE_KEYS, "roi") if k not in stages]
        if missing:
            raise ValueError(f"스테이지가 빠졌습니다: {missing}")
        self.recipe = recipe
        self.stages = dict(stages)

    @classmethod
    def from_recipe(cls, recipe: Recipe, deps: Mapping[str, Any] | None = None) -> Pipeline:
        """레지스트리에서 각 스테이지의 method 구현을 찾아 조립한다. 미구현이면 ``StageNotImplementedError``."""
        deps = dict(deps or {})
        pipe = recipe.pipeline
        stages: dict[str, registry.Stage] = {}
        for key in STAGE_KEYS:
            stages[key] = registry.build(key, getattr(pipe, key), deps)
        stages["roi"] = registry.build("roi", pipe.placement.roi, deps)
        return cls(recipe, stages)

    # ------------------------------------------------------------------

    def run_one(self, target: TargetImage, index: int) -> GraftResult:
        result, _ = self._run(target, index, trace=False)
        return result

    def run_one_traced(
        self, target: TargetImage, index: int
    ) -> tuple[GraftResult, list[TraceStep]]:
        """스테이지별 중간 Context를 함께 돌려준다 — GUI 신호 체인·디버깅용."""
        return self._run(target, index, trace=True)

    def _run(
        self, target: TargetImage, index: int, *, trace: bool
    ) -> tuple[GraftResult, list[TraceStep]]:
        steps: list[TraceStep] = []
        rec = self.recipe
        ctx = Context.initial(image_rng(rec.seed, index), target)

        ctx = self.stages["roi"].apply(ctx)
        if trace:
            steps.append(TraceStep("roi", None, ctx))

        lo, hi = rec.output.defects_per_image
        n_defects = int(ctx.rng.integers(lo, hi + 1))
        n_ok = 0
        for k in range(n_defects):
            ctx = ctx.begin_defect()
            aborted = False
            for stage_key, must_set in _DEFECT_LOOP:
                ctx = self.stages[stage_key].apply(ctx)
                if trace:
                    steps.append(TraceStep(stage_key, k, ctx))
                if getattr(ctx, must_set) is None:
                    aborted = True
                    break
            if aborted:
                # 실패한 결함은 placed에 들어가지 않도록 placed_mask를 비운 채 봉인
                ctx = replace(ctx, placed_mask=None)
            else:
                n_ok += 1
            ctx = ctx.end_defect()

        ctx = replace(ctx, pre_degrade=ctx.composite)
        for stage_key in ("degrade", "gtmask"):
            ctx = self.stages[stage_key].apply(ctx)
            if trace:
                steps.append(TraceStep(stage_key, None, ctx))

        return self._finish(ctx, index, n_defects, n_ok), steps

    # ------------------------------------------------------------------

    def _finish(self, ctx: Context, index: int, n_defects: int, n_ok: int) -> GraftResult:
        h, w = ctx.target.image.shape[:2]
        gt = ctx.gt_mask if ctx.gt_mask is not None else np.zeros((h, w), dtype=np.uint8)
        status: str = "ok" if n_ok > 0 else "skipped"
        reason = None if n_ok > 0 else (ctx.warnings[-1] if ctx.warnings else "배치된 결함 없음")
        sidecar: dict[str, Any] = {
            "index": index,
            "seed": self.recipe.seed,
            "recipe": self.recipe.name,
            "target": {
                "file": str(ctx.target.path),
                "shape": [h, w, 1 if ctx.target.gray else 3],
                "gray": ctx.target.gray,
                "um_per_px": ctx.target.um_per_px,
            },
            "defects_requested": n_defects,
            "defects": [dict(d) for d in ctx.defect_logs],
            "degrade": dict(ctx.log.get("degrade", {})),
            "gtmask": dict(ctx.log.get("gtmask", {})),
            "warnings": list(ctx.warnings),
        }
        return GraftResult(
            index=index,
            status=status,  # type: ignore[arg-type]
            image=demote_from_bgr(ctx.composite, ctx.target.gray),
            gt_mask=gt,
            instances=ctx.instances,
            sidecar=sidecar,
            warnings=ctx.warnings,
            reason=reason,
        )

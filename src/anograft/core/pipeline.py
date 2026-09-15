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

from collections.abc import Mapping, Sequence
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


def skip_reason(warnings: Sequence[str]) -> str:
    """skipped 사유 한 줄. ROI 스테이지 경고(``roi:`` 접두)가 있으면 그것이 근본 원인이다 — placement 의
    "ROI 없음"은 그 결과라서 마지막 경고를 그대로 쓰면 원인이 가려진다(KNOWN-ISSUES #1). 없으면 마지막 경고."""
    for w in warnings:
        if w.startswith("roi:"):
            return w
    return warnings[-1] if warnings else "배치된 결함 없음"


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
        """레지스트리에서 각 스테이지의 method 구현을 찾아 조립한다. 미구현이면 ``StageNotImplementedError``.

        ``deps`` 계약 — core는 파일을 읽지 않으므로 호출자(``runner``·GUI·테스트)가 넣어 준다. 전부 선택이고
        없으면 해당 스테이지가 fail-soft로 건너뛰거나 폴백한다:

        - ``bank``: ``classes``·``by_class(cls)``를 가진 은행 (``bank.Bank`` 또는 ``Bank.from_sources``) → source
        - ``class_probs``: ``{cls: p}`` (``Recipe.class_probabilities(bank)``) → source 의 클래스 추첨
        - ``class_ids``: ``{cls: id}`` (``bank.class_ids`` — 은행 classes 순서) → source 로그 · gtmask 인스턴스
        - ``read_mask``: ``Path -> HxW uint8`` (``imgio.read_mask``) → roi ``mask_dir``
        """
        deps = dict(deps or {})
        pipe = recipe.pipeline
        stages: dict[str, registry.Stage] = {}
        for key in STAGE_KEYS:
            stages[key] = registry.build(key, getattr(pipe, key), deps)
        stages["roi"] = registry.build("roi", pipe.placement.roi, deps)
        return cls(recipe, stages)

    # ------------------------------------------------------------------

    def run_one(
        self, target: TargetImage, index: int, *, rng: np.random.Generator | None = None
    ) -> GraftResult:
        """이미지 ``index`` 한 장. ``rng``를 주면 그 스트림을 이어 쓴다 — 실행기가 ``image_rng(seed, i)``로 먼저
        대상을 뽑고(설계 §7) 같은 스트림을 넘기는 경로. 안 주면 여기서 ``image_rng(seed, index)``를 만든다."""
        result, _ = self._run(target, index, trace=False, rng=rng)
        return result

    def run_one_traced(
        self, target: TargetImage, index: int, *, rng: np.random.Generator | None = None
    ) -> tuple[GraftResult, list[TraceStep]]:
        """스테이지별 중간 Context를 함께 돌려준다 — GUI 신호 체인·디버깅용."""
        return self._run(target, index, trace=True, rng=rng)

    def _run(
        self,
        target: TargetImage,
        index: int,
        *,
        trace: bool,
        rng: np.random.Generator | None = None,
    ) -> tuple[GraftResult, list[TraceStep]]:
        steps: list[TraceStep] = []
        rec = self.recipe
        ctx = Context.initial(rng if rng is not None else image_rng(rec.seed, index), target)

        ctx = self.stages["roi"].apply(ctx)
        roi_log = dict(ctx.log.get("roi", {}))  # begin_defect()가 log를 비우므로 여기서 붙잡아 둔다
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

        return self._finish(ctx, index, n_defects, n_ok, roi_log), steps

    # ------------------------------------------------------------------

    def _finish(
        self, ctx: Context, index: int, n_defects: int, n_ok: int, roi_log: Mapping[str, Any]
    ) -> GraftResult:
        h, w = ctx.target.image.shape[:2]
        gt = ctx.gt_mask if ctx.gt_mask is not None else np.zeros((h, w), dtype=np.uint8)
        status: str = "ok" if n_ok > 0 else "skipped"
        reason = None if n_ok > 0 else skip_reason(ctx.warnings)
        defects: list[dict[str, Any]] = [dict(d) for d in ctx.defect_logs]
        for (
            inst
        ) in ctx.instances:  # 인스턴스 GT를 그 결함의 항목으로 되돌린다 (설계 §8.3 defects[k].gt)
            if 0 <= inst.defect_index < len(defects):
                defects[inst.defect_index]["gt"] = {
                    "area_px": inst.area_px,
                    "bbox": list(inst.bbox),
                }
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
            "roi": dict(roi_log),
            "defects_requested": n_defects,
            "defects": defects,
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

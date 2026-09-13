"""7단계 정답 마스크 — ``source`` · ``diff`` · ``union`` (설계 §6). 결함마다 **따로** 계산해 인스턴스로 남기고 전체 GT는 합집합.

- ``source`` = 배치된 소스 마스크(``PlacedDefect.mask``).
- ``diff`` = ``max_c |pre_degrade − target.image| > diff_threshold`` 중 **그 결함의 소스 마스크를 ``dilate_px+2``만큼 팽창한 영역 안**만
  (먼 곳의 잡음·이웃 결함 제외).
- ``union`` = 둘의 합. 마지막에 ``dilate_px`` 팽창. Poisson은 마스크 내부를 다시 풀어 저대비 부분이 사라지므로(소스 마스크 과라벨)
  ``union``이 기본.
- ``class_id``는 ``deps["class_ids"]``(은행 classes 순서). 없으면 등장 클래스의 정렬 순서로 매기고 로그에 출처를 남긴다.
- 정책 결과 면적이 0인 결함(diff에서 변화 없음)은 인스턴스를 만들지 않고 경고 — writer가 빈 박스를 쓰지 않게.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, ClassVar

import cv2
import numpy as np

from anograft.core.recipe import GtMaskConfig
from anograft.core.registry import register
from anograft.core.stages.harmonize import dilate_mask
from anograft.core.types import Context, Instance, PlacedDefect


def change_map(before: np.ndarray, after: np.ndarray, threshold: int) -> np.ndarray:
    """채널 최대 절대차 > threshold (bool HxW)."""
    diff = np.abs(before.astype(np.int16) - after.astype(np.int16))
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    return diff > threshold


def instance_mask(
    policy: str, source: np.ndarray, changed: np.ndarray | None, dilate_px: int
) -> np.ndarray:
    """정책 하나를 결함 하나에 적용한 0/255 마스크."""
    src = source > 0
    if policy == "source":
        m = src
    else:
        assert changed is not None
        region = dilate_mask(source, dilate_px + 2) > 0
        local = changed & region
        m = local if policy == "diff" else (src | local)
    out = (m.astype(np.uint8)) * 255
    return dilate_mask(out, dilate_px)


def build_instance(p: PlacedDefect, mask: np.ndarray, class_id: int) -> Instance:
    x, y, w, h = cv2.boundingRect(mask)
    return Instance(
        cls=p.cls,
        class_id=class_id,
        mask=mask,
        bbox=(int(x), int(y), int(w), int(h)),
        area_px=int(np.count_nonzero(mask)),
        defect_index=p.defect_index,
    )


@register
class GtMask:
    stage: ClassVar[str] = "gtmask"
    methods: ClassVar[tuple[str, ...]] = ("source", "diff", "union")
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: GtMaskConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg
        ids = deps.get("class_ids")
        self.class_ids: dict[str, int] | None = dict(ids) if ids is not None else None

    def _ids(self, placed: tuple[PlacedDefect, ...]) -> tuple[dict[str, int], str]:
        if self.class_ids is not None:
            return self.class_ids, "deps"
        names = sorted({p.cls for p in placed})
        return {c: i for i, c in enumerate(names)}, "placed-sorted"

    def apply(self, ctx: Context) -> Context:
        cfg = self.cfg
        h, w = ctx.target.image.shape[:2]
        log: dict[str, Any] = {
            "policy": cfg.policy,
            "diff_threshold": cfg.diff_threshold,
            "dilate_px": cfg.dilate_px,
        }
        changed: np.ndarray | None = None
        if cfg.policy != "source":
            before = ctx.pre_degrade if ctx.pre_degrade is not None else ctx.composite
            changed = change_map(ctx.target.image, before, cfg.diff_threshold)
        ids, ids_source = self._ids(ctx.placed)
        log["class_ids_source"] = ids_source

        out = ctx
        instances: list[Instance] = []
        union = np.zeros((h, w), dtype=np.uint8)
        for p in ctx.placed:
            m = instance_mask(cfg.policy, p.mask, changed, cfg.dilate_px)
            if not np.any(m):
                out = out.warn(
                    f"gtmask: {p.source_id} 정책 '{cfg.policy}' 결과 면적 0 — 인스턴스 제외"
                )
                continue
            if p.cls not in ids:
                out = out.warn(f"gtmask: 클래스 '{p.cls}'의 id를 모릅니다 — 인스턴스 제외")
                continue
            instances.append(build_instance(p, m, ids[p.cls]))
            union = np.maximum(union, m)
        log["instances"] = [
            {
                "defect_index": i.defect_index,
                "class": i.cls,
                "class_id": i.class_id,
                "area_px": i.area_px,
                "bbox": list(i.bbox),
            }
            for i in instances
        ]
        log["area_px_total"] = int(np.count_nonzero(union))
        return replace(out, gt_mask=union, instances=tuple(instances)).with_log("gtmask", log)

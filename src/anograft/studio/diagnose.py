"""미리보기 한 장에 붙는 진단 — **Qt 없음**. Qt 스튜디오 탭과 웹 미리보기 화면이 **같은 문장**을 보게 한다.

원래 `gui/studio/tab.py` 의 private 헬퍼였다(`_source_ids`·`_low_confidence_sources`·`_flipped_instances`·
`_fit_warnings`·ROI 꺼내기). 웹이 같은 판단을 다시 구현하면 두 화면이 조용히 갈리므로 여기로 올렸다
— 규약대로 **판단은 순수 계층에, 화면은 부르기만** 한다.

전부 "미리보기 결과 하나 + `Prepared`" 만 보고, 없으면 빈 목록(fail-soft). 미리보기는 긴 변 1024
축소본이라 각도·폭은 **근사**다 — 그래서 여기 문장은 "의심"·"빠듯"까지만 말하고 채택을 바꾸지 않는다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from anograft import runner
from anograft.core import recipe as R
from anograft.core.appearance import flipped_instances
from anograft.core.channels import promote_to_bgr
from anograft.core.pipeline import TraceStep
from anograft.core.types import GraftResult


@dataclass(frozen=True)
class DefectLine:
    """이 장에 붙은 결함 하나 — 사이드카에 이미 있는 것만(새로 계산하지 않는다)."""

    cls: str
    source_id: str
    blend: str


def defect_lines(result: GraftResult) -> list[DefectLine]:
    """사이드카의 결함 목록(``gt`` 가 붙은 것 = 실제로 들어간 것)."""
    out: list[DefectLine] = []
    for d in result.sidecar.get("defects", []):
        if "gt" not in d:
            continue
        src = d.get("source", {})
        out.append(
            DefectLine(
                cls=str(src.get("class", "")),
                source_id=str(src.get("source_id", "")),
                blend=str(d.get("blend", {}).get("method", "")),
            )
        )
    return out


def source_ids(result: GraftResult) -> list[str]:
    return [d.source_id for d in defect_lines(result)]


def low_confidence_sources(prep: runner.Prepared | None, result: GraftResult) -> list[str]:
    """이 장에 쓰인 조각 중 ``confidence < 0.5``(KNOWN-ISSUES #3) — 그림이 그럴듯해도 마스크가 헐거울 수 있다."""
    if prep is None or prep.bank is None:
        return []
    low = {s.id for s in prep.bank.low_confidence()}
    return [sid for sid in source_ids(result) if sid in low]


def flipped(prep: runner.Prepared | None, result: GraftResult) -> list[int]:
    """조명이 실제 클래스 방향과 반대(> 90°)인 인스턴스 번호 — 검수 탭 '조명 뒤집힘 의심'을 미리보기에서.

    은행에 방향이 유의한 클래스가 없으면 빈 목록. 축소본이라 각도는 근사다.
    """
    if prep is None or prep.recipe.bankless or result.status != "ok" or not result.instances:
        return []
    real = prep.bank.real_lighting_direction()
    if not real:
        return []
    import cv2

    gray = cv2.cvtColor(promote_to_bgr(result.image), cv2.COLOR_BGR2GRAY)
    return flipped_instances(gray, [(i.cls, i.mask) for i in result.instances], real)


def roi_of(steps: Sequence[TraceStep]) -> np.ndarray | None:
    """추적의 첫 단계가 ROI면 그 허용 영역 — 캔버스 오버레이·배치 가능성 진단이 같은 것을 본다."""
    if steps and steps[0].stage == "roi":
        return steps[0].ctx.roi
    return None


def fit_for_preview(
    prep: runner.Prepared | None,
    recipe: R.Recipe,
    roi: np.ndarray | None,
    scale: float,
    target_name: str,
) -> Any | None:
    """이 바탕의 허용 영역 최대 폭 vs 클래스별 패치 폭(``runner.fit_from_widths``).

    미리보기 ROI 를 축소 배율로 나눠 **원본 px** 로 되돌린 뒤 묻는다 — 그래야 축소본에서 본 진단이
    원본 해상도 ``run`` 에 대해 맞는 말이 된다.
    """
    if prep is None or roi is None:
        return None
    from anograft.core.stages.placement import roi_max_width

    margin = int(getattr(recipe.pipeline.placement, "margin_px", 0))
    s = max(scale, 1e-6)
    width = roi_max_width(roi, margin) / s
    short_side = float(min(roi.shape[:2])) / s
    return runner.fit_from_widths(
        prep, [(target_name, round(width, 1))], target_short_side=short_side
    )


def fit_warnings(fit: Any | None) -> list[str]:
    """배치 가능성 경고(조각이 모자람 · 허용 영역이 빠듯/불가) — 없으면 빈 목록."""
    if fit is None:
        return []
    return [w for w in (fit.source_warning(), fit.warning()) if w]

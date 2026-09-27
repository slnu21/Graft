"""신뢰도 게이트 — **사람 없이 받아들여도 되는 마스크인가** (설계 §3 `gate`, T6).

판정은 한 줄이지만 **두 곳이 같은 답을 봐야 한다**:

- `bank` 임포트 게이트 — `bank import-yolo --min-confidence` (박스→마스크 추정의 타당성이 낮으면 은행에
  넣지 않는다). 강제는 `BankWriter.add` 한 지점이다(`holdout.txt` 거부와 같은 자리).
- `loop` 자동 편입 — `loop auto` 가 확신 높은 예측을 사람 큐를 거치지 않고 편입한다.

둘이 갈리면 "은행이 거부한 조각을 루프가 자동으로 넣는" 모순이 생긴다. `core` 에 두는 이유는
`core.classes` 와 같이 **방향뿐**이다: `bank`·`loop` 는 `core` 를 import 하지만 그 반대는 안 된다.

점수의 출처는 `bank.mask_from_box.mask_confidence`(0~1 타당성 + flags)다 — 여기서는 그 값을 **쓰는
규칙**만 정한다. 임계값은 코드가 정하지 않는다(기본 0 = 제한 없음, 설계 §8 확인 게이트).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: 게이트를 끈 값. 0 이면 "신뢰도로 막지 않는다" — `confidence` 가 없는 사람 마스크와 같은 취급이다.
OFF = 0.0


def gate(confidence: float | None, threshold: float) -> bool:
    """자동으로 은행에 넣어도 되는가.

    ``confidence`` 가 ``None`` 이면 **사람이 그린 정확한 마스크**라는 뜻이라 통과시킨다
    (`mask_origin` 이 ``png``·``manual:*`` 인 경우 — 은행 메타에서 confidence 는 그때 비어 있다).
    ``NaN`` 은 "재려다 실패했다"이므로 **거부**한다(모르는 것을 통과시키면 게이트가 있으나 마나다).
    """
    if confidence is None:
        return True
    if math.isnan(confidence):
        return False
    return confidence >= threshold


def gate_reason(confidence: float | None, threshold: float) -> str:
    """통과·거부의 **사유 한 줄**. 조용히 빠지는 조각을 만들지 않기 위한 것이다(로그·요약이 이걸 쓴다).

    순서가 `gate` 와 같아야 한다 — **NaN 은 게이트를 꺼도 거부**이므로 "꺼져 있습니다"보다 먼저 답한다
    (거부해 놓고 "게이트가 꺼져 있다"고 말하면 사유가 자기모순이 된다).
    """
    if confidence is None:
        return "사람이 그린 마스크입니다(추정 점수 없음)"
    if math.isnan(confidence):
        # 크기 불일치·읽기 실패로 타당성을 못 쟀다는 뜻이다 — 문턱과 무관하게 받지 않는다
        return "추정 마스크를 재지 못했습니다(크기 불일치·읽기 실패) — 모르는 것은 받지 않습니다"
    if threshold <= OFF:
        return "신뢰도 게이트가 꺼져 있습니다"
    if confidence >= threshold:
        return f"추정 마스크 신뢰도 {confidence:.2f} ≥ {threshold:.2f}"
    return f"추정 마스크 신뢰도 {confidence:.2f} < {threshold:.2f}"


def partition_by_gate(
    items: Sequence[tuple[str, float | None]], threshold: float
) -> tuple[list[str], list[str]]:
    """``(자동 편입, 사람 큐)`` 로 가른다."""
    auto: list[str] = []
    queue: list[str] = []
    for item_id, conf in items:
        (auto if gate(conf, threshold) else queue).append(item_id)
    return auto, queue


__all__ = ["OFF", "gate", "gate_reason", "partition_by_gate"]

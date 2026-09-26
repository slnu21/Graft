"""클래스 목록 규약 — **미분류**(`__unsorted__`)와 class id 불변식 (설계 §2b.5(2)·§6.7, T15).

`classes` 순서가 곧 class id 이고 그게 출력 `data.yaml` 순서다. 그래서 목록을 만지는 규칙은 한 곳에 있어야
한다 — 보관함(`bank`)·레시피(`core.recipe`)·writer(`io.writers`)가 같은 답을 봐야 하기 때문이다.
`core` 에 두는 이유는 방향뿐이다: `bank` 는 `core` 를 import 하지만 그 반대는 안 된다.
"""

from __future__ import annotations

from collections.abc import Sequence

#: 아직 이름을 주지 않은 결함 조각들이 사는 클래스(화면 용어 **미분류**).
#: 합성에 쓰이지 않고 writer 출력에도 나가지 않는다 — 사람이 이름을 주면(`bank promote`) 그때부터 쓰인다.
UNSORTED = "__unsorted__"


def is_unsorted(cls: str) -> bool:
    """미분류 클래스인가 — 합성·출력에서 빼는 판정이 이 한 곳을 지난다."""
    return cls == UNSORTED


def order_classes(classes: Sequence[str]) -> list[str]:
    """미분류를 **항상 맨 끝으로**. 이 불변식이 class id 를 지킨다.

    미분류가 가운데 있으면 그것을 빼는 순간 뒤 클래스들의 id 가 밀린다(= 기존 모델과 비호환, §6.7).
    끝에 있으면 빼도 아무 id 가 움직이지 않으므로 "출력에서 제외"가 꼬리를 자르는 일이 된다.
    """
    named = [c for c in classes if not is_unsorted(c)]
    return named + [c for c in classes if is_unsorted(c)]


def named_classes(classes: Sequence[str]) -> list[str]:
    """합성·writer 가 쓰는 클래스 — 미분류를 뺀 것(순서 그대로)."""
    return [c for c in classes if not is_unsorted(c)]


__all__ = ["UNSORTED", "is_unsorted", "named_classes", "order_classes"]

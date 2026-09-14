"""1단계 소스 선택 — ``bank`` (설계 §6). 결함 루프마다 클래스 → 소스 순으로 뽑는다.

- 클래스는 ``deps["class_probs"]``(``Recipe.class_probabilities(bank)``, {cls: p})로 ``rng.choice``. 없으면
  ``cfg.classes``(없으면 은행 전체) 균등.
- 소스는 ``deps["bank"].by_class(cls)``(이름 정렬 — 은행 로더가 보장)에서 ``rng.integers(n)``.
- 무작위 소비 순서(고정): ``choice(클래스)`` → ``integers(소스)``. 클래스가 하나뿐이어도 ``choice``를 소비해
  클래스 수가 바뀌어도 스트림 위치가 밀리지 않게 한다.
- 은행이 없거나(``deps["bank"]`` 부재) 그 클래스에 소스가 0개면 ``source=None`` + 경고 → 파이프라인이 이 결함을 건너뛴다.
  예외를 던지지 않는다(fail-soft — 레지스트리 스모크 테스트가 deps 없이 apply를 부른다).
- 로그 ``source = {method, source_id, class, class_id, mask_origin}`` — 사이드카 ``defects[k].source``, manifest의
  ``classes``·``source_ids`` 열이 여기서 나온다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from anograft.core.recipe import BankSourceConfig
from anograft.core.registry import register
from anograft.core.types import Context, DefectSource


@runtime_checkable
class SourceBank(Protocol):
    """이 스테이지가 은행에 요구하는 최소 인터페이스 (``bank.Bank``가 만족; 테스트는 ``Bank.from_sources``)."""

    @property
    def classes(self) -> Sequence[str]: ...

    def by_class(self, cls: str) -> Sequence[DefectSource]: ...


def _skip(ctx: Context, reason: str) -> Context:
    return (
        replace(ctx, source=None)
        .warn(f"source: {reason}")
        .with_log("source", {"method": "bank", "skipped": True, "reason": reason})
    )


@register
class BankSource:
    stage: ClassVar[str] = "source"
    methods: ClassVar[tuple[str, ...]] = ("bank",)
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: BankSourceConfig, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.bank: SourceBank | None = deps.get("bank")
        probs: Mapping[str, float] | None = deps.get("class_probs")
        ids: Mapping[str, int] | None = deps.get("class_ids")
        if probs is None:
            classes = (
                list(cfg.classes)
                if cfg.classes is not None
                else (list(self.bank.classes) if self.bank is not None else [])
            )
            probs = {c: 1.0 / len(classes) for c in classes} if classes else {}
        self.classes: list[str] = list(probs)
        total = float(sum(probs.values()))
        self.probs: np.ndarray = (
            np.array([probs[c] / total for c in self.classes], dtype=np.float64)
            if total > 0
            else np.zeros(len(self.classes))
        )
        self.class_ids: dict[str, int] = (
            dict(ids) if ids is not None else {c: i for i, c in enumerate(self.classes)}
        )

    def apply(self, ctx: Context) -> Context:
        if self.bank is None:
            return _skip(ctx, "deps['bank'] 이 없습니다")
        if not self.classes or float(self.probs.sum()) <= 0:
            return _skip(ctx, "뽑을 클래스가 없습니다")
        cls = self.classes[int(ctx.rng.choice(len(self.classes), p=self.probs))]
        sources = self.bank.by_class(cls)
        if not sources:
            return _skip(ctx, f"클래스 '{cls}' 소스가 0개")
        src = sources[int(ctx.rng.integers(len(sources)))]
        log = {
            "method": "bank",
            "source_id": src.id,
            "class": src.cls,
            "class_id": self.class_ids.get(src.cls, -1),
            "mask_origin": src.mask_origin,
        }
        return replace(ctx, source=src).with_log("source", log)

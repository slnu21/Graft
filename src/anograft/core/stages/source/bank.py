"""1단계 소스 선택 — ``bank`` (설계 §6). 결함 루프마다 클래스 → 소스 순으로 뽑는다.

- 클래스는 ``deps["class_probs"]``(``Recipe.class_probabilities(bank)``, {cls: p})로 ``rng.choice``. 없으면
  ``cfg.classes``(없으면 은행 전체) 균등.
- 소스는 ``deps["bank"].by_class(cls)``(이름 정렬 — 은행 로더가 보장)에서 ``rng.integers(n)``. ``cfg.tags`` 가 있으면
  그 목록을 태그로 거른 뒤(순서 유지) 뽑는다 — 풀이 0개면 skip 사유에 필터를 적는다.
- 무작위 소비 순서(고정): ``choice(클래스)`` → ``integers(소스)``. 클래스가 하나뿐이어도 ``choice``를 소비해
  클래스 수가 바뀌어도 스트림 위치가 밀리지 않게 한다.
- 은행이 없거나(``deps["bank"]`` 부재) 그 클래스에 소스가 0개면 ``source=None`` + 경고 → 파이프라인이 이 결함을 건너뛴다.
  예외를 던지지 않는다(fail-soft — 레지스트리 스모크 테스트가 deps 없이 apply를 부른다).
- 로그 ``source = {method, source_id, class, class_id, mask_origin, tags}`` — 사이드카 ``defects[k].source``, manifest의
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
        # 태그 필터가 켜져 있으면 클래스별 풀을 미리 거른다(은행은 prepare 뒤 바뀌지 않는다). 꺼져 있으면 None.
        self.pool: dict[str, list[DefectSource]] | None = None
        if cfg.tags.active and self.bank is not None:
            self.pool = {
                c: [s for s in self.bank.by_class(c) if cfg.tags.accepts(s.tags)]
                for c in self.classes
            }

    def _image_class(self, ctx: Context) -> str | None:
        """``single_class_per_image``: 이 이미지에서 이미 뽑힌(봉인된 첫 결함 로그의) 클래스. 없으면 None → 추첨."""
        if not getattr(self.cfg, "single_class_per_image", False):
            return None
        for d in ctx.defect_logs:
            c = (d.get("source") or {}).get("class")
            if isinstance(c, str) and c in self.classes:
                return c
        return None

    def apply(self, ctx: Context) -> Context:
        if self.bank is None:
            return _skip(ctx, "deps['bank'] 이 없습니다")
        if not self.classes or float(self.probs.sum()) <= 0:
            return _skip(ctx, "뽑을 클래스가 없습니다")
        cls = self._image_class(ctx)
        if cls is None:
            cls = self.classes[int(ctx.rng.choice(len(self.classes), p=self.probs))]
        if self.pool is not None:
            sources = self.pool.get(cls, [])
            if not sources:
                return _skip(
                    ctx,
                    f"클래스 '{cls}' 소스가 0개 (태그 필터 {self.cfg.tags.describe()} 후, "
                    f"원래 {len(self.bank.by_class(cls))}개)",
                )
        else:
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
            "tags": list(src.tags),
        }
        return replace(ctx, source=src).with_log("source", log)

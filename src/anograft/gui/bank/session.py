"""은행 세션 — **Qt 없음**. 은행 하나를 열어 소스 목록을 필터·정렬하고, 삭제·마스크 교체를 파일 계층(``BankWriter``)에 위임한다
(v0.7 은행 탭).

- 읽기는 ``Bank.load``(임포터·runner 와 같은 로더) — 소스 순서·마스크 이진화·경고가 같다.
- ``rows()`` 는 표시용 요약(``SourceRow``), ``filtered(...)`` 가 클래스·태그·저신뢰·추정·검색·정렬을 적용한다(순수 함수).
- ``delete(ids)`` 는 세 파일(png·mask.png·json)을 지우고 ``bank.yaml`` 의 ``classes`` 는 **유지**(id 순서 = class id — 출력
  ``data.yaml`` 이 흔들리면 안 된다), imports 이력에 남긴다. ``replace_mask(id, mask, tool)`` 는 라벨 탭에서 다듬은 마스크를 같은
  id 에 덮어쓴다(``mask_origin: manual:<tool>``, 추정 점수는 지움). 둘 다 끝나면 ``reload()`` 로 메모리 은행을 다시 읽는다.
- 위젯은 이 클래스의 상태만 그리고, 클릭을 여기 메서드 호출로 바꾼다(테스트는 여기서 끝난다).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from anograft.bank import Bank
from anograft.bank.bank import ESTIMATED_PREFIX, BankError, is_low_confidence
from anograft.bank.importers.common import BankWriter
from anograft.core.types import DefectSource

SORT_KEYS: tuple[str, ...] = ("id", "confidence", "area", "class")


class BankSessionError(ValueError):
    pass


@dataclass(frozen=True)
class SourceRow:
    """목록 한 줄 — ``DefectSource`` 의 표시용 요약."""

    id: str  # "<class>/<name>"
    cls: str
    name: str  # id 의 뒷부분(파일 stem)
    area_px: int
    mask_origin: str
    estimated: bool
    confidence: float | None
    low_confidence: bool
    flags: tuple[str, ...]
    tags: tuple[str, ...]
    um_per_px: float | None
    origin: str
    size: tuple[int, int]  # 크롭 (h, w)


def row_of(s: DefectSource) -> SourceRow:
    return SourceRow(
        id=s.id,
        cls=s.cls,
        name=s.id.split("/", 1)[1] if "/" in s.id else s.id,
        area_px=int(np.count_nonzero(s.mask)),
        mask_origin=s.mask_origin,
        estimated=s.mask_origin.startswith(ESTIMATED_PREFIX),
        confidence=s.confidence,
        low_confidence=is_low_confidence(s),
        flags=tuple(s.flags),
        tags=tuple(s.tags),
        um_per_px=s.um_per_px,
        origin=s.origin,
        size=(int(s.mask.shape[0]), int(s.mask.shape[1])),
    )


def filter_rows(
    rows: Sequence[SourceRow],
    *,
    cls: str | None = None,
    tag: str | None = None,
    only_low: bool = False,
    only_estimated: bool = False,
    text: str = "",
    sort: str = "id",
    descending: bool = False,
) -> list[SourceRow]:
    """순수 필터·정렬. ``sort`` 는 ``SORT_KEYS`` — confidence 는 None 을 끝으로."""
    if sort not in SORT_KEYS:
        raise BankSessionError(f"정렬 키 {sort!r} (선택: {', '.join(SORT_KEYS)})")
    q = text.strip().lower()
    out = [
        r
        for r in rows
        if (cls is None or r.cls == cls)
        and (tag is None or tag in r.tags)
        and (not only_low or r.low_confidence)
        and (not only_estimated or r.estimated)
        and (
            not q
            or q in r.id.lower()
            or q in r.origin.lower()
            or any(q in t.lower() for t in r.tags)
        )
    ]
    if sort == "confidence":
        out.sort(
            key=lambda r: (r.confidence is None, r.confidence or 0.0, r.id), reverse=descending
        )
        if descending:  # None 은 항상 끝
            out.sort(key=lambda r: r.confidence is None)
    elif sort == "area":
        out.sort(key=lambda r: (r.area_px, r.id), reverse=descending)
    elif sort == "class":
        out.sort(key=lambda r: (r.cls, r.id), reverse=descending)
    else:
        out.sort(key=lambda r: r.id, reverse=descending)
    return out


class BankSession:
    def __init__(self) -> None:
        self.root: Path | None = None
        self.bank: Bank | None = None
        self.warnings: list[str] = []
        self._rows: list[SourceRow] = []

    # ------------------------------------------------------------------ 열기

    @property
    def loaded(self) -> bool:
        return self.bank is not None

    def load(self, root: str | Path) -> Bank:
        """``Bank.load`` — 은행이 아니면 ``BankSessionError``(fail-soft 경고는 ``warnings``)."""
        try:
            bank = Bank.load(root)
        except BankError as e:
            raise BankSessionError(str(e)) from e
        self.root, self.bank = Path(root), bank
        self.warnings = list(bank.warnings)
        self._rows = [row_of(s) for s in bank.sources()]
        return bank

    def reload(self) -> Bank:
        if self.root is None:
            raise BankSessionError("은행을 먼저 여세요")
        return self.load(self.root)

    def close(self) -> None:
        self.root, self.bank, self.warnings, self._rows = None, None, [], []

    # ------------------------------------------------------------------ 조회

    def rows(self) -> list[SourceRow]:
        return list(self._rows)

    def filtered(self, **kw) -> list[SourceRow]:
        return filter_rows(self._rows, **kw)

    def source(self, source_id: str) -> DefectSource:
        assert self.bank is not None
        for s in self.bank.sources():
            if s.id == source_id:
                return s
        raise BankSessionError(f"소스가 없습니다: {source_id}")

    def classes(self) -> list[str]:
        return list(self.bank.classes) if self.bank is not None else []

    def tags(self) -> list[str]:
        return list(self.bank.tag_counts()) if self.bank is not None else []

    def summary_text(self) -> str:
        if self.bank is None:
            return "은행 없음"
        b = self.bank
        low = len(b.low_confidence())
        est = sum(r.estimated for r in b.summary())
        parts = [f"{b.name}: 소스 {len(b)} · 클래스 {len(b.classes)}"]
        if est:
            parts.append(f"추정 {est}" + (f" (저신뢰 {low})" if low else ""))
        if b.no_pitch_count():
            parts.append(f"um_per_px 미지정 {b.no_pitch_count()}")
        directional = self.directional_classes()
        if directional:
            parts.append(
                "조명 의존(lightR) "
                + ", ".join(f"{c} {r:.2f}" for c, r in directional)
                + " → dent-graft"
            )
        return " · ".join(parts)

    def directional_classes(self) -> list[tuple[str, float]]:
        """조명 일관성 R ≥ LIGHT_REAL_MIN 인 클래스(실제 소스 n ≥ 3) — ±180/flip 프리셋이 하이라이트를 뒤집는 것들."""
        if self.bank is None:
            return []
        return [(r.cls, r.light_r) for r in self.bank.summary() if r.directional]

    # ------------------------------------------------------------------ 편집 (파일 계층 → reload)

    def delete(self, ids: Sequence[str]) -> int:
        """소스 삭제. 반환 = 지운 수. 없는 id 는 건너뛴다."""
        if self.root is None or self.bank is None:
            raise BankSessionError("은행을 먼저 여세요")
        writer = BankWriter(self.root)
        n = 0
        for sid in ids:
            cls, _, name = sid.partition("/")
            if not name:
                continue
            n += writer.delete(cls, name)
        if n:
            writer.finish({"importer": "bank-tab", "action": "delete", "ids": list(ids)})
        self.reload()
        return n

    def replace_mask(self, source_id: str, mask: np.ndarray, *, tool: str = "brush") -> None:
        """라벨 탭에서 다듬은 마스크를 같은 id 에 덮어쓴다(크롭 크기 그대로). 빈 마스크는 거부."""
        if self.root is None or self.bank is None:
            raise BankSessionError("은행을 먼저 여세요")
        src = self.source(source_id)
        if mask.shape[:2] != src.mask.shape[:2]:
            raise BankSessionError(f"마스크 크기 {mask.shape[:2]} ≠ 크롭 {src.mask.shape[:2]}")
        if not np.any(mask):
            raise BankSessionError("마스크가 비어 있습니다 — 소스를 지우려면 삭제를 쓰세요")
        cls, _, name = source_id.partition("/")
        writer = BankWriter(self.root)
        writer.replace_mask(cls, name, mask, mask_origin=f"manual:{tool}")
        writer.finish({"importer": "bank-tab", "action": "replace_mask", "ids": [source_id]})
        self.reload()

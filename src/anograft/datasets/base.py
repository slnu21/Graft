"""표준 산업 데이터셋 어댑터 계약 (설계 §3.4) — **변환기**일 뿐이다.

어댑터는 자기 폴더 레이아웃을 ``PairRecord(image, mask, class)`` 목록으로 바꾸고, 은행 쓰기는
``bank/importers/pairs.import_pair_records``가 한다(어댑터 = 파일 1개 + 테스트 1개). 앱은 데이터를 내려받지도
재배포하지도 않는다 — ``info``의 라이선스·URL은 안내용이고, 임포터는 로컬 사본을 읽기만 한다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from anograft.bank.importers.pairs import PairRecord


class DatasetError(RuntimeError):
    """레이아웃이 기대와 다르다 — 메시지에 기대 구조를 함께 싣는다."""


@dataclass(frozen=True)
class DatasetInfo:
    name: str  # "mvtec-ad"
    title: str
    license: str
    url: str
    layout_help: str  # 기대 폴더 구조 (여러 줄)
    categories: tuple[str, ...]
    note: str = ""


@runtime_checkable
class DatasetAdapter(Protocol):
    info: DatasetInfo

    def defects(
        self, root: Path, *, warn: Callable[[str], None] | None = None
    ) -> Iterable[PairRecord]:
        """``root`` = ``<dataset>/<category>``. 결함 이미지·GT 마스크·클래스(결함 유형) 쌍. ``good``은 제외.
        빠진 마스크 등은 ``warn``으로 알리고 건너뛴다(fail-soft)."""
        ...

    def normals(self, root: Path) -> list[Path]:
        """정상 이미지(합성 대상 후보) — 이름 정렬."""
        ...

    def category(self, root: Path) -> str: ...


REGISTRY: dict[str, DatasetAdapter] = {}


def register(adapter: DatasetAdapter) -> DatasetAdapter:
    REGISTRY[adapter.info.name] = adapter
    return adapter


def get_adapter(name: str) -> DatasetAdapter:
    try:
        return REGISTRY[name]
    except KeyError:
        raise DatasetError(
            f"알 수 없는 데이터셋 {name!r} — 선택: {', '.join(sorted(REGISTRY))}"
        ) from None


def adapter_names() -> list[str]:
    return sorted(REGISTRY)


def info_lines(info: DatasetInfo) -> list[str]:
    """``anograft dataset info`` 출력 행."""
    lines = [
        f"{info.name} — {info.title}",
        f"  라이선스: {info.license}",
        f"  URL: {info.url}",
        f"  카테고리({len(info.categories)}): {', '.join(info.categories)}",
        "  기대 구조:",
    ]
    lines += [f"    {ln}" for ln in info.layout_help.splitlines()]
    if info.note:
        lines.append(f"  참고: {info.note}")
    lines.append(
        "  앱은 내려받지도 재배포하지도 않습니다 — 로컬 사본을 읽기만 합니다: "
        f"anograft bank import-dataset {info.name} <root>/<category> --out bank/<category>"
    )
    return lines

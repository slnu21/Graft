"""표준 데이터셋 임포터 (설계 §3.4) — 어댑터가 낸 ``PairRecord``를 ``pairs.import_pair_records``로. 은행 쓰기 로직 0.

``anograft bank import-dataset mvtec-ad <root>/<category> --out bank/<category>``. 결과는 같은 쌍을
``import-pairs``로 넣은 것과 **동일**(테스트로 고정 — 어댑터 = 변환기).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from anograft.bank.importers.common import DEFAULT_MARGIN, DEFAULT_MIN_AREA
from anograft.bank.importers.pairs import PairsImportResult, import_pair_records
from anograft.datasets import DatasetError, get_adapter

Logger = Callable[[str], None]


@dataclass
class DatasetImportResult:
    pairs: PairsImportResult
    dataset: str
    category: str
    normals: list[Path]  # 어댑터가 아는 정상 이미지(합성 대상 후보) — 복사하지 않는다


def import_dataset(
    name: str,
    root: str | Path,
    out: str | Path,
    *,
    margin: int = DEFAULT_MARGIN,
    min_area: int = DEFAULT_MIN_AREA,
    keep_whole: bool = False,
    tags: Sequence[str] = (),
    log: Logger | None = None,
) -> DatasetImportResult:
    """``DatasetError``(레이아웃 불일치·미지 이름)는 그대로 올린다 — CLI가 기대 구조를 출력한다."""
    log = log or (lambda _m: None)
    adapter = get_adapter(name)
    r = Path(root)
    pre: list[str] = []

    def warn(msg: str) -> None:
        pre.append(msg)
        log(msg)

    pairs = list(adapter.defects(r, warn=warn))
    category = adapter.category(r)
    res = import_pair_records(
        pairs,
        out,
        margin=margin,
        min_area=min_area,
        keep_whole=keep_whole,
        um_per_px=None,  # 표준셋은 픽셀 피치 정보가 없다
        tags=tags,
        entry={"importer": "dataset", "dataset": name, "root": str(r), "category": category},
        log=log,
    )
    res.warnings = pre + res.warnings
    try:
        normals = adapter.normals(r)
    except DatasetError as e:  # 정상 폴더가 없어도 은행 임포트는 유효
        warn(f"정상 이미지 목록 실패: {e}")
        normals = []
    return DatasetImportResult(res, name, category, normals)

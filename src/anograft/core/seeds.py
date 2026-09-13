"""시드·재현성 계약 (설계 §5). 모든 무작위성은 여기서 만든 Generator 하나를 통과한다.

- 이미지 ``i``의 결과는 ``image_rng(seed, i)``만으로 정해진다 — 워커 배정과 무관.
- ``spawn_key``의 첫 원소가 용도(0 = split, 1 = image)라 용도끼리 스트림이 겹치지 않는다.
- 전역 ``np.random.*``·``random`` 모듈은 코드베이스 어디서도 쓰지 않는다(테스트가 grep으로 막는다).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np

_PURPOSE_SPLIT = 0
_PURPOSE_IMAGE = 1


def split_rng(seed: int) -> np.random.Generator:
    """대상 이미지 목록을 나눌 때(mvtec writer 등) 쓰는 rng."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(_PURPOSE_SPLIT,)))


def image_rng(seed: int, index: int) -> np.random.Generator:
    """이미지 인덱스 ``index``의 파생 rng. 같은 (seed, index) → 같은 스트림."""
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(_PURPOSE_IMAGE, index)))


def bank_fingerprint(entries: Iterable[tuple[str, int]]) -> str:
    """은행 지문 — ``(id, area_px)`` 쌍을 id 순으로 정렬해 해시. 은행이 바뀌면 파이프라인 해시도 바뀐다."""
    lines = sorted(f"{sid},{area}" for sid, area in entries)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def pipeline_hash(resolved_recipe_yaml: str, version: str, bank_fp: str) -> str:
    """사이드카·manifest에 남기는 파이프라인 해시. 레시피(resolved) + 패키지 버전 + 은행 지문."""
    h = hashlib.sha256()
    h.update(resolved_recipe_yaml.encode("utf-8"))
    h.update(b"\0")
    h.update(version.encode("utf-8"))
    h.update(b"\0")
    h.update(bank_fp.encode("utf-8"))
    return h.hexdigest()[:16]

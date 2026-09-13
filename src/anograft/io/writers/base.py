"""Writer 계약 (설계 §8) — 합성과 무관한 마지막 변환 단계. core와는 ``GraftResult``로만 만난다.

정본(``pairs`` 레이아웃: images/masks/meta + manifest + recipe.resolved.yaml)은 형식과 무관하게 항상 기록되고,
형식별 writer(``yolo`` 등)는 그 위에 학습 프레임워크가 읽는 파일을 덧붙인다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from anograft.core.recipe import Recipe
from anograft.core.types import GraftResult


@dataclass
class WriterSummary:
    root: Path
    n_ok: int = 0
    n_skipped: int = 0
    n_normals: int = 0
    n_fallback: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)  # manifest · recipe 등 부속 파일 경로


@runtime_checkable
class Writer(Protocol):
    format: ClassVar[str]

    def begin(
        self,
        root: Path,
        recipe: Recipe,
        pipeline_hash: str,
        classes: Sequence[str],
        *,
        bank_fingerprint: str = "",
    ) -> None: ...

    def write_normal(self, path: Path) -> None: ...

    def write_synthetic(self, result: GraftResult) -> None: ...

    def finish(self) -> WriterSummary: ...


class WriterNotImplementedError(NotImplementedError):
    pass


def sidecar_with_header(
    sidecar: Any, *, version: str, pipeline_hash: str, writer: dict[str, Any]
) -> dict[str, Any]:
    """§8.3 순서: ``anograft`` · ``pipeline_hash`` 가 맨 앞, ``writer`` 는 ``warnings`` 앞."""
    body = dict(sidecar)
    warnings = body.pop("warnings", [])
    out: dict[str, Any] = {"anograft": version, "pipeline_hash": pipeline_hash}
    out.update(body)
    out["writer"] = writer
    out["warnings"] = warnings
    return out

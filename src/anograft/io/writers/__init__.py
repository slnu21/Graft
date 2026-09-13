"""출력 writer 계층 (설계 §8). ``make_writer(cfg)``가 레시피 ``output.writer.format``으로 고른다.

v0.1 first-run: ``pairs``(정본). ``yolo``는 bank-writer-parallel(#758)에서 — 그 전까지 ``yolo`` 레시피는 정본만 기록하고
경고를 남긴다(fail-soft: 정본은 어떤 형식에서도 항상 나가므로 "첫 합성 이미지"가 writer 때문에 막히지 않는다).
"""

from __future__ import annotations

from anograft.core.recipe import PairsWriterConfig, YoloWriterConfig
from anograft.io.writers.base import Writer, WriterSummary, sidecar_with_header
from anograft.io.writers.pairs import PairsWriter

__all__ = [
    "PairsWriter",
    "Writer",
    "WriterSummary",
    "make_writer",
    "sidecar_with_header",
]

WriterCfg = PairsWriterConfig | YoloWriterConfig


def make_writer(cfg: WriterCfg) -> tuple[Writer, str | None]:
    """``(writer, warning)``. 형식이 아직 없으면 정본 writer + 사유 문자열."""
    if cfg.format == "pairs":
        return PairsWriter(), None
    return (
        PairsWriter(),
        f"writer '{cfg.format}' 은(는) 아직 구현되지 않았습니다 — 정본(images/masks/meta)만 기록합니다",
    )

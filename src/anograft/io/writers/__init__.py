"""출력 writer 계층 (설계 §8). ``make_writer(cfg)``가 레시피 ``output.writer.format``으로 고른다.

v0.1: ``pairs``(정본) · ``yolo``(정본 + labels/·data.yaml). v0.2 ``mvtec`` · v0.7 ``coco``. 스키마에 있는데 구현이 없는
형식은 정본 writer로 폴백하고 경고를 남긴다(fail-soft: 정본은 어떤 형식에서도 항상 나간다).
"""

from __future__ import annotations

from anograft.core.recipe import PairsWriterConfig, YoloWriterConfig
from anograft.io.writers.base import Writer, WriterSummary, sidecar_with_header
from anograft.io.writers.pairs import PairsWriter
from anograft.io.writers.yolo import YoloWriter

__all__ = [
    "PairsWriter",
    "Writer",
    "WriterSummary",
    "YoloWriter",
    "make_writer",
    "sidecar_with_header",
]

WriterCfg = PairsWriterConfig | YoloWriterConfig


def make_writer(cfg: WriterCfg) -> tuple[Writer, str | None]:
    """``(writer, warning)``. 형식이 아직 없으면 정본 writer + 사유 문자열."""
    if cfg.format == "pairs":
        return PairsWriter(), None
    if cfg.format == "yolo":
        return YoloWriter(cfg), None
    return (
        PairsWriter(),
        f"writer '{cfg.format}' 은(는) 아직 구현되지 않았습니다 — 정본(images/masks/meta)만 기록합니다",
    )

"""런타임 합성 픽스처 (설계 §11) — MVTec 다운로드 없이 스테이지·파이프라인을 시험한다.

- ``disk_target(size, gray)``: 어두운 배경 + 밝은 원판(물체는 중앙, 배경은 테두리 — Otsu ``auto``의 전제).
- ``line_defect(length, width)``: 밝은 선 결함 크롭 + 마스크 (``DefectSource``).
- ``context(...)``: 스테이지 단위 테스트용 Context 생성.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
from pydantic import BaseModel

from anograft.core.types import Context, DefectSource, PlacedDefect, TargetImage


def disk_image(size: int = 128, *, invert: bool = False, radius: int | None = None) -> np.ndarray:
    """``size×size×3`` — 배경 40, 중앙 원판 200. ``invert``면 밝은 배경·어두운 원판."""
    img = np.full((size, size, 3), 40, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius or size // 3, (200, 200, 200), -1)
    return (255 - img) if invert else img


def disk_target(
    size: int = 128,
    gray: bool = False,
    *,
    invert: bool = False,
    um_per_px: float | None = None,
    name: str = "target.png",
) -> TargetImage:
    return TargetImage(
        path=Path(name), image=disk_image(size, invert=invert), gray=gray, um_per_px=um_per_px
    )


def line_defect(
    length: int = 16,
    width: int = 3,
    *,
    margin: int = 4,
    cls: str = "scratch",
    um_per_px: float | None = None,
) -> DefectSource:
    """가로 밝은 선 + 마스크. 크롭 = 선 bbox + ``margin``(은행 포맷과 같은 형태)."""
    h, w = width + 2 * margin, length + 2 * margin
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    img[margin : margin + width, margin : margin + length] = 230
    mask[margin : margin + width, margin : margin + length] = 255
    return DefectSource(
        id=f"{cls}/000", cls=cls, image=img, mask=mask, um_per_px=um_per_px, origin="fixture"
    )


def context(
    target: TargetImage | None = None,
    *,
    seed: int = 0,
    source: DefectSource | None = None,
    roi: np.ndarray | None = None,
    patch: np.ndarray | None = None,
    patch_mask: np.ndarray | None = None,
    placed: tuple[PlacedDefect, ...] = (),
) -> Context:
    t = target or disk_target()
    return Context(
        rng=np.random.default_rng(seed),
        target=t,
        composite=t.image,
        roi=roi,
        source=source,
        patch=patch,
        patch_mask=patch_mask,
        placed=placed,
    )


# ---------------------------------------------------------------------------
# 파이프라인 통합 테스트용 더미 — 아직 구현 안 된 스테이지 자리(source: first-run에서 은행 로더로 교체)
# ---------------------------------------------------------------------------


class _Dummy:
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: BaseModel, deps: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.deps = dict(deps)


class SourceFixture(_Dummy):
    """``deps["sources"]``(DefectSource 목록)에서 rng로 하나."""

    stage = "source"
    methods = ("bank",)

    def apply(self, ctx: Context) -> Context:
        sources: list[DefectSource] = self.deps["sources"]
        src = sources[int(ctx.rng.integers(len(sources)))]
        return replace(ctx, source=src).with_log("source", {"source_id": src.id, "class": src.cls})

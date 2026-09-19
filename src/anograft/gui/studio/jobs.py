"""미리보기 작업 — **Qt 없음**. 워커 스레드가 부르는 순수 함수와, "최신 것만 남기는" 큐.

- ``preview_target(target, long_side)``: 대상을 긴 변 ``long_side``로 축소하고 ``um_per_px``를 같은 배율로 키운다(축소 = 픽셀당
  물리 길이 증가). **축소본에서 합성하므로 원본 해상도 ``run``과 결과가 다르다** — 배치 좌표·Poisson 결과는 축척에 따라 달라진다.
  GUI는 "느낌"을, 정확한 결과는 CLI ``run``이 낸다(TASKS 열어둔 것). 상태바에 축소 배율을 표시한다.
- ``run_preview(prep, job)``: ``image_rng(seed, k)`` → **사용자가 고른 대상**에 파이프라인(``run_one_traced``). CLI ``run``은
  대상을 rng로 뽑지만 GUI는 대상을 고정하고 시드 변형 ``k``만 돌린다.
- ``LatestOnlyQueue``: 같은 키(캔버스/변형 k/썸네일 경로)의 작업은 최신 것으로 교체, 세대(``generation``)가 지난 작업은 폐기.
  캔버스 작업이 변형 작업보다 먼저 나온다.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from anograft import runner
from anograft.core.pipeline import TraceStep
from anograft.core.recipe import Recipe
from anograft.core.seeds import image_rng
from anograft.core.types import GraftResult, TargetImage
from anograft.io.targets import load_target
from anograft.preview import fit_long_side

KIND_PREPARE = "prepare"
KIND_PREVIEW = "preview"
KIND_THUMB = "thumb"


@dataclass(frozen=True)
class PreviewJob:
    kind: str  # prepare | preview | thumb
    generation: int
    target: Path | None = None
    index: int = 0  # 변형 k
    long_side: int = 1024  # 0 = 원본
    priority: int = 1  # 낮을수록 먼저 (캔버스 = 0, 변형 = 1, 썸네일 = 2)
    recipe: Recipe | None = None  # prepare 전용

    @property
    def key(self) -> Hashable:
        if self.kind == KIND_PREVIEW:
            return (self.kind, self.index)
        if self.kind == KIND_THUMB:
            return (self.kind, str(self.target))
        return (self.kind,)


@dataclass
class PreviewResult:
    job: PreviewJob
    result: GraftResult
    steps: list[TraceStep]
    target: TargetImage  # 축소된 대상 (캔버스 A면)
    scale: float  # 축소 배율 (1.0 = 원본)
    elapsed_s: float


@dataclass
class ThumbResult:
    job: PreviewJob
    image: np.ndarray  # HxWx3 BGR
    shape: tuple[int, int]  # 원본 (h, w)


@dataclass
class JobError:
    job: PreviewJob
    message: str


def preview_target(target: TargetImage, long_side: int) -> tuple[TargetImage, float]:
    """긴 변 축소 + um_per_px 보정. ``long_side <= 0``이면 원본."""
    h, w = target.image.shape[:2]
    if long_side <= 0 or max(h, w) <= long_side:
        return target, 1.0
    s = long_side / float(max(h, w))
    small = fit_long_side(target.image, long_side)
    um = target.um_per_px / s if target.um_per_px is not None else None
    return TargetImage(path=target.path, image=small, gray=target.gray, um_per_px=um), s


def run_preview(prep: runner.Prepared, job: PreviewJob) -> PreviewResult:
    assert job.target is not None
    t0 = time.perf_counter()
    full = load_target(job.target, prep.recipe.inputs.um_per_px)
    small, s = preview_target(full, job.long_side)
    rng = image_rng(prep.recipe.seed, job.index)
    result, steps = prep.pipeline.run_one_traced(small, job.index, rng=rng)
    return PreviewResult(job, result, steps, small, s, time.perf_counter() - t0)


def make_thumb(job: PreviewJob, long_side: int = 192) -> ThumbResult:
    assert job.target is not None
    full = load_target(job.target)
    return ThumbResult(job, fit_long_side(full.image, long_side), full.image.shape[:2])


def run_job(prep: runner.Prepared | None, job: PreviewJob) -> Any:
    """워커의 디스패치. 반환 = ``Prepared`` | ``PreviewResult`` | ``ThumbResult``."""
    if job.kind == KIND_PREPARE:
        assert job.recipe is not None
        return runner.prepare(job.recipe)
    if job.kind == KIND_THUMB:
        return make_thumb(job)
    if prep is None:
        raise runner.PrepareError("보관함·바탕 이미지가 준비되지 않았습니다")
    return run_preview(prep, job)


@dataclass
class LatestOnlyQueue:
    """키당 최신 작업 하나. ``pop()``은 (priority, 삽입 순) 순으로. ``drop_before(gen)``으로 지난 세대 폐기."""

    _items: OrderedDict[Hashable, PreviewJob] = field(default_factory=OrderedDict)

    def put(self, job: PreviewJob) -> None:
        self._items.pop(job.key, None)
        self._items[job.key] = job

    def pop(self) -> PreviewJob | None:
        if not self._items:
            return None
        key = min(self._items, key=lambda k: (self._items[k].priority, list(self._items).index(k)))
        return self._items.pop(key)

    def drop_before(self, generation: int) -> int:
        stale = [
            k for k, j in self._items.items() if j.kind != KIND_THUMB and j.generation < generation
        ]
        for k in stale:
            del self._items[k]
        return len(stale)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


def overlay_bgra(mask: np.ndarray, color_bgr: tuple[int, int, int], alpha: int) -> np.ndarray:
    """마스크(0/255) → 캔버스 오버레이용 BGRA(마스크 안만 불투명 ``alpha``). 윤곽은 캔버스가 그린다."""
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    on = mask > 0
    out[on, 0], out[on, 1], out[on, 2] = color_bgr
    out[on, 3] = alpha
    return out


def contours_of(mask: np.ndarray) -> list[np.ndarray]:
    cnts, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [c.reshape(-1, 2) for c in cnts if len(c) >= 2]

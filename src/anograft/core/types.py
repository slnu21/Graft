"""파이프라인을 흐르는 값 객체. 전부 frozen dataclass — 스테이지는 ``replace()``로 새 Context를 돌려준다.

설계 §2. 배열 필드의 규약:
- 이미지는 항상 ``HxWx3 uint8`` (흑백도 승격). 출력 직전에만 ``TargetImage.gray``로 복원한다.
- 마스크는 ``HxW uint8``, 값은 0/255만.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

BBox = tuple[int, int, int, int]  # x, y, w, h


@dataclass(frozen=True)
class DefectSource:
    """결함 은행의 소스 하나 — bbox+margin 크롭과 마스크."""

    id: str  # "<class>/<name>"  예: "scratch/img0042-01"
    cls: str
    image: np.ndarray  # HxWx3 uint8
    mask: np.ndarray  # HxW uint8 0/255
    um_per_px: float | None = None
    tags: tuple[str, ...] = ()
    origin: str = ""  # 원본 파일 상대경로 (감사용)
    mask_origin: str = "png"  # png | yolo-polygon | yolo-box:<method>
    confidence: float | None = (
        None  # 추정 마스크 타당성 0..1 (bank.mask_from_box.mask_confidence) — 정확 마스크는 None
    )
    flags: tuple[str, ...] = ()  # low-contrast · box-edge · fragmented · saturated · area-out


@dataclass(frozen=True)
class TargetImage:
    """합성 대상(정상) 이미지. 로드 시 3ch로 승격되고 ``gray``가 원래 채널을 기억한다."""

    path: Path
    image: np.ndarray  # HxWx3 uint8
    gray: bool
    um_per_px: float | None = None


@dataclass(frozen=True)
class Placement:
    """패치가 대상 어디에 놓였나. ``offset``이 근본값(패치 캔버스 → 대상 좌표 변환), ``bbox``·``center``는 그로부터 유도.

    패치 **캔버스**(회전 bbox·소스 크롭 여유 포함)는 이미지 밖으로 걸쳐도 된다 — 마스크만 ROI 안·테두리 여유 안이면 채택.
    blend는 ``composite[y, x] ↔ patch[y - oy, x - ox]``로 대응시키고 캔버스 창은 이미지에 맞춰 잘라 쓴다.
    """

    center: tuple[int, int]  # (x, y) 대상 좌표 — 패치 마스크 bbox의 중심
    bbox: BBox  # 패치 마스크의 대상 내 bbox (x, y, w, h)
    offset: tuple[int, int]  # (ox, oy) 패치 캔버스 좌상단의 대상 좌표 (음수 가능)
    tries: int
    shrink_rounds: int = 0


@dataclass(frozen=True)
class PlacedDefect:
    """결함 루프 한 바퀴의 산출 — 소스 마스크를 대상 좌표에 놓은 것. 7단계(gtmask)가 이걸 인스턴스 GT로 바꾼다."""

    cls: str
    source_id: str
    mask: np.ndarray  # HxW uint8 0/255 (대상 크기)
    defect_index: int = -1  # 결함 루프 순번 k — 사이드카 defects[k].gt 로 되돌아간다


@dataclass(frozen=True)
class Instance:
    """결함 인스턴스 하나의 GT — YOLO writer가 줄 하나로 쓴다."""

    cls: str
    class_id: int
    mask: np.ndarray  # HxW uint8 0/255 (대상 크기)
    bbox: BBox
    area_px: int
    defect_index: int = -1  # 어느 결함에서 왔나 (PlacedDefect.defect_index)


@dataclass(frozen=True)
class Context:
    """스테이지 사이를 흐르는 상태. 결함 루프(1~5) 동안 source/patch/placement가 결함마다 갈아끼워지고,
    composite·instances·log는 누적된다."""

    rng: np.random.Generator
    target: TargetImage
    composite: np.ndarray  # 현재까지 합성 결과. 시작값 = target.image
    roi: np.ndarray | None = None  # HxW bool — 배치 허용 영역
    source: DefectSource | None = None
    patch: np.ndarray | None = None  # 기하 변환 후 패치 HxWx3
    patch_mask: np.ndarray | None = None  # 패치 크기 마스크
    placement: Placement | None = None
    placed_mask: np.ndarray | None = None  # 현재 결함의 소스 마스크를 대상 좌표에 놓은 것 (HxW)
    pre_degrade: np.ndarray | None = None  # 6단계 직전 스냅샷 (GT diff 정책용)
    placed: tuple[PlacedDefect, ...] = ()  # 성공한 결함들의 소스 마스크(대상 좌표) — gtmask 입력
    gt_mask: np.ndarray | None = None  # 전체 GT (인스턴스 합집합)
    instances: tuple[Instance, ...] = ()
    defect_logs: tuple[Mapping[str, Any], ...] = ()  # 결함마다 스테이지 로그 묶음
    log: Mapping[str, Any] = field(default_factory=dict)  # 현재 결함/이미지 단계의 로그
    warnings: tuple[str, ...] = ()

    @classmethod
    def initial(cls, rng: np.random.Generator, target: TargetImage) -> Context:
        return cls(rng=rng, target=target, composite=target.image)

    # --- 스테이지가 쓰는 작은 도우미 (전부 새 Context 반환) ---

    def with_log(self, key: str, value: Any) -> Context:
        """``log[key] = value``를 기록한 새 Context."""
        new_log = dict(self.log)
        new_log[key] = value
        return replace(self, log=new_log)

    def warn(self, message: str) -> Context:
        return replace(self, warnings=(*self.warnings, message))

    def begin_defect(self) -> Context:
        """결함 루프 한 바퀴 시작 — 결함 단위 필드를 비운다. composite·instances·warnings는 유지."""
        return replace(
            self,
            source=None,
            patch=None,
            patch_mask=None,
            placement=None,
            placed_mask=None,
            log={},
        )

    def end_defect(self) -> Context:
        """결함 루프 한 바퀴 종료 — 현재 log를 defect_logs에 봉인하고, 배치까지 성공했으면 placed에 추가한다."""
        placed = self.placed
        if self.placed_mask is not None and self.source is not None:
            k = len(self.defect_logs)  # 지금 봉인하는 로그의 순번 = 이 결함의 k
            placed = (*placed, PlacedDefect(self.source.cls, self.source.id, self.placed_mask, k))
        return replace(self, defect_logs=(*self.defect_logs, dict(self.log)), log={}, placed=placed)


@dataclass(frozen=True)
class GraftResult:
    """이미지 한 장의 최종 결과. writer는 이것만 본다(core ↔ writer 유일 접점)."""

    index: int
    status: Literal["ok", "skipped"]
    image: np.ndarray  # 출력 채널로 복원됨 (gray면 HxW)
    gt_mask: np.ndarray  # HxW uint8 0/255 — 인스턴스 합집합
    instances: tuple[Instance, ...]
    sidecar: Mapping[str, Any]
    warnings: tuple[str, ...] = ()
    reason: str | None = None  # skipped 사유

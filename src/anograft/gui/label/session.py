"""라벨 세션 — **Qt 없음**. 이미지 한 장 + 마스크 편집 상태(되돌리기/다시하기) + 통계 + 은행 저장 (v0.5 라벨 탭).

- 편집은 전부 ``mask``(HxW uint8 0/255)에 대한 순수 배열 연산: 브러시/지우개 ``stroke``, ``fill_polygon``, 박스 → ``auto_select``
  (``bank.mask_from_box`` 재사용 — grabcut/otsu/ellipse/rect + 폴백 사슬), ``dilate``/``erode``, ``clear``.
- 변경 전에 ``push_undo``(위젯이 스트로크 시작에 한 번 부른다). 되돌리기 깊이 ``UNDO_DEPTH``.
- ``save_to_bank``는 임포터와 같은 화폐(``ImportRecord`` → ``BankWriter.add``)를 탄다 — 라벨 탭이 은행 포맷을 따로 알지
  않는다. ``mask_origin = "manual:<tool>"``(brush·polygon·auto:<method>·mixed). 성분마다 소스 하나(``keep_whole`` 아니면).
- 위젯은 이 클래스의 상태만 그리고, 마우스 이벤트를 편집 함수 호출로 바꾼다(테스트는 여기서 끝난다).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from anograft.bank.importers.common import (
    DEFAULT_MARGIN,
    DEFAULT_MIN_AREA,
    AddedSource,
    BankWriter,
    ImportOptions,
    ImportRecord,
)
from anograft.bank.mask_from_box import METHODS as AUTO_METHODS
from anograft.bank.mask_from_box import mask_from_box
from anograft.core.channels import binarize
from anograft.core.seeds import stable_seed
from anograft.io import imgio

UNDO_DEPTH = 40
Point = tuple[float, float]
Box = tuple[int, int, int, int]


class LabelError(ValueError):
    pass


@dataclass(frozen=True)
class LabelStats:
    area_px: int
    area_ratio: float
    bbox: Box | None
    n_components: int
    length_px: float  # 최소 외접 사각형의 긴 변 (전체 마스크)
    length_um: float | None  # um_per_px 가 있을 때
    contrast: float | None  # 마스크 안 평균 그레이 − 링(8px) 평균 그레이


class LabelSession:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.image: np.ndarray | None = None  # HxWx3
        self.gray = False
        self.mask: np.ndarray | None = None  # HxW 0/255
        self._undo: list[np.ndarray] = []
        self._redo: list[np.ndarray] = []
        self.tools_used: set[str] = set()  # mask_origin 계산용
        self.auto_methods_used: set[str] = set()
        self.dirty = False

    # ------------------------------------------------------------------ 로드

    @property
    def loaded(self) -> bool:
        return self.image is not None and self.mask is not None

    @property
    def shape(self) -> tuple[int, int]:
        assert self.image is not None
        return (int(self.image.shape[0]), int(self.image.shape[1]))

    def load_image(self, path: str | Path) -> None:
        """이미지를 읽고 마스크를 비운다(되돌리기 스택도)."""
        image, gray = imgio.read_image(path)
        self.path = Path(path)
        self.image, self.gray = image, gray
        self.mask = np.zeros(image.shape[:2], dtype=np.uint8)
        self._undo.clear()
        self._redo.clear()
        self.tools_used.clear()
        self.auto_methods_used.clear()
        self.dirty = False

    def set_image(self, image: np.ndarray, gray: bool = False, path: Path | None = None) -> None:
        """배열로 직접(테스트·은행 크롭 편집)."""
        img = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.path = path
        self.image, self.gray = np.ascontiguousarray(img), gray
        self.mask = np.zeros(img.shape[:2], dtype=np.uint8)
        self._undo.clear()
        self._redo.clear()
        self.tools_used.clear()
        self.auto_methods_used.clear()
        self.dirty = False

    def load_mask(self, path: str | Path) -> None:
        """기존 마스크 PNG(같은 크기)를 현재 마스크로. 되돌리기 가능."""
        self._require()
        m = imgio.read_mask(path)
        if m.shape[:2] != self.shape:
            raise LabelError(f"마스크 크기 {m.shape[:2]} ≠ 이미지 {self.shape}")
        self.push_undo()
        self.mask = m
        self.tools_used.add("png")
        self.dirty = True

    def _require(self) -> None:
        if not self.loaded:
            raise LabelError("이미지를 먼저 여세요")

    # ------------------------------------------------------------------ 되돌리기

    def push_undo(self) -> None:
        self._require()
        assert self.mask is not None
        self._undo.append(self.mask.copy())
        if len(self._undo) > UNDO_DEPTH:
            del self._undo[0]
        self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        assert self.mask is not None
        self._redo.append(self.mask)
        self.mask = self._undo.pop()
        self.dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        assert self.mask is not None
        self._undo.append(self.mask)
        self.mask = self._redo.pop()
        self.dirty = True
        return True

    # ------------------------------------------------------------------ 편집 (전부 mask 를 제자리에서 바꾼다 — 호출 전 push_undo)

    def stroke(self, points: Sequence[Point], radius: float, *, erase: bool = False) -> None:
        """점 목록을 잇는 둥근 선(지름 ≈ 2·radius). 점 하나면 원. ``erase``면 지운다."""
        self._require()
        assert self.mask is not None
        if not points:
            return
        value = 0 if erase else 255
        r = max(1, round(radius))
        pts = [(round(x), round(y)) for x, y in points]
        if len(pts) == 1:
            cv2.circle(self.mask, pts[0], r, value, -1, lineType=cv2.LINE_8)
        else:
            for a, b in itertools.pairwise(pts):
                cv2.line(self.mask, a, b, value, 2 * r, lineType=cv2.LINE_8)
            for p in (pts[0], pts[-1]):
                cv2.circle(self.mask, p, r, value, -1, lineType=cv2.LINE_8)
        self.tools_used.add("eraser" if erase else "brush")
        self.dirty = True

    def fill_polygon(self, points: Sequence[Point], *, erase: bool = False) -> None:
        self._require()
        assert self.mask is not None
        if len(points) < 3:
            raise LabelError("폴리곤은 점 3개 이상")
        poly = np.array([[round(x), round(y)] for x, y in points], dtype=np.int32)
        cv2.fillPoly(self.mask, [poly], 0 if erase else 255, lineType=cv2.LINE_8)
        self.tools_used.add("polygon")
        self.dirty = True

    def auto_select(self, box: Box, method: str = "grabcut", *, margin: int = 16) -> str:
        """박스 안 결함을 추정해 마스크에 **더한다**. 반환 = 실제 쓴 방법(폴백 반영). 시드는 (파일명, 박스)의 crc32."""
        self._require()
        assert self.image is not None and self.mask is not None
        if method not in AUTO_METHODS:
            raise LabelError(
                f"알 수 없는 자동 선택 방법 {method!r} (선택: {', '.join(AUTO_METHODS)})"
            )
        x, y, w, h = (int(v) for v in box)
        if w <= 0 or h <= 0:
            raise LabelError("박스가 비어 있습니다")
        key = f"{self.path.name if self.path else 'image'}:{x},{y},{w},{h}"
        try:
            est, used = mask_from_box(
                self.image, (x, y, w, h), method, margin=margin, seed=stable_seed(key)
            )
        except ValueError as e:
            raise LabelError(str(e)) from e
        self.mask = np.maximum(self.mask, est)
        self.tools_used.add("auto")
        self.auto_methods_used.add(used)
        self.dirty = True
        return used

    def dilate(self, px: int = 1) -> None:
        self._morph(px, grow=True)

    def erode(self, px: int = 1) -> None:
        self._morph(px, grow=False)

    def _morph(self, px: int, *, grow: bool) -> None:
        self._require()
        assert self.mask is not None
        if px <= 0:
            return
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        self.mask = (cv2.dilate if grow else cv2.erode)(self.mask, k)
        self.tools_used.add("morph")
        self.dirty = True

    def clear(self) -> None:
        self._require()
        assert self.mask is not None
        self.mask[:] = 0
        self.dirty = True

    # ------------------------------------------------------------------ 통계

    def stats(self, um_per_px: float | None = None) -> LabelStats:
        self._require()
        assert self.image is not None and self.mask is not None
        m = self.mask > 0
        area = int(m.sum())
        h, w = self.shape
        if area == 0:
            return LabelStats(0, 0.0, None, 0, 0.0, None, None)
        x, y, bw, bh = cv2.boundingRect(self.mask)
        n, _ = cv2.connectedComponents((m).astype(np.uint8), connectivity=8)
        pts = cv2.findNonZero(self.mask)
        (_cx, _cy), (rw, rh), _ang = cv2.minAreaRect(pts)
        length = float(max(rw, rh))
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ring = cv2.dilate(self.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
        ring &= ~m
        contrast = float(gray[m].mean() - gray[ring].mean()) if ring.any() else None
        return LabelStats(
            area_px=area,
            area_ratio=area / float(h * w),
            bbox=(int(x), int(y), int(bw), int(bh)),
            n_components=int(n - 1),
            length_px=length,
            length_um=(length * um_per_px) if um_per_px else None,
            contrast=contrast,
        )

    # ------------------------------------------------------------------ 은행 저장

    def mask_origin(self) -> str:
        used = self.tools_used - {"morph"}
        if not used:
            return "manual"
        if used == {"auto"} and len(self.auto_methods_used) == 1:
            return f"manual:auto:{next(iter(self.auto_methods_used))}"
        if len(used) == 1:
            return f"manual:{next(iter(used))}"
        return "manual:mixed"

    def save_to_bank(
        self,
        root: str | Path,
        cls: str,
        *,
        tags: Sequence[str] = (),
        um_per_px: float | None = None,
        margin: int = DEFAULT_MARGIN,
        min_area: int = DEFAULT_MIN_AREA,
        keep_whole: bool = False,
        id_hint: str | None = None,
    ) -> tuple[list[AddedSource], list[str]]:
        """현재 마스크를 은행에 소스로. 반환 ``(추가된 소스, 경고)``. 빈 마스크·빈 클래스는 ``LabelError``."""
        self._require()
        assert self.image is not None and self.mask is not None
        cls = cls.strip()
        if not cls:
            raise LabelError("클래스 이름을 입력하세요")
        if "/" in cls or "\\" in cls:
            raise LabelError("클래스 이름에 경로 구분자는 쓸 수 없습니다")
        if not np.any(self.mask):
            raise LabelError("마스크가 비어 있습니다")
        stem = id_hint or (self.path.stem if self.path else "label")
        rec = ImportRecord(
            image=self.image,
            gray=self.gray,
            mask=binarize(self.mask),
            cls=cls,
            origin=self.path.as_posix() if self.path else stem,
            id_hint=stem,
            mask_origin=self.mask_origin(),
            um_per_px=um_per_px,
            tags=tuple(t.strip() for t in tags if t.strip()),
        )
        writer = BankWriter(root)
        added = writer.add(
            rec, ImportOptions(margin=margin, min_area=min_area, keep_whole=keep_whole)
        )
        writer.finish(
            {
                "importer": "label",
                "origin": rec.origin,
                "class": cls,
                "mask_origin": rec.mask_origin,
                "n_sources": len(added),
            }
        )
        if added:
            self.dirty = False
        return added, list(writer.stats.warnings)

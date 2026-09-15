"""라벨 세션 — **Qt 없음**. 이미지 한 장 + 마스크 편집 상태(되돌리기/다시하기) + 통계 + 은행 저장 (v0.5 라벨 탭).

- 편집은 전부 ``mask``(HxW uint8 0/255)에 대한 순수 배열 연산: 브러시/지우개 ``stroke``, ``fill_polygon``, 박스 → ``auto_select``
  (``bank.mask_from_box`` 재사용 — grabcut/otsu/ellipse/rect + 폴백 사슬), ``dilate``/``erode``, ``clear``.
- 변경 전에 ``push_undo``(위젯이 스트로크 시작에 한 번 부른다). 되돌리기 깊이 ``UNDO_DEPTH``.
- ``save_to_bank``는 임포터와 같은 화폐(``ImportRecord`` → ``BankWriter.add``)를 탄다 — 라벨 탭이 은행 포맷을 따로 알지
  않는다. ``mask_origin = "manual:<tool>"``(brush·polygon·auto:<method>·mixed). 성분마다 소스 하나(``keep_whole`` 아니면).
- 위젯은 이 클래스의 상태만 그리고, 마우스 이벤트를 편집 함수 호출로 바꾼다(테스트는 여기서 끝난다).
- (v0.6) **YOLO 초안**: 이미지 옆의 기존 라벨(``images/…`` ↔ ``labels/….txt`` · 같은 폴더 ``<stem>.txt``)을 찾아 박스는
  ``mask_from_box`` 추정, 폴리곤은 채움으로 마스크를 **미리 채운다**(``load_yolo_draft`` → ``apply_draft(class_id)``). 초안만
  있으면 ``mask_origin`` 은 임포터와 같은 ``yolo-box:<method>``/``yolo-polygon``(추정 = ``bank ls`` est), 손을 대면 ``manual:mixed``.
- (v0.6) **ROI 모드**: ``save_roi_png(path)`` — 크롭·메타 없이 마스크를 원본 크기 PNG 로(``mask_dir`` ROI 가 읽는 형식).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
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
from anograft.bank.importers.yolo import (
    denormalize_box,
    parse_label_text,
    parse_names,
    polygon_mask,
)
from anograft.bank.mask_from_box import METHODS as AUTO_METHODS
from anograft.bank.mask_from_box import MaskConfidence, mask_confidence, mask_from_box
from anograft.core.appearance import mask_lighting
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
    lighting_deg: float | None = (
        None  # 둘레 2 px 링에서 밝은 쪽 각도(0 = →, 90 = ↓) — 검수 탭·bank ls lightR 와 같은 정의
    )


# ---------------------------------------------------------------------------
# YOLO 초안 (기존 라벨 재활용 — KNOWN-ISSUES #8)
# ---------------------------------------------------------------------------

NAMES_FILES: tuple[str, ...] = ("data.yaml", "data.yml", "classes.txt")


@dataclass(frozen=True)
class DraftItem:
    line_no: int
    class_id: int
    kind: str  # "box" | "polygon"
    box: Box | None  # 픽셀 (x, y, w, h) — 이미지 경계로 클립. 폴리곤은 None
    coords: tuple[float, ...]  # 정규화 원본(폴리곤 채움용)


@dataclass
class YoloDraft:
    label_path: Path
    items: list[DraftItem]
    names: list[str] | None = None
    names_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def class_ids(self) -> list[int]:
        return sorted({it.class_id for it in self.items})

    def class_name(self, class_id: int) -> str:
        if self.names is not None and 0 <= class_id < len(self.names):
            return self.names[class_id]
        return f"class {class_id}"

    def items_for(self, class_id: int) -> list[DraftItem]:
        return [it for it in self.items if it.class_id == class_id]

    def summary(self) -> str:
        n_box = sum(1 for it in self.items if it.kind == "box")
        n_poly = len(self.items) - n_box
        per = ", ".join(f"{self.class_name(c)} {len(self.items_for(c))}" for c in self.class_ids)
        parts = [self.label_path.name, f"박스 {n_box}"]
        if n_poly:
            parts.append(f"폴리곤 {n_poly}")
        if per:
            parts.append(per)
        return " · ".join(parts)


def find_yolo_label(image_path: str | Path) -> Path | None:
    """이미지 옆의 YOLO 라벨 파일. 순서: 경로의 마지막 ``images`` 세그먼트를 ``labels`` 로 바꾼 ``….txt``(표준 배치, 하위 폴더 유지)
    → 같은 폴더 ``<stem>.txt`` → ``<folder>/labels/<stem>.txt``. 없으면 None."""
    p = Path(image_path)
    parts = p.parts
    candidates: list[Path] = []
    for i in range(len(parts) - 2, -1, -1):
        if parts[i].lower() == "images":
            candidates.append(Path(*parts[:i], "labels", *parts[i + 1 :]).with_suffix(".txt"))
            break
    candidates.append(p.with_suffix(".txt"))
    candidates.append(p.parent / "labels" / f"{p.stem}.txt")
    for c in candidates:
        if c.is_file():
            return c
    return None


def find_names_file(image_path: str | Path, *, levels: int = 4) -> Path | None:
    """``data.yaml``/``data.yml``/``classes.txt`` — 이미지 폴더부터 위로 ``levels`` 단계."""
    d = Path(image_path).parent
    for _ in range(levels):
        for name in NAMES_FILES:
            p = d / name
            if p.is_file():
                return p
        if d.parent == d:
            break
        d = d.parent
    return None


def parse_yolo_draft(
    label_path: str | Path, shape: tuple[int, int], names_path: str | Path | None = None
) -> YoloDraft:
    """라벨 텍스트 → ``YoloDraft``(박스는 픽셀로, 폴리곤은 정규화 유지). 깨진 줄·names 는 경고로(fail-soft)."""
    lp = Path(label_path)
    parsed = parse_label_text(lp.read_text(encoding="utf-8"))
    items: list[DraftItem] = []
    warnings = list(parsed.warnings)
    for lab in parsed.labels:
        if lab.kind == "box":
            box, clipped = denormalize_box(lab.coords, shape)
            if box is None:
                warnings.append(f"{lab.line_no}행: 박스가 이미지 밖 — 건너뜀")
                continue
            if clipped:
                warnings.append(f"{lab.line_no}행: 박스가 경계를 넘어 잘라냄")
            items.append(DraftItem(lab.line_no, lab.class_id, "box", box, lab.coords))
        else:
            items.append(DraftItem(lab.line_no, lab.class_id, "polygon", None, lab.coords))
    names: list[str] | None = None
    np_ = Path(names_path) if names_path else None
    if np_ is not None:
        # parse_names 는 파일이 아니면 "a,b,c" 문자열로 해석하므로 파일 존재를 먼저 본다
        if not np_.is_file():
            warnings.append(f"names 파일 없음: {np_.name}")
            np_ = None
        else:
            try:
                names = parse_names(np_)
            except (OSError, ValueError) as e:
                warnings.append(f"names 읽기 실패 {np_.name}: {e}")
                np_ = None
    return YoloDraft(lp, items, names, np_, warnings)


def roi_png_path(mask_dir: str | Path, image_path: str | Path) -> Path:
    """``mask_dir`` ROI 가 찾는 파일 이름 — ``<mask_dir>/<대상 stem>.png``."""
    return Path(mask_dir) / f"{Path(image_path).stem}.png"


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
        self.draft: YoloDraft | None = None  # 현재 이미지 옆의 YOLO 라벨(있으면)
        self.draft_confidence: list[
            MaskConfidence
        ] = []  # apply_draft 박스마다 — 손대지 않고 저장하면 메타로
        self.last_auto_confidence: MaskConfidence | None = (
            None  # 마지막 auto_select 의 타당성(상태줄 안내용)
        )

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
        self.draft = None
        self.last_auto_confidence = None

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
        self.draft = None
        self.last_auto_confidence = None

    def set_mask(self, mask: np.ndarray, *, tool: str = "png") -> None:
        """배열 마스크를 현재 마스크로(은행 소스 편집 — 크롭과 같은 크기). 되돌리기 가능."""
        self._require()
        if mask.shape[:2] != self.shape:
            raise LabelError(f"마스크 크기 {mask.shape[:2]} ≠ 이미지 {self.shape}")
        self.push_undo()
        self.mask = binarize(mask)
        self.tools_used.add(tool)
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
        assert self.mask is not None and self.image is not None
        before = self.mask.copy()
        used = self._estimate_box(box, method, margin=margin)
        self.last_auto_confidence = mask_confidence(
            self.image, self.mask & ~before, tuple(int(v) for v in box), margin=margin
        )
        self.tools_used.add("auto")
        self.dirty = True
        return used

    def _estimate_box(self, box: Box, method: str, *, margin: int = 16) -> str:
        """박스 → 추정 마스크를 현재 마스크에 더한다(도구 표시는 호출자가). 시드는 (파일명, 박스)의 crc32."""
        assert self.image is not None and self.mask is not None
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
        self.auto_methods_used.add(used)
        return used

    # ------------------------------------------------------------------ YOLO 초안

    def load_yolo_draft(
        self,
        label_path: str | Path | None = None,
        names_path: str | Path | None = None,
        *,
        find_names: bool = True,
    ) -> YoloDraft | None:
        """현재 이미지의 YOLO 라벨을 찾아(또는 지정) 파싱만 한다 — 마스크는 건드리지 않는다. 없으면 None."""
        self._require()
        lp = Path(label_path) if label_path else (find_yolo_label(self.path) if self.path else None)
        if lp is None or not lp.is_file():
            self.draft = None
            return None
        np_ = names_path
        if np_ is None and find_names and self.path is not None:
            np_ = find_names_file(self.path)
        try:
            self.draft = parse_yolo_draft(lp, self.shape, np_)
        except OSError as e:
            raise LabelError(f"라벨 파일을 읽을 수 없습니다: {e}") from e
        return self.draft

    def apply_draft(self, class_id: int, method: str = "grabcut", *, margin: int = 16) -> list[str]:
        """초안의 ``class_id`` 항목으로 마스크를 **새로 채운다**(기존 마스크는 되돌리기 스택으로). 박스는 추정, 폴리곤은 채움.
        반환 = 박스마다 실제 쓴 방법. 항목이 없으면 ``LabelError``."""
        self._require()
        assert self.mask is not None
        if self.draft is None:
            raise LabelError("YOLO 초안이 없습니다")
        if method not in AUTO_METHODS:
            raise LabelError(
                f"알 수 없는 자동 선택 방법 {method!r} (선택: {', '.join(AUTO_METHODS)})"
            )
        items = self.draft.items_for(class_id)
        if not items:
            raise LabelError(f"초안에 클래스 {self.draft.class_name(class_id)} 항목이 없습니다")
        self.push_undo()
        self.mask[:] = 0
        self.tools_used.clear()
        self.auto_methods_used.clear()
        self.draft_confidence = []
        used: list[str] = []
        for it in items:
            if it.kind == "box" and it.box is not None:
                before = self.mask.copy()
                used.append(self._estimate_box(it.box, method, margin=margin))
                self.tools_used.add("draft-box")
                assert self.image is not None
                self.draft_confidence.append(
                    mask_confidence(self.image, self.mask & ~before, it.box, margin=margin)
                )
            else:
                self.mask = np.maximum(self.mask, polygon_mask(it.coords, self.mask.shape))
                self.tools_used.add("draft-polygon")
        self.dirty = True
        return used

    # ------------------------------------------------------------------ ROI 모드

    def save_roi_png(self, path: str | Path) -> Path:
        """마스크를 원본 크기 0/255 PNG 로 — ``placement.roi: {method: mask_dir, path: …}`` 가 읽는 형식(크롭·메타 없음).
        빈 마스크는 ``LabelError``(빈 ROI 는 "배치 불가"와 같아 실수일 가능성이 높다)."""
        self._require()
        assert self.mask is not None
        if not np.any(self.mask):
            raise LabelError("마스크가 비어 있습니다 — ROI 는 배치를 허용할 영역을 칠한 것입니다")
        out = Path(path)
        if out.suffix.lower() != ".png":
            raise LabelError("ROI 마스크는 PNG 로 저장합니다")
        imgio.write_image(out, binarize(self.mask))
        self.dirty = False
        return out

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
        lighting = mask_lighting(gray, self.mask)
        return LabelStats(
            area_px=area,
            area_ratio=area / float(h * w),
            bbox=(int(x), int(y), int(bw), int(bh)),
            n_components=int(n - 1),
            length_px=length,
            length_um=(length * um_per_px) if um_per_px else None,
            contrast=contrast,
            lighting_deg=lighting,
        )

    # ------------------------------------------------------------------ 은행 저장

    def mask_origin(self) -> str:
        used = self.tools_used - {"morph"}
        if not used:
            return "manual"
        if used <= {"draft-box", "draft-polygon"}:
            # 손대지 않은 YOLO 초안 = 임포터와 같은 출처(박스 추정은 bank ls 의 est 로 센다)
            if "draft-box" in used:
                methods = sorted(self.auto_methods_used)
                return f"yolo-box:{methods[0] if len(methods) == 1 else 'mixed'}"
            return "yolo-polygon"
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
        origin = self.mask_origin()
        conf: float | None = None
        flags: tuple[str, ...] = ()
        if origin.startswith("yolo-box:") and self.draft_confidence:
            # 손대지 않은 초안 = 임포터와 같은 추정 → 가장 낮은 박스 점수와 flags 합집합
            conf = min(c.score for c in self.draft_confidence)
            flags = tuple(sorted({f for c in self.draft_confidence for f in c.flags}))
        rec = ImportRecord(
            image=self.image,
            gray=self.gray,
            mask=binarize(self.mask),
            cls=cls,
            origin=self.path.as_posix() if self.path else stem,
            id_hint=stem,
            mask_origin=origin,
            um_per_px=um_per_px,
            tags=tuple(t.strip() for t in tags if t.strip()),
            confidence=conf,
            flags=flags,
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

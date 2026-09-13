"""임포터 공통 처리 (설계 §3.5) — 은행 **쓰기** 로직은 이 파일에만 있다.

세 임포터(yolo · pairs · dataset)는 ``ImportRecord(image, mask, class, mask_origin, box_in_origin, …)``를 만들어
``BankWriter.add``에 넘긴다.

- 마스크의 **연결 성분마다 소스 하나**(``keep_whole``이면 통째로 하나 — yolo는 항상 이쪽). ``min_area`` 미만 성분은 버리고 로그.
- 크롭 = 성분 bbox를 ``margin``만큼 확장, 이미지 경계에서 잘림. 원본 채널 유지(흑백은 1ch PNG).
- **``margin >= MIN_MARGIN``(= Poisson ``mask_dilate_px`` 기본값 + 1 = 6)** — 블렌딩이 소스 마스크를 팽창한 풀이 영역을
  캔버스 안에 담을 수 있어야 한다. 더 작으면 임포트 자체를 거부한다(``check_margin``).
- 같은 id가 이미 있으면 덮어쓰지 않고 ``-dup<n>``으로 저장 + 경고(은행은 여러 소스에서 누적된다).
- ``bank.yaml``의 ``classes``는 **이름 기준 병합**: 새 이름은 끝에 붙고, 들어온 names 순서가 은행 순서와 다르면 경고
  (출력 ``data.yaml``은 은행 순서를 따른다).
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from anograft.bank.bank import (
    BANK_FILE,
    IMAGE_SUFFIX,
    MASK_SUFFIX,
    META_SUFFIX,
    read_bank_meta,
)
from anograft.core.channels import binarize, demote_from_bgr
from anograft.core.recipe import PoissonBlendConfig
from anograft.core.types import BBox
from anograft.io import imgio

# Poisson 풀이 마스크 팽창(기본 5)이 크롭 캔버스에서 잘리지 않으려면 margin이 그보다 커야 한다.
MIN_MARGIN: int = int(PoissonBlendConfig.model_fields["mask_dilate_px"].default) + 1
DEFAULT_MARGIN = 16
DEFAULT_MIN_AREA = 16

Logger = Callable[[str], None]


def check_margin(margin: int) -> None:
    if margin < MIN_MARGIN:
        raise ValueError(
            f"--margin {margin} < {MIN_MARGIN} — Poisson 풀이 영역(mask_dilate_px {MIN_MARGIN - 1} 팽창)이 크롭에서 잘립니다"
        )


@dataclass(frozen=True)
class ImportRecord:
    """임포터 하나가 만드는 소스 후보. ``image``는 3ch 승격 상태, ``gray``가 원래 채널을 기억한다."""

    image: np.ndarray  # HxWx3 uint8
    gray: bool
    mask: np.ndarray  # HxW uint8 0/255 (이미지 크기)
    cls: str
    origin: str  # 원본 상대경로 (감사용)
    id_hint: str  # 소스 id 기반 (성분 분리 시 -k 접미)
    mask_origin: str = "png"  # png | yolo-polygon | yolo-box:<method>
    box_in_origin: BBox | None = None  # 라벨 박스 원본 (박스 라벨일 때)
    um_per_px: float | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImportOptions:
    margin: int = DEFAULT_MARGIN
    min_area: int = DEFAULT_MIN_AREA
    keep_whole: bool = False


@dataclass(frozen=True)
class AddedSource:
    cls: str
    source_id: str
    area_px: int
    bbox_in_origin: BBox
    mask_origin: str


@dataclass
class ImportStats:
    added: list[AddedSource] = field(default_factory=list)
    dropped_small: int = 0
    duplicates: int = 0
    warnings: list[str] = field(default_factory=list)

    def per_class(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.added:
            out[a.cls] = out.get(a.cls, 0) + 1
        return out


def components(mask: np.ndarray, keep_whole: bool) -> list[np.ndarray]:
    """마스크를 성분 목록으로. ``keep_whole``이면 [mask] 하나. 빈 마스크는 []."""
    if not np.any(mask):
        return []
    if keep_whole:
        return [mask]
    n, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return [((labels == k).astype(np.uint8) * 255) for k in range(1, n)]


def crop_box(mask: np.ndarray, margin: int) -> BBox:
    """성분 bbox + margin, 이미지 경계에서 잘림."""
    h, w = mask.shape[:2]
    x, y, bw, bh = cv2.boundingRect(mask)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


class BankWriter:
    """은행 폴더에 소스를 누적한다. 기존 은행이면 ``bank.yaml``을 읽어 이어 쓴다."""

    def __init__(
        self, root: str | Path, *, name: str | None = None, log: Logger | None = None
    ) -> None:
        self.root = Path(root)
        self.log: Logger = log or (lambda _m: None)
        self.stats = ImportStats()
        meta_path = self.root / BANK_FILE
        if meta_path.is_file():
            self.meta = read_bank_meta(meta_path)
        else:
            self.meta = {
                "name": name or self.root.name,
                "classes": [],
                "um_per_px": None,
                "imports": [],
            }
        self.meta.setdefault("classes", [])
        self.meta.setdefault("imports", [])
        self.meta.setdefault("um_per_px", None)
        if name:
            self.meta["name"] = name

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.meta["classes"]]

    def ensure_classes(self, names: Sequence[str]) -> list[str]:
        """이름 기준 병합. 새 이름은 끝에. 들어온 순서가 은행 순서와 다르면 경고 문자열을 돌려준다(그리고 stats에도)."""
        warnings: list[str] = []
        existing = self.classes
        for n in names:
            if n not in existing:
                existing.append(n)
        shared = [n for n in names if n in self.meta["classes"]]
        bank_order = [n for n in self.meta["classes"] if n in shared]
        if shared and bank_order != shared:
            warnings.append(
                f"클래스 id 순서가 원본과 다릅니다 — 은행 {bank_order} vs 들어온 {shared}. 출력 data.yaml은 은행 순서를 따릅니다"
            )
        self.meta["classes"] = existing
        for w in warnings:
            self._warn(w)
        return warnings

    def _warn(self, msg: str) -> None:
        self.stats.warnings.append(msg)
        self.log(msg)

    def _unique_id(self, cls: str, source_id: str) -> str:
        cdir = self.root / cls
        if not (cdir / f"{source_id}{META_SUFFIX}").exists():
            return source_id
        n = 1
        while (cdir / f"{source_id}-dup{n}{META_SUFFIX}").exists():
            n += 1
        self.stats.duplicates += 1
        self._warn(f"같은 id가 이미 있어 {cls}/{source_id}-dup{n} 으로 저장")
        return f"{source_id}-dup{n}"

    def add(self, rec: ImportRecord, opts: ImportOptions) -> list[AddedSource]:
        """레코드 하나 → 성분 분리 → 크롭·마스크·메타 기록. 기록된 소스 목록을 돌려준다."""
        check_margin(opts.margin)
        if rec.cls not in self.meta["classes"]:
            self.meta["classes"].append(rec.cls)
        parts = components(binarize(rec.mask), opts.keep_whole)
        multi = len(parts) > 1
        added: list[AddedSource] = []
        for k, part in enumerate(parts):
            area = int(np.count_nonzero(part))
            if area < opts.min_area:
                self.stats.dropped_small += 1
                self.log(f"{rec.origin}: 성분 {k} 면적 {area}px < {opts.min_area} — 버림")
                continue
            sid = f"{rec.id_hint}-{k}" if multi else rec.id_hint
            sid = self._unique_id(rec.cls, sid)
            x, y, w, h = crop_box(part, opts.margin)
            crop = rec.image[y : y + h, x : x + w]
            mcrop = part[y : y + h, x : x + w]
            cdir = self.root / rec.cls
            cdir.mkdir(parents=True, exist_ok=True)
            imgio.write_image(cdir / f"{sid}{IMAGE_SUFFIX}", demote_from_bgr(crop, rec.gray))
            imgio.write_image(cdir / f"{sid}{MASK_SUFFIX}", mcrop)
            meta: dict[str, Any] = {
                "class": rec.cls,
                "origin": rec.origin,
                "bbox_in_origin": [x, y, w, h],
                "box_in_origin": list(rec.box_in_origin) if rec.box_in_origin else None,
                "mask_origin": rec.mask_origin,
                "margin_px": opts.margin,
                "um_per_px": rec.um_per_px,
                "area_px": area,
                "tags": list(rec.tags),
            }
            (cdir / f"{sid}{META_SUFFIX}").write_text(
                json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            a = AddedSource(rec.cls, sid, area, (x, y, w, h), rec.mask_origin)
            added.append(a)
            self.stats.added.append(a)
        return added

    def finish(self, entry: dict[str, Any] | None = None) -> Path:
        """임포트 이력 한 줄을 붙이고 ``bank.yaml``을 쓴다."""
        if entry is not None:
            e = {"when": _dt.datetime.now().isoformat(timespec="seconds")}
            e.update(entry)
            e.setdefault("n_sources", len(self.stats.added))
            self.meta["imports"].append(e)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / BANK_FILE
        path.write_text(
            yaml.safe_dump(self.meta, sort_keys=False, allow_unicode=True, default_flow_style=None),
            encoding="utf-8",
        )
        return path

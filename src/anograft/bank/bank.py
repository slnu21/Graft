"""결함 은행 로더 (설계 §3.1).

디스크 포맷::

    bank/<name>/
      bank.yaml                 # name · classes(순서 = id) · um_per_px(기본값) · imports(임포트 이력)
      <class>/<id>.png          # bbox+margin 크롭, 원본 채널 유지(흑백은 1ch)
      <class>/<id>.mask.png     # 크롭과 같은 크기, 0/255
      <class>/<id>.json         # origin · bbox_in_origin · box_in_origin · mask_origin · margin_px · um_per_px · area_px · tags

규칙:
- ``classes`` 순서가 곧 class id. 은행에 없던 클래스는 임포트 때 끝에 붙는다(``importers/common``).
- 소스는 **이름 정렬 순**으로 로드한다 — 디렉터리 열람 순서는 OS마다 달라 재현성을 깬다.
- 마스크는 ``>127 → 255`` 이진화, 흑백 크롭은 3ch로 승격(코어 규약: 이미지는 항상 HxWx3).
- ``fingerprint()`` = ``seeds.bank_fingerprint`` — 은행이 바뀌면 파이프라인 해시가 바뀐다.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from anograft.core.seeds import bank_fingerprint
from anograft.core.types import DefectSource
from anograft.io import imgio

BANK_FILE = "bank.yaml"
MASK_SUFFIX = ".mask.png"
META_SUFFIX = ".json"
IMAGE_SUFFIX = ".png"

ESTIMATED_PREFIX = "yolo-box:"  # 이 접두사의 mask_origin은 추정 마스크


class BankError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClassSummary:
    """``bank ls`` 한 행."""

    cls: str
    class_id: int
    count: int
    area_median: float
    exact: int  # mask_origin이 png/yolo-polygon
    estimated: int  # mask_origin이 yolo-box:*
    origins: Mapping[str, int]


class Bank:
    """메모리에 올린 은행. ``load``가 정본 경로, ``from_sources``는 테스트·GUI용."""

    def __init__(
        self,
        name: str,
        classes: Sequence[str],
        sources: Iterable[DefectSource],
        *,
        um_per_px: float | None = None,
        root: Path | None = None,
        imports: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.name = name
        self._classes = list(dict.fromkeys(classes))  # 중복 제거, 순서 유지
        self.um_per_px = um_per_px
        self.root = root
        self.imports = [dict(i) for i in imports]
        self.warnings: list[str] = []  # 로드 중 경고 (깨진 소스 등)
        self._by_class: dict[str, list[DefectSource]] = {c: [] for c in self._classes}
        for s in sorted(sources, key=lambda s: s.id):
            if s.cls not in self._by_class:
                # classes에 없는 클래스의 소스 — 끝에 붙인다(fail-soft, 로더가 경고를 남긴다)
                self._classes.append(s.cls)
                self._by_class[s.cls] = []
            self._by_class[s.cls].append(s)

    # ------------------------------------------------------------------ 인터페이스 (BankLike)

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    @property
    def class_ids(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self._classes)}

    def counts(self) -> dict[str, int]:
        return {c: len(v) for c, v in self._by_class.items()}

    def by_class(self, cls: str) -> list[DefectSource]:
        return list(self._by_class.get(cls, []))

    def sources(self) -> list[DefectSource]:
        return [s for c in self._classes for s in self._by_class[c]]

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_class.values())

    def fingerprint(self) -> str:
        return bank_fingerprint((s.id, int(np.count_nonzero(s.mask))) for s in self.sources())

    def summary(self) -> list[ClassSummary]:
        out: list[ClassSummary] = []
        for i, c in enumerate(self._classes):
            srcs = self._by_class[c]
            areas = [int(np.count_nonzero(s.mask)) for s in srcs]
            origins: dict[str, int] = {}
            for s in srcs:
                origins[s.mask_origin] = origins.get(s.mask_origin, 0) + 1
            est = sum(n for o, n in origins.items() if o.startswith(ESTIMATED_PREFIX))
            out.append(
                ClassSummary(
                    cls=c,
                    class_id=i,
                    count=len(srcs),
                    area_median=float(statistics.median(areas)) if areas else 0.0,
                    exact=len(srcs) - est,
                    estimated=est,
                    origins=origins,
                )
            )
        return out

    # ------------------------------------------------------------------ 로드

    @classmethod
    def from_sources(
        cls,
        sources: Iterable[DefectSource],
        *,
        classes: Sequence[str] | None = None,
        name: str = "memory",
        um_per_px: float | None = None,
    ) -> Bank:
        """디스크 없이 소스 목록으로 은행을 만든다. ``classes``를 안 주면 등장 순서(정렬된 id 기준)."""
        srcs = sorted(sources, key=lambda s: s.id)
        if classes is None:
            classes = list(dict.fromkeys(s.cls for s in srcs))
        return cls(name, classes, srcs, um_per_px=um_per_px)

    @classmethod
    def load(cls, path: str | Path, *, warn: Any = None) -> Bank:
        """``bank.yaml``과 클래스 폴더를 읽는다. 깨진 소스 하나는 경고 후 건너뛴다(fail-soft); 은행 자체가 없으면 ``BankError``."""
        root = Path(path)
        meta_path = root / BANK_FILE
        if not meta_path.is_file():
            raise BankError(f"은행이 아닙니다 ({BANK_FILE} 없음): {root}")
        meta = read_bank_meta(meta_path)
        classes: list[str] = [str(c) for c in meta.get("classes") or []]
        default_um = meta.get("um_per_px")
        warnings: list[str] = []
        sources: list[DefectSource] = []
        # 폴더는 있는데 classes에 없는 클래스 — 끝에 붙이고 경고 (수동으로 폴더를 만든 경우)
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            if d.name not in classes and any(d.glob(f"*{META_SUFFIX}")):
                warnings.append(f"bank.yaml classes에 없는 클래스 폴더 '{d.name}' — 끝에 추가")
                classes.append(d.name)
        for c in classes:
            cdir = root / c
            if not cdir.is_dir():
                continue
            for meta_file in sorted(cdir.glob(f"*{META_SUFFIX}")):
                try:
                    sources.append(
                        load_source(cdir, meta_file.name[: -len(META_SUFFIX)], c, default_um)
                    )
                except (OSError, ValueError, imgio.ImageReadError) as e:
                    warnings.append(f"소스 로드 실패 {c}/{meta_file.stem}: {e}")
        bank = cls(
            str(meta.get("name") or root.name),
            classes,
            sources,
            um_per_px=float(default_um) if default_um is not None else None,
            root=root,
            imports=meta.get("imports") or [],
        )
        bank.warnings = warnings
        if warn is not None:
            for w in warnings:
                warn(w)
        return bank


def read_bank_meta(meta_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise BankError(f"{BANK_FILE} 형식 오류(매핑이 아님): {meta_path}")
    return data


def load_source(
    class_dir: Path, source_id: str, cls: str, default_um: float | None
) -> DefectSource:
    """``<class>/<id>.{png,mask.png,json}`` 세 파일 → ``DefectSource``. 흑백 크롭은 3ch 승격, 마스크는 이진화."""
    image, _gray = imgio.read_image(class_dir / f"{source_id}{IMAGE_SUFFIX}")
    mask = imgio.read_mask(class_dir / f"{source_id}{MASK_SUFFIX}")
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError(f"마스크 크기 {mask.shape[:2]} ≠ 크롭 크기 {image.shape[:2]}")
    meta = json.loads((class_dir / f"{source_id}{META_SUFFIX}").read_text(encoding="utf-8"))
    um = meta.get("um_per_px", default_um)
    return DefectSource(
        id=f"{cls}/{source_id}",
        cls=cls,
        image=image,
        mask=mask,
        um_per_px=float(um) if um is not None else None,
        tags=tuple(str(t) for t in meta.get("tags") or ()),
        origin=str(meta.get("origin") or ""),
        mask_origin=str(meta.get("mask_origin") or "png"),
    )

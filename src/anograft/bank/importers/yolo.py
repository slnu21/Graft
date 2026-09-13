"""YOLO 임포터 (설계 §3.2) ★ 보유 라벨 형식의 기본 경로.

라벨 ``.txt``의 한 줄 = 소스 하나:

| 줄 | 해석 | 마스크 |
|---|---|---|
| ``c cx cy w h`` (5 토큰, 정규화) | 박스 | 추정(``mask_from``) → ``mask_origin: yolo-box:<method_used>`` |
| ``c x1 y1 … xn yn`` (≥7 토큰, 홀수) | YOLO-seg 폴리곤 | ``cv2.fillPoly`` → ``yolo-polygon`` |
| 빈 파일 / 파일 없음 | 결함 없는 이미지 | 소스 없음 · ``list_normals``에 기록 |

- 매칭: ``<labels>/<이미지 상대경로 stem>.txt`` — 하위 폴더 구조를 그대로 미러링.
- 좌표는 이미지 크기로 역정규화 후 반올림·클립. 박스가 이미지 밖으로 나가면 잘라내고 경고.
- 박스 하나 = 소스 하나(``keep_whole``). ``id = f"{stem}-{line_no:02d}"``(라벨 파일의 1-기반 줄 번호).
- ``names``: data.yaml(``names`` list|dict) · classes.txt(한 줄 하나) · ``a,b,c`` 직접.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from anograft.bank.importers.common import (
    DEFAULT_MARGIN,
    DEFAULT_MIN_AREA,
    BankWriter,
    ImportOptions,
    ImportRecord,
    ImportStats,
    check_margin,
)
from anograft.bank.mask_from_box import METHODS, mask_from_box, stable_seed
from anograft.core.types import BBox
from anograft.io import imgio

Logger = Callable[[str], None]


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------


def parse_names(spec: str | Path) -> list[str]:
    """``data.yaml``(names: list|dict) · ``classes.txt``(한 줄 하나) · ``a,b,c``. id = 목록 순서."""
    p = Path(spec)
    if p.is_file():
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(text)
            names = data.get("names") if isinstance(data, dict) else None
            if isinstance(names, dict):
                keys = sorted(names, key=lambda k: int(k))
                if [int(k) for k in keys] != list(range(len(keys))):
                    raise ValueError(f"names 딕셔너리의 id가 0..n-1 연속이 아닙니다: {keys}")
                return [str(names[k]) for k in keys]
            if isinstance(names, list):
                return [str(n) for n in names]
            raise ValueError(f"{p}: 'names' 키(list 또는 dict)가 없습니다")
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    names = [s.strip() for s in str(spec).split(",") if s.strip()]
    if not names:
        raise ValueError(f"names를 해석할 수 없습니다: {spec!r} (data.yaml · classes.txt · a,b,c)")
    return names


# ---------------------------------------------------------------------------
# 라벨 파싱
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Label:
    line_no: int  # 1-기반
    class_id: int
    kind: str  # "box" | "polygon"
    coords: tuple[float, ...]  # 정규화. box: (cx, cy, w, h) / polygon: (x1, y1, …)


@dataclass
class ParsedLabels:
    labels: list[Label] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def parse_label_text(text: str) -> ParsedLabels:
    """빈 줄은 무시. 5토큰 = 박스, ≥7 홀수 = 폴리곤, 그 외는 경고 + 건너뜀."""
    out = ParsedLabels()
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        try:
            cid = int(float(tokens[0]))
            vals = tuple(float(t) for t in tokens[1:])
        except ValueError:
            out.warnings.append(f"{i}행: 숫자가 아닌 토큰 — 건너뜀: {line[:40]}")
            continue
        if len(tokens) == 5:
            out.labels.append(Label(i, cid, "box", vals))
        elif len(tokens) >= 7 and len(tokens) % 2 == 1:
            out.labels.append(Label(i, cid, "polygon", vals))
        else:
            out.warnings.append(
                f"{i}행: 토큰 {len(tokens)}개 — 박스(5)도 폴리곤(홀수 ≥7)도 아님, 건너뜀"
            )
    return out


def denormalize_box(coords: Sequence[float], shape: tuple[int, ...]) -> tuple[BBox | None, bool]:
    """``(cx, cy, w, h)`` 정규화 → 정수 ``(x, y, w, h)``, 클립. 반환 ``(box | None, clipped)``."""
    h, w = shape[:2]
    cx, cy, bw, bh = coords
    x0 = round((cx - bw / 2.0) * w)
    y0 = round((cy - bh / 2.0) * h)
    x1 = round((cx + bw / 2.0) * w)
    y1 = round((cy + bh / 2.0) * h)
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    clipped = (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return None, clipped
    return (cx0, cy0, cx1 - cx0, cy1 - cy0), clipped


def polygon_mask(coords: Sequence[float], shape: tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    pts = np.array(coords, dtype=np.float64).reshape(-1, 2) * np.array([w, h], dtype=np.float64)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
    return m


# ---------------------------------------------------------------------------
# 매칭
# ---------------------------------------------------------------------------


def find_pairs(images: Path, labels: Path) -> list[tuple[Path, Path | None]]:
    """이미지(재귀, 이름 정렬)와 ``<labels>/<상대경로 stem>.txt``. 라벨이 없으면 None."""
    out: list[tuple[Path, Path | None]] = []
    for img in sorted(
        p for p in images.rglob("*") if p.is_file() and p.suffix.lower() in imgio.IMAGE_SUFFIXES
    ):
        rel = img.relative_to(images)
        lab = labels / rel.with_suffix(".txt")
        out.append((img, lab if lab.is_file() else None))
    return out


# ---------------------------------------------------------------------------
# 임포트
# ---------------------------------------------------------------------------


@dataclass
class YoloImportResult:
    bank_root: Path
    classes: list[str]
    n_images: int
    normals: list[Path]
    stats: ImportStats
    mask_methods: dict[str, int]
    warnings: list[str]


def _rel_for_list(path: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:  # 드라이브가 다르면 상대경로 불가
        return path.resolve().as_posix()


def write_normals_list(list_file: Path, normals: Sequence[Path]) -> None:
    """``--list-normals`` — 목록 파일 기준 상대경로(POSIX). ``imgio.read_path_list``가 그대로 읽는다."""
    list_file.parent.mkdir(parents=True, exist_ok=True)
    base = list_file.resolve().parent
    lines = [
        "# anograft import-yolo — 라벨이 빈 이미지(정상 후보). inputs.targets에 이 파일을 지정"
    ]
    lines += [_rel_for_list(p.resolve(), base) for p in normals]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def import_yolo(
    images: str | Path,
    labels: str | Path,
    names: str | Path | Sequence[str],
    out: str | Path,
    *,
    mask_from: str = "grabcut",
    box_margin: int = 6,
    min_box: int = 8,
    margin: int = DEFAULT_MARGIN,
    min_area: int = DEFAULT_MIN_AREA,
    um_per_px: float | None = None,
    tags: Sequence[str] = (),
    list_normals: str | Path | None = None,
    log: Logger | None = None,
) -> YoloImportResult:
    """보유 YOLO 라벨 → 은행. 이미지 하나가 깨져도 경고 후 계속(fail-soft)."""
    log = log or (lambda _m: None)
    if mask_from not in METHODS:
        raise ValueError(f"--mask-from {mask_from!r}: 선택은 {', '.join(METHODS)}")
    check_margin(margin)
    images_dir, labels_dir = Path(images), Path(labels)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"라벨 폴더가 없습니다: {labels_dir}")
    name_list = list(names) if not isinstance(names, (str, Path)) else parse_names(names)

    writer = BankWriter(out, log=log)
    writer.ensure_classes(name_list)
    opts = ImportOptions(margin=margin, min_area=min_area, keep_whole=True)
    warnings: list[str] = []
    normals: list[Path] = []
    methods_used: dict[str, int] = {}
    pairs = find_pairs(images_dir, labels_dir)

    def warn(msg: str) -> None:
        warnings.append(msg)
        log(msg)

    for img_path, lab_path in pairs:
        rel = img_path.relative_to(images_dir).as_posix()
        parsed = (
            parse_label_text(lab_path.read_text(encoding="utf-8")) if lab_path else ParsedLabels()
        )
        for w in parsed.warnings:
            warn(f"{rel}: {w}")
        if not parsed.labels:
            normals.append(img_path)
            continue
        try:
            image, gray = imgio.read_image(img_path)
        except imgio.ImageReadError as e:
            warn(f"{rel}: 이미지 읽기 실패 — {e}")
            continue
        stem = img_path.stem
        for lab in parsed.labels:
            if not 0 <= lab.class_id < len(name_list):
                warn(
                    f"{rel} {lab.line_no}행: class id {lab.class_id}가 names 범위 밖(0..{len(name_list) - 1}) — 건너뜀"
                )
                continue
            cls = name_list[lab.class_id]
            sid = f"{stem}-{lab.line_no:02d}"
            box: BBox | None = None
            if lab.kind == "box":
                box, clipped = denormalize_box(lab.coords, image.shape)
                if box is None:
                    warn(f"{rel} {lab.line_no}행: 박스가 이미지 밖 — 건너뜀")
                    continue
                if clipped:
                    warn(f"{rel} {lab.line_no}행: 박스가 이미지 경계를 넘어 잘라냄 → {list(box)}")
                mask, used = mask_from_box(
                    image,
                    box,
                    mask_from,
                    box_margin,
                    min_box=min_box,
                    seed=stable_seed(f"{cls}/{sid}"),
                )
                origin = f"yolo-box:{used}"
                methods_used[used] = methods_used.get(used, 0) + 1
            else:
                mask = polygon_mask(lab.coords, image.shape)
                origin = "yolo-polygon"
            rec = ImportRecord(
                image=image,
                gray=gray,
                mask=mask,
                cls=cls,
                origin=rel,
                id_hint=sid,
                mask_origin=origin,
                box_in_origin=box,
                um_per_px=um_per_px,
                tags=tuple(tags),
            )
            if not writer.add(rec, opts):
                warn(f"{rel} {lab.line_no}행: 마스크 면적이 0이거나 min_area 미만 — 소스 없음")

    entry: dict[str, Any] = {
        "importer": "yolo",
        "images": str(images_dir),
        "labels": str(labels_dir),
        "names": name_list,
        "mask_from": mask_from,
        "n_images": len(pairs),
        "n_normals": len(normals),
        "mask_methods_used": methods_used,
    }
    writer.finish(entry)
    if list_normals is not None:
        write_normals_list(Path(list_normals), normals)
    return YoloImportResult(
        bank_root=Path(out),
        classes=writer.classes,
        n_images=len(pairs),
        normals=normals,
        stats=writer.stats,
        mask_methods=methods_used,
        warnings=warnings + writer.stats.warnings,
    )

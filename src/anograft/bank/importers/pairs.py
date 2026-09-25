"""마스크 PNG 쌍 임포터 (설계 §3.3) — ``(image, mask, class)`` 쌍 목록을 은행으로.

두 진입점이 같은 공통 처리(``import_pair_records``)를 탄다:

- ``import_pairs(images, masks, out, …)``: 폴더에서 쌍을 **찾는다** — ``<masks>/<상대 stem><suffix>.<ext>`` → 없으면
  ``<masks>/<상대 stem>.<ext>`` → 없으면 경고 후 건너뜀. 클래스는 ``cls`` 하나 고정 또는 ``class_from_dir``
  (``images/<class>/x.png`` 첫 폴더명). ``csv``(``image,mask,class`` 열, 경로는 CSV 파일 기준)를 주면 탐색 대신 그 목록.
- 표준 데이터셋 어댑터(``datasets/``)는 자기 레이아웃을 ``PairRecord`` 목록으로 바꿔 ``import_pair_records``에 넘긴다 —
  어댑터에는 은행 쓰기 로직이 없다(변환기).

- 마스크 값은 ``>127``을 결함으로. 팔레트 값으로 다중 클래스를 담은 마스크는 v0.1 미지원(``csv``로 클래스별 마스크를 따로).
- 기본은 **연결 성분마다 소스 하나**(``keep_whole``이면 통째로). ``id = <stem>[-k]``, ``mask_origin: png``.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anograft.bank.importers.common import (
    DEFAULT_MARGIN,
    DEFAULT_MIN_AREA,
    BankWriter,
    ImportOptions,
    ImportRecord,
    ImportStats,
    check_margin,
)
from anograft.io import imgio

Logger = Callable[[str], None]
DEFAULT_MASK_SUFFIX = "_mask"


@dataclass(frozen=True)
class PairRecord:
    """쌍 하나 — 임포터·어댑터 공통 화폐. ``origin``은 감사용 상대경로(없으면 이미지 파일명)."""

    image: Path
    mask: Path
    cls: str
    origin: str | None = None
    id_hint: str | None = None  # 기본 = 이미지 stem
    tags: tuple[str, ...] = ()
    mask_threshold: int = 127  # 마스크 이진화 문턱(``> threshold``). 0/1 라벨맵(VisA)은 0


@dataclass
class PairsImportResult:
    bank_root: Path
    classes: list[str]
    n_pairs: int  # 공통 처리에 들어간 쌍 수
    n_missing_mask: int  # 마스크를 못 찾아 건너뛴 이미지 수 (폴더 탐색 모드)
    stats: ImportStats
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 쌍 찾기
# ---------------------------------------------------------------------------


def find_mask(masks: Path, rel: Path, suffix: str) -> Path | None:
    """``<masks>/<rel-stem><suffix>.<ext>`` → ``<masks>/<rel-stem>.<ext>`` (ext는 이미지 확장자 전부)."""
    base = masks / rel.parent
    for stem in (f"{rel.stem}{suffix}", rel.stem) if suffix else (rel.stem,):
        for ext in sorted(imgio.IMAGE_SUFFIXES):
            for cand in (base / f"{stem}{ext}", base / f"{stem}{ext.upper()}"):
                if cand.is_file():
                    return cand
    return None


def discover_pairs(
    images: Path,
    masks: Path,
    *,
    cls: str | None,
    class_from_dir: bool,
    mask_suffix: str,
    warn: Logger,
) -> tuple[list[PairRecord], int]:
    """폴더 탐색. 반환 ``(쌍 목록, 마스크 없는 이미지 수)``. 이미지는 재귀·이름 정렬."""
    if (cls is None) == (not class_from_dir):
        raise ValueError("--class NAME 또는 --class-from-dir 중 하나를 주세요")
    out: list[PairRecord] = []
    missing = 0
    for img in sorted(
        p for p in images.rglob("*") if p.is_file() and p.suffix.lower() in imgio.IMAGE_SUFFIXES
    ):
        rel = img.relative_to(images)
        if class_from_dir:
            if len(rel.parts) < 2:
                warn(f"{rel.as_posix()}: --class-from-dir 인데 상위 클래스 폴더가 없음 — 건너뜀")
                continue
            c = rel.parts[0]
        else:
            c = str(cls)
        m = find_mask(masks, rel, mask_suffix)
        if m is None:
            missing += 1
            warn(
                f"{rel.as_posix()}: 마스크 없음 ({masks.as_posix()}/{rel.with_suffix('').as_posix()}{mask_suffix}.*)"
            )
            continue
        out.append(PairRecord(img, m, c, origin=rel.as_posix()))
    return out, missing


def read_pairs_csv(path: Path) -> list[PairRecord]:
    """``image,mask,class`` 헤더 CSV. 상대경로는 CSV 파일 위치 기준."""
    base = path.resolve().parent
    out: list[PairRecord] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        need = {"image", "mask", "class"}
        if reader.fieldnames is None or not need <= set(reader.fieldnames):
            raise ValueError(f"{path}: CSV 헤더에 image,mask,class 열이 필요합니다")
        for i, row in enumerate(reader, start=2):
            img, m, c = [(row.get(k) or "").strip() for k in ("image", "mask", "class")]
            if not (img and m and c):
                raise ValueError(f"{path}:{i}: image/mask/class 가 비어 있음")
            ip, mp = Path(img), Path(m)
            ip = ip if ip.is_absolute() else base / ip
            mp = mp if mp.is_absolute() else base / mp
            out.append(PairRecord(ip, mp, c, origin=img))
    return out


# ---------------------------------------------------------------------------
# 공통 처리 — 어댑터도 여기로
# ---------------------------------------------------------------------------


def import_pair_records(
    pairs: Iterable[PairRecord],
    out: str | Path,
    *,
    margin: int = DEFAULT_MARGIN,
    min_area: int = DEFAULT_MIN_AREA,
    keep_whole: bool = False,
    um_per_px: float | None = None,
    tags: Sequence[str] = (),
    mask_origin: str = "png",
    entry: dict[str, Any] | None = None,
    log: Logger | None = None,
) -> PairsImportResult:
    """쌍 목록 → 은행. 쌍 하나가 깨져도 경고 후 계속(fail-soft). ``entry``는 bank.yaml imports 이력에 덧붙일 키.

    ``mask_origin`` 기본값 ``png`` 는 "사람이 준 정확한 마스크"라는 뜻이다. 모델이 낸 마스크를 넣을 때는
    ``pred:<학습기>`` 를 준다(루프 T5) — 추정 마스크는 추정이라고 적어야 `bank ls` 의 ``est`` 가 사실이 된다.
    """
    log = log or (lambda _m: None)
    check_margin(margin)
    writer = BankWriter(out, log=log)
    opts = ImportOptions(margin=margin, min_area=min_area, keep_whole=keep_whole)
    warnings: list[str] = []
    n = 0

    def warn(msg: str) -> None:
        warnings.append(msg)
        log(msg)

    for pr in pairs:
        origin = pr.origin or pr.image.name
        try:
            image, gray = imgio.read_image(pr.image)
            mask = imgio.read_mask(pr.mask, threshold=pr.mask_threshold)
        except imgio.ImageReadError as e:
            warn(f"{origin}: 읽기 실패 — {e}")
            continue
        if mask.shape[:2] != image.shape[:2]:
            warn(f"{origin}: 마스크 크기 {mask.shape[:2]} ≠ 이미지 {image.shape[:2]} — 건너뜀")
            continue
        n += 1
        rec = ImportRecord(
            image=image,
            gray=gray,
            mask=mask,
            cls=pr.cls,
            origin=origin,
            id_hint=pr.id_hint or pr.image.stem,
            mask_origin=mask_origin,
            um_per_px=um_per_px,
            tags=tuple(tags) + tuple(pr.tags),
        )
        if not writer.add(rec, opts):
            warn(f"{origin}: 마스크가 비었거나 성분이 전부 min_area 미만 — 소스 없음")
    e: dict[str, Any] = {"importer": "pairs", "n_pairs": n}
    if entry:
        e.update(entry)
    writer.finish(e)
    return PairsImportResult(
        bank_root=Path(out),
        classes=writer.classes,
        n_pairs=n,
        n_missing_mask=0,
        stats=writer.stats,
        warnings=warnings + writer.stats.warnings,
    )


def import_pairs(
    images: str | Path,
    masks: str | Path | None,
    out: str | Path,
    *,
    cls: str | None = None,
    class_from_dir: bool = False,
    mask_suffix: str = DEFAULT_MASK_SUFFIX,
    csv_path: str | Path | None = None,
    margin: int = DEFAULT_MARGIN,
    min_area: int = DEFAULT_MIN_AREA,
    keep_whole: bool = False,
    um_per_px: float | None = None,
    tags: Sequence[str] = (),
    log: Logger | None = None,
) -> PairsImportResult:
    """마스크 PNG 쌍 → 은행. ``csv_path``가 있으면 폴더 탐색 대신 그 목록(``images``·``masks``는 무시)."""
    log = log or (lambda _m: None)
    check_margin(margin)
    pre_warnings: list[str] = []

    def warn(msg: str) -> None:
        pre_warnings.append(msg)
        log(msg)

    if csv_path is not None:
        pairs = read_pairs_csv(Path(csv_path))
        missing = 0
        entry: dict[str, Any] = {"csv": str(csv_path)}
    else:
        images_dir = Path(images)
        if not images_dir.is_dir():
            raise FileNotFoundError(f"이미지 폴더가 없습니다: {images_dir}")
        if masks is None or not Path(masks).is_dir():
            raise FileNotFoundError(f"마스크 폴더가 없습니다: {masks}")
        pairs, missing = discover_pairs(
            images_dir,
            Path(masks),
            cls=cls,
            class_from_dir=class_from_dir,
            mask_suffix=mask_suffix,
            warn=warn,
        )
        entry = {
            "images": str(images_dir),
            "masks": str(masks),
            "class": cls,
            "class_from_dir": class_from_dir,
            "mask_suffix": mask_suffix,
            "n_missing_mask": missing,
        }
    res = import_pair_records(
        pairs,
        out,
        margin=margin,
        min_area=min_area,
        keep_whole=keep_whole,
        um_per_px=um_per_px,
        tags=tags,
        entry=entry,
        log=log,
    )
    res.n_missing_mask = missing
    res.warnings = pre_warnings + res.warnings
    return res

"""합성 대상(정상 이미지) 풀 (설계 §7 2단계).

``inputs.targets``는 **폴더**(이미지 파일 이름 정렬) 또는 **경로 목록 ``.txt``**(줄 순서, 빈 줄·``#`` 무시, 상대경로는
목록 파일 기준 — ``import-yolo --list-normals`` 산출물). 어느 쪽이든 순서가 곧 인덱스라 재현성의 일부다.
"""

from __future__ import annotations

from pathlib import Path

from anograft.core.types import TargetImage
from anograft.io import imgio


class TargetsError(RuntimeError):
    pass


def list_targets(spec: str | Path) -> list[Path]:
    """폴더 → ``imgio.list_images``(이름 정렬) · ``.txt`` → ``imgio.read_path_list``(줄 순서). 비어 있으면 ``TargetsError``."""
    p = Path(spec)
    if p.is_dir():
        paths = imgio.list_images(p)
        what = f"폴더 {p}"
    elif p.is_file() and p.suffix.lower() == ".txt":
        paths = imgio.read_path_list(p)
        what = f"목록 {p}"
    elif p.is_file():
        raise TargetsError(f"inputs.targets 는 폴더 또는 .txt 목록이어야 합니다: {p}")
    else:
        raise TargetsError(f"inputs.targets 가 없습니다: {p}")
    if not paths:
        raise TargetsError(f"대상 이미지가 0장입니다 ({what})")
    return paths


def stem_collisions(paths: list[Path]) -> list[tuple[Path, Path]]:
    """같은 폴더·같은 stem 인 파일 쌍(예: ``a.jpg`` + ``a.png``) — 대상 폴더에 마스크 PNG 가 이미지 옆에 놓인 흔한 배치
    (Magnetic Tile ``MT_Free/Imgs``)에서 마스크까지 대상으로 뽑히는 것을 prepare 경고로. 순서는 ``paths`` 순."""
    seen: dict[tuple[Path, str], Path] = {}
    out: list[tuple[Path, Path]] = []
    for q in paths:
        key = (q.parent, q.stem)
        if key in seen:
            out.append((seen[key], q))
        else:
            seen[key] = q
    return out


def targets_warning(paths: list[Path]) -> str | None:
    """``stem_collisions`` 가 있으면 한 줄(``targets:`` 접두) — 대상 수·예시·조치(.txt 목록)."""
    pairs = stem_collisions(paths)
    if not pairs:
        return None
    a, b = pairs[0]
    return (
        f"targets: 같은 이름의 파일이 확장자만 다르게 {len(pairs)}쌍 있습니다(예: {a.name} + {b.name}) — "
        "마스크 PNG 가 이미지 옆에 있으면 마스크도 대상으로 뽑힙니다 → inputs.targets 를 .txt 목록으로(한 확장자만)"
    )


def load_target(path: Path, um_per_px: float | None = None) -> TargetImage:
    """파일 → ``TargetImage``(3ch 승격, ``gray`` 기억). 읽기 실패는 ``imgio.ImageReadError`` — 실행기가 그 인덱스를 skipped로."""
    image, gray = imgio.read_image(path)
    return TargetImage(path=Path(path), image=image, gray=gray, um_per_px=um_per_px)

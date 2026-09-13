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


def load_target(path: Path, um_per_px: float | None = None) -> TargetImage:
    """파일 → ``TargetImage``(3ch 승격, ``gray`` 기억). 읽기 실패는 ``imgio.ImageReadError`` — 실행기가 그 인덱스를 skipped로."""
    image, gray = imgio.read_image(path)
    return TargetImage(path=Path(path), image=image, gray=gray, um_per_px=um_per_px)

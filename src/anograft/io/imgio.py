"""한글 경로 안전 이미지 IO + 채널 승격/복원.

``cv2.imread``/``cv2.imwrite``는 Windows 비-ASCII 경로에서 조용히 실패한다(None 반환). 코드베이스 어디서도
직접 부르지 않고(테스트가 grep으로 막는다) 여기의 ``np.fromfile + imdecode`` / ``imencode + write_bytes``만 쓴다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from anograft.core.channels import binarize, demote_from_bgr, promote_to_bgr

__all__ = [
    "IMAGE_SUFFIXES",
    "ImageReadError",
    "binarize",
    "demote_from_bgr",
    "list_images",
    "promote_to_bgr",
    "read_image",
    "read_mask",
    "read_path_list",
    "write_image",
]

IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)


class ImageReadError(RuntimeError):
    pass


def read_image(path: str | Path) -> tuple[np.ndarray, bool]:
    """이미지를 ``HxWx3 uint8``로 읽는다. 반환 ``(image, gray)`` — gray는 원본이 1채널이었는지.

    - 4채널(BGRA)은 알파를 버리고 BGR로.
    - 16-bit는 v0.1 범위 밖 — 명확한 에러.
    """
    p = Path(path)
    if not p.is_file():
        raise ImageReadError(f"파일이 없습니다: {p}")
    buf = np.fromfile(str(p), dtype=np.uint8)
    if buf.size == 0:
        raise ImageReadError(f"빈 파일: {p}")
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageReadError(f"이미지로 디코드할 수 없습니다: {p}")
    if img.dtype != np.uint8:
        raise ImageReadError(f"8-bit만 지원합니다 (v0.1). {p}: dtype={img.dtype}")
    if img.ndim == 2:
        return promote_to_bgr(img), True
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), False
    if img.ndim == 3 and img.shape[2] == 3:
        return img, False
    raise ImageReadError(f"지원하지 않는 채널 구성 {img.shape}: {p}")


def read_mask(path: str | Path) -> np.ndarray:
    """마스크를 ``HxW uint8 0/255``로 읽는다. 회색값은 ``>127``로 이진화 — 회색이 섞인 마스크를 그대로 믿지 않는다."""
    p = Path(path)
    if not p.is_file():
        raise ImageReadError(f"파일이 없습니다: {p}")
    buf = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageReadError(f"마스크로 디코드할 수 없습니다: {p}")
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.dtype != np.uint8:
        img = (img > 0).astype(np.uint8) * 255
    return binarize(img)


def write_image(path: str | Path, image: np.ndarray) -> None:
    """확장자로 인코더를 고르고 바이트로 쓴다. 상위 폴더는 만들어 준다."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError(f"지원하지 않는 이미지 확장자: {suffix} ({p})")
    ok, buf = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"이미지 인코딩 실패: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(buf.tobytes())


def list_images(folder: str | Path) -> list[Path]:
    """폴더의 이미지 파일을 **이름 정렬**로. 디렉터리 열람 순서는 OS마다 달라 재현성을 깨므로 항상 정렬."""
    d = Path(folder)
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def read_path_list(list_file: str | Path) -> list[Path]:
    """``.txt`` 경로 목록(한 줄 하나, 빈 줄·``#`` 무시, 상대경로는 목록 파일 기준). 줄 순서 유지."""
    lf = Path(list_file)
    base = lf.parent
    out: list[Path] = []
    for raw in lf.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        out.append(p if p.is_absolute() else (base / p))
    return out

"""YOLO 박스 → 픽셀 마스크 추정 (설계 §3.2). 순수 함수 — 파일 IO 없음.

| 방법 | 동작 |
|---|---|
| ``grabcut`` | 박스를 ``margin``만큼 넓힌 창에서 ``cv2.grabCut(rect=박스, iter=5)`` → FGD ∪ PR_FGD |
| ``otsu`` | 박스 바깥 링(폭 ``margin``)의 중앙값을 국소 배경으로, ``|gray − 배경|``을 박스 안에서 Otsu → 열림 |
| ``ellipse`` | 박스 내접 타원 |
| ``rect`` | 박스 전체 (CutPaste·NSA 방식, ``hard-paste`` 대조군) |

- **폴백 사슬**: 추정(grabcut·otsu) 결과 면적이 박스의 ``[5%, 95%]`` 밖이면 ``grabcut → otsu → ellipse`` 순으로 내려간다.
  ``ellipse``가 비면(1~2px 박스) ``rect``. 짧은 변 ``< min_box``면 바로 ``ellipse``. 실제 쓴 방법을 함께 돌려준다.
- **``cv2.grabCut``은 내부 k-means가 OpenCV 전역 RNG를 쓴다** → 호출 직전 ``cv2.setRNGSeed(seed)``. 호출자는 소스 id의
  **안정 해시**(``stable_seed`` — 파이썬 ``hash()``는 프로세스마다 달라 쓸 수 없다)를 넘긴다. 같은 입력 → 같은 마스크.
- 추정 마스크는 **박스 밖으로 나가지 않는다**(박스가 라벨러의 의도 경계). 반환 마스크는 이미지 크기 ``HxW uint8 0/255``.
"""

from __future__ import annotations

import zlib
from typing import Literal

import cv2
import numpy as np

from anograft.core.channels import promote_to_bgr

Box = tuple[int, int, int, int]  # x, y, w, h (이미지 좌표, 정수)
Method = Literal["grabcut", "otsu", "ellipse", "rect"]

METHODS: tuple[str, ...] = ("grabcut", "otsu", "ellipse", "rect")
_CHAIN: tuple[str, ...] = ("grabcut", "otsu", "ellipse", "rect")  # 폴백 순서
AREA_RATIO_MIN = 0.05
AREA_RATIO_MAX = 0.95
GRABCUT_ITERS = 5


def stable_seed(key: str) -> int:
    """문자열 → 16-bit 시드. ``zlib.crc32``라 프로세스·플랫폼과 무관하게 같다."""
    return zlib.crc32(key.encode("utf-8")) & 0xFFFF


def clip_box(box: Box, shape: tuple[int, ...]) -> Box | None:
    """박스를 이미지 안으로 자른다. 면적이 0이면 None."""
    h, w = shape[:2]
    x, y, bw, bh = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _window(box: Box, shape: tuple[int, ...], margin: int) -> Box:
    h, w = shape[:2]
    x, y, bw, bh = box
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
    return (x0, y0, x1 - x0, y1 - y0)


def mask_rect(shape: tuple[int, ...], box: Box) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    x, y, bw, bh = box
    m[y : y + bh, x : x + bw] = 255
    return m


def mask_ellipse(shape: tuple[int, ...], box: Box) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    x, y, bw, bh = box
    center = (round(x + (bw - 1) / 2.0), round(y + (bh - 1) / 2.0))
    axes = (max(1, bw // 2), max(1, bh // 2))
    cv2.ellipse(m, center, axes, 0.0, 0.0, 360.0, 255, -1)
    m &= mask_rect(shape, box)  # 반올림으로 박스를 넘지 않게
    return m


def mask_otsu(image: np.ndarray, box: Box, margin: int) -> np.ndarray:
    """링 중앙값 배경 → ``|gray − bg|`` 박스 안 Otsu → 3×3 열림. 링이 없으면(박스 = 이미지) 창 전체 중앙값."""
    gray = cv2.cvtColor(promote_to_bgr(image), cv2.COLOR_BGR2GRAY)
    wx, wy, ww, wh = _window(box, gray.shape, margin)
    win = gray[wy : wy + wh, wx : wx + ww]
    bx, by, bw, bh = box[0] - wx, box[1] - wy, box[2], box[3]
    ring = np.ones(win.shape, dtype=bool)
    ring[by : by + bh, bx : bx + bw] = False
    bg = float(np.median(win[ring])) if ring.any() else float(np.median(win))
    inner = win[by : by + bh, bx : bx + bw].astype(np.float32)
    diff = np.abs(inner - bg)
    diff8 = np.clip(diff, 0, 255).astype(np.uint8)
    if diff8.max() == diff8.min():
        return np.zeros(gray.shape, dtype=np.uint8)  # 편차 없음 → 빈 마스크 → 폴백
    _thr, binary = cv2.threshold(diff8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if min(bw, bh) >= 3:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    m = np.zeros(gray.shape, dtype=np.uint8)
    m[box[1] : box[1] + bh, box[0] : box[0] + bw] = binary
    return m


def mask_grabcut(image: np.ndarray, box: Box, margin: int, seed: int) -> np.ndarray:
    """``GC_INIT_WITH_RECT`` 5회. 창 = 박스 + margin(배경 표본). 창이 박스와 같으면(배경 없음) cv2.error → 호출자가 폴백."""
    bgr = promote_to_bgr(image)
    wx, wy, ww, wh = _window(box, bgr.shape, margin)
    win = np.ascontiguousarray(bgr[wy : wy + wh, wx : wx + ww])
    rect = (box[0] - wx, box[1] - wy, box[2], box[3])
    gc = np.zeros((wh, ww), dtype=np.uint8)
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    cv2.setRNGSeed(int(seed))  # k-means 초기화가 전역 RNG를 쓴다 — 결정성
    cv2.grabCut(win, gc, rect, bgd, fgd, GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)
    fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    m = np.zeros(bgr.shape[:2], dtype=np.uint8)
    m[wy : wy + wh, wx : wx + ww] = fg.astype(np.uint8) * 255
    return m & mask_rect(bgr.shape, box)


def _area_ok(mask: np.ndarray, box: Box) -> bool:
    ratio = float(np.count_nonzero(mask)) / float(box[2] * box[3])
    return AREA_RATIO_MIN <= ratio <= AREA_RATIO_MAX


def mask_from_box(
    image: np.ndarray,
    box: Box,
    method: str = "grabcut",
    margin: int = 6,
    *,
    min_box: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, str]:
    """박스 하나 → ``(mask HxW 0/255, method_used)``. 박스는 이미지 안으로 잘라 쓰고, 면적 0이면 ``ValueError``."""
    if method not in METHODS:
        raise ValueError(f"알 수 없는 방법 {method!r} (선택: {', '.join(METHODS)})")
    clipped = clip_box(box, image.shape)
    if clipped is None:
        raise ValueError(f"박스가 이미지 밖입니다: {box}")
    box = clipped
    shape = image.shape[:2]

    if method == "rect":
        return mask_rect(shape, box), "rect"
    start = "ellipse" if min(box[2], box[3]) < min_box else method
    for m in _CHAIN[_CHAIN.index(start) :]:
        if m == "rect":
            return mask_rect(shape, box), "rect"
        if m == "ellipse":
            mask = mask_ellipse(shape, box)
            if np.any(mask):
                return mask, "ellipse"
            continue
        try:
            mask = (
                mask_grabcut(image, box, margin, seed)
                if m == "grabcut"
                else mask_otsu(image, box, margin)
            )
        except cv2.error:
            continue
        if _area_ok(mask, box):
            return mask, m
    return mask_rect(shape, box), "rect"  # 도달 불가 — 형식상

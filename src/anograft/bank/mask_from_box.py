"""YOLO 박스 → 픽셀 마스크 추정 (설계 §3.2). 순수 함수 — 파일 IO 없음.

| 방법 | 동작 |
|---|---|
| ``grabcut`` | 박스를 ``margin``만큼 넓힌 창에서 ``cv2.grabCut(rect=박스, iter=5)`` → FGD ∪ PR_FGD |
| ``otsu`` | 박스 바깥 링(폭 ``margin``)의 중앙값을 국소 배경으로, ``|gray − 배경|``을 박스 안에서 Otsu → 열림 |
| ``ellipse`` | 박스 내접 타원 |
| ``rect`` | 박스 전체 (CutPaste·NSA 방식, ``hard-paste`` 대조군) |
| ``hybrid`` | ``grabcut`` 사슬을 돌리고 결과의 ``mask_confidence`` 가 ``LOW_CONFIDENCE`` 미만이면 ``ellipse`` 로(v0.8 옵션 — 공개 데이터 벤치에서 사슬 0.37 → 0.50 IoU, 실패 0.43 → 0.21; 기본은 그대로 ``grabcut``) |

- **폴백 사슬**: 추정(grabcut·otsu) 결과 면적이 박스의 ``[5%, 95%]`` 밖이면 ``grabcut → otsu → ellipse`` 순으로 내려간다.
  ``ellipse``가 비면(1~2px 박스) ``rect``. 짧은 변 ``< min_box``면 바로 ``ellipse``. 실제 쓴 방법을 함께 돌려준다.
- **``cv2.grabCut``은 내부 k-means가 OpenCV 전역 RNG를 쓴다** → 호출 직전 ``cv2.setRNGSeed(seed)``. 호출자는 소스 id의
  **안정 해시**(``stable_seed`` — 파이썬 ``hash()``는 프로세스마다 달라 쓸 수 없다)를 넘긴다. 같은 입력 → 같은 마스크.
- 추정 마스크는 **박스 밖으로 나가지 않는다**(박스가 라벨러의 의도 경계). 반환 마스크는 이미지 크기 ``HxW uint8 0/255``.
- (v0.6) **``mask_confidence``** — 면적 비율만 보던 폴백 사슬을 보완하는 **타당성 점수**(KNOWN-ISSUES #3: 경면 금속에서
  그럴듯한 면적의 엉뚱한 영역이 채택됨). 마스크 안 평균 vs 박스 바깥 링 평균의 분리도(링 σ 단위) · 박스 테두리 접촉 비율 ·
  성분 수 · 포화 비율을 0..1 점수와 flags 로. 채택 여부는 바꾸지 않고(재현성) 은행 메타 ``confidence``/``flags`` 에 남겨
  ``bank ls``·``bank preview``·``run`` 경고가 쓴다. 휴리스틱이다 — ``LOW_CONFIDENCE`` 미만은 "눈으로 확인" 신호.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from anograft.core.channels import promote_to_bgr
from anograft.core.seeds import stable_seed

__all__ = [
    "LOW_CONFIDENCE",
    "METHODS",
    "MaskConfidence",
    "mask_confidence",
    "mask_from_box",
    "stable_seed",
]

Box = tuple[int, int, int, int]  # x, y, w, h (이미지 좌표, 정수)
Method = Literal["grabcut", "otsu", "ellipse", "rect"]

METHODS: tuple[str, ...] = ("grabcut", "otsu", "ellipse", "rect", "hybrid")
_CHAIN: tuple[str, ...] = ("grabcut", "otsu", "ellipse", "rect")  # 폴백 순서
AREA_RATIO_MIN = 0.05
AREA_RATIO_MAX = 0.95
GRABCUT_ITERS = 5
LOW_CONFIDENCE = 0.5  # 이 미만이면 bank ls/preview/run 이 "확인 필요"로 센다
SEPARATION_FULL = 2.0  # 링 σ 의 이 배수 이상 떨어지면 대비 점수 1.0
SATURATED_GRAY = 250


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


@dataclass(frozen=True)
class MaskConfidence:
    """추정 마스크의 타당성 — ``score`` 0..1 와 사람이 읽을 ``flags``. 나머지는 근거 수치."""

    score: float
    flags: tuple[str, ...]
    contrast: float  # 마스크 안 평균 그레이 − 박스 바깥 링 평균 (부호 있음)
    separation: float  # |contrast| / (링 σ + 1)
    touch: float  # 박스 테두리 픽셀 중 마스크가 닿은 비율
    n_components: int
    saturated: float  # 마스크 안 포화(≥ SATURATED_GRAY) 비율
    area_ratio: float  # 마스크 면적 / 박스 면적


def mask_confidence(
    image: np.ndarray, mask: np.ndarray, box: Box, *, margin: int = 6
) -> MaskConfidence:
    """박스 추정 마스크가 "결함을 잡았는지"의 휴리스틱 점수.

    - **분리도**: 마스크 안 평균 그레이가 박스 **바깥** 링(폭 ``margin``)의 평균에서 링 σ 의 몇 배 떨어졌나. 밝은 금속을 밝은 금속
      위에서 잡으면 0 근처 → ``low-contrast``. 점수 = min(1, 분리도 / SEPARATION_FULL).
    - **테두리 접촉**: 박스 테두리 픽셀 중 마스크가 닿은 비율. 반 넘게 닿으면 박스를 그냥 채운 것에 가깝다 → ``box-edge``, ×(1 − touch/2).
    - **성분 수**: 4개 넘으면 텍스처를 주워 담은 모양 → ``fragmented``, ×min(1, 3/n).
    - **포화**: 마스크 안 ≥ 250 비율이 반 넘으면 하이라이트를 잡은 것 → ``saturated``, ×(1 − sat/2).
    - 면적 비율이 사슬 범위 밖이면 ``area-out`` (점수엔 이미 반영된 셈이라 감점 없음).
    빈 마스크는 점수 0 + ``empty``."""
    clipped = clip_box(box, image.shape)
    if clipped is None:
        return MaskConfidence(0.0, ("empty",), 0.0, 0.0, 0.0, 0, 0.0, 0.0)
    x, y, bw, bh = clipped
    gray = cv2.cvtColor(promote_to_bgr(image), cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = mask > 0
    inner = m[y : y + bh, x : x + bw]
    area = int(inner.sum())
    if area == 0:
        return MaskConfidence(0.0, ("empty",), 0.0, 0.0, 0.0, 0, 0.0, 0.0)
    wx, wy, ww, wh = _window(clipped, gray.shape, margin)
    win = gray[wy : wy + wh, wx : wx + ww]
    ring = np.ones(win.shape, dtype=bool)
    ring[y - wy : y - wy + bh, x - wx : x - wx + bw] = False
    if not ring.any():  # 박스가 이미지 전체 — 박스 안 마스크 밖을 배경으로
        ring = ~m[wy : wy + wh, wx : wx + ww]
    bg = win[ring]
    bg_mean = float(bg.mean()) if bg.size else float(win.mean())
    bg_std = float(bg.std()) if bg.size else float(win.std())
    box_gray = gray[y : y + bh, x : x + bw]
    in_mean = float(box_gray[inner].mean())
    contrast = in_mean - bg_mean
    separation = abs(contrast) / (bg_std + 1.0)
    s_sep = min(1.0, separation / SEPARATION_FULL)

    edge = np.zeros(inner.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    touch = float((inner & edge).sum()) / float(edge.sum())

    n, _ = cv2.connectedComponents(inner.astype(np.uint8), connectivity=8)
    n_comp = int(n - 1)
    saturated = float((box_gray[inner] >= SATURATED_GRAY).mean())
    area_ratio = area / float(bw * bh)

    flags: list[str] = []
    if s_sep < 0.5:
        flags.append("low-contrast")
    if touch > 0.5:
        flags.append("box-edge")
    if n_comp > 4:
        flags.append("fragmented")
    if saturated > 0.5:
        flags.append("saturated")
    if not (AREA_RATIO_MIN <= area_ratio <= AREA_RATIO_MAX):
        flags.append("area-out")
    score = s_sep * (1.0 - touch / 2.0) * min(1.0, 3.0 / max(1, n_comp)) * (1.0 - saturated / 2.0)
    return MaskConfidence(
        score=round(float(score), 3),
        flags=tuple(flags),
        contrast=round(contrast, 2),
        separation=round(separation, 3),
        touch=round(touch, 3),
        n_components=n_comp,
        saturated=round(saturated, 3),
        area_ratio=round(area_ratio, 3),
    )


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
    if method == "hybrid":
        mask, used = mask_from_box(image, box, "grabcut", margin, min_box=min_box, seed=seed)
        if used in ("grabcut", "otsu") and (
            mask_confidence(image, mask, box, margin=margin).score < LOW_CONFIDENCE
        ):
            ell = mask_ellipse(shape, box)
            if np.any(ell):
                return ell, "ellipse"
        return mask, used
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

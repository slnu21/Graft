"""배치 허용 영역(ROI) 추출 — 순수 배열 함수 (설계 §6 roi). 파일 IO 없음.

배경에 붙은 결함은 학습에 해롭다 — ROI 제약은 옵션이 아니라 기본값이다. v0.1은 ``otsu``·``none``·``mask_dir``,
나중에 ``grabcut``·``sam``이 같은 반환 규약(``HxW bool``)으로 붙는다.

테두리 ``margin_px``는 여기서 빼지 않는다 — ROI 방법과 무관하게 ``placement``가 공통으로 적용한다
(``margin_px``는 placement 설정이라 ROI 스테이지가 볼 수 없다).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from anograft.core.channels import binarize

Invert = Literal["auto", "yes", "no"]


@dataclass(frozen=True)
class OtsuRoi:
    roi: np.ndarray  # HxW bool
    threshold: float
    inverted: bool  # True면 어두운 쪽이 전경
    border_touch_bright: float  # Otsu 밝은 쪽의 테두리 접촉 비율
    border_touch_dark: float
    area_before_erode: int


def border_touch_ratio(mask: np.ndarray) -> float:
    """마스크가 이미지 테두리(1px 링)에 닿는 비율 — 테두리 픽셀 중 True인 것의 비율.

    물체는 보통 중앙에, 배경은 테두리에 있으므로 이 값이 낮은 쪽이 물체다.
    """
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return 0.0
    if h <= 2 or w <= 2:
        border = mask.astype(bool)
    else:
        ring = np.zeros((h, w), dtype=bool)
        ring[0, :] = ring[-1, :] = True
        ring[:, 0] = ring[:, -1] = True
        border = mask.astype(bool)[ring]
    return float(border.mean()) if border.size else 0.0


def erode_bool(mask: np.ndarray, px: int) -> np.ndarray:
    """bool 마스크를 ``px``만큼 침식. 0이면 그대로."""
    if px <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask.astype(np.uint8), k) > 0


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def roi_otsu(image: np.ndarray, invert: Invert = "auto", erode_px: int = 8) -> OtsuRoi:
    """그레이 Otsu 이진화 → 전경 선택(``auto`` = 테두리 접촉 비율이 낮은 쪽) → 침식.

    한쪽이 비어 있으면(균일 이미지 등) 비어 있지 않은 쪽을 고른다. 둘 다 비면 면적 0 ROI.
    """
    gray = to_gray(image)
    thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = binary > 0
    dark = ~bright
    touch_bright = border_touch_ratio(bright)
    touch_dark = border_touch_ratio(dark)
    if invert == "yes":
        inverted = True
    elif invert == "no":
        inverted = False
    else:
        n_bright, n_dark = int(bright.sum()), int(dark.sum())
        # 한쪽이 비면(균일 이미지) 남은 쪽, 아니면 테두리 접촉이 적은 쪽
        inverted = (
            (n_bright == 0) if (n_bright == 0 or n_dark == 0) else (touch_dark < touch_bright)
        )
    fg = dark if inverted else bright
    area_before = int(fg.sum())
    roi = erode_bool(fg, erode_px)
    return OtsuRoi(
        roi=roi,
        threshold=float(thr),
        inverted=inverted,
        border_touch_bright=touch_bright,
        border_touch_dark=touch_dark,
        area_before_erode=area_before,
    )


def roi_none(shape: tuple[int, int]) -> np.ndarray:
    """전체 허용. 텍스처 카테고리(carpet·grid·leather·tile·wood)용."""
    return np.ones(shape[:2], dtype=bool)


def roi_from_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """외부 마스크(PNG 등)를 ROI로. ``>127`` 이진화. 크기가 다르면 ``ValueError``."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] != tuple(shape[:2]):
        raise ValueError(f"ROI 마스크 크기 {mask.shape[:2]} 가 대상 {tuple(shape[:2])} 와 다릅니다")
    return binarize(mask) > 0


def distance_to_edge(allowed: np.ndarray) -> np.ndarray:
    """허용 영역 각 픽셀에서 가장 가까운 비허용 픽셀(또는 이미지 테두리)까지의 거리. float32, 비허용은 0.

    이미지 테두리도 경계로 세기 위해 1px 0 테두리를 두르고 계산한다.
    """
    h, w = allowed.shape[:2]
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = allowed.astype(np.uint8)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.ascontiguousarray(dist[1:-1, 1:-1], dtype=np.float32)

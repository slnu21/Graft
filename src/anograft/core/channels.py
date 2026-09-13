"""채널 승격/복원 + 마스크 이진화 — 순수 배열 연산 (파일 IO 없음).

코어 내부는 항상 ``HxWx3 uint8``. 흑백 입력은 세 채널 동일하게 승격하고, 출력 직전에 채널 0만 취해 복원한다
(세 채널이 동일하므로 무손실 — 테스트로 고정).
"""

from __future__ import annotations

import cv2
import numpy as np


def promote_to_bgr(image: np.ndarray) -> np.ndarray:
    """1ch → 3ch(세 채널 동일). 이미 3ch면 그대로."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    return image


def demote_from_bgr(image: np.ndarray, gray: bool) -> np.ndarray:
    """출력 직전 채널 복원. gray면 채널 0만 취한다."""
    if gray and image.ndim == 3:
        return np.ascontiguousarray(image[:, :, 0])
    return image


def binarize(mask: np.ndarray) -> np.ndarray:
    """``>127 → 255``, 나머지 0. dtype uint8 보장."""
    return np.where(mask > 127, 255, 0).astype(np.uint8)

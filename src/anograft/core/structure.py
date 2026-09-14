"""구조 필드 — 그래디언트 크기 맵 · 창 구조 텐서 방향/일관성 · 마스크 주축 (설계 §6 placement ``structure-aware``). 순수 배열.

각도 규약(전부 같은 좌표계): 이미지 좌표(x 오른쪽, y **아래**)에서 +x 축으로부터 +y 쪽으로 잰 각도(도, ``(-90, 90]``).
- ``window_orientation``이 주는 ``theta_g``는 **지배 그래디언트 방향**이고, 에지·결(grain)의 방향은 그에 수직(``theta_g + 90``).
- ``mask_principal_axis``의 ``phi``는 마스크 주축(긴 쪽) 방향.
- ``align_rotation(phi, target)``은 그 주축을 ``target``에 맞추기 위해 ``warp_affine``(= ``cv2.getRotationMatrix2D``,
  **양수 = 화면상 반시계**)에 넣을 각도. 화면 반시계 회전은 이 좌표계에서 각도를 **줄이므로** 부호가 뒤집힌다
  (``test_structure``가 가로 막대를 실제로 돌려 고정).

``coherence``·``anisotropy``는 둘 다 ``((λ1-λ2)/(λ1+λ2))²``: 0 = 등방(방향 없음), 1 = 완전히 한 방향.
"""

from __future__ import annotations

import math
from typing import Literal

import cv2
import numpy as np

Prefer = Literal["edges", "flat", "uniform"]  # 레시피 스키마와 같은 값
WEIGHT_EPS = 1e-3  # 가중 0 픽셀도 아주 낮은 확률로는 뽑히게(누적합 0 방지)


def to_gray_f32(image: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 또는 HxW → float32 그레이 [0, 255]."""
    g = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32)


def gradient_magnitude(gray: np.ndarray, smooth_px: float) -> np.ndarray:
    """Sobel 3×3 그래디언트 크기 → 가우시안 평활(``smooth_px`` σ, 0이면 생략). float32 HxW, 정규화 안 함."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    if smooth_px > 0:
        mag = cv2.GaussianBlur(mag, (0, 0), smooth_px)
    return mag


def structure_weights(
    mag: np.ndarray, allowed: np.ndarray, prefer: Prefer, strength: float
) -> np.ndarray | None:
    """허용 픽셀(row-major 순)의 위치 가중치. ``uniform``은 None(균등 정수 추첨).

    ``m`` = 크기를 허용 영역 안 99퍼센타일로 나눠 [0, 1] 클립. ``edges`` = ``m^strength``, ``flat`` = ``(1-m)^strength``,
    둘 다 ``+ WEIGHT_EPS``(전 픽셀 0 방지). ``strength`` 0 이면 사실상 균등.
    """
    if prefer == "uniform":
        return None
    m = mag[allowed].astype(np.float64)
    if m.size == 0:
        return None
    p99 = float(np.percentile(m, 99))
    m = np.clip(m / p99, 0.0, 1.0) if p99 > 0 else np.zeros_like(m)
    if prefer == "edges":
        w = m**strength
    elif prefer == "flat":
        w = (1.0 - m) ** strength
    else:
        raise ValueError(f"알 수 없는 prefer: {prefer!r}")
    return w + WEIGHT_EPS


def _tensor_orientation(jxx: float, jyy: float, jxy: float) -> tuple[float, float]:
    """2×2 대칭 텐서 → (지배 방향 각도 deg, 일관성 [0,1])."""
    trace = jxx + jyy
    if trace <= 1e-12:
        return 0.0, 0.0
    theta = 0.5 * math.degrees(math.atan2(2.0 * jxy, jxx - jyy))
    root = math.sqrt(((jxx - jyy) / 2.0) ** 2 + jxy**2)
    coherence = (2.0 * root / trace) ** 2  # (λ1-λ2)/(λ1+λ2) = 2·root/trace
    return wrap_180(theta), float(min(1.0, max(0.0, coherence)))


def window_orientation(gray: np.ndarray, cx: int, cy: int, half: int) -> tuple[float, float]:
    """``(cx, cy)`` 중심 ``(2·half+1)²`` 창의 구조 텐서 → ``(theta_g, coherence)``.

    ``theta_g`` = 지배 **그래디언트** 방향(에지·결 방향은 +90°). 창은 이미지 안으로 잘라 쓰고 Sobel 경계 오염을 피하려
    1px 여유를 둔 뒤 안쪽만 합산한다. 창이 비거나 평탄하면 ``(0, 0)``.
    """
    h, w = gray.shape[:2]
    x0, x1 = max(0, cx - half - 1), min(w, cx + half + 2)
    y0, y1 = max(0, cy - half - 1), min(h, cy + half + 2)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0, 0.0
    win = gray[y0:y1, x0:x1]
    gx = cv2.Sobel(win, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1]
    gy = cv2.Sobel(win, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1]
    jxx = float(np.sum(gx * gx))
    jyy = float(np.sum(gy * gy))
    jxy = float(np.sum(gx * gy))
    return _tensor_orientation(jxx, jyy, jxy)


def mask_principal_axis(mask: np.ndarray) -> tuple[float, float]:
    """마스크(0/255 또는 bool)의 2차 모멘트 주축 → ``(phi, anisotropy)``. 비어 있거나 점이면 ``(0, 0)``."""
    m = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    if m["m00"] <= 0:
        return 0.0, 0.0
    return _tensor_orientation(m["mu20"], m["mu02"], m["mu11"])


def wrap_180(deg: float) -> float:
    """각도를 ``(-90, 90]``로 — 축(방향 없는 선)의 각도."""
    d = (deg + 90.0) % 180.0 - 90.0
    return 90.0 if d == -90.0 else d


def align_rotation(phi_deg: float, target_deg: float) -> float:
    """마스크 주축 ``phi``를 ``target`` 축에 맞추는 ``warp_affine`` 회전각(양수 = 화면 반시계), ``(-90, 90]``."""
    return wrap_180(phi_deg - target_deg)

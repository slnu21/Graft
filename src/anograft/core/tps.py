"""Thin-plate spline 워프 (v0.8.x, 설계 §6.1 의 v0.2 열 ``tps``) — headless OpenCV 에는 ``shape`` 모듈(TPS)이 없어 numpy 로.

- 제어점 격자 ``n×n`` 를 패치 위에 놓고 각 점을 ``rng.normal(0, jitter × 짧은 변)`` 만큼 흔든다 → 매끈한 전역 변형(elastic 이
  국소 잔물결이면 tps 는 휘어짐·늘어남). 무작위 소비: ``normal`` 2·n² 번(``jitter > 0`` 일 때만 — 기본 0 = off = rng 0회).
- ``cv2.remap`` 은 **출력 → 입력** 좌표가 필요하므로 TPS 를 (흔든 점 → 원래 점) 방향으로 푼다. 커널 U(r) = r² log r².
- 순수 numpy/cv2 — ``tps_maps`` · ``tps_solve`` 는 테스트(항등·평행이동·마스크 이진 유지)가 고정한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from anograft.core.channels import binarize


def _kernel(r2: np.ndarray) -> np.ndarray:
    out = np.zeros_like(r2)
    nz = r2 > 0
    out[nz] = r2[nz] * np.log(r2[nz])
    return out


def tps_solve(src: np.ndarray, dst: np.ndarray, *, reg: float = 0.0) -> np.ndarray:
    """``src``(n×2, 매핑의 입력 점) → ``dst``(n×2) 를 지나는 TPS 계수 (n+3)×2. ``reg`` 는 대각 정규화(0 = 정확 보간)."""
    n = src.shape[0]
    d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    k = _kernel(d2) + reg * np.eye(n)
    p = np.hstack([np.ones((n, 1)), src])
    a = np.zeros((n + 3, n + 3))
    a[:n, :n] = k
    a[:n, n:] = p
    a[n:, :n] = p.T
    b = np.zeros((n + 3, 2))
    b[:n] = dst
    return np.linalg.solve(a, b)


def tps_apply(coef: np.ndarray, src: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """계수·제어점(입력 쪽)·평가점(m×2) → 매핑된 점(m×2)."""
    n = src.shape[0]
    d2 = ((pts[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    u = _kernel(d2)
    w, aff = coef[:n], coef[n:]
    return u @ w + aff[0] + pts @ aff[1:]


def control_grid(shape: tuple[int, int], n: int) -> np.ndarray:
    """패치 안쪽 ``n×n`` 격자 제어점 (x, y) — 가장자리를 살짝 안쪽으로."""
    h, w = shape
    xs = np.linspace(0.5, w - 1.5, n)
    ys = np.linspace(0.5, h - 1.5, n)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)


def tps_maps(
    shape: tuple[int, int], ctrl: np.ndarray, moved: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``ctrl``(원래) → ``moved``(흔든) 변형의 **역매핑** 맵: 출력 픽셀 → 입력 좌표. 반환 ``(map_x, map_y)`` float32."""
    h, w = shape
    coef = tps_solve(moved, ctrl)  # 출력(흔든 자리) → 입력(원래 자리)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    out = tps_apply(coef, moved, pts)
    return (
        out[:, 0].reshape(h, w).astype(np.float32),
        out[:, 1].reshape(h, w).astype(np.float32),
    )


def tps_pad(shape: tuple[int, int], jitter: float) -> int:
    """제어점이 3σ 까지 움직여도 마스크가 캔버스 밖으로 안 나가게 두를 여유(px)."""
    if jitter <= 0:
        return 0
    return int(np.ceil(3.0 * jitter * min(shape))) + 1


def tps_deform(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    points: int,
    jitter: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """이미지·마스크에 같은 TPS 변형. 반환 ``(image, mask, ctrl, moved)`` — 로그용 제어점 포함. ``jitter <= 0`` 이면 rng 0회, 원본 그대로."""
    if jitter <= 0 or points < 2:
        return image, mask, np.zeros((0, 2)), np.zeros((0, 2))
    pad = tps_pad(mask.shape[:2], jitter)
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    h, w = mask.shape[:2]
    ctrl = control_grid((h - 2 * pad, w - 2 * pad), points) + pad
    sigma = jitter * float(min(h - 2 * pad, w - 2 * pad))
    moved = ctrl + rng.normal(0.0, sigma, ctrl.shape)
    map_x, map_y = tps_maps((h, w), ctrl, moved)
    out_img = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    out_mask = cv2.remap(
        mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return out_img, binarize(out_mask), ctrl, moved

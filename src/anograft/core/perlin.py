"""펄린 노이즈 (DRAEM식 이상 마스크). 순수 numpy·cv2 — 파일·Qt 의존 0.

DRAEM(Zavrtanik 2021)은 축별 해상도 ``2^k``(k ∈ [0, 5])의 2D 펄린 노이즈를 만들고 ``> 0.5``로 임계해 이상 영역 마스크로
쓴다(회전 −90~90°). 여기서는 그 절차를 ``rng`` 하나로 결정적으로 재현한다.

- ``perlin_noise(rng, h, w, res_y, res_x)``: 격자 ``res_y × res_x``의 무작위 단위 그래디언트 → 각 픽셀에서 네 모서리
  내적 → quintic fade ``6t⁵−15t⁴+10t³`` → 이중 선형 보간. 값 범위 ≈ [−1, 1](√2 배 정규화 전 ±0.7). rng 소비:
  ``rng.random((res_y+1, res_x+1))`` 1회(각도).
- ``perlin_mask(rng, h, w, ...)``: ``k_y, k_x = rng.integers(lo, hi+1)`` → 노이즈 → ``rng.uniform(rotate)`` 회전
  (NEAREST, 캔버스 유지) → ``> threshold`` → uint8 0/255. rng 소비 순서: **integers×2 → random(그래디언트) → uniform(회전)**.
  창 크기가 해상도로 나누어떨어지지 않아도 되도록 노이즈는 ``ceil``한 크기로 만들고 자른다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def perlin_noise(rng: np.random.Generator, h: int, w: int, res_y: int, res_x: int) -> np.ndarray:
    """``h×w`` float32 펄린 노이즈, 격자 해상도 ``res_y × res_x``(≥ 1). 대략 [−1, 1]."""
    if h < 1 or w < 1:
        raise ValueError("h, w는 1 이상이어야 합니다")
    res_y, res_x = max(1, int(res_y)), max(1, int(res_x))
    # 격자 한 칸의 픽셀 크기 — 나누어떨어지지 않으면 올림해서 만들고 자른다
    cell_y, cell_x = math.ceil(h / res_y), math.ceil(w / res_x)
    hh, ww = cell_y * res_y, cell_x * res_x
    angles = rng.random((res_y + 1, res_x + 1)).astype(np.float32) * (2.0 * np.pi)
    grad = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (res_y+1, res_x+1, 2)

    ys = (np.arange(hh, dtype=np.float32) / cell_y)[:, None]  # 격자 좌표
    xs = (np.arange(ww, dtype=np.float32) / cell_x)[None, :]
    iy = np.minimum(np.floor(ys).astype(np.int64), res_y - 1)
    ix = np.minimum(np.floor(xs).astype(np.int64), res_x - 1)
    fy, fx = ys - iy, xs - ix  # 칸 안 좌표 [0, 1)
    fy = np.broadcast_to(fy, (hh, ww))
    fx = np.broadcast_to(fx, (hh, ww))
    iy = np.broadcast_to(iy, (hh, ww))
    ix = np.broadcast_to(ix, (hh, ww))

    def dot(dy: int, dx: int) -> np.ndarray:
        g = grad[iy + dy, ix + dx]  # (hh, ww, 2)
        return g[..., 0] * (fx - dx) + g[..., 1] * (fy - dy)

    n00, n10, n01, n11 = dot(0, 0), dot(1, 0), dot(0, 1), dot(1, 1)
    uy, ux = _fade(fy), _fade(fx)
    nx0 = n00 * (1 - ux) + n01 * ux
    nx1 = n10 * (1 - ux) + n11 * ux
    out = (nx0 * (1 - uy) + nx1 * uy) * math.sqrt(2.0)
    return out[:h, :w].astype(np.float32)


def rotate_keep(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    """중심 회전, 캔버스 크기 유지, NEAREST(마스크는 회색값 금지). 캔버스 밖은 0."""
    if abs(angle_deg) < 1e-9:
        return mask
    h, w = mask.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0 - 0.5, h / 2.0 - 0.5), angle_deg, 1.0)
    return cv2.warpAffine(
        mask, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def perlin_mask(
    rng: np.random.Generator,
    h: int,
    w: int,
    *,
    scale_range: tuple[int, int] = (0, 5),
    threshold: float = 0.5,
    rotate: tuple[float, float] = (-90.0, 90.0),
) -> tuple[np.ndarray, dict[str, float | int]]:
    """DRAEM식 이상 마스크 ``h×w`` uint8 0/255 + 로그(``res_y·res_x·rotate·area_px``).

    rng 소비 순서: ``integers``(k_y) → ``integers``(k_x) → ``random``(그래디언트) → ``uniform``(회전)."""
    lo, hi = int(scale_range[0]), int(scale_range[1])
    if lo < 0 or hi < lo:
        raise ValueError("scale_range는 0 <= lo <= hi 여야 합니다")
    k_y = int(rng.integers(lo, hi + 1))
    k_x = int(rng.integers(lo, hi + 1))
    res_y, res_x = 2**k_y, 2**k_x
    noise = perlin_noise(rng, h, w, res_y, res_x)
    angle = float(rng.uniform(rotate[0], rotate[1]))
    mask = (noise > threshold).astype(np.uint8) * 255
    mask = rotate_keep(mask, angle)
    return mask, {
        "res_y": res_y,
        "res_x": res_x,
        "rotate": angle,
        "area_px": int(np.count_nonzero(mask)),
    }

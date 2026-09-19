"""외형 지표 — 그레이 이미지 + 마스크(둘 다 같은 창) → 스칼라. 순수 numpy/opencv.

검수 탭(합성 vs 실제 분포) · 은행 요약(``bank ls`` 의 lightR) · ``runner.lighting_warning`` 이 같은 정의를 쓴다.
전부 빈 마스크(또는 링 없음)면 None — 호출 쪽이 건너뛴다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

RING_PX = 8  # 대비 링 폭
CONTRAST_FADE_RATIO = 0.5  # 클래스의 합성 대비 중앙값이 실제의 이 비율 미만이면 '옅어짐' — MT blowhole −17/−48 = 0.35 가 mAP −0.30 의 신호였다
CONTRAST_MIN_N = (
    5  # 중앙값 비교는 양쪽 다 이보다 적으면 판단하지 않는다(퀵스타트·작은 은행의 헛경고 방지)
)
CONTRAST_REAL_MIN = 8.0  # 실제 대비 중앙값 절댓값이 이보다 작으면(저대비 결함 — screw 스크래치) 비율은 잡음이라 판단하지 않는다
CONTRAST_GAP_MIN = 12.0  # 중앙값 차이(gray)가 이보다 작으면 비율이 낮아도 판단하지 않는다 — MT crack 은 −4.5 vs −13.7(비율 0.33, 차 9)인데 블렌딩 3종 mAP 무차별(0.50~0.54); blowhole 은 차 31~41 에서 −0.30 → +0.47. 카메라 노이즈·JPEG 수준의 차이는 프리셋을 가를 근거가 아니다
LIGHT_RING_PX = 2  # 조명 방향은 얇은 링 — 하이라이트 림이 1~2 px 라 8 px 링에선 질감에 묻힌다(샘플 pit: R 0.78→0.99)
LIGHT_REAL_MIN = 0.5  # 실제 소스의 R 이 이 이상이면 '조명 방향이 있는 클래스'
LIGHT_SYNTH_MAX = 0.3  # 그 클래스의 합성 R 이 이 미만이면 회전이 방향을 뒤집고 있다
LIGHT_MIN_N = 3  # R 은 n=1 이면 항상 1 — 이보다 적으면 판단하지 않는다
LIGHT_RAYLEIGH_Z = 2.9  # 유의성: n·R² ≥ z (Rayleigh, p ≈ e^−z ≈ 0.05). 무작위 각도의 R 은 ≈ 1/√n 이라 n 이 작으면 0.5 를 우연히 넘는다


def mask_contrast(gray: np.ndarray, mask: np.ndarray, *, ring_px: int = RING_PX) -> float | None:
    """마스크 안 평균 그레이 − 둘레 링(폭 ``ring_px``, 마스크 제외) 평균. 어느 쪽이든 비면 None. 라벨 탭 통계와 같은 정의."""
    m = mask > 0
    if not m.any():
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    ring = cv2.dilate(m.astype(np.uint8), k) > 0
    ring &= ~m
    if not ring.any():
        return None
    g = gray.astype(np.float32)
    return round(float(g[m].mean() - g[ring].mean()), 2)


def mask_texture(gray: np.ndarray, mask: np.ndarray) -> float | None:
    """마스크 안 Sobel 그래디언트 크기 평균(결함 내부의 질감·에지 에너지). 빈 마스크면 None."""
    m = mask > 0
    if not m.any():
        return None
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return round(float(np.sqrt(gx * gx + gy * gy)[m].mean()), 2)


def mask_sharpness(gray: np.ndarray, mask: np.ndarray) -> float | None:
    """마스크 안 라플라시안 분산(선명도 — 열화 블러가 결함을 뭉갰는지). 빈 마스크면 None."""
    m = mask > 0
    if not m.any():
        return None
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    return round(float(lap[m].var()), 2)


def mask_lighting(
    gray: np.ndarray, mask: np.ndarray, *, ring_px: int = LIGHT_RING_PX
) -> float | None:
    """둘레 링에서 밝은 쪽이 어느 방향인지 — 각도(°, 이미지 좌표: 0 = 오른쪽, 90 = 아래). KI #5 근거.

    링 픽셀의 (밝기 − 링 평균) 을 무게로 중심→픽셀 단위벡터를 합해 방향을 얻는다. 조명이 한쪽에서 오는
    움푹/볼록 결함은 하이라이트 림이 한 방향에 몰리므로 실제 소스는 각도가 한 곳에 모이고, 회전 ±180 으로
    합성하면 고르게 퍼진다(→ ``circular_concentration``). 빈 마스크·링이면 None."""
    m = mask > 0
    if not m.any():
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    ring = cv2.dilate(m.astype(np.uint8), k) > 0
    ring &= ~m
    if not ring.any():
        return None
    ys, xs = np.nonzero(m)
    cy, cx = float(ys.mean()), float(xs.mean())
    ry, rx = np.nonzero(ring)
    g = gray.astype(np.float32)
    w = g[ry, rx] - float(g[ring].mean())
    dy, dx = ry.astype(np.float32) - cy, rx.astype(np.float32) - cx
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    vx, vy = float((w * dx / norm).sum()), float((w * dy / norm).sum())
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return None
    return round(math.degrees(math.atan2(vy, vx)), 1)


def circular_concentration(angles_deg: Sequence[float]) -> float | None:
    """각도 집합의 평균 합벡터 길이 R (1 = 전부 같은 방향, 0 = 고르게 퍼짐). 비면 None."""
    if not angles_deg:
        return None
    th = np.radians(np.asarray(list(angles_deg), dtype=np.float64))
    return round(float(np.hypot(np.cos(th).mean(), np.sin(th).mean())), 3)


APPEARANCE_FN = {
    "contrast": mask_contrast,
    "texture": mask_texture,
    "sharpness": mask_sharpness,
    "lighting": mask_lighting,
}


def is_directional(r: float | None, n: int) -> bool:
    """'조명 방향이 있는 클래스' 판정 — R ≥ LIGHT_REAL_MIN **이고** n·R² ≥ LIGHT_RAYLEIGH_Z (n ≥ LIGHT_MIN_N).
    n=5 무작위면 R ≈ 0.45(임계 근처)라 R 만으로는 소표본에서 오판한다: n=3 은 R ≥ 0.98, n=5 는 0.76, n=12 는 0.5 가 필요."""
    if r is None or n < LIGHT_MIN_N:
        return False
    return r >= LIGHT_REAL_MIN and n * r * r >= LIGHT_RAYLEIGH_Z


def safe_flip(light_dir_deg: float | None) -> str:
    """조명 방향을 아는 클래스에 허용되는 가장 넉넉한 flip — 위/아래 조명이면 ``horizontal``, 옆 조명이면 ``vertical``,
    모르면 ``none``. (``flip_breaks_lighting`` 의 역.)"""
    if light_dir_deg is None:
        return "none"
    vertical_light = abs(math.sin(math.radians(light_dir_deg))) >= math.cos(math.radians(45.0))
    return "horizontal" if vertical_light else "vertical"


def flip_breaks_lighting(flip: str, light_dir_deg: float | None) -> bool:
    """flip 모드가 그 클래스의 하이라이트 방향을 뒤집는가. ``both`` 는 항상; ``vertical`` 은 조명이 위/아래에서 올 때
    (방향이 세로축에서 45° 안), ``horizontal`` 은 옆에서 올 때. 방향을 모르면(None) horizontal/vertical 도 위험으로 본다."""
    if flip == "none":
        return False
    if flip == "both" or light_dir_deg is None:
        return True
    vertical_light = abs(math.sin(math.radians(light_dir_deg))) >= math.cos(math.radians(45.0))
    return vertical_light if flip == "vertical" else not vertical_light


def circular_mean(angles_deg: Sequence[float]) -> float | None:
    """각도 집합의 평균 방향(°). 비면 None."""
    if not angles_deg:
        return None
    th = np.radians(np.asarray(list(angles_deg), dtype=np.float64))
    return round(math.degrees(math.atan2(np.sin(th).mean(), np.cos(th).mean())), 1)


def angle_diff(a: float, b: float) -> float:
    """두 각도의 차이 절댓값(0..180)."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def gray_of(image: np.ndarray) -> np.ndarray:
    """BGR/그레이 어느 쪽이든 2D 그레이로."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def lighting_of_sources(pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[float | None, int]:
    """(image, mask) 쌍들의 조명 일관성 — ``(R, n)``. n < LIGHT_MIN_N 이면 R 은 None(판단 보류)."""
    r, n, _ = lighting_stats(pairs)
    return r, n


def lighting_stats(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[float | None, int, float | None]:
    """``(R, n, 평균 방향°)`` — 평균 방향은 n ≥ 1 이면 준다(판정은 ``is_directional(R, n)`` 으로)."""
    angles = [v for img, m in pairs if (v := mask_lighting(gray_of(img), m)) is not None]
    mean = circular_mean(angles)
    if len(angles) < LIGHT_MIN_N:
        return None, len(angles), mean
    return circular_concentration(angles), len(angles), mean


def flipped_instances(
    gray: np.ndarray,
    instances: Sequence[tuple[str, np.ndarray]],
    real_dir: Mapping[str, float],
    *,
    max_deg: float = 90.0,
) -> list[int]:
    """합성 이미지의 (클래스, 마스크) 인스턴스 중 조명 방향이 그 클래스의 실제 평균 방향에서 ``max_deg`` 넘게 벗어난 것의
    인덱스. ``real_dir`` 에 없는 클래스(방향 없는 클래스)는 건너뛴다. 검수 탭 '조명 뒤집힘 의심'과 같은 규칙."""
    out: list[int] = []
    for i, (cls, mask) in enumerate(instances):
        if cls not in real_dir:
            continue
        v = mask_lighting(gray, mask)
        if v is not None and angle_diff(v, real_dir[cls]) > max_deg:
            out.append(i)
    return out


@dataclass(frozen=True)
class ContrastHint:
    """한 클래스의 합성 대비가 실제(은행·실측)보다 옅다는 판정 — 중앙값끼리 비교. ``ratio`` < 0 은 극성 반전(배경보다 밝은 구멍)."""

    cls: str
    synth_median: float
    real_median: float
    n_synth: int
    n_real: int

    @property
    def ratio(self) -> float:
        return self.synth_median / self.real_median

    @property
    def flipped(self) -> bool:
        return self.ratio < 0.0


def contrast_hints(
    synth: Mapping[str, Sequence[float]],
    real: Mapping[str, Sequence[float]],
    *,
    fade_ratio: float = CONTRAST_FADE_RATIO,
    min_n: int = CONTRAST_MIN_N,
    real_min: float = CONTRAST_REAL_MIN,
    gap_min: float = CONTRAST_GAP_MIN,
) -> list[ContrastHint]:
    """클래스별 대비(``mask_contrast``: 마스크 안 − 링, 부호 있음) 합성 vs 실제 → 옅어진 클래스 목록(클래스 이름순).

    판정: 양쪽 n ≥ ``min_n`` · |실제 중앙값| ≥ ``real_min`` · |합성 − 실제 중앙값| ≥ ``gap_min`` · 합성 중앙값 / 실제 중앙값 <
    ``fade_ratio``(부호가 반대면 음수라 항상 포함). poisson·stats 계열 조화는 정의상 결함 톤을 대상에 맞춰 옅게 만들고(MT blowhole 합성 −17 vs 실제 −48 →
    mAP −0.30), hard-paste 는 노출이 다른 대상에서 절반이 극성 반전(중앙값 0) — 어느 쪽이든 그 클래스만
    ``relative-paste`` 로 갈라(``recipe init --classes``) ``dataset merge`` 하라는 것이 BENCHMARKS §2 의 답(+0.15).
    무작위성 없음 · 임계는 상수(스키마 아님)."""
    out: list[ContrastHint] = []
    for cls in sorted(set(synth) & set(real)):
        a, b = [float(v) for v in synth[cls]], [float(v) for v in real[cls]]
        if len(a) < min_n or len(b) < min_n:
            continue
        ma, mb = float(np.median(a)), float(np.median(b))
        if abs(mb) < real_min or abs(ma - mb) < gap_min:
            continue
        if ma / mb < fade_ratio:
            out.append(ContrastHint(cls, round(ma, 1), round(mb, 1), len(a), len(b)))
    return out

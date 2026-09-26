"""처음 보는 형상 — 보관함 어느 조각과도 닮지 않은 결함 찾기 (설계 §2b.5, 작업 단위 T15).

루프의 가장 약한 고리는 **새 결함 유형**이다. 보관함은 기존 클래스로 조직돼 있고, 지도 모델은 배운 외형에만
반응하고 비지도 모델은 "이상하다"까지만 말한다. 그래서 "이건 지금까지 본 것들과 다르다"는 신호를 따로 둔다.

재료는 이미 있었다 — `core/appearance.py`(대비·질감·선명도)와 마스크 모양. 여기서 하는 일은 그것들을 **한
벡터로 묶고, 보관함 조각들과의 최근접 거리를 로버스트 단위로 재는 것**뿐이다.

**이 점수는 분류기가 아니다.** 기존 클래스 안의 변형과 진짜 새 유형을 완벽히 가르지 못한다. 쓰임은 하나 —
**사람에게 먼저 보여 줄 우선순위**와 **"이름을 아직 주지 말자"는 판단**(미분류, `__unsorted__`)이다.

규율 둘:

- **비교할 조각이 적으면 판정하지 않는다**(`MIN_REFS` = 10). 조각 몇 개짜리 보관함에서는 무엇이든 멀어
  보이고 편차 추정도 흔들린다 — 그 상태로 임계를 넘기면 들어오는 것이 전부 미분류가 되어 신호가 아니라
  소음이 된다. 초기 라운드에는 조용한 것이 맞다.
- **폭은 보관함이 정한다**(차원별 MAD). 대비·질감·선명도의 스케일이 데이터셋마다 다르므로 절대 거리로는
  임계를 정할 수 없다. 그래서 "이 보관함의 흔한 편차 몇 배인가"로 잰다.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import astuple, dataclass

import cv2
import numpy as np

from anograft.core.appearance import gray_of, mask_contrast, mask_sharpness, mask_texture

#: 비교할 보관함 조각이 이보다 적으면 점수를 내지 않는다(항상 0.0).
#: 열 개로 잡은 이유: 폭을 MAD 로 재는데 표본이 그보다 적으면 편차 추정이 너무 흔들려 **멀쩡한 조각도
#: 미분류로 밀려난다**. 신호가 없는 것보다 틀린 신호가 나쁘다 — 초기 라운드의 작은 보관함은 그냥 조용하다.
MIN_REFS = 10
#: 로버스트 단위 이만큼 떨어지면 점수 0.5 — 즉 임계 0.6 은 "흔한 편차의 3배"다.
DIST_UNIT = 2.0
#: 기본 임계. 보수적으로 둔다 — 실무 값은 확인 게이트(설계 §8).
DEFAULT_THRESHOLD = 0.6
#: MAD → 표준편차 환산(정규분포 가정).
_MAD_TO_SIGMA = 1.4826
#: 장변/단변 상한 — 한 픽셀 폭 선은 길이만큼 늘어나므로 차원 하나가 거리를 독차지하지 않게 자른다.
MAX_ELONGATION = 50.0
#: 차원별 폭 바닥값 — 보관함이 한 유형뿐이면 MAD 가 0 이 되어 어떤 차이도 무한히 멀어진다.
_SCALE_FLOOR: tuple[float, ...] = (2.0, 2.0, 5.0, 0.25, 0.15, 0.05)


@dataclass(frozen=True)
class Feature:
    """결함 조각 하나의 외형 벡터. 여섯 차원 전부 **한 조각만 보고** 잴 수 있는 것들이다.

    조명 방향(`mask_lighting`)은 일부러 넣지 않았다 — 원형 값이라 유클리드 거리에 그대로 못 넣고,
    같은 유형이 방향만 달라도 "새 유형"으로 잡힌다(그건 `appearance.is_directional` 의 일이다).
    """

    contrast: float  # 마스크 안 − 둘레 링 평균 그레이(부호 있음 — 밝은 결함/어두운 결함이 다르다)
    texture: float  # 마스크 안 그래디언트 크기 평균
    sharpness: float  # 마스크 안 라플라시안 분산
    log_area: float  # log10(면적 px + 1) — 면적은 자릿수로 다르다
    elongation: float  # 최소외접사각의 장변/단변(가는 스크래치 ↔ 둥근 기공)
    fill: float  # 면적 / 최소외접사각 면적(선은 낮고 덩어리는 높다)

    def vector(self) -> tuple[float, ...]:
        return astuple(self)

    @classmethod
    def of(cls, image: np.ndarray, mask: np.ndarray) -> Feature | None:
        """크롭+마스크 → 외형 벡터. 빈 마스크·링 없음이면 ``None``(비교 대상에서 빠진다)."""
        m = np.asarray(mask)
        binary = (m > 0).astype(np.uint8)
        area = int(np.count_nonzero(binary))
        if area <= 0:
            return None
        gray = gray_of(np.asarray(image))
        contrast = mask_contrast(gray, binary)
        texture = mask_texture(gray, binary)
        sharpness = mask_sharpness(gray, binary)
        if contrast is None or texture is None or sharpness is None:
            return None
        elongation, fill = shape_of(binary, area)
        return cls(
            contrast=float(contrast),
            texture=float(texture),
            sharpness=float(sharpness),
            log_area=math.log10(area + 1.0),
            elongation=elongation,
            fill=fill,
        )


def shape_of(binary: np.ndarray, area: int) -> tuple[float, float]:
    """``(장변/단변, 채움 비율)`` — 최소외접사각 기준.

    bbox 가 아니라 최소외접사각인 이유: **대각선 스크래치는 bbox 가 정사각형**이라 가늘다는 사실이 사라진다.
    """
    points = cv2.findNonZero(binary)
    if points is None:
        return 1.0, 1.0
    (_cx, _cy), (w, h) = cv2.minAreaRect(points)[:2]
    long_side, short_side = max(w, h), min(w, h)
    short = max(short_side, 1.0)  # 한 픽셀 폭 선도 폭이 1 이다(0 으로 나누지 않는다)
    elongation = min(float(long_side / short), MAX_ELONGATION)
    rect_area = float(long_side * short)
    fill = float(area / rect_area) if rect_area > 0 else 1.0
    return elongation, min(fill, 1.0)


def features_of(pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> list[Feature]:
    """``(image, mask)`` 쌍들 → 벡터 목록. 잴 수 없는 것은 조용히 빠진다(fail-soft)."""
    out: list[Feature] = []
    for image, mask in pairs:
        feat = Feature.of(image, mask)
        if feat is not None:
            out.append(feat)
    return out


def scales(refs: Sequence[Feature]) -> tuple[float, ...]:
    """차원별 로버스트 폭(MAD×1.4826, 바닥값 적용) — **이 보관함의 흔한 편차**.

    평균·표준편차가 아니라 중앙값·MAD 인 이유: 보관함에는 이미 이상한 조각이 섞여 있다(추정 마스크가
    엉뚱한 곳을 잡은 것 등). 표준편차로 재면 그 몇 개가 폭을 부풀려 **아무것도 새롭지 않게** 된다.
    """
    if not refs:
        return _SCALE_FLOOR
    matrix = np.asarray([r.vector() for r in refs], dtype=np.float64)
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0) * _MAD_TO_SIGMA
    return tuple(float(max(m, floor)) for m, floor in zip(mad, _SCALE_FLOOR, strict=True))


def distance(a: Feature, b: Feature, scale: Sequence[float]) -> float:
    """로버스트 단위 유클리드 거리 — 차원마다 그 보관함의 편차로 나눈다."""
    va, vb = a.vector(), b.vector()
    total = 0.0
    for x, y, s in zip(va, vb, scale, strict=True):
        d = (x - y) / (s if s > 0 else 1.0)
        total += d * d
    return math.sqrt(total)


def nearest(feat: Feature, refs: Sequence[Feature], scale: Sequence[float]) -> float:
    """가장 가까운 조각까지의 거리(로버스트 단위). 참조가 없으면 ``inf``."""
    if not refs:
        return math.inf
    return min(distance(feat, r, scale) for r in refs)


def novelty_score(
    feat: Feature | None,
    refs: Sequence[Feature],
    *,
    scale: Sequence[float] | None = None,
    min_refs: int = MIN_REFS,
) -> float:
    """0~1. 클수록 "보관함 어느 조각과도 닮지 않았다".

    **참조가 `min_refs` 보다 적으면 0.0** — 작은 보관함에서는 무엇이든 멀어 보이므로 판정을 하지 않는다
    (신호가 아니라 소음이 된다). 잴 수 없는 조각(빈 마스크)도 0.0 이다.
    """
    if feat is None or len(refs) < max(1, min_refs):
        return 0.0
    d = nearest(feat, refs, scale if scale is not None else scales(refs))
    if not math.isfinite(d):
        return 1.0
    return round(d / (d + DIST_UNIT), 4)


def is_novel(score: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """임계 0 이하는 **끔**(어떤 점수도 새 유형으로 보지 않는다)."""
    return threshold > 0 and score >= threshold


__all__ = [
    "DEFAULT_THRESHOLD",
    "DIST_UNIT",
    "MAX_ELONGATION",
    "MIN_REFS",
    "Feature",
    "distance",
    "features_of",
    "is_novel",
    "nearest",
    "novelty_score",
    "scales",
    "shape_of",
]

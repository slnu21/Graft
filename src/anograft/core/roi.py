"""배치 허용 영역(ROI) 추출 — 순수 배열 함수 (설계 §6 roi). 파일 IO 없음.

배경에 붙은 결함은 학습에 해롭다 — ROI 제약은 옵션이 아니라 기본값이다. v0.1은 ``otsu``·``none``·``mask_dir``,
v0.4 ``grabcut``, v0.6 ``annulus``(원형 부품 링 면 — 물체 vs 배경이 아니라 **물체 안에서** 검사 면을 고른다,
KNOWN-ISSUES #2). 나중에 ``sam``이 같은 반환 규약(``HxW bool``)으로 붙는다.

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


@dataclass(frozen=True)
class GrabCutRoi:
    roi: np.ndarray  # HxW bool
    init_used: str  # "rect" | "otsu" | "otsu→rect"(Otsu 가 비어 rect 로 폴백)
    inverted: bool | None  # otsu 초기화일 때 Otsu 극성
    work_scale: float  # 작업 해상도 / 원본 (1.0 = 원본)
    area_before_erode: int
    fallback: str | None = None  # GrabCut 자체가 실패해 Otsu 결과로 대체했을 때 사유


def _rect_inside(h: int, w: int, margin: float) -> tuple[int, int, int, int]:
    mx, my = max(1, round(w * margin)), max(1, round(h * margin))
    mx, my = min(mx, (w - 2) // 2), min(my, (h - 2) // 2)
    return (mx, my, w - 2 * mx, h - 2 * my)


def roi_grabcut(
    image: np.ndarray,
    *,
    init: Literal["rect", "otsu"] = "rect",
    invert: Invert = "auto",
    rect_margin: float = 0.03,
    iters: int = 5,
    work_px: int = 1024,
    erode_px: int = 8,
    seed: int = 0,
) -> GrabCutRoi:
    """GrabCut 전경 → ROI. ``rect`` = 테두리 ``rect_margin`` 바깥은 확정 배경, 안쪽 사각형에서 전경 탐색(``GC_INIT_WITH_RECT``).
    ``otsu`` = Otsu 전경/배경을 ``GC_PR_FGD``/``GC_PR_BGD``로, 테두리 링은 ``GC_BGD``(``GC_INIT_WITH_MASK``) — Otsu 한쪽이 비면
    ``rect``로 폴백. ``work_px``(긴 변)로 줄여 풀고 NEAREST 로 되돌린 뒤 ``erode_px`` 침식.

    ``cv2.grabCut``은 전역 RNG 를 쓰므로 호출 직전 ``cv2.setRNGSeed(seed)`` — 같은 입력·시드 → 같은 ROI.
    ``cv2.error``(이미지가 너무 작거나 라벨이 한쪽뿐)는 예외로 새지 않고 Otsu(auto) 결과로 대체 + ``fallback`` 기록.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    scale = 1.0
    if work_px > 0 and max(h, w) > work_px:
        scale = work_px / float(max(h, w))
    if scale < 1.0:
        sw, sh = max(4, round(w * scale)), max(4, round(h * scale))
        small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        small = np.ascontiguousarray(image)
    sh, sw = small.shape[:2]

    rect = _rect_inside(sh, sw, rect_margin)
    gc = np.zeros((sh, sw), dtype=np.uint8)
    inverted: bool | None = None
    init_used = init
    mode = cv2.GC_INIT_WITH_RECT
    if init == "otsu":
        o = roi_otsu(small, invert, 0)
        fg = o.roi
        n_fg = int(fg.sum())
        if 0 < n_fg < fg.size:
            inverted = o.inverted
            gc[:] = cv2.GC_PR_BGD
            gc[fg] = cv2.GC_PR_FGD
            rx, ry, rw, rh = rect
            ring = np.ones((sh, sw), dtype=bool)
            ring[ry : ry + rh, rx : rx + rw] = False
            gc[ring] = cv2.GC_BGD
            mode = cv2.GC_INIT_WITH_MASK
        else:
            init_used = "otsu→rect"
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    fallback: str | None = None
    try:
        cv2.setRNGSeed(int(seed))  # k-means 초기화가 전역 RNG 를 쓴다 — 결정성
        cv2.grabCut(small, gc, rect, bgd, fgd, int(iters), mode)
        fg_small = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    except cv2.error as e:  # 예: 4px 이미지, 한쪽 라벨뿐 — 이 대상은 Otsu 로 대신 (fail-soft)
        fallback = f"cv2.error: {str(e).strip().splitlines()[-1][:120]}"
        fg_small = roi_otsu(small, "auto", 0).roi
    if scale < 1.0:
        fg = cv2.resize(fg_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        fg = fg_small
    area_before = int(fg.sum())
    return GrabCutRoi(
        roi=erode_bool(fg, erode_px),
        init_used=init_used,
        inverted=inverted,
        work_scale=scale,
        area_before_erode=area_before,
        fallback=fallback,
    )


# ---------------------------------------------------------------------------
# annulus (v0.6) — 파라메트릭 ROI: 원형 부품의 링 면 (KNOWN-ISSUES #2 #4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiskFit:
    """Otsu 전경의 가장 큰 연결 성분에 씌운 최소외접원 — 원형 부품의 중심·바깥 반경 추정."""

    center: tuple[float, float]  # (cx, cy) px
    radius: float  # px
    inverted: bool  # Otsu 극성 (roi_otsu 와 같은 규칙)
    area_px: int  # 그 성분의 면적


def detect_disk(image: np.ndarray, invert: Invert = "auto") -> DiskFit | None:
    """부품 중심·반경 자동 검출. 전경이 없으면 None. 잡음 성분에 흔들리지 않게 **가장 큰 성분 하나**만 쓴다 —
    중앙에 홈(어두운 리세스)이 있어 전경이 링 모양이어도 최소외접원은 바깥 원을 준다."""
    fg = roi_otsu(image, invert, 0)
    mask = fg.roi.astype(np.uint8)
    if not mask.any():
        return None
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    comp = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    return DiskFit((float(cx), float(cy)), float(r), fg.inverted, int(comp.sum()))


def annulus_mask(
    shape: tuple[int, int], center: tuple[float, float], r_inner: float, r_outer: float
) -> np.ndarray:
    """``r_inner ≤ 거리 < r_outer`` 인 픽셀 (HxW bool). ``r_inner=0`` 이면 원판."""
    h, w = int(shape[0]), int(shape[1])
    cx, cy = center
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return (d2 >= r_inner * r_inner) & (d2 < r_outer * r_outer)


@dataclass(frozen=True)
class AnnulusRoi:
    roi: np.ndarray  # HxW bool
    center: tuple[float, float]
    radius: float  # 비율의 기준 반경(px) — units: px 면 참고값
    r_inner_px: float
    r_outer_px: float
    center_source: str  # "auto" | "fixed" | "auto→image-center"
    radius_source: str  # "auto" | "fixed" | "auto→half-min-side" | "unused"
    area_before_erode: int
    fallback: str | None = None  # 자동 검출 실패 사유 (있으면 경고)


def roi_annulus(
    image: np.ndarray,
    *,
    center: tuple[float, float] | None = None,
    radius: float | None = None,
    r_inner: float,
    r_outer: float,
    units: Literal["ratio", "px"] = "ratio",
    invert: Invert = "auto",
    erode_px: int = 0,
) -> AnnulusRoi:
    """링(annulus) ROI. ``center``/``radius`` 가 None 이면 ``detect_disk`` 로 자동 — 촬영마다 부품이 수십 px 움직여도
    링이 따라간다(KNOWN-ISSUES #4). ``units: ratio`` 면 ``r_inner``·``r_outer`` 는 기준 반경의 배율, ``px`` 면 그대로.
    자동 검출이 실패하면 이미지 중심·짧은 변의 절반으로 대체하고 ``fallback`` 에 사유."""
    h, w = image.shape[:2]
    need_fit = center is None or (units == "ratio" and radius is None)
    fit = detect_disk(image, invert) if need_fit else None
    fallback: str | None = None
    if center is not None:
        c = (float(center[0]), float(center[1]))
        center_source = "fixed"
    elif fit is not None:
        c, center_source = fit.center, "auto"
    else:
        c, center_source = ((w - 1) / 2.0, (h - 1) / 2.0), "auto→image-center"
        fallback = "Otsu 전경 없음 — 이미지 중심으로 대체"
    if units == "px":
        base, radius_source = (
            (float(radius) if radius is not None else 0.0),
            ("fixed" if radius is not None else "unused"),
        )
        r_in, r_out = float(r_inner), float(r_outer)
    else:
        if radius is not None:
            base, radius_source = float(radius), "fixed"
        elif fit is not None:
            base, radius_source = fit.radius, "auto"
        else:
            base, radius_source = min(h, w) / 2.0, "auto→half-min-side"
            fallback = fallback or "Otsu 전경 없음 — 반경을 짧은 변의 절반으로 대체"
        r_in, r_out = float(r_inner) * base, float(r_outer) * base
    mask = annulus_mask((h, w), c, r_in, r_out)
    area_before = int(mask.sum())
    return AnnulusRoi(
        roi=erode_bool(mask, erode_px),
        center=c,
        radius=base,
        r_inner_px=r_in,
        r_outer_px=r_out,
        center_source=center_source,
        radius_source=radius_source,
        area_before_erode=area_before,
        fallback=fallback,
    )


def roi_from_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """외부 마스크(PNG 등)를 ROI로. ``>127`` 이진화.

    크기가 다르면 ``INTER_NEAREST`` 로 대상 크기에 맞춘다 — ROI 는 대략적 허용 영역이라 픽셀 정밀도가 필요 없고,
    GUI 미리보기(긴 변 1024 축소)에서 원본 크기 마스크가 그대로 쓰여야 한다(KNOWN-ISSUES #1).
    가로세로 비가 다르면(마스크가 다른 이미지의 것일 가능성) ``ValueError``.
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    h, w = int(shape[0]), int(shape[1])
    mh, mw = mask.shape[:2]
    if (mh, mw) != (h, w):
        # 축소 반올림 오차 |mh·w − mw·h| ≤ ½(mh+mw) 는 허용, 그 이상은 다른 비율
        if mh <= 0 or mw <= 0 or abs(mh * w - mw * h) > max(h, w, mh, mw):
            raise ValueError(
                f"ROI 마스크 크기 {(mh, mw)} 가 대상 {(h, w)} 와 비율이 다릅니다 — 다른 이미지의 마스크인지 확인"
            )
        mask = cv2.resize(binarize(mask), (w, h), interpolation=cv2.INTER_NEAREST)
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

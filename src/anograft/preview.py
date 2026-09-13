"""미리보기 렌더 (설계 §9 ``preview``) — ``원본 | 합성 | GT 오버레이`` 3패널 한 장, 긴 변 ``long_side``.

순수 배열 연산(numpy/cv2). ``run``과 **같은 결과**(같은 인덱스·같은 시드)를 축소해 보여 준다 — 축소본에서 다시 합성하지 않는다.
GT 오버레이 = 합성 위에 GT 반투명 붉은 채움 + 윤곽 + 인스턴스 bbox·클래스 라벨. 패널 라벨은 cv2 텍스트라 영문.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from anograft.core.channels import promote_to_bgr
from anograft.core.types import GraftResult, Instance

GT_FILL = (60, 60, 230)  # BGR — 붉은 채움
GT_EDGE = (40, 230, 40)  # 초록 윤곽·bbox
GAP_PX = 6
LABEL_BG = (0, 0, 0)


def overlay_gt(
    image: np.ndarray, gt_mask: np.ndarray, instances: Sequence[Instance], *, alpha: float = 0.35
) -> np.ndarray:
    out = promote_to_bgr(image).copy()
    on = gt_mask > 0
    if on.any():
        tint = np.empty_like(out)
        tint[:] = GT_FILL
        out[on] = cv2.addWeighted(out, 1.0 - alpha, tint, alpha, 0.0)[on]
        contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, GT_EDGE, 1)
    for inst in instances:
        x, y, w, h = inst.bbox
        cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), GT_EDGE, 1)
        _label(out, f"{inst.cls}#{inst.class_id}", (x, max(0, y - 4)))
    return out


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.45) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    y = max(th + 2, y)
    cv2.rectangle(img, (x, y - th - 2), (x + tw + 2, y + base), LABEL_BG, -1)
    cv2.putText(
        img, text, (x + 1, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA
    )


def fit_long_side(image: np.ndarray, long_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= long_side:
        return image
    s = long_side / float(max(h, w))
    return cv2.resize(
        image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA
    )


def render_preview(
    target_image: np.ndarray, result: GraftResult, *, long_side: int = 1024
) -> np.ndarray:
    """``원본 | 합성 | GT`` — 세 패널을 같은 축척으로 축소해 가로로 붙인다. skipped면 합성·GT 자리에 원본과 안내."""
    original = promote_to_bgr(target_image)
    if result.status == "ok":
        synthetic = promote_to_bgr(result.image)
        gt = overlay_gt(result.image, result.gt_mask, result.instances)
    else:
        synthetic = original.copy()
        gt = original.copy()
    panels = [fit_long_side(p, long_side) for p in (original, synthetic, gt)]
    names = ["original", "synthetic", "GT overlay"]
    if result.status != "ok":
        names[1] = f"skipped: {(result.reason or '')[:40]}"
    for p, n in zip(panels, names, strict=True):
        _label(p, n, (4, 16), 0.5)
    h = max(p.shape[0] for p in panels)
    w = sum(p.shape[1] for p in panels) + GAP_PX * (len(panels) - 1)
    canvas = np.full((h, w, 3), 24, dtype=np.uint8)
    x = 0
    for p in panels:
        canvas[: p.shape[0], x : x + p.shape[1]] = p
        x += p.shape[1] + GAP_PX
    return canvas


def render_grid(tiles: Sequence[np.ndarray], *, cols: int = 4, gap: int = 4) -> np.ndarray:
    """같은 크기가 아니어도 되는 타일들을 격자로 — ``bank preview``(#758)·비교 그리드 공용."""
    if not tiles:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    tiles = [promote_to_bgr(t) for t in tiles]
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * th + (rows - 1) * gap, cols * tw + (cols - 1) * gap, 3), 24, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = r * (th + gap), c * (tw + gap)
        canvas[y : y + t.shape[0], x : x + t.shape[1]] = t
    return canvas

"""미리보기 렌더 (설계 §9 ``preview``) — ``원본 | 합성 | GT 오버레이`` 3패널 한 장, 긴 변 ``long_side``.

순수 배열 연산(numpy/cv2). ``run``과 **같은 결과**(같은 인덱스·같은 시드)를 축소해 보여 준다 — 축소본에서 다시 합성하지 않는다.
GT 오버레이 = 합성 위에 GT 반투명 붉은 채움 + 윤곽 + 인스턴스 bbox·클래스 라벨. 패널 라벨은 cv2 텍스트라 영문.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from anograft.bank.mask_from_box import LOW_CONFIDENCE
from anograft.core.channels import promote_to_bgr
from anograft.core.types import GraftResult, Instance

GT_FILL = (60, 60, 230)  # BGR — 붉은 채움
GT_EDGE = (40, 230, 40)  # 초록 윤곽·bbox
EST_COLOR = (28, 132, 200)  # BGR amber — 추정 마스크 출처(bank preview)
LOW_COLOR = (60, 60, 230)  # BGR red — 저신뢰 추정 마스크 테두리(bank preview)
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


def _label(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.45,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    y = max(th + 2, y)
    cv2.rectangle(img, (x, y - th - 2), (x + tw + 2, y + base), LABEL_BG, -1)
    cv2.putText(img, text, (x + 1, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def fit_long_side(image: np.ndarray, long_side: int, *, upscale: bool = False) -> np.ndarray:
    """긴 변을 ``long_side``로 축소(INTER_AREA). ``upscale``이면 작은 이미지도 키운다(NEAREST — 픽셀을 그대로 보이게)."""
    h, w = image.shape[:2]
    if max(h, w) == long_side or (max(h, w) < long_side and not upscale):
        return image
    s = long_side / float(max(h, w))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
    return cv2.resize(image, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp)


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


def _fit_tile(image: np.ndarray, tile: int) -> tuple[np.ndarray, tuple[int, int, float]]:
    """긴 변 ``tile``로 (작으면 NEAREST 확대 — 결함 크롭은 대개 작다), 정사각 캔버스 가운데 배치."""
    img = promote_to_bgr(image)
    h, w = img.shape[:2]
    s = tile / float(max(h, w))
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
    img = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((tile, tile, 3), 24, dtype=np.uint8)
    y, x = (tile - nh) // 2, (tile - nw) // 2
    canvas[y : y + nh, x : x + nw] = img
    return canvas, (x, y, s)


def tile_id_label(source_id: str, max_chars: int) -> str:
    """타일 id 라벨 — 클래스 접두(``cls/``)는 격자에서 이미 보이므로 떼고, 길면 **앞을 잘라** 번호(꼬리)가 남게(``~``)."""
    name = source_id.rsplit("/", 1)[-1]
    if len(name) <= max_chars:
        return name
    return "~" + name[-(max_chars - 1) :]  # cv2 Hershey 폰트는 ASCII 만


def source_tile(
    image: np.ndarray,
    mask: np.ndarray,
    source_id: str,
    mask_origin: str,
    *,
    tile: int = 128,
    confidence: float | None = None,
    lighting_deg: float | None = None,
) -> np.ndarray:
    """``bank preview`` 타일 — 크롭 + 마스크 윤곽(초록) + 아래 두 줄(id · 마스크 출처). 추정 마스크(``yolo-box:*``)는
    출처를 amber 로 찍어 한눈에 셀 수 있게 한다(폴백 ellipse = 박스 내접 타원 = 과라벨). ``confidence`` 가 있으면 출처 뒤에
    점수를 찍고, ``LOW_CONFIDENCE`` 미만이면 **빨간 테두리**(KNOWN-ISSUES #3 — 면적은 정상인데 엉뚱한 곳을 잡은 마스크).
    ``lighting_deg`` 가 있으면 오른쪽 위에 **밝은 쪽을 가리키는 화살표**(KI #5 — 클래스 안에서 화살표가 한 방향이면 조명 의존)."""
    canvas, (x0, y0, s) = _fit_tile(image, tile)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        pts = np.round(c.reshape(-1, 2).astype(np.float64) * s + [x0, y0]).astype(np.int32)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], True, GT_EDGE, 1)
    estimated = mask_origin.startswith("yolo-box:")
    origin = mask_origin.split(":", 1)[1] if estimated else mask_origin.replace("yolo-", "")
    max_chars = max(6, int(tile / 7.2))  # 0.36 스케일 글자 ≈ 7 px — 타일 밖으로 넘치지 않게
    _label(canvas, tile_id_label(source_id, max_chars), (2, tile - 14), 0.36)
    low = confidence is not None and confidence < LOW_CONFIDENCE
    text = origin[:max_chars]
    if confidence is not None:
        text = f"{origin[: max(3, max_chars - 4)]} {confidence:.2f}"
    _label(
        canvas,
        text,
        (2, tile - 3),
        0.36,
        color=LOW_COLOR if low else (EST_COLOR if estimated else GT_EDGE),
    )
    if low:
        cv2.rectangle(canvas, (0, 0), (tile - 1, tile - 1), LOW_COLOR, 2)
    if lighting_deg is not None:
        draw_lighting_arrow(canvas, lighting_deg, tile)
    return canvas


LIGHT_ARROW = (40, 200, 255)  # 화살표(노랑) — 밝은 쪽


def draw_lighting_arrow(canvas: np.ndarray, deg: float, tile: int) -> None:
    """타일 오른쪽 위 구석의 작은 화살표: 중심에서 ``deg``(이미지 좌표, 0 = →, 90 = ↓) 방향으로."""
    r = max(6, tile // 12)
    cx, cy = tile - r - 4, r + 4
    dx, dy = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    cv2.circle(canvas, (cx, cy), r + 2, (0, 0, 0), -1)
    cv2.arrowedLine(
        canvas,
        (round(cx - dx * r * 0.6), round(cy - dy * r * 0.6)),
        (round(cx + dx * r), round(cy + dy * r)),
        LIGHT_ARROW,
        1,
        cv2.LINE_AA,
        tipLength=0.45,
    )


def render_compare(
    entries: Sequence[tuple[str, np.ndarray | None, str | None]], *, long_side: int = 512
) -> np.ndarray:
    """``preview --compare-methods`` — ``(method, 합성 이미지 | None, 사유)``를 격자로. None이면 검은 타일에 사유."""
    tiles: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for _m, img, _r in entries:
        if img is not None:
            shape = promote_to_bgr(img).shape[:2]
            break
    for method, img, reason in entries:
        if img is None:
            h, w = shape or (long_side, long_side)
            t = np.full((h, w, 3), 24, dtype=np.uint8)
            t = fit_long_side(t, long_side, upscale=True)
            _label(t, f"{method}: {reason or 'n/a'}"[:60], (4, 16), 0.5)
        else:
            t = fit_long_side(promote_to_bgr(img), long_side, upscale=True)
            _label(t, method, (4, 16), 0.5)
        tiles.append(t)
    cols = max(1, int(np.ceil(np.sqrt(len(tiles))))) if len(tiles) > 2 else len(tiles)
    return render_grid(tiles, cols=cols, gap=GAP_PX)


def union_bbox(
    results: Sequence[GraftResult | None],
    shape: tuple[int, int],
    *,
    pad_ratio: float = 0.6,
    min_side: int = 96,
) -> tuple[int, int, int, int] | None:
    """모든 결과의 인스턴스 bbox 합집합 + 여유(긴 변의 ``pad_ratio``, 최소 ``min_side``) — 비교 격자의 공통 크롭 창.
    인스턴스가 하나도 없으면 None(전체를 쓴다)."""
    boxes = [i.bbox for r in results if r is not None and r.status == "ok" for i in r.instances]
    if not boxes:
        return None
    x0 = min(x for x, _y, _w, _h in boxes)
    y0 = min(y for _x, y, _w, _h in boxes)
    x1 = max(x + w for x, _y, w, _h in boxes)
    y1 = max(y + h for _x, y, _w, h in boxes)
    side = max(x1 - x0, y1 - y0)
    pad = max(int(side * pad_ratio), (min_side - side) // 2, 0)
    hh, ww = shape
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(ww, x1 + pad), min(hh, y1 + pad)
    return (cx0, cy0, cx1 - cx0, cy1 - cy0)


def crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return image[y : y + h, x : x + w]

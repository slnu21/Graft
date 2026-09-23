"""``CompareCanvas`` — 원본(A) ↔ 합성(B) 와이프 비교 + GT/ROI 오버레이 + 휠 줌·드래그 팬. 목업 ``.stagewrap``.

- 와이프 선(amber) 왼쪽은 원본, 오른쪽은 합성(+오버레이). 선 근처를 드래그하면 와이프, 그 밖은 팬. ←/→ 키로 4%씩.
- 오버레이는 합성 위에만(마스크는 합성의 것이므로). GT = ``MASK_COLOR`` 반투명 채움 + 윤곽 + 인스턴스 bbox·라벨, ROI = teal 반투명.
- 이미지 좌표 ↔ 위젯 좌표는 ``image_rect()`` 하나로 — 픽셀 정확도가 필요한 라벨 탭(v0.5)이 같은 위젯을 물려받는다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from anograft.core.types import Instance
from anograft.gui.qt_image import to_qpixmap
from anograft.gui.theme import COLORS
from anograft.studio.jobs import contours_of, overlay_bgra

WIPE_GRAB_PX = 9
ZOOM_MIN, ZOOM_MAX = 0.2, 12.0


class CompareCanvas(QWidget):
    wipe_changed = Signal(float)
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Stage")
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._orig: QPixmap | None = None
        self._synth: QPixmap | None = None
        self._gt: QPixmap | None = None
        self._roi: QPixmap | None = None
        self._contours: list[np.ndarray] = []
        self._instances: list[Instance] = []
        self.wipe = 0.5
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.show_gt = True
        self.show_roi = False
        self.show_labels = True
        self._drag: str | None = None  # "wipe" | "pan"
        self._last = QPointF()
        self.message = "왼쪽에서 바탕 이미지를 고르면 미리보기가 여기에 나타납니다"

    # ------------------------------------------------------------------ 데이터

    def clear(self, message: str | None = None) -> None:
        self._orig = self._synth = self._gt = self._roi = None
        self._contours, self._instances = [], []
        if message is not None:
            self.message = message
        self.update()

    def set_images(
        self,
        original: np.ndarray,
        synthetic: np.ndarray | None,
        gt_mask: np.ndarray | None = None,
        instances: Sequence[Instance] = (),
        roi: np.ndarray | None = None,
    ) -> None:
        self._orig = to_qpixmap(original)
        self._synth = to_qpixmap(synthetic) if synthetic is not None else None
        if gt_mask is not None and gt_mask.any():
            c = QColor(COLORS["mask"])
            self._gt = to_qpixmap(overlay_bgra(gt_mask, (c.blue(), c.green(), c.red()), 96))
            self._contours = contours_of(gt_mask)
        else:
            self._gt, self._contours = None, []
        if roi is not None and roi.any():
            c = QColor(COLORS["teal"])
            self._roi = to_qpixmap(
                overlay_bgra((roi > 0).astype(np.uint8) * 255, (c.blue(), c.green(), c.red()), 44)
            )
        else:
            self._roi = None
        self._instances = list(instances)
        self.update()

    def set_original_only(self, original: np.ndarray) -> None:
        self.set_images(original, None)

    def set_overlays(
        self, *, gt: bool | None = None, roi: bool | None = None, labels: bool | None = None
    ) -> None:
        if gt is not None:
            self.show_gt = gt
        if roi is not None:
            self.show_roi = roi
        if labels is not None:
            self.show_labels = labels
        self.update()

    def reset_view(self) -> None:
        self.zoom, self.pan = 1.0, QPointF(0, 0)
        self.zoom_changed.emit(self.zoom)
        self.update()

    # ------------------------------------------------------------------ 좌표

    def image_size(self) -> tuple[int, int]:
        if self._orig is None:
            return (0, 0)
        return (self._orig.width(), self._orig.height())

    def fit_scale(self) -> float:
        w, h = self.image_size()
        if w == 0 or h == 0:
            return 1.0
        return min((self.width() - 24) / w, (self.height() - 24) / h)

    def image_rect(self) -> QRectF:
        """이미지가 위젯에 그려지는 사각형(줌·팬 반영)."""
        w, h = self.image_size()
        s = self.fit_scale() * self.zoom
        dw, dh = w * s, h * s
        cx, cy = self.width() / 2 + self.pan.x(), self.height() / 2 + self.pan.y()
        return QRectF(cx - dw / 2, cy - dh / 2, dw, dh)

    def wipe_x(self) -> float:
        r = self.image_rect()
        return r.left() + r.width() * self.wipe

    def to_image(self, p: QPointF) -> tuple[float, float]:
        r = self.image_rect()
        w, h = self.image_size()
        if r.width() == 0 or r.height() == 0:
            return (0.0, 0.0)
        return ((p.x() - r.left()) / r.width() * w, (p.y() - r.top()) / r.height() * h)

    # ------------------------------------------------------------------ 그리기

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt 규약)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(COLORS["stage"]))
        if self._orig is None:
            p.setPen(QColor(COLORS["tx3"]))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 2.0)
        r = self.image_rect()
        p.drawPixmap(r, self._orig, QRectF(self._orig.rect()))
        if self._synth is not None:
            wx = self.wipe_x()
            right = QRectF(wx, r.top(), r.right() - wx, r.height())
            p.save()
            p.setClipRect(right)
            p.drawPixmap(r, self._synth, QRectF(self._synth.rect()))
            if self.show_roi and self._roi is not None:
                p.drawPixmap(r, self._roi, QRectF(self._roi.rect()))
            if self.show_gt and self._gt is not None:
                p.drawPixmap(r, self._gt, QRectF(self._gt.rect()))
                self._draw_contours(p, r)
            if self.show_labels:
                self._draw_labels(p, r)
            p.restore()
            self._draw_wipe(p, r, wx)
        p.end()

    def _draw_contours(self, p: QPainter, r: QRectF) -> None:
        w, h = self.image_size()
        sx, sy = r.width() / w, r.height() / h
        p.setPen(QPen(QColor(COLORS["mask"]), 1.2))
        for c in self._contours:
            poly = QPolygonF([QPointF(r.left() + x * sx, r.top() + y * sy) for x, y in c])
            p.drawPolygon(poly)

    def _draw_labels(self, p: QPainter, r: QRectF) -> None:
        w, h = self.image_size()
        sx, sy = r.width() / w, r.height() / h
        font = QFont(self.font())
        font.setPointSizeF(8.5)
        p.setFont(font)
        for inst in self._instances:
            x, y, bw, bh = inst.bbox
            box = QRectF(r.left() + x * sx, r.top() + y * sy, bw * sx, bh * sy)
            p.setPen(QPen(QColor(COLORS["ok"]), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(box)
            text = f"{inst.cls}#{inst.class_id} · {inst.area_px}px"
            tw = p.fontMetrics().horizontalAdvance(text) + 8
            lab = QRectF(box.left(), max(r.top(), box.top() - 16), tw, 15)
            p.fillRect(lab, QColor(0, 0, 0, 170))
            p.setPen(QColor("#FFFFFF"))
            p.drawText(lab, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_wipe(self, p: QPainter, r: QRectF, wx: float) -> None:
        amber = QColor(COLORS["amber"])
        p.setPen(QPen(amber, 2))
        p.drawLine(QPointF(wx, r.top()), QPointF(wx, r.bottom()))
        p.setBrush(amber)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(wx, r.center().y()), 9, 9)
        # 하단 라벨
        font = QFont(self.font())
        font.setPointSizeF(8.5)
        p.setFont(font)
        for text, x in (("원본 original", r.left() + 8), ("합성 synthetic", r.right() - 8 - 84)):
            lab = QRectF(x, r.bottom() - 22, 84, 16)
            p.fillRect(lab, QColor(0, 0, 0, 140))
            p.setPen(QColor("#FFFFFF"))
            p.drawText(lab, Qt.AlignmentFlag.AlignCenter, text)

    # ------------------------------------------------------------------ 입력

    def _near_wipe(self, pos: QPointF) -> bool:
        return self._synth is not None and abs(pos.x() - self.wipe_x()) <= WIPE_GRAB_PX

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if self._orig is None:
            return
        self._last = e.position()
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = "wipe" if self._near_wipe(e.position()) else "pan"
        elif e.button() == Qt.MouseButton.MiddleButton:
            self._drag = "pan"
        self.setFocus()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        pos = e.position()
        if self._drag == "wipe":
            self._set_wipe_from_x(pos.x())
        elif self._drag == "pan":
            self.pan += pos - self._last
            self._last = pos
            self.update()
        else:
            self.setCursor(
                Qt.CursorShape.SplitHCursor if self._near_wipe(pos) else Qt.CursorShape.ArrowCursor
            )

    def mouseReleaseEvent(self, _e: QMouseEvent) -> None:  # noqa: N802
        self._drag = None

    def mouseDoubleClickEvent(self, _e: QMouseEvent) -> None:  # noqa: N802
        self.reset_view()

    def wheelEvent(self, e: QWheelEvent) -> None:  # noqa: N802
        if self._orig is None:
            return
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        new = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * factor))
        if new == self.zoom:
            return
        # 커서 아래 픽셀이 제자리에 있도록 팬 보정
        pos = e.position()
        center = QPointF(self.width() / 2, self.height() / 2)
        rel = pos - center - self.pan
        self.pan -= rel * (new / self.zoom - 1.0)
        self.zoom = new
        self.zoom_changed.emit(self.zoom)
        self.update()

    def keyPressEvent(self, e: QKeyEvent) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_Left:
            self.set_wipe(self.wipe - 0.04)
        elif e.key() == Qt.Key.Key_Right:
            self.set_wipe(self.wipe + 0.04)
        elif e.key() == Qt.Key.Key_0:
            self.reset_view()
        else:
            super().keyPressEvent(e)

    def _set_wipe_from_x(self, x: float) -> None:
        r = self.image_rect()
        if r.width() <= 0:
            return
        self.set_wipe((x - r.left()) / r.width())

    def set_wipe(self, value: float) -> None:
        value = max(0.02, min(0.98, float(value)))
        if value != self.wipe:
            self.wipe = value
            self.wipe_changed.emit(value)
            self.update()

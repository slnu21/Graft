"""``LabelCanvas`` — 이미지 + 마스크 오버레이 + 도구 상호작용(브러시·지우개·폴리곤·자동 선택 박스·팬). ``CompareCanvas``의
좌표계(``image_rect``·``to_image``)·줌/팬을 물려받고, 마우스 이벤트를 ``LabelSession`` 편집 호출로 바꾼다.

- 오버레이는 **numpy 버퍼를 공유하는 QImage**(``Format_ARGB32``, 메모리 BGRA) — 스트로크가 닿은 bbox 만 다시 칠하고 그대로
  그린다. 4K 마스크를 스트로크마다 통째로 변환하지 않는다(``refresh_overlay(rect)``). 전체 갱신은 undo·폴리곤·자동 선택 뒤.
- 브러시/지우개: 누르면 ``push_undo`` 1회, 움직일 때마다 마지막 점 → 현재 점 선분을 그린다(``session.stroke``).
  폴리곤: 클릭 = 점 추가, 더블클릭/Enter = 채움, Esc = 취소, 우클릭 = 마지막 점 삭제. 자동 선택: 드래그 박스 → ``auto_select``.
  팬: 가운데 버튼 또는 팬 도구/Space 로 왼쪽 드래그. 휠 = 줌(상속). ``[`` ``]`` = 브러시 크기.
- 편집 뒤 ``edited`` 시그널(탭이 통계·되돌리기 버튼을 갱신). 자동 선택은 ``auto_done(method_used)``.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from anograft.gui.qt_image import to_qpixmap
from anograft.gui.studio.canvas import CompareCanvas
from anograft.gui.theme import COLORS
from anograft.labeling import LabelError, LabelSession

TOOLS: tuple[str, ...] = ("brush", "eraser", "polygon", "auto", "pan")
BRUSH_MIN, BRUSH_MAX = 1, 200


class LabelCanvas(CompareCanvas):
    edited = Signal()
    auto_done = Signal(str)
    error = Signal(str)
    brush_changed = Signal(int)
    tool_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session: LabelSession | None = None
        self.tool = "brush"
        self.brush_px = 12
        self.auto_method = "grabcut"
        self.mask_visible = True
        self.opacity = 110  # 0..255
        self._ov: np.ndarray | None = None  # HxWx4 BGRA — QImage 와 버퍼 공유
        self._qov: QImage | None = None
        self._stroke_last: tuple[float, float] | None = None
        self._poly: list[tuple[float, float]] = []
        self._box_start: tuple[float, float] | None = None
        self._box_cur: tuple[float, float] | None = None
        self._space = False
        self._cursor_img: tuple[float, float] | None = None
        self.message = "결함 이미지를 열면 여기에 나타납니다 — 왼쪽 목록 또는 폴더 열기"

    # ------------------------------------------------------------------ 세션·이미지

    def set_session(self, session: LabelSession | None) -> None:
        self.session = session
        self._poly.clear()
        self._box_start = self._box_cur = None
        if session is None or not session.loaded:
            self._ov = self._qov = None
            self.clear()
            return
        assert session.image is not None
        self._orig = to_qpixmap(session.image)
        self._synth = self._gt = self._roi = None
        h, w = session.shape
        self._ov = np.zeros((h, w, 4), dtype=np.uint8)
        self._qov = QImage(self._ov.data, w, h, w * 4, QImage.Format.Format_ARGB32)
        self.refresh_overlay()
        self.reset_view()

    def refresh_overlay(self, rect: tuple[int, int, int, int] | None = None) -> None:
        """마스크 → 오버레이 버퍼. ``rect``(x, y, w, h)만 다시 칠하면 4K 에서도 스트로크가 가볍다."""
        if self.session is None or self._ov is None or self.session.mask is None:
            return
        c = QColor(COLORS["mask"])
        mask = self.session.mask
        if rect is None:
            x, y, w, h = 0, 0, mask.shape[1], mask.shape[0]
        else:
            x, y, w, h = rect
        sub = self._ov[y : y + h, x : x + w]
        on = mask[y : y + h, x : x + w] > 0
        sub[...] = 0
        sub[on, 0], sub[on, 1], sub[on, 2] = c.blue(), c.green(), c.red()
        sub[on, 3] = self.opacity
        self.update()

    def set_opacity(self, value: int) -> None:
        self.opacity = max(0, min(255, int(value)))
        self.refresh_overlay()

    def set_mask_visible(self, on: bool) -> None:
        self.mask_visible = bool(on)
        self.update()

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            raise ValueError(tool)
        if tool != self.tool:
            self.cancel_polygon()
            self.tool = tool
            self.tool_changed.emit(tool)
            self.update()

    def set_brush(self, px: int) -> None:
        px = max(BRUSH_MIN, min(BRUSH_MAX, int(px)))
        if px != self.brush_px:
            self.brush_px = px
            self.brush_changed.emit(px)
            self.update()

    # ------------------------------------------------------------------ 편집 도우미

    def _rect_of(self, pts: list[tuple[float, float]], pad: int) -> tuple[int, int, int, int]:
        assert self.session is not None
        h, w = self.session.shape
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0 = max(0, int(min(xs)) - pad - 1)
        y0 = max(0, int(min(ys)) - pad - 1)
        x1 = min(w, int(max(xs)) + pad + 2)
        y1 = min(h, int(max(ys)) + pad + 2)
        return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))

    def _paint(self, a: tuple[float, float], b: tuple[float, float]) -> None:
        assert self.session is not None
        r = self.brush_px / 2.0
        self.session.stroke([a, b], r, erase=self.tool == "eraser")
        self.refresh_overlay(self._rect_of([a, b], int(r)))

    def commit_polygon(self) -> None:
        if self.session is None or len(self._poly) < 3:
            self.cancel_polygon()
            return
        try:
            self.session.push_undo()
            self.session.fill_polygon(self._poly)
        except LabelError as e:
            self.error.emit(str(e))
        self._poly.clear()
        self.refresh_overlay()
        self.edited.emit()

    def cancel_polygon(self) -> None:
        if self._poly:
            self._poly.clear()
            self.update()

    def _commit_box(self) -> None:
        if self.session is None or self._box_start is None or self._box_cur is None:
            self._box_start = self._box_cur = None
            return
        (x0, y0), (x1, y1) = self._box_start, self._box_cur
        box = (int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)), int(abs(y1 - y0)))
        self._box_start = self._box_cur = None
        if box[2] < 2 or box[3] < 2:
            self.update()
            return
        try:
            self.session.push_undo()
            used = self.session.auto_select(box, self.auto_method)
        except LabelError as e:
            self.session.undo()
            self.error.emit(str(e))
            self.update()
            return
        self.refresh_overlay()
        self.auto_done.emit(used)
        self.edited.emit()

    def undo(self) -> None:
        if self.session is not None and self.session.undo():
            self.refresh_overlay()
            self.edited.emit()

    def redo(self) -> None:
        if self.session is not None and self.session.redo():
            self.refresh_overlay()
            self.edited.emit()

    def apply(self, fn) -> None:
        """되돌리기 한 단계로 묶인 편집(팽창·침식·지우기·마스크 로드)."""
        if self.session is None or not self.session.loaded:
            return
        try:
            self.session.push_undo()
            fn(self.session)
        except LabelError as e:
            self.session.undo()
            self.error.emit(str(e))
            return
        self.refresh_overlay()
        self.edited.emit()

    # ------------------------------------------------------------------ 그리기

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(COLORS["stage"]))
        if self._orig is None:
            p.setPen(QColor(COLORS["tx3"]))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            p.end()
            return
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 2.0)
        r = self.image_rect()
        p.drawPixmap(r, self._orig, QRectF(self._orig.rect()))
        if self.mask_visible and self._qov is not None:
            p.drawImage(r, self._qov, QRectF(self._qov.rect()))
        w, h = self.image_size()
        sx, sy = r.width() / w, r.height() / h
        if self._poly:
            pts = [QPointF(r.left() + x * sx, r.top() + y * sy) for x, y in self._poly]
            p.setPen(QPen(QColor(COLORS["amber"]), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF(pts))
            if self._cursor_img is not None:
                cx, cy = self._cursor_img
                p.setPen(QPen(QColor(COLORS["amber"]), 1, Qt.PenStyle.DashLine))
                p.drawLine(pts[-1], QPointF(r.left() + cx * sx, r.top() + cy * sy))
            p.setBrush(QColor(COLORS["amber"]))
            p.setPen(Qt.PenStyle.NoPen)
            for q in pts:
                p.drawEllipse(q, 3, 3)
        if self._box_start is not None and self._box_cur is not None:
            (x0, y0), (x1, y1) = self._box_start, self._box_cur
            box = QRectF(
                r.left() + min(x0, x1) * sx,
                r.top() + min(y0, y1) * sy,
                abs(x1 - x0) * sx,
                abs(y1 - y0) * sy,
            )
            p.setPen(QPen(QColor(COLORS["teal"]), 1.5, Qt.PenStyle.DashLine))
            p.setBrush(QColor(0, 161, 136, 40))
            p.drawRect(box)
        if self.tool in ("brush", "eraser") and self._cursor_img is not None:
            cx, cy = self._cursor_img
            rad = self.brush_px / 2.0 * sx
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor("#FFFFFF" if self.tool == "brush" else COLORS["amber"]), 1))
            p.drawEllipse(QPointF(r.left() + cx * sx, r.top() + cy * sy), rad, rad)
        p.end()

    # ------------------------------------------------------------------ 입력

    def _pan_mode(self, e: QMouseEvent) -> bool:
        return (
            e.button() == Qt.MouseButton.MiddleButton
            or self.tool == "pan"
            or self._space
            or self.session is None
            or not self.session.loaded
        )

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if self._orig is None:
            return
        self.setFocus()
        self._last = e.position()
        if self._pan_mode(e):
            self._drag = "pan"
            return
        if e.button() == Qt.MouseButton.RightButton:
            if self.tool == "polygon" and self._poly:
                self._poly.pop()
                self.update()
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        pt = self.to_image(e.position())
        assert self.session is not None
        if self.tool in ("brush", "eraser"):
            self.session.push_undo()
            self._stroke_last = pt
            self._paint(pt, pt)
            self._drag = "stroke"
        elif self.tool == "polygon":
            self._poly.append(pt)
            self.update()
        elif self.tool == "auto":
            self._box_start = self._box_cur = pt
            self._drag = "box"
            self.update()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        pos = e.position()
        self._cursor_img = self.to_image(pos) if self._orig is not None else None
        if self._drag == "pan":
            self.pan += pos - self._last
            self._last = pos
        elif self._drag == "stroke" and self._stroke_last is not None:
            pt = self.to_image(pos)
            self._paint(self._stroke_last, pt)
            self._stroke_last = pt
            return  # refresh_overlay 가 update 했다
        elif self._drag == "box":
            self._box_cur = self.to_image(pos)
        self.update()

    def mouseReleaseEvent(self, _e: QMouseEvent) -> None:  # noqa: N802
        drag, self._drag = self._drag, None
        if drag == "stroke":
            self._stroke_last = None
            self.edited.emit()
        elif drag == "box":
            self._commit_box()

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if self.tool == "polygon" and self._orig is not None:
            self.commit_polygon()
        elif self.tool == "pan":
            self.reset_view()

    def leaveEvent(self, _e) -> None:  # noqa: N802
        self._cursor_img = None
        self.update()

    def keyPressEvent(self, e: QKeyEvent) -> None:  # noqa: N802
        k = e.key()
        if k == Qt.Key.Key_Space:
            self._space = True
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.commit_polygon()
        elif k == Qt.Key.Key_Escape:
            self.cancel_polygon()
            self._box_start = self._box_cur = None
            self.update()
        elif k == Qt.Key.Key_BracketLeft:
            self.set_brush(round(self.brush_px / 1.25))
        elif k == Qt.Key.Key_BracketRight:
            self.set_brush(round(self.brush_px * 1.25) or 2)
        elif k == Qt.Key.Key_0:
            self.reset_view()
        elif k == Qt.Key.Key_Z and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.redo()
            else:
                self.undo()
        elif k == Qt.Key.Key_Y and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.redo()
        elif k == Qt.Key.Key_B:
            self.set_tool("brush")
        elif k == Qt.Key.Key_E:
            self.set_tool("eraser")
        elif k == Qt.Key.Key_P:
            self.set_tool("polygon")
        elif k == Qt.Key.Key_A:
            self.set_tool("auto")
        elif k == Qt.Key.Key_H:
            self.set_tool("pan")
        else:
            QWidget.keyPressEvent(self, e)

    def keyReleaseEvent(self, e: QKeyEvent) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_Space:
            self._space = False
        else:
            super().keyReleaseEvent(e)

"""기하 카드의 ``geometry.per_class`` 편집 표(0.7.5+) — 클래스별 회전 범위·flip 오버라이드.

카드 폼(`params.py`)은 dict 필드를 못 그리므로 전용 위젯. 행 = 은행 클래스(준비된 뒤) ∪ 레시피에 이미 있는 키.
열: 클래스 · 적용 · rotate lo/hi · flip(기본/켬/끔). scale 오버라이드는 YAML 로만(표를 좁게) — 있으면 그대로 보존한다.
값이 바뀌면 ``changed(dict)`` — 탭이 ``session.set_stage_field("geometry", "per_class", dict)`` 로 재검증한다.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QSignalBlocker, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from anograft.core import recipe as R

DEBOUNCE_MS = 350
FLIP_CHOICES = ("기본", "켬", "끔")  # None / True / False


class _Row(QWidget):
    changed = Signal()

    def __init__(self, cls: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cls = cls
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        self.use = QCheckBox(cls)
        self.use.setToolTip(
            "이 클래스만 다른 기하(회전 범위·flip)로 — 스크래치는 그대로, 찍힘만 좁힐 때"
        )
        self.lo = QDoubleSpinBox()
        self.hi = QDoubleSpinBox()
        for sp, v in ((self.lo, -15.0), (self.hi, 15.0)):
            sp.setRange(-360.0, 360.0)
            sp.setDecimals(1)
            sp.setSingleStep(5.0)
            sp.setValue(v)
            sp.setSuffix("°")
            sp.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            sp.setMinimumWidth(64)
        self.flip = QComboBox()
        self.flip.addItems(list(FLIP_CHOICES))
        self.flip.setCurrentIndex(2)  # 조명 의존 클래스의 기본 = 끔
        self.flip.setToolTip("flip — 기본(전체 설정) / 켬 / 끔")
        h.addWidget(self.use, 1)
        h.addWidget(QLabel("회전"))
        h.addWidget(self.lo)
        h.addWidget(QLabel("~"))
        h.addWidget(self.hi)
        h.addWidget(QLabel("flip"))
        h.addWidget(self.flip)
        self.use.toggled.connect(self._sync_enabled)
        self.use.toggled.connect(lambda _v: self.changed.emit())
        self.lo.valueChanged.connect(lambda _v: self.changed.emit())
        self.hi.valueChanged.connect(lambda _v: self.changed.emit())
        self.flip.currentIndexChanged.connect(lambda _i: self.changed.emit())
        self._sync_enabled(False)

    def _sync_enabled(self, on: bool) -> None:
        for w in (self.lo, self.hi, self.flip):
            w.setEnabled(on)

    def set_override(self, o: R.GeometryOverride | None) -> None:
        """레시피 값 → 위젯(시그널 막고). None = 미적용."""
        blockers = [QSignalBlocker(w) for w in (self.use, self.lo, self.hi, self.flip)]
        self.use.setChecked(o is not None)
        if o is not None:
            if o.rotate is not None:
                self.lo.setValue(float(o.rotate[0]))
                self.hi.setValue(float(o.rotate[1]))
            self.flip.setCurrentIndex(0 if o.flip is None else (1 if o.flip else 2))
        self._sync_enabled(o is not None)
        del blockers

    def override(self, keep: R.GeometryOverride | None) -> dict[str, Any] | None:
        """위젯 → 오버라이드 dict(적용이 꺼져 있으면 None). ``keep`` 의 scale 은 보존."""
        if not self.use.isChecked():
            return None
        d: dict[str, Any] = {"rotate": [self.lo.value(), self.hi.value()]}
        idx = self.flip.currentIndex()
        d["flip"] = None if idx == 0 else idx == 1
        if keep is not None and keep.scale is not None:
            d["scale"] = list(keep.scale)
        return d


class PerClassEditor(QWidget):
    """``geometry.per_class`` 표. ``set_classes`` 로 행을 만들고 ``set_value`` 로 채운다; 편집은 디바운스 뒤 ``changed``."""

    changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 4, 0, 0)
        v.setSpacing(3)
        self.title = QLabel("클래스별 per_class — 회전·flip 만 (scale 은 YAML)")
        self.title.setObjectName("Muted")
        self.title.setWordWrap(True)
        v.addWidget(self.title)
        self.rows_box = QVBoxLayout()
        self.rows_box.setSpacing(2)
        v.addLayout(self.rows_box)
        self.rows: dict[str, _Row] = {}
        self._current: dict[str, R.GeometryOverride] = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self._emit)
        self.hide()

    # ------------------------------------------------------------------

    def set_classes(self, classes: list[str]) -> None:
        """행 재구성(은행 클래스 ∪ 현재 키). 클래스가 없으면 숨긴다."""
        wanted = list(dict.fromkeys([*classes, *self._current]))
        if list(self.rows) == wanted:
            return
        for r in self.rows.values():
            r.hide()
            r.deleteLater()
        self.rows = {}
        for c in wanted:
            row = _Row(c)
            row.changed.connect(self._schedule)
            self.rows_box.addWidget(row)
            self.rows[c] = row
        self.setVisible(bool(wanted))
        self._fill()

    def set_value(self, per_class: dict[str, R.GeometryOverride]) -> None:
        """세션 → 표(디바운스 대기 중이면 덮어쓰지 않는다 — 카드 폼과 같은 규칙)."""
        self._current = dict(per_class)
        missing = [c for c in per_class if c not in self.rows]
        if missing:
            self.set_classes([*self.rows, *missing])
            return
        if self._timer.isActive():
            return
        self._fill()

    def value(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for c, row in self.rows.items():
            d = row.override(self._current.get(c))
            if d is not None:
                out[c] = d
        return out

    # ------------------------------------------------------------------

    def _fill(self) -> None:
        for c, row in self.rows.items():
            row.set_override(self._current.get(c))

    def _schedule(self) -> None:
        self._timer.start()

    def flush(self) -> None:
        """테스트·즉시 반영용 — 디바운스를 기다리지 않고 지금 emit."""
        if self._timer.isActive():
            self._timer.stop()
        self._emit()

    def _emit(self) -> None:
        self.changed.emit(self.value())

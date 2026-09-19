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
FLIP_CHOICES: tuple[tuple[str, str], ...] = (  # (라벨, 값) — "" = 기본(전체 설정)
    ("기본", ""),
    ("없음", "none"),
    ("좌우", "horizontal"),
    ("상하", "vertical"),
    ("둘 다", "both"),
)


class _Row(QWidget):
    changed = Signal()

    def __init__(self, cls: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.cls = cls
        outer = QVBoxLayout(
            self
        )  # 두 줄: [☑ 클래스] / [회전 lo ~ hi · 뒤집기] — 한 줄이면 카드 폭(≈300 px)을 넘긴다
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        h = QHBoxLayout()
        h.setContentsMargins(14, 0, 0, 0)
        h.setSpacing(6)
        self.use = QCheckBox(cls)
        self.use.setToolTip(
            "이 클래스만 다른 크기·회전(회전 범위·뒤집기)으로 — 스크래치는 그대로, 찍힘만 좁힐 때"
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
            sp.setMinimumWidth(54)
        self.flip = QComboBox()
        for label, val in FLIP_CHOICES:
            self.flip.addItem(label, val)
        self.flip.setCurrentIndex(self.flip.findData("none"))  # 조명 의존 클래스의 기본 = 없음
        self.flip.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )  # 콤보 폭이 카드 폭을 넘기지 않게
        self.flip.setMinimumWidth(48)
        self.flip.setToolTip(
            "flip(뒤집기) — 기본(전체 설정) / 없음 / 좌우 / 상하 / 둘 다. 빛이 위·아래에서 오면 좌우는 안전합니다"
        )
        outer.addWidget(self.use)
        h.addWidget(QLabel("회전"))
        h.addWidget(self.lo, 1)
        h.addWidget(QLabel("~"))
        h.addWidget(self.hi, 1)
        h.addWidget(QLabel("뒤집기"))
        h.addWidget(self.flip, 1)
        outer.addLayout(h)
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
            self.flip.setCurrentIndex(self.flip.findData("" if o.flip is None else o.flip))
        self._sync_enabled(o is not None)
        del blockers

    def override(self, keep: R.GeometryOverride | None) -> dict[str, Any] | None:
        """위젯 → 오버라이드 dict(적용이 꺼져 있으면 None). ``keep`` 의 scale 은 보존."""
        if not self.use.isChecked():
            return None
        d: dict[str, Any] = {"rotate": [self.lo.value(), self.hi.value()]}
        fv = self.flip.currentData()
        d["flip"] = None if fv in ("", None) else fv
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
        self.title = QLabel("클래스별 예외 — 회전·뒤집기 (크기는 YAML 에서)")
        self.title.setToolTip(
            "geometry.per_class — 클래스마다 회전·뒤집기를 따로 둡니다(찍힘만 ±15° 등)"
        )
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

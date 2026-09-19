"""``FieldSpec`` 목록 → 편집 위젯 폼(``ParamForm``). 값 변경은 ``value_changed(name, value)`` 로만 나가고 상태는
``StudioSession`` 이 가진다(다른 패널과 같은 규칙).

- 숫자 스핀박스는 ``valueChanged`` 를 350 ms 디바운스해 커밋(화살표 연타·타이핑 중 미리보기 재계산 방지). 콤보·체크는 즉시,
  텍스트·경로는 ``editingFinished``.
- ``X | None`` 필드는 체크박스가 앞에 있다 — 끄면 ``None``, 켜면 ``spec.on_default``.
- ``set_specs`` 는 이름·종류가 같으면 위젯을 재사용하고 값만 갱신(시그널 억제) — 재검증 실패 후 되돌리기가 깜빡이지 않게.
- 열린 구간(``gt``/``lt``)은 스핀 한계를 한 step 안쪽으로.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QWidget,
)

from anograft.gui.studio.params import FieldSpec, coerce, spin_bounds, spin_step

DEBOUNCE_MS = 350
LABEL_W = 108  # 라벨 열 고정 폭 — 긴 이름(shrink_on_fail.factor)은 가운데 생략, 전체 이름은 툴팁
_INT_LIMIT = 2_000_000_000  # QSpinBox 는 int32


def _decimals(spec: FieldSpec) -> int:
    return 3 if spin_step(spec) < 0.1 else 2


def _make_spin(spec: FieldSpec) -> QSpinBox | QDoubleSpinBox:
    lo, hi = spin_bounds(spec)
    step = spin_step(spec)
    if spec.kind in ("int", "int_range"):
        w: QSpinBox | QDoubleSpinBox = QSpinBox()
        w.setRange(
            int(max(lo, -_INT_LIMIT)) + (1 if spec.lo_open else 0),
            int(min(hi, _INT_LIMIT)) - (1 if spec.hi_open else 0),
        )
    else:
        w = QDoubleSpinBox()
        w.setDecimals(_decimals(spec))
        eps = 10 ** (-_decimals(spec))
        w.setRange(lo + (eps if spec.lo_open else 0.0), hi - (eps if spec.hi_open else 0.0))
        w.setSingleStep(step)
    w.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)  # 폭을 아낀다 — 화살표 키·휠로 조절
    w.setAlignment(Qt.AlignmentFlag.AlignRight)
    w.setKeyboardTracking(False)  # 타이핑 중간값은 커밋하지 않는다
    # 스핀박스는 범위(±1e9)의 자릿수로 sizeHint 를 잡아 카드 폭(300px)을 넘긴다 → 힌트를 무시하고 레이아웃이 준 폭을 쓴다
    w.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    w.setMinimumWidth(44)
    return w


class _Row(QWidget):
    """필드 하나 — 라벨·(optional 체크)·편집기. ``changed(name, value)`` 는 이미 ``coerce`` 된 값."""

    changed = Signal(str, object)

    def __init__(self, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self._commit)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.toggle: QCheckBox | None = None
        if spec.optional:
            self.toggle = QCheckBox()
            self.toggle.setToolTip(
                "켜면 이 옵션을 씁니다. 끄면 null — 난수도 쓰지 않아 기존 결과가 그대로입니다"
            )
            self.toggle.toggled.connect(self._on_toggle)
            lay.addWidget(self.toggle)
        self.editors: list[QWidget] = []
        k = spec.kind
        if k in ("int", "float"):
            self._add(lay, _make_spin(spec), "valueChanged")
        elif k in ("int_range", "range"):
            self._add(lay, _make_spin(spec), "valueChanged")
            dash = QLabel("–")
            dash.setObjectName("Muted")
            lay.addWidget(dash)
            self._add(lay, _make_spin(spec), "valueChanged")
        elif k == "bool":
            cb = QCheckBox()
            cb.toggled.connect(lambda _v: self._commit())
            self.editors.append(cb)
            lay.addWidget(cb)
        elif k == "choice":
            combo = QComboBox()
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(6)
            for c in spec.choices:
                combo.addItem(c, c)
            combo.currentIndexChanged.connect(lambda _i: self._commit())
            self.editors.append(combo)
            lay.addWidget(combo, 1)
        else:  # text · list · path
            edit = QLineEdit()
            edit.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            edit.setMinimumWidth(44)
            edit.setPlaceholderText({"list": "a, b, c", "path": "폴더 경로", "text": ""}.get(k, ""))
            edit.editingFinished.connect(self._commit)
            self.editors.append(edit)
            lay.addWidget(edit, 1)
            if k == "path":
                btn = QPushButton("…")
                btn.setFixedWidth(26)
                btn.setToolTip("폴더 고르기")
                btn.clicked.connect(self._pick_dir)
                lay.addWidget(btn)
        self.setToolTip(spec.hint)
        self.set_value(spec.value)

    def _add(self, lay: QHBoxLayout, w: QSpinBox | QDoubleSpinBox, signal: str) -> None:
        getattr(w, signal).connect(lambda _v: self._timer.start())
        self.editors.append(w)
        lay.addWidget(w, 1)

    # ------------------------------------------------------------------ 값

    def set_value(self, value: Any) -> None:
        """세션 → 위젯 (시그널 억제). 디바운스 대기 중인 사용자 편집이 있으면 덮어쓰지 않는다 — 다른 필드의 커밋이
        폼을 재동기화할 때 아직 커밋 안 된 값이 되돌아가 사라지는 것을 막는다(그 커밋은 곧 따라온다)."""
        if self._timer.isActive():
            return
        blockers = [QSignalBlocker(w) for w in self.editors]
        if self.toggle is not None:
            blockers.append(QSignalBlocker(self.toggle))
            self.toggle.setChecked(value is not None)
        enabled = not (self.spec.optional and value is None)
        for w in self.editors:
            w.setEnabled(enabled)
        if value is None:
            del blockers
            return
        k = self.spec.kind
        if k in ("int", "float"):
            self.editors[0].setValue(value)  # type: ignore[attr-defined]
        elif k in ("int_range", "range"):
            self.editors[0].setValue(value[0])  # type: ignore[attr-defined]
            self.editors[1].setValue(value[1])  # type: ignore[attr-defined]
        elif k == "bool":
            self.editors[0].setChecked(bool(value))  # type: ignore[attr-defined]
        elif k == "choice":
            combo: QComboBox = self.editors[0]  # type: ignore[assignment]
            i = combo.findData(str(value))
            if i >= 0:
                combo.setCurrentIndex(i)
        elif k == "list":
            self.editors[0].setText(", ".join(str(s) for s in value))  # type: ignore[attr-defined]
        else:
            self.editors[0].setText(str(value))  # type: ignore[attr-defined]
        del blockers

    def value(self) -> Any:
        """위젯 → 레시피 값(``coerce`` 적용). optional 이 꺼져 있으면 None."""
        if self.toggle is not None and not self.toggle.isChecked():
            return None
        k = self.spec.kind
        if k in ("int", "float"):
            raw: Any = self.editors[0].value()  # type: ignore[attr-defined]
        elif k in ("int_range", "range"):
            raw = (self.editors[0].value(), self.editors[1].value())  # type: ignore[attr-defined]
        elif k == "bool":
            raw = self.editors[0].isChecked()  # type: ignore[attr-defined]
        elif k == "choice":
            raw = self.editors[0].currentData()  # type: ignore[attr-defined]
        else:
            raw = self.editors[0].text()  # type: ignore[attr-defined]
        return coerce(self.spec, raw)

    # ------------------------------------------------------------------ 이벤트

    def _on_toggle(self, on: bool) -> None:
        if on:
            self.set_value(self.spec.on_default)
            if self.toggle is not None:
                with QSignalBlocker(self.toggle):
                    self.toggle.setChecked(True)
        else:
            for w in self.editors:
                w.setEnabled(False)
        self._timer.stop()
        self.changed.emit(self.spec.name, self.value())

    def _commit(self) -> None:
        self._timer.stop()
        self.changed.emit(self.spec.name, self.value())

    def _pick_dir(self) -> None:
        edit: QLineEdit = self.editors[0]  # type: ignore[assignment]
        d = QFileDialog.getExistingDirectory(self, "폴더", edit.text() or ".")
        if d:
            edit.setText(d)
            self._commit()

    def set_error(self, on: bool) -> None:
        for w in self.editors:
            w.setProperty("error", on)
            w.style().unpolish(w)
            w.style().polish(w)


class ParamForm(QWidget):
    """스펙 목록을 2열 그리드(라벨 | 편집기)로. 스펙이 비면 "옵션 없음"."""

    value_changed = Signal(str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ParamForm")
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(4)
        self.grid.setColumnStretch(1, 1)
        self.rows: dict[str, _Row] = {}
        self._labels: dict[str, QLabel] = {}
        self._signature: tuple[tuple[str, str, bool, tuple[str, ...]], ...] = ()
        self.empty = QLabel("이 방법에는 조정할 값이 없습니다")
        self.empty.setObjectName("Muted")
        self.empty.hide()
        self.grid.addWidget(self.empty, 0, 0, 1, 2)

    @staticmethod
    def _sig(specs: Sequence[FieldSpec]) -> tuple[tuple[str, str, bool, tuple[str, ...]], ...]:
        return tuple((s.name, s.kind, s.optional, s.choices) for s in specs)

    def set_specs(self, specs: Sequence[FieldSpec]) -> None:
        sig = self._sig(specs)
        if sig == self._signature:
            for s in specs:  # 값만 갱신
                self.rows[s.name].set_value(s.value)
            return
        self._signature = sig
        for w in list(self.rows.values()) + list(self._labels.values()):
            self.grid.removeWidget(w)
            w.hide()  # deleteLater 는 이벤트 루프가 돌아야 지워진다 — 그때까지 옛 라벨이 새 라벨 위에 겹쳐 그려진다
            w.setParent(None)
            w.deleteLater()
        self.rows.clear()
        self._labels.clear()
        self.empty.setVisible(not specs)
        for i, s in enumerate(specs, start=1):
            lab = QLabel()
            lab.setObjectName("StageParams")
            lab.setFixedWidth(LABEL_W)
            lab.setText(
                QFontMetrics(lab.font()).elidedText(s.name, Qt.TextElideMode.ElideMiddle, LABEL_W)
            )
            lab.setToolTip(f"{s.name}\n{s.hint}")
            row = _Row(s)
            row.changed.connect(self.value_changed.emit)
            self.grid.addWidget(lab, i, 0, Qt.AlignmentFlag.AlignVCenter)
            self.grid.addWidget(row, i, 1)
            self.rows[s.name] = row
            self._labels[s.name] = lab

    def values(self) -> dict[str, Any]:
        return {n: r.value() for n, r in self.rows.items()}

    def set_error(self, name: str | None) -> None:
        """``name`` 행만 오류 표시(None 이면 전부 해제)."""
        for n, r in self.rows.items():
            r.set_error(name is not None and (n == name or n.startswith(f"{name}.")))

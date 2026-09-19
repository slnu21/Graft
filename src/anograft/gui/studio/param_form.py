"""``FieldSpec`` 목록 → 편집 위젯 폼(``ParamForm``). 값 변경은 ``value_changed(name, value)`` 로만 나가고 상태는
``StudioSession`` 이 가진다(다른 패널과 같은 규칙).

- 숫자 스핀박스는 ``valueChanged`` 를 350 ms 디바운스해 커밋(화살표 연타·타이핑 중 미리보기 재계산 방지). 콤보·체크는 즉시,
  텍스트·경로는 ``editingFinished``.
- ``X | None`` 필드는 체크박스가 앞에 있다 — 끄면 ``None``, 켜면 ``spec.on_default``.
- ``set_specs`` 는 이름·종류가 같으면 위젯을 재사용하고 값·기준값만 갱신(시그널 억제) — 재검증 실패 후 되돌리기가 깜빡이지 않게.
- 열린 구간(``gt``/``lt``)은 스핀 한계를 한 step 안쪽으로.
- **v0.9(param-form-v2)**: 기준값(프리셋)과 다른 행은 라벨이 teal + ↺ 되돌리기 버튼(``spec.modified``) · 좁은 실수 범위(폭 ≤ 10)는
  슬라이더 + 스핀(더블클릭 = 되돌리기) · ``advanced`` 필드는 "고급 옵션 (n)" 아래 접힘(열림 여부는 QSettings 로 기억).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import QSettings, QSignalBlocker, Qt, QTimer, Signal
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
    QSlider,
    QSpinBox,
    QToolButton,
    QWidget,
)

from anograft.gui.studio.params import FieldSpec, _norm, coerce, spin_bounds, spin_step

DEBOUNCE_MS = 350
LABEL_W = 116  # 라벨 열 고정 폭 — 긴 라벨(정렬 최소 일관성)은 오른쪽 생략, 전체는 툴팁
_INT_LIMIT = 2_000_000_000  # QSpinBox 는 int32
SLIDER_MAX_SPAN = 10.0  # 실수 범위 폭이 이 이하면(세기 0..1 · 임계 −1..1) 슬라이더를 함께 둔다
_SLIDER_STEPS = 1000
ADVANCED_KEY = "studio/advanced_open"  # QSettings — 고급 옵션 펼침 기억(전 카드 공통 기본값)
SETTINGS: QSettings | None = (
    None  # 앱이 `use_settings()` 로 준다. None(테스트) 이면 기억하지 않는다
)


def use_settings(store: QSettings | None) -> None:
    global SETTINGS
    SETTINGS = store


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


def wants_slider(spec: FieldSpec) -> bool:
    """실수 한 값 + 양끝 유한 + 폭 ≤ SLIDER_MAX_SPAN — 세기·임계·비율 류."""
    return (
        spec.kind == "float"
        and spec.lo is not None
        and spec.hi is not None
        and 0 < (spec.hi - spec.lo) <= SLIDER_MAX_SPAN
    )


class _Slider(QSlider):
    """더블클릭 = 되돌리기(Lightroom 식)."""

    reset_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setRange(0, _SLIDER_STEPS)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(40)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt 규약)
        self.reset_requested.emit()
        event.accept()


class _Row(QWidget):
    """필드 하나 — (optional 체크)·편집기·(슬라이더)·↺. ``changed(name, value)`` 는 이미 ``coerce`` 된 값.
    ``modified_changed`` 는 기준값 대비 바뀜 여부가 달라졌을 때(폼이 라벨 색·헤더 수를 갱신)."""

    changed = Signal(str, object)
    modified_changed = Signal(str, bool)

    def __init__(self, spec: FieldSpec, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = spec
        self._modified = False
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
        self.slider: _Slider | None = None
        k = spec.kind
        if k in ("int", "float"):
            self._add(lay, _make_spin(spec), "valueChanged")
            if wants_slider(spec):
                self.slider = _Slider()
                self.slider.valueChanged.connect(self._on_slider)
                self.slider.reset_requested.connect(self.reset)
                self.editors[0].valueChanged.connect(self._sync_slider)  # type: ignore[attr-defined]
                lay.addWidget(self.slider, 2)
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
            lay.addStretch(1)
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
        # ↺ 되돌리기 — 기준값(프리셋)과 다를 때만 보인다
        self.btn_reset = QToolButton()
        self.btn_reset.setObjectName("Reset")
        self.btn_reset.setText("↺")
        self.btn_reset.setFixedSize(18, 18)
        self.btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset.clicked.connect(self.reset)
        self.btn_reset.hide()
        lay.addWidget(self.btn_reset)
        self.setToolTip(spec.tooltip)
        self.set_value(spec.value)
        self._refresh_modified(force=True)

    def _add(self, lay: QHBoxLayout, w: QSpinBox | QDoubleSpinBox, signal: str) -> None:
        getattr(w, signal).connect(lambda _v: self._timer.start())
        self.editors.append(w)
        lay.addWidget(w, 1)

    # ------------------------------------------------------------------ 값

    def update_spec(self, spec: FieldSpec) -> None:
        """같은 이름·종류의 새 스펙(값·기준값) — 위젯은 그대로 두고 값만 맞춘다."""
        self.spec = spec
        self.set_value(spec.value)
        self._refresh_modified()

    def set_value(self, value: Any) -> None:
        """세션 → 위젯 (시그널 억제). 디바운스 대기 중인 사용자 편집이 있으면 덮어쓰지 않는다 — 다른 필드의 커밋이
        폼을 재동기화할 때 아직 커밋 안 된 값이 되돌아가 사라지는 것을 막는다(그 커밋은 곧 따라온다)."""
        if self._timer.isActive():
            return
        blockers = [QSignalBlocker(w) for w in self.editors]
        if self.slider is not None:
            blockers.append(QSignalBlocker(self.slider))
        if self.toggle is not None:
            blockers.append(QSignalBlocker(self.toggle))
            self.toggle.setChecked(value is not None)
        enabled = not (self.spec.optional and value is None)
        for w in self.editors:
            w.setEnabled(enabled)
        if self.slider is not None:
            self.slider.setEnabled(enabled)
        if value is None:
            del blockers
            return
        k = self.spec.kind
        if k in ("int", "float"):
            self.editors[0].setValue(value)  # type: ignore[attr-defined]
            self._sync_slider(value)
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

    @property
    def modified(self) -> bool:
        return self._modified

    def _refresh_modified(self, *, force: bool = False) -> None:
        """현재 위젯 값 vs 기준값 → ↺ 표시 + 시그널(변할 때만)."""
        now = self.spec.has_baseline and _norm(self.value()) != _norm(self.spec.baseline)
        if now != self._modified or force:
            self._modified = now
            self.btn_reset.setVisible(now)
            if now:
                self.btn_reset.setToolTip(
                    f"프리셋 값으로 되돌리기 · Reset to preset ({self.spec.baseline!r})"
                )
            self.modified_changed.emit(self.spec.name, now)

    def reset(self) -> None:
        """기준값으로 — 기준이 없으면 무시."""
        if not self.spec.has_baseline:
            return
        self._timer.stop()
        self.set_value(self.spec.baseline)
        self._commit()

    # ------------------------------------------------------------------ 슬라이더

    def _slider_pos(self, v: float) -> int:
        lo, hi = float(self.spec.lo), float(self.spec.hi)  # type: ignore[arg-type]
        return round((v - lo) / (hi - lo) * _SLIDER_STEPS)

    def _sync_slider(self, v: float) -> None:
        if self.slider is None:
            return
        with QSignalBlocker(self.slider):
            self.slider.setValue(self._slider_pos(float(v)))

    def _on_slider(self, pos: int) -> None:
        lo, hi = float(self.spec.lo), float(self.spec.hi)  # type: ignore[arg-type]
        v = lo + (hi - lo) * pos / _SLIDER_STEPS
        spin: QDoubleSpinBox = self.editors[0]  # type: ignore[assignment]
        with QSignalBlocker(spin):
            spin.setValue(v)
        self._timer.start()

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
            if self.slider is not None:
                self.slider.setEnabled(False)
        self._timer.stop()
        self._refresh_modified()
        self.changed.emit(self.spec.name, self.value())

    def _commit(self) -> None:
        self._timer.stop()
        self._refresh_modified()
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


def _repolish(w: QWidget) -> None:
    w.style().unpolish(w)
    w.style().polish(w)


class ParamForm(QWidget):
    """스펙 목록을 2열 그리드(라벨 | 편집기)로. 스펙이 비면 "조정할 값 없음". ``advanced`` 스펙은 "고급 옵션 (n)" 아래 접힌다."""

    value_changed = Signal(str, object)
    modified_count_changed = Signal(int)

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
        self._advanced: list[str] = []
        self._signature: tuple[tuple[str, str, bool, tuple[str, ...]], ...] = ()
        self.empty = QLabel("이 방법에는 조정할 값이 없습니다")
        self.empty.setObjectName("Muted")
        self.empty.hide()
        self.grid.addWidget(self.empty, 0, 0, 1, 2)
        self.btn_advanced = QToolButton()
        self.btn_advanced.setObjectName("Advanced")
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_advanced.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_advanced.setToolTip(
            "프리셋을 고른 뒤 보통 만지지 않는 값들 · Advanced options (state is remembered)"
        )
        self.btn_advanced.setChecked(
            bool(SETTINGS.value(ADVANCED_KEY, False, type=bool)) if SETTINGS is not None else False
        )
        self.btn_advanced.toggled.connect(self._on_advanced_toggled)
        self.btn_advanced.hide()

    @staticmethod
    def _sig(specs: Sequence[FieldSpec]) -> tuple[tuple[str, str, bool, tuple[str, ...]], ...]:
        return tuple((s.name, s.kind, s.optional, s.choices) for s in specs)

    def set_specs(self, specs: Sequence[FieldSpec]) -> None:
        sig = self._sig(specs)
        if sig == self._signature:
            for s in specs:  # 값·기준값만 갱신
                self.rows[s.name].update_spec(s)
            self._emit_count()
            return
        self._signature = sig
        for w in [*self.rows.values(), *self._labels.values()]:
            self.grid.removeWidget(w)
            w.hide()  # deleteLater 는 이벤트 루프가 돌아야 지워진다 — 그때까지 옛 라벨이 새 라벨 위에 겹쳐 그려진다
            w.setParent(None)
            w.deleteLater()
        self.grid.removeWidget(self.btn_advanced)
        self.rows.clear()
        self._labels.clear()
        self._advanced = []
        self.empty.setVisible(not specs)
        basic = [s for s in specs if not s.advanced]
        advanced = [s for s in specs if s.advanced]
        i = 1
        for s in basic:
            self._add_row(s, i)
            i += 1
        if advanced:
            self.grid.addWidget(self.btn_advanced, i, 0, 1, 2)
            self.btn_advanced.show()
            i += 1
            for s in advanced:
                self._add_row(s, i)
                self._advanced.append(s.name)
                i += 1
        else:
            self.btn_advanced.hide()
        self._apply_advanced_visibility()
        self._emit_count()

    def _add_row(self, s: FieldSpec, i: int) -> None:
        lab = QLabel()
        lab.setObjectName("StageParams")
        lab.setFixedWidth(LABEL_W)
        lab.setText(
            QFontMetrics(lab.font()).elidedText(s.title, Qt.TextElideMode.ElideRight, LABEL_W)
        )
        lab.setToolTip(s.tooltip)
        row = _Row(s)
        row.changed.connect(self.value_changed.emit)
        row.modified_changed.connect(self._on_row_modified)
        self.grid.addWidget(lab, i, 0, Qt.AlignmentFlag.AlignVCenter)
        self.grid.addWidget(row, i, 1)
        self.rows[s.name] = row
        self._labels[s.name] = lab
        self._mark_label(s.name, row.modified)

    # ------------------------------------------------------------------ 바뀜 표시

    def _mark_label(self, name: str, modified: bool) -> None:
        lab = self._labels.get(name)
        if lab is None:
            return
        lab.setProperty("modified", modified)
        _repolish(lab)

    def _on_row_modified(self, name: str, modified: bool) -> None:
        self._mark_label(name, modified)
        self._emit_count()

    def modified_names(self) -> list[str]:
        return [n for n, r in self.rows.items() if r.modified]

    def _emit_count(self) -> None:
        n = len(self.modified_names())
        self.btn_advanced.setText(self._advanced_text())
        self.modified_count_changed.emit(n)

    def reset_all(self) -> None:
        """바뀐 행 전부 기준값으로(행마다 value_changed 가 난다)."""
        for n in self.modified_names():
            self.rows[n].reset()

    # ------------------------------------------------------------------ 고급 접기

    def _advanced_text(self) -> str:
        n = len(self._advanced)
        mod = sum(1 for a in self._advanced if a in self.rows and self.rows[a].modified)
        arrow = "▾" if self.btn_advanced.isChecked() else "▸"
        return f"{arrow} 고급 옵션 ({n})" + (f" · 바뀜 {mod}" if mod else "")

    def _on_advanced_toggled(self, on: bool) -> None:
        if SETTINGS is not None:
            SETTINGS.setValue(ADVANCED_KEY, on)
        self._apply_advanced_visibility()

    def _apply_advanced_visibility(self) -> None:
        show = self.btn_advanced.isChecked()
        for n in self._advanced:
            if n in self.rows:
                self.rows[n].setVisible(show)
                self._labels[n].setVisible(show)
        self.btn_advanced.setText(self._advanced_text())

    def set_advanced_open(self, on: bool) -> None:
        self.btn_advanced.setChecked(on)

    # ------------------------------------------------------------------ 값·오류

    def values(self) -> dict[str, Any]:
        return {n: r.value() for n, r in self.rows.items()}

    def set_error(self, name: str | None) -> None:
        """``name`` 행만 오류 표시(None 이면 전부 해제)."""
        for n, r in self.rows.items():
            r.set_error(name is not None and (n == name or n.startswith(f"{name}.")))

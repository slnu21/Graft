"""스튜디오 패널 위젯 — ``StripBar``(프리셋·시드·변형 수·축소) · ``InputsPanel``(은행·대상 경로) · ``TargetRail``(대상 썸네일) ·
``PipelinePanel``(7단계 카드: method 콤보 + **파라미터 폼**(스키마 자동 생성, `params.py`/`param_form.py`) + 단계별 썸네일 + 경고/오류). 전부 "값이 바뀌었다"는 시그널만 내고 상태는 ``StudioSession``이 가진다.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from anograft.core import recipe as R
from anograft.core import registry
from anograft.core.pipeline import TraceStep
from anograft.gui.qt_image import to_qpixmap
from anograft.gui.studio.param_form import ParamForm
from anograft.gui.studio.params import field_specs
from anograft.gui.theme import COLORS
from anograft.preview import fit_long_side

STAGE_TITLES: tuple[tuple[str, str], ...] = (
    ("source", "소스 선택 Source"),
    ("geometry", "기하 변환 Geometry"),
    ("placement", "배치 Placement"),
    ("blend", "블렌딩 Blend"),
    ("harmonize", "조화 Harmonize"),
    ("degrade", "열화 Degrade"),
    ("gtmask", "정답 마스크 GT mask"),
)
LONG_SIDES: tuple[tuple[str, int], ...] = (
    ("1024px 축소", 1024),
    ("768px 축소", 768),
    ("512px 축소", 512),
    ("원본 해상도", 0),
)


def flat_icon(pm: QPixmap) -> QIcon:
    """선택돼도 틴트되지 않는 아이콘 — Qt는 Selected 모드 픽스맵이 없으면 하이라이트 색을 덧씌운다."""
    icon = QIcon()
    for mode in (QIcon.Mode.Normal, QIcon.Mode.Selected, QIcon.Mode.Active):
        icon.addPixmap(pm, mode, QIcon.State.Off)
        icon.addPixmap(pm, mode, QIcon.State.On)
    return icon


def h4(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("H4")
    return lab


# ---------------------------------------------------------------------------
# 상단 스트립
# ---------------------------------------------------------------------------


class StripBar(QWidget):
    preset_changed = Signal(str)
    seed_changed = Signal(int)
    variants_changed = Signal(int)
    long_side_changed = Signal(int)
    open_clicked = Signal()
    save_clicked = Signal()
    batch_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Strip")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 7, 18, 7)
        lay.setSpacing(14)

        self.preset = QComboBox()
        for name in R.preset_names():
            self.preset.addItem(name, name)
        self.preset.currentIndexChanged.connect(
            lambda _i: self.preset_changed.emit(self.preset.currentData())
        )
        lay.addWidget(self._field("프리셋 Preset", self.preset))

        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.seed.setFixedWidth(96)
        self.seed.editingFinished.connect(lambda: self.seed_changed.emit(self.seed.value()))
        lay.addWidget(self._field("시드 Seed", self.seed))

        self.variants = QSpinBox()
        self.variants.setRange(1, 12)
        self.variants.setValue(6)
        self.variants.setFixedWidth(52)
        self.variants.valueChanged.connect(self.variants_changed.emit)
        lay.addWidget(self._field("변형 Variants", self.variants))

        self.long_side = QComboBox()
        for label, px in LONG_SIDES:
            self.long_side.addItem(label, px)
        self.long_side.currentIndexChanged.connect(
            lambda _i: self.long_side_changed.emit(int(self.long_side.currentData()))
        )
        lay.addWidget(self._field("미리보기 Preview", self.long_side))

        self.note = QLabel("")
        self.note.setObjectName("Muted")
        self.note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.note, 1)

        self.btn_open = QPushButton("레시피 열기")
        self.btn_save = QPushButton("레시피 저장")
        self.btn_batch = QPushButton("배치로 보내기")
        self.btn_batch.setObjectName("Primary")
        self.btn_open.clicked.connect(self.open_clicked.emit)
        self.btn_save.clicked.connect(self.save_clicked.emit)
        self.btn_batch.clicked.connect(self.batch_clicked.emit)
        for b in (self.btn_open, self.btn_save, self.btn_batch):
            lay.addWidget(b)

    @staticmethod
    def _field(label: str, widget: QWidget) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(7)
        lab = QLabel(label)
        lab.setObjectName("Muted")
        h.addWidget(lab)
        h.addWidget(widget)
        return box

    def sync(self, recipe: R.Recipe, n_variants: int, long_side: int) -> None:
        """세션 → 위젯 (시그널 억제)."""
        for w in (self.preset, self.seed, self.variants, self.long_side):
            w.blockSignals(True)
        i = self.preset.findData(recipe.pipeline.preset)
        if i >= 0:
            self.preset.setCurrentIndex(i)
        self.seed.setValue(recipe.seed)
        self.variants.setValue(n_variants)
        j = self.long_side.findData(long_side)
        if j >= 0:
            self.long_side.setCurrentIndex(j)
        for w in (self.preset, self.seed, self.variants, self.long_side):
            w.blockSignals(False)


# ---------------------------------------------------------------------------
# 입력(은행·대상)
# ---------------------------------------------------------------------------


class InputsPanel(QWidget):
    open_requested = Signal(str, str)  # bank, targets

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Inputs")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 6)
        lay.setSpacing(6)
        lay.addWidget(h4("입력 Inputs"))
        self.bank = QLineEdit()
        self.bank.setPlaceholderText("결함 은행 폴더 (bank.yaml)")
        self.targets = QLineEdit()
        self.targets.setPlaceholderText("정상 이미지 폴더 또는 목록 .txt")
        lay.addLayout(self._row("은행", self.bank, ("폴더", self._pick_bank)))
        lay.addLayout(
            self._row(
                "대상",
                self.targets,
                ("폴더", self._pick_targets_dir),
                ("목록", self._pick_targets_file),
            )
        )
        self.btn_open = QPushButton("열기 Open")
        self.btn_open.clicked.connect(
            lambda: self.open_requested.emit(self.bank.text().strip(), self.targets.text().strip())
        )
        lay.addWidget(self.btn_open)
        self.summary = QLabel("")
        self.summary.setObjectName("Muted")
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)

    def _row(self, label: str, edit: QLineEdit, *buttons: tuple[str, Any]) -> QHBoxLayout:
        h = QHBoxLayout()
        h.setSpacing(5)
        lab = QLabel(label)
        lab.setObjectName("Muted")
        lab.setFixedWidth(28)
        h.addWidget(lab)
        h.addWidget(edit, 1)
        for text, slot in buttons:
            b = QPushButton(text)
            b.setFixedWidth(44)
            b.clicked.connect(slot)
            h.addWidget(b)
        return h

    def _pick_bank(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "결함 은행 폴더", self.bank.text() or ".")
        if d:
            self.bank.setText(Path(d).as_posix())

    def _pick_targets_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "정상 이미지 폴더", self.targets.text() or ".")
        if d:
            self.targets.setText(Path(d).as_posix())

    def _pick_targets_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "정상 이미지 목록", self.targets.text() or ".", "목록 (*.txt)"
        )
        if f:
            self.targets.setText(Path(f).as_posix())

    def sync(self, recipe: R.Recipe, summary: str) -> None:
        self.bank.setText(recipe.inputs.bank_key())  # 빈칸 = 은행 없음(self-cut·perlin)
        self.targets.setText(recipe.inputs.targets.as_posix())
        self.summary.setText(summary)


# ---------------------------------------------------------------------------
# 대상 레일
# ---------------------------------------------------------------------------


class TargetRail(QWidget):
    selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Rail")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 10)
        lay.setSpacing(6)
        self.title = h4("대상 정상 이미지 · 0장")
        lay.addWidget(self.title)
        self.list = QListWidget()
        # IconMode + 세로 흐름 = 아이콘 아래 캡션이 붙는 카드 목록 (목업 .tgt)
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setFlow(QListWidget.Flow.TopToBottom)
        self.list.setWrapping(False)
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setMovement(QListWidget.Movement.Static)
        self.list.setIconSize(QSize(176, 118))
        self.list.setSpacing(4)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.currentRowChanged.connect(self._on_row)
        lay.addWidget(self.list, 1)
        self._paths: list[Path] = []

    def _on_row(self, row: int) -> None:
        if row >= 0:
            self.selected.emit(row)

    def set_targets(self, paths: Sequence[Path], current: int = 0) -> None:
        self._paths = list(paths)
        self.list.blockSignals(True)
        self.list.clear()
        for p in self._paths:
            item = QListWidgetItem(p.name)
            item.setToolTip(p.as_posix())
            item.setSizeHint(QSize(192, 162))
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            self.list.addItem(item)
        if self._paths:
            self.list.setCurrentRow(min(current, len(self._paths) - 1))
        self.list.blockSignals(False)
        self.title.setText(f"대상 정상 이미지 · {len(self._paths)}장")

    def set_thumb(self, path: Path, image: np.ndarray, shape: tuple[int, int]) -> None:
        try:
            i = self._paths.index(path)
        except ValueError:
            return
        item = self.list.item(i)
        if item is None:
            return
        item.setIcon(flat_icon(to_qpixmap(image)))
        item.setText(f"{path.name}\n{shape[1]}×{shape[0]}")

    def select(self, index: int) -> None:
        self.list.blockSignals(True)
        self.list.setCurrentRow(index)
        self.list.blockSignals(False)


# ---------------------------------------------------------------------------
# 파이프라인 카드
# ---------------------------------------------------------------------------


def params_text(cfg: Any) -> str:
    """설정 모델 → ``key: v · key: v`` 한 줄(method/policy 제외, 범위는 ``a–b``)."""
    d = cfg.model_dump(mode="json") if hasattr(cfg, "model_dump") else dict(cfg)
    parts: list[str] = []
    for k, v in d.items():
        if k in ("method", "policy", "roi"):
            continue
        if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            parts.append(f"{k} {v[0]:g}–{v[1]:g}")
        elif isinstance(v, dict):
            parts.append(f"{k} " + "/".join(f"{a}={b}" for a, b in v.items()))
        elif v is None:
            parts.append(f"{k} –")
        else:
            parts.append(f"{k} {v}")
    return " · ".join(parts)


def stage_thumbnail(stage: str, steps: Sequence[TraceStep], size: int = 132) -> np.ndarray | None:
    """스테이지의 마지막 TraceStep에서 보여 줄 이미지 하나. 결함 루프 단계는 배치 bbox 주변 크롭."""
    mine = [s for s in steps if s.stage == stage]
    if not mine:
        return None
    ctx = mine[-1].ctx
    img: np.ndarray | None
    if stage == "source":
        img = ctx.source.image if ctx.source is not None else None
    elif stage == "geometry":
        img = ctx.patch
    elif stage == "roi":
        img = (ctx.roi.astype(np.uint8) * 255) if ctx.roi is not None else None
    elif stage == "gtmask":
        img = ctx.gt_mask
    elif stage == "degrade":
        img = ctx.composite
    else:  # placement · blend · harmonize — 배치 bbox 주변 크롭
        if ctx.placement is None:
            return None
        x, y, w, h = ctx.placement.bbox
        m = 24
        hh, ww = ctx.composite.shape[:2]
        img = ctx.composite[max(0, y - m) : min(hh, y + h + m), max(0, x - m) : min(ww, x + w + m)]
        if stage == "placement" and ctx.placed_mask is not None:
            img = img.copy()
            pm = (
                ctx.placed_mask[
                    max(0, y - m) : min(hh, y + h + m), max(0, x - m) : min(ww, x + w + m)
                ]
                > 0
            )
            img[pm] = (img[pm] * 0.5 + np.array([109, 77, 255]) * 0.5).astype(np.uint8)
    if img is None or img.size == 0:
        return None
    return fit_long_side(np.ascontiguousarray(img), size)


class StageCard(QFrame):
    """파이프라인 카드 하나 — 헤더(번호·제목·method 콤보) · 썸네일 + 경고/오류 · **파라미터 폼**(스키마에서 자동 생성).
    배치 카드는 하위 스테이지 ROI(자기 method 콤보 + 폼)를 함께 가진다."""

    method_changed = Signal(str, str)  # stage, method
    field_changed = Signal(str, str, object)  # stage, field(점 경로), value

    def __init__(self, no: int, stage: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StageCard")
        self.stage = stage
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(8)
        num = QLabel(str(no))
        num.setObjectName("StageNo")
        head.addWidget(num)
        self.title = QLabel(title)
        self.title.setObjectName("StageTitle")
        head.addWidget(self.title, 1)
        self.method = QComboBox()
        self.method.setMinimumWidth(96)
        head.addWidget(self.method)
        lay.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(8)
        self.thumb = QLabel()
        self.thumb.setFixedSize(132, 88)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet(
            f"background: {COLORS['bg']}; border: 1px solid {COLORS['line']}; border-radius: 4px;"
        )
        body.addWidget(self.thumb)
        notes = QVBoxLayout()
        notes.setSpacing(4)
        # fail-soft 경고(`<stage>: …`)를 그 스테이지 카드에 바로 보인다 — 로그에만 남으면 사용자는 원인을 못 본다
        self.note = QLabel("")
        self.note.setObjectName("Warn")
        self.note.setWordWrap(True)
        self.note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.note.hide()
        notes.addWidget(self.note)
        # 파라미터 재검증 오류 — 값은 세션 값으로 되돌아가고 사유만 여기 남는다
        self.error = QLabel("")
        self.error.setObjectName("Bad")
        self.error.setWordWrap(True)
        self.error.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.error.hide()
        notes.addWidget(self.error)
        notes.addStretch(1)
        body.addLayout(notes, 1)
        lay.addLayout(body)

        self.form = ParamForm()
        self.form.value_changed.connect(lambda n, v: self.field_changed.emit(self.stage, n, v))
        lay.addWidget(self.form)

        # 배치 카드 = ROI 하위 스테이지 포함 (레시피 `placement.roi`, 실행은 이미지당 1회 별도 스테이지)
        self.roi_method: QComboBox | None = None
        self.roi_form: ParamForm | None = None
        if stage == "placement":
            sub = QHBoxLayout()
            sub.setSpacing(8)
            sub_title = QLabel("배치 허용 영역 ROI")
            sub_title.setObjectName("H4")
            sub.addWidget(sub_title, 1)
            self.roi_method = QComboBox()
            self.roi_method.setMinimumWidth(96)
            sub.addWidget(self.roi_method)
            lay.addLayout(sub)
            self.roi_form = ParamForm()
            self.roi_form.value_changed.connect(lambda n, v: self.field_changed.emit("roi", n, v))
            lay.addWidget(self.roi_form)

        self._fill_methods(self.method, self.stage)
        self.method.currentIndexChanged.connect(
            lambda i: self._on_method(self.method, self.stage, i)
        )
        if self.roi_method is not None:
            self._fill_methods(self.roi_method, "roi")
            self.roi_method.currentIndexChanged.connect(
                lambda i: self._on_method(self.roi_method, "roi", i)  # type: ignore[arg-type]
            )

    @staticmethod
    def _fill_methods(combo: QComboBox, stage: str) -> None:
        model = QStandardItemModel(combo)
        for info in registry.list_methods(stage):
            item = QStandardItem(info.method)
            item.setData(info.method, Qt.ItemDataRole.UserRole)
            if not info.usable:
                item.setEnabled(False)
                item.setToolTip(info.reason or "미구현")
                item.setText(f"{info.method} — {info.reason or '미구현'}")
            model.appendRow(item)
        combo.setModel(model)

    def _on_method(self, combo: QComboBox, stage: str, i: int) -> None:
        m = combo.itemData(i, Qt.ItemDataRole.UserRole)
        if m:
            self.method_changed.emit(stage, str(m))

    @staticmethod
    def _select(combo: QComboBox, method: str) -> None:
        with QSignalBlocker(combo):
            for i in range(combo.count()):
                if combo.itemData(i, Qt.ItemDataRole.UserRole) == method:
                    combo.setCurrentIndex(i)
                    break

    def sync(self, cfg: Any) -> None:
        """세션 → 카드. method 콤보·폼(스펙은 스키마에서)·제목 툴팁(한 줄 요약)."""
        self._select(self.method, registry.config_method(cfg))
        self.form.set_specs(field_specs(cfg))
        self.title.setToolTip(params_text(cfg))
        if self.roi_method is not None and self.roi_form is not None:
            roi = cfg.roi
            self._select(self.roi_method, registry.config_method(roi))
            self.roi_form.set_specs(field_specs(roi))
            self.roi_method.setToolTip(params_text(roi))

    def set_error(self, text: str | None, field: str | None = None, *, roi: bool = False) -> None:
        """재검증 오류 한 줄(None 이면 해제) + 해당 필드 행 강조."""
        self.error.setText(text or "")
        self.error.setVisible(bool(text))
        self.form.set_error(None if roi else field)
        if self.roi_form is not None:
            self.roi_form.set_error(field if roi else None)

    def set_warnings(self, messages: Sequence[str]) -> None:
        """이 스테이지의 경고만(자기 접두 ``<stage>:`` 는 떼고). ROI 는 배치 카드의 하위 블록이라 ``roi:`` 경고는
        배치 카드에 접두를 남긴 채 보인다. 비면 숨긴다."""
        own = f"{self.stage}:"
        prefixes = (own, "roi:") if self.stage == "placement" else (own,)
        mine = [
            (m[len(own) :].strip() if m.startswith(own) else m.strip())
            for m in messages
            if m.startswith(prefixes)
        ]
        if not mine:
            self.note.hide()
            self.note.setText("")
            return
        self.note.setText("\n".join(f"⚠ {m}" for m in mine))
        self.note.setToolTip("\n".join(mine))
        self.note.show()

    def set_thumb(self, image: np.ndarray | None) -> None:
        if image is None:
            self.thumb.setPixmap(QPixmap())
            self.thumb.setText("–")
            return
        self.thumb.setText("")
        self.thumb.setPixmap(
            to_qpixmap(image).scaled(
                self.thumb.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class PipelinePanel(QScrollArea):
    method_changed = Signal(str, str)
    field_changed = Signal(str, str, object)  # stage("roi" 포함), field, value

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Pipe")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner.setObjectName("Pipe")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 10, 12, 24)
        lay.setSpacing(12)
        lay.addWidget(h4("파이프라인 · 7단계 Pipeline"))
        self.cards: dict[str, StageCard] = {}
        for i, (stage, title) in enumerate(STAGE_TITLES, start=1):
            card = StageCard(i, stage, title)
            card.method_changed.connect(self.method_changed.emit)
            card.field_changed.connect(self.field_changed.emit)
            self.cards[stage] = card
            lay.addWidget(card)
        lay.addStretch(1)
        self.setWidget(inner)

    def card_for(self, stage: str) -> StageCard:
        """``roi`` 는 배치 카드."""
        return self.cards["placement" if stage == "roi" else stage]

    def sync(self, recipe: R.Recipe) -> None:
        pipe = recipe.pipeline
        for stage, card in self.cards.items():
            card.sync(getattr(pipe, stage))

    def set_error(self, stage: str, field: str, text: str) -> None:
        self.clear_errors()
        self.card_for(stage).set_error(text, field, roi=(stage == "roi"))

    def clear_errors(self) -> None:
        for card in self.cards.values():
            card.set_error(None)

    def set_trace(self, steps: Sequence[TraceStep] | None, warnings: Sequence[str] = ()) -> None:
        for stage, card in self.cards.items():
            card.set_thumb(stage_thumbnail(stage, steps) if steps else None)
            card.set_warnings(warnings)

    def stage_warnings(self) -> dict[str, str]:
        """카드에 표시 중인 경고 {stage: text} — 테스트·상태 표시용."""
        return {s: c.note.text() for s, c in self.cards.items() if not c.note.isHidden()}

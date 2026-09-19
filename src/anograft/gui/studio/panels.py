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
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from anograft.core import recipe as R
from anograft.core import registry
from anograft.core.help import STAGE_HELP, method_help
from anograft.core.pipeline import TraceStep
from anograft.gui.qt_image import to_qpixmap
from anograft.gui.studio.param_form import ParamForm
from anograft.gui.studio.params import baseline_config, field_specs
from anograft.gui.studio.per_class import PerClassEditor
from anograft.gui.theme import COLORS
from anograft.preview import fit_long_side

STAGE_ORDER: tuple[str, ...] = (
    "source",
    "geometry",
    "placement",
    "blend",
    "harmonize",
    "degrade",
    "gtmask",
)
# (스테이지 키, 한국어 제목) — 문안은 core/help.STAGE_HELP 한 원천. 영어·YAML 키는 STAGE_TIPS 툴팁으로
STAGE_TITLES: tuple[tuple[str, str], ...] = tuple((k, STAGE_HELP[k].label) for k in STAGE_ORDER)
STAGE_TIPS: dict[str, str] = {
    k: f"{h.en} · YAML pipeline.{k} — {h.desc}" for k, h in STAGE_HELP.items() if k != "roi"
}
LONG_SIDES: tuple[tuple[str, int], ...] = (
    ("긴 변 1024px", 1024),
    ("긴 변 768px", 768),
    ("긴 변 512px", 512),
    ("원본 해상도", 0),
)


def _compact_combo(combo: QComboBox) -> None:
    """긴 한국어 표시명(경계 자연스럽게(Poisson))이 카드 최소 폭을 밀어 올리지 않게 — 내용 길이 대신 최소 8자, 남는 폭은 채운다."""
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(4)
    combo.setSizePolicy(
        QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
    )  # 남는 폭을 채우되 최소 폭은 고정
    combo.setMinimumWidth(96)


def preset_tip(name: str) -> str:
    """프리셋 콤보 항목 툴팁 — meta(제목 · 한 줄 · 이럴 때 · 피할 때)."""
    m = R.preset_meta(name)
    lines = [f"{name} — {m['title']}", m["summary"]]
    if m["use_for"]:
        lines.append(f"이럴 때: {m['use_for']}")
    if m["avoid"]:
        lines.append(f"피할 때: {m['avoid']}")
    return "\n".join(lines)


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
        for i, name in enumerate(R.preset_names()):
            self.preset.addItem(name, name)
            self.preset.setItemData(i, preset_tip(name), Qt.ItemDataRole.ToolTipRole)
        self.preset.currentIndexChanged.connect(
            lambda _i: self.preset_changed.emit(self.preset.currentData())
        )
        self.preset.setToolTip(
            "Preset — 7단계 설정을 묶어 둔 시작점. 고르면 아래 카드 값이 바뀝니다"
        )
        lay.addWidget(self._field("프리셋", self.preset))

        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.seed.setFixedWidth(96)
        self.seed.editingFinished.connect(lambda: self.seed_changed.emit(self.seed.value()))
        self.seed.setToolTip("Seed — 난수 시드. 같은 시드·같은 레시피면 항상 같은 결과가 나옵니다")
        lay.addWidget(self._field("난수 시드", self.seed))

        self.variants = QSpinBox()
        self.variants.setRange(1, 12)
        self.variants.setValue(6)
        self.variants.setFixedWidth(52)
        self.variants.valueChanged.connect(self.variants_changed.emit)
        self.variants.setToolTip("Variants — 같은 바탕 이미지를 시드만 바꿔 몇 장 미리 볼지")
        lay.addWidget(self._field("시드 변형", self.variants))

        self.long_side = QComboBox()
        for label, px in LONG_SIDES:
            self.long_side.addItem(label, px)
        self.long_side.currentIndexChanged.connect(
            lambda _i: self.long_side_changed.emit(int(self.long_side.currentData()))
        )
        self.long_side.setToolTip(
            "Preview size — 미리보기는 긴 변을 이만큼 줄여 합성합니다(빠름). 최종 출력은 항상 원본 해상도"
        )
        lay.addWidget(self._field("미리보기 크기", self.long_side))

        self.note = QLabel("")
        self.note.setObjectName("Muted")
        self.note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.note, 1)

        self.btn_open = QPushButton("레시피 열기")
        self.btn_open.setToolTip("Open recipe — 레시피(설정 파일, YAML) 열기")
        self.btn_save = QPushButton("레시피 저장")
        self.btn_save.setToolTip("Save recipe — 지금 값을 레시피(YAML)로 저장")
        self.btn_batch = QPushButton("일괄 생성으로 보내기")
        self.btn_batch.setToolTip("Send to batch — 이 레시피로 데이터셋을 한꺼번에 만들러 갑니다")
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
        if recipe.pipeline.preset:
            self.preset.setToolTip(preset_tip(recipe.pipeline.preset))
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
    um_per_px_changed = Signal(object)  # float | None — 대상 픽셀 피치(축척 정합, KNOWN-ISSUES #6)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Inputs")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 6)
        lay.setSpacing(6)
        lay.addWidget(h4("입력"))
        self.bank = QLineEdit()
        self.bank.setPlaceholderText("결함 보관함 폴더 (bank.yaml)")
        self.bank.setToolTip("Bank · inputs.bank — 결함 조각을 모아 둔 폴더")
        self.targets = QLineEdit()
        self.targets.setPlaceholderText("바탕(정상) 이미지 폴더 또는 목록 .txt")
        self.targets.setToolTip(
            "Targets · inputs.targets — 결함을 붙일 정상 이미지 폴더, 또는 경로 목록 .txt"
        )
        lay.addLayout(self._row("보관함", self.bank, ("폴더", self._pick_bank)))
        lay.addLayout(
            self._row(
                "바탕",
                self.targets,
                ("폴더", self._pick_targets_dir),
                ("목록", self._pick_targets_file),
            )
        )
        self.um = QDoubleSpinBox()
        self.um.setDecimals(3)
        self.um.setRange(0.0, 10000.0)
        self.um.setSingleStep(0.5)
        self.um.setSpecialValueText("모름")
        self.um.setToolTip(
            "Pixel size · inputs.um_per_px — 바탕 이미지 1픽셀이 몇 µm 인지. 결함 조각에도 값이 있으면 실제 크기로 자동 축척합니다. 0 = 모름(축척 끔)"
        )
        self.um.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        um_row = QHBoxLayout()
        um_row.setSpacing(5)
        um_lab = QLabel("µm/px")
        um_lab.setObjectName("Muted")
        um_lab.setFixedWidth(40)
        um_row.addWidget(um_lab)
        um_row.addWidget(self.um, 1)
        lay.addLayout(um_row)
        self.um.editingFinished.connect(
            lambda: self.um_per_px_changed.emit(self.um.value() or None)
        )
        self.btn_open = QPushButton("열기")
        self.btn_open.setToolTip("Open — 보관함과 바탕 이미지를 읽어 미리보기를 준비합니다")
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
        lab.setFixedWidth(40)
        h.addWidget(lab)
        h.addWidget(edit, 1)
        for text, slot in buttons:
            b = QPushButton(text)
            b.setFixedWidth(44)
            b.clicked.connect(slot)
            h.addWidget(b)
        return h

    def _pick_bank(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "결함 보관함 폴더", self.bank.text() or ".")
        if d:
            self.bank.setText(Path(d).as_posix())

    def _pick_targets_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "바탕(정상) 이미지 폴더", self.targets.text() or "."
        )
        if d:
            self.targets.setText(Path(d).as_posix())

    def _pick_targets_file(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "바탕(정상) 이미지 목록", self.targets.text() or ".", "목록 (*.txt)"
        )
        if f:
            self.targets.setText(Path(f).as_posix())

    def sync(self, recipe: R.Recipe, summary: str) -> None:
        self.bank.setText(recipe.inputs.bank_key())  # 빈칸 = 은행 없음(self-cut·perlin)
        self.targets.setText(recipe.inputs.targets.as_posix())
        self.um.blockSignals(True)
        self.um.setValue(float(recipe.inputs.um_per_px or 0.0))
        self.um.blockSignals(False)
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
        self.title = h4("바탕 이미지 · 0장")
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
        self.title.setText(f"바탕 이미지 · {len(self._paths)}장")

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
    fix_requested = Signal(str)  # stage — 경고 옆 '고치기' 버튼(탭이 무엇을 고칠지 안다)

    def __init__(self, no: int, stage: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StageCard")
        self.stage = stage
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(6)
        num = QLabel(str(no))
        num.setObjectName("StageNo")
        head.addWidget(num)
        self.title = QLabel(title)
        self.title.setObjectName("StageTitle")
        self.title.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )  # 헤더 폭은 콤보가 채운다
        self.title.setMinimumWidth(60)
        head.addWidget(self.title, 1)
        # 바뀜 n · 프리셋 값으로 — 폼(과 ROI 폼)의 바뀐 행 수. 0 이면 숨김
        self.modified_label = QLabel("")
        self.modified_label.setObjectName("Modified")
        self.modified_label.hide()
        head.addWidget(self.modified_label)
        self.btn_reset_all = QToolButton()
        self.btn_reset_all.setObjectName("ResetAll")
        self.btn_reset_all.setText("↺")
        self.btn_reset_all.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_reset_all.setFixedSize(22, 18)
        self.btn_reset_all.setToolTip(
            "이 단계의 바뀐 값을 전부 프리셋 값으로 되돌립니다 · Reset stage to preset"
        )
        self.btn_reset_all.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset_all.clicked.connect(self.reset_all)
        self.btn_reset_all.hide()
        head.addWidget(self.btn_reset_all)
        # 켬/끔 — `none` method 가 있는 단계(색·밝기 맞추기 · 카메라 효과)만. 끄면 none, 켜면 마지막 method 로
        self.enabled_toggle: QCheckBox | None = None
        self._last_method: str | None = None
        if "none" in registry.schema_methods(stage):
            self.enabled_toggle = QCheckBox()
            self.enabled_toggle.setToolTip("이 단계를 켜고 끕니다(끄면 method none) · Enable stage")
            self.enabled_toggle.toggled.connect(self._on_enabled_toggled)
            head.addWidget(self.enabled_toggle)
        self.method = QComboBox()
        _compact_combo(self.method)
        head.addWidget(self.method, 1)
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
        # 경고를 한 번에 고치는 버튼(탭이 라벨과 동작을 정한다 — 예: 조명 의존 클래스만 per_class 로) · 정보 한 줄(per_class 등)
        self.fix = QPushButton("")
        self.fix.setObjectName("Fix")
        self.fix.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )  # 긴 라벨이 카드 폭을 넘기지 않게
        self.fix.hide()
        self.fix.clicked.connect(lambda: self.fix_requested.emit(self.stage))
        notes.addWidget(self.fix)
        self.info = QLabel("")
        self.info.setObjectName("Muted")
        self.info.setWordWrap(True)
        self.info.hide()
        notes.addWidget(self.info)
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
        self.form.modified_count_changed.connect(lambda _n: self._refresh_modified())
        lay.addWidget(self.form)

        # 기하 카드 = per_class 편집 표(폼이 못 그리는 dict) — 변경은 field_changed("geometry", "per_class", dict)
        self.per_class: PerClassEditor | None = None
        if stage == "geometry":
            self.per_class = PerClassEditor()
            self.per_class.changed.connect(
                lambda d: self.field_changed.emit("geometry", "per_class", d)
            )
            lay.addWidget(self.per_class)

        # 배치 카드 = ROI 하위 스테이지 포함 (레시피 `placement.roi`, 실행은 이미지당 1회 별도 스테이지)
        self.roi_method: QComboBox | None = None
        self.roi_form: ParamForm | None = None
        if stage == "placement":
            sub = QHBoxLayout()
            sub.setSpacing(8)
            sub_title = QLabel("붙일 수 있는 영역")
            sub_title.setObjectName("H4")
            sub_title.setToolTip(f"ROI · placement.roi — {STAGE_HELP['roi'].desc}")
            sub_title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            sub_title.setMinimumWidth(60)
            sub.addWidget(sub_title, 1)
            self.roi_method = QComboBox()
            _compact_combo(self.roi_method)
            sub.addWidget(self.roi_method, 2)
            lay.addLayout(sub)
            self.roi_form = ParamForm()
            self.roi_form.modified_count_changed.connect(lambda _n: self._refresh_modified())
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
            h = method_help(stage, info.method)
            item = QStandardItem(h.label if h else info.method)
            item.setData(info.method, Qt.ItemDataRole.UserRole)
            if h:
                tip = f"{info.method} · {h.en}\n{h.summary}" + (
                    f"\n이럴 때: {h.when}" if h.when else ""
                )
                item.setToolTip(tip)
            if not info.usable:
                item.setEnabled(False)
                item.setToolTip(info.reason or "아직 구현되지 않음")
                item.setText(f"{info.method} — {info.reason or '아직 구현되지 않음'}")
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

    def sync(self, cfg: Any, recipe: R.Recipe | None = None) -> None:
        """세션 → 카드. method 콤보·폼(스펙은 스키마에서, 기준값은 레시피 프리셋)·제목 툴팁(한 줄 요약)."""
        method = registry.config_method(cfg)
        self._select(self.method, method)
        if self.enabled_toggle is not None:
            with QSignalBlocker(self.enabled_toggle):
                self.enabled_toggle.setChecked(method != "none")
            if method != "none":
                self._last_method = method
        base = baseline_config(recipe, self.stage, cfg) if recipe is not None else None
        self.form.set_specs(field_specs(cfg, baseline=base))
        self.title.setToolTip(f"{STAGE_TIPS[self.stage]}\n{params_text(cfg)}")
        self.method.setToolTip(self._method_tip(self.stage, registry.config_method(cfg)))
        if self.per_class is not None:
            self.per_class.set_value(dict(getattr(cfg, "per_class", {}) or {}))
        if self.roi_method is not None and self.roi_form is not None:
            roi = cfg.roi
            self._select(self.roi_method, registry.config_method(roi))
            roi_base = baseline_config(recipe, "roi", roi) if recipe is not None else None
            self.roi_form.set_specs(field_specs(roi, baseline=roi_base))
            self.roi_method.setToolTip(
                f"{self._method_tip('roi', registry.config_method(roi))}\n{params_text(roi)}"
            )

    def modified_count(self) -> int:
        n = len(self.form.modified_names())
        if self.roi_form is not None:
            n += len(self.roi_form.modified_names())
        return n

    def _refresh_modified(self) -> None:
        n = self.modified_count()
        self.modified_label.setText(f"바뀜 {n}" if n else "")
        self.modified_label.setVisible(n > 0)
        self.btn_reset_all.setVisible(n > 0)

    def reset_all(self) -> None:
        """이 카드의 바뀐 행 전부 프리셋 값으로(행마다 field_changed)."""
        self.form.reset_all()
        if self.roi_form is not None:
            self.roi_form.reset_all()

    def _on_enabled_toggled(self, on: bool) -> None:
        if on:
            target = self._last_method or next(
                (
                    i.method
                    for i in registry.list_methods(self.stage)
                    if i.usable and i.method != "none"
                ),
                None,
            )
            if target:
                self.method_changed.emit(self.stage, target)
        else:
            self.method_changed.emit(self.stage, "none")

    @staticmethod
    def _method_tip(stage: str, method: str) -> str:
        h = method_help(stage, method)
        if h is None:
            return method
        return f"{method} · {h.en}\n{h.summary}" + (f"\n이럴 때: {h.when}" if h.when else "")

    def set_fix(self, label: str | None) -> None:
        """경고 옆 '고치기' 버튼 — None 이면 숨김."""
        self.fix.setText(label or "")
        self.fix.setVisible(bool(label))

    def set_info(self, text: str | None) -> None:
        """카드 아래 정보 한 줄(경고 아님) — None/빈 문자열이면 숨김."""
        self.info.setText(text or "")
        self.info.setVisible(bool(text))

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
    fix_requested = Signal(str)  # stage

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
        lay.addWidget(h4("7단계 파이프라인"))
        self.cards: dict[str, StageCard] = {}
        for i, (stage, title) in enumerate(STAGE_TITLES, start=1):
            card = StageCard(i, stage, title)
            card.title.setToolTip(STAGE_TIPS[stage])
            card.method_changed.connect(self.method_changed.emit)
            card.field_changed.connect(self.field_changed.emit)
            card.fix_requested.connect(self.fix_requested.emit)
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
            card.sync(getattr(pipe, stage), recipe)
        geo = self.cards["geometry"]
        # 표가 있으면(클래스를 알 때) 정보 줄은 숨기고, 아직 은행을 모르면 한 줄 요약만
        geo.set_info(
            ""
            if geo.per_class is not None and geo.per_class.isVisible()
            else per_class_text(pipe.geometry)
        )

    def set_classes(self, classes: list[str]) -> None:
        """준비된 은행의 클래스 → 기하 카드 per_class 표의 행."""
        geo = self.cards["geometry"]
        if geo.per_class is not None:
            geo.per_class.set_classes(classes)

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


def per_class_text(geo: Any) -> str:
    """기하 카드 정보 줄 — `geometry.per_class`(카드 폼엔 안 나오는 dict) 를 한 줄로. 없으면 빈 문자열."""
    per = getattr(geo, "per_class", None) or {}
    parts = []
    for cls, o in per.items():
        bits = []
        if o.rotate is not None:
            bits.append(f"회전 {o.rotate[0]:g}~{o.rotate[1]:g}°")
        if o.flip is not None:
            bits.append("뒤집기 " + ("켬" if o.flip else "끔"))
        if o.scale is not None:
            bits.append(f"크기 {o.scale[0]:g}~{o.scale[1]:g}")
        parts.append(f"{cls}: {' · '.join(bits) or '(변경 없음)'}")
    return ("클래스별 예외(per_class) — " + " / ".join(parts)) if parts else ""

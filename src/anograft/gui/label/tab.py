"""``LabelTab`` — 라벨 탭 조립·배선 (목업 ``.labwrap``): 왼쪽 도구열 + 이미지 목록 · 가운데 ``LabelCanvas`` + 오버레이 바 ·
오른쪽 결함 정보(클래스·픽셀 피치·태그)·마스크 통계·은행에 저장.

흐름: 폴더 열기 → 목록에서 결함 사진 선택 → 브러시/폴리곤/자동 선택으로 마스크 → 클래스 입력 → **은행에 저장**(``LabelSession.save_to_bank``
— 임포터와 같은 화폐) → 스튜디오가 같은 은행을 쓰고 있으면 ``bank_saved`` 시그널로 다시 준비한다. 저장 뒤 마스크는 비운다
(Ctrl+Z 로 복구 가능) — 같은 사진의 다음 결함/클래스를 이어서 라벨링.

단축키: B 브러시 · E 지우개 · P 폴리곤 · A 자동 선택 · H 팬 · [ ] 브러시 크기 · Ctrl+Z/Y · Enter 폴리곤 확정 · Esc 취소 ·
Ctrl+S 은행에 저장 · PageDown/PageUp 다음/이전 이미지.

(v0.6) **YOLO 초안** — 이미지를 열 때 옆의 ``labels/<stem>.txt``(+ ``data.yaml`` names)를 찾으면 오른쪽 "YOLO 초안" 에 박스·폴리곤
수와 클래스가 뜨고, "열 때 자동 채우기" 가 켜져 있으면 첫 클래스의 초안으로 마스크를 미리 채운다(클래스 콤보 동기). 목록에서
라벨이 있는 이미지는 ``▸`` 표시. **ROI 모드** — 오른쪽 "저장 대상" 을 ROI 로 바꾸면 은행 대신 ``<mask_dir>/<stem>.png``
(``placement.roi: mask_dir`` 형식) 로 저장하고, 이미지를 열 때 같은 이름의 ROI 가 있으면 불러온다. 정상 이미지 폴더를 열어
"결함이 생겨도 되는 면" 을 칠한다.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from anograft.bank.bank import BANK_FILE, read_bank_meta
from anograft.bank.mask_from_box import METHODS as AUTO_METHODS
from anograft.gui.label.canvas import LabelCanvas
from anograft.gui.label.session import (
    LabelError,
    LabelSession,
    find_yolo_label,
    roi_png_path,
)
from anograft.gui.studio.panels import h4
from anograft.io import imgio

TOOL_BUTTONS: tuple[tuple[str, str, str], ...] = (
    ("brush", "✎", "브러시 (B) — 왼쪽 드래그로 칠하기"),
    ("eraser", "⌫", "지우개 (E)"),
    ("polygon", "⬡", "폴리곤 (P) — 클릭으로 점, Enter/더블클릭 확정, Esc 취소"),
    ("auto", "✧", "자동 선택 (A) — 결함 둘레로 박스를 끌면 마스크 추정(GrabCut/Otsu)"),
    ("pan", "✋", "이동 (H) — 드래그로 이동, 휠로 확대. 가운데 버튼·Space 는 어느 도구에서나"),
)


MODE_BANK = "bank"
MODE_ROI = "roi"
LABELED_MARK = "▸ "  # 목록에서 YOLO 라벨이 있는 이미지 표시


class LabelTab(QWidget):
    status = Signal(str)
    bank_saved = Signal(str)  # 은행 루트(posix) — 스튜디오가 같은 은행이면 다시 준비
    roi_saved = Signal(str)  # 저장된 ROI PNG(posix)

    def __init__(self, session: LabelSession | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session or LabelSession()
        self.folder: Path | None = None
        self.files: list[Path] = []
        self.confirm_discard: Callable[[], bool] | None = self._ask_discard  # 테스트는 None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._strip())

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._left())
        split.addWidget(self._center())
        split.addWidget(self._right())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([240, 900, 320])
        root.addWidget(split, 1)
        self._wire()
        self.canvas.set_session(self.session)
        self.refresh()

    # ------------------------------------------------------------------ 조립

    def _strip(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("Strip")
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 7, 14, 7)
        h.setSpacing(10)
        self.btn_folder = QPushButton("폴더 열기 Folder")
        self.btn_image = QPushButton("이미지 열기 Image")
        self.btn_mask = QPushButton("마스크 PNG 불러오기")
        h.addWidget(self.btn_folder)
        h.addWidget(self.btn_image)
        h.addWidget(self.btn_mask)
        lab = QLabel("은행 Bank")
        lab.setObjectName("Muted")
        h.addWidget(lab)
        self.bank_edit = QLineEdit()
        self.bank_edit.setPlaceholderText("저장할 결함 은행 폴더 (없으면 새로 만든다)")
        self.bank_edit.setMinimumWidth(240)
        self.btn_bank = QPushButton("폴더")
        self.btn_bank.setFixedWidth(44)
        h.addWidget(self.bank_edit, 1)
        h.addWidget(self.btn_bank)
        self.note = QLabel("")
        self.note.setObjectName("Muted")
        self.note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        h.addWidget(self.note, 1)
        return bar

    def _left(self) -> QWidget:
        left = QWidget()
        left.setObjectName("Rail")
        h = QHBoxLayout(left)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        tools = QWidget()
        tools.setObjectName("Inputs")
        tv = QVBoxLayout(tools)
        tv.setContentsMargins(6, 10, 6, 10)
        tv.setSpacing(6)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_buttons: dict[str, QPushButton] = {}
        for key, glyph, tip in TOOL_BUTTONS:
            b = QPushButton(glyph)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setFixedSize(34, 34)
            self.tool_group.addButton(b)
            self.tool_buttons[key] = b
            tv.addWidget(b)
        self.tool_buttons["brush"].setChecked(True)
        tv.addSpacing(8)
        self.btn_dilate = QPushButton("＋")
        self.btn_dilate.setToolTip("팽창 1px")
        self.btn_erode = QPushButton("－")
        self.btn_erode.setToolTip("침식 1px")
        self.btn_undo = QPushButton("↶")
        self.btn_undo.setToolTip("되돌리기 (Ctrl+Z)")
        self.btn_redo = QPushButton("↷")
        self.btn_redo.setToolTip("다시하기 (Ctrl+Y)")
        self.btn_clear = QPushButton("✕")
        self.btn_clear.setToolTip("마스크 비우기")
        for b in (self.btn_dilate, self.btn_erode, self.btn_undo, self.btn_redo, self.btn_clear):
            b.setFixedSize(34, 34)
            tv.addWidget(b)
        tv.addStretch(1)
        h.addWidget(tools)
        listwrap = QWidget()
        lv = QVBoxLayout(listwrap)
        lv.setContentsMargins(8, 10, 8, 8)
        lv.setSpacing(6)
        self.list_title = h4("결함 이미지 · 0장")
        lv.addWidget(self.list_title)
        self.list = QListWidget()
        lv.addWidget(self.list, 1)
        h.addWidget(listwrap, 1)
        left.setMinimumWidth(220)
        return left

    def _center(self) -> QWidget:
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        self.canvas = LabelCanvas()
        ml.addWidget(self.canvas, 1)
        bar = QWidget()
        bar.setObjectName("Strip")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 5, 14, 5)
        bl.setSpacing(10)
        self.cb_mask = QCheckBox("마스크 표시")
        self.cb_mask.setChecked(True)
        bl.addWidget(self.cb_mask)
        bl.addWidget(self._muted("불투명도"))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(45)
        self.opacity.setFixedWidth(110)
        bl.addWidget(self.opacity)
        bl.addWidget(self._muted("브러시"))
        self.brush = QSlider(Qt.Orientation.Horizontal)
        self.brush.setRange(1, 200)
        self.brush.setValue(self.canvas.brush_px)
        self.brush.setFixedWidth(140)
        bl.addWidget(self.brush)
        self.brush_label = self._muted(f"{self.canvas.brush_px}px")
        self.brush_label.setFixedWidth(40)
        bl.addWidget(self.brush_label)
        bl.addWidget(self._muted("자동 선택"))
        self.auto_method = QComboBox()
        self.auto_method.addItems(list(AUTO_METHODS))
        bl.addWidget(self.auto_method)
        bl.addStretch(1)
        self.zoom_info = self._muted("")
        self.zoom_info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.zoom_info.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bl.addWidget(self.zoom_info, 1)
        ml.addWidget(bar)
        return mid

    def _right(self) -> QWidget:
        side = QWidget()
        side.setObjectName("Pipe")
        v = QVBoxLayout(side)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)
        v.addWidget(h4("저장 대상 Save as"))
        mode_row = QHBoxLayout()
        mode_row.setSpacing(10)
        self.rb_bank = QRadioButton("은행 소스 Bank")
        self.rb_bank.setToolTip("결함 마스크를 은행에 소스로 저장 (성분별 크롭 + 메타)")
        self.rb_roi = QRadioButton("ROI 마스크 ROI")
        self.rb_roi.setToolTip(
            "정상 이미지에서 결함이 생겨도 되는 면을 칠해 <mask_dir>/<stem>.png 로 저장 — "
            "레시피 placement.roi: {method: mask_dir, path: <mask_dir>} 가 읽는다"
        )
        self.rb_bank.setChecked(True)
        mode_row.addWidget(self.rb_bank)
        mode_row.addWidget(self.rb_roi)
        mode_row.addStretch(1)
        v.addLayout(mode_row)
        self.roi_box = QWidget()
        rv = QVBoxLayout(self.roi_box)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(6)
        dir_row = QHBoxLayout()
        dir_row.setSpacing(6)
        self.roi_dir = QLineEdit()
        self.roi_dir.setPlaceholderText("ROI 마스크 폴더 (mask_dir) — 비면 <이미지 폴더>/../roi")
        self.btn_roi_dir = QPushButton("폴더")
        self.btn_roi_dir.setFixedWidth(44)
        dir_row.addWidget(self._muted("mask_dir"))
        dir_row.addWidget(self.roi_dir, 1)
        dir_row.addWidget(self.btn_roi_dir)
        rv.addLayout(dir_row)
        self.roi_hint = self._muted("")
        self.roi_hint.setWordWrap(True)
        rv.addWidget(self.roi_hint)
        v.addWidget(self.roi_box)
        self.roi_box.hide()
        v.addSpacing(4)

        v.addWidget(h4("YOLO 초안 Draft"))
        self.draft_info = self._muted("없음 — labels/<stem>.txt 가 옆에 있으면 자동으로 찾는다")
        self.draft_info.setWordWrap(True)
        v.addWidget(self.draft_info)
        draft_row = QHBoxLayout()
        draft_row.setSpacing(6)
        self.draft_cls = QComboBox()
        self.draft_cls.setToolTip(
            "초안에 있는 클래스 — 고르면 그 클래스의 박스/폴리곤으로 마스크를 채운다"
        )
        self.btn_draft = QPushButton("채우기 Fill")
        self.btn_draft.setToolTip("현재 마스크를 초안으로 교체 (Ctrl+Z 로 복구)")
        draft_row.addWidget(self.draft_cls, 1)
        draft_row.addWidget(self.btn_draft)
        v.addLayout(draft_row)
        self.cb_draft_auto = QCheckBox("열 때 자동 채우기")
        self.cb_draft_auto.setChecked(True)
        self.cb_draft_auto.setToolTip(
            "이미지를 열 때 라벨이 있으면 첫 클래스 초안으로 마스크를 미리 채운다"
        )
        v.addWidget(self.cb_draft_auto)
        self._set_draft_enabled(False)
        v.addSpacing(4)

        self.defect_title = h4("결함 정보 Defect")
        v.addWidget(self.defect_title)
        self.cls = QComboBox()
        self.cls.setEditable(True)
        self.cls.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.cls.lineEdit().setPlaceholderText("클래스 이름 (예: scratch)")
        self.defect_box = QWidget()
        dv = QVBoxLayout(self.defect_box)
        dv.setContentsMargins(0, 0, 0, 0)
        dv.setSpacing(6)
        dv.addLayout(self._kv("클래스", self.cls))
        self.origin = self._muted("–")
        self.origin.setWordWrap(True)
        v.addLayout(self._kv("원본", self.origin))
        self.um = QDoubleSpinBox()
        self.um.setDecimals(3)
        self.um.setRange(0.0, 10000.0)
        self.um.setSingleStep(0.5)
        self.um.setSpecialValueText("모름")
        self.um.setToolTip(
            "픽셀 피치 µm/px — 은행 소스에 저장돼 대상과의 축척 정합에 쓰인다. 0 = 모름"
        )
        dv.addLayout(self._kv("µm/px", self.um))
        self.tags = QLineEdit()
        self.tags.setPlaceholderText("태그, 쉼표로 (예: 가공면, 직선형)")
        dv.addLayout(self._kv("태그", self.tags))
        self.cb_whole = QCheckBox("성분을 나누지 않고 하나로 저장")
        self.cb_whole.setToolTip(
            "끄면 떨어진 조각마다 소스 하나(기본). 한 결함이 여러 조각이면 켠다"
        )
        dv.addWidget(self.cb_whole)
        v.addWidget(self.defect_box)
        v.addSpacing(8)
        v.addWidget(h4("마스크 통계 Mask"))
        self.stats = QLabel("–")
        self.stats.setObjectName("Hint")
        self.stats.setWordWrap(True)
        v.addWidget(self.stats)
        v.addStretch(1)
        self.result = QLabel("")
        self.result.setObjectName("Muted")
        self.result.setWordWrap(True)
        v.addWidget(self.result)
        self.btn_save = QPushButton("은행에 저장 Save to bank  (Ctrl+S)")
        self.btn_save.setObjectName("Primary")
        v.addWidget(self.btn_save)
        side.setMinimumWidth(300)
        return side

    @staticmethod
    def _muted(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("Muted")
        return lab

    def _kv(self, key: str, widget: QWidget) -> QHBoxLayout:
        h = QHBoxLayout()
        h.setSpacing(6)
        lab = self._muted(key)
        lab.setFixedWidth(46)
        h.addWidget(lab)
        h.addWidget(widget, 1)
        return h

    # ------------------------------------------------------------------ 배선

    def _wire(self) -> None:
        c = self.canvas
        self.btn_folder.clicked.connect(self.open_folder_dialog)
        self.btn_image.clicked.connect(self.open_image_dialog)
        self.btn_mask.clicked.connect(self.open_mask_dialog)
        self.btn_bank.clicked.connect(self._pick_bank)
        self.bank_edit.editingFinished.connect(self._reload_classes)
        for key, b in self.tool_buttons.items():
            b.clicked.connect(lambda _=False, k=key: c.set_tool(k))
        c.tool_changed.connect(self._on_tool)
        self.btn_dilate.clicked.connect(lambda: c.apply(lambda s: s.dilate(1)))
        self.btn_erode.clicked.connect(lambda: c.apply(lambda s: s.erode(1)))
        self.btn_clear.clicked.connect(lambda: c.apply(lambda s: s.clear()))
        self.btn_undo.clicked.connect(c.undo)
        self.btn_redo.clicked.connect(c.redo)
        self.list.currentRowChanged.connect(self._on_row)
        self.cb_mask.toggled.connect(c.set_mask_visible)
        self.opacity.valueChanged.connect(lambda v: c.set_opacity(round(v * 2.55)))
        self.brush.valueChanged.connect(c.set_brush)
        c.brush_changed.connect(self._on_brush)
        self.auto_method.currentTextChanged.connect(lambda m: setattr(c, "auto_method", m))
        c.zoom_changed.connect(lambda _z: self._update_zoom_info())
        c.edited.connect(self.refresh)
        c.auto_done.connect(lambda m: self.status.emit(f"자동 선택: {m}"))
        c.error.connect(self._on_error)
        self.um.valueChanged.connect(lambda _v: self._refresh_stats())
        self.btn_save.clicked.connect(self.save)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save)
        self.rb_roi.toggled.connect(self._on_mode)
        self.btn_roi_dir.clicked.connect(self._pick_roi_dir)
        self.roi_dir.editingFinished.connect(self._refresh_roi_hint)
        self.btn_draft.clicked.connect(self.fill_draft)
        QShortcut(QKeySequence(Qt.Key.Key_PageDown), self, activated=lambda: self.step(1))
        QShortcut(QKeySequence(Qt.Key.Key_PageUp), self, activated=lambda: self.step(-1))

    # ------------------------------------------------------------------ 열기

    def _ask_discard(self) -> bool:
        r = QMessageBox.question(
            self,
            "마스크 버리기",
            "저장하지 않은 마스크가 있습니다. 버리고 다른 이미지를 열까요?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return r == QMessageBox.StandardButton.Yes

    def _may_discard(self) -> bool:
        if not self.session.dirty or not self.session.loaded:
            return True
        return self.confirm_discard() if self.confirm_discard is not None else True

    def open_folder(self, folder: str | Path) -> None:
        d = Path(folder)
        try:
            files = imgio.list_images(d)
        except OSError as e:
            self._on_error(f"폴더를 열 수 없습니다: {e}")
            return
        self.folder, self.files = d, files
        n_labeled = 0
        self.list.blockSignals(True)
        self.list.clear()
        for p in files:
            lp = find_yolo_label(p)
            labeled = lp is not None and lp.stat().st_size > 0  # 빈 라벨 파일 = 정상 이미지
            n_labeled += labeled
            self.list.addItem((LABELED_MARK if labeled else "") + p.name)
        self.list.blockSignals(False)
        title = f"{'정상' if self.mode == MODE_ROI else '결함'} 이미지 · {len(files)}장 — {d.name}"
        if n_labeled:
            title += f" · YOLO 라벨 {n_labeled}"
        self.list_title.setText(title)
        self.list_title.setToolTip(f"{d.as_posix()}\n{title}")
        self._refresh_roi_hint()
        if files:
            self.list.setCurrentRow(0)
        else:
            self.status.emit(f"이미지가 없습니다: {d.as_posix()}")

    def open_image(self, path: str | Path) -> bool:
        if not self._may_discard():
            return False
        try:
            self.session.load_image(path)
        except imgio.ImageReadError as e:
            self._on_error(str(e))
            return False
        self.canvas.set_session(self.session)
        self.origin.setText(Path(path).name)
        self.result.setText("")
        msg = f"열림: {Path(path).as_posix()} · {self.session.shape[1]}×{self.session.shape[0]}"
        extra = self._load_draft() if self.mode == MODE_BANK else self._load_roi()
        if extra:
            msg += " · " + extra
        self.refresh()
        self.status.emit(msg)
        return True

    # ------------------------------------------------------------------ YOLO 초안

    def _set_draft_enabled(self, on: bool) -> None:
        self.draft_cls.setEnabled(on)
        self.btn_draft.setEnabled(on)

    def _load_draft(self) -> str:
        """이미지 옆 라벨을 찾아 패널을 채우고, 자동이면 첫 클래스로 마스크를 미리 채운다. 반환 = 상태줄에 덧붙일 문구."""
        try:
            draft = self.session.load_yolo_draft()
        except LabelError as e:
            draft = None
            self._on_error(str(e))
        self.draft_cls.blockSignals(True)
        self.draft_cls.clear()
        if draft is None or not draft.items:
            self.draft_cls.blockSignals(False)
            self._set_draft_enabled(False)
            self.draft_info.setText(
                "없음 — labels/<stem>.txt 가 옆에 있으면 자동으로 찾는다"
                if draft is None
                else f"{draft.label_path.name}: 항목 없음"
                + (f" · {draft.warnings[0]}" if draft.warnings else "")
            )
            return ""
        for cid in draft.class_ids:
            self.draft_cls.addItem(draft.class_name(cid), cid)
        self.draft_cls.blockSignals(False)
        self._set_draft_enabled(True)
        info = draft.summary()
        if draft.names_path is not None:
            info += f" · names {draft.names_path.name}"
        if draft.warnings:
            info += f" · ⚠ {draft.warnings[0]}"
        self.draft_info.setText(info)
        if self.cb_draft_auto.isChecked():
            return self.fill_draft() or ""
        return f"YOLO 초안 {len(draft.items)}개 (채우기로 적용)"

    def fill_draft(self) -> str | None:
        """선택한 클래스의 초안으로 마스크를 교체하고 클래스 콤보를 맞춘다. 반환 = 상태 문구(실패 None)."""
        draft = self.session.draft
        if draft is None or self.draft_cls.count() == 0:
            self._on_error("YOLO 초안이 없습니다")
            return None
        cid = int(self.draft_cls.currentData())
        try:
            used = self.session.apply_draft(cid, self.auto_method.currentText())
        except LabelError as e:
            self._on_error(str(e))
            return None
        self.canvas.refresh_overlay()
        name = draft.class_name(cid)
        self.cls.setEditText(name)
        n = len(draft.items_for(cid))
        methods = ", ".join(sorted(set(used))) if used else "폴리곤"
        msg = f"YOLO 초안 채움: {name} {n}개 ({methods}) — 다듬은 뒤 저장"
        self.refresh()
        self.status.emit(msg)
        return msg

    # ------------------------------------------------------------------ ROI 모드

    @property
    def mode(self) -> str:
        return MODE_ROI if self.rb_roi.isChecked() else MODE_BANK

    def set_mode(self, mode: str) -> None:
        (self.rb_roi if mode == MODE_ROI else self.rb_bank).setChecked(True)

    def _on_mode(self, _checked: bool = False) -> None:
        roi = self.mode == MODE_ROI
        self.roi_box.setVisible(roi)
        self.defect_box.setVisible(not roi)
        self.defect_title.setVisible(not roi)
        self.btn_save.setText(
            "ROI 저장 Save ROI  (Ctrl+S)" if roi else "은행에 저장 Save to bank  (Ctrl+S)"
        )
        if self.folder is not None:
            self.list_title.setText(
                self.list_title.text().replace("결함 이미지", "정상 이미지")
                if roi
                else self.list_title.text().replace("정상 이미지", "결함 이미지")
            )
        self._refresh_roi_hint()
        if roi and self.session.loaded and not self.session.dirty:
            extra = self._load_roi()
            if extra:
                self.status.emit(extra)

    def roi_dir_path(self) -> Path | None:
        text = self.roi_dir.text().strip()
        if text:
            return Path(text)
        if self.folder is not None:
            return self.folder.parent / "roi"
        return None

    def _pick_roi_dir(self) -> None:
        start = self.roi_dir.text().strip() or (self.folder.as_posix() if self.folder else ".")
        d = QFileDialog.getExistingDirectory(self, "ROI 마스크 폴더 (mask_dir)", start)
        if d:
            self.roi_dir.setText(Path(d).as_posix())
            self._refresh_roi_hint()

    def _refresh_roi_hint(self) -> None:
        d = self.roi_dir_path()
        if d is None:
            self.roi_hint.setText("정상 이미지 폴더를 열면 기본 mask_dir 은 <폴더>/../roi")
            return
        n = len(list(d.glob("*.png"))) if d.is_dir() else 0
        self.roi_hint.setText(
            f"{d.as_posix()} — ROI {n}장"
            + (" (폴더가 만들어진다)" if not d.is_dir() else "")
            + f"\n레시피: placement.roi: {{method: mask_dir, path: {d.as_posix()}}}"
        )

    def _load_roi(self) -> str:
        """ROI 모드에서 이미지를 열 때 같은 이름의 ROI PNG 가 있으면 불러온다."""
        d = self.roi_dir_path()
        if d is None or self.session.path is None:
            return ""
        p = roi_png_path(d, self.session.path)
        if not p.is_file():
            return "ROI 없음 — 결함이 생겨도 되는 면을 칠한다"
        try:
            self.session.load_mask(p)
        except (LabelError, imgio.ImageReadError) as e:
            self._on_error(f"ROI 불러오기 실패: {e}")
            return ""
        self.session.dirty = False
        self.canvas.refresh_overlay()
        return f"ROI 불러옴: {p.name}"

    def save_roi(self) -> bool:
        d = self.roi_dir_path()
        if d is None or self.session.path is None:
            self._on_error("정상 이미지를 먼저 여세요")
            return False
        out = roi_png_path(d, self.session.path)
        try:
            self.session.save_roi_png(out)
        except (LabelError, OSError) as e:
            self._on_error(str(e))
            return False
        msg = f"ROI 저장: {out.as_posix()}"
        self.result.setText(msg)
        self.status.emit(msg)
        self._refresh_roi_hint()
        self.roi_saved.emit(out.as_posix())
        return True

    def save(self) -> bool:
        return self.save_roi() if self.mode == MODE_ROI else self.save_to_bank()

    def step(self, delta: int) -> None:
        if not self.files:
            return
        row = self.list.currentRow()
        new = max(0, min(len(self.files) - 1, row + delta))
        if new != row:
            self.list.setCurrentRow(new)

    def _on_row(self, row: int) -> None:
        if 0 <= row < len(self.files) and not self.open_image(self.files[row]):
            # 버리지 않기로 했으면 이전 행으로 되돌린다
            prev = self.files.index(self.session.path) if self.session.path in self.files else -1
            self.list.blockSignals(True)
            self.list.setCurrentRow(prev)
            self.list.blockSignals(False)

    def open_folder_dialog(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "결함 이미지 폴더", self.folder.as_posix() if self.folder else "."
        )
        if d:
            self.open_folder(d)

    def open_image_dialog(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self,
            "결함 이미지",
            self.folder.as_posix() if self.folder else ".",
            "이미지 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if f:
            self.open_image(f)

    def open_mask_dialog(self) -> None:
        if not self.session.loaded:
            self._on_error("이미지를 먼저 여세요")
            return
        f, _ = QFileDialog.getOpenFileName(self, "마스크 PNG", ".", "마스크 (*.png)")
        if f:
            self.canvas.apply(lambda s: s.load_mask(f))

    def _pick_bank(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "결함 은행 폴더", self.bank_edit.text() or ".")
        if d:
            self.set_bank(d)

    def set_bank(self, root: str) -> None:
        self.bank_edit.setText(Path(root).as_posix() if root else "")
        self._reload_classes()

    def _reload_classes(self) -> None:
        root = self.bank_edit.text().strip()
        current = self.cls.currentText()
        classes: list[str] = []
        if root:
            meta = Path(root) / BANK_FILE
            if meta.is_file():
                try:
                    classes = [str(c) for c in read_bank_meta(meta).get("classes", [])]
                except (
                    Exception
                ) as e:  # 깨진 bank.yaml 은 안내만 — 저장 때 BankWriter 가 다시 판단한다
                    self._on_error(f"bank.yaml 읽기 실패: {e}")
        self.cls.blockSignals(True)
        self.cls.clear()
        self.cls.addItems(classes)
        self.cls.setEditText(current)
        self.cls.blockSignals(False)
        self.note.setText(
            f"은행 클래스 {len(classes)}: {', '.join(classes)}"
            if classes
            else ("새 은행(폴더가 만들어진다)" if root else "")
        )

    # ------------------------------------------------------------------ 저장

    def save_to_bank(self) -> bool:
        root = self.bank_edit.text().strip()
        if not root:
            self._on_error("은행 폴더를 지정하세요 (상단 은행 칸)")
            return False
        tags = [t for t in self.tags.text().split(",") if t.strip()]
        um = self.um.value() or None
        try:
            added, warns = self.session.save_to_bank(
                root,
                self.cls.currentText(),
                tags=tags,
                um_per_px=um,
                keep_whole=self.cb_whole.isChecked(),
            )
        except LabelError as e:
            self._on_error(str(e))
            return False
        ids = ", ".join(f"{a.cls}/{a.source_id}" for a in added)
        msg = (
            f"은행 저장 {len(added)}개: {ids}"
            if added
            else "저장된 소스 없음(성분이 전부 min_area 미만)"
        )
        if warns:
            msg += " · " + " · ".join(warns)
        self.result.setText(msg)
        self.status.emit(msg)
        if added:
            self.canvas.apply(lambda s: s.clear())  # 다음 결함을 이어서 — Ctrl+Z 로 복구
            self.session.dirty = False
        self._reload_classes()
        self.bank_saved.emit(Path(root).as_posix())
        return bool(added)

    # ------------------------------------------------------------------ 표시

    def _on_tool(self, tool: str) -> None:
        b = self.tool_buttons.get(tool)
        if b is not None and not b.isChecked():
            b.setChecked(True)
        self._update_zoom_info()

    def _on_brush(self, px: int) -> None:
        self.brush.blockSignals(True)
        self.brush.setValue(px)
        self.brush.blockSignals(False)
        self.brush_label.setText(f"{px}px")
        self._update_zoom_info()

    def _on_error(self, msg: str) -> None:
        self.result.setText(msg)
        self.status.emit(msg)

    def _update_zoom_info(self) -> None:
        c = self.canvas
        self.zoom_info.setText(f"{c.tool} · 브러시 {c.brush_px}px · {c.zoom * 100:.0f}%")

    def _refresh_stats(self) -> None:
        if not self.session.loaded:
            self.stats.setText("–")
            return
        st = self.session.stats(self.um.value() or None)
        if st.area_px == 0:
            self.stats.setText("마스크 없음 — 브러시로 칠하거나 자동 선택(A)으로 박스를 끄세요")
            return
        parts = [
            f"면적 {st.area_px:,} px · {st.area_ratio * 100:.2f}%",
            f"성분 {st.n_components}개"
            + (
                ""
                if self.mode == MODE_ROI
                else (" → 소스 하나로 저장" if self.cb_whole.isChecked() else " → 소스 각각")
            ),
            f"길이 {st.length_px:.0f} px"
            + (f" · {st.length_um / 1000:.2f} mm" if st.length_um else ""),
        ]
        if st.bbox:
            x, y, w, h = st.bbox
            parts.append(f"bbox {x},{y} {w}×{h}")
        if st.contrast is not None:
            parts.append(
                f"평균 대비 {st.contrast:+.0f} ({'배경보다 어두움' if st.contrast < 0 else '배경보다 밝음'})"
            )
        self.stats.setText("\n".join(parts))

    def refresh(self) -> None:
        self.btn_undo.setEnabled(self.session.can_undo)
        self.btn_redo.setEnabled(self.session.can_redo)
        self.btn_save.setEnabled(self.session.loaded)
        self._refresh_stats()
        self._update_zoom_info()

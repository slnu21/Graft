"""``BankTab`` — 은행 탭(v0.7): 상단 은행 경로·열기·새로고침·요약 · 왼쪽 필터(클래스·태그·저신뢰만·추정만·검색·정렬) ·
가운데 소스 타일 그리드(``preview.source_tile`` — 마스크 윤곽, 추정 amber, 저신뢰 빨간 테두리) · 오른쪽 상세(크롭+마스크 오버레이
확대, 메타) + **삭제** / **라벨 탭에서 다듬기**.

- 파일을 바꾸는 것은 ``BankSession.delete``/``replace_mask``(→ ``BankWriter``)뿐이고, 바뀌면 ``bank_changed(root)`` 를 내보내
  같은 은행을 쓰는 스튜디오가 다시 준비한다.
- "라벨 탭에서 다듬기"는 ``edit_requested(root, source_id)`` — 메인 창이 라벨 탭을 **은행 소스 편집 모드**로 열고(크롭 + 현재
  마스크), 거기서 저장하면 같은 id 에 덮어쓴다(``LabelTab.begin_bank_edit`` → ``source_updated``).
- 그리드는 필터 결과만 다시 그린다(타일은 소스 id 로 캐시). 4K 은행이 아니라 크롭 은행이라 타일 수백 장도 즉시.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from anograft.core.channels import promote_to_bgr
from anograft.gui.bank.session import SORT_KEYS, BankSession, BankSessionError, SourceRow
from anograft.gui.qt_image import to_qpixmap
from anograft.gui.studio.panels import flat_icon, h4
from anograft.preview import GT_EDGE, source_tile

TILE = 128
ALL = "(전체)"
SORT_LABELS: dict[str, str] = {
    "id": "id",
    "confidence": "신뢰도 confidence",
    "area": "면적 area",
    "class": "클래스 class",
}


def detail_image(image: np.ndarray, mask: np.ndarray, *, long_side: int = 360) -> np.ndarray:
    """상세 패널용 — 크롭을 긴 변 ``long_side`` 로 키우고(NEAREST, 크롭은 작다) 마스크 윤곽(초록) + 반투명 채움."""
    img = promote_to_bgr(image).copy()
    h, w = img.shape[:2]
    s = long_side / float(max(h, w))
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
    big = cv2.resize(img, (nw, nh), interpolation=interp)
    m = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
    tint = big.copy()
    tint[m] = (0.55 * tint[m] + 0.45 * np.array([60, 60, 230], dtype=np.float32)).astype(np.uint8)
    out = tint
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, GT_EDGE, 1)
    return out


class BankTab(QWidget):
    status = Signal(str)
    bank_changed = Signal(str)  # 은행 루트(posix) — 삭제·마스크 교체 뒤
    edit_requested = Signal(str, str)  # (은행 루트 posix, source id) — 라벨 탭에서 다듬기

    def __init__(self, session: BankSession | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session or BankSession()
        self.confirm_delete: Callable[[list[str]], bool] | None = self._ask_delete  # 테스트는 None
        self._tiles: dict[str, QPixmap] = {}
        self._rows: list[SourceRow] = []

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
        split.setSizes([240, 900, 380])
        root.addWidget(split, 1)
        self._wire()
        self.refresh()

    # ------------------------------------------------------------------ 조립

    def _strip(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("Strip")
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 7, 14, 7)
        h.setSpacing(10)
        lab = QLabel("은행 Bank")
        lab.setObjectName("Muted")
        h.addWidget(lab)
        self.bank_edit = QLineEdit()
        self.bank_edit.setPlaceholderText("결함 은행 폴더 (bank.yaml 이 있는 곳)")
        self.bank_edit.setMinimumWidth(260)
        self.btn_pick = QPushButton("폴더")
        self.btn_pick.setFixedWidth(44)
        self.btn_open = QPushButton("열기 Open")
        self.btn_reload = QPushButton("새로고침")
        h.addWidget(self.bank_edit, 1)
        h.addWidget(self.btn_pick)
        h.addWidget(self.btn_open)
        h.addWidget(self.btn_reload)
        self.summary = QLabel("")
        self.summary.setObjectName("Muted")
        self.summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        h.addWidget(self.summary, 1)
        return bar

    def _left(self) -> QWidget:
        left = QWidget()
        left.setObjectName("Inputs")
        v = QVBoxLayout(left)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)
        v.addWidget(h4("필터 Filter"))
        self.f_cls = QComboBox()
        self.f_tag = QComboBox()
        self.f_text = QLineEdit()
        self.f_text.setPlaceholderText("id · 원본 · 태그 검색")
        self.f_low = QCheckBox("저신뢰만 (confidence < 0.5)")
        self.f_est = QCheckBox("추정 마스크만 (yolo-box:*)")
        v.addLayout(self._kv("클래스", self.f_cls))
        v.addLayout(self._kv("태그", self.f_tag))
        v.addLayout(self._kv("검색", self.f_text))
        v.addWidget(self.f_low)
        v.addWidget(self.f_est)
        v.addSpacing(8)
        v.addWidget(h4("정렬 Sort"))
        self.sort = QComboBox()
        for k in SORT_KEYS:
            self.sort.addItem(SORT_LABELS[k], k)
        self.sort_desc = QCheckBox("내림차순")
        v.addWidget(self.sort)
        v.addWidget(self.sort_desc)
        v.addStretch(1)
        self.count = QLabel("")
        self.count.setObjectName("Hint")
        self.count.setWordWrap(True)
        v.addWidget(self.count)
        left.setMinimumWidth(220)
        return left

    def _center(self) -> QWidget:
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        self.grid = QListWidget()
        self.grid.setObjectName("BankGrid")
        self.grid.setViewMode(QListWidget.ViewMode.IconMode)
        self.grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.grid.setMovement(QListWidget.Movement.Static)
        self.grid.setIconSize(QSize(TILE, TILE))
        self.grid.setSpacing(6)
        self.grid.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.grid.setUniformItemSizes(True)
        self.grid.setWordWrap(False)
        ml.addWidget(self.grid, 1)
        return mid

    def _right(self) -> QWidget:
        side = QWidget()
        side.setObjectName("Pipe")
        v = QVBoxLayout(side)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(6)
        v.addWidget(h4("소스 Source"))
        self.detail = QLabel("")
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setMinimumHeight(240)
        self.detail.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        v.addWidget(self.detail)
        self.meta = QLabel("소스를 고르면 상세가 여기에")
        self.meta.setObjectName("Hint")
        self.meta.setWordWrap(True)
        self.meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.meta)
        v.addStretch(1)
        self.result = QLabel("")
        self.result.setObjectName("Muted")
        self.result.setWordWrap(True)
        v.addWidget(self.result)
        self.btn_edit = QPushButton("라벨 탭에서 다듬기 Refine in Label")
        self.btn_edit.setObjectName("Primary")
        self.btn_delete = QPushButton("삭제 Delete (Del)")
        v.addWidget(self.btn_edit)
        v.addWidget(self.btn_delete)
        side.setMinimumWidth(340)
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

    def _wire(self) -> None:
        self.btn_pick.clicked.connect(self._pick_bank)
        self.btn_open.clicked.connect(lambda: self.open_bank(self.bank_edit.text()))
        self.bank_edit.returnPressed.connect(lambda: self.open_bank(self.bank_edit.text()))
        self.btn_reload.clicked.connect(self.reload)
        for w in (self.f_cls, self.f_tag, self.sort):
            w.currentIndexChanged.connect(lambda _i: self.refresh_grid())
        self.f_text.textChanged.connect(lambda _t: self.refresh_grid())
        self.f_low.toggled.connect(lambda _b: self.refresh_grid())
        self.f_est.toggled.connect(lambda _b: self.refresh_grid())
        self.sort_desc.toggled.connect(lambda _b: self.refresh_grid())
        self.grid.itemSelectionChanged.connect(self._on_select)
        self.grid.itemDoubleClicked.connect(lambda _it: self.request_edit())
        self.btn_edit.clicked.connect(self.request_edit)
        self.btn_delete.clicked.connect(self.delete_selected)

    # ------------------------------------------------------------------ 열기

    def _pick_bank(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "결함 은행 폴더", self.bank_edit.text() or ".")
        if d:
            self.open_bank(d)

    def open_bank(self, root: str | Path) -> bool:
        text = str(root).strip()
        if not text:
            self._on_error("은행 폴더를 지정하세요")
            return False
        try:
            self.session.load(text)
        except BankSessionError as e:
            self._on_error(str(e))
            return False
        self.bank_edit.setText(Path(text).as_posix())
        self._tiles.clear()
        self._sync_filters()
        self.refresh()
        msg = f"은행 열림: {self.session.summary_text()}"
        if self.session.warnings:
            msg += f" · 경고 {len(self.session.warnings)}건: {self.session.warnings[0]}"
        self.status.emit(msg)
        return True

    def reload(self) -> None:
        if not self.session.loaded:
            return
        selected = self.selected_ids()
        self.session.reload()
        self._tiles.clear()
        self._sync_filters()
        self.refresh()
        self.select_ids(selected)

    def _sync_filters(self) -> None:
        for combo, items in (
            (self.f_cls, self.session.classes()),
            (self.f_tag, self.session.tags()),
        ):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(ALL)
            combo.addItems(items)
            idx = combo.findText(cur)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

    # ------------------------------------------------------------------ 그리드

    def filter_kwargs(self) -> dict:
        cls = self.f_cls.currentText()
        tag = self.f_tag.currentText()
        return {
            "cls": None if cls in (ALL, "") else cls,
            "tag": None if tag in (ALL, "") else tag,
            "only_low": self.f_low.isChecked(),
            "only_estimated": self.f_est.isChecked(),
            "text": self.f_text.text(),
            "sort": str(self.sort.currentData() or "id"),
            "descending": self.sort_desc.isChecked(),
        }

    def _tile(self, row: SourceRow) -> QPixmap:
        pm = self._tiles.get(row.id)
        if pm is None:
            s = self.session.source(row.id)
            pm = to_qpixmap(
                source_tile(
                    s.image, s.mask, s.id, s.mask_origin, tile=TILE, confidence=s.confidence
                )
            )
            self._tiles[row.id] = pm
        return pm

    def refresh_grid(self) -> None:
        self.grid.blockSignals(True)
        self.grid.clear()
        self._rows = self.session.filtered(**self.filter_kwargs()) if self.session.loaded else []
        for r in self._rows:
            item = QListWidgetItem(flat_icon(self._tile(r)), r.name)
            item.setData(Qt.ItemDataRole.UserRole, r.id)
            tip = f"{r.id} · {r.mask_origin} · {r.area_px:,} px"
            if r.confidence is not None:
                tip += f" · confidence {r.confidence:.2f}"
                if r.flags:
                    tip += f" ({', '.join(r.flags)})"
            if r.tags:
                tip += f" · 태그 {', '.join(r.tags)}"
            item.setToolTip(tip)
            self.grid.addItem(item)
        self.grid.blockSignals(False)
        total = len(self.session.rows())
        self.count.setText(
            f"표시 {len(self._rows)} / 전체 {total}"
            + (f" · 저신뢰 {sum(1 for r in self._rows if r.low_confidence)}" if self._rows else "")
        )
        self._on_select()

    def selected_ids(self) -> list[str]:
        return [str(it.data(Qt.ItemDataRole.UserRole)) for it in self.grid.selectedItems()]

    def select_ids(self, ids: list[str]) -> None:
        want = set(ids)
        for i in range(self.grid.count()):
            it = self.grid.item(i)
            it.setSelected(str(it.data(Qt.ItemDataRole.UserRole)) in want)

    def current_id(self) -> str | None:
        ids = self.selected_ids()
        return ids[0] if ids else None

    # ------------------------------------------------------------------ 상세

    def _on_select(self) -> None:
        ids = self.selected_ids()
        self.btn_delete.setEnabled(bool(ids))
        self.btn_edit.setEnabled(len(ids) == 1)
        if not ids:
            self.detail.clear()
            self.meta.setText(
                "소스를 고르면 상세가 여기에"
                if self.session.loaded
                else "은행을 열어 주세요 — 상단 경로 → 열기"
            )
            return
        if len(ids) > 1:
            self.detail.clear()
            self.meta.setText(f"{len(ids)}개 선택 — 삭제할 수 있습니다")
            return
        s = self.session.source(ids[0])
        self.detail.setPixmap(to_qpixmap(detail_image(s.image, s.mask)))
        r = next(x for x in self._rows if x.id == ids[0])
        lines = [
            f"<b>{r.id}</b>",
            f"클래스 {r.cls} · 크롭 {r.size[1]}×{r.size[0]} · 면적 {r.area_px:,} px",
            f"마스크 출처 {r.mask_origin}"
            + (
                f" · confidence <b>{r.confidence:.2f}</b>"
                + (" ⚠ 저신뢰" if r.low_confidence else "")
                if r.confidence is not None
                else ""
            ),
        ]
        if r.flags:
            lines.append("flags " + ", ".join(r.flags))
        lines.append(f"µm/px {r.um_per_px if r.um_per_px is not None else '미지정'}")
        if r.tags:
            lines.append("태그 " + ", ".join(r.tags))
        if r.origin:
            lines.append(f"원본 {r.origin}")
        self.meta.setText("<br>".join(lines))

    # ------------------------------------------------------------------ 편집

    def _ask_delete(self, ids: list[str]) -> bool:
        r = QMessageBox.question(
            self,
            "소스 삭제",
            f"{len(ids)}개 소스를 은행에서 지울까요? (png·mask·json 세 파일, 되돌릴 수 없음)\n"
            + "\n".join(ids[:6])
            + ("\n…" if len(ids) > 6 else ""),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return r == QMessageBox.StandardButton.Yes

    def delete_selected(self) -> int:
        ids = self.selected_ids()
        if not ids:
            return 0
        if self.confirm_delete is not None and not self.confirm_delete(ids):
            return 0
        try:
            n = self.session.delete(ids)
        except BankSessionError as e:
            self._on_error(str(e))
            return 0
        self._tiles.clear()
        self._sync_filters()
        self.refresh()
        msg = f"삭제 {n}개 · {self.session.summary_text()}"
        self.result.setText(msg)
        self.status.emit(msg)
        if n and self.session.root is not None:
            self.bank_changed.emit(self.session.root.as_posix())
        return n

    def request_edit(self) -> None:
        sid = self.current_id()
        if sid is None or self.session.root is None:
            return
        self.edit_requested.emit(self.session.root.as_posix(), sid)

    def apply_mask(self, source_id: str, mask: np.ndarray, *, tool: str = "brush") -> bool:
        """라벨 탭이 다듬은 마스크를 넘겨준다 — 같은 id 에 덮어쓰고 그리드를 갱신."""
        try:
            self.session.replace_mask(source_id, mask, tool=tool)
        except BankSessionError as e:
            self._on_error(str(e))
            return False
        self._tiles.pop(source_id, None)
        self.refresh()
        self.select_ids([source_id])
        msg = f"마스크 갱신: {source_id} (manual:{tool})"
        self.result.setText(msg)
        self.status.emit(msg)
        if self.session.root is not None:
            self.bank_changed.emit(self.session.root.as_posix())
        return True

    # ------------------------------------------------------------------ 표시

    def keyPressEvent(self, e) -> None:  # noqa: N802 (Qt 규약)
        if e.key() == Qt.Key.Key_Delete and self.grid.hasFocus():
            self.delete_selected()
            return
        super().keyPressEvent(e)

    def _on_error(self, msg: str) -> None:
        self.result.setText(msg)
        self.status.emit(msg)

    def refresh(self) -> None:
        self.summary.setText(self.session.summary_text() if self.session.loaded else "")
        self.btn_reload.setEnabled(self.session.loaded)
        self.refresh_grid()

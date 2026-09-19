"""프리셋 갤러리 대화상자(v0.9 사용성 ④) — 카드(썸네일 · 제목 · 한 줄 · 이럴 때 · 피할 때 · 근거 · 단계 요약)를 2열로.
문안·썸네일은 ``gallery.py``(Qt 없음). 더블클릭 또는 '이 프리셋으로' → ``chosen(name)``.

썸네일은 현재 보관함·바탕 이미지로 프리셋마다 한 장 합성(10장 ≈ 0.3 s, 256 px) — 준비된 세션이 없으면 문안만."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from anograft.gui.qt_image import to_qpixmap
from anograft.gui.studio.gallery import PresetCard, preset_cards
from anograft.gui.theme import COLORS
from anograft.preview import fit_long_side

THUMB_W, THUMB_H = 168, 118
COLUMNS = 2


class PresetCardWidget(QFrame):
    """카드 하나 — 클릭으로 선택, 더블클릭으로 확정."""

    clicked = Signal(str)
    activated = Signal(str)

    def __init__(self, card: PresetCard, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.card = card
        self.setObjectName("PresetCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(10)
        self.thumb = QLabel()
        self.thumb.setFixedSize(THUMB_W, THUMB_H)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet(
            f"background: {COLORS['bg']}; border: 1px solid {COLORS['line']}; border-radius: 4px; color: {COLORS['tx3']};"
        )
        self.thumb.setText("썸네일 없음")
        lay.addWidget(self.thumb, 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout()
        text.setSpacing(3)
        head = QLabel(f"<b>{card.title}</b> <span style='color:{COLORS['tx3']}'>{card.name}</span>")
        head.setTextFormat(Qt.TextFormat.RichText)
        text.addWidget(head)
        self.summary = self._muted(card.summary, COLORS["tx2"])
        text.addWidget(self.summary)
        if card.use_for:
            text.addWidget(self._muted(f"이럴 때: {card.use_for}", COLORS["teal"]))
        if card.avoid:
            text.addWidget(self._muted(f"피할 때: {card.avoid}", COLORS["amber"]))
        stages = " · ".join(f"{lab} {ml}" for lab, _m, ml in card.stages)
        text.addWidget(self._muted(stages, COLORS["tx3"]))
        if card.evidence:
            text.addWidget(self._muted(f"근거: {card.evidence}", COLORS["tx3"]))
        text.addStretch(1)
        lay.addLayout(text, 1)
        self.set_selected(False)

    @staticmethod
    def _muted(text: str, color: str) -> QLabel:
        lab = QLabel(text)
        lab.setWordWrap(True)
        lab.setStyleSheet(f"color: {color}; font-size: 11.5px;")
        lab.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        return lab

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", on)
        self.setStyleSheet(
            f"QFrame#PresetCard {{ background: {COLORS['panel2']}; border: 1px solid "
            f"{COLORS['teal'] if on else COLORS['line']}; border-radius: 7px; }}"
        )

    def set_thumb(self, image: np.ndarray | None) -> None:
        if image is None:
            self.thumb.setPixmap(QPixmap())
            self.thumb.setText("이 입력으로는 돌 수 없음")
            return
        small = fit_long_side(image, THUMB_W)
        self.thumb.setText("")
        self.thumb.setPixmap(to_qpixmap(small))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt 규약)
        self.clicked.emit(self.card.name)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt 규약)
        self.activated.emit(self.card.name)
        super().mouseDoubleClickEvent(event)


class PresetGalleryDialog(QDialog):
    """``PresetGalleryDialog(current, thumbs).exec()`` → ``chosen`` (Accepted 면 고른 이름)."""

    def __init__(
        self,
        current: str | None = None,
        thumbs: dict[str, np.ndarray | None] | None = None,
        parent: QWidget | None = None,
        *,
        cards: Sequence[PresetCard] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("프리셋 고르기")
        self.resize(1040, 720)
        self.chosen: str | None = current
        self.cards: dict[str, PresetCardWidget] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)
        intro = QLabel(
            "프리셋 = 7단계 설정을 묶어 둔 시작점. 고르면 카드 값이 그 프리셋으로 바뀌고, 그 뒤 값을 손봐도 됩니다(바뀐 값은 ↺). "
            "썸네일은 지금 고른 바탕 이미지에 각 프리셋을 적용한 것(시드 변형 v1)입니다."
        )
        intro.setWordWrap(True)
        intro.setObjectName("Muted")
        outer.addWidget(intro)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        for i, card in enumerate(cards if cards is not None else preset_cards()):
            w = PresetCardWidget(card)
            w.clicked.connect(self.select)
            w.activated.connect(self._activate)
            if thumbs is not None and card.name in thumbs:
                w.set_thumb(thumbs[card.name])
            grid.addWidget(w, i // COLUMNS, i % COLUMNS)
            self.cards[card.name] = w
        grid.setRowStretch(grid.rowCount(), 1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        buttons = QDialogButtonBox()
        self.btn_ok = buttons.addButton("이 프리셋으로", QDialogButtonBox.ButtonRole.AcceptRole)
        self.btn_ok.setObjectName("Primary")
        buttons.addButton("취소", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        if current in self.cards:
            self.select(current)
        self.btn_ok.setEnabled(self.chosen is not None)

    def select(self, name: str) -> None:
        self.chosen = name
        for n, w in self.cards.items():
            w.set_selected(n == name)
        self.btn_ok.setEnabled(True)

    def _activate(self, name: str) -> None:
        self.select(name)
        self.accept()

"""``VariantStrip`` — 같은 대상에 ``image_rng(seed, k)`` k = 0..N-1 로 돌린 결과 썸네일 카드. 목업 ``.variants``.

클릭 → ``selected(k)``. 아직 계산 안 된 카드는 자리표(“계산 중…”), 실패는 사유. 세대가 바뀌면 전부 자리표로 되돌린다.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QWidget

from anograft.gui.qt_image import to_qpixmap
from anograft.gui.studio.panels import flat_icon
from anograft.preview import fit_long_side

THUMB_W, THUMB_H = 146, 96


class VariantStrip(QWidget):
    selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Variants")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 8, 18, 8)
        lay.setSpacing(9)
        lab = QLabel("시드 변형 Variants")
        lab.setObjectName("Muted")
        lay.addWidget(lab)
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setFlow(QListWidget.Flow.LeftToRight)
        self.list.setWrapping(False)
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setMovement(QListWidget.Movement.Static)
        self.list.setIconSize(QSize(THUMB_W, THUMB_H))
        self.list.setSpacing(4)
        self.list.setFixedHeight(THUMB_H + 56)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.currentRowChanged.connect(self._on_row)
        lay.addWidget(self.list, 1)
        self._n = 0

    def _on_row(self, row: int) -> None:
        if row >= 0:
            self.selected.emit(row)

    def reset(self, n: int, current: int = 0) -> None:
        self._n = n
        self.list.blockSignals(True)
        self.list.clear()
        for k in range(n):
            item = QListWidgetItem(f"v{k + 1} · 계산 중…")
            item.setSizeHint(QSize(THUMB_W + 16, THUMB_H + 40))
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            self.list.addItem(item)
        if n:
            self.list.setCurrentRow(min(current, n - 1))
        self.list.blockSignals(False)

    def set_result(self, k: int, image: np.ndarray, caption: str) -> None:
        item = self.list.item(k)
        if item is None:
            return
        item.setIcon(flat_icon(to_qpixmap(fit_long_side(image, THUMB_W))))
        item.setText(f"v{k + 1} · {caption}")
        item.setToolTip("")

    def set_failed(self, k: int, reason: str) -> None:
        item = self.list.item(k)
        if item is not None:
            item.setIcon(QIcon())
            item.setText(f"v{k + 1} · {reason[:28]}")
            item.setToolTip(reason)  # 28자에서 잘린 원문 (KNOWN-ISSUES #1: 원인이 보여야 한다)

    def select(self, k: int) -> None:
        self.list.blockSignals(True)
        self.list.setCurrentRow(k)
        self.list.blockSignals(False)

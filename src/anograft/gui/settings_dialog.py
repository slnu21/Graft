"""설정 대화상자(v0.9 사용성 ⑦) — 테마(다크/라이트) · 글자 크기(100/115/130 %). 저장만 하고 **다시 시작하면 적용**
(팔레트는 위젯이 만들어질 때 읽히므로). ``QSettings`` 키는 ``theme.THEME_KEY``/``FONT_SCALE_KEY``."""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QWidget,
)

from anograft.gui.theme import FONT_SCALES, save_theme_settings, theme_settings

THEME_LABELS: tuple[tuple[str, str], ...] = (("dark", "다크"), ("light", "라이트"))


class SettingsDialog(QDialog):
    def __init__(self, store: QSettings | None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.store = store
        mode, scale = theme_settings(store)
        form = QFormLayout(self)
        self.theme = QComboBox()
        for key, label in THEME_LABELS:
            self.theme.addItem(label, key)
        self.theme.setCurrentIndex(max(self.theme.findData(mode), 0))
        self.theme.setToolTip("Theme — 다크(기본) / 라이트")
        form.addRow("테마", self.theme)
        self.scale = QComboBox()
        for s in FONT_SCALES:
            self.scale.addItem(f"{s * 100:.0f} %", s)
        self.scale.setCurrentIndex(max(self.scale.findData(scale), 0))
        self.scale.setToolTip("Font size — 모든 글자를 이 배율로(4K·고배율 화면용)")
        form.addRow("글자 크기", self.scale)
        note = QLabel("저장하면 다음에 앱을 열 때 적용됩니다 · Applied on next start")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        form.addRow(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("저장")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple[str, float]:
        return str(self.theme.currentData()), float(self.scale.currentData())

    def accept(self) -> None:  # Qt 규약
        if self.store is not None:
            mode, scale = self.values()
            save_theme_settings(self.store, mode, scale)
        super().accept()

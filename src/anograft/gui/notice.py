"""경고·힌트·오류 한 컴포넌트(v0.9 사용성 ⑥) — 아이콘 · 문장(무엇이 · 왜 · 이렇게) · 행동 버튼 하나.

카드 note+고치기, 검수 대비/조명 힌트, 보관함 신뢰도 안내가 전부 이것을 쓴다. 문장은 만들지 않고 보여만 준다(문장은 core/runner ·
review/session 등 Qt 없는 쪽이 만든다 — `<stage>:` 접두 규약 그대로)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

LEVELS: tuple[str, ...] = ("warn", "info", "error", "hint")
ICONS: dict[str, str] = {"warn": "⚠", "info": "ⓘ", "error": "✖", "hint": "💡"}


class Notice(QFrame):
    """``label``(문장)·``button``(행동)은 바깥에서 직접 만져도 된다(기존 카드 API 호환) — 바꾼 뒤 ``refresh()``."""

    action = Signal()

    def __init__(self, level: str = "warn", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Notice")
        self.level = level
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(6)
        self.icon = QLabel(ICONS.get(level, "⚠"))
        self.icon.setObjectName("NoticeIcon")
        self.icon.setFixedWidth(16)
        self.icon.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(self.icon)
        body = QVBoxLayout()
        body.setSpacing(4)
        self.label = QLabel("")
        self.label.setObjectName("NoticeText")
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        body.addWidget(self.label)
        self.button = QPushButton("")
        self.button.setObjectName("NoticeAction")
        self.button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.clicked.connect(self.action.emit)
        self.button.hide()
        body.addWidget(self.button)
        lay.addLayout(body, 1)
        self.set_level(level)
        self.hide()

    def set_level(self, level: str) -> None:
        self.level = level
        self.icon.setText(ICONS.get(level, "⚠"))
        self.setProperty("level", level)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_text(self, text: str | None) -> None:
        self.label.setText(text or "")
        self.label.setVisible(bool(text))
        self.refresh()

    def set_action(self, label: str | None) -> None:
        self.button.setText(label or "")
        self.button.setVisible(bool(label))
        self.refresh()

    def refresh(self) -> None:
        """문장이나 버튼이 하나라도 보이면 프레임도 보인다."""
        self.setVisible(bool(self.label.text()) or bool(self.button.text()))

    def text(self) -> str:
        return self.label.text()

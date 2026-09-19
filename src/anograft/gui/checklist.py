"""시작 체크리스트 위젯(v0.9 사용성 ⑤) — 레시피가 없을 때 미리보기 캔버스 자리에. 항목·완료 판정은 ``workflow.checklist``(Qt 없음).
버튼은 ``action(key)`` 만 내보내고 메인 창이 탭 이동·샘플·레시피 열기를 한다."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from anograft.gui.theme import COLORS
from anograft.gui.workflow import WorkflowState, checklist


class ChecklistPanel(QWidget):
    action = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(40, 30, 40, 30)
        outer.addStretch(1)
        box = QFrame()
        box.setObjectName("Checklist")
        box.setMinimumWidth(720)
        box.setMaximumWidth(860)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(10)
        title = QLabel("시작하기 — 결함 사진에서 학습 데이터셋까지")
        title.setObjectName("H3")
        lay.addWidget(title)
        sub = QLabel(
            "열린 레시피가 없습니다. 아래 순서대로 가면 됩니다 — 이미 된 단계는 ✓. "
            "위 탭의 번호가 같은 순서이고, 오른쪽 위 '다음 →' 로 넘어갑니다."
        )
        sub.setObjectName("Muted")
        sub.setWordWrap(True)
        lay.addWidget(sub)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(8)
        self.grid.setColumnStretch(1, 1)
        lay.addLayout(self.grid)
        outer.addWidget(box, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(2)
        self.rows: list[tuple[QLabel, QLabel, QPushButton]] = []
        self._keys: list[str] = []
        self.set_state(WorkflowState())

    def set_state(self, state: WorkflowState) -> None:
        items = checklist(state)
        self._keys = [it.key for it in items]
        while len(self.rows) < len(items):
            i = len(self.rows)
            mark = QLabel()
            mark.setFixedWidth(22)
            mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
            text = QLabel()
            text.setWordWrap(True)
            text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            text.setMinimumWidth(360)
            btn = QPushButton()
            btn.setFixedWidth(130)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _c=False, i=i: self.action.emit(self._keys[i]))
            self.grid.addWidget(mark, i, 0)
            self.grid.addWidget(text, i, 1)
            self.grid.addWidget(btn, i, 2)
            self.rows.append((mark, text, btn))
        for (mark, text, btn), it in zip(self.rows, items, strict=True):
            mark.setText("✓" if it.done else "○")
            mark.setStyleSheet(
                f"color: {COLORS['teal'] if it.done else COLORS['tx3']}; font-weight: 700;"
            )
            text.setText(it.text)
            text.setStyleSheet(f"color: {COLORS['tx3'] if it.done else COLORS['tx']};")
            btn.setText(it.action or "이동")
            btn.setObjectName("Primary" if not it.done and self._first_open(items) is it else "")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    @staticmethod
    def _first_open(items) -> object | None:
        return next((it for it in items if not it.done), None)

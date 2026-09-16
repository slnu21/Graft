"""샘플 데이터 옵션 대화상자 (v0.8.x) — 상단 '샘플 데이터' 버튼. 모양·폴더·정상/결함 장수·크기·합성 장수·시드를 한 번에.

``QuickstartDialog.values()`` 가 ``samples.quickstart.quickstart(**)`` 인자 dict 를 돌려준다(Qt 없는 부분은 ``quickstart_options`` 로
분리 — 테스트가 폼 값 → 인자 변환을 고정한다).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

SHAPES: dict[str, str] = {
    "판 plate — 브러시드 메탈 사각 판": "plate",
    "원형 ring — 가공 링 면 + 리세스(annulus·dent-graft)": "ring",
}


def quickstart_options(
    *,
    folder: str,
    shape: str,
    n_normal: int,
    n_defect: int,
    width: int,
    height: int,
    count: int,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    """폼 값 → ``(root, quickstart 키워드 인자)``. root = ``<folder>/<shape>``. 검증(장수·크기 최소)은 ``ValueError``."""
    if not folder.strip():
        raise ValueError("폴더를 고르세요")
    if n_normal < 1 or n_defect < 1:
        raise ValueError("정상·결함 이미지는 각각 1장 이상")
    if width < 64 or height < 64:
        raise ValueError("크기는 64 px 이상")
    root = Path(folder) / shape
    return root, {
        "shape": shape,
        "seed": int(seed),
        "n_normal": int(n_normal),
        "n_defect": int(n_defect),
        "size": (int(width), int(height)),
        "count": int(count),
    }


class QuickstartDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, *, default_folder: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("샘플 데이터 Sample data")
        form = QFormLayout(self)
        self.shape = QComboBox()
        for label, key in SHAPES.items():
            self.shape.addItem(label, key)
        form.addRow("부품 모양 Part shape", self.shape)
        row = QHBoxLayout()
        self.folder = QLineEdit(default_folder)
        self.folder.setPlaceholderText("샘플을 만들 폴더 (그 아래 <모양>/ 이 생깁니다)")
        self.btn_folder = QPushButton("폴더… Browse")
        self.btn_folder.clicked.connect(self._pick_folder)
        row.addWidget(self.folder, 1)
        row.addWidget(self.btn_folder)
        form.addRow("폴더 Folder", row)
        self.n_normal = self._spin(1, 500, 12)
        self.n_defect = self._spin(1, 500, 10)
        form.addRow("정상 이미지 normal images", self.n_normal)
        form.addRow("결함 이미지 defect images", self.n_defect)
        size_row = QHBoxLayout()
        self.width = self._spin(64, 8192, 640)
        self.height = self._spin(64, 8192, 480)
        size_row.addWidget(self.width)
        size_row.addWidget(self.height)
        form.addRow("크기 size (w × h)", size_row)
        self.count = self._spin(1, 100000, 24)
        form.addRow("합성 장수 synthetic count", self.count)
        self.seed = self._spin(0, 2**31 - 1, 7)
        form.addRow("시드 seed", self.seed)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    @staticmethod
    def _spin(lo: int, hi: int, value: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(value)
        return s

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "샘플을 만들 폴더 Folder for the sample set", self.folder.text() or "."
        )
        if d:
            self.folder.setText(d)

    def values(self) -> tuple[Path, dict[str, Any]]:
        return quickstart_options(
            folder=self.folder.text(),
            shape=str(self.shape.currentData()),
            n_normal=self.n_normal.value(),
            n_defect=self.n_defect.value(),
            width=self.width.value(),
            height=self.height.value(),
            count=self.count.value(),
            seed=self.seed.value(),
        )

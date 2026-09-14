"""``MainWindow`` — 상단 바(브랜드·컨텍스트) · 5탭(은행·라벨·스튜디오·배치·검수, 스튜디오만 실물) · 상태바. 목업 ``.bar``·``.tabs``.

``run_app(recipe=)``가 ``QApplication``을 만들고 테마를 입힌 뒤 창을 띄운다. 워커 스레드는 창이 닫힐 때 멈춘다.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from anograft import __version__
from anograft.gui.studio.session import StudioSession
from anograft.gui.studio.tab import StudioTab
from anograft.gui.studio.worker import PreviewWorker
from anograft.gui.theme import apply_theme

TABS: tuple[tuple[str, str], ...] = (
    ("bank", "은행  Bank"),
    ("label", "라벨  Label"),
    ("studio", "스튜디오  Studio"),
    ("batch", "배치  Batch"),
    ("review", "검수  Review"),
)
PLACEHOLDER: dict[str, str] = {
    "bank": "결함 은행 보기·가져오기 — v0.3 (지금은 CLI: anograft bank import-yolo / bank ls)",
    "label": "브러시·폴리곤·자동 선택으로 결함 마스크 만들기 — v0.5",
    "batch": "레시피로 데이터셋 생성·진행률 — v0.7 (지금은 CLI: anograft run <recipe>)",
    "review": "합성 결과 검수·실제 결함 분포 비교 — v0.7",
}


class MainWindow(QMainWindow):
    def __init__(self, session: StudioSession | None = None, *, start_worker: bool = True) -> None:
        super().__init__()
        self.setWindowTitle(f"Graft — anograft {__version__}")
        self.resize(1480, 920)
        self.session = session or StudioSession()
        self.worker = PreviewWorker(self)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._top_bar())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.studio = StudioTab(self.session, self.worker)
        for key, label in TABS:
            page = self.studio if key == "studio" else self._placeholder(PLACEHOLDER[key])
            self.tabs.addTab(page, label)
        self.tabs.setCurrentIndex(2)
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.studio.status.connect(self.status_bar.showMessage)
        self.studio.context.connect(self.ctx.setText)
        self.worker.busy.connect(self._on_busy)
        self.studio.sync_widgets()
        if start_worker:
            self.worker.start()

    def _top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("TopBar")
        h = QHBoxLayout(bar)
        h.setContentsMargins(18, 9, 18, 8)
        h.setSpacing(16)
        brand = QLabel("Graft")
        brand.setObjectName("Brand")
        sub = QLabel(f"anograft {__version__} · 오프라인 · CPU")
        sub.setObjectName("BrandSub")
        self.ctx = QLabel("")
        self.ctx.setObjectName("Ctx")
        self.ctx.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.busy = QLabel("")
        self.busy.setObjectName("Muted")
        h.addWidget(brand)
        h.addWidget(sub)
        h.addWidget(self.ctx, 1)
        h.addWidget(self.busy)
        return bar

    @staticmethod
    def _placeholder(text: str) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        lab = QLabel(text)
        lab.setObjectName("Hint")
        lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(lab)
        return w

    def _on_busy(self, busy: bool) -> None:
        self.busy.setText("● 계산 중" if busy else "")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.worker.stop()
        super().closeEvent(event)


def run_app(recipe: str | None = None, argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.show()
    if recipe:
        win.studio.open_recipe(Path(recipe))
    return app.exec()

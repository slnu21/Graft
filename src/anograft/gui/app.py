"""``MainWindow`` — 상단 바(브랜드·컨텍스트) · 5탭(은행·라벨·스튜디오·배치·검수 — v0.7 에서 전부 실물) · 상태바. 목업 ``.bar``·``.tabs``.

은행 탭(v0.7): 소스 보기·필터·삭제, "라벨 탭에서 다듬기"(``edit_requested`` → ``LabelTab.begin_bank_edit`` → ``source_updated`` →
``BankTab.apply_mask``). 은행이 바뀌면(``bank_changed``·라벨 탭 ``bank_saved``) 같은 은행을 쓰는 스튜디오가 다시 준비하고 은행 탭이 새로고침한다.

스튜디오 "배치로 보내기" → 현재 레시피를 배치 탭에 넘기고 탭을 전환한다(``send_to_batch`` 시그널). 배치 탭은 ``BatchWorker`` 스레드로
``runner.run`` 을 돌린다 — CLI ``anograft run`` 과 결과 동일.

라벨 탭에서 은행에 저장하면(``bank_saved``) 스튜디오가 같은 은행을 쓰고 있을 때 다시 준비한다(새 소스가 미리보기에 바로 반영).

``run_app(recipe=)``가 ``QApplication``을 만들고 테마를 입힌 뒤 창을 띄운다. 워커 스레드는 창이 닫힐 때 멈춘다.
레시피 인자가 없으면 **최근 레시피**(``QSettings`` slnu21/Graft ``recent_recipe``, 열기·저장 때 기록)를 복원하고, 그것도 없으면
스튜디오 캔버스에 **빈 상태 안내**(ko/en — 라벨 → 입력 → 프리셋 → 배치)를 띄운다(KNOWN-ISSUES #9).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
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
from anograft.gui.bank.tab import BankTab
from anograft.gui.batch.tab import BatchTab
from anograft.gui.label.tab import LabelTab
from anograft.gui.review.tab import ReviewTab
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
PLACEHOLDER: dict[str, str] = {}
SETTINGS_ORG, SETTINGS_APP = "slnu21", "Graft"
RECENT_RECIPE_KEY = "recent_recipe"
EMPTY_STATE = (
    "레시피가 열려 있지 않습니다 · No recipe loaded\n\n"
    "①  라벨 탭에서 결함 사진 → 마스크 → 은행에 저장   Label tab: defect photo → mask → save to bank\n"
    "②  왼쪽 '입력'에 은행 폴더와 정상 이미지 폴더 → 열기   Inputs (left): bank + normal-image folder → Open\n"
    "③  프리셋을 고르고 대상을 클릭 → 미리보기   Pick a preset, click a target → preview\n"
    "④  배치로 보내기 → 생성 시작   Send to batch → run\n\n"
    "또는 상단 '레시피 열기'로 YAML (예: recipes/sample-poisson.yaml)   or 'Open recipe' at the top"
)


def settings() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def recent_recipe(store: QSettings | None = None) -> Path | None:
    """최근 레시피 경로 — 파일이 아직 있을 때만."""
    v = (store or settings()).value(RECENT_RECIPE_KEY, "", type=str)
    if not v:
        return None
    p = Path(v)
    return p if p.is_file() else None


def remember_recipe(path: str | Path | None, store: QSettings | None = None) -> None:
    st = store or settings()
    if path is None:
        st.remove(RECENT_RECIPE_KEY)
    else:
        st.setValue(RECENT_RECIPE_KEY, Path(path).resolve().as_posix())
    st.sync()


class MainWindow(QMainWindow):
    def __init__(
        self,
        session: StudioSession | None = None,
        *,
        start_worker: bool = True,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(f"Graft — anograft {__version__}")
        self.resize(1480, 920)
        self.session = session or StudioSession()
        self.settings = settings  # None = 최근 레시피를 기억하지 않는다(테스트)
        self.worker = PreviewWorker(self)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._top_bar())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.studio = StudioTab(self.session, self.worker)
        self.label = LabelTab()
        self.batch = BatchTab()
        self.bank = BankTab()
        self.review = ReviewTab()
        for key, label in TABS:
            if key == "studio":
                page: QWidget = self.studio
            elif key == "label":
                page = self.label
            elif key == "bank":
                page = self.bank
            elif key == "review":
                page = self.review
            elif key == "batch":
                page = self.batch
            else:
                page = self._placeholder(PLACEHOLDER[key])
            self.tabs.addTab(page, label)
        self.tabs.setCurrentIndex(2)
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.studio.status.connect(self.status_bar.showMessage)
        self.studio.context.connect(self.ctx.setText)
        self.label.status.connect(self.status_bar.showMessage)
        self.label.bank_saved.connect(self._on_bank_saved)
        self.batch.status.connect(self.status_bar.showMessage)
        self.bank.status.connect(self.status_bar.showMessage)
        self.review.status.connect(self.status_bar.showMessage)
        self.batch.run_finished.connect(self._on_batch_finished)
        self.batch.review_requested.connect(self._on_review_requested)
        self.bank.bank_changed.connect(self._on_bank_saved)
        self.bank.edit_requested.connect(self._on_bank_edit_requested)
        self.label.source_updated.connect(self._on_source_updated)
        self.label.bank_saved.connect(self._on_label_bank_saved)
        self.studio.send_to_batch_requested.connect(self._on_send_to_batch)
        self.studio.recipe_opened.connect(self._on_recipe_opened)
        self.label.set_bank(self.session.recipe.inputs.bank_key())
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

    def _on_recipe_opened(self, path: str) -> None:
        """스튜디오가 레시피를 열거나 저장했다 — 최근 레시피로 기억, 라벨 탭 은행 동기."""
        if self.settings is not None:
            remember_recipe(path, self.settings)
        bank = self.session.recipe.inputs.bank_key()
        self.label.set_bank(bank)
        if bank and not self.bank.session.loaded:
            self.bank.open_bank(bank)

    def open_recipe(self, path: str | Path) -> None:
        self.studio.open_recipe(Path(path))

    def show_empty_state(self) -> None:
        self.studio.canvas.clear(EMPTY_STATE)
        self.status_bar.showMessage(
            "레시피 없음 — 라벨 탭에서 시작하거나 스튜디오 '열기'로 레시피를 여세요 · "
            "No recipe: start in the Label tab or open a recipe"
        )

    def _on_batch_finished(self, summary) -> None:
        """배치가 끝나면 검수 탭이 그 출력 폴더를 가리킨다(자동으로 열지는 않는다 — 큰 출력은 썸네일 로드가 걸린다)."""
        if summary is not None and getattr(summary, "writer", None) is not None:
            self.review.root_edit.setText(Path(summary.writer.root).as_posix())

    def _on_review_requested(self, root: str) -> None:
        """배치 탭 '검수 탭에서 열기' → 검수 탭이 그 출력을 열고 탭 전환."""
        if self.review.open_root(root):
            self.tabs.setCurrentWidget(self.review)

    def _on_bank_saved(self, root: str) -> None:
        """라벨 탭이 은행에 소스를 더했다(또는 은행 탭이 지우거나 고쳤다) — 스튜디오가 그 은행을 쓰면 다시 준비."""
        ses = self.session
        if ses.recipe.inputs.bank_key() == root:
            self.studio.open_inputs(root, ses.recipe.inputs.targets.as_posix())

    def _on_label_bank_saved(self, root: str) -> None:
        """라벨 탭 저장 → 은행 탭이 같은 은행을 보고 있으면 새로고침."""
        if self.bank.session.root is not None and self.bank.session.root.as_posix() == root:
            self.bank.reload()

    def _on_bank_edit_requested(self, root: str, source_id: str) -> None:
        """은행 탭 → 라벨 탭 은행 소스 편집 모드."""
        try:
            s = self.bank.session.source(source_id)
        except Exception as e:  # BankSessionError — 상태바에만
            self.status_bar.showMessage(str(e))
            return
        gray = bool(
            s.image.ndim == 3
            and (s.image[..., 0] == s.image[..., 1]).all()
            and (s.image[..., 1] == s.image[..., 2]).all()
        )
        if self.label.begin_bank_edit(root, source_id, s.image, gray, s.mask):
            self.tabs.setCurrentWidget(self.label)

    def _on_source_updated(self, root: str, source_id: str, mask, tool: str) -> None:
        """라벨 탭 은행 소스 편집 저장 → 은행 탭이 같은 id 에 덮어쓴다(→ bank_changed → 스튜디오 재준비)."""
        if self.bank.session.root is None or self.bank.session.root.as_posix() != root:
            self.bank.open_bank(root)
        self.bank.apply_mask(source_id, mask, tool=tool)

    def _on_send_to_batch(self) -> None:
        """스튜디오 → 배치: 현재 레시피(저장 여부 무관)를 넘기고 배치 탭으로."""
        ses = self.session
        self.batch.set_recipe(ses.recipe, ses.recipe_path)
        self.tabs.setCurrentWidget(self.batch)

    def _on_busy(self, busy: bool) -> None:
        self.busy.setText("● 계산 중" if busy else "")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.worker.stop()
        if self.batch.worker is not None:
            self.batch.stop()
            self.batch.wait(10000)
        super().closeEvent(event)


def run_app(recipe: str | None = None, argv: list[str] | None = None) -> int:
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    apply_theme(app)
    store = settings()
    win = MainWindow(settings=store)
    win.show()
    start = Path(recipe) if recipe else recent_recipe(store)
    if start is not None:
        win.open_recipe(start)
    else:
        win.show_empty_state()
    return app.exec()

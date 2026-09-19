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
from typing import Any

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
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
from anograft.gui.studio import param_form
from anograft.gui.studio.session import StudioSession
from anograft.gui.studio.tab import StudioTab
from anograft.gui.studio.worker import PreviewWorker
from anograft.gui.theme import apply_theme
from anograft.gui.workflow import (
    NEXT_HINT,
    STEPS,
    WorkflowState,
    next_step,
    tab_label,
)
from anograft.samples.quickstart import Quickstart

# (키, 한국어 라벨) — 흐름 순서(v0.9 stepper: 결함 표시 → 보관함 → 미리보기 → 일괄 생성 → 검수). 영어 이름은 TAB_TIPS 툴팁으로.
# 실제 탭 텍스트는 workflow.tab_label(번호 + 배지) — 이 상수는 순서·기본 이름의 정본
TABS: tuple[tuple[str, str], ...] = STEPS
TAB_TIPS: dict[str, str] = {
    "bank": "Bank — 라벨링한 결함 조각(이미지·마스크·메타)을 모아 두는 곳 · CLI anograft bank",
    "label": "Label — 결함 사진에 마스크를 그려 보관함에 넣습니다",
    "studio": "Studio — 프리셋·값을 바꿔 가며 한 장씩 미리 봅니다",
    "batch": "Batch — 레시피대로 데이터셋을 한꺼번에 만듭니다 · CLI anograft run",
    "review": "Review — 만든 결과를 채택/반려하고 분포를 봅니다",
}
PLACEHOLDER: dict[str, str] = {}
SETTINGS_ORG, SETTINGS_APP = "slnu21", "Graft"
RECENT_RECIPE_KEY = "recent_recipe"
EMPTY_STATE = (
    "열린 레시피가 없습니다\n\n"
    "⓪  데이터가 하나도 없으면 상단 '샘플 데이터…' 로 시작하세요\n"
    "①  결함 표시 탭에서 결함 사진에 마스크를 그려 결함 보관함에 저장합니다\n"
    "②  왼쪽 '입력'에 결함 보관함 폴더와 바탕(정상) 이미지 폴더를 넣고 '열기'\n"
    "③  프리셋을 고르고 바탕 이미지를 클릭하면 미리보기가 나옵니다\n"
    "④  '일괄 생성으로 보내기' → '생성 시작'\n\n"
    "또는 상단 '레시피 열기'로 YAML 을 엽니다 (예: recipes/sample-poisson.yaml)"
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
        param_form.use_settings(
            settings
        )  # 카드 폼 '고급 옵션' 펼침 기억도 같은 저장소(None 이면 기억 안 함)
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
            self.tabs.setTabToolTip(self.tabs.count() - 1, TAB_TIPS[key])
        self.tabs.setCurrentIndex(self.tab_index("studio"))
        # 코너 '다음 →' — 흐름의 다음 탭으로(툴팁 = 이 탭에서 끝낼 것)
        self.btn_next = QPushButton("")
        self.btn_next.setObjectName("NextStep")
        self.btn_next.clicked.connect(self.go_next)
        self.tabs.setCornerWidget(self.btn_next, Qt.Corner.TopRightCorner)
        self.tabs.currentChanged.connect(lambda _i: self._sync_next())
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
        self.label.next_edit_requested.connect(self._on_next_edit_requested)
        self.label.bank_saved.connect(self._on_label_bank_saved)
        self.studio.send_to_batch_requested.connect(self._on_send_to_batch)
        self.studio.recipe_opened.connect(self._on_recipe_opened)
        self.label.set_bank(self.session.recipe.inputs.bank_key())
        self.worker.busy.connect(self._on_busy)
        # 배지·체크리스트는 각 탭의 상태 시그널 뒤에 다시 계산(값싼 갱신)
        for sig in (
            self.studio.status,
            self.label.status,
            self.bank.status,
            self.review.status,
            self.batch.status,
        ):
            sig.connect(lambda _m: self.refresh_workflow())
        self.batch.run_finished.connect(lambda _s: self.refresh_workflow())
        self.bank.bank_changed.connect(lambda _r: self.refresh_workflow())
        self.studio.checklist.action.connect(self._on_checklist_action)
        self.studio.sync_widgets()
        self.refresh_workflow()
        self._sync_next()
        if start_worker:
            self.worker.start()

    # ------------------------------------------------------------------ 흐름(탭 순서·배지·다음)

    def tab_index(self, key: str) -> int:
        return [k for k, _n in STEPS].index(key)

    def current_key(self) -> str:
        return STEPS[self.tabs.currentIndex()][0]

    def workflow_state(self) -> WorkflowState:
        """각 탭 세션에서 읽은 사실 — 배지·체크리스트의 입력(Qt 없는 workflow.py 가 문구를 만든다)."""
        ses = self.session
        prep = ses.prepared
        pieces: int | None = None
        if self.bank.session.loaded and self.bank.session.bank is not None:
            pieces = len(self.bank.session.bank)
        elif prep is not None and not ses.recipe.bankless:
            pieces = len(prep.bank)
        rs = self.review.session
        counts = rs.counts() if rs.loaded else {}
        return WorkflowState(
            bank_pieces=pieces,
            recipe_open=ses.recipe_path is not None or prep is not None,
            prepared=prep is not None,
            targets=len(prep.targets) if prep is not None else 0,
            batch_ok=self.batch.last_summary.writer.n_ok
            if self.batch.last_summary is not None
            else None,
            review_open=rs.loaded,
            review_unreviewed=counts.get("unreviewed") if rs.loaded else None,
            label_images=self.label.list.count(),
        )

    def refresh_workflow(self) -> None:
        state = self.workflow_state()
        for i, (key, _name) in enumerate(STEPS):
            self.tabs.setTabText(i, tab_label(key, state))
        self.studio.checklist.set_state(state)

    def _sync_next(self) -> None:
        key = self.current_key()
        nxt = next_step(key)
        if nxt is None:
            self.btn_next.setText("흐름 끝 — 데이터셋 내보내기")
            self.btn_next.setToolTip(NEXT_HINT[key])
            self.btn_next.setEnabled(False)
            return
        self.btn_next.setEnabled(True)
        self.btn_next.setText(f"다음: {nxt[1]} →")
        self.btn_next.setToolTip(NEXT_HINT[key])

    def go_next(self) -> None:
        nxt = next_step(self.current_key())
        if nxt is not None:
            self.tabs.setCurrentIndex(self.tab_index(nxt[0]))

    def _on_checklist_action(self, key: str) -> None:
        """체크리스트 버튼 — 탭으로 가거나(키 = 탭) 상단 동작(sample · 레시피 열기)."""
        if key == "sample":
            self._on_sample_clicked()
        elif key == "studio":
            self.studio.open_recipe_dialog()
        elif key in dict(STEPS):
            self.tabs.setCurrentIndex(self.tab_index(key))

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
        self.btn_sample = QPushButton("샘플 데이터…")
        self.btn_sample.setToolTip(
            "데이터가 없을 때 — 샘플 이미지와 라벨을 만들어 결함 보관함에 넣고 레시피까지 열어 줍니다\n"
            "Sample data — generate sample images + labels, import them into a bank and open a recipe"
        )
        self.btn_sample.clicked.connect(self._on_sample_clicked)
        h.addWidget(brand)
        h.addWidget(sub)
        h.addWidget(self.ctx, 1)
        h.addWidget(self.btn_sample)
        h.addWidget(self.busy)
        return bar

    # ------------------------------------------------------------------ 샘플 데이터

    def _on_sample_clicked(self) -> None:
        from anograft.gui.quickstart_dialog import QuickstartDialog

        dlg = QuickstartDialog(self, default_folder=str(Path.cwd() / "samples"))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            root, opts = dlg.values()
        except ValueError as e:
            QMessageBox.warning(self, "샘플 데이터", str(e))
            return
        self.make_sample(root, **opts)

    def make_sample(self, root: str | Path, shape: str = "plate", **opts: Any) -> Quickstart | None:
        """샘플 → 은행 → 레시피(``samples.quickstart``, Qt 없음) → 스튜디오·은행 탭에 연다. 몇 초 걸린다(박스→마스크 grabcut).
        ``opts`` = seed·n_normal·n_defect·size·count(대화상자 값)."""
        from anograft.samples.quickstart import quickstart

        self.status_bar.showMessage(f"샘플 데이터 만드는 중… {Path(root).as_posix()}")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            q = quickstart(root, shape=shape, **opts)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "샘플 생성 실패", str(e))
            return None
        finally:
            QApplication.restoreOverrideCursor()
        self.open_recipe(q.recipe)
        self.bank.open_bank(q.bank.as_posix())
        self.tabs.setCurrentWidget(self.studio)
        self.status_bar.showMessage(q.line())
        return q

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
        self.studio.show_checklist(False)
        self.refresh_workflow()
        bank = self.session.recipe.inputs.bank_key()
        self.label.set_bank(bank)
        if bank and not self.bank.session.loaded:
            self.bank.open_bank(bank)

    def open_recipe(self, path: str | Path) -> None:
        self.studio.open_recipe(Path(path))

    def show_empty_state(self) -> None:
        self.studio.canvas.clear(EMPTY_STATE)
        self.studio.show_checklist(True)
        self.refresh_workflow()
        self.status_bar.showMessage(
            "열린 레시피가 없습니다 — 결함 표시 탭에서 시작하거나 미리보기 탭의 '열기'로 레시피를 여세요"
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

    def _on_next_edit_requested(self, root: str, after: str) -> None:
        """라벨 탭 '다음 저신뢰 소스' → 은행 탭이 다음 id 를 골라 편집 요청(같은 은행이 아니면 먼저 연다)."""
        if self.bank.session.root is None or self.bank.session.root.as_posix() != root:
            self.bank.open_bank(root)
        if not self.bank.request_edit_next_low(after):
            self.tabs.setCurrentWidget(self.bank)

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
        self.busy.setText("● 계산 중…" if busy else "")

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

"""``BatchTab`` — 목업 ``.batch``: 생성 설정 카드(레시피 · 출력 폴더 · 장수 · 시드 · 워커 · 출력 형식) + 실행(진행률 바 · 로그 · 중지 ·
출력 폴더 열기). 스튜디오 "배치로 보내기"가 현재 레시피를 여기로 넘긴다(``set_recipe``); 파일에서 열 수도 있다.

실행은 ``BatchWorker`` 스레드 — GUI 는 시그널만 받는다. 같은 레시피·시드면 CLI ``anograft run`` 과 결과가 같다(로그 첫 줄에
그 명령을 남긴다).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from anograft import runner
from anograft.core import recipe as R
from anograft.gui.batch.session import WRITER_FORMATS, BatchError, BatchSession, summary_text
from anograft.gui.batch.worker import BatchWorker
from anograft.gui.studio.panels import h4


class BatchTab(QWidget):
    status = Signal(str)
    run_finished = Signal(object)  # runner.RunSummary | None(실패)

    def __init__(self, session: BatchSession | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session or BatchSession()
        self.worker: BatchWorker | None = None
        self.last_summary: runner.RunSummary | None = None

        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)
        root.addWidget(self._settings_card(), 0)
        root.addWidget(self._run_card(), 1)
        self._wire()
        self.sync_widgets()

    # ------------------------------------------------------------------ 조립

    def _settings_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("StageCard")
        card.setMinimumWidth(420)
        card.setMaximumWidth(520)
        v = QVBoxLayout(card)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)
        v.addWidget(h4("생성 설정 Batch"))
        self.hint = QLabel(
            "스튜디오에서 확정한 레시피를 그대로 돌립니다. 시드와 레시피가 같으면 언제 다시 돌려도 같은 데이터셋이 나옵니다 — "
            "CLI `anograft run` 과 동일. 이미지마다 어떤 결함이 어디에 들어갔는지 meta/·manifest.csv 에 남습니다."
        )
        self.hint.setObjectName("Hint")
        self.hint.setWordWrap(True)
        v.addWidget(self.hint)
        form = QFormLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(7)
        rec_row = QHBoxLayout()
        self.recipe_label = QLabel("(없음)")
        self.recipe_label.setObjectName("Muted")
        self.recipe_label.setWordWrap(True)
        self.btn_recipe = QPushButton("파일…")
        self.btn_recipe.setFixedWidth(52)
        rec_row.addWidget(self.recipe_label, 1)
        rec_row.addWidget(self.btn_recipe)
        form.addRow("레시피", rec_row)
        out_row = QHBoxLayout()
        self.out = QLineEdit()
        self.out.setPlaceholderText("출력 폴더 (예: out/run-01)")
        self.btn_out = QPushButton("폴더")
        self.btn_out.setFixedWidth(44)
        out_row.addWidget(self.out, 1)
        out_row.addWidget(self.btn_out)
        form.addRow("출력 폴더", out_row)
        self.count = QSpinBox()
        self.count.setRange(1, 1_000_000)
        form.addRow("생성 장수", self.count)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        form.addRow("시드", self.seed)
        self.workers = QSpinBox()
        self.workers.setRange(0, 64)
        self.workers.setToolTip(
            "0 = 이 프로세스에서 순서대로. N = spawn 워커 N개 — 20장 넘으면 빠르다(결과는 동일)"
        )
        form.addRow("워커", self.workers)
        self.writer = QComboBox()
        self.writer.addItems(list(WRITER_FORMATS))
        form.addRow("출력 형식", self.writer)
        self.category = QLineEdit()
        self.category.setPlaceholderText("mvtec 카테고리 폴더 이름 (예: graft)")
        form.addRow("mvtec 카테고리", self.category)
        v.addLayout(form)
        self.summary = QLabel("")
        self.summary.setObjectName("Muted")
        self.summary.setWordWrap(True)
        v.addWidget(self.summary)
        v.addStretch(1)
        self.btn_run = QPushButton("생성 시작 Run")
        self.btn_run.setObjectName("Primary")
        self.btn_stop = QPushButton("중지 Stop")
        self.btn_stop.setEnabled(False)
        row = QHBoxLayout()
        row.addWidget(self.btn_run, 1)
        row.addWidget(self.btn_stop)
        v.addLayout(row)
        return card

    def _run_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("StageCard")
        v = QVBoxLayout(card)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)
        v.addWidget(h4("진행 Progress"))
        self.bar = QProgressBar()
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.bar.setTextVisible(True)
        v.addWidget(self.bar)
        self.progress_label = QLabel("대기")
        self.progress_label.setObjectName("Muted")
        v.addWidget(self.progress_label)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        v.addWidget(self.log, 1)
        row = QHBoxLayout()
        self.btn_open_out = QPushButton("출력 폴더 열기")
        self.btn_open_out.setEnabled(False)
        row.addStretch(1)
        row.addWidget(self.btn_open_out)
        v.addLayout(row)
        return card

    def _wire(self) -> None:
        self.btn_recipe.clicked.connect(self.open_recipe_dialog)
        self.btn_out.clicked.connect(self._pick_out)
        self.btn_run.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_open_out.clicked.connect(self.open_output)
        self.writer.currentTextChanged.connect(lambda f: self.category.setEnabled(f == "mvtec"))

    # ------------------------------------------------------------------ 레시피

    def set_recipe(self, recipe: R.Recipe, path: Path | None = None) -> None:
        self.session.set_recipe(recipe, path)
        self.sync_widgets()
        self.status.emit(
            f"배치: 레시피 {recipe.name} ({recipe.pipeline.preset}, seed {recipe.seed}) — 출력 {self.session.out}"
        )

    def open_recipe(self, path: str | Path) -> bool:
        try:
            self.session.load(path)
        except BatchError as e:
            self._append(str(e))
            self.status.emit(str(e))
            return False
        self.sync_widgets()
        return True

    def open_recipe_dialog(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "레시피 열기", "recipes", "레시피 (*.yaml *.yml)")
        if f:
            self.open_recipe(f)

    def _pick_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "출력 폴더", self.out.text() or ".")
        if d:
            self.out.setText(Path(d).as_posix())

    def sync_widgets(self) -> None:
        s = self.session
        if s.recipe is None:
            self.recipe_label.setText("(없음 — 스튜디오 '배치로 보내기' 또는 파일…)")
            self.summary.setText("")
        else:
            src = s.recipe_path.as_posix() if s.recipe_path else "(스튜디오 레시피, 저장 안 됨)"
            self.recipe_label.setText(f"{s.recipe.name} · {s.recipe.pipeline.preset}\n{src}")
            r = s.recipe
            self.summary.setText(
                f"은행 {r.inputs.bank_key() or '(없음)'} · 대상 {r.inputs.targets.as_posix()} · "
                f"결함/이미지 {r.output.defects_per_image[0]}–{r.output.defects_per_image[1]} · 정상 포함 {r.output.include_normals}"
            )
        self.out.setText(s.out)
        self.count.setValue(max(1, s.count))
        self.seed.setValue(s.seed)
        self.workers.setValue(s.workers)
        self.writer.setCurrentText(s.writer)
        self.category.setText(s.mvtec_category)
        self.category.setEnabled(s.writer == "mvtec")
        self.btn_run.setEnabled(s.recipe is not None and self.worker is None)

    def _pull(self) -> None:
        s = self.session
        s.out = self.out.text().strip()
        s.count = int(self.count.value())
        s.seed = int(self.seed.value())
        s.workers = int(self.workers.value())
        s.writer = self.writer.currentText()
        s.mvtec_category = self.category.text().strip() or "graft"

    # ------------------------------------------------------------------ 실행

    def start(self) -> bool:
        if self.worker is not None:
            return False
        self._pull()
        try:
            rec = self.session.build_recipe()
        except BatchError as e:
            self._append(f"오류: {e}")
            self.status.emit(str(e))
            return False
        self.log.clear()
        self._append(f"$ {self.session.run_command()}")
        self._append(
            f"레시피 {rec.name} · seed {rec.seed} · {rec.output.count}장 → {rec.output.root.as_posix()} ({rec.output.writer.format})"
        )
        self.bar.setRange(0, rec.output.count)
        self.bar.setValue(0)
        self.progress_label.setText("준비 중(은행·대상 로드)…")
        self.last_summary = None
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_open_out.setEnabled(False)
        w = BatchWorker(rec, self.session.workers, self)
        w.progress.connect(self._on_progress)
        w.warning.connect(lambda m: self._append(f"경고: {m}"))
        w.finished_run.connect(self._on_finished)
        w.failed.connect(self._on_failed)
        w.finished.connect(self._on_thread_done)
        self.worker = w
        w.start()
        self.status.emit("배치 실행 중…")
        return True

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.progress_label.setText("중지 요청 — 현재 이미지까지 기록하고 멈춥니다")
            self.btn_stop.setEnabled(False)

    @property
    def running(self) -> bool:
        return self.worker is not None

    def _on_progress(self, done: int, total: int, status: str, reason: str) -> None:
        self.bar.setRange(0, total)
        self.bar.setValue(done)
        self.progress_label.setText(f"{done}/{total}")
        if status != "ok":
            self._append(f"[{done - 1:06d}] {status}: {reason}")

    def _on_finished(self, summary: runner.RunSummary) -> None:
        self.last_summary = summary
        self.session.last_summary = summary
        self._append(summary_text(summary))
        for w in summary.warnings:
            self._append(f"경고: {w}")
        self.progress_label.setText("취소됨" if summary.cancelled else "완료")
        self.btn_open_out.setEnabled(True)
        self.status.emit(
            f"배치 {'취소' if summary.cancelled else '완료'}: ok {summary.writer.n_ok} · skipped {summary.writer.n_skipped} → {summary.writer.root.as_posix()}"
        )
        self.run_finished.emit(summary)

    def _on_failed(self, msg: str) -> None:
        self._append(f"실패: {msg}")
        self.progress_label.setText("실패")
        self.status.emit(f"배치 실패: {msg.splitlines()[0]}")
        self.run_finished.emit(None)

    def _on_thread_done(self) -> None:
        w = self.worker
        self.worker = None
        if w is not None:
            w.deleteLater()
        self.btn_run.setEnabled(self.session.recipe is not None)
        self.btn_stop.setEnabled(False)

    def open_output(self) -> None:
        root = self.last_summary.writer.root if self.last_summary else Path(self.out.text().strip())
        if root and Path(root).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(root).resolve())))

    def _append(self, text: str) -> None:
        self.log.appendPlainText(text)

    def wait(self, ms: int = 60000) -> bool:
        """테스트용 — 실행 스레드가 끝날 때까지."""
        return self.worker is None or self.worker.wait(ms)

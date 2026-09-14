"""``PreviewWorker(QThread)`` — 큐(``LatestOnlyQueue``)에서 작업을 꺼내 ``jobs.run_job``을 돌리고 시그널로 돌려준다.

- 스레드는 하나. 예외는 ``failed(JobError)``로 올리고 스레드는 죽지 않는다.
- ``submit(job)``: 같은 키의 대기 작업을 교체. ``invalidate(generation)``: 지난 세대의 대기 작업 폐기(실행 중인 것은 완료 후
  결과의 ``job.generation``으로 탭이 걸러낸다).
- ``Prepared``는 스레드 사이에 공유되지만 워커만 쓰는 동안 GUI는 읽기만 한다(pydantic frozen · numpy 읽기 전용 사용).
"""

from __future__ import annotations

import traceback

from PySide6.QtCore import QMutex, QThread, QWaitCondition, Signal

from anograft import runner
from anograft.gui.studio.jobs import (
    KIND_PREPARE,
    JobError,
    LatestOnlyQueue,
    PreviewJob,
    PreviewResult,
    ThumbResult,
    run_job,
)


class PreviewWorker(QThread):
    prepared = Signal(object)  # runner.Prepared
    finished_preview = Signal(object)  # PreviewResult
    finished_thumb = Signal(object)  # ThumbResult
    failed = Signal(object)  # JobError
    busy = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._queue = LatestOnlyQueue()
        self._mutex = QMutex()
        self._cond = QWaitCondition()
        self._stop = False
        self._prep: runner.Prepared | None = None

    # ------------------------------------------------------------------ GUI 스레드에서 부르는 것

    def set_prepared(self, prep: runner.Prepared | None) -> None:
        self._mutex.lock()
        self._prep = prep
        self._mutex.unlock()

    def submit(self, job: PreviewJob) -> None:
        self._mutex.lock()
        self._queue.put(job)
        self._cond.wakeOne()
        self._mutex.unlock()

    def invalidate(self, generation: int) -> int:
        self._mutex.lock()
        n = self._queue.drop_before(generation)
        self._mutex.unlock()
        return n

    def pending(self) -> int:
        self._mutex.lock()
        n = len(self._queue)
        self._mutex.unlock()
        return n

    def stop(self, wait_ms: int = 5000) -> None:
        self._mutex.lock()
        self._stop = True
        self._queue.clear()
        self._cond.wakeAll()
        self._mutex.unlock()
        self.wait(wait_ms)

    # ------------------------------------------------------------------ 워커 스레드

    def run(self) -> None:
        while True:
            self._mutex.lock()
            while not self._stop and len(self._queue) == 0:
                self._cond.wait(self._mutex)
            if self._stop:
                self._mutex.unlock()
                return
            job = self._queue.pop()
            prep = self._prep
            self._mutex.unlock()
            if job is None:
                continue
            self.busy.emit(True)
            try:
                out = run_job(prep, job)
            except Exception as e:  # 작업 하나의 실패가 스레드를 죽이면 안 된다
                msg = f"{type(e).__name__}: {e}"
                if not isinstance(e, runner.PrepareError):
                    msg += "\n" + traceback.format_exc(limit=3)
                self.failed.emit(JobError(job, msg))
            else:
                if job.kind == KIND_PREPARE:
                    self.set_prepared(out)
                    self.prepared.emit(out)
                elif isinstance(out, ThumbResult):
                    self.finished_thumb.emit(out)
                elif isinstance(out, PreviewResult):
                    self.finished_preview.emit(out)
            finally:
                self.busy.emit(self.pending() > 0)

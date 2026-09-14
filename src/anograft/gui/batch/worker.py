"""``BatchWorker(QThread)`` — 실행 한 번 = 스레드 한 번. ``run_batch`` 를 돌리며 진행·경고를 시그널로, 끝나면 ``finished_run``
(``RunSummary``) 또는 ``failed(str)``. ``request_stop()`` 은 ``should_stop`` 플래그 — 다음 결과에서 멈춘다(spawn 풀은 종료).

runner 의 콜백은 워커 스레드에서 불리므로 여기서 시그널로만 바꾼다(위젯을 만지지 않는다).
"""

from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QThread, Signal

from anograft.core import recipe as R
from anograft.gui.batch.session import BatchError, run_batch


class BatchWorker(QThread):
    progress = Signal(int, int, str, str)  # done, total, status, reason
    warning = Signal(str)
    finished_run = Signal(object)  # runner.RunSummary
    failed = Signal(str)

    def __init__(self, recipe: R.Recipe, workers: int, parent=None) -> None:
        super().__init__(parent)
        self.recipe = recipe
        self.workers = workers
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def run(self) -> None:
        try:
            summary = run_batch(
                self.recipe,
                workers=self.workers,
                progress=lambda d, t, r: self.progress.emit(d, t, r.status, r.reason or ""),
                warn=self.warning.emit,
                should_stop=self._stop.is_set,
            )
        except BatchError as e:
            self.failed.emit(str(e))
        except RuntimeError as e:
            if "bootstrapping phase" in str(e):
                # spawn 풀을 메인 모듈 가드 없이 만들었다 — 스크립트에서 GUI 를 띄운 경우. 안내로 바꾼다
                self.failed.emit(
                    "spawn 워커를 만들 수 없습니다(메인 모듈에 `if __name__ == '__main__'` 가드가 없음) — "
                    "워커를 0 으로 두거나 `anograft-gui` / `python -m anograft.gui` 로 실행하세요"
                )
            else:
                self.failed.emit(f"RuntimeError: {e}\n" + traceback.format_exc(limit=3))
        except Exception as e:  # 예상 밖 — 메시지 + 짧은 트레이스, 스레드는 정상 종료
            self.failed.emit(f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=3))
        else:
            self.finished_run.emit(summary)

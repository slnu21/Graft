"""일괄 생성 API (U6) — `anograft.batch` 를 **감싸기만** 한다.

앞선 화면들과 다른 점 하나: **오래 걸리는 작업**이다(수백 장 × 수백 ms). Qt 는 `QThread` + 시그널이지만
웹은 스트리밍을 두지 않기로 했으므로(U5 확정) 서버가 **스레드 하나**에서 돌리고 화면은 `state` 를
**짧게 폴링**한다. 진행 상황은 스레드가 `_Run` 에 쌓고, 폴링은 그 스냅샷을 읽는다.

지켜야 할 것 셋:

* **실행 스레드는 API 락을 잡지 않는다.** 잡으면 도는 동안 `state` 폴링이 막혀 화면이 얼어붙는다.
  세션(설정)은 `_LOCK`, 진행 상황은 `_Run` 자신의 락으로 나눠 지킨다.
* **`runner.run` 이 하는 일을 여기서 다시 하지 않는다** — 진행·경고·취소는 이미 콜백으로 나온다.
  `batch.run_batch` 가 그 콜백을 받는 순수 함수이므로 스레드는 그것만 부른다.
* **디스크에 쓰는 작업**이라 시작은 POST(쓰기)다. 링크 한 번에 수백 장이 써지면 안 된다.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any

from anograft import batch, recent, runner
from anograft.core import recipe as R
from anograft.web.api import ApiResult, Handler, Request, register

#: 화면에 돌려주는 로그 줄 수 상한 — 수백 장이면 줄이 길어진다(오래된 줄은 버린다).
LOG_LIMIT = 400

#: 세션 락. **재진입 가능(RLock)** — `_serialized` 로 감싼 핸들러가 안에서 또 잠그는 실수가
#: `Lock` 이면 조용한 교착이 된다(U6 에서 `_start` 가 실제로 그랬다).
_LOCK = threading.RLock()
_SESSION: batch.BatchSession | None = None
_RUN: _Run | None = None  # type: ignore[name-defined]
_REGISTERED = False


class _Run:
    """실행 한 번. 스레드가 쓰고 폴링이 읽는다 — 그래서 **자기 락**을 들고 다닌다."""

    def __init__(self, recipe: R.Recipe, workers: int) -> None:
        self.recipe = recipe
        self.workers = workers
        self.total = int(recipe.output.count)
        self.done = 0
        self.log: list[str] = []
        self.warnings: set[str] = set()
        self.summary: runner.RunSummary | None = None
        self.error: str | None = None
        self.finished = False
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="anograft-batch", daemon=True)

    # ------------------------------------------------------------ 스레드에서

    def _append(self, line: str) -> None:
        with self._lock:
            self.log.append(line)
            if len(self.log) > LOG_LIMIT:
                del self.log[: len(self.log) - LOG_LIMIT]

    def _on_progress(self, done: int, total: int, result: Any) -> None:
        with self._lock:
            self.done, self.total = done, total
        if result.status != "ok":
            self._append(f"[{done - 1:06d}] {result.status}: {result.reason or ''}")

    def _on_warn(self, message: str) -> None:
        with self._lock:
            self.warnings.add(message)
        self._append(f"경고: {message}")

    def _run(self) -> None:
        try:
            summary = batch.run_batch(
                self.recipe,
                workers=self.workers,
                progress=self._on_progress,
                warn=self._on_warn,
                should_stop=self._stop.is_set,
            )
        except batch.BatchError as exc:
            self.error = str(exc)
        except RuntimeError as exc:
            # spawn 풀을 메인 모듈 가드 없이 만들었다 — `anograft serve` 가 아닌 방식으로 띄운 경우
            self.error = (
                "spawn 워커를 만들 수 없습니다(메인 모듈 가드 없음) — 워커를 0 으로 두세요"
                if "bootstrapping phase" in str(exc)
                else f"RuntimeError: {exc}"
            )
        except Exception as exc:  # fail-soft — 서버는 살아 있어야 한다
            self.error = f"{type(exc).__name__}: {exc}\n" + traceback.format_exc(limit=3)
        else:
            self.summary = summary
            self._append(batch.summary_text(summary))
            for w in summary.warnings:  # 실행 중 스트리밍된 것 말고 새 경고만(writer 요약 등)
                if w not in self.warnings:
                    self._append(f"경고: {w}")
        finally:
            if self.error:
                self._append(f"실패: {self.error}")
            self.finished = True

    # ------------------------------------------------------------ 폴링에서

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set() and not self.finished

    def snapshot(self) -> dict:
        with self._lock:
            done, total, log = self.done, self.total, list(self.log)
        s = self.summary
        return {
            "running": not self.finished,
            "stopping": self.stopping,
            "done": done,
            "total": total,
            "log": log,
            "error": self.error,
            "summary": _summary_json(s) if s is not None else None,
        }


def _summary_json(s: runner.RunSummary) -> dict:
    w = s.writer
    return {
        "root": w.root.as_posix(),
        "nOk": w.n_ok,
        "nSkipped": w.n_skipped,
        "nNormals": w.n_normals,
        "nFallback": w.n_fallback,
        "perClass": dict(w.per_class),
        "files": dict(w.files),
        "warnings": list(s.warnings),
        "cancelled": s.cancelled,
        "done": s.done,
        "count": s.count,
        "text": batch.summary_text(s),
    }


def reset() -> None:
    """테스트·재시작용. 도는 스레드가 있으면 멈추라고 이르고 기다린다(파일을 쓰는 중이므로 죽이지 않는다)."""
    global _SESSION, _RUN
    run = _RUN
    if run is not None and not run.finished:
        run.stop()
        run.thread.join(timeout=30)
    with _LOCK:
        _SESSION = None
        _RUN = None


def _session() -> batch.BatchSession:
    if _SESSION is None:
        raise batch.BatchError("먼저 레시피를 여세요.")
    return _SESSION


# ---------------------------------------------------------------- 상태


def _state_payload(session: batch.BatchSession) -> dict:
    r = session.recipe
    out: dict[str, Any] = {
        "open": r is not None,
        "recipePath": session.recipe_path.as_posix() if session.recipe_path else "",
        "writerFormats": list(batch.WRITER_FORMATS),
        "settings": {
            "out": session.out,
            "count": session.count,
            "seed": session.seed,
            "workers": session.workers,
            "writer": session.writer,
            "mvtecCategory": session.mvtec_category,
        },
        "runCommand": session.run_command(),
        "run": _RUN.snapshot() if _RUN is not None else None,
    }
    if r is not None:
        out["recipe"] = {
            "name": r.name,
            "preset": r.pipeline.preset or "",
            "bank": r.inputs.bank_key(),
            "targets": r.inputs.targets.as_posix(),
            "defectsPerImage": list(r.output.defects_per_image),
            "includeNormals": r.output.include_normals,
        }
    return out


def _state(_req: Request) -> ApiResult:
    """화면이 **짧게 폴링**하는 자리 — 실행 중에도 막히지 않아야 한다(실행 스레드는 이 락을 안 잡는다)."""
    if _SESSION is None:
        return ApiResult(200, {"open": False, "writerFormats": list(batch.WRITER_FORMATS)})
    return ApiResult(200, _state_payload(_SESSION))


# ---------------------------------------------------------------- 쓰기


def _open(req: Request) -> ApiResult:
    """레시피 파일로 연다. 미리보기 화면에서 저장한 레시피를 그대로 받는 자리이기도 하다."""
    global _SESSION
    path = str(req.json.get("recipe", "")).strip()
    if not path:
        return ApiResult(400, {"error": "레시피 파일 경로가 필요합니다."})
    session = batch.BatchSession()
    try:
        session.load(path)
    except batch.BatchError as exc:
        return ApiResult(400, {"error": str(exc)})
    with _LOCK:
        _SESSION = session
    recent.remember("recipe", path)
    return ApiResult(200, _state_payload(session))


def _settings(req: Request) -> ApiResult:
    """실행 오버라이드(출력·장수·시드·워커·형식) — CLI `run --out --count --seed` 와 같은 자리."""
    session = _session()
    s = req.json.get("settings")
    if not isinstance(s, dict):
        return ApiResult(400, {"error": "설정이 필요합니다."})
    try:
        if "out" in s:
            session.out = str(s["out"]).strip()
        if "count" in s:
            session.count = int(s["count"])
        if "seed" in s:
            session.seed = int(s["seed"])
        if "workers" in s:
            session.workers = max(0, int(s["workers"]))
        if "writer" in s:
            session.writer = str(s["writer"])
        if "mvtecCategory" in s:
            session.mvtec_category = str(s["mvtecCategory"]).strip() or "graft"
    except (TypeError, ValueError) as exc:
        return ApiResult(400, {"error": f"값을 읽지 못했습니다: {exc}"})
    return ApiResult(200, _state_payload(session))


def _start(req: Request) -> ApiResult:
    """실행 시작 — **디스크에 쓴다**. 이미 돌고 있으면 거절한다(한 번에 하나)."""
    global _RUN
    session = _session()
    if _RUN is not None and not _RUN.finished:
        return ApiResult(409, {"error": "이미 일괄 생성이 돌고 있습니다."})
    try:
        recipe = session.build_recipe()
    except batch.BatchError as exc:
        return ApiResult(400, {"error": str(exc)})
    run = _Run(recipe, session.workers)
    run._append(f"$ {session.run_command()}")
    run._append(
        f"레시피 {recipe.name} · seed {recipe.seed} · {recipe.output.count}장 → "
        f"{recipe.output.root.as_posix()} ({recipe.output.writer.format})"
    )
    _RUN = run  # 이 핸들러는 이미 `_serialized` 안이다 — 여기서 또 잠그지 않는다
    run.thread.start()
    # 출력 폴더는 곧 ⑤ 검수에서 열 곳이다 — 그때 다시 치지 않게 기억해 둔다(U7).
    recent.remember("output", recipe.output.root.as_posix())
    return ApiResult(200, _state_payload(session))


def _stop(_req: Request) -> ApiResult:
    """중지 요청 — `runner.run` 이 **다음 결과에서** 멈추고 그때까지의 파일·manifest 는 남는다."""
    session = _session()
    if _RUN is None or _RUN.finished:
        return ApiResult(400, {"error": "돌고 있는 일괄 생성이 없습니다."})
    _RUN.stop()
    return ApiResult(200, _state_payload(session))


# ---------------------------------------------------------------- 인계 (U7)


def adopt(recipe: R.Recipe, path: Path | None) -> dict:
    """③ 미리보기에서 넘어온 레시피를 받는다 — Qt ``BatchTab.set_recipe`` 와 같은 자리.

    세션을 새로 만들지 않고 **지금 것에 얹는다**: 출력·장수·시드·형식은 레시피 값으로 되돌아가지만
    워커 수처럼 레시피 밖 설정은 사람이 맞춰 둔 대로 남는다(Qt 와 같다).

    **돌고 있는 중이면 거절한다** — 설정이 발밑에서 바뀌면 진행 중인 실행을 설명하는 화면이
    거짓말을 하게 된다(취소·재시작은 사람이 정한다).
    """
    global _SESSION
    with _LOCK:
        if _RUN is not None and not _RUN.finished:
            raise batch.BatchError("일괄 생성이 돌고 있습니다 — 끝난 뒤에 보내세요.")
        session = _SESSION or batch.BatchSession()
        session.set_recipe(recipe, path)
        _SESSION = session
        return _state_payload(session)


def _serialized(handler: Handler) -> Handler:
    """세션을 만지는 핸들러만 직렬화한다. **실행 스레드는 이 락을 잡지 않는다** — 잡으면 폴링이 막힌다."""

    def wrapped(req: Request) -> ApiResult:
        with _LOCK:
            return handler(req)

    wrapped.__name__ = handler.__name__
    wrapped.__doc__ = handler.__doc__
    return wrapped


def ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    register("/api/batch/state", _state)  # 폴링 — 락 없이(스냅샷만 읽는다)
    register("/api/batch/open", _open, write=True)
    register("/api/batch/settings", _serialized(_settings), write=True)
    register("/api/batch/start", _serialized(_start), write=True)
    register("/api/batch/stop", _serialized(_stop), write=True)
    _REGISTERED = True


def output_root() -> Path | None:
    """마지막 실행의 출력 루트 — 검수 화면으로 넘길 때 화면이 쓴다."""
    if _RUN is not None and _RUN.summary is not None:
        return _RUN.summary.writer.root
    return None

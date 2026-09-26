"""학습 루프 API (U8) — `anograft.loop` 을 **감싸기만** 한다.

읽기 셋(`state`·`rounds`·`config`) + 쓰기 하나(`open`)뿐이다. **라운드를 웹에서 돌리지 않는다** —
한 바퀴는 학습기를 프로세스로 띄워 몇 시간을 돌고, 그것을 부르는 자리는 이미 `anograft loop run` 과
스케줄러가 부르는 `loop tick` 이다(설계 §2b.1: Graft 는 상주하지 않는다). 그래서 이 화면은 **지금
어디인가를 보여 주고, 사람이 할 차례면 그 자리로 보내 준다**(검토 대기 → ⑤ 검수 화면).

판단은 전부 파이썬 쪽에 이미 있다: 트리거 `policy.should_start_round` · 자동 정지 `policy.circuit_break`
· 사람 수정률 `policy.correction_window` · 표와 추이 모양 `loop.board`. 여기서 새로 하는 계산은 없다.

상태: 다른 화면과 같이 **서버가 세션 하나**를 든다(로컬 1인용). 다만 여기 세션은 열어 둔 `loop.yaml`
하나뿐이고, 화면이 보는 사실은 **매번 디스크에서** 읽는다 — 루프는 이 서버 밖(스케줄러·CLI)에서도
돌기 때문에 캐시를 들면 화면이 옛 라운드를 보여 준다.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from anograft import recent
from anograft.loop import board
from anograft.loop import ledger as L
from anograft.web.api import ApiResult, Handler, Request, register

#: 재진입 가능 — `_serialized` 안에서 또 잠그는 실수를 조용한 교착으로 만들지 않는다(U6 에서 겪었다).
_LOCK = threading.RLock()
_LOOP: Any = None  # ResolvedLoop — 지연 import 라 타입을 묶지 않는다
_PATH: Path | None = None
_REGISTERED = False


def reset() -> None:
    """테스트가 서버 상태를 지울 때."""
    global _LOOP, _PATH
    with _LOCK:
        _LOOP = None
        _PATH = None


def _loop() -> Any:
    if _LOOP is None:
        from anograft.loop.round import LoopError

        raise LoopError("먼저 loop.yaml 을 여세요.")
    return _LOOP


# ---------------------------------------------------------------- 상태


def _champion_json(state: Any) -> dict | None:
    champ = state.champion
    return None if champ is None else champ.to_dict()


def _round_json(status: Any, labels: dict[str, str]) -> dict | None:
    """진행 중(또는 마지막) 라운드 — 단계 줄까지 함께. 라운드가 없으면 ``None``."""
    record = status.record
    if record is None:
        return None
    steps = board.phase_steps(record.phases, record.done, labels)
    nxt = status.next
    return {
        "number": record.number,
        "bootstrap": record.bootstrap,
        "started": record.started,
        "updated": record.updated,
        "finished": record.finished,
        "next": nxt,
        "nextLabel": labels.get(nxt or "", "") if nxt else "",
        "steps": [s.to_json() for s in steps],
        "warnings": list(record.warnings),
    }


def _review_json(loop: Any, status: Any) -> dict:
    """사람 판정 자리 — **검수 화면이 그대로 여는 폴더**를 함께 준다(T5 의 요점: 새 포맷 0)."""
    from anograft.loop.round import round_name

    record = status.record
    queue_dir: Path | None = None
    if record is not None:
        candidate = loop.out / round_name(record.number) / "queue"
        if candidate.is_dir():
            queue_dir = candidate
    return {
        "judged": status.judged,
        "total": status.total,
        "waiting": status.next == "review" and status.total > 0 and status.judged < status.total,
        "queueDir": queue_dir.as_posix() if queue_dir else "",
    }


def _state_payload(loop: Any, path: Path | None) -> dict:
    from anograft.loop.round import (
        PHASE_LABEL,
        bank_facts,
        breaker_history,
        ledger_path,
        status,
    )

    st = status(loop)
    led = L.read(ledger_path(loop))
    rows = board.round_rows(led)
    facts = bank_facts(loop)  # 지금 보관함(라운드 뒤에 사람이 넣은 것도 들어 있다)
    last = rows[0] if rows else None
    outcomes = breaker_history(loop, led)
    from anograft.loop.policy import correction_window

    corrections = correction_window(outcomes)
    review = _review_json(loop, st)
    action = board.next_action(
        breaker=st.breaker,
        waiting_review=bool(review["waiting"]),
        judged=st.judged,
        total=st.total,
        lock=st.lock,
        trigger=st.trigger,
        has_round=st.record is not None,
        config_path=path.as_posix() if path else "",
    )
    per_class = {str(k): int(v) for k, v in (facts.get("bank_per_class") or {}).items()}
    return {
        "open": True,
        "configPath": path.as_posix() if path else "",
        "out": loop.out.as_posix(),
        "bankPath": loop.bank.as_posix(),
        "recipePath": loop.recipe.as_posix(),
        "fieldPath": loop.field.as_posix() if loop.field else "",
        "trainer": loop.config.trainer,
        "metricName": loop.config.promote.metric,
        # 라벨·풀이는 서버가 준다 — 용어의 한 원천은 파이썬 쪽이다(프론트에 사전 사본을 두지 않는다)
        "metricLabel": board.metric_label(loop.config.promote.metric)[0],
        "metricHint": board.metric_label(loop.config.promote.metric)[1],
        "champion": _champion_json(st.state),
        "round": _round_json(st, PHASE_LABEL),
        "review": review,
        "trigger": (
            None if st.trigger is None else {"start": st.trigger.start, "reason": st.trigger.reason}
        ),
        "breaker": (
            None
            if st.breaker is None
            else {
                "tripped": st.breaker.tripped,
                "reason": st.breaker.reason,
                "kinds": list(st.breaker.kinds),
            }
        ),
        "tick": st.tick.to_json() if st.tick else None,
        "lock": st.lock.to_json() if st.lock else None,
        "lockText": st.lock.text() if st.lock else "",
        "processed": st.processed,
        "action": action.to_json(),
        "corrections": {
            "rate": corrections.rate,
            "drafted": corrections.drafted,
            "corrected": corrections.corrected,
            "text": corrections.text(),
        },
        "bank": {
            "sources": sum(per_class.values()),
            "classes": list(facts.get("bank_classes") or []),
            "perClass": per_class,
            # 증가분은 **마지막 라운드가 적어 둔 분포와의 차이**다(트리거의 "새 조각"과 같은 셈법 — 보관함에
            # 추가 시각이 없어서 이렇게 센다). 앞 라운드가 없으면 delta 는 null(0 이라고 말하지 않는다).
            "rows": [
                r.to_json() for r in board.class_rows(per_class, last.per_class if last else None)
            ],
        },
        "failures": [{"round": e.round, "phase": e.phase, "reason": e.reason} for e in st.failures],
        "warnings": list(led.warnings),
    }


def _state(_req: Request) -> ApiResult:
    """화면이 여는 순간과 **돌고 있는 동안** 부르는 자리. 열기 전에도 200 으로 답한다(안내를 띄운다)."""
    if _LOOP is None:
        from anograft.loop.config import find_loop_file

        found = find_loop_file()
        return ApiResult(
            200,
            {
                "open": False,
                # 찾아 둔 기본 후보(cwd/loop.yaml → ~/.anograft/loop.yaml) — 사람이 경로를 몰라도 열 수 있게
                "suggest": found.as_posix() if found else "",
                "example": "loop.example.yaml",
            },
        )
    return ApiResult(200, _state_payload(_LOOP, _PATH))


def _rounds(req: Request) -> ApiResult:
    """라운드 기록 — 원장 `rounds.jsonl` 한 줄이 표 한 행. **최신이 위**, 추이 점은 옛것부터."""
    from anograft.loop.round import ledger_path

    loop = _loop()
    led = L.read(ledger_path(loop))
    rows = board.round_rows(led, limit=req.int_of("limit", board.DEFAULT_LIMIT))
    return ApiResult(
        200,
        {
            "rounds": [r.to_json() for r in rows],
            "points": [p.to_json() for p in board.metric_points(rows)],
            "metricName": loop.config.promote.metric,
            "metricLabel": board.metric_label(loop.config.promote.metric)[0],
            "metricHint": board.metric_label(loop.config.promote.metric)[1],
            "ledger": ledger_path(loop).as_posix(),
            "warnings": list(led.warnings),
        },
    )


# ---------------------------------------------------------------- 쓰기


def _open(req: Request) -> ApiResult:
    """`loop.yaml` 을 연다. 경로가 비면 CLI 와 **같은 순서**로 찾는다(cwd → `~/.anograft`)."""
    global _LOOP, _PATH
    from anograft.loop.config import LoopConfigError, find_loop_file, load_loop_config

    raw = str(req.json.get("config", "")).strip()
    path = Path(raw) if raw else find_loop_file()
    if path is None:
        return ApiResult(
            400,
            {
                "error": "loop.yaml 을 찾지 못했습니다 — 경로를 적거나 loop.example.yaml 을 복사해 만드세요."
            },
        )
    try:
        loop = load_loop_config(path)
    except LoopConfigError as exc:
        return ApiResult(400, {"error": str(exc)})
    with _LOCK:
        _LOOP = loop
        _PATH = Path(path)
    recent.remember("loop", Path(path).as_posix())
    return ApiResult(200, _state_payload(loop, Path(path)))


def _serialized(handler: Handler) -> Handler:
    """세션을 보는 핸들러를 직렬화한다(세션 교체 중에 반쯤 바뀐 상태를 읽지 않게)."""

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
    register("/api/loop/state", _serialized(_state))
    register("/api/loop/rounds", _serialized(_rounds))
    register("/api/loop/open", _open, write=True)  # 자체 락으로 세션을 바꾼다
    _REGISTERED = True

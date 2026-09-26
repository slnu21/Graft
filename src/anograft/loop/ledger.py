"""루프 원장 — 라운드에 무슨 일이 있었는지 **더하기만 하는** 기록 (설계 §2b.4, 작업 단위 T14).

`.atlas/ledger.jsonl` 과 같은 정신이다: 한 줄이 사실 하나이고, 고치지 않고 **더한다**. T10 이 남긴 상태 두
파일 중 **이력**이 이쪽으로 승격됐다 —

- ``<out>/rounds.jsonl`` — 라운드 시작 · 단계 완료 · 끝(지표·승급·champion·해시·은행 스냅샷) · 실패 · 안 돎.
  **중단·재개·감사·조회·롤백이 전부 이 파일에서 나온다.** 사람이 읽을 수 있어야 한다.
- ``<out>/loop.state.json`` — 남는 것은 **움직이는 포인터 둘**뿐이다(지금 라운드 번호 · champion).
  롤백이 포인터를 되돌리는 일이라 이건 덮어쓰는 파일이어야 하고, 이력은 원장이 든다.
- ``<out>/round-NNN/round.json`` — 그 라운드의 `done` 목록(부분 진행 재사용의 진실). 원장은 감사용이고
  **다음 단계를 고르는 것은 여전히 `round.json`** 이다 — 재생(replay)으로 상태를 복원하는 구조를 만들면
  한 줄이 깨질 때마다 루프가 멈춘다.

그리고 ``<out>/tick.json`` — **마지막 tick 한 번**의 기록(돌았나 · 안 돌았으면 왜). 스케줄러가 5분마다
부르는 것을 원장에 다 적으면 원장이 tick 으로 덮이므로, 같은 사유가 이어지는 동안은 이 파일만 갱신하고
**사유가 바뀔 때만** 원장에 `skipped` 를 남긴다("조용히 안 도는 루프가 제일 나쁘다"는 규율은 지키면서).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: 라운드 원장 파일 이름.
ROUNDS_FILE = "rounds.jsonl"
#: 마지막 tick 기록 파일 이름.
TICK_FILE = "tick.json"

EVENT_ROUND_START = "round_start"
EVENT_PHASE = "phase"
EVENT_ROUND_END = "round_end"
EVENT_FAILED = "failed"
EVENT_SKIPPED = "skipped"
#: 기준선 리셋(클래스 신설 등) — **T15 가 쓴다**. 이 지점 앞뒤로 Δ 를 비교하지 않는다(설계 §2b.5(3)).
EVENT_BASELINE_RESET = "baseline_reset"
#: **자동 정지 해제**(T12) — 사람이 "무엇을 바꿨는지" 적고 다시 돌린 지점. 이 앞의 라운드는 자동 정지
#: 판정에서 빠진다(기준선 재설정과 같은 규율 — 조건이 바뀌었으면 그 앞과 견주지 않는다).
EVENT_BREAKER_RESET = "breaker_reset"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class Event:
    """원장 한 줄. 정해진 자리는 `event`·`at`·`round` 뿐이고 나머지는 `data` 에 그대로 담는다.

    스키마를 좁게 못 박지 않는 이유: T12(자동 정지 지표) · T15(`baseline_reset`) · T16(롤링 골든)이 각자
    필드를 더할 것이고, **읽는 쪽이 모르는 키를 만나도 깨지지 않아야** 원장이 오래 산다. T12 가 실제로
    `round_end` 에 `intake`·`drafted`·`corrected`·`bank_per_class` 를 더했고, 옛 줄에 그 키가 없으면
    판정하지 않는다(모르면 막지 않는다).
    """

    event: str
    at: str = ""
    round: int = 0
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def reason(self) -> str:
        return str(self.data.get("reason", "") or "")

    @property
    def phase(self) -> str:
        return str(self.data.get("phase", "") or "")

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"at": self.at or _now(), "event": self.event}
        if self.round:
            out["round"] = self.round
        out.update(self.data)
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Event | None:
        """모르는/깨진 줄은 ``None`` — 원장 한 줄이 상해도 나머지는 읽힌다(fail-soft)."""
        name = str(data.get("event", "") or "")
        if not name:
            return None
        rest = {k: v for k, v in data.items() if k not in ("event", "at", "round")}
        try:
            number = int(data.get("round", 0) or 0)
        except (TypeError, ValueError):
            number = 0
        return cls(event=name, at=str(data.get("at", "") or ""), round=number, data=rest)


def append(
    path: str | Path, event: str, *, round_no: int = 0, at: str | None = None, **fields: Any
) -> Event:
    """원장에 한 줄 더한다. 실패는 호출부가 경고로 삼는다(원장 때문에 라운드를 죽이지 않는다)."""
    record = Event(event=event, at=at or _now(), round=round_no, data=dict(fields))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
    return record


@dataclass
class Ledger:
    """읽어 온 원장 — 순서는 적힌 순서 그대로(시간순이라고 믿지 않는다: 사람이 손으로 고칠 수 있다)."""

    events: list[Event] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.events)

    def of(self, event: str) -> list[Event]:
        return [e for e in self.events if e.event == event]

    def last(self, event: str) -> Event | None:
        for e in reversed(self.events):
            if e.event == event:
                return e
        return None

    def for_round(self, number: int) -> list[Event]:
        return [e for e in self.events if e.round == number]

    @property
    def last_end(self) -> Event | None:
        """마지막으로 **끝난** 라운드 — 트리거의 "마지막 라운드로부터 몇 시간"이 여기서 나온다."""
        return self.last(EVENT_ROUND_END)

    def history(self, limit: int = 5) -> list[dict[str, Any]]:
        """`loop status` 가 찍는 최근 라운드 줄들(옛 `loop.state.json` 의 `history` 자리)."""
        out: list[dict[str, Any]] = []
        for e in self.of(EVENT_ROUND_END)[-limit:]:
            out.append(
                {
                    "round": e.round,
                    "metric": e.get("metric"),
                    "promoted": bool(e.get("promoted")),
                    "reason": e.reason,
                    "finished": e.at,
                }
            )
        return out

    def failures(self, limit: int = 3) -> list[Event]:
        return self.of(EVENT_FAILED)[-limit:]


def read(path: str | Path) -> Ledger:
    """원장을 읽는다. 없으면 빈 것, 깨진 줄은 사유와 함께 건너뛴다."""
    p = Path(path)
    if not p.is_file():
        return Ledger()
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        return Ledger(warnings=[f"{p} 를 읽을 수 없습니다: {exc}"])
    events: list[Event] = []
    warns: list[str] = []
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            data = json.loads(line)
        except ValueError:
            warns.append(f"{p.name} {n}번째 줄을 읽지 못했습니다(건너뜀)")
            continue
        parsed = Event.from_json(data) if isinstance(data, Mapping) else None
        if parsed is None:
            warns.append(f"{p.name} {n}번째 줄에 event 가 없습니다(건너뜀)")
            continue
        events.append(parsed)
    return Ledger(events=events, warnings=warns)


# --------------------------------------------------------------------------------------
# 마지막 tick — 스케줄러가 부른 흔적 하나
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tick:
    """`tick.json` — 마지막으로 불렸을 때 돌았나, 안 돌았으면 왜."""

    at: str = ""
    ran: bool = False
    reason: str = ""
    round: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "at": self.at or _now(),
            "ran": self.ran,
            "reason": self.reason,
            "round": self.round,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Tick:
        return cls(
            at=str(data.get("at", "") or ""),
            ran=bool(data.get("ran", False)),
            reason=str(data.get("reason", "") or ""),
            round=int(data.get("round", 0) or 0),
        )

    def line(self) -> str:
        when = self.at or "(시각 없음)"
        head = "돌았습니다" if self.ran else "돌지 않았습니다"
        return (
            f"마지막 tick {when} — {head}: {self.reason}"
            if self.reason
            else f"마지막 tick {when} — {head}"
        )


def read_tick(path: str | Path) -> Tick | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return Tick.from_json(data) if isinstance(data, Mapping) else None


def write_tick(path: str | Path, tick: Tick) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(tick.to_json(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return p


__all__ = [
    "EVENT_BASELINE_RESET",
    "EVENT_BREAKER_RESET",
    "EVENT_FAILED",
    "EVENT_PHASE",
    "EVENT_ROUND_END",
    "EVENT_ROUND_START",
    "EVENT_SKIPPED",
    "ROUNDS_FILE",
    "TICK_FILE",
    "Event",
    "Ledger",
    "Tick",
    "append",
    "read",
    "read_tick",
    "write_tick",
]

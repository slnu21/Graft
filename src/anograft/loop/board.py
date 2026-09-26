"""루프 현황판 — 원장에 쌓인 사실을 **화면이 그릴 모양**으로 바꾼다 (작업 단위 U8 `web-loop`).

`loop status` 가 찍는 것과 **같은 사실**을 쓴다. 다른 것은 모양뿐이다: 터미널은 줄이고 화면은 표와 추이라
행이 필요하다. 그래서 여기서 **판단을 새로 하지 않는다** — 승급은 `policy.should_promote`, 트리거는
`policy.should_start_round`, 자동 정지는 `policy.circuit_break`, 사람 수정률은 `policy.correction_window`
가 이미 답을 냈고 이 모듈은 그 답을 행으로 옮긴다.

**순수하다**(파일·Qt·HTTP 를 모른다) — 읽기는 `round.py`·`ledger.py` 가 하고 그 결과를 받는다. 그래서
웹 화면이든 나중에 생길 Qt 탭이든 같은 행을 보고, 테스트가 원장 한 줄을 손으로 만들어 넣을 수 있다.

주의 둘:

* **행은 최신이 위**다(표는 사람이 위부터 읽는다). 추이 점은 반대로 **옛것부터**다(선을 왼쪽부터 긋는다).
  둘을 한 함수가 내지 않고 이름으로 갈라 두었다 — 섞이면 그래프가 거꾸로 그려진다.
* **기준선 재설정·자동 정지 해제 지점은 선을 끊는다**(`segment`). 그 앞뒤는 견주지 않기로 한 구간이라
  이어 그리면 화면이 "떨어졌다/올랐다"는 거짓말을 한다(설계 §2b.5(3)).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from anograft.core.classes import UNSORTED
from anograft.loop import ledger as L
from anograft.loop.policy import CorrectionStats, RoundOutcome, correction_window

#: 화면이 기본으로 보여 주는 라운드 수. 원장은 계속 자라므로 표는 끝에서 잘라 본다.
DEFAULT_LIMIT = 12

#: 지표 이름 → (화면 라벨, 풀이). 용어 사전 §3.5 — ``mAP50`` 은 지표의 고유명이라 남기되 **풀이가
#: 필수**다(압축 기호 금지 원칙의 예외). 모르는 이름은 그대로 쓴다(어댑터가 무슨 지표를 낼지 모른다).
METRIC_HELP: dict[str, tuple[str, str]] = {
    "mAP50": (
        "검출 점수 (mAP50)",
        "0~1, 높을수록 잘 찾습니다 — 정답과 절반 이상 겹치면 맞힌 것으로 봅니다",
    ),
}


def metric_label(name: str) -> tuple[str, str]:
    """``(라벨, 풀이)``. 화면이 표 머리글·카드 제목에 같은 말을 쓰게 하는 한 지점."""
    return METRIC_HELP.get(name, (name, ""))


#: 지점 표시 — 원장 이벤트 이름 → 사람이 읽는 말.
MARKER_LABEL: dict[str, str] = {
    L.EVENT_BASELINE_RESET: "기준선 재설정",
    L.EVENT_BREAKER_RESET: "자동 정지 해제",
}


# --------------------------------------------------------------------------------------
# 1. 단계 — 지금 어디까지 왔나
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseStep:
    """단계 하나. ``state`` 는 ``done``·``current``·``todo`` 셋뿐이다."""

    phase: str
    label: str
    state: str

    def to_json(self) -> dict[str, Any]:
        return {"phase": self.phase, "label": self.label, "state": self.state}


def phase_steps(
    phases: Sequence[str], done: Sequence[str], labels: Mapping[str, str] | None = None
) -> list[PhaseStep]:
    """이 라운드의 단계 줄 — 끝난 것 · 지금 것 · 남은 것.

    "지금"은 **아직 안 끝난 첫 단계**다(`round.next_phase` 와 같은 규칙 — 그것이 `round.json` 의 `done`
    을 보는 유일한 판정이고, 여기서 다른 규칙을 쓰면 화면과 실행이 갈린다).
    """
    labels = labels or {}
    finished = set(done)
    current_seen = False
    out: list[PhaseStep] = []
    for p in phases:
        if p in finished:
            state = "done"
        elif not current_seen:
            state, current_seen = "current", True
        else:
            state = "todo"
        out.append(PhaseStep(phase=p, label=str(labels.get(p, p)), state=state))
    return out


# --------------------------------------------------------------------------------------
# 2. 라운드 기록 — 원장 `round_end` 한 줄 = 표 한 행
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundRow:
    """라운드 기록 표의 한 행. 없는 값은 ``None`` 으로 둔다 — **모르는 것과 0 은 다르다**."""

    round: int
    at: str = ""
    metric: float | None = None
    metric_name: str = ""
    promoted: bool = False
    reason: str = ""
    #: 이 라운드에 보관함으로 편입된 조각 수(`accept` 단계)
    intake: int = 0
    #: 합성이 본 보관함 조각 수 · 그 라운드의 스냅샷 파일
    sources: int = 0
    snapshot: str = ""
    pipeline_hash: str = ""
    bootstrap: bool = False
    #: 이름 있는 클래스 수(미분류 제외)
    classes: int = 0
    per_class: Mapping[str, int] = field(default_factory=dict)
    #: **이 라운드와 그 앞 라운드 사이**의 사람 수정률(누계가 아니라 차이 — 설계 §2 규약 4)
    corrections: CorrectionStats = field(default_factory=CorrectionStats)
    #: 이 라운드 **뒤에** 사람이 찍은 지점(기준선 재설정·자동 정지 해제). 없으면 빈 문자열
    marker: str = ""
    marker_note: str = ""

    @property
    def marker_label(self) -> str:
        return MARKER_LABEL.get(self.marker, "")

    def to_json(self) -> dict[str, Any]:
        rate = self.corrections.rate
        return {
            "round": self.round,
            "at": self.at,
            "metric": self.metric,
            "metricName": self.metric_name,
            "promoted": self.promoted,
            "reason": self.reason,
            "intake": self.intake,
            "sources": self.sources,
            "snapshot": self.snapshot,
            "pipelineHash": self.pipeline_hash,
            "bootstrap": self.bootstrap,
            "classes": self.classes,
            "perClass": dict(self.per_class),
            "correctionRate": rate,
            "drafted": self.corrections.drafted,
            "corrected": self.corrections.corrected,
            "marker": self.marker,
            "markerLabel": self.marker_label,
            "markerNote": self.marker_note,
        }


def _outcome(event: L.Event) -> RoundOutcome:
    metric = event.get("metric")
    return RoundOutcome(
        round=event.round,
        promoted=bool(event.get("promoted")),
        metric=float(metric) if isinstance(metric, (int, float)) else None,
        intake=int(event.get("intake", 0) or 0),
        drafted=int(event.get("drafted", 0) or 0),
        corrected=int(event.get("corrected", 0) or 0),
        per_class={str(k): int(v) for k, v in (event.get("bank_per_class") or {}).items()},
        bootstrap=bool(event.get("bootstrap")),
    )


def _markers(led: L.Ledger) -> dict[int, tuple[str, str]]:
    """``after_round`` → (이벤트 이름, 메모). 같은 라운드에 둘이 찍혔으면 뒤엣것이 남는다."""
    out: dict[int, tuple[str, str]] = {}
    for name in (L.EVENT_BASELINE_RESET, L.EVENT_BREAKER_RESET):
        for e in led.of(name):
            out[int(e.get("after_round", 0) or 0)] = (name, str(e.get("note", "") or ""))
    return out


def round_rows(led: L.Ledger, *, limit: int = DEFAULT_LIMIT) -> list[RoundRow]:
    """라운드 기록 — **최신이 위**. 사람 수정률은 바로 앞 라운드와의 차이로 채운다.

    `limit` 은 표시 상한이고 차이 계산은 잘리기 **전** 목록으로 한다(잘린 자리에서 앞 라운드가 사라지면
    첫 행의 수정률이 사라진다).
    """
    ends = led.of(L.EVENT_ROUND_END)
    markers = _markers(led)
    rows: list[RoundRow] = []
    previous: RoundOutcome | None = None
    for e in ends:
        outcome = _outcome(e)
        stats = (
            correction_window([previous, outcome]) if previous is not None else CorrectionStats()
        )
        per_class = dict(outcome.per_class)
        marker, note = markers.get(e.round, ("", ""))
        rows.append(
            RoundRow(
                round=e.round,
                at=e.at,
                metric=outcome.metric,
                metric_name=str(e.get("metric_name", "") or ""),
                promoted=outcome.promoted,
                reason=e.reason,
                intake=outcome.intake,
                sources=int(e.get("bank_sources", 0) or 0),
                snapshot=str(e.get("bank_snapshot", "") or ""),
                pipeline_hash=str(e.get("pipeline_hash", "") or ""),
                bootstrap=outcome.bootstrap,
                classes=len([c for c in per_class if c != UNSORTED]),
                per_class=per_class,
                corrections=stats,
                marker=marker,
                marker_note=note,
            )
        )
        previous = outcome
    rows.reverse()
    return rows[:limit] if limit > 0 else rows


# --------------------------------------------------------------------------------------
# 3. 추이 — 선은 지점에서 끊는다
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricPoint:
    """추이 점 하나. ``segment`` 가 달라지면 **선을 잇지 않는다**."""

    round: int
    metric: float
    promoted: bool
    segment: int
    marker: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "metric": self.metric,
            "promoted": self.promoted,
            "segment": self.segment,
            "marker": self.marker,
        }


def metric_points(rows: Sequence[RoundRow]) -> list[MetricPoint]:
    """지표가 있는 라운드만, **옛것부터**. 지점 뒤로는 `segment` 가 하나 올라간다.

    지표가 없는 라운드(학습 전에 멈춘 것)는 건너뛴다 — 0 으로 그리면 "점수가 0 이 됐다"가 된다.
    """
    out: list[MetricPoint] = []
    segment = 0
    for row in sorted(rows, key=lambda r: r.round):
        if row.metric is not None:
            out.append(
                MetricPoint(
                    round=row.round,
                    metric=float(row.metric),
                    promoted=row.promoted,
                    segment=segment,
                    marker=row.marker,
                )
            )
        if row.marker:  # 이 라운드 **뒤에** 찍힌 지점이므로 다음 점부터 끊는다
            segment += 1
    return out


# --------------------------------------------------------------------------------------
# 4. 클래스 — 무엇이 얼마나 들어와 있나
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassRow:
    """클래스 한 줄 — 지금 조각 수와 **지난 라운드 대비 증가분**."""

    name: str
    count: int
    delta: int | None = None
    share: float = 0.0
    unsorted: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "delta": self.delta,
            "share": self.share,
            "unsorted": self.unsorted,
        }


def class_rows(
    current: Mapping[str, int], previous: Mapping[str, int] | None = None
) -> list[ClassRow]:
    """지금 보관함 분포 — 조각 많은 순, **미분류는 언제나 맨 끝**(`core.classes` 의 불변식과 같은 자리).

    `previous` 가 없으면 `delta` 는 ``None``(증가분을 모른다 — 0 이라고 말하지 않는다).
    """
    total = sum(max(0, int(v)) for v in current.values())
    rows = [
        ClassRow(
            name=name,
            count=int(count),
            delta=(int(count) - int(previous.get(name, 0))) if previous is not None else None,
            share=(int(count) / total) if total > 0 else 0.0,
            unsorted=name == UNSORTED,
        )
        for name, count in current.items()
    ]
    rows.sort(key=lambda r: (r.unsorted, -r.count, r.name))
    return rows


# --------------------------------------------------------------------------------------
# 5. 지금 무엇을 할 차례인가 — 화면 맨 위 한 줄
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """다음에 할 일 하나. ``kind`` 로 화면이 강조를 정하고, ``command`` 가 있으면 그대로 치면 된다."""

    kind: str
    text: str
    command: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "command": self.command}


def next_action(
    *,
    breaker: Any = None,
    waiting_review: bool = False,
    judged: int = 0,
    total: int = 0,
    lock: Any = None,
    trigger: Any = None,
    has_round: bool = False,
    config_path: str = "",
) -> Action:
    """화면 맨 위 한 줄 — **CLI `loop status` 와 같은 우선순위**로 고른다.

    순서에 뜻이 있다: 돌고 있으면 기다리는 것이 할 일이고(먼저), 자동 정지는 트리거보다 앞선다(사람이
    무엇을 바꿔야 한다 — 정지 중에 "지금 돌 때입니다"를 같이 찍으면 화면이 자기모순이 된다, T12).
    사람 판정 대기는 정지·트리거보다 **앞**이다 — 이미 열린 라운드의 큐는 썩기 전에 봐야 한다.
    """
    suffix = f" --config {config_path}" if config_path else ""
    if lock is not None:
        return Action("running", "지금 한 바퀴가 돌고 있습니다 — 끝나면 여기 반영됩니다.")
    if waiting_review:
        left = max(0, total - judged)
        return Action(
            "review",
            f"검토 대기 {total}장 중 {judged}장을 판정했습니다 — 남은 {left}장을 검수 화면에서 보세요.",
        )
    if breaker is not None and getattr(breaker, "tripped", False):
        return Action(
            "breaker",
            f"자동 정지: {breaker.reason}",
            'anograft loop breaker-reset --note "무엇을 바꿨는지"',
        )
    if not has_round:
        return Action(
            "start",
            "아직 라운드가 없습니다 — 첫 라운드를 열면 여기에 채워집니다.",
            f"anograft loop run{suffix}",
        )
    if trigger is not None and not getattr(trigger, "start", False):
        return Action(
            "wait", f"지금은 돌지 않습니다 — {trigger.reason}", f"anograft loop tick{suffix}"
        )
    reason = getattr(trigger, "reason", "") if trigger is not None else ""
    return Action(
        "run",
        "지금 돌 때입니다" + (f" — {reason}" if reason else ""),
        f"anograft loop run{suffix}",
    )


__all__ = [
    "DEFAULT_LIMIT",
    "MARKER_LABEL",
    "METRIC_HELP",
    "Action",
    "ClassRow",
    "MetricPoint",
    "PhaseStep",
    "RoundRow",
    "class_rows",
    "metric_label",
    "metric_points",
    "next_action",
    "phase_steps",
    "round_rows",
]

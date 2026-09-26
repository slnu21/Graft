"""루프 현황판(U8) — 원장 줄이 표 행·추이 점·다음 할 일로 바뀌는 자리.

여기가 지키는 것: 사람 수정률은 **라운드 사이의 차이**(누계가 아니라) · 지점(기준선 재설정·자동 정지
해제)에서 **선이 끊긴다** · 지표가 없는 라운드는 점을 찍지 않는다(0 으로 그리면 거짓말) · 미분류는
클래스 표의 **맨 끝** · 다음 할 일의 **우선순위**(돌고 있음 → 사람 판정 → 자동 정지 → 트리거).
"""

from __future__ import annotations

from pathlib import Path

from anograft.core.classes import UNSORTED
from anograft.loop import board
from anograft.loop import ledger as L
from anograft.loop.policy import BreakerVerdict, TriggerDecision
from anograft.loop.round import PHASE_LABEL, PHASES


def _end(path: Path, number: int, **fields: object) -> None:
    L.append(path, L.EVENT_ROUND_END, round_no=number, **fields)


def _ledger(tmp_path: Path) -> Path:
    return tmp_path / "rounds.jsonl"


# --------------------------------------------------------------------- 단계


def test_phase_steps_mark_done_current_and_todo() -> None:
    steps = board.phase_steps(PHASES, ["predict", "queue"], PHASE_LABEL)
    assert [s.state for s in steps] == ["done", "done", "current", "todo", "todo", "todo", "todo"]
    assert steps[2].phase == "review" and steps[2].label == "사람 판정 대기"


def test_phase_steps_all_done_has_no_current() -> None:
    steps = board.phase_steps(("synth", "train"), ["synth", "train"])
    assert [s.state for s in steps] == ["done", "done"]


def test_phase_steps_label_falls_back_to_the_phase_name() -> None:
    assert board.phase_steps(("synth",), [])[0].label == "synth"


# --------------------------------------------------------------------- 라운드 기록


def test_round_rows_are_newest_first_and_carry_the_facts(tmp_path: Path) -> None:
    p = _ledger(tmp_path)
    _end(
        p, 1, metric=0.30, metric_name="mAP50", promoted=True, reason="기준선 없음", bootstrap=True
    )
    _end(
        p,
        2,
        metric=0.34,
        metric_name="mAP50",
        promoted=True,
        reason="개선",
        intake=6,
        bank_sources=40,
    )

    rows = board.round_rows(L.read(p))
    assert [r.round for r in rows] == [2, 1]
    assert rows[0].metric == 0.34 and rows[0].intake == 6 and rows[0].sources == 40
    assert rows[1].bootstrap is True and rows[0].metric_name == "mAP50"


def test_correction_rate_is_the_difference_between_rounds(tmp_path: Path) -> None:
    """누계가 아니라 **차이**다 — 초기에 열심히 고친 이력이 "지금 아무도 안 본다"를 가리면 안 된다."""
    p = _ledger(tmp_path)
    _end(p, 1, drafted=10, corrected=8)  # 옛날에 8/10 을 고쳤다
    _end(p, 2, drafted=20, corrected=9)  # 이번 라운드 사이에는 10 개 중 1 개뿐

    rows = board.round_rows(L.read(p))
    assert rows[0].corrections.drafted == 10 and rows[0].corrections.corrected == 1
    assert rows[0].corrections.rate == 0.1
    # 첫 라운드는 견줄 앞이 없다 — 0 이 아니라 "모름"
    assert rows[1].corrections.rate is None


def test_limit_cuts_the_table_but_not_the_difference(tmp_path: Path) -> None:
    """잘린 자리에서 앞 라운드가 사라지면 첫 행의 수정률이 통째로 없어진다."""
    p = _ledger(tmp_path)
    _end(p, 1, drafted=4, corrected=4)
    _end(p, 2, drafted=10, corrected=5)

    rows = board.round_rows(L.read(p), limit=1)
    assert [r.round for r in rows] == [2]
    assert rows[0].corrections.drafted == 6 and rows[0].corrections.corrected == 1


def test_reset_marker_is_attached_to_the_round_it_follows(tmp_path: Path) -> None:
    p = _ledger(tmp_path)
    _end(p, 1, metric=0.3)
    L.append(p, L.EVENT_BASELINE_RESET, after_round=1, note="blowhole 추가")
    _end(p, 2, metric=0.2)

    rows = {r.round: r for r in board.round_rows(L.read(p))}
    assert rows[1].marker == L.EVENT_BASELINE_RESET and rows[1].marker_label == "기준선 재설정"
    assert rows[1].marker_note == "blowhole 추가"
    assert rows[2].marker == "" and rows[2].to_json()["markerLabel"] == ""


def test_row_json_keeps_none_apart_from_zero(tmp_path: Path) -> None:
    p = _ledger(tmp_path)
    _end(p, 1, promoted=False, reason="학습 전에 멈췄습니다")
    row = board.round_rows(L.read(p))[0].to_json()
    assert row["metric"] is None and row["correctionRate"] is None and row["intake"] == 0


def test_classes_column_does_not_count_unsorted(tmp_path: Path) -> None:
    p = _ledger(tmp_path)
    _end(p, 1, bank_per_class={"scratch": 3, "pit": 2, UNSORTED: 5})
    assert board.round_rows(L.read(p))[0].classes == 2


# --------------------------------------------------------------------- 추이


def test_metric_points_are_oldest_first_and_break_at_a_marker(tmp_path: Path) -> None:
    p = _ledger(tmp_path)
    _end(p, 1, metric=0.30)
    _end(p, 2, metric=0.34)
    L.append(p, L.EVENT_BREAKER_RESET, after_round=2, note="링 조명 교체")
    _end(p, 3, metric=0.21)

    points = board.metric_points(board.round_rows(L.read(p)))
    assert [pt.round for pt in points] == [1, 2, 3]
    assert [pt.segment for pt in points] == [0, 0, 1]  # 지점 앞뒤는 잇지 않는다


def test_metric_points_skip_rounds_without_a_score(tmp_path: Path) -> None:
    p = _ledger(tmp_path)
    _end(p, 1, metric=0.30)
    _end(p, 2)  # 학습 전에 멈춘 라운드
    _end(p, 3, metric=0.33)
    assert [pt.round for pt in board.metric_points(board.round_rows(L.read(p)))] == [1, 3]


# --------------------------------------------------------------------- 클래스


def test_class_rows_sort_by_count_and_put_unsorted_last() -> None:
    rows = board.class_rows({"pit": 2, UNSORTED: 9, "scratch": 5})
    assert [r.name for r in rows] == ["scratch", "pit", UNSORTED]
    assert rows[-1].unsorted is True
    assert abs(rows[0].share - 5 / 16) < 1e-9


def test_class_delta_is_none_without_a_previous_round() -> None:
    rows = board.class_rows({"pit": 2})
    assert rows[0].delta is None
    rows = board.class_rows({"pit": 5}, {"pit": 2})
    assert rows[0].delta == 3


def test_class_row_for_a_brand_new_class_counts_all_of_it() -> None:
    rows = {r.name: r for r in board.class_rows({"pit": 2, "blowhole": 4}, {"pit": 2})}
    assert rows["blowhole"].delta == 4


# --------------------------------------------------------------------- 지표 라벨


def test_metric_label_glosses_map50_and_passes_others_through() -> None:
    """용어 사전 §3.5 — `mAP50` 은 고유명이라 남기되 **풀이가 필수**다."""
    label, hint = board.metric_label("mAP50")
    assert label == "검출 점수 (mAP50)" and "높을수록" in hint
    assert board.metric_label("image_auroc") == ("image_auroc", "")


# --------------------------------------------------------------------- 다음 할 일


def test_running_round_comes_first() -> None:
    action = board.next_action(
        lock=object(), waiting_review=True, breaker=BreakerVerdict(True, "x")
    )
    assert action.kind == "running"


def test_human_judgement_comes_before_the_breaker() -> None:
    """이미 열린 라운드의 큐는 **썩기 전에** 봐야 한다 — 정지가 그 앞을 막으면 안 된다."""
    action = board.next_action(
        waiting_review=True, judged=2, total=5, breaker=BreakerVerdict(True, "연속 미승급")
    )
    assert action.kind == "review" and "남은 3장" in action.text


def test_breaker_beats_the_trigger_and_tells_how_to_clear_it() -> None:
    action = board.next_action(
        breaker=BreakerVerdict(True, "연속 3회 미승급"),
        trigger=TriggerDecision(True, "양 기준이 없어 바로 돕니다"),
        has_round=True,
    )
    assert action.kind == "breaker" and "breaker-reset" in action.command


def test_trigger_reason_is_shown_when_it_says_no() -> None:
    action = board.next_action(
        trigger=TriggerDecision(False, "새 조각 0/40개 — 아직 양이 안 찼습니다"),
        has_round=True,
        config_path="loop.yaml",
    )
    assert action.kind == "wait" and "0/40" in action.text
    assert action.command == "anograft loop tick --config loop.yaml"


def test_first_round_gets_the_run_command() -> None:
    action = board.next_action(has_round=False, config_path="c/loop.yaml")
    assert action.kind == "start" and action.command.endswith("loop run --config c/loop.yaml")

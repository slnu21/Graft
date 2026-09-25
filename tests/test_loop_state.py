"""루프 원장 + 트리거 정책 (설계 §2b.3·2b.4, 작업 단위 T14).

여기가 못 박는 것:

1. **이력은 append-only 원장에 있다** — `loop.state.json` 에 남는 것은 움직이는 포인터 둘(라운드·champion)뿐.
2. **실패도 기록이다** — 세 시간 뒤 실패를 아침에 읽을 사람이 있다. 그리고 실패해도 **합성을 다시 하지 않는다**.
3. **안 도는 이유를 항상 남긴다** — 조용히 안 도는 루프가 제일 나쁘다. 다만 5분마다 같은 사유를 원장에
   쌓지는 않는다(사유가 바뀔 때만).
4. **트리거는 순수 함수다** — 파일을 읽는 쪽과 판정하는 쪽이 갈려 있어야 `noop` 없이도 규칙을 시험할 수 있다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from anograft.cli import EXIT_OK, main
from anograft.io import imgio
from anograft.io.manifest import MANIFEST_FILE, read_manifest
from anograft.io.prune import REVIEW_FILE, write_review
from anograft.loop import ledger
from anograft.loop.config import LoopConfigError, load_loop_config
from anograft.loop.policy import (
    TriggerPolicy,
    TriggerState,
    should_start_round,
)
from anograft.loop.round import (
    LoopError,
    check_trigger,
    ledger_path,
    load_record,
    load_state,
    round_name,
    run_round,
    status,
    tick_path,
)
from tests.fixtures import blob_image, blob_mask, loop_workspace

# --------------------------------------------------------------------- 트리거 (순수)


def test_in_progress_round_is_always_continued() -> None:
    """이어 가는 것은 트리거가 막지 않는다 — 반쯤 진행된 라운드를 세워 두면 사람이 판정한 것이 썩는다."""
    facts = TriggerState(has_round=True, in_progress=True, new_labels=0, hours_since=0.0)
    decision = should_start_round(facts, TriggerPolicy(min_labels=99, min_interval_hours=168))
    assert decision.start and "이어" in decision.reason


def test_first_round_always_runs() -> None:
    decision = should_start_round(TriggerState(), TriggerPolicy(min_labels=99))
    assert decision.start and "첫 라운드" in decision.reason


def test_minimum_interval_blocks_with_numbers() -> None:
    facts = TriggerState(has_round=True, new_labels=100, hours_since=12.0)
    decision = should_start_round(facts, TriggerPolicy(min_labels=1, min_interval_hours=168))
    assert not decision.start
    assert "12.0시간" in decision.reason and "168.0시간" in decision.reason  # 숫자와 함께 말한다


def test_quantity_criteria_fire_and_report_which() -> None:
    policy = TriggerPolicy(min_labels=20, min_images=50, min_interval_hours=1.0)
    yes = should_start_round(
        TriggerState(has_round=True, new_labels=25, new_images=3, hours_since=48.0), policy
    )
    assert yes.start and "새 조각 25개" in yes.reason and "새 이미지" not in yes.reason

    no = should_start_round(
        TriggerState(has_round=True, new_labels=3, new_images=10, hours_since=48.0), policy
    )
    assert not no.start
    assert "3/20" in no.reason and "10/50" in no.reason  # 얼마나 남았는지 보인다


def test_no_quantity_criteria_means_run() -> None:
    decision = should_start_round(TriggerState(has_round=True, hours_since=1.0), TriggerPolicy())
    assert decision.start and "양 기준이 없어" in decision.reason


def test_max_interval_runs_even_without_new_data() -> None:
    """드리프트 감시 — 양이 안 차도 한참 지나면 한 바퀴 돈다."""
    facts = TriggerState(has_round=True, new_labels=0, hours_since=800.0)
    decision = should_start_round(
        facts, TriggerPolicy(min_labels=20, min_interval_hours=168, max_interval_hours=720)
    )
    assert decision.start and "최대 간격" in decision.reason


def test_unknown_elapsed_time_does_not_block() -> None:
    facts = TriggerState(has_round=True, new_labels=50, hours_since=None)
    assert should_start_round(facts, TriggerPolicy(min_labels=20, min_interval_hours=168)).start


# --------------------------------------------------------------------- 원장


def test_ledger_round_trip_keeps_unknown_fields(tmp_path: Path) -> None:
    """T12·T15·T16 이 필드를 더할 것이다 — 모르는 키를 만나도 깨지지 않아야 원장이 오래 산다."""
    path = tmp_path / ledger.ROUNDS_FILE
    ledger.append(path, ledger.EVENT_ROUND_START, round_no=1, phases=["synth"], bootstrap=True)
    ledger.append(
        path, ledger.EVENT_ROUND_END, round_no=1, metric=0.42, promoted=True, breaker="open"
    )
    led = ledger.read(path)
    assert len(led) == 2 and not led.warnings
    end = led.last_end
    assert end is not None and end.round == 1 and end.get("breaker") == "open"
    assert led.history() == [
        {"round": 1, "metric": 0.42, "promoted": True, "reason": "", "finished": end.at}
    ]


def test_ledger_skips_broken_lines_and_says_so(tmp_path: Path) -> None:
    path = tmp_path / ledger.ROUNDS_FILE
    ledger.append(path, ledger.EVENT_FAILED, round_no=2, phase="train", reason="학습 실패")
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("{망가진 줄}\n")
        fh.write(json.dumps({"at": "…"}) + "\n")  # event 가 없다
    led = ledger.read(path)
    assert len(led) == 1 and len(led.warnings) == 2
    assert led.failures()[0].phase == "train" and "학습 실패" in led.failures()[0].reason
    assert ledger.read(tmp_path / "없는파일.jsonl").events == []


def test_tick_record_round_trip(tmp_path: Path) -> None:
    path = tmp_path / ledger.TICK_FILE
    assert ledger.read_tick(path) is None
    ledger.write_tick(path, ledger.Tick(ran=False, reason="아직 양이 안 찼습니다"))
    back = ledger.read_tick(path)
    assert back is not None and not back.ran and "양이 안 찼" in back.line()
    path.write_text("JSON 아님", encoding="utf-8")
    assert ledger.read_tick(path) is None  # fail-soft


# --------------------------------------------------------------------- 설정


def _write_loop_yaml(base: Path, path: Path, **changes: object) -> Path:
    doc = yaml.safe_load(base.read_text(encoding="utf-8"))
    for key, value in changes.items():
        doc[key] = value
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_trigger_block_parses_and_rejects_typos(tmp_path: Path) -> None:
    ws = loop_workspace(tmp_path)
    good = _write_loop_yaml(
        ws["loop"],
        tmp_path / "loop-trigger.yaml",
        trigger={"min_labels": 20, "min_interval_hours": 168, "max_interval_hours": 720},
    )
    policy = load_loop_config(good).config.trigger.policy()
    assert policy.min_labels == 20 and policy.max_interval_hours == 720

    bad = _write_loop_yaml(ws["loop"], tmp_path / "loop-typo.yaml", trigger={"min_label": 20})
    with pytest.raises(LoopConfigError):
        load_loop_config(bad)


# --------------------------------------------------------------------- 라운드와 함께


@pytest.fixture
def loop_ws(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Path]:
    ws = loop_workspace(tmp_path)
    capsys.readouterr()
    return ws


def test_round_writes_the_ledger_and_a_bank_snapshot(loop_ws: dict[str, Path]) -> None:
    loop = load_loop_config(loop_ws["loop"])
    result = run_round(loop)
    led = ledger.read(ledger_path(loop))

    kinds = [e.event for e in led.events]
    assert kinds[0] == ledger.EVENT_ROUND_START and kinds[-1] == ledger.EVENT_ROUND_END
    assert [e.phase for e in led.of(ledger.EVENT_PHASE)] == ["synth", "train", "judge"]

    end = led.last_end
    assert end is not None and end.round == 1
    assert end.get("promoted") is True and end.get("metric") > 0
    assert end.get("pipeline_hash") and end.get("bank_sources") > 0
    assert end.get("champion", {}).get("round") == 1

    # 이 라운드가 **어떤 보관함으로** 만들었는가 — 복사가 아니라 id+해시 목록
    snap = json.loads((result.round_dir / "bank.snapshot.json").read_text(encoding="utf-8"))
    assert len(snap["sources"]) == end.get("bank_sources")

    # 포인터 파일에는 이력이 없다
    saved = json.loads((loop_ws["out"] / "loop.state.json").read_text(encoding="utf-8"))
    assert set(saved) == {"round", "champion"}
    assert load_state(loop_ws["out"]).champion is not None


def test_failure_is_recorded_and_synthesis_is_not_redone(
    loop_ws: dict[str, Path], tmp_path: Path
) -> None:
    """학습이 3시간 뒤 실패해도 다음 실행이 합성을 다시 하지 않는다(설계 §2b.4 부분 진행 재사용)."""
    broken = _write_loop_yaml(
        loop_ws["loop"], tmp_path / "loop-badmetric.yaml", promote={"metric": "없는지표"}
    )
    with pytest.raises(LoopError, match="없는지표"):
        run_round(load_loop_config(broken))

    loop = load_loop_config(broken)
    led = ledger.read(ledger_path(loop))
    failed = led.failures()
    assert failed and failed[0].phase == "judge" and "없는지표" in failed[0].reason

    record = load_record(loop_ws["out"] / round_name(1))
    assert record is not None and record.done == ["synth", "train"]  # 합성·학습은 살아 있다
    synth_before = dict(record.data["synth"])

    # 지표 이름을 맞추고 다시 부르면 **판정부터** 이어 간다
    good = load_loop_config(loop_ws["loop"])
    done = run_round(good)
    assert done.record.number == 1 and done.record.data["synth"] == synth_before
    assert ledger.read(ledger_path(good)).last_end is not None


def _judge_all(queue_dir: Path) -> int:
    rows = [r for r in read_manifest(queue_dir / MANIFEST_FILE) if r.get("status") == "ok"]
    write_review(queue_dir / REVIEW_FILE, {r["index"]: ("accept", "") for r in rows})
    return len(rows)


def test_trigger_blocks_the_second_tick_and_says_why_once(
    loop_ws: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """양 기준이 안 찼으면 안 돈다 — 이유는 남기지만 5분마다 원장에 쌓지는 않는다."""
    gated = _write_loop_yaml(
        loop_ws["loop"], tmp_path / "loop-gated.yaml", trigger={"min_labels": 99}
    )
    assert main(["loop", "tick", "--config", str(gated), "--json"]) == EXIT_OK
    first = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert first["ran"] is True and first["round"] == 1  # 첫 라운드는 언제나 돈다

    assert main(["loop", "tick", "--config", str(gated), "--json"]) == EXIT_OK
    second = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert second["ran"] is False and second["busy"] is False
    assert "0/99" in second["reason"] and second["newLabels"] == 0

    loop = load_loop_config(gated)
    tick = ledger.read_tick(tick_path(loop))
    assert tick is not None and not tick.ran and "99" in tick.reason
    assert len(ledger.read(ledger_path(loop)).of(ledger.EVENT_SKIPPED)) == 1

    # 같은 사유로 또 불려도 원장은 늘지 않는다(tick.json 만 갱신)
    assert main(["loop", "tick", "--config", str(gated), "--json"]) == EXIT_OK
    capsys.readouterr()
    assert len(ledger.read(ledger_path(loop)).of(ledger.EVENT_SKIPPED)) == 1

    # --force 는 기준을 무시하고 돈다(잠금은 그대로 지킨다)
    assert main(["loop", "tick", "--config", str(gated), "--force", "--json"]) == EXIT_OK
    forced = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert forced["ran"] is True and forced["round"] == 2


def test_status_and_check_trigger_agree(loop_ws: dict[str, Path], tmp_path: Path) -> None:
    gated = _write_loop_yaml(
        loop_ws["loop"], tmp_path / "loop-gated2.yaml", trigger={"min_labels": 99}
    )
    loop = load_loop_config(gated)
    run_round(loop)  # 부트스트랩 한 바퀴

    decision, facts = check_trigger(loop)
    assert not decision.start and facts.new_labels == 0
    st = status(loop)
    assert st.trigger is not None and st.trigger.reason == decision.reason
    text = " ".join(st.lines())
    assert "트리거: 지금은 돌지 않습니다" in text and "라운드 1" in text
    assert st.history and st.history[0]["round"] == 1


def test_pieces_added_between_rounds_count_as_new_labels(
    loop_ws: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ "새 조각"은 **마지막 라운드가 적어 둔 보관함 조각 수와의 차이**다.

    그래서 사람이 결함 표시 화면에서 넣은 조각도 라운드를 여는 이유가 된다(라운드 안에서 `accept` 한
    조각은 그 라운드가 이미 먹었으므로 다시 세지 않는다 — 그게 맞다).
    """
    gated = _write_loop_yaml(
        loop_ws["loop"], tmp_path / "loop-labels.yaml", trigger={"min_labels": 1}
    )
    loop = load_loop_config(gated)
    run_round(loop)  # 1라운드(부트스트랩) — 그때의 보관함 조각 수가 원장에 적힌다
    blocked, before = check_trigger(loop)
    assert before.new_labels == 0 and not blocked.start

    # 사람이 조각 하나를 더 넣었다(결함 표시 화면이 하는 일 = import-pairs 와 같은 경로)
    pair = tmp_path / "hand"
    (pair / "images").mkdir(parents=True)
    (pair / "masks").mkdir(parents=True)
    imgio.write_image(pair / "images" / "hand0.png", blob_image(64, [(32, 32, 8)]))
    imgio.write_image(pair / "masks" / "hand0.png", blob_mask(64, [(32, 32, 8)]))
    assert (
        main(
            [
                "bank",
                "import-pairs",
                "--images",
                str(pair / "images"),
                "--masks",
                str(pair / "masks"),
                "--class",
                "spot",
                "--out",
                str(loop_ws["bank"]),
            ]
        )
        == EXIT_OK
    )
    capsys.readouterr()

    decision, facts = check_trigger(loop)
    assert facts.new_labels == 1
    assert decision.start and "새 조각 1개" in decision.reason

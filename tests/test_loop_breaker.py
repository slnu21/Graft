"""자동 정지 + 사람 수정률 (설계 §6.5 · §2 규약 4, 작업 단위 T12).

여기가 못 박는 것:

1. **기준은 코드가 정하지 않는다** — 기본값 0 = 제한 없음(T14 트리거와 같은 규율). 숫자는 `loop.yaml`.
2. **사람 수정률은 라운드 *사이*로 잰다** — 누계 비율만 보면 초기에 열심히 고친 이력이 "지금 아무도 안
   본다"를 영원히 가려 주고, 정지를 한 번 풀어도 옛 누계가 곧바로 다시 정지를 부른다.
3. **모르는 것과 0 은 다르다** — 들어온 초안이 없으면 수정률을 판정하지 않는다.
4. **정지는 라운드를 열기 *전에*** — 빈 라운드 폴더·원장 줄을 남기지 않는다(T15 가 겪은 것).
5. **진행 중인 라운드는 막지 않는다** — 사람이 판정해 둔 큐가 썩는다(트리거의 첫 규칙과 같은 이유).
6. **스케줄러에게 정지는 오류가 아니라 상태다**(exit 0) — 5분마다 실패 알림이 오는 운영은 아무도 안 본다.
   대신 `--force` 로도 넘어가지 않는다: 다시 돌리는 길은 "무엇을 바꿨는지 적는 것" 하나다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from anograft.cli import EXIT_OK, EXIT_RECIPE_ERROR, main
from anograft.loop import ledger
from anograft.loop.config import LoopConfigError, load_loop_config
from anograft.loop.policy import (
    BreakerPolicy,
    RoundOutcome,
    circuit_break,
    class_shift,
    correction_window,
)
from anograft.loop.round import (
    LoopError,
    bank_facts,
    breaker_history,
    breaker_reset,
    breaker_verdict,
    ledger_path,
    round_name,
    run_round,
    status,
)
from tests.fixtures import loop_workspace

# --------------------------------------------------------------------- 순수 판정


def _rounds(*promoted: bool) -> list[RoundOutcome]:
    return [RoundOutcome(round=i + 1, promoted=p) for i, p in enumerate(promoted)]


def test_no_criteria_never_stops() -> None:
    """기본값은 제한 없음 — 코드가 정한 숫자가 루프를 세우면 "왜 멈췄는지" 모르는 사람이 먼저 생긴다."""
    verdict = circuit_break(_rounds(False, False, False, False), BreakerPolicy())
    assert not verdict.tripped and "기준이 없습니다" in verdict.reason


def test_consecutive_rounds_without_promotion_stop() -> None:
    verdict = circuit_break(_rounds(True, False, False, False), BreakerPolicy(stale_rounds=3))
    assert verdict.tripped and verdict.kinds == ("stale",)
    assert "3라운드 연속 승급 없음" in verdict.reason
    assert "조명" in verdict.reason  # 무엇을 바꿔야 하는지 함께 말한다


def test_promotion_resets_the_streak() -> None:
    verdict = circuit_break(_rounds(False, False, True, False), BreakerPolicy(stale_rounds=2))
    assert not verdict.tripped and "연속 유지 1/2회" in verdict.reason


def test_correction_rate_is_measured_between_rounds() -> None:
    """누계로 재면 "초기에 열심히 고쳤다"가 "지금 아무도 안 본다"를 영원히 가려 준다."""
    history = [
        RoundOutcome(round=1, promoted=True, drafted=100, corrected=20),
        RoundOutcome(round=2, promoted=True, drafted=120, corrected=20),
    ]
    assert correction_window(history, 2).rate == 0.0  # 새로 들어온 20개 중 0개
    assert history[-1].corrected / history[-1].drafted > 0.15  # 누계로는 17% (안 걸린다)

    verdict = circuit_break(history, BreakerPolicy(min_correction_rate=0.05))
    assert verdict.tripped and verdict.kinds == ("correction",)
    assert "사람 수정률 0%" in verdict.reason and "아무도 안 보고" in verdict.reason


def test_someone_is_looking_so_it_keeps_running() -> None:
    history = [
        RoundOutcome(round=1, promoted=True, drafted=10, corrected=2),
        RoundOutcome(round=2, promoted=True, drafted=20, corrected=8),
    ]
    verdict = circuit_break(history, BreakerPolicy(min_correction_rate=0.05))
    assert not verdict.tripped and "사람 수정률 60%" in verdict.reason


def test_unknown_correction_rate_does_not_stop() -> None:
    """모르는 것과 0 은 다르다 — 견줄 앞 라운드가 없거나 새 초안이 없으면 판정하지 않는다."""
    single = [RoundOutcome(round=1, promoted=True, drafted=50, corrected=0)]
    assert correction_window(single, 2).rate is None
    assert not circuit_break(single, BreakerPolicy(min_correction_rate=0.5)).tripped

    idle = [*single, RoundOutcome(round=2, promoted=True, drafted=50, corrected=0)]
    assert correction_window(idle, 2).rate is None  # 들어온 초안이 0개
    assert not circuit_break(idle, BreakerPolicy(min_correction_rate=0.5)).tripped


def test_correction_window_honours_the_window_length() -> None:
    history = [
        RoundOutcome(round=1, drafted=0, corrected=0),
        RoundOutcome(round=2, drafted=10, corrected=10),  # 다 고쳤다
        RoundOutcome(round=3, drafted=20, corrected=10),  # 아무것도 안 고쳤다
    ]
    assert correction_window(history, 1).rate == 0.0  # 마지막 구간만
    assert correction_window(history, 2).rate == 0.5  # 두 구간
    assert correction_window(history, 0).rate == 0.5  # 0 = 전 구간


def test_deleted_pieces_do_not_become_negative_counts() -> None:
    history = [
        RoundOutcome(round=1, drafted=30, corrected=10),
        RoundOutcome(round=2, drafted=20, corrected=5),  # 사람이 조각을 지웠다
    ]
    assert correction_window(history, 1).rate is None  # 새로 들어온 것 0개로 본다


def test_class_shift_compares_ratios_not_counts() -> None:
    """보관함은 라운드마다 커진다 — 개수 차이는 언제나 크고, 알고 싶은 것은 **비중** 변화다."""
    assert class_shift({"a": 10, "b": 10}, {"a": 20, "b": 20}) == 0.0
    assert class_shift({"a": 10, "b": 10}, {"a": 10, "b": 40}) == pytest.approx(0.3)
    assert class_shift({}, {"a": 1}) is None  # 모르면 막지 않는다

    history = [
        RoundOutcome(round=1, promoted=True, per_class={"a": 10, "b": 10}),
        RoundOutcome(round=2, promoted=True, per_class={"a": 10, "b": 40}),
    ]
    verdict = circuit_break(history, BreakerPolicy(max_class_shift=0.2))
    assert verdict.tripped and verdict.kinds == ("class_shift",)
    assert "0.30" in verdict.reason and "로트" in verdict.reason


def test_every_tripped_reason_is_reported_together() -> None:
    """한 번 멈출 때 사람이 알아야 하는 것을 하나만 보여 주지 않는다."""
    history = [
        RoundOutcome(round=1, promoted=False, drafted=10, corrected=1),
        RoundOutcome(round=2, promoted=False, drafted=30, corrected=1),
    ]
    verdict = circuit_break(
        history, BreakerPolicy(stale_rounds=2, min_correction_rate=0.1, correction_rounds=1)
    )
    assert verdict.tripped and verdict.kinds == ("stale", "correction")
    assert "연속 승급 없음" in verdict.reason and "사람 수정률" in verdict.reason


def test_empty_history_does_not_stop() -> None:
    assert not circuit_break([], BreakerPolicy(stale_rounds=1)).tripped


# --------------------------------------------------------------------- 설정


def _write_loop_yaml(base: Path, path: Path, **changes: object) -> Path:
    doc = yaml.safe_load(base.read_text(encoding="utf-8"))
    for key, value in changes.items():
        doc[key] = value
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def test_breaker_block_parses_and_rejects_typos(tmp_path: Path) -> None:
    ws = loop_workspace(tmp_path)
    good = _write_loop_yaml(
        ws["loop"],
        tmp_path / "loop-breaker.yaml",
        breaker={"stale_rounds": 3, "min_correction_rate": 0.05, "max_class_shift": 0.35},
    )
    policy = load_loop_config(good).config.breaker.policy()
    assert policy.stale_rounds == 3 and policy.enabled
    assert not load_loop_config(ws["loop"]).config.breaker.policy().enabled  # 기본은 제한 없음

    bad = _write_loop_yaml(ws["loop"], tmp_path / "loop-typo.yaml", breaker={"stale_round": 3})
    with pytest.raises(LoopConfigError):
        load_loop_config(bad)


# --------------------------------------------------------------------- 라운드와 함께


@pytest.fixture
def loop_ws(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Path]:
    ws = loop_workspace(tmp_path)
    capsys.readouterr()
    return ws


def _stalling(ws: dict[str, Path], tmp_path: Path, **breaker: object) -> Path:
    """승급이 절대 안 되는(시드 요동을 크게 잡은) 수집 없는 설정 — 자동 정지만 시험한다."""
    return _write_loop_yaml(
        ws["loop"],
        tmp_path / "loop-stall.yaml",
        field=None,  # 현장 이미지 없음 = 부트스트랩 라운드만(사람을 기다리지 않는다)
        promote={"metric": "mAP50", "noise": 5.0},
        breaker=breaker,
    )


def test_breaker_stops_before_opening_a_round(loop_ws: dict[str, Path], tmp_path: Path) -> None:
    config = _stalling(loop_ws, tmp_path, stale_rounds=1)
    loop = load_loop_config(config)

    run_round(loop)  # 1라운드 — 기준선이 없어 champion 이 된다(승급)
    second = run_round(loop)  # 2라운드 — Δ 가 시드 요동 이하라 유지
    assert not second.record.data["judge"]["promote"]
    assert [o.promoted for o in breaker_history(loop)] == [True, False]

    verdict = breaker_verdict(loop)
    assert verdict.tripped and verdict.kinds == ("stale",)

    with pytest.raises(LoopError) as exc:
        run_round(loop)
    assert "자동 정지" in str(exc.value) and "breaker-reset" in str(exc.value)

    # **빈 라운드 폴더도 원장 줄도 남지 않는다**(T15 가 겪은 함정)
    assert not (loop_ws["out"] / round_name(3)).exists()
    led = ledger.read(ledger_path(loop))
    assert [e.round for e in led.of(ledger.EVENT_ROUND_START)] == [1, 2]


def test_reset_records_what_was_wrong_and_releases_the_loop(
    loop_ws: dict[str, Path], tmp_path: Path
) -> None:
    config = _stalling(loop_ws, tmp_path, stale_rounds=1)
    loop = load_loop_config(config)
    run_round(loop)
    run_round(loop)
    assert breaker_verdict(loop).tripped

    result = breaker_reset(loop, note="링 조명 교체")
    assert result["tripped"] and result["kinds"] == ["stale"] and result["after_round"] == 2
    mark = ledger.read(ledger_path(loop)).last(ledger.EVENT_BREAKER_RESET)
    assert mark is not None and mark.get("note") == "링 조명 교체"
    assert "연속 승급 없음" in mark.get("reason")

    # 해제 지점 앞 라운드는 판정에서 빠진다 — 그래야 해제가 무언가를 사 준다
    assert breaker_history(loop) == []
    assert not breaker_verdict(loop).tripped
    third = run_round(loop)
    assert third.record.number == 3

    # 그리고 그 다음 라운드부터 다시 센다(한 번 풀면 영원히 안 멈추는 게 아니다)
    assert breaker_verdict(loop).tripped


def test_tick_treats_the_breaker_as_a_state_not_an_error(
    loop_ws: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """5분마다 실패 알림이 오는 운영은 아무도 안 본다 — busy 와 같은 규율(exit 0)."""
    config = _stalling(loop_ws, tmp_path, stale_rounds=1)
    for _ in range(2):
        assert main(["loop", "tick", "--config", str(config), "--json"]) == EXIT_OK
        assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["ran"] is True

    assert main(["loop", "tick", "--config", str(config), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["ran"] is False and payload["busy"] is False
    assert payload["breaker"] is True and payload["kinds"] == ["stale"]
    assert "자동 정지" in payload["reason"]

    # `--force` 는 트리거를 무시하는 스위치다 — 자동 정지는 넘지 못한다
    assert main(["loop", "tick", "--config", str(config), "--force", "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["ran"] is False

    loop = load_loop_config(config)
    tick = ledger.read_tick(loop.out / ledger.TICK_FILE)
    assert tick is not None and not tick.ran and "자동 정지" in tick.reason
    # 사유가 그대로면 원장은 늘지 않는다(tick.json 만 갱신) — T14 의 규율을 그대로 탄다
    assert len(ledger.read(ledger_path(loop)).of(ledger.EVENT_SKIPPED)) == 1

    # 사람이 부른 run 에게는 오류다("내가 시킨 일이 안 됐다")
    assert main(["loop", "run", "--config", str(config)]) == EXIT_RECIPE_ERROR
    assert "자동 정지" in capsys.readouterr().err

    # 해제하면 다시 돈다
    assert (
        main(["loop", "breaker-reset", "--config", str(config), "--note", "모델 교체"]) == EXIT_OK
    )
    assert "해제했습니다" in capsys.readouterr().out
    assert main(["loop", "tick", "--config", str(config), "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["ran"] is True


def test_status_shows_the_breaker_and_agrees_with_tick(
    loop_ws: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _stalling(loop_ws, tmp_path, stale_rounds=1)
    loop = load_loop_config(config)
    run_round(loop)
    run_round(loop)

    st = status(loop)
    assert st.breaker is not None and st.breaker.tripped
    text = " ".join(st.lines())
    assert "자동 정지: " in text and "breaker-reset" in text

    assert main(["loop", "status", "--config", str(config), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["breaker"]["tripped"] and payload["breaker"]["kinds"] == ["stale"]
    assert payload["breaker"]["reason"] == st.breaker.reason


def test_status_shows_what_is_being_watched_before_it_trips(
    loop_ws: dict[str, Path], tmp_path: Path
) -> None:
    config = _stalling(loop_ws, tmp_path, stale_rounds=3)
    loop = load_loop_config(config)
    run_round(loop)
    run_round(loop)
    text = " ".join(status(loop).lines())
    assert "자동 정지 감시: 연속 유지 1/3회" in text  # 조용히 지켜보지 않는다


# --------------------------------------------------------------------- 사람 수정률 (실물)


def test_round_end_records_the_facts_the_breaker_needs(loop_ws: dict[str, Path]) -> None:
    """사람 수정률의 분모·분자는 **보관함이 답한다** — `origin:field` 태그 × ``mask_origin``."""
    from anograft.bank.browse import BankSession

    loop = load_loop_config(loop_ws["loop"])
    run_round(loop)  # 1라운드 부트스트랩
    second = run_round(loop)  # 2라운드 — 사람 판정 앞에서 멈춘다
    assert second.waiting_for_human

    from anograft.io.manifest import MANIFEST_FILE, read_manifest
    from anograft.io.prune import REVIEW_FILE, write_review

    queue_dir = second.round_dir / "queue"
    rows = [r for r in read_manifest(queue_dir / MANIFEST_FILE) if r.get("status") == "ok"]
    write_review(queue_dir / REVIEW_FILE, {r["index"]: ("accept", "") for r in rows})
    done = run_round(loop)
    assert done.record.data["accept"]["imported"] > 0

    facts = bank_facts(loop)
    assert facts["drafted"] == done.record.data["accept"]["imported"]
    assert facts["corrected"] == 0  # 아직 아무도 다듬지 않았다
    assert sum(facts["bank_per_class"].values()) > facts["drafted"]  # 원래 있던 조각 + 편입분

    end = ledger.read(ledger_path(loop)).last_end
    assert end is not None and end.round == 2
    assert end.get("drafted") == facts["drafted"] and end.get("corrected") == 0
    assert end.get("intake") == done.record.data["accept"]["imported"]
    assert end.get("bank_per_class") == facts["bank_per_class"]

    # 사람이 결함 표시 화면에서 마스크를 다듬으면 그 한 장이 분자가 된다
    session = BankSession()
    session.load(loop_ws["bank"])
    field_rows = [r for r in session.rows() if r.mask_origin.startswith("pred:")]
    assert field_rows
    source = session.source(field_rows[0].id)
    session.replace_mask(field_rows[0].id, source.mask, tool="brush")
    assert bank_facts(loop)["corrected"] == 1

    # 부트스트랩 라운드는 편입이 없으니 분모가 0 — 그래서 라운드 1·2 구간은 판정할 수 있다
    history = breaker_history(loop)
    assert [o.round for o in history] == [1, 2]
    assert history[0].drafted == 0 and history[0].intake == 0
    assert history[1].drafted == facts["drafted"] and history[1].intake > 0
    assert correction_window(history, 1).rate == 0.0  # 편입된 초안을 아무도 안 다듬은 상태

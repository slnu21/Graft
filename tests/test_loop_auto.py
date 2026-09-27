"""자동 편입 게이트 (T6 · 설계 §3 `gate`·§4 L2).

순수 판정(`core.gate`·`loop.auto`) → 재기(`mask_confidence` 재사용) → 은행 편입 → CLI · 그리고 자동 편입
비율이 **사람 수정률의 여집합이 아니라 독립 지표**가 되는 자리(`policy.auto_window`·`circuit_break`).

여기서 지키려는 규율: 게이트는 **닫히는 쪽이 기본**이고(모르면 안 받는다) 못 받은 것은 **사람 큐로** 간다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from anograft.bank import Bank
from anograft.bank.bank import is_estimated
from anograft.cli import EXIT_OK, main
from anograft.core.gate import gate, gate_reason, partition_by_gate
from anograft.io import imgio
from anograft.io.manifest import MANIFEST_FILE, read_manifest
from anograft.io.prune import REVIEW_FILE, write_review
from anograft.loop import board, ledger
from anograft.loop.auto import (
    AUTO_FILE,
    AUTO_TAG,
    AutoCandidate,
    AutoPolicy,
    auto_admit,
    cheap_candidates,
    measure,
    partition_auto,
    read_auto_csv,
    run_auto,
    write_auto_csv,
)
from anograft.loop.config import load_loop_config
from anograft.loop.policy import (
    AutoStats,
    BreakerPolicy,
    RoundOutcome,
    auto_window,
    circuit_break,
)
from anograft.loop.queue import (
    Instance,
    Prediction,
    PredictionSet,
    index_images,
    read_predictions,
    read_queue_csv,
)
from anograft.loop.round import run_round, status
from tests.fixtures import blob_image, blob_mask, fake_yolo_dataset, loop_workspace

SIZE = 64
BLOB = (32, 32, 8)
BOX = (24.0, 24.0, 16.0, 16.0)


def _field(root: Path, stems: list[str]) -> Path:
    images = root / "field"
    images.mkdir(parents=True, exist_ok=True)
    for s in stems:
        imgio.write_image(images / f"{s}.png", blob_image(SIZE, [BLOB]))
    return images


def _pred(
    root: Path,
    scores: dict[str, float],
    *,
    name: str = "pred",
    cls: str = "scratch",
    masks: bool = True,
    boxes: dict[str, list[tuple[float, float, float, float]]] | None = None,
) -> Path:
    pred = root / name
    (pred / "scores").mkdir(parents=True, exist_ok=True)
    if masks:
        (pred / "masks").mkdir(parents=True, exist_ok=True)
    for stem, score in scores.items():
        bb = (boxes or {}).get(stem, [BOX])
        (pred / "scores" / f"{stem}.json").write_text(
            json.dumps(
                {
                    "image_score": score,
                    "instances": [{"bbox": list(b), "score": score, "class": cls} for b in bb],
                }
            ),
            encoding="utf-8",
        )
        if masks:
            imgio.write_image(pred / "masks" / f"{stem}.png", blob_mask(SIZE, [BLOB]))
    return pred


def _pset(scores: dict[str, float], *, cls: str = "scratch", mask: Path | None = None):
    return PredictionSet(
        root=Path("pred"),
        items={
            stem: Prediction(
                stem=stem,
                image_score=v,
                instances=(Instance(bbox=BOX, score=v, cls=cls),),
                mask=mask,
            )
            for stem, v in scores.items()
        },
    )


# --------------------------------------------------------------------- core.gate


def test_gate_passes_human_masks_and_refuses_unmeasurable() -> None:
    """``None`` 은 사람이 그린 마스크(통과), ``NaN`` 은 "재려다 실패"(거부) — 모르는 것을 통과시키지 않는다."""
    assert gate(None, 0.8) is True
    assert gate(0.9, 0.8) is True
    assert gate(0.8, 0.8) is True
    assert gate(0.42, 0.8) is False
    assert gate(float("nan"), 0.8) is False


def test_gate_reason_always_says_something() -> None:
    assert "꺼져" in gate_reason(0.1, 0.0)
    assert "사람이 그린" in gate_reason(None, 0.5)
    assert "재지 못했" in gate_reason(float("nan"), 0.5)
    assert "0.90" in gate_reason(0.9, 0.5) and "≥" in gate_reason(0.9, 0.5)
    assert "<" in gate_reason(0.1, 0.5)


def test_gate_reason_does_not_say_off_while_refusing() -> None:
    """**NaN 은 게이트를 꺼도 거부**다 — 거부해 놓고 "꺼져 있습니다"라고 말하면 사유가 자기모순이 된다."""
    assert gate(float("nan"), 0.0) is False
    assert "재지 못했" in gate_reason(float("nan"), 0.0)
    verdict = auto_admit(
        _cand(confidence=float("nan"), disagreement=None),
        AutoPolicy(score=0.5, require_agreement=False),
        known_classes=["scratch"],
    )
    assert not verdict.admit and "꺼져" not in verdict.reason


def test_partition_by_gate() -> None:
    auto, queue = partition_by_gate([("a", 0.9), ("b", 0.4), ("c", None)], 0.8)
    assert auto == ["a", "c"] and queue == ["b"]


def test_policy_reexports_the_same_function() -> None:
    """`loop.policy.gate` 는 `core.gate.gate` 그 자체여야 한다 — 은행과 루프가 같은 판정을 본다."""
    from anograft.core import gate as core_gate
    from anograft.loop import policy

    assert policy.gate is core_gate.gate


# --------------------------------------------------------------------- 판정(순수)


def _cand(**over) -> AutoCandidate:
    base = {
        "stem": "a",
        "score": 0.96,
        "cls": "scratch",
        "has_mask": True,
        "confidence": 0.8,
        "disagreement": 0,
    }
    base.update(over)
    return AutoCandidate(**base)


def test_auto_admit_opens_only_when_everything_passes() -> None:
    policy = AutoPolicy(score=0.95, min_confidence=0.6)
    verdict = auto_admit(_cand(), policy, known_classes=["scratch"])
    assert verdict.admit and "0.960" in verdict.reason and "일치" in verdict.reason


@pytest.mark.parametrize(
    ("over", "needle"),
    [
        ({"has_mask": False}, "마스크가 없습니다"),
        ({"score": 0.5}, "문턱"),
        ({"confidence": 0.2}, "신뢰도"),
        ({"confidence": float("nan")}, "재지 못했"),
        ({"cls": "blowhole"}, "보관함에 없는 클래스"),
        ({"novelty": 0.9}, "처음 보는 형상"),
        ({"disagreement": 2}, "어긋났습니다"),
        ({"disagreement": None}, "두 번째 예측이 없습니다"),
    ],
)
def test_auto_admit_closes_with_a_reason(over: dict, needle: str) -> None:
    """게이트가 닫힐 때는 **왜**를 말한다 — 조용히 빠지는 조각을 만들지 않는다."""
    policy = AutoPolicy(score=0.95, min_confidence=0.6)
    verdict = auto_admit(_cand(**over), policy, known_classes=["scratch"], novelty_threshold=0.6)
    assert not verdict.admit and needle in verdict.reason


def test_auto_admit_is_off_by_default() -> None:
    """기본값으로는 아무것도 받지 않는다 — 임계값은 코드가 정하지 않는다(설계 §8)."""
    assert AutoPolicy().enabled is False
    assert not auto_admit(_cand(), AutoPolicy()).admit


def test_auto_admit_without_second_prediction_needs_the_switch() -> None:
    """상대가 없으면 **닫힌다**. 열려면 사람이 `require_agreement` 를 끄는 결정을 해야 한다."""
    policy = AutoPolicy(score=0.9)
    assert not auto_admit(_cand(disagreement=None), policy, known_classes=["scratch"]).admit
    open_policy = AutoPolicy(score=0.9, require_agreement=False)
    assert auto_admit(_cand(disagreement=None), open_policy, known_classes=["scratch"]).admit


def test_auto_admit_can_allow_novel_explicitly() -> None:
    policy = AutoPolicy(score=0.9, require_agreement=False, allow_novel=True)
    assert auto_admit(
        _cand(novelty=0.95, disagreement=None),
        policy,
        known_classes=["scratch"],
        novelty_threshold=0.6,
    ).admit


def test_auto_admit_skips_the_class_check_when_the_bank_is_new() -> None:
    """빈 보관함(클래스 목록이 없음)에서는 클래스 검사를 하지 않는다 — 첫 임포트를 막지 않는다."""
    policy = AutoPolicy(score=0.9, require_agreement=False)
    assert auto_admit(_cand(disagreement=None, cls="anything"), policy, known_classes=[]).admit


def test_partition_auto_sends_the_cut_to_the_human_queue() -> None:
    """상한에서 잘린 것은 **버리지 않고** 사람 큐로 — 자동이 못 받은 것도 판정 대상이다."""
    cands = [_cand(stem=s, score=v) for s, v in (("low", 0.96), ("high", 0.99), ("mid", 0.97))]
    policy = AutoPolicy(score=0.95, max_per_round=2)
    admit, held = partition_auto(cands, policy, known_classes=["scratch"])
    assert [c.stem for c in admit] == ["high", "mid"]  # 점수 높은 순
    assert [(c.stem, "상한" in r) for c, r in held] == [("low", True)]


def test_cheap_candidates_counts_what_it_did_not_measure() -> None:
    """점수 문턱 아래는 **이미지를 읽지 않는다**(4K 수천 장을 전부 재면 라운드마다 디스크를 통째로 읽는다)."""
    preds = _pset({"a": 0.99, "b": 0.10, "c": 0.97})
    cands, below = cheap_candidates(preds, AutoPolicy(score=0.95))
    assert [c.stem for c in cands] == ["a", "c"] and below == 1
    assert all(c.confidence is None for c in cands)  # 아직 안 쟀다


def test_cheap_candidates_marks_missing_comparison_as_none() -> None:
    """한쪽에만 있는 stem 은 불일치가 아니라 **결측**이다 — ``None`` 으로 두어 게이트가 닫게 한다."""
    a = _pset({"both": 0.99, "only_a": 0.99})
    b = _pset({"both": 0.99})
    cands, _ = cheap_candidates(a, AutoPolicy(score=0.9), other=b)
    by = {c.stem: c.disagreement for c in cands}
    assert by == {"both": 0, "only_a": None}


def test_cheap_candidates_is_empty_when_off() -> None:
    preds = _pset({"a": 0.99})
    assert cheap_candidates(preds, AutoPolicy()) == ([], 1)


# --------------------------------------------------------------------- 재기(IO)


def test_measure_fills_confidence_from_mask_confidence(tmp_path: Path) -> None:
    """모델 마스크도 추정이라 `mask_confidence` 를 그대로 쓴다(설계 §3 "재사용")."""
    images_dir = _field(tmp_path, ["a"])
    pred = _pred(tmp_path, {"a": 0.99})
    preds = read_predictions(pred)
    images, _ = index_images(sorted(images_dir.glob("*.png")))
    cands, _ = cheap_candidates(preds, AutoPolicy(score=0.9))
    measured = measure(cands, preds, images, box_margin=6)
    assert measured[0].confidence is not None and measured[0].confidence > 0.5


def test_measure_refuses_a_mask_of_the_wrong_size(tmp_path: Path) -> None:
    """이상맵을 모델 해상도로 낸 어댑터(T2 함정의 하류) — NaN 으로 두어 게이트가 거부한다."""
    images_dir = _field(tmp_path, ["a"])
    pred = _pred(tmp_path, {"a": 0.99})
    imgio.write_image(pred / "masks" / "a.png", blob_mask(SIZE // 2, [(16, 16, 4)]))
    preds = read_predictions(pred)
    images, _ = index_images(sorted(images_dir.glob("*.png")))
    cands, _ = cheap_candidates(preds, AutoPolicy(score=0.9))
    measured = measure(cands, preds, images, box_margin=6)
    assert np.isnan(measured[0].confidence)
    assert not auto_admit(
        measured[0], AutoPolicy(score=0.9, min_confidence=0.5, require_agreement=False)
    ).admit


# --------------------------------------------------------------------- 은행 편입


def test_run_auto_imports_with_the_auto_tag_and_pred_origin(tmp_path: Path) -> None:
    """자동으로 들어온 조각은 ``auto`` 태그 + ``pred:*`` — **아무도 안 본 조각**을 나중에 찾을 수 있다."""
    images_dir = _field(tmp_path, ["a", "b"])
    pred = _pred(tmp_path, {"a": 0.99, "b": 0.2})
    preds = read_predictions(pred)
    images, _ = index_images(sorted(images_dir.glob("*.png")))
    bank_dir = tmp_path / "bank"

    summary = run_auto(
        preds,
        images,
        bank_dir,
        AutoPolicy(score=0.95, min_confidence=0.5, require_agreement=False),
        trainer="noop",
        round_no=3,
    )
    assert summary.stems == ["a"] and summary.imported == 1
    assert summary.below_score == 1 and summary.held == []

    bank = Bank.load(bank_dir)
    src = bank.sources()[0]
    assert src.cls == "scratch"
    assert src.mask_origin == "pred:noop" and is_estimated(src.mask_origin)
    assert AUTO_TAG in src.tags and "origin:field" in src.tags and "round-3" in src.tags
    meta = json.loads((bank_dir / "scratch" / f"{src.id.split('/')[-1]}.json").read_text("utf-8"))
    assert meta["confidence"] is not None  # 게이트가 본 점수를 그대로 남긴다


def test_run_auto_dry_run_does_not_touch_the_bank(tmp_path: Path) -> None:
    images_dir = _field(tmp_path, ["a"])
    preds = read_predictions(_pred(tmp_path, {"a": 0.99}))
    images, _ = index_images(sorted(images_dir.glob("*.png")))
    bank_dir = tmp_path / "bank"

    summary = run_auto(
        preds,
        images,
        bank_dir,
        AutoPolicy(score=0.9, require_agreement=False),
        dry_run=True,
    )
    assert summary.stems == ["a"] and summary.imported == 0
    assert not bank_dir.exists()


def test_run_auto_respects_holdout(tmp_path: Path) -> None:
    """평가셋은 자동 편입도 못 지난다 — 막는 지점은 `BankWriter.add` 하나다(T4)."""
    images_dir = _field(tmp_path, ["a"])
    preds = read_predictions(_pred(tmp_path, {"a": 0.99}))
    images, _ = index_images(sorted(images_dir.glob("*.png")))
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    (bank_dir / "holdout.txt").write_text("a\n", encoding="utf-8")

    summary = run_auto(preds, images, bank_dir, AutoPolicy(score=0.9, require_agreement=False))
    assert summary.stems == ["a"] and summary.imported == 0
    # 기록은 **원본 경로**다(감사 + stem 대조가 원본으로 걸리게) — 조용히 빼지 않는다
    assert len(summary.held_out) == 1 and summary.held_out[0].endswith("field/a.png")


def test_run_auto_gate_refuses_low_confidence_at_the_writer(tmp_path: Path) -> None:
    """게이트를 통과한 것만 넘기지만, **쓰기 지점도** 같은 문턱을 본다(임포트 게이트와 한 규칙)."""
    from anograft.bank.importers.pairs import PairRecord, import_pair_records

    images_dir = _field(tmp_path, ["a"])
    mask = tmp_path / "m.png"
    imgio.write_image(mask, blob_mask(SIZE, [BLOB]))
    res = import_pair_records(
        [PairRecord(images_dir / "a.png", mask, "scratch", confidence=0.2)],
        tmp_path / "bank",
        mask_origin="pred:noop",
        min_confidence=0.6,
    )
    assert res.stats.added == []
    assert res.stats.low_confidence and "신뢰도" in res.stats.low_confidence[0][1]


def test_auto_csv_round_trips(tmp_path: Path) -> None:
    cands = [_cand(stem="a", confidence=0.81, novelty=0.12)]
    path = write_auto_csv(tmp_path / AUTO_FILE, cands, {"a": "왜냐하면"}, {"a": tmp_path / "a.png"})
    rows = read_auto_csv(path)
    assert rows[0]["stem"] == "a" and rows[0]["reason"] == "왜냐하면"
    assert rows[0]["confidence"].startswith("0.81") and rows[0]["class"] == "scratch"


# --------------------------------------------------------------------- 비율·자동 정지


def test_auto_stats_rate_is_none_without_intake() -> None:
    """편입이 없으면 ``None`` — **모르는 것과 0 은 다르다**."""
    assert AutoStats().rate is None
    assert "편입된 조각이 없습니다" in AutoStats().text()
    assert AutoStats(auto=3, reviewed=1).rate == pytest.approx(0.75)


def test_auto_window_sums_rounds_instead_of_diffing_them() -> None:
    """편입은 라운드 **안에서** 일어난다 — 사람 수정률처럼 누계의 차이를 볼 필요가 없다."""
    history = [
        RoundOutcome(round=1, intake=4, auto=0),
        RoundOutcome(round=2, intake=1, auto=3),
        RoundOutcome(round=3, intake=0, auto=5),
    ]
    assert auto_window(history).rate == pytest.approx(8 / 13)
    assert auto_window(history, 2).rate == pytest.approx(8 / 9)


def test_circuit_break_trips_on_too_much_automation() -> None:
    """설계 §6.5 의 네 번째 조건 — T6 이 붙어서야 **독립 지표**가 됐다."""
    history = [RoundOutcome(round=1, intake=1, auto=9)]
    tripped = circuit_break(history, BreakerPolicy(max_auto_rate=0.8))
    assert tripped.tripped and "auto_rate" in tripped.kinds and "90%" in tripped.reason
    ok = circuit_break(history, BreakerPolicy(max_auto_rate=0.95))
    assert not ok.tripped and "자동 편입 90%" in ok.reason


def test_breaker_is_still_off_by_default() -> None:
    assert BreakerPolicy().enabled is False
    assert BreakerPolicy(max_auto_rate=0.8).enabled is True


# --------------------------------------------------------------------- CLI


def test_cli_auto_dry_run_then_import(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`loop auto` 는 라운드 단계와 **같은 함수**를 부른다 — 먼저 --dry-run 으로 볼 수 있어야 한다."""
    images_dir = _field(tmp_path, ["a", "b"])
    pred = _pred(tmp_path, {"a": 0.99, "b": 0.30})
    bank_dir = tmp_path / "bank"
    args = [
        "loop",
        "auto",
        "--pred",
        str(pred),
        "--images",
        str(images_dir),
        "--bank",
        str(bank_dir),
        "--score",
        "0.95",
        "--min-confidence",
        "0.5",
        "--no-agreement",
        "--trainer",
        "noop",
        "--json",
    ]
    assert main([*args, "--dry-run"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["admitted"] == ["a"] and payload["imported"] == 0 and payload["belowScore"] == 1
    assert not bank_dir.exists()

    assert main([*args, "--out", str(tmp_path / AUTO_FILE)]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["imported"] == 1 and payload["perClass"] == {"scratch": 1}
    assert read_auto_csv(tmp_path / AUTO_FILE)[0]["stem"] == "a"
    assert AUTO_TAG in Bank.load(bank_dir).sources()[0].tags


def test_cli_auto_needs_agreement_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """교차 검증이 기본이라, 상대 예측 없이는 아무것도 들어가지 않는다(사유는 남는다)."""
    images_dir = _field(tmp_path, ["a"])
    pred = _pred(tmp_path, {"a": 0.99})
    assert (
        main(
            [
                "loop",
                "auto",
                "--pred",
                str(pred),
                "--images",
                str(images_dir),
                "--bank",
                str(tmp_path / "bank"),
                "--score",
                "0.9",
                "--json",
            ]
        )
        == EXIT_OK
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["admitted"] == []
    assert "두 번째 예측" in payload["held"][0]["reason"]


def test_cli_auto_with_two_predictions_only_takes_agreement(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    images_dir = _field(tmp_path, ["same", "off"])
    a = _pred(tmp_path, {"same": 0.99, "off": 0.99}, name="pred-a")
    _pred(
        tmp_path,
        {"same": 0.99, "off": 0.99},
        name="pred-b",
        boxes={"same": [BOX], "off": [BOX, (0.0, 0.0, 8.0, 8.0)]},
    )
    assert (
        main(
            [
                "loop",
                "auto",
                "--pred",
                str(a),
                "--pred-b",
                str(tmp_path / "pred-b"),
                "--images",
                str(images_dir),
                "--bank",
                str(tmp_path / "bank"),
                "--score",
                "0.9",
                "--json",
            ]
        )
        == EXIT_OK
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["admitted"] == ["same"]
    assert payload["held"][0]["stem"] == "off" and "어긋났" in payload["held"][0]["reason"]


# --------------------------------------------------------------------- 라운드 안에서


@pytest.fixture
def loop_ws(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Path]:
    ws = loop_workspace(tmp_path)
    capsys.readouterr()
    return ws


def _enable_auto(loop_yaml: Path, **over: object) -> None:
    """작업장의 `loop.yaml` 에서 자동 편입을 켠다(문턱은 낮게 — noop 점수는 파일 이름에서 파생된다)."""
    data = yaml.safe_load(loop_yaml.read_text(encoding="utf-8"))
    data["auto"] = {"score": 0.05, "require_agreement": False, **over}
    loop_yaml.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def _judge_all(queue_dir: Path, verdict: str = "accept") -> int:
    rows = [r for r in read_manifest(queue_dir / MANIFEST_FILE) if r.get("status") == "ok"]
    write_review(queue_dir / REVIEW_FILE, {r["index"]: (verdict, "") for r in rows})
    return len(rows)


def test_round_runs_the_auto_phase_before_the_queue(loop_ws: dict[str, Path]) -> None:
    """라운드 단계 `auto` 는 `queue` **앞**에 서고, 받은 것은 검토 대기에서 빠진다.

    자동으로 받은 것을 큐에 또 넣으면 사람이 **이미 은행에 있는 것**을 판정하게 된다.
    """
    # 상한 1 — 현장 이미지가 둘뿐이라 상한이 없으면 자동이 전부 받아 큐가 빈다(아래 테스트가 그 경우다)
    _enable_auto(loop_ws["loop"], max_per_round=1)
    loop = load_loop_config(loop_ws["loop"])

    first = run_round(loop)  # 부트스트랩 — 모델이 없으니 수집 단계를 건너뛴다
    assert "auto" not in first.record.phases and first.state.champion is not None

    second = run_round(loop)  # champion 이 있다 → predict → auto → queue → (사람)
    assert second.record.phases[:3] == ("predict", "auto", "queue")
    auto = second.record.data["auto"]
    assert auto["imported"] >= 1 and auto["stems"]
    csv_rows = read_auto_csv(second.round_dir / AUTO_FILE)
    assert csv_rows and all(r["reason"] for r in csv_rows)

    # 큐에 들어간 stem 과 자동으로 받은 stem 은 겹치지 않는다
    assert second.waiting_for_human  # 상한에서 잘린 것이 사람 큐로 갔다
    in_queue = {r["stem"] for r in read_queue_csv(second.round_dir / "queue" / "queue.csv")}
    assert in_queue and not (in_queue & set(auto["stems"]))

    # 보관함에는 auto 태그 + 모델 초안으로 들어갔다
    tagged = [s for s in Bank.load(loop_ws["bank"]).sources() if AUTO_TAG in s.tags]
    assert len(tagged) == auto["imported"]
    assert all(s.mask_origin == "pred:noop" for s in tagged)

    # 끝까지 돌리면 원장 `round_end.auto` 가 남는다(자동 편입 비율의 분자)
    _judge_all(second.round_dir / "queue")
    third = run_round(loop)
    assert third.record.number == second.record.number and third.record.done[-1] == "judge"
    end = ledger.read(loop_ws["out"] / ledger.ROUNDS_FILE).last_end
    assert end is not None and end.get("auto") == auto["imported"]


def test_auto_taking_everything_leaves_an_empty_queue(loop_ws: dict[str, Path]) -> None:
    """자동이 전부 받으면 **사람을 기다리지 않는다** — 큐에 넣을 것이 없으면 라운드가 그대로 끝까지 간다.

    이것이 게이트를 너무 헐겁게 열었을 때의 모습이고, 그래서 `breaker.max_auto_rate` 가 짝으로 있다.
    """
    _enable_auto(loop_ws["loop"])
    loop = load_loop_config(loop_ws["loop"])
    run_round(loop)
    second = run_round(loop)
    assert not second.waiting_for_human and second.record.done[-1] == "judge"
    assert "전부" in second.record.data["queue"].get("note", "")
    assert second.record.data["accept"]["imported"] == 0
    end = ledger.read(loop_ws["out"] / ledger.ROUNDS_FILE).last_end
    assert end is not None and end.get("auto") >= 1 and end.get("intake") == 0


def test_status_and_board_report_the_auto_rate(loop_ws: dict[str, Path]) -> None:
    """`loop status` 와 현황판이 **같은 사실**을 본다 — 판정은 `policy.auto_window` 하나다."""
    _enable_auto(loop_ws["loop"], max_per_round=1)
    loop = load_loop_config(loop_ws["loop"])
    run_round(loop)  # 부트스트랩
    second = run_round(loop)
    _judge_all(second.round_dir / "queue")
    run_round(loop)

    st = status(loop)
    assert st.auto_enabled and st.auto is not None and st.auto.auto >= 1
    assert any("자동 편입" in line for line in st.lines())

    rows = board.round_rows(ledger.read(loop_ws["out"] / ledger.ROUNDS_FILE))
    assert rows[0].auto >= 1 and rows[0].to_json()["autoRate"] is not None
    assert board.auto_totals(rows).auto == sum(r.auto for r in rows)


def test_round_without_auto_has_no_phase_and_no_ledger_noise(loop_ws: dict[str, Path]) -> None:
    """켜지 않으면 단계가 서지 않고 원장에도 0 만 남는다 — 화면이 없는 것을 묻지 않게."""
    loop = load_loop_config(loop_ws["loop"])
    run_round(loop)
    end = ledger.read(loop_ws["out"] / ledger.ROUNDS_FILE).last_end
    assert end is not None and end.get("auto") == 0
    assert not status(loop).auto_enabled


# --------------------------------------------------------------------- 임포트 게이트(CLI)


def test_cli_import_yolo_min_confidence_refuses_and_says_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`bank import-yolo --min-confidence` — 조용히 건너뛰지 않는다(holdout 거부와 같은 규율)."""
    d = fake_yolo_dataset(tmp_path / "ds")

    def run(out: Path, *extra: str) -> int:
        return main(
            [
                "bank",
                "import-yolo",
                "--images",
                str(d["images"]),
                "--labels",
                str(d["labels"]),
                "--names",
                str(d["names"]),
                "--out",
                str(out),
                "--mask-from",
                "otsu",
                *extra,
            ]
        )

    plain = tmp_path / "plain"
    assert run(plain) == EXIT_OK
    capsys.readouterr()
    estimated = [s for s in Bank.load(plain).sources() if is_estimated(s.mask_origin)]
    assert estimated  # 게이트를 끄면 박스 추정도 들어온다
    scores = [c for c in (_confidence_of(plain, s) for s in estimated) if c is not None]
    assert scores

    # 관측된 최고 점수보다 높은 문턱 — 추정 마스크는 **전부** 막히고 사유가 남는다
    gated = tmp_path / "gated"
    assert run(gated, "--min-confidence", f"{max(scores) + 0.01:.3f}") == EXIT_OK
    assert "신뢰도 게이트" in capsys.readouterr().err
    # 게이트는 **추정 마스크에만** 걸린다 — 폴리곤 라벨(정확한 마스크)은 그대로 들어온다
    left = Bank.load(gated).sources()
    assert left and all(not is_estimated(s.mask_origin) for s in left)


def _confidence_of(bank_root: Path, source) -> float | None:
    """은행 메타의 ``confidence`` — 게이트가 보는 그 값이다."""
    cls, sid = source.id.split("/", 1)
    meta = json.loads((bank_root / cls / f"{sid}.json").read_text(encoding="utf-8"))
    value = meta.get("confidence")
    return None if value is None else float(value)

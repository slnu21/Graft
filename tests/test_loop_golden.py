"""이중 평가셋 — 고정(동결)과 최근(갱신) (설계 §2b.6, 작업 단위 T16).

여기가 못 박는 것:

1. **지문은 내용까지 본다** — 크기만 같은 다른 사진, 정답만 고친 경우도 "다른 평가셋"이다. 여기서
   "안 바뀌었다"를 잘못 말하면 서로 다른 평가셋의 점수가 나란히 그려진다(이 단위가 막는 유일한 거짓말).
2. **갱신은 추이를 끊고 판정은 살린다** — champion 을 새 평가셋에서 다시 재기 때문이다. 그래서
   `metric_points`(고정)는 갱신에 끊기지 않고 `rolling_points`(최근)만 끊긴다.
3. **선택 verb 다** — `eval` 을 선언하지 않은 학습기로도 라운드는 끝까지 돈다(고정만으로 판정 + 사유).
4. **최근에서 개선이 없으면 승급하지 않는다** — 고정이 올랐어도 그렇다(설계 §2b.6 의 승급 조건).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from anograft.io import imgio
from anograft.io.manifest import MANIFEST_FILE, read_manifest
from anograft.io.prune import REVIEW_FILE, write_review
from anograft.loop import ledger as L
from anograft.loop.board import metric_points, rolling_points, round_rows
from anograft.loop.config import load_loop_config
from anograft.loop.contract import TrainerError, evaluate, parse_eval, parse_info
from anograft.loop.golden import (
    RollingBaseline,
    file_entry,
    fingerprint,
    rolling_plan,
    split_entries,
    split_fingerprint,
)
from anograft.loop.round import (
    PHASES,
    LoopState,
    assemble_eval_dataset,
    load_state,
    round_name,
    round_phases,
    run_round,
    save_state,
    status,
)
from tests.fixtures import blob_image, blob_mask, loop_workspace

REPO_ROOT = Path(__file__).resolve().parents[1]
NOOP = REPO_ROOT / "adapters" / "noop.py"


# --------------------------------------------------------------------- 지문 (순수 + 파일)


def _split(root: Path, stems: list[str], *, radius: int = 7) -> dict[str, Path | None]:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    for s in stems:
        imgio.write_image(root / "images" / f"{s}.png", blob_image(64, [(32, 32, radius)]))
        imgio.write_image(root / "masks" / f"{s}.png", blob_mask(64, [(32, 32, radius)]))
    return {"images": root / "images", "masks": root / "masks", "labels": None}


def test_fingerprint_is_order_independent_but_content_sensitive() -> None:
    a = [("img/a.png", 10, "aaaaaaaa"), ("img/b.png", 20, "bbbbbbbb")]
    assert fingerprint(a) == fingerprint(list(reversed(a)))
    assert fingerprint(a) != fingerprint([("img/a.png", 10, "aaaaaaaa")])
    # 같은 이름·같은 크기라도 내용이 다르면 다른 평가셋이다
    assert fingerprint(a) != fingerprint([("img/a.png", 10, "cccccccc"), a[1]])


def test_file_entry_marks_unreadable_files_instead_of_guessing(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    p.write_bytes(b"1234")
    assert file_entry("img/a.png", p) == ("img/a.png", 4, file_entry("img/a.png", p)[2])
    assert file_entry("img/missing.png", tmp_path / "missing.png") == ("img/missing.png", -1, "")


def test_split_fingerprint_sees_labels_too(tmp_path: Path) -> None:
    """정답만 고친 것도 **다른 평가셋**이다 — 라벨을 지문에서 빼면 갱신을 놓친다."""
    split = _split(tmp_path / "recent", ["a", "b"])
    before, n = split_fingerprint(split)
    assert n == 2 and before

    masks = split["masks"]
    assert masks is not None
    imgio.write_image(masks / "a.png", blob_mask(64, [(30, 30, 9)]))
    after, _ = split_fingerprint(split)
    assert after != before
    assert len(split_entries(split)) == 4  # 이미지 둘 + 마스크 둘


def test_empty_split_has_no_fingerprint(tmp_path: Path) -> None:
    """지문이 없으면 판정도 하지 않는다 — 빈 폴더를 "갱신" 으로 읽으면 매 라운드 끊긴다."""
    empty = tmp_path / "none"
    (empty / "images").mkdir(parents=True)
    assert split_fingerprint({"images": empty / "images", "masks": None}) == ("", 0)


# --------------------------------------------------------------------- 판정 (순수)


def test_rolling_plan_without_a_champion_does_not_measure() -> None:
    plan = rolling_plan(baseline=None, fingerprint="aa", champion_model="")
    assert plan.remeasure is False and plan.changed is False
    assert "champion" in plan.reason


def test_rolling_plan_measures_when_there_is_no_baseline() -> None:
    plan = rolling_plan(baseline=None, fingerprint="aa", champion_model="m.pt")
    assert plan.remeasure is True and plan.changed is False


def test_rolling_plan_breaks_the_trend_when_the_set_changed() -> None:
    base = RollingBaseline(fingerprint="aa", metric=0.4, round=3, model="m.pt")
    plan = rolling_plan(baseline=base, fingerprint="bb", champion_model="m.pt")
    assert plan.remeasure is True and plan.changed is True
    assert "aa" in plan.reason and "bb" in plan.reason


def test_rolling_plan_remeasures_a_baseline_from_another_model() -> None:
    base = RollingBaseline(fingerprint="aa", metric=0.4, round=3, model="old.pt")
    plan = rolling_plan(baseline=base, fingerprint="aa", champion_model="new.pt")
    assert plan.remeasure is True and plan.changed is False


def test_rolling_plan_keeps_the_baseline_when_nothing_moved() -> None:
    base = RollingBaseline(fingerprint="aa", metric=0.4, round=3, model="m.pt")
    plan = rolling_plan(baseline=base, fingerprint="aa", champion_model="m.pt")
    assert plan.remeasure is False and plan.changed is False


def test_rolling_baseline_round_trip_and_broken_shapes(tmp_path: Path) -> None:
    base = RollingBaseline(fingerprint="aa", metric=0.42, round=2, model="m.pt")
    assert RollingBaseline.from_dict(base.to_dict()) == base
    assert RollingBaseline.from_dict({"fingerprint": "aa"}) is None  # 점수 없음
    assert RollingBaseline.from_dict({"metric": 0.4}) is None  # 무엇을 잰 건지 모름

    state = LoopState(round=2, rolling=base)
    save_state(tmp_path, state)
    back = load_state(tmp_path)
    assert back.rolling == base


# --------------------------------------------------------------------- 계약 (선택 verb)


def test_parse_eval_refuses_a_result_without_numbers() -> None:
    assert parse_eval({"metrics": {"mAP50": 0.4, "note": "x"}}).metrics == {"mAP50": 0.4}
    with pytest.raises(TrainerError, match="숫자 지표"):
        parse_eval({"metrics": {}})


def test_capabilities_accept_eval_and_expose_it() -> None:
    info = parse_info(
        {
            "name": "x",
            "dataset_format": "pairs",
            "trains_on": "labeled",
            "capabilities": ["score", "eval"],
        }
    )
    assert info.can_eval is True
    plain = parse_info(
        {"name": "x", "dataset_format": "pairs", "trains_on": "labeled", "capabilities": ["score"]}
    )
    assert plain.can_eval is False


def test_noop_adapter_evaluates_without_training(tmp_path: Path) -> None:
    """더미가 `eval` 을 하면 루프 전체를 torch 없이 시험할 수 있다(§1.6 의 연장)."""
    from anograft.loop.contract import fit

    dataset = tmp_path / "ds"
    (dataset / "val").mkdir(parents=True)
    (dataset / "val" / "a.png").write_bytes(b"x")
    trained = fit([sys.executable, str(NOOP)], dataset=dataset, out=tmp_path / "m", seed=7)

    # 같은 모델·같은 데이터셋이면 fit 의 지표와 같은 값(같은 규칙을 쓴다)
    same = evaluate([sys.executable, str(NOOP)], model=trained.model, dataset=dataset, seed=7)
    assert same.metrics["mAP50"] == pytest.approx(trained.metrics["mAP50"])

    # 데이터셋이 달라지면 값이 달라진다 → 평가셋을 갈면 추이가 끊겨야 하는 이유
    other = tmp_path / "ds2"
    (other / "val").mkdir(parents=True)
    for k in range(5):
        (other / "val" / f"{k}.png").write_bytes(b"x")
    assert evaluate(
        [sys.executable, str(NOOP)], model=trained.model, dataset=other, seed=7
    ).metrics["mAP50"] != pytest.approx(trained.metrics["mAP50"])


# --------------------------------------------------------------------- 데이터셋 조립


def test_assemble_eval_dataset_has_val_only(tmp_path: Path) -> None:
    """**학습이 없는 것이 정상**인 데이터셋 — `assemble_dataset` 과 갈라 둔 이유다."""
    split = _split(tmp_path / "recent", ["a", "b"])
    out, warns = assemble_eval_dataset("pairs", tmp_path / "ds", split=split, names=["scratch"])
    assert not warns
    assert sorted(p.name for p in (out / "val" / "images").iterdir()) == ["a.png", "b.png"]
    assert not (out / "train").exists()

    yolo_split = {"images": split["images"], "labels": None, "masks": None}
    out2, warns2 = assemble_eval_dataset(
        "yolo", tmp_path / "ds2", split=yolo_split, names=["scratch"]
    )
    doc = yaml.safe_load((out2 / "data.yaml").read_text(encoding="utf-8"))
    assert doc["val"] == "images/val" and doc["train"] == "images/val"
    assert any("라벨이 없어" in w for w in warns2)


def test_assemble_eval_dataset_refuses_an_empty_set(tmp_path: Path) -> None:
    empty = tmp_path / "none"
    (empty / "images").mkdir(parents=True)
    with pytest.raises(Exception, match="0장"):
        assemble_eval_dataset(
            "pairs",
            tmp_path / "ds",
            split={"images": empty / "images", "masks": None},
            names=["scratch"],
        )


# --------------------------------------------------------------------- 현황판 (순수)


def _ledger(
    tmp_path: Path, rows: list[dict], events: list[tuple[str, dict]] | None = None
) -> L.Ledger:
    path = tmp_path / "rounds.jsonl"
    for row in rows:
        L.append(path, L.EVENT_ROUND_END, round_no=row.pop("round"), **row)
    for name, data in events or []:
        L.append(path, name, round_no=int(data.get("after_round", 0)) + 1, **data)
    return L.read(path)


def test_rolling_update_breaks_only_the_recent_trend(tmp_path: Path) -> None:
    """고정 평가셋은 그대로다 — 그 선까지 끊으면 "회귀 감시가 끊겼다"는 없는 사실이 화면에 생긴다."""
    led = _ledger(
        tmp_path,
        [
            {"round": 1, "metric": 0.30, "rolling_metric": 0.31, "rolling_fingerprint": "aa"},
            {"round": 2, "metric": 0.34, "rolling_metric": 0.35, "rolling_fingerprint": "aa"},
            {"round": 3, "metric": 0.36, "rolling_metric": 0.28, "rolling_fingerprint": "bb"},
        ],
        [(L.EVENT_ROLLING_UPDATE, {"after_round": 2, "fingerprint": "bb", "note": "9월분"})],
    )
    rows = round_rows(led)
    assert [r.round for r in rows] == [3, 2, 1]  # 표는 최신이 위
    assert rows[0].rolling == pytest.approx(0.28) and rows[0].rolling_fingerprint == "bb"
    assert rows[1].rolling_marker == L.EVENT_ROLLING_UPDATE
    assert rows[1].rolling_marker_label == "최근 평가셋 갱신"
    assert rows[1].marker == ""  # 판정 지점이 아니다

    fixed = metric_points(rows)
    assert [p.segment for p in fixed] == [0, 0, 0]  # 고정 추이는 이어진다
    recent = rolling_points(rows)
    assert [p.round for p in recent] == [1, 2, 3]
    assert [p.segment for p in recent] == [0, 0, 1]  # 갱신 뒤로 끊긴다


def test_baseline_reset_breaks_both_trends(tmp_path: Path) -> None:
    led = _ledger(
        tmp_path,
        [
            {"round": 1, "metric": 0.30, "rolling_metric": 0.31, "rolling_fingerprint": "aa"},
            {"round": 2, "metric": 0.34, "rolling_metric": 0.35, "rolling_fingerprint": "aa"},
        ],
        [(L.EVENT_BASELINE_RESET, {"after_round": 1, "note": "blowhole 신설"})],
    )
    rows = round_rows(led)
    assert [p.segment for p in metric_points(rows)] == [0, 1]
    assert [p.segment for p in rolling_points(rows)] == [0, 1]


def test_rounds_without_a_recent_score_have_no_recent_point(tmp_path: Path) -> None:
    """옛 원장·최근 평가셋을 안 쓰던 라운드에는 키가 없다 — 0 으로 그리지 않는다."""
    led = _ledger(tmp_path, [{"round": 1, "metric": 0.3}, {"round": 2, "metric": 0.35}])
    rows = round_rows(led)
    assert all(r.rolling is None for r in rows)
    assert rolling_points(rows) == []
    assert len(metric_points(rows)) == 2


# --------------------------------------------------------------------- 한 바퀴 (noop 어댑터)


@pytest.fixture
def loop_ws(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Path]:
    ws = loop_workspace(tmp_path)
    capsys.readouterr()
    return ws


def _with_rolling(ws: dict[str, Path], *, radius: int = 7, stems: tuple[str, ...] = ("c0", "c1")):
    """`loop.yaml` 에 최근 평가셋을 붙인다(고정 평가셋과 다른 폴더·다른 장수)."""
    root = ws["root"] / "recent"
    _split(root, list(stems), radius=radius)
    doc = yaml.safe_load(ws["loop"].read_text(encoding="utf-8"))
    doc["eval_rolling"] = {
        "images": str(root / "images"),
        "masks": str(root / "masks"),
    }
    ws["loop"].write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return root


def _set_bias(ws: dict[str, Path], bias: float) -> None:
    """더미 학습기의 점수를 옮긴다 — **모델마다 다른 점수**가 나야 champion vs challenger 가 시험된다."""
    doc = yaml.safe_load(ws["loop"].read_text(encoding="utf-8"))
    doc["spec"] = {"bias": bias}
    ws["loop"].write_text(
        yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _judge_all(queue_dir: Path, verdict: str = "accept") -> int:
    rows = [r for r in read_manifest(queue_dir / MANIFEST_FILE) if r.get("status") == "ok"]
    write_review(queue_dir / REVIEW_FILE, {r["index"]: (verdict, "") for r in rows})
    return len(rows)


def test_rolling_phase_only_exists_when_configured() -> None:
    assert "rolling" not in round_phases(has_champion=True, has_field=True)
    assert "rolling" in round_phases(has_champion=True, has_field=True, has_rolling=True)
    assert round_phases(has_champion=False, has_field=False, has_rolling=True) == (
        "synth",
        "train",
        "rolling",
        "judge",
    )
    assert "rolling" in PHASES  # 전체 목록에는 있다(옛 라운드는 자기 목록을 들고 있다)


def test_first_round_measures_the_recent_set_and_stores_the_baseline(
    loop_ws: dict[str, Path],
) -> None:
    _with_rolling(loop_ws)
    logs: list[str] = []
    first = run_round(load_loop_config(loop_ws["loop"]), on_log=logs.append)

    assert first.record.done == ["synth", "train", "rolling", "judge"]
    data = first.record.data["rolling"]
    assert data["images"] == 2 and data["fingerprint"]
    assert data["metric"] is not None and data["changed"] is False
    # champion 이 없었으니 재측정은 없다 — 견줄 것이 없는 라운드다
    assert data["remeasured"] is False and data["champion_metric"] is None

    # 승급(첫 모델)과 함께 **기준선이 옮겨진다** — 안 옮기면 다음 라운드가 champion 을 또 잰다
    baseline = first.state.rolling
    assert baseline is not None
    assert baseline.fingerprint == data["fingerprint"]
    assert baseline.metric == pytest.approx(data["metric"])
    assert baseline.round == 1

    end = L.read(loop_ws["out"] / L.ROUNDS_FILE).last_end
    assert end is not None and end.get("rolling_metric") == pytest.approx(data["metric"])
    assert end.get("rolling_fingerprint") == data["fingerprint"]
    assert "최근" in first.message


def test_recent_set_gates_promotion_even_when_the_fixed_one_improves(
    loop_ws: dict[str, Path],
) -> None:
    """승급 = **고정 회귀 없음 ∧ 최근 개선**. 더미는 모델이 같으면 최근 점수가 그대로다 → 유지."""
    _with_rolling(loop_ws)
    loop = load_loop_config(loop_ws["loop"])
    run_round(loop)  # 1라운드 — 부트스트랩
    second = run_round(loop)
    assert second.waiting_for_human
    assert _judge_all(second.round_dir / "queue") > 0

    done = run_round(load_loop_config(loop_ws["loop"]))
    judge = done.record.data["judge"]
    rolling = done.record.data["rolling"]
    assert rolling["remeasured"] is False  # 평가셋이 그대로라 champion 을 다시 재지 않는다
    assert rolling["champion_metric"] == pytest.approx(judge["rolling_champion"])
    assert judge["promote"] is False and "최근 평가셋" in judge["reason"]
    # 유지했으므로 champion·기준선은 1라운드 것 그대로다
    assert done.state.champion is not None and done.state.champion.round == 1
    assert done.state.rolling is not None and done.state.rolling.round == 1


def test_a_refreshed_recent_set_breaks_the_trend_but_not_the_verdict(
    loop_ws: dict[str, Path],
) -> None:
    """갱신 지점은 원장에 남고(추이가 끊긴다) champion 은 **새 평가셋에서 다시 재진다**(판정은 산다)."""
    _with_rolling(loop_ws)
    loop = load_loop_config(loop_ws["loop"])
    run_round(loop)  # 1라운드
    first_fingerprint = load_state(loop_ws["out"]).rolling
    assert first_fingerprint is not None

    # 사람이 최근 평가셋을 새로 채웠다 + 새 모델은 더 좋다(더미의 bias)
    root = loop_ws["root"] / "recent"
    imgio.write_image(root / "images" / "c2.png", blob_image(64, [(28, 28, 9)]))
    imgio.write_image(root / "masks" / "c2.png", blob_mask(64, [(28, 28, 9)]))
    _set_bias(loop_ws, 0.2)

    second = run_round(load_loop_config(loop_ws["loop"]))
    assert second.waiting_for_human
    _judge_all(second.round_dir / "queue")
    done = run_round(load_loop_config(loop_ws["loop"]))

    rolling = done.record.data["rolling"]
    assert rolling["changed"] is True and rolling["remeasured"] is True
    assert rolling["fingerprint"] != first_fingerprint.fingerprint
    assert rolling["images"] == 3
    # champion 을 새 평가셋에서 다시 쟀으므로 둘을 견줄 수 있다 → 개선이 보이면 승급
    assert rolling["champion_metric"] is not None
    assert rolling["metric"] > rolling["champion_metric"]
    assert done.record.data["judge"]["promote"] is True

    led = L.read(loop_ws["out"] / L.ROUNDS_FILE)
    update = led.last(L.EVENT_ROLLING_UPDATE)
    assert update is not None
    assert update.get("previous") == first_fingerprint.fingerprint
    assert update.get("after_round") == 1  # 이 라운드 앞에서 끊긴다
    rows = round_rows(led)
    assert [p.segment for p in metric_points(rows)] == [0, 0]  # 고정은 이어진다
    assert [p.segment for p in rolling_points(rows)] == [0, 1]  # 최근은 끊긴다

    # 승급했으니 기준선이 새 모델·새 지문으로 옮겨졌다
    baseline = done.state.rolling
    assert baseline is not None and baseline.round == 2
    assert baseline.fingerprint == rolling["fingerprint"]


def test_a_trainer_without_eval_still_finishes_the_round(
    loop_ws: dict[str, Path], tmp_path: Path
) -> None:
    """선택 verb 다 — 못 재면 **사유를 남기고** 고정 평가셋만으로 판정한다(라운드를 죽이지 않는다)."""
    _with_rolling(loop_ws)
    stub = tmp_path / "noeval.py"
    stub.write_text(
        "\n".join(
            [
                "import json, sys",
                "verb = sys.argv[1]",
                "args = dict(zip(sys.argv[2::2], sys.argv[3::2]))",
                "if verb == 'info':",
                "    print(json.dumps({'name': 'noeval', 'version': '0',",
                "        'capabilities': ['score', 'mask'], 'dataset_format': 'pairs',",
                "        'trains_on': 'labeled', 'deterministic': True}))",
                "elif verb == 'fit':",
                "    from pathlib import Path",
                "    out = Path(args['--out']); out.mkdir(parents=True, exist_ok=True)",
                "    (out / 'model.json').write_text('{}', encoding='utf-8')",
                "    print(json.dumps({'model': str(out / 'model.json'),",
                "        'metrics': {'mAP50': 0.5}}))",
                "else:",
                "    print('eval 은 못 합니다', file=sys.stderr)",
                "    raise SystemExit(2)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trainers = tmp_path / "trainers-noeval.yaml"
    trainers.write_text(
        yaml.safe_dump({"trainers": {"noeval": {"command": [sys.executable, str(stub)]}}}),
        encoding="utf-8",
    )
    doc = yaml.safe_load(loop_ws["loop"].read_text(encoding="utf-8"))
    doc["trainer"] = "noeval"
    doc["trainers_file"] = str(trainers)
    loop_yaml = tmp_path / "loop-noeval.yaml"
    loop_yaml.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")

    first = run_round(load_loop_config(loop_yaml))
    assert first.record.done == ["synth", "train", "rolling", "judge"]
    assert "eval 을 선언하지 않아" in first.record.data["rolling"]["skipped"]
    assert any("eval 을 선언하지 않아" in w for w in first.record.warnings)
    assert first.record.data["judge"]["promote"] is True  # 첫 모델 — 고정만으로 판정
    assert (
        "rolling_metric"
        not in (
            L.read(load_loop_config(loop_yaml).out / L.ROUNDS_FILE).last_end or L.Event("x")
        ).data
    )


def test_an_empty_recent_folder_is_a_warning_not_a_failure(loop_ws: dict[str, Path]) -> None:
    root = _with_rolling(loop_ws)
    for p in (root / "images").iterdir():
        p.unlink()
    first = run_round(load_loop_config(loop_ws["loop"]))
    assert "이미지가 없습니다" in first.record.data["rolling"]["skipped"]
    assert first.state.rolling is None  # 못 쟀으면 기준선을 세우지 않는다


def test_status_and_cli_report_both_sets(
    loop_ws: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    from anograft.cli import EXIT_OK, main

    _with_rolling(loop_ws)
    run_round(load_loop_config(loop_ws["loop"]))
    st = status(load_loop_config(loop_ws["loop"]))
    text = " ".join(st.lines())
    assert "고정" in text and "최근 평가셋:" in text and st.rolling_configured

    assert main(["loop", "status", "--config", str(loop_ws["loop"]), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["rolling"]["configured"] is True
    assert payload["rolling"]["metric"] is not None
    assert payload["rolling"]["fingerprint"]


def test_status_says_nothing_about_the_recent_set_when_it_is_not_configured(
    loop_ws: dict[str, Path],
) -> None:
    """안 쓰는 것을 묻지 않는다 — 루프는 선택 계층이고 최근 평가셋은 그 안의 선택이다."""
    run_round(load_loop_config(loop_ws["loop"]))
    st = status(load_loop_config(loop_ws["loop"]))
    assert st.rolling_configured is False
    assert all("최근 평가셋" not in line for line in st.lines())
    assert (loop_ws["out"] / round_name(1) / "rolling").exists() is False

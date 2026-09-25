"""라운드 오케스트레이션 (설계 T10) — 단계 계획(순수) · 데이터셋 조립 · **noop 어댑터로 한 바퀴**.

여기가 못 박는 것:

1. **부트스트랩 라운드**(모델 없음)는 수집 넷을 건너뛰고 합성부터 돈다 — 첫 라운드엔 스코어링할 모델이 없다.
2. **사람 앞에서 멈춘다** — 검토 대기가 판정되지 않으면 `run_round` 가 진행률과 함께 서고, 판정 뒤 다시
   부르면 **그 다음 단계부터** 이어 간다(멱등·부분 진행 재사용).
3. **평가는 학습 데이터셋의 val = 동결 평가셋**이고 `fit` 의 지표가 곧 라운드 점수다.
4. 정상만 학습하는 어댑터(`trains_on: normal_only`)는 **거부**한다 — 합성 결함을 먹이면 오염이다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from anograft.bank import Bank
from anograft.cli import EXIT_OK, EXIT_RECIPE_ERROR, main
from anograft.io import imgio
from anograft.io.manifest import MANIFEST_FILE, read_manifest
from anograft.io.prune import REVIEW_FILE, write_review
from anograft.loop.config import load_loop_config
from anograft.loop.round import (
    COLLECT_PHASES,
    PHASES,
    Champion,
    LoopError,
    LoopState,
    RoundRecord,
    assemble_dataset,
    check_label_classes,
    load_record,
    load_state,
    next_phase,
    round_name,
    round_phases,
    run_round,
    save_record,
    save_state,
    status,
)
from tests.fixtures import blob_image, blob_mask, fake_yolo_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
NOOP = REPO_ROOT / "adapters" / "noop.py"


# --------------------------------------------------------------------- 순수


def test_round_phases_skips_collection_without_a_model_or_field() -> None:
    """첫 라운드엔 스코어링할 모델이 없다 — 그게 부트스트랩이고, 합성이 그 자리를 메운다."""
    assert round_phases(has_champion=True, has_field=True) == PHASES
    boot = round_phases(has_champion=False, has_field=True)
    assert boot == ("synth", "train", "judge")
    assert not set(boot) & set(COLLECT_PHASES)
    assert round_phases(has_champion=True, has_field=False) == boot


def test_next_phase_is_the_first_unfinished_one() -> None:
    assert next_phase(PHASES, []) == "predict"
    assert next_phase(PHASES, ["predict", "queue"]) == "review"
    assert next_phase(PHASES, list(PHASES)) is None
    # 건너뛴 단계가 done 에 있어도 순서는 phases 가 정한다
    assert next_phase(("synth", "train"), ["synth"]) == "train"


def test_round_record_round_trip(tmp_path: Path) -> None:
    rec = RoundRecord(number=3, phases=PHASES)
    rec.mark("predict", {"count": 12})
    save_record(tmp_path, rec)
    back = load_record(tmp_path)
    assert back is not None
    assert back.number == 3 and back.done == ["predict"] and back.data["predict"]["count"] == 12
    assert not back.finished


def test_loop_state_round_trip(tmp_path: Path) -> None:
    state = LoopState(round=2, champion=Champion(round=1, model="m.pt", metric=0.42))
    save_state(tmp_path, state)
    back = load_state(tmp_path)
    assert back.round == 2 and back.champion is not None
    assert back.champion.model == "m.pt" and back.champion.metric == pytest.approx(0.42)


def test_check_label_classes_flags_ids_outside_the_bank(tmp_path: Path) -> None:
    """평가셋 라벨이 보관함 클래스 순서와 어긋나면 **평가가 조용히 무의미해진다**."""
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2" + chr(10), encoding="utf-8")
    assert check_label_classes(labels, 2) == []
    (labels / "b.txt").write_text("5 0.5 0.5 0.2 0.2" + chr(10), encoding="utf-8")
    assert any("class id" in w for w in check_label_classes(labels, 2))


# --------------------------------------------------------------------- 데이터셋 조립


def _yolo_split(root: Path, stems: list[str], *, labels: bool = True) -> dict[str, Path]:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for s in stems:
        imgio.write_image(root / "images" / f"{s}.png", blob_image(48, [(24, 24, 6)]))
        if labels:
            (root / "labels" / f"{s}.txt").write_text(
                "0 0.5 0.5 0.25 0.25" + chr(10), encoding="utf-8"
            )
    return {"images": root / "images", "labels": root / "labels", "masks": None}


def test_assemble_dataset_yolo_merges_real_and_synthetic(tmp_path: Path) -> None:
    real = _yolo_split(tmp_path / "real", ["r0", "r1"])
    gold = _yolo_split(tmp_path / "gold", ["g0"])
    synth = tmp_path / "synth"
    _yolo_split(synth, ["000000"])

    out, warns = assemble_dataset(
        "yolo",
        tmp_path / "ds",
        synth=synth,
        train_base=real,
        eval_split=gold,
        names=["scratch", "dent"],
    )
    train = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert train == ["r0.png", "r1.png", "syn_000000.png"]  # 합성은 syn_ 접두로 구분된다
    assert (out / "labels" / "train" / "syn_000000.txt").is_file()
    assert [p.name for p in (out / "images" / "val").iterdir()] == ["g0.png"]
    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == ["scratch", "dent"] and data["nc"] == 2
    assert data["train"] == "images/train" and data["val"] == "images/val"
    assert warns == []


def test_assemble_dataset_makes_empty_labels_for_background_images(tmp_path: Path) -> None:
    """라벨 없는 이미지 = 배경(정상)이라는 YOLO 규약 — 빈 라벨을 만들고 **몇 장인지 말한다**."""
    real = _yolo_split(tmp_path / "real", ["r0"], labels=False)
    gold = _yolo_split(tmp_path / "gold", ["g0"])
    out, warns = assemble_dataset(
        "yolo", tmp_path / "ds", synth=None, train_base=real, eval_split=gold, names=["scratch"]
    )
    assert (out / "labels" / "train" / "r0.txt").read_text(encoding="utf-8") == ""
    assert any("빈 라벨" in w for w in warns)


def test_assemble_dataset_requires_an_eval_set(tmp_path: Path) -> None:
    real = _yolo_split(tmp_path / "real", ["r0"])
    empty = {"images": tmp_path / "none", "labels": None, "masks": None}
    (tmp_path / "none").mkdir()
    with pytest.raises(LoopError, match="평가셋이 0장"):
        assemble_dataset(
            "yolo", tmp_path / "ds", synth=None, train_base=real, eval_split=empty, names=["a"]
        )


def test_assemble_dataset_pairs_layout(tmp_path: Path) -> None:
    def pairs(root: Path, stems: list[str]) -> dict[str, Path]:
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "masks").mkdir(parents=True, exist_ok=True)
        for s in stems:
            imgio.write_image(root / "images" / f"{s}.png", blob_image(48, [(24, 24, 6)]))
            imgio.write_image(root / "masks" / f"{s}.png", blob_mask(48, [(24, 24, 6)]))
        return {"images": root / "images", "masks": root / "masks", "labels": None}

    real = pairs(tmp_path / "real", ["r0"])
    gold = pairs(tmp_path / "gold", ["g0"])
    synth = tmp_path / "synth"
    pairs(synth, ["000000"])
    out, _ = assemble_dataset(
        "pairs", tmp_path / "ds", synth=synth, train_base=real, eval_split=gold, names=[]
    )
    assert sorted(p.name for p in (out / "train" / "images").iterdir()) == [
        "r0.png",
        "syn_000000.png",
    ]
    assert (out / "train" / "masks" / "syn_000000.png").is_file()
    assert [p.name for p in (out / "val" / "images").iterdir()] == ["g0.png"]


# --------------------------------------------------------------------- 한 바퀴 (noop)


@pytest.fixture
def loop_ws(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Path]:
    """은행 + 레시피 + 동결 평가셋 + 현장 이미지 + trainers.yaml + loop.yaml."""
    d = fake_yolo_dataset(tmp_path / "ds")
    bank = tmp_path / "bank"
    normals = tmp_path / "normals.txt"
    assert (
        main(
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
                str(bank),
                "--mask-from",
                "otsu",
                "--list-normals",
                str(normals),
            ]
        )
        == EXIT_OK
    )
    recipe = tmp_path / "r.yaml"
    assert (
        main(
            [
                "recipe",
                "init",
                "--preset",
                "hard-paste",
                "--bank",
                str(bank),
                "--targets",
                str(normals),
                "--out",
                str(tmp_path / "unused"),
                "--count",
                "2",
                "--seed",
                "11",
                "--write",
                str(recipe),
            ]
        )
        == EXIT_OK
    )
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    data["pipeline"]["placement"]["roi"]["erode_px"] = 2
    data["pipeline"]["placement"]["margin_px"] = 4
    data["pipeline"]["source"]["min_sources_warn"] = 1
    recipe.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # 동결 평가셋(pairs — noop 은 pairs 를 선언한다)
    gold = tmp_path / "golden"
    (gold / "images").mkdir(parents=True)
    (gold / "masks").mkdir(parents=True)
    for i in range(2):
        imgio.write_image(gold / "images" / f"g{i}.png", blob_image(64, [(32, 32, 7)]))
        imgio.write_image(gold / "masks" / f"g{i}.png", blob_mask(64, [(32, 32, 7)]))

    # 현장 이미지 — 정상 목록을 그대로 쓴다(배포 모델이 스코어링할 대상)
    field = tmp_path / "field"
    field.mkdir()
    for i, src in enumerate(imgio.read_path_list(normals)):  # 목록에는 주석 줄이 있다
        imgio.write_image(field / f"f{i}.png", imgio.read_image(src)[0])

    trainers = tmp_path / "trainers.yaml"
    trainers.write_text(
        yaml.safe_dump(
            {"trainers": {"noop": {"command": [sys.executable, str(NOOP)]}}},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    classes = Bank.load(bank).classes
    loop_yaml = tmp_path / "loop.yaml"
    loop_yaml.write_text(
        yaml.safe_dump(
            {
                "trainer": "noop",
                "bank": str(bank),
                "recipe": str(recipe),
                "out": str(tmp_path / "loop"),
                "field": str(field),
                "eval": {"images": str(gold / "images"), "masks": str(gold / "masks")},
                "review": {"n": 3, "threshold": 0.5, "accept_class": classes[0]},
                "promote": {"metric": "mAP50", "noise": 0.001},
                "trainers_file": str(trainers),
                "seed": 5,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    capsys.readouterr()
    return {"root": tmp_path, "bank": bank, "loop": loop_yaml, "out": tmp_path / "loop"}


def _judge_all(queue_dir: Path, verdict: str = "accept") -> int:
    rows = [r for r in read_manifest(queue_dir / MANIFEST_FILE) if r.get("status") == "ok"]
    write_review(queue_dir / REVIEW_FILE, {r["index"]: (verdict, "") for r in rows})
    return len(rows)


def test_full_round_trip_with_the_noop_adapter(loop_ws: dict[str, Path]) -> None:
    """더미로 루프가 안 돌면 설계가 틀린 것이다(§1.6) — 여기서 한 바퀴를 실제로 돈다."""
    loop = load_loop_config(loop_ws["loop"])
    logs: list[str] = []

    # 1라운드 — 모델이 없으니 부트스트랩(합성 → 학습 → 승급)
    first = run_round(loop, on_log=logs.append)
    assert first.record.bootstrap and first.record.done == ["synth", "train", "judge"]
    assert not first.waiting_for_human
    assert first.state.champion is not None and first.state.champion.round == 1
    assert (loop_ws["out"] / round_name(1) / "dataset" / "val" / "images").is_dir()
    assert Path(first.state.champion.model).exists()
    assert "기준선 없음" in first.record.data["judge"]["reason"]

    # 2라운드 — 이제 champion 이 있으니 현장 이미지를 스코어링하고 사람 앞에서 **멈춘다**
    second = run_round(loop, on_log=logs.append)
    assert second.record.number == 2 and not second.record.bootstrap
    assert second.waiting_for_human and second.record.done == ["predict", "queue"]
    assert "판정" in second.message
    queue_dir = second.round_dir / "queue"
    assert (queue_dir / MANIFEST_FILE).is_file()

    # 멈춘 자리에서 다시 불러도 같은 자리 — 상태가 앞으로 가지 않는다
    again = run_round(loop, on_log=logs.append)
    assert again.record.number == 2 and again.waiting_for_human
    assert again.record.done == ["predict", "queue"]

    # 사람이 판정하면 그 다음 단계부터 이어 간다
    n = _judge_all(queue_dir)
    assert n > 0
    third = run_round(loop, on_log=logs.append)
    assert third.record.number == 2 and not third.waiting_for_human
    assert third.record.done == list(PHASES)
    assert third.record.data["accept"]["accepted"] == n
    assert third.record.data["train"]["metrics"]["mAP50"] > 0
    assert len(third.state.history) == 2

    # 채택분이 보관함에 들어갔다(모델 초안이므로 추정으로 센다)
    sources = Bank.load(loop_ws["bank"]).sources()
    assert any(s.mask_origin.startswith("pred:") for s in sources)
    assert any("round-2" in s.tags for s in sources)

    # 조립된 학습셋: train 에 합성이 섞이고 val 은 평가셋 그대로
    ds = Path(third.record.data["train"]["dataset"])
    assert any(p.name.startswith("syn_") for p in (ds / "train" / "images").iterdir())
    assert len(list((ds / "val" / "images").iterdir())) == 2


def test_status_reports_where_we_are(loop_ws: dict[str, Path]) -> None:
    loop = load_loop_config(loop_ws["loop"])
    before = status(loop)
    assert before.record is None and before.state.champion is None
    assert "champion: 없음" in before.lines()[0]

    run_round(loop)
    run_round(loop)  # 2라운드 — 사람 대기에서 선다
    st = status(loop)
    assert st.next == "review" and st.total > 0
    text = " ".join(st.lines())
    assert "round-002" in text and "판정됨" in text and "champion" in text


def test_accept_partial_lets_a_half_judged_queue_through(loop_ws: dict[str, Path]) -> None:
    loop = load_loop_config(loop_ws["loop"])
    run_round(loop)
    second = run_round(loop)
    assert second.waiting_for_human
    rows = [
        r
        for r in read_manifest(second.round_dir / "queue" / MANIFEST_FILE)
        if r.get("status") == "ok"
    ]
    write_review(second.round_dir / "queue" / REVIEW_FILE, {rows[0]["index"]: ("accept", "")})

    assert run_round(loop).waiting_for_human  # 기본은 전부 판정될 때까지 기다린다
    done = run_round(loop, accept_partial=True)
    assert not done.waiting_for_human and done.record.data["accept"]["accepted"] == 1


def test_normal_only_trainer_is_refused(loop_ws: dict[str, Path], tmp_path: Path) -> None:
    """비지도 어댑터에 합성 결함을 먹이는 건 오염이다(설계 §5·§6.4) — 데이터를 만들기 전에 막는다."""
    fake = tmp_path / "unsup.py"
    fake.write_text(
        "import json, sys" + chr(10) + "print(json.dumps({'name': 'unsup', 'version': '0',"
        " 'capabilities': ['score'], 'dataset_format': 'mvtec',"
        " 'trains_on': 'normal_only', 'deterministic': True}))" + chr(10),
        encoding="utf-8",
    )
    trainers = tmp_path / "trainers2.yaml"
    trainers.write_text(
        yaml.safe_dump({"trainers": {"unsup": {"command": [sys.executable, str(fake)]}}}),
        encoding="utf-8",
    )
    doc = yaml.safe_load(loop_ws["loop"].read_text(encoding="utf-8"))
    doc["trainer"] = "unsup"
    doc["trainers_file"] = str(trainers)
    loop_yaml = tmp_path / "loop2.yaml"
    loop_yaml.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")

    with pytest.raises(LoopError, match="normal_only"):
        run_round(load_loop_config(loop_yaml))


# --------------------------------------------------------------------- CLI


def test_cli_loop_run_and_status(
    loop_ws: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["loop", "run", "--config", str(loop_ws["loop"]), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["round"] == 1 and payload["waitingForHuman"] is False
    assert payload["champion"]["metric"] > 0

    assert main(["loop", "status", "--config", str(loop_ws["loop"]), "--json"]) == EXIT_OK
    st = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert st["round"] == 1 and st["next"] is None and st["champion"]["round"] == 1


def test_cli_loop_run_without_config_is_fail_soft(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["loop", "run", "--config", "없는파일.yaml"]) == EXIT_RECIPE_ERROR
    assert "loop.example.yaml" in capsys.readouterr().err

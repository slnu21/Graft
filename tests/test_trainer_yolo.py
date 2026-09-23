"""T3 — 계약 헬퍼(`fit`/`predict`/`merge_spec`/스트리밍)와 **YOLO 어댑터의 순수 부분**.

어댑터 본체는 ultralytics 가 있는 별도 venv 에서 돌지만, spec 병합·경로 해석·지표 평탄화는 순수 함수라
코어 venv 에서 그대로 고정된다. 학습이 필요한 왕복은 데브로그의 실측(짧은 학습 한 바퀴)이 맡는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from anograft.loop.contract import (
    TrainerError,
    call,
    fit,
    merge_spec,
    predict,
    spec_file,
)
from tests.fixtures import load_adapter

REPO_ROOT = Path(__file__).resolve().parents[1]
NOOP_CMD = [sys.executable, str(REPO_ROOT / "adapters" / "noop.py")]

ya = load_adapter("yolo")


# --------------------------------------------------------------------- spec (불투명)


def test_merge_spec_call_wins_and_is_shallow() -> None:
    """등록부 spec 위에 호출 spec 을 얹는다. 깊은 병합은 하지 않는다 — spec 은 불투명이다(§1.3)."""
    base = {"model": "yolov8n.pt", "epochs": 40, "nested": {"a": 1}}
    assert merge_spec(base, {"epochs": 3, "nested": {"b": 2}}) == {
        "model": "yolov8n.pt",
        "epochs": 3,
        "nested": {"b": 2},
    }
    assert merge_spec(None, None) == {}
    assert merge_spec(base, None)["epochs"] == 40


def test_spec_file_is_written_and_cleaned_up() -> None:
    with spec_file({"epochs": 3}) as p:
        assert p is not None and json.loads(p.read_text(encoding="utf-8")) == {"epochs": 3}
        kept = p
    assert not kept.exists()  # 임시 폴더째 지운다


def test_spec_file_is_none_when_empty() -> None:
    """빈 spec 이면 ``--spec`` 인자를 아예 붙이지 않는다(어댑터 기본값이 살아야 한다)."""
    with spec_file({}) as p:
        assert p is None


# --------------------------------------------------------------------- fit·predict 헬퍼


def test_fit_helper_delivers_spec_to_adapter(tmp_path: Path) -> None:
    """noop 의 ``bias`` 는 spec 으로만 들어간다 → 지표가 달라지면 spec 이 전달된 것이다."""
    # 데이터셋과 출력은 갈라 둔다 — noop 의 기본 지표가 데이터셋 **파일 수**에서 나오므로,
    # 출력 폴더가 데이터셋 안에 있으면 두 번째 호출의 기준선이 달라진다.
    dataset = tmp_path / "ds"
    dataset.mkdir()
    plain = fit(NOOP_CMD, dataset=dataset, out=tmp_path / "m0", seed=7, timeout=120)
    biased = fit(
        NOOP_CMD, dataset=dataset, out=tmp_path / "m1", seed=7, spec={"bias": 0.2}, timeout=120
    )
    assert biased.metrics["mAP50"] == pytest.approx(plain.metrics["mAP50"] + 0.2, abs=1e-6)
    assert Path(plain.model).is_file()


def test_predict_helper_writes_importable_outputs(tmp_path: Path) -> None:
    """예측 출력은 ``bank import-pairs``/``import-yolo`` 가 그대로 받는 형식이어야 한다(§1.4)."""
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # 헤더만 — noop 은 크기를 못 읽으면 기본값을 쓴다
    listing = tmp_path / "imgs.txt"
    listing.write_text(str(img), encoding="utf-8")

    res = predict(
        NOOP_CMD,
        model="dummy",
        images=listing,
        out=tmp_path / "pred",
        spec={"imgsz": 320},  # 학습 때와 같은 spec 이 예측에도 닿는다(어댑터가 모르는 키는 무시)
        timeout=120,
    )
    assert res.count == 1
    assert (res.predictions / "masks" / "a.png").is_file()
    payload = json.loads((res.predictions / "scores" / "a.json").read_text(encoding="utf-8"))
    assert 0.0 <= payload["image_score"] <= 1.0 and payload["instances"]


def test_on_log_streams_stderr_lines(tmp_path: Path) -> None:
    """학습은 길다 — stderr 는 끝나고 한 번에가 아니라 **줄 단위로** 와야 한다."""
    lines: list[str] = []
    payload, err = call(
        NOOP_CMD,
        "fit",
        ["--dataset", str(tmp_path), "--out", str(tmp_path / "m"), "--seed", "1"],
        timeout=120,
        on_log=lines.append,
    )
    assert payload["model"]  # stdout 은 임시 파일로 받아도 결과 JSON 이 살아 있다
    assert any("noop" in ln for ln in lines)
    assert err.splitlines() == lines  # 모아 준 것과 흘려 준 것이 같다


def test_streaming_timeout_kills_a_silent_adapter() -> None:
    """로그 한 줄 없이 붙잡혀 있는 어댑터도 타임아웃에 걸려야 한다(감시 타이머).

    읽기 루프 안에서 시계를 보는 구현이면 stderr 가 조용할 때 영원히 기다린다 — 학습 어댑터에서
    실제로 날 수 있는 모양이라 테스트로 고정한다.
    """
    hang = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(TrainerError, match="안에 끝나지 않았습니다"):
        call(hang, "info", timeout=1.0, on_log=lambda _: None)


def test_streaming_failure_still_raises_with_hint(tmp_path: Path) -> None:
    with pytest.raises(TrainerError, match="predict"):
        call(
            NOOP_CMD,
            "predict",
            ["--model", "x", "--images", str(tmp_path / "없음.txt"), "--out", str(tmp_path)],
            timeout=60,
            on_log=lambda _: None,
        )


# --------------------------------------------------------------------- YOLO 어댑터 순수 부분


def test_adapter_spec_defaults_and_override(tmp_path: Path) -> None:
    assert ya.load_spec(None)["model"] == "yolov8n.pt"
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"epochs": 3, "device": "cpu"}), encoding="utf-8")
    spec = ya.load_spec(str(p))
    assert spec["epochs"] == 3 and spec["imgsz"] == ya.DEFAULTS["imgsz"]
    # 읽을 수 없는 spec 은 기본값으로 — 어댑터도 fail-soft 다
    assert ya.load_spec(str(tmp_path / "없음.json")) == ya.DEFAULTS


def test_adapter_data_yaml_and_seg_detection(tmp_path: Path) -> None:
    (tmp_path / "data.yaml").write_text("names: []\n", encoding="utf-8")
    assert ya.data_yaml_of(tmp_path) == tmp_path / "data.yaml"
    assert ya.data_yaml_of(tmp_path / "data.yaml") == tmp_path / "data.yaml"
    assert ya.is_seg("yolov8n-seg.pt") and not ya.is_seg("yolov8n.pt")


def test_adapter_flattens_metrics_with_class_keys() -> None:
    """계약의 지표는 평평한 float 맵 — 클래스별은 ``mAP50/<class>``(중첩은 parse_fit 이 버린다)."""
    metrics = SimpleNamespace(
        box=SimpleNamespace(map50=0.4123, map=0.2011, ap_class_index=[0, 2], ap50=[0.61, 0.22]),
        names={0: "bent", 1: "color", 2: "scratch"},
    )
    flat = ya.flat_metrics(metrics)
    assert flat == {"mAP50": 0.4123, "mAP50-95": 0.2011, "mAP50/bent": 0.61, "mAP50/scratch": 0.22}
    assert all(isinstance(v, float) for v in flat.values())


def test_cli_fit_then_predict_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``anograft trainer fit|predict`` — 벤치·루프가 공용으로 쓰는 진입점(도구는 ``--json`` 을 읽는다)."""
    from anograft.cli import EXIT_OK, main

    cfg = _trainers_yaml(tmp_path)
    assert (
        main(
            [
                "trainer",
                "fit",
                "noop",
                "--dataset",
                str(tmp_path),
                "--out",
                str(tmp_path / "model"),
                "--seed",
                "7",
                "--file",
                str(cfg),
                "--json",
            ]
        )
        == EXIT_OK
    )
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert Path(payload["model"]).is_file() and payload["metrics"]["mAP50"] > 0

    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    listing = tmp_path / "imgs.txt"
    listing.write_text(str(img), encoding="utf-8")
    assert (
        main(
            [
                "trainer",
                "predict",
                "noop",
                "--model",
                payload["model"],
                "--images",
                str(listing),
                "--out",
                str(tmp_path / "pred"),
                "--file",
                str(cfg),
                "--json",
            ]
        )
        == EXIT_OK
    )
    pred = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert pred["count"] == 1 and (Path(pred["predictions"]) / "scores" / "a.json").is_file()


def test_cli_fit_spec_file_overrides_registry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--spec`` 은 등록부 spec 위에 얹힌다 — noop 의 bias 로 확인."""
    from anograft.cli import EXIT_OK, main

    cfg = _trainers_yaml(tmp_path, spec="{bias: 0.1}")
    override = tmp_path / "over.json"
    override.write_text(json.dumps({"bias": 0.3}), encoding="utf-8")
    dataset = tmp_path / "ds"
    dataset.mkdir()
    args = ["trainer", "fit", "noop", "--dataset", str(dataset), "--file", str(cfg), "--json"]
    assert main([*args, "--out", str(tmp_path / "m0")]) == EXIT_OK
    base = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["metrics"]["mAP50"]
    assert main([*args, "--out", str(tmp_path / "m1"), "--spec", str(override)]) == EXIT_OK
    with_over = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["metrics"]["mAP50"]
    assert with_over == pytest.approx(base + 0.2, abs=1e-6)


def test_cli_unknown_trainer_lists_registered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from anograft.cli import EXIT_RECIPE_ERROR, main

    cfg = _trainers_yaml(tmp_path)
    code = main(
        [
            "trainer",
            "fit",
            "ghost",
            "--dataset",
            str(tmp_path),
            "--out",
            str(tmp_path / "m"),
            "--file",
            str(cfg),
        ]
    )
    assert code == EXIT_RECIPE_ERROR
    assert "noop" in capsys.readouterr().err


def _trainers_yaml(tmp_path: Path, *, spec: str | None = None) -> Path:
    cfg = tmp_path / "trainers.yaml"
    body = (
        "trainers:\n  noop:\n"
        f"    command: [{json.dumps(sys.executable)}, {json.dumps(str(NOOP_CMD[1]))}]\n"
    )
    if spec:
        body += f"    spec: {spec}\n"
    cfg.write_text(body, encoding="utf-8")
    return cfg


def test_adapter_declares_contract_vocabulary() -> None:
    """``info`` 선언은 계약 어휘 안이어야 한다(ultralytics 없이 값만 본다)."""
    from anograft.loop.contract import CAPABILITIES, DATASET_FORMATS, TRAINS_ON

    assert ya.NAME == "yolo"
    assert set(["score", "box"]) <= set(CAPABILITIES)
    assert "yolo" in DATASET_FORMATS and "labeled" in TRAINS_ON

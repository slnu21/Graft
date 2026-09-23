"""학습기 계약 — 파싱은 순수 함수로, 그리고 **`adapters/noop` 을 실제로 호출해** 계약을 실증한다.

더미로 3 verb 가 돌지 않으면 계약 설계가 틀린 것이다(설계 §1.6).
"""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import pytest

from anograft.loop.contract import (
    TrainerError,
    call,
    last_json_line,
    parse_fit,
    parse_info,
    parse_predict,
)
from anograft.loop.registry import (
    TrainerSpec,
    find_trainers_file,
    load_trainers,
    probe,
    resolve_cwd,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOOP = REPO_ROOT / "adapters" / "noop.py"
NOOP_CMD = [sys.executable, str(NOOP)]


# --------------------------------------------------------------------- 순수 파싱


def test_last_json_line_ignores_progress_output() -> None:
    """어댑터가 앞줄에 진행 상황을 찍어도 된다 — 마지막 JSON 오브젝트가 결과다."""
    out = 'epoch 1/40\n[1, 2, 3]\n{"model": "a.pt"}\n'
    assert last_json_line(out) == {"model": "a.pt"}


def test_last_json_line_without_json_raises() -> None:
    with pytest.raises(TrainerError, match="JSON"):
        last_json_line("학습 중...\n끝\n")


def test_parse_info_requires_contract_vocabulary() -> None:
    ok = parse_info(
        {
            "name": "x",
            "capabilities": ["mask"],
            "dataset_format": "mvtec",
            "trains_on": "normal_only",
            "deterministic": False,
        }
    )
    assert ok.normal_only is True
    assert ok.can("mask") and not ok.can("box")
    assert ok.deterministic is False

    with pytest.raises(TrainerError, match="dataset_format"):
        parse_info(
            {
                "name": "x",
                "capabilities": ["mask"],
                "dataset_format": "parquet",
                "trains_on": "labeled",
            }
        )
    with pytest.raises(TrainerError, match="trains_on"):
        parse_info(
            {"name": "x", "capabilities": ["mask"], "dataset_format": "yolo", "trains_on": "maybe"}
        )
    with pytest.raises(TrainerError, match="capabilities"):
        parse_info(
            {
                "name": "x",
                "capabilities": ["heatmap"],
                "dataset_format": "yolo",
                "trains_on": "labeled",
            }
        )
    with pytest.raises(TrainerError, match="빠진"):
        parse_info({"capabilities": ["mask"], "dataset_format": "yolo", "trains_on": "labeled"})


def test_normal_only_is_the_guard_against_contamination() -> None:
    """이 선언이 없으면 비지도 어댑터에 합성 결함을 먹이는 오염이 조용히 일어난다(§6.4)."""
    unsup = parse_info(
        {
            "name": "patchcore",
            "capabilities": ["score", "mask"],
            "dataset_format": "mvtec",
            "trains_on": "normal_only",
        }
    )
    sup = parse_info(
        {"name": "yolo", "capabilities": ["box"], "dataset_format": "yolo", "trains_on": "labeled"}
    )
    assert unsup.normal_only and not sup.normal_only


def test_parse_fit_drops_non_numeric_metrics() -> None:
    r = parse_fit({"model": "m.pt", "metrics": {"mAP50": 0.41, "note": "좋음"}})
    assert r.model == "m.pt"
    assert r.metrics == {"mAP50": 0.41}
    with pytest.raises(TrainerError, match="model"):
        parse_fit({"metrics": {}})


def test_parse_predict() -> None:
    r = parse_predict({"predictions": "out/p", "count": "12"})
    assert r.predictions == Path("out/p") and r.count == 12
    with pytest.raises(TrainerError, match="predictions"):
        parse_predict({"count": 1})


# --------------------------------------------------------------------- 실제 호출 (noop)


def test_noop_adapter_exists() -> None:
    assert NOOP.exists(), "adapters/noop.py 가 있어야 계약이 실증된다"


def test_call_info_roundtrip() -> None:
    payload, _ = call(NOOP_CMD, "info", timeout=60)
    info = parse_info(payload)
    assert info.name == "noop"
    assert info.deterministic is True
    assert info.dataset_format == "pairs"


def test_call_missing_command_is_fail_soft() -> None:
    with pytest.raises(TrainerError, match="실행할 수 없습니다"):
        call(["__no_such_binary__"], "info", timeout=10)


def test_call_nonzero_exit_raises_with_hint(tmp_path: Path) -> None:
    with pytest.raises(TrainerError, match="실패"):
        call(
            NOOP_CMD,
            "predict",
            ["--model", "m", "--images", "nope", "--out", str(tmp_path)],
            timeout=60,
        )


def _png(path: Path, w: int, h: int) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(bytearray(w)) for _ in range(h))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def test_fit_then_predict_produces_importable_outputs(tmp_path: Path) -> None:
    """predict 출력이 **Graft 가 이미 임포트하는 형식**이어야 루프가 닫힌다(§1.4)."""
    ds = tmp_path / "ds"
    (ds / "images").mkdir(parents=True)
    for i in range(3):
        _png(ds / "images" / f"img_{i}.png", 64, 48)

    fit_payload, _ = call(
        NOOP_CMD,
        "fit",
        ["--dataset", str(ds), "--out", str(tmp_path / "model"), "--seed", "7"],
        timeout=120,
    )
    fit = parse_fit(fit_payload)
    assert Path(fit.model).exists()
    assert "mAP50" in fit.metrics

    listing = tmp_path / "images.txt"
    listing.write_text(
        "\n".join(str(p) for p in sorted((ds / "images").glob("*.png"))), encoding="utf-8"
    )
    pred_payload, _ = call(
        NOOP_CMD,
        "predict",
        ["--model", fit.model, "--images", str(listing), "--out", str(tmp_path / "pred")],
        timeout=120,
    )
    pred = parse_predict(pred_payload)
    assert pred.count == 3

    masks = sorted((pred.predictions / "masks").glob("*.png"))
    scores = sorted((pred.predictions / "scores").glob("*.json"))
    assert len(masks) == 3 and len(scores) == 3
    # 마스크는 bank import-pairs 가 그대로 받는 PNG 여야 한다
    assert masks[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    one = json.loads(scores[0].read_text(encoding="utf-8"))
    assert 0.0 <= one["image_score"] <= 1.0
    assert len(one["instances"][0]["bbox"]) == 4


def test_noop_is_deterministic(tmp_path: Path) -> None:
    """같은 입력이면 같은 결과 — 루프의 재현성이 어댑터에서부터 깨지지 않게."""
    ds = tmp_path / "ds"
    ds.mkdir()
    _png(ds / "a.png", 32, 32)
    listing = tmp_path / "l.txt"
    listing.write_text(str(ds / "a.png"), encoding="utf-8")

    digests = []
    for k in (1, 2):
        out = tmp_path / f"p{k}"
        call(
            NOOP_CMD,
            "predict",
            ["--model", "m", "--images", str(listing), "--out", str(out)],
            timeout=60,
        )
        digests.append((out / "masks" / "a.png").read_bytes())
    assert digests[0] == digests[1]


def test_spec_bias_changes_metrics(tmp_path: Path) -> None:
    """--spec 은 불투명 dict — 어댑터만 해석한다. 루프 테스트에서 challenger 를 만드는 손잡이."""
    ds = tmp_path / "ds"
    ds.mkdir()
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"bias": 0.1}), encoding="utf-8")

    base, _ = call(
        NOOP_CMD,
        "fit",
        ["--dataset", str(ds), "--out", str(tmp_path / "m1"), "--seed", "0"],
        timeout=60,
    )
    biased, _ = call(
        NOOP_CMD,
        "fit",
        ["--dataset", str(ds), "--out", str(tmp_path / "m2"), "--seed", "0", "--spec", str(spec)],
        timeout=60,
    )
    assert parse_fit(biased).metrics["mAP50"] > parse_fit(base).metrics["mAP50"]


# --------------------------------------------------------------------- 등록부


def test_load_trainers_and_probe(tmp_path: Path) -> None:
    cfg = tmp_path / "trainers.yaml"
    cfg.write_text(
        "trainers:\n"
        "  noop:\n"
        f"    command: [{json.dumps(sys.executable)}, {json.dumps(str(NOOP))}]\n"
        "    spec: {bias: 0.0}\n",
        encoding="utf-8",
    )
    trainers = load_trainers(cfg)
    assert set(trainers) == {"noop"}

    status = probe("noop", trainers["noop"], tmp_path, timeout=60)
    assert status.available is True
    assert status.info is not None and status.info.name == "noop"
    assert "pairs" in status.summary


def test_probe_is_fail_soft_for_broken_entry(tmp_path: Path) -> None:
    """어댑터가 없거나 깨져도 앱은 죽지 않고 사유만 남긴다."""
    spec = TrainerSpec(command=["__missing__"])
    status = probe("ghost", spec, tmp_path, timeout=10)
    assert status.available is False
    assert status.info is None
    assert "쓸 수 없음" in status.summary


def test_trainers_file_rejects_unknown_keys(tmp_path: Path) -> None:
    cfg = tmp_path / "trainers.yaml"
    cfg.write_text("trainers:\n  x:\n    command: [echo]\n    epochs: 40\n", encoding="utf-8")
    with pytest.raises(TrainerError, match="형식 오류"):
        load_trainers(cfg)


def test_find_trainers_file_prefers_cwd(tmp_path: Path) -> None:
    assert find_trainers_file(cwd=tmp_path) is None
    here = tmp_path / "trainers.yaml"
    here.write_text("trainers: {}\n", encoding="utf-8")
    assert find_trainers_file(cwd=tmp_path) == here
    assert find_trainers_file(tmp_path / "nope.yaml", cwd=tmp_path) is None


def test_resolve_cwd_is_relative_to_config(tmp_path: Path) -> None:
    spec = TrainerSpec(command=["x"], cwd="sub")
    assert resolve_cwd(spec, tmp_path) == tmp_path / "sub"
    assert resolve_cwd(TrainerSpec(command=["x"]), tmp_path) == tmp_path

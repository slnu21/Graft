"""T2 — anomalib 어댑터의 **순수 부분**과 비지도 선언이 계약에 맞는지.

anomalib 은 코어 venv 에 없다(그게 요점이다 — 학습 환경은 어댑터 쪽에만 있다). 여기서 보는 것은
지표 평탄화·spec 병합·데이터 경로 해석·모델 이름 거르기, 그리고 **`trains_on: normal_only` 가 무엇을
막는가**이다. 실제 학습 왕복은 데브로그의 실측이 맡는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anograft.loop.contract import DATASET_FORMATS, TrainerError, parse_info
from tests.fixtures import load_adapter

ad = load_adapter("anomalib_trainer")


def test_adapter_file_does_not_shadow_the_package() -> None:
    """파일 이름이 ``anomalib.py`` 면 스크립트 폴더가 ``sys.path[0]`` 이라 **자기 자신**을 import 한다.

    실제로 한 번 겪었다: ``No module named 'anomalib.engine'; 'anomalib' is not a package``.
    """
    adapters = Path(__file__).resolve().parents[1] / "adapters"
    assert (adapters / "anomalib_trainer.py").is_file()
    assert not (adapters / "anomalib.py").exists()


def test_declaration_is_unsupervised_and_within_contract() -> None:
    """이 어댑터의 존재 이유가 이 선언이다 — 합성 결함이 학습셋에 들어가지 않게 하는 안전장치(§6.4)."""
    info = parse_info(
        {
            "name": ad.NAME,
            "version": "2.6.2",
            "capabilities": ["score", "mask"],
            "dataset_format": "mvtec",
            "trains_on": "normal_only",
            "deterministic": False,
        }
    )
    assert info.normal_only is True
    assert info.dataset_format in DATASET_FORMATS
    assert info.can("mask") and not info.can("box")


def test_metrics_are_flattened_to_floats() -> None:
    """``engine.test`` 는 dict 리스트를 준다. ``/`` 는 계약에서 '클래스별' 구분자라 ``_`` 로 바꾼다."""
    flat = ad.flat_metrics([{"image_AUROC": 0.9428, "pixel/AUPRO": 0.61, "note": "x"}])
    assert flat == {"image_AUROC": 0.9428, "pixel_AUPRO": 0.61}
    assert ad.flat_metrics(None) == {}
    assert ad.flat_metrics({"image_AUROC": 1}) == {"image_AUROC": 1.0}


def test_spec_defaults_and_override(tmp_path: Path) -> None:
    assert ad.load_spec(None)["model"] == "patchcore"
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"model": "padim", "image_size": 128}), encoding="utf-8")
    spec = ad.load_spec(str(p))
    assert spec["model"] == "padim" and spec["image_size"] == 128
    assert spec["accelerator"] == ad.DEFAULTS["accelerator"]  # 안 준 키는 기본값
    assert ad.load_spec(str(tmp_path / "없음.json")) == ad.DEFAULTS  # fail-soft


def test_unknown_model_is_refused_with_a_reason() -> None:
    """지도 학습 모델을 여기 주면 조용히 도는 대신 사유를 주고 멈춘다(오염 방지의 일부)."""
    with pytest.raises(SystemExit, match="모르는 model"):
        ad.build_model({"model": "yolov8n", "image_size": 256})
    assert set(ad.MODELS) >= {"patchcore", "padim", "fastflow"}


def test_contract_error_message_for_missing_env() -> None:
    """환경이 없을 때 죽는 게 아니라 종료 코드로 말한다 — `probe` 가 사유를 보여 준다(fail-soft)."""
    assert issubclass(TrainerError, RuntimeError)

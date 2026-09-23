"""학습기 플러그인 계약 (설계 `v1.x-training-loop.md` §1) — **프로세스 경계 + JSON in/out**.

Python ABC 가 아니다. ABC 면 학습 코드가 ``anograft`` 를 import 해야 하고 → torch·ultralytics·anomalib 이
코어와 한 venv → "의존성은 전부 순수 wheel" 원칙이 깨진다. 프로세스 경계여야 **어떤 언어·어떤 환경·원격
GPU 박스**든 붙는다. 여기서 쓰는 것은 ``subprocess``·``json`` 뿐 — 새 의존성 0.

계약은 verb 세 개::

    <command> info
      → {"name", "version", "capabilities":[score|mask|box], "dataset_format":mvtec|yolo|coco|pairs,
         "trains_on":normal_only|labeled, "deterministic":bool}

    <command> fit --dataset <p> --out <model_dir> --seed <N> --spec <json-path>
      → {"model":"<opaque>", "metrics":{…}}

    <command> predict --model <opaque> --images <list.txt> --out <pred_dir>
      → {"predictions":"<pred_dir>", "count":N}

약속:

- 종료 코드 0 = 성공, **stdout 의 마지막 JSON 줄**이 결과다(그 앞 줄은 진행 로그로 흘려도 된다).
- stderr 는 로그. 호출자가 그대로 배치 로그에 흘린다.
- 실패는 **fail-soft** — `TrainerError` 로 올리고 루프가 "라운드 실패"로 기록한다. 앱은 죽지 않는다.
- ``model`` 과 ``--spec`` 은 **불투명**하다. Graft 는 해석하지 않고 그대로 들고 다닌다(설계 §1.3).

파싱은 전부 순수 함수라 어댑터 없이 테스트된다(`tests/test_trainer_contract.py`).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 계약이 허용하는 어휘. 어댑터가 다른 값을 주면 계약 위반으로 거른다(오타를 조용히 넘기지 않는다).
CAPABILITIES: tuple[str, ...] = ("score", "mask", "box")
DATASET_FORMATS: tuple[str, ...] = ("mvtec", "yolo", "coco", "pairs")
TRAINS_ON: tuple[str, ...] = ("normal_only", "labeled")

DEFAULT_TIMEOUT_S = 60.0 * 60 * 12  # 학습은 길다. info/predict 는 호출부가 따로 줄인다.


class TrainerError(RuntimeError):
    """어댑터 호출 실패 — 종료 코드 ≠ 0, JSON 없음, 계약 위반, 타임아웃.

    루프는 이것을 잡아 라운드 실패로 기록하고 다음 tick 까지 기다린다(fail-soft).
    """


@dataclass(frozen=True)
class TrainerInfo:
    """``info`` 가 선언하는 것 — 루프가 이걸로 **무엇을 어떻게 줄지** 정한다(설계 §1.2)."""

    name: str
    version: str
    capabilities: tuple[str, ...]
    dataset_format: str
    trains_on: str
    deterministic: bool

    @property
    def normal_only(self) -> bool:
        """정상만으로 학습하는가.

        ``True`` 면 **합성 결함을 학습셋에 넣지 않는다.** PatchCore·PaDiM·FastFlow 계열이 여기 해당하고,
        이 선언이 없으면 비지도 어댑터에 합성을 먹이는 오염이 조용히 일어난다(설계 §6.4).
        """
        return self.trains_on == "normal_only"

    def can(self, capability: str) -> bool:
        return capability in self.capabilities


@dataclass(frozen=True)
class FitResult:
    model: str  # 불투명 — .pt / .ckpt / memory bank 무엇이든 Graft 는 해석하지 않는다
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictResult:
    predictions: Path
    count: int


def last_json_line(stdout: str) -> dict[str, Any]:
    """stdout 의 **마지막 JSON 오브젝트 줄**을 돌려준다.

    어댑터가 진행 상황을 앞줄에 찍어도 되게 하려는 것. 마지막 줄부터 거슬러 올라가며 처음 파싱되는
    오브젝트를 취한다(배열·숫자는 결과가 아니므로 건너뛴다).
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise TrainerError(
        "어댑터가 JSON 결과를 내지 않았습니다 (stdout 마지막 줄이 JSON 오브젝트여야 합니다)"
    )


def parse_info(payload: Mapping[str, Any]) -> TrainerInfo:
    """``info`` 응답을 검증해 `TrainerInfo` 로. 계약 위반은 사유를 붙여 `TrainerError`."""
    missing = [k for k in ("name", "dataset_format", "trains_on") if not payload.get(k)]
    if missing:
        raise TrainerError(f"info 에 빠진 항목: {', '.join(missing)}")

    fmt = str(payload["dataset_format"])
    if fmt not in DATASET_FORMATS:
        raise TrainerError(
            f"dataset_format 이 계약 밖입니다: {fmt!r} (가능: {', '.join(DATASET_FORMATS)})"
        )

    trains_on = str(payload["trains_on"])
    if trains_on not in TRAINS_ON:
        raise TrainerError(
            f"trains_on 이 계약 밖입니다: {trains_on!r} (가능: {', '.join(TRAINS_ON)})"
        )

    raw_caps = payload.get("capabilities") or ()
    if isinstance(raw_caps, str):
        raw_caps = (raw_caps,)
    caps = tuple(str(c) for c in raw_caps)
    bad = [c for c in caps if c not in CAPABILITIES]
    if bad:
        raise TrainerError(
            f"capabilities 가 계약 밖입니다: {', '.join(bad)} (가능: {', '.join(CAPABILITIES)})"
        )
    if not caps:
        raise TrainerError("capabilities 가 비어 있습니다 — 최소 하나는 선언해야 합니다")

    return TrainerInfo(
        name=str(payload["name"]),
        version=str(payload.get("version", "")),
        capabilities=caps,
        dataset_format=fmt,
        trains_on=trains_on,
        deterministic=bool(payload.get("deterministic", False)),
    )


def parse_fit(payload: Mapping[str, Any]) -> FitResult:
    model = payload.get("model")
    if not model:
        raise TrainerError("fit 결과에 model 이 없습니다")
    raw = payload.get("metrics") or {}
    if not isinstance(raw, Mapping):
        raise TrainerError("fit 결과의 metrics 는 오브젝트여야 합니다")
    metrics: dict[str, float] = {}
    for k, v in raw.items():
        try:
            metrics[str(k)] = float(v)
        except (TypeError, ValueError):
            continue  # 숫자가 아닌 지표는 조용히 버린다(어댑터가 문자열을 섞어도 죽지 않게)
    return FitResult(model=str(model), metrics=metrics)


def parse_predict(payload: Mapping[str, Any]) -> PredictResult:
    pred = payload.get("predictions")
    if not pred:
        raise TrainerError("predict 결과에 predictions 경로가 없습니다")
    try:
        count = int(payload.get("count", 0))
    except (TypeError, ValueError):
        count = 0
    return PredictResult(predictions=Path(str(pred)), count=count)


def call(
    command: Sequence[str],
    verb: str,
    args: Sequence[str] = (),
    *,
    cwd: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    env: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], str]:
    """어댑터를 한 번 부르고 ``(결과 JSON, stderr)`` 를 돌려준다.

    호출자는 stderr 를 로그로 흘린다. 실패는 전부 `TrainerError` — 호출부가 라운드 실패로 기록한다.
    """
    argv = [*command, verb, *args]
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise TrainerError(f"어댑터를 실행할 수 없습니다: {argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TrainerError(f"어댑터가 {timeout:.0f}s 안에 끝나지 않았습니다 ({verb})") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        hint = tail[-1] if tail else "(출력 없음)"
        raise TrainerError(f"어댑터 {verb} 실패 (exit {proc.returncode}): {hint}")

    return last_json_line(proc.stdout or ""), proc.stderr or ""

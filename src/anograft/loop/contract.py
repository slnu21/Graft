"""학습기 플러그인 계약 (설계 `v1.x-training-loop.md` §1) — **프로세스 경계 + JSON in/out**.

Python ABC 가 아니다. ABC 면 학습 코드가 ``anograft`` 를 import 해야 하고 → torch·ultralytics·anomalib 이
코어와 한 venv → "의존성은 전부 순수 wheel" 원칙이 깨진다. 프로세스 경계여야 **어떤 언어·어떤 환경·원격
GPU 박스**든 붙는다. 여기서 쓰는 것은 ``subprocess``·``json`` 뿐 — 새 의존성 0.

계약은 verb 셋 + **선택 하나**(`eval`)::

    <command> info
      → {"name", "version", "capabilities":[score|mask|box|eval], "dataset_format":mvtec|yolo|coco|pairs,
         "trains_on":normal_only|labeled, "deterministic":bool}

    <command> fit --dataset <p> --out <model_dir> --seed <N> --spec <json-path>
      → {"model":"<opaque>", "metrics":{…}}

    <command> predict --model <opaque> --images <list.txt> --out <pred_dir> [--spec <json-path>]
      → {"predictions":"<pred_dir>", "count":N}

    <command> eval --model <opaque> --dataset <p> [--seed <N>] [--spec <json-path>]   # 선택
      → {"metrics":{…}}

약속:

- 종료 코드 0 = 성공, **stdout 의 마지막 JSON 줄**이 결과다(그 앞 줄은 진행 로그로 흘려도 된다).
- stderr 는 로그. 호출자가 그대로 배치 로그에 흘린다(`call(..., on_log=…)` 은 **줄 단위 실시간**).
  학습 프레임워크는 진행을 stdout 에 찍는 것이 많다 — 어댑터가 그것을 stderr 로 돌리고 stdout 은
  결과 JSON 만 남기는 게 규약이다(`adapters/yolo.py` 의 ``redirect_stdout``).
- ``fit`` 의 ``metrics`` 는 **평평한 float 맵**이다. 클래스별 값은 ``"<지표>/<클래스>"`` 키로 편다
  (예 ``{"mAP50": 0.41, "mAP50/bent": 0.62}``) — 중첩 오브젝트는 `parse_fit` 이 버린다.
- ``eval`` 은 **선택**이다(`capabilities` 에 ``eval`` 을 선언한 어댑터만 부른다). 학습하지 않고 **이미
  학습된 모델**을 다른 평가셋에서 재는 것이라 지표 모양은 `fit` 과 같다 — 이중 평가셋(T16)의 "최근
  평가셋" 점수가 여기서 나온다. 선언이 없으면 루프는 고정 평가셋만으로 판정하고 사유를 남긴다(fail-soft).
- 실패는 **fail-soft** — `TrainerError` 로 올리고 루프가 "라운드 실패"로 기록한다. 앱은 죽지 않는다.
- ``model`` 과 ``--spec`` 은 **불투명**하다. Graft 는 해석하지 않고 그대로 들고 다닌다(설계 §1.3).
  ``--spec`` 은 **fit 과 predict 에 같은 값**이 간다(추론 해상도가 학습과 달라지면 조용히 나빠진다) —
  어댑터는 쓰지 않더라도 ``predict --spec`` 을 **거부하지 않아야** 한다.

파싱은 전부 순수 함수라 어댑터 없이 테스트된다(`tests/test_trainer_contract.py`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LogSink = Callable[[str], None]
"""어댑터 stderr 한 줄을 받는 콜백 — 호출자가 배치 로그·터미널로 흘린다."""

# 계약이 허용하는 어휘. 어댑터가 다른 값을 주면 계약 위반으로 거른다(오타를 조용히 넘기지 않는다).
# ``score``·``mask``·``box`` 는 **예측 축**(검토 큐의 교차 검증에 쓸 수 있는 축, §1.2)이고
# ``eval`` 은 **선택 verb 의 선언**이다 — "학습 없이 모델을 다시 잴 수 있다"(T16 이중 평가셋).
# 한 칸에 둔 이유: 루프가 묻는 것은 언제나 "이 어댑터가 무엇을 해 줄 수 있나" 하나이고, 선언 필드를
# 늘리면 어댑터마다 채워야 할 자리가 늘어난다(없으면 못 하는 것으로 읽는 규칙은 그대로다).
CAPABILITY_EVAL = "eval"
CAPABILITIES: tuple[str, ...] = ("score", "mask", "box", CAPABILITY_EVAL)
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

    @property
    def can_eval(self) -> bool:
        """학습 없이 모델을 다시 잴 수 있는가 — **이중 평가셋의 최근 쪽이 이것에 달려 있다**(T16).

        없으면 루프는 고정 평가셋만으로 판정한다(라운드를 죽이지 않는다 — 선택 verb 다).
        """
        return self.can(CAPABILITY_EVAL)


@dataclass(frozen=True)
class FitResult:
    model: str  # 불투명 — .pt / .ckpt / memory bank 무엇이든 Graft 는 해석하지 않는다
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictResult:
    predictions: Path
    count: int


@dataclass(frozen=True)
class EvalResult:
    """``eval`` 결과 — 모델 참조를 새로 내지 않는다(학습하지 않았으니 모델은 그대로다)."""

    metrics: Mapping[str, float] = field(default_factory=dict)


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


def flat_metrics(payload: Mapping[str, Any], verb: str) -> dict[str, float]:
    """``metrics`` → **평평한 float 맵**. `fit` 과 `eval` 이 같은 규칙을 쓴다(한 지점에서 판정한다)."""
    raw = payload.get("metrics") or {}
    if not isinstance(raw, Mapping):
        raise TrainerError(f"{verb} 결과의 metrics 는 오브젝트여야 합니다")
    metrics: dict[str, float] = {}
    for k, v in raw.items():
        try:
            metrics[str(k)] = float(v)
        except (TypeError, ValueError):
            continue  # 숫자가 아닌 지표는 조용히 버린다(어댑터가 문자열을 섞어도 죽지 않게)
    return metrics


def parse_fit(payload: Mapping[str, Any]) -> FitResult:
    model = payload.get("model")
    if not model:
        raise TrainerError("fit 결과에 model 이 없습니다")
    return FitResult(model=str(model), metrics=flat_metrics(payload, "fit"))


def parse_eval(payload: Mapping[str, Any]) -> EvalResult:
    """``eval`` 응답. 지표가 하나도 없으면 **오류**다 — 재지 않은 것을 "쟀다"로 넘기면 안 된다."""
    metrics = flat_metrics(payload, "eval")
    if not metrics:
        raise TrainerError("eval 결과에 숫자 지표가 없습니다")
    return EvalResult(metrics=metrics)


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
    on_log: LogSink | None = None,
) -> tuple[dict[str, Any], str]:
    """어댑터를 한 번 부르고 ``(결과 JSON, stderr)`` 를 돌려준다.

    호출자는 stderr 를 로그로 흘린다. 실패는 전부 `TrainerError` — 호출부가 라운드 실패로 기록한다.

    ``on_log`` 를 주면 **stderr 를 줄 단위로 그때그때** 넘긴다(학습은 몇 시간이라 끝나고 한 번에 주면
    진행이 안 보인다). 이때 stdout 은 임시 파일로 받는다 — 파이프 두 개를 한 스레드로 읽으면
    한쪽이 가득 찰 때 교착이 나기 때문이다. 어댑터는 진행 로그를 stderr 로 보낸다(계약 §1.1).
    """
    argv = [*command, verb, *args]
    if on_log is None:
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
        code, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
    else:
        code, out, err = _call_streaming(
            argv, cwd=cwd, timeout=timeout, env=env, on_log=on_log, verb=verb
        )

    if code != 0:
        tail = (err or out).strip().splitlines()
        hint = tail[-1] if tail else "(출력 없음)"
        raise TrainerError(f"어댑터 {verb} 실패 (exit {code}): {hint}")

    return last_json_line(out), err


def _call_streaming(
    argv: list[str],
    *,
    cwd: Path | None,
    timeout: float,
    env: Mapping[str, str] | None,
    on_log: LogSink,
    verb: str,
) -> tuple[int, str, str]:
    """stdout 은 임시 파일, stderr 는 줄 단위로 ``on_log`` — 교착 없이 진행을 보여 준다.

    타임아웃은 **감시 타이머**로 건다. 읽기 루프 안에서 시계를 보면 *조용히 멈춘* 어댑터(로그 한 줄도
    안 내고 붙잡혀 있는 경우)에는 영원히 안 걸린다 — 학습 어댑터에서 실제로 있을 수 있는 모양이다.
    """
    lines: list[str] = []
    timed_out = threading.Event()
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace", newline="") as sink:
        try:
            proc = subprocess.Popen(  # 등록된 학습기 명령(trainers.yaml) — 셸 없이 argv 로만 부른다
                argv,
                cwd=str(cwd) if cwd else None,
                env=dict(env) if env else None,
                stdout=sink,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise TrainerError(f"어댑터를 실행할 수 없습니다: {argv[0]}") from exc

        def _kill() -> None:
            timed_out.set()
            proc.kill()

        watchdog = threading.Timer(timeout, _kill)
        watchdog.daemon = True
        watchdog.start()
        assert proc.stderr is not None
        try:
            with proc:
                for raw in proc.stderr:
                    line = raw.rstrip("\r\n")
                    lines.append(line)
                    on_log(line)
        finally:
            watchdog.cancel()
        if timed_out.is_set():
            raise TrainerError(f"어댑터가 {timeout:.0f}s 안에 끝나지 않았습니다 ({verb})")
        sink.seek(0)
        return proc.returncode, sink.read(), "\n".join(lines)


def merge_spec(
    base: Mapping[str, Any] | None, override: Mapping[str, Any] | None
) -> dict[str, Any]:
    """등록부 ``spec`` 위에 호출별 ``spec`` 을 얹는다(얕은 병합, 호출 쪽이 이긴다).

    spec 은 **불투명**이라 Graft 가 키를 해석하지 않는다(설계 §1.3) — 그래서 깊은 병합도 하지 않는다.
    """
    return {**dict(base or {}), **dict(override or {})}


@contextmanager
def spec_file(spec: Mapping[str, Any] | None) -> Iterator[Path | None]:
    """``--spec`` 으로 넘길 임시 JSON 파일. 빈 spec 이면 ``None``(인자를 아예 붙이지 않는다)."""
    if not spec:
        yield None
        return
    tmp = Path(tempfile.mkdtemp(prefix="anograft-spec-")) / "spec.json"
    tmp.write_text(json.dumps(dict(spec), ensure_ascii=False), encoding="utf-8")
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def fit(
    command: Sequence[str],
    *,
    dataset: Path,
    out: Path,
    seed: int = 0,
    spec: Mapping[str, Any] | None = None,
    cwd: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    on_log: LogSink | None = None,
) -> FitResult:
    """``fit`` verb — 학습기가 데이터셋으로 학습하고 **불투명한 모델 참조**를 돌려준다.

    ``metrics`` 는 **평평한 float 맵**이다(`parse_fit` 이 숫자 아닌 값을 버린다). 클래스별 값은
    ``"<지표>/<클래스>"`` 키로 편다 — 예 ``{"mAP50": 0.41, "mAP50/bent": 0.62}``.
    """
    out.mkdir(parents=True, exist_ok=True)
    with spec_file(spec) as sp:
        args = ["--dataset", str(dataset), "--out", str(out), "--seed", str(int(seed))]
        if sp is not None:
            args += ["--spec", str(sp)]
        payload, _ = call(command, "fit", args, cwd=cwd, timeout=timeout, on_log=on_log)
    return parse_fit(payload)


def predict(
    command: Sequence[str],
    *,
    model: str,
    images: Path,
    out: Path,
    spec: Mapping[str, Any] | None = None,
    cwd: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    on_log: LogSink | None = None,
) -> PredictResult:
    """``predict`` verb — 출력은 **Graft 가 이미 임포트하는 형식**(``masks/`` + ``scores/``, 설계 §1.4).

    ``spec`` 은 `fit` 과 **같은 불투명 dict** 를 준다 — 추론 해상도가 학습과 달라지면 조용히 나빠지므로
    등록부 spec 이 양쪽에 똑같이 닿아야 한다(어댑터가 모르는 키는 무시한다).
    """
    out.mkdir(parents=True, exist_ok=True)
    with spec_file(spec) as sp:
        args = ["--model", str(model), "--images", str(images), "--out", str(out)]
        if sp is not None:
            args += ["--spec", str(sp)]
        payload, _ = call(command, "predict", args, cwd=cwd, timeout=timeout, on_log=on_log)
    return parse_predict(payload)


def evaluate(
    command: Sequence[str],
    *,
    model: str,
    dataset: Path,
    seed: int = 0,
    spec: Mapping[str, Any] | None = None,
    cwd: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    on_log: LogSink | None = None,
) -> EvalResult:
    """``eval`` verb (선택) — **학습하지 않고** 이미 학습된 모델을 이 데이터셋의 ``val`` 에서 잰다.

    이중 평가셋(T16)이 이것을 부른다. 왜 `fit` 을 다시 부르지 않나: 한 바퀴의 학습은 몇 시간이고 평가는
    추론 한 번이다. 왜 Graft 가 직접 지표를 계산하지 않나: **지표 정의는 어댑터 몫**이라(계약 §1.1) 우리가
    따로 계산하면 champion 의 점수와 challenger 의 점수가 서로 다른 계산에서 나온다.

    ``spec`` 은 `fit`·`predict` 와 **같은 불투명 dict** 다(해상도가 달라지면 값이 조용히 달라진다).
    함수 이름이 `eval` 이 아닌 이유는 파이썬 내장 이름과 겹치기 때문이고, verb 이름은 그대로 ``eval`` 이다.
    """
    with spec_file(spec) as sp:
        args = ["--model", str(model), "--dataset", str(dataset), "--seed", str(int(seed))]
        if sp is not None:
            args += ["--spec", str(sp)]
        payload, _ = call(command, "eval", args, cwd=cwd, timeout=timeout, on_log=on_log)
    return parse_eval(payload)

"""학습기 등록부 — **레시피 밖** 로컬 ``trainers.yaml`` (설계 §1.5, 사용자 결정 2026-09-23).

어댑터 등록에는 *실행할 명령*이 들어간다. 레시피는 사람이 읽고 diff 하고 **공유하는** 물건(README·samples·
이슈 첨부)이라, 거기에 임의 명령이 들어가면 YAML 이 실행 가능한 파일이 된다. 그래서 명령은 여기 두고
레시피는 **이름만** 참조한다::

    # trainers.yaml  (gitignore — 사람마다 경로가 다르다)
    trainers:
      noop:
        command: [python, adapters/noop.py]
      yolo-seg:
        command: ["C:/…/graft-train/Scripts/python.exe", "adapters/yolo_seg.py"]
        spec: {epochs: 40, imgsz: 640}

찾는 순서는 **명시 경로 → cwd/trainers.yaml → ~/.anograft/trainers.yaml**. 레시피 기준 폴백은 두지 않는다
(레시피와 학습 환경은 서로 독립이다).

`probe` 는 `core/registry.py` 의 "가용 여부 + 사유" 패턴을 그대로 따른다 — 없으면 앱을 죽이지 않고
사유를 노출한다(fail-soft).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from anograft.loop.contract import TrainerError, TrainerInfo, call, parse_info

TRAINERS_FILENAME = "trainers.yaml"
PROBE_TIMEOUT_S = 30.0


class TrainerSpec(BaseModel):
    """``trainers.yaml`` 의 항목 하나."""

    model_config = ConfigDict(extra="forbid")

    command: list[str] = Field(min_length=1)
    spec: dict[str, Any] = Field(default_factory=dict)  # 불투명 — 어댑터가 해석한다
    cwd: str | None = None
    timeout: float | None = Field(default=None, gt=0)

    @field_validator("command")
    @classmethod
    def _no_blank(cls, v: list[str]) -> list[str]:
        if any(not str(x).strip() for x in v):
            raise ValueError("command 에 빈 항목이 있습니다")
        return [str(x) for x in v]


class TrainersFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trainers: dict[str, TrainerSpec] = Field(default_factory=dict)


@dataclass(frozen=True)
class TrainerStatus:
    """``anograft trainer list`` 가 보여 주는 한 줄 — `core.registry.MethodInfo` 와 같은 역할."""

    name: str
    available: bool
    info: TrainerInfo | None
    reason: str | None

    @property
    def summary(self) -> str:
        if not self.available:
            return f"쓸 수 없음 — {self.reason}"
        assert self.info is not None
        bits = [
            self.info.dataset_format,
            "정상만 학습" if self.info.normal_only else "라벨 학습",
            "·".join(self.info.capabilities),
        ]
        if not self.info.deterministic:
            bits.append("비결정적(시드 평균 필요)")
        return " · ".join(bits)


def find_trainers_file(explicit: Path | None = None, *, cwd: Path | None = None) -> Path | None:
    """명시 경로 → ``cwd/trainers.yaml`` → ``~/.anograft/trainers.yaml`` 순으로 찾는다."""
    if explicit is not None:
        return explicit if explicit.exists() else None
    here = (cwd or Path.cwd()) / TRAINERS_FILENAME
    if here.exists():
        return here
    home = Path.home() / ".anograft" / TRAINERS_FILENAME
    return home if home.exists() else None


def load_trainers(path: Path) -> dict[str, TrainerSpec]:
    """``trainers.yaml`` 을 읽어 이름 → 스펙. 스키마 오류는 `TrainerError` 로 올린다."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise TrainerError(f"{path} 를 읽을 수 없습니다: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise TrainerError(f"{path} 의 최상위는 매핑이어야 합니다")
    try:
        parsed = TrainersFile.model_validate(dict(raw))
    except Exception as exc:  # pydantic ValidationError
        raise TrainerError(f"{path} 형식 오류: {exc}") from exc
    return dict(parsed.trainers)


def resolve_cwd(spec: TrainerSpec, base: Path) -> Path:
    """스펙의 ``cwd`` 를 ``trainers.yaml`` 이 있는 폴더 기준으로 푼다."""
    if not spec.cwd:
        return base
    p = Path(spec.cwd)
    return p if p.is_absolute() else (base / p)


def probe(
    name: str, spec: TrainerSpec, base: Path, *, timeout: float = PROBE_TIMEOUT_S
) -> TrainerStatus:
    """``info`` 를 불러 가용 여부를 판정한다. **예외를 던지지 않는다**(fail-soft)."""
    try:
        payload, _ = call(spec.command, "info", cwd=resolve_cwd(spec, base), timeout=timeout)
        info = parse_info(payload)
    except TrainerError as exc:
        return TrainerStatus(name=name, available=False, info=None, reason=str(exc))

    if info.name and info.name != name:
        # 이름이 달라도 막지는 않는다 — 등록 이름이 정본이고, 어댑터 이름은 참고다.
        pass
    return TrainerStatus(name=name, available=True, info=info, reason=None)


def probe_all(
    trainers: Mapping[str, TrainerSpec], base: Path, *, timeout: float = PROBE_TIMEOUT_S
) -> list[TrainerStatus]:
    return [probe(n, s, base, timeout=timeout) for n, s in sorted(trainers.items())]

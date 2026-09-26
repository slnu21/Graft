"""라운드 설정 — **레시피 밖** 로컬 ``loop.yaml`` (설계 §2·§2b, T10. 사용자 결정 2026-09-25).

`trainers.yaml` 과 같은 자리·같은 이유다: 여기 적히는 것은 **이 사람의 환경**(현장 이미지 폴더 · 평가셋 ·
학습기 이름 · 출력 루트)이라 공유되는 레시피에 들어가면 안 된다. 레시피는 "어떻게 합성하는가"만 들고 있고,
루프는 "그 레시피를 어떤 데이터로 몇 번 돌리는가"를 여기서 읽는다.

`loop tick`(T13)을 **스케줄러가 인자 없이** 부르므로 설정이 파일에 있어야 한다 — 트리거 정책
(신규 N개·최소 간격, T14 의 `trigger` 블록)도 그래서 여기 얹혀 있다.

찾는 순서는 `trainers.yaml` 과 같다: **명시 경로 → cwd/loop.yaml → ~/.anograft/loop.yaml**.

경로 해석은 **loop.yaml 파일 기준**이다(레시피의 cwd 우선 규칙과 다르다). 스케줄러가 어느 폴더에서
부르든 같은 라운드를 가리켜야 하기 때문이다 — 그래서 `resolved()` 가 전부 절대경로로 바꿔 돌려준다.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from anograft.core.novelty import DEFAULT_THRESHOLD as NOVELTY_THRESHOLD

LOOP_FILENAME = "loop.yaml"

#: 라운드가 만드는 학습 데이터셋 형식 — 학습기가 `info` 로 선언한 것과 맞춰야 한다(계약 §1.2).
SUPPORTED_FORMATS: tuple[str, ...] = ("yolo", "pairs")


class LoopConfigError(ValueError):
    pass


class DataSplit(BaseModel):
    """이미지 + 라벨(또는 마스크) 폴더 한 쌍 — 실제 학습분·동결 평가셋이 같은 모양이다."""

    model_config = ConfigDict(extra="forbid")

    images: str
    labels: str | None = None  # yolo 형식일 때
    masks: str | None = None  # pairs 형식일 때

    @field_validator("images")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("images 경로가 비어 있습니다")
        return str(v)


class ReviewSettings(BaseModel):
    """검토 대기 큐(T5) 설정 — 사람이 한 라운드에 볼 양."""

    model_config = ConfigDict(extra="forbid")

    n: int = Field(default=30, ge=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou: float = Field(default=0.3, gt=0.0, le=1.0)
    #: 경계·확신·무작위 비율(합은 자유 — 정규화한다). 불일치 몫의 상한이 경계 비율이다.
    mix: tuple[float, float, float] = (0.6, 0.2, 0.2)
    #: 채택분을 은행에 넣을 때 클래스 이름을 고정할지(비지도 예측처럼 클래스를 모를 때)
    accept_class: str | None = None
    #: **처음 보는 형상** 임계(T15) — 이 이상이면 큐에서 맨 앞으로 오고, 채택하면 **미분류**로 들어간다.
    #: 0 = 끔. 기본값은 보수적이고 실무 값은 확인 게이트(설계 §8).
    novelty_threshold: float = Field(default=NOVELTY_THRESHOLD, ge=0.0, le=1.0)
    #: 마스크를 성분으로 쪼개지 않고 통째로 한 조각으로 넣을지
    keep_whole: bool = False

    @field_validator("mix")
    @classmethod
    def _positive_sum(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if sum(v) <= 0:
            raise ValueError("mix 의 합이 0 이하입니다")
        if any(x < 0 for x in v):
            raise ValueError("mix 에 음수가 있습니다")
        return v


class PromoteSettings(BaseModel):
    """승급 판정 — 설계 §2 규약 2("Δ 가 시드 요동 이하면 개선이 아니다")."""

    model_config = ConfigDict(extra="forbid")

    #: 비교할 지표 이름. `fit` 의 **평평한 지표 맵**에서 이 키를 읽는다(계약 §1.1).
    metric: str = "mAP50"
    #: 학습 시드 요동(BENCHMARKS §2 에서 잰 값). 이보다 작은 Δ 는 개선이 아니다.
    noise: float = Field(default=0.04, ge=0.0)


class TriggerSettings(BaseModel):
    """언제 새 라운드를 여는가 (설계 §2b.3, T14). `loop tick` 만 본다 — 사람이 부른 `loop run` 은 그냥 돈다.

    **기본값은 전부 0 = 제한 없음**이다. 임계값은 현장마다 다르고(설계 §8 확인 게이트) 기본값이 라운드를
    막으면 "왜 안 도는지" 모르는 사람이 먼저 생긴다. 실무 제안은 `loop.example.yaml` 주석에 있다.
    """

    model_config = ConfigDict(extra="forbid")

    #: 지난 라운드 뒤로 보관함에 들어온 조각 수(0 = 안 봄)
    min_labels: int = Field(default=0, ge=0)
    #: 아직 스코어링하지 않은 현장 이미지 장수(0 = 안 봄)
    min_images: int = Field(default=0, ge=0)
    #: 마지막 라운드로부터 최소 간격(시간)
    min_interval_hours: float = Field(default=0.0, ge=0.0)
    #: 이만큼 지나면 양과 무관하게 돈다(드리프트 감시). null = 안 봄
    max_interval_hours: float | None = Field(default=None, gt=0.0)

    def policy(self) -> Any:
        """`loop.policy.TriggerPolicy` 로 — 판정은 순수 함수가 한다(여기는 검증·기본값만)."""
        from anograft.loop.policy import TriggerPolicy

        return TriggerPolicy(
            min_labels=self.min_labels,
            min_images=self.min_images,
            min_interval_hours=self.min_interval_hours,
            max_interval_hours=self.max_interval_hours,
        )


class LoopConfig(BaseModel):
    """``loop.yaml`` 전체."""

    model_config = ConfigDict(extra="forbid")

    trainer: str  # trainers.yaml 의 이름 (명령이 아니다)
    bank: str
    recipe: str
    out: str  # 라운드 폴더 루트
    #: 스코어링할 현장 이미지(폴더 또는 .txt 목록). 없으면 **수집 단계를 건너뛴다**(합성만 도는 라운드)
    field: str | None = None
    #: 동결 평가셋 — 조립한 학습 데이터셋의 **val** 로 들어가고, `fit` 의 지표가 곧 라운드 점수다
    eval: DataSplit
    #: 실제 학습분(있으면 train 에 함께 들어간다). 없으면 train 은 합성만
    train_base: DataSplit | None = None
    review: ReviewSettings = Field(default_factory=ReviewSettings)
    promote: PromoteSettings = Field(default_factory=PromoteSettings)
    #: `loop tick` 이 "지금 돌 때인가"를 판정하는 기준(T14). 기본값은 제한 없음
    trigger: TriggerSettings = Field(default_factory=TriggerSettings)
    #: 한 라운드에 만들 합성 장수. 없으면 레시피의 `output.count` 그대로
    synth_count: int | None = Field(default=None, ge=1)
    seed: int = 7
    #: 학습기에 넘길 spec(불투명) — 등록부 spec 위에 얕게 병합된다
    spec: dict[str, Any] = Field(default_factory=dict)
    #: trainers.yaml 경로(기본: 찾는 순서대로)
    trainers_file: str | None = None

    def resolved(self, base: Path) -> ResolvedLoop:
        """경로를 **loop.yaml 기준 절대경로**로 바꾼다 — 어느 폴더에서 불러도 같은 라운드를 가리키게."""
        return ResolvedLoop(config=self, base=base.resolve())


class ResolvedLoop:
    """경로가 풀린 설정. 값은 `config` 그대로 두고 경로 해석만 여기서 한다(원본을 변형하지 않는다)."""

    def __init__(self, config: LoopConfig, base: Path) -> None:
        self.config = config
        self.base = base

    def path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self.base / p)

    @property
    def bank(self) -> Path:
        return self.path(self.config.bank)

    @property
    def recipe(self) -> Path:
        return self.path(self.config.recipe)

    @property
    def out(self) -> Path:
        return self.path(self.config.out)

    @property
    def field(self) -> Path | None:
        return self.path(self.config.field) if self.config.field else None

    @property
    def trainers_file(self) -> Path | None:
        return self.path(self.config.trainers_file) if self.config.trainers_file else None

    def split(self, split: DataSplit) -> dict[str, Path | None]:
        return {
            "images": self.path(split.images),
            "labels": self.path(split.labels) if split.labels else None,
            "masks": self.path(split.masks) if split.masks else None,
        }


def find_loop_file(explicit: Path | None = None, *, cwd: Path | None = None) -> Path | None:
    """명시 경로 → ``cwd/loop.yaml`` → ``~/.anograft/loop.yaml``. `trainers.yaml` 과 같은 규칙."""
    if explicit is not None:
        return explicit if explicit.exists() else None
    here = (cwd or Path.cwd()) / LOOP_FILENAME
    if here.exists():
        return here
    home = Path.home() / ".anograft" / LOOP_FILENAME
    return home if home.exists() else None


def load_loop_config(path: str | Path) -> ResolvedLoop:
    """``loop.yaml`` 을 읽고 검증해 **경로가 풀린** 설정을 돌려준다."""
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise LoopConfigError(f"{p} 를 읽을 수 없습니다: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise LoopConfigError(f"{p} 의 최상위는 매핑이어야 합니다")
    try:
        config = LoopConfig.model_validate(dict(raw))
    except Exception as exc:  # pydantic ValidationError
        raise LoopConfigError(f"{p} 형식 오류: {exc}") from exc
    return config.resolved(p.parent)

"""스테이지 레지스트리 (설계 §6) — ``(stage, method)`` → 클래스, 가용 여부, 사유.

"무엇을 고를 수 있나"의 정본은 **레시피 스키마의 discriminated union**이다. 레지스트리는 그 선택지 각각이
구현됐는지·선택 의존성이 있는지를 덧붙인다. ``anograft methods``와 GUI 툴팁이 같은 정보를 쓴다.

새 알고리즘 = 스테이지 클래스에 ``@register`` + 프리셋 YAML 한 장. 파이프라인·스키마·writer는 건드리지 않는다.
"""

from __future__ import annotations

import importlib
import importlib.util
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel

from anograft.core import recipe as R
from anograft.core.types import Context

# 레지스트리가 아는 스테이지 순서. ``roi``는 레시피에선 ``placement.roi`` 하위 설정이지만 실행은 별도 스테이지(이미지당 1회).
STAGE_ORDER: tuple[str, ...] = (
    "source",
    "geometry",
    "roi",
    "placement",
    "blend",
    "harmonize",
    "degrade",
    "gtmask",
)


@runtime_checkable
class Stage(Protocol):
    """모든 스테이지의 계약. ``apply``는 Context를 바꾸지 않고 새 것을 돌려준다."""

    stage: ClassVar[str]
    methods: ClassVar[tuple[str, ...]]
    requires: ClassVar[tuple[str, ...]]

    def __init__(self, cfg: BaseModel, deps: Mapping[str, Any]) -> None: ...

    def apply(self, ctx: Context) -> Context: ...


@dataclass(frozen=True)
class MethodInfo:
    stage: str
    method: str
    implemented: bool
    available: bool
    reason: str | None  # 미구현/불가 사유
    requires: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return self.implemented and self.available


class StageNotImplementedError(NotImplementedError):
    pass


class StageUnavailableError(RuntimeError):
    pass


_REGISTRY: dict[tuple[str, str], type] = {}
_LOADED = False


def register(cls: type) -> type:
    """스테이지 클래스 데코레이터. ``stage``·``methods``·``requires`` 클래스 속성이 필요하다."""
    for attr in ("stage", "methods", "requires"):
        if not hasattr(cls, attr):
            raise TypeError(f"{cls.__name__}: 클래스 속성 {attr!r}가 필요합니다")
    if cls.stage not in STAGE_ORDER:
        raise ValueError(f"{cls.__name__}: 알 수 없는 스테이지 {cls.stage!r}")
    for m in cls.methods:
        key = (cls.stage, m)
        if key in _REGISTRY and _REGISTRY[key] is not cls:
            raise ValueError(f"{key} 는 이미 {_REGISTRY[key].__name__} 가 등록했습니다")
        _REGISTRY[key] = cls
    return cls


def unregister(stage: str, method: str) -> None:
    """테스트용."""
    _REGISTRY.pop((stage, method), None)


def ensure_loaded() -> None:
    """스테이지 모듈을 import해 등록을 일으킨다(지연 — 순환 import 회피)."""
    global _LOADED
    if _LOADED:
        return
    importlib.import_module("anograft.core.stages")
    _LOADED = True


def availability(requires: tuple[str, ...]) -> tuple[bool, str | None]:
    """선택 의존성 모듈이 전부 import 가능한가. 없으면 (False, 사유)."""
    missing = [m for m in requires if importlib.util.find_spec(m) is None]
    if missing:
        return False, f"선택 의존성 없음: {', '.join(missing)}"
    return True, None


def config_method(cfg: BaseModel) -> str:
    """설정 모델에서 method 키를 읽는다. gtmask는 ``policy``가 그 역할."""
    m = getattr(cfg, "method", None)
    if m is None:
        m = getattr(cfg, "policy", None)
    if m is None:
        raise ValueError(f"{type(cfg).__name__}: method/policy 필드가 없습니다")
    return str(m)


def build(stage: str, cfg: BaseModel, deps: Mapping[str, Any] | None = None) -> Stage:
    """``(stage, method)``에 등록된 클래스를 설정으로 생성한다. 미구현·불가면 이유가 담긴 예외."""
    ensure_loaded()
    method = config_method(cfg)
    cls = _REGISTRY.get((stage, method))
    if cls is None:
        raise StageNotImplementedError(f"{stage}.{method} 은(는) 아직 구현되지 않았습니다")
    ok, reason = availability(cls.requires)
    if not ok:
        raise StageUnavailableError(f"{stage}.{method}: {reason}")
    return cls(cfg, dict(deps or {}))


def _union_variants(annotation: Any) -> list[type[BaseModel]]:
    """``Annotated[Union[A, B], Field(...)]`` 또는 ``Annotated[A, ...]``에서 pydantic 모델 목록."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
        origin = typing.get_origin(annotation)
    if origin is typing.Union or (origin is not None and origin.__name__ == "UnionType"):
        return [
            a
            for a in typing.get_args(annotation)
            if isinstance(a, type) and issubclass(a, BaseModel)
        ]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return []


def schema_methods(stage: str) -> list[str]:
    """레시피 스키마가 허용하는 method 목록 — 선택지의 정본."""
    if stage == "roi":
        ann = R.SampledPlacementConfig.model_fields["roi"].annotation
    elif stage == "gtmask":
        return list(typing.get_args(R.GtMaskConfig.model_fields["policy"].annotation))
    else:
        ann = R.PipelineConfig.model_fields[stage].annotation
    out: list[str] = []
    for variant in _union_variants(ann):
        default = variant.model_fields["method"].default
        out.append(str(default))
    return out


def list_methods(stage: str | None = None) -> list[MethodInfo]:
    """스테이지별 선택지 + 구현/가용 상태. ``anograft methods``가 그대로 출력한다."""
    ensure_loaded()
    infos: list[MethodInfo] = []
    for s in STAGE_ORDER:
        if stage is not None and s != stage:
            continue
        for m in schema_methods(s):
            cls = _REGISTRY.get((s, m))
            if cls is None:
                infos.append(MethodInfo(s, m, False, False, "미구현", ()))
                continue
            ok, reason = availability(cls.requires)
            infos.append(MethodInfo(s, m, True, ok, reason, tuple(cls.requires)))
    return infos


def registered() -> dict[tuple[str, str], type]:
    ensure_loaded()
    return dict(_REGISTRY)

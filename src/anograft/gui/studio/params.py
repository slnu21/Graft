"""스테이지 설정(pydantic 모델) → 위젯 스펙. Qt 없음 — ``panels.py`` 의 ``ParamForm`` 이 스펙대로 위젯을 만든다.

"스키마가 곧 UI": 새 method 의 설정 모델을 ``recipe.py`` 에 추가하면 카드 편집기는 저절로 생긴다(GUI 수정 0).
필드 이름은 레시피 YAML 키 그대로 보인다 — CLI·레시피 파일과 같은 어휘라 번역하지 않는다.

- ``method``/``policy`` 는 카드 콤보가 맡고, ``placement.roi`` 는 하위 스테이지(자기 method 콤보 + 폼)라 여기서 제외.
- 중첩 모델(``elastic``·``shrink_on_fail``·``jitter``)은 ``elastic.alpha`` 처럼 점 경로로 평탄화.
- ``X | None`` 은 ``optional`` — 폼이 "끔" 체크박스를 앞에 둔다. 켤 때 시작값은 ``on_default``(이름별 표, 없으면 형식별 기본).
- 숫자 제약은 ``Ge/Le/Gt/Lt`` 메타데이터만 읽는다. 커스텀 validator(rotate ±360 등)는 재검증 오류로 카드에 표시된다.
"""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from pydantic.fields import FieldInfo

Kind = Literal["int", "float", "int_range", "range", "bool", "choice", "text", "list", "path"]

EXCLUDE: frozenset[str] = frozenset({"method", "policy", "roi"})

# X | None 필드를 켤 때 시작값 — 없으면 형식별 기본(_kind_default). 값은 레시피 YAML 표현(list·str).
ON_DEFAULTS: dict[str, Any] = {
    "opacity": [0.5, 1.0],
    "jpeg_quality": [60, 95],
    "motion_blur_px": [0.0, 6.0],
    "vignette": [0.0, 0.3],
    "gamma": [0.8, 1.25],
    "classes": [],
    "texture_dir": "",
    "center": [0.0, 0.0],  # annulus — 고정 중심(px). 켠 뒤 대상 좌표로 고친다
    "radius": 100.0,  # annulus — 고정 기준 반경(px)
}

_BIG = 1_000_000_000


@dataclass(frozen=True)
class FieldSpec:
    name: str  # 점 경로 ("elastic.alpha")
    kind: Kind
    value: Any  # 현재 값 (YAML 표현: 범위는 list, 경로는 str). optional 이 꺼져 있으면 None
    optional: bool = False
    choices: tuple[str, ...] = ()
    lo: float | None = None  # ge/gt (gt 는 열린 구간이라 폼이 한 step 안쪽으로)
    hi: float | None = None  # le/lt
    lo_open: bool = False
    hi_open: bool = False
    on_default: Any = None  # optional 을 켤 때 넣을 값
    hint: str = ""  # 툴팁 — 형식·제약 요약

    @property
    def enabled(self) -> bool:
        return not (self.optional and self.value is None)


@dataclass
class _Ann:
    """annotation 분해 결과."""

    base: Any  # Annotated·Optional 을 벗긴 형
    optional: bool = False
    metadata: list[Any] = field(default_factory=list)


def _strip(ann: Any) -> _Ann:
    out = _Ann(ann)
    changed = True
    while changed:
        changed = False
        origin = typing.get_origin(out.base)
        if origin is typing.Annotated:
            args = typing.get_args(out.base)
            out.base = args[0]
            out.metadata.extend(args[1:])
            changed = True
        elif origin is typing.Union or origin is types.UnionType:
            args = [a for a in typing.get_args(out.base) if a is not type(None)]
            if len(args) < len(typing.get_args(out.base)):
                out.optional = True
            if len(args) == 1:
                out.base = args[0]
                changed = True
            else:  # 모델 union(roi 등) — 호출자가 제외한다
                out.base = typing.Union[tuple(args)]  # noqa: UP007 — 런타임 재조립
    return out


def _bounds(metadata: list[Any]) -> tuple[float | None, float | None, bool, bool]:
    lo: float | None = None
    hi: float | None = None
    lo_open = hi_open = False
    # annotated_types 의 Ge/Gt/Le/Lt — 직접 import 하지 않고(pydantic 의 간접 의존성) 속성으로 읽는다
    for m in metadata:
        for attr, is_lo, is_open in (
            ("ge", True, False),
            ("gt", True, True),
            ("le", False, False),
            ("lt", False, True),
        ):
            v = getattr(m, attr, None)
            if v is None:
                continue
            if is_lo:
                lo, lo_open = float(v), is_open
            else:
                hi, hi_open = float(v), is_open
    return lo, hi, lo_open, hi_open


def _kind_of(base: Any) -> tuple[Kind, tuple[str, ...]] | None:
    origin = typing.get_origin(base)
    if origin is typing.Literal:
        return "choice", tuple(str(a) for a in typing.get_args(base))
    if base is bool:
        return "bool", ()
    if base is int:
        return "int", ()
    if base is float:
        return "float", ()
    if base is str:
        return "text", ()
    if base is Path:
        return "path", ()
    if origin is tuple:
        args = typing.get_args(base)
        if len(args) == 2 and args[0] is args[1] and args[0] in (int, float):
            return ("int_range" if args[0] is int else "range"), ()
    if origin is list:
        return "list", ()
    return None


def _kind_default(kind: Kind, lo: float | None, hi: float | None) -> Any:
    if kind in ("range", "int_range"):
        a = 0 if lo is None else lo
        b = 1 if hi is None else hi
        return [int(a), int(b)] if kind == "int_range" else [float(a), float(b)]
    return {"int": 0, "float": 0.0, "bool": False, "text": "", "list": [], "path": ""}.get(kind)


def _fmt(v: float) -> str:
    return f"{v:g}"


def _hint(
    kind: Kind, lo: float | None, hi: float | None, lo_open: bool, hi_open: bool, optional: bool
) -> str:
    names = {
        "int": "정수",
        "float": "실수",
        "int_range": "정수 범위 [lo, hi]",
        "range": "실수 범위 [lo, hi]",
        "bool": "on/off",
        "choice": "선택",
        "text": "문자열",
        "list": "목록 (쉼표로 구분)",
        "path": "경로",
    }
    parts = [names[kind]]
    if lo is not None or hi is not None:
        left = ("(" if lo_open else "[") + (_fmt(lo) if lo is not None else "−∞")
        right = (_fmt(hi) if hi is not None else "∞") + (")" if hi_open else "]")
        parts.append(f"{left}, {right}")
    if optional:
        parts.append("null = 끔")
    return " · ".join(parts)


def _to_yaml_value(v: Any) -> Any:
    if isinstance(v, Path):
        return v.as_posix()
    if isinstance(v, tuple):
        return list(v)
    return v


def field_specs(
    cfg: BaseModel, *, prefix: str = "", exclude: frozenset[str] = EXCLUDE
) -> list[FieldSpec]:
    """설정 모델의 편집 가능한 필드 스펙 목록(선언 순서). 중첩 모델은 점 경로로 평탄화, 모델 union(roi)은 건너뛴다."""
    specs: list[FieldSpec] = []
    fields: dict[str, FieldInfo] = type(cfg).model_fields
    for name, fi in fields.items():
        if not prefix and name in exclude:
            continue
        a = _strip(fi.annotation)
        a.metadata = list(fi.metadata) + a.metadata
        value = getattr(cfg, name)
        if isinstance(a.base, type) and issubclass(a.base, BaseModel):
            if isinstance(value, BaseModel):
                specs.extend(field_specs(value, prefix=f"{prefix}{name}.", exclude=exclude))
            continue
        kind_choices = _kind_of(a.base)
        if kind_choices is None:
            continue  # 모델 union 등 — 카드가 따로 다룬다
        kind, choices = kind_choices
        lo, hi, lo_open, hi_open = _bounds(a.metadata)
        on_default = ON_DEFAULTS.get(name, _kind_default(kind, lo, hi)) if a.optional else None
        specs.append(
            FieldSpec(
                name=f"{prefix}{name}",
                kind=kind,
                value=_to_yaml_value(value),
                optional=a.optional,
                choices=choices,
                lo=lo,
                hi=hi,
                lo_open=lo_open,
                hi_open=hi_open,
                on_default=on_default,
                hint=_hint(kind, lo, hi, lo_open, hi_open, a.optional),
            )
        )
    return specs


def coerce(spec: FieldSpec, raw: Any) -> Any:
    """위젯에서 온 값을 레시피 dict 에 넣을 표현으로. 폼이 만든 값은 대부분 그대로지만 문자열 입력은 여기서 정리한다."""
    if raw is None:
        return None
    if spec.kind == "list":
        if isinstance(raw, str):
            return [s.strip() for s in raw.split(",") if s.strip()]
        return [str(s) for s in raw]
    if spec.kind == "path":
        return Path(str(raw)).as_posix() if str(raw).strip() else ""
    if spec.kind == "text":
        return str(raw)
    if spec.kind in ("int_range", "range"):
        a, b = raw
        return [int(a), int(b)] if spec.kind == "int_range" else [float(a), float(b)]
    if spec.kind == "int":
        return int(raw)
    if spec.kind == "float":
        return float(raw)
    if spec.kind == "bool":
        return bool(raw)
    return raw


def spin_bounds(spec: FieldSpec) -> tuple[float, float]:
    """스핀박스 min/max — 제약이 없으면 ±1e9, 열린 구간은 폼이 한 step 안쪽으로 조정한다."""
    lo = -_BIG if spec.lo is None else spec.lo
    hi = _BIG if spec.hi is None else spec.hi
    return lo, hi


def spin_step(spec: FieldSpec) -> float:
    """실수 스핀 step — [0,1] 계열은 0.05, 그 외 0.1. 정수는 1."""
    if spec.kind in ("int", "int_range"):
        return 1.0
    if spec.hi is not None and spec.lo is not None and spec.hi - spec.lo <= 2.0:
        return 0.05
    return 0.1


def required_placeholders(model: type[BaseModel]) -> dict[str, Any]:
    """기본값 없는 필수 필드(``mask_dir.path`` 등)에 넣을 자리표시 값 — method 를 콤보로 바꿀 때 검증을 통과시키고
    사용자가 폼에서 채우게 한다. 경로는 ``"."``(없는 파일이라 fail-soft 경고가 바로 카드에 뜬다), 문자열은 ``""``."""
    out: dict[str, Any] = {}
    for name, fi in model.model_fields.items():
        if name in EXCLUDE or not fi.is_required():
            continue
        a = _strip(fi.annotation)
        kc = _kind_of(a.base)
        if kc is None:
            continue
        kind, choices = kc
        if kind == "path":
            out[name] = "."
        elif kind == "choice" and choices:
            out[name] = choices[0]
        else:
            lo, hi, _, _ = _bounds(list(fi.metadata) + a.metadata)
            out[name] = _kind_default(kind, lo, hi)
    return out

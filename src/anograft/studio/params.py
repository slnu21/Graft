"""스테이지 설정(pydantic 모델) → 폼 스펙. Qt 없음 — ``gui/studio/param_form.py`` 가 위젯을, 웹 폼이 입력을 스펙대로 만든다.

"스키마가 곧 UI": 새 method 의 설정 모델을 ``recipe.py`` 에 추가하면 카드 편집기는 저절로 생긴다(GUI 수정 0).
라벨·설명·단위는 ``core/help.py``(한 원천)에서 — 라벨은 한국어, YAML 키·영어·형식은 툴팁(v0.9). 도움말이 없는 필드는
``tests/test_param_help.py`` 가 잡는다.

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

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from anograft.core import recipe as R
from anograft.core import registry
from anograft.core.help import field_help

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
    "max_align_deg": 30.0,  # structure-aware — 정렬 회전 상한(dent-graft 값)
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
    label: str = ""  # 한국어 라벨(core/help.py). 비면 name 을 그대로 보인다
    desc: str = ""  # 무엇 · 올리면/내리면
    unit: str = ""  # px · ° · 배 · gray · 회 · 개
    en: str = ""  # 영어 한 줄
    advanced: bool = False  # 카드에서 "고급 옵션"으로 접히는 필드
    baseline: Any = None  # 프리셋(없으면 스키마 기본) 값 — 바뀜 표시·되돌리기 기준. has_baseline 이 False 면 무의미
    has_baseline: bool = False

    @property
    def enabled(self) -> bool:
        return not (self.optional and self.value is None)

    @property
    def modified(self) -> bool:
        """프리셋 값과 다른가(YAML 표현으로 비교). 기준이 없으면 False."""
        return self.has_baseline and _norm(self.value) != _norm(self.baseline)

    @property
    def title(self) -> str:
        """폼 라벨 — 한국어 라벨 + 단위(없으면 YAML 키)."""
        base = self.label or self.name
        return f"{base} ({self.unit})" if self.unit else base

    @property
    def tooltip(self) -> str:
        """3줄 툴팁 — 라벨 · YAML 키 / 설명 / 영어 · 형식·범위."""
        head = f"{self.label} · {self.name}" if self.label else self.name
        lines = [head]
        if self.desc:
            lines.append(self.desc)
        tail = " · ".join(x for x in (self.en, self.hint) if x)
        if tail:
            lines.append(tail)
        return "\n".join(lines)


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


def _norm(v: Any) -> Any:
    """비교용 정규화 — tuple/list 동일, 실수는 6자리."""
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, float):
        return round(v, 6)
    return v


def baseline_config(recipe: R.Recipe, stage: str, cfg: BaseModel) -> BaseModel | None:
    """카드의 **기준 설정** — 레시피 프리셋의 같은 method 블록(있으면), 아니면 스키마 기본값. 기본값이 없는 필수 필드가 있으면
    None(기준 없음 → 바뀜 표시 안 함). Qt 없음 — 폼의 ● 표시·되돌리기가 이것과 비교한다."""
    cls = type(cfg)
    method = registry.config_method(cfg)
    name = recipe.pipeline.preset
    if name:
        try:
            pipe = R.load_preset(name)
        except (KeyError, ValueError):
            pipe = {}
        block = (pipe.get("placement") or {}).get("roi") if stage == "roi" else pipe.get(stage)
        key = "policy" if stage == "gtmask" else "method"
        if isinstance(block, dict) and str(block.get(key, cls.model_fields[key].default)) == method:
            try:
                return cls.model_validate(block)
            except ValidationError:
                pass
    try:
        return cls()
    except ValidationError:
        return None


def _to_yaml_value(v: Any) -> Any:
    if isinstance(v, Path):
        return v.as_posix()
    if isinstance(v, tuple):
        return list(v)
    return v


def field_specs(
    cfg: BaseModel,
    *,
    prefix: str = "",
    exclude: frozenset[str] = EXCLUDE,
    baseline: BaseModel | None = None,
) -> list[FieldSpec]:
    """설정 모델의 편집 가능한 필드 스펙 목록(선언 순서). 중첩 모델은 점 경로로 평탄화, 모델 union(roi)은 건너뛴다.
    ``baseline``(같은 클래스의 기준 설정, `baseline_config`)을 주면 각 스펙에 기준값이 붙어 ``modified`` 를 판단할 수 있다."""
    specs: list[FieldSpec] = []
    fields: dict[str, FieldInfo] = type(cfg).model_fields
    for name, fi in fields.items():
        if not prefix and name in exclude:
            continue
        a = _strip(fi.annotation)
        a.metadata = list(fi.metadata) + a.metadata
        value = getattr(cfg, name)
        base_value = getattr(baseline, name, None) if baseline is not None else None
        if isinstance(a.base, type) and issubclass(a.base, BaseModel):
            if isinstance(value, BaseModel):
                specs.extend(
                    field_specs(
                        value,
                        prefix=f"{prefix}{name}.",
                        exclude=exclude,
                        baseline=base_value if isinstance(base_value, BaseModel) else None,
                    )
                )
            continue
        kind_choices = _kind_of(a.base)
        if kind_choices is None:
            continue  # 모델 union 등 — 카드가 따로 다룬다
        kind, choices = kind_choices
        lo, hi, lo_open, hi_open = _bounds(a.metadata)
        on_default = ON_DEFAULTS.get(name, _kind_default(kind, lo, hi)) if a.optional else None
        h = field_help(type(cfg), name)
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
                label=h.label if h else "",
                desc=h.desc if h else "",
                unit=h.unit if h else "",
                en=h.en if h else "",
                advanced=h.advanced if h else False,
                baseline=_to_yaml_value(base_value) if baseline is not None else None,
                has_baseline=baseline is not None,
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
        # 레시피 경로는 posix 표기 — Windows 백슬래시는 OS 와 무관하게 '/' 로(POSIX 에선 Path.as_posix 가 '\\' 를 안 바꾼다)
        text = str(raw).strip()
        return Path(text.replace("\\", "/")).as_posix() if text else ""
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

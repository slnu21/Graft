"""``anograft explain`` · ``PARAMS.md`` — 도움말(core/help.py) + 스키마(recipe.py) + 프리셋 meta 를 사람이 읽는 글로. Qt·GUI 없음.

- ``explain_text(query)``: ``geometry`` (스테이지) · ``geometry.scale`` (필드) · ``placement.roi.erode_px`` · ``output.count`` ·
  ``preset:dent-graft`` · ``blend:poisson`` (method) — 라벨 · 설명 · 단위 · 형식·범위 · YAML 키 · 프리셋별 값.
- ``markdown()``: 전체 표(= ``PARAMS.md``). ``tests/test_param_help.py`` 가 저장된 파일과 비교한다 — 문안을 고치면
  ``anograft explain --out PARAMS.md`` 로 재생성(LF).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from anograft.core import recipe as R
from anograft.core import registry
from anograft.core.help import (
    METHOD_HELP,
    STAGE_HELP,
    FieldHelp,
    field_help,
    method_help,
)

TOP_STAGES: tuple[str, ...] = (
    "source",
    "geometry",
    "placement",
    "blend",
    "harmonize",
    "degrade",
    "gtmask",
)
EXTRA_MODELS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("inputs", R.Inputs),
    ("output", R.Output),
    ("output.writer[yolo]", R.YoloWriterConfig),
    ("output.writer[pairs]", R.PairsWriterConfig),
    ("output.writer[mvtec]", R.MvtecWriterConfig),
    ("output.writer[coco]", R.CocoWriterConfig),
)
SKIP_FIELDS: frozenset[str] = frozenset({"method", "policy", "roi", "format", "writer"})


def _flatten(cls: type[BaseModel], prefix: str = "") -> list[tuple[str, type[BaseModel], str, Any]]:
    """(점 경로, 필드를 정의한 모델, 로컬 이름, FieldInfo) — 중첩 모델은 펼치고 union(roi)은 건너뛴다."""
    out: list[tuple[str, type[BaseModel], str, Any]] = []
    for name, fi in cls.model_fields.items():
        if not prefix and name in SKIP_FIELDS:
            continue
        ann = fi.annotation
        variants = registry._union_variants(ann)
        if len(variants) == 1 and not _is_optional(ann):
            out.extend(_flatten(variants[0], f"{prefix}{name}."))
            continue
        if len(variants) > 1:
            continue
        out.append((f"{prefix}{name}", cls, name, fi))
    return out


def _is_optional(ann: Any) -> bool:
    import types
    import typing

    origin = typing.get_origin(ann)
    if origin is typing.Annotated:
        ann = typing.get_args(ann)[0]
        origin = typing.get_origin(ann)
    if origin is typing.Union or origin is types.UnionType:
        return type(None) in typing.get_args(ann)
    return False


REQUIRED = object()  # 기본값 없는 필수 필드 표시


def _fmt(v: Any) -> str:
    if v is REQUIRED:
        return "(필수)"
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, BaseModel):
        return "{…}"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _default_of(cls: type[BaseModel], path: str) -> Any:
    """모델 기본값(중첩은 default_factory 인스턴스에서)."""
    cur: Any = cls
    obj: Any = None
    for i, part in enumerate(path.split(".")):
        fi = cur.model_fields[part]
        if fi.default_factory is not None:
            obj = fi.default_factory()
        elif fi.default is PydanticUndefined:
            obj = REQUIRED
        else:
            obj = fi.default
        if isinstance(obj, BaseModel):
            cur = type(obj)
        elif i < path.count("."):
            return None
    return obj


def _preset_values(stage: str, method: str, path: str) -> dict[str, Any]:
    """프리셋마다 이 필드 값(그 method 를 쓰는 프리셋만)."""
    out: dict[str, Any] = {}
    for name in R.preset_names():
        pipe = R.load_preset(name)
        block = pipe.get(stage)
        if stage == "roi":
            block = (pipe.get("placement") or {}).get("roi")
        if not isinstance(block, dict):
            continue
        key = "policy" if stage == "gtmask" else "method"
        if str(block.get(key)) != method:
            continue
        cur: Any = block
        for part in path.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur is not None:
            out[name] = cur
    return out


def field_rows(stage: str, method: str) -> list[dict[str, Any]]:
    """한 method 의 필드 행 목록(PARAMS.md·explain 공용)."""
    cls = registry.config_class(stage, method)
    rows = []
    for path, owner, local, _fi in _flatten(cls):
        h = field_help(owner, local)
        rows.append(
            {
                "path": path,
                "label": h.label if h else path,
                "unit": h.unit if h else "",
                "desc": h.desc if h else "",
                "en": h.en if h else "",
                "advanced": h.advanced if h else False,
                "default": _fmt(_default_of(cls, path)),
                "presets": _preset_values(stage, method, path),
                "help": h,
            }
        )
    return rows


def _yaml_key(stage: str, path: str) -> str:
    return f"pipeline.placement.roi.{path}" if stage == "roi" else f"pipeline.{stage}.{path}"


def explain_text(query: str) -> str:
    """질의 하나 → 여러 줄 텍스트. 모르는 질의는 ``KeyError``."""
    q = query.strip()
    if q.startswith("preset:"):
        return _explain_preset(q.split(":", 1)[1])
    if ":" in q:
        stage, method = q.split(":", 1)
        return _explain_method(stage, method)
    parts = q.split(".")
    head = parts[0]
    if head in ("inputs", "output"):
        return _explain_extra(head, parts[1:])
    if head not in STAGE_HELP:
        raise KeyError(
            f"모르는 스테이지: {head!r} (선택: {', '.join(TOP_STAGES)}, inputs, output, preset:<이름>, <stage>:<method>)"
        )
    stage = head
    rest = parts[1:]
    if stage == "placement" and rest and rest[0] == "roi":
        stage, rest = "roi", rest[1:]
    if not rest:
        return _explain_stage(stage)
    return _explain_field(stage, ".".join(rest))


def _explain_stage(stage: str) -> str:
    sh = STAGE_HELP[stage]
    lines = [
        f"{sh.label} ({sh.en}) — {sh.desc}",
        f"YAML: {_yaml_key(stage, '<field>').rsplit('.', 1)[0]}",
        "",
        "방법(method):",
    ]
    for info in registry.list_methods(stage):
        mh = method_help(stage, info.method)
        state = "" if info.usable else f"  [사용 불가 — {info.reason}]"
        lines.append(
            f"  {info.method:<16} {mh.label if mh else ''} — {mh.summary if mh else ''}{state}"
        )
        if mh and mh.when:
            lines.append(f"  {'':<16} 이럴 때: {mh.when}")
    lines.append("")
    lines.append(
        f"필드는 `anograft explain {stage}:<method>` 또는 `anograft explain {stage}.<field>`"
    )
    return "\n".join(lines)


def _explain_method(stage: str, method: str) -> str:
    if stage not in STAGE_HELP:
        raise KeyError(f"모르는 스테이지: {stage!r}")
    mh = method_help(stage, method)
    if mh is None:
        raise KeyError(
            f"모르는 method: {stage}:{method} (선택: {', '.join(registry.schema_methods(stage))})"
        )
    lines = [f"{STAGE_HELP[stage].label} › {mh.label} ({method} · {mh.en})", mh.summary]
    if mh.when:
        lines.append(f"이럴 때: {mh.when}")
    lines.append("")
    for r in field_rows(stage, method):
        adv = " [고급]" if r["advanced"] else ""
        unit = f" ({r['unit']})" if r["unit"] else ""
        lines.append(f"  {r['label']}{unit}{adv} — {r['path']} · 기본 {r['default']}")
        if r["desc"]:
            lines.append(f"      {r['desc']}")
    return "\n".join(lines)


def _explain_field(stage: str, path: str) -> str:
    found: list[tuple[str, dict[str, Any]]] = []
    for info in registry.list_methods(stage):
        for r in field_rows(stage, info.method):
            if r["path"] == path:
                found.append((info.method, r))
    if not found:
        raise KeyError(f"{stage}.{path}: 그런 필드가 없습니다 (`anograft explain {stage}` 로 목록)")
    _method, r = found[0]
    h: FieldHelp | None = r["help"]
    lines = [
        f"{r['label']}" + (f" ({r['unit']})" if r["unit"] else "") + f" — {_yaml_key(stage, path)}"
    ]
    if h:
        lines.append(h.desc)
        if h.en:
            lines.append(f"EN: {h.en}")
    methods = ", ".join(m for m, _ in found)
    lines.append(
        f"method: {methods} · 기본 {r['default']}" + (" · 고급 옵션" if r["advanced"] else "")
    )
    presets = {}
    for _m, rr in found:
        presets.update(rr["presets"])
    if presets:
        lines.append("프리셋 값: " + " · ".join(f"{k} {_fmt(v)}" for k, v in presets.items()))
    return "\n".join(lines)


def _explain_extra(head: str, rest: list[str]) -> str:
    cls = R.Inputs if head == "inputs" else R.Output
    rows = _flatten(cls)
    if not rest:
        lines = [f"{head}:"]
        for p, o, local, _fi in rows:
            h = field_help(o, local)
            lines.append(
                f"  {(h.label if h else p):<14} — {head}.{p}" + (f" · {h.desc}" if h else "")
            )
        if head == "output":
            lines.append(
                "  학습 형식(writer): "
                + ", ".join(k.split("[")[1][:-1] for k, _ in EXTRA_MODELS if "[" in k)
            )
        return "\n".join(lines)
    path = ".".join(rest)
    for p, o, local, _fi in rows:
        if p == path:
            h = field_help(o, local)
            if h is None:
                return f"{head}.{path}"
            return "\n".join(
                [f"{h.label}" + (f" ({h.unit})" if h.unit else "") + f" — {head}.{path}", h.desc]
                + ([f"EN: {h.en}"] if h.en else [])
                + [f"기본 {_fmt(_default_of(cls, path))}"]
            )
    raise KeyError(f"{head}.{path}: 그런 필드가 없습니다")


def _explain_preset(name: str) -> str:
    meta = R.preset_meta(name)
    pipe = R.load_preset(name)
    lines = [f"{name} — {meta['title']}", meta["summary"]]
    if meta["use_for"]:
        lines.append(f"이럴 때: {meta['use_for']}")
    if meta["avoid"]:
        lines.append(f"피할 때: {meta['avoid']}")
    if meta["evidence"]:
        lines.append(f"근거: {meta['evidence']}")
    lines.append("")
    for stage in TOP_STAGES:
        block = pipe.get(stage)
        if not isinstance(block, dict):
            continue
        key = "policy" if stage == "gtmask" else "method"
        m = str(block.get(key))
        mh = method_help(stage, m)
        lines.append(f"  {STAGE_HELP[stage].label:<10} {m}" + (f" — {mh.label}" if mh else ""))
        if stage == "placement" and isinstance(block.get("roi"), dict):
            rm = str(block["roi"].get("method"))
            rh = method_help("roi", rm)
            lines.append(f"  {'붙일 수 있는 영역':<10} {rm}" + (f" — {rh.label}" if rh else ""))
    return "\n".join(lines)


def markdown() -> str:
    """PARAMS.md 전체(ko + en 열)."""
    out = [
        "# PARAMS — 레시피 파라미터 도움말 / Recipe parameter reference",
        "",
        "> `anograft explain --out PARAMS.md` 가 만든다(원천 `src/anograft/core/help.py` · 프리셋 `meta:`). 손으로 고치지 말고 문안을 고친 뒤 재생성. "
        "GUI 카드 툴팁 · `anograft explain <stage>.<field>` 와 같은 글. 기본값은 스키마 기본(프리셋은 다를 수 있음 — 프리셋 열).",
        "",
        "## 프리셋 / Presets",
        "",
        "| 프리셋 | 제목 | 한 줄 | 이럴 때 | 피할 때 | 근거 |",
        "|---|---|---|---|---|---|",
    ]
    for name in R.preset_names():
        m = R.preset_meta(name)
        out.append(
            f"| `{name}` | {m['title']} | {m['summary']} | {m['use_for']} | {m['avoid']} | {m['evidence']} |"
        )
    out += [
        "",
        "## 입력 · 출력 / Inputs · Output",
        "",
        "| 키 | 라벨 | 단위 | 설명 | EN | 기본 |",
        "|---|---|---|---|---|---|",
    ]
    for head, cls in EXTRA_MODELS:
        for p, o, local, _fi in _flatten(cls):
            h = field_help(o, local)
            key = f"{head}.{p}"
            out.append(
                f"| `{key}` | {h.label if h else ''} | {h.unit if h else ''} | {h.desc if h else ''} | {h.en if h else ''} | `{_fmt(_default_of(cls, p))}` |"
            )
    for stage in (*TOP_STAGES, "roi"):
        sh = STAGE_HELP[stage]
        out += ["", f"## {sh.label} / {sh.en} — `{_yaml_key(stage, '').rstrip('.')}`", "", sh.desc]
        for info in registry.list_methods(stage):
            mh = METHOD_HELP.get((stage, info.method))
            out += [
                "",
                f"### `{info.method}` — {mh.label if mh else ''}"
                + ("" if info.usable else f" (사용 불가: {info.reason})"),
                "",
            ]
            if mh:
                out.append(
                    mh.summary
                    + (f" · **이럴 때:** {mh.when}" if mh.when else "")
                    + f" · EN: {mh.en}"
                )
                out.append("")
            rows = field_rows(stage, info.method)
            if not rows:
                out.append("(조정할 값 없음)")
                continue
            out += [
                "| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for r in rows:
                presets = " · ".join(f"{k} `{_fmt(v)}`" for k, v in r["presets"].items())
                out.append(
                    f"| `{r['path']}` | {r['label']} | {r['unit']} | {r['desc']} | {r['en']} | `{r['default']}` | {presets} | {'✓' if r['advanced'] else ''} |"
                )
    return "\n".join(out) + "\n"

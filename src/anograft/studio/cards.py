"""파이프라인 카드의 **Qt 없는** 부분 — 카드 모델·스테이지 썸네일·경고 라우팅·`per_class` 표 규칙.

"스키마가 곧 UI" 의 나머지 절반이다. `params.py` 가 필드 하나하나의 스펙을 내면, 여기서는 **카드 한 장**
(번호·제목·method 선택지·필드 목록·바뀜 수)을 만든다. Qt `gui/studio/panels.py` 와 웹 폼이 같은 것을 본다 —
스테이지 순서·제목·툴팁·경고 라우팅을 화면마다 다시 쓰면 둘이 조용히 갈린다(U5a `diagnose.py` 와 같은 이유).

한국어 문안의 한 원천은 여전히 `core/help.py` 다. 여기서는 그걸 **조립**만 한다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from anograft.core import recipe as R
from anograft.core import registry
from anograft.core.help import STAGE_HELP, method_help
from anograft.core.pipeline import TraceStep
from anograft.preview import fit_long_side
from anograft.studio.params import FieldSpec, baseline_config, field_specs

#: 카드 순서 = 사람이 보는 7단계(`roi` 는 배치 카드의 하위 블록).
STAGE_ORDER: tuple[str, ...] = (
    "source",
    "geometry",
    "placement",
    "blend",
    "harmonize",
    "degrade",
    "gtmask",
)
#: (스테이지 키, 한국어 제목) — 문안은 `core/help.py` 한 원천. 영어·YAML 키는 툴팁(`STAGE_TIPS`)으로.
STAGE_TITLES: tuple[tuple[str, str], ...] = tuple((k, STAGE_HELP[k].label) for k in STAGE_ORDER)
STAGE_TIPS: dict[str, str] = {
    k: f"{h.en} · YAML pipeline.{k} — {h.desc}" for k, h in STAGE_HELP.items() if k != "roi"
}
#: 미리보기 축소 선택지(0 = 원본 해상도).
LONG_SIDES: tuple[tuple[str, int], ...] = (
    ("긴 변 1024px", 1024),
    ("긴 변 768px", 768),
    ("긴 변 512px", 512),
    ("원본 해상도", 0),
)
#: `geometry.per_class` 표의 뒤집기 선택지. `""` = 기본(전체 설정을 따름).
FLIP_CHOICES: tuple[tuple[str, str], ...] = (
    ("기본", ""),
    ("없음", "none"),
    ("좌우", "horizontal"),
    ("상하", "vertical"),
    ("둘 다", "both"),
)
#: per_class 를 처음 켤 때의 회전 범위(조명 의존 결함 기준 — `dent-graft` 와 같은 값).
PER_CLASS_DEFAULT_ROTATE: tuple[float, float] = (-15.0, 15.0)


# ---------------------------------------------------------------- 문구


def params_text(cfg: Any) -> str:
    """설정 모델 → ``key: v · key: v`` 한 줄(method/policy 제외, 범위는 ``a–b``)."""
    d = cfg.model_dump(mode="json") if hasattr(cfg, "model_dump") else dict(cfg)
    parts: list[str] = []
    for k, v in d.items():
        if k in ("method", "policy", "roi"):
            continue
        if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            parts.append(f"{k} {v[0]:g}–{v[1]:g}")
        elif isinstance(v, dict):
            parts.append(f"{k} " + "/".join(f"{a}={b}" for a, b in v.items()))
        elif v is None:
            parts.append(f"{k} –")
        else:
            parts.append(f"{k} {v}")
    return " · ".join(parts)


def per_class_text(geo: Any) -> str:
    """``geometry.per_class`` 한 줄 요약 — 표를 못 그리는 자리(은행을 아직 모를 때)의 읽기 전용 표시."""
    per = dict(getattr(geo, "per_class", {}) or {})
    if not per:
        return ""
    parts = []
    for cls, o in per.items():
        bits = []
        if o.rotate is not None:
            bits.append(f"±{max(abs(o.rotate[0]), abs(o.rotate[1])):g}°")
        if o.flip is not None:
            bits.append(f"flip {o.flip}")
        if o.scale is not None:
            bits.append(f"크기 {o.scale[0]:g}–{o.scale[1]:g}")
        parts.append(f"{cls}({', '.join(bits) or '기본'})")
    return "클래스별 예외: " + " · ".join(parts)


def stage_warnings(stage: str, messages: Sequence[str]) -> list[str]:
    """이 스테이지 것만 고른다(자기 접두 ``<stage>:`` 는 뗀다).

    스테이지 경고는 ``<stage>: …`` 접두가 규약이고, **`roi:` 는 배치 카드가 받는다**(ROI 는 배치의 하위
    블록이라 자기 카드가 없다 — 접두를 남긴 채 보여 어디서 온 말인지 알린다).
    """
    own = f"{stage}:"
    prefixes = (own, "roi:") if stage == "placement" else (own,)
    return [
        (m[len(own) :].strip() if m.startswith(own) else m.strip())
        for m in messages
        if m.startswith(prefixes)
    ]


# ---------------------------------------------------------------- 카드 모델


@dataclass(frozen=True)
class MethodChoice:
    method: str
    label: str
    usable: bool
    reason: str
    summary: str


@dataclass(frozen=True)
class StageCardModel:
    """카드 한 장. `fields` 는 `params.FieldSpec` 그대로 — 폼은 스펙대로 그리기만 한다."""

    stage: str  # YAML 키. `roi` 는 배치 카드의 하위 블록으로 따로 들어간다
    no: int  # 1~7 (roi 는 0)
    label: str
    tip: str
    method: str
    method_label: str
    methods: tuple[MethodChoice, ...]
    fields: tuple[FieldSpec, ...]
    summary: str  # params_text 한 줄

    @property
    def modified(self) -> int:
        return sum(1 for f in self.fields if f.modified)


def method_choices(stage: str) -> tuple[MethodChoice, ...]:
    """이 스테이지에서 고를 수 있는 알고리즘 — 못 쓰는 것도 **이유와 함께** 남긴다(조용히 숨기지 않는다)."""
    out = []
    for info in registry.list_methods(stage):
        h = method_help(stage, info.method)
        out.append(
            MethodChoice(
                method=info.method,
                label=h.label if h else info.method,
                usable=info.usable,
                reason=info.reason or ("" if info.usable else "아직 구현되지 않음"),
                summary=h.summary if h else "",
            )
        )
    return tuple(out)


def stage_card(recipe: R.Recipe, stage: str, no: int = 0) -> StageCardModel:
    """레시피의 한 스테이지 → 카드 모델. ``stage="roi"`` 는 ``placement.roi`` 블록을 본다."""
    cfg = recipe.pipeline.placement.roi if stage == "roi" else getattr(recipe.pipeline, stage)
    method = registry.config_method(cfg)
    h = method_help(stage, method)
    sh = STAGE_HELP[stage]
    return StageCardModel(
        stage=stage,
        no=no,
        label=sh.label,
        tip=STAGE_TIPS.get(stage, f"{sh.en} · YAML {stage} — {sh.desc}"),
        method=method,
        method_label=h.label if h else method,
        methods=method_choices(stage),
        fields=tuple(field_specs(cfg, baseline=baseline_config(recipe, stage, cfg))),
        summary=params_text(cfg),
    )


def stage_cards(recipe: R.Recipe) -> list[StageCardModel]:
    """7단계 카드 + 배치 바로 뒤의 ``roi``(하위 블록) — 화면이 보는 순서 그대로."""
    out: list[StageCardModel] = []
    for i, stage in enumerate(STAGE_ORDER, start=1):
        out.append(stage_card(recipe, stage, i))
        if stage == "placement":
            out.append(stage_card(recipe, "roi", 0))
    return out


# ---------------------------------------------------------------- per_class 표


@dataclass(frozen=True)
class PerClassRow:
    """`geometry.per_class` 한 줄. ``on=False`` 면 그 클래스는 전체 설정을 따른다(dict 에서 빠진다)."""

    cls: str
    on: bool
    rotate: tuple[float, float]
    flip: str  # "" = 기본
    scale: tuple[float, float] | None  # 표에 없는 값 — 그대로 **보존**한다


def per_class_rows(geo: Any, classes: Sequence[str]) -> list[PerClassRow]:
    """행 = 은행 클래스 ∪ 레시피에 이미 있는 키(순서 유지). 크기(scale)는 표에 없지만 값은 들고 다닌다."""
    per = dict(getattr(geo, "per_class", {}) or {})
    out = []
    for cls in dict.fromkeys([*classes, *per]):
        o = per.get(cls)
        rotate = tuple(o.rotate) if o is not None and o.rotate is not None else None
        out.append(
            PerClassRow(
                cls=cls,
                on=o is not None,
                rotate=rotate or PER_CLASS_DEFAULT_ROTATE,  # type: ignore[arg-type]
                flip="" if o is None or o.flip is None else str(o.flip),
                scale=tuple(o.scale) if o is not None and o.scale is not None else None,  # type: ignore[arg-type]
            )
        )
    return out


def per_class_dict(rows: Sequence[PerClassRow | Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """표 → 레시피 dict. **꺼진 행은 빠진다**(= 그 클래스는 전체 설정) · `flip ""` 은 `null`(기본) ·
    표에 없는 `scale` 은 보존. Qt `gui/studio/per_class.py` 가 위젯으로 하는 것과 같은 규칙이다."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        r = row if isinstance(row, PerClassRow) else _row_of(row)
        if not r.on:
            continue
        d: dict[str, Any] = {"rotate": [float(r.rotate[0]), float(r.rotate[1])]}
        d["flip"] = r.flip or None
        if r.scale is not None:
            d["scale"] = [float(r.scale[0]), float(r.scale[1])]
        out[r.cls] = d
    return out


def _row_of(d: Mapping[str, Any]) -> PerClassRow:
    rot = d.get("rotate") or PER_CLASS_DEFAULT_ROTATE
    scale = d.get("scale")
    return PerClassRow(
        cls=str(d.get("cls", "")),
        on=bool(d.get("on", False)),
        rotate=(float(rot[0]), float(rot[1])),
        flip=str(d.get("flip") or ""),
        scale=(float(scale[0]), float(scale[1])) if scale else None,
    )


# ---------------------------------------------------------------- 스테이지 썸네일


def stage_thumbnail(stage: str, steps: Sequence[TraceStep], size: int = 132) -> np.ndarray | None:
    """스테이지의 마지막 TraceStep에서 보여 줄 이미지 하나. 결함 루프 단계는 배치 bbox 주변 크롭."""
    mine = [s for s in steps if s.stage == stage]
    if not mine:
        return None
    ctx = mine[-1].ctx
    img: np.ndarray | None
    if stage == "source":
        img = ctx.source.image if ctx.source is not None else None
    elif stage == "geometry":
        img = ctx.patch
    elif stage == "roi":
        img = (ctx.roi.astype(np.uint8) * 255) if ctx.roi is not None else None
    elif stage == "gtmask":
        img = ctx.gt_mask
    elif stage == "degrade":
        img = ctx.composite
    else:  # placement · blend · harmonize — 배치 bbox 주변 크롭
        if ctx.placement is None:
            return None
        x, y, w, h = ctx.placement.bbox
        m = 24
        hh, ww = ctx.composite.shape[:2]
        img = ctx.composite[max(0, y - m) : min(hh, y + h + m), max(0, x - m) : min(ww, x + w + m)]
        if stage == "placement" and ctx.placed_mask is not None:
            img = img.copy()
            pm = (
                ctx.placed_mask[
                    max(0, y - m) : min(hh, y + h + m), max(0, x - m) : min(ww, x + w + m)
                ]
                > 0
            )
            img[pm] = (img[pm] * 0.5 + np.array([109, 77, 255]) * 0.5).astype(np.uint8)
    if img is None or img.size == 0:
        return None
    return fit_long_side(np.ascontiguousarray(img), size)

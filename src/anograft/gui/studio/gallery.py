"""프리셋 갤러리의 **Qt 없는** 부분(v0.9 사용성 ④) — 카드 문안(프리셋 meta)과 현재 바탕 이미지로 프리셋별 썸네일 합성.

- ``preset_cards()``: 이름순 ``PresetCard``(제목·한 줄·이럴 때·피할 때·근거·단계 method 요약).
- ``render_preset_thumbs(prep, recipe, target, names, long_side)``: 현재 ``Prepared``(보관함·바탕)에서 레시피의 프리셋만 바꿔
  ``reprepare`` → 같은 바탕·같은 시드로 한 장씩. 프리셋이 그 입력으로 돌 수 없으면(보관함 없음 등) None — 갤러리는 회색 칸.
  미리보기와 같은 규칙(축소본 합성, ``image_rng(seed, 0)``)이라 스튜디오 v1 과 같은 그림이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from anograft import runner
from anograft.core import recipe as R
from anograft.core.help import METHOD_HELP, STAGE_HELP
from anograft.core.seeds import image_rng
from anograft.gui.studio.jobs import preview_target
from anograft.io.targets import load_target

THUMB_LONG_SIDE = 256


@dataclass(frozen=True)
class PresetCard:
    name: str
    title: str
    summary: str
    use_for: str
    avoid: str
    evidence: str
    stages: tuple[tuple[str, str, str], ...]  # (스테이지 라벨, method, method 라벨)


def _stage_summary(pipe: dict) -> tuple[tuple[str, str, str], ...]:
    out = []
    for stage in ("source", "geometry", "placement", "blend", "harmonize", "degrade", "gtmask"):
        block = pipe.get(stage)
        if not isinstance(block, dict):
            continue
        key = "policy" if stage == "gtmask" else "method"
        m = str(block.get(key, ""))
        mh = METHOD_HELP.get((stage, m))
        out.append((STAGE_HELP[stage].label, m, mh.label if mh else m))
        if stage == "placement" and isinstance(block.get("roi"), dict):
            rm = str(block["roi"].get("method", ""))
            rh = METHOD_HELP.get(("roi", rm))
            out.append((STAGE_HELP["roi"].label, rm, rh.label if rh else rm))
    return tuple(out)


def preset_cards() -> list[PresetCard]:
    cards = []
    for name in R.preset_names():
        m = R.preset_meta(name)
        cards.append(
            PresetCard(
                name=name,
                title=m["title"],
                summary=m["summary"],
                use_for=m["use_for"],
                avoid=m["avoid"],
                evidence=m["evidence"],
                stages=_stage_summary(R.load_preset(name)),
            )
        )
    return cards


def recipe_with_preset(recipe: R.Recipe, name: str) -> R.Recipe:
    """레시피의 ``pipeline`` 을 그 프리셋 기본값으로 통째로(스튜디오 ``set_preset`` 과 같은 규칙)."""
    d = recipe.to_dict()
    d["pipeline"] = {"preset": name}
    return R.Recipe.from_dict(d)


def render_preset_thumbs(
    prep: runner.Prepared,
    recipe: R.Recipe,
    target: Path,
    names: list[str] | None = None,
    *,
    long_side: int = THUMB_LONG_SIDE,
    index: int = 0,
) -> dict[str, np.ndarray | None]:
    """프리셋별 합성 썸네일(BGR). 돌 수 없는 프리셋은 None."""
    out: dict[str, np.ndarray | None] = {}
    full = load_target(target, recipe.inputs.um_per_px)
    small, _s = preview_target(full, long_side)
    for name in names or R.preset_names():
        try:
            rec = recipe_with_preset(recipe, name)
            p = runner.reprepare(prep, rec)
            rng = image_rng(rec.seed, index)
            result, _steps = p.pipeline.run_one_traced(small, index, rng=rng)
            out[name] = result.image if result.status == "ok" else None
        except Exception:  # fail-soft: 갤러리는 회색 칸으로
            out[name] = None
    return out

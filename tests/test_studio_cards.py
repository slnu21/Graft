"""`studio.cards` — 파이프라인 카드의 Qt 없는 부분. Qt 탭과 웹 폼이 **같은 표**를 본다.

`params_text`·`stage_thumbnail` 은 `gui/studio/panels.py` 에 있어 Qt 가 있어야만 돌던 검사였다(U5b 승격).
CI ubuntu 는 PySide6 없이 도니까 이제 거기서도 지켜진다.
"""

from __future__ import annotations

import pytest

from anograft.core.pipeline import Pipeline
from anograft.core.recipe import Recipe
from anograft.studio import cards
from anograft.studio.session import default_recipe
from tests.fixtures import disk_target, line_defect, memory_bank, pipeline_deps


def test_params_text_skips_method_and_folds_ranges():
    rec = default_recipe()
    text = cards.params_text(rec.pipeline.geometry)
    assert "scale 0.8–1.25" in text and "method" not in text and "flip both" in text
    assert cards.params_text(rec.pipeline.blend).startswith("poisson_mode normal")


def test_stage_thumbnail_for_every_stage():
    rec = default_recipe()
    bank = memory_bank([line_defect(14, 3)])
    d = rec.to_dict()
    d["pipeline"]["placement"]["margin_px"] = 4
    d["pipeline"]["placement"]["roi"]["erode_px"] = 2
    rec = Recipe.from_dict(d)
    pipe = Pipeline.from_recipe(rec, pipeline_deps(rec, bank))
    _r, steps = pipe.run_one_traced(disk_target(96), 0)
    for stage in cards.STAGE_ORDER:
        th = cards.stage_thumbnail(stage, steps, 40)
        assert th is not None and max(th.shape[:2]) <= 40, stage
    assert cards.stage_thumbnail("blend", [], 40) is None


def test_stage_warnings_route_roi_to_the_placement_card():
    ws = ["blend: 경계가 닿음", "roi: 허용 영역이 없습니다", "placement: 배치 실패", "접두 없는 말"]
    assert cards.stage_warnings("blend", ws) == ["경계가 닿음"]
    # ROI 는 배치 카드의 하위 블록이라 자기 카드가 없다 — 접두를 남긴 채 배치 카드로 간다
    assert cards.stage_warnings("placement", ws) == ["roi: 허용 영역이 없습니다", "배치 실패"]
    assert cards.stage_warnings("degrade", ws) == []


def test_stage_cards_cover_seven_stages_plus_roi():
    rec = default_recipe()
    models = cards.stage_cards(rec)
    keys = [m.stage for m in models]
    assert keys == [
        "source",
        "geometry",
        "placement",
        "roi",  # 배치 바로 뒤 — 하위 블록
        "blend",
        "harmonize",
        "degrade",
        "gtmask",
    ]
    by = {m.stage: m for m in models}
    assert by["source"].label == "결함 고르기" and by["source"].no == 1
    assert by["roi"].no == 0
    assert by["blend"].method == "poisson" and by["blend"].method_label
    # 못 쓰는 method 도 이유와 함께 남긴다(조용히 숨기지 않는다)
    assert all(c.reason for c in by["blend"].methods if not c.usable)
    assert {c.method for c in by["blend"].methods} >= {"paste", "alpha", "poisson", "multiband"}
    assert any(f.name == "feather_px" for f in by["blend"].fields)
    assert all(f.name not in ("method", "policy", "roi") for m in models for f in m.fields)


def test_card_counts_changed_fields_against_the_preset():
    rec = default_recipe()
    assert cards.stage_card(rec, "blend").modified == 0
    d = rec.to_dict()
    d["pipeline"]["blend"]["feather_px"] = 9
    changed = cards.stage_card(Recipe.from_dict(d), "blend")
    assert changed.modified == 1
    assert [f.name for f in changed.fields if f.modified] == ["feather_px"]


# ---------------------------------------------------------------- per_class 표


def _geo(per: dict) -> object:
    d = default_recipe().to_dict()
    d["pipeline"]["geometry"]["per_class"] = per
    return Recipe.from_dict(d).pipeline.geometry


def test_per_class_rows_are_bank_classes_plus_existing_keys():
    geo = _geo({"dent": {"rotate": [-15, 15], "flip": "none"}})
    rows = cards.per_class_rows(geo, ["scratch", "dent"])
    assert [r.cls for r in rows] == ["scratch", "dent"]
    assert rows[0].on is False and rows[0].rotate == cards.PER_CLASS_DEFAULT_ROTATE
    assert rows[1].on is True and rows[1].flip == "none"
    # 레시피에만 있는 클래스도 행으로 남는다(은행이 아직 없을 때)
    assert [r.cls for r in cards.per_class_rows(geo, [])] == ["dent"]


def test_per_class_dict_drops_off_rows_and_keeps_scale():
    geo = _geo({"dent": {"rotate": [-15, 15], "flip": "none", "scale": [0.9, 1.1]}})
    rows = cards.per_class_rows(geo, ["scratch", "dent"])
    out = cards.per_class_dict(rows)
    assert set(out) == {"dent"}, "꺼진 행은 빠진다 = 그 클래스는 전체 설정을 따른다"
    assert out["dent"]["scale"] == [0.9, 1.1], "표에 없는 값은 보존한다"
    assert out["dent"]["flip"] == "none"
    # "기본"(빈 문자열)은 null 로 — 전체 설정을 따른다
    rows[1] = cards.PerClassRow("dent", True, (-30.0, 30.0), "", None)
    out = cards.per_class_dict(rows)
    assert out["dent"] == {"rotate": [-30.0, 30.0], "flip": None}


def test_per_class_dict_takes_plain_dicts_too():
    """웹은 JSON 으로 행을 보낸다 — 같은 규칙을 타야 한다."""
    out = cards.per_class_dict(
        [{"cls": "dent", "on": True, "rotate": [-5, 5], "flip": "horizontal"}]
    )
    assert out == {"dent": {"rotate": [-5.0, 5.0], "flip": "horizontal"}}
    assert cards.per_class_dict([{"cls": "dent", "on": False}]) == {}


def test_per_class_dict_result_validates_into_the_recipe():
    """표가 낸 dict 가 곧 레시피 값이어야 한다(재검증 실패면 화면이 값을 못 돌려준다)."""
    rows = [cards.PerClassRow("dent", True, (-15.0, 15.0), "none", (0.9, 1.1))]
    geo = _geo(cards.per_class_dict(rows))
    assert geo.per_class["dent"].flip == "none"
    assert geo.per_class["dent"].rotate == pytest.approx((-15.0, 15.0))

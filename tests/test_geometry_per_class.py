"""``geometry.per_class`` (0.7.3) — 클래스별 기하 오버라이드: 스크래치는 ±180, 찍힘은 ±15·flip 끔을 레시피 하나로.
스키마(부분 오버라이드·검증) · 스테이지가 클래스에 맞는 범위를 쓰고 로그에 표시 · 오버라이드 없는 클래스는 종전과 바이트 동일 ·
lighting_warning/patch_sides 가 오버라이드를 반영 · 은행에 없는 클래스는 경고."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from anograft import runner
from anograft.bank import Bank
from anograft.core import recipe as R
from anograft.core.stages.geometry import AffineGeometry
from tests.fixtures import context, line_defect
from tests.test_lighting_warning import _dent, _flat, _recipe


def _cfg(**kw) -> R.AffineGeometryConfig:
    return R.AffineGeometryConfig(**kw)


def test_schema_partial_override_and_validation() -> None:
    g = _cfg(per_class={"pit": {"rotate": [-15, 15], "flip": False}})
    e = g.for_class("pit")
    assert e.rotate == (-15.0, 15.0) and e.flip is False and e.scale == (0.8, 1.25) and e.overridden
    d = g.for_class("scratch")
    assert d.rotate == (-180.0, 180.0) and d.flip is True and not d.overridden
    assert not g.for_class(None).overridden
    with pytest.raises(ValidationError):
        _cfg(per_class={"pit": {"rotate": [-400, 0]}})
    with pytest.raises(ValidationError):
        _cfg(per_class={"pit": {"scale": [0, 1]}})
    with pytest.raises(ValidationError):
        _cfg(per_class={"pit": {"rotat": [-1, 1]}})  # 오타는 strict 가 막는다
    # 레시피 왕복
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": "t",
            "seed": 1,
            "inputs": {"bank": "b", "targets": "n"},
            "output": {"root": "o", "count": 1},
            "pipeline": {
                "preset": "poisson-graft",
                "geometry": {
                    "method": "affine",
                    "per_class": {"pit": {"rotate": [-15, 15], "flip": False}},
                },
            },
        }
    )
    again = R.Recipe.from_yaml(rec.to_yaml())
    assert again.pipeline.geometry.per_class["pit"].rotate == (-15.0, 15.0)
    assert again.pipeline.geometry.per_class["pit"].scale is None


def test_stage_uses_class_override_and_keeps_others_identical() -> None:
    base = AffineGeometry(_cfg(), {})
    over = AffineGeometry(_cfg(per_class={"dent": {"rotate": [-15, 15], "flip": False}}), {})
    scratch, dent = line_defect(18, 4), line_defect(10, 6, cls="dent")
    # 오버라이드 없는 클래스: 바이트 동일
    a, b = base.apply(context(source=scratch, seed=3)), over.apply(context(source=scratch, seed=3))
    assert (
        np.array_equal(a.patch, b.patch)
        and a.log["geometry"]["rotate"] == b.log["geometry"]["rotate"]
    )
    assert a.log["geometry"]["per_class"] is False and b.log["geometry"]["per_class"] is False
    # 오버라이드 클래스: 회전 ±15 · flip 없음 · 로그 표시
    for seed in range(8):
        out = over.apply(context(source=dent, seed=seed))
        g = out.log["geometry"]
        assert (
            -15.0 <= g["rotate"] <= 15.0 and g["flip"] == [False, False] and g["per_class"] is True
        )
    # 같은 시드 → 같은 결과(결정성)
    x, y = over.apply(context(source=dent, seed=4)), over.apply(context(source=dent, seed=4))
    assert np.array_equal(x.patch, y.patch)


def test_lighting_warning_and_patch_sides_respect_override() -> None:
    bank = Bank.from_sources(
        [_dent("down", k=i) for i in range(3)] + [_flat(k=i) for i in range(3)],
        classes=["pit", "stain"],
    )
    rec = _recipe("poisson-graft")
    assert runner.lighting_warning(rec, bank)  # ±180 + flip → 경고
    d = rec.to_dict()
    d["pipeline"]["geometry"]["per_class"] = {"pit": {"rotate": [-15, 15], "flip": False}}
    fixed = R.Recipe.from_dict(d)
    assert (
        runner.lighting_warning(fixed, bank) is None
    )  # pit 만 좁히면 경고 없음(stain 은 방향 없음)
    d["pipeline"]["geometry"]["per_class"] = {
        "pit": {"rotate": [-15, 15]}
    }  # flip 은 그대로 → 여전히 경고
    w = runner.lighting_warning(R.Recipe.from_dict(d), bank)
    assert w and "flip" in w and "rotate" not in w.split("로 합성하면")[0].split(" 을 ")[1]
    assert "per_class" in w
    # 은행에 없는 클래스는 validate_against 경고
    d["pipeline"]["geometry"]["per_class"] = {"nope": {"flip": False}}
    warns = R.Recipe.from_dict(d).validate_against(bank)
    assert any("per_class" in x and "nope" in x for x in warns)
    # patch_sides 는 클래스별 scale 상한
    d["pipeline"]["geometry"]["per_class"] = {"pit": {"scale": [0.5, 0.5]}}
    d["pipeline"]["source"] = {"method": "bank", "min_sources_warn": 1}
    prep = runner.Prepared(R.Recipe.from_dict(d), bank, [], None, "h", {"class_probs": {}}, [])  # type: ignore[arg-type]
    sides = runner.patch_sides(prep)
    assert sides["pit"] == pytest.approx(sides["stain"] * 0.5 / 1.25 * (12 / 10), rel=0.2)

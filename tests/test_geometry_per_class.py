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
    assert (
        e.rotate == (-15.0, 15.0) and e.flip == "none" and e.scale == (0.8, 1.25) and e.overridden
    )
    d = g.for_class("scratch")
    assert d.rotate == (-180.0, 180.0) and d.flip == "both" and not d.overridden
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
    assert w and "flip both" in w and "rotate" not in w.split("로 합성하면")[0].split(" 을 ")[1]
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


def test_recipe_init_dent_class_and_auto_dent(tmp_path, capsys) -> None:
    """``recipe init --dent-class pit`` 은 per_class 에 ±15·flip 끔, ``--auto-dent`` 는 은행 lightR ≥ 0.5 클래스를 찾는다."""
    import yaml

    from anograft.bank.importers.common import BankWriter, ImportOptions, ImportRecord
    from anograft.cli import EXIT_OK, auto_dent_classes, main

    d = R.init_recipe_dict("poisson-graft", dent_classes=["pit", "dent"])
    assert d["pipeline"]["geometry"]["per_class"]["pit"] == {
        "scale": None,
        "rotate": [-15.0, 15.0],
        "flip": "none",
    }
    assert set(d["pipeline"]["geometry"]["per_class"]) == {"pit", "dent"}
    assert R.init_recipe_dict("poisson-graft")["pipeline"]["geometry"]["per_class"] == {}
    # 은행: pit 3(아래 림) + stain 3(균일)
    w = BankWriter(tmp_path / "b", name="b")
    w.ensure_classes(["pit", "stain"])
    for s in [_dent("down", k=i) for i in range(3)] + [_flat(k=i) for i in range(3)]:
        rec = ImportRecord(
            image=s.image, gray=False, mask=s.mask, cls=s.cls, origin="t", id_hint=s.id[-3:]
        )
        w.add(rec, ImportOptions(margin=8))
    w.finish({"importer": "test"})
    assert (
        auto_dent_classes(tmp_path / "b") == ["pit"] and auto_dent_classes(tmp_path / "nope") == []
    )
    target = tmp_path / "r.yaml"
    assert (
        main(
            [
                "recipe",
                "init",
                "--bank",
                str(tmp_path / "b"),
                "--auto-dent",
                "--dent-class",
                "stain",
                "--write",
                str(target),
            ]
        )
        == EXIT_OK
    )
    cap = capsys.readouterr()
    assert "['pit']" in cap.err
    text = target.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert set(data["pipeline"]["geometry"]["per_class"]) == {"pit", "stain"}
    assert "--dent-class stain pit" in text.splitlines()[0]
    # 은행 없으면 안내만
    assert main(["recipe", "init", "--bank", str(tmp_path / "nope"), "--auto-dent"]) == EXIT_OK
    assert "찾지 못했습니다" in capsys.readouterr().err


def test_flip_modes_compat_rng_and_lighting() -> None:
    """0.7.5+ ``flip``: none/horizontal/vertical/both — YAML true/false 호환(both/none), rng 소비(both 2·h/v 1·none 0),
    조명 경고는 방향에 따라(위/아래 조명이면 horizontal 안전)."""
    from anograft.core.appearance import flip_breaks_lighting
    from anograft.core.stages.geometry import AffineGeometry
    from tests.fixtures import context, line_defect

    assert _cfg(flip=True).flip == "both" and _cfg(flip=False).flip == "none"
    assert _cfg(flip="horizontal").flip == "horizontal"
    with pytest.raises(ValidationError):
        _cfg(flip="sideways")
    assert (
        R.GeometryOverride(flip=True).flip == "both"
        and R.GeometryOverride(flip="vertical").flip == "vertical"
    )
    e = _cfg(flip="horizontal").for_class(None)
    assert e.flip_h and not e.flip_v
    # rng 소비: 같은 시드에서 both 는 h·v 두 번, horizontal 은 h 한 번 → 뒤따르는 난수가 달라진다(결정성은 유지)
    src = line_defect(18, 4)
    outs = {
        m: AffineGeometry(_cfg(rotate=(0.0, 0.0), scale=(1.0, 1.0), flip=m), {}).apply(
            context(source=src, seed=9)
        )
        for m in ("none", "horizontal", "vertical", "both")
    }
    assert outs["none"].log["geometry"]["flip"] == [False, False]
    assert outs["horizontal"].log["geometry"]["flip"][1] is False
    assert outs["vertical"].log["geometry"]["flip"][0] is False
    for m in ("none", "horizontal", "vertical", "both"):
        again = AffineGeometry(_cfg(rotate=(0.0, 0.0), scale=(1.0, 1.0), flip=m), {}).apply(
            context(source=src, seed=9)
        )
        assert again.log["geometry"]["flip"] == outs[m].log["geometry"]["flip"]
    # 조명 방향 vs flip
    assert not flip_breaks_lighting("none", 90.0) and flip_breaks_lighting("both", 90.0)
    assert not flip_breaks_lighting("horizontal", 90.0) and flip_breaks_lighting(
        "vertical", 90.0
    )  # 아래 조명
    assert flip_breaks_lighting("horizontal", 0.0) and not flip_breaks_lighting(
        "vertical", 0.0
    )  # 옆 조명
    assert flip_breaks_lighting("horizontal", None) and flip_breaks_lighting("vertical", None)
    # 경고: 아래 조명 pit + horizontal → 없음, vertical → 있음
    bank = Bank.from_sources([_dent("down", k=i) for i in range(3)], classes=["pit"])
    assert (
        runner.lighting_warning(_recipe("poisson-graft", rotate=[-15, 15], flip="horizontal"), bank)
        is None
    )
    w = runner.lighting_warning(_recipe("poisson-graft", rotate=[-15, 15], flip="vertical"), bank)
    assert w and "flip vertical" in w
    assert R.DENT_OVERRIDE["flip"] == "none"

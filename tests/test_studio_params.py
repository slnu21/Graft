"""스테이지 파라미터 편집기(v0.6 stage-params) — Qt 없는 스펙 층(`params.py`)과 세션 점 경로. 위젯은 ``test_gui_smoke``."""

from __future__ import annotations

import pytest

from anograft.core import recipe as R
from anograft.core import registry
from anograft.core.registry import STAGE_ORDER, schema_methods
from anograft.studio.params import (
    EXCLUDE,
    ON_DEFAULTS,
    FieldSpec,
    coerce,
    field_specs,
    required_placeholders,
    spin_bounds,
    spin_step,
)
from anograft.studio.session import SessionError, StudioSession, default_recipe


def _by_name(specs: list[FieldSpec]) -> dict[str, FieldSpec]:
    return {s.name: s for s in specs}


def test_field_specs_cover_every_kind_from_schema() -> None:
    geo = _by_name(field_specs(R.AffineGeometryConfig()))
    assert list(geo) == [
        "scale",
        "rotate",
        "flip",
        "elastic.alpha",
        "elastic.sigma",
        "tps.points",
        "tps.jitter",
    ]  # 선언 순서 · 중첩 평탄화 (0.8.x: tps 휘어짐)
    assert geo["scale"].kind == "range" and geo["scale"].value == [0.8, 1.25]
    assert (
        geo["flip"].kind == "choice" and geo["flip"].value == "both"
    )  # 0.7.5+: none/horizontal/vertical/both
    assert set(geo["flip"].choices) == {"none", "horizontal", "vertical", "both"}
    per = _by_name(field_specs(R.PerlinSourceConfig()))
    assert (
        per["augment"].kind == "bool" and per["augment"].value is True
    )  # bool 종류는 여전히 체크박스
    assert geo["elastic.alpha"].kind == "float" and geo["elastic.alpha"].lo == 0.0
    assert (
        geo["elastic.sigma"].lo == 0.0 and geo["elastic.sigma"].lo_open is True
    )  # gt=0 → 열린 구간
    assert "(0, ∞]" in geo["elastic.sigma"].hint

    cam = _by_name(field_specs(R.CameraDegradeConfig()))
    assert cam["jpeg_quality"].kind == "int_range" and cam["jpeg_quality"].optional
    assert (
        cam["jpeg_quality"].value is None
        and cam["jpeg_quality"].on_default == ON_DEFAULTS["jpeg_quality"]
    )
    assert cam["gamma"].kind == "range" and cam["gamma"].enabled is False
    assert cam["noise_sigma"].enabled is True and "null = 끔" in cam["gamma"].hint

    sa = _by_name(field_specs(R.StructureAwarePlacementConfig()))
    assert sa["prefer"].kind == "choice" and sa["prefer"].choices == ("edges", "flat", "uniform")
    assert sa["min_coherence"].lo == 0.0 and sa["min_coherence"].hi == 1.0
    assert sa["shrink_on_fail.factor"].lo_open and sa["shrink_on_fail.factor"].hi_open  # gt=0, lt=1
    assert "roi" not in sa and "method" not in sa  # 콤보·하위 스테이지 몫

    src = _by_name(field_specs(R.BankSourceConfig()))
    assert (
        src["classes"].kind == "list"
        and src["classes"].optional
        and src["classes"].on_default == []
    )
    per = _by_name(field_specs(R.PerlinSourceConfig()))
    assert per["texture_dir"].kind == "path" and per["texture_dir"].optional
    assert per["cls"].kind == "text" and per["scale_range"].kind == "int_range"
    assert (
        field_specs(R.MaskDirRoiConfig(path="roi/masks"))[0].value == "roi/masks"
    )  # Path → posix 문자열
    gt = _by_name(field_specs(R.GtMaskConfig()))
    assert (
        set(gt) == {"diff_threshold", "dilate_px"} and gt["diff_threshold"].hi == 255.0
    )  # policy 는 콤보
    assert field_specs(R.PasteBlendConfig()) == [] and field_specs(R.NoneRoiConfig()) == []


@pytest.mark.parametrize("stage", [s for s in STAGE_ORDER])
def test_every_schema_method_yields_specs_without_error(stage: str) -> None:
    """스키마가 곧 UI — 모든 (stage, method) 설정 모델이 스펙으로 변환된다(새 method 가 GUI 를 깨뜨리지 않게 감시)."""
    for method in schema_methods(stage):
        ses = StudioSession(default_recipe())
        ses.set_method(stage, method)  # 필수 필드(mask_dir.path)는 자리표시로 채워진다
        rec = ses.recipe
        cfg = rec.pipeline.placement.roi if stage == "roi" else getattr(rec.pipeline, stage)
        assert registry.config_method(cfg) == method and registry.config_class(
            stage, method
        ) is type(cfg)
        specs = field_specs(cfg)
        if method == "mask_dir":
            assert [s.name for s in specs] == ["path"] and specs[0].value == "."
        assert all(s.name not in EXCLUDE for s in specs)
        for s in specs:
            assert s.kind in (
                "int",
                "float",
                "int_range",
                "range",
                "bool",
                "choice",
                "text",
                "list",
                "path",
            )
            assert s.hint


def test_coerce_and_spin_helpers() -> None:
    lst = FieldSpec("classes", "list", None, optional=True)
    assert coerce(lst, " a, b ,,c ") == ["a", "b", "c"] and coerce(lst, None) is None
    rng = FieldSpec("scale", "range", [0.8, 1.25])
    assert coerce(rng, (1, 2)) == [1.0, 2.0]
    irng = FieldSpec("jpeg", "int_range", None, optional=True)
    assert coerce(irng, (60.0, 95.0)) == [60, 95]
    pth = FieldSpec("path", "path", "")
    assert coerce(pth, "a\\b") == "a/b" and coerce(pth, "  ") == ""
    unit = FieldSpec("strength", "float", 0.5, lo=0.0, hi=1.0)
    assert spin_bounds(unit) == (0.0, 1.0) and spin_step(unit) == 0.05
    free = FieldSpec("smooth_px", "float", 3.0, lo=0.0)
    assert spin_bounds(free)[1] > 1e8 and spin_step(free) == 0.1
    assert spin_step(FieldSpec("n", "int", 1)) == 1.0


def test_session_set_stage_field_accepts_dotted_paths_and_roi() -> None:
    ses = StudioSession(default_recipe())
    ses.set_stage_field("geometry", "elastic.alpha", 2.5)
    assert ses.recipe.pipeline.geometry.elastic.alpha == 2.5
    ses.set_stage_field("placement", "shrink_on_fail.rounds", 1)
    assert ses.recipe.pipeline.placement.shrink_on_fail.rounds == 1
    ses.set_stage_field("roi", "erode_px", 20)
    assert ses.recipe.pipeline.placement.roi.erode_px == 20
    ses.set_stage_field("degrade", "gamma", [0.8, 1.25])
    assert ses.recipe.pipeline.degrade.gamma == (0.8, 1.25)
    ses.set_stage_field("degrade", "gamma", None)
    assert ses.recipe.pipeline.degrade.gamma is None
    with pytest.raises(SessionError, match="rotate"):
        ses.set_stage_field("geometry", "rotate", [-400.0, 0.0])
    assert ses.recipe.pipeline.geometry.rotate == (-180.0, 180.0)  # 실패하면 그대로
    # 폼이 만든 값은 레시피 저장(YAML)에도 그대로 반영된다
    assert "alpha: 2.5" in ses.recipe.to_yaml()


def test_required_placeholders_only_for_required_fields() -> None:
    assert required_placeholders(R.MaskDirRoiConfig) == {"path": "."}
    assert (
        required_placeholders(R.OtsuRoiConfig) == {}
        and required_placeholders(R.CameraDegradeConfig) == {}
    )

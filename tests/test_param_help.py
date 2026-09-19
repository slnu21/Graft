"""v0.9 사용성 ② — 파라미터 도움말 한 원천(core/help.py)의 **완전성**과 소비자(FieldSpec · explain · PARAMS.md · CLI).
Qt 없음. 문안이 빠지면 여기서 막힌다."""

from __future__ import annotations

from pathlib import Path

import pytest

from anograft.cli import EXIT_OK, main
from anograft.core import explain as E
from anograft.core import recipe as R
from anograft.core import registry
from anograft.core.help import (
    FIELD_HELP,
    METHOD_HELP,
    STAGE_HELP,
    FieldHelp,
    field_help,
    method_help,
)
from anograft.gui.studio.params import field_specs

ROOT = Path(__file__).resolve().parents[1]


def _all_fields() -> list[tuple[str, type, str]]:
    """(표시용 키, 필드를 정의한 모델, 로컬 이름) — 스테이지 method 전부 + inputs/output/writer."""
    out: list[tuple[str, type, str]] = []
    for info in registry.list_methods():
        cls = registry.config_class(info.stage, info.method)
        for path, owner, local, _fi in E._flatten(cls):
            out.append((f"{info.stage}:{info.method}.{path}", owner, local))
    for head, cls in E.EXTRA_MODELS:
        for path, owner, local, _fi in E._flatten(cls):
            out.append((f"{head}.{path}", owner, local))
    return out


def test_every_field_has_help() -> None:
    missing = [k for k, owner, local in _all_fields() if field_help(owner, local) is None]
    assert not missing, f"도움말 없는 필드 {len(missing)}: {missing[:10]}"


def test_help_text_rules() -> None:
    """라벨은 한국어(YAML 키 아님) · 설명 1~200자 · 영어 한 줄 · 단위는 짧게."""
    for key, h in FIELD_HELP.items():
        assert h.label and "_" not in h.label and h.label != key.split(".")[-1], key
        assert 1 <= len(h.desc) <= 200, (key, len(h.desc))
        assert h.en, key
        assert len(h.unit) <= 6, key
    for (stage, method), mh in METHOD_HELP.items():
        assert mh.label and mh.summary and mh.en, (stage, method)
        assert len(mh.summary) <= 120, (stage, method)


def test_every_method_and_stage_has_help() -> None:
    for info in registry.list_methods():
        assert method_help(info.stage, info.method) is not None, (info.stage, info.method)
    for stage in registry.STAGE_ORDER:
        assert stage in STAGE_HELP, stage
    # 반대로 도움말에만 있는 유령 method 도 없어야 한다
    real = {(i.stage, i.method) for i in registry.list_methods()}
    assert set(METHOD_HELP) <= real, set(METHOD_HELP) - real


def test_every_preset_has_meta() -> None:
    for name in R.preset_names():
        m = R.preset_meta(name)
        assert m["title"] and m["summary"] and m["use_for"], name
        assert set(m) == set(R.PRESET_META_KEYS)
    with pytest.raises(KeyError):
        R.preset_meta("nope")
    # meta 는 pipeline 에 섞이지 않는다(스키마·해시 불변)
    assert "meta" not in R.load_preset("poisson-graft")


def test_field_help_inheritance_and_nested() -> None:
    assert field_help(R.SampledPlacementConfig, "margin_px") is not None  # _PlacementBase 에서
    assert field_help(R.StructureAwarePlacementConfig, "max_tries") is not None
    assert field_help(R.ColorJitterConfig, "brightness").label == "밝기 흔들기"
    assert field_help(R.AffineGeometryConfig, "nope") is None
    assert isinstance(FIELD_HELP["AffineGeometryConfig.scale"], FieldHelp)


def test_field_specs_carry_help() -> None:
    specs = {s.name: s for s in field_specs(R.AffineGeometryConfig())}
    sc = specs["scale"]
    assert sc.label == "크기 배율" and sc.unit == "배" and not sc.advanced
    assert sc.title == "크기 배율 (배)"
    assert sc.tooltip.startswith("크기 배율 · scale\n") and "Scale range" in sc.tooltip
    assert "실수 범위" in sc.tooltip  # 형식·범위 줄은 그대로
    assert specs["elastic.alpha"].advanced and specs["elastic.alpha"].label == "잔물결 세기"
    # 도움말이 없는(가상) 필드는 이름을 그대로 보인다
    from anograft.gui.studio.params import FieldSpec

    bare = FieldSpec(name="x.y", kind="int", value=1)
    assert bare.title == "x.y" and bare.tooltip == "x.y"


def test_explain_text_queries() -> None:
    t = E.explain_text("geometry.scale")
    assert t.startswith("크기 배율 (배) — pipeline.geometry.scale") and "dent-graft [0.9, 1.1]" in t
    assert "경계 깎기" in E.explain_text("placement.roi.erode_px")
    assert "annulus-graft 4" in E.explain_text("placement.roi.erode_px")
    m = E.explain_text("blend:poisson")
    assert "붙이기 › 경계 자연스럽게(Poisson)" in m and "[고급]" in m and "poisson_mode" in m
    p = E.explain_text("preset:relative-paste")
    assert "대비 지키며 붙이기" in p and "relative — 노출만 맞춤" in p and "이럴 때:" in p
    assert "(필수)" in E.explain_text("output.count")
    assert "학습 형식(writer): yolo, pairs, mvtec, coco" in E.explain_text("output")
    assert "바탕 픽셀 크기" in E.explain_text("inputs.um_per_px")
    s = E.explain_text("harmonize")
    assert "방법(method):" in s and "relative" in s and "노출만 맞춤" in s
    for bad in ("bogus", "geometry.nope", "blend:nope", "output.nope"):
        with pytest.raises(KeyError):
            E.explain_text(bad)


def test_params_md_is_up_to_date() -> None:
    """PARAMS.md = `anograft explain --markdown`. 문안을 고쳤으면 재생성해 커밋."""
    md = E.markdown()
    assert md.startswith("# PARAMS") and "## 프리셋" in md and "### `poisson`" in md
    stored = (ROOT / "PARAMS.md").read_text(encoding="utf-8")
    assert stored == md, "PARAMS.md 가 오래됐습니다: anograft explain --markdown > PARAMS.md"


def test_cli_explain(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["explain"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "스테이지:" in out and "geometry" in out
    assert main(["explain", "geometry.rotate", "preset:dent-graft"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "회전 (°)" in out and "찍힘·덴트" in out
    assert main(["explain", "--markdown"]) == EXIT_OK
    assert capsys.readouterr().out.startswith("# PARAMS")
    assert main(["explain", "bogus"]) != EXIT_OK
    assert "모르는 스테이지" in capsys.readouterr().err
    assert main(["methods", "--stage", "harmonize"]) == EXIT_OK
    assert "노출만 맞춤" in capsys.readouterr().out

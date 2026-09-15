"""v0.6 ``source-tags-scale``(KNOWN-ISSUES #7 #6) — ``source.tags`` 필터(스키마·검증·스테이지·dry-run) ·
µm/px 미지정 표면화(``bank ls`` no_um 열 · ``runner.scale_warning``)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from anograft import runner
from anograft.bank import Bank
from anograft.cli import EXIT_OK, main
from anograft.core import recipe as R
from anograft.core.stages.source import BankSource
from anograft.core.types import DefectSource
from tests.fixtures import context, fake_yolo_dataset, line_defect


def _src(
    n: int, cls: str = "scratch", tags: tuple[str, ...] = (), um: float | None = None
) -> DefectSource:
    s = line_defect(12 + n, 3, cls=cls, um_per_px=um)
    return DefectSource(f"{cls}/{n:03d}", cls, s.image, s.mask, um, tags, s.origin, s.mask_origin)


def _bank() -> Bank:
    return Bank.from_sources(
        [
            _src(0, tags=("A",)),
            _src(1, tags=("B",)),
            _src(2, tags=("A", "old")),
            _src(3),
            _src(0, cls="dent", tags=("B",)),
        ],
        classes=["scratch", "dent"],
    )


# --- TagFilter 의미 ---------------------------------------------------------


def test_tag_filter_include_any_exclude_any_and_inactive_default() -> None:
    f = R.TagFilter()
    assert not f.active and f.accepts(()) and f.accepts(("x",)) and f.describe() == "(없음)"
    inc = R.TagFilter(include=["A", "C"])
    assert (
        inc.active and inc.accepts(("A",)) and inc.accepts(("C", "z")) and not inc.accepts(("B",))
    )
    assert not inc.accepts(())  # include 가 있으면 태그 없는 소스는 탈락
    exc = R.TagFilter(exclude=["old"])
    assert exc.accepts(()) and exc.accepts(("A",)) and not exc.accepts(("A", "old"))
    both = R.TagFilter(include=["A"], exclude=["old"])
    assert both.accepts(("A",)) and not both.accepts(("A", "old"))
    assert both.describe() == "include=['A'] exclude=['old']"


def test_recipe_tags_null_means_no_filter_and_unknown_key_rejected() -> None:
    base = {
        "version": 1,
        "name": "t",
        "seed": 1,
        "inputs": {"bank": "b", "targets": "n"},
        "output": {"root": "o", "count": 1},
        "pipeline": {"preset": "hard-paste", "source": {"method": "bank", "tags": None}},
    }
    rec = R.Recipe.from_dict(base)
    assert not rec.pipeline.source.tags.active
    base["pipeline"]["source"]["tags"] = {"include": ["A"]}
    assert R.Recipe.from_dict(base).pipeline.source.tags.include == ["A"]
    # 왕복: YAML 에 그대로 남는다
    d = R.Recipe.from_dict(base).to_dict()
    assert d["pipeline"]["source"]["tags"] == {"include": ["A"], "exclude": []}
    base["pipeline"]["source"]["tags"] = {"includes": ["A"]}
    with pytest.raises(ValidationError):
        R.Recipe.from_dict(base)


# --- validate_against ---------------------------------------------------------


def _recipe(tags: dict, classes: list[str] | None = None) -> R.Recipe:
    src: dict = {"method": "bank", "tags": tags, "min_sources_warn": 2}
    if classes is not None:
        src["classes"] = classes
    return R.Recipe.from_dict(
        {
            "version": 1,
            "name": "t",
            "seed": 1,
            "inputs": {"bank": "b", "targets": "n"},
            "output": {"root": "o", "count": 1},
            "pipeline": {"preset": "hard-paste", "source": src},
        }
    )


def test_validate_against_counts_after_filter() -> None:
    bank = _bank()
    # A: scratch 2개(000·002) · dent 0개 → dent 는 경고(건너뜀), scratch 는 2 → 경고 없음
    w = _recipe({"include": ["A"]}).validate_against(bank)
    assert any("dent" in x and "태그 필터" in x and "0개" in x for x in w)
    assert not any("'scratch'" in x for x in w)
    # exclude 로 scratch 1개만 남으면 min_sources_warn 경고
    w = _recipe({"exclude": ["A", "B"]}).validate_against(bank)
    assert any("'scratch' 소스가 1개" in x for x in w)
    # 전부 0 → 치명
    with pytest.raises(ValueError, match="태그 필터"):
        _recipe({"include": ["nope"]}).validate_against(bank)
    # 필터 없음 → 종전과 같은 경로(태그 언급 없음)
    assert all("태그" not in x for x in _recipe({}).validate_against(bank))


# --- BankSource 스테이지 -------------------------------------------------------


def test_stage_draws_only_from_filtered_pool_and_keeps_order() -> None:
    bank = _bank()
    cfg = R.BankSourceConfig(tags={"include": ["A"]}, classes=["scratch"])
    st = BankSource(cfg, {"bank": bank, "class_ids": bank.class_ids})
    assert st.pool is not None and [s.id for s in st.pool["scratch"]] == [
        "scratch/000",
        "scratch/002",
    ]
    ctx = context(seed=4)
    seen = set()
    for _ in range(30):
        ctx = st.apply(ctx.begin_defect())
        assert ctx.source is not None and "A" in ctx.source.tags
        assert ctx.log["source"]["tags"] == list(ctx.source.tags)
        seen.add(ctx.source.id)
    assert seen == {"scratch/000", "scratch/002"}


def test_stage_skips_with_filter_reason_when_pool_empty() -> None:
    bank = _bank()
    cfg = R.BankSourceConfig(tags={"include": ["A"]}, classes=["dent"])
    st = BankSource(cfg, {"bank": bank, "class_ids": bank.class_ids})
    out = st.apply(context(seed=0))
    reason = out.log["source"]["reason"]
    assert out.source is None and "태그 필터" in reason and "원래 1개" in reason
    assert out.warnings[0].startswith("source:")


def test_stage_without_filter_has_no_pool_and_same_draws_as_before() -> None:
    bank = _bank()
    st = BankSource(R.BankSourceConfig(), {"bank": bank})
    assert st.pool is None
    a = [st.apply(context(seed=s)).source.id for s in range(6)]  # type: ignore[union-attr]
    st2 = BankSource(R.BankSourceConfig(tags=None), {"bank": bank})
    b = [st2.apply(context(seed=s)).source.id for s in range(6)]  # type: ignore[union-attr]
    assert a == b


# --- 은행 요약 · µm/px ---------------------------------------------------------


def test_bank_summary_counts_no_pitch_and_tags() -> None:
    bank = Bank.from_sources(
        [_src(0, tags=("A",), um=2.0), _src(1, tags=("A", "B")), _src(0, cls="dent")],
        classes=["scratch", "dent"],
    )
    rows = {r.cls: r for r in bank.summary()}
    assert rows["scratch"].no_pitch == 1 and rows["dent"].no_pitch == 1
    assert rows["scratch"].tags == {"A": 2, "B": 1} and rows["dent"].tags == {}
    assert bank.no_pitch_count() == 2 and bank.tag_counts() == {"A": 2, "B": 1}


def _recipe_for(bank: Bank, um: float | None) -> R.Recipe:
    return R.Recipe.from_dict(
        {
            "version": 1,
            "name": "t",
            "seed": 1,
            "inputs": {"bank": "b", "targets": "n", "um_per_px": um},
            "output": {"root": "o", "count": 1},
            "pipeline": {"preset": "hard-paste"},
        }
    )


def test_scale_warning_levels() -> None:
    none = Bank.from_sources([_src(0), _src(1)])
    some = Bank.from_sources([_src(0, um=1.5), _src(1)])
    full = Bank.from_sources([_src(0, um=1.5), _src(1, um=1.5)])
    w = runner.scale_warning(_recipe_for(none, None), none)
    assert w and "축척 정합 꺼짐" in w and "전부" in w
    w = runner.scale_warning(_recipe_for(full, None), full)
    assert w and "대상 inputs.um_per_px" in w
    w = runner.scale_warning(_recipe_for(some, 1.5), some)
    assert w and "일부" in w and "1/2" in w
    assert runner.scale_warning(_recipe_for(full, 1.5), full) is None
    empty = Bank.from_sources([], name="(없음)")
    assert runner.scale_warning(_recipe_for(empty, None), empty) is None


# --- CLI e2e: import --tags → bank ls no_um/tags 열 → dry-run 태그 풀 -------------


def test_cli_bank_ls_shows_no_um_and_tags_and_run_dry_run_filters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    bank = tmp_path / "bank"
    normals = tmp_path / "normals.txt"
    args = [
        "bank", "import-yolo", "--images", str(d["images"]), "--labels", str(d["labels"]),
        "--names", str(d["names"]), "--out", str(bank), "--mask-from", "otsu",
        "--list-normals", str(normals), "--tags", "prodA,lot1",
    ]  # fmt: skip
    assert main(args) == EXIT_OK
    capsys.readouterr()
    assert main(["bank", "ls", str(bank)]) == EXIT_OK
    cap = capsys.readouterr()
    assert "no_um" in cap.out and "미지정 5/5" in cap.out and "prodA:" in cap.out
    assert "um_per_px 없는 소스 5개" in cap.err

    recipe = tmp_path / "r.yaml"
    assert (
        main(
            [
                "recipe",
                "init",
                "--preset",
                "hard-paste",
                "--bank",
                str(bank),
                "--targets",
                str(normals),
                "--out",
                str(tmp_path / "out"),
                "--count",
                "2",
                "--write",
                str(recipe),
            ]
        )
        == EXIT_OK
    )
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    data["pipeline"]["source"]["tags"] = {"include": ["prodA"], "exclude": ["nope"]}
    data["pipeline"]["source"]["min_sources_warn"] = 1
    recipe.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    capsys.readouterr()
    assert main(["run", str(recipe), "--dry-run"]) == EXIT_OK
    cap = capsys.readouterr()
    assert "source.tags" in cap.out and "include=['prodA']" in cap.out
    assert "축척 정합 꺼짐" in cap.err  # prepare 경고가 CLI 표면에

    # 태그가 전혀 안 맞으면 prepare 단계에서 명확히 실패
    data["pipeline"]["source"]["tags"] = {"include": ["prodZ"]}
    recipe.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert main(["run", str(recipe), "--dry-run"]) != EXIT_OK
    assert "태그 필터" in capsys.readouterr().err

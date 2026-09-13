from pathlib import Path

import pytest
import yaml

from anograft import __version__
from anograft.cli import EXIT_OK, EXIT_RECIPE_ERROR, build_parser, main


def test_version_is_semver() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


def test_no_command_prints_help_and_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == EXIT_RECIPE_ERROR
    assert "anograft" in capsys.readouterr().err


def test_methods_lists_all_v01_choices(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["methods"]) == EXIT_OK
    out = capsys.readouterr().out
    for token in [
        "[blend]",
        "paste",
        "alpha",
        "poisson",
        "multiband",
        "[harmonize]",
        "histmatch",
        "프리셋:",
    ]:
        assert token in out


def test_methods_single_stage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["methods", "--stage", "roi"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "[roi]" in out and "[blend]" not in out


def test_recipe_init_then_check_roundtrip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "r.yaml"
    assert (
        main(["recipe", "init", "--preset", "multiband-graft", "--write", str(target)]) == EXIT_OK
    )
    text = target.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert data["pipeline"]["blend"] == {"method": "multiband", "levels": 4}
    capsys.readouterr()

    code = main(["recipe", "check", str(target), "--resolved"])
    captured = capsys.readouterr()
    assert "레시피 OK: multiband-graft" in captured.out
    # 스테이지 구현 여부에 따라 실행 불가 목록이 있을 수 있다 — 검증 자체는 통과해야 한다
    assert code in (EXIT_OK, EXIT_RECIPE_ERROR)
    assert "preset: multiband-graft" in captured.out


def test_recipe_init_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["recipe", "init"]) == EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("# anograft") and "method: poisson" in out


def test_recipe_init_unknown_preset(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["recipe", "init", "--preset", "nope"]) == EXIT_RECIPE_ERROR
    assert "프리셋이 없습니다" in capsys.readouterr().err


def test_recipe_check_reports_validation_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "name": "x",
                "seed": 1,
                "inputs": {"bank": "b", "targets": "t"},
                "output": {"root": "o", "count": 1},
                "pipeline": {"blend": {"method": "alpha", "levels": 3}},
            }
        ),
        encoding="utf-8",
    )
    assert main(["recipe", "check", str(bad)]) == EXIT_RECIPE_ERROR
    err = capsys.readouterr().err
    assert "레시피 검증 실패" in err and "levels" in err


def test_parser_prog() -> None:
    assert build_parser().prog == "anograft"

from pathlib import Path

import numpy as np
import pytest
import yaml

from anograft import __version__
from anograft.cli import EXIT_ALL_SKIPPED, EXIT_OK, EXIT_RECIPE_ERROR, build_parser, main
from anograft.io import imgio
from anograft.io.manifest import read_manifest
from tests.fixtures import fake_yolo_dataset


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


# ---------------------------------------------------------------------------
# 엔드투엔드 — import-yolo → bank ls → recipe → run → preview (설계 §11 test_cli)
# ---------------------------------------------------------------------------


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def workspace(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Path]:
    """은행 + 정상 목록 + pairs 레시피(count 5)가 준비된 작업 폴더."""
    d = fake_yolo_dataset(tmp_path / "ds")
    bank = tmp_path / "bank"
    normals = tmp_path / "normals.txt"
    code = main(
        [
            "bank",
            "import-yolo",
            "--images",
            str(d["images"]),
            "--labels",
            str(d["labels"]),
            "--names",
            str(d["names"]),
            "--out",
            str(bank),
            "--mask-from",
            "otsu",
            "--list-normals",
            str(normals),
        ]
    )
    out = capsys.readouterr().out
    assert code == EXIT_OK and "소스 5개" in out and "정상(라벨 없음) 2장" in out
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
                "5",
                "--seed",
                "11",
                "--write",
                str(recipe),
            ]
        )
        == EXIT_OK
    )
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    data["output"]["writer"] = {"format": "pairs"}
    data["pipeline"]["placement"]["roi"]["erode_px"] = 2
    data["pipeline"]["placement"]["margin_px"] = 4
    data["pipeline"]["source"]["min_sources_warn"] = 1
    recipe.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    capsys.readouterr()
    return {"bank": bank, "normals": normals, "recipe": recipe, "root": tmp_path}


def test_bank_ls_table(workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["bank", "ls", str(workspace["bank"])]) == EXIT_OK
    out = capsys.readouterr().out
    assert "spot" in out and "crack" in out and "yolo-polygon:1" in out and "소스 5" in out
    assert main(["bank", "ls", str(workspace["root"] / "nope")]) == EXIT_RECIPE_ERROR


def test_recipe_check_validates_against_bank(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["recipe", "check", str(workspace["recipe"])]) == EXIT_OK
    assert "은행 대조 OK" in capsys.readouterr().out


def test_run_writes_triplets_manifest_and_is_reproducible(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    root = workspace["root"] / "out"
    assert main(["run", str(workspace["recipe"]), "--workers", "0"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "완료: ok 5" in out and "정상 2" in out
    rows = read_manifest(root / "manifest.csv")
    assert len(rows) == 7 and sum(r["status"] == "ok" for r in rows) == 5
    for r in rows:
        if r["status"] != "ok":
            continue
        for col in ("image", "mask", "sidecar"):
            assert (root / r[col]).is_file(), col
        m = imgio.read_mask(root / r["mask"])
        assert set(np.unique(m)) == {0, 255} and int(r["area_px"]) == int(np.count_nonzero(m))
    assert (root / "recipe.resolved.yaml").is_file()
    assert sorted(p.name for p in (root / "images").iterdir())[:2] == ["000000.png", "000001.png"]
    # 같은 레시피를 다른 폴더에 다시 돌리면 트리가 바이트 동일 (recipe.resolved.yaml의 root만 다르다)
    root2 = workspace["root"] / "out2"
    assert main(["run", str(workspace["recipe"]), "--out", str(root2)]) == EXIT_OK
    a, b = _tree_bytes(root), _tree_bytes(root2)
    assert set(a) == set(b)
    for k in a:
        if k in ("recipe.resolved.yaml", "manifest.csv") or k.startswith("meta/"):
            continue
        assert a[k] == b[k], k
    # 시드를 바꾸면 달라진다
    root3 = workspace["root"] / "out3"
    assert main(["run", str(workspace["recipe"]), "--out", str(root3), "--seed", "12"]) == EXIT_OK
    assert _tree_bytes(root3)["images/000000.png"] != a["images/000000.png"]


def test_run_dry_run_writes_nothing(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", str(workspace["recipe"]), "--dry-run"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "pipeline_hash" in out and "class spot" in out and "dry-run" in out
    assert not (workspace["root"] / "out").exists()


def test_run_all_skipped_exits_2(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    recipe = workspace["recipe"]
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    data["pipeline"]["geometry"]["scale"] = [6.0, 6.0]  # 패치가 대상보다 커서 배치 실패
    data["pipeline"]["placement"]["shrink_on_fail"]["rounds"] = 0
    recipe.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert main(["run", str(recipe), "--count", "2"]) == EXIT_ALL_SKIPPED
    out = capsys.readouterr().out
    assert "ok 0" in out and "skipped 2" in out
    rows = read_manifest(workspace["root"] / "out" / "manifest.csv")
    assert all(r["status"] != "ok" for r in rows) and any(r["reason"] for r in rows)


def test_run_reports_prepare_errors(
    workspace: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recipe = workspace["recipe"]
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    data["output"]["class_ratio"] = {"ghost": 1.0}
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert main(["run", str(bad)]) == EXIT_RECIPE_ERROR
    assert "은행에 없는 클래스" in capsys.readouterr().err
    data["output"].pop("class_ratio")
    data["inputs"]["bank"] = str(tmp_path / "nobank")
    bad.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert main(["run", str(bad)]) == EXIT_RECIPE_ERROR
    assert "bank.yaml" in capsys.readouterr().err


def test_preview_renders_three_panels(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out = workspace["root"] / "pv.png"
    assert (
        main(
            [
                "preview",
                str(workspace["recipe"]),
                "--index",
                "1",
                "--out",
                str(out),
                "--long-side",
                "200",
            ]
        )
        == EXIT_OK
    )
    text = capsys.readouterr().out
    assert "미리보기" in text and "#" in text and "→" in text
    img, _ = imgio.read_image(out)
    assert img.shape[0] <= 200 and img.shape[1] > 3 * 90  # 세 패널 가로 배치
    # run 결과와 같은 대상·같은 소스를 골랐다 (같은 시드·인덱스)
    assert main(["run", str(workspace["recipe"]), "--count", "2"]) == EXIT_OK
    rows = read_manifest(workspace["root"] / "out" / "manifest.csv")
    r1 = next(r for r in rows if r["index"] == "1")
    assert r1["source_ids"] in text and Path(r1["target"]).name in text


def test_import_yolo_rejects_small_margin(
    workspace: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = fake_yolo_dataset(tmp_path / "ds2")
    code = main(
        [
            "bank",
            "import-yolo",
            "--images",
            str(d["images"]),
            "--labels",
            str(d["labels"]),
            "--names",
            "a,b",
            "--out",
            str(tmp_path / "b2"),
            "--margin",
            "3",
        ]
    )
    assert code == EXIT_RECIPE_ERROR and "margin" in capsys.readouterr().err

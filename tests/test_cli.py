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
    assert "미구현" not in out and "불가" not in out  # v0.1 method 전부 구현 (stages-more-methods)


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


# ---------------------------------------------------------------------------
# #758 — 멀티프로세싱 결정성 · yolo writer · compare-methods · import-pairs/import-dataset · bank preview · dataset info
# ---------------------------------------------------------------------------


def _set_writer(recipe: Path, fmt: str, **extra: object) -> None:
    data = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    data["output"]["writer"] = {"format": fmt, **extra}
    recipe.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_run_workers_0_and_2_produce_identical_trees(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """설계 §7·§11: 프로세스 풀 결과 = 인프로세스 결과 (images·masks·meta·labels·manifest 바이트 동일)."""
    recipe = workspace["recipe"]
    _set_writer(recipe, "yolo")
    # 다른 --out 으로 두 번 — pipeline_hash 는 output.root 를 빼고 계산하므로(데브로그 07) 사이드카·meta 도 같아야 한다.
    # recipe.resolved.yaml 만 root 줄이 다르다.
    root0, root = workspace["root"] / "w0", workspace["root"] / "out"
    assert main(["run", str(recipe), "--workers", "0", "--out", str(root0)]) == EXIT_OK
    assert main(["run", str(recipe), "--workers", "2", "--out", str(root)]) == EXIT_OK
    out = capsys.readouterr().out
    assert out.count("완료: ok 5") == 2 and "data.yaml" in out
    a, b = _tree_bytes(root0), _tree_bytes(root)
    assert set(a) == set(b) and {"labels/000000.txt", "data.yaml", "labels/n_n0.txt"} <= set(a)
    assert [k for k in a if a[k] != b[k]] == [
        "recipe.resolved.yaml"
    ]  # images·masks·meta(해시)·labels 전부 동일
    ra, rb = a["recipe.resolved.yaml"].decode("utf-8"), b["recipe.resolved.yaml"].decode("utf-8")
    assert ra.splitlines()[0] == rb.splitlines()[0]  # 헤더의 pipeline_hash 동일
    assert [ln for ln in ra.splitlines() if ln not in rb.splitlines()] == [
        "  root: " + root0.as_posix()
    ]
    # 라벨: ok 행마다 한 줄 이상, 정상 이미지는 빈 파일
    rows = read_manifest(root0 / "manifest.csv")
    for r in rows:
        if r["status"] == "ok":
            lines = (root0 / r["label"]).read_text(encoding="utf-8").splitlines()
            assert len(lines) == int(r["n_defects"]) >= 1
            assert all(len(ln.split()) == 5 for ln in lines)
        elif r["status"] == "normal":
            assert (root0 / "labels" / (Path(r["image"]).stem + ".txt")).read_text() == ""
    assert yaml.safe_load((root0 / "data.yaml").read_text(encoding="utf-8"))["names"] == [
        "spot",
        "crack",
    ]


def test_preview_compare_methods_grid(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out = workspace["root"] / "cmp.png"
    code = main(
        [
            "preview",
            str(workspace["recipe"]),
            "--index",
            "0",
            "--compare-methods",
            "blend",
            "--out",
            str(out),
            "--long-side",
            "160",
        ]
    )
    assert code == EXIT_OK
    text = capsys.readouterr().out
    assert "비교:" in text and all(m in text for m in ("paste", "alpha", "poisson", "multiband"))
    img, _ = imgio.read_image(out)
    # 4 method → 2×2 격자, 타일은 결함 주변 크롭을 long-side 로 확대(정사각이 아닐 수 있다), gap 6
    assert "크롭" in text and max(img.shape[:2]) == 2 * 160 + 6 and min(img.shape[:2]) >= 2 * 60
    assert (
        main(
            [
                "preview",
                str(workspace["recipe"]),
                "--compare-methods",
                "blend",
                "--out",
                str(out),
                "--full",
                "--long-side",
                "96",
            ]
        )
        == EXIT_OK
    )
    assert "전체" in capsys.readouterr().out
    img, _ = imgio.read_image(out)
    assert img.shape[:2] == (2 * 96 + 6, 2 * 96 + 6)  # 전체 96px, gap 6
    # harmonize 도 된다 (4 method)
    assert (
        main(
            [
                "preview",
                str(workspace["recipe"]),
                "--compare-methods",
                "harmonize",
                "--out",
                str(out),
                "--long-side",
                "120",
            ]
        )
        == EXIT_OK
    )
    assert "histmatch" in capsys.readouterr().out


def test_bank_import_pairs_and_preview_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.fixtures import fake_pairs_dataset

    d = fake_pairs_dataset(tmp_path / "ds")
    bank = tmp_path / "bank"
    code = main(
        [
            "bank",
            "import-pairs",
            "--images",
            str(d["images"]),
            "--masks",
            str(d["masks"]),
            "--out",
            str(bank),
            "--class-from-dir",
        ]
    )
    out = capsys.readouterr()
    assert (
        code == EXIT_OK
        and "3쌍" in out.out
        and "소스 4개" in out.out
        and "마스크 없음 1" in out.out
    )
    # CSV 모드 누적
    code = main(["bank", "import-pairs", "--csv", str(d["csv"]), "--out", str(bank)])
    assert code == EXIT_OK and "중복 id 2" in capsys.readouterr().out
    # 둘 다 없으면 오류
    code = main(
        [
            "bank",
            "import-pairs",
            "--images",
            str(d["images"]),
            "--masks",
            str(d["masks"]),
            "--out",
            str(bank),
        ]
    )
    assert code == EXIT_RECIPE_ERROR and "--class" in capsys.readouterr().err
    # bank preview 그리드
    grid = tmp_path / "grid.png"
    assert (
        main(["bank", "preview", str(bank), "--out", str(grid), "--cols", "3", "--tile", "64"])
        == EXIT_OK
    )
    text = capsys.readouterr().out
    assert "소스 6개" in text and "추정 마스크 0" in text
    img, _ = imgio.read_image(grid)
    assert img.shape[1] == 3 * 64 + 2 * 4 and img.shape[0] == 2 * 64 + 4
    assert main(["bank", "preview", str(bank), "--class", "crack", "--out", str(grid)]) == EXIT_OK
    assert "소스 2개" in capsys.readouterr().out
    assert (
        main(["bank", "preview", str(bank), "--class", "nope", "--out", str(grid)])
        == EXIT_RECIPE_ERROR
    )


def test_dataset_info_and_import_dataset_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.fixtures import fake_mvtec_tree

    assert main(["dataset", "info"]) == EXIT_OK
    assert "mvtec-ad" in capsys.readouterr().out
    assert main(["dataset", "info", "mvtec-ad"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "CC BY-NC-SA" in text and "ground_truth" in text and "재배포" in text
    assert main(["dataset", "info", "nope"]) == EXIT_RECIPE_ERROR
    assert "mvtec-ad" in capsys.readouterr().err

    cat = fake_mvtec_tree(tmp_path / "mvtec")
    bank = tmp_path / "bank"
    assert main(["bank", "import-dataset", "mvtec-ad", str(cat), "--out", str(bank)]) == EXIT_OK
    out = capsys.readouterr()
    assert (
        "mvtec-ad/metal_nut 3쌍" in out.out
        and "정상 이미지 4장" in out.out
        and "train/good" in out.out
    )
    assert "경고 1건" in out.err  # hole/001 마스크 없음
    assert main(["bank", "ls", str(bank)]) == EXIT_OK
    assert "png:" in capsys.readouterr().out
    assert (
        main(["bank", "import-dataset", "mvtec-ad", str(tmp_path / "nope"), "--out", str(bank)])
        == EXIT_RECIPE_ERROR
    )
    assert "카테고리 폴더" in capsys.readouterr().err

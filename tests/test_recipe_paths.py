"""v0.7.x ``resolve_recipe_paths``(KNOWN-ISSUES #9 보완) — 상대 입력 경로는 cwd 우선, 없으면 레시피 파일 기준. 절대경로·null·둘 다
없음은 그대로 · output.root 는 제외 · `Recipe.load_with_notes` 노트 · CLI `recipe check`/`run` 이 다른 cwd 에서 동작 ·
스튜디오 세션 `path_notes`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from anograft.bank.importers import yolo as Y
from anograft.cli import EXIT_OK, main
from anograft.core import recipe as R
from anograft.gui.studio.session import StudioSession
from tests.fixtures import fake_yolo_dataset


def _base(**inputs: object) -> dict:
    return {
        "version": 1,
        "name": "t",
        "seed": 1,
        "inputs": {"bank": "bank", "targets": "normals.txt", **inputs},
        "output": {"root": "out", "count": 1},
        "pipeline": {
            "preset": "hard-paste",
            "placement": {"roi": {"method": "mask_dir", "path": "roi"}},
        },
    }


def test_resolve_prefers_cwd_then_recipe_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proj = tmp_path / "proj"
    (proj / "bank").mkdir(parents=True)
    (proj / "roi").mkdir()
    (proj / "normals.txt").write_text("", encoding="utf-8")
    data = _base()
    data["pipeline"]["source"] = {
        "method": "perlin-texture",
        "texture": "dir",
        "texture_dir": "tex",
    }
    (proj / "tex").mkdir()
    # 레시피 폴더 기준으로만 있는 경우 → 바뀌고 노트
    monkeypatch.chdir(tmp_path)
    out, notes = R.resolve_recipe_paths(data, proj)
    assert out["inputs"]["bank"] == (proj / "bank").as_posix()
    assert out["inputs"]["targets"] == (proj / "normals.txt").as_posix()
    assert out["pipeline"]["placement"]["roi"]["path"] == (proj / "roi").as_posix()
    assert out["pipeline"]["source"]["texture_dir"] == (proj / "tex").as_posix()
    assert len(notes) == 5 and all("레시피 파일 기준" in n for n in notes)
    assert data["inputs"]["bank"] == "bank"  # 원본 dict 불변
    # v0.8.x: 입력이 폴백했으면 output.root(상대)도 레시피 파일 기준 — 출력이 cwd 에 흩어지지 않게
    assert out["output"]["root"] == (proj / "out").as_posix() and data["output"]["root"] == "out"
    assert notes[-1].startswith("output.root:")
    # cwd 에 있으면 그대로(하위 호환)
    monkeypatch.chdir(proj)
    out2, notes2 = R.resolve_recipe_paths(data, tmp_path / "elsewhere")
    assert out2["inputs"]["bank"] == "bank" and notes2 == [] and out2["output"]["root"] == "out"
    # 둘 다 없으면 그대로 · 절대경로·null 은 그대로
    monkeypatch.chdir(tmp_path)
    out3, notes3 = R.resolve_recipe_paths(_base(bank="nope"), tmp_path / "nowhere")
    assert out3["inputs"]["bank"] == "nope" and notes3 == []
    absolute = (proj / "bank").resolve().as_posix()
    out4, notes4 = R.resolve_recipe_paths(_base(bank=absolute, targets=None), proj)
    assert out4["inputs"]["bank"] == absolute and out4["inputs"]["targets"] is None
    # 절대경로·null 은 노트 없음 — roi 만 바뀐다(cwd=tmp_path 에 없고 proj 에 있음)
    assert len(notes4) == 2 and notes4[0].startswith("pipeline.placement.roi.path")
    assert notes4[1].startswith("output.root:")
    # output.root 가 절대경로면 그대로
    abs_out = _base()
    abs_out["output"]["root"] = (tmp_path / "abs-out").as_posix()
    out6, notes6 = R.resolve_recipe_paths(abs_out, proj)
    assert out6["output"]["root"] == (tmp_path / "abs-out").as_posix()
    assert not any(n.startswith("output.root") for n in notes6)
    # 잘못된 구조(placement 가 dict 아님)는 건너뛴다
    bad = _base()
    bad["pipeline"]["placement"] = "x"
    out5, _ = R.resolve_recipe_paths(bad, proj)
    assert out5["pipeline"]["placement"] == "x"


def test_load_from_other_cwd_cli_and_studio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    proj = tmp_path / "proj"
    d = fake_yolo_dataset(proj / "ds")
    normals = proj / "normals.txt"
    Y.import_yolo(
        d["images"], d["labels"], d["names"], proj / "bank", mask_from="rect", list_normals=normals
    )
    data = {
        "version": 1,
        "name": "rel",
        "seed": 2,
        "inputs": {"bank": "bank", "targets": "normals.txt"},
        "output": {"root": "out", "count": 2},
        "pipeline": {
            "preset": "hard-paste",
            "source": {"method": "bank", "min_sources_warn": 1},
            "placement": {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 4},
        },
    }
    recipe = proj / "recipes" / "r.yaml"
    recipe.parent.mkdir()
    # 레시피는 recipes/ 안, 경로는 프로젝트 루트 기준 — 루트에서 돌리는 종전 사용법
    data_root = dict(data)
    recipe.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    monkeypatch.chdir(proj)
    rec, notes = R.Recipe.load_with_notes(recipe)
    assert notes == [] and rec.inputs.bank_key() == "bank"
    # 다른 cwd: cwd 에 bank 가 없고 레시피 폴더에도 없다(recipes/ 안이 아님) → 원래대로 실패 경로(변화 없음)
    monkeypatch.chdir(tmp_path)
    _rec2, notes2 = R.Recipe.load_with_notes(recipe)
    assert notes2 == []
    # 레시피 옆에 자료가 있는 배치(레시피 = 프로젝트 루트) → 다른 cwd 에서도 열린다
    recipe2 = proj / "r.yaml"
    recipe2.write_text(
        yaml.safe_dump(data_root, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    rec3, notes3 = R.Recipe.load_with_notes(recipe2)
    assert len(notes3) == 3 and rec3.inputs.bank_key() == (proj / "bank").as_posix()  # +output.root
    assert main(["recipe", "check", str(recipe2)]) == EXIT_OK
    cap = capsys.readouterr()
    assert "은행 대조 OK" in cap.out and "경로: inputs.bank" in cap.err
    assert main(["run", str(recipe2), "--workers", "0", "--out", str(tmp_path / "o")]) == EXIT_OK
    assert (tmp_path / "o" / "manifest.csv").is_file()  # --out 은 그대로(cwd 기준 명시)
    # --out 없이 돌리면 출력도 레시피 파일 기준(v0.8.x) — cwd(tmp_path)/out 이 아니라 proj/out
    assert main(["run", str(recipe2), "--workers", "0"]) == EXIT_OK
    assert (proj / "out" / "manifest.csv").is_file() and not (tmp_path / "out").exists()
    # 스튜디오 세션도 같은 규칙 + path_notes
    s = StudioSession()
    s.load(recipe2)
    assert len(s.path_notes) == 3 and os.path.isdir(s.recipe.inputs.bank_key())

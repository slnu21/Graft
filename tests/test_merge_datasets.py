"""``io/merge_datasets.py`` · CLI ``dataset merge`` — 출력 폴더 병합(접두어 · index 재부여 · coco/yolo 파일 · 호환성 검사)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from anograft.bank.importers import yolo as Y
from anograft.cli import EXIT_OK, main
from anograft.io.manifest import read_manifest
from anograft.io.merge_datasets import (
    MergeError,
    check_compatible,
    merge_datasets,
    prefixed,
    writer_files,
)
from anograft.io.prune import read_review, write_review
from tests.fixtures import fake_yolo_dataset


def test_prefixed_and_writer_files() -> None:
    assert prefixed("images/000000.png", "d0_") == "images/d0_000000.png"
    assert (
        prefixed("mvtec/x/ground_truth/c/000000_mask.png", "d1_")
        == "mvtec/x/ground_truth/c/d1_000000_mask.png"
    )
    assert prefixed("annotations.json#3", "d0_") == "annotations.json#3"
    assert prefixed("", "d0_") == "" and prefixed("labels/", "d0_") == "labels/"
    row = {"image": "images/a.png", "mask": "masks/a.png"}
    doc = {
        "writer": {
            "format": "mvtec",
            "image": "mvtec/c/test/x/a.png",
            "mask": "mvtec/c/ground_truth/x/a_mask.png",
        }
    }
    assert writer_files(doc, row) == ["mvtec/c/test/x/a.png", "mvtec/c/ground_truth/x/a_mask.png"]
    assert writer_files({"writer": {"format": "yolo", "label": "labels/a.txt"}}, row) == [
        "labels/a.txt"
    ]
    assert writer_files({"writer": {"format": "coco", "label": "annotations.json#1"}}, row) == []
    assert writer_files({}, row) == []


def _make_output(tmp_path: Path, name: str, *, seed: int, writer: str, count: int = 3) -> Path:
    d = fake_yolo_dataset(tmp_path / f"ds-{name}")
    normals = tmp_path / f"normals-{name}.txt"
    Y.import_yolo(
        d["images"],
        d["labels"],
        d["names"],
        tmp_path / f"bank-{name}",
        mask_from="rect",
        list_normals=normals,
    )
    data = {
        "version": 1,
        "name": name,
        "seed": seed,
        "inputs": {"bank": (tmp_path / f"bank-{name}").as_posix(), "targets": normals.as_posix()},
        "output": {
            "root": (tmp_path / name).as_posix(),
            "count": count,
            "writer": {"format": writer},
        },
        "pipeline": {
            "preset": "hard-paste",
            "source": {"method": "bank", "min_sources_warn": 1},
            "placement": {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 4},
        },
    }
    recipe = tmp_path / f"{name}.yaml"
    recipe.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert main(["run", str(recipe), "--workers", "0"]) == EXIT_OK
    return tmp_path / name


@pytest.mark.parametrize("writer", ["yolo", "coco", "mvtec"])
def test_merge_two_outputs(tmp_path: Path, writer: str) -> None:
    a = _make_output(tmp_path, "a", seed=1, writer=writer)
    b = _make_output(tmp_path, "b", seed=2, writer=writer)
    rows_a, rows_b = read_manifest(a / "manifest.csv"), read_manifest(b / "manifest.csv")
    first_ok_b = next(r["index"] for r in rows_b if r["status"] == "ok")
    write_review(b / "review.csv", {first_ok_b: ("reject", "bad")})
    out = tmp_path / "merged"
    s = merge_datasets([a, b], out)
    assert s.synthetic == sum(r["status"] == "ok" for r in rows_a + rows_b)
    assert s.normals == sum(r["status"] == "normal" for r in rows_a + rows_b)
    assert s.skipped == sum(r["status"] == "skipped" for r in rows_a + rows_b)
    rows = read_manifest(out / "manifest.csv")
    assert len(rows) == len(rows_a) + len(rows_b)
    # 합성·skipped 행은 0..N-1 로 다시 번호, 정상 행은 index 빈 채
    idx = [int(r["index"]) for r in rows if r["status"] != "normal"]
    assert idx == list(range(len(idx)))
    assert all(r["index"] == "" for r in rows if r["status"] == "normal")
    # 파일 이름 접두어 + 실제 존재 + 사이드카 index/merged_from 일치
    for r in rows:
        if r["status"] == "skipped":
            continue
        for col in ("image", "mask", "sidecar"):
            assert Path(r[col]).name.startswith(("d0_", "d1_")) and (out / r[col]).is_file(), (
                col,
                r[col],
            )
        doc = json.loads((out / r["sidecar"]).read_text(encoding="utf-8"))
        assert doc["merged_from"]["root"] in {"a", "b"}
        if r["status"] == "ok":
            assert doc["index"] == int(r["index"])
        if writer == "yolo":
            assert r["label"].startswith("labels/d") and (out / r["label"]).is_file()
            assert doc.get("writer", {}).get("label", r["label"]) == r["label"]
        elif writer == "mvtec":
            # 레이아웃 사본(test/<cls>/d0_x.png · ground_truth/<cls>/d0_x_mask.png · train|test/good/d0_n_x.png)도 접두어 + 존재
            assert r["label"].startswith("mvtec/") and Path(r["label"]).name.startswith(
                ("d0_", "d1_")
            )
            assert (out / r["label"]).is_file()
            for key in ("image", "mask"):
                rel = doc.get("writer", {}).get(key)
                if rel:
                    assert Path(rel).name.startswith(("d0_", "d1_")) and (out / rel).is_file()
        else:
            assert r["label"].startswith("annotations.json#")
    assert (out / "recipe.resolved.yaml").is_file() and (out / "merge.json").is_file()
    merge_doc = json.loads((out / "merge.json").read_text(encoding="utf-8"))
    assert merge_doc["writer"] == writer and len(merge_doc["per_root"]) == 2
    # review.csv 는 새 index 로 옮겨진다(b 의 첫 합성 → 반려 유지)
    rv = read_review(out / "review.csv")
    assert len(rv) == 1 and next(iter(rv.values())) == ("reject", "bad")
    new_idx = next(iter(rv))
    doc = json.loads(
        (out / next(r["sidecar"] for r in rows if r["index"] == new_idx)).read_text(
            encoding="utf-8"
        )
    )
    assert doc["merged_from"] == {"root": "b", "index": first_ok_b}
    if writer == "coco":
        ann = json.loads((out / "annotations.json").read_text(encoding="utf-8"))
        names = {Path(r["image"]).name for r in rows if r["status"] != "skipped"}
        assert {im["file_name"] for im in ann["images"]} == names
        ids = [im["id"] for im in ann["images"]]
        assert len(ids) == len(set(ids))
        aids = [x["id"] for x in ann["annotations"]]
        assert len(aids) == len(set(aids)) and all(
            x["image_id"] in set(ids) for x in ann["annotations"]
        )
        na = json.loads((a / "annotations.json").read_text(encoding="utf-8"))
        nb = json.loads((b / "annotations.json").read_text(encoding="utf-8"))
        assert len(ann["annotations"]) == len(na["annotations"]) + len(nb["annotations"])
        assert ann["categories"] == na["categories"]
    elif writer == "yolo":
        assert (out / "data.yaml").is_file()
    # 병합 결과를 다시 병합에 넣어도 된다(접두어가 겹쳐 붙는다)
    s2 = merge_datasets([out, a], tmp_path / "merged2")
    assert s2.synthetic == s.synthetic + sum(r["status"] == "ok" for r in rows_a)


def test_merge_rejects_incompatible(tmp_path: Path) -> None:
    a = _make_output(tmp_path, "a", seed=1, writer="yolo")
    b = _make_output(tmp_path, "b", seed=2, writer="coco")
    with pytest.raises(MergeError, match="writer 형식"):
        merge_datasets([a, b], tmp_path / "m")
    with pytest.raises(MergeError, match="둘 이상"):
        check_compatible([a])
    with pytest.raises(MergeError, match="다른 폴더"):
        merge_datasets([a, a], a)
    # 클래스 이름이 다르면 거부
    dy = b / "data.yaml"
    dy.write_text(yaml.safe_dump({"names": ["x", "y"]}), encoding="utf-8")
    (b / "recipe.resolved.yaml").write_text(
        (a / "recipe.resolved.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(MergeError, match="클래스 이름"):
        merge_datasets([a, b], tmp_path / "m")


def test_cli_dataset_merge(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _make_output(tmp_path, "a", seed=1, writer="yolo")
    b = _make_output(tmp_path, "b", seed=2, writer="yolo")
    assert main(["dataset", "merge", str(a), str(b), "--out", str(tmp_path / "m")]) == EXIT_OK
    out = capsys.readouterr().out
    assert "병합 →" in out and "a: 합성" in out and (tmp_path / "m" / "manifest.csv").is_file()
    assert main(["dataset", "merge", str(a), "--out", str(tmp_path / "m2")]) != EXIT_OK
    assert "병합 실패" in capsys.readouterr().err

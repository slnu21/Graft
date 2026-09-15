"""v0.7 검수(Qt 없음) — ``io.prune``(review.csv 읽기/쓰기 · 정리본: 반려 제외·정상 유지·skipped 제거·manifest 재작성·coco 필터·
mvtec 사본) · ``ReviewSession``(열기·지연 사이드카·필터·판정·저장·분포 히스토그램·prune) · CLI ``dataset prune``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from anograft.bank.importers import yolo as Y
from anograft.cli import EXIT_OK, main
from anograft.gui.review.session import ReviewError, ReviewSession, histogram
from anograft.io.manifest import read_manifest
from anograft.io.prune import PruneError, prune_dataset, read_review, write_review
from tests.fixtures import fake_yolo_dataset


@pytest.fixture
def output_root(tmp_path: Path) -> Path:
    """샘플 YOLO → 은행 → run(count 4, coco writer) 출력 폴더. cwd 무관하게 은행 경로는 절대."""
    d = fake_yolo_dataset(tmp_path / "ds")
    normals = tmp_path / "normals.txt"
    Y.import_yolo(
        d["images"],
        d["labels"],
        d["names"],
        tmp_path / "bank",
        mask_from="rect",
        list_normals=normals,
    )
    data = {
        "version": 1,
        "name": "rv",
        "seed": 4,
        "inputs": {"bank": (tmp_path / "bank").as_posix(), "targets": normals.as_posix()},
        "output": {"root": (tmp_path / "out").as_posix(), "count": 4, "writer": {"format": "coco"}},
        "pipeline": {
            "preset": "hard-paste",
            "source": {"method": "bank", "min_sources_warn": 1},
            "placement": {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 4},
        },
    }
    recipe = tmp_path / "r.yaml"
    recipe.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert main(["run", str(recipe), "--workers", "0"]) == EXIT_OK
    return tmp_path / "out"


def test_review_csv_roundtrip_and_validation(tmp_path: Path) -> None:
    p = tmp_path / "review.csv"
    assert read_review(p) == {}
    write_review(p, {"3": ("accept", ""), "10": ("reject", "흐림"), "1": ("", "메모만")})
    assert read_review(p) == {"1": ("", "메모만"), "3": ("accept", ""), "10": ("reject", "흐림")}
    assert p.read_text(encoding="utf-8").splitlines()[1].startswith("1,")  # 숫자 순
    p.write_text("index,verdict,note\n0,maybe,\n", encoding="utf-8")
    with pytest.raises(PruneError, match="verdict"):
        read_review(p)


def test_histogram_shared_log_bins() -> None:
    h = histogram([10, 100, 1000], [20, 200], bins=4)
    assert h.bins == 4 and sum(h.a) == 3 and sum(h.b) == 2 and h.log
    assert h.edges[0] == pytest.approx(10.0) and h.edges[-1] == pytest.approx(1000.0)
    empty = histogram([], [], bins=3)
    assert sum(empty.a) == 0 and sum(empty.b) == 0 and len(empty.edges) == 4
    one = histogram([5.0], [], bins=2)
    assert sum(one.a) == 1
    lin = histogram([0, 1, 2], [3], bins=3, log=False)
    assert sum(lin.a) == 3 and not lin.log


def test_session_load_verdict_save_filters_and_distribution(output_root: Path) -> None:
    s = ReviewSession()
    with pytest.raises(ReviewError):
        s.load(output_root.parent / "nope")
    items = s.load(output_root)
    c = s.counts()
    assert c["ok"] + c["skipped"] == 4 and c["normal"] == 2 and c["unreviewed"] == c["ok"]
    assert s.bank is not None and s.recipe_meta["inputs"]["bank"]
    ok = [it for it in items if it.status == "ok"]
    assert ok and not ok[0].loaded
    it = s.item(ok[0].index)
    assert it.loaded and it.instances and "area_px" in it.instances[0]
    # 판정 · 저장 · 재로드
    s.set_verdict(ok[0].index, "accept")
    s.set_verdict(ok[1].index, "reject", "너무 큼") if len(ok) > 1 else None
    with pytest.raises(ReviewError, match="verdict"):
        s.set_verdict(ok[0].index, "maybe")
    normal = next(it for it in items if it.status == "normal")
    with pytest.raises(ReviewError, match="합성"):
        s.set_verdict(normal.index, "accept")
    assert s.dirty
    p = s.save()
    assert p.name == "review.csv" and not s.dirty
    s2 = ReviewSession()
    s2.load(output_root, load_bank=False)
    assert s2.item(ok[0].index).verdict == "accept" and s2.bank is None
    if len(ok) > 1:
        assert s2.item(ok[1].index).note == "너무 큼"
        assert [x.index for x in s2.filtered("reject")] == [ok[1].index]
    assert [x.index for x in s2.filtered("accept")] == [ok[0].index]
    assert len(s2.filtered("unreviewed")) == c["ok"] - (2 if len(ok) > 1 else 1)
    assert len(s2.filtered("all")) == len(items) and len(s2.filtered("skipped")) == c["skipped"]
    assert s2.filtered("all", cls="spot") == [x for x in items if "spot" in x.classes]
    with pytest.raises(ReviewError):
        s2.filtered("bogus")
    # 분포: 반려는 합성에서 빠지고, 실제는 은행 소스 수
    syn = s.synthetic_values("area")
    assert len(syn) == sum(len(s.item(x.index).instances) for x in ok if x.verdict != "reject")
    assert len(s.real_values("area")) == 5 and all(v > 0 for v in s.real_values("length"))
    h = s.distribution("length")
    assert sum(h.a) == len(syn) and sum(h.b) == 5
    assert "채택 1" in s.summary_text()


def test_prune_drops_rejected_keeps_normals_filters_coco(output_root: Path, tmp_path: Path) -> None:
    s = ReviewSession()
    items = s.load(output_root, load_bank=False)
    ok = [it for it in items if it.status == "ok"]
    s.set_verdict(ok[0].index, "reject", "x")
    out = tmp_path / "pruned"
    summary = s.prune(out)
    assert summary.dropped == 1 and summary.kept == len(ok) - 1 and summary.normals == 2
    assert summary.skipped == 4 - len(ok) and (out / "review.csv").is_file()
    rows = read_manifest(out / "manifest.csv")
    assert all(r["status"] != "skipped" for r in rows) and ok[0].index not in {
        r["index"] for r in rows
    }
    assert (
        not (out / ok[0].image).exists() and (out / ok[1].image).is_file() if len(ok) > 1 else True
    )
    assert (out / "recipe.resolved.yaml").is_file()
    doc = json.loads((out / "annotations.json").read_text(encoding="utf-8"))
    kept = {Path(r["image"]).name for r in rows}
    assert {im["file_name"] for im in doc["images"]} == kept
    ids = {im["id"] for im in doc["images"]}
    assert all(a["image_id"] in ids for a in doc["annotations"])
    assert read_review(out / "review.csv") == {}  # 반려는 정리본 review 에서도 빠진다
    with pytest.raises(ReviewError, match="다른 폴더"):
        s.prune(output_root)
    # 미검수까지 빼면 채택만
    s.set_verdict(ok[1].index, "accept") if len(ok) > 1 else None
    s2 = prune_dataset(output_root, tmp_path / "strict", s.review, drop_unreviewed=True)
    assert s2.kept == (1 if len(ok) > 1 else 0)


def test_cli_dataset_prune(
    output_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = read_manifest(output_root / "manifest.csv")
    first_ok = next(r["index"] for r in rows if r["status"] == "ok")
    write_review(output_root / "review.csv", {first_ok: ("reject", "")})
    assert main(["dataset", "prune", str(output_root), "--out", str(tmp_path / "p")]) == EXIT_OK
    out = capsys.readouterr().out
    assert "1 제외" in out and (tmp_path / "p" / "manifest.csv").is_file()
    assert main(["dataset", "prune", str(output_root), "--out", str(output_root)]) != EXIT_OK
    assert "정리 실패" in capsys.readouterr().err

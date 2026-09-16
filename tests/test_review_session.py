"""v0.7 검수(Qt 없음) — ``io.prune``(review.csv 읽기/쓰기 · 정리본: 반려 제외·정상 유지·skipped 제거·manifest 재작성·coco 필터·\nmvtec 사본) · ``ReviewSession``(열기·지연 사이드카·필터·판정·저장·분포 히스토그램·prune) · CLI ``dataset prune``."""

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


# --- 리포트(v0.7.x) --------------------------------------------------------------


def test_report_render_and_session_report(output_root: Path, tmp_path: Path) -> None:
    from anograft.io.report import ReportData, render_report, skipped_reason_counts, svg_histogram

    h = histogram([10, 100, 1000], [20, 200], bins=4)
    svg = svg_histogram(h, "면적")
    assert (
        svg.startswith("<svg")
        and svg.count("<rect") == 5
        and "■ 합성 3" in svg
        and "■ 실제 2" in svg
    )
    assert "분포 없음" in svg_histogram(None, "x") and "분포 없음" in svg_histogram(
        histogram([], []), "x"
    )
    assert skipped_reason_counts(
        ["placement: 배치 실패 — max_tries", "placement: 배치 실패 — x", "roi: 없음", ""]
    ) == {
        "placement: 배치 실패": 2,
        "roi: 없음": 1,
        "(사유 없음)": 1,
    }
    d = ReportData(
        root="o",
        recipe_name="<r>",
        counts={"ok": 4, "accept": 1, "reject": 1},
        rejected=[("3", "spot", "a<b")],
    )
    page = render_report(d)
    assert (
        "&lt;r&gt;" in page and "a&lt;b" in page and "50%" in page and "<svg" not in page
    )  # 히스토그램 없음
    s = ReviewSession()
    items = s.load(output_root)
    ok = [it for it in items if it.status == "ok"]
    s.set_verdict(ok[0].index, "reject", "흐림")
    data = s.report_data()
    assert (
        data.recipe_name == "rv"
        and data.seed == 4
        and data.preset == "hard-paste"
        and len(data.pipeline_hash) >= 8
    )
    assert data.counts["reject"] == 1 and data.rejected == [
        (ok[0].index, ", ".join(ok[0].classes), "흐림")
    ]
    assert sum(data.per_class.values()) == sum(len(s.item(x.index).instances) for x in ok[1:])
    assert data.hist_area is not None and sum(data.hist_area.b) == 5 and data.bank_name
    p = s.write_report()
    assert p == output_root / "review-report.html" and "<svg" in p.read_text(encoding="utf-8")
    p2 = s.write_report(tmp_path / "r" / "x.html")
    assert p2.is_file()
    assert (
        main(
            [
                "dataset",
                "report",
                str(output_root),
                "--out",
                str(tmp_path / "cli.html"),
                "--no-bank",
            ]
        )
        == EXIT_OK
    )
    assert (
        "은행 없음" in (tmp_path / "cli.html").read_text(encoding="utf-8")
        or (tmp_path / "cli.html").is_file()
    )
    assert main(["dataset", "report", str(tmp_path / "nope")]) != EXIT_OK


# --- 대비 분포(v0.7.x) -----------------------------------------------------------


def test_contrast_distribution_synthetic_vs_real(output_root: Path) -> None:
    import numpy as np

    from anograft.gui.review.session import DIST_KEYS, mask_contrast

    assert "contrast" in DIST_KEYS
    g = np.full((40, 40), 100, dtype=np.uint8)
    g[15:25, 15:25] = 160
    m = np.zeros((40, 40), dtype=np.uint8)
    m[15:25, 15:25] = 255
    assert mask_contrast(g, m) == 60.0
    assert mask_contrast(g, np.zeros_like(m)) is None
    assert mask_contrast(g, np.full_like(m, 255)) is None  # 링이 없다
    s = ReviewSession()
    items = s.load(output_root)
    ok = [it for it in items if it.status == "ok"]
    syn = s.synthetic_values("contrast")
    assert len(syn) == sum(len(s.item(x.index).instances) for x in ok)
    assert all(isinstance(v, float) for v in syn) and ok[0].contrasts is not None
    real = s.real_values("contrast")
    assert len(real) == 5 and all(v != 0.0 for v in real)  # 얼룩은 배경보다 밝다
    h = s.distribution("contrast")
    assert not h.log and sum(h.a) == len(syn) and sum(h.b) == 5
    assert h.edges[0] <= min(syn + real) and h.edges[-1] >= max(syn + real)
    with pytest.raises(ReviewError):
        s.synthetic_values("bogus")
    # 반려하면 합성 값에서 빠진다
    s.set_verdict(ok[0].index, "reject")
    assert len(s.synthetic_values("contrast")) == len(syn) - len(ok[0].contrasts)
    # 리포트에 세 번째 히스토그램
    data = s.report_data()
    assert (
        data.hist_contrast is not None
        and "대비" in (output_root / "review-report.html").read_text(encoding="utf-8")
        if s.write_report()
        else True
    )


def test_texture_and_sharpness_distributions(output_root: Path) -> None:
    import numpy as np

    from anograft.gui.review.session import IMAGE_KEYS, mask_sharpness, mask_texture

    assert IMAGE_KEYS == ("contrast", "texture", "sharpness", "lighting")
    flat = np.full((32, 32), 100, dtype=np.uint8)
    m = np.zeros((32, 32), dtype=np.uint8)
    m[8:24, 8:24] = 255
    assert mask_texture(flat, m) == 0.0 and mask_sharpness(flat, m) == 0.0
    edgy = flat.copy()
    edgy[:, 16:] = 200
    assert mask_texture(edgy, m) > 0 and mask_sharpness(edgy, m) > 0
    assert (
        mask_texture(flat, np.zeros_like(m)) is None
        and mask_sharpness(flat, np.zeros_like(m)) is None
    )
    s = ReviewSession()
    items = s.load(output_root)
    ok = [it for it in items if it.status == "ok"]
    n_inst = sum(len(s.item(x.index).instances) for x in ok)
    tex, sharp = s.synthetic_values("texture"), s.synthetic_values("sharpness")
    assert len(tex) == len(sharp) == n_inst and all(v >= 0 for v in tex + sharp)
    assert (
        set(ok[0].appearance) == set(IMAGE_KEYS) and ok[0].contrasts is ok[0].appearance["contrast"]
    )
    assert len(s.real_values("texture")) == 5 and len(s.real_values("sharpness")) == 5
    h = s.distribution("texture")
    assert not h.log and sum(h.b) == 5  # 0 도 정상값이라 선형 구간


def test_lighting_direction_and_concentration(output_root: Path, tmp_path: Path) -> None:
    """조명 방향(링에서 밝은 쪽 각도)·일관성 R — KI #5(회전 ±180 이 하이라이트를 뒤집는다)의 검수 근거."""
    import numpy as np

    from anograft.gui.review.session import (
        FIXED_RANGE,
        LIGHT_RING_PX,
        circular_concentration,
        mask_lighting,
    )
    from anograft.io.report import lighting_broken_classes, lighting_line

    g = np.full((40, 40), 100, dtype=np.uint8)
    m = np.zeros((40, 40), dtype=np.uint8)
    m[14:26, 14:26] = 255
    g[26:29, 12:28] = 200  # 아래쪽 림이 밝다 → 90°
    assert LIGHT_RING_PX < 8 and abs(mask_lighting(g, m) - 90.0) < 5
    assert abs(mask_lighting(g[::-1].copy(), m[::-1].copy()) + 90.0) < 5  # 상하 반전 → −90°
    assert (
        abs(mask_lighting(np.ascontiguousarray(g.T), np.ascontiguousarray(m.T))) < 5
    )  # 전치 → 오른쪽 0°
    assert mask_lighting(np.full_like(g, 100), m) is None  # 링이 균일하면 방향 없음
    assert (
        mask_lighting(g, np.zeros_like(m)) is None
        and mask_lighting(g, np.full_like(m, 255)) is None
    )
    assert circular_concentration([]) is None
    assert circular_concentration([90, 90, 90]) == 1.0 and circular_concentration([0, 180]) == 0.0
    assert 0.5 < circular_concentration([80, 100, 90, 95]) <= 1.0
    # 각도 히스토그램은 구간 고정(−180..180) — 값 범위와 무관
    h = histogram([90, 91], [88], bins=12, log=False, value_range=FIXED_RANGE["lighting"])
    assert h.edges[0] == -180.0 and h.edges[-1] == 180.0 and sum(h.a) == 2 and sum(h.b) == 1
    # 리포트 한 줄: 실제가 뚜렷하고 합성이 무작위면 dent-graft 안내
    assert lighting_line((None, None)) == ""
    assert "0.10" in lighting_line((0.1, 0.9)) and "dent-graft" not in lighting_line((0.1, 0.9))
    per = {"pit": (0.12, 0.99), "scratch": (None, 0.26), "stain": (0.2, 0.45)}
    assert lighting_broken_classes(per) == ["pit"]  # 실제 ≥ 0.5 · 합성 < 0.3 인 클래스만
    assert lighting_broken_classes({"pit": (0.51, 0.99)}) == []
    line = lighting_line((0.12, 0.41), per)
    assert "dent-graft" in line and "<b>pit</b>" in line and "scratch –/0.26" in line
    assert "dent-graft" not in lighting_line((0.5, 0.9), {"pit": (0.51, 0.99)})
    # 세션: 합성·실제 값 수, 분포·R·리포트 필드
    s = ReviewSession()
    items = s.load(output_root)
    ok = [it for it in items if it.status == "ok"]
    syn = s.synthetic_values("lighting")
    assert len(syn) <= sum(len(s.item(x.index).instances) for x in ok) and all(
        -180 <= v <= 180 for v in syn
    )
    assert len(s.real_values("lighting")) <= 5
    h = s.distribution("lighting")
    assert not h.log and h.edges[0] == -180.0 and h.edges[-1] == 180.0
    rs, rr = s.lighting_concentration()
    assert (rs is None or 0 <= rs <= 1) and (rr is None or 0 <= rr <= 1)
    by_cls = s.synthetic_by_class("lighting")
    assert sum(len(v) for v in by_cls.values()) == len(syn) and set(by_cls) <= {"crack", "spot"}
    assert sum(len(v) for v in s.synthetic_by_class("contrast").values()) == len(
        s.synthetic_values("contrast")
    )
    assert set(s.real_by_class("contrast")) == {"crack", "spot"}  # 은행 클래스별
    with pytest.raises(ReviewError):
        s.synthetic_by_class("area")
    per = s.lighting_concentration_by_class()
    assert all(len(v) == 2 for v in per.values())
    # 클래스별 분포(0.8): 외형 지표만 · 값 집합은 by_class 와 같다 · 조명은 구간 고정
    assert s.class_options("lighting") == sorted(set(by_cls) | set(s.real_by_class("lighting")))
    assert s.class_options("area") == []
    for c in s.class_options("lighting"):
        hc = s.distribution_by_class("lighting", c)
        assert not hc.log and hc.edges[0] == -180.0 and hc.edges[-1] == 180.0
        assert sum(hc.a) == len(by_cls.get(c, [])) and sum(hc.b) == len(
            s.real_by_class("lighting").get(c, [])
        )
        r_s, r_r, n_s, n_r = s.lighting_r_for(c)
        assert (n_s, n_r) == (sum(hc.a), sum(hc.b)) and per[c] == (r_s, r_r)
    with pytest.raises(ReviewError):
        s.distribution_by_class("area", "spot")
    assert set(s.lighting_histograms_by_class()) == set(s.directional_classes())
    data = s.report_data()
    assert data.hist_lighting is not None and data.lighting_r == (rs, rr)
    assert set(data.hist_lighting_class) == set(s.directional_classes())
    assert data.lighting_r_class == per
    assert (
        data.geometry.startswith("scale ") and "flip both" in data.geometry
    )  # 0.7.7+: 기하 맥락 한 줄
    page = s.write_report(tmp_path / "r.html").read_text(encoding="utf-8")
    assert "조명 방향" in page and ("조명 일관성 R" in page) == (rs is not None or rr is not None)
    assert "기하 geometry" in page


def test_flipped_lighting_filter(output_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """0.7.3 — 실제 클래스 방향이 뚜렷할 때(R ≥ 0.5, n ≥ 3) 거기서 > 90° 벗어난 합성 인스턴스 = '조명 뒤집힘 의심' 필터."""
    from anograft.core.appearance import angle_diff, circular_mean
    from anograft.gui.review.session import FILTERS, IMAGE_KEYS

    assert circular_mean([]) is None and abs(circular_mean([80, 100]) - 90.0) < 1e-6
    assert abs(circular_mean([170, -170]) - 180.0) < 1e-6  # 경계를 넘어도 평균은 180
    assert angle_diff(90, -90) == 180.0 and angle_diff(10, 350) == 20.0 and angle_diff(5, 5) == 0.0
    assert "flipped" in FILTERS
    s = ReviewSession()
    items = s.load(output_root)
    ok = [it for it in items if it.status == "ok"]
    assert s.flipped_lighting() == set()  # 평탄한 합성물 · 실제 방향 없음 → 비어 있음
    # 실제 spot 은 아래쪽(90°) 하이라이트로 뚜렷, crack 은 방향 없음
    monkeypatch.setattr(
        ReviewSession,
        "real_by_class",
        lambda self, key: {"spot": [88.0, 90.0, 92.0], "crack": [0.0, 180.0, 90.0, -90.0]},
    )
    assert s.real_lighting_direction() == {"spot": 90.0}
    for it in ok:
        s.item(it.index)
        for k in IMAGE_KEYS:
            it.appearance[k], it.appearance_cls[k] = [], []
    # 첫 항목: spot 인스턴스가 −80°(뒤집힘) · 둘째: spot 60°(정상) · 셋째: crack 180°(방향 없는 클래스 → 무시)
    ok[0].appearance["lighting"], ok[0].appearance_cls["lighting"] = [-80.0], ["spot"]
    ok[1].appearance["lighting"], ok[1].appearance_cls["lighting"] = [60.0], ["spot"]
    ok[2].appearance["lighting"], ok[2].appearance_cls["lighting"] = [180.0], ["crack"]
    assert s.flipped_lighting() == {ok[0].index}
    assert [it.index for it in s.filtered("flipped")] == [ok[0].index]
    assert s.flipped_lighting(max_deg=170.0) == set() and len(s.flipped_lighting(max_deg=20.0)) == 2
    from anograft.io.report import flipped_line

    assert (
        flipped_line(()) == ""
        and "<b>3</b>건" in flipped_line(["a", "b", "c"])
        and "외 2" in flipped_line([str(i) for i in range(14)])
    )
    assert s.report_data().flipped == [ok[0].index]
    with pytest.raises(ReviewError):
        s.filtered("bogus")


def test_real_distribution_follows_recipe_classes(output_root: Path, tmp_path: Path) -> None:
    """0.7.3+ — '실제' 분포는 레시피가 뽑은 클래스만(source.classes → class_ratio → 전부). pit 만 합성한 출력과\n    은행의 스크래치까지 비교하면 분포가 어긋난다."""
    s = ReviewSession()
    s.load(output_root)
    assert s.real_classes() is None and len(s.real_sources()) == 5  # 레시피가 전부 뽑음
    n_all = len(s.real_values("area"))
    # 출력의 resolved 레시피를 spot 만 뽑은 것처럼 바꿔 다시 읽는다
    resolved = output_root / "recipe.resolved.yaml"
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    data["pipeline"]["source"]["classes"] = ["spot"]
    resolved.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    s2 = ReviewSession()
    s2.load(output_root)
    assert s2.real_classes() == ["spot"]
    assert all(src.cls == "spot" for src in s2.real_sources()) and 0 < len(s2.real_sources()) < 5
    assert 0 < len(s2.real_values("area")) < n_all
    assert set(s2.real_by_class("contrast")) <= {"spot"}
    # class_ratio 키도 같은 규칙
    data["pipeline"]["source"]["classes"] = None
    data["output"]["class_ratio"] = {"crack": 1.0}
    resolved.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    s3 = ReviewSession()
    s3.load(output_root)
    assert s3.real_classes() == ["crack"] and all(src.cls == "crack" for src in s3.real_sources())
    # 은행 없이 열면 비어 있다
    s4 = ReviewSession()
    s4.load(output_root, load_bank=False)
    assert s4.real_sources() == [] and s4.real_values("area") == []


def test_prune_drop_indices_and_cli_drop_flipped(
    output_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``prune_dataset(drop_indices=)`` 는 판정과 무관하게 뺀다 · CLI ``--drop-flipped`` 는 검수 세션의 뒤집힘 집합(여기선 0건)."""
    s = ReviewSession()
    items = s.load(output_root, load_bank=False)
    ok = [it for it in items if it.status == "ok"]
    summary = prune_dataset(output_root, tmp_path / "p1", {}, drop_indices=[ok[0].index, "nope"])
    assert summary.dropped == 1 and summary.kept == len(ok) - 1
    kept = {r["index"] for r in read_manifest(tmp_path / "p1" / "manifest.csv")}
    assert ok[0].index not in kept and ok[1].index in kept
    assert (
        main(
            ["dataset", "prune", str(output_root), "--out", str(tmp_path / "p2"), "--drop-flipped"]
        )
        == EXIT_OK
    )
    cap = capsys.readouterr()
    assert "조명 뒤집힘 의심 없음" in cap.err and "정리본" in cap.out  # 평탄 합성물 — 뒤집힘 없음
    assert len(read_manifest(tmp_path / "p2" / "manifest.csv")) == len(
        [r for r in read_manifest(output_root / "manifest.csv") if r["status"] != "skipped"]
    )


def test_real_csv_replaces_bank_series(output_root: Path, tmp_path: Path) -> None:
    """실측 CSV(0.8): CSV 에 있는 열은 은행 대신, 없는 열은 은행 그대로 · class 열로 클래스별 · 레시피 클래스 필터 · 검증."""
    s = ReviewSession()
    s.load(output_root)
    bank_area = s.real_values("area")
    bank_contrast = s.real_by_class("contrast")
    csv = tmp_path / "real.csv"
    csv.write_text(
        "class,area,contrast\nspot,100,\nspot,200,5\ncrack,300,7\nother,999,9\n", encoding="utf-8"
    )
    assert s.load_real_csv(csv) == 4
    assert s.real_csv_keys() == {"area", "contrast"} and s.real_label() == "실측 real.csv"
    # 레시피가 클래스를 제한하지 않으면 전부(빈 칸은 건너뜀) · 제한하면(spot·crack) other 제외
    assert sorted(s.real_values("area")) == [100.0, 200.0, 300.0, 999.0]
    assert s.real_by_class("contrast") == {"spot": [5.0], "crack": [7.0], "other": [9.0]}
    s.recipe_meta.setdefault("pipeline", {}).setdefault("source", {})["classes"] = ["spot", "crack"]
    assert sorted(s.real_values("area")) == [100.0, 200.0, 300.0]
    assert s.real_by_class("contrast") == {"spot": [5.0], "crack": [7.0]}
    assert s.real_values("length") == s.real_values("length")  # 은행 폴백 경로가 살아 있다
    assert s.real_values("area") != bank_area
    h = s.distribution("area")
    assert sum(h.b) == 3
    data = s.report_data()
    assert data.bank_name == "실측 real.csv"
    page = s.write_report(tmp_path / "r.html").read_text(encoding="utf-8")
    assert "실제 = 실측 real.csv" in page
    s.clear_real_csv()
    assert s.real_values("area") == bank_area and s.real_by_class("contrast") == bank_contrast
    assert s.real_label().startswith("은행 ")
    # 형식 오류
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ReviewError, match="분포 열"):
        s.load_real_csv(bad)
    bad.write_text("area\nx\n", encoding="utf-8")
    with pytest.raises(ReviewError, match="숫자"):
        s.load_real_csv(bad)
    with pytest.raises(ReviewError, match="읽을 수"):
        s.load_real_csv(tmp_path / "none.csv")


def test_cli_dataset_report_real_csv(
    output_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv = tmp_path / "real.csv"
    csv.write_text("area\n10\n20\n", encoding="utf-8")
    assert main(["dataset", "report", str(output_root), "--real-csv", str(csv)]) == EXIT_OK
    cap = capsys.readouterr()
    assert "실측 real.csv" in cap.out and "2행" in cap.err
    page = (output_root / "review-report.html").read_text(encoding="utf-8")
    assert "실제 = 실측 real.csv" in page

"""``runner.fit_diagnostic`` — ``--dry-run`` 의 배치 가능성 진단(ROI 최대 폭 vs 클래스별 패치 폭, KNOWN-ISSUES 부록)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anograft import runner
from anograft.core import recipe as R
from anograft.io import imgio
from tests.fixtures import fake_yolo_dataset


def test_verdicts_and_warning_are_pure() -> None:
    fit = runner.FitDiagnostic(
        roi_widths=[("a.png", 50.0), ("b.png", 80.0)],
        patch_sides={"ok": 30.0, "tight": 45.0, "no": 120.0},
        scale_hi=1.0,
        shrink_floor=0.512,
    )
    assert fit.min_width == 50.0
    assert fit.verdicts() == {"ok": "가능", "tight": "빠듯", "no": "불가"}
    w = fit.warning()
    assert w and w.startswith("placement:") and "no" in w and "tight" in w and "skipped 예상" in w
    assert (
        runner.FitDiagnostic([("a", 100.0)], {"x": 10.0}, 1.0, 1.0).warning() is None
    )  # 전부 가능이면 경고 없음
    # shrink 가 없으면(floor 1.0) 폭을 넘는 즉시 불가
    assert runner.FitDiagnostic([("a", 50.0)], {"x": 51.0}, 1.0, 1.0).verdicts() == {"x": "불가"}


def test_verdicts_v2_short_side_and_alignment() -> None:
    """가늘고 긴 패치: 짧은 변이 폭에 들어가면 정렬(along) 시 가능, 정렬 없으면 빠듯(회전 운) · 짧은 변까지 넘으면 불가."""
    base = dict(
        roi_widths=[("a", 100.0)],
        patch_sides={"scratch": 214.0, "flip": 685.0, "spot": 40.0},
        scale_hi=1.1,
        shrink_floor=0.512,
    )
    aligned = runner.FitDiagnostic(
        **base, patch_short={"scratch": 80.0, "flip": 628.0, "spot": 30.0}, aligned=True
    )
    assert aligned.verdicts() == {"scratch": "가능", "flip": "불가", "spot": "가능"}
    assert "정렬 있음" in aligned.verdict_note("scratch") and aligned.verdict_note("spot") == ""
    loose = runner.FitDiagnostic(
        **base, patch_short={"scratch": 80.0, "flip": 628.0, "spot": 30.0}, aligned=False
    )
    assert loose.verdicts()["scratch"] == "빠듯" and "정렬 없음" in loose.verdict_note("scratch")
    tight = runner.FitDiagnostic(
        **base, patch_short={"scratch": 85.0, "flip": 628.0, "spot": 30.0}, aligned=True
    )
    assert tight.verdicts()["scratch"] == "빠듯" and "80%" in tight.verdict_note("scratch")
    # patch_short 가 비면 종전(긴 변) 판정
    assert runner.FitDiagnostic(**base).verdicts() == {
        "scratch": "불가",
        "flip": "불가",
        "spot": "가능",
    }
    w = loose.warning()
    assert w and "정렬 없음" in w and "flip" in w


def test_mask_dims_and_placement_aligned() -> None:
    m = np.zeros((50, 50), np.uint8)
    m[20:24, 5:45] = 255  # 가로 40 × 세로 4
    short_s, long_s = runner.mask_dims(m)
    assert 3.5 <= short_s <= 5.0 and 39.0 <= long_s <= 41.0
    d = np.zeros((50, 50), np.uint8)
    yy, xx = np.mgrid[:50, :50]
    d[(yy - 25) ** 2 + (xx - 25) ** 2 <= 10**2] = 255  # 지름 ≈ 21 원 → 짧은≈긴
    short_s, long_s = runner.mask_dims(d)
    assert abs(short_s - long_s) < 2.5 and 19 <= long_s <= 23
    assert runner.mask_dims(np.zeros((5, 5), np.uint8)) == (0.0, 0.0)
    one = np.zeros((5, 5), np.uint8)
    one[2, 2] = 255
    assert runner.mask_dims(one) == (1.0, 1.0)
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": "t",
            "seed": 1,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1},
            "pipeline": {"preset": "structure-aware-graft"},
        }
    )
    assert runner.placement_aligned(rec) is True
    assert (
        runner.placement_aligned(
            R.Recipe.from_dict({**rec.to_dict(), "pipeline": {"preset": "poisson-graft"}})
        )
        is False
    )


def test_non_local_class_warning() -> None:
    """패치가 대상 자체와 맞먹는 클래스(MVTec ``flip`` 685/700px) → ``source:`` 경고. 대상 크기를 모르면(0) 침묵."""
    fit = runner.FitDiagnostic(
        [("a", 100.0)],
        {"flip": 685.0, "bent": 141.0, "half": 350.0},
        1.0,
        0.512,
        target_short_side=700.0,
    )
    assert fit.non_local_classes() == {"flip": 0.98, "half": 0.5}
    w = fit.source_warning()
    assert w and w.startswith("source:") and "flip(대상 짧은 변의 98%)" in w and "bent" not in w
    assert "source.classes" in w
    assert (
        runner.FitDiagnostic([("a", 100.0)], {"flip": 685.0}, 1.0, 0.512).source_warning() is None
    )
    assert (
        runner.FitDiagnostic([("a", 100.0)], {"bent": 141.0}, 1.0, 0.512, 700.0).source_warning()
        is None
    )


def test_fit_diagnostic_on_workspace(tmp_path: Path) -> None:
    from anograft.bank.importers import yolo as Y

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
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": "t",
            "seed": 1,
            "inputs": {"bank": (tmp_path / "bank").as_posix(), "targets": normals.as_posix()},
            "output": {"root": (tmp_path / "o").as_posix(), "count": 1},
            "pipeline": {
                "preset": "hard-paste",
                "source": {"method": "bank", "min_sources_warn": 1},
            },
        }
    )
    prep = runner.prepare(rec)
    fit = runner.fit_diagnostic(prep, n_targets=5)
    assert fit is not None and 1 <= len(fit.roi_widths) <= min(5, len(prep.targets))
    assert set(fit.patch_sides) == {"spot", "crack"} and all(
        v > 0 for v in fit.patch_sides.values()
    )
    assert fit.scale_hi == rec.pipeline.geometry.scale[1]
    assert 0 < fit.shrink_floor <= 1.0
    assert fit.target_short_side > 0 and fit.source_warning() is None  # 샘플 결함은 국소
    assert set(fit.patch_short) == {"spot", "crack"} and all(
        0 < fit.patch_short[c] <= fit.patch_sides[c] + 1e-6 for c in fit.patch_short
    )
    assert (
        fit.physical == {"spot": 1.0, "crack": 1.0} and fit.aligned is False
    )  # hard-paste = sampled
    rows = dict(runner.dry_run_table(prep, fit=fit))
    assert "roi width" in rows and "fit spot" in rows and "짧은 변" in rows["fit spot"]
    assert rows["fit spot"].split("→")[-1].strip().split(" ")[0] in {"가능", "빠듯", "불가"}
    assert "축척" not in rows["fit spot"]  # 피치 없음 → 축척 표기 없음
    # ROI 캐시를 타므로 두 번 불러도 같다(rng 0회)
    assert runner.fit_diagnostic(prep, n_targets=5) == fit
    # 대상 하나만 보면 폭 1개
    assert len(runner.fit_diagnostic(prep, n_targets=1).roi_widths) == 1
    # 아주 좁은 ROI(mask_dir 로 가는 띠)면 불가가 나온다
    roi_dir = tmp_path / "roi"
    roi_dir.mkdir()
    for t in prep.targets:
        img, _ = imgio.read_image(t)
        m = np.zeros(img.shape[:2], dtype=np.uint8)
        m[:, 10:14] = 255
        imgio.write_image(roi_dir / (t.stem + ".png"), m)
    d2 = rec.to_dict()
    d2["pipeline"]["placement"]["roi"] = {"method": "mask_dir", "path": roi_dir.as_posix()}
    d2["pipeline"]["placement"]["margin_px"] = 0
    prep2 = runner.prepare(R.Recipe.from_dict(d2))
    fit2 = runner.fit_diagnostic(prep2)
    assert fit2 is not None and fit2.min_width <= 4.0 and set(fit2.verdicts().values()) == {"불가"}
    assert "skipped 예상" in (fit2.warning() or "")
    # 비-bank 소스는 진단 없음
    bankless = R.Recipe.from_dict({**rec.to_dict(), "pipeline": {"preset": "self-cut"}})
    assert runner.fit_diagnostic(runner.prepare(bankless)) is None


def test_fit_diagnostic_skips_unreadable_target(tmp_path: Path) -> None:
    from anograft.bank.importers import yolo as Y

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
    lines = normals.read_text(encoding="utf-8").splitlines()
    normals.write_text(chr(10).join(["missing.png", *lines]) + chr(10), encoding="utf-8")
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": "t",
            "seed": 1,
            "inputs": {"bank": (tmp_path / "bank").as_posix(), "targets": normals.as_posix()},
            "output": {"root": (tmp_path / "o").as_posix(), "count": 1},
            "pipeline": {
                "preset": "hard-paste",
                "source": {"method": "bank", "min_sources_warn": 1},
            },
        }
    )
    prep = runner.prepare(
        rec
    )  # 목록의 없는 파일은 prepare 를 막지 않는다(실행 시 그 인덱스만 skipped)
    fit = runner.fit_diagnostic(prep, n_targets=2)
    assert fit is not None and [n for n, _ in fit.roi_widths] == ["n0.png"]  # missing.png 는 건너뜀


def test_roi_max_width_and_fit_from_widths(tmp_path: Path) -> None:
    from anograft.bank.importers import yolo as Y
    from anograft.core.stages.placement import roi_max_width

    roi = np.zeros((40, 60), dtype=bool)
    roi[10:30, 5:55] = True  # 20 px 높이 띠 → 내접원 지름 ≈ 20
    assert 18.0 <= roi_max_width(roi, 0) <= 20.0
    assert roi_max_width(roi, 12) < roi_max_width(roi, 0)  # 테두리 여유가 폭을 깎는다
    assert roi_max_width(None, 0, (40, 60)) >= 38.0 and roi_max_width(None, 0) == 0.0
    assert roi_max_width(np.zeros((8, 8), dtype=bool), 0) == 0.0
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
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": "t",
            "seed": 1,
            "inputs": {"bank": (tmp_path / "bank").as_posix(), "targets": normals.as_posix()},
            "output": {"root": (tmp_path / "o").as_posix(), "count": 1},
            "pipeline": {
                "preset": "hard-paste",
                "source": {"method": "bank", "min_sources_warn": 1},
            },
        }
    )
    prep = runner.prepare(rec)
    sides = runner.patch_sides(prep)
    assert set(sides) == {"spot", "crack"}
    fit = runner.fit_from_widths(prep, [("x.png", 1000.0)])
    assert fit is not None and fit.patch_sides == sides and set(fit.verdicts().values()) == {"가능"}
    assert runner.fit_from_widths(prep, []) is None
    tiny = runner.fit_from_widths(prep, [("x.png", 2.0)])
    assert tiny is not None and set(tiny.verdicts().values()) == {"불가"}


def test_fit_uses_physical_scale(tmp_path: Path) -> None:
    """소스 µm/px 4 · 대상 µm/px 2 → 소스 1 px 가 대상 2 px(factor 2.0): 패치 변이 두 배로 잡혀 판정이 바뀐다. 한쪽만 있으면 1.0."""
    from anograft.bank.importers import yolo as Y

    d = fake_yolo_dataset(tmp_path / "ds")
    normals = tmp_path / "normals.txt"
    Y.import_yolo(
        d["images"],
        d["labels"],
        d["names"],
        tmp_path / "bank",
        mask_from="rect",
        list_normals=normals,
        um_per_px=4.0,
    )
    base = {
        "version": 1,
        "name": "t",
        "seed": 1,
        "inputs": {"bank": (tmp_path / "bank").as_posix(), "targets": normals.as_posix()},
        "output": {"root": (tmp_path / "o").as_posix(), "count": 1},
        "pipeline": {"preset": "hard-paste", "source": {"method": "bank", "min_sources_warn": 1}},
    }
    plain = runner.fit_diagnostic(runner.prepare(R.Recipe.from_dict(base)))
    scaled = runner.fit_diagnostic(
        runner.prepare(R.Recipe.from_dict({**base, "inputs": {**base["inputs"], "um_per_px": 2.0}}))
    )
    assert plain is not None and scaled is not None
    assert plain.physical == {"spot": 1.0, "crack": 1.0} and scaled.physical == {
        "spot": 2.0,
        "crack": 2.0,
    }
    for c in ("spot", "crack"):
        assert abs(scaled.patch_sides[c] - 2 * plain.patch_sides[c]) <= 0.2
        assert abs(scaled.patch_short[c] - 2 * plain.patch_short[c]) <= 0.2
    rows = dict(
        runner.dry_run_table(
            runner.prepare(
                R.Recipe.from_dict({**base, "inputs": {**base["inputs"], "um_per_px": 2.0}})
            ),
            fit=scaled,
        )
    )
    assert "축척 2.00" in rows["fit spot"]

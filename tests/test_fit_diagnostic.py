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
    rows = dict(runner.dry_run_table(prep, fit=fit))
    assert (
        "roi width" in rows
        and "fit spot" in rows
        and rows["fit spot"].split("→")[-1].strip() in {"가능", "빠듯", "불가"}
    )
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

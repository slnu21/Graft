"""v0.6 ``mask-confidence``(KNOWN-ISSUES #3) — 박스→마스크 추정의 타당성 점수 ``mask_confidence`` · 임포터/은행 메타 왕복 ·
``bank ls``/``bank preview`` 표시 · ``runner.confidence_warning`` · 라벨 탭 초안 저장 시 점수 기록."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anograft import runner
from anograft.bank import Bank
from anograft.bank.importers import yolo as Y
from anograft.bank.mask_from_box import (
    LOW_CONFIDENCE,
    mask_confidence,
    mask_from_box,
)
from anograft.cli import EXIT_OK, main
from anograft.core import recipe as R
from anograft.core.types import DefectSource
from anograft.gui.label.session import LabelSession
from anograft.preview import source_tile
from tests.fixtures import blob_image, fake_yolo_dataset, line_defect

BOX = (30, 30, 36, 36)


def _texture(seed: int = 0, mean: float = 180.0, std: float = 12.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.normal(mean, std, (96, 96)).clip(0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


# --- 점수 ---------------------------------------------------------------------


def test_clean_blob_scores_high_and_texture_pickup_scores_low() -> None:
    img = blob_image(96, [(48, 48, 12)])
    m, used = mask_from_box(img, BOX, "grabcut", seed=1)
    c = mask_confidence(img, m, BOX)
    assert used == "grabcut" and c.score >= 0.9 and c.flags == ()
    assert c.separation > 5 and c.touch == 0.0 and c.n_components == 1
    tex = _texture()
    m2, _ = mask_from_box(tex, BOX, "grabcut", seed=1)
    c2 = mask_confidence(tex, m2, BOX)
    assert c2.score < LOW_CONFIDENCE and "low-contrast" in c2.flags


def test_flags_box_edge_fragmented_saturated_empty() -> None:
    img = blob_image(96, [(48, 48, 12)])
    rect, _ = mask_from_box(img, BOX, "rect")
    c = mask_confidence(img, rect, BOX)
    assert c.touch == 1.0 and "box-edge" in c.flags and "area-out" in c.flags and c.score <= 0.5
    # 조각난 마스크 — 점 6개
    frag = np.zeros((96, 96), dtype=np.uint8)
    for i in range(6):
        frag[34 + i * 5, 34 + i * 5] = 255
    c = mask_confidence(img, frag, BOX)
    assert c.n_components == 6 and "fragmented" in c.flags
    # 포화 — 흰 얼룩(255) 위 마스크
    sat = np.full((96, 96, 3), 200, dtype=np.uint8)
    sat[40:56, 40:56] = 255
    m = np.zeros((96, 96), dtype=np.uint8)
    m[40:56, 40:56] = 255
    c = mask_confidence(sat, m, BOX)
    assert c.saturated == 1.0 and "saturated" in c.flags
    # 빈 마스크 · 박스 밖
    c = mask_confidence(img, np.zeros((96, 96), dtype=np.uint8), BOX)
    assert c.score == 0.0 and c.flags == ("empty",)
    assert mask_confidence(img, m, (200, 200, 10, 10)).flags == ("empty",)


def test_confidence_is_deterministic_and_within_unit_interval() -> None:
    tex = _texture(3)
    for method in ("grabcut", "otsu", "ellipse", "rect"):
        m, _ = mask_from_box(tex, BOX, method, seed=7)
        a, b = mask_confidence(tex, m, BOX), mask_confidence(tex, m, BOX)
        assert a == b and 0.0 <= a.score <= 1.0


# --- 임포터 → 은행 메타 → 로더 --------------------------------------------------------


def test_import_yolo_records_confidence_and_bank_reports_low(tmp_path: Path) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    res = Y.import_yolo(
        d["images"], d["labels"], d["names"], tmp_path / "bank", mask_from="grabcut"
    )
    bank = Bank.load(tmp_path / "bank")
    boxes = [s for s in bank.sources() if s.mask_origin.startswith("yolo-box:")]
    polys = [s for s in bank.sources() if s.mask_origin == "yolo-polygon"]
    assert boxes and all(s.confidence is not None and 0 <= s.confidence <= 1 for s in boxes)
    assert polys and all(s.confidence is None and s.flags == () for s in polys)
    assert res.low_confidence == len(bank.low_confidence())
    rows = {r.cls: r for r in bank.summary()}
    assert sum(r.low_conf for r in rows.values()) == len(bank.low_confidence())
    # 메타 JSON 에 남는다
    import json

    meta = json.loads(
        (tmp_path / "bank" / boxes[0].cls / f"{boxes[0].id.split('/')[1]}.json").read_text("utf-8")
    )
    assert "confidence" in meta and "flags" in meta
    # rect 로 임포트하면 전부 box-edge → 0.5 이하 · 저신뢰 판단은 < 0.5 만
    Y.import_yolo(d["images"], d["labels"], d["names"], tmp_path / "bank-rect", mask_from="rect")
    b2 = Bank.load(tmp_path / "bank-rect")
    assert all("box-edge" in s.flags for s in b2.sources() if s.confidence is not None)


def _bank_with(scores: list[float | None]) -> Bank:
    srcs = []
    for i, sc in enumerate(scores):
        s = line_defect(12 + i, 3)
        origin = "yolo-box:grabcut" if sc is not None else "png"
        srcs.append(
            DefectSource(
                f"scratch/{i:03d}",
                "scratch",
                s.image,
                s.mask,
                None,
                (),
                "",
                origin,
                sc,
                ("low-contrast",) if sc is not None and sc < 0.5 else (),
            )
        )
    return Bank.from_sources(srcs, classes=["scratch"])


def _recipe() -> R.Recipe:
    return R.Recipe.from_dict(
        {
            "version": 1,
            "name": "t",
            "seed": 1,
            "inputs": {"bank": "b", "targets": "n"},
            "output": {"root": "o", "count": 1},
            "pipeline": {"preset": "hard-paste"},
        }
    )


def test_runner_confidence_warning_levels() -> None:
    assert runner.confidence_warning(_recipe(), _bank_with([0.9, None, 0.8])) is None
    mild = runner.confidence_warning(_recipe(), _bank_with([0.9, 0.2, 0.8, None]))
    assert mild and mild.startswith("마스크 신뢰도 낮은 조각 1/3개") and "scratch/001" in mild
    strong = runner.confidence_warning(_recipe(), _bank_with([0.1, 0.2, 0.8]))
    assert strong and "절반이 넘습니다" in strong and "67%" in strong
    empty = Bank.from_sources([], name="(없음)")
    assert runner.confidence_warning(_recipe(), empty) is None


def test_source_tile_marks_low_confidence_with_red_border() -> None:
    s = line_defect(16, 4)
    ok = source_tile(s.image, s.mask, "scratch/a", "yolo-box:grabcut", tile=64, confidence=0.95)
    low = source_tile(s.image, s.mask, "scratch/b", "yolo-box:grabcut", tile=64, confidence=0.2)
    none = source_tile(s.image, s.mask, "scratch/c", "png", tile=64)
    assert tuple(int(v) for v in low[0, 32]) == (60, 60, 230)  # 위쪽 테두리가 빨강
    assert tuple(int(v) for v in ok[0, 32]) != (60, 60, 230)
    assert not np.array_equal(ok, none) and not np.array_equal(ok, low)


def test_cli_bank_ls_and_preview_show_low_confidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    bank = tmp_path / "bank"
    assert (
        main(
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
                "ellipse",
            ]
        )
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "저신뢰(<0.5)" in out
    assert main(["bank", "ls", str(bank)]) == EXIT_OK
    cap = capsys.readouterr()
    assert "lowconf" in cap.out
    assert main(["bank", "preview", str(bank), "--out", str(tmp_path / "g.png")]) == EXIT_OK
    assert "저신뢰" in capsys.readouterr().out


# --- 라벨 탭 초안 → 저장 시 점수 ------------------------------------------------------


def test_label_session_draft_confidence_saved_only_when_untouched(tmp_path: Path) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    s = LabelSession()
    s.load_image(d["images"] / "d1.png")
    s.load_yolo_draft()
    s.apply_draft(0, "grabcut")
    assert len(s.draft_confidence) == 1
    s.save_to_bank(tmp_path / "bank", "spot")
    src = Bank.load(tmp_path / "bank").by_class("spot")[0]
    assert src.mask_origin == "yolo-box:grabcut" and src.confidence == s.draft_confidence[0].score
    # 손대면 manual:mixed → 점수 없음
    s.apply_draft(0, "grabcut")
    s.stroke([(5.0, 5.0)], 2)
    s.save_to_bank(tmp_path / "bank2", "spot")
    src2 = Bank.load(tmp_path / "bank2").by_class("spot")[0]
    assert src2.mask_origin == "manual:mixed" and src2.confidence is None


def test_source_tile_lighting_arrow() -> None:
    """0.7.3 — ``lighting_deg`` 가 있으면 오른쪽 위에 밝은 쪽 화살표(노랑). 없으면 종전과 동일."""
    from anograft.preview import LIGHT_ARROW, draw_lighting_arrow

    s = line_defect(16, 4)
    base = source_tile(s.image, s.mask, "scratch/a", "png", tile=96)
    down = source_tile(s.image, s.mask, "scratch/a", "png", tile=96, lighting_deg=90.0)
    right = source_tile(s.image, s.mask, "scratch/a", "png", tile=96, lighting_deg=0.0)
    assert not np.array_equal(base, down) and not np.array_equal(down, right)
    corner = down[:24, 96 - 24 :]
    assert (corner == np.array(LIGHT_ARROW, dtype=np.uint8)).all(axis=2).any()  # 화살표 색이 구석에
    assert not (base[:24, 96 - 24 :] == np.array(LIGHT_ARROW, dtype=np.uint8)).all(axis=2).any()
    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
    draw_lighting_arrow(canvas, 90.0, 64)
    ys, xs = np.nonzero((canvas == np.array(LIGHT_ARROW, dtype=np.uint8)).all(axis=2))
    assert (
        len(ys) and ys.max() > ys.min() and xs.max() - xs.min() <= 6
    )  # 아래를 가리키는 세로 화살표


def test_tile_id_label_keeps_tail_number() -> None:
    from anograft.preview import tile_id_label

    assert tile_id_label("bent/metal_nut-bent-000", 20) == "metal_nut-bent-000"
    assert tile_id_label("bent/metal_nut-bent-000", 10) == "~-bent-000"
    assert tile_id_label("a/b", 6) == "b" and tile_id_label("noslash-12", 6) == "~sh-12"

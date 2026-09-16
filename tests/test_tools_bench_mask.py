"""``tools/bench_mask_from_box.py`` — 순수 부분(IoU · 박스 패딩 · 성분 · 요약) + 작은 합성 이미지 한 바퀴."""

from __future__ import annotations

import numpy as np

from tests.fixtures import load_tool

bench = load_tool("bench_mask_from_box")


def test_iou_and_pad_box() -> None:
    a = np.zeros((10, 10), np.uint8)
    b = np.zeros((10, 10), np.uint8)
    a[2:6, 2:6] = 255
    b[4:8, 4:8] = 255
    assert (
        bench.iou(a, a) == 1.0
        and bench.iou(a, b) == 4 / 28
        and bench.iou(a, np.zeros_like(a)) == 0.0
    )
    assert bench.iou(np.zeros_like(a), np.zeros_like(a)) == 1.0
    assert bench.pad_box((10, 10, 20, 10), 0.1, (100, 100)) == (8, 9, 24, 12)
    assert bench.pad_box((0, 0, 20, 10), 0.5, (12, 15)) == (0, 0, 15, 12)  # 이미지 안으로


def test_components_sorted_and_filtered() -> None:
    m = np.zeros((20, 20), np.uint8)
    m[1:3, 1:3] = 255  # 4 px
    m[5:15, 5:15] = 255  # 100 px
    comps = bench.components(m, min_area=5)
    assert (
        len(comps) == 1 and comps[0][1] == (5, 5, 10, 10) and np.count_nonzero(comps[0][0]) == 100
    )
    assert [b for _, b in bench.components(m, min_area=1)] == [(5, 5, 10, 10), (1, 1, 2, 2)]


def test_summarize_rows() -> None:
    def rec(cls: str, chain_iou: float, conf: float, used: str = "grabcut") -> bench.Record:
        return bench.Record(
            "d",
            cls,
            "i.png",
            0,
            (0, 0, 1, 1),
            dict.fromkeys(bench.METHODS, chain_iou),
            used,
            chain_iou,
            conf,
            (),
        )

    rows = bench.summarize([rec("a", 0.8, 0.9), rec("a", 0.1, 0.2), rec("b", 0.1, 0.9, "otsu")])
    by = {(r["dataset"], r["class"]): r for r in rows}
    a = by[("d", "a")]
    assert a["n"] == 2 and a["fail_rate"] == 0.5 and a["lowconf_rate"] == 0.5
    assert a["lowconf_precision"] == 1.0 and a["lowconf_recall"] == 1.0
    b = by[("d", "b")]
    assert b["fail_rate"] == 1.0 and b["lowconf_rate"] == 0.0 and b["lowconf_precision"] is None
    assert b["lowconf_recall"] == 0.0 and b["chain_used"] == {"otsu": 1}
    tot = by[("(전체)", "")]
    assert tot["n"] == 3 and tot["iou_chain"] == 0.1
    md = bench.to_markdown(rows, box_pad=0.1)
    assert md.count("\n|") >= 4 and "10%" in md


def test_bench_instance_on_synthetic_blob() -> None:
    """밝은 배경에 어두운 원 → grabcut/otsu 가 원을 잡아 IoU 가 rect 보다 높고 confidence 가 높다."""
    img = np.full((64, 64, 3), 200, np.uint8)
    gt = np.zeros((64, 64), np.uint8)
    yy, xx = np.mgrid[:64, :64]
    disk = (yy - 32) ** 2 + (xx - 32) ** 2 <= 12**2
    img[disk] = 40
    gt[disk] = 255
    (comp, bbox), *_ = bench.components(gt, 16)
    box = bench.pad_box(bbox, 0.1, img.shape)
    ious, used, chain_iou, conf, flags = bench.bench_instance(img, comp, box, seed=1)
    assert used in {"grabcut", "otsu"} and chain_iou > 0.8 and ious["rect"] < chain_iou
    assert conf >= bench.LOW_CONFIDENCE and "low-contrast" not in flags

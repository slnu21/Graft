"""3단계 placement(sampled) — 설계 §11: 채택된 배치는 항상 마스크 ⊆ ROI ∧ 테두리 여유 · ``edge`` 평균 경계거리 < ``uniform``
< ``center``(고정 시드, 200회) · ROI가 너무 작으면 shrink → skip 경로와 로그. 추가로 기존 결함과 겹침 금지 · ``offset`` 규약 ·
실제 roi→geometry→placement 사슬을 파이프라인에 꽂은 통합 테스트."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anograft.core import recipe as R
from anograft.core.pipeline import Pipeline
from anograft.core.roi import distance_to_edge, roi_otsu
from anograft.core.stages.placement import (
    SampledPlacement,
    allowed_centers,
    candidate_cdf,
    draw_index,
    shrink_patch,
)
from anograft.core.types import Context, PlacedDefect, TargetImage
from tests.fixtures import context, disk_image, disk_target, line_defect, memory_bank, pipeline_deps


def _square_patch(size: int = 9, inner: int = 5) -> tuple[np.ndarray, np.ndarray]:
    off = (size - inner) // 2
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[off : off + inner, off : off + inner] = 255
    return np.full((size, size, 3), 230, dtype=np.uint8), mask


def _disk_roi(size: int = 128, erode: int = 0) -> np.ndarray:
    return roi_otsu(disk_image(size), "auto", erode).roi


def _stage(**kw) -> SampledPlacement:
    return SampledPlacement(R.SampledPlacementConfig(**kw), {})


def _ctx(roi: np.ndarray, patch: np.ndarray, mask: np.ndarray, seed: int = 0, **kw) -> Context:
    t = disk_target(roi.shape[0])
    return context(t, seed=seed, roi=roi, patch=patch, patch_mask=mask, **kw)


# ---------------------------------------------------------------------------
# 순수 함수
# ---------------------------------------------------------------------------


def test_allowed_centers_applies_margin_and_existing() -> None:
    roi = np.ones((20, 20), dtype=bool)
    a = allowed_centers(roi, 4, None)
    assert a.sum() == 12 * 12 and not a[3, :].any() and a[4, 4]
    existing = np.zeros((20, 20), dtype=bool)
    existing[8:12, 8:12] = True
    a2 = allowed_centers(roi, 4, existing)
    assert a2.sum() == 12 * 12 - 16 and not a2[9, 9]
    assert allowed_centers(roi, 10, None).sum() == 0  # 여유가 이미지를 다 먹으면 후보 0


def test_candidate_cdf_and_draw_index_consume_one_random_each() -> None:
    allowed = np.zeros((9, 9), dtype=bool)
    allowed[1:8, 1:8] = True
    assert candidate_cdf(allowed, "uniform") is None
    cdf_e, cdf_c = candidate_cdf(allowed, "edge"), candidate_cdf(allowed, "center")
    assert cdf_e.shape == cdf_c.shape == (49,)
    assert np.all(np.diff(cdf_e) > 0) and np.all(np.diff(cdf_c) > 0)
    for cdf in (None, cdf_e):
        a, b = np.random.default_rng(3), np.random.default_rng(3)
        idx = [draw_index(a, 49, cdf) for _ in range(5)]
        assert all(0 <= i < 49 for i in idx)
        b.random(5) if cdf is not None else b.integers(49, size=5)
        assert a.bit_generator.state == b.bit_generator.state  # 시도 5회 = 소비 5회
    with pytest.raises(ValueError):
        candidate_cdf(allowed, "nope")


def test_shrink_patch_keeps_mask_binary() -> None:
    patch, mask = _square_patch(20, 10)
    p, m = shrink_patch(patch, mask, 0.5)
    assert p.shape == (10, 10, 3) and m.shape == (10, 10)
    assert set(np.unique(m).tolist()) <= {0, 255} and np.count_nonzero(m) == 25


# ---------------------------------------------------------------------------
# 스테이지 — 채택 조건
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("distribution", ["uniform", "edge", "center"])
@pytest.mark.parametrize("margin", [0, 8])
def test_accepted_placement_is_inside_roi_and_margin(distribution: str, margin: int) -> None:
    roi = _disk_roi(96)
    patch, mask = _square_patch()
    st = _stage(distribution=distribution, margin_px=margin, max_tries=50)
    for seed in range(30):
        out = st.apply(_ctx(roi, patch, mask, seed))
        assert out.placement is not None, out.warnings
        pl, placed = out.placement, out.placed_mask
        assert placed.shape == roi.shape and set(np.unique(placed).tolist()) == {0, 255}
        assert roi[placed > 0].all()
        x, y, w, h = pl.bbox
        assert x >= margin and y >= margin and x + w <= 96 - margin and y + h <= 96 - margin
        assert np.count_nonzero(placed) == np.count_nonzero(mask)
        # bbox·center·offset 정합: 캔버스 좌표 (2,2)~(6,6)가 대상 bbox로
        assert (x, y, w, h) == (pl.offset[0] + 2, pl.offset[1] + 2, 5, 5)
        assert pl.center == (x + 2, y + 2)
        assert placed[y : y + h, x : x + w].all()
        log = out.log["placement"]
        assert log["bbox"] == list(pl.bbox) and log["tries"] == pl.tries >= 1
        assert log["shrink_rounds"] == 0 and "failed" not in log


def test_distribution_ordering_edge_lt_uniform_lt_center() -> None:
    roi = _disk_roi(128)
    dist = distance_to_edge(roi)
    patch, mask = _square_patch()

    def mean_edge_distance(distribution: str, n: int = 200) -> float:
        st = _stage(distribution=distribution, margin_px=4)
        ds = []
        for seed in range(n):
            out = st.apply(_ctx(roi, patch, mask, seed))
            assert out.placement is not None
            cx, cy = out.placement.center
            ds.append(dist[cy, cx])
        return float(np.mean(ds))

    e, u, c = (mean_edge_distance(d) for d in ("edge", "uniform", "center"))
    assert e < u < c, (e, u, c)


def test_deterministic_for_same_seed() -> None:
    roi = _disk_roi(96)
    patch, mask = _square_patch()
    a = _stage().apply(_ctx(roi, patch, mask, 11))
    b = _stage().apply(_ctx(roi, patch, mask, 11))
    assert a.placement == b.placement and np.array_equal(a.placed_mask, b.placed_mask)
    assert _stage().apply(_ctx(roi, patch, mask, 12)).placement != a.placement


def test_no_overlap_with_existing_defects() -> None:
    roi = np.ones((40, 40), dtype=bool)
    patch, mask = _square_patch(9, 9)
    # 이미 놓인 결함이 중앙 큰 영역을 차지 → 남은 자리에만 놓인다
    big = np.zeros((40, 40), dtype=np.uint8)
    big[4:36, 4:24] = 255
    placed = (PlacedDefect("scratch", "scratch/000", big),)
    st = _stage(margin_px=2, max_tries=200)
    for seed in range(20):
        out = st.apply(_ctx(roi, patch, mask, seed, placed=placed))
        assert out.placement is not None, out.warnings
        assert not (out.placed_mask > 0)[big > 0].any()


def test_canvas_may_overhang_image_when_mask_fits() -> None:
    """캔버스(여유 포함)는 이미지 밖으로 걸쳐도 되고, 마스크만 안에 있으면 채택."""
    roi = np.ones((24, 24), dtype=bool)
    patch, mask = _square_patch(31, 5)  # 캔버스 31 > 이미지 24, 마스크 5
    out = _stage(margin_px=2, max_tries=100).apply(_ctx(roi, patch, mask, 1))
    assert out.placement is not None, out.warnings
    ox, oy = out.placement.offset
    assert ox < 0 and oy < 0
    assert np.count_nonzero(out.placed_mask) == 25


# ---------------------------------------------------------------------------
# 스테이지 — 실패 경로 (예외 없음, 로그·경고)
# ---------------------------------------------------------------------------


def test_shrink_then_accept_returns_shrunk_patch() -> None:
    roi = np.zeros((40, 40), dtype=bool)
    roi[10:30, 10:30] = True  # 20×20 허용 영역
    patch, mask = _square_patch(30, 30)  # 30px 마스크는 안 들어감 → 0.8배씩 축소 (24 → 19)
    st = _stage(margin_px=0, max_tries=20, shrink_on_fail={"factor": 0.8, "rounds": 3})
    out = st.apply(_ctx(roi, patch, mask, 0))
    assert out.placement is not None, out.warnings
    assert out.placement.shrink_rounds == 2 and out.patch_mask.shape[0] < 30
    assert out.patch.shape[:2] == out.patch_mask.shape
    assert roi[out.placed_mask > 0].all()
    log = out.log["placement"]
    assert log["shrink_rounds"] == 2 and log["shrink_scale"] == pytest.approx(0.64)


def test_too_small_roi_fails_soft_with_log_and_warning() -> None:
    roi = np.zeros((40, 40), dtype=bool)
    roi[18:22, 18:22] = True  # 4×4 — 이미지에는 들어가지만 ROI에는 어떤 축소로도 안 들어감
    patch, mask = _square_patch(30, 30)
    st = _stage(margin_px=0, max_tries=10, shrink_on_fail={"factor": 0.8, "rounds": 2})
    out = st.apply(_ctx(roi, patch, mask, 0, source=line_defect()))
    assert out.placement is None and out.placed_mask is None
    log = out.log["placement"]
    assert log["failed"] is True and log["shrink_rounds"] == 2 and log["tries"] == 30
    assert log["reason"] == "max_tries 소진"
    assert any("placement: scratch/000 배치 실패" in w for w in out.warnings)
    # 실패하면 patch는 입력 그대로 (축소본은 채택될 때만 돌려준다)
    assert out.patch is patch and out.patch_mask is mask


def test_mask_larger_than_image_shrinks_without_trying() -> None:
    roi = np.ones((40, 40), dtype=bool)
    patch, mask = _square_patch(50, 50)  # 이미지보다 큰 마스크 → 시도 없이 축소, 끝까지 안 들어감
    st = _stage(margin_px=4, max_tries=10, shrink_on_fail={"factor": 0.9, "rounds": 2})
    out = st.apply(_ctx(roi, patch, mask, 0))
    log = out.log["placement"]
    assert out.placement is None and log["tries"] == 0 and log["shrink_rounds"] == 2
    assert "큼" in log["reason"]


def test_exhausting_tries_is_logged() -> None:
    roi = np.zeros((40, 40), dtype=bool)
    roi[10:30, 10:30] = True
    patch, mask = _square_patch(21, 19)  # 이미지에는 들어가지만 ROI 20×20에는 못 들어감
    st = _stage(margin_px=0, max_tries=7, shrink_on_fail={"factor": 0.9, "rounds": 0})
    out = st.apply(_ctx(roi, patch, mask, 0))
    assert out.placement is None
    log = out.log["placement"]
    assert log["tries"] == 7 and log["reason"] == "max_tries 소진"


def test_missing_roi_or_patch() -> None:
    patch, mask = _square_patch()
    out = _stage().apply(context(patch=patch, patch_mask=mask))
    assert out.placement is None and out.log["placement"]["reason"] == "ROI 없음"
    out = _stage().apply(context(roi=np.ones((8, 8), dtype=bool)))
    assert out.placement is None and out.log["placement"]["skipped"] == "patch 없음"
    assert out.warnings == ()


def test_empty_roi_has_no_candidates() -> None:
    patch, mask = _square_patch()
    out = _stage().apply(_ctx(np.zeros((32, 32), dtype=bool), patch, mask))
    assert out.placement is None and out.log["placement"]["candidates"] == 0
    assert "후보 중심 없음" in out.log["placement"]["reason"]


# ---------------------------------------------------------------------------
# 통합 — 실물 스테이지 전부(source는 메모리 은행)로 hard-paste 경로
# ---------------------------------------------------------------------------


def _pipeline(defects: tuple[int, int] = (2, 2), distribution: str = "uniform") -> Pipeline:
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": "t",
            "seed": 20260914,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1, "defects_per_image": list(defects)},
            "pipeline": {
                "preset": "hard-paste",
                "placement": {
                    "roi": {"method": "otsu", "erode_px": 2},
                    "distribution": distribution,
                    "margin_px": 4,
                },
            },
        }
    )
    bank = memory_bank([line_defect(16, 3), line_defect(10, 5, cls="dent")])
    return Pipeline.from_recipe(rec, pipeline_deps(rec, bank))


def test_pipeline_places_two_disjoint_defects_inside_disk() -> None:
    p = _pipeline((2, 2))
    r, steps = p.run_one_traced(disk_target(128), 0)
    assert r.status == "ok" and len(r.instances) == 2
    roi = steps[0].ctx.roi
    assert roi is not None
    a, b = (inst.mask > 0 for inst in r.instances)
    assert not (a & b).any()
    assert roi[a].all() and roi[b].all()
    # GT 밖은 대상 그대로, GT 안은 밝은 선(회전 보간으로 230 아래 값도 섞인다)
    assert np.array_equal(r.image[r.gt_mask == 0], disk_target(128).image[r.gt_mask == 0])
    assert r.image[r.gt_mask > 0].mean() > 150
    sc = r.sidecar
    assert sc["roi"]["method"] == "otsu" and sc["roi"]["area_px"] == int(roi.sum())
    for d in sc["defects"]:
        assert set(d) == {"source", "geometry", "placement", "blend", "harmonize", "gt"}
        assert d["geometry"]["method"] == "affine" and "failed" not in d["placement"]
        assert d["placement"]["offset"] and d["placement"]["bbox"]
        assert d["blend"] == {"method": "paste"} and d["harmonize"] == {"method": "none"}
        assert d["gt"]["area_px"] > 0 and len(d["gt"]["bbox"]) == 4
    assert sc["degrade"] == {"method": "none"} and sc["gtmask"]["policy"] == "source"


def test_pipeline_is_deterministic_and_varies_by_index() -> None:
    p = _pipeline((1, 3))
    a = p.run_one(disk_target(96), 3)
    b = p.run_one(disk_target(96), 3)
    assert np.array_equal(a.image, b.image) and a.sidecar == b.sidecar
    c = p.run_one(disk_target(96), 4)
    assert a.sidecar["defects"] != c.sidecar["defects"]


def test_pipeline_skips_image_when_roi_empty() -> None:
    p = _pipeline((1, 1))
    dot = TargetImage(path=Path("dot.png"), image=disk_image(64, radius=2), gray=False)
    r = p.run_one(dot, 0)  # 점 하나짜리 물체는 erode_px=2에 다 깎여 ROI 면적 0
    assert r.status == "skipped" and r.instances == ()
    assert r.sidecar["roi"]["area_px"] == 0
    assert any("면적 0" in w for w in r.warnings) and "후보 중심 없음" in (r.reason or "")
    assert np.array_equal(r.image, dot.image)

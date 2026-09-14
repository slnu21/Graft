"""3단계 placement(structure-aware, v0.4) — 채택 배치는 sampled 와 같은 불변식(마스크 ⊆ ROI ∧ 여유 ∧ 겹침 없음) · 정렬:
줄무늬 대상에서 긴 패치의 주축이 줄 방향(jitter 0)으로 · 둥근 패치·``align: none``·낮은 일관성이면 정렬 안 함 · 위치 가중
``edges`` > ``uniform`` > ``flat``(고정 시드, 채택 중심의 평균 그래디언트 크기) · 시도마다 rng 정확히 2회 · 이미지당 구조 맵 캐시 ·
``with_method("placement", …)``가 roi 블록을 유지 · 프리셋 ``structure-aware-graft`` 파이프라인 e2e."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from anograft.core import recipe as R
from anograft.core import structure as S
from anograft.core.pipeline import Pipeline
from anograft.core.stages.placement import StructureAwarePlacement, allowed_centers
from anograft.core.types import Context, TargetImage
from tests.fixtures import context, disk_target, line_defect, memory_bank, pipeline_deps
from tests.test_structure import stripes


def _bar_patch(length: int = 24, width: int = 4, margin: int = 4) -> tuple[np.ndarray, np.ndarray]:
    src = line_defect(length, width, margin=margin)
    return src.image, src.mask


def _disk_patch(r: int = 6, margin: int = 3) -> tuple[np.ndarray, np.ndarray]:
    size = 2 * (r + margin) + 1
    yy, xx = np.mgrid[0:size, 0:size]
    mask = (((yy - size // 2) ** 2 + (xx - size // 2) ** 2) <= r * r).astype(np.uint8) * 255
    return np.full((size, size, 3), 230, dtype=np.uint8), mask


def striped_target(size: int = 160, angle: float = -30.0, name: str = "stripes.png") -> TargetImage:
    g = stripes(size, angle).astype(np.uint8)
    return TargetImage(path=Path(name), image=np.repeat(g[:, :, None], 3, axis=2), gray=False)


def _stage(**kw) -> StructureAwarePlacement:
    return StructureAwarePlacement(R.StructureAwarePlacementConfig(**kw), {})


def _ctx(target: TargetImage, patch: np.ndarray, mask: np.ndarray, seed: int = 0, **kw) -> Context:
    h, w = target.image.shape[:2]
    roi = kw.pop("roi", np.ones((h, w), dtype=bool))
    return context(target, seed=seed, roi=roi, patch=patch, patch_mask=mask, **kw)


# ---------------------------------------------------------------------------
# 불변식 · 정렬
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefer", ["edges", "flat", "uniform"])
def test_accepted_placement_inside_roi_and_margin(prefer: str) -> None:
    t = striped_target()
    patch, mask = _bar_patch()
    st = _stage(prefer=prefer, margin_px=6, jitter_deg=0.0)
    for seed in range(6):
        out = st.apply(_ctx(t, patch, mask, seed))
        assert out.placement is not None, out.warnings
        x, y, w, h = out.placement.bbox
        assert x >= 6 and y >= 6 and x + w <= 160 - 6 and y + h <= 160 - 6
        placed = out.placed_mask > 0
        assert placed.sum() == (out.patch_mask > 0).sum()
        log = out.log["placement"]
        assert log["method"] == "structure-aware" and log["prefer"] == prefer
        assert "aligned" in log and "coherence" in log and "angle_deg" in log


@pytest.mark.parametrize("angle", [-30.0, 20.0, 70.0])
def test_bar_aligns_along_stripes(angle: float) -> None:
    t = striped_target(angle=angle)
    patch, mask = _bar_patch()
    out = _stage(jitter_deg=0.0, prefer="uniform").apply(_ctx(t, patch, mask, 1))
    assert out.placement is not None
    log = out.log["placement"]
    assert log["aligned"] is True and log["coherence"] > 0.5
    phi, _ = S.mask_principal_axis(out.patch_mask)
    assert abs(S.wrap_180(phi - angle)) < 4.0, (angle, phi)
    # across 는 그에 수직
    out2 = _stage(jitter_deg=0.0, prefer="uniform", align="across").apply(_ctx(t, patch, mask, 1))
    phi2, _ = S.mask_principal_axis(out2.patch_mask)
    assert abs(S.wrap_180(phi2 - angle - 90.0)) < 4.0, (angle, phi2)


def test_no_alignment_for_round_patch_align_none_or_low_coherence() -> None:
    t = striped_target()
    bar, bar_mask = _bar_patch()
    disk, disk_mask = _disk_patch()
    out = _stage(jitter_deg=0.0).apply(_ctx(t, disk, disk_mask, 2))
    assert out.placement is not None and out.log["placement"]["aligned"] is False
    assert out.log["placement"]["angle_deg"] == 0.0 and out.patch is disk
    out = _stage(jitter_deg=0.0, align="none").apply(_ctx(t, bar, bar_mask, 2))
    assert out.placement is not None and out.log["placement"]["aligned"] is False
    assert out.patch_mask is bar_mask
    # 평탄한 대상(일관성 0)에서는 정렬하지 않는다
    flat = TargetImage(path=Path("f.png"), image=np.full((96, 96, 3), 120, np.uint8), gray=False)
    out = _stage(jitter_deg=0.0).apply(_ctx(flat, bar, bar_mask, 2))
    assert out.placement is not None and out.log["placement"]["aligned"] is False
    assert out.log["placement"]["coherence"] == 0.0


def test_jitter_spreads_axis_around_stripe_direction() -> None:
    t = striped_target(angle=0.0)
    patch, mask = _bar_patch()
    st = _stage(jitter_deg=30.0, prefer="uniform")
    angles = []
    for seed in range(12):
        out = st.apply(_ctx(t, patch, mask, seed))
        assert out.placement is not None
        angles.append(S.mask_principal_axis(out.patch_mask)[0])
    assert max(abs(a) for a in angles) <= 33.0 and max(abs(a) for a in angles) > 5.0
    assert len({round(a) for a in angles}) > 3


# ---------------------------------------------------------------------------
# 위치 가중
# ---------------------------------------------------------------------------


def test_prefer_edges_lands_on_stronger_gradient_than_flat() -> None:
    # 왼쪽 반은 평탄, 오른쪽 반은 줄무늬(그래디언트 큼)
    g = np.full((160, 160), 120.0, dtype=np.float32)
    g[:, 80:] = stripes(160, 0.0)[:, 80:]
    t = TargetImage(
        path=Path("half.png"),
        image=np.repeat(g.astype(np.uint8)[:, :, None], 3, axis=2),
        gray=False,
    )
    patch, mask = _disk_patch(4)
    mag = S.gradient_magnitude(S.to_gray_f32(t.image), 3.0)

    def mean_mag(prefer: str, strength: float = 1.0) -> float:
        st = _stage(prefer=prefer, strength=strength, align="none")
        vals = []
        for seed in range(40):
            out = st.apply(_ctx(t, patch, mask, seed))
            assert out.placement is not None
            cx, cy = out.placement.center
            vals.append(float(mag[cy, cx]))
        return float(np.mean(vals))

    e, u, f = mean_mag("edges"), mean_mag("uniform"), mean_mag("flat")
    assert e > u > f, (e, u, f)
    assert mean_mag("edges", 3.0) >= e  # 지수를 올리면 더 강하게 쏠린다


# ---------------------------------------------------------------------------
# rng 소비 · 캐시 · 결정성
# ---------------------------------------------------------------------------


def test_consumes_exactly_two_randoms_per_try() -> None:
    t = striped_target()
    patch, mask = _bar_patch()
    st = _stage(prefer="uniform", jitter_deg=5.0)
    out = st.apply(_ctx(t, patch, mask, 7))
    assert out.placement is not None
    tries = out.placement.tries
    # 같은 시드로 수동 재현: 시도마다 integers(n) + uniform
    allowed = allowed_centers(np.ones((160, 160), dtype=bool), 8, None)
    n = int(np.count_nonzero(allowed))
    rng = np.random.default_rng(7)
    for _ in range(tries):
        rng.integers(n)
        rng.uniform(-5.0, 5.0)
    assert out.rng.bit_generator.state == rng.bit_generator.state


def test_field_cached_per_image_object() -> None:
    t = striped_target()
    st = _stage()
    f1 = st.field_for(t.image)
    assert st.field_for(t.image) is f1
    other = striped_target(angle=45.0)
    f2 = st.field_for(other.image)
    assert f2 is not f1 and f2.image is other.image
    assert f1.mag.shape == (160, 160) and f1.gray.dtype == np.float32


def test_deterministic_for_same_seed_and_differs_by_seed() -> None:
    t = striped_target()
    patch, mask = _bar_patch()
    a = _stage().apply(_ctx(t, patch, mask, 11))
    b = _stage().apply(_ctx(t, patch, mask, 11))
    assert a.placement == b.placement and np.array_equal(a.patch_mask, b.patch_mask)
    assert _stage().apply(_ctx(t, patch, mask, 12)).placement != a.placement


def test_fail_soft_when_patch_does_not_fit() -> None:
    t = striped_target(size=40)
    patch, mask = _bar_patch(length=60, width=6)
    out = _stage(shrink_on_fail={"factor": 0.5, "rounds": 1}, max_tries=3).apply(
        _ctx(t, patch, mask, 0)
    )
    assert out.placement is None and out.log["placement"]["failed"]
    assert any("배치 실패" in w for w in out.warnings)


# ---------------------------------------------------------------------------
# 레시피 · 파이프라인
# ---------------------------------------------------------------------------


def test_with_method_placement_keeps_roi_block() -> None:
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": "t",
            "seed": 1,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1},
            "pipeline": {"placement": {"roi": {"method": "none"}, "distribution": "edge"}},
        }
    )
    sa = rec.with_method("placement", "structure-aware")
    assert sa.pipeline.placement.method == "structure-aware"
    assert sa.pipeline.placement.roi.method == "none"  # roi 유지
    back = sa.with_method("placement", "sampled")
    assert (
        back.pipeline.placement.distribution == "uniform"
        and back.pipeline.placement.roi.method == "none"
    )
    with pytest.raises(ValidationError):
        R.StructureAwarePlacementConfig(distribution="edge")  # 다른 method 의 키는 에러


def test_preset_pipeline_end_to_end_on_disk_target() -> None:
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": "sa",
            "seed": 20260914,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1, "defects_per_image": [2, 2]},
            "pipeline": {
                "preset": "structure-aware-graft",
                "placement": {
                    "roi": {"method": "grabcut", "erode_px": 2, "work_px": 0},
                    "margin_px": 4,
                },
            },
        }
    )
    assert rec.pipeline.placement.method == "structure-aware"
    assert rec.pipeline.placement.roi.method == "grabcut"
    bank = memory_bank([line_defect(18, 4), line_defect(10, 6, cls="dent")])
    p = Pipeline.from_recipe(rec, pipeline_deps(rec, bank))
    r, steps = p.run_one_traced(disk_target(128), 0)
    assert r.status == "ok" and len(r.instances) == 2, r.warnings
    roi = steps[0].ctx.roi
    assert roi is not None and r.sidecar["roi"]["method"] == "grabcut"
    # 배치된 소스 마스크는 ROI 안 (GT 는 union 정책이라 팽창 링만큼 밖으로 나갈 수 있다)
    placed = [
        s.ctx.placed_mask for s in steps if s.stage == "placement" and s.ctx.placed_mask is not None
    ]
    assert len(placed) == 2
    for pm in placed:
        assert roi[pm > 0].all()
    for d in r.sidecar["defects"]:
        assert d["placement"]["method"] == "structure-aware" and "aligned" in d["placement"]
    r2 = p.run_one(disk_target(128), 0)
    assert np.array_equal(r.image, r2.image)

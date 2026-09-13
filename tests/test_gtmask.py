"""7단계 gtmask — 설계 §11: `source ⊆ union`; `diff`는 변화 픽셀만(합성 전후 동일 영역은 0); 먼 곳 잡음 제외; dilate 면적 증가.
추가: 인스턴스별 GT·class_id(deps 또는 정렬 폴백)·면적 0 인스턴스 제외+경고·defect_index 전달."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from anograft.core import recipe as R
from anograft.core.stages.gtmask import GtMask, change_map, instance_mask
from anograft.core.types import Context, PlacedDefect, TargetImage
from tests.fixtures import context

H, W = 48, 64


def _scene() -> tuple[Context, np.ndarray, np.ndarray]:
    """대상 = 균일 100. 결함 A(소스 마스크 사각)는 마스크의 **왼쪽 절반만** 실제로 바뀜(저대비 부분 = Poisson이 지운 셈).
    결함 B는 다른 클래스. 먼 곳(우하단)에 결함과 무관한 변화(잡음)."""
    target = np.full((H, W, 3), 100, dtype=np.uint8)
    a = np.zeros((H, W), dtype=np.uint8)
    a[10:20, 10:30] = 255
    b = np.zeros((H, W), dtype=np.uint8)
    b[30:40, 40:56] = 255
    comp = target.copy()
    comp[10:20, 10:20] = 200  # A의 왼쪽 절반만 변화
    comp[30:40, 40:56] = 30  # B 전체 변화
    comp[44:47, 58:62] = 255  # 먼 곳 잡음
    t = TargetImage(path=Path("t.png"), image=target, gray=False)
    ctx = replace(
        context(t),
        composite=comp,
        pre_degrade=comp,
        placed=(
            PlacedDefect("scratch", "scratch/000", a, defect_index=0),
            PlacedDefect("dent", "dent/003", b, defect_index=2),
        ),
    )
    return ctx, a, b


def _stage(policy: str, dilate_px: int = 0, thr: int = 12, **deps) -> GtMask:
    return GtMask(R.GtMaskConfig(policy=policy, diff_threshold=thr, dilate_px=dilate_px), deps)


def test_change_map_threshold_and_channel_max() -> None:
    before = np.zeros((2, 2, 3), dtype=np.uint8)
    after = before.copy()
    after[0, 0, 2] = 13  # 한 채널만 13 → 임계 12 초과
    after[0, 1] = 12  # 정확히 임계 → 미포함
    assert change_map(before, after, 12).tolist() == [[True, False], [False, False]]


def test_instance_mask_policies() -> None:
    src = np.zeros((20, 20), dtype=np.uint8)
    src[5:15, 5:15] = 255
    changed = np.zeros((20, 20), dtype=bool)
    changed[5:15, 5:10] = True  # 왼쪽 절반
    changed[18, 18] = True  # 먼 곳
    s = instance_mask("source", src, None, 0)
    d = instance_mask("diff", src, changed, 0)
    u = instance_mask("union", src, changed, 0)
    assert np.array_equal(s, src)
    assert d.sum() // 255 == 50 and not d[18, 18]
    assert np.array_equal(u, src)  # diff ⊆ source 이므로 union == source
    assert instance_mask("source", src, None, 2).sum() > s.sum()


def test_source_policy_instances_and_union() -> None:
    ctx, a, b = _scene()
    out = _stage("source", class_ids={"scratch": 0, "dent": 1}).apply(ctx)
    assert len(out.instances) == 2
    ia, ib = out.instances
    assert (ia.cls, ia.class_id, ia.defect_index) == ("scratch", 0, 0)
    assert (ib.cls, ib.class_id, ib.defect_index) == ("dent", 1, 2)
    assert np.array_equal(ia.mask, a) and ia.bbox == (10, 10, 20, 10) and ia.area_px == 200
    assert np.array_equal(out.gt_mask, np.maximum(a, b))
    log = out.log["gtmask"]
    assert log["policy"] == "source" and log["area_px_total"] == 200 + 160
    assert log["class_ids_source"] == "deps" and len(log["instances"]) == 2
    assert out.warnings == ()


def test_diff_policy_only_changed_pixels_and_excludes_far_noise() -> None:
    ctx, _, _ = _scene()
    out = _stage("diff").apply(ctx)
    ia, ib = out.instances
    assert ia.area_px == 100 and ia.bbox == (10, 10, 10, 10)  # 왼쪽 절반만
    assert ib.area_px == 160
    assert not out.gt_mask[44:47, 58:62].any()  # 먼 곳 잡음 제외
    # diff ⊆ source 팽창 영역
    assert np.array_equal(out.gt_mask > 0, np.maximum(ia.mask, ib.mask) > 0)


def test_union_contains_source_and_dilate_grows() -> None:
    ctx, _, _ = _scene()
    src = _stage("source").apply(ctx)
    uni = _stage("union").apply(ctx)
    assert not ((src.gt_mask > 0) & ~(uni.gt_mask > 0)).any()  # source ⊆ union
    grown = _stage("union", dilate_px=2).apply(ctx)
    assert grown.log["gtmask"]["area_px_total"] > uni.log["gtmask"]["area_px_total"]
    assert not ((uni.gt_mask > 0) & ~(grown.gt_mask > 0)).any()
    assert set(np.unique(grown.gt_mask).tolist()) <= {0, 255}


def test_diff_uses_pre_degrade_not_degraded_composite() -> None:
    ctx, _, _ = _scene()
    noisy = ctx.composite.copy()
    noisy[:, :, 0] ^= 0x7F  # 열화된 composite는 전부 바뀐 것처럼
    out = _stage("diff").apply(replace(ctx, composite=noisy))
    assert out.instances[0].area_px == 100  # pre_degrade 기준


def test_zero_area_instance_is_dropped_with_warning() -> None:
    ctx, _, _ = _scene()
    comp = ctx.target.image.copy()
    comp[30:40, 40:56] = 30  # A는 변화 없음
    out = _stage("diff").apply(replace(ctx, composite=comp, pre_degrade=comp))
    assert len(out.instances) == 1 and out.instances[0].cls == "dent"
    assert any("scratch/000" in w and "면적 0" in w for w in out.warnings)


def test_class_ids_fallback_sorted_and_unknown_class() -> None:
    ctx, _, _ = _scene()
    out = _stage("source").apply(ctx)
    ids = {i.cls: i.class_id for i in out.instances}
    assert (
        ids == {"dent": 0, "scratch": 1}
        and out.log["gtmask"]["class_ids_source"] == "placed-sorted"
    )
    out = _stage("source", class_ids={"scratch": 0}).apply(ctx)
    assert [i.cls for i in out.instances] == ["scratch"]
    assert any("dent" in w for w in out.warnings)


def test_no_placed_defects_gives_empty_gt() -> None:
    out = _stage("union").apply(context())
    assert out.gt_mask is not None and out.gt_mask.max() == 0 and out.instances == ()
    assert out.log["gtmask"]["area_px_total"] == 0


@pytest.mark.parametrize("policy", ["source", "diff", "union"])
def test_registry_builds_every_policy(policy: str) -> None:
    from anograft.core import registry

    st = registry.build("gtmask", R.GtMaskConfig(policy=policy))
    assert isinstance(st, GtMask)

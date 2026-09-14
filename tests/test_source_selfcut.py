"""1단계 ``self-cut``(CutPaste) — 은행 없이: 크기 범위·ROI 안 자르기·스카 치수·크롭 여유·색 지터·결정성·fail-soft·흑백 불변식."""

from __future__ import annotations

import numpy as np
import pytest

from anograft.core import registry
from anograft.core.recipe import ColorJitterConfig, SelfCutSourceConfig
from anograft.core.stages.source import SelfCutSource
from anograft.core.stages.source.selfcut import color_jitter
from tests.fixtures import context, disk_image, disk_target


def _roi(size: int = 128) -> np.ndarray:
    return disk_image(size)[..., 0] > 100  # 원판


def test_registered_and_buildable_without_deps() -> None:
    st = registry.build("source", SelfCutSourceConfig(), {})
    assert isinstance(st, SelfCutSource)
    out = st.apply(context(seed=1))
    assert out.source is not None and out.source.cls == "cutpaste"
    assert out.log["source"]["method"] == "self-cut" and out.log["source"]["class_id"] == 0


def test_rect_size_within_ratio_and_aspect_and_inside_roi() -> None:
    cfg = SelfCutSourceConfig(
        shape="rect",
        area_ratio=(0.02, 0.05),
        aspect=(0.5, 2.0),
        jitter=ColorJitterConfig(brightness=0, contrast=0, saturation=0, hue=0),
    )
    st = SelfCutSource(cfg, {"class_ids": {"cutpaste": 3}})
    roi = _roi()
    for seed in range(20):
        out = st.apply(context(seed=seed, roi=roi))
        log = out.log["source"]
        x, y, w, h = log["cut_box"]
        assert 0.015 * 128 * 128 <= w * h <= 0.06 * 128 * 128  # 반올림 여유
        assert 0.4 <= w / h <= 2.5
        assert roi[y : y + h, x : x + w].all() and not log["roi_fallback"]
        assert log["class_id"] == 3 and log["mask_origin"] == "self-cut:rect"
        src = out.source
        assert src is not None and src.mask.shape == src.image.shape[:2]
        assert np.count_nonzero(src.mask) == w * h
        # 크롭 = 사각형 ± margin (이미지 안으로 클립) — 마스크 bbox 가 크롭 안에 여유를 두고 들어간다
        assert src.image.shape[0] >= h and src.image.shape[1] >= w
        assert (
            src.image.shape[0] <= h + 2 * cfg.margin_px
            and src.image.shape[1] <= w + 2 * cfg.margin_px
        )
        assert src.um_per_px is None and src.origin == "target.png"


def test_scar_dimensions_and_mixed_choice() -> None:
    scar = SelfCutSource(
        SelfCutSourceConfig(shape="scar", scar_width_px=(2, 4), scar_length_px=(20, 30)), {}
    )
    for seed in range(10):
        _x, _y, w, h = scar.apply(context(seed=seed)).log["source"]["cut_box"]
        assert 2 <= w <= 4 and 20 <= h <= 30
    mixed = SelfCutSource(SelfCutSourceConfig(shape="mixed"), {})
    shapes = {mixed.apply(context(seed=s)).log["source"]["shape"] for s in range(30)}
    assert shapes == {"rect", "scar"}


def test_patch_is_cut_from_target_pixels() -> None:
    cfg = SelfCutSourceConfig(
        shape="rect",
        jitter=ColorJitterConfig(brightness=0, contrast=0, saturation=0, hue=0),
        margin_px=0,
    )
    st = SelfCutSource(cfg, {})
    t = disk_target(96)
    out = st.apply(context(t, seed=2))
    x, y, w, h = out.log["source"]["cut_box"]
    assert out.source is not None
    assert np.array_equal(out.source.image, t.image[y : y + h, x : x + w])
    assert out.log["source"]["jitter"] is None


def test_roi_fallback_when_rect_cannot_fit() -> None:
    roi = np.zeros((128, 128), dtype=bool)
    roi[60:64, 60:64] = True  # 4x4 — 어떤 rect 도 못 들어간다
    st = SelfCutSource(SelfCutSourceConfig(shape="rect", max_tries=5), {})
    out = st.apply(context(seed=0, roi=roi))
    assert out.source is not None and out.log["source"]["roi_fallback"]
    assert any("아무 자리" in w for w in out.warnings)


def test_deterministic_and_seed_sensitive() -> None:
    st = SelfCutSource(SelfCutSourceConfig(), {})
    a, b = st.apply(context(seed=9)), st.apply(context(seed=9))
    assert a.log["source"] == b.log["source"]
    assert np.array_equal(a.source.image, b.source.image)  # type: ignore[union-attr]
    c = st.apply(context(seed=10))
    assert c.log["source"]["cut_box"] != a.log["source"]["cut_box"]


def test_color_jitter_ranges_and_gray_invariant() -> None:
    cfg = ColorJitterConfig(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1)
    img = np.random.default_rng(0).integers(0, 255, (24, 24, 3), dtype=np.uint8)
    out, log = color_jitter(img, np.random.default_rng(1), cfg)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert 0.8 <= log["brightness"] <= 1.2 and 0.8 <= log["contrast"] <= 1.2
    assert 0.7 <= log["saturation"] <= 1.3 and -18.0 <= log["hue_deg"] <= 18.0
    assert not np.array_equal(out, img)
    # 흑백(3채널 동일)은 채도·색상이 항등 → 3채널 동일 유지
    g = np.repeat(np.random.default_rng(2).integers(0, 255, (24, 24, 1), dtype=np.uint8), 3, axis=2)
    og, _ = color_jitter(g, np.random.default_rng(1), cfg)
    assert (og[..., 0] == og[..., 1]).all() and (og[..., 1] == og[..., 2]).all()
    # 전부 0 이면 항등 (rng 는 그래도 4회 소비)
    rng = np.random.default_rng(3)
    same, _ = color_jitter(
        img, rng, ColorJitterConfig(brightness=0, contrast=0, saturation=0, hue=0)
    )
    assert np.array_equal(same, img)


def test_tiny_target_is_skipped() -> None:
    from pathlib import Path

    from anograft.core.types import TargetImage

    t = TargetImage(Path("t.png"), np.zeros((3, 3, 3), dtype=np.uint8), False)
    out = SelfCutSource(SelfCutSourceConfig(), {}).apply(context(t))
    assert out.source is None and out.log["source"]["skipped"]


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        SelfCutSourceConfig(area_ratio=(0.0, 0.5))
    with pytest.raises(ValueError):
        SelfCutSourceConfig(aspect=(0.0, 2.0))
    with pytest.raises(ValueError):
        SelfCutSourceConfig(scar_width_px=(0, 4))
    with pytest.raises(ValueError):
        SelfCutSourceConfig(unknown=1)  # type: ignore[call-arg]

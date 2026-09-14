"""1단계 ``perlin-texture``(DRAEM) — 은행 없이: 창 크기·마스크 면적·bbox 크롭·텍스처 self/dir(폴백)·증강·흑백 불변식·결정성·fail-soft."""

from __future__ import annotations

import numpy as np
import pytest

from anograft.core import registry
from anograft.core.recipe import PerlinSourceConfig
from anograft.core.stages.source import PerlinSource
from anograft.core.stages.source.perlin import AUGMENTS, augment_texture
from tests.fixtures import context, disk_target


def test_registered_and_buildable_without_deps() -> None:
    st = registry.build("source", PerlinSourceConfig(), {})
    assert isinstance(st, PerlinSource)
    out = st.apply(context(seed=1))
    assert out.source is not None and out.source.cls == "anomaly"
    log = out.log["source"]
    assert log["method"] == "perlin-texture" and log["mask_origin"] == "perlin"
    assert log["texture"] == "self" and log["texture_index"] is None and len(log["augment"]) == 3


def test_window_size_mask_area_and_bbox_crop() -> None:
    cfg = PerlinSourceConfig(size_ratio=(0.25, 0.25), augment=False, min_area_px=1)
    st = PerlinSource(cfg, {"class_ids": {"anomaly": 2}})
    t = disk_target(128)
    for seed in range(8):
        out = st.apply(context(t, seed=seed))
        src = out.source
        assert src is not None
        log = out.log["source"]
        assert log["window_px"] == 32 and log["class_id"] == 2
        assert log["perlin"]["res_y"] in {1, 2, 4, 8, 16, 32}
        # 크롭 = 마스크 bbox ± 4 (창 안으로 클립) → 마스크 면적은 로그와 같고 크롭은 창보다 작거나 같다
        assert np.count_nonzero(src.mask) == log["perlin"]["area_px"] >= 1
        assert src.image.shape[:2] == src.mask.shape and src.mask.shape[0] <= 32
        ys, xs = np.nonzero(src.mask)
        assert ys.min() <= 4 and xs.min() <= 4  # bbox 가 크롭 앞쪽에 붙어 있다
        assert set(np.unique(src.mask).tolist()) <= {0, 255}


def test_texture_dir_used_and_falls_back_to_self_when_empty() -> None:
    tex = np.full((64, 64, 3), (10, 200, 30), dtype=np.uint8)
    cfg = PerlinSourceConfig(
        texture="dir", texture_dir="x", augment=False, size_ratio=(0.25, 0.25), min_area_px=1
    )
    st = PerlinSource(cfg, {"textures": [tex]})
    out = st.apply(context(seed=4))
    assert out.source is not None and out.log["source"]["texture"] == "dir"
    assert out.log["source"]["texture_index"] == 0
    assert (out.source.image == (10, 200, 30)).all() and not out.warnings
    # 텍스처가 없으면 self 로 폴백 + 경고
    st2 = PerlinSource(cfg, {"textures": []})
    out2 = st2.apply(context(seed=4))
    assert out2.source is not None and out2.log["source"]["texture"] == "self"
    assert any("폴백" in w for w in out2.warnings)
    # 창보다 작은 텍스처는 리사이즈
    small = np.full((8, 8, 3), 77, dtype=np.uint8)
    out3 = PerlinSource(cfg, {"textures": [small]}).apply(context(seed=4))
    assert out3.source is not None and (out3.source.image == 77).all()


def test_gray_target_keeps_three_channels_identical() -> None:
    tex = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)  # 컬러 텍스처
    cfg = PerlinSourceConfig(texture="dir", texture_dir="x", augment=True, min_area_px=1)
    st = PerlinSource(cfg, {"textures": [tex]})
    out = st.apply(context(disk_target(96, gray=True), seed=2))
    src = out.source
    assert src is not None
    assert (src.image[..., 0] == src.image[..., 1]).all() and (
        src.image[..., 1] == src.image[..., 2]
    ).all()


def test_augment_picks_three_distinct_and_is_deterministic() -> None:
    img = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    a, names_a = augment_texture(img.copy(), np.random.default_rng(5))
    b, names_b = augment_texture(img.copy(), np.random.default_rng(5))
    assert names_a == names_b and np.array_equal(a, b) and len(names_a) == 3
    assert len({n.split()[0] for n in names_a}) == 3  # 비복원
    assert a.shape == img.shape and a.dtype == np.uint8
    # 9종 각각이 모양·dtype 을 지킨다
    for fn in AUGMENTS:
        y, name = fn(img.copy(), np.random.default_rng(1))
        assert y.shape == img.shape and y.dtype == np.uint8 and name


def test_deterministic_and_skips_when_mask_too_small() -> None:
    st = PerlinSource(PerlinSourceConfig(), {})
    a, b = st.apply(context(seed=3)), st.apply(context(seed=3))
    assert a.log["source"] == b.log["source"]
    assert np.array_equal(a.source.mask, b.source.mask)  # type: ignore[union-attr]
    # threshold 1.0 → 마스크가 늘 비어 max_tries 뒤 skip
    st2 = PerlinSource(PerlinSourceConfig(threshold=1.0, max_tries=2), {})
    out = st2.apply(context(seed=3))
    assert out.source is None and out.log["source"]["skipped"]
    assert "2번" in out.log["source"]["reason"]


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        PerlinSourceConfig(texture="dir")  # texture_dir 필요
    with pytest.raises(ValueError):
        PerlinSourceConfig(size_ratio=(0.0, 0.5))
    with pytest.raises(ValueError):
        PerlinSourceConfig(scale_range=(0, 9))
    with pytest.raises(ValueError):
        PerlinSourceConfig(threshold=1.5)

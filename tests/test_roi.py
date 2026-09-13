"""3단계(앞) roi — 설계 §11: 원판 이미지에서 ``auto``가 원판을 고름(밝은 배경·어두운 배경 양쪽) · ``erode_px``로 면적 감소 ·
``none``은 전체 · ``mask_dir`` 로드(+ 로더 없음/실패는 fail-soft)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anograft.core import roi as R
from anograft.core.recipe import MaskDirRoiConfig, NoneRoiConfig, OtsuRoiConfig
from anograft.core.stages.roi import MaskDirRoi, NoneRoi, OtsuRoi
from anograft.io import imgio
from tests.fixtures import context, disk_image, disk_target


def _disk_mask(size: int = 128) -> np.ndarray:
    img = disk_image(size)
    return img[:, :, 0] > 100


# ---------------------------------------------------------------------------
# 순수 함수
# ---------------------------------------------------------------------------


def test_border_touch_ratio() -> None:
    m = np.zeros((10, 10), dtype=bool)
    assert R.border_touch_ratio(m) == 0.0
    m[:] = True
    assert R.border_touch_ratio(m) == 1.0
    m[:] = False
    m[0, :] = True  # 위쪽 변만: 테두리 픽셀 36개 중 10개
    assert R.border_touch_ratio(m) == pytest.approx(10 / 36)


@pytest.mark.parametrize("invert", [False, True])
def test_otsu_auto_picks_disk_on_both_polarities(invert: bool) -> None:
    res = R.roi_otsu(disk_image(invert=invert), "auto", 0)
    disk = _disk_mask()
    assert res.inverted is invert
    assert np.array_equal(res.roi, disk)
    assert res.area_before_erode == int(disk.sum())


def test_otsu_explicit_invert_overrides_auto() -> None:
    img = disk_image()
    disk = _disk_mask()
    assert np.array_equal(R.roi_otsu(img, "no", 0).roi, disk)
    assert np.array_equal(R.roi_otsu(img, "yes", 0).roi, ~disk)


def test_otsu_erode_reduces_area_and_stays_inside() -> None:
    r0 = R.roi_otsu(disk_image(), "auto", 0).roi
    r8 = R.roi_otsu(disk_image(), "auto", 8).roi
    assert 0 < r8.sum() < r0.sum()
    assert not (r8 & ~r0).any()


def test_otsu_uniform_image_gives_empty_or_full_without_error() -> None:
    flat = np.full((32, 32, 3), 77, dtype=np.uint8)
    res = R.roi_otsu(flat, "auto", 0)
    assert res.roi.shape == (32, 32)
    assert res.roi.sum() in (0, 32 * 32)


def test_roi_none_and_from_mask() -> None:
    assert R.roi_none((5, 7)).all() and R.roi_none((5, 7)).shape == (5, 7)
    m = np.array([[0, 127, 128, 255]], dtype=np.uint8)
    assert R.roi_from_mask(m, (1, 4)).tolist() == [[False, False, True, True]]
    with pytest.raises(ValueError, match="크기"):
        R.roi_from_mask(m, (2, 4))


def test_distance_to_edge_counts_image_border() -> None:
    allowed = np.ones((5, 5), dtype=bool)
    d = R.distance_to_edge(allowed)
    assert d[0, 0] == pytest.approx(1.0) and d[2, 2] > d[1, 1] > d[0, 0]
    allowed[2, 2] = False
    assert R.distance_to_edge(allowed)[2, 2] == 0.0


# ---------------------------------------------------------------------------
# 스테이지
# ---------------------------------------------------------------------------


def test_otsu_stage_sets_roi_and_log() -> None:
    out = OtsuRoi(OtsuRoiConfig(erode_px=4), {}).apply(context(disk_target()))
    assert out.roi is not None and out.roi.dtype == bool and out.roi.any()
    log = out.log["roi"]
    assert log["method"] == "otsu" and log["inverted"] is False and log["erode_px"] == 4
    assert log["area_px"] == int(out.roi.sum()) and 0 < log["area_ratio"] < 1
    assert out.warnings == ()


def test_none_stage_is_full_image() -> None:
    out = NoneRoi(NoneRoiConfig(), {}).apply(context(disk_target(64)))
    assert out.roi.shape == (64, 64) and out.roi.all()
    assert out.log["roi"] == {"method": "none", "area_px": 64 * 64, "area_ratio": 1.0}


def test_empty_roi_warns_but_does_not_raise() -> None:
    # 원판이 작아 erode_px가 다 먹어 버리는 경우
    out = OtsuRoi(OtsuRoiConfig(erode_px=60), {}).apply(context(disk_target(96)))
    assert out.roi is not None and not out.roi.any()
    assert any("면적 0" in w for w in out.warnings)


def test_mask_dir_stage_loads_target_stem_png(tmp_path: Path) -> None:
    t = disk_target(48, name="sample_07.png")
    m = np.zeros((48, 48), dtype=np.uint8)
    m[10:30, 5:40] = 255
    imgio.write_image(tmp_path / "sample_07.png", m)
    st = MaskDirRoi(MaskDirRoiConfig(path=tmp_path), {"read_mask": imgio.read_mask})
    out = st.apply(context(t))
    assert np.array_equal(out.roi, m > 0)
    assert out.log["roi"]["path"].endswith("sample_07.png") and out.warnings == ()


def test_mask_dir_stage_without_loader_or_file_is_fail_soft(tmp_path: Path) -> None:
    t = disk_target(48, name="missing.png")
    out = MaskDirRoi(MaskDirRoiConfig(path=tmp_path), {}).apply(context(t))
    assert out.roi is None and out.log["roi"]["failed"] is True
    assert "read_mask" in out.log["roi"]["reason"] and out.warnings

    st = MaskDirRoi(MaskDirRoiConfig(path=tmp_path), {"read_mask": imgio.read_mask})
    out = st.apply(context(t))
    assert out.roi is None and "ImageReadError" in out.log["roi"]["reason"]

    # 크기가 다른 마스크도 예외 대신 skip
    imgio.write_image(tmp_path / "missing.png", np.zeros((10, 10), dtype=np.uint8))
    out = st.apply(context(t))
    assert out.roi is None and "ValueError" in out.log["roi"]["reason"]

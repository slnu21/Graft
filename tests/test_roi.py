"""3단계(앞) roi — 설계 §11: 원판 이미지에서 ``auto``가 원판을 고름(밝은 배경·어두운 배경 양쪽) · ``erode_px``로 면적 감소 ·
``none``은 전체 · ``mask_dir`` 로드(+ 로더 없음/실패는 fail-soft) · ``grabcut``(v0.4): rect/otsu 초기화가 원판을 찾음(Otsu 와 IoU),
같은 입력 → 같은 ROI(시드), ``work_px`` 축소 후 원본 크기, 너무 작은 이미지는 Otsu 로 대체(fail-soft), 스테이지 로그·시드."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from anograft.core import roi as R
from anograft.core.recipe import GrabCutRoiConfig, MaskDirRoiConfig, NoneRoiConfig, OtsuRoiConfig
from anograft.core.seeds import stable_seed
from anograft.core.stages.roi import GrabCutRoi, MaskDirRoi, NoneRoi, OtsuRoi
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
    with pytest.raises(ValueError, match="비율"):
        R.roi_from_mask(m, (4, 4))  # 1×4 → 4×4 는 비율이 다르다 (1px 반올림 허용 범위 밖)


def test_roi_from_mask_resizes_same_ratio_with_nearest() -> None:
    """KNOWN-ISSUES #1: 원본 크기(1400²) 마스크가 미리보기 축소(1024²)에서도 같은 링을 가리켜야 한다. NEAREST + 이진화."""
    big = np.zeros((1400, 1400), dtype=np.uint8)
    cv2.circle(big, (700, 700), 510, 255, -1)
    cv2.circle(big, (700, 700), 319, 0, -1)
    small = R.roi_from_mask(big, (1024, 1024))
    assert small.shape == (1024, 1024) and small.dtype == bool
    s = 1024 / 1400
    yy, xx = np.nonzero(small)
    r = np.hypot(yy - 511.5, xx - 511.5)
    assert r.min() >= 319 * s - 2 and r.max() <= 510 * s + 2  # 링 안쪽·바깥 반경이 배율대로
    # 반올림으로 1px 어긋난 대상 크기(1400×1000 → 1024×731)도 같은 비율로 본다
    rect = np.zeros((1000, 1400), dtype=np.uint8)
    rect[200:800, 300:1100] = 255
    assert R.roi_from_mask(rect, (731, 1024)).shape == (731, 1024)
    # 회색값 마스크도 이진화되어 들어간다(LINEAR 오염 없음)
    grey = np.full((200, 200), 200, dtype=np.uint8)
    grey[:100] = 50
    out = R.roi_from_mask(grey, (100, 100))
    assert out[:50].sum() == 0 and out[50:].all()


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

    # 비율이 다른 마스크는 예외 대신 skip — 경고에 원인이 그대로 남는다
    imgio.write_image(tmp_path / "missing.png", np.zeros((10, 20), dtype=np.uint8))
    out = st.apply(context(t))
    assert out.roi is None and "ValueError" in out.log["roi"]["reason"]
    assert out.warnings[-1].startswith("roi: mask_dir 로드 실패") and "비율" in out.warnings[-1]


def test_mask_dir_stage_resizes_full_size_mask_to_preview_target(tmp_path: Path) -> None:
    """KNOWN-ISSUES #1: 대상이 축소본(48²)이고 마스크가 원본 크기(96²)여도 ROI 가 살아 있고 ``resized_from`` 이 남는다."""
    t = disk_target(48, name="big.png")
    m = np.zeros((96, 96), dtype=np.uint8)
    m[20:60, 10:80] = 255
    imgio.write_image(tmp_path / "big.png", m)
    out = MaskDirRoi(MaskDirRoiConfig(path=tmp_path), {"read_mask": imgio.read_mask}).apply(
        context(t)
    )
    assert out.roi is not None and out.roi.shape == (48, 48) and out.warnings == ()
    assert out.log["roi"]["resized_from"] == [96, 96] and "failed" not in out.log["roi"]
    assert out.roi[10:30, 5:40].all() and not out.roi[:9].any()


# ---------------------------------------------------------------------------
# grabcut (v0.4)
# ---------------------------------------------------------------------------


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum()) / float((a | b).sum() or 1)


@pytest.mark.parametrize("init", ["rect", "otsu"])
@pytest.mark.parametrize("invert", [False, True])
def test_grabcut_finds_disk_like_otsu(init: str, invert: bool) -> None:
    img = disk_image(128, invert=invert)
    res = R.roi_grabcut(img, init=init, erode_px=0, work_px=0)  # type: ignore[arg-type]
    ref = R.roi_otsu(img, "auto", 0).roi
    assert res.fallback is None and res.init_used == init and res.work_scale == 1.0
    assert _iou(res.roi, ref) > 0.9
    assert res.area_before_erode == int(res.roi.sum())
    if init == "otsu":
        assert res.inverted is invert
    else:
        assert res.inverted is None


def test_grabcut_is_deterministic_and_seed_matters_little() -> None:
    img = disk_image(96)
    a = R.roi_grabcut(img, seed=1, work_px=0).roi
    b = R.roi_grabcut(img, seed=1, work_px=0).roi
    assert np.array_equal(a, b)
    c = R.roi_grabcut(img, seed=2, work_px=0).roi
    assert _iou(a, c) > 0.9  # 시드는 k-means 초기화만 바꾼다


def test_grabcut_work_px_downscales_and_restores_shape() -> None:
    img = disk_image(160)
    res = R.roi_grabcut(img, work_px=64, erode_px=0)
    assert res.roi.shape == (160, 160) and abs(res.work_scale - 0.4) < 1e-9
    assert _iou(res.roi, R.roi_otsu(img, "auto", 0).roi) > 0.85
    assert res.roi.dtype == bool


def test_grabcut_erode_reduces_area_and_gray_input_ok() -> None:
    img = disk_image(128)
    a = R.roi_grabcut(img, erode_px=0, work_px=0).roi
    b = R.roi_grabcut(img, erode_px=6, work_px=0).roi
    assert b.sum() < a.sum() and not (b & ~a).any()
    g = R.roi_grabcut(img[:, :, 0], erode_px=0, work_px=0).roi
    assert _iou(g, a) > 0.95


def test_grabcut_otsu_init_falls_back_to_rect_on_uniform_image() -> None:
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    res = R.roi_grabcut(img, init="otsu", work_px=0, erode_px=0)
    assert res.init_used == "otsu→rect" and res.inverted is None
    assert res.roi.shape == (64, 64)  # 예외 없이 끝난다 (내용은 GrabCut 의 몫)


def test_grabcut_tiny_image_is_fail_soft() -> None:
    img = disk_image(4, radius=1)
    res = R.roi_grabcut(img, work_px=0, erode_px=0)
    assert res.roi.shape == (4, 4)
    # cv2.error 가 났으면 fallback 사유가 있고, 아니면 None — 어느 쪽이든 예외로 새지 않는다
    assert res.fallback is None or res.fallback.startswith("cv2.error")


def test_grabcut_stage_logs_and_seeds_by_file_name() -> None:
    t = disk_target(96, name="plate_007.png")
    out = GrabCutRoi(GrabCutRoiConfig(erode_px=2, work_px=64), {}).apply(context(t))
    assert out.roi is not None and out.roi.any() and out.warnings == ()
    log = out.log["roi"]
    assert log["method"] == "grabcut" and log["init"] == "rect" and log["init_used"] == "rect"
    assert log["seed"] == stable_seed("plate_007.png") and log["work_scale"] == 0.6667
    assert log["area_px"] == int(out.roi.sum()) and "fallback" not in log
    # 다른 폴더의 같은 파일 이름 → 같은 시드 (데이터셋을 옮겨도 ROI 동일)
    t2 = disk_target(96, name="plate_007.png")
    out2 = GrabCutRoi(GrabCutRoiConfig(erode_px=2, work_px=64), {}).apply(context(t2))
    assert np.array_equal(out.roi, out2.roi)


def test_grabcut_stage_warns_on_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    def boom(*a, **k):
        raise cv2.error("(-215:Assertion failed) fake")

    monkeypatch.setattr(cv2, "grabCut", boom)
    out = GrabCutRoi(GrabCutRoiConfig(work_px=0), {}).apply(context(disk_target(64)))
    assert out.roi is not None and out.roi.any()  # Otsu 대체
    assert out.log["roi"]["fallback"].startswith("cv2.error")
    assert any("grabcut 실패" in w for w in out.warnings)

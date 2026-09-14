"""4단계 blend — 설계 §11: **4방법 공통** 마스크 bbox 밖 불변 · dtype uint8 · src==dst면 항등(±1). paste: 내부 == 패치;
alpha: 페더 밖 = 대상, 페더 대역 단조; poisson: 정렬(변경 픽셀 = bbox 내부) · 경계 접촉 → 폴백 `fallback=true` · cv2.error 폴백.
multiband: 작은 창에서 `levels` 자동 축소 + 로그 `levels_used` · 경계 근처는 저주파가 대상 쪽, 깊은 안쪽은 패치 · 피라미드 왕복 정확.
추가: 캔버스 overhang 창 클리핑 · 입력 배열 불변(seamlessClone이 mask를 제자리에서 망가뜨리는 함정)."""

from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from anograft.core import recipe as R
from anograft.core.stages.blend import common as C
from anograft.core.stages.blend.alpha import AlphaBlend
from anograft.core.stages.blend.multiband import (
    MultibandBlend,
    collapse,
    crop_window,
    laplacian_pyramid,
    levels_for,
    mask_thickness,
)
from anograft.core.stages.blend.paste import HardPaste
from anograft.core.stages.blend.poisson import (
    PoissonBlend,
    clone_center,
    precheck,
    solve_bbox,
    solve_mask,
)
from anograft.core.types import Context, Placement, TargetImage
from tests.fixtures import context

H, W = 64, 80
PH, PW = 20, 30
MASK_BOX = (6, 4, 18, 11)  # 패치 캔버스 안 마스크 bbox (x, y, w, h)


def _patch(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    patch = rng.integers(0, 255, (PH, PW, 3), dtype=np.uint8)
    mask = np.zeros((PH, PW), dtype=np.uint8)
    x, y, w, h = MASK_BOX
    mask[y : y + h, x : x + w] = 255
    return patch, mask


def _target(flat: bool = True, seed: int = 1) -> np.ndarray:
    if flat:
        return np.full((H, W, 3), 100, dtype=np.uint8)
    return np.random.default_rng(seed).integers(0, 255, (H, W, 3), dtype=np.uint8)


def _placement(offset: tuple[int, int]) -> Placement:
    x, y, w, h = MASK_BOX
    bbox = (offset[0] + x, offset[1] + y, w, h)
    return Placement(center=(bbox[0] + w // 2, bbox[1] + h // 2), bbox=bbox, offset=offset, tries=1)


def _ctx(
    target: np.ndarray, patch: np.ndarray, mask: np.ndarray, offset: tuple[int, int]
) -> Context:
    t = TargetImage(path=Path("t.png"), image=target, gray=False)
    pl = _placement(offset)
    placed = np.zeros((H, W), dtype=np.uint8)
    x, y, w, h = pl.bbox
    placed[max(y, 0) : y + h, max(x, 0) : x + w] = 255
    ctx = context(t, patch=patch, patch_mask=mask)
    return replace(ctx, placement=pl, placed_mask=placed)


def _stages() -> dict[str, object]:
    """공통 계약 테스트용 — poisson은 팽창 0으로 두어 '마스크 bbox 밖 불변'을 소스 bbox 기준으로 잰다."""
    return {
        "paste": HardPaste(R.PasteBlendConfig(), {}),
        "alpha": AlphaBlend(R.AlphaBlendConfig(feather_px=3), {}),
        "poisson": PoissonBlend(R.PoissonBlendConfig(mask_dilate_px=0), {}),
        "multiband": MultibandBlend(R.MultibandBlendConfig(levels=3), {}),
    }


def _poisson(**kw) -> PoissonBlend:
    return PoissonBlend(R.PoissonBlendConfig(**kw), {})


def _bbox_mask(bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    m = np.zeros((H, W), dtype=bool)
    m[y : y + h, x : x + w] = True
    return m


# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["paste", "alpha", "poisson", "multiband"])
@pytest.mark.parametrize("flat", [True, False])
def test_outside_bbox_unchanged_and_uint8(method: str, flat: bool) -> None:
    patch, mask = _patch()
    target = _target(flat)
    out = _stages()[method].apply(_ctx(target, patch, mask, (10, 12)))
    assert out.composite.dtype == np.uint8 and out.composite.shape == target.shape
    inside = _bbox_mask(out.placement.bbox)
    assert np.array_equal(out.composite[~inside], target[~inside])
    assert not np.array_equal(out.composite[inside], target[inside])
    assert out.log["blend"]["method"] == method and "fallback" not in {
        k for k, v in out.log["blend"].items() if v is True
    }


@pytest.mark.parametrize("method", ["paste", "alpha", "poisson", "multiband"])
def test_identity_when_patch_equals_target(method: str) -> None:
    target = _target(flat=False)
    _, mask = _patch()
    patch = target[12 : 12 + PH, 10 : 10 + PW].copy()
    out = _stages()[method].apply(_ctx(target, patch, mask, (10, 12)))
    assert np.abs(out.composite.astype(int) - target.astype(int)).max() <= 1


@pytest.mark.parametrize("method", ["paste", "alpha", "poisson", "multiband"])
def test_inputs_are_not_mutated(method: str) -> None:
    patch, mask = _patch()
    target = _target(flat=False)
    p0, m0, t0 = patch.copy(), mask.copy(), target.copy()
    ctx = _ctx(target, patch, mask, (10, 12))
    out = _stages()[method].apply(ctx)
    assert np.array_equal(patch, p0) and np.array_equal(mask, m0) and np.array_equal(target, t0)
    assert ctx.composite is target and out.composite is not target


@pytest.mark.parametrize("method", ["paste", "alpha", "poisson", "multiband"])
def test_skips_without_placement(method: str) -> None:
    out = _stages()[method].apply(context())
    assert np.array_equal(out.composite, out.target.image)
    assert out.log["blend"]["skipped"]


# ---------------------------------------------------------------------------
# paste · alpha
# ---------------------------------------------------------------------------


def test_paste_interior_equals_patch_with_overhang() -> None:
    patch, mask = _patch()
    target = _target()
    offset = (-3, -1)  # 캔버스가 이미지 밖으로 걸침 — 마스크는 안
    out = HardPaste(R.PasteBlendConfig(), {}).apply(_ctx(target, patch, mask, offset))
    x, y, w, h = _placement(offset).bbox
    assert (x, y) == (3, 3)
    assert np.array_equal(out.composite[y : y + h, x : x + w], patch[4 : 4 + h, 6 : 6 + w])


def test_canvas_window_clips_to_image() -> None:
    win = C.canvas_window((H, W), (PH, PW), (-5, -7))
    assert (win.ys, win.xs) == (slice(0, PH - 7), slice(0, PW - 5))
    assert (win.pys, win.pxs) == (slice(7, PH), slice(5, PW))
    win = C.canvas_window((H, W), (PH, PW), (W - 4, H - 2))
    assert (win.ys, win.xs) == (slice(H - 2, H), slice(W - 4, W))
    assert C.canvas_window((H, W), (PH, PW), (W + 1, 0)).empty


def test_alpha_feather_monotone_and_zero_feather_is_paste() -> None:
    patch, mask = _patch()
    a = C.feather_alpha(mask, 3)
    x, y, _w, h = MASK_BOX
    row = a[y + h // 2, x - 1 : x + 5]
    assert row.tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0, 1.0, 1.0])
    assert a.max() == 1.0 and not a[mask == 0].any()
    assert np.array_equal(C.feather_alpha(mask, 0), (mask > 0).astype(np.float32))

    target = _target()
    inp0 = C.BlendInputs(
        target, patch, mask, _placement((10, 12)), C.canvas_window((H, W), (PH, PW), (10, 12))
    )
    assert np.array_equal(C.alpha_blend(inp0, 0), C.paste(inp0))


def test_alpha_band_is_monotone_between_target_and_patch() -> None:
    _, mask = _patch()
    patch = np.full((PH, PW, 3), 250, dtype=np.uint8)
    target = _target()  # 100
    out = AlphaBlend(R.AlphaBlendConfig(feather_px=4), {}).apply(
        _ctx(target, patch, mask, (10, 12))
    )
    x, y, _w, h = out.placement.bbox
    row = out.composite[y + h // 2, x - 1 : x + 6, 0].astype(int)
    assert row[0] == 100 and row[-1] == 250
    assert all(a <= b for a, b in itertools.pairwise(row))
    assert out.log["blend"] == {"method": "alpha", "feather_px": 4}


# ---------------------------------------------------------------------------
# poisson
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [(10, 12), (-3, 5), (55, 40)])
@pytest.mark.parametrize("mode", ["normal", "mixed"])
def test_poisson_alignment_changed_pixels_fill_bbox_interior(
    offset: tuple[int, int], mode: str
) -> None:
    patch, mask = _patch()
    target = _target()
    st = _poisson(poisson_mode=mode, mask_dilate_px=0)
    out = st.apply(_ctx(target, patch, mask, offset))
    assert out.log["blend"] == {
        "method": "poisson",
        "mode": mode,
        "mask_dilate_px": 0,
        "fallback": False,
    }
    changed = np.any(out.composite != target, axis=2)
    x, y, w, h = out.placement.bbox
    interior = _bbox_mask((x + 1, y + 1, w - 2, h - 2))
    # 경계 1px는 대상값으로 고정(디리클레) → 변화는 안쪽에만, 안쪽은 거의 전부 바뀐다(우연히 같은 값인 픽셀만 예외)
    assert not (changed & ~interior).any()
    assert changed[interior].mean() > 0.95


def test_clone_center_formula() -> None:
    assert clone_center((16, 16, 18, 11)) == (25, 21)
    assert clone_center((0, 0, 4, 4)) == (2, 2)


def test_poisson_falls_back_when_bbox_touches_target_border() -> None:
    patch, mask = _patch()
    target = _target()
    offset = (-6, -4)  # 마스크 bbox가 (0,0)에서 시작
    st = _poisson(feather_px=2, mask_dilate_px=0)
    out = st.apply(_ctx(target, patch, mask, offset))
    log = out.log["blend"]
    assert log["fallback"] is True and "대상 테두리" in log["fallback_reason"]
    assert log["feather_px"] == 2
    ref = AlphaBlend(R.AlphaBlendConfig(feather_px=2), {}).apply(_ctx(target, patch, mask, offset))
    assert np.array_equal(out.composite, ref.composite)


def test_poisson_dilation_keeps_thin_defect_that_tight_mask_erases() -> None:
    """OpenCV seamlessClone은 마스크를 3px 침식한다 — 팽창 0이면 3px 선이 통째로 사라지고, 기본 5면 살아남는다."""
    target = np.full((H, W, 3), 100, dtype=np.uint8)
    patch = np.full((PH, PW, 3), 100, dtype=np.uint8)
    mask = np.zeros((PH, PW), dtype=np.uint8)
    patch[9:12, 6:24] = 30  # 3px 두께 어두운 선
    mask[9:12, 6:24] = 255
    pl = Placement(center=(25, 22), bbox=(16, 21, 18, 3), offset=(10, 12), tries=1)
    placed = np.zeros((H, W), dtype=np.uint8)
    placed[21:24, 16:34] = 255
    t = TargetImage(Path("t.png"), target, False)
    ctx = replace(context(t, patch=patch, patch_mask=mask), placement=pl, placed_mask=placed)
    inside = placed > 0

    erased = _poisson(mask_dilate_px=0).apply(ctx)
    kept = _poisson(mask_dilate_px=5).apply(ctx)
    d_erased = np.abs(erased.composite.astype(int) - target.astype(int))[inside].mean()
    d_kept = np.abs(kept.composite.astype(int) - target.astype(int))[inside].mean()
    assert d_erased < 2 and d_kept > 50, (d_erased, d_kept)
    assert erased.log["blend"]["fallback"] is False and kept.log["blend"]["fallback"] is False
    # 팽창 마스크 bbox 밖은 불변
    outside = ~_bbox_mask(solve_bbox(solve_mask(mask, 5), (10, 12)))
    assert np.array_equal(kept.composite[outside], target[outside])


def test_solve_mask_and_bbox() -> None:
    _, mask = _patch()
    m0 = solve_mask(mask, 0)
    assert np.array_equal(m0, mask) and m0 is not mask
    m5 = solve_mask(mask, 5)
    assert m5.sum() > mask.sum() and not C.touches_border(m5)
    x, y, w, h = solve_bbox(m5, (10, 12))
    assert (x, y) == (10 + 1, 12 + 1) and (w, h) == (PW - 2, PH - 2)  # 캔버스 테두리 1px에서 잘림


def test_poisson_mask_touching_canvas_border_is_clipped_not_fallback() -> None:
    patch, mask = _patch()
    mask[0:15, 6:24] = 255  # 캔버스 위 테두리에 닿음 — OpenCV가 깎을 1px를 먼저 비우고 진행
    target = _target()
    out = _poisson(mask_dilate_px=0).apply(_ctx(target, patch, mask, (10, 12)))
    assert out.log["blend"]["fallback"] is False
    changed = np.any(out.composite != target, axis=2)
    assert not changed[12, :].any()  # 캔버스 row 0 = 대상 row 12 는 잘려서 안 바뀜


def test_poisson_precheck_reasons() -> None:
    _, mask = _patch()
    assert precheck(mask, (16, 16, 18, 11), (H, W)) is None
    assert precheck(np.zeros_like(mask), (16, 16, 18, 11), (H, W)) == "마스크 면적 0"
    assert "대상 테두리" in precheck(mask, (0, 16, 18, 11), (H, W))
    assert "대상 테두리" in precheck(mask, (16, 16, W - 16, 11), (H, W))


def test_poisson_falls_back_on_cv2_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        raise cv2.error("simulated solver failure")

    monkeypatch.setattr(cv2, "seamlessClone", boom)
    patch, mask = _patch()
    out = PoissonBlend(R.PoissonBlendConfig(), {}).apply(_ctx(_target(), patch, mask, (10, 12)))
    log = out.log["blend"]
    assert log["fallback"] is True and log["fallback_reason"].startswith("cv2.error")
    inside = _bbox_mask(out.placement.bbox)
    assert not np.array_equal(out.composite[inside], _target()[inside])


# ---------------------------------------------------------------------------
# multiband
# ---------------------------------------------------------------------------


def _multiband(levels: int = 4) -> MultibandBlend:
    return MultibandBlend(R.MultibandBlendConfig(levels=levels), {})


def _box_mask(
    h: int, w: int, ph: int = PH, pw: int = PW, at: tuple[int, int] = (2, 2)
) -> np.ndarray:
    m = np.zeros((ph, pw), dtype=np.uint8)
    m[at[0] : at[0] + h, at[1] : at[1] + w] = 255
    return m


def _ctx_for(
    target: np.ndarray, patch: np.ndarray, mask: np.ndarray, offset: tuple[int, int]
) -> Context:
    """``_ctx``와 같되 bbox를 실제 마스크에서 잰다(MASK_BOX가 아닌 마스크용)."""
    x, y, w, h = cv2.boundingRect(mask)
    bbox = (offset[0] + x, offset[1] + y, w, h)
    pl = Placement(center=(bbox[0] + w // 2, bbox[1] + h // 2), bbox=bbox, offset=offset, tries=1)
    placed = np.zeros(target.shape[:2], dtype=np.uint8)
    placed[bbox[1] : bbox[1] + h, bbox[0] : bbox[0] + w] = mask[y : y + h, x : x + w]
    t = TargetImage(path=Path("t.png"), image=target, gray=False)
    return replace(context(t, patch=patch, patch_mask=mask), placement=pl, placed_mask=placed)


def test_mask_thickness_is_inscribed_diameter() -> None:
    assert mask_thickness(_box_mask(4, 24)) == 4.0
    assert mask_thickness(_box_mask(11, 18)) == 12.0  # 홀수 두께: 중앙 행 거리 6
    assert mask_thickness(np.full((32, 32), 255, dtype=np.uint8)) == 32.0
    assert mask_thickness(np.zeros((8, 8), dtype=np.uint8)) == 0.0


def test_levels_for_caps_by_thickness_then_window() -> None:
    assert levels_for(4, 64.0) == 4 and levels_for(8, 64.0) == 4  # floor(log2 64) − 2
    assert levels_for(4, 32.0) == 3 and levels_for(4, 16.0) == 2 and levels_for(4, 8.0) == 1
    assert levels_for(4, 4.0) == 1 and levels_for(4, 0.0) == 1  # 최소 1
    assert levels_for(4, 256.0, (64, 80)) == 4 and levels_for(8, 256.0, (64, 80)) == 5  # 창 캡
    assert levels_for(8, 256.0, (7, 7)) == 1 and levels_for(8, 256.0, (1, 1)) == 1


def test_laplacian_pyramid_round_trips_exactly() -> None:
    img = _target(flat=False).astype(np.float32)
    for levels in (1, 3, 5):
        assert np.abs(collapse(laplacian_pyramid(img, levels)) - img).max() < 1e-3
    odd = img[:63, :77]  # 홀수 크기 — pyrUp dstsize 규약
    assert np.abs(collapse(laplacian_pyramid(odd, 4)) - odd).max() < 1e-3


def test_multiband_logs_levels_used_and_window() -> None:
    patch, mask = _patch()  # 마스크 18×11 → 두께 12 → 캡 1
    out = _multiband(4).apply(_ctx(_target(), patch, mask, (10, 12)))
    log = out.log["blend"]
    assert log["method"] == "multiband" and log["levels"] == 4
    assert log["levels_used"] == 1 and log["thickness_px"] == 12.0
    assert log["window_px"] == [18 + 2 * 2, 11 + 2 * 2]  # bbox ± 2**1

    big = np.full((PH, PW), 255, dtype=np.uint8)  # 20×30 전부 → 두께 20 → 캡 2
    out = _multiband(4).apply(_ctx(_target(), patch, big, (10, 12)))
    assert out.log["blend"]["levels_used"] == 2 and out.log["blend"]["window_px"] == [
        30 + 8,
        20 + 8,
    ]
    assert (
        _multiband(1).apply(_ctx(_target(), patch, big, (10, 12))).log["blend"]["levels_used"] == 1
    )


def test_multiband_window_cap_when_target_is_tiny() -> None:
    """대상이 작아 창이 잘리면 창 캡이 두께 캡보다 먼저 걸린다: 6×6 대상 → floor(log2 6) − 1 = 1."""
    tiny = np.full((6, 6, 3), 100, dtype=np.uint8)
    patch = np.full((6, 6, 3), 250, dtype=np.uint8)
    mask = np.full((6, 6), 255, dtype=np.uint8)  # 두께 6 → 캡 max(1, 2−2) = 1 … 창 캡도 1
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 0  # 4×4 → 두께 4 → 캡 1
    t = TargetImage(path=Path("t.png"), image=tiny, gray=False)
    pl = Placement(center=(3, 3), bbox=(1, 1, 4, 4), offset=(0, 0), tries=1)
    ctx = replace(context(t, patch=patch, patch_mask=mask), placement=pl, placed_mask=mask)
    out = _multiband(8).apply(ctx)
    assert out.log["blend"]["levels_used"] == 1 and out.log["blend"]["window_px"] == [6, 6]
    assert np.array_equal(out.composite[mask == 0], tiny[mask == 0])
    assert (out.composite[mask > 0, 0] > 100).all()


def test_multiband_low_frequency_follows_target_near_edge_and_patch_deep_inside() -> None:
    """마스크 안 경계 픽셀은 저주파가 대상 쪽(paste는 250), 깊은 안쪽은 패치 값의 ~90%+ (두께 캡 덕)."""
    mask = _box_mask(16, 26)
    patch = np.full((PH, PW, 3), 250, dtype=np.uint8)
    target = _target()  # 100
    out = _multiband(4).apply(_ctx_for(target, patch, mask, (10, 12)))
    assert out.log["blend"]["levels_used"] == 2
    x, y, w, h = out.placement.bbox
    cy = y + h // 2
    edge = int(out.composite[cy, x, 0])
    deep = int(out.composite[cy, x + w // 2, 0])
    assert 100 < edge < deep and deep >= 230
    inside = _bbox_mask(out.placement.bbox)
    assert np.array_equal(out.composite[~inside], target[~inside])


def test_multiband_thin_line_keeps_contrast() -> None:
    """4px 스크래치: 두께 캡 → levels_used 1 → 내부 평균이 결함 값의 85% 이상(캡 없이 levels 4면 ~70%)."""
    mask = _box_mask(4, 24)
    patch = np.full((PH, PW, 3), 100, dtype=np.uint8)
    patch[mask > 0] = 250
    out = _multiband(4).apply(_ctx_for(_target(), patch, mask, (10, 12)))
    assert out.log["blend"]["levels_used"] == 1
    x, y, w, h = out.placement.bbox
    vals = out.composite[y : y + h, x : x + w, 0].astype(float)
    assert (vals.mean() - 100) / 150 > 0.85


def test_multiband_crop_window_fills_outside_canvas_with_target() -> None:
    patch, mask = _patch()
    target = _target(flat=False)
    inp = C.BlendInputs(
        target, patch, mask, _placement((10, 12)), C.canvas_window((H, W), (PH, PW), (10, 12))
    )
    crop = crop_window(inp, 16)
    assert crop is not None
    assert (crop.ys, crop.xs) == (slice(0, 12 + 4 + 11 + 16), slice(0, 10 + 6 + 18 + 16))
    # 캔버스 밖(창 안)은 대상 자신, 캔버스 안은 패치, 마스크는 그 자리에
    assert np.array_equal(crop.src[:12], crop.dst[:12]) and np.array_equal(
        crop.dst, target[:43, :50]
    )
    assert np.array_equal(crop.src[12 : 12 + PH, 10 : 10 + PW], patch)
    assert (
        np.array_equal(crop.mask[12 : 12 + PH, 10 : 10 + PW], mask)
        and crop.mask.sum() == mask.sum()
    )
    empty = C.BlendInputs(target, patch, np.zeros_like(mask), inp.placement, inp.window)
    assert crop_window(empty, 8) is None


def test_multiband_is_deterministic_and_differs_from_alpha() -> None:
    patch, mask = _patch()
    target = _target(flat=False)
    a = _multiband(3).apply(_ctx(target, patch, mask, (10, 12)))
    b = _multiband(3).apply(_ctx(target, patch, mask, (10, 12)))
    assert np.array_equal(a.composite, b.composite) and a.log == b.log
    alpha = AlphaBlend(R.AlphaBlendConfig(feather_px=3), {}).apply(
        _ctx(target, patch, mask, (10, 12))
    )
    assert not np.array_equal(a.composite, alpha.composite)

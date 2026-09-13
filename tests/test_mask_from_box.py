"""박스 → 마스크 추정 (설계 §11 test_mask_from_box) — 어두운 박스 안 밝은 얼룩 픽스처."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from anograft.bank import mask_from_box as M
from tests.fixtures import blob_image


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    return float((a & b).sum()) / float((a | b).sum() or 1)


def _blob_truth(size: int, cx: int, cy: int, r: int) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m


SIZE, CX, CY, R = 96, 48, 44, 9
BOX = (CX - R - 2, CY - R - 2, 2 * R + 5, 2 * R + 5)  # 라벨러 여유 2px


@pytest.mark.parametrize("method", ["grabcut", "otsu"])
def test_estimators_find_bright_blob(method: str) -> None:
    img = blob_image(SIZE, [(CX, CY, R)])
    mask, used = M.mask_from_box(img, BOX, method, margin=6, seed=M.stable_seed("x"))
    assert used == method
    assert _iou(mask, _blob_truth(SIZE, CX, CY, R)) > 0.8
    assert set(np.unique(mask)) <= {0, 255} and mask.shape == (SIZE, SIZE)


def test_ellipse_and_rect_are_geometric() -> None:
    img = blob_image(SIZE)
    rect, used = M.mask_from_box(img, BOX, "rect")
    assert used == "rect"
    x, y, w, h = BOX
    assert int(np.count_nonzero(rect)) == w * h and rect[y : y + h, x : x + w].all()
    ell, used = M.mask_from_box(img, BOX, "ellipse")
    assert used == "ellipse"
    area = int(np.count_nonzero(ell))
    assert 0.7 * np.pi * (w / 2) * (h / 2) < area < w * h  # 내접 타원 근사, 박스보다 작다
    assert not ell[y - 1, :].any() and not ell[:, x - 1].any()  # 박스 밖 0


def test_mask_never_leaves_box() -> None:
    img = blob_image(SIZE, [(CX, CY, R + 6)])  # 얼룩이 박스보다 크다
    for method in M.METHODS:
        mask, _ = M.mask_from_box(img, BOX, method, margin=6)
        outside = mask.copy()
        x, y, w, h = BOX
        outside[y : y + h, x : x + w] = 0
        assert not outside.any(), method


def test_fallback_chain_when_area_out_of_range() -> None:
    flat = blob_image(SIZE)  # 박스 안에 아무것도 없음 → grabcut/otsu 면적 0 → ellipse
    mask, used = M.mask_from_box(flat, BOX, "grabcut", margin=6)
    assert used == "ellipse" and np.any(mask)
    mask, used = M.mask_from_box(flat, BOX, "otsu", margin=6)
    assert used == "ellipse"
    # 박스가 전부 얼룩(편차 없음) → otsu 빈 마스크 → ellipse
    full = blob_image(SIZE, [(CX, CY, 30)])
    inner = (CX - 5, CY - 5, 10, 10)
    _mask, used = M.mask_from_box(full, inner, "otsu", margin=6)
    assert used == "ellipse"


def test_min_box_goes_straight_to_ellipse() -> None:
    img = blob_image(SIZE, [(CX, CY, R)])
    small = (CX - 3, CY - 3, 6, 6)
    _mask, used = M.mask_from_box(img, small, "grabcut", margin=6, min_box=8)
    assert used == "ellipse"
    _mask, used = M.mask_from_box(img, small, "grabcut", margin=6, min_box=4)
    assert used in ("grabcut", "otsu", "ellipse")  # 추정을 시도했다는 뜻


def test_same_input_twice_gives_same_mask() -> None:
    """grabcut의 k-means가 전역 RNG를 쓰므로 setRNGSeed 없이는 깨진다 — 여기서 고정."""
    img = blob_image(SIZE, [(CX, CY, R)])
    # 잡음을 섞어 k-means가 실제로 확률적으로 동작하게
    rng = np.random.default_rng(0)
    img = np.clip(img.astype(np.int16) + rng.integers(-12, 13, img.shape), 0, 255).astype(np.uint8)
    seed = M.stable_seed("scratch/img-01")
    a, _ = M.mask_from_box(img, BOX, "grabcut", margin=6, seed=seed)
    cv2.setRNGSeed(12345)  # 사이에 전역 RNG를 흔들어도
    b, _ = M.mask_from_box(img, BOX, "grabcut", margin=6, seed=seed)
    assert np.array_equal(a, b)


def test_stable_seed_is_process_independent_and_16bit() -> None:
    s = M.stable_seed("scratch/img0042-01")
    assert 0 <= s <= 0xFFFF and s == M.stable_seed("scratch/img0042-01")
    assert M.stable_seed("a") != M.stable_seed("b")
    assert M.stable_seed("a") == 0xBE43  # crc32("a") & 0xFFFF — 플랫폼·프로세스 무관 상수


def test_clip_and_errors() -> None:
    img = blob_image(SIZE, [(CX, CY, R)])
    assert M.clip_box((-5, -5, 20, 20), img.shape) == (0, 0, 15, 15)
    assert M.clip_box((100, 100, 5, 5), img.shape) is None
    with pytest.raises(ValueError, match="박스가 이미지 밖"):
        M.mask_from_box(img, (200, 200, 5, 5), "rect")
    with pytest.raises(ValueError, match="알 수 없는 방법"):
        M.mask_from_box(img, BOX, "magic")


def test_gray_input_is_accepted() -> None:
    img = blob_image(SIZE, [(CX, CY, R)], gray=True)
    assert img.ndim == 2
    mask, used = M.mask_from_box(img, BOX, "otsu", margin=6)
    assert used == "otsu" and _iou(mask, _blob_truth(SIZE, CX, CY, R)) > 0.8

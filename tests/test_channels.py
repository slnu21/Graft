from pathlib import Path

import cv2
import numpy as np
import pytest

from anograft.core.channels import binarize, demote_from_bgr, promote_to_bgr
from anograft.io import imgio


def _color(h: int = 12, w: int = 16) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 10
    img[..., 1] = 120
    img[..., 2] = 230
    return img


def _gray(h: int = 12, w: int = 16) -> np.ndarray:
    return (np.arange(h * w, dtype=np.uint8).reshape(h, w) * 3).astype(np.uint8)


def test_promote_then_demote_is_lossless_for_gray() -> None:
    g = _gray()
    p = promote_to_bgr(g)
    assert p.shape == (12, 16, 3)
    assert np.array_equal(p[..., 0], p[..., 1]) and np.array_equal(p[..., 1], p[..., 2])
    assert np.array_equal(demote_from_bgr(p, gray=True), g)


def test_demote_keeps_color_when_not_gray() -> None:
    c = _color()
    assert demote_from_bgr(c, gray=False) is c


def test_binarize_thresholds_at_127() -> None:
    m = np.array([[0, 127, 128, 255]], dtype=np.uint8)
    assert binarize(m).tolist() == [[0, 0, 255, 255]]
    assert binarize(m).dtype == np.uint8


@pytest.mark.parametrize("suffix", [".png", ".bmp"])
def test_write_read_roundtrip_color_and_gray(tmp_path: Path, suffix: str) -> None:
    c = _color()
    imgio.write_image(tmp_path / f"c{suffix}", c)
    back, gray = imgio.read_image(tmp_path / f"c{suffix}")
    assert gray is False and np.array_equal(back, c)

    g = _gray()
    imgio.write_image(tmp_path / f"g{suffix}", g)
    back, gray = imgio.read_image(tmp_path / f"g{suffix}")
    assert gray is True and back.shape == (12, 16, 3)
    assert np.array_equal(demote_from_bgr(back, gray=True), g)


def test_korean_path_roundtrip(tmp_path: Path) -> None:
    """cv2.imread가 실패하는 경로에서도 imgio는 동작해야 한다."""
    d = tmp_path / "한글 폴더" / "결함"
    d.mkdir(parents=True)
    p = d / "이미지 01.png"
    imgio.write_image(p, _color())
    back, _ = imgio.read_image(p)
    assert np.array_equal(back, _color())


def test_read_mask_binarizes_and_drops_channels(tmp_path: Path) -> None:
    m = np.zeros((8, 8, 3), dtype=np.uint8)
    m[2:5, 2:5] = 200
    m[6, 6] = 100  # 회색 → 0
    imgio.write_image(tmp_path / "m.png", m)
    mask = imgio.read_mask(tmp_path / "m.png")
    assert mask.shape == (8, 8) and set(np.unique(mask).tolist()) <= {0, 255}
    assert mask[3, 3] == 255 and mask[6, 6] == 0


def test_alpha_channel_is_dropped(tmp_path: Path) -> None:
    bgra = np.dstack([_color(), np.full((12, 16), 77, dtype=np.uint8)])
    ok, buf = cv2.imencode(".png", bgra)
    assert ok
    (tmp_path / "a.png").write_bytes(buf.tobytes())
    back, gray = imgio.read_image(tmp_path / "a.png")
    assert gray is False and back.shape == (12, 16, 3)


def test_16bit_is_rejected(tmp_path: Path) -> None:
    img16 = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 1000).astype(np.uint16)
    ok, buf = cv2.imencode(".png", img16)
    assert ok
    (tmp_path / "x.png").write_bytes(buf.tobytes())
    with pytest.raises(imgio.ImageReadError, match="8-bit"):
        imgio.read_image(tmp_path / "x.png")


def test_missing_file_and_bad_suffix(tmp_path: Path) -> None:
    with pytest.raises(imgio.ImageReadError):
        imgio.read_image(tmp_path / "nope.png")
    with pytest.raises(ValueError):
        imgio.write_image(tmp_path / "x.txt", _color())


def test_list_images_sorted_by_name(tmp_path: Path) -> None:
    for name in ["b.png", "a.PNG", "c.jpg", "notes.txt"]:
        imgio.write_image(tmp_path / name, _color()) if name != "notes.txt" else (
            tmp_path / name
        ).write_text("x")
    assert [p.name for p in imgio.list_images(tmp_path)] == ["a.PNG", "b.png", "c.jpg"]


def test_read_path_list_relative_to_list_file(tmp_path: Path) -> None:
    lst = tmp_path / "normals.txt"
    lst.write_text("# 주석\nimgs/a.png\n\n  imgs/b.png  \n", encoding="utf-8")
    paths = imgio.read_path_list(lst)
    assert paths == [tmp_path / "imgs" / "a.png", tmp_path / "imgs" / "b.png"]

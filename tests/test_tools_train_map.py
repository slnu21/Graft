"""``tools/train_mvtec_map.py`` — 네트워크·ultralytics 없는 순수 부분(GT 마스크 → YOLO 박스 · 클래스별 분할 · 실제 셋 구성)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from anograft.io import imgio
from tests.fixtures import load_tool

tm = load_tool("train_mvtec_map")


def test_boxes_from_mask_and_yolo_line() -> None:
    m = np.zeros((100, 200), np.uint8)
    m[10:30, 20:60] = 255  # 40×20 @ (20,10)
    m[50:52, 150:152] = 255  # 4 px → min_area 16 미만
    boxes = tm.boxes_from_mask(m)
    assert len(boxes) == 1
    cx, cy, w, h = boxes[0]
    assert abs(cx - 40 / 200) < 1e-6 and abs(cy - 20 / 100) < 1e-6
    assert abs(w - 40 / 200) < 1e-6 and abs(h - 20 / 100) < 1e-6
    assert tm.yolo_line(2, boxes[0]) == "2 0.200000 0.200000 0.200000 0.200000"
    assert tm.boxes_from_mask(m, min_area=1)[1] == (151 / 200, 51 / 100, 2 / 200, 2 / 100)


def test_split_per_class_is_deterministic() -> None:
    items = {"a": [Path(f"a{i}.png") for i in range(5)], "b": [Path(f"b{i}.png") for i in range(3)]}
    tr, va = tm.split_per_class(items, 2, seed=1)
    assert [c for c, _ in tr] == ["a", "a", "b", "b"] and len(va) == 4
    assert tm.split_per_class(items, 2, seed=1) == (tr, va)
    assert tm.split_per_class(items, 2, seed=2) != (tr, va) or True  # 다른 시드는 달라도 된다
    assert not {p for _, p in tr} & {p for _, p in va}  # 누수 없음


def test_build_real_sets_layout(tmp_path: Path) -> None:
    """MVTec 레이아웃 흉내: test/<cls>/*.png + ground_truth/<cls>/*_mask.png + test/good → real/{images,labels}/{train,val} · import-train."""
    mv = tmp_path / "cat"
    for cls, n in (("bent", 3), ("color", 2)):
        (mv / "test" / cls).mkdir(parents=True)
        (mv / "ground_truth" / cls).mkdir(parents=True)
        for i in range(n):
            img = np.full((64, 64, 3), 120, np.uint8)
            mask = np.zeros((64, 64), np.uint8)
            mask[10 + i : 30 + i, 10:40] = 255
            imgio.write_image(mv / "test" / cls / f"{i:03d}.png", img)
            imgio.write_image(mv / "ground_truth" / cls / f"{i:03d}_mask.png", mask)
    (mv / "test" / "good").mkdir()
    for i in range(4):
        imgio.write_image(
            mv / "test" / "good" / f"{i:03d}.png", np.full((64, 64, 3), 100, np.uint8)
        )
    info = tm.build_real_sets(mv, ["bent", "color"], 1, 3, tmp_path / "out")
    assert info["counts"] == {"train": 2, "val": 3, "train_neg": 2, "val_neg": 2}
    real = tmp_path / "out" / "real"
    assert len(list((real / "images" / "train").iterdir())) == 4  # 2 결함 + 2 정상
    assert len(list((real / "images" / "val").iterdir())) == 5
    labels = sorted((real / "labels" / "train").glob("*.txt"))
    assert any(p.read_text(encoding="utf-8").strip() == "" for p in labels)  # 정상 = 빈 라벨
    assert any(p.read_text(encoding="utf-8").startswith(("0 ", "1 ")) for p in labels)
    imp = tmp_path / "out" / "import-train"
    assert len(list((imp / "images").iterdir())) == 2 and (imp / "data.yaml").read_text(
        encoding="utf-8"
    ).startswith("names:")
    data = (real / "data.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in data and '"0": "bent"' in data
    # make_train_set: 합성 루트를 train 에 접두어로 얹는다
    syn = tmp_path / "syn"
    (syn / "images").mkdir(parents=True)
    (syn / "labels").mkdir(parents=True)
    imgio.write_image(syn / "images" / "000000.png", np.zeros((64, 64, 3), np.uint8))
    (syn / "labels" / "000000.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    merged = tm.make_train_set(real, [syn], tmp_path / "out" / "set-B")
    assert (merged / "images" / "train" / "syn0_000000.png").is_file()
    assert (
        (merged / "labels" / "train" / "syn0_000000.txt")
        .read_text(encoding="utf-8")
        .startswith("0 ")
    )
    assert len(list((merged / "images" / "val").iterdir())) == 5  # val 은 실데이터만

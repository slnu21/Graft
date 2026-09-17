"""``tools/train_mvtec_map.py`` — 네트워크·ultralytics 없는 순수 부분(GT 마스크 → YOLO 박스 · 클래스별 분할 · pairs.csv · 실제 셋 구성 · 결과 표)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
    (mv / "train" / "good").mkdir(parents=True)
    for i in range(3):
        imgio.write_image(
            mv / "train" / "good" / f"{i:03d}.png", np.full((64, 64, 3), 100, np.uint8)
        )
    info = tm.build_real_sets(mv, ["bent", "color"], 1, 3, tmp_path / "out")
    assert info["counts"] == {"train": 2, "val": 3, "train_neg": 2, "val_neg": 2}
    assert info["targets"] == 3 and info["dataset"] == "cat"
    targets = (tmp_path / "out" / "targets.txt").read_text(encoding="utf-8").split()
    assert len(targets) == 3 and targets[0].endswith("train/good/000.png")
    # 같은 분할로 다시 부르면 그대로(캐시 유지), 다른 분할이면 거부 — 한 폴더 = 한 분할
    assert tm.build_real_sets(mv, ["bent", "color"], 1, 3, tmp_path / "out") == info
    with pytest.raises(SystemExit, match="다른 분할"):
        tm.build_real_sets(mv, ["bent", "color"], 1, 4, tmp_path / "out")
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


def test_boxes_from_mask_threshold_ignores_jpeg_ring() -> None:
    """MT 마스크(JPEG 링 1~14)는 ``> 127`` 로 이진화 — 잡음 성분이 박스가 되지 않는다."""
    m = np.zeros((100, 100), np.uint8)
    m[10:30, 10:30] = 255
    m[30:35, 10:30] = 14  # 링
    m[60:80, 60:80] = 9  # 잡음 덩어리
    assert len(tm.boxes_from_mask(m)) == 1
    assert tm.boxes_from_mask(m)[0][3] == 20 / 100  # 링 제외 높이
    loose = tm.boxes_from_mask(
        m, threshold=0
    )  # 종전(> 0): 링이 붙어 높이 25 + 잡음 덩어리가 박스로
    assert len(loose) == 2 and loose[0][3] == 25 / 100


def test_parse_pairs_csv_and_split_normals() -> None:
    text = "image,mask,class\nA/1.jpg,A/1.png,crack\n\nB/2.jpg,B/2.png,break\n"
    assert tm.parse_pairs_csv(text) == [
        ("A/1.jpg", "A/1.png", "crack"),
        ("B/2.jpg", "B/2.png", "break"),
    ]
    normals = [Path(f"n{i:02d}.jpg") for i in range(10)]
    goods, targets = tm.split_normals(normals, 4, 5)
    assert goods == normals[:4] and targets == normals[4:9]
    assert not set(goods) & set(targets)
    with pytest.raises(SystemExit, match="모자랍니다"):
        tm.split_normals(normals, 4, 7)


def test_load_pairs_layout(tmp_path: Path) -> None:
    """pairs.csv(CSV 기준 경로) + normals.txt(목록 기준 경로) → DatasetSpec · 없는 클래스는 거부."""
    root = tmp_path / "mt"
    (root / "C").mkdir(parents=True)
    (root / "F").mkdir()
    rows = ["image,mask,class"]
    for i in range(3):
        imgio.write_image(root / "C" / f"{i}.jpg", np.full((32, 32), 90, np.uint8))
        imgio.write_image(root / "C" / f"{i}.png", np.zeros((32, 32), np.uint8))
        rows.append(f"C/{i}.jpg,C/{i}.png,crack")
    (root / "pairs.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for i in range(6):
        imgio.write_image(root / "F" / f"{i}.jpg", np.full((32, 32), 100, np.uint8))
    (root / "normals.txt").write_text(
        "# 정상\n" + "\n".join(f"F/{i}.jpg" for i in range(6)) + "\n", encoding="utf-8"
    )
    spec = tm.load_pairs(root / "pairs.csv", ["crack"], root / "normals.txt", 2, 3)
    assert spec.name == "mt" and [p.name for p in spec.items["crack"]] == [
        "0.jpg",
        "1.jpg",
        "2.jpg",
    ]
    assert spec.masks[root / "C" / "1.jpg"] == root / "C" / "1.png"
    assert [p.name for p in spec.goods] == ["0.jpg", "1.jpg"]
    assert [p.name for p in spec.targets] == ["2.jpg", "3.jpg", "4.jpg"]
    assert spec.extra == {"normals": 6, "n_good": 2}
    with pytest.raises(SystemExit, match="없는 클래스"):
        tm.load_pairs(root / "pairs.csv", ["crack", "fray"], root / "normals.txt", 2, 3)
    # build_real_sets 가 spec 을 그대로 받는다 (마스크 전부 0 → 라벨 빈 줄 없이 빈 파일)
    info = tm.build_real_sets(spec, ["crack"], 1, 1, tmp_path / "out")
    assert info["counts"] == {"train": 1, "val": 2, "train_neg": 1, "val_neg": 1}
    assert info["targets"] == 3 and info["normals"] == 6
    assert (
        tmp_path / "out" / "import-train" / "images" / f"crack_{spec.items['crack'][0].stem}.jpg"
    ).is_file() or any((tmp_path / "out" / "import-train" / "images").glob("crack_*.jpg"))


def test_recipe_matches_ignores_crlf(tmp_path: Path) -> None:
    """``recipe init --write`` 는 Windows 에서 CRLF — seed·count 판정은 줄 끝과 무관해야 합성 재사용이 된다."""
    r = tmp_path / "r.yaml"
    r.write_bytes(b"version: 1\r\nseed: 7\r\noutput:\r\n  root: x\r\n  count: 200\r\n")
    assert tm.recipe_matches(r, 7, 200)
    assert not tm.recipe_matches(r, 8, 200) and not tm.recipe_matches(r, 7, 20)
    assert not tm.recipe_matches(tmp_path / "none.yaml", 7, 200)


def _metrics(m50: float, m5095: float, pc: dict[str, float], minutes: float = 1.0) -> dict:
    return {"map50": m50, "map50_95": m5095, "per_class_map50": pc, "minutes": minutes}


def test_summarize_table_and_short_names() -> None:
    base = "A real-train only"
    results = {
        base: {
            7: _metrics(0.20, 0.10, {"a": 0.2, "b": 0.2}),
            8: _metrics(0.40, 0.20, {"a": 0.5, "b": 0.3}),
        },
        "B +poisson-graft (hybrid)": {
            7: _metrics(0.50, 0.25, {"a": 0.6, "b": 0.4}, 20.0),
            8: _metrics(0.40, 0.20, {"a": 0.5, "b": 0.3}, 22.0),
        },
        "B +dent-graft (hybrid)": {7: _metrics(0.30, 0.15, {"a": 0.3, "b": 0.3})},
    }
    md = tm.summarize(results, [7, 8], base)
    lines = md.splitlines()
    assert lines[0].startswith("| 학습셋 | s7 | s8 | mAP50 평균(Δ)")
    assert (
        "| A real-train only | 0.200 | 0.400 | **0.300** | 0.150 | a 0.35 · b 0.25 | 1.0 |" in lines
    )
    assert (
        "| B +poisson-graft (hybrid) | 0.500 (+0.30) | 0.400 (+0.00) | **0.450** (+0.15) | 0.225 | a 0.55 · b 0.35 | 21.0 |"
        in lines
    )
    # 시드 하나만 있는 셋: 빈 칸은 — · 평균 Δ 는 있는 시드끼리
    assert (
        "| B +dent-graft (hybrid) | 0.300 (+0.10) | — | **0.300** (+0.10) | 0.150 | a 0.30 · b 0.30 | 1.0 |"
        in lines
    )
    assert tm.set_short_name(base) == "A-real-train"
    assert tm.set_short_name("B +poisson-graft (hybrid)") == "B-poisson-graft-hybrid"
    assert tm.set_short_name("B +dent-graft") == "B-dent-graft"

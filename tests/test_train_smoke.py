"""``tools/train_smoke.py`` — anograft 출력을 기존 YOLO 셋에 합치는 순수 로직(``merge_yolo_sets``).

학습 자체(ultralytics)는 선택 의존성이라 여기서 돌리지 않는다 — 없으면 종료 코드 2 + 안내만 확인.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from tools.train_smoke import main, merge_yolo_sets, read_names


def _yolo_set(root: Path, names: list[str], stems: list[str], *, labeled: set[str]) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for s in stems:
        (root / "images" / f"{s}.png").write_bytes(b"\x89PNG" + s.encode())
        if s in labeled:
            (root / "labels" / f"{s}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        elif s.endswith("0"):  # 정상 두 표현: 빈 파일 / 파일 없음
            (root / "labels" / f"{s}.txt").write_text("", encoding="utf-8")
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names}), encoding="utf-8")
    return root


def test_merge_with_base_uses_base_as_val_and_prefixes(tmp_path: Path) -> None:
    syn = _yolo_set(tmp_path / "syn", ["a", "b"], ["000", "001", "n_x"], labeled={"000", "001"})
    base = _yolo_set(tmp_path / "base", ["a", "b"], ["img0", "img1"], labeled={"img1"})
    s = merge_yolo_sets(syn, tmp_path / "m", base=base)
    assert (s.n_train, s.n_val, s.names) == (5, 2, ("a", "b"))
    assert s.n_labels_empty == 2  # n_x(합성 정상) + img0(base 빈 라벨)
    train = sorted(p.name for p in (tmp_path / "m" / "images" / "train").iterdir())
    assert train == ["base_img0.png", "base_img1.png", "syn_000.png", "syn_001.png", "syn_n_x.png"]
    val = sorted(p.name for p in (tmp_path / "m" / "images" / "val").iterdir())
    assert val == ["base_img0.png", "base_img1.png"]
    # 라벨은 이미지마다 하나(정상은 빈 파일), 내용 보존
    labels = sorted(p.name for p in (tmp_path / "m" / "labels" / "train").iterdir())
    assert labels == [n.replace(".png", ".txt") for n in train]
    assert (
        tmp_path / "m" / "labels" / "train" / "syn_000.txt"
    ).read_text() == "0 0.5 0.5 0.1 0.1\n"
    assert (tmp_path / "m" / "labels" / "train" / "syn_n_x.txt").read_text() == ""
    data = yaml.safe_load((tmp_path / "m" / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == ["a", "b"] and data["nc"] == 2
    assert data["train"] == "images/train" and data["val"] == "images/val"
    assert Path(data["path"]).resolve() == (tmp_path / "m").resolve()


def test_merge_without_base_holds_out_synthetic_slice(tmp_path: Path) -> None:
    stems = [f"{i:03d}" for i in range(10)]
    syn = _yolo_set(tmp_path / "syn", ["a"], stems, labeled=set(stems))
    s = merge_yolo_sets(syn, tmp_path / "m")
    assert (s.n_train, s.n_val) == (10, 2)  # 20% 앞에서
    val = sorted(p.name for p in (tmp_path / "m" / "images" / "val").iterdir())
    assert val == ["syn_000.png", "syn_001.png"]


def test_merge_rejects_class_mismatch_and_empty(tmp_path: Path) -> None:
    syn = _yolo_set(tmp_path / "syn", ["a", "b"], ["000"], labeled={"000"})
    base = _yolo_set(tmp_path / "base", ["b", "a"], ["x"], labeled={"x"})
    with pytest.raises(ValueError, match="클래스 이름/순서"):
        merge_yolo_sets(syn, tmp_path / "m", base=base)
    empty = tmp_path / "empty"
    (empty / "images").mkdir(parents=True)
    (empty / "data.yaml").write_text(yaml.safe_dump({"names": ["a"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="합성 이미지가 없습니다"):
        merge_yolo_sets(empty, tmp_path / "m2")


def test_read_names_accepts_dict_form(tmp_path: Path) -> None:
    p = tmp_path / "d.yaml"
    p.write_text(yaml.safe_dump({"names": {1: "b", 0: "a"}}), encoding="utf-8")
    assert read_names(p) == ["a", "b"]
    p.write_text(yaml.safe_dump({"names": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        read_names(p)


def test_main_dry_run_and_missing_ultralytics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    syn = _yolo_set(tmp_path / "syn", ["a"], ["000", "001"], labeled={"000"})
    assert main(["--synthetic", str(syn), "--out", str(tmp_path / "m"), "--dry-run"]) == 0
    assert "train 2장" in capsys.readouterr().out
    # ultralytics 없음 → 2 + 안내 (있는 환경이면 import 를 막아 같은 경로를 강제)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    assert main(["--synthetic", str(syn), "--out", str(tmp_path / "m2")]) == 2
    assert "pip install ultralytics" in capsys.readouterr().err
    # 잘못된 입력 → 1
    assert main(["--synthetic", str(tmp_path / "nope"), "--out", str(tmp_path / "m3")]) == 1

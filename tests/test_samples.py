"""``anograft.samples.yolo`` — 샘플 YOLO 세트 생성기(zip 배포의 "5분 시작" 진입점).

- 레이아웃(images/·labels/·data.yaml) · 정상 이미지의 두 표현(빈 라벨 파일 / 파일 없음) · 라벨 줄 형식과 범위
- 같은 인자 → 같은 바이트(재현) · ``gray_every`` → 1ch PNG
- ``anograft sample`` 서브커맨드 · ``tools/make_sample_yolo.py`` 셔틀이 같은 인자로 같은 결과
- 생성 결과가 ``import-yolo``로 그대로 은행이 된다(5분 시작의 첫 두 단계)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from anograft.bank import Bank
from anograft.bank.importers import yolo as yolo_importer
from anograft.cli import EXIT_OK, main
from anograft.io import imgio
from anograft.samples import yolo as sample_yolo

SMALL = {"n_normal": 3, "n_defect": 2, "size": (240, 180), "seed": 3}


def _tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_generate_layout_labels_and_normals(tmp_path: Path) -> None:
    s = sample_yolo.generate(tmp_path / "s", **SMALL)
    assert s.total == 5 and s.names == ("scratch", "pit", "stain") and s.n_boxes >= 2
    root = tmp_path / "s"
    imgs = sorted((root / "images").glob("*.png"))
    assert [p.stem for p in imgs] == [f"plate_{i:03d}" for i in range(5)]
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == ["scratch", "pit", "stain"] and data["train"] == "images"
    # 정상: 짝수 인덱스는 빈 라벨 파일, 홀수는 파일 없음
    assert (root / "labels" / "plate_000.txt").read_text() == ""
    assert not (root / "labels" / "plate_001.txt").exists()
    assert (root / "labels" / "plate_002.txt").read_text() == ""
    # 결함: 줄마다 5토큰, 클래스 0..2, 좌표 0..1
    n_lines = 0
    for i in (3, 4):
        lines = (root / "labels" / f"plate_{i:03d}.txt").read_text(encoding="utf-8").splitlines()
        assert lines
        for ln in lines:
            cid, cx, cy, w, h = ln.split()
            assert int(cid) in (0, 1, 2)
            assert all(0.0 < float(v) <= 1.0 for v in (cx, cy, w, h))
        n_lines += len(lines)
    assert n_lines == s.n_boxes
    img, gray = imgio.read_image(imgs[0])
    assert img.shape == (180, 240, 3) and not gray


def test_generate_is_reproducible_and_seed_matters(tmp_path: Path) -> None:
    sample_yolo.generate(tmp_path / "a", **SMALL)
    sample_yolo.generate(tmp_path / "b", **SMALL)
    assert _tree(tmp_path / "a") == _tree(tmp_path / "b")
    sample_yolo.generate(tmp_path / "c", **{**SMALL, "seed": 4})
    assert _tree(tmp_path / "c") != _tree(tmp_path / "a")


def test_gray_every_writes_single_channel(tmp_path: Path) -> None:
    sample_yolo.generate(tmp_path / "g", **{**SMALL, "gray_every": 2})
    img1, gray1 = imgio.read_image(tmp_path / "g" / "images" / "plate_001.png")
    img0, gray0 = imgio.read_image(tmp_path / "g" / "images" / "plate_000.png")
    # read_image 는 3ch 로 승격해 돌려주고 gray 플래그로 원본 채널을 알린다
    assert gray1 and img1.ndim == 3 and (img1[..., 0] == img1[..., 2]).all()
    assert not gray0 and img0.ndim == 3 and (img0[..., 0] != img0[..., 2]).any()


def test_generate_rejects_bad_args(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        sample_yolo.generate(tmp_path / "x", n_normal=0, n_defect=0)
    with pytest.raises(ValueError):
        sample_yolo.generate(tmp_path / "x", size=(64, 64))


def test_cli_sample_and_tool_shim_match(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["--n-normal", "3", "--n-defect", "2", "--size", "240", "180", "--seed", "3"]
    assert main(["sample", "--out", str(tmp_path / "cli"), *args]) == EXIT_OK
    out = capsys.readouterr().out
    assert "이미지 5장" in out and "names ['scratch', 'pit', 'stain']" in out
    assert sample_yolo.main(["--out", str(tmp_path / "tool"), *args]) == 0
    assert _tree(tmp_path / "cli") == _tree(tmp_path / "tool")
    # 잘못된 인자는 종료 코드 1 + stderr
    assert (
        main(["sample", "--out", str(tmp_path / "bad"), "--n-normal", "0", "--n-defect", "0"]) == 1
    )
    assert "샘플 생성 실패" in capsys.readouterr().err


def test_sample_feeds_import_yolo(tmp_path: Path) -> None:
    """5분 시작의 첫 두 단계: sample → import-yolo 가 클래스 3개 은행을 만든다."""
    root = tmp_path / "s"
    sample_yolo.generate(root, n_normal=2, n_defect=4, size=(320, 240), seed=5)
    normals = root / "normals.txt"
    yolo_importer.import_yolo(
        images=root / "images",
        labels=root / "labels",
        names=root / "data.yaml",
        out=tmp_path / "bank",
        list_normals=normals,
    )
    bank = Bank.load(tmp_path / "bank")
    assert bank.classes == ["scratch", "pit", "stain"] and len(bank) >= 4
    assert len(imgio.read_path_list(normals)) == 2

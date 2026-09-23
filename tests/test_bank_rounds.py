"""T4 — 은행 스냅샷과 평가셋(holdout) 거부.

여기가 지키는 것: **평가셋은 은행에 못 들어간다**(코드가 막는다, 사람 규율이 아니라) ·
스냅샷은 **복사가 아니라 목록**이고 내용이 바뀌면 잡아낸다 · 폴더를 옮겨도 같은 은행으로 본다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anograft.bank import holdout as ho
from anograft.bank import snapshot as snap
from anograft.bank.bank import Bank, BankError
from anograft.bank.importers import yolo as Y
from anograft.cli import EXIT_OK, EXIT_RECIPE_ERROR, main
from tests.fixtures import fake_yolo_dataset


@pytest.fixture
def imported(tmp_path: Path) -> tuple[Path, dict]:
    """샘플 YOLO → 은행. (은행 경로, 데이터셋 정보)"""
    d = fake_yolo_dataset(tmp_path / "ds")
    bank = tmp_path / "bank"
    Y.import_yolo(d["images"], d["labels"], d["names"], bank, mask_from="rect")
    return bank, d


# ---------------------------------------------------------------- holdout 파싱(순수)


def test_parse_holdout_forms():
    text = "\n".join(
        [
            "# 고정 평가셋",
            "plate_0007",
            "plate_0031.png",
            "images/plate_0042.png",
            "  spaced_name.jpg  # 뒤 주석",
            "",
            "   ",
        ]
    )
    assert ho.parse_holdout(text) == {
        "plate_0007",
        "plate_0031",
        "plate_0042",
        "spaced_name",
    }


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("images/plate_0007.png", True),
        ("plate_0007.png", True),
        ("plate_0007", True),
        ("other/plate_0007.jpg", True),  # 폴더·확장자가 달라도 같은 사진으로 본다
        ("plate_00070.png", False),
        ("", False),  # 출처를 모르면 막지 않는다
    ],
)
def test_is_held_out(origin, expected):
    assert ho.is_held_out(origin, {"plate_0007"}) is expected


def test_read_holdout_missing_file_is_empty(tmp_path: Path):
    assert ho.read_holdout(tmp_path) == set()


# ---------------------------------------------------------------- 거부(모든 임포터 공통 경로)


def test_import_refuses_holdout_and_says_why(tmp_path: Path):
    d = fake_yolo_dataset(tmp_path / "ds")
    bank = tmp_path / "bank"
    bank.mkdir()
    stems = sorted(p.stem for p in Path(d["images"]).glob("*.png"))
    (bank / ho.HOLDOUT_FILE).write_text(f"# 평가셋\n{stems[0]}\n", encoding="utf-8")

    result = Y.import_yolo(d["images"], d["labels"], d["names"], bank, mask_from="rect")

    assert result.stats.held_out, "거부 기록이 없다"
    assert all(stems[0] in o for o in result.stats.held_out)
    assert any("평가셋" in w for w in result.stats.warnings), "사유를 남기지 않았다"
    # 그 원본에서 나온 소스는 은행에 없다
    loaded = Bank.load(bank)
    assert not [s for s in loaded.sources() if ho.normalize_stem(s.origin) == stems[0]]
    assert len(loaded) > 0, "나머지는 정상으로 들어가야 한다"


def test_cli_import_says_it_refused(tmp_path, capsys):
    """거부를 조용히 하지 않는다 — 요약에 몇 건을 왜 뺐는지 나와야 한다."""
    d = fake_yolo_dataset(tmp_path / "ds")
    bank = tmp_path / "bank"
    bank.mkdir()
    stem = sorted(p.stem for p in Path(d["images"]).glob("*.png"))[0]
    (bank / ho.HOLDOUT_FILE).write_text(stem + "\n", encoding="utf-8")

    assert (
        main(
            [
                "bank",
                "import-yolo",
                "--images",
                str(d["images"]),
                "--labels",
                str(d["labels"]),
                "--names",
                str(d["names"]),
                "--out",
                str(bank),
            ]
        )
        == EXIT_OK
    )
    captured = capsys.readouterr()
    assert "평가셋(holdout.txt)이라" in captured.err
    assert stem in captured.err


def test_violations_catches_already_imported(imported):
    """목록을 **나중에** 만든 경우 — 거부만으로는 못 잡는다."""
    bank, _ = imported
    loaded = Bank.load(bank)
    stem = ho.normalize_stem(loaded.sources()[0].origin)
    assert ho.violations(loaded.sources(), {stem})
    assert not ho.violations(loaded.sources(), {"없는이름"})


# ---------------------------------------------------------------- 스냅샷


def test_snapshot_lists_sources_not_copies(imported, tmp_path: Path):
    bank, _ = imported
    shot = snap.take(bank)
    assert shot.sources and len(shot.sources) == len(Bank.load(bank))
    assert all(len(s.digest) == snap.DIGEST_CHARS for s in shot.sources)

    out = snap.write(shot, tmp_path / "snap.json")
    assert out.stat().st_size < 100_000, "스냅샷은 목록이지 사본이 아니다"
    assert snap.load(out).to_dict() == shot.to_dict()


def test_snapshot_diff_is_empty_for_untouched_bank(imported, tmp_path: Path):
    bank, _ = imported
    shot = snap.take(bank)
    d = snap.diff(shot, bank)
    assert d.same and d.text() == "스냅샷과 같습니다."


def test_snapshot_survives_moving_the_bank(imported, tmp_path: Path):
    """경로는 해시에 안 들어간다 — 폴더를 옮겨도 같은 은행이다(설계 §6.2)."""
    bank, _ = imported
    shot = snap.take(bank)
    moved = tmp_path / "moved-bank"
    bank.rename(moved)
    assert snap.diff(shot, moved).same


def test_snapshot_detects_changed_mask(imported):
    """마스크를 다듬으면 '내용 바뀜' 으로 잡힌다 — 라운드 재현이 그 차이를 알아야 한다."""
    import numpy as np

    from anograft.io import imgio

    bank, _ = imported
    shot = snap.take(bank)
    first = sorted(shot.sources, key=lambda s: s.id)[0]
    cls, _, name = first.id.partition("/")
    mask_path = bank / cls / f"{name}.mask.png"
    mask = imgio.read_mask(mask_path)
    ys, xs = np.nonzero(
        mask
    )  # 크롭 위쪽은 이미 배경이라 **실제로 칠해진 픽셀**을 지워야 내용이 바뀐다
    mask[ys[0], xs[0]] = np.uint8(0)
    imgio.write_image(mask_path, mask)

    d = snap.diff(shot, bank)
    assert d.changed == [first.id]
    assert not d.missing and not d.added
    assert "내용 바뀜" in d.text()


def test_snapshot_detects_missing_and_added(imported, tmp_path: Path):
    bank, _ = imported
    shot = snap.take(bank)
    gone = sorted(shot.sources, key=lambda s: s.id)[0]
    cls, _, name = gone.id.partition("/")
    for suffix in (".png", ".mask.png", ".json"):
        (bank / cls / f"{name}{suffix}").unlink()

    d = snap.diff(shot, bank)
    assert d.missing == [gone.id]
    assert not d.added

    # 새 소스가 들어오면 added
    d2 = snap.diff(snap.Snapshot(name="x", created="", classes=[], sources=[]), bank)
    assert d2.added and not d2.missing


def test_snapshot_rejects_non_bank(tmp_path: Path):
    with pytest.raises(BankError):
        snap.take(tmp_path)


# ---------------------------------------------------------------- CLI


def test_cli_snapshot_and_verify(imported, tmp_path: Path, capsys):
    bank, _ = imported
    out = tmp_path / "snap.json"
    assert main(["bank", "snapshot", str(bank), "--out", str(out)]) == EXIT_OK
    assert out.is_file()

    assert main(["bank", "verify", str(bank), "--snapshot", str(out)]) == EXIT_OK
    text = capsys.readouterr().out
    assert "스냅샷과 같습니다" in text

    # 소스를 하나 지우면 verify 가 실패(1)로 알린다
    loaded = Bank.load(bank)
    cls, _, name = loaded.sources()[0].id.partition("/")
    for suffix in (".png", ".mask.png", ".json"):
        (bank / cls / f"{name}{suffix}").unlink()
    assert main(["bank", "verify", str(bank), "--snapshot", str(out)]) == EXIT_RECIPE_ERROR


def test_cli_verify_reports_holdout_leak(imported, capsys):
    bank, _ = imported
    stem = ho.normalize_stem(Bank.load(bank).sources()[0].origin)
    (bank / ho.HOLDOUT_FILE).write_text(f"{stem}\n", encoding="utf-8")

    assert main(["bank", "verify", str(bank)]) == EXIT_RECIPE_ERROR
    captured = capsys.readouterr()
    assert "평가셋이 은행에 있습니다" in captured.err
    assert "자동으로 지우지 않습니다" in captured.err

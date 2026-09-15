"""v0.7.x ``bank merge`` — 세 파일 그대로 복사(크롭·점수 보존) · 클래스 이름 병합(대상 순서 유지, 새 이름 끝) · id 충돌 -dup ·
--rename · --tags · 은행 기본 um_per_px 실체화 · 대상=소스 거부 · 이어 쓰기 · CLI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from anograft.bank import BANK_FILE, Bank
from anograft.bank.importers import yolo as Y
from anograft.bank.merge import MergeError, merge_banks, parse_rename
from anograft.cli import EXIT_OK, main
from tests.fixtures import fake_yolo_dataset


@pytest.fixture
def two_banks(tmp_path: Path) -> tuple[Path, Path]:
    d = fake_yolo_dataset(tmp_path / "ds")
    Y.import_yolo(
        d["images"], d["labels"], d["names"], tmp_path / "a", mask_from="rect", tags=["prodA"]
    )
    Y.import_yolo(
        d["images"],
        d["labels"],
        d["names"],
        tmp_path / "b",
        mask_from="ellipse",
        um_per_px=2.5,
        tags=["prodB"],
    )
    # b 는 은행 기본 um_per_px 만 두고 소스 메타의 um 을 지운다(실체화 확인용)
    meta = yaml.safe_load((tmp_path / "b" / BANK_FILE).read_text(encoding="utf-8"))
    meta["um_per_px"] = 2.5
    (tmp_path / "b" / BANK_FILE).write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    import json

    for p in (tmp_path / "b").rglob("*.json"):
        m = json.loads(p.read_text(encoding="utf-8"))
        m["um_per_px"] = None
        p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return tmp_path / "a", tmp_path / "b"


def test_parse_rename() -> None:
    assert parse_rename(["spot=dent", " crack = scratch "]) == {"spot": "dent", "crack": "scratch"}
    for bad in ("nope", "=x", "a=", "a=b/c"):
        with pytest.raises(MergeError):
            parse_rename([bad])


def test_merge_copies_preserves_and_dedups(two_banks: tuple[Path, Path], tmp_path: Path) -> None:
    a, b = two_banks
    out = tmp_path / "c"
    s = merge_banks([a, b], out, tags=["all"], rename={"crack": "scratch"})
    assert s.banks == 2 and s.copied == 10 and s.renamed == 2 and s.classes == ["spot", "scratch"]
    assert s.duplicates == 5  # 같은 원본 이름 5개 → -dup1
    bank = Bank.load(out)
    assert len(bank) == 10 and bank.classes == ["spot", "scratch"]
    ba = Bank.load(a)
    # a 의 소스는 바이트 그대로(크롭·마스크·점수), 태그 누적
    for src in ba.sources():
        new_cls = "scratch" if src.cls == "crack" else src.cls
        dst = next(x for x in bank.sources() if x.id == f"{new_cls}/{src.id.split('/', 1)[1]}")
        assert np.array_equal(dst.image, src.image) and np.array_equal(dst.mask, src.mask)
        assert dst.confidence == src.confidence and dst.mask_origin == src.mask_origin
        assert dst.tags == (*src.tags, "all") and dst.um_per_px is None
    # b 의 소스는 -dup1 로, crack → scratch, 은행 기본 um 2.5 실체화
    dups = [x for x in bank.sources() if x.id.endswith("-dup1")]
    assert len(dups) == 5 and all(x.um_per_px == 2.5 and "prodB" in x.tags for x in dups)
    assert {x.cls for x in dups} == {"spot", "scratch"} and not any(
        x.cls == "crack" for x in bank.sources()
    )
    hist = yaml.safe_load((out / BANK_FILE).read_text(encoding="utf-8"))["imports"][-1]
    assert (
        hist["importer"] == "merge"
        and hist["rename"] == {"crack": "scratch"}
        and hist["n_sources"] == 10
    )
    # 대상에 이어 쓰기 — 클래스 순서 유지, 다시 병합하면 또 dup
    s2 = merge_banks([a], out, rename={"crack": "scratch"})
    assert s2.copied == 5 and s2.duplicates == 5 and Bank.load(out).classes == ["spot", "scratch"]
    assert len(Bank.load(out)) == 15
    # 오류: 소스=대상 · 은행 아님 · 빈 목록
    with pytest.raises(MergeError, match="같습니다"):
        merge_banks([a, out], out)
    with pytest.raises(MergeError, match="은행이 아닙니다"):
        merge_banks([tmp_path / "nope"], tmp_path / "d")
    with pytest.raises(MergeError):
        merge_banks([], tmp_path / "d")


def test_cli_bank_merge(
    two_banks: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = two_banks
    assert (
        main(
            [
                "bank",
                "merge",
                str(a),
                str(b),
                "--out",
                str(tmp_path / "m"),
                "--rename",
                "crack=scratch",
                "--tags",
                "x",
            ]
        )
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "소스 10개 복사" in out and "id 충돌 5" in out and "['spot', 'scratch']" in out
    assert main(["bank", "merge", str(a), "--out", str(a)]) != EXIT_OK
    assert "병합 실패" in capsys.readouterr().err
    assert (
        main(["bank", "merge", str(a), "--out", str(tmp_path / "n"), "--rename", "bad"]) != EXIT_OK
    )


def test_merge_dedupe_by_content(
    two_banks: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """0.7.6+ ``dedupe`` — 같은 클래스에서 이미지·마스크 바이트가 같은 소스는 한 번만. a+a 는 절반, a+b(마스크 방법이 달라 내용 다름)는 그대로."""
    a, b = two_banks
    s = merge_banks([a, a], tmp_path / "aa", dedupe=True)
    assert s.copied == 5 and s.skipped_same == 5 and s.duplicates == 0
    assert len(Bank.load(tmp_path / "aa")) == 5
    s2 = merge_banks([a, a], tmp_path / "aa2")  # dedupe 없으면 -dup 로 둘 다
    assert s2.copied == 10 and s2.skipped_same == 0 and s2.duplicates == 5
    s3 = merge_banks([a, b], tmp_path / "ab", dedupe=True)  # 내용이 다르면 id 가 같아도 둘 다
    assert (
        s3.copied + s3.skipped_same == 10 and s3.copied >= 9
    )  # rect/ellipse 가 우연히 같은 작은 박스 하나까지
    from anograft.cli import EXIT_OK, main

    assert (
        main(["bank", "merge", str(a), str(a), "--out", str(tmp_path / "cli"), "--dedupe"])
        == EXIT_OK
    )
    assert "내용 중복 건너뜀 5" in capsys.readouterr().out

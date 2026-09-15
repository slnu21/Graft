"""v0.7 은행 탭 세션(Qt 없음) — ``row_of``·``filter_rows``(클래스·태그·저신뢰·추정·검색·정렬) · ``BankSession`` 열기/새로고침 ·
``delete``(세 파일 삭제, classes 유지, imports 이력) · ``replace_mask``(마스크 덮어쓰기, 메타 갱신, 점수 제거) · 오류. 파일 계층은
``BankWriter.delete/replace_mask``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from anograft.bank import BANK_FILE, Bank
from anograft.bank.importers import yolo as Y
from anograft.core.types import DefectSource
from anograft.gui.bank.session import BankSession, BankSessionError, filter_rows, row_of
from anograft.io import imgio
from tests.fixtures import fake_yolo_dataset, line_defect


def _src(
    i: int, cls: str = "a", conf: float | None = None, tags=(), area: int = 12
) -> DefectSource:
    s = line_defect(area, 3, cls=cls)
    origin = "yolo-box:grabcut" if conf is not None else "png"
    return DefectSource(
        f"{cls}/{i:03d}", cls, s.image, s.mask, None, tuple(tags), f"img{i}.png", origin, conf, ()
    )


def test_row_of_and_filter_rows() -> None:
    rows = [
        row_of(_src(0, conf=0.9, tags=("p1",), area=10)),
        row_of(_src(1, conf=0.2, tags=("p2",), area=30)),
        row_of(_src(2, cls="b", area=20)),
        row_of(_src(3, cls="b", conf=0.6, tags=("p1", "p2"), area=5)),
    ]
    r = rows[1]
    assert (
        r.id == "a/001"
        and r.name == "001"
        and r.estimated
        and r.low_confidence
        and r.size == (11, 38)
    )
    assert not rows[2].estimated and rows[2].confidence is None and not rows[2].low_confidence
    ids = lambda xs: [x.id for x in xs]  # noqa: E731
    assert ids(filter_rows(rows)) == ["a/000", "a/001", "b/002", "b/003"]
    assert ids(filter_rows(rows, cls="b")) == ["b/002", "b/003"]
    assert ids(filter_rows(rows, tag="p1")) == ["a/000", "b/003"]
    assert ids(filter_rows(rows, only_low=True)) == ["a/001"]
    assert ids(filter_rows(rows, only_estimated=True)) == ["a/000", "a/001", "b/003"]
    assert ids(filter_rows(rows, text="img2")) == ["b/002"] and ids(
        filter_rows(rows, text="P2")
    ) == ["a/001", "b/003"]
    assert ids(filter_rows(rows, sort="area")) == ["b/003", "a/000", "b/002", "a/001"]
    assert ids(filter_rows(rows, sort="area", descending=True)) == [
        "a/001",
        "b/002",
        "a/000",
        "b/003",
    ]
    # confidence: None 은 항상 끝
    assert ids(filter_rows(rows, sort="confidence")) == ["a/001", "b/003", "a/000", "b/002"]
    assert ids(filter_rows(rows, sort="confidence", descending=True)) == [
        "a/000",
        "b/003",
        "a/001",
        "b/002",
    ]
    assert ids(filter_rows(rows, sort="class", descending=True))[0].startswith("b/")
    with pytest.raises(BankSessionError):
        filter_rows(rows, sort="nope")


@pytest.fixture
def bank_root(tmp_path: Path) -> Path:
    d = fake_yolo_dataset(tmp_path / "ds")
    Y.import_yolo(d["images"], d["labels"], d["names"], tmp_path / "bank", mask_from="rect")
    return tmp_path / "bank"


def test_session_load_reload_and_summary(bank_root: Path, tmp_path: Path) -> None:
    s = BankSession()
    assert not s.loaded and s.summary_text() == "은행 없음" and s.classes() == []
    with pytest.raises(BankSessionError):
        s.load(tmp_path / "nope")
    with pytest.raises(BankSessionError):
        s.reload()
    b = s.load(bank_root)
    assert s.loaded and len(s.rows()) == len(b) == 5 and s.classes() == ["spot", "crack"]
    assert "소스 5" in s.summary_text() and "추정 4" in s.summary_text()
    assert s.source("spot/d0-01").cls == "spot"
    with pytest.raises(BankSessionError):
        s.source("spot/zzz")
    s.close()
    assert not s.loaded


def test_delete_removes_files_keeps_classes_and_logs(bank_root: Path) -> None:
    s = BankSession()
    s.load(bank_root)
    ids = [r.id for r in s.rows() if r.cls == "spot"]
    assert len(ids) == 2
    n = s.delete([*ids, "spot/ghost", "bad"])
    assert n == 2 and s.loaded
    assert [r.cls for r in s.rows()] == ["crack", "crack", "crack"]
    assert s.classes() == ["spot", "crack"]  # id 순서 유지
    for sid in ids:
        cls, name = sid.split("/")
        assert not list((bank_root / cls).glob(f"{name}.*"))
    meta = yaml.safe_load((bank_root / BANK_FILE).read_text(encoding="utf-8"))
    last = meta["imports"][-1]
    assert last["importer"] == "bank-tab" and last["action"] == "delete" and last["ids"][:2] == ids
    assert Bank.load(bank_root).classes == ["spot", "crack"]
    assert s.delete(["spot/ghost"]) == 0


def test_replace_mask_overwrites_same_id_and_clears_score(bank_root: Path) -> None:
    s = BankSession()
    s.load(bank_root)
    row = next(r for r in s.rows() if r.estimated)
    src = s.source(row.id)
    assert src.confidence is not None
    new = np.zeros(src.mask.shape, dtype=np.uint8)
    new[2:6, 2:9] = 200  # 이진화되어 저장
    s.replace_mask(row.id, new, tool="brush")
    after = s.source(row.id)
    assert after.mask_origin == "manual:brush" and after.confidence is None and after.flags == ()
    assert np.count_nonzero(after.mask) == 4 * 7 and set(np.unique(after.mask)) <= {0, 255}
    cls, name = row.id.split("/")
    meta = json.loads((bank_root / cls / f"{name}.json").read_text(encoding="utf-8"))
    assert meta["area_px"] == 28 and meta["confidence"] is None and meta["flags"] == []
    assert imgio.read_mask(bank_root / cls / f"{name}.mask.png").sum() // 255 == 28
    # 크롭은 그대로 · 이력
    assert np.array_equal(after.image, src.image)
    hist = yaml.safe_load((bank_root / BANK_FILE).read_text(encoding="utf-8"))["imports"][-1]
    assert hist["action"] == "replace_mask" and hist["ids"] == [row.id]
    # 오류: 크기 다름 · 빈 마스크 · 없는 id
    with pytest.raises(BankSessionError, match="크기"):
        s.replace_mask(row.id, np.zeros((3, 3), dtype=np.uint8))
    with pytest.raises(BankSessionError, match="비어"):
        s.replace_mask(row.id, np.zeros(src.mask.shape, dtype=np.uint8))
    with pytest.raises(BankSessionError):
        s.replace_mask("spot/zzz", new)


def test_summary_text_names_directional_classes(tmp_path: Path) -> None:
    """0.7.2+ — 조명 의존 클래스(lightR ≥ 0.5, n ≥ 3)는 요약 한 줄에 dent-graft 안내와 함께."""
    from anograft.bank.importers.common import BankWriter, ImportOptions, ImportRecord
    from tests.test_lighting_warning import _dent, _flat

    w = BankWriter(tmp_path / "b", name="b")
    w.ensure_classes(["pit", "stain"])
    for s in [_dent("down", k=i) for i in range(3)] + [_flat(k=i) for i in range(3)]:
        rec = ImportRecord(
            image=s.image, gray=False, mask=s.mask, cls=s.cls, origin="t", id_hint=s.id[-3:]
        )
        w.add(rec, ImportOptions(margin=8))
    w.finish({"importer": "test"})
    s = BankSession()
    assert s.directional_classes() == []
    s.load(tmp_path / "b")
    assert s.directional_classes() == [("pit", 1.0)]
    assert "조명 의존(lightR) pit 1.00 → dent-graft" in s.summary_text()

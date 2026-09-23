"""보관함·결함 표시 API (U4) — Qt 탭과 **같은 순수 로직**을 부르는지, 쓰기가 규칙을 지키는지.

여기가 지키는 것: 필터·정렬이 `bank.browse` 와 일치 · 삭제는 `BankWriter` 를 지나 **클래스 목록을 유지** ·
브러시가 마스크를 만들고 되돌릴 수 있다 · 오버레이는 **투명 배경 + 반투명 빨강** ·
**평가셋(holdout)은 라벨 탭 저장으로도 못 들어간다**(T4 가 `BankWriter` 한 지점을 막았으므로).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anograft.bank import holdout as ho
from anograft.bank.bank import Bank
from anograft.bank.browse import BankSession, filter_rows
from anograft.bank.importers import yolo as Y
from anograft.io import imgio
from anograft.web import bank_api, label_api
from anograft.web.api import WRITE_ROUTES, Request, handle
from tests.fixtures import fake_yolo_dataset


@pytest.fixture(autouse=True)
def _clean():
    bank_api.reset()
    label_api.reset()
    yield
    bank_api.reset()
    label_api.reset()


@pytest.fixture
def bank_root(tmp_path: Path) -> Path:
    d = fake_yolo_dataset(tmp_path / "ds")
    root = tmp_path / "bank"
    Y.import_yolo(d["images"], d["labels"], d["names"], root, mask_from="rect")
    return root


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    d = fake_yolo_dataset(tmp_path / "ds2")
    return sorted(Path(d["images"]).glob("*.png"))[0]


def _open_bank(root: Path) -> dict:
    res = bank_api._open(Request("/api/bank/open", "POST", json={"root": str(root)}))
    assert res.status == 200, res.payload
    return res.payload


# ---------------------------------------------------------------- 보관함


def test_bank_needs_open_first():
    res = handle(Request("/api/bank/sources"))
    assert res.status == 500 and "먼저 결함 보관함을 여세요" in res.payload["error"]


def test_bank_list_matches_session(bank_root: Path):
    _open_bank(bank_root)
    session = BankSession()
    session.load(bank_root)

    res = bank_api._sources(Request("/api/bank/sources", query={"sort": "area"}))
    assert res.status == 200
    expected = [r.id for r in filter_rows(session.rows(), sort="area")]
    assert [s["id"] for s in res.payload["sources"]] == expected


def test_bank_filter_by_class(bank_root: Path):
    state = _open_bank(bank_root)
    cls = state["classes"][0]
    res = bank_api._sources(Request("/api/bank/sources", query={"class": cls}))
    assert res.payload["sources"]
    assert {s["cls"] for s in res.payload["sources"]} == {cls}


def test_bank_rejects_unknown_sort(bank_root: Path):
    _open_bank(bank_root)
    assert bank_api._sources(Request("/api/bank/sources", query={"sort": "없음"})).status == 400


def test_bank_tile_is_png(bank_root: Path):
    state = _open_bank(bank_root)
    first = bank_api._sources(Request("/api/bank/sources")).payload["sources"][0]
    res = bank_api._image(Request("/api/bank/image", query={"id": first["id"]}))
    assert res.status == 200 and res.body[:8] == b"\x89PNG\r\n\x1a\n"
    assert state["total"] >= 1


def test_bank_delete_keeps_class_list(bank_root: Path):
    """클래스는 삭제해도 유지된다 — id 순서 = class id = 출력 `data.yaml`."""
    state = _open_bank(bank_root)
    classes_before = state["classes"]
    victim = bank_api._sources(Request("/api/bank/sources")).payload["sources"][0]["id"]

    res = bank_api._delete(Request("/api/bank/delete", "POST", json={"ids": [victim]}))
    assert res.status == 200 and res.payload["removed"] == 1
    assert res.payload["state"]["classes"] == classes_before

    assert victim not in [s.id for s in Bank.load(bank_root).sources()]


def test_bank_delete_needs_ids(bank_root: Path):
    _open_bank(bank_root)
    assert bank_api._delete(Request("/api/bank/delete", "POST", json={"ids": []})).status == 400
    assert bank_api._delete(Request("/api/bank/delete", "POST", json={"ids": "x"})).status == 400


def test_bank_state_reports_holdout_leak(bank_root: Path):
    """T4 를 화면에 잇는다 — 평가셋이 섞였으면 보관함을 여는 순간 보인다."""
    stem = ho.normalize_stem(Bank.load(bank_root).sources()[0].origin)
    (bank_root / ho.HOLDOUT_FILE).write_text(stem + "\n", encoding="utf-8")

    state = _open_bank(bank_root)
    assert state["holdout"]["listed"] == 1
    assert state["holdout"]["leaked"], "누수를 보고하지 않았다"


# ---------------------------------------------------------------- 결함 표시


def _open_label(photo: Path) -> dict:
    res = label_api._open(Request("/api/label/open", "POST", json={"path": str(photo)}))
    assert res.status == 200, res.payload
    return res.payload


def test_label_needs_open_first():
    res = handle(Request("/api/label/state"))
    assert res.status == 200 and res.payload == {"open": False}


def test_label_open_reports_size(photo: Path):
    state = _open_label(photo)
    assert state["open"] and state["width"] > 0 and state["height"] > 0
    assert state["stats"]["areaPx"] == 0  # 아직 칠한 것 없음


def test_stroke_paints_and_undo_restores(photo: Path):
    _open_label(photo)
    res = label_api._stroke(
        Request("/api/label/stroke", "POST", json={"points": [[20, 20], [60, 40]], "radius": 9})
    )
    assert res.status == 200
    painted = res.payload["stats"]["areaPx"]
    assert painted > 0
    assert res.payload["canUndo"] is True

    back = label_api._undo(Request("/api/label/undo", "POST", json={}))
    assert back.status == 200
    assert back.payload["stats"]["areaPx"] == 0
    assert back.payload["moved"] is True


def test_eraser_removes(photo: Path):
    _open_label(photo)
    label_api._stroke(
        Request("/api/label/stroke", "POST", json={"points": [[30, 30], [80, 30]], "radius": 12})
    )
    res = label_api._stroke(
        Request(
            "/api/label/stroke",
            "POST",
            json={"points": [[30, 30], [80, 30]], "radius": 14, "erase": True},
        )
    )
    assert res.payload["stats"]["areaPx"] == 0


def test_polygon_needs_points(photo: Path):
    _open_label(photo)
    assert label_api._polygon(Request("/api/label/polygon", "POST", json={})).status == 400
    bad = label_api._stroke(
        Request("/api/label/stroke", "POST", json={"points": [[1]], "radius": 3})
    )
    assert bad.status == 400


def test_overlay_is_transparent_outside_mask(photo: Path, tmp_path: Path):
    """CSS 블렌드로 0/255 마스크를 물들이면 사진 전체가 물든다 — 알파는 서버가 만든다."""
    _open_label(photo)
    label_api._stroke(
        Request("/api/label/stroke", "POST", json={"points": [[40, 40], [90, 60]], "radius": 10})
    )
    res = label_api._image(Request("/api/label/image", query={"kind": "overlay"}))
    assert res.status == 200 and res.content_type == "image/png"

    out = tmp_path / "overlay.png"
    out.write_bytes(res.body)
    import cv2

    rgba = cv2.imdecode(np.fromfile(str(out), np.uint8), cv2.IMREAD_UNCHANGED)
    assert rgba.shape[2] == 4, "알파 채널이 없다"
    alpha = rgba[:, :, 3]
    assert alpha.max() > 0 and alpha.min() == 0  # 칠한 곳만 불투명
    painted = rgba[alpha > 0]
    assert (painted[:, 2] > painted[:, 0]).all()  # 빨강(BGR 에서 R 이 가장 크다)


def test_clear_empties_mask(photo: Path):
    _open_label(photo)
    label_api._stroke(
        Request("/api/label/stroke", "POST", json={"points": [[10, 10], [40, 40]], "radius": 8})
    )
    res = label_api._clear(Request("/api/label/clear", "POST", json={}))
    assert res.payload["stats"]["areaPx"] == 0


def test_save_puts_a_source_in_the_bank(photo: Path, tmp_path: Path):
    _open_label(photo)
    label_api._stroke(
        Request("/api/label/stroke", "POST", json={"points": [[40, 40], [80, 60]], "radius": 12})
    )
    bank = tmp_path / "newbank"
    res = label_api._save(
        Request("/api/label/save", "POST", json={"bank": str(bank), "cls": "scratch"})
    )
    assert res.status == 200 and res.payload["added"]
    loaded = Bank.load(bank)
    assert len(loaded) == 1 and loaded.sources()[0].cls == "scratch"


def test_save_needs_class_and_mask(photo: Path, tmp_path: Path):
    _open_label(photo)
    empty = label_api._save(
        Request("/api/label/save", "POST", json={"bank": str(tmp_path / "b"), "cls": "x"})
    )
    assert empty.status == 400 and "마스크" in empty.payload["error"]

    label_api._stroke(
        Request("/api/label/stroke", "POST", json={"points": [[10, 10], [30, 30]], "radius": 8})
    )
    no_cls = label_api._save(
        Request("/api/label/save", "POST", json={"bank": str(tmp_path / "b"), "cls": " "})
    )
    assert no_cls.status == 400


def test_holdout_also_blocks_label_save(photo: Path, tmp_path: Path):
    """T4 가 `BankWriter` 한 지점을 막았으므로 **라벨 탭 저장도** 막힌다."""
    _open_label(photo)
    label_api._stroke(
        Request("/api/label/stroke", "POST", json={"points": [[40, 40], [80, 60]], "radius": 12})
    )
    bank = tmp_path / "guarded"
    bank.mkdir()
    (bank / ho.HOLDOUT_FILE).write_text(photo.stem + "\n", encoding="utf-8")

    res = label_api._save(
        Request("/api/label/save", "POST", json={"bank": str(bank), "cls": "scratch"})
    )
    # 거부되어 아무것도 안 들어간다(경고는 남는다)
    assert res.status == 200 and not res.payload["added"]
    assert any("평가셋" in w for w in res.payload["warnings"])


def test_label_mask_roundtrip_matches_session(photo: Path, tmp_path: Path):
    """서버가 그린 마스크가 곧 저장되는 마스크다 — 화면이 따로 계산하지 않는다."""
    _open_label(photo)
    label_api._stroke(
        Request("/api/label/stroke", "POST", json={"points": [[50, 50], [70, 70]], "radius": 10})
    )
    res = label_api._image(Request("/api/label/image", query={"kind": "mask"}))
    out = tmp_path / "m.png"
    out.write_bytes(res.body)
    mask = imgio.read_mask(out)
    assert (
        int(np.count_nonzero(mask))
        == label_api._state(Request("/api/label/state")).payload["stats"]["areaPx"]
    )


# ---------------------------------------------------------------- 쓰기 등록


def test_all_mutating_routes_are_write_registered():
    """쓰기가 GET 으로 열리면 CSRF 방어가 통째로 무의미해진다 — 등록을 테스트로 고정한다."""
    handle(Request("/api/health"))  # 라우트 등록을 강제
    for path in (
        "/api/bank/open",
        "/api/bank/delete",
        "/api/label/open",
        "/api/label/stroke",
        "/api/label/polygon",
        "/api/label/auto",
        "/api/label/undo",
        "/api/label/clear",
        "/api/label/save",
    ):
        assert path in WRITE_ROUTES, path
        assert handle(Request(path, "GET")).status == 405

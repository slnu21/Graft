"""파이프라인 카드 편집 API (U5b) — 폼이 **스키마에서 나오고**, 편집이 Qt 와 같은 규칙으로 막히는지.

여기가 지키는 것: 카드·필드 스펙은 `studio.cards`/`params` 가 낸 그대로(웹이 폼을 따로 만들지 않는다) ·
라벨·툴팁은 **서버가 실어 준다**(한국어 문안의 한 원천은 `core/help.py`) · 잘못된 값은 **이전 레시피를
유지**하고 어느 필드인지 알린다 · method 교체는 그 블록을 새 기본값으로 갈아끼운다 · 되돌리기는 프리셋 값 ·
`per_class` 는 꺼진 행을 빼고 `scale` 을 보존한다 · 쓰기는 전부 `write=True`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anograft.bank.importers import yolo as Y
from anograft.studio import cards
from anograft.studio.session import StudioSession, default_recipe
from anograft.web import studio_api
from anograft.web.api import WRITE_ROUTES, Request, handle
from tests.fixtures import fake_yolo_dataset

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def _clean():
    studio_api.reset()
    yield
    studio_api.reset()


@pytest.fixture
def opened(tmp_path: Path) -> dict:
    d = fake_yolo_dataset(tmp_path / "ds")
    normals = tmp_path / "normals.txt"
    Y.import_yolo(
        d["images"],
        d["labels"],
        d["names"],
        tmp_path / "bank",
        mask_from="rect",
        list_normals=normals,
    )
    ses = StudioSession(
        default_recipe(bank=(tmp_path / "bank").as_posix(), targets=normals.as_posix())
    )
    ses.set_stage_field("placement", "margin_px", 4)
    ses.set_stage_field("roi", "erode_px", 2)
    ses.set_field(("pipeline", "source", "min_sources_warn"), 1)
    path = ses.save(tmp_path / "studio.yaml")
    res = handle(Request("/api/studio/open", "POST", json={"recipe": str(path)}))
    assert res.status == 200, res.payload
    return res.payload


def _cards() -> dict:
    res = handle(Request("/api/studio/cards"))
    assert res.status == 200, res.payload
    return res.payload


def _card(stage: str) -> dict:
    return next(c for c in _cards()["cards"] if c["stage"] == stage)


def _post(path: str, **body) -> tuple[int, dict]:
    res = handle(Request(path, "POST", json=body))
    return res.status, res.payload


# ---------------------------------------------------------------- 카드 목록


def test_cards_need_open_first():
    res = handle(Request("/api/studio/cards"))
    assert res.status == 500 and "먼저 레시피" in res.payload["error"]


def test_cards_are_the_schema_with_korean_labels(opened: dict):
    payload = _cards()
    stages = [c["stage"] for c in payload["cards"]]
    assert stages == [m.stage for m in cards.stage_cards(studio_api._SESSION.recipe)]
    blend = _card("blend")
    assert blend["label"] == "붙이기" and blend["method"] == "poisson"
    feather = next(f for f in blend["fields"] if f["name"] == "feather_px")
    # 라벨·툴팁을 서버가 만든다 — 프론트에 용어 사전 사본을 두지 않는다
    assert feather["title"] and feather["tooltip"].splitlines()[0].endswith("feather_px")
    assert feather["kind"] == "int" and feather["modified"] is False
    assert any(not m["usable"] and m["reason"] for m in blend["methods"]) or all(
        m["usable"] for m in blend["methods"]
    )
    assert payload["longSides"][0]["px"] == 1024


def test_optional_field_is_marked_with_its_on_default(opened: dict):
    degrade = _card("degrade")
    jpeg = next(f for f in degrade["fields"] if f["name"] == "jpeg_quality")
    assert jpeg["optional"] and jpeg["onDefault"] == [60, 95]


# ---------------------------------------------------------------- 편집


def test_field_edit_revalidates_and_marks_modified(opened: dict):
    status, payload = _post("/api/studio/field", stage="blend", name="feather_px", value=9)
    assert status == 200, payload
    blend = next(c for c in payload["cards"]["cards"] if c["stage"] == "blend")
    feather = next(f for f in blend["fields"] if f["name"] == "feather_px")
    assert feather["value"] == 9 and feather["modified"] is True and blend["modified"] == 1
    assert payload["state"]["recipe"]["preset"] == "poisson-graft"


def test_bad_value_keeps_the_recipe_and_says_which_field(opened: dict):
    status, payload = _post("/api/studio/field", stage="blend", name="feather_px", value=-5)
    assert status == 400 and payload["name"] == "feather_px" and payload["stage"] == "blend"
    feather = next(f for f in _card("blend")["fields"] if f["name"] == "feather_px")
    assert feather["value"] != -5, "실패하면 이전 레시피를 유지한다"


def test_unknown_field_or_stage_is_rejected(opened: dict):
    assert _post("/api/studio/field", stage="blend", name="levels", value=3)[0] == 400
    assert _post("/api/studio/field", stage="nope", name="x", value=1)[0] == 400


def test_method_change_replaces_the_block_with_its_defaults(opened: dict):
    status, payload = _post("/api/studio/field", stage="blend", name="feather_px", value=9)
    assert status == 200
    status, payload = _post("/api/studio/method", stage="blend", method="alpha")
    assert status == 200, payload
    blend = next(c for c in payload["cards"]["cards"] if c["stage"] == "blend")
    assert blend["method"] == "alpha"
    feather = next(f for f in blend["fields"] if f["name"] == "feather_px")
    assert feather["value"] == 3, "새 method 의 기본값만 남는다"
    # 지난 그림은 버린다(옛 그림을 새 설정의 결과로 보여 주면 거짓말)
    assert handle(Request("/api/studio/image", "GET", query={"kind": "synth"})).status == 404


def test_roi_is_edited_as_its_own_stage(opened: dict):
    status, payload = _post("/api/studio/method", stage="roi", method="none")
    assert status == 200, payload
    roi = next(c for c in payload["cards"]["cards"] if c["stage"] == "roi")
    assert roi["method"] == "none" and roi["no"] == 0


def test_reset_puts_the_preset_value_back(opened: dict):
    _post("/api/studio/field", stage="blend", name="feather_px", value=9)
    status, payload = _post("/api/studio/reset", stage="blend")
    assert status == 200, payload
    blend = next(c for c in payload["cards"]["cards"] if c["stage"] == "blend")
    assert blend["modified"] == 0
    # 한 필드만 되돌리기도 같은 규칙. 전체 `modified` 는 다른 카드(레시피가 프리셋에서 바꾼 것)까지 센다
    total = payload["cards"]["modified"]
    _post("/api/studio/field", stage="blend", name="feather_px", value=9)
    status, payload = _post("/api/studio/reset", stage="blend", name="feather_px")
    assert status == 200 and payload["cards"]["modified"] == total


# ---------------------------------------------------------------- per_class 표


def test_per_class_rows_come_from_the_bank_classes(opened: dict):
    per = _cards()["perClass"]
    assert per["classes"] and [r["cls"] for r in per["rows"]] == per["classes"]
    assert all(r["on"] is False for r in per["rows"])
    assert per["flipChoices"][0]["value"] == "", "첫 선택지는 '기본'(전체 설정)"


def test_per_class_write_drops_off_rows(opened: dict):
    per = _cards()["perClass"]
    rows = [dict(r) for r in per["rows"]]
    rows[0].update(on=True, rotate=[-15, 15], flip="none")
    status, payload = _post("/api/studio/per-class", rows=rows)
    assert status == 200, payload
    after = payload["cards"]["perClass"]
    on = [r for r in after["rows"] if r["on"]]
    assert [r["cls"] for r in on] == [rows[0]["cls"]] and on[0]["flip"] == "none"
    assert after["text"].startswith("클래스별 예외")
    # `per_class` 는 dict 라 카드 폼에 행이 없다(Qt 도 전용 표다) — 바뀜 수에는 안 들어간다
    geo = next(c for c in payload["cards"]["cards"] if c["stage"] == "geometry")
    assert not any(f["name"] == "per_class" for f in geo["fields"])
    assert _post("/api/studio/per-class", rows="nope")[0] == 400


def test_per_class_rejects_an_impossible_range(opened: dict):
    per = _cards()["perClass"]
    rows = [dict(r) for r in per["rows"]]
    rows[0].update(on=True, rotate=[400, 500])
    status, payload = _post("/api/studio/per-class", rows=rows)
    assert status == 400 and payload["stage"] == "geometry"


# ---------------------------------------------------------------- 그림


def test_stage_thumbnail_needs_a_preview_first(opened: dict):
    res = handle(Request("/api/studio/stage-image", "GET", query={"stage": "blend"}))
    assert res.status == 404
    assert handle(Request("/api/studio/preview", "POST", json={})).status == 200
    for stage in ("source", "geometry", "placement", "roi", "blend", "gtmask"):
        res = handle(Request("/api/studio/stage-image", "GET", query={"stage": stage}))
        assert res.status == 200 and res.body.startswith(PNG_MAGIC), stage
    assert handle(Request("/api/studio/stage-image", "GET", query={"stage": "nope"})).status == 404


def test_variant_image_does_not_disturb_the_canvas(opened: dict):
    assert handle(Request("/api/studio/preview", "POST", json={})).status == 200
    before = handle(Request("/api/studio/image", "GET", query={"kind": "synth"})).body
    res = handle(Request("/api/studio/variant-image", "GET", query={"k": "2", "tile": "160"}))
    assert res.status == 200 and res.body.startswith(PNG_MAGIC)
    after = handle(Request("/api/studio/image", "GET", query={"kind": "synth"})).body
    assert after == before, "변형 그리드는 캔버스의 미리보기를 갈아치우지 않는다"
    assert res.body != before, "다른 시드 변형이면 다른 그림"


# ---------------------------------------------------------------- 쓰기 등록


def test_all_mutating_routes_are_write_registered():
    handle(Request("/api/health"))
    for path in (
        "/api/studio/field",
        "/api/studio/method",
        "/api/studio/reset",
        "/api/studio/per-class",
    ):
        assert path in WRITE_ROUTES, path
        assert handle(Request(path, "GET")).status == 405
    for path in ("/api/studio/cards", "/api/studio/stage-image", "/api/studio/variant-image"):
        assert path not in WRITE_ROUTES
        assert handle(Request(path, "POST")).status == 405

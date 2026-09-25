"""화면 사이 손잡이 (U7) — 인계는 **서버가 상태를 든다**.

v0.10.0 을 직접 써 보고 나온 지적("매 화면마다 뭘 열어야 하는거야?")의 답이라, 여기서 지키는 것은
"버튼이 있다"가 아니라 **다음 화면이 열어 보기만 해도 이미 그것을 보고 있다**는 쪽이다:

* ③ → ④ 미리보기 레시피를 저장하고 일괄 생성 세션이 그대로 받는다(CLI 한 줄이 거짓말이 되지 않게).
* ② → ① 결함 표시가 조각을 더하면 보관함 세션이 새로 읽는다 — 단, **다른 보관함을 보고 있으면 건드리지 않는다**.
* ① → ② 조각을 열어 다듬고 저장하면 ``BankWriter.replace_mask`` 한 지점을 지나 같은 id 를 덮어쓴다.
* 열기 칸 최근 경로는 **연 행위가 곧 기록**이다(따로 "기억해 줘" 쓰기 API 가 없다).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anograft import recent
from anograft.bank.browse import BankSession
from anograft.bank.importers import yolo as Y
from anograft.core import recipe as R
from anograft.web import bank_api, batch_api, label_api, review_api, studio_api
from anograft.web.api import WRITE_ROUTES, Request, handle
from tests.fixtures import fake_yolo_dataset


@pytest.fixture(autouse=True)
def _clean():
    for mod in (bank_api, label_api, studio_api, batch_api, review_api):
        mod.reset()
    yield
    for mod in (bank_api, label_api, studio_api, batch_api, review_api):
        mod.reset()


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


def _post(route: str, **json) -> tuple[int, dict]:
    """`path=` 를 본문 키로 쓰는 라우트가 있어 첫 인자 이름은 `route` 다."""
    res = handle(Request(route, "POST", json=json))
    return res.status, (res.payload or {})


def _open_bank(root: Path) -> dict:
    status, payload = _post("/api/bank/open", root=str(root))
    assert status == 200, payload
    return payload


def _open_label(path: Path) -> dict:
    status, payload = _post("/api/label/open", path=str(path))
    assert status == 200, payload
    return payload


def _paint(radius: float = 6.0) -> dict:
    """가운데에 획 하나 — 마스크가 비어 있으면 저장이 거부되므로 어느 시험이든 먼저 칠한다."""
    status, payload = _post(
        "/api/label/stroke", points=[[20, 20], [30, 30], [40, 32]], radius=radius
    )
    assert status == 200, payload
    return payload


# ---------------------------------------------------------------- ③ → ④ 레시피 인계


@pytest.fixture
def studio_recipe(tmp_path: Path, bank_root: Path) -> Path:
    """은행 + 대상 몇 장으로 최소 레시피 하나 — 미리보기 세션이 열 수 있는 것."""
    d = fake_yolo_dataset(tmp_path / "targets")
    rec = R.Recipe.from_dict(
        {
            "name": "handoff",
            "seed": 7,
            "inputs": {"bank": str(bank_root), "targets": str(d["images"])},
            "pipeline": {"preset": "hard-paste"},
            "output": {"root": str(tmp_path / "out"), "count": 2},
        }
    )
    path = tmp_path / "recipes" / "handoff.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rec.to_yaml(), encoding="utf-8", newline="\n")
    return path


def test_to_batch_saves_and_hands_over(studio_recipe: Path, tmp_path: Path) -> None:
    assert _post("/api/studio/open", recipe=str(studio_recipe))[0] == 200
    assert _post("/api/studio/seed", seed=123)[0] == 200

    out = tmp_path / "recipes" / "sent.yaml"
    status, payload = _post("/api/studio/to-batch", path=str(out))
    assert status == 200, payload

    # 1) 저장됐고 2) 저장된 파일이 방금 바꾼 시드를 담고 있고 3) 일괄 생성이 그걸 들고 있다
    assert out.exists()
    assert R.Recipe.load(out).seed == 123
    assert payload["batch"]["open"] is True
    assert payload["batch"]["recipePath"] == out.as_posix()
    assert payload["batch"]["settings"]["seed"] == 123

    # 화면을 새로 열어도(= state 만 물어도) 이미 그것을 보고 있다 — 인계의 핵심
    state = handle(Request("/api/batch/state")).payload
    assert state["recipePath"] == out.as_posix()
    assert state["recipe"]["name"] == "handoff"


def test_to_batch_without_path_uses_the_opened_recipe(studio_recipe: Path) -> None:
    assert _post("/api/studio/open", recipe=str(studio_recipe))[0] == 200
    status, payload = _post("/api/studio/to-batch")
    assert status == 200, payload
    assert payload["path"] == studio_recipe.as_posix()
    # CLI 한 줄이 진짜 있는 파일을 가리킨다 — 이걸 위해 인계가 저장을 한다
    assert studio_recipe.as_posix() in payload["runCommand"]


def test_to_batch_keeps_worker_count(studio_recipe: Path, tmp_path: Path) -> None:
    """레시피 밖 설정(워커 수)은 사람이 맞춰 둔 대로 남는다 — Qt ``set_recipe`` 와 같다."""
    assert _post("/api/batch/open", recipe=str(studio_recipe))[0] == 200
    assert _post("/api/batch/settings", settings={"workers": 3})[0] == 200
    assert _post("/api/studio/open", recipe=str(studio_recipe))[0] == 200
    status, payload = _post("/api/studio/to-batch", path=str(tmp_path / "r2.yaml"))
    assert status == 200, payload
    assert payload["batch"]["settings"]["workers"] == 3


def test_to_batch_needs_a_path_when_nothing_was_opened_from_file(
    bank_root: Path, tmp_path: Path
) -> None:
    d = fake_yolo_dataset(tmp_path / "t2")
    assert _post("/api/studio/open", bank=str(bank_root), targets=str(d["images"]))[0] == 200
    status, payload = _post("/api/studio/to-batch")
    assert status == 400 and "경로가 필요" in payload["error"]


def test_to_batch_refused_while_running(studio_recipe: Path) -> None:
    """돌고 있는 중에 설정을 갈아 끼우면 진행 중인 실행을 설명하는 화면이 거짓말을 한다."""
    assert _post("/api/studio/open", recipe=str(studio_recipe))[0] == 200

    class _Busy:
        finished = False

    batch_api._RUN = _Busy()  # type: ignore[assignment]
    try:
        status, payload = _post("/api/studio/to-batch", path=str(studio_recipe))
        assert status == 409 and "돌고 있습니다" in payload["error"]
    finally:
        batch_api._RUN = None


# ---------------------------------------------------------------- ② → ① 저장 반영


def test_label_save_refreshes_an_open_bank(bank_root: Path, photo: Path) -> None:
    _open_bank(bank_root)
    n_before = handle(Request("/api/bank/state")).payload["total"]

    _open_label(photo)
    _paint()
    status, payload = _post("/api/label/save", bank=str(bank_root), cls="scratch")
    assert status == 200, payload
    assert payload["bankShown"] is True

    # 보관함 화면은 **다시 열지 않아도** 새 조각을 본다
    after = handle(Request("/api/bank/state")).payload
    assert after["total"] == n_before + 1
    ids = [s["id"] for s in handle(Request("/api/bank/sources")).payload["sources"]]
    assert any(i.startswith("scratch/") for i in ids)


def test_label_save_opens_the_bank_when_none_is_open(bank_root: Path, photo: Path) -> None:
    assert handle(Request("/api/bank/state")).payload == {"open": False}
    _open_label(photo)
    _paint()
    status, payload = _post("/api/label/save", bank=str(bank_root), cls="scratch")
    assert status == 200, payload
    assert payload["bankShown"] is True
    state = handle(Request("/api/bank/state")).payload
    assert state["open"] is True and state["root"] == bank_root.as_posix()


def test_label_save_does_not_steal_a_different_bank(
    bank_root: Path, photo: Path, tmp_path: Path
) -> None:
    """다른 보관함을 보고 있으면 말없이 빼앗지 않는다 — 화면의 버튼이 사람의 뜻으로 바꾼다."""
    d = fake_yolo_dataset(tmp_path / "ds3")
    other = tmp_path / "other-bank"
    Y.import_yolo(d["images"], d["labels"], d["names"], other, mask_from="rect")
    _open_bank(other)

    _open_label(photo)
    _paint()
    status, payload = _post("/api/label/save", bank=str(bank_root), cls="scratch")
    assert status == 200, payload
    assert payload["bankShown"] is False
    assert handle(Request("/api/bank/state")).payload["root"] == other.as_posix()


def test_holdout_still_blocks_the_save(bank_root: Path, photo: Path) -> None:
    """T4 는 인계가 생겨도 그대로다 — 평가셋은 ``BankWriter.add`` 한 지점에서 거부된다."""
    (bank_root / "holdout.txt").write_text(photo.stem + "\n", encoding="utf-8")
    _open_bank(bank_root)
    n_before = handle(Request("/api/bank/state")).payload["total"]
    _open_label(photo)
    _paint()
    status, payload = _post("/api/label/save", bank=str(bank_root), cls="scratch")
    assert status == 200, payload
    assert payload["added"] == []
    assert any("평가셋" in w for w in payload["warnings"]), payload["warnings"]
    assert handle(Request("/api/bank/state")).payload["total"] == n_before


# ---------------------------------------------------------------- ① → ② 조각 다듬기


def _source(root: Path, source_id: str):
    """디스크에서 다시 읽은 조각 — 세션 캐시가 아니라 **파일이 정말 바뀌었나**를 본다."""
    session = BankSession()
    session.load(root)
    return session.source(source_id)


def _first_source_id(root: Path) -> str:
    _open_bank(root)
    return handle(Request("/api/bank/sources")).payload["sources"][0]["id"]


def test_edit_source_opens_the_crop_and_its_mask(bank_root: Path) -> None:
    sid = _first_source_id(bank_root)
    src = _source(bank_root, sid)

    status, payload = _post("/api/label/edit-source", root=str(bank_root), id=sid)
    assert status == 200, payload
    assert payload["editTarget"] == {"root": bank_root.as_posix(), "id": sid}
    assert (payload["height"], payload["width"]) == src.image.shape[:2]
    # 빈 캔버스가 아니라 **지금 마스크**를 열었다
    assert payload["stats"]["areaPx"] == int(np.count_nonzero(src.mask))
    assert payload["cls"] == src.cls


def test_edit_source_opens_the_bank_if_needed(bank_root: Path) -> None:
    sid = _first_source_id(bank_root)
    bank_api.reset()
    assert handle(Request("/api/bank/state")).payload == {"open": False}
    status, payload = _post("/api/label/edit-source", root=str(bank_root), id=sid)
    assert status == 200, payload
    assert handle(Request("/api/bank/state")).payload["root"] == bank_root.as_posix()


def test_update_source_overwrites_the_same_id(bank_root: Path) -> None:
    sid = _first_source_id(bank_root)
    before = _source(bank_root, sid)
    assert _post("/api/label/edit-source", root=str(bank_root), id=sid)[0] == 200
    _paint(radius=3)

    status, payload = _post("/api/label/update-source")
    assert status == 200, payload
    assert payload["id"] == sid and payload["tool"] == "brush"

    after = _source(bank_root, sid)
    assert after.mask_origin == "manual:brush"  # 추정 출처가 사람 손으로 바뀐다
    assert after.confidence is None  # 사람이 손봤으므로 추정 점수는 지워진다
    assert not np.array_equal(after.mask, before.mask)
    assert after.image.shape == before.image.shape  # 크롭은 그대로 — 마스크만 갈린다


def test_update_source_does_not_add_a_new_one(bank_root: Path) -> None:
    sid = _first_source_id(bank_root)
    n_before = handle(Request("/api/bank/state")).payload["total"]
    assert _post("/api/label/edit-source", root=str(bank_root), id=sid)[0] == 200
    _paint(radius=3)
    assert _post("/api/label/update-source")[0] == 200
    assert handle(Request("/api/bank/state")).payload["total"] == n_before


def test_update_source_rejects_an_empty_mask(bank_root: Path) -> None:
    """조각을 지우려면 보관함에서 삭제해야 한다 — 빈 마스크로 덮어쓰지 않는다."""
    sid = _first_source_id(bank_root)
    assert _post("/api/label/edit-source", root=str(bank_root), id=sid)[0] == 200
    assert _post("/api/label/clear")[0] == 200
    status, payload = _post("/api/label/update-source")
    assert status == 400 and "비어" in payload["error"]


def test_update_source_needs_edit_mode(photo: Path) -> None:
    _open_label(photo)
    _paint()
    status, payload = _post("/api/label/update-source")
    assert status == 400 and "다듬는 중이 아닙니다" in payload["error"]


def test_opening_a_photo_ends_edit_mode(bank_root: Path, photo: Path) -> None:
    sid = _first_source_id(bank_root)
    assert _post("/api/label/edit-source", root=str(bank_root), id=sid)[0] == 200
    assert _open_label(photo)["editTarget"] is None
    assert handle(Request("/api/label/state")).payload["editTarget"] is None


def test_clear_leaves_no_web_only_tool_name(bank_root: Path) -> None:
    """전부 지우기는 Qt 와 같은 메서드를 쓴다 — ``tools_used`` 에 웹에만 있는 이름이 남으면 갈린다."""
    sid = _first_source_id(bank_root)
    assert _post("/api/label/edit-source", root=str(bank_root), id=sid)[0] == 200
    assert _post("/api/label/clear")[0] == 200
    assert label_api._SESSION is not None
    assert "clear" not in label_api._SESSION.tools_used
    assert label_api._SESSION.edit_tool() == "brush"


# ---------------------------------------------------------------- 최근 경로


def test_opening_remembers_the_path(bank_root: Path, photo: Path) -> None:
    _open_bank(bank_root)
    _open_label(photo)
    payload = handle(Request("/api/recent")).payload
    assert payload["recent"]["bank"] == [str(bank_root)]
    assert payload["recent"]["image"] == [str(photo)]


def test_saving_a_recipe_shows_up_in_the_batch_list(studio_recipe: Path, tmp_path: Path) -> None:
    """③ 에서 저장한 레시피가 ④ 의 목록에 그대로 뜬다 — 종류를 화면이 아니라 **칸의 뜻**으로 나눈 이유."""
    assert _post("/api/studio/open", recipe=str(studio_recipe))[0] == 200
    out = tmp_path / "saved.yaml"
    assert _post("/api/studio/save", path=str(out))[0] == 200
    assert recent.recent("recipe")[0] == out.as_posix()


def test_recent_is_read_only() -> None:
    assert "/api/recent" not in WRITE_ROUTES
    assert handle(Request("/api/recent", "POST")).status == 405


# ---------------------------------------------------------------- 규약


def test_handoff_writes_are_post_only() -> None:
    for path in ("/api/studio/to-batch", "/api/label/edit-source", "/api/label/update-source"):
        assert path in WRITE_ROUTES, path
        # 링크를 잘못 눌러 데이터가 바뀌면 안 된다
        assert handle(Request(path, "GET")).status == 405, path

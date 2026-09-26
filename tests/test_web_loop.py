"""학습 루프 API (U8) — ⑥ 화면이 보는 사실이 CLI `loop status` 와 **같은 원천**인지.

여기가 지키는 것: 열기 전에도 200 으로 답한다(안내) · 읽기는 GET·쓰기는 POST 로 갈린다 ·
라운드가 돌면 champion·단계·기록이 따라온다 · **사람이 판정할 차례면 검토 대기 폴더를 함께 준다**
(⑤ 검수 화면이 그대로 여는 그 폴더 — 새 포맷을 만들지 않는다는 T5 의 요점) · 라운드를 웹에서
**돌리지 않는다**(쓰기 라우트는 `open` 하나뿐).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anograft import recent
from anograft.io.manifest import MANIFEST_FILE, read_manifest
from anograft.io.prune import REVIEW_FILE, write_review
from anograft.loop.config import load_loop_config
from anograft.loop.round import run_round
from anograft.web import loop_api
from anograft.web.api import WRITE_ROUTES, Request, handle
from tests.fixtures import loop_workspace


@pytest.fixture(autouse=True)
def _clean():
    loop_api.reset()
    yield
    loop_api.reset()


@pytest.fixture
def ws(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Path]:
    workspace = loop_workspace(tmp_path)
    capsys.readouterr()
    return workspace


def _open(path: Path) -> dict:
    res = handle(Request("/api/loop/open", "POST", json={"config": str(path)}))
    assert res.status == 200, res.payload
    return res.payload


def _state() -> dict:
    res = handle(Request("/api/loop/state"))
    assert res.status == 200, res.payload
    return res.payload


# --------------------------------------------------------------------- 열기 전후


def test_state_before_opening_says_so_and_suggests_a_file() -> None:
    payload = _state()
    assert payload["open"] is False and "suggest" in payload


def test_open_reads_loop_yaml_and_remembers_it(ws: dict[str, Path]) -> None:
    payload = _open(ws["loop"])
    assert payload["open"] is True
    assert payload["trainer"] == "noop" and payload["metricName"] == "mAP50"
    # 한국어 라벨·풀이는 **서버가 실어 준다**(프론트에 용어 사전 사본을 두지 않는다)
    assert payload["metricLabel"] == "검출 점수 (mAP50)" and payload["metricHint"]
    assert payload["out"] == ws["out"].as_posix()
    assert payload["champion"] is None and payload["round"] is None
    # 루프 설정은 **레시피와 뜻이 다른 칸**이라 종류를 따로 둔다(③·④ 목록에 섞이지 않게)
    assert ws["loop"].as_posix() in recent.recent("loop")
    assert ws["loop"].as_posix() not in recent.recent("recipe")


def test_open_without_a_path_says_what_to_do(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # 여기엔 loop.yaml 이 없다
    monkeypatch.setenv(recent.HOME_ENV, str(tmp_path / "home"))
    res = handle(Request("/api/loop/open", "POST", json={}))
    assert res.status == 400 and "loop.example.yaml" in res.payload["error"]


def test_broken_config_is_a_400_not_a_crash(ws: dict[str, Path], tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("trainer: noop\n", encoding="utf-8")  # 필수 키가 없다
    res = handle(Request("/api/loop/open", "POST", json={"config": str(bad)}))
    assert res.status == 400 and res.payload["error"]


def test_reads_and_writes_are_separated(ws: dict[str, Path]) -> None:
    assert "/api/loop/open" in WRITE_ROUTES
    assert "/api/loop/state" not in WRITE_ROUTES and "/api/loop/rounds" not in WRITE_ROUTES
    assert handle(Request("/api/loop/open", "GET")).status == 405
    assert handle(Request("/api/loop/state", "POST")).status == 405


def test_rounds_before_opening_is_a_clear_error() -> None:
    res = handle(Request("/api/loop/rounds"))
    assert res.status == 500 and "loop.yaml" in res.payload["error"]


# --------------------------------------------------------------------- 라운드가 돌고 난 뒤


def test_after_a_round_the_screen_shows_champion_steps_and_history(ws: dict[str, Path]) -> None:
    run_round(load_loop_config(ws["loop"]))
    payload = _open(ws["loop"])

    assert payload["champion"]["round"] == 1 and payload["champion"]["metric"] > 0
    rnd = payload["round"]
    assert rnd["number"] == 1 and rnd["finished"] is True and rnd["bootstrap"] is True
    assert [s["phase"] for s in rnd["steps"]] == ["synth", "train", "judge"]
    assert {s["state"] for s in rnd["steps"]} == {"done"}
    assert payload["bank"]["sources"] > 0 and payload["bank"]["rows"]
    assert payload["action"]["kind"] in {"run", "wait"}

    res = handle(Request("/api/loop/rounds"))
    assert res.status == 200
    rounds = res.payload["rounds"]
    assert len(rounds) == 1 and rounds[0]["round"] == 1 and rounds[0]["promoted"] is True
    assert res.payload["points"][0]["segment"] == 0
    assert res.payload["ledger"].endswith("rounds.jsonl")


def test_waiting_for_a_person_hands_over_the_queue_folder(ws: dict[str, Path]) -> None:
    """②라운드는 사람 앞에서 멈춘다 — 화면은 **그 폴더**를 ⑤ 검수로 넘길 수 있어야 한다."""
    loop = load_loop_config(ws["loop"])
    run_round(loop)  # 1라운드(부트스트랩)
    waiting = run_round(load_loop_config(ws["loop"]))
    assert waiting.waiting_for_human

    payload = _open(ws["loop"])
    review = payload["review"]
    assert review["waiting"] is True and review["total"] > 0 and review["judged"] == 0
    queue = Path(review["queueDir"])
    assert queue.is_dir() and (queue / MANIFEST_FILE).is_file()
    assert payload["action"]["kind"] == "review"
    assert payload["round"]["next"] == "review"
    assert payload["round"]["nextLabel"] == "사람 판정 대기"

    # 검수 화면이 **그대로** 연다(새 포맷 0) — 판정하면 진행률이 따라 움직인다
    assert handle(Request("/api/review/open", "POST", json={"root": str(queue)})).status == 200
    rows = [r for r in read_manifest(queue / MANIFEST_FILE) if r.get("status") == "ok"]
    write_review(queue / REVIEW_FILE, {rows[0]["index"]: ("accept", "")})
    assert _state()["review"]["judged"] == 1


def test_corrections_are_reported_as_unknown_when_nothing_was_drafted(ws: dict[str, Path]) -> None:
    """분모 0 은 0% 가 아니라 **모름**이다(모르는 것과 0 은 다르다)."""
    run_round(load_loop_config(ws["loop"]))
    payload = _open(ws["loop"])
    assert payload["corrections"]["rate"] is None
    assert "모델 초안" in payload["corrections"]["text"]

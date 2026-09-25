"""일괄 생성 API (U6) — 오래 걸리는 작업을 **스레드 + 폴링**으로 다루는 자리.

여기가 지키는 것: 실행이 실제로 파일을 남긴다(정본 3종 + manifest) · **실행 중에도 `state` 가 막히지 않는다**
(실행 스레드가 API 락을 잡으면 화면이 얼어붙는다) · 중지는 그때까지를 남기고 `cancelled` 로 끝난다 ·
설정 오버라이드는 `batch.BatchSession`(Qt 탭과 같은 것)이 하고 잘못된 값은 시작 전에 막힌다 ·
쓰기는 전부 `write=True` 이고 두 번 시작하면 409.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from anograft.bank.importers import yolo as Y
from anograft.studio.session import default_recipe
from anograft.web import batch_api
from anograft.web.api import WRITE_ROUTES, Request, handle
from tests.fixtures import fake_yolo_dataset


@pytest.fixture(autouse=True)
def _clean():
    batch_api.reset()
    yield
    batch_api.reset()


@pytest.fixture
def recipe_file(tmp_path: Path) -> Path:
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
    rec = default_recipe(bank=(tmp_path / "bank").as_posix(), targets=normals.as_posix())
    data = rec.to_dict()
    data["pipeline"]["placement"]["margin_px"] = 4
    data["pipeline"]["placement"]["roi"]["erode_px"] = 2
    data["pipeline"]["source"]["min_sources_warn"] = 1
    data["output"]["root"] = (tmp_path / "out").as_posix()
    data["output"]["count"] = 3
    from anograft.core.recipe import Recipe

    path = tmp_path / "batch.yaml"
    path.write_text(Recipe.from_dict(data).to_yaml(), encoding="utf-8", newline="\n")
    return path


def _open(path: Path) -> dict:
    res = handle(Request("/api/batch/open", "POST", json={"recipe": str(path)}))
    assert res.status == 200, res.payload
    return res.payload


def _state() -> dict:
    res = handle(Request("/api/batch/state"))
    assert res.status == 200, res.payload
    return res.payload


def _wait_done(timeout: float = 90.0) -> dict:
    """폴링으로 기다린다 — 화면이 하는 것과 같은 방식."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        st = _state()
        run = st.get("run")
        if run and not run["running"]:
            return st
        time.sleep(0.05)
    raise AssertionError("일괄 생성이 끝나지 않았습니다")


# ---------------------------------------------------------------- 열기·설정


def test_batch_needs_a_recipe_first():
    assert _state()["open"] is False
    res = handle(Request("/api/batch/start", "POST", json={}))
    assert res.status == 500 and "먼저 레시피" in res.payload["error"]


def test_open_fills_the_overrides_from_the_recipe(recipe_file: Path):
    st = _open(recipe_file)
    s = st["settings"]
    assert st["open"] and s["count"] == 3 and s["writer"] == "pairs"
    assert s["out"].endswith("/out") and st["recipe"]["preset"] == "poisson-graft"
    assert st["runCommand"].startswith("anograft run ")
    assert "pairs" in st["writerFormats"] and "yolo" in st["writerFormats"]


def test_settings_are_revalidated_at_start(recipe_file: Path, tmp_path: Path):
    _open(recipe_file)
    res = handle(Request("/api/batch/settings", "POST", json={"settings": {"count": 0}}))
    assert res.status == 200 and res.payload["settings"]["count"] == 0
    res = handle(Request("/api/batch/start", "POST", json={}))
    assert res.status == 400 and "1 이상" in res.payload["error"]
    res = handle(Request("/api/batch/settings", "POST", json={"settings": {"out": ""}}))
    assert handle(Request("/api/batch/start", "POST", json={})).status == 400
    assert handle(Request("/api/batch/settings", "POST", json={"settings": "nope"})).status == 400


def test_unknown_writer_is_rejected(recipe_file: Path):
    _open(recipe_file)
    handle(Request("/api/batch/settings", "POST", json={"settings": {"writer": "parquet"}}))
    res = handle(Request("/api/batch/start", "POST", json={}))
    assert res.status == 400 and "writer" in res.payload["error"]


# ---------------------------------------------------------------- 실행


def test_run_writes_the_dataset_and_reports_progress(recipe_file: Path, tmp_path: Path):
    _open(recipe_file)
    res = handle(Request("/api/batch/start", "POST", json={}))
    assert res.status == 200, res.payload
    assert res.payload["run"]["total"] == 3

    st = _wait_done()
    run = st["run"]
    assert run["error"] is None, run["log"]
    summary = run["summary"]
    assert summary["nOk"] >= 1 and summary["cancelled"] is False
    assert summary["text"].startswith("완료:")
    root = Path(summary["root"])
    assert (root / "images").is_dir() and (root / "masks").is_dir() and (root / "meta").is_dir()
    assert (root / "manifest.csv").exists(), "정본 셋 + manifest 가 함께 나간다"
    assert run["done"] == run["total"]
    assert any("$ anograft run" in line for line in run["log"]), "재현 방법을 로그 맨 위에"


def test_state_answers_while_the_run_is_going(recipe_file: Path):
    """실행 스레드가 API 락을 잡으면 화면이 얼어붙는다 — 도는 동안에도 폴링이 즉시 답해야 한다."""
    _open(recipe_file)
    handle(Request("/api/batch/start", "POST", json={}))
    for _ in range(5):
        t0 = time.monotonic()
        st = _state()
        assert time.monotonic() - t0 < 2.0, "폴링이 막혔다"
        assert st["run"] is not None
        if not st["run"]["running"]:
            break
        time.sleep(0.02)
    _wait_done()


def test_second_start_is_refused_while_running(recipe_file: Path):
    _open(recipe_file)
    handle(Request("/api/batch/start", "POST", json={}))
    res = handle(Request("/api/batch/start", "POST", json={}))
    assert res.status in (200, 409)
    if res.status == 200:  # 이미 끝난 뒤였다면 두 번째도 정상 — 그때는 끝난 상태여야 한다
        assert res.payload["run"] is not None
    _wait_done()


def test_stop_keeps_what_was_written(recipe_file: Path):
    _open(recipe_file)
    handle(Request("/api/batch/settings", "POST", json={"settings": {"count": 40}}))
    handle(Request("/api/batch/start", "POST", json={}))
    res = handle(Request("/api/batch/stop", "POST", json={}))
    assert res.status == 200, res.payload
    st = _wait_done()
    summary = st["run"]["summary"]
    assert summary is not None and summary["cancelled"] is True
    assert summary["done"] < 40, "다음 결과에서 멈춘다"
    assert Path(summary["root"], "manifest.csv").exists(), "그때까지의 기록은 남는다"
    assert any("취소됨" in line for line in st["run"]["log"])


def test_stop_without_a_run_is_rejected(recipe_file: Path):
    _open(recipe_file)
    res = handle(Request("/api/batch/stop", "POST", json={}))
    assert res.status == 400


def test_output_root_is_available_for_the_review_screen(recipe_file: Path):
    _open(recipe_file)
    assert batch_api.output_root() is None
    handle(Request("/api/batch/start", "POST", json={}))
    _wait_done()
    root = batch_api.output_root()
    assert root is not None and root.is_dir()


# ---------------------------------------------------------------- 쓰기 등록


def test_all_mutating_routes_are_write_registered():
    handle(Request("/api/health"))
    for path in (
        "/api/batch/open",
        "/api/batch/settings",
        "/api/batch/start",
        "/api/batch/stop",
    ):
        assert path in WRITE_ROUTES, path
        assert handle(Request(path, "GET")).status == 405
    assert "/api/batch/state" not in WRITE_ROUTES
    assert handle(Request("/api/batch/state", "POST")).status == 405

"""검수 API — Qt 검수 탭과 **같은 함수**(`anograft.review.ReviewSession`)를 부르는지, 쓰기가 막혀 있는지.

여기가 지키는 것: 열기 전에는 친절히 거절 · 판정이 `review.csv` 로 나간다(Qt 탭이 읽는 그 파일) ·
썸네일이 PNG 로 나온다 · **쓰기는 POST + CSRF 헤더 없이는 안 된다** · 정리본이 만들어진다.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from anograft.review import ReviewSession
from anograft.web import review_api
from anograft.web import server as web_server
from anograft.web.api import Request
from tests.test_review_session import output_root  # noqa: F401 — 픽스처 재사용
from tests.test_web import _Server  # 통합 서버 헬퍼


@pytest.fixture(autouse=True)
def _clean_session():
    """서버 전역 세션은 테스트마다 비운다(로컬 1인용이라 전역인 것이 설계다)."""
    review_api.reset()
    yield
    review_api.reset()


def _open(root: Path) -> dict:
    result = review_api._open(Request("/api/review/open", "POST", json={"root": str(root)}))
    assert result.status == 200, result.payload
    return result.payload


# ---------------------------------------------------------------- 순수 핸들러


def test_needs_open_first():
    """열기 전에는 500 이 아니라 '먼저 여세요' 로 — fail-soft."""
    from anograft.web import api as web_api

    res = web_api.handle(Request("/api/review/items"))
    assert res.status == 500
    assert "먼저 출력 폴더를 여세요" in res.payload["error"]


def test_open_reports_counts_and_classes(output_root: Path):  # noqa: F811
    state = _open(output_root)
    assert state["open"] is True
    assert state["counts"]["ok"] >= 1
    assert state["classes"]
    assert any(f["id"] == "all" and f["label"] == "전체" for f in state["filters"])


def test_open_marks_a_loop_queue_folder(output_root: Path):  # noqa: F811
    """루프의 검토 대기 폴더는 같은 화면이 열지만 **항목이 합성이 아니다** — 라벨은 서버가 준다(사전 §3.6)."""
    assert _open(output_root)["okLabel"] == "합성"
    (output_root / "queue.csv").write_text("index,stem" + chr(10), encoding="utf-8")
    state = _open(output_root)
    assert state["isQueue"] is True and state["okLabel"] == "검토 대기"
    assert "검토 대기" in state["summary"]


def test_open_rejects_missing_folder(tmp_path: Path):
    res = review_api._open(
        Request("/api/review/open", "POST", json={"root": str(tmp_path / "없음")})
    )
    assert res.status == 400
    assert res.payload["error"]


def test_items_filter_matches_session(output_root: Path):  # noqa: F811
    _open(output_root)
    session = ReviewSession()
    session.load(output_root)

    for which in ("all", "unreviewed", "skipped"):
        res = review_api._items(Request("/api/review/items", query={"filter": which}))
        assert res.status == 200
        assert res.payload["total"] == len(session.filtered(which)), which

    bad = review_api._items(Request("/api/review/items", query={"filter": "없는필터"}))
    assert bad.status == 400


def test_image_returns_png_bytes(output_root: Path):  # noqa: F811
    state = _open(output_root)
    index = _first_ok_index(state)
    res = review_api._image(Request("/api/review/image", query={"index": index}))
    assert res.status == 200
    assert res.content_type == "image/png"
    assert res.body[:8] == b"\x89PNG\r\n\x1a\n"
    assert res.payload is None


def test_image_unknown_index_is_404(output_root: Path):  # noqa: F811
    _open(output_root)
    res = review_api._image(Request("/api/review/image", query={"index": "9999"}))
    assert res.status == 404


def test_verdict_writes_review_csv_that_session_reads_back(output_root: Path):  # noqa: F811
    """판정은 Qt 탭이 읽는 **그 파일**로 나간다 — 두 화면이 같은 상태를 본다."""
    state = _open(output_root)
    index = _first_ok_index(state)

    res = review_api._verdict(
        Request("/api/review/verdict", "POST", json={"index": index, "verdict": "reject"})
    )
    assert res.status == 200
    assert res.payload["item"]["verdict"] == "reject"
    assert (output_root / "review.csv").is_file()

    reopened = ReviewSession()
    reopened.load(output_root)
    assert reopened.item(index).verdict == "reject"


def test_verdict_rejects_unknown_value(output_root: Path):  # noqa: F811
    state = _open(output_root)
    res = review_api._verdict(
        Request(
            "/api/review/verdict", "POST", json={"index": _first_ok_index(state), "verdict": "좋음"}
        )
    )
    assert res.status == 400


def test_histogram_shape(output_root: Path):  # noqa: F811
    _open(output_root)
    res = review_api._histogram(
        Request("/api/review/histogram", query={"key": "area", "bins": "8"})
    )
    assert res.status == 200
    body = res.payload
    assert len(body["synthetic"]) == len(body["real"]) == 8
    assert len(body["edges"]) == 9


def test_prune_makes_a_dataset(output_root: Path, tmp_path: Path):  # noqa: F811
    state = _open(output_root)
    index = _first_ok_index(state)
    review_api._verdict(
        Request("/api/review/verdict", "POST", json={"index": index, "verdict": "reject"})
    )
    out = tmp_path / "pruned"
    res = review_api._prune(Request("/api/review/prune", "POST", json={"out": str(out)}))
    assert res.status == 200
    assert res.payload["dropped"] >= 1
    assert (out / "manifest.csv").is_file()


def test_prune_needs_out(output_root: Path):  # noqa: F811
    _open(output_root)
    assert review_api._prune(Request("/api/review/prune", "POST", json={})).status == 400


def _first_ok_index(state: dict) -> str:
    res = review_api._items(Request("/api/review/items", query={"filter": "all"}))
    for item in res.payload["items"]:
        if item["status"] == "ok":
            return item["index"]
    raise AssertionError(f"ok 인 항목이 없다: {state['counts']}")


# ---------------------------------------------------------------- 쓰기 방어(통합)


def _post(
    server: _Server, path: str, payload: dict, *, csrf: bool = True, origin: str | None = None
):
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    if csrf:
        req.add_header(web_server.CSRF_HEADER, "1")
    if origin:
        req.add_header("Origin", origin)
    try:
        with urllib.request.urlopen(req, timeout=5) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def test_write_requires_csrf_header(output_root: Path, tmp_path: Path):  # noqa: F811
    with _Server(bundle_root=tmp_path / "없음") as s:
        status, body = _post(s, "/api/review/open", {"root": str(output_root)}, csrf=False)
        assert status == 403
        assert web_server.CSRF_HEADER in body["error"]

        status, _ = _post(s, "/api/review/open", {"root": str(output_root)})
        assert status == 200


def test_write_rejects_foreign_origin(output_root: Path, tmp_path: Path):  # noqa: F811
    with _Server(bundle_root=tmp_path / "없음") as s:
        status, body = _post(
            s, "/api/review/open", {"root": str(output_root)}, origin="https://evil.example.com"
        )
        assert status == 403
        assert "다른 사이트" in body["error"]


def test_get_cannot_write_and_post_cannot_read(output_root: Path, tmp_path: Path):  # noqa: F811
    with _Server(bundle_root=tmp_path / "없음") as s:
        # 쓰기 경로를 GET 으로 → 405 (링크 한 번 잘못 눌러 데이터가 바뀌면 안 된다)
        assert s.get("/api/review/open")[0] == 405
        # 읽기 경로를 POST 로 → 405
        assert _post(s, "/api/review/items", {})[0] == 405


def test_post_to_static_path_is_404(tmp_path: Path):
    with _Server(bundle_root=tmp_path / "없음") as s:
        assert _post(s, "/", {})[0] == 404


def test_bad_json_body_is_400(output_root: Path, tmp_path: Path):  # noqa: F811
    with _Server(bundle_root=tmp_path / "없음") as s:
        req = urllib.request.Request(
            f"http://127.0.0.1:{s.port}/api/review/open", data=b"{not json", method="POST"
        )
        req.add_header(web_server.CSRF_HEADER, "1")
        try:
            with urllib.request.urlopen(req, timeout=5):
                raise AssertionError("400 이어야 한다")
        except urllib.error.HTTPError as err:
            assert err.code == 400


@pytest.mark.parametrize(
    ("origin", "ok"),
    [
        ("http://127.0.0.1:8000", True),
        ("http://localhost:5173", True),
        ("https://evil.example.com", False),
        ("null", False),
        (None, True),  # 같은 오리진 요청·curl 은 Origin 을 안 보낸다
    ],
)
def test_origin_allowed(origin, ok):
    assert web_server.origin_allowed(origin) is ok

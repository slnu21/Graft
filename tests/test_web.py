"""웹 계층 — 순수 함수(라우팅·경로 해석·Host 검사) + 실제로 띄워 보는 통합 한 바퀴.

여기가 지키는 것: **번들이 없어도 서버가 죽지 않는다**(소스 체크아웃의 정상 상태) · **번들 밖 파일은 절대 안 나간다** ·
**로컬이 아닌 Host 는 거부**(DNS rebinding) · 한국어가 깨지지 않는다(charset).
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from anograft.web import api as web_api
from anograft.web import server as web_server
from anograft.web import static as web_static


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """빌드된 번들 흉내 — index.html + 해시 붙은 자산."""
    root = tmp_path / "static"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>Graft</title>", encoding="utf-8")
    (root / "assets" / "index-abc123.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("비밀", encoding="utf-8")
    return root


# ---------------------------------------------------------------- 순수: API 라우팅


def test_health_has_version():
    result = web_api.handle_api("/api/health")
    assert result.status == 200
    assert result.payload["name"] == "anograft"
    assert result.payload["ok"] is True


def test_methods_carry_korean_labels_from_help():
    """라벨의 한 원천은 `core/help.py` — 프론트에 사전 사본을 두지 않으려고 서버가 실어 준다."""
    payload = web_api.handle_api("/api/methods").payload
    assert payload["methods"], "method 가 하나도 없다"
    assert all("label" in m for m in payload["methods"])
    stages = {s["stage"]: s for s in payload["stages"]}
    assert stages["source"]["label"] == "결함 고르기"
    # 파이프라인 순서(1~7)를 지킨다 — registry 등록 순서가 아니다
    order = [s["stage"] for s in payload["stages"]]
    assert order.index("source") < order.index("blend") < order.index("gtmask")


def test_unknown_api_is_404_json_not_static():
    result = web_api.handle_api("/api/nope")
    assert result.status == 404
    assert "routes" in result.payload


def test_handler_failure_is_500_not_crash(monkeypatch):
    def boom(_query):
        raise RuntimeError("터졌다")

    monkeypatch.setitem(web_api.ROUTES, "/api/boom", boom)
    result = web_api.handle_api("/api/boom")
    assert result.status == 500
    assert "터졌다" in result.payload["error"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [("/api", True), ("/api/health", True), ("/apifoo", False), ("/", False), ("/assets/x", False)],
)
def test_is_api_path(path, expected):
    assert web_api.is_api_path(path) is expected


# ---------------------------------------------------------------- 순수: 정적 경로


def test_resolve_static_serves_index_and_assets(bundle: Path):
    assert web_static.resolve_static(bundle, "/") == (bundle / "index.html").resolve()
    assert web_static.resolve_static(bundle, "/index.html") == (bundle / "index.html").resolve()
    assert (
        web_static.resolve_static(bundle, "/assets/index-abc123.js")
        == (bundle / "assets" / "index-abc123.js").resolve()
    )


@pytest.mark.parametrize(
    "attack",
    [
        "/../secret.txt",
        "/assets/../../secret.txt",
        "/./../../secret.txt",
        "/assets/../../../../../../etc/passwd",
    ],
)
def test_resolve_static_refuses_to_leave_the_bundle(bundle: Path, attack: str):
    assert web_static.resolve_static(bundle, attack) is None


def test_resolve_static_returns_none_for_missing(bundle: Path):
    assert web_static.resolve_static(bundle, "/nope.js") is None


def test_content_type_keeps_utf8(bundle: Path):
    assert "charset=utf-8" in web_static.content_type(bundle / "index.html")
    assert "charset=utf-8" in web_static.content_type(bundle / "assets" / "index-abc123.js")
    assert web_static.content_type(Path("x.png")) == "image/png"


def test_bundle_status_reports_missing(tmp_path: Path):
    missing = web_static.bundle_status(tmp_path / "없음")
    assert not missing.present and missing.reason
    # 안내 페이지는 API 가 살아 있다는 것을 알려 준다
    assert "/api/health" in web_static.missing_bundle_html()


@pytest.mark.parametrize(
    ("header", "ok"),
    [
        ("127.0.0.1:8000", True),
        ("localhost:8000", True),
        ("127.0.0.1", True),
        ("[::1]:8000", True),
        ("evil.example.com", False),
        ("evil.example.com:8000", False),
        (None, False),
        ("", False),
    ],
)
def test_host_allowed(header, ok):
    assert web_server.host_allowed(header, 8000) is ok


# ---------------------------------------------------------------- 통합: 실제로 띄운다


class _Server:
    def __init__(self, **kwargs):
        self.srv = web_server.make_server(0, **kwargs)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(timeout=5)

    def get(self, path: str, host: str | None = None) -> tuple[int, bytes, dict]:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}")
        if host is not None:
            req.add_header("Host", host)
        try:
            with urllib.request.urlopen(req, timeout=5) as res:
                return res.status, res.read(), dict(res.headers)
        except urllib.error.HTTPError as err:
            return err.code, err.read(), dict(err.headers)

    def raw(self, request_line: str) -> bytes:
        """urllib 이 정규화해 버리는 경로를 그대로 보내려면 소켓으로 쏴야 한다."""
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.sendall(
                f"{request_line} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n".encode()
            )
            chunks = []
            while chunk := sock.recv(4096):
                chunks.append(chunk)
            return b"".join(chunks)


def test_server_serves_api_and_bundle(bundle: Path):
    with _Server(bundle_root=bundle) as s:
        status, body, headers = s.get("/api/health")
        assert status == 200
        assert json.loads(body)["name"] == "anograft"
        assert headers["Content-Type"] == "application/json; charset=utf-8"

        status, body, headers = s.get("/")
        assert status == 200 and b"Graft" in body
        assert headers["Cache-Control"] == "no-store"  # 진입점은 캐시하지 않는다

        status, _, headers = s.get("/assets/index-abc123.js")
        assert status == 200
        assert "immutable" in headers["Cache-Control"]  # 해시 붙은 자산만 오래 캐시


def test_server_without_bundle_explains_instead_of_dying(tmp_path: Path):
    with _Server(bundle_root=tmp_path / "없음") as s:
        status, body, _ = s.get("/")
        assert status == 200
        assert "웹 번들" in body.decode("utf-8")
        assert s.get("/api/health")[0] == 200  # API 는 그대로 산다


def test_server_rejects_foreign_host(bundle: Path):
    with _Server(bundle_root=bundle) as s:
        assert s.get("/api/health", host="evil.example.com")[0] == 403


def test_server_refuses_path_traversal(bundle: Path):
    with _Server(bundle_root=bundle) as s:
        assert b"\xeb\xb9\x84\xeb\xb0\x80" not in s.raw("GET /../secret.txt")
        assert b"\xeb\xb9\x84\xeb\xb0\x80" not in s.raw("GET /assets/%2e%2e/%2e%2e/secret.txt")


def test_api_only_does_not_serve_files(bundle: Path):
    with _Server(bundle_root=bundle, api_only=True) as s:
        assert s.get("/api/health")[0] == 200
        status, body, _ = s.get("/")
        assert status == 404
        assert "npm run dev" in json.loads(body)["error"]


def test_head_has_no_body(bundle: Path):
    with _Server(bundle_root=bundle) as s:
        head = s.raw("HEAD /api/health")
        assert b"200" in head.split(b"\r\n", 1)[0]
        assert b'{"name"' not in head

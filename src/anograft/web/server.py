"""로컬 웹 서버 — **stdlib `http.server` 만**. 런타임 의존성 0 원칙(순수 wheel 4개)을 지킨다.

- **`127.0.0.1` 고정 바인딩.** 오프라인·로컬 표명과 직결이라 v0.10 에는 `--host` 를 열지 않는다
  (현장 작업자가 브라우저로 붙는 그림은 v1.x 의 확인 게이트).
- `Host` 헤더를 검사한다 — 로컬 바인딩만으로는 **DNS rebinding**(외부 도메인이 127.0.0.1 로 해석되게 해
  브라우저가 이 서버를 대신 두드리게 하는 것)을 막지 못한다.
- 프론트가 없어도 **API 는 뜬다**(`--api-only`, 그리고 번들 없을 때의 안내 페이지).

JSON API 는 `api.py`(순수), 파일 해석은 `static.py`(순수) — 이 모듈은 그 둘을 소켓에 잇기만 한다.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from anograft.web import api as web_api
from anograft.web import static as web_static

DEFAULT_PORT = 8000
DEFAULT_DEV_PORT = 5173
HOST = "127.0.0.1"


def host_allowed(header: str | None, port: int) -> bool:
    """`Host` 헤더가 로컬을 가리키는가 — 아니면 브라우저가 남의 페이지를 대신 실어 나르는 중이다."""
    if not header:
        return False
    name = header.rsplit(":", 1)[0] if not header.startswith("[") else header.split("]", 1)[0] + "]"
    return name in {"127.0.0.1", "localhost", "[::1]", "::1"}


def port_in_use(port: int, host: str = HOST) -> bool:
    """이미 떠 있는지 — 에러 메시지를 '주소 사용 중' 대신 사람 말로 내기 위한 사전 확인."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


class GraftHandler(BaseHTTPRequestHandler):
    """GET·HEAD 만 받는다. 쓰기는 U3 이후에 POST 로 들어온다(그때 CSRF 를 같이 본다)."""

    server_version = "anograft"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # 서버가 인스턴스에 심어 준다
    bundle_root: Path
    api_only: bool = False
    verbose: bool = False
    dev_port: int = DEFAULT_DEV_PORT

    def do_GET(self) -> None:
        self._handle(body=True)

    def do_HEAD(self) -> None:
        self._handle(body=False)

    # ------------------------------------------------------------------ 내부
    def _handle(self, *, body: bool) -> None:
        if not host_allowed(self.headers.get("Host"), self.server.server_address[1]):
            self._send_json(403, {"error": "로컬(127.0.0.1)에서만 쓸 수 있습니다."}, body=body)
            return

        parts = urlsplit(self.path)
        path = parts.path

        if web_api.is_api_path(path):
            query = dict(parse_qsl(parts.query))
            result = web_api.handle_api(path, query)
            self._send_json(result.status, result.payload, body=body, extra=result.headers)
            return

        if self.api_only:
            self._send_json(
                404,
                {"error": "API 전용으로 떠 있습니다 — 화면은 개발 서버(npm run dev)에서 봅니다."},
                body=body,
            )
            return

        target = web_static.resolve_static(self.bundle_root, path)
        if target is None:
            status = web_static.bundle_status(self.bundle_root)
            if not status.present:
                self._send_html(200, web_static.missing_bundle_html(self.dev_port), body=body)
            else:
                self._send_json(404, {"error": f"없는 파일입니다: {path}"}, body=body)
            return

        self._send_file(target, body=body)

    def _send_file(self, target: Path, *, body: bool) -> None:
        data = target.read_bytes()
        # 해시가 붙은 자원(assets/)만 오래 캐시하고, 진입점은 항상 새로 받는다.
        cache = (
            "public, max-age=31536000, immutable"
            if "assets" in target.parts and target.name != "index.html"
            else "no-store"
        )
        self._respond(200, web_static.content_type(target), data, body=body, cache=cache)

    def _send_json(
        self, status: int, payload: dict, *, body: bool, extra: dict[str, str] | None = None
    ) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._respond(status, "application/json; charset=utf-8", data, body=body, extra=extra)

    def _send_html(self, status: int, html: str, *, body: bool) -> None:
        self._respond(status, "text/html; charset=utf-8", html.encode("utf-8"), body=body)

    def _respond(
        self,
        status: int,
        ctype: str,
        data: bytes,
        *,
        body: bool,
        cache: str = "no-store",
        extra: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        # 로컬 전용 — 외부 자원을 부르지 않는다는 것을 헤더로도 못 박는다(CDN 금지 규약의 런타임 판).
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        if self.verbose:
            super().log_message(fmt, *args)


def make_server(
    port: int = DEFAULT_PORT,
    *,
    api_only: bool = False,
    verbose: bool = False,
    bundle_root: Path | None = None,
    dev_port: int = DEFAULT_DEV_PORT,
) -> ThreadingHTTPServer:
    """서버 객체만 만든다(테스트가 포트 0 으로 띄워 쓴다). 실행 루프는 `serve`."""
    handler = type(
        "BoundGraftHandler",
        (GraftHandler,),
        {
            "bundle_root": (bundle_root or web_static.BUNDLE_DIR),
            "api_only": api_only,
            "verbose": verbose,
            "dev_port": dev_port,
        },
    )
    server = ThreadingHTTPServer((HOST, port), handler)
    server.daemon_threads = True
    return server


def serve(
    port: int = DEFAULT_PORT,
    *,
    api_only: bool = False,
    open_browser: bool = True,
    verbose: bool = False,
    bundle_root: Path | None = None,
    dev_port: int = DEFAULT_DEV_PORT,
    on_ready=None,
) -> int:
    """블로킹 실행. 종료 코드를 돌려준다(포트 충돌 등은 1)."""
    if port_in_use(port):
        print(f"{HOST}:{port} 는 이미 쓰는 중입니다 — 다른 포트를 주세요(--port).")
        return 1

    server = make_server(
        port, api_only=api_only, verbose=verbose, bundle_root=bundle_root, dev_port=dev_port
    )
    actual = server.server_address[1]
    url = f"http://{HOST}:{actual}/"
    status = web_static.bundle_status(bundle_root or web_static.BUNDLE_DIR)

    print(f"Graft 웹 서버 — {url}  (Ctrl+C 로 끝냅니다)")
    if api_only:
        print("  API 전용입니다 — 화면은 web/ 에서 npm run dev 로 띄우세요.")
    elif not status.present:
        print(f"  {status.reason}")
        print(
            f"  화면은 개발 서버(http://{HOST}:{dev_port}/)에서 보거나 npm run build 로 만드세요."
        )
    if on_ready is not None:
        on_ready(actual)
    if open_browser and not api_only:
        threading.Timer(0.3, _open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n끝냅니다.")
    finally:
        server.shutdown()
        server.server_close()
    return 0


def _open(url: str) -> None:
    import webbrowser

    webbrowser.open(url)

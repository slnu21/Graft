"""웹 API — **순수 계층**: `http.server`·소켓을 모르고, 요청 값을 받아 결과 값을 돌려준다.

규약(`gui/` 의 웹판): **`web/` 프론트는 API 를 호출만 한다 — 합성·판정 로직 0.** 그 뒤를 받치는 이 모듈도
계산을 새로 하지 않는다. `core`·`runner`·`review` 가 이미 내는 것을 JSON 모양으로 옮길 뿐이고,
새 판단이 필요하면 그건 `core` 에 순수 함수로 들어가야 한다.

핸들러는 **fail-soft** 다 — 예외를 던져 서버를 죽이지 않고 `{"error": ...}` 와 5xx 를 돌려준다
(선택 의존성이 없는 PC 에서도 나머지 화면은 떠야 한다).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from anograft import __version__


@dataclass(frozen=True)
class ApiResult:
    """핸들러 결과.

    보통은 `payload`(JSON 이 될 dict)를 채우고, 이미지처럼 **바이트를 그대로** 보내야 하면
    `body`+`content_type` 을 채운다(검수 썸네일이 그렇다). 둘 다 채우지는 않는다.
    """

    status: int
    payload: dict | None = None
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | None = None
    content_type: str = ""


@dataclass(frozen=True)
class Request:
    """핸들러가 보는 요청 — **소켓이 아니라 값**이라 테스트가 그대로 만들어 넣는다."""

    path: str
    method: str = "GET"
    query: Mapping[str, str] = field(default_factory=dict)
    json: dict = field(default_factory=dict)

    def get(self, key: str, default: str = "") -> str:
        return self.query.get(key, default)

    def int_of(self, key: str, default: int) -> int:
        """질의 문자열의 숫자 — 사람이 손으로 URL 을 고칠 수 있으니 조용히 기본값으로."""
        try:
            return int(self.query.get(key, default))
        except (TypeError, ValueError):
            return default


Handler = Callable[[Request], ApiResult]

#: 쓰기(POST)를 받는 경로. GET 은 부수효과가 없어야 하고, 쓰기는 반드시 여기에 등록된다
#: (`server.py` 가 이 집합을 보고 CSRF 검사를 건다).
WRITE_ROUTES: set[str] = set()


def _health(_req: Request) -> ApiResult:
    """서버가 살아 있는지 + 어느 버전인지. 프론트가 가장 먼저 부른다."""
    return ApiResult(200, {"name": "anograft", "version": __version__, "ok": True})


def _doctor(_req: Request) -> ApiResult:
    """환경 진단 — CLI `anograft doctor` 와 **같은 원천**(`cli.doctor_info`)."""
    from anograft.cli import doctor_info  # 지연 import: cli → web(serve) 순환을 피한다

    return ApiResult(200, doctor_info())


def _methods(_req: Request) -> ApiResult:
    """스테이지별 알고리즘과 가용 여부 — CLI `anograft methods` 와 같은 원천(`core.registry`).

    **한국어 라벨도 여기서 붙인다.** 용어 사전의 한 원천은 `core/help.py` 이므로 프론트에 사전 사본을
    두지 않는다(두면 둘이 갈린다 — v0.9 가 라벨을 한 곳으로 모은 이유).
    """
    from anograft.core import registry
    from anograft.core.help import method_help, stage_help

    items = []
    for i in registry.list_methods(None):
        mh = method_help(i.stage, i.method)
        items.append(
            {
                "stage": i.stage,
                "method": i.method,
                "usable": i.usable,
                "reason": i.reason,
                "label": mh.label if mh else i.method,
                "summary": mh.summary if mh else "",
            }
        )
    stages = [
        {"stage": name, "label": sh.label, "desc": sh.desc, "en": sh.en}
        for name, sh in ((n, stage_help(n)) for n in _stage_order(items))
        if sh is not None
    ]
    return ApiResult(200, {"methods": items, "stages": stages})


def _stage_order(items: list[dict]) -> list[str]:
    """파이프라인 순서를 지킨다 — registry 등록 순서가 아니라 사람이 보는 1~7 순서."""
    order = ["source", "geometry", "roi", "placement", "blend", "harmonize", "degrade", "gtmask"]
    seen = {i["stage"] for i in items}
    return [s for s in order if s in seen] + sorted(seen - set(order))


ROUTES: dict[str, Handler] = {
    "/api/health": _health,
    "/api/doctor": _doctor,
    "/api/methods": _methods,
}


def register(path: str, handler: Handler, *, write: bool = False) -> None:
    """라우트 한 줄 등록. `write=True` 는 POST 전용이 되고 CSRF 검사를 받는다."""
    ROUTES[path] = handler
    if write:
        WRITE_ROUTES.add(path)


def is_api_path(path: str) -> bool:
    """`/api` 로 시작하면 API — 없는 하위 경로라도 정적 파일로 흘려보내지 않는다(404 JSON)."""
    return path == "/api" or path.startswith("/api/")


def handle(req: Request) -> ApiResult:
    """요청 하나를 처리한다. 모르는 경로는 404, 핸들러가 터지면 500 — 둘 다 JSON 으로."""
    from anograft.web import review_api  # 지연 등록: import 만으로 라우트가 붙는다

    review_api.ensure_registered()

    path = req.path.rstrip("/") or "/"
    handler = ROUTES.get(path)
    if handler is None:
        return ApiResult(
            404, {"error": f"그런 API 가 없습니다: {req.path}", "routes": sorted(ROUTES)}
        )
    if path in WRITE_ROUTES and req.method != "POST":
        return ApiResult(405, {"error": "이 API 는 POST 로 부릅니다."})
    if path not in WRITE_ROUTES and req.method == "POST":
        return ApiResult(405, {"error": "이 API 는 읽기 전용입니다."})
    try:
        return handler(req)
    except Exception as exc:  # fail-soft — 한 화면이 못 뜨는 것이 서버가 죽는 것보다 낫다
        return ApiResult(500, {"error": f"{type(exc).__name__}: {exc}", "path": req.path})


def handle_api(path: str, query: Mapping[str, str] | None = None) -> ApiResult:
    """예전 진입점(GET 전용) — 호출부를 한꺼번에 바꾸지 않으려고 남겨 둔다."""
    return handle(Request(path, "GET", query or {}))

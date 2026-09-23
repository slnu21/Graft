"""웹 API — **순수 계층**: `http.server`·소켓·파일 응답을 모르고, 경로+질의를 받아 JSON 이 될 dict 를 돌려준다.

규약(`gui/` 의 웹판): **`web/` 프론트는 API 를 호출만 한다 — 합성·판정 로직 0.** 그 뒤를 받치는 이 모듈도
계산을 새로 하지 않는다. `core`·`runner`·`loop` 가 이미 내는 것을 JSON 모양으로 옮길 뿐이고,
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
    """핸들러 결과 — 상태 코드와 JSON 본문. 파일·바이트 응답은 정적 계층(`static.py`)이 맡는다."""

    status: int
    payload: dict
    headers: dict[str, str] = field(default_factory=dict)


Handler = Callable[[Mapping[str, str]], ApiResult]


def _health(_query: Mapping[str, str]) -> ApiResult:
    """서버가 살아 있는지 + 어느 버전인지. 프론트가 가장 먼저 부른다."""
    return ApiResult(200, {"name": "anograft", "version": __version__, "ok": True})


def _doctor(_query: Mapping[str, str]) -> ApiResult:
    """환경 진단 — CLI `anograft doctor` 와 **같은 원천**(`cli.doctor_info`)."""
    from anograft.cli import doctor_info  # 지연 import: cli → web(serve) 순환을 피한다

    return ApiResult(200, doctor_info())


def _methods(_query: Mapping[str, str]) -> ApiResult:
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


def is_api_path(path: str) -> bool:
    """`/api` 로 시작하면 API — 없는 하위 경로라도 정적 파일로 흘려보내지 않는다(404 JSON)."""
    return path == "/api" or path.startswith("/api/")


def handle_api(path: str, query: Mapping[str, str] | None = None) -> ApiResult:
    """경로 하나를 처리한다. 모르는 경로는 404, 핸들러가 터지면 500 — 둘 다 JSON 으로."""
    handler = ROUTES.get(path.rstrip("/") or "/")
    if handler is None:
        return ApiResult(404, {"error": f"그런 API 가 없습니다: {path}", "routes": sorted(ROUTES)})
    try:
        return handler(query or {})
    except Exception as exc:  # fail-soft — 한 화면이 못 뜨는 것이 서버가 죽는 것보다 낫다
        return ApiResult(500, {"error": f"{type(exc).__name__}: {exc}", "path": path})

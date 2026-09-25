"""최근 경로 API (U7) — `anograft.recent` 를 **감싸기만** 한다.

읽기 하나뿐이다. **"기억해 줘" 쓰기 API 는 일부러 두지 않았다** — 무언가를 연 행위가 곧 기록이라서,
각 화면의 ``open`` 핸들러가 성공한 뒤에 ``recent.remember`` 를 부른다. 쓰기 라우트를 따로 두면
프론트가 "무엇을 기억할지"를 판단하게 되고, 그건 화면이 상태를 드는 것이다(규약 위반).
"""

from __future__ import annotations

from anograft import recent
from anograft.web.api import ApiResult, Request, register

_REGISTERED = False


def _recent(_req: Request) -> ApiResult:
    """종류별 최근 경로. 파일이 없거나 깨졌으면 빈 목록이 온다(fail-soft — 안내를 띄우지 않는다)."""
    return ApiResult(200, {"recent": recent.load(), "kinds": list(recent.KINDS)})


def ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    register("/api/recent", _recent)
    _REGISTERED = True

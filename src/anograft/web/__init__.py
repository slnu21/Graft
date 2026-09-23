"""웹 UI 계층(v0.10~) — stdlib `http.server` 백엔드 + `web/`(Vite+React) 프론트.

규약 둘:
1. **프론트는 API 를 호출만 한다 — 합성·판정 로직 0**(`gui/` 의 "Qt 는 gui 안에서만" 규약의 웹판).
   비즈니스 로직은 계속 `core`/`runner` 에 있고 pytest 가 지킨다. vitest 는 프론트 순수 함수(포맷·좌표)만.
2. **런타임 의존성은 여전히 0.** 서버는 stdlib 이고, node 는 **빌드 때만** 필요하다
   (사용자는 node 의 존재를 모른다 — `pip install anograft` 가 그대로다).

`from anograft.web.server import serve` 로 쓴다. 이 패키지를 import 하는 것만으로는
소켓을 열지 않는다(CLI `anograft serve` 가 부를 때만).
"""

from __future__ import annotations

__all__ = ["api", "server", "static"]

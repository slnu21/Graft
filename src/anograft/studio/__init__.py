"""미리보기 계층 — **Qt 없음**. 화면(Qt 탭·웹)이 공유하는 상태·작업·스펙·갤러리.

`gui/studio/` 에서 승격했다(U5a). U3 의 `review.py`·U4 의 `bank/browse.py`·`labeling.py` 와 같은 이유:
원래 Qt 가 없는 순수 로직인데 `gui/` 안에 있어서, 웹이 그걸 import 하면 "웹이 GUI 에 의존"하는
거꾸로 된 경계가 생긴다. Qt 은퇴를 생각하면 어차피 나와야 할 코드다.

- `session.py` — `StudioSession`: 레시피(검증된 것만) + `Prepared` 캐시 + 선택(대상·변형) + `generation`.
- `jobs.py` — 미리보기 작업(축소·합성·썸네일)과 "최신만 살리기" 큐, 캔버스 오버레이(BGRA).
- `params.py` — 스테이지 설정 모델(pydantic) → `FieldSpec`(위젯/폼 스펙). "스키마가 곧 UI".
- `gallery.py` — 프리셋 카드 문안 + 현재 바탕으로 프리셋별 썸네일.
- `diagnose.py` — 미리보기 한 장에 붙는 진단(결함 줄·저신뢰 조각·빛 뒤집힘·배치 가능성).

여기에 Qt·`http.server`·경로 대화상자를 들이지 않는다 — 그러면 다시 화면 하나에 묶인다.
"""

from __future__ import annotations

__all__ = ["diagnose", "gallery", "jobs", "params", "session"]

"""결함 보관함 API — `anograft.bank.browse.BankSession` 을 **감싸기만** 한다.

U3 의 검수 API 와 같은 모양: 읽기는 목록·타일 PNG, 쓰기는 POST + CSRF. 필터·정렬·요약은
Qt 보관함 탭이 부르는 **그 순수 함수**(`filter_rows`·`row_of`)가 그대로 낸다.

T4 를 여기서 쓴다 — 보관함을 열면 **평가셋(holdout) 위반**을 함께 보고한다. 은행에 평가셋이 섞이면
그 뒤 라운드 비교가 조용히 무의미해지므로, 사람이 가장 자주 여는 화면이 그걸 말해 줘야 한다.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from anograft.bank.browse import SORT_KEYS, BankSession, BankSessionError, filter_rows
from anograft.web.api import ApiResult, Handler, Request, register

#: 타일 긴 변(px) — 보관함 그리드는 크롭이라 작다.
TILE_LONG_SIDE = 240
DETAIL_LONG_SIDE = 900

_LOCK = threading.Lock()
_SESSION: BankSession | None = None
_ROOT: Path | None = None
_REGISTERED = False


def reset() -> None:
    global _SESSION, _ROOT
    with _LOCK:
        _SESSION = None
        _ROOT = None


def _session() -> BankSession:
    if _SESSION is None:
        raise BankSessionError("먼저 결함 보관함을 여세요.")
    return _SESSION


def _row_json(r: Any) -> dict:
    return {
        "id": r.id,
        "cls": r.cls,
        "name": r.name,
        "areaPx": r.area_px,
        "maskOrigin": r.mask_origin,
        "estimated": r.estimated,
        "confidence": r.confidence,
        "lowConfidence": r.low_confidence,
        "flags": list(r.flags),
        "tags": list(r.tags),
        "umPerPx": r.um_per_px,
        "origin": r.origin,
        "size": list(r.size),
    }


def _state_payload(session: BankSession) -> dict:
    from anograft.bank import holdout as ho

    root = _ROOT or Path(".")
    held = ho.read_holdout(root)
    leaked = ho.violations([session.source(r.id) for r in session.rows()], held) if held else []
    return {
        "open": True,
        "root": root.as_posix(),
        "summary": session.summary_text(),
        "classes": session.classes(),
        "tags": session.tags(),
        "sortKeys": list(SORT_KEYS),
        "total": len(session.rows()),
        # T4 — 평가셋이 은행에 섞였는지. 지우는 건 사람이 정한다(자동 삭제 없음).
        "holdout": {"listed": len(held), "leaked": leaked},
        "directional": [{"cls": c, "r": r} for c, r in session.directional_classes()],
    }


# ---------------------------------------------------------------- 읽기


def _open(req: Request) -> ApiResult:
    global _SESSION, _ROOT
    root = str(req.json.get("root", "")).strip()
    if not root:
        return ApiResult(400, {"error": "보관함 폴더 경로가 필요합니다."})
    session = BankSession()
    try:
        session.load(root)
    except BankSessionError as exc:
        return ApiResult(400, {"error": str(exc)})
    with _LOCK:
        _SESSION = session
        _ROOT = Path(root)
    return ApiResult(200, _state_payload(session))


def _state(_req: Request) -> ApiResult:
    if _SESSION is None:
        return ApiResult(200, {"open": False})
    return ApiResult(200, _state_payload(_SESSION))


def _sources(req: Request) -> ApiResult:
    """필터·정렬은 Qt 탭과 **같은 순수 함수**가 한다."""
    session = _session()
    sort = req.get("sort", "id")
    if sort not in SORT_KEYS:
        return ApiResult(400, {"error": f"모르는 정렬입니다: {sort}", "sortKeys": list(SORT_KEYS)})
    rows = filter_rows(
        session.rows(),
        cls=req.get("class") or None,
        tag=req.get("tag") or None,
        only_low=req.get("low") == "1",
        only_estimated=req.get("estimated") == "1",
        text=req.get("q", ""),
        sort=sort,
        descending=req.get("desc") == "1",
    )
    return ApiResult(200, {"sources": [_row_json(r) for r in rows], "total": len(rows)})


def _image(req: Request) -> ApiResult:
    """소스 크롭 + 마스크 윤곽 PNG. `kind=tile|detail|mask`."""
    import cv2

    from anograft.preview import overlay_image

    session = _session()
    source_id = req.get("id")
    try:
        source = session.source(source_id)
    except BankSessionError as exc:
        return ApiResult(404, {"error": str(exc)})

    kind = req.get("kind", "tile")
    if kind == "mask":
        canvas = source.mask
    else:
        long_side = DETAIL_LONG_SIDE if kind == "detail" else TILE_LONG_SIDE
        canvas = overlay_image(source.image, source.mask, long_side=long_side)
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        return ApiResult(500, {"error": "PNG 로 만들지 못했습니다."})
    return ApiResult(
        200,
        body=buf.tobytes(),
        content_type="image/png",
        # 마스크를 다듬으면 그림이 바뀐다 — 보관함 타일은 캐시하지 않는다(검수와 다른 점).
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------- 쓰기


def _delete(req: Request) -> ApiResult:
    """소스 삭제 — `BankWriter.delete` 만 쓴다(파일을 직접 만지지 않는다).

    **클래스는 삭제해도 유지된다**(id 순서 = class id = 출력 `data.yaml`). 되돌릴 수 없으므로
    화면이 먼저 확인을 받고, 서버는 무엇을 지웠는지 그대로 돌려준다.
    """
    session = _session()
    ids = req.json.get("ids") or []
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        return ApiResult(400, {"error": "지울 소스 id 목록이 필요합니다."})
    if not ids:
        return ApiResult(400, {"error": "지울 것이 없습니다."})
    try:
        removed = session.delete(ids)
    except BankSessionError as exc:
        return ApiResult(400, {"error": str(exc)})
    return ApiResult(200, {"removed": removed, "state": _state_payload(session)})


def _serialized(handler: Handler) -> Handler:
    def wrapped(req: Request) -> ApiResult:
        with _LOCK:
            return handler(req)

    wrapped.__name__ = handler.__name__
    wrapped.__doc__ = handler.__doc__
    return wrapped


def ensure_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    register("/api/bank/state", _serialized(_state))
    register("/api/bank/sources", _serialized(_sources))
    register("/api/bank/image", _serialized(_image))
    register("/api/bank/open", _open, write=True)
    register("/api/bank/delete", _serialized(_delete), write=True)
    _REGISTERED = True

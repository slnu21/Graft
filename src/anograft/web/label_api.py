"""결함 표시(라벨) API — `anograft.labeling.LabelSession` 을 **감싸기만** 한다.

브러시 한 획마다 서버를 부른다: 프론트는 포인터를 따라 로컬 캔버스에 미리 그려 주고, 획이 끝나면
(`pointerup`) 점 목록을 한 번에 보낸다. 마스크 연산(스트로크·폴리곤·자동 선택·되돌리기)은 전부
파이썬이 하므로 **Qt 라벨 탭과 같은 결과**가 나오고, 프론트에는 그리기 로직이 없다(규약 그대로).

로컬 서버라 왕복이 수 ms 다 — 획 단위 전송이면 체감이 없다. 점 단위로 보내면 요청이 폭주하므로
**획이 끝날 때 한 번**이 규칙이다.
"""

from __future__ import annotations

import threading
from pathlib import Path

from anograft.labeling import LabelError, LabelSession, lighting_word
from anograft.web.api import ApiResult, Handler, Request, register

VIEW_LONG_SIDE = 1400

_LOCK = threading.Lock()
_SESSION: LabelSession | None = None
_REGISTERED = False


def reset() -> None:
    global _SESSION
    with _LOCK:
        _SESSION = None


def _session() -> LabelSession:
    if _SESSION is None or not _SESSION.loaded:
        raise LabelError("먼저 이미지를 여세요.")
    return _SESSION


def _points(raw) -> list[tuple[float, float]]:
    """`[[x, y], …]` 만 받는다 — 좌표는 **원본 픽셀 기준**(축소는 화면이 되돌려 보낸다)."""
    if not isinstance(raw, list):
        raise LabelError("점 목록이 필요합니다.")
    out: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise LabelError("점은 [x, y] 형태여야 합니다.")
        out.append((float(item[0]), float(item[1])))
    if not out:
        raise LabelError("점이 없습니다.")
    return out


def _state_payload(session: LabelSession) -> dict:
    stats = session.stats()
    h, w = session.shape
    return {
        "open": True,
        "path": str(session.path) if session.path else "",
        "width": w,
        "height": h,
        "canUndo": session.can_undo,
        "canRedo": session.can_redo,
        "stats": {
            "areaPx": stats.area_px,
            "areaRatio": stats.area_ratio,
            "components": stats.n_components,
            "lengthPx": stats.length_px,
            "contrast": stats.contrast,
            "lightingDeg": stats.lighting_deg,
            "lightingWord": lighting_word(stats.lighting_deg)
            if stats.lighting_deg is not None
            else "",
        },
    }


# ---------------------------------------------------------------- 읽기


def _state(_req: Request) -> ApiResult:
    if _SESSION is None or not _SESSION.loaded:
        return ApiResult(200, {"open": False})
    return ApiResult(200, _state_payload(_SESSION))


#: 칠한 영역 색(BGR) — 목업 토큰 `--mask: #D8294A` 와 같다.
MASK_BGR = (0x4A, 0x29, 0xD8)
MASK_ALPHA = 140


def _image(req: Request) -> ApiResult:
    """`kind=base` 원본 · `mask` 0/255 · `overlay` **투명 배경 + 빨강 반투명 RGBA**.

    화면은 `overlay` 를 원본 위에 그냥 얹는다. CSS 블렌드로 0/255 마스크를 물들이려 하면
    **검은 배경까지 곱해져 사진 전체가 물든다**(처음에 그렇게 했다) — 알파는 서버가 만드는 게 맞다.
    """
    import cv2
    import numpy as np

    session = _session()
    kind = req.get("kind", "base")
    if kind == "overlay":
        mask = session.mask
        if mask is None:
            return ApiResult(404, {"error": "마스크가 없습니다."})
        h, w = mask.shape[:2]
        canvas = np.zeros((h, w, 4), np.uint8)
        on = mask > 0
        canvas[on, 0], canvas[on, 1], canvas[on, 2] = MASK_BGR
        canvas[on, 3] = MASK_ALPHA
    else:
        canvas = session.mask if kind == "mask" else session.image
    if canvas is None:
        return ApiResult(404, {"error": "이미지가 없습니다."})
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        return ApiResult(500, {"error": "PNG 로 만들지 못했습니다."})
    return ApiResult(
        200, body=buf.tobytes(), content_type="image/png", headers={"Cache-Control": "no-store"}
    )


# ---------------------------------------------------------------- 쓰기


def _open(req: Request) -> ApiResult:
    global _SESSION
    path = str(req.json.get("path", "")).strip()
    if not path:
        return ApiResult(400, {"error": "이미지 경로가 필요합니다."})
    session = LabelSession()
    try:
        session.load_image(path)
        mask = str(req.json.get("mask", "")).strip()
        if mask:
            session.load_mask(mask)
    except (LabelError, OSError) as exc:
        return ApiResult(400, {"error": str(exc)})
    with _LOCK:
        _SESSION = session
    return ApiResult(200, _state_payload(session))


def _stroke(req: Request) -> ApiResult:
    """획 하나 — 브러시 또는 지우개. `radius` 는 원본 픽셀."""
    session = _session()
    try:
        points = _points(req.json.get("points"))
        radius = float(req.json.get("radius", 8))
        # `stroke` 는 되돌리기 지점을 스스로 찍지 않는다(Qt 캔버스가 눌림 시점에 부르는 구조) —
        # 획 단위로 보내는 웹에서는 **여기가 그 시점**이다.
        session.push_undo()
        session.stroke(points, radius, erase=bool(req.json.get("erase", False)))
    except (LabelError, TypeError, ValueError) as exc:
        return ApiResult(400, {"error": str(exc)})
    return ApiResult(200, _state_payload(session))


def _polygon(req: Request) -> ApiResult:
    session = _session()
    try:
        points = _points(req.json.get("points"))
        session.push_undo()
        session.fill_polygon(points, erase=bool(req.json.get("erase", False)))
    except (LabelError, TypeError, ValueError) as exc:
        return ApiResult(400, {"error": str(exc)})
    return ApiResult(200, _state_payload(session))


def _auto(req: Request) -> ApiResult:
    """박스 하나로 자동 선택 — `bank.mask_from_box` 와 **같은 추정기**(결과도 같다)."""
    session = _session()
    box = req.json.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return ApiResult(400, {"error": "박스는 [x, y, w, h] 형태여야 합니다."})
    try:
        session.push_undo()
        used = session.auto_select(
            tuple(int(v) for v in box), str(req.json.get("method", "grabcut"))
        )
    except (LabelError, ValueError) as exc:
        return ApiResult(400, {"error": str(exc)})
    payload = _state_payload(session)
    payload["method"] = used
    return ApiResult(200, payload)


def _undo(req: Request) -> ApiResult:
    session = _session()
    moved = session.redo() if req.json.get("redo") else session.undo()
    payload = _state_payload(session)
    payload["moved"] = moved
    return ApiResult(200, payload)


def _clear(_req: Request) -> ApiResult:
    import numpy as np

    session = _session()
    session.push_undo()
    session.set_mask(np.zeros(session.shape, np.uint8), tool="clear")
    return ApiResult(200, _state_payload(session))


def _save(req: Request) -> ApiResult:
    """은행에 소스로 저장 — `BankWriter` 를 지나므로 **평가셋(holdout)이면 여기서도 거부된다**(T4)."""
    session = _session()
    root = str(req.json.get("bank", "")).strip()
    cls = str(req.json.get("cls", "")).strip()
    if not root:
        return ApiResult(400, {"error": "저장할 보관함 폴더가 필요합니다."})
    tags = req.json.get("tags") or []
    um = req.json.get("umPerPx")
    try:
        added, warnings = session.save_to_bank(
            root,
            cls,
            tags=[str(t) for t in tags],
            um_per_px=float(um) if um not in (None, "") else None,
        )
    except (LabelError, OSError, ValueError) as exc:
        return ApiResult(400, {"error": str(exc)})
    return ApiResult(
        200,
        {
            "added": [{"id": a.source_id, "cls": a.cls, "areaPx": a.area_px} for a in added],
            "warnings": list(warnings),
            "bank": str(Path(root).as_posix()),
        },
    )


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
    register("/api/label/state", _serialized(_state))
    register("/api/label/image", _serialized(_image))
    register("/api/label/open", _open, write=True)
    register("/api/label/stroke", _serialized(_stroke), write=True)
    register("/api/label/polygon", _serialized(_polygon), write=True)
    register("/api/label/auto", _serialized(_auto), write=True)
    register("/api/label/undo", _serialized(_undo), write=True)
    register("/api/label/clear", _serialized(_clear), write=True)
    register("/api/label/save", _serialized(_save), write=True)
    _REGISTERED = True

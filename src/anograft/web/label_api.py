"""결함 표시(라벨) API — `anograft.labeling.LabelSession` 을 **감싸기만** 한다.

브러시 한 획마다 서버를 부른다: 프론트는 포인터를 따라 로컬 캔버스에 미리 그려 주고, 획이 끝나면
(`pointerup`) 점 목록을 한 번에 보낸다. 마스크 연산(스트로크·폴리곤·자동 선택·되돌리기)은 전부
파이썬이 하므로 **Qt 라벨 탭과 같은 결과**가 나오고, 프론트에는 그리기 로직이 없다(규약 그대로).

로컬 서버라 왕복이 수 ms 다 — 획 단위 전송이면 체감이 없다. 점 단위로 보내면 요청이 폭주하므로
**획이 끝날 때 한 번**이 규칙이다.

U7 인계 — 이 화면은 보관함(①)과 양쪽으로 손을 잡는다. ``edit-source`` 가 보관함 조각을 열고
(``_EDIT`` 에 대상 id 를 기억), ``update-source`` 가 다듬은 마스크를 **같은 id 에 덮어쓴다**.
새 조각을 더하는 ``save`` 와 라우트를 나눈 이유: 한 라우트가 상황에 따라 다른 일을 하면
"지금 무엇이 저장되는가"가 화면에서도 서버에서도 흐려진다(Qt 도 메서드를 나눠 두었다).
"""

from __future__ import annotations

import threading
from pathlib import Path

from anograft import recent
from anograft.bank.browse import BankSessionError
from anograft.core.channels import is_gray
from anograft.labeling import LabelError, LabelSession, lighting_word
from anograft.web.api import ApiResult, Handler, Request, register

VIEW_LONG_SIDE = 1400

#: 세션 락. **재진입 가능(RLock)** — 인계 핸들러가 보관함 쪽을 부르는 동안에도 자기 락을 들고 있다.
#: 락 순서는 언제나 **label → bank** 한 방향이라(보관함은 라벨을 부르지 않는다) 교착이 없다.
_LOCK = threading.RLock()
_SESSION: LabelSession | None = None
#: 보관함 조각을 다듬는 중이면 ``(보관함 경로, 조각 id)`` — Qt ``LabelTab.edit_target`` 에 해당.
_EDIT: tuple[str, str] | None = None
_REGISTERED = False


def reset() -> None:
    global _SESSION, _EDIT
    with _LOCK:
        _SESSION = None
        _EDIT = None


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
        # 보관함 조각을 다듬는 중인가 — 화면이 저장 카드를 "갱신"으로 바꾼다(Qt 는 버튼 문구를 바꾼다).
        "editTarget": {"root": _EDIT[0], "id": _EDIT[1]} if _EDIT else None,
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
    global _SESSION, _EDIT
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
        _EDIT = None  # 파일을 새로 열면 보관함 조각 편집은 끝난다
    recent.remember("image", path)
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
    """전부 지우기 — Qt 와 **같은 메서드**(``LabelSession.clear``)를 쓴다.

    예전엔 ``set_mask(zeros, tool="clear")`` 였는데, 그러면 ``tools_used`` 에 웹에만 있는 도구 이름이
    남아 ``mask_origin``·``edit_tool`` 이 Qt 와 갈린다(U7 에서 맞췄다). ``set_mask`` 가 안에서
    되돌리기 지점을 또 찍던 중복도 함께 사라진다.
    """
    session = _session()
    session.push_undo()
    session.clear()
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
    recent.remember("bank", root)
    # ② → ① 인계: 보관함 화면이 새 조각을 보게 한다(Qt ``bank_saved`` 신호).
    from anograft.web import bank_api

    shown = bank_api.bank_changed(root)
    return ApiResult(
        200,
        {
            "added": [{"id": a.source_id, "cls": a.cls, "areaPx": a.area_px} for a in added],
            "warnings": list(warnings),
            "bank": str(Path(root).as_posix()),
            # ① 이 지금 이 보관함을 보고 있는가 — 화면이 "보러 가기"를 권할지 정한다.
            "bankShown": shown,
        },
    )


# ---------------------------------------------------------------- 보관함 조각 다듬기 (U7)


def _edit_source(req: Request) -> ApiResult:
    """① 조각 "다듬기" → 그 크롭과 현재 마스크를 연다 (Qt ``LabelTab.begin_bank_edit``).

    ``gray`` 는 메타에 없어서 그림에서 되살린다(``core.channels.is_gray`` — Qt 도 같은 판정).
    """
    global _SESSION, _EDIT
    from anograft.web import bank_api

    source_id = str(req.json.get("id", "")).strip()
    root = str(req.json.get("root", "")).strip()
    if not source_id:
        return ApiResult(400, {"error": "다듬을 조각 id 가 필요합니다."})
    try:
        source, bank_root = bank_api.source_for_edit(root, source_id)
    except BankSessionError as exc:
        return ApiResult(400, {"error": str(exc)})
    session = LabelSession()
    try:
        session.set_image(source.image, is_gray(source.image), path=None)
        session.set_mask(source.mask)
    except LabelError as exc:
        return ApiResult(400, {"error": str(exc)})
    with _LOCK:
        _SESSION = session
        _EDIT = (bank_root, source_id)
    payload = _state_payload(session)
    payload["cls"] = source.cls
    return ApiResult(200, payload)


def _update_source(_req: Request) -> ApiResult:
    """다듬은 마스크를 같은 id 에 덮어쓴다 (Qt ``source_updated`` → ``BankTab.apply_mask``).

    새 조각을 더하지 않는다 — ``BankWriter.replace_mask`` 한 지점을 지나므로 ``mask_origin`` 은
    ``manual:<도구>`` 가 되고 추정 ``confidence`` 는 지워진다(사람이 손봤으므로).
    """
    from anograft.web import bank_api

    session = _session()
    if _EDIT is None:
        return ApiResult(400, {"error": "보관함 조각을 다듬는 중이 아닙니다."})
    root, source_id = _EDIT
    tool = session.edit_tool()
    assert session.mask is not None
    try:
        bank_state = bank_api.replace_mask(root, source_id, session.mask, tool=tool)
    except BankSessionError as exc:
        return ApiResult(400, {"error": str(exc)})
    session.dirty = False
    return ApiResult(
        200,
        {
            "id": source_id,
            "bank": root,
            "tool": tool,
            "state": _state_payload(session),
            "bankState": bank_state,
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
    register("/api/label/edit-source", _serialized(_edit_source), write=True)
    register("/api/label/update-source", _serialized(_update_source), write=True)
    _REGISTERED = True

"""검수 API — `anograft.review.ReviewSession` 을 **감싸기만** 한다.

규약(U2 가 세운 것): 여기서 판정하지 않는다. 필터·통계·히스토그램·힌트·정리본은 전부 `review.py` 의
순수 로직이 이미 하는 일이고, 이 모듈은 그 결과를 JSON·PNG 로 옮긴다. Qt 검수 탭과 **같은 함수**를
부르므로 두 화면의 판단이 갈릴 수 없다.

상태: 이 도구는 **로컬 1인용**이라 서버 프로세스가 세션 하나를 들고 있는다(`_SESSION`).
여러 사람이 붙는 그림은 v1.x 의 확인 게이트이고, 그때는 세션 키가 필요해진다.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from anograft.review import FILTER_LABELS, FILTERS, ReviewError, ReviewSession
from anograft.web.api import ApiResult, Handler, Request, register

#: 썸네일 긴 변(px). 그리드 타일이 176 이라 2배까지 선명하게.
THUMB_LONG_SIDE = 352
#: 상세 보기 긴 변 — 원본을 그대로 보내면 4K 가 오간다.
DETAIL_LONG_SIDE = 1400

_LOCK = threading.Lock()
_SESSION: ReviewSession | None = None
_ROOT: Path | None = None
_REGISTERED = False


def reset() -> None:
    """테스트가 서버 상태를 지울 때."""
    global _SESSION, _ROOT
    with _LOCK:
        _SESSION = None
        _ROOT = None


def _session() -> ReviewSession:
    if _SESSION is None:
        raise ReviewError("먼저 출력 폴더를 여세요.")
    return _SESSION


def _item_json(it: Any) -> dict:
    """목록 한 줄 — 그리드가 그리는 데 필요한 것만(사이드카 지연 로드를 강제하지 않는다)."""
    return {
        # manifest 의 정상(normal) 행은 index 가 비어 있다 — 화면이 행을 구분할 안정된 키가 따로 필요하다.
        "key": it.index or it.image or f"{it.status}:{it.target}",
        "index": it.index,
        "status": it.status,
        "verdict": it.verdict,
        "note": it.note,
        "classes": list(it.classes),
        "areaPx": it.area_px,
        "blend": it.blend,
        "fallback": it.fallback,
        "reason": it.reason,
        "target": it.target,
        "warnings": list(it.warnings),
    }


# ---------------------------------------------------------------- 읽기


def _open(req: Request) -> ApiResult:
    """출력 폴더를 연다(POST — 서버 상태가 바뀐다)."""
    global _SESSION, _ROOT
    root = str(req.json.get("root", "")).strip()
    if not root:
        return ApiResult(400, {"error": "출력 폴더 경로가 필요합니다."})
    session = ReviewSession()
    try:
        session.load(root)
    except ReviewError as exc:
        return ApiResult(400, {"error": str(exc)})
    with _LOCK:
        _SESSION = session
        _ROOT = Path(root)
    return ApiResult(200, _state_payload(session))


def _state(_req: Request) -> ApiResult:
    """지금 열려 있는 세션 요약 — 새로고침해도 화면이 돌아온다."""
    if _SESSION is None:
        return ApiResult(200, {"open": False})
    return ApiResult(200, _state_payload(_SESSION))


def _state_payload(session: ReviewSession) -> dict:
    counts = session.counts()
    hints = [
        {"cls": h.cls, "synthetic": h.synthetic, "real": h.real, "n": h.n}
        for h in session.contrast_hints()
    ]
    return {
        "open": True,
        "root": str(_ROOT or session.root),
        "summary": session.summary_text(),
        "counts": counts,
        "classes": session.classes(),
        # 라벨까지 함께 — 프론트에 용어 사본을 두지 않는다(사전 §3.6)
        "filters": [{"id": f, "label": FILTER_LABELS.get(f, f)} for f in FILTERS],
        "contrastHints": hints,
        "directional": session.directional_classes(),
    }


def _items(req: Request) -> ApiResult:
    """필터에 걸린 목록. `class` 는 빈 문자열이면 전체."""
    session = _session()
    which = req.get("filter", "all")
    if which not in FILTERS:
        return ApiResult(400, {"error": f"모르는 필터입니다: {which}", "filters": list(FILTERS)})
    cls = req.get("class") or None
    items = session.filtered(which, cls)
    return ApiResult(200, {"items": [_item_json(it) for it in items], "total": len(items)})


def _item(req: Request) -> ApiResult:
    """한 장의 상세 — 사이드카를 이때 읽는다(지연 로드)."""
    session = _session()
    index = req.get("index")
    try:
        it = session.item(index)
    except ReviewError as exc:
        return ApiResult(404, {"error": str(exc)})
    data = _item_json(it)
    data.update(
        {
            "image": it.image,
            "mask": it.mask,
            "sidecar": it.sidecar,
            "sourceIds": list(it.source_ids),
            "instances": it.instances,
            "appearance": it.appearance,
            "appearanceClasses": it.appearance_cls,
        }
    )
    return ApiResult(200, data)


def _image(req: Request) -> ApiResult:
    """합성 이미지(+GT 윤곽) PNG. `kind=thumb|detail|mask`, 없으면 thumb."""
    import cv2

    from anograft.io.imgio import ImageReadError, read_image, read_mask
    from anograft.preview import overlay_image

    session = _session()
    index = req.get("index")
    kind = req.get("kind", "thumb")
    try:
        it = session.item(index)
    except ReviewError as exc:
        return ApiResult(404, {"error": str(exc)})

    root = session.root
    try:
        image, _gray = read_image(root / it.image)
    except (ImageReadError, OSError) as exc:
        return ApiResult(404, {"error": f"이미지를 읽지 못했습니다: {exc}"})

    mask = None
    if it.mask:
        try:
            mask = read_mask(root / it.mask)
        except (ImageReadError, OSError):
            mask = None  # 마스크가 없어도 원본은 보여 준다(fail-soft)

    if kind == "mask":
        if mask is None:
            return ApiResult(404, {"error": "정답 영역 파일이 없습니다."})
        canvas = mask
    else:
        long_side = DETAIL_LONG_SIDE if kind == "detail" else THUMB_LONG_SIDE
        long_side = min(long_side, max(64, req.int_of("max", long_side)))
        canvas = overlay_image(image, mask, long_side=long_side)

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        return ApiResult(500, {"error": "PNG 로 만들지 못했습니다."})
    # 같은 index 의 그림은 판정이 바뀌어도 안 바뀐다 — 그리드 스크롤이 매번 다시 받지 않게.
    return ApiResult(
        200,
        body=buf.tobytes(),
        content_type="image/png",
        headers={"Cache-Control": "private, max-age=300"},
    )


def _histogram(req: Request) -> ApiResult:
    """합성 vs 실제 분포 — 막대만 그리면 되게 숫자로 준다(판정은 서버가 이미 했다)."""
    session = _session()
    key = req.get("key", "area")
    cls = req.get("class") or None
    bins = req.int_of("bins", 12)
    hist = (
        session.distribution_by_class(key, cls, bins=bins)
        if cls
        else session.distribution(key, bins=bins)
    )
    return ApiResult(
        200,
        {
            "key": key,
            "cls": cls or "",
            "edges": list(hist.edges),
            "synthetic": list(hist.a),
            "real": list(hist.b),
            "log": hist.log,
            "realLabel": session.real_label(),
            "classOptions": session.class_options(key),
        },
    )


# ---------------------------------------------------------------- 쓰기


def _verdict(req: Request) -> ApiResult:
    """채택/반려/판정 취소 → `review.csv` 에 바로 저장(Qt 탭과 같은 파일)."""
    session = _session()
    index = str(req.json.get("index", ""))
    verdict = str(req.json.get("verdict", ""))
    if verdict not in {"accept", "reject", ""}:
        return ApiResult(400, {"error": "판정은 accept · reject · 빈 문자열(취소)만 됩니다."})
    note = req.json.get("note")
    try:
        it = session.set_verdict(index, verdict, note if note is None else str(note))
    except ReviewError as exc:
        return ApiResult(404, {"error": str(exc)})
    path = session.save()
    return ApiResult(200, {"item": _item_json(it), "counts": session.counts(), "saved": str(path)})


def _prune(req: Request) -> ApiResult:
    """반려를 뺀 정리본 — `io.prune.prune_dataset` 을 그대로 부른다."""
    session = _session()
    out = str(req.json.get("out", "")).strip()
    if not out:
        return ApiResult(400, {"error": "정리본을 만들 폴더가 필요합니다."})
    drop_unreviewed = bool(req.json.get("dropUnreviewed", False))
    summary = session.prune(out, drop_unreviewed=drop_unreviewed)
    return ApiResult(
        200,
        {
            "out": str(summary.out),
            "kept": summary.kept,
            "dropped": summary.dropped,
            "normals": summary.normals,
            "skipped": summary.skipped,
            "files": summary.files,
            "warnings": list(summary.warnings),
        },
    )


def _report(req: Request) -> ApiResult:
    """검수 리포트 HTML — 경로만 돌려주고 파일은 디스크에 남긴다(사람이 공유한다)."""
    session = _session()
    path = req.json.get("path")
    written = session.write_report(path)
    return ApiResult(200, {"path": str(written)})


def _serialized(handler: Handler) -> Handler:
    """요청을 직렬화한다 — `ReviewSession` 은 지연 로드 캐시를 들고 있어 스레드 안전하지 않다.

    로컬 1인용 도구라 이 단순함이 맞다(`ThreadingHTTPServer` 는 브라우저가 여러 썸네일을 동시에
    받으려 할 때 필요할 뿐이고, 그 순차화 비용은 수십 ms 다).
    """

    def wrapped(req: Request) -> ApiResult:
        with _LOCK:
            return handler(req)

    wrapped.__name__ = handler.__name__
    wrapped.__doc__ = handler.__doc__
    return wrapped


def ensure_registered() -> None:
    """라우트를 한 번만 등록한다(`api.handle` 이 부른다)."""
    global _REGISTERED
    if _REGISTERED:
        return
    register("/api/review/state", _serialized(_state))
    register("/api/review/items", _serialized(_items))
    register("/api/review/item", _serialized(_item))
    register("/api/review/image", _serialized(_image))
    register("/api/review/histogram", _serialized(_histogram))
    register("/api/review/open", _open, write=True)  # 자체 락(_LOCK)으로 세션을 바꾼다
    register("/api/review/verdict", _serialized(_verdict), write=True)
    register("/api/review/prune", _serialized(_prune), write=True)
    register("/api/review/report", _serialized(_report), write=True)
    _REGISTERED = True

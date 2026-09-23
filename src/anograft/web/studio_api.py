"""미리보기 API (U5a) — `anograft.studio` 를 **감싸기만** 한다.

Qt 스튜디오 탭과 같은 것을 부른다: 상태는 `StudioSession`, 합성은 `jobs.run_preview`, 진단 문장은
`studio.diagnose`, 프리셋 카드·썸네일은 `studio.gallery`. 여기서 새로 계산하는 것은 **PNG 인코딩뿐**이다.

설계 판단(TASKS U5, 확정):

* **실시간 스트리밍은 WebSocket·SSE 없이 단순 요청/응답.** 파라미터가 바뀌면 프론트가 `preview` 를 POST 하고
  결과 메타를 받는다(그림은 `image` 로 뒤따라 받는다 — 브라우저가 `<img>` 로 알아서 받게 두는 게 낫다).
  stdlib `http.server` 로 WebSocket 을 직접 짜는 건 과하고, 로컬 왕복은 수 ms 다.
* **"최신만 살리기"(Qt `LatestOnlyQueue`)의 웹판은 프론트의 `AbortController`** 다. 서버는 큐를 두지 않고
  락으로 직렬화한다 — 요청을 중간에 끊어도 계산은 끝까지 가지만(그게 stdlib 서버의 한계다) 1024 축소본
  한 장은 수십~수백 ms 라 다음 요청이 밀리지 않는다.
* **미리보기는 긴 변 1024 축소본** — 배치 좌표·Poisson 결과가 원본 해상도 `run` 과 다르다. 그래서 응답이
  `scale` 을 늘 실어 주고 화면이 그걸 상시 표시한다(규약).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from anograft import runner
from anograft.core import recipe as R
from anograft.core.channels import promote_to_bgr
from anograft.studio import diagnose, gallery
from anograft.studio.jobs import (
    KIND_PREVIEW,
    PreviewJob,
    PreviewResult,
    gt_overlay_bgra,
    make_thumb,
    roi_overlay_bgra,
    run_preview,
)
from anograft.studio.session import DEFAULT_PRESET, SessionError, StudioSession, default_recipe
from anograft.web.api import ApiResult, Handler, Request, register

#: 대상 목록 썸네일 긴 변(px) — 목업의 184px 열.
THUMB_LONG_SIDE = 192

_LOCK = threading.Lock()
_SESSION: StudioSession | None = None
_PREVIEW: PreviewResult | None = None
_VERSION = 0  # 미리보기가 갈릴 때마다 +1 — 프론트가 그림 URL 의 캐시를 깬다
_THUMBS: dict[str, bytes] = {}
_REGISTERED = False


def reset() -> None:
    """테스트·재시작용 — 세션과 캐시를 버린다."""
    global _SESSION, _PREVIEW, _VERSION
    with _LOCK:
        _SESSION = None
        _PREVIEW = None
        _VERSION = 0
        _THUMBS.clear()


def _session() -> StudioSession:
    if _SESSION is None:
        raise SessionError("먼저 레시피(또는 보관함·바탕 이미지)를 여세요.")
    return _SESSION


# ---------------------------------------------------------------- 상태


def _stage_rows(recipe: R.Recipe) -> list[dict]:
    """지금 파이프라인의 단계 → method 표. 한국어 라벨은 `core/help.py` 가 원천이라 **서버가 실어 준다**
    (프론트에 용어 사전 사본을 두면 둘이 갈린다)."""
    return [
        {"stage": r.stage, "label": r.label, "method": r.method, "methodLabel": r.method_label}
        for r in gallery.stage_rows(recipe.to_dict().get("pipeline", {}))
    ]


def _state_payload(session: StudioSession) -> dict:
    r = session.recipe
    prep = session.prepared
    return {
        "open": True,
        "recipe": {
            "name": r.name,
            "preset": r.pipeline.preset or "",
            "seed": r.seed,
            "bank": r.inputs.bank_key(),
            "targets": r.inputs.targets.as_posix(),
            "out": r.output.root.as_posix(),
            "count": r.output.count,
            "writer": r.output.writer.format,
            "defectsPerImage": list(r.output.defects_per_image),
        },
        "recipePath": session.recipe_path.as_posix() if session.recipe_path else "",
        "summary": dict(session.summary()),
        "stages": _stage_rows(r),
        "presets": list(R.preset_names()),
        "warnings": list(session.warnings),
        "pathNotes": list(session.path_notes),
        "longSide": session.long_side,
        "targetIndex": session.target_index,
        "targets": [
            {"index": i, "name": p.name, "path": p.as_posix()}
            for i, p in enumerate(session.targets)
        ],
        "bankName": prep.bank.name if prep is not None else "",
        "bankN": len(prep.bank) if prep is not None else 0,
        "classes": [] if prep is None or r.bankless else list(prep.bank.classes),
        "pipelineHash": prep.pipeline_hash if prep is not None else "",
        "runCommand": session.run_command(),
        "version": _VERSION,
    }


def _state(_req: Request) -> ApiResult:
    if _SESSION is None:
        return ApiResult(200, {"open": False, "presets": list(R.preset_names())})
    return ApiResult(200, _state_payload(_SESSION))


def _presets(_req: Request) -> ApiResult:
    """프리셋 카드 문안 — Qt 갤러리(v0.9 ⑱)와 **같은 `gallery.preset_cards()`**."""
    cards = [
        {
            "name": c.name,
            "title": c.title,
            "summary": c.summary,
            "useFor": c.use_for,
            "avoid": c.avoid,
            "evidence": c.evidence,
            "stages": [{"label": s[0], "method": s[1], "methodLabel": s[2]} for s in c.stages],
        }
        for c in gallery.preset_cards()
    ]
    current = _SESSION.recipe.pipeline.preset if _SESSION is not None else DEFAULT_PRESET
    return ApiResult(200, {"presets": cards, "current": current or ""})


# ---------------------------------------------------------------- 그림


def _png(canvas: np.ndarray) -> ApiResult:
    import cv2

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        return ApiResult(500, {"error": "PNG 로 만들지 못했습니다."})
    return ApiResult(
        200,
        body=buf.tobytes(),
        content_type="image/png",
        # 미리보기는 파라미터 한 번에 바뀐다 — 캐시는 `v` 로만 깨고 저장은 하지 않는다.
        headers={"Cache-Control": "no-store"},
    )


def _image(req: Request) -> ApiResult:
    """`kind=base|synth|gt|roi` = 마지막 미리보기 · `thumb` = 대상 목록 썸네일(`index`)."""
    session = _session()
    kind = req.get("kind", "synth")
    if kind == "thumb":
        i = req.int_of("index", -1)
        targets = session.targets
        if not (0 <= i < len(targets)):
            return ApiResult(404, {"error": f"그런 바탕 이미지가 없습니다: {i}"})
        path = targets[i].as_posix()
        if path not in _THUMBS:
            thumb = make_thumb(PreviewJob("thumb", 0, target=targets[i]), THUMB_LONG_SIDE)
            res = _png(thumb.image)
            if res.body is None:
                return res
            _THUMBS[path] = res.body
        return ApiResult(
            200,
            body=_THUMBS[path],
            content_type="image/png",
            headers={"Cache-Control": "max-age=300"},
        )

    res = _PREVIEW
    if res is None:
        return ApiResult(404, {"error": "아직 미리보기를 만들지 않았습니다."})
    r = res.result
    if kind == "base":
        return _png(promote_to_bgr(res.target.image))
    if kind == "synth":
        if r.status != "ok":
            return ApiResult(404, {"error": r.reason or "이 바탕에는 결함을 붙이지 못했습니다."})
        return _png(promote_to_bgr(r.image))
    if kind == "gt":
        if r.status != "ok" or not r.gt_mask.any():
            return ApiResult(404, {"error": "정답 영역이 없습니다."})
        return _png(gt_overlay_bgra(r.gt_mask))
    if kind == "roi":
        roi = diagnose.roi_of(res.steps)
        if roi is None or not roi.any():
            return ApiResult(404, {"error": "붙일 수 있는 영역이 없습니다."})
        return _png(roi_overlay_bgra(roi))
    return ApiResult(400, {"error": f"모르는 그림 종류입니다: {kind}"})


#: 갤러리 타일 표시 크기(px). **합성은 미리보기와 같은 축척에서** 하고 여기까지 줄이기만 한다.
PRESET_TILE_PX = 256


def _preset_image(req: Request) -> ApiResult:
    """프리셋 하나를 지금 바탕에 적용한 썸네일 — Qt 갤러리와 같은 `gallery.render_preset_thumbs`.

    **합성은 미리보기 축척(긴 변 1024)에서 하고 타일 크기로 줄인다.** Qt 갤러리처럼 256 에서 바로 합성하면
    패치가 줄어든 ROI 에 못 들어가 죄다 회색이 된다(metal_nut 링 ROI 에서 실제로 그랬다: "ROI 최대 폭 32px
    vs 패치 52×58"). 고르는 사람이 보는 것은 **그 프리셋으로 실제로 나올 그림**이어야 한다.

    돌 수 없는 프리셋(보관함 없음·배치 실패)은 404 로 답하고 화면이 빈 칸을 둔다(fail-soft).
    """
    from anograft.preview import fit_long_side

    session = _session()
    name = req.get("name")
    target = session.target
    if session.prepared is None or target is None:
        return ApiResult(404, {"error": "바탕 이미지가 준비되지 않았습니다."})
    if name not in R.preset_names():
        return ApiResult(404, {"error": f"그런 프리셋이 없습니다: {name}"})
    thumbs = gallery.render_preset_thumbs(
        session.prepared,
        session.recipe,
        target,
        [name],
        long_side=session.long_side or gallery.THUMB_LONG_SIDE,
    )
    image = thumbs.get(name)
    if image is None:
        return ApiResult(404, {"error": "이 입력으로는 이 프리셋을 돌릴 수 없습니다."})
    return _png(fit_long_side(promote_to_bgr(image), PRESET_TILE_PX))


# ---------------------------------------------------------------- 쓰기


def _open(req: Request) -> ApiResult:
    """레시피 파일로, 또는 보관함·바탕 경로로 연다. 은행 로드는 여기서 **동기**로(로컬 1인용)."""
    global _SESSION, _PREVIEW
    recipe_path = str(req.json.get("recipe", "")).strip()
    bank = str(req.json.get("bank", "")).strip()
    targets = str(req.json.get("targets", "")).strip()
    session = StudioSession()
    try:
        if recipe_path:
            session.load(recipe_path)
            if bank or targets:
                session.set_paths(bank=bank or None, targets=targets or None)
        elif targets:
            session.replace_recipe(default_recipe(bank=bank, targets=targets))
        else:
            return ApiResult(400, {"error": "레시피 파일 또는 바탕 이미지 경로가 필요합니다."})
        session.prepare_now()
    except SessionError as exc:
        return ApiResult(400, {"error": str(exc)})
    with _LOCK:
        _SESSION = session
        _PREVIEW = None
        _THUMBS.clear()
    return ApiResult(200, _state_payload(session))


def _preview_payload(session: StudioSession, res: PreviewResult) -> dict:
    r = res.result
    prep = session.prepared
    roi = diagnose.roi_of(res.steps)
    fit = diagnose.fit_for_preview(prep, session.recipe, roi, res.scale, res.target.path.name)
    low = diagnose.low_confidence_sources(prep, r)
    flipped = diagnose.flipped(prep, r)
    h, w = res.target.image.shape[:2]
    return {
        "version": _VERSION,
        "status": r.status,
        "reason": r.reason,
        "elapsedMs": round(res.elapsed_s * 1000, 1),
        "scale": res.scale,
        "width": w,
        "height": h,
        "longSide": session.long_side,
        "target": {"index": session.target_index, "name": res.target.path.name},
        "variant": res.job.index,
        "seed": session.recipe.seed,
        "defects": [
            {"cls": d.cls, "sourceId": d.source_id, "blend": d.blend}
            for d in diagnose.defect_lines(r)
        ],
        "instances": [
            {"cls": i.cls, "classId": i.class_id, "bbox": list(i.bbox), "areaPx": i.area_px}
            for i in r.instances
        ],
        # prepare 경고 + 결과의 fail-soft 경고 + 배치 가능성 진단 — Qt 파이프라인 카드가 받는 그 목록.
        "warnings": [*session.warnings, *r.warnings, *diagnose.fit_warnings(fit)],
        "lowConfidence": low,
        "flipped": [
            {"n": i + 1, "cls": r.instances[i].cls} for i in flipped if i < len(r.instances)
        ],
        "hasGt": bool(r.status == "ok" and r.gt_mask.any()),
        "hasRoi": bool(roi is not None and roi.any()),
    }


def _preview(req: Request) -> ApiResult:
    """합성 한 장. `targetIndex`·`variant`·`longSide` 는 주면 세션 선택을 먼저 바꾼다."""
    global _PREVIEW, _VERSION
    session = _session()
    if session.prepared is None:
        return ApiResult(400, {"error": "보관함·바탕 이미지가 준비되지 않았습니다."})
    if "longSide" in req.json:
        session.set_long_side(int(req.json["longSide"]))
    if "targetIndex" in req.json:
        i = int(req.json["targetIndex"])
        if not (0 <= i < len(session.targets)):
            return ApiResult(404, {"error": f"그런 바탕 이미지가 없습니다: {i}"})
        session.select_target(i)
    variant = int(req.json.get("variant", session.variant_index))
    target = session.target
    if target is None:
        return ApiResult(400, {"error": "바탕 이미지가 없습니다."})
    job = PreviewJob(
        KIND_PREVIEW,
        session.generation,
        target=target,
        index=max(0, variant),
        long_side=session.long_side,
    )
    try:
        res = run_preview(session.prepared, job)
    except (runner.PrepareError, ValueError, OSError) as exc:
        return ApiResult(400, {"error": f"미리보기 실패: {exc}"})
    _PREVIEW = res
    _VERSION += 1
    return ApiResult(200, _preview_payload(session, res))


def _edit(req: Request, mutate: Any) -> ApiResult:
    """레시피 편집 하나 — 실패하면 **이전 레시피를 유지**하고 한국어 메시지를 돌려준다(GUI 와 같은 규칙)."""
    global _PREVIEW
    session = _session()
    try:
        mutate(session)
    except SessionError as exc:
        return ApiResult(400, {"error": str(exc)})
    _PREVIEW = None  # 파라미터가 바뀌었으니 지난 그림은 버린다
    return ApiResult(200, _state_payload(session))


def _preset(req: Request) -> ApiResult:
    name = str(req.json.get("name", "")).strip()
    if name not in R.preset_names():
        return ApiResult(400, {"error": f"그런 프리셋이 없습니다: {name}"})
    return _edit(req, lambda s: s.set_preset(name))


def _seed(req: Request) -> ApiResult:
    try:
        seed = int(req.json.get("seed"))
    except (TypeError, ValueError):
        return ApiResult(400, {"error": "시드는 0 이상의 정수입니다."})
    return _edit(req, lambda s: s.set_seed(seed))


def _save(req: Request) -> ApiResult:
    """레시피 YAML 저장 — 일괄 생성(`anograft run`)으로 넘기는 손잡이."""
    session = _session()
    path = str(req.json.get("path", "")).strip()
    if not path:
        return ApiResult(400, {"error": "저장할 레시피 경로가 필요합니다."})
    try:
        saved = session.save(path)
    except OSError as exc:
        return ApiResult(400, {"error": f"레시피를 저장하지 못했습니다: {exc}"})
    return ApiResult(200, {"path": Path(saved).as_posix(), "runCommand": session.run_command()})


def _serialized(handler: Handler) -> Handler:
    """`ThreadingHTTPServer` 이므로 세션을 만지는 핸들러는 직렬화한다(`StudioSession` 은 스레드 안전하지 않다)."""

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
    register("/api/studio/state", _serialized(_state))
    register("/api/studio/presets", _serialized(_presets))
    register("/api/studio/image", _serialized(_image))
    register("/api/studio/preset-image", _serialized(_preset_image))
    register("/api/studio/open", _open, write=True)
    register("/api/studio/preview", _serialized(_preview), write=True)
    register("/api/studio/preset", _serialized(_preset), write=True)
    register("/api/studio/seed", _serialized(_seed), write=True)
    register("/api/studio/save", _serialized(_save), write=True)
    _REGISTERED = True

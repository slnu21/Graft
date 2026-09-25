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

from anograft import recent, runner
from anograft.core import recipe as R
from anograft.core.channels import promote_to_bgr
from anograft.studio import cards, diagnose, gallery
from anograft.studio.jobs import (
    KIND_PREVIEW,
    PreviewJob,
    PreviewResult,
    gt_overlay_bgra,
    make_thumb,
    roi_overlay_bgra,
    run_preview,
)
from anograft.studio.params import FieldSpec, coerce, spin_step
from anograft.studio.session import DEFAULT_PRESET, SessionError, StudioSession, default_recipe
from anograft.web.api import ApiResult, Handler, Request, register

#: 대상 목록 썸네일 긴 변(px) — 목업의 184px 열.
THUMB_LONG_SIDE = 192
#: 카드 안 단계별 중간 결과 썸네일(px).
CARD_THUMB_PX = 132
#: 시드 변형 그리드 한 칸의 **표시** 크기(px). 합성은 미리보기 축척에서 하고 여기까지 줄인다
#: — 작은 축척에서 합성하면 줄어든 ROI 에 패치가 못 들어가 "이 시드는 안 나온다"는 거짓말이 된다
#: (프리셋 썸네일에서 겪은 것과 같은 함정, U5a).
VARIANT_TILE_PX = 320

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
    if recipe_path:
        recent.remember("recipe", recipe_path)
    if bank:
        recent.remember("bank", bank)
    return ApiResult(200, _state_payload(session))


def _preview_payload(session: StudioSession, res: PreviewResult) -> dict:
    r = res.result
    prep = session.prepared
    roi = diagnose.roi_of(res.steps)
    fit = diagnose.fit_for_preview(prep, session.recipe, roi, res.scale, res.target.path.name)
    warnings = [*session.warnings, *r.warnings, *diagnose.fit_warnings(fit)]
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
        "warnings": warnings,
        # 카드별로 나눠 둔 것(U5b) — 고르는 규칙은 `cards.stage_warnings`(Qt 카드와 같다).
        # `roi:` 는 자기 카드가 없어 배치 카드가 받는다. 프론트가 접두를 다시 해석하지 않게 서버가 나눈다.
        "stageWarnings": {
            s: w for s in (*cards.STAGE_ORDER, "roi") if (w := cards.stage_warnings(s, warnings))
        },
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
    recent.remember("recipe", Path(saved).as_posix())
    return ApiResult(200, {"path": Path(saved).as_posix(), "runCommand": session.run_command()})


def _to_batch(req: Request) -> ApiResult:
    """③ → ④ "일괄 생성으로 보내기" — **저장하고** 넘긴다 (U7).

    Qt 는 ``Recipe`` 객체를 그대로 탭에 넘기지만(파일이 없어도 된다), 웹에서는 한 번 저장한다.
    이유는 화면 밖에 있다: 일괄 생성 화면이 보여 주는 CLI 한 줄(``anograft run <레시피>``)을
    사람이 나중에 그대로 돌리고, 사이드카의 ``pipeline_hash`` 로 "같은 합성인가"를 대조한다 —
    저장하지 않으면 그 한 줄이 없는 파일을 가리키는 거짓말이 된다.

    경로는 주면 그곳, 안 주면 열었던 레시피 자리. 둘 다 없으면 어디에 둘지 사람이 정해야 한다.
    """
    from anograft.batch import BatchError
    from anograft.web import batch_api

    session = _session()
    path = str(req.json.get("path", "")).strip() or (
        session.recipe_path.as_posix() if session.recipe_path else ""
    )
    if not path:
        return ApiResult(400, {"error": "레시피를 저장할 경로가 필요합니다."})
    try:
        saved = session.save(path)
    except OSError as exc:
        return ApiResult(400, {"error": f"레시피를 저장하지 못했습니다: {exc}"})
    recent.remember("recipe", Path(saved).as_posix())
    try:
        # 파일을 다시 읽지 않고 **메모리의 레시피**를 넘긴다 — 방금 쓴 것과 같은 객체라
        # 경로 해석(`resolve_recipe_paths`)을 한 번 더 태워 어긋날 일이 없다.
        state = batch_api.adopt(session.recipe, Path(saved))
    except BatchError as exc:  # 돌고 있는 중이면 거절 — 설정을 발밑에서 바꾸지 않는다
        return ApiResult(409, {"error": str(exc)})
    return ApiResult(
        200,
        {"path": Path(saved).as_posix(), "runCommand": session.run_command(), "batch": state},
    )


# ---------------------------------------------------------------- 파이프라인 카드(U5b)


def _field_json(f: FieldSpec) -> dict:
    """`FieldSpec` 하나 → 폼이 그릴 수 있는 JSON. **라벨·툴팁도 서버가 만든다**(문안의 한 원천은
    `core/help.py` — 프론트가 라벨 규칙을 다시 쓰면 Qt 폼과 갈린다)."""
    return {
        "name": f.name,
        "kind": f.kind,
        "value": f.value,
        "optional": f.optional,
        "enabled": f.enabled,
        "choices": list(f.choices),
        "lo": f.lo,
        "hi": f.hi,
        "loOpen": f.lo_open,
        "hiOpen": f.hi_open,
        "onDefault": f.on_default,
        "title": f.title,
        "tooltip": f.tooltip,
        "desc": f.desc,
        "advanced": f.advanced,
        "baseline": f.baseline if f.has_baseline else None,
        "hasBaseline": f.has_baseline,
        "modified": f.modified,
        "step": spin_step(f),
        # 좁은 실수 범위(세기 0..1 · 임계 −1..1)는 슬라이더를 함께 — Qt 폼과 같은 규칙
        "slider": bool(
            f.kind == "float" and f.lo is not None and f.hi is not None and f.hi - f.lo <= 10.0
        ),
    }


def _card_json(m: cards.StageCardModel) -> dict:
    return {
        "stage": m.stage,
        "no": m.no,
        "label": m.label,
        "tip": m.tip,
        "method": m.method,
        "methodLabel": m.method_label,
        "summary": m.summary,
        "modified": m.modified,
        "methods": [
            {
                "method": c.method,
                "label": c.label,
                "usable": c.usable,
                "reason": c.reason,
                "summary": c.summary,
            }
            for c in m.methods
        ],
        "fields": [_field_json(f) for f in m.fields],
    }


def _per_class_payload(session: StudioSession) -> dict:
    """기하 카드의 `per_class` 표 — 행은 **은행 클래스 ∪ 레시피에 이미 있는 키**(규칙은 `studio.cards`)."""
    prep = session.prepared
    classes = [] if prep is None or session.recipe.bankless else list(prep.bank.classes)
    rows = cards.per_class_rows(session.recipe.pipeline.geometry, classes)
    return {
        "classes": classes,
        "flipChoices": [{"label": lab, "value": val} for lab, val in cards.FLIP_CHOICES],
        "rows": [
            {
                "cls": r.cls,
                "on": r.on,
                "rotate": list(r.rotate),
                "flip": r.flip,
                "scale": list(r.scale) if r.scale else None,
            }
            for r in rows
        ],
        "text": cards.per_class_text(session.recipe.pipeline.geometry),
    }


def _cards_payload(session: StudioSession) -> dict:
    models = cards.stage_cards(session.recipe)
    return {
        "cards": [_card_json(m) for m in models],
        "perClass": _per_class_payload(session),
        "longSides": [{"label": lab, "px": px} for lab, px in cards.LONG_SIDES],
        "modified": sum(m.modified for m in models),
        "version": _VERSION,
    }


def _cards(_req: Request) -> ApiResult:
    return ApiResult(200, _cards_payload(_session()))


def _spec_for(session: StudioSession, stage: str, name: str) -> FieldSpec | None:
    return next((f for f in cards.stage_card(session.recipe, stage).fields if f.name == name), None)


def _edit_result(session: StudioSession) -> ApiResult:
    return ApiResult(200, {"state": _state_payload(session), "cards": _cards_payload(session)})


def _known_stage(stage: str) -> bool:
    return stage in {*cards.STAGE_ORDER, "roi"}


def _field(req: Request) -> ApiResult:
    """필드 하나. 값은 **그 필드의 스펙으로 정리**해서(`params.coerce`) 넣는다 — Qt 폼과 같은 변환이다.

    재검증에 실패하면 **이전 레시피를 유지**하고 어느 필드인지 함께 돌려준다(화면이 그 행을 강조한다).
    """
    global _PREVIEW
    session = _session()
    stage = str(req.json.get("stage", ""))
    name = str(req.json.get("name", ""))
    if not _known_stage(stage):
        return ApiResult(400, {"error": f"모르는 스테이지입니다: {stage}"})
    spec = _spec_for(session, stage, name)
    if spec is None:
        return ApiResult(400, {"error": f"이 알고리즘에 없는 설정입니다: {stage}.{name}"})
    raw = req.json.get("value")
    try:
        value = None if raw is None else coerce(spec, raw)
    except (TypeError, ValueError) as exc:
        return ApiResult(
            400, {"error": f"값을 읽지 못했습니다: {exc}", "stage": stage, "name": name}
        )
    try:
        session.set_stage_field(stage, name, value)
    except SessionError as exc:
        return ApiResult(400, {"error": str(exc), "stage": stage, "name": name})
    _PREVIEW = None
    return _edit_result(session)


def _method(req: Request) -> ApiResult:
    """스테이지 알고리즘 교체 — 그 블록은 새 method 의 기본값만 남는다(`Recipe.with_method` 규칙)."""
    global _PREVIEW
    session = _session()
    stage = str(req.json.get("stage", ""))
    method = str(req.json.get("method", ""))
    if not _known_stage(stage):
        return ApiResult(400, {"error": f"모르는 스테이지입니다: {stage}"})
    try:
        session.set_method(stage, method)
    except SessionError as exc:
        return ApiResult(400, {"error": str(exc), "stage": stage})
    _PREVIEW = None
    return _edit_result(session)


def _reset(req: Request) -> ApiResult:
    """프리셋 값으로 되돌리기 — 필드 하나(`name`) 또는 그 카드의 바뀐 행 전부. 기준이 없으면 아무것도 안 한다."""
    global _PREVIEW
    session = _session()
    stage = str(req.json.get("stage", ""))
    if not _known_stage(stage):
        return ApiResult(400, {"error": f"모르는 스테이지입니다: {stage}"})
    name = req.json.get("name")
    fields = cards.stage_card(session.recipe, stage).fields
    targets = [
        f for f in fields if f.has_baseline and (f.modified if name is None else f.name == name)
    ]
    for f in targets:
        try:
            session.set_stage_field(stage, f.name, f.baseline)
        except SessionError as exc:
            return ApiResult(400, {"error": str(exc), "stage": stage, "name": f.name})
    if targets:
        _PREVIEW = None
    return _edit_result(session)


def _per_class(req: Request) -> ApiResult:
    """`geometry.per_class` 표 → 레시피. **꺼진 행은 빠지고**(그 클래스는 전체 설정) `scale` 은 보존된다."""
    global _PREVIEW
    session = _session()
    rows = req.json.get("rows")
    if not isinstance(rows, list):
        return ApiResult(400, {"error": "클래스별 예외 행 목록이 필요합니다."})
    try:
        value = cards.per_class_dict(rows)
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        return ApiResult(400, {"error": f"값을 읽지 못했습니다: {exc}"})
    try:
        session.set_stage_field("geometry", "per_class", value)
    except SessionError as exc:
        return ApiResult(400, {"error": str(exc), "stage": "geometry", "name": "per_class"})
    _PREVIEW = None
    return _edit_result(session)


def _stage_image(req: Request) -> ApiResult:
    """단계별 중간 결과 썸네일 — 마지막 미리보기의 추적에서. Qt 카드 썸네일과 **같은 함수**."""
    res = _PREVIEW
    if res is None:
        return ApiResult(404, {"error": "아직 미리보기를 만들지 않았습니다."})
    image = cards.stage_thumbnail(req.get("stage"), res.steps, req.int_of("size", CARD_THUMB_PX))
    if image is None:
        return ApiResult(404, {"error": "이 단계는 보여 줄 중간 결과가 없습니다."})
    return _png(promote_to_bgr(image))


def _variant_image(req: Request) -> ApiResult:
    """시드 변형 k 한 장(작은 그림) — 캔버스의 마지막 미리보기는 건드리지 않는다.

    `image_rng(seed, k)` 로 **같은 바탕**에 다시 합성한다(CLI `run` 은 바탕도 rng 로 뽑지만 미리보기는
    사용자가 고른 바탕에 고정한다 — Qt 변형 카드와 같은 규칙).
    """
    session = _session()
    if session.prepared is None or session.target is None:
        return ApiResult(404, {"error": "바탕 이미지가 준비되지 않았습니다."})
    from anograft.preview import fit_long_side

    job = PreviewJob(
        KIND_PREVIEW,
        session.generation,
        target=session.target,
        index=max(0, req.int_of("k", 0)),
        long_side=session.long_side,
    )
    try:
        res = run_preview(session.prepared, job)
    except (runner.PrepareError, ValueError, OSError) as exc:
        return ApiResult(400, {"error": f"미리보기 실패: {exc}"})
    if res.result.status != "ok":
        return ApiResult(404, {"error": res.result.reason or "이 시드로는 붙이지 못했습니다."})
    tile = req.int_of("tile", VARIANT_TILE_PX)
    return _png(fit_long_side(promote_to_bgr(res.result.image), tile))


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
    register("/api/studio/to-batch", _serialized(_to_batch), write=True)
    register("/api/studio/cards", _serialized(_cards))
    register("/api/studio/stage-image", _serialized(_stage_image))
    register("/api/studio/variant-image", _serialized(_variant_image))
    register("/api/studio/field", _serialized(_field), write=True)
    register("/api/studio/method", _serialized(_method), write=True)
    register("/api/studio/reset", _serialized(_reset), write=True)
    register("/api/studio/per-class", _serialized(_per_class), write=True)
    _REGISTERED = True

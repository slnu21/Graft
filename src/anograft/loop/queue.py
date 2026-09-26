"""검토 대기 큐 (설계 `v1.x-training-loop.md` §2 규약 1 · §3 · §4 열쇠 1, 작업 단위 T5).

모델 예측을 **사람이 볼 목록**으로 바꾸고, 판정이 끝난 것을 은행으로 되돌린다. 루프가 닫히는 자리다.

**새 포맷을 만들지 않는다** — 큐 폴더는

    <out>/manifest.csv · review.csv(빈 판정) · queue.csv · images/ · masks/ · meta/

라서 **검수 화면(Qt 검수 탭 · 웹 ⑤)이 그대로 연다**(`ReviewSession.load` 는 레시피·은행이 없으면 fail-soft).
판정은 이미 있는 ``review.csv`` 계약(accept · reject · 빈칸)이고, 채택분은 ``bank import-pairs`` 와 **같은
공통 처리**(`import_pair_records`)로 은행에 들어간다(`accept_to_bank`).

무엇을 큐에 넣는가(§2 규약 1 · §4):

- **불일치 우선** — 비지도(정상 분포 이탈에 반응)와 지도(학습한 외형에 반응)는 오류가 독립적이라, 두 모델이
  어긋난 집합이 곧 **정보량 최대 집합**이다. 예측 폴더를 둘 주면(`--pred-b`) 이것이 큐 맨 앞에 온다.
- 나머지는 `policy.select_for_review` 그대로 — 경계 60% + 확신 20% + 무작위 20%. **TP 만 모으면 은행이
  모델의 거울이 되어 수렴한다**(무작위 몫이 드리프트를 잡는 자리).
- 불일치가 큐를 통째로 먹지 않게 **경계 몫을 상한으로** 둔다 — 불일치도 경계의 한 종류(더 강한 신호)로 보고,
  확신·무작위 몫은 지킨다.

선정은 순수 함수(§3 규율), 폴더 쓰기와 은행 되돌리기만 파일을 만진다.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from anograft.io.prune import QUEUE_FILE
from anograft.loop.policy import ReviewMix, agreement, select_for_review

#: 큐에 온 이유 — `policy.ReviewPick.reason`("경계"·"확신"·"무작위") 에 하나를 더한다.
REASON_DISAGREE = "불일치"

IMAGES_DIR = "images"
MASKS_DIR = "masks"
META_DIR = "meta"

DEFAULT_THRESHOLD = 0.5
DEFAULT_N = 30  # 설계 §4 — 하루 20~30장이 사람이 감당하는 크기
DEFAULT_IOU = 0.3


class QueueError(ValueError):
    pass


# --------------------------------------------------------------------------------------
# 1. 예측 읽기 — 파싱은 순수, 폴더 훑기만 IO
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """``scores/<stem>.json`` 의 인스턴스 하나."""

    bbox: tuple[float, float, float, float]  # x, y, w, h
    score: float
    cls: str

    def area_px(self) -> int:
        return round(max(0.0, self.bbox[2]) * max(0.0, self.bbox[3]))


@dataclass(frozen=True)
class Prediction:
    """이미지 한 장에 대한 예측 — ``scores/<stem>.json`` (+ 있으면 ``masks/<stem>.png``)."""

    stem: str
    image_score: float
    instances: tuple[Instance, ...] = ()
    mask: Path | None = None

    def boxes(self) -> list[tuple[float, float, float, float]]:
        return [i.bbox for i in self.instances]

    def classes(self) -> tuple[str, ...]:
        seen: list[str] = []
        for i in self.instances:
            if i.cls and i.cls not in seen:
                seen.append(i.cls)
        return tuple(seen)


def parse_scores(stem: str, payload: Any) -> tuple[Prediction, list[str]]:
    """``scores/<stem>.json`` 한 장 → ``Prediction``. 깨진 필드는 경고하고 건너뛴다(fail-soft).

    ``image_score`` 가 없으면 인스턴스 최고 점수로 본다(지도 모델의 관례 — `adapters/yolo.py` 가 그렇게 쓴다).
    """
    warnings: list[str] = []
    if not isinstance(payload, Mapping):
        return Prediction(stem=stem, image_score=0.0), [
            f"{stem}: scores JSON 이 오브젝트가 아닙니다"
        ]
    instances: list[Instance] = []
    raw = payload.get("instances")
    for k, item in enumerate(raw if isinstance(raw, Sequence) and not isinstance(raw, str) else []):
        if not isinstance(item, Mapping):
            warnings.append(f"{stem}: instances[{k}] 가 오브젝트가 아닙니다 — 건너뜀")
            continue
        box = item.get("bbox")
        if not (isinstance(box, Sequence) and not isinstance(box, str) and len(box) == 4):
            warnings.append(f"{stem}: instances[{k}].bbox 가 [x, y, w, h] 가 아닙니다 — 건너뜀")
            continue
        try:
            bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            warnings.append(f"{stem}: instances[{k}] 의 숫자를 읽지 못했습니다 — 건너뜀")
            continue
        instances.append(Instance(bbox=bbox, score=score, cls=str(item.get("class", "") or "")))
    try:
        image_score = float(payload["image_score"])
    except (KeyError, TypeError, ValueError):
        image_score = max((i.score for i in instances), default=0.0)
    return Prediction(stem=stem, image_score=image_score, instances=tuple(instances)), warnings


@dataclass
class PredictionSet:
    """예측 폴더 하나 — ``{stem: Prediction}`` + 경고."""

    root: Path
    items: dict[str, Prediction] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def scored(self) -> list[tuple[str, float]]:
        """``select_for_review`` 가 받는 ``(id, score)`` 목록 — stem 순(재현성)."""
        return [(s, self.items[s].image_score) for s in sorted(self.items)]


def read_predictions(root: str | Path) -> PredictionSet:
    """``<pred>/scores/*.json`` 을 읽고 ``<pred>/masks/<stem>.png`` 가 있으면 붙인다.

    한 장이 깨져도 나머지는 살린다 — 예측 수백 장 중 하나 때문에 큐를 못 만들면 루프가 멈춘다.
    """
    r = Path(root)
    scores_dir = r / "scores"
    if not scores_dir.is_dir():
        raise QueueError(f"예측 폴더가 아닙니다 (scores/ 없음): {r}")
    out = PredictionSet(root=r)
    masks_dir = r / MASKS_DIR
    for p in sorted(scores_dir.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            out.warnings.append(f"{p.name}: 읽기 실패 — {exc}")
            continue
        pred, warns = parse_scores(p.stem, payload)
        out.warnings.extend(warns)
        mask = masks_dir / f"{p.stem}.png"
        out.items[p.stem] = (
            pred
            if not mask.is_file()
            else Prediction(pred.stem, pred.image_score, pred.instances, mask)
        )
    if not out.items:
        raise QueueError(f"예측이 0장입니다: {scores_dir}")
    return out


# --------------------------------------------------------------------------------------
# 2. 무엇을 사람에게 보낼까 — 순수
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueItem:
    """큐 한 줄. ``index`` 순서가 곧 **검토 우선순위**다(처음 보는 형상 → 불일치 → 경계 → 확신 → 무작위)."""

    stem: str
    score: float
    reason: str
    detail: str  # 사람이 읽는 한 줄 — 사이드카 warnings 로 들어가 검수 화면 상세에 뜬다
    disagreement: int = 0  # 두 모델이 어긋난 검출 수(0 이면 일치 또는 비교 안 함)
    #: 처음 보는 형상 점수 0~1 (T15). 0 = 재지 않았거나 보관함 조각과 닮았다
    novelty: float = 0.0


def disagreement_counts(
    a: Mapping[str, Prediction],
    b: Mapping[str, Prediction],
    *,
    iou_thresh: float = DEFAULT_IOU,
) -> dict[str, int]:
    """두 예측을 이미지마다 맞춰 보고 **어긋난 검출 수**를 돌려준다(0 은 넣지 않는다).

    한쪽에만 있는 stem 은 비교 대상이 아니다(모델 B 가 그 이미지를 안 본 것 — 불일치가 아니라 결측).
    """
    out: dict[str, int] = {}
    for stem in sorted(set(a) & set(b)):
        ag = agreement(a[stem].boxes(), b[stem].boxes(), iou_thresh=iou_thresh)
        n = len(ag.only_a) + len(ag.only_b)
        if n:
            out[stem] = n
    return out


def select_queue(
    a: PredictionSet,
    b: PredictionSet | None = None,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    n: int = DEFAULT_N,
    mix: ReviewMix | None = None,
    iou_thresh: float = DEFAULT_IOU,
    rng: np.random.Generator | None = None,
) -> list[QueueItem]:
    """예측 → 검토 대기 목록. **불일치를 앞에, 나머지는 `select_for_review` 그대로.**

    불일치 몫의 상한은 ``mix`` 의 경계 비율이다 — 불일치도 경계의 한 종류로 보되 확신·무작위 몫을 지킨다
    (§2 규약 1: TP 만 모으면 은행이 모델의 거울이 된다).
    """
    if n <= 0 or not a.items:
        return []
    mix = mix or ReviewMix()
    w_boundary, w_confident, w_random = mix.weights()

    picked: list[QueueItem] = []
    if b is not None:
        counts = disagreement_counts(a.items, b.items, iou_thresh=iou_thresh)
        cap = min(n, round(n * w_boundary))
        # 많이 어긋난 것부터, 동점은 임계값에 가까운 것 → stem(재현성)
        order = sorted(
            counts,
            key=lambda s: (-counts[s], abs(a.items[s].image_score - threshold), s),
        )
        for stem in order[:cap]:
            score = a.items[stem].image_score
            picked.append(
                QueueItem(
                    stem=stem,
                    score=score,
                    reason=REASON_DISAGREE,
                    detail=(
                        f"두 모델이 어긋났습니다 — 검출 {counts[stem]}건이 한쪽에만 있습니다 "
                        f"(점수 {score:.3f})"
                    ),
                    disagreement=counts[stem],
                )
            )

    taken = {q.stem for q in picked}
    rest = [(s, v) for s, v in a.scored() if s not in taken]
    remaining = n - len(picked)
    if rest and remaining > 0:
        # 불일치가 이미 먹은 만큼 경계 몫을 줄인다(비율은 ReviewMix 가 정규화한다)
        rest_mix = ReviewMix(
            boundary=max(0.0, n * w_boundary - len(picked)),
            confident=n * w_confident,
            random=n * w_random,
        )
        for pick in select_for_review(
            rest, threshold=threshold, n=remaining, mix=rest_mix, rng=rng
        ):
            picked.append(
                QueueItem(
                    stem=pick.item_id,
                    score=pick.score,
                    reason=pick.reason,
                    detail=_detail_for(pick.reason, pick.score, threshold),
                )
            )
    return picked


def _detail_for(reason: str, score: float, threshold: float) -> str:
    if reason == "경계":
        return f"운영 임계값 근처입니다 — 점수 {score:.3f} (임계 {threshold:.3f})"
    if reason == "확신":
        return f"모델이 확신한 검출입니다 — 점수 {score:.3f} (품질 확인용)"
    return f"무작위로 뽑았습니다 — 점수 {score:.3f} (드리프트 감지용)"


def novelty_scores(
    items: Sequence[QueueItem],
    preds: PredictionSet,
    images: Mapping[str, Path],
    refs: Sequence[Any],
    *,
    scale: Sequence[float] | None = None,
) -> dict[str, float]:
    """고른 항목들의 **처음 보는 형상** 점수 (설계 §2b.5(1), T15).

    **고른 것만 잰다.** 후보 전체를 재려면 현장 이미지를 모두 읽어야 하고(4K 수천 장) 그건 tick 마다
    디스크를 통째로 읽는 일이다. 대신 이미 뽑힌 n 장(기본 30)만 읽어 **사람에게 보여 줄 순서**를 정한다 —
    진짜 새 유형은 대개 경계(낮은 확신)나 무작위 몫으로 이미 뽑혀 있다. 한계는 그대로 남으므로 문서에 남긴다.

    마스크가 없으면(검출만 하는 학습기) 0.0 — 모양을 못 재면 판정하지 않는다.
    """
    from anograft.core.novelty import Feature, novelty_score, scales
    from anograft.io import imgio

    if not refs:
        return {}
    unit = tuple(scale) if scale is not None else scales(refs)
    out: dict[str, float] = {}
    for item in items:
        src = images.get(item.stem)
        pred = preds.items.get(item.stem)
        if src is None or pred is None or pred.mask is None:
            continue
        try:
            image, _gray = imgio.read_image(src)
            mask = imgio.read_mask(pred.mask)
        except (imgio.ImageReadError, OSError):
            continue
        if mask.shape[:2] != image.shape[:2]:
            continue  # 크기가 다른 이상맵 — 마스크 없이 큐에 들어간다(T2 함정의 하류)
        out[item.stem] = novelty_score(Feature.of(image, mask), refs, scale=unit)
    return out


def order_by_novelty(
    items: Sequence[QueueItem], scores: Mapping[str, float], *, threshold: float
) -> list[QueueItem]:
    """처음 보는 형상을 **맨 앞으로** — 순수. 나머지 순서는 그대로 둔다(안정 정렬).

    `reason` 은 바꾸지 않는다(몫 회계·태그가 그것을 쓴다) — 점수만 항목에 붙고 순서가 바뀐다.
    """
    from anograft.core.novelty import is_novel

    scored = [replace(it, novelty=float(scores.get(it.stem, it.novelty))) for it in items]
    novel = [it for it in scored if is_novel(it.novelty, threshold)]
    rest = [it for it in scored if not is_novel(it.novelty, threshold)]
    novel.sort(key=lambda it: -it.novelty)
    return novel + rest


def reason_counts(items: Sequence[QueueItem]) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        out[it.reason] = out.get(it.reason, 0) + 1
    return out


# --------------------------------------------------------------------------------------
# 3. 큐 폴더 쓰기 — 검수 화면이 그대로 여는 형식
# --------------------------------------------------------------------------------------


@dataclass
class QueueSummary:
    out: Path
    written: list[QueueItem] = field(default_factory=list)
    missing_image: list[str] = field(default_factory=list)
    without_mask: list[str] = field(default_factory=list)
    novel: list[str] = field(default_factory=list)  # 처음 보는 형상으로 표시된 stem (T15)
    warnings: list[str] = field(default_factory=list)

    def reasons(self) -> dict[str, int]:
        return reason_counts(self.written)


def index_images(paths: Iterable[Path]) -> tuple[dict[str, Path], list[str]]:
    """``{stem: 경로}``. 같은 stem 이 둘이면 **먼저 온 것**을 쓰고 경고한다(예측 stem 과 1:1 이어야 한다)."""
    out: dict[str, Path] = {}
    warnings: list[str] = []
    for p in paths:
        if p.stem in out:
            warnings.append(
                f"{p.stem}: 같은 이름의 이미지가 둘 이상입니다 — {out[p.stem]} 을 씁니다"
            )
            continue
        out[p.stem] = p
    return out, warnings


def build_queue(
    items: Sequence[QueueItem],
    preds: PredictionSet,
    images: Mapping[str, Path],
    out: str | Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    trainer: str = "",
    round_no: int | None = None,
    pred_b: Path | None = None,
    novelty_threshold: float = 0.0,
) -> QueueSummary:
    """검토 대기 폴더를 만든다 — 이미지·마스크 사본 + 사이드카 + ``manifest.csv`` + 빈 ``review.csv``.

    이미지를 **복사**하는 이유: 큐는 사람에게 넘기는 묶음이라 자족해야 하고(폴더 하나만 열면 된다), 원본이
    옮겨가거나 지워져도 판정 이력이 남는다. 원본 경로는 사이드카·``queue.csv`` 에 그대로 적힌다.
    """
    from anograft.io import imgio
    from anograft.io.manifest import MANIFEST_FILE, write_manifest
    from anograft.io.prune import REVIEW_FILE, write_review

    root = Path(out)
    if root.exists() and any(root.iterdir()):
        raise QueueError(f"빈 폴더여야 합니다: {root}")
    summary = QueueSummary(out=root)
    (root / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    (root / META_DIR).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    for item in items:
        src = images.get(item.stem)
        if src is None:
            summary.missing_image.append(item.stem)
            summary.warnings.append(f"{item.stem}: 원본 이미지를 찾지 못했습니다 — 큐에서 뺍니다")
            continue
        pred = preds.items[item.stem]
        try:
            image, gray = imgio.read_image(src)
        except imgio.ImageReadError as exc:
            summary.missing_image.append(item.stem)
            summary.warnings.append(f"{item.stem}: 이미지를 읽지 못했습니다 — {exc}")
            continue
        h, w = image.shape[:2]

        rel_image = f"{IMAGES_DIR}/{src.name}"
        shutil.copyfile(src, root / rel_image)

        rel_mask, area_px, mask_warn = _copy_mask(pred, root, item.stem, (h, w))
        if mask_warn:
            summary.warnings.append(mask_warn)
        if not rel_mask:
            summary.without_mask.append(item.stem)
        if area_px == 0:
            area_px = sum(i.area_px() for i in pred.instances)

        index = str(len(rows))
        rel_meta = f"{META_DIR}/{item.stem}.json"
        (root / rel_meta).write_text(
            json.dumps(
                _sidecar(
                    item,
                    pred,
                    src,
                    shape=[h, w, 1 if gray else 3],
                    area_px=area_px,
                    threshold=threshold,
                    trainer=trainer,
                    round_no=round_no,
                    pred_root=preds.root,
                    pred_b=pred_b,
                    has_mask=bool(rel_mask),
                    novelty_threshold=novelty_threshold,
                ),
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "index": index,
                "status": "ok",
                "image": rel_image,
                "mask": rel_mask,
                "sidecar": rel_meta,
                "label": "",
                "target": src.as_posix(),
                "classes": ";".join(pred.classes()),
                "source_ids": "",
                "n_defects": len(pred.instances),
                "area_px": area_px,
                "blend": "",
                "fallback": 0,
                "reason": "",
            }
        )
        queue_rows.append(
            {
                "index": index,
                "stem": item.stem,
                "reason": item.reason,
                "score": f"{item.score:.4f}",
                "disagreement": item.disagreement,
                "novelty": f"{item.novelty:.4f}",
                "classes": ";".join(pred.classes()),
                "image": src.as_posix(),
            }
        )
        summary.written.append(item)
        if _novel(item, novelty_threshold):
            summary.novel.append(item.stem)

    write_manifest(root / MANIFEST_FILE, rows)
    # 빈 판정 — 검수 화면이 "미검수" 로 세고, 사람이 A/R 을 찍으면 그대로 덮어쓴다
    write_review(root / REVIEW_FILE, {r["index"]: ("", "") for r in rows})
    _write_queue_csv(root / QUEUE_FILE, queue_rows)
    return summary


def _novel(item: QueueItem, threshold: float) -> bool:
    from anograft.core.novelty import is_novel

    return is_novel(item.novelty, threshold)


def _copy_mask(
    pred: Prediction, root: Path, stem: str, shape: tuple[int, int]
) -> tuple[str, int, str]:
    """예측 마스크를 큐로 복사하고 ``(상대경로, 면적, 경고)``. 없거나 크기가 다르면 상대경로는 빈 문자열."""
    from anograft.io import imgio

    if pred.mask is None:
        return "", 0, ""
    try:
        mask = imgio.read_mask(pred.mask)
    except (imgio.ImageReadError, OSError) as exc:
        return "", 0, f"{stem}: 예측 마스크를 읽지 못했습니다 — {exc}"
    if mask.shape[:2] != shape:
        # 이상맵을 모델 입력 해상도 그대로 낸 어댑터 — 원본 크기로 되돌리는 건 어댑터의 책임이다(T2 함정)
        return (
            "",
            0,
            f"{stem}: 예측 마스크 크기 {mask.shape[:2]} ≠ 이미지 {shape} — 마스크 없이 큐에 넣습니다",
        )
    (root / MASKS_DIR).mkdir(parents=True, exist_ok=True)
    rel = f"{MASKS_DIR}/{stem}.png"
    shutil.copyfile(pred.mask, root / rel)
    return rel, int(np.count_nonzero(mask)), ""


def _sidecar(
    item: QueueItem,
    pred: Prediction,
    src: Path,
    *,
    shape: list[int],
    area_px: int,
    threshold: float,
    trainer: str,
    round_no: int | None,
    pred_root: Path,
    pred_b: Path | None,
    has_mask: bool,
    novelty_threshold: float = 0.0,
) -> dict[str, Any]:
    """큐 사이드카 — 합성 사이드카(§8.3)와 **같은 자리**에 ``gtmask.instances`` 를 둔다(검수 화면이 그걸 읽는다).

    합성이 아니므로 ``defects``·``blend`` 는 없고, 대신 ``queue`` 블록이 **왜 이게 왔는지**를 들고 있다.
    """
    warnings = [f"검토 대기 사유: {item.reason} — {item.detail}"]
    if _novel(item, novelty_threshold):
        # 이게 맨 앞에 온 이유이므로 **첫 줄**에 둔다(검수 화면이 첫 줄을 사유로 보여 준다)
        warnings.insert(
            0,
            f"처음 보는 형상입니다({item.novelty:.2f}) — 보관함 어느 조각과도 닮지 않았습니다. "
            "채택하면 이름 없이 미분류로 들어갑니다(나중에 bank promote 로 이름을 주세요)",
        )
    if not has_mask:
        warnings.append(
            "이 예측에는 마스크가 없습니다 — 채택해도 은행에 넣으려면 결함 표시 화면에서 그려야 합니다"
        )
    return {
        "queue": {
            "reason": item.reason,
            "detail": item.detail,
            "score": round(item.score, 6),
            "threshold": threshold,
            "disagreement": item.disagreement,
            "novelty": round(item.novelty, 4),
            "novelty_threshold": novelty_threshold,
            "trainer": trainer,
            "round": round_no,
            "predictions": pred_root.as_posix(),
            "predictions_b": pred_b.as_posix() if pred_b else None,
        },
        # 경로는 posix — 합성 사이드카·manifest 와 같은 규약(Windows 역슬래시 금지)
        "target": {"file": src.as_posix(), "shape": shape, "gray": shape[2] == 1},
        "gtmask": {
            "policy": "pred",  # 모델 초안이다 — 사람이 다듬기 전까지 정답이 아니다
            "area_px_total": area_px,
            "instances": [
                {
                    "class": i.cls,
                    "bbox": [round(v, 1) for v in i.bbox],
                    "area_px": i.area_px(),
                    "score": i.score,
                }
                for i in pred.instances
            ],
        },
        "warnings": warnings,
    }


def _write_queue_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    cols = ("index", "stem", "reason", "score", "disagreement", "novelty", "classes", "image")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def read_queue_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------------------
# 4. 판정된 큐 → 은행 (루프가 닫히는 곳)
# --------------------------------------------------------------------------------------


@dataclass
class AcceptSummary:
    bank: Path
    accepted: int = 0  # review.csv 가 채택한 항목 수
    imported: int = 0  # 실제로 은행에 들어간 쌍 수
    no_mask: list[str] = field(default_factory=list)
    unsorted: list[str] = field(default_factory=list)  # 처음 보는 형상이라 미분류로 들어간 것 (T15)
    classes: list[str] = field(default_factory=list)
    per_class: dict[str, int] = field(default_factory=dict)
    held_out: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def accept_to_bank(
    queue_root: str | Path,
    bank_out: str | Path,
    *,
    cls: str | None = None,
    tags: Sequence[str] = (),
    round_no: int | None = None,
    keep_whole: bool = False,
    min_area: int | None = None,
    margin: int | None = None,
    um_per_px: float | None = None,
    novelty_threshold: float = 0.0,
    log: Any = None,
) -> AcceptSummary:
    """``review.csv`` 가 **채택**한 것만 은행에 넣는다 — 임포터 공통 처리를 그대로 탄다.

    - ``mask_origin`` 은 ``pred:<학습기>`` 다. 모델 마스크는 **추정**이라 `bank ls` 의 ``est`` 로 세어야 하고,
      사람이 결함 표시 화면에서 다듬으면 ``manual:*`` 가 된다 — 그 비율이 설계 §2 규약 4 의 **사람 수정률**이다.
    - ``origin`` 은 큐 사본이 아니라 **원본 경로**다(감사 + `holdout.txt` 대조가 원본 stem 으로 걸리게).
    - 평가셋 거부·중복 id·작은 성분 버리기는 전부 `BankWriter.add` 한 지점에서 일어난다(T4).
    - **처음 보는 형상**(사이드카 `queue.novelty` ≥ `novelty_threshold`)은 **미분류**로 들어간다(T15).
      점수는 큐를 만들 때 이미 재어 사이드카에 적혀 있으므로 여기서 이미지를 다시 읽지 않는다.
      틀린 이름을 붙이는 것은 되돌릴 수 없고(그 클래스가 오염된다) 미분류는 `bank promote` 한 줄로 되돌린다.
    """
    from anograft.bank.importers.common import DEFAULT_MARGIN, DEFAULT_MIN_AREA
    from anograft.bank.importers.pairs import PairRecord, import_pair_records
    from anograft.core.classes import UNSORTED
    from anograft.core.novelty import is_novel
    from anograft.io.manifest import MANIFEST_FILE, read_manifest
    from anograft.io.prune import REVIEW_FILE, read_review

    root = Path(queue_root)
    if not (root / MANIFEST_FILE).is_file():
        raise QueueError(f"큐 폴더가 아닙니다 ({MANIFEST_FILE} 없음): {root}")
    review = read_review(root / REVIEW_FILE)
    summary = AcceptSummary(bank=Path(bank_out))

    records: list[PairRecord] = []
    trainers: set[str] = set()
    for row in read_manifest(root / MANIFEST_FILE):
        index = str(row.get("index", ""))
        if review.get(index, ("", ""))[0] != "accept":
            continue
        summary.accepted += 1
        meta = _read_sidecar(root, row.get("sidecar", ""))
        stem = Path(row.get("image", "")).stem
        if not row.get("mask"):
            summary.no_mask.append(stem)
            summary.warnings.append(
                f"{stem}: 마스크가 없어 은행에 넣지 못했습니다 — 결함 표시 화면에서 그린 뒤 저장하세요"
            )
            continue
        trainers.add(str((meta.get("queue") or {}).get("trainer") or ""))
        origin = str((meta.get("target") or {}).get("file") or row.get("target") or stem)
        item_cls = cls or _class_of(meta, row)
        item_tags = _item_tags(meta, round_no)
        novelty = float((meta.get("queue") or {}).get("novelty") or 0.0)
        if is_novel(novelty, novelty_threshold):
            item_cls = UNSORTED
            item_tags.append("novel")
            summary.unsorted.append(stem)
            summary.warnings.append(
                f"{stem}: 처음 보는 형상({novelty:.2f})이라 미분류로 넣습니다 — "
                "`bank promote` 로 이름을 주면 합성·출력에 쓰입니다"
            )
        records.append(
            PairRecord(
                image=root / row["image"],
                mask=root / row["mask"],
                cls=item_cls,
                origin=origin,
                id_hint=stem,
                tags=tuple(item_tags),
            )
        )

    trainer = next(iter(sorted(t for t in trainers if t)), "")
    result = import_pair_records(
        records,
        bank_out,
        keep_whole=keep_whole,
        min_area=DEFAULT_MIN_AREA if min_area is None else min_area,
        margin=DEFAULT_MARGIN if margin is None else margin,
        um_per_px=um_per_px,
        tags=tuple(tags),
        mask_origin=f"pred:{trainer}" if trainer else "pred",
        entry={"importer": "loop-queue", "queue": str(root), "round": round_no},
        log=log,
    )
    summary.imported = len(result.stats.added)
    summary.classes = list(result.classes)
    summary.per_class = result.stats.per_class()
    summary.held_out = list(result.stats.held_out)
    summary.warnings.extend(result.warnings)
    return summary


def _read_sidecar(root: Path, rel: str) -> dict[str, Any]:
    if not rel:
        return {}
    try:
        data = json.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _class_of(meta: Mapping[str, Any], row: Mapping[str, str]) -> str:
    """예측 클래스. 여러 개면 **면적이 가장 큰 인스턴스**의 클래스, 없으면 ``anomaly``.

    (비지도 모델은 클래스를 모른다 — 이름은 사람이 결함 보관함에서 준다.)
    """
    instances = (meta.get("gtmask") or {}).get("instances") or []
    best = ""
    best_area = -1
    for i in instances:
        if not isinstance(i, Mapping):
            continue
        area = int(i.get("area_px") or 0)
        name = str(i.get("class") or "")
        if name and area > best_area:
            best, best_area = name, area
    if best:
        return best
    from_row = (row.get("classes") or "").split(";")[0]
    return from_row or "anomaly"


def _item_tags(meta: Mapping[str, Any], round_no: int | None) -> list[str]:
    """설계 §2 4단계 — ``round-n`` · ``origin:field``. 드리프트가 오면 태그로 옛것을 감쇠·배제한다."""
    queue = meta.get("queue") or {}
    r = round_no if round_no is not None else queue.get("round")
    tags = ["origin:field"]
    if r is not None:
        tags.insert(0, f"round-{r}")
    reason = str(queue.get("reason") or "")
    if reason == REASON_DISAGREE:
        tags.append("disagree")
    return tags

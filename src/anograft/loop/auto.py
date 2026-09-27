"""자동 편입 — 확신이 아주 높은 예측을 **사람 큐를 거치지 않고** 보관함으로 (설계 §3 `gate` · §4 L2, T6).

검토 대기(T5)가 "무엇을 사람에게 보낼까"를 정한다면 여기는 그 앞에서 **"이건 안 보여 줘도 되나"** 를 정한다.
루프의 자동화 수준이 L1(마스크 초안)에서 L2(고확신 자동 편입)로 올라가는 자리이고, 사람은 매 장 판정에서
**경계 판정**으로 옮겨간다(설계 §4).

게이트는 다섯을 **전부** 넘어야 열린다 — 하나라도 어긋나면 그 조각은 사람 큐로 간다:

1. **마스크가 있다.** 검출 모델(박스만)의 예측은 GT 를 못 주므로 애초에 은행에 못 들어간다.
2. **처음 보는 형상이 아니다**(T15). 새 유형에 이름을 자동으로 붙이는 것은 되돌릴 수 없다 — 틀린 이름보다
   이름 없음이 낫고, 그 판단은 사람이 한다.
3. **보관함에 이미 있는 클래스다.** 게이트가 **클래스를 만들지 않는다** — 새 클래스는 class id 를 밀고
   평가셋에 정답이 없어 라운드 비교를 끊는다(설계 §6.7). 자동으로 넘길 일이 아니다.
4. **검출 점수가 문턱 위다.** 경계는 정보량이 가장 많은 자리라 사람에게 보내야 한다 — 자동은 **확신** 몫만.
5. **추정 마스크의 타당성**(`mask_confidence`)이 문턱 위다(`core.gate`). 점수가 높아도 마스크가 엉뚱한 곳을
   잡으면(경면 금속) GT 가 오염된다 — 은행 임포트 게이트와 **같은 판정**을 쓴다.

그리고 교차 검증(설계 §4 열쇠 1): ``require_agreement`` 가 켜져 있으면 **두 번째 예측이 있어야 하고**
어긋난 검출이 없어야 한다. 비지도와 지도는 오류가 독립적이라 둘이 같은 곳을 지목한 것은 정밀도가 높다 —
반대로 비교할 상대가 없으면 **열지 않는다**(모르면 막지 않는 트리거와 달리, 은행에 쓰는 게이트는 모르면
닫는 쪽이 맞다 — `holdout.txt` 거부와 같은 정신).

**기본값은 전부 0 = 끔**이다(설계 §8 확인 게이트) — 임계값은 현장이 정한다. 꺼져 있으면 라운드에
`auto` 단계가 아예 서지 않으므로, 아무 일도 안 하는 단계를 화면에서 묻게 되지 않는다.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from anograft.core.gate import gate, gate_reason
from anograft.loop.queue import DEFAULT_IOU, FIELD_TAG, Prediction, PredictionSet

#: 자동으로 들어온 조각에 붙는 태그 — `bank ls` 의 tags 열에 `auto:N` 으로 찍혀
#: **아무도 안 본 조각**이 몇 개인지 그대로 보인다(웹 ① 보관함에서도 태그로 보인다).
AUTO_TAG = "auto"

#: 자동 편입 기록 파일(감사용) — 라운드 폴더에 남는다. 큐의 `queue.csv` 와 같은 자리·같은 정신이다.
AUTO_FILE = "auto.csv"

_COLUMNS: tuple[str, ...] = (
    "stem",
    "class",
    "score",
    "confidence",
    "novelty",
    "disagreement",
    "reason",
    "image",
)


# --------------------------------------------------------------------------------------
# 1. 정책과 판정 — 순수
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AutoPolicy:
    """자동 편입 문턱. **기본값은 전부 0/끔** — 코드가 임계값을 정하지 않는다(설계 §8)."""

    #: 검출 점수 하한(0 = 자동 편입을 하지 않는다). 운영 임계값보다 **한참 위**여야 뜻이 있다
    score: float = 0.0
    #: 추정 마스크 타당성(`mask_confidence`) 하한. 0 = 신뢰도로 막지 않는다
    min_confidence: float = 0.0
    #: 두 번째 예측과 **일치**해야 여는가(설계 §4 열쇠 1). 켜져 있고 상대가 없으면 **열지 않는다**
    require_agreement: bool = True
    #: 처음 보는 형상도 자동으로 받을지 — 기본 거부(T15: 틀린 이름보다 이름 없음이 낫다)
    allow_novel: bool = False
    #: 한 라운드에 자동으로 받을 최대 장수(0 = 상한 없음). 점수 높은 것부터 채운다
    max_per_round: int = 0

    @property
    def enabled(self) -> bool:
        """켜졌는가 — **점수 문턱이 있어야** 자동 편입이 일어난다(나머지는 그 위의 조건이다)."""
        return self.score > 0.0


@dataclass(frozen=True)
class AutoCandidate:
    """자동 편입 판정에 필요한 사실만. 파일은 호출부가 읽어서 넣는다(이 판정은 순수)."""

    stem: str
    score: float
    cls: str = ""
    has_mask: bool = False
    #: 추정 마스크 타당성(`mask_confidence`). ``None`` = 아직 재지 않았거나 사람 마스크다
    confidence: float | None = None
    #: 처음 보는 형상 점수(T15). 0 = 재지 않았거나 보관함 조각과 닮았다
    novelty: float = 0.0
    #: 두 모델이 어긋난 검출 수. ``None`` = **비교하지 않았다**(상대 예측이 없다)
    disagreement: int | None = None


@dataclass(frozen=True)
class AutoVerdict:
    """``(받을까?, 왜)`` — 게이트도 트리거·자동 정지와 같이 **이유를 항상 돌려준다**."""

    admit: bool
    reason: str


def auto_admit(
    candidate: AutoCandidate,
    policy: AutoPolicy,
    *,
    known_classes: Sequence[str] = (),
    novelty_threshold: float = 0.0,
) -> AutoVerdict:
    """이 예측을 사람 없이 보관함에 넣어도 되나.

    순서에 뜻이 있다 — **못 넣는 것**(마스크 없음)부터, 그다음 **넣으면 되돌릴 수 없는 것**(새 유형·새
    클래스), 마지막이 문턱 둘(점수·마스크 타당성)이다. 사람이 사유를 읽고 무엇을 고칠지 알 수 있어야 한다.
    """
    from anograft.core.novelty import is_novel

    if not policy.enabled:
        return AutoVerdict(False, "자동 편입이 꺼져 있습니다")
    if not candidate.has_mask:
        return AutoVerdict(False, "마스크가 없습니다 — 결함 표시 화면에서 그려야 은행에 들어갑니다")
    if not policy.allow_novel and is_novel(candidate.novelty, novelty_threshold):
        return AutoVerdict(
            False,
            f"처음 보는 형상입니다({candidate.novelty:.2f}) — 이름은 사람이 줘야 합니다",
        )
    if known_classes and candidate.cls and candidate.cls not in known_classes:
        return AutoVerdict(
            False,
            f"보관함에 없는 클래스입니다({candidate.cls}) — 클래스 신설은 사람 결정입니다",
        )
    if candidate.score < policy.score:
        return AutoVerdict(
            False, f"검출 점수 {candidate.score:.3f} < 자동 편입 문턱 {policy.score:.3f}"
        )
    if policy.require_agreement:
        if candidate.disagreement is None:
            return AutoVerdict(
                False,
                "교차 검증할 두 번째 예측이 없습니다 — --pred-b 를 주거나 require_agreement 를 끄세요",
            )
        if candidate.disagreement > 0:
            return AutoVerdict(
                False,
                f"두 모델이 어긋났습니다 — 검출 {candidate.disagreement}건이 한쪽에만 있습니다",
            )
    if not gate(candidate.confidence, policy.min_confidence):
        return AutoVerdict(False, gate_reason(candidate.confidence, policy.min_confidence))
    parts = [f"검출 점수 {candidate.score:.3f} ≥ {policy.score:.3f}"]
    if policy.min_confidence > 0:
        parts.append(gate_reason(candidate.confidence, policy.min_confidence))
    if candidate.disagreement == 0:
        parts.append("두 모델이 일치")
    return AutoVerdict(True, " · ".join(parts))


def partition_auto(
    candidates: Sequence[AutoCandidate],
    policy: AutoPolicy,
    *,
    known_classes: Sequence[str] = (),
    novelty_threshold: float = 0.0,
) -> tuple[list[AutoCandidate], list[tuple[AutoCandidate, str]]]:
    """``(자동 편입, [(사람 큐로, 사유)])`` — 순수. 상한을 넘은 것은 **점수 낮은 쪽부터** 사람 큐로.

    상한(`max_per_round`)에서 잘린 것을 조용히 버리지 않는다 — 사람이 볼 목록으로 넘긴다(자동이 못 받은
    것은 여전히 판정 대상이다).
    """
    admit: list[tuple[AutoCandidate, str]] = []
    held: list[tuple[AutoCandidate, str]] = []
    for cand in candidates:
        verdict = auto_admit(
            cand, policy, known_classes=known_classes, novelty_threshold=novelty_threshold
        )
        (admit if verdict.admit else held).append((cand, verdict.reason))
    # 점수 높은 순(동점은 stem — 재현성). 상한이 있으면 그 뒤는 사람 큐로 돌린다.
    admit.sort(key=lambda t: (-t[0].score, t[0].stem))
    if policy.max_per_round > 0 and len(admit) > policy.max_per_round:
        cut = admit[policy.max_per_round :]
        admit = admit[: policy.max_per_round]
        held.extend(
            (cand, f"이번 라운드 자동 편입 상한 {policy.max_per_round}장을 넘었습니다")
            for cand, _reason in cut
        )
    held.sort(key=lambda t: t[0].stem)
    return [cand for cand, _ in admit], held


def cheap_candidates(
    preds: PredictionSet,
    policy: AutoPolicy,
    *,
    other: PredictionSet | None = None,
    iou_thresh: float = DEFAULT_IOU,
) -> tuple[list[AutoCandidate], int]:
    """**이미지를 읽기 전에** 후보를 좁힌다 — 점수·마스크 유무·불일치만 보고(순수).

    왜 두 단계인가: `mask_confidence` 와 처음 보는 형상은 이미지·마스크를 읽어야 재는 값이라, 현장 수천
    장을 전부 재면 라운드마다 디스크를 통째로 읽는다(T15 가 큐에서 "고른 것만 잰다"로 푼 것과 같은 함정).
    점수 문턱은 공짜로 걸러지므로 **확신 꼬리만** 재면 된다.

    돌려주는 둘째 값은 **문턱 아래로 떨어진 장수**다 — 요약이 "몇 장을 안 봤는지" 말할 수 있어야 한다.
    """
    from anograft.loop.queue import disagreement_counts

    if not policy.enabled:
        return [], len(preds.items)
    counts = (
        disagreement_counts(preds.items, other.items, iou_thresh=iou_thresh)
        if other is not None
        else {}
    )
    comparable = set(preds.items) & set(other.items) if other is not None else set()
    out: list[AutoCandidate] = []
    below = 0
    for stem in sorted(preds.items):
        pred = preds.items[stem]
        if pred.image_score < policy.score:
            below += 1
            continue
        out.append(
            AutoCandidate(
                stem=stem,
                score=pred.image_score,
                cls=_class_of(pred),
                has_mask=pred.mask is not None,
                disagreement=(counts.get(stem, 0) if stem in comparable else None),
            )
        )
    return out, below


def _class_of(pred: Prediction) -> str:
    """면적이 가장 큰 인스턴스의 클래스. 없으면 빈 문자열(비지도 모델은 클래스를 모른다).

    `queue._class_of` 와 같은 규칙이다 — 다른 규칙을 쓰면 같은 예측이 경로에 따라 다른 클래스로 들어간다.
    """
    best = ""
    best_area = -1
    for inst in pred.instances:
        if inst.cls and inst.area_px() > best_area:
            best, best_area = inst.cls, inst.area_px()
    return best


# --------------------------------------------------------------------------------------
# 2. 재기 — 좁혀진 후보만 이미지를 읽는다
# --------------------------------------------------------------------------------------


def measure(
    candidates: Sequence[AutoCandidate],
    preds: PredictionSet,
    images: Mapping[str, Path],
    *,
    refs: Sequence[Any] = (),
    box_margin: int = 6,
) -> list[AutoCandidate]:
    """후보에 **마스크 타당성**과 **처음 보는 형상** 점수를 채운다(IO — 후보만 읽는다).

    `mask_confidence` 는 원래 "박스→마스크 추정이 결함을 잡았나"를 재는 함수인데, 모델이 낸 마스크도 추정이라
    그대로 쓴다(설계 §3 의 "mask_confidence 재사용"). 박스는 예측 인스턴스의 것을 쓰고, 없으면 마스크의
    외접 사각형을 쓴다 — 점수가 "마스크 안 vs 박스 바깥 링"의 분리도라 박스가 있어야 정의된다.

    못 읽거나 크기가 어긋나면 ``confidence`` 를 **NaN** 으로 둔다 — 게이트가 그것을 거부한다(모르는 것을
    통과시키지 않는다). 마스크가 이미지와 크기가 다른 것은 T2 함정의 하류다(어댑터가 원복할 책임).
    """
    from anograft.bank.mask_from_box import mask_confidence
    from anograft.core.novelty import Feature, novelty_score, scales
    from anograft.io import imgio

    unit = scales(refs) if refs else None
    out: list[AutoCandidate] = []
    for cand in candidates:
        pred = preds.items.get(cand.stem)
        src = images.get(cand.stem)
        if pred is None or pred.mask is None or src is None:
            out.append(replace(cand, confidence=float("nan")))
            continue
        try:
            image, _gray = imgio.read_image(src)
            mask = imgio.read_mask(pred.mask)
        except (imgio.ImageReadError, OSError):
            out.append(replace(cand, confidence=float("nan")))
            continue
        if mask.shape[:2] != image.shape[:2]:
            out.append(replace(cand, confidence=float("nan")))
            continue
        box = _box_for(pred, mask)
        if box is None:
            out.append(replace(cand, confidence=float("nan")))
            continue
        conf = mask_confidence(image, mask, box, margin=box_margin)
        novelty = (
            novelty_score(Feature.of(image, mask), refs, scale=unit)
            if refs and unit is not None
            else 0.0
        )
        out.append(replace(cand, confidence=float(conf.score), novelty=float(novelty)))
    return out


def _box_for(pred: Prediction, mask: Any) -> tuple[int, int, int, int] | None:
    """타당성을 재는 기준 박스 — 예측 인스턴스의 것(면적 최대), 없으면 마스크 외접 사각형."""
    import cv2

    best = None
    best_area = -1
    for inst in pred.instances:
        if inst.area_px() > best_area:
            best, best_area = inst.bbox, inst.area_px()
    if best is not None and best_area > 0:
        return (int(best[0]), int(best[1]), int(best[2]), int(best[3]))
    x, y, w, h = cv2.boundingRect((mask > 0).astype("uint8"))
    return (int(x), int(y), int(w), int(h)) if w > 0 and h > 0 else None


# --------------------------------------------------------------------------------------
# 3. 보관함으로 — 임포터 공통 처리를 그대로 탄다
# --------------------------------------------------------------------------------------


@dataclass
class AutoSummary:
    """한 번의 자동 편입이 남긴 사실. `loop status`·원장·화면이 같은 숫자를 본다."""

    bank: Path
    admitted: list[AutoCandidate] = field(default_factory=list)
    imported: int = 0
    #: 사람 큐로 돌린 것 ``[(stem, 사유)]`` — **조용히 빠지는 조각을 만들지 않는다**
    held: list[tuple[str, str]] = field(default_factory=list)
    #: 받은 것의 사유 ``{stem: 왜 열렸나}`` — `auto.csv` 와 로그가 이걸 쓴다
    reasons: dict[str, str] = field(default_factory=dict)
    #: 점수 문턱 아래라 재지도 않은 장수
    below_score: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    held_out: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def stems(self) -> list[str]:
        """자동으로 받은 stem — **검토 대기에서 빼야 하는 목록**이다(이미 은행에 있는 것을 판정시키지 않는다)."""
        return [c.stem for c in self.admitted]


def admit_to_bank(
    admitted: Sequence[AutoCandidate],
    preds: PredictionSet,
    images: Mapping[str, Path],
    bank_out: str | Path,
    *,
    cls: str | None = None,
    trainer: str = "",
    round_no: int | None = None,
    tags: Sequence[str] = (),
    keep_whole: bool = False,
    min_area: int | None = None,
    margin: int | None = None,
    um_per_px: float | None = None,
    min_confidence: float = 0.0,
    log: Any = None,
) -> AutoSummary:
    """자동 편입분을 은행에 넣는다 — `loop accept` 와 **같은 공통 처리**(`import_pair_records`).

    큐처럼 이미지를 복사하지 않는다: 자동 편입은 사람에게 넘기는 묶음이 아니라 곧바로 은행에 들어가는
    일이고, 감사에 필요한 것은 사본이 아니라 **원본 경로와 사유**(`auto.csv` · 메타 `origin`)다.

    마스크 출처는 ``pred:<학습기>`` 그대로다(사람이 다듬으면 `manual:*` 가 된다 — 사람 수정률의 분자).
    태그에 ``auto`` 를 더해 **아무도 안 본 조각**을 나중에 그대로 찾을 수 있게 한다.
    """
    from anograft.bank.importers.common import DEFAULT_MARGIN, DEFAULT_MIN_AREA
    from anograft.bank.importers.pairs import PairRecord, import_pair_records

    summary = AutoSummary(bank=Path(bank_out), admitted=list(admitted))
    records: list[PairRecord] = []
    for cand in admitted:
        pred = preds.items.get(cand.stem)
        src = images.get(cand.stem)
        if pred is None or pred.mask is None or src is None:
            summary.warnings.append(f"{cand.stem}: 예측 마스크·원본을 찾지 못해 건너뜁니다")
            continue
        item_tags = [AUTO_TAG, FIELD_TAG]
        if round_no is not None:
            item_tags.insert(0, f"round-{round_no}")
        records.append(
            PairRecord(
                image=src,
                mask=pred.mask,
                cls=cls or cand.cls or "anomaly",
                origin=src.as_posix(),
                id_hint=cand.stem,
                tags=tuple(item_tags),
                confidence=cand.confidence,
            )
        )
    if not records:
        return summary
    result = import_pair_records(
        records,
        bank_out,
        keep_whole=keep_whole,
        min_area=DEFAULT_MIN_AREA if min_area is None else min_area,
        margin=DEFAULT_MARGIN if margin is None else margin,
        um_per_px=um_per_px,
        tags=tuple(tags),
        mask_origin=f"pred:{trainer}" if trainer else "pred",
        min_confidence=min_confidence,
        entry={"importer": "loop-auto", "round": round_no, "trainer": trainer},
        log=log,
    )
    summary.imported = len(result.stats.added)
    summary.per_class = result.stats.per_class()
    summary.held_out = list(result.stats.held_out)
    summary.warnings.extend(result.warnings)
    return summary


def write_auto_csv(
    path: str | Path,
    admitted: Sequence[AutoCandidate],
    reasons: Mapping[str, str],
    images: Mapping[str, Path],
) -> Path:
    """감사 기록 — 무엇을 왜 사람 없이 받았나. 큐의 `queue.csv` 와 같은 자리·같은 열 정신이다."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for cand in admitted:
            src = images.get(cand.stem)
            writer.writerow(
                {
                    "stem": cand.stem,
                    "class": cand.cls,
                    "score": f"{cand.score:.4f}",
                    "confidence": "" if cand.confidence is None else f"{cand.confidence:.4f}",
                    "novelty": f"{cand.novelty:.4f}",
                    "disagreement": "" if cand.disagreement is None else cand.disagreement,
                    "reason": reasons.get(cand.stem, ""),
                    "image": src.as_posix() if src is not None else "",
                }
            )
    return p


def read_auto_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------------------
# 4. 한 번에 — 고르고 재고 넣는다 (CLI `loop auto` · 라운드 `auto` 단계가 같이 쓴다)
# --------------------------------------------------------------------------------------


def run_auto(
    preds: PredictionSet,
    images: Mapping[str, Path],
    bank_out: str | Path,
    policy: AutoPolicy,
    *,
    other: PredictionSet | None = None,
    iou_thresh: float = DEFAULT_IOU,
    known_classes: Sequence[str] = (),
    refs: Sequence[Any] = (),
    novelty_threshold: float = 0.0,
    cls: str | None = None,
    trainer: str = "",
    round_no: int | None = None,
    tags: Sequence[str] = (),
    keep_whole: bool = False,
    min_area: int | None = None,
    margin: int | None = None,
    um_per_px: float | None = None,
    box_margin: int = 6,
    dry_run: bool = False,
    log: Any = None,
) -> AutoSummary:
    """예측 → (게이트) → 보관함. ``dry_run`` 이면 판정만 하고 은행을 만지지 않는다.

    `loop auto`(CLI)와 라운드 `auto` 단계가 **같은 함수**를 부른다 — 화면·라운드가 다른 판정을 하면 사람이
    "왜 CLI 로는 들어가는데 라운드에서는 안 들어가나"를 묻게 된다.
    """
    cheap, below = cheap_candidates(preds, policy, other=other, iou_thresh=iou_thresh)
    if cls:
        # 이름을 고정했으면(비지도 예측처럼 클래스를 모를 때) **그것이 실제 클래스**다 —
        # 예측이 뭐라 부르든 이 이름으로 들어가므로 "보관함에 없는 클래스" 검사도 이 이름으로 한다.
        cheap = [replace(c, cls=cls) for c in cheap]
    measured = measure(cheap, preds, images, refs=refs, box_margin=box_margin) if cheap else []
    admitted, held = partition_auto(
        measured,
        policy,
        known_classes=known_classes,
        novelty_threshold=novelty_threshold,
    )
    reasons = {
        cand.stem: auto_admit(
            cand, policy, known_classes=known_classes, novelty_threshold=novelty_threshold
        ).reason
        for cand in admitted
    }
    if dry_run:
        summary = AutoSummary(bank=Path(bank_out), admitted=admitted)
    else:
        summary = admit_to_bank(
            admitted,
            preds,
            images,
            bank_out,
            cls=cls,
            trainer=trainer,
            round_no=round_no,
            tags=tags,
            keep_whole=keep_whole,
            min_area=min_area,
            margin=margin,
            um_per_px=um_per_px,
            min_confidence=policy.min_confidence,
            log=log,
        )
    summary.below_score = below
    summary.held = [(cand.stem, reason) for cand, reason in held]
    summary.reasons = reasons
    return summary


__all__ = [
    "AUTO_FILE",
    "AUTO_TAG",
    "AutoCandidate",
    "AutoPolicy",
    "AutoSummary",
    "AutoVerdict",
    "admit_to_bank",
    "auto_admit",
    "cheap_candidates",
    "measure",
    "partition_auto",
    "read_auto_csv",
    "run_auto",
    "write_auto_csv",
]

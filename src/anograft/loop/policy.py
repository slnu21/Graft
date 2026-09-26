"""루프의 순수 로직 (설계 `v1.x-training-loop.md` §3) — Qt·torch·경로·파일 IO 의존 0.

여기 있는 함수들이 "루프가 무엇을 사람에게 보내고, 무엇을 자동으로 받아들이고, 언제 승급하는가"를 정한다.
전부 순수 함수라 **`adapters/noop` 더미만으로 루프 전체가 테스트된다** — 더미로 안 돌면 설계가 틀린 것이다.

무작위성은 호출자가 넘긴 ``rng``(numpy Generator)만 쓴다. 전역 ``np.random`` 은 쓰지 않는다(코어 규약).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

Box = tuple[float, float, float, float]  # x, y, w, h


# --------------------------------------------------------------------------------------
# 1. 검토 대상 고르기 — 확신 높은 것이 아니라 "경계"를 보낸다
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewMix:
    """검토 큐의 구성비 (설계 §2 규약 1).

    **TP 만 모으면 은행이 모델의 거울이 되어 수렴한다** — 모델이 잡은 것만 넣으면 미검출 유형은 영원히
    들어오지 않고, 평가셋까지 같은 경로면 recall 이 과대평가된다. 그래서 경계를 주력으로 두고 무작위를 섞는다.
    (네 번째 경로인 FN 회수 채널은 외부에서 들어오므로 이 함수 밖이다.)
    """

    boundary: float = 0.6  # 임계값 근처 — 정보량이 가장 많다
    confident: float = 0.2  # 확신 높은 검출 — 품질 확인용
    random: float = 0.2  # 검출과 무관한 무작위 — 드리프트 감지

    def weights(self) -> tuple[float, float, float]:
        total = self.boundary + self.confident + self.random
        if total <= 0:
            raise ValueError("ReviewMix 의 합이 0 이하입니다")
        return (self.boundary / total, self.confident / total, self.random / total)


@dataclass(frozen=True)
class ReviewPick:
    item_id: str
    score: float
    reason: str  # "경계" · "확신" · "무작위" — 큐에서 왜 이게 왔는지 보여 준다


def select_for_review(
    scored: Sequence[tuple[str, float]],
    *,
    threshold: float,
    n: int,
    mix: ReviewMix | None = None,
    rng: np.random.Generator | None = None,
) -> list[ReviewPick]:
    """스코어 목록에서 사람이 볼 ``n`` 개를 고른다.

    ``scored`` 는 ``(id, score)``. ``threshold`` 는 운영 임계값이고, 그 **근처**가 경계다.
    ``rng`` 가 없으면 무작위 몫도 결정적으로(앞에서부터) 채운다 — 테스트와 재현을 위해.
    몫 하나가 비어 남으면(임계값 위 검출이 하나도 없는 초기 라운드 등) **경계 순서로 채워** n 을 맞춘다.
    """
    if n <= 0 or not scored:
        return []
    mix = mix or ReviewMix()
    w_boundary, w_confident, _ = mix.weights()

    n_boundary = min(len(scored), round(n * w_boundary))
    n_confident = min(len(scored) - n_boundary, round(n * w_confident))
    n_random = max(0, n - n_boundary - n_confident)

    picked: list[ReviewPick] = []
    taken: set[str] = set()

    def take(pool: Sequence[tuple[str, float]], count: int, reason: str) -> None:
        for item_id, score in pool:
            if count <= 0:
                return
            if item_id in taken:
                continue
            taken.add(item_id)
            picked.append(ReviewPick(item_id=item_id, score=score, reason=reason))
            count -= 1

    # 경계: |score − threshold| 가 작은 순. 동점은 id 로 안정 정렬(재현성).
    by_boundary = sorted(scored, key=lambda s: (abs(s[1] - threshold), s[0]))
    take(by_boundary, n_boundary, "경계")

    # 확신: 임계값을 넘은 것 중 점수가 높은 순
    by_confident = sorted((s for s in scored if s[1] >= threshold), key=lambda s: (-s[1], s[0]))
    take(by_confident, n_confident, "확신")

    # 무작위: 남은 것에서
    rest = [s for s in scored if s[0] not in taken]
    if rest and n_random > 0:
        if rng is not None:
            idx = rng.permutation(len(rest))
            rest = [rest[i] for i in idx]
        else:
            rest = sorted(rest, key=lambda s: s[0])
        take(rest, n_random, "무작위")

    # 당이 비었으면(임계값을 넘긴 검출이 없는 라운드 등) **경계 순서로 채운다** —
    # 사람에게 보낼 자리를 비워 두면 큐가 요청한 n 보다 작아진다(루프 첫 라운드에서 실제로 겪음).
    if len(picked) < n:
        take(by_boundary, n - len(picked), "경계")

    return picked


# --------------------------------------------------------------------------------------
# 2. 자동 편입 게이트 — 신뢰도가 확실한 것만 사람 없이 받는다
# --------------------------------------------------------------------------------------


def gate(confidence: float | None, threshold: float) -> bool:
    """자동으로 은행에 넣어도 되는가.

    ``confidence`` 가 ``None`` 이면 **사람이 그린 정확한 마스크**라는 뜻이라 통과시킨다
    (`mask_origin` 이 ``png``·``manual:*`` 인 경우 — 은행 메타에서 confidence 는 그때 비어 있다).
    """
    if confidence is None:
        return True
    if math.isnan(confidence):
        return False
    return confidence >= threshold


def partition_by_gate(
    items: Sequence[tuple[str, float | None]], threshold: float
) -> tuple[list[str], list[str]]:
    """``(자동 편입, 사람 큐)`` 로 가른다."""
    auto: list[str] = []
    queue: list[str] = []
    for item_id, conf in items:
        (auto if gate(conf, threshold) else queue).append(item_id)
    return auto, queue


# --------------------------------------------------------------------------------------
# 3. 교차 검증 — 비지도 × 지도는 오류가 독립이다
# --------------------------------------------------------------------------------------


def iou(a: Box, b: Box) -> float:
    """두 박스(x, y, w, h)의 IoU."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class Agreement:
    """두 모델의 검출을 맞춰 본 결과 (설계 §4 열쇠 1)."""

    matched: list[tuple[int, int]] = field(default_factory=list)  # (a 인덱스, b 인덱스)
    only_a: list[int] = field(default_factory=list)
    only_b: list[int] = field(default_factory=list)

    @property
    def agreed(self) -> bool:
        """둘이 같은 것을 지목했고 어긋난 게 없는가 → **자동 NG 확정** 후보."""
        return bool(self.matched) and not self.only_a and not self.only_b

    @property
    def needs_human(self) -> bool:
        """불일치가 있는가 → 사람 큐. 이 집합이 곧 **정보량 최대 집합**이다."""
        return bool(self.only_a or self.only_b)


def agreement(a: Sequence[Box], b: Sequence[Box], *, iou_thresh: float = 0.3) -> Agreement:
    """탐욕적 1:1 매칭. IoU 가 큰 쌍부터 붙인다(동점은 인덱스 순으로 안정적).

    비지도(정상 분포 이탈에 반응, 먼지·조명 얼룩에 속음)와 지도(학습한 외형에 반응, 새 유형에 장님)는
    **오류가 독립적**이라 교집합의 정밀도가 높고, 불일치는 능동학습 샘플링과 같은 집합이 된다.
    """
    pairs = sorted(
        ((iou(box_a, box_b), i, j) for i, box_a in enumerate(a) for j, box_b in enumerate(b)),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int]] = []
    for score, i, j in pairs:
        if score < iou_thresh:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j))
    matched.sort()
    return Agreement(
        matched=matched,
        only_a=[i for i in range(len(a)) if i not in used_a],
        only_b=[j for j in range(len(b)) if j not in used_b],
    )


# --------------------------------------------------------------------------------------
# 4. 승급 판정 — 고정 골든에서 회귀 없음 ∧ 롤링에서 개선
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionVerdict:
    promote: bool
    reason: str


def should_promote(
    *,
    fixed_champion: float,
    fixed_challenger: float,
    rolling_champion: float | None = None,
    rolling_challenger: float | None = None,
    noise: float = 0.04,
) -> PromotionVerdict:
    """새 모델(challenger)을 배포할지.

    ``noise`` 기본값 0.04 는 BENCHMARKS §2 에서 잰 **학습 시드 요동**이다 — 그보다 작은 Δ 는 개선이 아니다.
    고정 골든은 회귀 감시(떨어지면 무조건 정지), 롤링 골든은 현재 공정 성능(승급 판단). 둘 다 본다(설계 §2b.6).
    """
    d_fixed = fixed_challenger - fixed_champion
    if d_fixed < -noise:
        return PromotionVerdict(False, f"고정 골든셋이 회귀했습니다 (Δ {d_fixed:+.3f})")

    if rolling_champion is not None and rolling_challenger is not None:
        d_roll = rolling_challenger - rolling_champion
        if d_roll > noise:
            return PromotionVerdict(True, f"롤링 골든셋 개선 (Δ {d_roll:+.3f}) · 고정 회귀 없음")
        return PromotionVerdict(
            False, f"롤링 개선이 시드 노이즈 이하입니다 (Δ {d_roll:+.3f} ≤ {noise:.3f})"
        )

    # 롤링이 아직 없는 초기 라운드 — 고정만으로 판단한다
    if d_fixed > noise:
        return PromotionVerdict(True, f"고정 골든셋 개선 (Δ {d_fixed:+.3f})")
    return PromotionVerdict(
        False, f"개선이 시드 노이즈 이하입니다 (Δ {d_fixed:+.3f} ≤ {noise:.3f})"
    )


# --------------------------------------------------------------------------------------
# 5. 라운드 합성 배분 — 합성의 장기 역할은 양이 아니라 불균형 해소
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundPlan:
    synthetic_total: int
    per_class: dict[str, int]
    note: str


def synthetic_ratio(real_total: int, *, full_below: int = 10, none_above: int = 200) -> float:
    """실제 결함 수 → 합성 비율(0~1). 실제가 쌓일수록 합성을 줄인다(설계 §2 규약 3).

    ``full_below`` 이하면 1.0(거의 전부 합성), ``none_above`` 이상이면 하한 0.15 로 수렴한다 —
    0 으로 만들지 않는 이유는 **희소 클래스·희귀 조건 보강**이라는 장기 역할이 남기 때문이다.
    """
    if real_total <= full_below:
        return 1.0
    if real_total >= none_above:
        return 0.15
    span = none_above - full_below
    t = (real_total - full_below) / span
    return 1.0 - t * (1.0 - 0.15)


def round_plan(
    real_counts: Mapping[str, int],
    *,
    total: int | None = None,
    target_per_class: int | None = None,
) -> RoundPlan:
    """클래스별 실제 조각 수 → 이번 라운드에 만들 합성 장수 배분.

    부족분(``target − real``)에 비례해 나눈다. 새 유형 1~2 장이 들어온 라운드에서 합성이 그쪽으로 몰리는데,
    그게 **부트스트랩**이고 루프에서 Graft 의 값어치가 가장 큰 순간이다(설계 §2b.5(4)).
    """
    counts = {k: max(0, int(v)) for k, v in real_counts.items()}
    if not counts:
        return RoundPlan(0, {}, "클래스가 없습니다")

    real_total = sum(counts.values())
    target = target_per_class if target_per_class is not None else max(counts.values())
    deficits = {k: max(0, target - v) for k, v in counts.items()}
    deficit_total = sum(deficits.values())

    if total is None:
        total = round(real_total * synthetic_ratio(real_total))
    total = max(0, int(total))

    if deficit_total == 0:
        # 이미 균형 — 고르게 나눈다
        per = {k: total // len(counts) for k in counts}
        for k in sorted(counts)[: total % len(counts)]:
            per[k] += 1
        return RoundPlan(total, per, f"클래스가 균형 상태입니다(각 {target}) — 고르게 배분")

    per_class: dict[str, int] = {}
    assigned = 0
    for k in sorted(deficits):
        share = int(total * deficits[k] / deficit_total)
        per_class[k] = share
        assigned += share
    # 남은 장수는 부족분이 큰 클래스부터
    for k in sorted(deficits, key=lambda k: (-deficits[k], k)):
        if assigned >= total:
            break
        per_class[k] += 1
        assigned += 1

    scarce = [k for k, v in counts.items() if v < target / 2]
    note = (
        f"희소 클래스에 집중: {', '.join(sorted(scarce))}"
        if scarce
        else f"부족분 비례 배분(목표 각 {target})"
    )
    return RoundPlan(total, per_class, note)


# --------------------------------------------------------------------------------------
# 6. 트리거 — 지금 돌 때인가, 아니면 왜 안 도는가 (설계 §2b.3, T14)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TriggerPolicy:
    """언제 새 라운드를 여는가. 실무 기본값은 **양 + 최소 간격**("N개 모이면, 단 7일에 한 번 이하").

    기본값은 **전부 0 = 제한 없음**이다 — 임계값은 현장마다 다르고(설계 §8 확인 게이트) 기본값이 라운드를
    막으면 "왜 안 도는지" 모르는 사람이 먼저 생긴다. 숫자는 `loop.yaml` 의 `trigger` 에서 사람이 정한다.
    """

    #: 지난 라운드 뒤로 **보관함에 들어온 조각** 수(0 = 이 기준 안 봄)
    min_labels: int = 0
    #: 아직 스코어링하지 않은 **현장 이미지** 장수(0 = 이 기준 안 봄)
    min_images: int = 0
    #: 마지막 라운드로부터 최소 이만큼은 지나야 한다(양 기준이 너무 자주 넘기지 않게)
    min_interval_hours: float = 0.0
    #: 이만큼 지나면 양과 무관하게 한 바퀴 돈다(드리프트 감시 — 없으면 양만 본다)
    max_interval_hours: float | None = None


@dataclass(frozen=True)
class TriggerState:
    """판정에 필요한 사실만. 파일·경로는 호출부가 읽어서 넣는다(이 함수는 순수)."""

    has_round: bool = False
    #: 끝나지 않은 라운드가 있다 — 이어 가는 것은 트리거가 막지 않는다
    in_progress: bool = False
    new_labels: int = 0
    new_images: int = 0
    #: 마지막으로 **끝난** 라운드로부터 지난 시간. ``None`` = 모른다(막지 않는다)
    hours_since: float | None = None


@dataclass(frozen=True)
class TriggerDecision:
    start: bool
    reason: str


def should_start_round(state: TriggerState, policy: TriggerPolicy) -> TriggerDecision:
    """``(돌까?, 왜)`` — **이유를 항상 남긴다**. 조용히 안 도는 루프가 제일 나쁘다(설계 §2b.3).

    순서에 뜻이 있다: 진행 중인 라운드 → 첫 라운드 → 최대 간격(양 무시) → 최소 간격 → 양.
    """
    if state.in_progress:
        return TriggerDecision(True, "진행 중인 라운드를 이어 갑니다")
    if not state.has_round:
        return TriggerDecision(True, "첫 라운드입니다")

    hours = state.hours_since
    if policy.max_interval_hours is not None and (
        hours is None or hours >= policy.max_interval_hours
    ):
        since = "모름" if hours is None else f"{hours:.1f}시간"
        return TriggerDecision(
            True,
            f"마지막 라운드로부터 {since} — 최대 간격 {policy.max_interval_hours:.1f}시간을 넘었습니다",
        )
    if policy.min_interval_hours > 0 and hours is not None and hours < policy.min_interval_hours:
        return TriggerDecision(
            False,
            f"마지막 라운드로부터 {hours:.1f}시간 — 최소 간격 {policy.min_interval_hours:.1f}시간이 안 됐습니다",
        )

    if policy.min_labels <= 0 and policy.min_images <= 0:
        return TriggerDecision(True, "양 기준이 없어 바로 돕니다")

    fired: list[str] = []
    if policy.min_labels > 0 and state.new_labels >= policy.min_labels:
        fired.append(f"새 조각 {state.new_labels}개 ≥ {policy.min_labels}")
    if policy.min_images > 0 and state.new_images >= policy.min_images:
        fired.append(f"새 이미지 {state.new_images}장 ≥ {policy.min_images}")
    if fired:
        return TriggerDecision(True, " · ".join(fired))

    parts: list[str] = []
    if policy.min_labels > 0:
        parts.append(f"새 조각 {state.new_labels}/{policy.min_labels}개")
    if policy.min_images > 0:
        parts.append(f"새 이미지 {state.new_images}/{policy.min_images}장")
    return TriggerDecision(False, " · ".join(parts) + " — 아직 양이 안 찼습니다")


# --------------------------------------------------------------------------------------
# 7. 자동 정지 — 망가진 루프는 망가진 채로 계속 돈다 (설계 §6.5, T12)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundOutcome:
    """한 라운드가 남긴 사실. 원장 ``round_end`` 한 줄에서 읽어 오고, **파일은 호출부가 읽는다**.

    ``drafted``·``corrected`` 는 그 라운드가 끝난 **시점의 누계**다(사람은 라운드가 끝난 뒤에 마스크를
    다듬으므로 라운드 안에서는 잴 수 없다). 그래서 판정은 **라운드 사이의 차이**로 한다 — `circuit_break`.
    """

    round: int
    promoted: bool = False
    metric: float | None = None
    #: 이 라운드에 보관함으로 편입된 조각 수(`accept` 단계의 결과)
    intake: int = 0
    #: 모델 초안(`mask_origin: pred:*`)으로 들어온 조각 **누계** — 사람 수정률의 분모
    drafted: int = 0
    #: 그중 사람이 다듬은 것(`manual:*`) **누계** — 분자
    corrected: int = 0
    #: 보관함 클래스별 조각 수(분포 변화 감시)
    per_class: Mapping[str, int] = field(default_factory=dict)
    bootstrap: bool = False


@dataclass(frozen=True)
class CorrectionStats:
    """사람 수정률 (설계 §2 규약 4) — **0 으로 수렴하면 모델이 좋아진 게 아니라 아무도 안 보는 중**이다."""

    drafted: int = 0
    corrected: int = 0

    @property
    def rate(self) -> float | None:
        """분모가 0 이면 ``None`` — **모르는 것과 0 은 다르다**(들어온 초안이 없으면 판정하지 않는다)."""
        return (self.corrected / self.drafted) if self.drafted > 0 else None

    def text(self) -> str:
        rate = self.rate
        if rate is None:
            return "사람 수정률: 모델 초안이 들어온 적이 없습니다"
        return f"사람 수정률 {rate:.0%} (모델 초안 {self.drafted}개 중 {self.corrected}개를 다듬음)"


def correction_window(history: Sequence[RoundOutcome], rounds: int = 0) -> CorrectionStats:
    """최근 ``rounds`` 라운드 **사이에** 들어온 초안과 그중 다듬어진 수. ``rounds`` ≤ 0 이면 전 구간.

    누계의 차이를 쓰는 이유: 사람은 라운드가 끝난 **뒤에** 보관함에서 마스크를 다듬는다. 누계 비율만 보면
    초기에 열심히 고친 이력이 "지금 아무도 안 본다"를 영원히 가려 주고, 반대로 자동 정지를 한 번 해제한
    직후에 옛 누계가 그대로 다시 정지를 부른다(해제가 아무것도 사 주지 않는다).
    차이가 음수면(조각을 지웠다) 0 으로 본다 — 삭제는 이 지표의 관심사가 아니다.

    **견줄 앞 라운드가 없으면 빈 것**(``rate`` = ``None``)이다 — 모르는 것과 0 은 다르다.
    """
    window = list(history) if rounds <= 0 else list(history[-(rounds + 1) :])
    if len(window) < 2:
        return CorrectionStats()
    drafted = corrected = 0
    for previous, current in pairwise(window):
        drafted += max(0, current.drafted - previous.drafted)
        corrected += max(0, current.corrected - previous.corrected)
    return CorrectionStats(drafted=drafted, corrected=corrected)


def class_shift(before: Mapping[str, int], after: Mapping[str, int]) -> float | None:
    """두 클래스 분포의 거리(0~1, total variation). 한쪽이 비면 ``None``(모르면 막지 않는다).

    **수가 아니라 비율**을 본다 — 보관함은 라운드마다 커지므로 개수 차이는 언제나 크다. 여기서 보고 싶은
    것은 "무엇이 들어오는 비중이 달라졌는가"(로트·공정 변화)다.
    """
    total_b = sum(max(0, int(v)) for v in before.values())
    total_a = sum(max(0, int(v)) for v in after.values())
    if total_b <= 0 or total_a <= 0:
        return None
    keys = set(before) | set(after)
    return 0.5 * sum(
        abs(max(0, int(before.get(k, 0))) / total_b - max(0, int(after.get(k, 0))) / total_a)
        for k in keys
    )


@dataclass(frozen=True)
class BreakerPolicy:
    """자동 정지 기준 (설계 §6.5). **기본값은 전부 0 = 제한 없음**이다.

    T14 트리거와 같은 규율이다 — 실무 임계값은 현장마다 다르고(설계 §8 확인 게이트) 코드가 정한 숫자가
    루프를 세우면 "왜 멈췄는지" 모르는 사람이 먼저 생긴다. 숫자는 `loop.yaml` 의 `breaker` 에서 정한다.
    """

    #: 이만큼 연속으로 승급이 없으면 정지(0 = 안 봄)
    stale_rounds: int = 0
    #: 사람 수정률 하한 — 최근 구간이 이보다 낮으면 정지(0 = 안 봄)
    min_correction_rate: float = 0.0
    #: 수정률을 몇 라운드 구간으로 볼지
    correction_rounds: int = 2
    #: 보관함 클래스 분포가 이보다 많이 바뀌면 정지(0~1, 0 = 안 봄)
    max_class_shift: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.stale_rounds > 0 or self.min_correction_rate > 0 or self.max_class_shift > 0


@dataclass(frozen=True)
class BreakerVerdict:
    """``(멈출까?, 왜)`` — 트리거와 같이 **이유를 항상 돌려준다**."""

    tripped: bool
    reason: str
    kinds: tuple[str, ...] = ()


def circuit_break(
    history: Sequence[RoundOutcome], policy: BreakerPolicy | None = None
) -> BreakerVerdict:
    """루프를 멈춰야 하나 (설계 §6.5).

    **멈추는 건 실패가 아니라 "데이터 말고 다른 걸 바꿀 때"(조명·해상도·모델)라는 신호다.** 자동 루프는
    망가져도 계속 돌기 때문에 이 판정이 필요하다.

    ``history`` 는 **견줄 수 있는 구간**만 들어와야 한다 — 기준선 재설정(T15) 앞의 라운드는 호출부가
    걸러 낸다(그 지점 앞뒤로는 점수를 비교하지 않는다는 규약).
    """
    policy = policy or BreakerPolicy()
    if not policy.enabled:
        return BreakerVerdict(False, "자동 정지 기준이 없습니다")
    if not history:
        return BreakerVerdict(False, "견줄 라운드가 없습니다")

    reasons: list[str] = []
    kinds: list[str] = []
    notes: list[str] = []

    if policy.stale_rounds > 0:
        stale = 0
        for outcome in reversed(history):
            if outcome.promoted:
                break
            stale += 1
        if stale >= policy.stale_rounds:
            reasons.append(
                f"{stale}라운드 연속 승급 없음 (기준 {policy.stale_rounds}회) — "
                "데이터를 더 넣는 것으로는 나아지지 않습니다(조명·해상도·모델을 보세요)"
            )
            kinds.append("stale")
        else:
            notes.append(f"연속 유지 {stale}/{policy.stale_rounds}회")

    if policy.min_correction_rate > 0:
        stats = correction_window(history, policy.correction_rounds)
        rate = stats.rate
        if rate is not None and rate <= policy.min_correction_rate:
            reasons.append(
                f"사람 수정률 {rate:.0%} ≤ 기준 {policy.min_correction_rate:.0%} "
                f"(최근 {policy.correction_rounds}라운드 · 모델 초안 {stats.drafted}개 중 "
                f"{stats.corrected}개만 다듬음) — 모델이 좋아진 게 아니라 아무도 안 보고 있을 수 있습니다"
            )
            kinds.append("correction")
        else:
            notes.append(stats.text())

    if policy.max_class_shift > 0 and len(history) >= 2:
        shift = class_shift(history[-2].per_class, history[-1].per_class)
        if shift is not None and shift > policy.max_class_shift:
            reasons.append(
                f"보관함 클래스 분포가 {shift:.2f} 만큼 바뀌었습니다 "
                f"(기준 {policy.max_class_shift:.2f}) — 로트·공정이 바뀌었는지 확인하세요"
            )
            kinds.append("class_shift")
        elif shift is not None:
            notes.append(f"클래스 분포 변화 {shift:.2f}/{policy.max_class_shift:.2f}")

    if reasons:
        return BreakerVerdict(True, " · ".join(reasons), tuple(kinds))
    return BreakerVerdict(False, " · ".join(notes) or "자동 정지 기준에 걸린 것이 없습니다")

"""루프의 순수 로직 (설계 `v1.x-training-loop.md` §3) — Qt·torch·경로·파일 IO 의존 0.

여기 있는 함수들이 "루프가 무엇을 사람에게 보내고, 무엇을 자동으로 받아들이고, 언제 승급하는가"를 정한다.
전부 순수 함수라 **`adapters/noop` 더미만으로 루프 전체가 테스트된다** — 더미로 안 돌면 설계가 틀린 것이다.

무작위성은 호출자가 넘긴 ``rng``(numpy Generator)만 쓴다. 전역 ``np.random`` 은 쓰지 않는다(코어 규약).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

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

"""`anograft.loop.policy` — 루프의 순수 로직. torch·어댑터 없이 전부 여기서 고정된다."""

from __future__ import annotations

import numpy as np
import pytest

from anograft.loop.policy import (
    ReviewMix,
    agreement,
    gate,
    iou,
    partition_by_gate,
    round_plan,
    select_for_review,
    should_promote,
    synthetic_ratio,
)

# --------------------------------------------------------------------- 검토 대상 고르기


def _scored(n: int = 20) -> list[tuple[str, float]]:
    return [(f"i{k:02d}", k / (n - 1)) for k in range(n)]


def test_boundary_picks_are_nearest_to_threshold() -> None:
    """경계 몫은 임계값에 가장 가까운 것부터 — 정보량이 거기 있다."""
    picks = select_for_review(_scored(), threshold=0.5, n=4, mix=ReviewMix(1, 0, 0))
    assert [p.reason for p in picks] == ["경계"] * 4
    scores = [p.score for p in picks]
    assert max(abs(s - 0.5) for s in scores) <= 0.12


def test_confident_mix_takes_high_scores_above_threshold() -> None:
    picks = select_for_review(_scored(), threshold=0.5, n=3, mix=ReviewMix(0, 1, 0))
    assert [p.reason for p in picks] == ["확신"] * 3
    assert picks[0].score == pytest.approx(1.0)
    assert all(p.score >= 0.5 for p in picks)


def test_mix_is_respected_and_ids_unique() -> None:
    picks = select_for_review(_scored(), threshold=0.5, n=10, mix=ReviewMix(0.6, 0.2, 0.2))
    assert len(picks) == 10
    assert len({p.item_id for p in picks}) == 10
    kinds = {r: sum(1 for p in picks if p.reason == r) for r in ("경계", "확신", "무작위")}
    assert kinds["경계"] == 6
    assert kinds["확신"] == 2
    assert kinds["무작위"] == 2


def test_random_share_uses_injected_rng_only() -> None:
    """무작위 몫도 rng 를 주면 재현된다 — 전역 난수를 쓰지 않는다(코어 규약)."""
    a = select_for_review(_scored(), threshold=0.5, n=8, rng=np.random.default_rng(7))
    b = select_for_review(_scored(), threshold=0.5, n=8, rng=np.random.default_rng(7))
    assert [p.item_id for p in a] == [p.item_id for p in b]


def test_without_rng_is_deterministic() -> None:
    a = select_for_review(_scored(), threshold=0.5, n=8)
    b = select_for_review(_scored(), threshold=0.5, n=8)
    assert [p.item_id for p in a] == [p.item_id for p in b]


def test_select_handles_small_pool_and_zero_n() -> None:
    assert select_for_review([], threshold=0.5, n=5) == []
    assert select_for_review(_scored(3), threshold=0.5, n=0) == []
    picks = select_for_review(_scored(3), threshold=0.5, n=99)
    assert len({p.item_id for p in picks}) == 3


# --------------------------------------------------------------------- 자동 편입 게이트


def test_gate_passes_when_confidence_missing() -> None:
    """confidence 가 None = 사람이 그린 정확한 마스크 → 통과."""
    assert gate(None, 0.8) is True


def test_gate_threshold_and_nan() -> None:
    assert gate(0.9, 0.8) is True
    assert gate(0.8, 0.8) is True
    assert gate(0.42, 0.8) is False
    assert gate(float("nan"), 0.8) is False


def test_partition_by_gate() -> None:
    auto, queue = partition_by_gate([("a", 0.9), ("b", 0.4), ("c", None)], 0.8)
    assert auto == ["a", "c"]
    assert queue == ["b"]


# --------------------------------------------------------------------- 교차 검증


def test_iou_basics() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(5 * 10 / (100 + 100 - 50))
    assert iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


def test_agreement_both_models_point_at_same_place() -> None:
    """교집합만 있으면 자동 NG 확정 후보."""
    a = agreement([(10, 10, 20, 20)], [(12, 11, 19, 21)])
    assert a.matched == [(0, 0)]
    assert a.agreed is True
    assert a.needs_human is False


def test_agreement_disagreement_goes_to_human() -> None:
    a = agreement([(10, 10, 20, 20)], [(200, 200, 15, 15)])
    assert a.matched == []
    assert a.only_a == [0] and a.only_b == [0]
    assert a.needs_human is True
    assert a.agreed is False


def test_agreement_is_one_to_one_greedy() -> None:
    """한 박스가 둘에 매칭되지 않는다."""
    a = agreement(
        [(0, 0, 10, 10), (0, 0, 9, 9)],
        [(0, 0, 10, 10)],
    )
    assert len(a.matched) == 1
    assert a.matched[0][1] == 0
    assert len(a.only_a) == 1


def test_agreement_empty_sides() -> None:
    assert agreement([], []).agreed is False
    assert agreement([(0, 0, 5, 5)], []).only_a == [0]
    assert agreement([], [(0, 0, 5, 5)]).only_b == [0]


# --------------------------------------------------------------------- 승급 판정


def test_promote_requires_rolling_gain_beyond_noise() -> None:
    v = should_promote(
        fixed_champion=0.39,
        fixed_challenger=0.41,
        rolling_champion=0.41,
        rolling_challenger=0.46,
        noise=0.04,
    )
    assert v.promote is True
    assert "롤링" in v.reason


def test_hold_when_gain_is_within_seed_noise() -> None:
    """BENCHMARKS §2 의 시드 요동 ±0.04 보다 작으면 개선이 아니다."""
    v = should_promote(
        fixed_champion=0.34,
        fixed_challenger=0.35,
        rolling_champion=0.34,
        rolling_challenger=0.35,
        noise=0.04,
    )
    assert v.promote is False
    assert "노이즈" in v.reason


def test_fixed_regression_blocks_even_with_rolling_gain() -> None:
    """고정 골든셋은 회귀 감시 — 떨어지면 무조건 정지."""
    v = should_promote(
        fixed_champion=0.41,
        fixed_challenger=0.30,
        rolling_champion=0.40,
        rolling_challenger=0.60,
        noise=0.04,
    )
    assert v.promote is False
    assert "회귀" in v.reason


def test_early_round_without_rolling_uses_fixed_only() -> None:
    assert should_promote(fixed_champion=0.30, fixed_challenger=0.37).promote is True
    assert should_promote(fixed_champion=0.30, fixed_challenger=0.32).promote is False


# --------------------------------------------------------------------- 라운드 배분


def test_synthetic_ratio_shrinks_as_real_data_accumulates() -> None:
    assert synthetic_ratio(3) == 1.0
    assert synthetic_ratio(500) == pytest.approx(0.15)
    mid = synthetic_ratio(100)
    assert 0.15 < mid < 1.0
    assert synthetic_ratio(50) > synthetic_ratio(150)


def test_round_plan_concentrates_on_scarce_classes() -> None:
    """새 유형 부트스트랩 — 합성이 희소 클래스로 몰리는 게 요점."""
    plan = round_plan({"blowhole": 2, "break": 80, "crack": 60}, total=100)
    assert plan.synthetic_total == 100
    assert sum(plan.per_class.values()) == 100
    assert plan.per_class["blowhole"] > plan.per_class["crack"] > plan.per_class["break"]
    assert "희소" in plan.note


def test_round_plan_balanced_classes_spread_evenly() -> None:
    plan = round_plan({"a": 10, "b": 10}, total=10)
    assert plan.per_class == {"a": 5, "b": 5}
    assert "균형" in plan.note


def test_round_plan_default_total_follows_ratio() -> None:
    plan = round_plan({"a": 4, "b": 4})
    assert plan.synthetic_total == 8  # 실제 8장 · 비율 1.0
    big = round_plan({"a": 300, "b": 300})
    assert big.synthetic_total == round(600 * 0.15)


def test_round_plan_empty() -> None:
    plan = round_plan({})
    assert plan.synthetic_total == 0 and plan.per_class == {}

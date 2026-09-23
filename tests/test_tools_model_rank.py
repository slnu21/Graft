"""``tools/bench_model_rank.py`` — 순수 부분(순위·Spearman·정상 분할·표).

anomalib 도 학습도 필요 없다. 이 실험의 주장("합성셋만으로 모델을 고를 수 있다")이 숫자로 성립하는지는
실측이 말하지만, **순위를 세는 방식**이 틀리면 실측이 의미가 없으므로 여기서 고정한다.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tests.fixtures import load_tool

bm = load_tool("bench_model_rank")


def test_ranks_are_one_based_and_average_ties() -> None:
    assert bm.ranks([0.9, 0.5, 0.7]) == [1.0, 3.0, 2.0]  # 큰 값이 1등
    assert bm.ranks([0.5, 0.5, 0.1]) == [1.5, 1.5, 3.0]  # 동점은 평균 순위
    assert bm.ranks([]) == []


def test_spearman_agreement_and_disagreement() -> None:
    assert bm.spearman([0.9, 0.7, 0.5], [0.8, 0.6, 0.4]) == pytest.approx(1.0)
    assert bm.spearman([0.9, 0.7, 0.5], [0.4, 0.6, 0.8]) == pytest.approx(-1.0)
    assert bm.spearman([0.9, 0.7, 0.5, 0.3], [0.8, 0.9, 0.3, 0.4]) == pytest.approx(
        0.6
    )  # d²=4 → 1-24/60


def test_spearman_is_nan_when_ranking_is_meaningless() -> None:
    """한쪽이 전부 동점이면 순위가 없다 — 0.0 으로 보고하면 '상관 없음'으로 읽혀 오해가 된다."""
    assert math.isnan(bm.spearman([0.5, 0.5, 0.5], [0.1, 0.2, 0.3]))
    assert math.isnan(bm.spearman([0.5], [0.3]))
    with pytest.raises(ValueError, match="길이"):
        bm.spearman([0.1, 0.2], [0.3])


def test_split_normals_never_overlaps(tmp_path: Path) -> None:
    """§5 함정: 학습 정상과 합성 대상이 겹치면 이상맵이 비현실적으로 깨끗해진다(낙관 편향)."""
    paths = [tmp_path / f"{i:03d}.png" for i in range(10)]
    train, targets = bm.split_normals(paths, 6, 3)
    assert train == paths[:6] and targets == paths[6:9]
    assert not set(train) & set(targets)
    # 모자라면 있는 만큼만 — 겹치게 채우지 않는다
    train, targets = bm.split_normals(paths, 8, 5)
    assert len(train) == 8 and len(targets) == 2 and not set(train) & set(targets)


def test_rank_table_reports_rho_and_rows() -> None:
    results = {
        "padim": {"real/image_AUROC": 0.80, "syn/image_AUROC": 0.71},
        "patchcore": {"real/image_AUROC": 0.95, "syn/image_AUROC": 0.88},
    }
    md = bm.rank_table(results, "image_AUROC")
    assert "| patchcore | 0.950 (1) | 0.880 (1) |" in md
    assert "| padim | 0.800 (2) | 0.710 (2) |" in md
    assert "Spearman ρ(실제, 합성) = **1.000**" in md


def test_rank_table_missing_metric_is_nan_not_zero() -> None:
    """지표 키가 없으면 nan — 0 으로 채우면 '그 모델이 최악'이라는 거짓 순위가 된다."""
    md = bm.rank_table({"a": {"real/x": 0.5}, "b": {"real/x": 0.7, "syn/x": 0.6}}, "x")
    assert "nan" in md

"""1단계 ``bank`` 소스 선택 — 클래스 확률·결정성·fail-soft(deps 없음·빈 클래스)."""

from __future__ import annotations

import numpy as np

from anograft.core import registry
from anograft.core.recipe import BankSourceConfig
from anograft.core.stages.source import BankSource
from tests.fixtures import context, line_defect, memory_bank


def _bank():
    return memory_bank([line_defect(12, 3), line_defect(14, 3), line_defect(10, 5, cls="dent")])


def test_registered_and_buildable_without_deps() -> None:
    st = registry.build("source", BankSourceConfig(), {})
    assert isinstance(st, BankSource)
    out = st.apply(context(seed=1))
    assert out.source is None
    assert out.log["source"]["skipped"] and "bank" in out.log["source"]["reason"]
    assert out.warnings and out.warnings[0].startswith("source:")


def test_picks_from_bank_and_logs_identity() -> None:
    bank = _bank()
    st = BankSource(BankSourceConfig(), {"bank": bank, "class_ids": bank.class_ids})
    out = st.apply(context(seed=3))
    assert out.source is not None and out.source in bank.by_class(out.source.cls)
    log = out.log["source"]
    assert log["source_id"] == out.source.id and log["class"] == out.source.cls
    assert log["class_id"] == bank.class_ids[out.source.cls]
    assert log["mask_origin"] == "png" and log["method"] == "bank"


def test_uniform_default_and_class_probs_respected() -> None:
    bank = _bank()
    n = 600
    uniform = BankSource(BankSourceConfig(), {"bank": bank})
    counts = {"dent": 0, "scratch": 0}
    ctx = context(seed=5)
    for _ in range(n):
        out = uniform.apply(ctx.begin_defect())
        assert out.source is not None
        counts[out.source.cls] += 1
        ctx = out
    assert abs(counts["dent"] / n - 0.5) < 0.08  # 균등 (클래스 2개)

    skewed = BankSource(
        BankSourceConfig(), {"bank": bank, "class_probs": {"dent": 0.9, "scratch": 0.1}}
    )
    counts = {"dent": 0, "scratch": 0}
    ctx = context(seed=5)
    for _ in range(n):
        out = skewed.apply(ctx.begin_defect())
        counts[out.source.cls] += 1  # type: ignore[union-attr]
        ctx = out
    assert counts["dent"] / n > 0.8


def test_cfg_classes_restricts_when_no_probs() -> None:
    bank = _bank()
    st = BankSource(BankSourceConfig(classes=["dent"]), {"bank": bank})
    ctx = context(seed=9)
    for _ in range(20):
        ctx = st.apply(ctx.begin_defect())
        assert ctx.source is not None and ctx.source.cls == "dent"


def test_deterministic_for_same_rng() -> None:
    bank = _bank()
    st = BankSource(BankSourceConfig(), {"bank": bank})
    a = [st.apply(context(seed=s)).source.id for s in range(10)]  # type: ignore[union-attr]
    b = [st.apply(context(seed=s)).source.id for s in range(10)]  # type: ignore[union-attr]
    assert a == b and len(set(a)) > 1


def test_empty_class_is_skipped_not_raised() -> None:
    bank = _bank()
    st = BankSource(BankSourceConfig(), {"bank": bank, "class_probs": {"ghost": 1.0}})
    out = st.apply(context(seed=0))
    assert out.source is None and "ghost" in out.log["source"]["reason"]
    # 확률 합 0 → 뽑을 클래스 없음
    st0 = BankSource(BankSourceConfig(), {"bank": bank, "class_probs": {"dent": 0.0}})
    assert st0.apply(context(seed=0)).source is None


def test_source_image_is_not_copied_or_mutated() -> None:
    bank = _bank()
    st = BankSource(BankSourceConfig(), {"bank": bank})
    ctx = context(seed=2)
    before = {s.id: s.image.copy() for s in bank.sources()}
    out = st.apply(ctx)
    assert out.source is not None
    assert np.array_equal(out.source.image, before[out.source.id])
    assert np.array_equal(out.composite, ctx.composite)

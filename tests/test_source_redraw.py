"""``source.redraw_on_empty``(v0.8.x) — 기하 변환 뒤 마스크가 비면(< 4 px) 같은 자리에서 소스를 다시 뽑는다.
실패했을 때만 rng 를 더 쓰므로 성공 경로(골든)는 불변."""

from __future__ import annotations

from anograft.core import recipe as R
from anograft.core.pipeline import Pipeline
from tests.fixtures import disk_target, line_defect, memory_bank, pipeline_deps


def _recipe(redraw: int, *, scale: tuple[float, float] = (0.05, 0.05)) -> R.Recipe:
    return R.Recipe.from_dict(
        {
            "version": 1,
            "name": "hard-paste",
            "seed": 11,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1, "defects_per_image": [1, 1]},
            "pipeline": {
                "preset": "hard-paste",
                "source": {"method": "bank", "redraw_on_empty": redraw, "min_sources_warn": 0},
                "geometry": {"scale": list(scale), "rotate": [0.0, 0.0], "flip": "none"},
                "placement": {"roi": {"method": "none"}, "margin_px": 2},
            },
        }
    )


def test_schema_default_and_bounds() -> None:
    assert R.BankSourceConfig().redraw_on_empty == 2
    r = _recipe(0)
    assert r.pipeline.source.redraw_on_empty == 0
    import pytest

    with pytest.raises(ValueError):
        _recipe(11)


def test_redraw_recovers_when_a_tiny_source_vanishes() -> None:
    """은행 = 작은 선(6×1, 0.05 배로 줄이면 0 px) + 큰 원판급 선(200×40 → 10×2 = 20 px). 대상은 클래스 하나라
    처음 뽑힌 소스가 작은 쪽이면 geometry 가 비고 → 재추첨이 큰 쪽을 뽑을 때까지 반복. 재추첨 0 이면 그 이미지는 skipped."""
    tiny = line_defect(6, 1, margin=2, cls="scratch")
    big = line_defect(200, 40, margin=4, cls="scratch")
    bank = memory_bank([tiny, big])
    target = disk_target(256)
    # 재추첨 없이: 어떤 시드에서든 작은 소스가 뽑히면 skipped. 여기서는 skipped 가 나오는 index 를 찾는다
    off = Pipeline.from_recipe(_recipe(0), pipeline_deps(_recipe(0), bank))
    failing = next((i for i in range(40) if off.run_one(target, i).status == "skipped"), None)
    assert failing is not None, "작은 소스가 한 번은 뽑혀야 테스트가 의미 있다"
    r_off = off.run_one(target, failing)
    assert r_off.status == "skipped" and any("geometry:" in w for w in r_off.sidecar["warnings"])
    # 재추첨 켜면 같은 index 가 살아난다(큰 소스로) · 사이드카 defects 는 요청 수(1) · source.redraws 기록 · 경고 남음
    on = Pipeline.from_recipe(_recipe(2), pipeline_deps(_recipe(2), bank))
    r_on = on.run_one(target, failing)
    if r_on.status == "ok":
        d = r_on.sidecar["defects"]
        assert (
            len(d) == 1
            and d[0]["source"]["redraws"] >= 1
            and d[0]["source"]["source_id"] == "scratch/001"
        )
        assert any("소스 재추첨" in w for w in r_on.sidecar["warnings"])
    else:  # 두 번 다 작은 소스가 뽑힌 경우 — 경고에 재추첨 2/2 가 남는다
        assert any("재추첨 2/2" in w for w in r_on.sidecar["warnings"])
    # 결정성
    again = Pipeline.from_recipe(_recipe(2), pipeline_deps(_recipe(2), bank)).run_one(
        target, failing
    )
    assert again.status == r_on.status and again.sidecar["defects"] == r_on.sidecar["defects"]


def test_success_path_consumes_no_extra_rng() -> None:
    """실패가 없으면 redraw_on_empty 0 과 2 의 결과가 바이트 동일(골든 불변 조건)."""
    bank = memory_bank([line_defect(18, 4), line_defect(10, 6, cls="dent")])
    target = disk_target(128)
    for i in range(4):
        a = Pipeline.from_recipe(
            _recipe(0, scale=(0.9, 1.1)), pipeline_deps(_recipe(0, scale=(0.9, 1.1)), bank)
        ).run_one(target, i)
        b = Pipeline.from_recipe(
            _recipe(2, scale=(0.9, 1.1)), pipeline_deps(_recipe(2, scale=(0.9, 1.1)), bank)
        ).run_one(target, i)
        assert a.status == b.status == "ok"
        assert (a.image == b.image).all() and (a.gt_mask == b.gt_mask).all()
        assert "redraws" not in a.sidecar["defects"][0]["source"]

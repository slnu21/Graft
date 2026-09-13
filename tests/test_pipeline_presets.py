"""프리셋 단위 엔드투엔드 — poisson-graft · hard-paste 가 ``Pipeline.from_recipe`` + 메모리 은행(실물 BankSource)으로 끝까지 돈다.
gray/color 양쪽, 사이드카 구조, 결정성. alpha-paste(reinhard)·multiband-graft(multiband)는 아직 미구현이라 명확한 예외
(stages-more-methods에서 골든 8장과 함께 이 목록에 들어온다)."""

from __future__ import annotations

import numpy as np
import pytest

from anograft.core import recipe as R
from anograft.core import registry
from anograft.core.pipeline import Pipeline
from tests.fixtures import disk_target, line_defect, memory_bank, pipeline_deps

IMPLEMENTED = ["poisson-graft", "hard-paste"]


def _recipe(preset: str, seed: int = 20260914) -> R.Recipe:
    return R.Recipe.from_dict(
        {
            "version": 1,
            "name": preset,
            "seed": seed,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1, "defects_per_image": [2, 2]},
            "pipeline": {
                "preset": preset,
                "placement": {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 6},
            },
        }
    )


def _pipeline(preset: str) -> Pipeline:
    rec = _recipe(preset)
    bank = memory_bank([line_defect(18, 4), line_defect(10, 6, cls="dent")])
    return Pipeline.from_recipe(rec, pipeline_deps(rec, bank))


@pytest.mark.parametrize("gray", [False, True])
@pytest.mark.parametrize("preset", IMPLEMENTED)
def test_preset_runs_end_to_end(preset: str, gray: bool) -> None:
    p = _pipeline(preset)
    t = disk_target(128, gray=gray)
    r = p.run_one(t, 0)
    assert r.status == "ok", r.warnings
    assert len(r.instances) == 2 and r.gt_mask.max() == 255
    assert r.image.dtype == np.uint8 and r.image.shape == ((128, 128) if gray else (128, 128, 3))
    sc = r.sidecar
    pipe = p.recipe.pipeline
    assert sc["roi"]["method"] == "otsu"
    for d in sc["defects"]:
        assert set(d) == {"source", "geometry", "placement", "blend", "harmonize", "gt"}
        assert d["blend"]["method"] == pipe.blend.method
        assert d["harmonize"]["method"] == pipe.harmonize.method
        assert d["gt"]["area_px"] > 0
    assert sc["degrade"]["method"] == pipe.degrade.method
    assert sc["gtmask"]["policy"] == pipe.gtmask.policy
    assert sc["gtmask"]["area_px_total"] == int(np.count_nonzero(r.gt_mask))
    # GT 인스턴스 합집합 = 전체 GT, 인스턴스는 결함 순번을 안다
    union = np.zeros_like(r.gt_mask)
    for inst in r.instances:
        union = np.maximum(union, inst.mask)
        assert inst.defect_index in (0, 1)
    assert np.array_equal(union, r.gt_mask)


def test_poisson_graft_defaults_do_not_fall_back_on_disk() -> None:
    r = _pipeline("poisson-graft").run_one(disk_target(128), 3)
    assert r.status == "ok"
    for d in r.sidecar["defects"]:
        assert d["blend"] == {
            "method": "poisson",
            "mode": "mixed",
            "mask_dilate_px": 5,
            "fallback": False,
        }
        assert d["harmonize"]["method"] == "stats" and "skipped" not in d["harmonize"]
    assert r.sidecar["degrade"]["method"] == "camera"


@pytest.mark.parametrize("preset", IMPLEMENTED)
def test_preset_is_deterministic(preset: str) -> None:
    p = _pipeline(preset)
    a, b = p.run_one(disk_target(96), 7), p.run_one(disk_target(96), 7)
    assert np.array_equal(a.image, b.image) and np.array_equal(a.gt_mask, b.gt_mask)
    assert a.sidecar == b.sidecar
    c = p.run_one(disk_target(96), 8)
    assert not np.array_equal(a.image, c.image)


def test_hard_paste_changes_only_inside_gt() -> None:
    """hard-paste = paste + none + none: GT(source) 밖은 대상과 바이트 동일해야 한다."""
    t = disk_target(128)
    r = _pipeline("hard-paste").run_one(t, 1)
    assert r.status == "ok"
    outside = r.gt_mask == 0
    assert np.array_equal(r.image[outside], t.image[outside])


@pytest.mark.parametrize(
    ("preset", "stage", "pattern"),
    [
        ("multiband-graft", "blend", r"blend\.multiband"),
        ("alpha-paste", "harmonize", r"harmonize\.reinhard"),
    ],
)
def test_remaining_presets_report_not_implemented(preset: str, stage: str, pattern: str) -> None:
    rec = _recipe(preset)
    with pytest.raises(registry.StageNotImplementedError, match=pattern):
        registry.build(stage, getattr(rec.pipeline, stage))

"""프리셋 단위 엔드투엔드 — 프리셋 4종 전부가 ``Pipeline.from_recipe`` + 메모리 은행(실물 BankSource)으로 끝까지 돈다.
gray/color 양쪽, 사이드카 구조, 결정성. 픽셀 회귀는 ``test_pipeline_golden``(골든 8장)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anograft.core import recipe as R
from anograft.core.pipeline import Pipeline, skip_reason
from anograft.io import imgio
from tests.fixtures import disk_target, line_defect, memory_bank, pipeline_deps

IMPLEMENTED = ["poisson-graft", "hard-paste", "alpha-paste", "multiband-graft"]


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
            "mode": "normal",
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
    ("preset", "blend", "harmonize"),
    [("alpha-paste", "alpha", "reinhard"), ("multiband-graft", "multiband", "histmatch")],
)
def test_new_presets_run_their_methods_without_skip(
    preset: str, blend: str, harmonize: str
) -> None:
    """alpha-paste · multiband-graft: 블렌딩·조화가 실제로 돌았다(skipped 없음, multiband는 levels_used ≥ 1)."""
    r = _pipeline(preset).run_one(disk_target(128), 3)
    assert r.status == "ok"
    for d in r.sidecar["defects"]:
        assert d["blend"]["method"] == blend and "skipped" not in d["blend"]
        assert d["harmonize"]["method"] == harmonize and "skipped" not in d["harmonize"]
        if blend == "multiband":
            assert 1 <= d["blend"]["levels_used"] <= d["blend"]["levels"] == 4


# ---------------------------------------------------------------------------
# skipped 사유 — 근본 원인이 보여야 한다 (KNOWN-ISSUES #1)
# ---------------------------------------------------------------------------


def test_skip_reason_prefers_roi_warning_over_placement_consequence() -> None:
    ws = (
        "roi: mask_dir 로드 실패 x.png — ValueError",
        "placement: s 배치 실패 — ROI 없음 (시도 0, 축소 0)",
    )
    assert skip_reason(ws) == ws[0]
    assert skip_reason(ws[1:]) == ws[1]  # ROI 경고가 없으면 마지막 경고
    assert skip_reason(()) == "배치된 결함 없음"


def test_mask_dir_failure_is_the_skipped_reason_and_resized_mask_works(tmp_path: Path) -> None:
    """mask_dir ROI 로 끝까지: (a) 마스크 파일이 없으면 reason 이 'roi: …' (placement 의 'ROI 없음'이 아니라),
    (b) 대상보다 큰 같은 비율 마스크는 축소되어 합성이 성공하고 사이드카 roi.resized_from 이 남는다."""
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": "hard-paste",
            "seed": 3,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1, "defects_per_image": [1, 1]},
            "pipeline": {
                "preset": "hard-paste",
                "placement": {
                    "roi": {"method": "mask_dir", "path": tmp_path.as_posix()},
                    "margin_px": 4,
                },
            },
        }
    )
    p = Pipeline.from_recipe(rec, pipeline_deps(rec, memory_bank()))
    r = p.run_one(disk_target(96, name="t01.png"), 0)
    assert r.status == "skipped" and r.reason is not None
    assert r.reason.startswith("roi: mask_dir 로드 실패") and r.sidecar["roi"]["failed"] is True
    assert any(w.startswith("placement:") for w in r.warnings)  # 결과 경고는 남되 사유는 원인

    big = np.zeros(
        (192, 192), dtype=np.uint8
    )  # 원본 크기 마스크(2배) → 미리보기 축소 대상에 맞춰진다
    big[40:150, 40:150] = 255
    imgio.write_image(tmp_path / "t01.png", big)
    r2 = p.run_one(disk_target(96, name="t01.png"), 0)
    assert r2.status == "ok" and r2.sidecar["roi"]["resized_from"] == [192, 192]
    assert r2.gt_mask[20:75, 20:75].any() and not r2.gt_mask[:18].any()

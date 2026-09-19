"""골든 회귀 — 설계 §11: 64×64 합성 대상 + 합성 은행, 시드 고정, **프리셋 10종 × gray/color** → ``tests/golden/<preset>-<gray|color>.png``
픽셀 바이트 일치(18 골든; self-cut·perlin-texture·structure-aware-graft 는 v0.4, annulus-graft·dent-graft 는 v0.6 에서 추가). 갱신은
``pytest --update-golden``으로만(``conftest``) — 알고리즘을 의도적으로 바꿨을 때, 데브로그에 사유.

PNG 바이트가 아니라 **디코드한 픽셀 배열**을 비교한다(zlib/OpenCV 버전에 따라 인코딩 바이트는 달라질 수 있다).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anograft.core import recipe as R
from anograft.core.pipeline import Pipeline
from anograft.io import imgio
from tests.fixtures import disk_target, line_defect, memory_bank, pipeline_deps

GOLDEN_DIR = Path(__file__).parent / "golden"
PRESETS = [
    "poisson-graft",
    "hard-paste",
    "alpha-paste",
    "multiband-graft",
    "self-cut",
    "perlin-texture",
    "structure-aware-graft",
    "annulus-graft",
    "dent-graft",
    "relative-paste",
]
SEED = 20260914
SIZE = 64


# 64px 원판(지름 42)에 맞춘 프리셋별 축소 — self-cut 기본 area_ratio(2~15%)는 회전하면 ROI 에 못 들어간다
SOURCE_OVERRIDES: dict[str, dict] = {
    "self-cut": {
        "method": "self-cut",
        "area_ratio": [0.01, 0.03],
        "scar_length_px": [8, 16],
        "scar_width_px": [2, 4],
    },
    "perlin-texture": {"method": "perlin-texture", "size_ratio": [0.25, 0.35]},
}


def _pipeline(preset: str) -> Pipeline:
    pipe: dict = {
        "preset": preset,
        "placement": {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 6},
    }
    if (
        preset == "structure-aware-graft"
    ):  # 프리셋의 grabcut ROI 는 별도 테스트 — 골든은 배치 정렬만 고정
        pipe["placement"]["roi"] = {"method": "grabcut", "erode_px": 2, "work_px": 0}
    if (
        preset == "annulus-graft"
    ):  # 64px 원판(반경 21)에 맞춘 링 — 기본 0.55~0.9·erode 4 는 두께가 0 에 가깝다
        pipe["placement"]["roi"] = {
            "method": "annulus",
            "r_inner": 0.3,
            "r_outer": 1.0,
            "erode_px": 0,
        }
    if preset in SOURCE_OVERRIDES:
        pipe["source"] = SOURCE_OVERRIDES[preset]
    rec = R.Recipe.from_dict(
        {
            "version": 1,
            "name": preset,
            "seed": SEED,
            "inputs": {"bank": "b", "targets": "t"},
            "output": {"root": "o", "count": 1, "defects_per_image": [2, 2]},
            "pipeline": pipe,
        }
    )
    bank = memory_bank([line_defect(18, 4), line_defect(10, 6, cls="dent")])
    return Pipeline.from_recipe(rec, pipeline_deps(rec, bank))


def golden_path(preset: str, gray: bool) -> Path:
    return GOLDEN_DIR / f"{preset}-{'gray' if gray else 'color'}.png"


@pytest.mark.parametrize("gray", [False, True], ids=["color", "gray"])
@pytest.mark.parametrize("preset", PRESETS)
def test_preset_matches_golden(preset: str, gray: bool, update_golden: bool) -> None:
    r = _pipeline(preset).run_one(disk_target(SIZE, gray=gray), 0)
    assert r.status == "ok", r.warnings
    assert len(r.instances) == 2  # 골든은 결함 2개가 다 붙은 결과여야 의미가 있다
    path = golden_path(preset, gray)
    if update_golden:
        path.parent.mkdir(parents=True, exist_ok=True)
        imgio.write_image(path, r.image)
        pytest.skip(f"golden 갱신: {path.name}")
    assert path.exists(), f"골든 없음: {path.name} — `pytest --update-golden`으로 만든다"
    expected, was_gray = imgio.read_image(path)  # 항상 HxWx3 승격
    assert was_gray == gray
    if gray:
        expected = expected[:, :, 0]
    assert expected.shape == r.image.shape and expected.dtype == r.image.dtype
    diff = np.abs(expected.astype(int) - r.image.astype(int))
    assert diff.max() == 0, (
        f"{path.name}: {int((diff > 0).sum())}px 다름 (max {int(diff.max())}) — 의도한 변경이면 --update-golden"
    )


def test_golden_files_are_small_and_complete() -> None:
    """전부 있고 각 ≤ 10KB(커밋 대상)."""
    for preset in PRESETS:
        for gray in (False, True):
            p = golden_path(preset, gray)
            assert p.exists(), p.name
            assert p.stat().st_size <= 10 * 1024, (p.name, p.stat().st_size)

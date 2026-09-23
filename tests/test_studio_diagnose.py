"""`studio.diagnose` — 미리보기 진단 순수 함수. Qt 스튜디오 탭과 웹이 같은 문장을 보게 하는 지점.

원래 `gui/studio/tab.py` 의 private 헬퍼라 테스트가 없었다(탭을 띄워야 했다). 승격(U5a) 하면서
Qt 없이 고정한다 — 특히 **축소본 → 원본 px 되돌리기**(÷ scale)는 틀리면 조용히 엉뚱한 경고가 난다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anograft.bank.importers import yolo as Y
from anograft.studio import diagnose
from anograft.studio.jobs import gt_overlay_bgra, roi_overlay_bgra
from anograft.studio.session import StudioSession, default_recipe
from tests.fixtures import fake_yolo_dataset


@pytest.fixture
def session(tmp_path: Path) -> StudioSession:
    d = fake_yolo_dataset(tmp_path / "ds")
    normals = tmp_path / "normals.txt"
    Y.import_yolo(
        d["images"],
        d["labels"],
        d["names"],
        tmp_path / "bank",
        mask_from="rect",
        list_normals=normals,
    )
    ses = StudioSession(
        default_recipe(bank=(tmp_path / "bank").as_posix(), targets=normals.as_posix())
    )
    ses.set_field(("pipeline", "source", "min_sources_warn"), 1)
    ses.prepare_now()
    return ses


class _Result:
    """사이드카만 있는 가짜 결과 — `defect_lines` 는 사이드카에 이미 있는 것만 읽는다."""

    status = "skipped"
    instances: tuple = ()

    def __init__(self, sidecar: dict) -> None:
        self.sidecar = sidecar


def test_defect_lines_skips_defects_without_gt():
    r = _Result(
        {
            "defects": [
                {"source": {"class": "scratch", "source_id": "a1"}, "blend": {"method": "poisson"}},
                {"source": {"class": "pit", "source_id": "b2"}},  # gt 없음 = 못 붙은 것
                {"gt": {}, "source": {"class": "stain", "source_id": "c3"}, "blend": {}},
            ]
        }
    )
    # 첫 항목도 gt 가 없다 → 실제로 들어간 것은 c3 하나
    assert [d.source_id for d in diagnose.defect_lines(r)] == ["c3"]
    assert diagnose.source_ids(r) == ["c3"]
    assert diagnose.defect_lines(r)[0].cls == "stain"


def test_no_bank_or_no_roi_is_fail_soft():
    r = _Result({})
    assert diagnose.low_confidence_sources(None, r) == []
    assert diagnose.flipped(None, r) == []
    assert diagnose.roi_of([]) is None
    assert diagnose.fit_for_preview(None, default_recipe(), None, 1.0, "t.png") is None
    assert diagnose.fit_warnings(None) == []


def test_fit_undoes_the_preview_shrink(session: StudioSession):
    """축소본 ROI 폭을 배율로 나눠 **원본 px** 로 되돌린다 — 배율 0.5 면 폭이 두 배로 읽혀야 한다."""
    roi = np.zeros((64, 64), np.uint8)
    roi[8:56, 8:56] = 255
    full = diagnose.fit_for_preview(session.prepared, session.recipe, roi, 1.0, "t.png")
    half = diagnose.fit_for_preview(session.prepared, session.recipe, roi, 0.5, "t.png")
    assert full is not None and half is not None
    assert half.min_width == pytest.approx(full.min_width * 2, rel=0.02)


def test_overlays_are_transparent_outside_the_mask():
    mask = np.zeros((20, 20), np.uint8)
    mask[5:15, 5:15] = 255
    for over in (gt_overlay_bgra(mask), roi_overlay_bgra(mask // 255)):
        assert over.shape == (20, 20, 4)
        assert (over[0, 0] == 0).all(), "마스크 밖은 완전 투명"
        assert over[10, 10, 3] > 0
    gt = gt_overlay_bgra(mask)
    assert gt[10, 10, 2] > gt[10, 10, 0], "정답 영역은 빨강"
    assert gt[5, 5, 3] == 255, "윤곽은 불투명"

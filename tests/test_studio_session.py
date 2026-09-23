"""``StudioSession``·``jobs`` — Qt 없이 도는 GUI 상태 로직. 레시피 변경은 전부 재검증, 은행 재사용(reprepare), 큐 규칙."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anograft.bank.importers import yolo as Y
from anograft.core.types import TargetImage
from anograft.studio.jobs import (
    KIND_PREVIEW,
    KIND_THUMB,
    LatestOnlyQueue,
    PreviewJob,
    make_thumb,
    preview_target,
    run_job,
    run_preview,
)
from anograft.studio.session import SessionError, StudioSession, default_recipe
from tests.fixtures import fake_yolo_dataset


@pytest.fixture
def prepared_session(tmp_path: Path) -> StudioSession:
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
    ses.set_stage_field("placement", "margin_px", 4)
    ses.set_stage_field("roi", "erode_px", 2)
    ses.set_field(("pipeline", "source", "min_sources_warn"), 1)
    ses.prepare_now()
    return ses


def test_default_recipe_is_valid_pairs_writer() -> None:
    r = default_recipe()
    assert r.output.writer.format == "pairs" and r.pipeline.preset == "poisson-graft"


def test_edits_revalidate_and_bump_generation() -> None:
    ses = StudioSession()
    g = ses.generation
    ses.set_seed(7)
    assert ses.recipe.seed == 7 and ses.generation == g + 1
    ses.set_seed(7)  # 같은 값 → 세대 불변
    assert ses.generation == g + 1
    with pytest.raises(SessionError, match="seed"):
        ses.set_seed(-1)
    assert ses.recipe.seed == 7  # 실패하면 이전 레시피 유지
    with pytest.raises(SessionError):
        ses.set_field(("pipeline", "blend", "levels"), 3)  # poisson에 multiband 키 → 거부
    assert ses.recipe.pipeline.blend.method == "poisson"


def test_set_preset_replaces_pipeline_and_set_method_resets_block() -> None:
    ses = StudioSession()
    ses.set_stage_field("blend", "feather_px", 9)
    ses.set_preset("hard-paste")
    p = ses.recipe.pipeline
    assert p.preset == "hard-paste" and p.blend.method == "paste" and p.harmonize.method == "none"
    ses.set_method("blend", "alpha")
    assert ses.recipe.pipeline.blend.method == "alpha" and ses.recipe.pipeline.blend.feather_px == 3
    ses.set_method("gtmask", "diff")
    assert ses.recipe.pipeline.gtmask.policy == "diff"
    ses.set_method("roi", "none")
    assert ses.recipe.pipeline.placement.roi.method == "none"
    with pytest.raises(SessionError):
        ses.set_method("blend", "nope")


def test_prepare_and_reprepare_reuse_bank(prepared_session: StudioSession) -> None:
    ses = prepared_session
    assert ses.prepared is not None and not ses.needs_prepare()
    bank = ses.prepared.bank
    assert len(ses.targets) == 2 and ses.target is not None
    old_hash = ses.prepared.pipeline_hash
    ses.set_seed(99)
    assert (
        ses.prepared.bank is bank and ses.prepared.pipeline_hash != old_hash
    )  # 은행 재사용, 해시 갱신
    ses.set_preset("hard-paste")
    assert ses.prepared.pipeline.stages["blend"].__class__.__name__ == "HardPaste"
    # 은행에 없는 클래스 → reprepare 실패 → 레시피 되돌림
    with pytest.raises(SessionError, match="은행"):
        ses.set_field(("output", "class_ratio"), {"ghost": 1.0})
    assert ses.recipe.output.class_ratio is None
    # 경로가 바뀌면 다시 prepare 필요
    ses.set_paths(bank="elsewhere")
    assert ses.needs_prepare()


def test_save_load_roundtrip(prepared_session: StudioSession, tmp_path: Path) -> None:
    ses = prepared_session
    ses.set_seed(5)
    p = ses.save(tmp_path / "r" / "studio.yaml")
    assert p.is_file() and ses.run_command().endswith("studio.yaml")
    other = StudioSession()
    other.load(p)
    assert other.recipe.to_yaml() == ses.recipe.to_yaml() and other.needs_prepare()
    with pytest.raises(SessionError):
        other.load(tmp_path / "missing.yaml")


def test_selection_helpers() -> None:
    ses = StudioSession()
    ses.set_n_variants(3)
    ses.select_variant(9)
    assert ses.variant_index == 2
    g = ses.generation
    ses.select_target(1)
    ses.set_long_side(512)
    assert ses.generation == g + 2 and ses.target is None  # prepared 없음
    assert ses.summary()["preset"] == "poisson-graft"


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


def test_preview_target_scales_image_and_um() -> None:
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    t = TargetImage(Path("t.png"), img, False, um_per_px=2.0)
    small, s = preview_target(t, 200)
    assert small.image.shape[:2] == (100, 200) and s == pytest.approx(1 / 3)
    assert small.um_per_px == pytest.approx(6.0)  # 축소 = 픽셀당 물리 길이 증가
    same, s1 = preview_target(t, 0)
    assert same is t and s1 == 1.0
    same2, _ = preview_target(t, 1000)
    assert same2 is t


def test_latest_only_queue_rules() -> None:
    q = LatestOnlyQueue()
    q.put(PreviewJob(KIND_PREVIEW, 1, index=0, priority=1))
    q.put(PreviewJob(KIND_PREVIEW, 1, index=1, priority=0))
    q.put(PreviewJob(KIND_PREVIEW, 2, index=0, priority=1))  # 같은 키 → 교체
    q.put(PreviewJob(KIND_THUMB, 0, target=Path("a.png"), priority=2))
    assert len(q) == 3
    first = q.pop()
    assert first is not None and first.index == 1  # priority 0 먼저
    assert q.drop_before(2) == 0  # index 0 은 세대 2, 썸네일은 세대와 무관
    nxt = q.pop()
    assert nxt is not None and nxt.index == 0 and nxt.generation == 2
    thumb = q.pop()
    assert thumb is not None and thumb.kind == KIND_THUMB
    assert q.pop() is None
    q.put(PreviewJob(KIND_PREVIEW, 1, index=3))
    assert q.drop_before(5) == 1 and len(q) == 0


def test_run_preview_and_thumb(prepared_session: StudioSession) -> None:
    ses = prepared_session
    prep = ses.prepared
    assert prep is not None
    target = ses.targets[0]
    job = PreviewJob(KIND_PREVIEW, ses.generation, target=target, index=0, long_side=64)
    res = run_preview(prep, job)
    assert res.result.status == "ok" and res.scale == pytest.approx(64 / 96)
    assert res.target.image.shape[:2] == (64, 64) and res.steps and res.steps[0].stage == "roi"
    again = run_job(prep, job)
    assert np.array_equal(again.result.image, res.result.image)  # 같은 (seed, k, 대상) → 같은 결과
    other = run_preview(
        prep, PreviewJob(KIND_PREVIEW, ses.generation, target=target, index=1, long_side=64)
    )
    assert not np.array_equal(other.result.image, res.result.image)
    thumb = make_thumb(PreviewJob(KIND_THUMB, 0, target=target), long_side=48)
    assert thumb.image.shape[:2] == (48, 48) and thumb.shape == (96, 96)


def test_gui_main_without_qt_prints_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import anograft.gui.__main__ as gm

    monkeypatch.setattr(gm, "qt_available", lambda: (False, "설치: pip install -e .[gui]"))
    assert gm.main([]) == 1
    assert "pip install" in capsys.readouterr().err

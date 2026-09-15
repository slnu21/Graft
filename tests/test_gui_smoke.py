"""GUI 스모크 — PySide6가 없으면 전부 skip. ``QT_QPA_PLATFORM=offscreen``으로 창을 만들어 워커까지 한 바퀴 돌린다.

- MainWindow: 탭 5개(ko/en 라벨), 스튜디오가 기본 탭.
- open_inputs → prepared → 레일 채움 → 변형 N장 결과 → 캔버스에 이미지 → 프리셋 변경 시 워커가 새 파이프라인을 쓴다.
- qt_image 왕복 바이트 동일 · 캔버스 와이프/줌 · params_text · stage_thumbnail.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication

from anograft.bank.importers import yolo as Y
from anograft.gui.app import TABS, MainWindow
from anograft.gui.qt_image import from_qimage, to_qimage, to_qpixmap
from anograft.gui.studio.canvas import CompareCanvas
from anograft.gui.studio.panels import PipelinePanel, params_text, stage_thumbnail
from anograft.gui.studio.session import StudioSession, default_recipe
from anograft.gui.studio.variants import VariantStrip
from anograft.gui.theme import apply_theme
from tests.fixtures import fake_yolo_dataset


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app


def _pump(qapp: QApplication, until, timeout: float = 60.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        qapp.processEvents()
        if until():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def bank_and_normals(tmp_path: Path) -> tuple[Path, Path]:
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
    return tmp_path / "bank", normals


def test_qimage_roundtrip(qapp: QApplication) -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (17, 31, 3), dtype=np.uint8)
    back = from_qimage(to_qimage(img))
    assert np.array_equal(back, img)
    gray = rng.integers(0, 256, (9, 13), dtype=np.uint8)
    q = to_qimage(gray)
    assert q.width() == 13 and q.height() == 9
    view = img[2:10, 3:20]  # 비연속 view 도 stride 처리
    assert np.array_equal(from_qimage(to_qimage(view)), view)
    assert to_qpixmap(img).width() == 31


def test_main_window_tabs_and_studio_flow(
    qapp: QApplication, bank_and_normals: tuple[Path, Path]
) -> None:
    bank, normals = bank_and_normals
    ses = StudioSession(default_recipe(bank=bank.as_posix(), targets=normals.as_posix()))
    ses.set_stage_field("placement", "margin_px", 4)
    ses.set_stage_field("roi", "erode_px", 2)
    ses.set_field(("pipeline", "source", "min_sources_warn"), 1)
    ses.set_n_variants(3)
    ses.set_long_side(64)
    win = MainWindow(ses)
    try:
        win.show()
        assert win.tabs.count() == len(TABS) == 5
        assert [win.tabs.tabText(i) for i in range(5)] == [t for _k, t in TABS]
        assert win.tabs.currentWidget() is win.studio
        tab = win.studio
        tab.open_inputs(bank.as_posix(), normals.as_posix())
        assert _pump(qapp, lambda: ses.prepared is not None)
        assert tab.rail.list.count() == 2 and tab.variants.list.count() == 3
        assert _pump(qapp, lambda: len(tab._results) >= 3)
        assert tab.canvas.image_size() == (64, 64)
        assert (
            "v1:" in win.status_bar.currentMessage()
            and "poisson" in win.status_bar.currentMessage()
        )
        assert _pump(qapp, lambda: win.worker.pending() == 0)
        assert tab.rail.list.item(0).text().startswith("n0.png") or tab.rail.list.item(
            0
        ).text().startswith("n1.png")
        # 프리셋 변경 → 세대 증가 → 새 파이프라인으로 재계산 (워커가 옛 Prepared를 쓰면 안 된다)
        gen = ses.generation
        tab.strip.preset.setCurrentIndex(tab.strip.preset.findData("hard-paste"))
        qapp.processEvents()
        assert ses.generation > gen and ses.recipe.pipeline.blend.method == "paste"
        assert _pump(qapp, lambda: len(tab._results) >= 3)
        assert "paste" in win.status_bar.currentMessage()
        res = tab._results[0]
        assert (
            res.job.generation == ses.generation
            and res.result.sidecar["defects"][0]["blend"]["method"] == "paste"
        )
        # 변형 카드 선택 → 캔버스 전환
        tab.variants.list.setCurrentRow(1)
        qapp.processEvents()
        assert ses.variant_index == 1 and "v2:" in win.status_bar.currentMessage()
        # 대상 변경 → 세대 증가
        gen = ses.generation
        tab.rail.list.setCurrentRow(1)
        qapp.processEvents()
        assert ses.generation > gen and ses.target_index == 1
        assert _pump(qapp, lambda: len(tab._results) >= 3)
        # 잘못된 값은 세션이 막고 위젯이 되돌아간다
        tab.session.set_seed(3)
        tab.sync_widgets()
        assert tab.strip.seed.value() == 3
        # 파이프라인 카드가 현재 method를 보여 준다
        assert tab.pipe.cards["blend"].method.currentData() == "paste"
        # 레시피 저장
        out = bank.parent / "saved.yaml"
        ses.save(out)
        assert out.is_file() and "preset: hard-paste" in out.read_text(encoding="utf-8")
    finally:
        win.close()
        qapp.processEvents()


def test_canvas_wipe_and_zoom(qapp: QApplication) -> None:
    c = CompareCanvas()
    c.resize(400, 300)
    img = np.full((100, 200, 3), 90, dtype=np.uint8)
    synth = img.copy()
    synth[40:60, 90:110] = 250
    gt = np.zeros((100, 200), dtype=np.uint8)
    gt[40:60, 90:110] = 255
    c.set_images(img, synth, gt, [], roi=np.ones((100, 200), dtype=bool))
    assert c.image_size() == (200, 100)
    r = c.image_rect()
    assert r.width() > 0 and abs(c.wipe_x() - (r.left() + r.width() / 2)) < 1e-6
    seen: list[float] = []
    c.wipe_changed.connect(seen.append)
    c.set_wipe(0.2)
    c.set_wipe(5.0)  # 클램프
    assert seen == [0.2, 0.98]
    x, y = c.to_image(QPointF(r.left(), r.top()))
    assert (round(x), round(y)) == (0, 0)
    c.zoom = 2.0
    assert c.image_rect().width() == pytest.approx(r.width() * 2)
    c.reset_view()
    assert c.zoom == 1.0
    c.grab()  # paintEvent 가 예외 없이 돈다
    c.set_original_only(img)
    c.clear("x")
    c.grab()


def test_params_text_and_stage_thumbnail(qapp: QApplication) -> None:
    from anograft.core.pipeline import Pipeline
    from tests.fixtures import disk_target, line_defect, memory_bank, pipeline_deps

    rec = default_recipe()
    text = params_text(rec.pipeline.geometry)
    assert "scale 0.8–1.25" in text and "method" not in text and "flip True" in text
    assert params_text(rec.pipeline.blend).startswith("poisson_mode normal")
    bank = memory_bank([line_defect(14, 3)])
    ses = StudioSession(rec)
    ses.set_stage_field("placement", "margin_px", 4)
    ses.set_stage_field("roi", "erode_px", 2)
    pipe = Pipeline.from_recipe(ses.recipe, pipeline_deps(ses.recipe, bank))
    _r, steps = pipe.run_one_traced(disk_target(96), 0)
    for stage in ("source", "geometry", "placement", "blend", "harmonize", "degrade", "gtmask"):
        th = stage_thumbnail(stage, steps, 40)
        assert th is not None and max(th.shape[:2]) <= 40, stage
    assert stage_thumbnail("blend", [], 40) is None


def test_stage_cards_show_fail_soft_warnings_and_variant_tooltip(qapp: QApplication) -> None:
    """KNOWN-ISSUES #1: 스테이지 경고(``<stage>: …``)가 그 카드에 보이고, 변형 카드의 잘린 사유는 툴팁에 원문."""
    panel = PipelinePanel()
    ws = (
        "roi: mask_dir 로드 실패 roi/t01.png — ValueError: 비율이 다릅니다",
        "placement: scratch/000 배치 실패 — ROI 없음 (시도 0, 축소 0)",
    )
    panel.set_trace([], ws)
    shown = panel.stage_warnings()
    assert set(shown) == {"placement"}  # ROI 는 배치 카드의 하위 블록 → roi: 경고도 배치 카드에
    lines = shown["placement"].splitlines()
    assert lines[0] == "⚠ " + ws[0] and lines[1] == "⚠ " + ws[1][len("placement:") :].strip()
    assert not panel.cards["blend"].note.isVisibleTo(panel)
    panel.set_trace([], ())  # 다음 결과가 정상이면 사라진다
    assert panel.stage_warnings() == {}

    variants = VariantStrip()
    variants.reset(2)
    variants.set_failed(0, ws[0])
    item = variants.list.item(0)
    assert item.text().startswith("v1 · ") and len(item.text()) <= len("v1 · ") + 28
    assert item.toolTip() == ws[0]
    variants.set_result(0, np.zeros((16, 16, 3), dtype=np.uint8), "ok")
    assert item.toolTip() == ""


def test_stage_param_form_edits_recipe_and_shows_errors_inline(
    qapp: QApplication, tmp_path: Path
) -> None:
    """stage-params: 카드 폼 → 세션 재검증 → 미리보기 요청. 잘못된 값은 대화상자 없이 카드 빨간 줄 + 위젯 되돌림.
    ROI 하위 콤보로 mask_dir 를 고르면 path 자리표시(".")가 채워지고 폼에 path 행이 생긴다."""
    from anograft.gui.studio.param_form import DEBOUNCE_MS

    ses = StudioSession(default_recipe())
    win = MainWindow(ses)
    try:
        tab = win.studio
        pipe = tab.pipe
        geo = pipe.cards["geometry"]
        assert list(geo.form.rows) == ["scale", "rotate", "flip", "elastic.alpha", "elastic.sigma"]
        # 1) 회전 범위 편집 → 디바운스 후 세션 반영
        row = geo.form.rows["rotate"]
        row.editors[0].setValue(-15.0)
        row.editors[1].setValue(15.0)
        assert ses.recipe.pipeline.geometry.rotate == (-180.0, 180.0)  # 아직(디바운스)
        assert _pump(qapp, lambda: ses.recipe.pipeline.geometry.rotate == (-15.0, 15.0), 5.0)
        assert geo.error.isHidden()
        # 2) 잘못된 값 → 카드 오류 줄 + 위젯 되돌림(대화상자 없음)
        row.editors[0].setValue(-400.0)
        assert _pump(qapp, lambda: not geo.error.isHidden(), 5.0)
        assert "rotate" in geo.error.text() and "360" in geo.error.text()
        assert row.editors[0].value() == -15.0 and row.editors[0].property("error") is True
        assert ses.recipe.pipeline.geometry.rotate == (-15.0, 15.0)
        # 3) 다음 정상 편집이 오류를 지운다 (체크박스는 즉시)
        geo.form.rows["flip"].editors[0].setChecked(False)
        assert _pump(qapp, lambda: ses.recipe.pipeline.geometry.flip is False, 5.0)
        assert geo.error.isHidden() and row.editors[0].property("error") is False
        # 4) 중첩 필드(점 경로)
        geo.form.rows["elastic.alpha"].editors[0].setValue(2.5)
        assert _pump(qapp, lambda: ses.recipe.pipeline.geometry.elastic.alpha == 2.5, 5.0)
        # 5) optional 켜기/끄기 — degrade.gamma
        gam = pipe.cards["degrade"].form.rows["gamma"]
        assert gam.toggle is not None and not gam.editors[0].isEnabled()
        gam.toggle.setChecked(True)
        assert _pump(qapp, lambda: ses.recipe.pipeline.degrade.gamma == (0.8, 1.25), 5.0)
        gam.toggle.setChecked(False)
        assert _pump(qapp, lambda: ses.recipe.pipeline.degrade.gamma is None, 5.0)
        # 6) ROI 하위 스테이지 — method 콤보 + 폼
        pl = pipe.cards["placement"]
        assert pl.roi_method is not None and pl.roi_form is not None
        assert list(pl.roi_form.rows) == ["invert", "erode_px"]
        pl.roi_form.rows["erode_px"].editors[0].setValue(20)
        assert _pump(qapp, lambda: ses.recipe.pipeline.placement.roi.erode_px == 20, 5.0)
        i = pl.roi_method.findData("mask_dir")
        pl.roi_method.setCurrentIndex(i)
        qapp.processEvents()
        assert ses.recipe.pipeline.placement.roi.method == "mask_dir"
        assert (
            list(pl.roi_form.rows) == ["path"] and pl.roi_form.rows["path"].editors[0].text() == "."
        )
        assert pl.method.currentData() == "sampled"  # 배치 method 는 그대로
        # 7) 디바운스 대기 중인 편집은 다른 필드의 즉시 커밋(재동기화)에 지워지지 않는다
        sc = geo.form.rows["scale"]
        sc.editors[0].setValue(0.5)
        sc.editors[1].setValue(2.0)
        geo.form.rows["flip"].editors[0].setChecked(True)  # 즉시 커밋 → 폼 sync
        assert sc.editors[0].value() == 0.5  # 되돌아가지 않았다
        assert _pump(qapp, lambda: ses.recipe.pipeline.geometry.scale == (0.5, 2.0), 5.0)
        # 8) 저장에 반영
        out = ses.save(tmp_path / "edited.yaml")
        text = out.read_text(encoding="utf-8")
        assert "rotate:" in text and "-15" in text and "alpha: 2.5" in text and "mask_dir" in text
        # 디바운스 타이머가 남아 있지 않게
        _pump(qapp, lambda: False, DEBOUNCE_MS / 1000 + 0.1)
    finally:
        win.close()
        qapp.processEvents()


def test_inputs_panel_um_per_px_edits_recipe(qapp: QApplication) -> None:
    """v0.7.x — 입력 패널 µm/px 스핀 → inputs.um_per_px(0 = 모름 = null) → 레시피 저장에 반영."""
    ses = StudioSession(default_recipe())
    win = MainWindow(ses, start_worker=False)
    try:
        panel = win.studio.inputs
        assert panel.um.value() == 0.0 and ses.recipe.inputs.um_per_px is None
        panel.um.setValue(2.5)
        panel.um.editingFinished.emit()
        assert ses.recipe.inputs.um_per_px == 2.5
        assert "um_per_px: 2.5" in ses.recipe.to_yaml()
        panel.um.setValue(0.0)
        panel.um.editingFinished.emit()
        assert ses.recipe.inputs.um_per_px is None
        # sync 가 스핀을 되돌린다
        ses.set_field(("inputs", "um_per_px"), 1.25)
        win.studio.sync_widgets()
        assert panel.um.value() == 1.25
    finally:
        win.close()
        qapp.processEvents()

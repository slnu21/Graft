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
from anograft.gui.studio.panels import params_text, stage_thumbnail
from anograft.gui.studio.session import StudioSession, default_recipe
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

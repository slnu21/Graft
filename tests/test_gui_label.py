"""라벨 탭 GUI 스모크 — PySide6 없으면 skip. offscreen 에서 ``LabelTab``을 만들고 **합성 마우스 이벤트**로 브러시·지우개·폴리곤·
자동 선택 박스를 실제로 굴려 세션 마스크가 바뀌는지, 오버레이 버퍼가 갱신되는지, 은행 저장 → 파일·bank.yaml, 메인 창에서
``bank_saved`` 가 스튜디오 재준비를 일으키는지 확인한다."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from anograft.bank import Bank
from anograft.bank.importers import yolo as Y
from anograft.gui.app import MainWindow
from anograft.gui.label.canvas import LabelCanvas
from anograft.gui.label.tab import LabelTab
from anograft.gui.studio.session import StudioSession, default_recipe
from anograft.gui.theme import apply_theme
from anograft.io import imgio
from tests.fixtures import blob_image, disk_image, fake_yolo_dataset


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


def _mouse(canvas: LabelCanvas, kind, pos: QPointF, button=Qt.MouseButton.LeftButton) -> None:
    buttons = button if kind != QEvent.Type.MouseButtonRelease else Qt.MouseButton.NoButton
    ev = QMouseEvent(kind, pos, pos, button, buttons, Qt.KeyboardModifier.NoModifier)
    if kind == QEvent.Type.MouseButtonPress:
        canvas.mousePressEvent(ev)
    elif kind == QEvent.Type.MouseMove:
        canvas.mouseMoveEvent(ev)
    elif kind == QEvent.Type.MouseButtonRelease:
        canvas.mouseReleaseEvent(ev)
    elif kind == QEvent.Type.MouseButtonDblClick:
        canvas.mouseDoubleClickEvent(ev)


def _drag(canvas: LabelCanvas, pts: list[tuple[float, float]]) -> None:
    """이미지 좌표 목록을 위젯 좌표로 바꿔 누름-이동-뗌."""
    r = canvas.image_rect()
    w, h = canvas.image_size()
    wp = [QPointF(r.left() + x / w * r.width(), r.top() + y / h * r.height()) for x, y in pts]
    _mouse(canvas, QEvent.Type.MouseButtonPress, wp[0])
    for p in wp[1:]:
        _mouse(canvas, QEvent.Type.MouseMove, p)
    _mouse(canvas, QEvent.Type.MouseButtonRelease, wp[-1])


@pytest.fixture
def tab(qapp: QApplication, tmp_path: Path) -> LabelTab:
    folder = tmp_path / "defects"
    imgio.write_image(folder / "b_plate.png", disk_image(128))
    imgio.write_image(folder / "a_blob.png", blob_image(128, [(64, 64, 14)]))
    t = LabelTab()
    t.confirm_discard = None  # 테스트에서 대화상자 없이 버린다
    t.resize(1200, 800)
    t.show()
    qapp.processEvents()
    t.open_folder(folder)
    qapp.processEvents()
    return t


def test_folder_list_and_open_first_image(tab: LabelTab) -> None:
    assert [p.name for p in tab.files] == ["a_blob.png", "b_plate.png"]  # 이름 정렬
    assert tab.session.loaded and tab.session.path.name == "a_blob.png"
    assert tab.canvas._qov is not None and tab.canvas._qov.width() == 128
    assert "마스크 없음" in tab.stats.text() and tab.btn_save.isEnabled()
    tab.step(1)
    assert tab.session.path.name == "b_plate.png"
    tab.step(-1)
    assert tab.session.path.name == "a_blob.png"


def test_brush_eraser_polygon_auto_via_mouse(tab: LabelTab, qapp: QApplication) -> None:
    c = tab.canvas
    c.set_tool("brush")
    c.set_brush(8)
    _drag(c, [(20.0, 100.0), (60.0, 100.0)])
    m = tab.session.mask
    assert m[100, 40] == 255 and m[100, 10] == 0
    assert (
        c._ov[100, 40, 3] == c.opacity and c._ov[100, 10, 3] == 0
    )  # 오버레이가 스트로크 bbox 만큼 갱신
    assert tab.session.can_undo and tab.btn_undo.isEnabled()
    assert "면적" in tab.stats.text()
    c.set_tool("eraser")
    _drag(c, [(40.0, 100.0), (40.0, 100.0)])
    assert m[100, 40] == 0 and c._ov[100, 40, 3] == 0
    # 폴리곤: 세 점 + 더블클릭
    c.set_tool("polygon")
    r = c.image_rect()
    w, h = c.image_size()

    def wp(x: float, y: float) -> QPointF:
        return QPointF(r.left() + x / w * r.width(), r.top() + y / h * r.height())

    for x, y in [(90, 20), (120, 20), (120, 50), (90, 50)]:
        _mouse(c, QEvent.Type.MouseButtonPress, wp(x, y))
        _mouse(c, QEvent.Type.MouseButtonRelease, wp(x, y))
    assert len(c._poly) == 4
    _mouse(c, QEvent.Type.MouseButtonDblClick, wp(90, 50))
    assert not c._poly and tab.session.mask[35, 105] == 255
    # 자동 선택: 얼룩(64,64,r14) 둘레 박스
    c.set_tool("auto")
    got: list[str] = []
    c.auto_done.connect(got.append)
    _drag(c, [(44.0, 44.0), (84.0, 84.0)])
    assert got and got[0] in ("grabcut", "otsu", "ellipse", "rect")
    assert tab.session.mask[64, 64] == 255 and tab.session.mask[64, 30] == 0
    # 되돌리기 3단계 → 폴리곤·자동 선택 취소
    c.undo()
    assert tab.session.mask[64, 64] == 0 and tab.session.mask[35, 105] == 255
    c.redo()
    assert tab.session.mask[64, 64] == 255
    # 팽창·침식·지우기
    n0 = int((tab.session.mask > 0).sum())
    tab.btn_dilate.click()
    assert int((tab.session.mask > 0).sum()) > n0
    tab.btn_erode.click()
    tab.btn_clear.click()
    assert not tab.session.mask.any() and tab.session.can_undo
    c.undo()
    assert tab.session.mask.any()


def test_keyboard_tools_and_brush_size(tab: LabelTab) -> None:
    from PySide6.QtGui import QKeyEvent

    c = tab.canvas

    def key(k, mods=Qt.KeyboardModifier.NoModifier) -> None:
        c.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, k, mods))

    key(Qt.Key.Key_E)
    assert c.tool == "eraser" and tab.tool_buttons["eraser"].isChecked()
    key(Qt.Key.Key_P)
    assert c.tool == "polygon"
    key(Qt.Key.Key_A)
    assert c.tool == "auto"
    key(Qt.Key.Key_H)
    assert c.tool == "pan"
    key(Qt.Key.Key_B)
    assert c.tool == "brush"
    c.set_brush(12)
    key(Qt.Key.Key_BracketRight)
    assert c.brush_px == 15 and tab.brush.value() == 15 and tab.brush_label.text() == "15px"
    key(Qt.Key.Key_BracketLeft)
    assert c.brush_px == 12
    tab.brush.setValue(30)
    assert c.brush_px == 30
    tab.opacity.setValue(100)
    assert c.opacity == 255


def test_save_to_bank_writes_sources_and_reloads_classes(tab: LabelTab, tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    assert not tab.save_to_bank()  # 은행 미지정
    assert "은행 폴더" in tab.result.text()
    tab.set_bank(bank.as_posix())
    assert not tab.save_to_bank()  # 클래스 없음
    assert "클래스" in tab.result.text()
    tab.cls.setEditText("scratch")
    assert not tab.save_to_bank()  # 빈 마스크
    assert "비어" in tab.result.text()
    tab.canvas.set_tool("brush")
    tab.canvas.set_brush(6)
    _drag(tab.canvas, [(20.0, 60.0), (50.0, 60.0)])
    _drag(tab.canvas, [(90.0, 90.0), (110.0, 90.0)])
    tab.cls.setEditText("scratch")
    tab.tags.setText("gui, 직선형")
    tab.um.setValue(4.0)
    saved: list[str] = []
    tab.bank_saved.connect(saved.append)
    assert tab.save_to_bank()
    assert saved == [bank.as_posix()]
    b = Bank.load(bank)
    assert b.classes == ["scratch"] and len(b) == 2
    src = b.by_class("scratch")[0]
    assert (
        src.mask_origin == "manual:brush" and src.um_per_px == 4.0 and src.tags == ("gui", "직선형")
    )
    assert "은행 저장 2개" in tab.result.text()
    assert not tab.session.mask.any() and tab.session.can_undo  # 저장 뒤 비움(되돌리기 가능)
    assert [tab.cls.itemText(i) for i in range(tab.cls.count())] == ["scratch"]
    assert "scratch" in tab.note.text()
    # 두 번째 클래스 — 성분 하나로
    tab.cb_whole.setChecked(True)
    _drag(tab.canvas, [(20.0, 20.0), (40.0, 20.0)])
    tab.cls.setEditText("dent")
    assert tab.save_to_bank()
    b = Bank.load(bank)
    assert b.classes == ["scratch", "dent"] and len(b) == 3
    assert [tab.cls.itemText(i) for i in range(tab.cls.count())] == ["scratch", "dent"]


def test_discard_guard_when_switching_images(tab: LabelTab) -> None:
    tab.canvas.set_tool("brush")
    _drag(tab.canvas, [(20.0, 60.0), (50.0, 60.0)])
    assert tab.session.dirty
    tab.confirm_discard = lambda: False
    tab.list.setCurrentRow(1)
    assert tab.session.path.name == "a_blob.png" and tab.list.currentRow() == 0  # 거부 → 그대로
    tab.confirm_discard = lambda: True
    tab.list.setCurrentRow(1)
    assert tab.session.path.name == "b_plate.png" and not tab.session.dirty


def test_main_window_bank_saved_reprepares_studio(qapp: QApplication, tmp_path: Path) -> None:
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
    win = MainWindow(ses, start_worker=True)
    try:
        win.show()
        win.studio.open_inputs((tmp_path / "bank").as_posix(), normals.as_posix())
        assert _pump(qapp, lambda: ses.prepared is not None)
        n_before = len(ses.prepared.bank)
        assert win.label.bank_edit.text() == (tmp_path / "bank").as_posix()
        # 라벨 탭에서 새 소스 저장 → 스튜디오 재준비 → 은행 소스 수 증가
        img = tmp_path / "new.png"
        imgio.write_image(img, disk_image(96))
        win.label.confirm_discard = None
        win.label.open_image(img)
        win.label.canvas.set_tool("brush")
        _drag(win.label.canvas, [(30.0, 48.0), (60.0, 48.0)])
        win.label.cls.setEditText("spot")
        gen = ses.generation
        assert win.label.save_to_bank()
        assert _pump(qapp, lambda: ses.prepared is not None and ses.generation > gen)
        assert len(ses.prepared.bank) == n_before + 1
        assert win.tabs.tabText(1).startswith("라벨")
    finally:
        win.worker.stop()
        win.close()
        qapp.processEvents()


def test_canvas_without_session_ignores_input(qapp: QApplication) -> None:
    c = LabelCanvas()
    c.resize(300, 200)
    _mouse(c, QEvent.Type.MouseButtonPress, QPointF(10, 10))
    _mouse(c, QEvent.Type.MouseButtonRelease, QPointF(10, 10))
    c.undo()
    c.apply(lambda s: s.clear())
    assert c.session is None and c._qov is None
    with pytest.raises(ValueError):
        c.set_tool("laser")
    assert np.array_equal(np.zeros(1), np.zeros(1))  # 자리표시

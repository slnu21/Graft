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
    assert "보관함 폴더" in tab.result.text()
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
    assert "보관함에 저장 2개" in tab.result.text()
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
        assert win.tabs.tabText(1).startswith("결함 표시")
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


# --- v0.6 label-yolo-roi: YOLO 초안 프리필 · ROI 모드 · 최근 레시피/빈 상태 --------------------


def test_folder_marks_labeled_images_and_prefills_yolo_draft(
    qapp: QApplication, tmp_path: Path
) -> None:
    d = fake_yolo_dataset(tmp_path / "ds")
    t = LabelTab()
    t.confirm_discard = None
    t.resize(1200, 800)
    t.show()
    t.auto_method.setCurrentText("rect")  # 결정적·빠름
    t.open_folder(d["images"])
    qapp.processEvents()
    names = [t.list.item(i).text() for i in range(t.list.count())]
    assert (
        names[0] == "▸ d0.png"
        and "n1.png" in names
        and not names[names.index("n1.png")].startswith("▸")
    )
    assert "YOLO 라벨" in t.list_title.text()
    # 첫 이미지 d0: 박스 1(spot) → 자동 채움 + 클래스 콤보 동기
    assert t.session.path.name == "d0.png" and t.session.draft is not None
    assert t.session.mask.any() and t.cls.currentText() == "spot"
    assert t.draft_cls.isEnabled() and t.draft_cls.count() == 1 and "박스 1" in t.draft_info.text()
    assert t.session.mask_origin() == "yolo-box:rect"
    # d1: 클래스 2개 — 첫 클래스(spot) 자동, 콤보로 crack 선택 → 채우기 → 폴리곤 자리 포함
    t.session.dirty = False
    t.list.setCurrentRow(names.index("▸ d1.png"))
    qapp.processEvents()
    assert t.draft_cls.count() == 2 and t.cls.currentText() == "spot"
    t.draft_cls.setCurrentIndex(1)
    assert t.fill_draft() is not None and t.cls.currentText() == "crack"
    assert t.session.mask[20:34, 20:34].any() and t.canvas._ov[..., 3].any()
    # 자동 채우기 끄면 열 때 비어 있고 초안만 안내
    t.cb_draft_auto.setChecked(False)
    t.session.dirty = False
    t.list.setCurrentRow(names.index("▸ d0.png"))
    qapp.processEvents()
    assert not t.session.mask.any() and t.draft_cls.isEnabled()
    # 라벨 없는 이미지 → 초안 비활성
    t.list.setCurrentRow(names.index("n1.png"))
    qapp.processEvents()
    assert not t.draft_cls.isEnabled() and "없음" in t.draft_info.text()
    t.close()


def test_roi_mode_saves_and_reloads_mask_dir_png(qapp: QApplication, tmp_path: Path) -> None:
    normals = tmp_path / "normals"
    imgio.write_image(normals / "n0.png", disk_image(128))
    imgio.write_image(normals / "n1.png", disk_image(128))
    t = LabelTab()
    t.confirm_discard = None
    t.resize(1200, 800)
    t.show()
    t.open_folder(normals)
    qapp.processEvents()
    assert t.mode == "bank" and t.defect_box.isVisible() and not t.roi_box.isVisible()
    t.set_mode("roi")
    qapp.processEvents()
    assert t.mode == "roi" and t.roi_box.isVisible() and not t.defect_box.isVisible()
    assert t.btn_save.text().startswith("ROI") and t.list_title.text().startswith("정상 이미지")
    assert t.roi_dir_path() == tmp_path / "roi" and "mask_dir" in t.roi_hint.text()
    saved: list[str] = []
    t.roi_saved.connect(saved.append)
    assert not t.save()  # 빈 마스크
    assert "비어" in t.result.text()
    t.canvas.set_tool("brush")
    t.canvas.set_brush(10)
    _drag(t.canvas, [(20.0, 64.0), (100.0, 64.0)])
    assert t.save() and saved == [(tmp_path / "roi" / "n0.png").as_posix()]
    m = imgio.read_mask(tmp_path / "roi" / "n0.png")
    assert m.shape == (128, 128) and m[64, 60] == 255 and not t.session.dirty
    assert "ROI 1장" in t.roi_hint.text()
    # 다른 이미지로 갔다가 돌아오면 ROI 를 다시 불러온다
    t.list.setCurrentRow(1)
    qapp.processEvents()
    assert not t.session.mask.any()
    t.list.setCurrentRow(0)
    qapp.processEvents()
    assert t.session.mask[64, 60] == 255 and not t.session.dirty
    # mask_dir 직접 지정
    t.roi_dir.setText((tmp_path / "custom").as_posix())
    t.roi_dir.editingFinished.emit()
    assert t.roi_dir_path() == tmp_path / "custom" and "만들어진다" in t.roi_hint.text()
    assert t.save() and (tmp_path / "custom" / "n0.png").is_file()
    t.close()


def test_main_window_recent_recipe_and_empty_state(qapp: QApplication, tmp_path: Path) -> None:
    from PySide6.QtCore import QSettings

    from anograft.gui.app import EMPTY_STATE, recent_recipe, remember_recipe

    store = QSettings((tmp_path / "s.ini").as_posix(), QSettings.Format.IniFormat)
    assert recent_recipe(store) is None
    remember_recipe(tmp_path / "gone.yaml", store)
    assert recent_recipe(store) is None  # 파일이 없으면 무시
    recipe = tmp_path / "r.yaml"
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
    ses.save(recipe)
    win = MainWindow(start_worker=False, settings=store)
    try:
        win.show_empty_state()
        assert win.studio.canvas.message == EMPTY_STATE and "레시피가 없습니다" in EMPTY_STATE
        assert "샘플 데이터" in EMPTY_STATE  # 0.7.3+: 데이터 없는 사용자의 첫 행동
        win.open_recipe(recipe)
        qapp.processEvents()
        assert recent_recipe(store) == recipe.resolve()
        assert win.label.bank_edit.text() == (tmp_path / "bank").as_posix()
        remember_recipe(None, store)
        assert recent_recipe(store) is None
    finally:
        win.close()
        qapp.processEvents()


def test_auto_select_status_shows_confidence(tab: LabelTab, qapp: QApplication) -> None:
    msgs: list[str] = []
    tab.status.connect(msgs.append)
    tab.canvas.set_tool("auto")
    tab.auto_method.setCurrentText("grabcut")
    _drag(tab.canvas, [(40.0, 40.0), (90.0, 90.0)])  # a_blob 의 얼룩(64,64 r14) 둘레
    qapp.processEvents()
    auto = [m for m in msgs if m.startswith("자동 선택")]
    assert auto and "confidence" in auto[-1] and tab.result.text() == auto[-1]

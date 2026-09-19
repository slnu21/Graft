"""v0.9 사용성 ⑤ workflow-stepper — 탭 순서·번호·배지·다음 버튼·시작 체크리스트. 순수 부분(workflow.py)은 Qt 없이."""

from __future__ import annotations

import os
import time

import pytest

from anograft.gui.workflow import (
    NEXT_HINT,
    STEPS,
    WorkflowState,
    checklist,
    next_step,
    tab_label,
)


def test_workflow_pure() -> None:
    assert [k for k, _n in STEPS] == ["label", "bank", "studio", "batch", "review"]
    empty = WorkflowState()
    assert tab_label("label", empty) == "① 결함 표시"
    assert tab_label("studio", empty) == "③ 미리보기"
    st = WorkflowState(
        bank_pieces=19,
        recipe_open=True,
        prepared=True,
        targets=12,
        batch_ok=24,
        review_open=True,
        review_unreviewed=3,
        label_images=22,
    )
    assert tab_label("label", st) == "① 결함 표시 · 22장"
    assert tab_label("bank", st) == "② 결함 보관함 · 19"
    assert tab_label("studio", st) == "③ 미리보기 · 준비됨"
    assert tab_label("batch", st) == "④ 일괄 생성 · 24장"
    assert tab_label("review", st) == "⑤ 검수 · 미검수 3"
    assert (
        tab_label("review", WorkflowState(review_open=True, review_unreviewed=0)) == "⑤ 검수 · 완료"
    )
    assert tab_label("studio", WorkflowState(recipe_open=True)) == "③ 미리보기 · 레시피"
    assert next_step("label") == ("bank", "결함 보관함") and next_step("review") is None
    assert set(NEXT_HINT) == {k for k, _n in STEPS}
    items = checklist(empty)
    assert [it.key for it in items] == ["sample", "label", "bank", "studio", "batch", "review"]
    assert not any(it.done for it in items)
    items = checklist(st)
    assert all(it.done for it in items)
    half = checklist(WorkflowState(bank_pieces=5))
    assert [it.done for it in half] == [True, True, True, False, False, False]


pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from anograft.gui.app import EMPTY_STATE, MainWindow  # noqa: E402
from anograft.gui.checklist import ChecklistPanel  # noqa: E402
from anograft.gui.studio.session import StudioSession, default_recipe  # noqa: E402
from anograft.gui.theme import apply_theme  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app


def _pump(qapp: QApplication, until, timeout: float = 5.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        qapp.processEvents()
        if until():
            return True
        time.sleep(0.01)
    return False


def test_checklist_panel(qapp: QApplication) -> None:
    panel = ChecklistPanel()
    got: list[str] = []
    panel.action.connect(got.append)
    assert len(panel.rows) == 6 and panel.rows[0][0].text() == "○"
    assert panel.rows[0][2].objectName() == "Primary"  # 첫 미완 항목이 강조
    panel.rows[0][2].click()
    panel.rows[3][2].click()
    assert got == ["sample", "studio"]
    panel.set_state(WorkflowState(bank_pieces=3))
    assert [r[0].text() for r in panel.rows] == ["✓", "✓", "✓", "○", "○", "○"]
    assert panel.rows[3][2].objectName() == "Primary" and panel.rows[0][2].objectName() == ""
    panel.rows[1][2].click()
    assert got[-1] == "label"  # 버튼은 한 번만 연결돼 있고 키는 현재 상태의 것
    panel.deleteLater()


def test_main_window_stepper(qapp: QApplication) -> None:
    ses = StudioSession(default_recipe())
    win = MainWindow(ses)
    try:
        assert win.tabs.currentWidget() is win.studio and win.current_key() == "studio"
        assert [win.tabs.tabText(i)[:1] for i in range(5)] == ["①", "②", "③", "④", "⑤"]
        assert win.btn_next.text() == "다음: 일괄 생성 →" and win.btn_next.isEnabled()
        assert win.btn_next.toolTip() == NEXT_HINT["studio"]
        win.go_next()
        assert win.tabs.currentWidget() is win.batch and win.btn_next.text() == "다음: 검수 →"
        win.go_next()
        assert win.tabs.currentWidget() is win.review and not win.btn_next.isEnabled()
        win.tabs.setCurrentIndex(win.tab_index("label"))
        assert win.btn_next.text() == "다음: 결함 보관함 →"
        # 빈 상태 → 체크리스트가 캔버스 자리에, 레시피가 열리면 캔버스로
        win.show_empty_state()
        assert win.studio.center.currentWidget() is win.studio.checklist
        assert win.studio.canvas.message == EMPTY_STATE
        win.studio.request_previews()  # 입력을 열면 불린다
        assert win.studio.center.currentWidget() is win.studio.canvas
        win.show_empty_state()
        win._on_checklist_action("bank")
        assert win.tabs.currentWidget() is win.bank
        win._on_checklist_action("review")
        assert win.tabs.currentWidget() is win.review
        # 배지: 세션 상태에서
        st = win.workflow_state()
        assert st.bank_pieces is None and not st.prepared and st.batch_ok is None
        win.refresh_workflow()
        assert win.tabs.tabText(win.tab_index("bank")) == "② 결함 보관함"
    finally:
        win.close()

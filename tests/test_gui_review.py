"""v0.7 검수 탭 GUI 스모크 — PySide6 없으면 skip. offscreen 에서 ``ReviewTab`` 을 열어 그리드·필터·상세·판정(A/R/U, 자동 저장 →\nreview.csv)·다음으로 이동·히스토그램·정리본 내보내기를 굴리고, 메인 창에서 검수 탭이 실물인지 확인한다."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from anograft.gui.app import MainWindow
from anograft.gui.review.session import histogram
from anograft.gui.review.tab import HistogramWidget, ReviewTab, overlay_image
from anograft.gui.theme import apply_theme
from anograft.io.prune import read_review
from tests.fixtures import disk_image
from tests.test_review_session import output_root  # noqa: F401 — 픽스처 재사용


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app


def test_overlay_image_and_histogram_widget(qapp: QApplication) -> None:
    img = disk_image(300)
    m = np.zeros((300, 300), dtype=np.uint8)
    m[100:140, 100:160] = 255
    out = overlay_image(img, m, long_side=120)
    assert out.shape[1] == 120 and not np.array_equal(out, overlay_image(img, None, long_side=120))
    w = HistogramWidget()
    w.resize(300, 160)
    w.show()
    w.set_histogram(histogram([10, 100], [50], bins=4), "면적")
    qapp.processEvents()
    w.set_histogram(None)
    qapp.processEvents()
    w.close()


def test_review_tab_flow(qapp: QApplication, output_root: Path, tmp_path: Path) -> None:  # noqa: F811
    t = ReviewTab()
    t.resize(1400, 850)
    t.show()
    qapp.processEvents()
    assert not t.open_root("") and not t.open_root(output_root.parent / "nope")
    assert t.open_root(output_root)
    items = t.session.items
    n_ok = sum(1 for it in items if it.status == "ok")
    assert t.grid.count() == len(items) and "합성" in t.summary.text()
    assert t.hist.hist is not None and sum(t.hist.hist.b) == 5  # 실제(은행) 소스 5
    # 필터: 미검수 = ok 전부 · skipped
    t.f_which.setCurrentIndex(1)
    assert t.grid.count() == n_ok
    t.f_which.setCurrentIndex(0)
    # 선택 → 상세 · 판정 A → 자동 저장 + 다음으로 이동
    first_ok = next(it.index for it in items if it.status == "ok")
    t.select_index(first_ok)
    qapp.processEvents()
    assert (
        t.detail.pixmap() is not None and not t.detail.pixmap().isNull() and "GT" in t.meta.text()
    )
    assert t.btn_accept.isEnabled()
    saved: list[str] = []
    t.review_saved.connect(saved.append)
    assert t.verdict("accept") == 1
    assert saved and read_review(output_root / "review.csv")[first_ok][0] == "accept"
    labels = [t.grid.item(i).text() for i in range(t.grid.count())]
    assert f"✓ {first_ok}" in labels
    assert t.current_index() != first_ok  # 판정 후 다음으로
    # 다중 선택 반려 · 히스토그램에서 반려 제외
    t.cb_next.setChecked(False)
    others = [it.index for it in items if it.status == "ok" and it.index != first_ok]
    if others:
        before = sum(t.hist.hist.a)
        for i in range(t.grid.count()):
            it = t.grid.item(i)
            it.setSelected(str(it.data(Qt.ItemDataRole.UserRole)) in others)
        qapp.processEvents()
        assert "한꺼번에" in t.meta.text() or len(others) == 1
        assert t.verdict("reject") == len(others)
        assert sum(t.hist.hist.a) < before
        assert t.session.counts()["reject"] == len(others)
    # 정상 이미지는 판정 불가(버튼 비활성)
    normal = next(it.index for it in items if it.status == "normal")
    t.select_index(normal)
    qapp.processEvents()
    assert not t.btn_accept.isEnabled() and t.verdict("accept") == 0
    # 메모
    t.select_index(first_ok)
    qapp.processEvents()
    t.note.setText("좋음")
    t.note.editingFinished.emit()
    assert read_review(output_root / "review.csv")[first_ok] == ("accept", "좋음")
    # 정리본
    assert t.export_pruned(tmp_path / "pruned")
    assert (tmp_path / "pruned" / "manifest.csv").is_file() and "정리본" in t.result.text()
    t.close()


def test_main_window_has_review_tab(qapp: QApplication) -> None:
    win = MainWindow(start_worker=False)
    try:
        assert win.tabs.tabText(4).startswith("검수") and isinstance(win.review, ReviewTab)
    finally:
        win.close()
        qapp.processEvents()


def test_review_tab_report_button(qapp: QApplication, output_root: Path, tmp_path: Path) -> None:  # noqa: F811
    t = ReviewTab()
    t.open_report_in_browser = False
    assert t.write_report() is None  # 열기 전
    t.open_root(output_root)
    p = t.write_report()
    assert p is not None and p.is_file() and "리포트" in t.result.text()
    # 분포 키에 조명 방향 — 제목에 일관성 R
    i = t.dist_key.findData("lighting")
    assert i >= 0
    t.dist_key.setCurrentIndex(i)
    assert "조명 방향" in t.hist.title and "일관성 R" in t.hist.title
    # 클래스 콤보(0.8): 외형 지표에서만 켜지고, 고르면 제목에 클래스·클래스별 R(n)
    # 클래스 콤보(0.8): 외형 지표에서만 켜지고(샘플은 조명 값이 없어 대비로), 고르면 제목에 클래스
    t.dist_key.setCurrentIndex(t.dist_key.findData("contrast"))
    assert t.dist_class.isEnabled() and t.dist_class.count() >= 2
    t.dist_class.setCurrentIndex(1)
    cls = str(t.dist_class.currentData())
    assert f"클래스 {cls}" in t.hist.title
    t.dist_key.setCurrentIndex(t.dist_key.findData("texture"))
    assert t.dist_class.currentData() == cls  # 키를 바꿔도 선택 유지
    t.dist_key.setCurrentIndex(t.dist_key.findData("area"))
    assert not t.dist_class.isEnabled() and t.dist_class.currentData() == ""  # 면적은 전체만
    # 실측 CSV(0.8): 버튼으로 실제 계열 교체 → 제목 · 되돌리기
    csv = tmp_path / "real.csv"
    csv.write_text("area\n10\n20\n30\n", encoding="utf-8")
    assert t.load_real_csv(csv) and t.btn_real_csv_clear.isEnabled()
    assert "실제 = 실측 real.csv" in t.hist.title and sum(t.hist.hist.b) == 3
    assert not t.load_real_csv(tmp_path / "none.csv") and "실측 CSV 실패" in t.result.text()
    t.clear_real_csv()
    assert not t.btn_real_csv_clear.isEnabled() and "실측" not in t.hist.title
    t.close()


def test_next_unreviewed_cycles(qapp: QApplication, output_root: Path) -> None:  # noqa: F811
    t = ReviewTab()
    t.open_root(output_root)
    t.cb_next.setChecked(False)
    oks = [it.index for it in t.session.items if it.status == "ok"]
    first = t.next_unreviewed()
    assert first == oks[0]
    t.verdict("accept")
    nxt = t.next_unreviewed()
    assert nxt == (oks[1] if len(oks) > 1 else None) or nxt != first
    for _ in oks:
        if t.current_index() in oks and t.session.item(t.current_index()).verdict == "":
            t.verdict("reject")
        t.next_unreviewed()
    assert t.session.counts()["unreviewed"] == 0 and t.next_unreviewed() is None
    t.close()


def test_reject_shown_applies_to_filtered_rows(qapp: QApplication, output_root: Path) -> None:  # noqa: F811
    """0.7.5+ — '표시된 것 전부 반려' 는 현재 필터의 합성 결과 전부(정상·skipped 제외)에 반려."""
    t = ReviewTab()
    t.autosave = False
    assert t.open_root(output_root)
    n_ok = sum(1 for it in t.session.items if it.status == "ok")
    t.f_which.setCurrentIndex(t.f_which.findData("unreviewed"))
    assert t.verdict_shown("reject") == n_ok
    assert t.session.counts()["reject"] == n_ok and t.session.counts()["unreviewed"] == 0
    # 미검수 필터엔 이제 아무것도 없다 → 0건
    assert t.verdict_shown("reject") == 0
    t.f_which.setCurrentIndex(t.f_which.findData("all"))
    assert t.verdict_shown("accept") == n_ok and t.session.counts()["accept"] == n_ok
    t.close()

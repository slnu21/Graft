"""배치 탭 GUI 스모크 — PySide6 없으면 skip. offscreen 에서 ``BatchTab`` 이 스레드로 실제 레시피를 돌려 진행률·로그·완료 시그널을
내고, 중지 요청이 부분 출력으로 끝나며, 메인 창의 스튜디오 "배치로 보내기"가 배치 탭으로 레시피를 넘기는지 확인한다."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from anograft.bank.importers import yolo as Y
from anograft.gui.app import MainWindow
from anograft.gui.batch.tab import BatchTab
from anograft.gui.studio.session import StudioSession, default_recipe
from anograft.gui.theme import apply_theme
from tests.fixtures import fake_yolo_dataset


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    return app


def _pump(qapp: QApplication, until, timeout: float = 90.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        qapp.processEvents()
        if until():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def recipe_file(tmp_path: Path) -> Path:
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
    data = {
        "version": 1,
        "name": "b",
        "seed": 3,
        "inputs": {"bank": (tmp_path / "bank").as_posix(), "targets": normals.as_posix()},
        "output": {"root": (tmp_path / "out").as_posix(), "count": 5, "include_normals": True},
        "pipeline": {
            "preset": "hard-paste",
            "source": {"method": "bank", "min_sources_warn": 1},
            "placement": {"roi": {"method": "otsu", "erode_px": 2}, "margin_px": 4},
        },
    }
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def test_batch_tab_runs_recipe_with_progress_and_log(
    qapp: QApplication, recipe_file: Path, tmp_path: Path
) -> None:
    tab = BatchTab()
    tab.show()
    assert not tab.btn_run.isEnabled() and not tab.start()  # 레시피 없음
    assert tab.open_recipe(recipe_file)
    assert tab.btn_run.isEnabled() and tab.count.value() == 5 and tab.writer.currentText() == "yolo"
    tab.out.setText((tmp_path / "gui-out").as_posix())
    tab.workers.setValue(0)
    finished: list[object] = []
    tab.run_finished.connect(finished.append)
    assert tab.start()
    assert tab.running and not tab.btn_run.isEnabled() and tab.btn_stop.isEnabled()
    assert _pump(qapp, lambda: bool(finished))
    assert _pump(qapp, lambda: tab.worker is None)
    summary = finished[0]
    assert summary is not None and not summary.cancelled and summary.done == 5
    assert tab.bar.value() == 5 and tab.progress_label.text() == "완료"
    log = tab.log.toPlainText()
    assert log.startswith("$ anograft run") and "완료: ok" in log and "manifest" in log
    assert (tmp_path / "gui-out" / "manifest.csv").is_file() and (
        tmp_path / "gui-out" / "data.yaml"
    ).is_file()
    assert tab.btn_run.isEnabled() and tab.btn_open_out.isEnabled()
    # 같은 설정으로 mvtec 형식
    tab.writer.setCurrentText("mvtec")
    assert tab.category.isEnabled()
    tab.category.setText("plate")
    tab.out.setText((tmp_path / "gui-mvtec").as_posix())
    finished.clear()
    assert tab.start()
    assert _pump(qapp, lambda: bool(finished)) and _pump(qapp, lambda: tab.worker is None)
    assert (tmp_path / "gui-mvtec" / "mvtec" / "plate" / "train" / "good").is_dir()


def test_batch_tab_stop_keeps_partial_output(
    qapp: QApplication, recipe_file: Path, tmp_path: Path
) -> None:
    tab = BatchTab()
    tab.show()
    tab.open_recipe(recipe_file)
    tab.count.setValue(40)
    tab.out.setText((tmp_path / "part").as_posix())
    finished: list[object] = []
    tab.run_finished.connect(finished.append)
    assert tab.start()
    assert _pump(qapp, lambda: tab.bar.value() >= 2)
    tab.stop()
    assert _pump(qapp, lambda: bool(finished)) and _pump(qapp, lambda: tab.worker is None)
    summary = finished[0]
    assert summary is not None and summary.cancelled and 2 <= summary.done < 40
    assert tab.progress_label.text() == "취소됨" and "취소됨" in tab.log.toPlainText()
    assert (tmp_path / "part" / "manifest.csv").is_file()


def test_batch_tab_reports_validation_error(qapp: QApplication, recipe_file: Path) -> None:
    tab = BatchTab()
    tab.open_recipe(recipe_file)
    tab.out.setText("")
    assert not tab.start() and "출력 폴더" in tab.log.toPlainText()
    assert not tab.open_recipe(recipe_file.parent / "nope.yaml")


def test_main_window_send_to_batch(qapp: QApplication, recipe_file: Path, tmp_path: Path) -> None:
    ses = StudioSession(
        default_recipe(
            bank=(tmp_path / "bank").as_posix(), targets=(tmp_path / "normals.txt").as_posix()
        )
    )
    win = MainWindow(ses, start_worker=False)
    try:
        win.show()
        assert win.batch.session.recipe is None
        win.studio.send_to_batch()
        assert win.tabs.currentWidget() is win.batch
        assert (
            win.batch.session.recipe is not None
            and win.batch.session.recipe.name == ses.recipe.name
        )
        assert "(스튜디오 레시피" in win.batch.recipe_label.text()
        assert win.tabs.tabText(3).startswith("배치")
    finally:
        win.close()
        qapp.processEvents()


def test_batch_finish_enables_review_and_main_window_opens_review_tab(
    qapp: QApplication, recipe_file: Path, tmp_path: Path
) -> None:
    """v0.7.x — 배치 완료 → '검수 탭에서 열기' → 검수 탭이 그 출력을 열고 전환."""
    win = MainWindow(start_worker=False)
    try:
        win.show()
        tab = win.batch
        assert not tab.btn_review.isEnabled()
        assert tab.open_recipe(recipe_file)
        tab.out.setText((tmp_path / "gui-review").as_posix())
        tab.workers.setValue(0)
        tab.count.setValue(2)
        finished: list[object] = []
        tab.run_finished.connect(finished.append)
        assert tab.start()
        assert _pump(qapp, lambda: bool(finished)) and _pump(qapp, lambda: tab.worker is None)
        assert tab.btn_review.isEnabled()
        assert win.review.root_edit.text() == (tmp_path / "gui-review").as_posix()  # 경로만 채워짐
        assert not win.review.session.loaded
        tab.request_review()
        qapp.processEvents()
        assert win.tabs.currentWidget() is win.review and win.review.session.loaded
        assert win.review.session.root == tmp_path / "gui-review"
    finally:
        win.close()
        qapp.processEvents()

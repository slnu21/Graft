"""v0.7 은행 탭 GUI 스모크 — PySide6 없으면 skip. offscreen 에서 ``BankTab`` 을 열어 그리드·필터·상세·삭제를 굴리고, 메인 창에서
"라벨 탭에서 다듬기" → 라벨 탭 은행 소스 편집 모드 → 저장 → 은행 탭이 같은 id 에 덮어쓰고 스튜디오가 다시 준비되는 왕복을 확인한다."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from anograft.bank import Bank
from anograft.bank.importers import yolo as Y
from anograft.gui.app import MainWindow
from anograft.gui.bank.tab import BankTab, detail_image
from anograft.gui.studio.session import StudioSession, default_recipe
from anograft.gui.theme import apply_theme
from tests.fixtures import fake_yolo_dataset, line_defect
from tests.test_gui_label import _drag


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
def bank_root(tmp_path: Path) -> Path:
    d = fake_yolo_dataset(tmp_path / "ds")
    Y.import_yolo(
        d["images"],
        d["labels"],
        d["names"],
        tmp_path / "bank",
        mask_from="ellipse",
        list_normals=tmp_path / "normals.txt",
        tags=["p1"],
    )
    return tmp_path / "bank"


def test_detail_image_scales_and_overlays() -> None:
    s = line_defect(16, 4)
    img = detail_image(s.image, s.mask, long_side=120)
    assert img.shape[1] == 120 and img.dtype == np.uint8
    assert not np.array_equal(img, img[..., ::-1])  # 색 오버레이가 들어감(회색 원본과 다름)


def test_bank_tab_grid_filter_detail_delete(qapp: QApplication, bank_root: Path) -> None:
    t = BankTab()
    t.confirm_delete = None
    t.resize(1300, 800)
    t.show()
    qapp.processEvents()
    assert not t.open_bank("") and "지정" in t.result.text()
    assert not t.open_bank(bank_root.parent / "nope")
    assert t.open_bank(bank_root)
    assert t.grid.count() == 5 and "조각 5" in t.summary.text()
    assert [t.f_cls.itemText(i) for i in range(t.f_cls.count())] == ["(전체)", "spot", "crack"]
    assert [t.f_tag.itemText(i) for i in range(t.f_tag.count())] == ["(전체)", "p1"]
    # 필터: 클래스 · 추정만 · 저신뢰만(ellipse 는 대비가 낮아 몇 개는 저신뢰) · 검색
    t.f_cls.setCurrentText("crack")
    assert t.grid.count() == 3 and "표시 3 / 전체 5" in t.count.text()
    t.f_cls.setCurrentIndex(0)
    t.f_est.setChecked(True)
    assert t.grid.count() == 4  # 폴리곤 1개 제외
    t.f_est.setChecked(False)
    t.f_text.setText("d2")
    assert t.grid.count() == 1 and t.grid.item(0).data(0x0100) == "crack/d2-01"
    t.f_text.setText("")
    t.sort.setCurrentIndex(2)  # area
    areas = [
        next(r for r in t._rows if r.id == t.grid.item(i).data(0x0100)).area_px
        for i in range(t.grid.count())
    ]
    assert areas == sorted(areas)
    # 상세
    t.grid.item(0).setSelected(True)
    qapp.processEvents()
    assert t.detail.pixmap() is not None and not t.detail.pixmap().isNull()
    assert "크롭" in t.meta.text() and t.btn_edit.isEnabled() and t.btn_delete.isEnabled()
    # 삭제 → 파일·그리드·시그널
    changed: list[str] = []
    t.bank_changed.connect(changed.append)
    sid = t.current_id()
    assert sid is not None
    assert t.delete_selected() == 1
    assert changed == [bank_root.as_posix()] and t.grid.count() == 4
    cls, name = sid.split("/")
    assert not list((bank_root / cls).glob(f"{name}.*")) and len(Bank.load(bank_root)) == 4
    assert t.delete_selected() == 0  # 선택 없음
    t.close()


def test_main_window_refine_roundtrip(qapp: QApplication, bank_root: Path, tmp_path: Path) -> None:
    normals = tmp_path / "normals.txt"
    ses = StudioSession(default_recipe(bank=bank_root.as_posix(), targets=normals.as_posix()))
    win = MainWindow(ses, start_worker=True)
    try:
        win.show()
        win.studio.open_inputs(bank_root.as_posix(), normals.as_posix())
        assert _pump(qapp, lambda: ses.prepared is not None)
        assert win.tabs.tabText(1).startswith("② 결함 보관함")  # 흐름 순서: 결함 표시 → 보관함 → …
        win.bank.open_bank(bank_root)
        win.bank.confirm_delete = None
        win.label.confirm_discard = None
        sid = "spot/d0-01"
        win.bank.select_ids([sid])
        qapp.processEvents()
        before = win.bank.session.source(sid)
        assert before.mask_origin.startswith("yolo-box:")
        # 라벨 탭에서 다듬기 → 편집 모드
        win.bank.request_edit()
        qapp.processEvents()
        assert win.tabs.currentWidget() is win.label and win.label.edit_target == (
            bank_root.as_posix(),
            sid,
        )
        assert win.label.session.shape == before.mask.shape and np.array_equal(
            win.label.session.mask, before.mask
        )
        assert win.label.btn_save.text().startswith("보관함 조각 갱신")
        # 브러시로 조금 더 칠하고 저장 → 은행 탭 apply_mask → 파일 갱신 + 스튜디오 재준비
        win.label.canvas.set_tool("brush")
        win.label.canvas.set_brush(3)
        w = before.mask.shape[1]
        _drag(win.label.canvas, [(2.0, 2.0), (float(w - 3), 2.0)])
        gen = ses.generation
        assert win.label.save()
        qapp.processEvents()
        after = Bank.load(bank_root)
        src = next(s for s in after.sources() if s.id == sid)
        assert src.mask_origin == "manual:brush" and src.confidence is None
        assert np.count_nonzero(src.mask) > np.count_nonzero(before.mask)
        # 편집 모드는 유지(계속 다듬어 다시 저장) — 다른 이미지를 열면 끝난다
        assert win.label.edit_target is not None and not win.label.session.dirty
        assert win.bank.session.source(sid).mask_origin == "manual:brush"
        assert _pump(qapp, lambda: ses.prepared is not None and ses.generation > gen)
        # 라벨 탭에서 새 소스를 저장하면 은행 탭도 새로고침
        n_before = win.bank.grid.count()
        win.label.open_image(next(iter((tmp_path / "ds" / "images").glob("n0.png"))))
        assert win.label.edit_target is None and win.label.btn_save.text().startswith(
            "보관함에 저장"
        )
        win.label.canvas.set_tool("brush")
        _drag(win.label.canvas, [(20.0, 40.0), (50.0, 40.0)])
        win.label.cls.setEditText("spot")
        assert win.label.save()
        qapp.processEvents()
        assert win.bank.grid.count() == n_before + 1
    finally:
        win.worker.stop()
        win.close()
        qapp.processEvents()


def test_refine_all_low_loop(qapp: QApplication, bank_root: Path, tmp_path: Path) -> None:
    """v0.7.x — 은행 탭 '저신뢰 전부 차례로 다듬기' → 라벨 탭 편집 → '다음 저신뢰 소스' → … → 없으면 은행 탭으로."""
    win = MainWindow(start_worker=False)
    try:
        win.show()
        win.label.confirm_discard = None
        # 픽스처 얼룩은 대비가 커서 저신뢰가 없다 — 메타를 고쳐 셋을 저신뢰로
        import json

        for k, meta_file in enumerate(sorted(bank_root.rglob("*.json"))[:3]):
            m = json.loads(meta_file.read_text(encoding="utf-8"))
            m["confidence"], m["flags"] = 0.1 * (k + 1), ["low-contrast"]
            meta_file.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
        win.bank.open_bank(bank_root)
        low = [r.id for r in win.bank.session.filtered(only_low=True, sort="confidence")]
        assert len(low) == 3 and win.bank.btn_edit_low.isEnabled()
        assert win.bank.next_low_confidence(None) == low[0]
        assert win.bank.next_low_confidence(low[0]) == low[1]
        assert win.bank.next_low_confidence(low[-1]) == low[0]  # 순환
        assert win.bank.request_edit_next_low(None)
        qapp.processEvents()
        assert win.tabs.currentWidget() is win.label and win.label.edit_target[1] == low[0]
        assert not win.label.btn_next_low.isHidden()
        # 저장 없이 다음으로 → 두 번째 저신뢰
        win.label.request_next_low()
        qapp.processEvents()
        assert win.label.edit_target[1] == low[1]
        # 다듬어 저장하면 저신뢰에서 빠진다 → 목록이 줄어든다
        w = win.label.session.mask.shape[1]
        win.label.canvas.set_tool("brush")
        _drag(win.label.canvas, [(1.0, 1.0), (float(w - 2), 1.0)])
        assert win.label.save()
        qapp.processEvents()
        remaining = [r.id for r in win.bank.session.filtered(only_low=True)]
        assert low[1] not in remaining and len(remaining) == len(low) - 1
        # 편집 모드 종료 시 버튼 숨김
        win.label.open_image(next(iter((tmp_path / "ds" / "images").glob("n0.png"))))
        assert win.label.btn_next_low.isHidden()
    finally:
        win.close()
        qapp.processEvents()

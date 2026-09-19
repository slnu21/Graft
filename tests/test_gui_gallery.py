"""v0.9 사용성 ④ preset-gallery — 카드 문안(Qt 없음) · 썸네일 합성 · 대화상자 선택/확정 · 스트립 '고르기…' → 콤보 → 세션."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from anograft import runner
from anograft.core import recipe as R
from anograft.gui.studio.gallery import (
    PresetCard,
    preset_cards,
    recipe_with_preset,
    render_preset_thumbs,
)
from tests.fixtures import fake_yolo_dataset


def test_preset_cards_from_meta() -> None:
    cards = preset_cards()
    assert [c.name for c in cards] == R.preset_names()
    by = {c.name: c for c in cards}
    rp = by["relative-paste"]
    assert rp.title == "대비 지키며 붙이기" and "블로우홀" in rp.use_for and rp.evidence
    labels = dict((m, ml) for _lab, m, ml in rp.stages)
    assert labels["relative"] == "노출만 맞춤" and labels["paste"] == "그대로 붙이기"
    assert any(lab == "붙일 수 있는 영역" for lab, _m, _ml in rp.stages)
    assert isinstance(rp, PresetCard)


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[runner.Prepared, R.Recipe]:
    d = fake_yolo_dataset(tmp_path / "ds")
    from anograft.bank.importers import yolo as Y

    normals = tmp_path / "normals.txt"
    Y.import_yolo(
        d["images"],
        d["labels"],
        d["names"],
        tmp_path / "bank",
        mask_from="rect",
        list_normals=normals,
    )
    data = R.init_recipe_dict(
        "poisson-graft",
        name="g",
        bank=(tmp_path / "bank").as_posix(),
        targets=normals.as_posix(),
        out=(tmp_path / "out").as_posix(),
        count=2,
    )
    rec = R.Recipe.from_dict(data)
    return runner.prepare(rec), rec


def test_render_preset_thumbs(prepared: tuple[runner.Prepared, R.Recipe]) -> None:
    prep, rec = prepared
    t0 = time.time()
    th = render_preset_thumbs(
        prep, rec, prep.targets[0], ["poisson-graft", "hard-paste", "self-cut"], long_side=128
    )
    assert set(th) == {"poisson-graft", "hard-paste", "self-cut"}
    for v in th.values():
        assert v is not None and v.ndim == 3 and max(v.shape[:2]) <= 128
    assert time.time() - t0 < 30
    # 같은 시드·같은 바탕이면 결정적
    again = render_preset_thumbs(prep, rec, prep.targets[0], ["poisson-graft"], long_side=128)
    assert (again["poisson-graft"] == th["poisson-graft"]).all()
    # 프리셋 교체는 pipeline 을 통째로
    r2 = recipe_with_preset(rec, "hard-paste")
    assert r2.pipeline.preset == "hard-paste" and r2.pipeline.blend.method == "paste"
    assert r2.inputs == rec.inputs and r2.seed == rec.seed


pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from anograft.gui.app import MainWindow  # noqa: E402
from anograft.gui.studio.preset_gallery import PresetGalleryDialog  # noqa: E402
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


def test_gallery_dialog_select_and_activate(qapp: QApplication) -> None:
    import numpy as np

    thumbs = {"poisson-graft": np.full((60, 80, 3), 90, dtype=np.uint8), "self-cut": None}
    dlg = PresetGalleryDialog("poisson-graft", thumbs)
    assert len(dlg.cards) == len(R.preset_names()) and dlg.chosen == "poisson-graft"
    assert dlg.cards["poisson-graft"].property("selected") is True
    assert dlg.cards["poisson-graft"].thumb.pixmap() is not None
    assert "돌 수 없음" in dlg.cards["self-cut"].thumb.text()
    assert dlg.cards["dent-graft"].thumb.text() == "썸네일 없음"  # 썸네일 dict 에 없음
    dlg.cards["dent-graft"].clicked.emit("dent-graft")
    assert dlg.chosen == "dent-graft" and dlg.cards["poisson-graft"].property("selected") is False
    accepted: list[int] = []
    dlg.accepted.connect(lambda: accepted.append(1))
    dlg.cards["relative-paste"].activated.emit("relative-paste")  # 더블클릭 = 확정
    assert dlg.chosen == "relative-paste" and accepted
    # 현재 없이 열면 확정 버튼 비활성
    d2 = PresetGalleryDialog(None, None)
    assert d2.chosen is None and not d2.btn_ok.isEnabled()
    d2.select("hard-paste")
    assert d2.btn_ok.isEnabled()


def test_studio_gallery_applies_preset(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    ses = StudioSession(default_recipe())
    win = MainWindow(ses)
    try:
        tab = win.studio
        assert tab.strip.btn_gallery.toolTip().startswith("Preset gallery")
        assert tab.gallery_thumbs() is None  # 준비 전
        # 대화상자를 띄우지 않고 '고름'을 흉내
        from anograft.gui.studio import preset_gallery as pg

        class Fake:
            def __init__(self, *a, **k):
                self.chosen = "dent-graft"

            def exec(self):
                from PySide6.QtWidgets import QDialog

                return QDialog.DialogCode.Accepted

        monkeypatch.setattr(pg, "PresetGalleryDialog", Fake)
        assert tab.open_preset_gallery() == "dent-graft"
        assert _pump(qapp, lambda: ses.recipe.pipeline.preset == "dent-graft")
        assert tab.strip.preset.currentData() == "dent-graft"
        assert ses.recipe.pipeline.geometry.rotate == (-15.0, 15.0)
    finally:
        win.close()

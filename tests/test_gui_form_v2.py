"""v0.9 사용성 ③ param-form-v2 — 카드 폼: 프리셋 대비 바뀜 ●/↺ 되돌리기 · 카드 헤더 "바뀜 n · 프리셋 값으로" · 고급 옵션 접기 ·
0..1 슬라이더(더블클릭 되돌리기) · 단계 켬/끔 토글. PySide6 없으면 skip, offscreen."""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from anograft.core import recipe as R
from anograft.gui.app import MainWindow
from anograft.gui.studio import param_form
from anograft.gui.studio.param_form import DEBOUNCE_MS, ParamForm, wants_slider
from anograft.gui.studio.params import baseline_config, field_specs
from anograft.gui.studio.session import StudioSession, default_recipe
from anograft.gui.theme import apply_theme


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


def _wait(qapp: QApplication, ms: int) -> None:
    t0 = time.time()
    while (time.time() - t0) * 1000 < ms:
        qapp.processEvents()
        time.sleep(0.01)


def test_baseline_config_pure() -> None:
    """레시피 프리셋의 같은 method 블록 → 기준; method 가 다르면 스키마 기본; 필수 필드가 있으면 None."""
    rec = default_recipe()
    assert rec.pipeline.preset == "poisson-graft"
    b = baseline_config(rec, "harmonize", rec.pipeline.harmonize)
    assert b is not None and b.strength == 0.3  # 프리셋 값(스키마 기본 0.5 가 아님)
    alpha = R.AlphaBlendConfig(feather_px=9)
    b2 = baseline_config(rec, "blend", alpha)
    assert b2 is not None and b2.feather_px == 3  # 프리셋은 poisson → 스키마 기본
    assert baseline_config(rec, "roi", R.MaskDirRoiConfig(path="x")) is None
    specs = {s.name: s for s in field_specs(alpha, baseline=b2)}
    assert specs["feather_px"].modified and specs["feather_px"].baseline == 3
    assert not specs["opacity"].modified
    assert not field_specs(alpha)[0].has_baseline


def test_form_modified_marker_reset_and_advanced(qapp: QApplication) -> None:
    form = ParamForm()
    rec = default_recipe()
    cfg = rec.pipeline.harmonize.model_copy(update={"strength": 0.7})
    changes: list[tuple[str, object]] = []
    form.value_changed.connect(lambda n, v: changes.append((n, v)))
    form.set_specs(field_specs(cfg, baseline=baseline_config(rec, "harmonize", cfg)))
    row = form.rows["strength"]
    assert (
        row.modified and row.btn_reset.isVisibleTo(form) and form.modified_names() == ["strength"]
    )
    assert form._labels["strength"].property("modified") is True
    # 고급 옵션: ring_px 는 advanced → 접힘, 버튼 텍스트에 개수
    assert form._advanced == ["ring_px"] and not form.btn_advanced.isChecked()
    assert form.rows["ring_px"].isHidden() and "고급 옵션 (1)" in form.btn_advanced.text()
    form.set_advanced_open(True)
    assert not form.rows["ring_px"].isHidden() and form.btn_advanced.text().startswith("▾")
    # ↺ → 기준값 + value_changed
    row.btn_reset.click()
    assert changes and changes[-1] == ("strength", 0.3) and not row.modified
    assert form._labels["strength"].property("modified") is False
    # 슬라이더: strength 는 0..1 → 슬라이더 있음, 끌면 스핀이 따라오고 디바운스 뒤 커밋
    assert row.slider is not None and wants_slider(row.spec)
    row.slider.setValue(500)
    assert abs(row.editors[0].value() - 0.5) < 1e-6
    _wait(qapp, DEBOUNCE_MS + 100)
    assert changes[-1][0] == "strength" and abs(changes[-1][1] - 0.5) < 1e-6 and row.modified
    row.slider.reset_requested.emit()  # 더블클릭과 같은 경로
    assert changes[-1] == ("strength", 0.3) and not row.modified
    # ring_px(정수) 에는 슬라이더 없음
    assert form.rows["ring_px"].slider is None
    form.deleteLater()


def test_advanced_state_remembered_with_settings(qapp: QApplication, tmp_path) -> None:
    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    param_form.use_settings(store)
    try:
        rec = default_recipe()
        f1 = ParamForm()
        f1.set_specs(field_specs(rec.pipeline.geometry))
        assert not f1.btn_advanced.isChecked()
        f1.set_advanced_open(True)
        f2 = ParamForm()
        f2.set_specs(field_specs(rec.pipeline.geometry))
        assert f2.btn_advanced.isChecked() and not f2.rows["elastic.alpha"].isHidden()
        f1.deleteLater()
        f2.deleteLater()
    finally:
        param_form.use_settings(None)


def test_card_header_and_stage_toggle(qapp: QApplication) -> None:
    """카드 헤더: 값이 바뀌면 "바뀜 n" + "프리셋 값으로"(전부 되돌림) · 색·밝기 맞추기/카메라 효과 카드는 켬/끔 토글."""
    ses = StudioSession(default_recipe())
    win = MainWindow(ses)
    try:
        pipe = win.studio.pipe
        harm = pipe.cards["harmonize"]
        geo = pipe.cards["geometry"]
        assert harm.enabled_toggle is not None and harm.enabled_toggle.isChecked()
        assert geo.enabled_toggle is None  # none 이 없는 단계
        assert harm.modified_label.isHidden() and harm.btn_reset_all.isHidden()
        # 값 편집 → 헤더
        harm.form.rows["strength"].editors[0].setValue(0.8)
        assert _pump(qapp, lambda: ses.recipe.pipeline.harmonize.strength == 0.8)
        assert not harm.modified_label.isHidden() and harm.modified_label.text() == "바뀜 1"
        assert not harm.btn_reset_all.isHidden()
        harm.btn_reset_all.click()
        assert _pump(qapp, lambda: ses.recipe.pipeline.harmonize.strength == 0.3)
        assert harm.modified_label.isHidden()
        # 끔 → none, 켬 → 마지막 method(stats)
        harm.enabled_toggle.setChecked(False)
        assert _pump(qapp, lambda: ses.recipe.pipeline.harmonize.method == "none")
        assert harm.method.currentData() == "none" and not harm.enabled_toggle.isChecked()
        harm.enabled_toggle.setChecked(True)
        assert _pump(qapp, lambda: ses.recipe.pipeline.harmonize.method == "stats")
        # 카메라 효과도 토글 있음 · ROI 부카드의 바뀜도 배치 카드 헤더에 합산
        assert pipe.cards["degrade"].enabled_toggle is not None
        pl = pipe.cards["placement"]
        assert pl.roi_form is not None
        pl.roi_form.rows["erode_px"].editors[0].setValue(20)
        assert _pump(qapp, lambda: ses.recipe.pipeline.placement.roi.erode_px == 20)
        assert pl.modified_label.text() == "바뀜 1"
        pl.btn_reset_all.click()
        assert _pump(qapp, lambda: ses.recipe.pipeline.placement.roi.erode_px == 8)
        # 프리셋을 바꾸면 기준이 그 프리셋으로 — 바뀜 0
        win.studio.strip.preset.setCurrentIndex(win.studio.strip.preset.findData("dent-graft"))
        assert _pump(qapp, lambda: ses.recipe.pipeline.preset == "dent-graft")
        assert geo.modified_count() == 0 and geo.form.rows["rotate"].spec.baseline == [-15.0, 15.0]
    finally:
        win.close()


def test_notice_component(qapp: QApplication) -> None:
    """v0.9 ⑥ Notice — 문장/버튼 중 하나라도 있으면 보이고, 둘 다 비면 숨는다 · level 속성 · action 시그널."""
    from anograft.gui.notice import Notice

    n = Notice("warn")
    assert n.isHidden() and n.property("level") == "warn" and n.icon.text() == "⚠"
    n.set_text("무엇이 · 왜 → 이렇게")
    assert not n.isHidden() and n.text() == "무엇이 · 왜 → 이렇게" and n.button.isHidden()
    fired: list[int] = []
    n.action.connect(lambda: fired.append(1))
    n.set_action("고치기")
    assert not n.button.isHidden() and n.button.text() == "고치기"
    n.button.click()
    assert fired == [1]
    n.set_text(None)
    assert not n.isHidden()  # 버튼이 남아 있다
    n.set_action(None)
    assert n.isHidden()
    n.set_level("hint")
    assert n.property("level") == "hint" and n.icon.text() == "💡"
    n.deleteLater()

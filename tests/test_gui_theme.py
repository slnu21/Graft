"""v0.9 사용성 ⑦ a11y-theme — 라이트/다크 팔레트 · 글자 배율 QSS · 설정 저장/복원 · 대비 · 설정 대화상자."""

from __future__ import annotations

import os
import re

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from anograft.gui import theme
from anograft.gui.settings_dialog import SettingsDialog


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    theme.apply_theme(app)
    return app


def _lum(hexcolor: str) -> float:
    r, g, b = (int(hexcolor[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def ch(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a: str, b: str) -> float:
    la, lb = _lum(a), _lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def test_palettes_contrast() -> None:
    """본문·보조 텍스트가 배경·패널에서 WCAG AA(4.5:1) 이상 — 두 테마 모두."""
    for name, pal in theme.THEMES.items():
        for txt in ("tx", "tx2", "tx3"):
            for bg in ("bg", "panel", "panel2"):
                assert contrast(pal[txt], pal[bg]) >= 4.5, (
                    name,
                    txt,
                    bg,
                    contrast(pal[txt], pal[bg]),
                )
        assert contrast(pal["teal"], pal["panel"]) >= 3.0, name  # 강조색은 큰 글자/아이콘 기준 3:1
        assert set(pal) == set(theme.DARK)


def test_apply_theme_switches_colors_in_place(qapp: QApplication) -> None:
    before = dict(theme.COLORS)
    try:
        theme.apply_theme(qapp, mode="light", font_scale=1.3)
        assert theme.COLORS["bg"] == theme.LIGHT["bg"] and theme.COLORS is not theme.LIGHT
        css = qapp.styleSheet()
        assert f"font-size: {12.5 * 1.3:.1f}px" in css  # QWidget 기본 12.5 × 1.3
        assert theme.LIGHT["panel"] in css and theme.DARK["bg"] not in css
        # 알 수 없는 테마 → 다크
        theme.apply_theme(qapp, mode="nope")
        assert theme.COLORS["bg"] == theme.DARK["bg"] and "font-size: 12.5px" in qapp.styleSheet()
    finally:
        theme.apply_theme(qapp, mode="dark", font_scale=1.0)
        assert before == theme.COLORS


def test_stylesheet_scale_only_touches_font_sizes() -> None:
    base = theme.stylesheet(1.0)
    big = theme.stylesheet(1.15)
    assert base.count("font-size") == big.count("font-size")
    assert re.sub(r"font-size:\s*[0-9.]+px", "F", base) == re.sub(
        r"font-size:\s*[0-9.]+px", "F", big
    )
    sizes_a = [float(x) for x in re.findall(r"font-size:\s*([0-9.]+)px", base)]
    sizes_b = [float(x) for x in re.findall(r"font-size:\s*([0-9.]+)px", big)]
    assert all(abs(b - a * 1.15) < 0.06 for a, b in zip(sizes_a, sizes_b, strict=True))


def test_settings_roundtrip_and_dialog(qapp: QApplication, tmp_path) -> None:
    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    assert theme.theme_settings(store) == ("dark", 1.0)
    assert theme.theme_settings(None) == ("dark", 1.0)
    theme.save_theme_settings(store, "light", 1.15)
    assert theme.theme_settings(store) == ("light", 1.15)
    store.setValue(theme.THEME_KEY, "bogus")
    store.setValue(theme.FONT_SCALE_KEY, "x")
    assert theme.theme_settings(store) == ("dark", 1.0)  # 이상한 값은 기본으로
    theme.save_theme_settings(store, "light", 1.3)
    dlg = SettingsDialog(store)
    assert dlg.values() == ("light", 1.3)
    dlg.theme.setCurrentIndex(dlg.theme.findData("dark"))
    dlg.scale.setCurrentIndex(dlg.scale.findData(1.0))
    dlg.accept()
    assert theme.theme_settings(store) == ("dark", 1.0)
    d2 = SettingsDialog(None)  # 저장소 없음(테스트 창) — 저장 안 함, 에러 없음
    d2.accept()


def test_main_window_settings_button(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    from anograft.gui import app as app_mod
    from anograft.gui.app import MainWindow
    from anograft.studio.session import StudioSession, default_recipe

    win = MainWindow(StudioSession(default_recipe()))
    try:
        assert win.btn_settings.toolTip().startswith("Settings")

        class Fake:
            def __init__(self, *a, **k):
                pass

            def exec(self):
                from PySide6.QtWidgets import QDialog

                return QDialog.DialogCode.Accepted

            def values(self):
                return ("light", 1.15)

        import anograft.gui.settings_dialog as sd

        monkeypatch.setattr(sd, "SettingsDialog", Fake)
        assert win.open_settings() is True
        assert "다시 열면 적용" in win.status_bar.currentMessage()
        assert app_mod.theme_settings(None) == ("dark", 1.0)
    finally:
        win.close()

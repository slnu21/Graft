"""다크 톤 — ``docs/mockup.html``의 CSS 변수를 그대로. 팔레트(``QPalette``) + 스타일시트(QSS) 한 쌍."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

COLORS: dict[str, str] = {
    "bg": "#14171B",
    "panel": "#1A1E23",
    "panel2": "#20252B",
    "raise": "#262C33",
    "line": "#2C333B",
    "line2": "#39424C",
    "tx": "#E3E8ED",
    "tx2": "#98A4B0",
    "tx3": "#697683",
    "stage": "#31363C",
    "teal": "#00A188",
    "amber": "#C8841C",
    "mask": "#FF4D6D",
    "ok": "#3FBF8F",
    "warn": "#D9822B",
    "bad": "#E05260",
}

# 검수 히스토그램 계열색(CLAUDE.md 규약): 합성 teal · 실제 amber. 캔버스 오버레이도 같은 계열을 쓴다.
SYNTH_COLOR = COLORS["teal"]
REAL_COLOR = COLORS["amber"]
MASK_COLOR = COLORS["mask"]


def qcolor(name: str) -> QColor:
    return QColor(COLORS[name])


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, qcolor("bg"))
    pal.setColor(QPalette.ColorRole.WindowText, qcolor("tx"))
    pal.setColor(QPalette.ColorRole.Base, qcolor("panel"))
    pal.setColor(QPalette.ColorRole.AlternateBase, qcolor("panel2"))
    pal.setColor(QPalette.ColorRole.Text, qcolor("tx"))
    pal.setColor(QPalette.ColorRole.Button, qcolor("panel2"))
    pal.setColor(QPalette.ColorRole.ButtonText, qcolor("tx2"))
    pal.setColor(QPalette.ColorRole.Highlight, qcolor("teal"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#06231E"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, qcolor("raise"))
    pal.setColor(QPalette.ColorRole.ToolTipText, qcolor("tx"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, qcolor("tx3"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, qcolor("tx3"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, qcolor("tx3"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, qcolor("tx3"))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)


STYLESHEET = f"""
QWidget {{ font-size: 12.5px; }}
QMainWindow, QDialog {{ background: {COLORS["bg"]}; }}
QToolTip {{ background: {COLORS["raise"]}; color: {COLORS["tx"]}; border: 1px solid {COLORS["line2"]}; padding: 4px 6px; }}

/* 상단 바 · 탭 */
#TopBar {{ background: {COLORS["panel"]}; border-bottom: 1px solid {COLORS["line"]}; }}
#Brand {{ font-size: 15px; font-weight: 600; color: {COLORS["tx"]}; }}
#BrandSub {{ font-size: 11.5px; color: {COLORS["tx3"]}; }}
#Ctx {{ color: {COLORS["tx2"]}; font-size: 12.5px; }}
QTabWidget::pane {{ border: none; background: {COLORS["bg"]}; }}
QTabBar {{ background: {COLORS["panel"]}; }}
QTabBar::tab {{ background: transparent; color: {COLORS["tx2"]}; padding: 8px 14px 7px; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {COLORS["tx"]}; border-bottom: 2px solid {COLORS["teal"]}; font-weight: 600; }}
QTabBar::tab:hover {{ color: {COLORS["tx"]}; }}

/* 패널 */
#Strip {{ background: {COLORS["panel2"]}; border-bottom: 1px solid {COLORS["line"]}; }}
#Rail, #Pipe, #Variants, #Inputs {{ background: {COLORS["panel"]}; }}
#Rail {{ border-right: 1px solid {COLORS["line"]}; }}
#Pipe {{ border-left: 1px solid {COLORS["line"]}; }}
#Variants {{ border-top: 1px solid {COLORS["line"]}; }}
#Stage {{ background: {COLORS["stage"]}; }}
QLabel#H4 {{ color: {COLORS["tx3"]}; font-size: 11.5px; font-weight: 600; }}
QLabel#Hint {{ color: {COLORS["tx2"]}; }}
QLabel#Muted {{ color: {COLORS["tx3"]}; font-size: 11.5px; }}
QLabel#Warn {{ color: {COLORS["warn"]}; }}
QLabel#Bad {{ color: {COLORS["bad"]}; }}

/* 입력 위젯 */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
  background: {COLORS["bg"]}; border: 1px solid {COLORS["line2"]}; border-radius: 5px; padding: 3px 7px; color: {COLORS["tx"]};
}}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {COLORS["tx3"]}; }}
QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: {COLORS["tx3"]}; border-color: {COLORS["line"]}; }}
QComboBox[error="true"], QLineEdit[error="true"], QSpinBox[error="true"], QDoubleSpinBox[error="true"] {{ border-color: {COLORS["bad"]}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {COLORS["panel2"]}; border: 1px solid {COLORS["line2"]}; selection-background-color: {COLORS["raise"]}; color: {COLORS["tx"]}; }}
QPushButton {{ background: {COLORS["panel2"]}; border: 1px solid {COLORS["line2"]}; border-radius: 5px; padding: 5px 12px; color: {COLORS["tx2"]}; }}
QPushButton:hover {{ border-color: {COLORS["tx3"]}; color: {COLORS["tx"]}; }}
QPushButton:disabled {{ color: {COLORS["tx3"]}; border-color: {COLORS["line"]}; }}
QPushButton#Primary {{ background: {COLORS["teal"]}; border-color: {COLORS["teal"]}; color: #06231E; font-weight: 600; }}
QPushButton#Primary:hover {{ background: #12B39A; }}
QPushButton#Fix {{ border-color: {COLORS["amber"]}; color: {COLORS["amber"]}; padding: 2px 8px; font-size: 11px; }}
QCheckBox {{ color: {COLORS["tx2"]}; spacing: 6px; }}
QCheckBox::indicator {{ width: 13px; height: 13px; border: 1px solid {COLORS["line2"]}; border-radius: 3px; background: {COLORS["bg"]}; }}
QCheckBox::indicator:checked {{ background: {COLORS["teal"]}; border-color: {COLORS["teal"]}; }}

/* 카드·리스트 */
QFrame#StageCard {{ background: {COLORS["panel2"]}; border: 1px solid {COLORS["line"]}; border-radius: 7px; }}
QFrame#StageCard QLabel#StageNo {{ color: #06231E; background: {COLORS["teal"]}; border-radius: 9px; min-width: 18px; max-width: 18px; min-height: 18px; max-height: 18px; font-size: 10.5px; font-weight: 700; qproperty-alignment: AlignCenter; }}
QFrame#StageCard QLabel#StageTitle {{ font-weight: 600; }}
QFrame#StageCard QLabel#StageParams {{ color: {COLORS["tx2"]}; font-size: 11.5px; }}
QFrame#StageCard QLabel#StageParams[modified="true"] {{ color: {COLORS["teal"]}; font-weight: 600; }}
QFrame#StageCard QLabel#Modified {{ color: {COLORS["teal"]}; font-size: 11px; }}
QToolButton#Reset {{ border: none; color: {COLORS["teal"]}; font-size: 13px; padding: 0; background: transparent; }}
QToolButton#Reset:hover {{ color: {COLORS["tx"]}; }}
QToolButton#ResetAll {{ border: 1px solid {COLORS["line2"]}; border-radius: 4px; color: {COLORS["teal"]}; font-size: 12px; padding: 0; background: transparent; }}
QToolButton#ResetAll:hover {{ border-color: {COLORS["teal"]}; }}
QToolButton#Advanced {{ border: none; color: {COLORS["tx3"]}; font-size: 11.5px; padding: 2px 0; background: transparent; text-align: left; }}
QToolButton#Advanced:hover, QToolButton#Advanced:checked {{ color: {COLORS["tx2"]}; }}
QSlider::groove:horizontal {{ height: 3px; background: {COLORS["line2"]}; border-radius: 1px; }}
QSlider::handle:horizontal {{ width: 10px; height: 10px; margin: -4px 0; border-radius: 5px; background: {COLORS["teal"]}; }}
QSlider::sub-page:horizontal {{ background: {COLORS["teal"]}; border-radius: 1px; }}
QSlider:disabled::handle:horizontal {{ background: {COLORS["line2"]}; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border: 1px solid {COLORS["line"]}; border-radius: 6px; background: {COLORS["panel2"]}; margin: 0 0 8px 0; padding: 4px; color: {COLORS["tx2"]}; }}
QListWidget::item:selected {{ background: {COLORS["panel2"]}; border: 1px solid {COLORS["teal"]}; color: {COLORS["tx"]}; }}
QListWidget::item:selected:active, QListWidget::item:selected:!active {{ background: {COLORS["panel2"]}; }}
QListWidget::item:hover {{ border-color: {COLORS["line2"]}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {COLORS["line2"]}; border-radius: 5px; min-height: 24px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {COLORS["line2"]}; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QStatusBar {{ background: {COLORS["panel"]}; color: {COLORS["tx3"]}; border-top: 1px solid {COLORS["line"]}; }}
QTableWidget {{ background: {COLORS["bg"]}; border: 1px solid {COLORS["line"]}; gridline-color: {COLORS["line"]}; color: {COLORS["tx2"]}; }}
QHeaderView::section {{ background: {COLORS["panel2"]}; color: {COLORS["tx3"]}; border: none; border-bottom: 1px solid {COLORS["line"]}; padding: 3px 6px; }}
QSplitter::handle {{ background: {COLORS["line"]}; }}
"""

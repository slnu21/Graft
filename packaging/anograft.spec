# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — 단일 폴더(onedir) 번들 하나에 exe 둘.

    anograft.exe        콘솔 — CLI 전부 (run·preview·bank·sample …)
    anograft-gui.exe    windowed — 스튜디오 (PySide6)

    pyinstaller packaging/anograft.spec --noconfirm --clean     # → dist/anograft/
    tools/build_zip.ps1                                          # 위 + recipes/·LICENSE·NOTICES 복사 + zip

- 프리셋(anograft/presets/*.yaml)은 ``collect_data_files`` 로 `_internal/anograft/presets/` 에 실린다 —
  ``importlib.resources.files("anograft.presets")`` 가 frozen 에서도 같은 경로를 돌려준다.
- multiprocessing spawn: 워커는 이 exe 를 다시 실행한다 → ``cli.main``/``gui.__main__.main`` 첫 줄의
  ``multiprocessing.freeze_support()`` 가 워커 분기를 잡는다(없으면 워커가 CLI 본체를 다시 돈다).
- PySide6 는 QtCore·QtGui·QtWidgets 만 쓴다 → 나머지 Qt 모듈은 excludes 로 크기 절감(LGPL 동적 링크 유지).
- UPX 는 쓰지 않는다(Qt DLL 을 깨뜨린다).
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("anograft")  # presets/*.yaml
# registry.ensure_loaded 가 스테이지 모듈을 importlib.import_module 로 지연 로드 → 정적 분석에 안 잡힌다
hidden = collect_submodules("anograft")
hidden_cli = [m for m in hidden if not m.startswith("anograft.gui")]  # CLI exe 는 Qt 를 모른다

EXCLUDES = [
    # 개발·테스트 도구
    "pytest", "_pytest", "ruff", "pip", "setuptools", "wheel",
    # 안 쓰는 표준/서드파티 대형 모듈
    "tkinter", "_tkinter", "matplotlib", "scipy", "pandas", "IPython", "jupyter",
    # PySide6 — 쓰는 건 QtCore·QtGui·QtWidgets 뿐
    "PySide6.Qt3DAnimation", "PySide6.Qt3DCore", "PySide6.Qt3DExtras", "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic", "PySide6.Qt3DRender", "PySide6.QtBluetooth", "PySide6.QtCharts",
    "PySide6.QtConcurrent", "PySide6.QtDataVisualization", "PySide6.QtDesigner", "PySide6.QtGraphs",
    "PySide6.QtHelp", "PySide6.QtHttpServer", "PySide6.QtLocation", "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets", "PySide6.QtNetwork", "PySide6.QtNetworkAuth", "PySide6.QtNfc",
    "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    "PySide6.QtPositioning", "PySide6.QtPrintSupport", "PySide6.QtQml", "PySide6.QtQuick",
    "PySide6.QtQuick3D", "PySide6.QtQuickControls2", "PySide6.QtQuickWidgets", "PySide6.QtRemoteObjects",
    "PySide6.QtScxml", "PySide6.QtSensors", "PySide6.QtSerialBus", "PySide6.QtSerialPort",
    "PySide6.QtSpatialAudio", "PySide6.QtSql", "PySide6.QtStateMachine", "PySide6.QtSvgWidgets",
    "PySide6.QtTest", "PySide6.QtTextToSpeech", "PySide6.QtUiTools", "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineQuick", "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebSockets", "PySide6.QtXml",
]

cli = Analysis(
    ["entry_cli.py"],
    pathex=["../src"],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_cli,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES + ["PySide6", "shiboken6"],  # CLI exe 는 Qt 를 모른다
    noarchive=False,
)
gui = Analysis(
    ["entry_gui.py"],
    pathex=["../src"],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

# --- 번들 다이어트 (2026-09-14 실측 252 MB → 아래 제거로 ~180 MB) -------------------------------------------
# 파일 단위로 빼는 것들 — 전부 "쓰지 않는데 훅이 끌고 오는" 것. 빼도 Qt 는 해당 플러그인/모듈을 조용히 건너뛴다.
import fnmatch  # noqa: E402

TRIM_PATTERNS = [
    "cv2/opencv_videoio_ffmpeg*.dll",     # 29 MB — 비디오 코덱, 이미지 IO 에 불필요
    "PySide6/opengl32sw.dll",             # 20 MB — Mesa 소프트웨어 GL 폴백, 위젯(raster) 에 불필요
    "PySide6/Qt6Quick*.dll", "PySide6/Qt6Qml*.dll",          # QML/Quick — 가상 키보드 플러그인이 끌고 옴
    "PySide6/Qt6VirtualKeyboard.dll", "PySide6/plugins/platforminputcontexts/qtvirtualkeyboardplugin.dll",
    "PySide6/Qt6Pdf.dll", "PySide6/plugins/imageformats/qpdf.dll",   # PDF 이미지 포맷
    "PySide6/Qt6Network.dll", "PySide6/plugins/networkinformation/*", "PySide6/plugins/tls/*",
    "PySide6/Qt6OpenGL.dll",              # QOpenGLWidget 전용
    "PySide6/plugins/generic/qtuiotouchplugin.dll",
    "PySide6/translations/qt_help_*",     # Qt Assistant 번역
]


def trim(toc):
    keep = []
    for entry in toc:
        dest = entry[0].replace("\\", "/")
        if any(fnmatch.fnmatch(dest, pat) for pat in TRIM_PATTERNS):
            continue
        keep.append(entry)
    return keep


cli.binaries = trim(cli.binaries)
cli.datas = trim(cli.datas)
gui.binaries = trim(gui.binaries)
gui.datas = trim(gui.datas)

cli_pyz = PYZ(cli.pure)
gui_pyz = PYZ(gui.pure)

cli_exe = EXE(
    cli_pyz,
    cli.scripts,
    [],
    exclude_binaries=True,
    name="anograft",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
gui_exe = EXE(
    gui_pyz,
    gui.scripts,
    [],
    exclude_binaries=True,
    name="anograft-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    cli_exe,
    cli.binaries,
    cli.datas,
    gui_exe,
    gui.binaries,
    gui.datas,
    strip=False,
    upx=False,
    name="anograft",
)

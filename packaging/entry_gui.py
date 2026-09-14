"""PyInstaller 진입점 — ``anograft-gui.exe`` (windowed, 콘솔 없음)."""

from __future__ import annotations

import sys

from anograft.gui.__main__ import main

if __name__ == "__main__":
    sys.exit(main())

"""PyInstaller 진입점 — ``anograft.exe`` (콘솔). ``anograft.cli.main`` 이 ``freeze_support()`` 를 부른다."""

from __future__ import annotations

import sys

from anograft.cli import main

if __name__ == "__main__":
    sys.exit(main())

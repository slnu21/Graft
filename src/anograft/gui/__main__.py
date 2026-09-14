"""``python -m anograft.gui [recipe.yaml]`` — Qt가 없으면 안내 한 줄 + 종료 코드 1."""

from __future__ import annotations

import argparse
import multiprocessing
import sys

from anograft import __version__
from anograft.gui import qt_available


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()  # frozen exe + spawn 워커 (cli.main과 같은 이유)
    parser = argparse.ArgumentParser(prog="anograft-gui", description="Graft GUI (PySide6)")
    parser.add_argument("recipe", nargs="?", default=None, help="시작할 때 열 레시피 YAML")
    parser.add_argument("--version", action="version", version=f"anograft {__version__}")
    args = parser.parse_args(argv)
    ok, reason = qt_available()
    if not ok:
        print(reason, file=sys.stderr)
        return 1
    from anograft.gui.app import run_app  # Qt는 여기서만 지연 import

    return run_app(recipe=args.recipe)


if __name__ == "__main__":
    raise SystemExit(main())

"""anograft CLI (argparse). 서브커맨드는 core-foundation 작업 단위부터 채운다."""

from __future__ import annotations

import argparse
import sys

from anograft import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anograft", description="Graft — 이상 이미지 합성 도구")
    parser.add_argument("--version", action="version", version=f"anograft {__version__}")
    parser.add_subparsers(dest="command")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 1
    return 0

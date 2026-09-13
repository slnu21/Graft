from anograft import __version__
from anograft.cli import build_parser


def test_version_is_semver() -> None:
    parts = __version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


def test_parser_builds() -> None:
    assert build_parser().prog == "anograft"

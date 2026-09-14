"""pytest 옵션 — ``--update-golden``: ``tests/golden/*.png``를 현재 결과로 다시 쓴다(의도한 알고리즘 변경 때만, 데브로그에 사유)."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="tests/golden/*.png 를 현재 파이프라인 결과로 갱신한다(비교 대신 기록)",
    )


@pytest.fixture
def update_golden(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-golden"))

"""pytest 옵션 — ``--update-golden``: ``tests/golden/*.png``를 현재 결과로 다시 쓴다(의도한 알고리즘 변경 때만, 데브로그에 사유).

그리고 **테스트는 사용자의 홈을 건드리지 않는다**: ``anograft.recent`` 는 ``~/.anograft/recent.json``
에 쓰는데, 웹 API 의 ``open`` 핸들러가 그걸 부르므로(U7) 아래 autouse 픽스처가 ``ANOGRAFT_HOME`` 을
임시 폴더로 돌려 둔다. 안 하면 테스트를 돌릴 때마다 개발자의 최근 경로 목록이 오염된다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anograft import recent


@pytest.fixture(autouse=True)
def _isolated_anograft_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    home = tmp_path_factory.mktemp("anograft-home")
    monkeypatch.setenv(recent.HOME_ENV, str(home))
    return home


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

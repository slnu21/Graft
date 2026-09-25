"""최근 경로 저장소 (U7) — `anograft.recent`.

지키는 것: 최근 것이 앞 · 같은 값은 중복되지 않고 위로 · 상한 ``LIMIT`` · 모르는 종류는 무시 ·
**깨진 파일·못 쓰는 폴더에서도 예외를 올리지 않는다**(편의 기능 하나 때문에 화면이 못 뜨면 안 된다).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anograft import recent


def test_empty_when_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(recent.HOME_ENV, str(tmp_path / "nowhere"))
    assert recent.load() == {k: [] for k in recent.KINDS}
    assert recent.recent("bank") == []


def test_remember_puts_newest_first() -> None:
    recent.remember("bank", "bank/a")
    recent.remember("bank", "bank/b")
    assert recent.recent("bank") == ["bank/b", "bank/a"]


def test_remember_moves_duplicate_up_instead_of_repeating() -> None:
    for v in ("bank/a", "bank/b", "bank/c"):
        recent.remember("bank", v)
    recent.remember("bank", "bank/a")
    assert recent.recent("bank") == ["bank/a", "bank/c", "bank/b"]


def test_limit_drops_the_oldest() -> None:
    for i in range(recent.LIMIT + 5):
        recent.remember("recipe", f"recipes/r{i}.yaml")
    items = recent.recent("recipe")
    assert len(items) == recent.LIMIT
    assert items[0] == f"recipes/r{recent.LIMIT + 4}.yaml"
    assert "recipes/r0.yaml" not in items


def test_kinds_are_separate() -> None:
    recent.remember("bank", "bank/a")
    recent.remember("recipe", "recipes/x.yaml")
    assert recent.recent("bank") == ["bank/a"]
    assert recent.recent("recipe") == ["recipes/x.yaml"]


def test_unknown_kind_and_blank_value_are_ignored() -> None:
    assert recent.remember("nonsense", "bank/a") == []
    assert recent.remember("bank", "   ") == []
    assert recent.load()["bank"] == []


def test_value_is_stripped() -> None:
    recent.remember("bank", "  bank/a  ")
    assert recent.recent("bank") == ["bank/a"]


def test_broken_file_reads_as_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(recent.HOME_ENV, str(tmp_path))
    recent.store_path().write_text("{ this is not json", encoding="utf-8")
    assert recent.load() == {k: [] for k in recent.KINDS}
    # 깨진 파일 위에도 계속 쓸 수 있어야 한다(사람이 손으로 지우게 하지 않는다)
    recent.remember("bank", "bank/a")
    assert recent.recent("bank") == ["bank/a"]


def test_wrong_shapes_in_file_are_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(recent.HOME_ENV, str(tmp_path))
    recent.store_path().write_text(
        '{"bank": ["ok", 3, "", null], "recipe": "not a list"}', encoding="utf-8"
    )
    data = recent.load()
    assert data["bank"] == ["ok"]
    assert data["recipe"] == []


def test_unwritable_store_is_fail_soft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """홈이 읽기 전용인 PC·샌드박스 — 기억하지 못할 뿐, 예외가 올라오지 않는다."""
    blocker = tmp_path / "blocked"
    blocker.write_text("파일이라 폴더를 못 만든다", encoding="utf-8")
    monkeypatch.setenv(recent.HOME_ENV, str(blocker / "inside"))
    assert recent.remember("bank", "bank/a") == ["bank/a"]  # 반환은 갱신된 모양
    assert recent.load()["bank"] == []  # 그러나 남지는 않았다

"""core/appearance 의 조명 일관성 → 은행 요약 lightR → runner.lighting_warning(KNOWN-ISSUES #5 를 은행에서 미리 잡기)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anograft import runner
from anograft.bank import Bank
from anograft.cli import EXIT_OK, main
from anograft.core import recipe as R
from anograft.core.appearance import (
    LIGHT_MIN_N,
    LIGHT_REAL_MIN,
    gray_of,
    lighting_of_sources,
    mask_lighting,
)
from anograft.core.types import DefectSource


def _dent(angle: str = "down", cls: str = "pit", k: int = 0) -> DefectSource:
    """어두운 원 + 한쪽 림이 밝은 소스 — 조명 의존 결함의 최소 모형."""
    img = np.full((40, 40, 3), 120, dtype=np.uint8)
    m = np.zeros((40, 40), dtype=np.uint8)
    m[14:26, 14:26] = 255
    img[14:26, 14:26] = 60
    if angle == "down":
        img[26:29, 12:28] = 220
    elif angle == "up":
        img[11:14, 12:28] = 220
    else:  # right
        img[12:28, 26:29] = 220
    return DefectSource(f"{cls}/{k:03d}", cls, img, m, None, (), "fixture", "png")


def _flat(cls: str = "stain", k: int = 0) -> DefectSource:
    """링이 균일한 소스 — 방향 없음(None)."""
    img = np.full((30, 30, 3), 100, dtype=np.uint8)
    m = np.zeros((30, 30), dtype=np.uint8)
    m[10:20, 10:20] = 255
    img[10:20, 10:20] = 160
    return DefectSource(f"{cls}/{k:03d}", cls, img, m, None, (), "fixture", "png")


def _recipe(preset: str, classes: list[str] | None = None, **geo) -> R.Recipe:
    d: dict = {
        "version": 1,
        "name": "t",
        "seed": 1,
        "inputs": {"bank": "b", "targets": "n"},
        "output": {"root": "o", "count": 1},
        "pipeline": {"preset": preset},
    }
    if classes is not None:
        d["pipeline"]["source"] = {"method": "bank", "classes": classes}
    if geo:
        d["pipeline"]["geometry"] = {"method": "affine", **geo}
    return R.Recipe.from_dict(d)


def test_lighting_of_sources_and_min_n() -> None:
    down = [_dent("down", k=i) for i in range(3)]
    assert abs(mask_lighting(gray_of(down[0].image), down[0].mask) - 90.0) < 5
    r, n = lighting_of_sources([(s.image, s.mask) for s in down])
    assert n == 3 and r is not None and r > 0.99
    # n < LIGHT_MIN_N 이면 판단 보류(R 은 n=1 이면 항상 1 이라 오판)
    r1, n1 = lighting_of_sources([(down[0].image, down[0].mask)])
    assert (r1, n1) == (None, 1) and LIGHT_MIN_N == 3
    # 방향이 제각각이면 R 낮음, 균일 링은 세지 않는다
    mixed = [_dent("down"), _dent("up"), _dent("right"), _dent("up")]
    r, n = lighting_of_sources([(s.image, s.mask) for s in mixed])
    assert n == 4 and r < 0.5
    assert lighting_of_sources([(f.image, f.mask) for f in [_flat(k=i) for i in range(3)]]) == (
        None,
        0,
    )


def test_bank_summary_light_r() -> None:
    srcs = [_dent("down", k=i) for i in range(4)] + [_flat(k=i) for i in range(3)]
    bank = Bank.from_sources(srcs, classes=["pit", "stain"])
    rows = {r.cls: r for r in bank.summary()}
    assert rows["pit"].light_r > 0.99 and rows["pit"].light_n == 4
    assert rows["stain"].light_r is None and rows["stain"].light_n == 0


def test_lighting_warning_rules() -> None:
    bank = Bank.from_sources(
        [_dent("down", k=i) for i in range(3)] + [_flat(k=i) for i in range(3)],
        classes=["pit", "stain"],
    )
    # ±180 + flip (poisson-graft 기본) → 경고, 클래스·원인 명시
    w = runner.lighting_warning(_recipe("poisson-graft"), bank)
    assert w and "pit(R 1.00, n 3)" in w and "rotate [-180, 180] + flip" in w and "dent-graft" in w
    assert w.startswith("geometry: ")  # 스튜디오 기하 카드 라우팅
    # dent-graft(±15, flip 끔) → 없음
    assert runner.lighting_warning(_recipe("dent-graft"), bank) is None
    # 회전만 좁혀도 flip 이 켜져 있으면 경고(원인은 flip 만)
    w = runner.lighting_warning(_recipe("poisson-graft", rotate=[-30, 30], flip=True), bank)
    assert w and "flip" in w and "rotate" not in w.split("로 합성하면")[0].split(" 을 ")[1]
    assert (
        runner.lighting_warning(_recipe("poisson-graft", rotate=[-45, 45], flip=False), bank)
        is None
    )
    # 조명 의존 클래스를 뽑지 않으면 없음
    assert runner.lighting_warning(_recipe("poisson-graft", classes=["stain"]), bank) is None
    # n < 3 이면 판단 보류
    small = Bank.from_sources([_dent("down", k=i) for i in range(2)], classes=["pit"])
    assert runner.lighting_warning(_recipe("poisson-graft"), small) is None
    # 은행 없음 / 비-bank 소스
    empty = Bank.from_sources([], name="(없음)")
    assert runner.lighting_warning(_recipe("poisson-graft"), empty) is None
    assert runner.lighting_warning(_recipe("self-cut"), bank) is None
    assert LIGHT_REAL_MIN == 0.5


def test_bank_ls_shows_light_r(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from anograft.bank.importers.common import BankWriter, ImportOptions, ImportRecord

    w = BankWriter(tmp_path / "b", name="b")
    w.ensure_classes(["pit", "stain"])
    for s in [_dent("down", k=i) for i in range(3)] + [_flat(k=i) for i in range(2)]:
        rec = ImportRecord(
            image=s.image,
            gray=False,
            mask=s.mask,
            cls=s.cls,
            origin="t",
            id_hint=s.id.split("/")[1],
        )
        w.add(rec, ImportOptions(margin=8))
    w.finish({"importer": "test"})
    assert main(["bank", "ls", str(tmp_path / "b")]) == EXIT_OK
    cap = capsys.readouterr()
    assert "lightR" in cap.out and "1.00" in cap.out and "–" in cap.out
    assert "조명 의존 클래스" in cap.err and "pit 1.00" in cap.err
    assert main(["bank", "ls", str(tmp_path / "b"), "--json"]) == EXIT_OK
    d = json.loads(capsys.readouterr().out)
    by = {c["class"]: c for c in d["classes"]}
    assert by["pit"]["light_r"] == 1.0 and by["pit"]["light_n"] == 3
    assert by["stain"]["light_r"] is None and by["stain"]["light_n"] == 0


def test_bank_summary_is_cached_and_copied() -> None:
    """요약(조명 R 포함)은 첫 호출 뒤 캐시 — 같은 내용, 호출자가 목록을 바꿔도 캐시는 안 바뀐다."""
    bank = Bank.from_sources([_dent("down", k=i) for i in range(3)], classes=["pit"])
    a = bank.summary()
    b = bank.summary()
    assert a == b and a is not b and bank._summary is not None
    a.clear()
    assert len(bank.summary()) == 1


def test_is_directional_requires_significance() -> None:
    """R 임계만으론 소표본에서 오판(무작위 n=5 의 R ≈ 0.45) — n·R² ≥ 2.9 도 요구. 요약 `directional`·JSON 에 반영."""
    from anograft.core.appearance import LIGHT_RAYLEIGH_Z, is_directional

    assert LIGHT_RAYLEIGH_Z == 2.9
    assert is_directional(1.0, 3) and not is_directional(0.9, 3)  # n=3 은 거의 완벽해야
    assert not is_directional(0.6, 5) and is_directional(0.8, 5)  # n=5: 0.76 이상
    assert is_directional(0.5, 12) and not is_directional(0.5, 11)  # n=12 부터 0.5 로 충분
    assert (
        not is_directional(None, 10)
        and not is_directional(0.99, 2)
        and not is_directional(0.49, 100)
    )
    # 요약: 뚜렷한 pit 3 → directional, 방향 제각각 4 → R 낮음 → False
    bank = Bank.from_sources(
        [_dent("down", k=i) for i in range(3)]
        + [_dent(a, cls="mix", k=i) for i, a in enumerate(("down", "up", "right", "up"))],
        classes=["pit", "mix"],
    )
    rows = {r.cls: r for r in bank.summary()}
    assert rows["pit"].directional and not rows["mix"].directional and rows["mix"].light_n == 4
    # 무작위 각도 소표본: R 이 0.5 를 넘어도 유의하지 않으면 False
    rng = np.random.default_rng(3)
    for n in (3, 4, 5, 6):
        false_pos = 0
        for _ in range(200):
            angles = rng.uniform(-180, 180, n)
            th = np.radians(angles)
            r = float(np.hypot(np.cos(th).mean(), np.sin(th).mean()))
            false_pos += is_directional(r, n)
        assert false_pos <= 20, (n, false_pos)  # ≤ 10 % (p ≈ 0.05 근사)

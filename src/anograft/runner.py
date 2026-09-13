"""실행기 (설계 §7) — ``Recipe → Bank → deps → targets → Pipeline → writer`` 를 한 줄로 잇는다. CLI ``run``·``preview``와
GUI가 같은 함수를 부른다. ``core``가 파일을 모르는 대신 여기서 ``io``를 주입한다.

::

    prep = prepare(recipe)                       # 은행 로드 → validate_against → 대상 목록 → Pipeline.from_recipe(deps) → 해시
    result = run_index(prep, i)                  # rng = image_rng(seed, i) → 대상 = targets[rng.integers(n)] → 파이프라인
    summary = run(prep, progress=...)            # writer.begin → (정상 이미지) → 0..count-1 → writer.finish

- 이미지 단위 예외는 ``status: skipped, reason`` 으로 manifest에 남기고 계속 간다. 전체 실패는 ``prepare`` 단계에서만
  (``PrepareError``).
- 레시피의 상대경로(``inputs.bank``·``inputs.targets``·``output.root``)는 **현재 작업 디렉터리** 기준.
- ``workers > 0``(프로세스 풀)은 bank-writer-parallel(#758)에서 — 지금은 인프로세스로 돌리고 경고 한 줄.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from anograft import __version__
from anograft.bank import Bank
from anograft.bank.bank import BankError
from anograft.core import registry
from anograft.core.pipeline import Pipeline
from anograft.core.recipe import Recipe
from anograft.core.seeds import image_rng, pipeline_hash
from anograft.core.types import GraftResult
from anograft.io import imgio
from anograft.io.targets import TargetsError, list_targets, load_target
from anograft.io.writers import WriterSummary, make_writer

Progress = Callable[[int, int, GraftResult], None]


class PrepareError(RuntimeError):
    """레시피·은행·대상 준비 단계의 치명적 오류 — CLI 종료 코드 1."""


@dataclass
class Prepared:
    recipe: Recipe
    bank: Bank
    targets: list[Path]
    pipeline: Pipeline
    pipeline_hash: str
    deps: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def class_probs(self) -> dict[str, float]:
        return dict(self.deps["class_probs"])


def build_deps(recipe: Recipe, bank: Bank) -> dict[str, Any]:
    """``Pipeline.from_recipe`` deps 계약(core/pipeline.py 참조)의 표준 구성."""
    return {
        "bank": bank,
        "class_ids": bank.class_ids,
        "class_probs": recipe.class_probabilities(bank),
        "read_mask": imgio.read_mask,
    }


def prepare(recipe: Recipe) -> Prepared:
    try:
        bank = Bank.load(recipe.inputs.bank)
    except BankError as e:
        raise PrepareError(str(e)) from e
    warnings = list(bank.warnings)
    try:
        warnings += recipe.validate_against(bank)
    except ValueError as e:
        raise PrepareError(f"레시피가 은행과 맞지 않습니다: {e}") from e
    try:
        targets = list_targets(recipe.inputs.targets)
    except TargetsError as e:
        raise PrepareError(str(e)) from e
    deps = build_deps(recipe, bank)
    try:
        pipeline = Pipeline.from_recipe(recipe, deps)
    except (registry.StageNotImplementedError, registry.StageUnavailableError) as e:
        raise PrepareError(f"실행할 수 없는 스테이지: {e}") from e
    ph = pipeline_hash(recipe.to_yaml(), __version__, bank.fingerprint())
    return Prepared(recipe, bank, targets, pipeline, ph, deps, warnings)


# ---------------------------------------------------------------------------
# 이미지 한 장
# ---------------------------------------------------------------------------


def skipped_result(index: int, recipe: Recipe, target: Path | None, reason: str) -> GraftResult:
    """파이프라인에 들어가기 전에 실패한 인덱스(대상 읽기 실패·예외)의 자리표. 이미지·마스크는 비어 있다."""
    sidecar: dict[str, Any] = {
        "index": index,
        "seed": recipe.seed,
        "recipe": recipe.name,
        "target": {"file": target.as_posix() if target else ""},
        "defects": [],
        "warnings": [reason],
    }
    empty = np.zeros((1, 1), dtype=np.uint8)
    return GraftResult(
        index=index,
        status="skipped",
        image=empty,
        gt_mask=empty,
        instances=(),
        sidecar=sidecar,
        warnings=(reason,),
        reason=reason,
    )


def pick_target(prep: Prepared, index: int) -> tuple[np.random.Generator, Path]:
    """``rng = image_rng(seed, i)`` → 대상 = ``targets[rng.integers(n)]``. 같은 rng를 파이프라인이 이어 쓴다(설계 §7)."""
    rng = image_rng(prep.recipe.seed, index)
    path = prep.targets[int(rng.integers(len(prep.targets)))]
    return rng, path


def run_index(prep: Prepared, index: int) -> GraftResult:
    rng, path = pick_target(prep, index)
    try:
        target = load_target(path, prep.recipe.inputs.um_per_px)
    except imgio.ImageReadError as e:
        return skipped_result(index, prep.recipe, path, f"대상 읽기 실패: {e}")
    try:
        return prep.pipeline.run_one(target, index, rng=rng)
    except Exception as e:  # 이미지 단위 격리 (설계 §7): 한 장의 예외가 배치를 멈추지 않는다
        return skipped_result(index, prep.recipe, path, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 배치
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    writer: WriterSummary
    count: int
    warnings: list[str]

    @property
    def all_skipped(self) -> bool:
        return self.count > 0 and self.writer.n_ok == 0


def run(
    prep: Prepared,
    *,
    workers: int = 0,
    progress: Progress | None = None,
    warn: Callable[[str], None] | None = None,
) -> RunSummary:
    recipe = prep.recipe
    warnings = list(prep.warnings)
    if workers > 0:
        warnings.append(
            f"--workers {workers}: 멀티프로세싱은 다음 단위(#758) — 인프로세스로 실행합니다"
        )
    writer, wwarn = make_writer(recipe.output.writer)
    if wwarn:
        warnings.append(wwarn)
    if warn is not None:
        for w in warnings:
            warn(w)
    root = Path(recipe.output.root)
    writer.begin(
        root,
        recipe,
        prep.pipeline_hash,
        prep.bank.classes,
        bank_fingerprint=prep.bank.fingerprint(),
    )
    if recipe.output.include_normals:
        for p in prep.targets:
            writer.write_normal(p)
    count = recipe.output.count
    for i in range(count):
        result = run_index(prep, i)
        writer.write_synthetic(result)
        if progress is not None:
            progress(i + 1, count, result)
    summary = writer.finish()
    warnings += summary.warnings
    return RunSummary(summary, count, warnings)


def dry_run_table(prep: Prepared) -> list[tuple[str, str]]:
    """``--dry-run`` 출력 행 ``(항목, 값)`` — 파일을 쓰지 않고 배분·경고까지만."""
    r = prep.recipe
    probs = prep.class_probs
    counts = prep.bank.counts()
    rows: list[tuple[str, str]] = [
        ("recipe", f"{r.name} (preset {r.pipeline.preset}, seed {r.seed})"),
        ("bank", f"{prep.bank.name} — {len(prep.bank)} 소스 / {len(prep.bank.classes)} 클래스"),
        ("targets", f"{len(prep.targets)} 장 ({Path(r.inputs.targets).as_posix()})"),
        (
            "output",
            f"{Path(r.output.root).as_posix()} · count {r.output.count} · writer {r.output.writer.format}",
        ),
        ("defects/image", f"{list(r.output.defects_per_image)}"),
        ("pipeline_hash", prep.pipeline_hash),
    ]
    for c, p in probs.items():
        expected = p * r.output.count * sum(r.output.defects_per_image) / 2.0
        rows.append(
            (f"class {c}", f"p={p:.3f} · 소스 {counts.get(c, 0)} · 기대 결함 수 ≈ {expected:.0f}")
        )
    return rows


def summary_lines(s: RunSummary) -> list[str]:
    w = s.writer
    lines = [
        f"완료: ok {w.n_ok} · skipped {w.n_skipped} · 정상 {w.n_normals} · 폴백 {w.n_fallback} → {w.root.as_posix()}",
    ]
    if w.per_class:
        lines.append(
            "클래스별 인스턴스: " + ", ".join(f"{c} {n}" for c, n in sorted(w.per_class.items()))
        )
    for name, rel in w.files.items():
        lines.append(f"  {name}: {rel}")
    return lines


def sidecar_defects(result: GraftResult) -> list[Mapping[str, Any]]:
    return [d for d in result.sidecar.get("defects", []) if "gt" in d]

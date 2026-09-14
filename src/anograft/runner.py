"""실행기 (설계 §7) — ``Recipe → Bank → deps → targets → Pipeline → writer`` 를 한 줄로 잇는다. CLI ``run``·``preview``와
GUI가 같은 함수를 부른다. ``core``가 파일을 모르는 대신 여기서 ``io``를 주입한다.

::

    prep = prepare(recipe)                       # 은행 로드 → validate_against → 대상 목록 → Pipeline.from_recipe(deps) → 해시
    result = run_index(prep, i)                  # rng = image_rng(seed, i) → 대상 = targets[rng.integers(n)] → 파이프라인
    summary = run(prep, progress=...)            # writer.begin → (정상 이미지) → 0..count-1 → writer.finish

- 이미지 단위 예외는 ``status: skipped, reason`` 으로 manifest에 남기고 계속 간다. 전체 실패는 ``prepare`` 단계에서만
  (``PrepareError``).
- 레시피의 상대경로(``inputs.bank``·``inputs.targets``·``output.root``)는 **현재 작업 디렉터리** 기준.
- ``workers > 0``: ``multiprocessing``(spawn) 풀. 워커는 ``initializer``에서 레시피 YAML로 ``prepare``를 **한 번** 돌려
  은행·대상·파이프라인을 자기 메모리에 두고(피클 왕복 없음), ``run_index(i)``만 받는다. 결과(``GraftResult``)는 순서대로
  메인에 오고 **메인만 파일을 쓴다**. 인덱스 ``i``의 결과는 워커 배정과 무관하게 ``image_rng(seed, i)``로 정해지므로
  ``--workers 0``과 바이트 동일(테스트 고정). 워커 수가 count보다 크면 count로 줄인다.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from anograft import __version__
from anograft.bank import Bank
from anograft.bank.bank import BankError
from anograft.core import recipe as R
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
    bank: Bank  # 비-bank 소스(self-cut·perlin)면 빈 은행 ``Bank.from_sources([], name="(없음)")``
    targets: list[Path]
    pipeline: Pipeline
    pipeline_hash: str
    deps: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def class_probs(self) -> dict[str, float]:
        return dict(self.deps["class_probs"])


def load_textures(folder: Path | None, warnings: list[str]) -> list[np.ndarray]:
    """``perlin-texture`` 의 ``texture_dir`` 이미지들 — 읽기만, 실패는 경고(fail-soft). 순서는 이름 정렬."""
    if folder is None:
        return []
    try:
        paths = imgio.list_images(folder)
    except (OSError, ValueError) as e:
        warnings.append(f"texture_dir 을 읽을 수 없습니다: {e}")
        return []
    out: list[np.ndarray] = []
    for p in paths:
        try:
            img, _gray = imgio.read_image(p)
        except imgio.ImageReadError as e:
            warnings.append(f"텍스처 읽기 실패: {e}")
            continue
        out.append(img)
    if not out:
        warnings.append(f"texture_dir 에 이미지가 없습니다: {folder}")
    return out


def prepared_classes(recipe: Recipe, bank: Bank) -> list[str]:
    """writer·class_ids 가 쓰는 클래스 순서 — bank 소스면 은행 전체, 비-bank 면 ``[cls]``."""
    return recipe.effective_classes(bank) if recipe.bankless else list(bank.classes)


def build_deps(recipe: Recipe, bank: Bank, warnings: list[str] | None = None) -> dict[str, Any]:
    """``Pipeline.from_recipe`` deps 계약(core/pipeline.py 참조)의 표준 구성. ``warnings``가 있으면 텍스처 로드 경고를 붙인다."""
    deps: dict[str, Any] = {
        "bank": bank,
        "class_ids": (
            {c: i for i, c in enumerate(prepared_classes(recipe, bank))}
            if recipe.bankless
            else bank.class_ids
        ),
        "class_probs": recipe.class_probabilities(bank),
        "read_mask": imgio.read_mask,
    }
    src = recipe.pipeline.source
    if src.method == "perlin-texture" and src.texture == "dir":
        deps["textures"] = load_textures(src.texture_dir, warnings if warnings is not None else [])
    return deps


def prepare(recipe: Recipe) -> Prepared:
    try:
        bank = (
            Bank.from_sources([], name="(없음)")
            if recipe.inputs.bank is None
            else Bank.load(recipe.inputs.bank)
        )
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
    deps = build_deps(recipe, bank, warnings)
    try:
        pipeline = Pipeline.from_recipe(recipe, deps)
    except (registry.StageNotImplementedError, registry.StageUnavailableError) as e:
        raise PrepareError(f"실행할 수 없는 스테이지: {e}") from e
    ph = pipeline_hash(recipe.hash_yaml(), __version__, bank.fingerprint())
    return Prepared(recipe, bank, targets, pipeline, ph, deps, warnings)


def reprepare(prep: Prepared, recipe: Recipe) -> Prepared:
    """은행·대상은 그대로 두고 레시피만 바뀐 경우(GUI 슬라이더·프리셋 변경) — deps·파이프라인·해시만 다시 만든다.
    ``inputs.bank``/``inputs.targets``가 바뀌었으면 ``prepare``를 다시 불러야 한다(여기서는 검사만)."""
    if (recipe.inputs.bank, recipe.inputs.targets) != (
        prep.recipe.inputs.bank,
        prep.recipe.inputs.targets,
    ):
        raise ValueError("은행 또는 대상 경로가 바뀌었습니다 — prepare()를 다시 부르세요")
    warnings = list(prep.bank.warnings)
    try:
        warnings += recipe.validate_against(prep.bank)
    except ValueError as e:
        raise PrepareError(f"레시피가 은행과 맞지 않습니다: {e}") from e
    deps = build_deps(recipe, prep.bank, warnings)
    try:
        pipeline = Pipeline.from_recipe(recipe, deps)
    except (registry.StageNotImplementedError, registry.StageUnavailableError) as e:
        raise PrepareError(f"실행할 수 없는 스테이지: {e}") from e
    ph = pipeline_hash(recipe.hash_yaml(), __version__, prep.bank.fingerprint())
    return Prepared(recipe, prep.bank, list(prep.targets), pipeline, ph, deps, warnings)


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
    cancelled: bool = False  # should_stop 으로 중간에 멈춤 — 그때까지의 결과·manifest 는 기록됨
    done: int = 0  # 실제로 처리한 인덱스 수 (취소면 count 보다 작다)

    @property
    def all_skipped(self) -> bool:
        return self.count > 0 and self.writer.n_ok == 0


# --- 워커 프로세스 (spawn: 모듈이 다시 import되므로 전부 모듈 수준 함수) ---

_WORKER_PREP: Prepared | None = None


def _worker_init(recipe_yaml: str) -> None:
    """풀 initializer — 레시피 YAML → ``prepare``(은행·대상·파이프라인) 1회. 실패는 각 인덱스에서 예외로 드러난다."""
    global _WORKER_PREP
    _WORKER_PREP = prepare(Recipe.from_yaml(recipe_yaml))


def _worker_run(index: int) -> GraftResult:
    assert _WORKER_PREP is not None, "워커가 초기화되지 않았습니다"
    return run_index(_WORKER_PREP, index)


def iter_results(prep: Prepared, indices: Iterable[int], workers: int) -> Iterator[GraftResult]:
    """``indices`` 순서대로 결과를 낸다. ``workers == 0``이면 인프로세스, 아니면 spawn 풀(``imap``, chunksize 4)."""
    idx = list(indices)
    if workers <= 0 or not idx:
        for i in idx:
            yield run_index(prep, i)
        return
    n = min(workers, len(idx))
    ctx = mp.get_context("spawn")
    with ctx.Pool(n, initializer=_worker_init, initargs=(prep.recipe.to_yaml(),)) as pool:
        yield from pool.imap(_worker_run, idx, chunksize=4)


def run(
    prep: Prepared,
    *,
    workers: int = 0,
    progress: Progress | None = None,
    warn: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> RunSummary:
    """레시피 전체를 돌려 writer 로 기록. ``should_stop()`` 이 True 를 돌려주면 다음 결과에서 멈춘다(GUI 취소) — 이미 쓴
    파일과 manifest 는 그대로 남고 ``RunSummary.cancelled`` 가 True. spawn 풀은 제너레이터가 닫히며 종료된다."""
    recipe = prep.recipe
    warnings = list(prep.warnings)
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
        prepared_classes(prep.recipe, prep.bank),
        bank_fingerprint=prep.bank.fingerprint(),
    )
    if recipe.output.include_normals:
        for p in prep.targets:
            writer.write_normal(p)
    count = recipe.output.count
    done = 0
    cancelled = False
    results = iter_results(prep, range(count), workers)
    try:
        for i, result in enumerate(results):
            writer.write_synthetic(result)
            done = i + 1
            if progress is not None:
                progress(done, count, result)
            if should_stop is not None and done < count and should_stop():
                cancelled = True
                break
    finally:
        results.close()  # 풀 컨텍스트 종료(terminate) — 취소·예외 모두
    summary = writer.finish()
    warnings += summary.warnings
    if cancelled:
        warnings.append(f"취소됨 — {done}/{count} 까지 기록")
    return RunSummary(summary, count, warnings, cancelled=cancelled, done=done)


# ---------------------------------------------------------------------------
# 같은 시드로 method 비교 (preview --compare-methods)
# ---------------------------------------------------------------------------


def compare_methods(
    prep: Prepared, stage: str, index: int
) -> list[tuple[str, GraftResult | None, str | None]]:
    """스테이지의 스키마 method 전부를 같은 대상·같은 rng로 돌린다. 반환 ``[(method, result | None, 사유)]`` —
    미구현·불가·레시피 오류는 result None + 사유(격자에 그대로 찍힌다). ``mask_dir`` 같이 필수 키가 있는 method는 건너뛴다."""
    out: list[tuple[str, GraftResult | None, str | None]] = []
    _rng0, path = pick_target(prep, index)
    try:
        target = load_target(path, prep.recipe.inputs.um_per_px)
    except imgio.ImageReadError as e:
        raise PrepareError(f"대상 읽기 실패: {e}") from e
    for info in registry.list_methods(stage):
        if not info.usable:
            out.append((info.method, None, info.reason or "미구현"))
            continue
        try:
            rec = prep.recipe.with_method(stage, info.method)
            p = reprepare(prep, rec)
        except (
            ValidationError
        ) as e:  # 예: 은행 없는 레시피에서 bank 소스 — 어느 키가 왜 틀렸는지 한 줄
            msg = R.format_validation_error(e).splitlines()[-1].strip()
            out.append((info.method, None, msg[:80]))
            continue
        except (PrepareError, ValueError) as e:
            out.append((info.method, None, str(e).splitlines()[0][:80]))
            continue
        rng, _ = pick_target(p, index)  # 같은 seed·index → 같은 스트림
        result = p.pipeline.run_one(target, index, rng=rng)
        out.append((info.method, result, None))
    return out


def dry_run_table(prep: Prepared) -> list[tuple[str, str]]:
    """``--dry-run`` 출력 행 ``(항목, 값)`` — 파일을 쓰지 않고 배분·경고까지만."""
    r = prep.recipe
    probs = prep.class_probs
    counts = prep.bank.counts()
    rows: list[tuple[str, str]] = [
        ("recipe", f"{r.name} (preset {r.pipeline.preset}, seed {r.seed})"),
        (
            "bank",
            f"(없음 — {r.pipeline.source.method}, 클래스 {prepared_classes(r, prep.bank)})"
            if r.bankless
            else f"{prep.bank.name} — {len(prep.bank)} 소스 / {len(prep.bank.classes)} 클래스",
        ),
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

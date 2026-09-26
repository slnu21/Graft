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

import cv2
import numpy as np
from pydantic import ValidationError

from anograft import __version__
from anograft.bank import Bank
from anograft.bank.bank import BankError
from anograft.core import recipe as R
from anograft.core import registry
from anograft.core.appearance import flip_breaks_lighting
from anograft.core.pipeline import Pipeline, RoiCache
from anograft.core.recipe import Recipe
from anograft.core.scale import physical_scale
from anograft.core.seeds import image_rng, pipeline_hash
from anograft.core.types import Context, GraftResult
from anograft.io import imgio
from anograft.io.targets import TargetsError, list_targets, load_target, targets_warning
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
    """``perlin-texture`` 의 ``texture_dir`` 이미지들 — 읽기만, 실패는 경고(fail-soft). 순서는 이름 정렬.
    폴더 대신 ``.txt`` 목록(한 줄 = 경로, 목록 파일 기준 상대 — ``dataset textures dtd`` 가 만든다)도 받는다(줄 순서)."""
    if folder is None:
        return []
    try:
        f = Path(folder)
        if f.is_file() and f.suffix.lower() == ".txt":
            paths = imgio.read_path_list(f)
        else:
            paths = imgio.list_images(f)
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
    """writer·class_ids 가 쓰는 클래스 순서 — bank 소스면 은행 전체(**미분류 제외**), 비-bank 면 ``[cls]``.

    미분류는 `bank.classes` 의 **맨 끝**이라(불변식, `core.classes.order_classes`) 빼도 다른 클래스의
    id 가 움직이지 않는다 — 출력 `data.yaml` 에 뜻 없는 이름이 나가지 않으면서 기존 모델과도 호환된다.
    """
    return recipe.effective_classes(bank) if recipe.bankless else list(bank.usable_classes)


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


def unsorted_warning(recipe: Recipe, bank: Bank) -> str | None:
    """미분류 조각이 있으면 한 줄 — **왜 내 조각이 안 쓰이는지** 알려 준다(T15).

    조용히 빼면 "보관함에 20개 넣었는데 합성에 5개만 쓰인다"가 되어 사람이 원인을 못 찾는다.
    """
    if recipe.bankless:
        return None
    n = len(bank.unsorted())
    if not n:
        return None
    return (
        f"미분류 조각 {n}개는 합성·출력에서 빠집니다 — 처음 보는 형상이라 이름이 없습니다. "
        f"`anograft bank promote <보관함> --to <클래스>` 로 이름을 주면 그때부터 쓰입니다"
    )


def scale_warning(recipe: Recipe, bank: Bank) -> str | None:
    """µm/px 축척 정합이 (일부라도) 꺼진 채 돌아가면 한 줄 경고(KNOWN-ISSUES #6). ``physical_scale`` 은 소스·대상 **양쪽**에
    피치가 있어야 동작하고 한쪽이라도 없으면 factor 1.0 으로 조용히 넘어가므로, prepare 에서 표면에 올린다.
    비-bank 소스(self-cut·perlin)는 대상 자신에서 만들므로 해당 없음."""
    if recipe.bankless or len(bank) == 0:
        return None
    no_pitch = bank.no_pitch_count()
    target_pitch = recipe.inputs.um_per_px
    if target_pitch is None and no_pitch == len(bank):
        return (
            "축척 정합 꺼짐 — 결함 조각 전부와 바탕 이미지(inputs.um_per_px) 모두 픽셀 크기(µm/px)가 없습니다. "
            "결함이 픽셀 크기 그대로 붙습니다. → 다른 카메라·배율의 결함이면 임포트 --um-per-px 와 레시피 inputs.um_per_px 를 지정하세요"
        )
    if target_pitch is None:
        return (
            f"축척 정합 꺼짐 — 바탕 이미지의 픽셀 크기(대상 inputs.um_per_px)가 없어 조각의 픽셀 크기({len(bank) - no_pitch}개)를 "
            "쓰지 못합니다. → 레시피 inputs.um_per_px 를 지정하세요"
        )
    if no_pitch:
        return (
            f"축척 정합 일부 꺼짐 — 픽셀 크기(um_per_px)가 없는 조각 {no_pitch}/{len(bank)}개는 배율 1.0 으로 붙습니다. "
            "→ bank ls 의 no_um 열에서 확인하세요"
        )
    return None


def confidence_warning(recipe: Recipe, bank: Bank) -> str | None:
    """저신뢰 추정 마스크(``confidence < LOW_CONFIDENCE``)가 있으면 한 줄(KNOWN-ISSUES #3). 추정 마스크의 절반을 넘으면 "은행을
    먼저 손보라"로 강화 — 헐거운 마스크는 결함이 아니라 소스 제품의 표면을 이식한다."""
    if recipe.bankless or len(bank) == 0:
        return None
    low = bank.low_confidence()
    if not low:
        return None
    est = sum(r.estimated for r in bank.summary())
    ratio = len(low) / est if est else 0.0
    examples = ", ".join(s.id for s in low[:3])
    if ratio > 0.5:
        return (
            f"추정 마스크 {est}개 중 신뢰도 낮음 {len(low)}개({ratio:.0%}) — 절반이 넘습니다. 이 보관함으로 합성하면 결함이 아니라 "
            f"원본 표면이 붙을 수 있습니다. → 보관함을 먼저 손보세요: 보관함 탭 '신뢰도 낮은 것 차례로 다듬기' 또는 "
            f"import-yolo --mask-from otsu (예: {examples})"
        )
    return (
        f"마스크 신뢰도 낮은 조각 {len(low)}/{est}개(추정 마스크 중) — 결함이 아닌 표면이 붙을 수 있습니다. "
        f"→ 보관함 탭에서 다듬거나 import-yolo --mask-from otsu (예: {examples})"
    )


LIGHT_ROTATE_MAX_DEG = 90.0  # rotate 범위 폭이 이보다 크면(±45 초과) 조명 방향이 뒤집힌다고 본다


def lighting_warning(recipe: Recipe, bank: Bank) -> str | None:
    """조명 의존 클래스(실제 소스의 하이라이트 방향 일관성 R ≥ LIGHT_REAL_MIN, n ≥ 3)를 rotate 폭 > 90° 또는 flip 으로
    합성하면 한 줄(KNOWN-ISSUES #5). 은행에서 미리 잡는다 — 검수 탭 '조명 방향' 분포는 사후 확인.
    ``geometry:`` 접두를 달아 스튜디오가 기하 카드에 표시한다(prepare 경고 중 스테이지 접두가 있는 것만 카드로)."""
    if recipe.bankless or len(bank) == 0:
        return None
    geo = recipe.pipeline.geometry
    selected = set(recipe.effective_classes(bank))
    bad: list[str] = []
    causes: set[str] = set()
    for r in bank.summary():
        if r.cls not in selected or not r.directional:
            continue
        eff = geo.for_class(r.cls)  # 클래스별 오버라이드가 있으면 그 범위로 판단
        lo, hi = eff.rotate
        wide = (hi - lo) > LIGHT_ROTATE_MAX_DEG
        flip_bad = flip_breaks_lighting(
            eff.flip, r.light_dir
        )  # 위/아래 조명이면 horizontal 은 안전
        if not wide and not flip_bad:
            continue
        bad.append(f"{r.cls}(R {r.light_r:.2f}, n {r.light_n})")
        if wide:
            causes.add(f"rotate [{lo:g}, {hi:g}]")
        if flip_bad:
            causes.add(f"flip {eff.flip}")
    if not bad:
        return None
    what = " · ".join(bad)
    cause = " + ".join(sorted(causes, key=lambda c: (c.startswith("flip"), c)))
    return (  # `geometry:` 접두 — 스튜디오가 크기·회전 카드에 ⚠ 로 라우팅(스테이지 경고 규약)
        f"geometry: 빛 방향이 정해진 결함 {what} 을 {cause} 로 합성하면 하이라이트가 뒤집힌 그림이 섞입니다. "
        f"→ 프리셋 dent-graft(±15°, 뒤집기 없음), 또는 그 클래스만 좁히기(크기·회전 카드 '빛 방향 클래스만 ±15°로 좁히기' = "
        f"geometry.per_class). 검수 탭 '밝은 쪽 방향' 분포로 확인"
    )


def prepare_warnings(recipe: Recipe, bank: Bank) -> list[str]:
    """은행 경고 + 레시피↔은행 대조 + 축척·저신뢰·조명 한 줄 경고 — ``prepare``·``reprepare``·``recipe check`` 공용.
    대조가 치명적이면 ``PrepareError``."""
    warnings = list(bank.warnings)
    try:
        warnings += recipe.validate_against(bank)
    except ValueError as e:
        raise PrepareError(f"레시피가 은행과 맞지 않습니다: {e}") from e
    for w in (
        unsorted_warning(recipe, bank),
        scale_warning(recipe, bank),
        confidence_warning(recipe, bank),
        lighting_warning(recipe, bank),
    ):
        if w:
            warnings.append(w)
    return warnings


DEFAULT_ROI_CACHE = (
    16  # 대상당 ROI 캐시 항목 수(LRU). 레시피가 아니라 실행 옵션 — pipeline_hash 에 안 들어간다
)


def prepare(recipe: Recipe, *, roi_cache_size: int = DEFAULT_ROI_CACHE) -> Prepared:
    try:
        bank = (
            Bank.from_sources([], name="(없음)")
            if recipe.inputs.bank is None
            else Bank.load(recipe.inputs.bank)
        )
    except BankError as e:
        raise PrepareError(str(e)) from e
    warnings = prepare_warnings(recipe, bank)
    try:
        targets = list_targets(recipe.inputs.targets)
    except TargetsError as e:
        raise PrepareError(str(e)) from e
    if tw := targets_warning(targets):
        warnings.append(tw)
    deps = build_deps(recipe, bank, warnings)
    try:
        pipeline = Pipeline.from_recipe(recipe, deps)
    except (registry.StageNotImplementedError, registry.StageUnavailableError) as e:
        raise PrepareError(f"실행할 수 없는 스테이지: {e}") from e
    # 대상당 ROI 1회(스튜디오 변형·run 대상 재추첨). 0 이면 캐시 없음(메모리가 빠듯한 4K 배치 등). 결과는 캐시 유무와 무관(ROI 는 rng 0회)
    pipeline.roi_cache = RoiCache(roi_cache_size) if roi_cache_size > 0 else None
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
    warnings = prepare_warnings(
        recipe, prep.bank
    )  # 축척·저신뢰·조명 경고는 레시피에 따라 바뀐다(카드 편집)
    if tw := targets_warning(prep.targets):  # 대상은 그대로라 경고도 유지
        warnings.append(tw)
    deps = build_deps(recipe, prep.bank, warnings)
    try:
        pipeline = Pipeline.from_recipe(recipe, deps)
    except (registry.StageNotImplementedError, registry.StageUnavailableError) as e:
        raise PrepareError(f"실행할 수 없는 스테이지: {e}") from e
    # ROI 캐시는 이어 받는다 — 키에 ROI 설정이 들어 있어 다른 스테이지 파라미터를 바꿔도(카드 편집) grabcut 을 다시 풀지 않는다
    pipeline.roi_cache = prep.pipeline.roi_cache  # None 이면(캐시 끔) 그대로 None
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


def _worker_init(recipe_yaml: str, roi_cache_size: int = DEFAULT_ROI_CACHE) -> None:
    """풀 initializer — 레시피 YAML → ``prepare``(은행·대상·파이프라인) 1회. 실패는 각 인덱스에서 예외로 드러난다."""
    global _WORKER_PREP
    _WORKER_PREP = prepare(Recipe.from_yaml(recipe_yaml), roi_cache_size=roi_cache_size)


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
    cache_size = prep.pipeline.roi_cache.max_items if prep.pipeline.roi_cache is not None else 0
    with ctx.Pool(
        n, initializer=_worker_init, initargs=(prep.recipe.to_yaml(), cache_size)
    ) as pool:
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


@dataclass(frozen=True)
class FitDiagnostic:
    """``--dry-run`` 의 배치 가능성 진단 — 대상 몇 장의 ROI 최대 폭(내접원 지름, 테두리 여유 뺀 것) vs 클래스별 패치 폭.
    KNOWN-ISSUES 부록("ROI 폭 대비 패치 크기를 사전에 알려 주는 진단"): 좁은 링에 큰 패치면 max_tries 를 다 쓰고 skipped 가 된다."""

    roi_widths: list[tuple[str, float]]  # (대상 파일명, 허용 영역 최대 폭 px)
    patch_sides: dict[str, float]  # 클래스 → 패치 긴 변 중앙값 × geometry.scale 상한 × 축척 (px)
    scale_hi: float
    shrink_floor: float  # shrink_on_fail 로 줄어드는 최소 배율(factor^rounds)
    target_short_side: float = 0.0  # 대상 짧은 변(px, 첫 대상) — 0 이면 모름
    patch_short: dict[str, float] = field(
        default_factory=dict
    )  # 클래스 → 짧은 변(minAreaRect) × scale × 축척. 비면 긴 변으로 판정
    aligned: bool = (
        False  # structure-aware align along/across — 긴 변이 결 방향으로 눕는다(링이면 접선)
    )
    physical: dict[str, float] = field(
        default_factory=dict
    )  # 클래스 → 축척 factor 중앙값(1.0 = 미적용)

    NON_LOCAL_RATIO = 0.5  # 패치 긴 변 ≥ 대상 짧은 변 × 이 비율 → "국소 결함이 아닐 수 있음"

    @property
    def min_width(self) -> float:
        return min((w for _, w in self.roi_widths), default=0.0)

    def non_local_classes(self) -> dict[str, float]:
        """패치가 대상 자체와 맞먹는 클래스 → 비율. 결함이 아니라 부품 전체 이상(MVTec ``flip`` 처럼 자세·누락)일 때 —
        ROI 를 넓혀도 답이 아니고 ``source.classes`` 로 빼는 게 맞다(리허설 2026-09-16: metal_nut ``flip`` 685/700px)."""
        if self.target_short_side <= 0:
            return {}
        return {
            c: round(side / self.target_short_side, 2)
            for c, side in self.patch_sides.items()
            if side >= self.target_short_side * self.NON_LOCAL_RATIO
        }

    def source_warning(self) -> str | None:
        big = self.non_local_classes()
        if not big:
            return None
        return (
            "source: 클래스 "
            + ", ".join(f"{c}(대상 짧은 변의 {int(r * 100)}%)" for c, r in big.items())
            + " 는 패치가 대상 자체와 맞먹습니다 — 국소 결함이 아니라 부품 전체 이상(자세·누락)일 수 있음 → source.classes 로 제외 검토"
        )

    def verdicts(self) -> dict[str, str]:
        """클래스 → '가능' · '빠듯' · '불가' (가장 좁은 대상 기준).

        v2(리허설 2026-09-16): 폭에 걸리는 건 **짧은 변**이다 — 가늘고 긴 스크래치(긴 변 214 px)가 링 폭 100 px 에 13/14 들어갔다.
        불가 = 짧은 변이 shrink 바닥까지 줄여도 폭 초과 · 빠듯 = 짧은 변이 폭의 80 % 초과, 또는 긴 변이 폭을 넘는데 정렬이 없어
        회전 운에 달림 · 그 외 가능. ``patch_short`` 가 비면(구 호출자) 긴 변으로 판정(종전 동작)."""
        w = self.min_width
        out: dict[str, str] = {}
        for c, long_s in self.patch_sides.items():
            short_s = self.patch_short.get(c, long_s)
            if short_s * self.shrink_floor > w:
                out[c] = "불가"
            elif short_s > w * 0.8 or (long_s > w and not self.aligned):
                out[c] = "빠듯"
            else:
                out[c] = "가능"
        return out

    def verdict_note(self, cls: str) -> str:
        """행 끝에 붙는 근거 한 토막 — 왜 그 판정인지."""
        w = self.min_width
        long_s = self.patch_sides.get(cls, 0.0)
        short_s = self.patch_short.get(cls, long_s)
        if short_s * self.shrink_floor > w:
            return "짧은 변이 shrink 뒤에도 폭 초과"
        if short_s > w * 0.8:
            return "짧은 변이 폭의 80% 초과"
        if long_s > w:
            return f"긴 변은 폭 초과 — 정렬 {'있음(결 방향으로 눕힘)' if self.aligned else '없음(회전에 따라)'}"
        return ""

    def warning(self) -> str | None:
        v = self.verdicts()
        bad = [c for c, s in v.items() if s == "불가"]
        tight = [c for c, s in v.items() if s == "빠듯"]
        if not bad and not tight:
            return None
        parts = []
        if bad:
            parts.append(
                f"클래스 {', '.join(bad)} 는 패치 짧은 변이 ROI 최대 폭 {self.min_width:.0f}px 보다 커서(shrink_on_fail 뒤에도) "
                "어디에도 못 들어갑니다 → skipped 예상"
            )
        if tight:
            parts.append(
                f"클래스 {', '.join(tight)} 는 빠듯합니다(짧은 변이 폭의 80% 초과, 또는 긴 변이 폭을 넘는데 정렬 없음)"
            )
        return (
            "placement: "
            + " · ".join(parts)
            + " — geometry.scale 상한을 낮추거나 placement.margin_px·roi.erode_px 를 줄이거나 ROI 를 넓히세요"
        )


def mask_dims(mask: np.ndarray) -> tuple[float, float]:
    """마스크의 (짧은 변, 긴 변) px — ``cv2.minAreaRect`` 의 회전 사각형(+1: 중심 간 거리 → 픽셀 수). 비면 (0, 0). 점 3개 미만이면 bbox."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    if len(xs) < 3:
        w, h = float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)
        return min(w, h), max(w, h)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    (_c, (w, h), _a) = cv2.minAreaRect(pts)
    w, h = float(w) + 1.0, float(h) + 1.0  # 픽셀 중심 간 거리 → 픽셀 수(bbox 와 같은 단위)
    return min(w, h), max(w, h)


def patch_dims(prep: Prepared) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """클래스별 (긴 변, 짧은 변, 축척 factor) 중앙값 — 긴/짧은 변은 ``geometry.scale`` 상한 × 소스별 µm/px 축척(양쪽 피치가 있을 때만,
    ``core.scale.physical_scale``)을 곱한 원본 해상도 px. 뽑는 클래스만."""
    r = prep.recipe
    tgt_um = r.inputs.um_per_px
    longs: dict[str, float] = {}
    shorts: dict[str, float] = {}
    phys: dict[str, float] = {}
    for c in r.effective_classes(prep.bank):
        scale_hi = float(r.pipeline.geometry.for_class(c).scale[1])  # 클래스별 오버라이드 반영
        ls, ss, fs = [], [], []
        for s in prep.bank.by_class(c):
            short_s, long_s = mask_dims(s.mask)
            if long_s <= 0:
                continue
            f = physical_scale(s.um_per_px, tgt_um).factor
            ls.append(long_s * f)
            ss.append(short_s * f)
            fs.append(f)
        if ls:
            longs[c] = round(float(np.median(ls)) * scale_hi, 1)
            shorts[c] = round(float(np.median(ss)) * scale_hi, 1)
            phys[c] = round(float(np.median(fs)), 3)
    return longs, shorts, phys


def patch_sides(prep: Prepared) -> dict[str, float]:
    """클래스별 패치 긴 변(호환) — ``patch_dims`` 의 첫 항."""
    return patch_dims(prep)[0]


def placement_aligned(recipe: Recipe) -> bool:
    """structure-aware ``align`` along/across 면 패치 긴 변이 결 방향으로 눕는다(링이면 접선) → 긴 변은 폭에 안 걸린다."""
    pl = recipe.pipeline.placement
    return getattr(pl, "method", "sampled") == "structure-aware" and getattr(
        pl, "align", "none"
    ) in (
        "along",
        "across",
    )


def fit_from_widths(
    prep: Prepared, widths: list[tuple[str, float]], *, target_short_side: float = 0.0
) -> FitDiagnostic | None:
    """이미 잰 허용 영역 폭(원본 px)으로 진단 — 스튜디오가 미리보기 ROI(축소본 ÷ 배율)로 부른다. 비-bank 면 None.
    ``target_short_side``(원본 px)를 주면 "국소 결함이 아닐 수 있음" 경고까지."""
    r = prep.recipe
    if r.bankless or len(prep.bank) == 0 or not widths:
        return None
    shrink = r.pipeline.placement.shrink_on_fail
    floor = float(shrink.factor**shrink.rounds) if shrink.rounds > 0 else 1.0
    longs, shorts, phys = patch_dims(prep)
    return FitDiagnostic(
        widths,
        longs,
        float(r.pipeline.geometry.scale[1]),
        floor,
        float(target_short_side),
        shorts,
        placement_aligned(r),
        phys,
    )


def fit_diagnostic(prep: Prepared, n_targets: int = 3) -> FitDiagnostic | None:
    """처음 ``n_targets`` 장의 ROI 를 실제로 풀어(캐시 사용, rng 0회) 허용 영역 최대 폭을 재고 클래스별 패치 폭과 견준다.
    비-bank 소스·대상 없음이면 None. 파일 읽기 실패한 대상은 건너뛴다."""
    r = prep.recipe
    if r.bankless or not prep.targets or len(prep.bank) == 0:
        return None
    from anograft.core.stages.placement import roi_max_width

    margin = int(getattr(r.pipeline.placement, "margin_px", 0))
    widths: list[tuple[str, float]] = []
    short_side = 0.0
    for path in prep.targets[: max(1, n_targets)]:
        try:
            target = load_target(path, r.inputs.um_per_px)
        except (OSError, imgio.ImageReadError):
            continue
        ctx = prep.pipeline._apply_roi(Context.initial(np.random.default_rng(0), target), target)
        width = roi_max_width(ctx.roi, margin, target.image.shape[:2])
        widths.append((path.name, round(width, 1)))
        short_side = short_side or float(min(target.image.shape[:2]))
    return fit_from_widths(prep, widths, target_short_side=short_side)


def dry_run_table(prep: Prepared, *, fit: FitDiagnostic | None = None) -> list[tuple[str, str]]:
    """``--dry-run`` 출력 행 ``(항목, 값)`` — 파일을 쓰지 않고 배분·경고까지만."""
    r = prep.recipe
    probs = prep.class_probs
    counts = prep.bank.counts()
    tags = getattr(r.pipeline.source, "tags", None)
    if tags is not None and tags.active:
        counts = {c: sum(1 for s in prep.bank.by_class(c) if tags.accepts(s.tags)) for c in counts}
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
    if tags is not None and tags.active:
        rows.append(("source.tags", tags.describe()))
    geo = r.pipeline.geometry
    rows.append(
        (
            "geometry",
            f"scale {geo.scale[0]:g}~{geo.scale[1]:g} · rotate {geo.rotate[0]:g}~{geo.rotate[1]:g}° · flip {geo.flip}",
        )
    )
    for c, o in geo.per_class.items():  # 클래스별 오버라이드 — 준 필드만
        bits = []
        if o.scale is not None:
            bits.append(f"scale {o.scale[0]:g}~{o.scale[1]:g}")
        if o.rotate is not None:
            bits.append(f"rotate {o.rotate[0]:g}~{o.rotate[1]:g}°")
        if o.flip is not None:
            bits.append(f"flip {o.flip}")
        rows.append((f"per_class {c}", " · ".join(bits) or "(변경 없음)"))
    for c, p in probs.items():
        expected = p * r.output.count * sum(r.output.defects_per_image) / 2.0
        rows.append(
            (f"class {c}", f"p={p:.3f} · 소스 {counts.get(c, 0)} · 기대 결함 수 ≈ {expected:.0f}")
        )
    if fit is not None:
        rows.append(
            (
                "roi width",
                " · ".join(f"{n} {w:.0f}px" for n, w in fit.roi_widths)
                + f" (허용 영역 최대 폭 = 내접원 지름, 테두리 여유 제외 · {len(fit.roi_widths)}장)",
            )
        )
        v = fit.verdicts()
        for c, side in fit.patch_sides.items():
            short_s = fit.patch_short.get(c, side)
            f = fit.physical.get(c, 1.0)
            scale_txt = f"× scale {fit.scale_hi:g}" + (
                f" × 축척 {f:.2f}" if abs(f - 1.0) > 1e-6 else ""
            )
            note = fit.verdict_note(c)
            rows.append(
                (
                    f"fit {c}",
                    f"패치 짧은 변 {short_s:.0f} · 긴 변 {side:.0f}px({scale_txt}) vs 폭 {fit.min_width:.0f}px → {v.get(c, '?')}"
                    + (f" ({note})" if note else ""),
                )
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

"""배치 세션 — **Qt 없음**. 레시피(스튜디오에서 넘어오거나 파일에서) + 실행 오버라이드(출력 폴더·장수·시드·워커·writer 형식) →
``build_recipe()`` → ``run_batch()``(= ``runner.prepare`` + ``runner.run``, 진행/경고/취소 콜백).

- 오버라이드는 CLI ``run --out --count --seed`` 와 같은 자리(``Recipe.to_dict`` 위에 덮어 ``from_dict`` — 재검증). writer 형식은
  ``output.writer = {format}`` 으로 갈아 끼운다(형식별 기본값; ``set_method_in_dict`` 와 같은 규칙 — 다른 형식의 키가 남지 않게).
- ``run_batch`` 는 스레드에서 돌리라고 만든 순수 함수: 콜백만 알고 위젯을 모른다. ``should_stop`` 이 True 면 ``runner.run`` 이
  다음 결과에서 멈추고 그때까지의 파일·manifest 를 남긴다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anograft import runner
from anograft.core import recipe as R
from anograft.core.types import GraftResult

WRITER_FORMATS: tuple[str, ...] = ("yolo", "pairs", "mvtec", "coco")
Progress = Callable[[int, int, GraftResult], None]


class BatchError(ValueError):
    pass


@dataclass
class BatchSession:
    recipe: R.Recipe | None = None
    recipe_path: Path | None = None
    out: str = ""
    count: int = 0
    seed: int = 0
    workers: int = 0
    writer: str = "yolo"
    mvtec_category: str = "graft"
    last_summary: runner.RunSummary | None = None
    log: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ 레시피

    def set_recipe(self, recipe: R.Recipe, path: Path | None = None) -> None:
        """스튜디오에서 넘어온 레시피(또는 파일). 오버라이드 칸을 레시피 값으로 채운다."""
        self.recipe = recipe
        self.recipe_path = path
        self.out = recipe.output.root.as_posix()
        self.count = recipe.output.count
        self.seed = recipe.seed
        self.writer = recipe.output.writer.format
        if self.writer == "mvtec":
            self.mvtec_category = getattr(recipe.output.writer, "category", "graft")

    def load(self, path: str | Path) -> R.Recipe:
        try:
            rec = R.Recipe.load(path)
        except Exception as e:  # 파일·검증 오류를 한 종류로 — 탭이 메시지로 보여 준다
            raise BatchError(f"레시피 로드 실패: {e}") from e
        self.set_recipe(rec, Path(path))
        return rec

    def build_recipe(self) -> R.Recipe:
        """오버라이드를 적용한 실행용 레시피(재검증). writer 형식이 바뀌면 그 형식 기본값으로."""
        if self.recipe is None:
            raise BatchError(
                "레시피가 없습니다 — 스튜디오에서 '배치로 보내기' 또는 레시피 파일 열기"
            )
        if not self.out.strip():
            raise BatchError("출력 폴더를 지정하세요")
        if self.count < 1:
            raise BatchError("생성 장수는 1 이상")
        d: dict[str, Any] = self.recipe.to_dict()
        d["output"]["root"] = self.out.strip()
        d["output"]["count"] = int(self.count)
        d["seed"] = int(self.seed)
        if self.writer not in WRITER_FORMATS:
            raise BatchError(f"알 수 없는 writer 형식 {self.writer!r}")
        if self.writer != self.recipe.output.writer.format:
            block: dict[str, Any] = {"format": self.writer}
            if self.writer == "mvtec":
                block["category"] = self.mvtec_category or "graft"
            d["output"]["writer"] = block
        elif self.writer == "mvtec":
            d["output"]["writer"]["category"] = self.mvtec_category or "graft"
        try:
            return R.Recipe.from_dict(d)
        except Exception as e:
            raise BatchError(f"레시피 검증 실패: {e}") from e

    def run_command(self) -> str:
        """같은 실행을 CLI 로 — 로그 맨 위에 찍어 재현 방법을 남긴다."""
        p = self.recipe_path.as_posix() if self.recipe_path else "<recipe.yaml>"
        return (
            f"anograft run {p} --out {self.out} --count {self.count} --seed {self.seed}"
            f" --workers {self.workers}"
        )


def run_batch(
    recipe: R.Recipe,
    *,
    workers: int = 0,
    progress: Progress | None = None,
    warn: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> runner.RunSummary:
    """``prepare`` + 배치 가능성 진단(경고로) + ``run``. ``PrepareError`` 는 ``BatchError`` 로."""
    try:
        prep = runner.prepare(recipe)
    except runner.PrepareError as e:
        raise BatchError(str(e)) from e
    if warn is not None:
        for w in (
            prep.warnings
        ):  # prepare 경고(축척·저신뢰·조명 …)도 로그에 — 종전엔 run 중 경고만 보였다
            warn(w)
        fit = runner.fit_diagnostic(
            prep
        )  # 워커 스레드라 ROI 몇 장은 괜찮다(캐시로 run 이 이어 쓴다)
        if fit is not None:
            if sw := fit.source_warning():
                warn(sw)
            fw = fit.warning()
            if fw:
                warn(fw)
            else:
                warn(
                    "placement: 배치 가능성 OK — ROI 최대 폭 "
                    + " · ".join(f"{n} {w:.0f}px" for n, w in fit.roi_widths)
                    + " vs 패치 "
                    + ", ".join(f"{c} {s:.0f}px" for c, s in fit.patch_sides.items())
                )
    return runner.run(prep, workers=workers, progress=progress, warn=warn, should_stop=should_stop)


def summary_text(summary: runner.RunSummary) -> str:
    lines = list(runner.summary_lines(summary))
    if summary.cancelled:
        lines.insert(0, f"취소됨 — {summary.done}/{summary.count} 까지 기록")
    return "\n".join(lines)

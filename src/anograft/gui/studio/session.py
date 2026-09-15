"""``StudioSession`` — GUI 상태의 단일 원천. **Qt 없음** (순수 파이썬, 테스트는 Qt 없이).

- 상태 = ``Recipe``(pydantic, 검증된 것만 보관) + ``Prepared``(은행·대상·파이프라인) 캐시 + 선택(대상 인덱스·변형 인덱스).
- 위젯 값 변경은 전부 ``update_*``/``set_*``를 거친다 → 딕셔너리로 고쳐 ``Recipe.from_dict``로 **다시 검증** → 실패하면
  ``SessionError``(메시지 = ``format_validation_error``)와 함께 이전 레시피 유지. GUI와 CLI가 같은 규칙으로 막는 지점.
- 은행/대상 경로가 바뀌면 ``needs_prepare()``가 True — IO는 호출자(워커)가 ``runner.prepare``로 하고 ``accept_prepared``로 넘긴다.
  파이프라인 설정만 바뀌면 ``runner.reprepare``(은행 재사용, 동기·가벼움)로 즉시 갱신한다.
- ``generation``은 "미리보기가 무효화된 횟수" — 워커 결과가 도착했을 때 이 값이 다르면 버린다.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from anograft import runner
from anograft.core import recipe as R
from anograft.core import registry
from anograft.gui.studio.params import required_placeholders

DEFAULT_PRESET = "poisson-graft"


class SessionError(ValueError):
    """레시피 검증 실패·준비 실패 — 사용자에게 그대로 보여 줄 한국어 메시지."""


def default_recipe(
    *, bank: str = "bank/sample", targets: str = "samples/metal/normals.txt"
) -> R.Recipe:
    data = R.init_recipe_dict(
        DEFAULT_PRESET, name="studio", bank=bank, targets=targets, out="out/studio", count=24
    )
    data["output"]["writer"] = {"format": "pairs"}
    return R.Recipe.from_dict(data)


def _set_nested(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur = data
    for k in path[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[path[-1]] = value


class StudioSession:
    def __init__(self, recipe: R.Recipe | None = None) -> None:
        self.recipe: R.Recipe = recipe or default_recipe()
        self.prepared: runner.Prepared | None = None
        self.recipe_path: Path | None = None
        self.target_index: int = 0  # 레일에서 고른 대상 (targets 목록 인덱스)
        self.variant_index: int = 0  # 캔버스에 보일 변형 k (= image_rng(seed, k))
        self.n_variants: int = 6
        self.long_side: int = 1024  # 미리보기 축소 긴 변 (0 = 원본)
        self.generation: int = 0
        self.warnings: list[str] = []
        self.path_notes: list[str] = []  # 레시피 파일 기준으로 다시 해석한 경로(load 때)

    # ------------------------------------------------------------------ 레시피 변경 (전부 재검증)

    def _apply(self, mutate: Any) -> R.Recipe:
        data = self.recipe.to_dict()
        mutate(data)
        try:
            new = R.Recipe.from_dict(data)
        except ValidationError as e:
            raise SessionError(R.format_validation_error(e)) from e
        except (KeyError, ValueError) as e:
            raise SessionError(f"레시피 오류: {e}") from e
        self._commit(new)
        return new

    def _commit(self, new: R.Recipe) -> None:
        changed = new.to_yaml() != self.recipe.to_yaml()
        old = self.recipe
        self.recipe = new
        if not changed:
            return
        self.generation += 1
        if self.prepared is not None and not self.needs_prepare():
            try:
                self.prepared = runner.reprepare(self.prepared, new)
                self.warnings = list(self.prepared.warnings)
            except runner.PrepareError as e:
                self.recipe = old
                self.generation += 1
                raise SessionError(str(e)) from e

    def set_field(self, path: tuple[str, ...], value: Any) -> R.Recipe:
        """임의 키 경로. 예: ``("seed",)`` · ``("output", "defects_per_image")`` · ``("pipeline", "blend", "feather_px")``."""
        return self._apply(lambda d: _set_nested(d, path, value))

    def set_seed(self, seed: int) -> R.Recipe:
        return self.set_field(("seed",), int(seed))

    def set_paths(
        self, bank: str | Path | None = None, targets: str | Path | None = None
    ) -> R.Recipe:
        """``bank=""``(빈 문자열)은 "은행 없음"(``inputs.bank: null`` — self-cut·perlin 프리셋). None 은 "그대로"."""

        def mutate(d: dict[str, Any]) -> None:
            if bank is not None:
                d["inputs"]["bank"] = Path(bank).as_posix() if str(bank) else None
            if targets is not None:
                d["inputs"]["targets"] = Path(targets).as_posix()

        return self._apply(mutate)

    def set_preset(self, name: str) -> R.Recipe:
        """프리셋을 바꾸면 ``pipeline``은 그 프리셋의 기본값으로 **통째로** 교체한다(다른 프리셋의 키가 섞이지 않게)."""
        return self._apply(lambda d: d.__setitem__("pipeline", {"preset": name}))

    def set_method(self, stage: str, method: str) -> R.Recipe:
        """스테이지 method 교체 — 그 스테이지 블록은 새 method의 기본값만 남긴다(``Recipe.with_method``와 같은 규칙).
        기본값 없는 필수 필드(``mask_dir.path``)는 자리표시 값으로 채워 카드 폼에서 고치게 한다(GUI 전용 관용)."""

        def mutate(d: dict[str, Any]) -> None:
            R.set_method_in_dict(d, stage, method)
            try:
                fill = required_placeholders(registry.config_class(stage, method))
            except KeyError:
                return
            if fill:
                block = (
                    d["pipeline"]["placement"]["roi"] if stage == "roi" else d["pipeline"][stage]
                )
                for k, v in fill.items():
                    block.setdefault(k, v)

        return self._apply(mutate)

    def set_stage_field(self, stage: str, field: str, value: Any) -> R.Recipe:
        """스테이지 필드 하나. ``field`` 는 점 경로 가능(``elastic.alpha``) — 카드 폼(`params.py`)이 그렇게 평탄화한다.
        ``roi`` 는 ``placement.roi`` 하위."""
        parts = tuple(field.split("."))
        if stage == "roi":
            return self.set_field(("pipeline", "placement", "roi", *parts), value)
        return self.set_field(("pipeline", stage, *parts), value)

    def replace_recipe(self, recipe: R.Recipe, path: Path | None = None) -> None:
        """레시피 파일 열기. 은행/대상이 다르면 다시 prepare가 필요하다."""
        self.recipe_path = path
        self._commit(recipe)
        if self.needs_prepare():
            self.prepared = None

    def load(self, path: str | Path) -> R.Recipe:
        try:
            rec, self.path_notes = R.Recipe.load_with_notes(path)
        except ValidationError as e:
            raise SessionError(R.format_validation_error(e)) from e
        except (KeyError, ValueError, OSError) as e:
            raise SessionError(f"레시피 로드 실패: {e}") from e
        self.replace_recipe(rec, Path(path))
        return rec

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.recipe.to_yaml(), encoding="utf-8")
        self.recipe_path = p
        return p

    # ------------------------------------------------------------------ 준비(은행·대상)

    def prepare_key(self) -> tuple[str, str]:
        return (self.recipe.inputs.bank_key(), self.recipe.inputs.targets.as_posix())

    def needs_prepare(self) -> bool:
        if self.prepared is None:
            return True
        return self.prepare_key() != (
            self.prepared.recipe.inputs.bank_key(),
            self.prepared.recipe.inputs.targets.as_posix(),
        )

    def prepare_now(self) -> runner.Prepared:
        """동기 준비(테스트·작은 은행). GUI는 워커에서 ``runner.prepare(session.recipe)``를 돌리고 ``accept_prepared``."""
        try:
            prep = runner.prepare(self.recipe)
        except runner.PrepareError as e:
            raise SessionError(str(e)) from e
        self.accept_prepared(prep)
        return prep

    def accept_prepared(self, prep: runner.Prepared) -> bool:
        """워커 결과 수용. 그 사이 은행/대상이 또 바뀌었으면 False(버림)."""
        if (
            prep.recipe.inputs.bank_key(),
            prep.recipe.inputs.targets.as_posix(),
        ) != self.prepare_key():
            return False
        if prep.recipe.to_yaml() != self.recipe.to_yaml():
            prep = runner.reprepare(prep, self.recipe)
        self.prepared = prep
        self.warnings = list(prep.warnings)
        self.target_index = min(self.target_index, max(0, len(prep.targets) - 1))
        self.generation += 1
        return True

    # ------------------------------------------------------------------ 선택

    @property
    def targets(self) -> list[Path]:
        return list(self.prepared.targets) if self.prepared is not None else []

    @property
    def target(self) -> Path | None:
        t = self.targets
        return t[self.target_index] if t and 0 <= self.target_index < len(t) else None

    def select_target(self, index: int) -> None:
        if index != self.target_index:
            self.target_index = index
            self.generation += 1

    def select_variant(self, k: int) -> None:
        self.variant_index = max(0, min(k, self.n_variants - 1))

    def set_n_variants(self, n: int) -> None:
        n = max(1, int(n))
        if n != self.n_variants:
            self.n_variants = n
            self.variant_index = min(self.variant_index, n - 1)
            self.generation += 1

    def set_long_side(self, px: int) -> None:
        if px != self.long_side:
            self.long_side = int(px)
            self.generation += 1

    # ------------------------------------------------------------------ 표시용

    def summary(self) -> Mapping[str, str]:
        r = self.recipe
        out: dict[str, str] = {
            "recipe": r.name,
            "preset": str(r.pipeline.preset),
            "seed": str(r.seed),
            "bank": r.inputs.bank_key() or "(없음)",
            "targets": r.inputs.targets.as_posix(),
        }
        if self.prepared is not None:
            out["bank_n"] = str(len(self.prepared.bank))
            out["targets_n"] = str(len(self.prepared.targets))
            out["pipeline_hash"] = self.prepared.pipeline_hash
        return out

    def run_command(self) -> str:
        """배치로 보내기 — 지금 레시피를 CLI로 돌리는 한 줄."""
        p = self.recipe_path.as_posix() if self.recipe_path else "<recipe.yaml>"
        return f"anograft run {p}"

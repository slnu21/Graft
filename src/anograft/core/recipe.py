"""레시피 스키마 (설계 §4) — YAML(사람이 읽고 diff) + pydantic v2(GUI·CLI 공용 검증).

규칙:
- ``extra="forbid"`` — 오타 키는 에러. 조용히 무시되는 옵션을 만들지 않는다.
- 스테이지 설정은 ``method``로 갈리는 discriminated union. 다른 method의 키를 주면 에러.
- ``[lo, hi]`` 범위는 lo <= hi. 비율·strength는 0..1.
- 프리셋은 ``pipeline`` 아래 기본값 딕셔너리(``anograft/presets/*.yaml``). 병합 순서:
  코드 기본값 → 프리셋 → 사용자 YAML → CLI 오버라이드. 사용자가 스테이지의 ``method``를 바꾸면
  그 스테이지는 사용자 블록을 통째로 쓴다(프리셋의 다른 method 키가 섞이지 않게).
- ``to_yaml()``은 필드 순서 고정(sort_keys=False) → 파이프라인 해시의 입력.
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# 공용 타입
# ---------------------------------------------------------------------------


def _check_range(v: tuple[float, float]) -> tuple[float, float]:
    lo, hi = v
    if lo > hi:
        raise ValueError(f"범위는 [lo, hi]에 lo <= hi 여야 합니다: {list(v)}")
    return (float(lo), float(hi))


def _check_int_range(v: tuple[int, int]) -> tuple[int, int]:
    lo, hi = v
    if lo > hi:
        raise ValueError(f"범위는 [lo, hi]에 lo <= hi 여야 합니다: {list(v)}")
    return (int(lo), int(hi))


Range = Annotated[tuple[float, float], AfterValidator(_check_range)]
IntRange = Annotated[tuple[int, int], AfterValidator(_check_int_range)]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _posix(p: Path) -> str:
    """경로는 항상 POSIX 구분자로 직렬화 — Windows/Linux에서 같은 YAML·같은 해시."""
    return p.as_posix()


# ---------------------------------------------------------------------------
# inputs / output
# ---------------------------------------------------------------------------


class Inputs(_Strict):
    bank: Path
    targets: Path  # 폴더 또는 경로 목록 .txt
    um_per_px: float | None = Field(default=None, gt=0)

    @field_serializer("bank", "targets")
    def _ser_paths(self, p: Path) -> str:
        return _posix(p)


class YoloWriterConfig(_Strict):
    format: Literal["yolo"] = "yolo"
    seg: bool = False  # true면 박스 대신 폴리곤 (YOLO-seg)
    names_from: Literal["bank"] = "bank"


class PairsWriterConfig(_Strict):
    format: Literal["pairs"] = "pairs"


WriterConfig = Annotated[YoloWriterConfig | PairsWriterConfig, Field(discriminator="format")]


class Output(_Strict):
    root: Path
    count: int = Field(ge=1)
    class_ratio: dict[str, float] | None = None  # null = 은행 클래스 균등
    defects_per_image: IntRange = (1, 1)
    include_normals: bool = True
    copy_mode: Literal["copy", "hardlink"] = "copy"
    writer: WriterConfig = Field(default_factory=YoloWriterConfig)

    @field_serializer("root")
    def _ser_root(self, p: Path) -> str:
        return _posix(p)

    @field_validator("defects_per_image")
    @classmethod
    def _at_least_one(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] < 1:
            raise ValueError("defects_per_image 최소값은 1 이상이어야 합니다")
        return v

    @field_validator("class_ratio")
    @classmethod
    def _normalize_ratio(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        """합이 1이 아니면 정규화(에러 아님 — 경고는 ``Recipe.warnings``에서). 음수·전부 0은 에러."""
        if v is None:
            return None
        if not v:
            raise ValueError("class_ratio가 비어 있습니다 — null(균등) 또는 클래스 하나 이상")
        if any(r < 0 for r in v.values()):
            raise ValueError("class_ratio 값은 0 이상이어야 합니다")
        total = sum(v.values())
        if total <= 0:
            raise ValueError("class_ratio 합이 0입니다")
        return {k: r / total for k, r in v.items()}


# ---------------------------------------------------------------------------
# 스테이지 설정 — method별 discriminated union (설계 §6.1 행렬의 v0.1 열)
# ---------------------------------------------------------------------------


class BankSourceConfig(_Strict):
    method: Literal["bank"] = "bank"
    classes: list[str] | None = None  # null = class_ratio 키 (그것도 null이면 은행 전체)
    min_sources_warn: int = Field(default=10, ge=0)


SourceConfig = Annotated[BankSourceConfig, Field(discriminator="method")]


class ElasticConfig(_Strict):
    alpha: float = Field(default=0.0, ge=0.0)  # 0 = off
    sigma: float = Field(default=4.0, gt=0.0)


class AffineGeometryConfig(_Strict):
    method: Literal["affine"] = "affine"
    scale: Range = (0.8, 1.25)  # 물리 축척 × 이 배율
    rotate: Range = (-180.0, 180.0)  # deg
    flip: bool = True
    elastic: ElasticConfig = Field(default_factory=ElasticConfig)

    @field_validator("scale")
    @classmethod
    def _positive_scale(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] <= 0:
            raise ValueError("scale 범위는 양수여야 합니다")
        return v

    @field_validator("rotate")
    @classmethod
    def _rotate_bounds(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] < -360 or v[1] > 360:
            raise ValueError("rotate 범위는 [-360, 360] 안이어야 합니다")
        return v


GeometryConfig = Annotated[AffineGeometryConfig, Field(discriminator="method")]


class OtsuRoiConfig(_Strict):
    method: Literal["otsu"] = "otsu"
    invert: Literal["auto", "yes", "no"] = "auto"
    erode_px: int = Field(default=8, ge=0)


class NoneRoiConfig(_Strict):
    method: Literal["none"] = "none"


class MaskDirRoiConfig(_Strict):
    method: Literal["mask_dir"] = "mask_dir"
    path: Path

    @field_serializer("path")
    def _ser_path(self, p: Path) -> str:
        return _posix(p)


RoiConfig = Annotated[
    OtsuRoiConfig | NoneRoiConfig | MaskDirRoiConfig, Field(discriminator="method")
]


class ShrinkConfig(_Strict):
    factor: float = Field(default=0.8, gt=0.0, lt=1.0)
    rounds: int = Field(default=3, ge=0)


class SampledPlacementConfig(_Strict):
    method: Literal["sampled"] = "sampled"
    roi: RoiConfig = Field(default_factory=OtsuRoiConfig)
    distribution: Literal["uniform", "edge", "center"] = "uniform"
    margin_px: int = Field(default=8, ge=0)
    max_tries: int = Field(default=50, ge=1)
    shrink_on_fail: ShrinkConfig = Field(default_factory=ShrinkConfig)


PlacementConfig = Annotated[SampledPlacementConfig, Field(discriminator="method")]


class PasteBlendConfig(_Strict):
    method: Literal["paste"] = "paste"


class AlphaBlendConfig(_Strict):
    method: Literal["alpha"] = "alpha"
    feather_px: int = Field(default=3, ge=0)


class PoissonBlendConfig(_Strict):
    method: Literal["poisson"] = "poisson"
    poisson_mode: Literal["normal", "mixed"] = "mixed"
    # 소스 마스크를 이만큼 팽창한 영역을 Poisson 풀이 영역으로. OpenCV seamlessClone이 내부에서 마스크를 3px 침식하므로
    # 0~2면 얇은 결함(스크래치)이 통째로 사라진다 — 실측 5부터 대비 100% 보존. GT는 여전히 소스 마스크(+정책).
    mask_dilate_px: int = Field(default=5, ge=0)
    feather_px: int = Field(default=3, ge=0)  # 폴백(alpha)에서 사용


class MultibandBlendConfig(_Strict):
    method: Literal["multiband"] = "multiband"
    levels: int = Field(default=4, ge=1, le=8)


BlendConfig = Annotated[
    PasteBlendConfig | AlphaBlendConfig | PoissonBlendConfig | MultibandBlendConfig,
    Field(discriminator="method"),
]


class NoneHarmonizeConfig(_Strict):
    method: Literal["none"] = "none"


class StatsHarmonizeConfig(_Strict):
    method: Literal["stats"] = "stats"
    strength: Unit = 0.5
    ring_px: int = Field(default=12, ge=1)


class ReinhardHarmonizeConfig(_Strict):
    method: Literal["reinhard"] = "reinhard"
    strength: Unit = 0.5
    ring_px: int = Field(default=12, ge=1)


class HistmatchHarmonizeConfig(_Strict):
    method: Literal["histmatch"] = "histmatch"
    strength: Unit = 0.5
    ring_px: int = Field(default=12, ge=1)


HarmonizeConfig = Annotated[
    NoneHarmonizeConfig | StatsHarmonizeConfig | ReinhardHarmonizeConfig | HistmatchHarmonizeConfig,
    Field(discriminator="method"),
]


class NoneDegradeConfig(_Strict):
    method: Literal["none"] = "none"


class CameraDegradeConfig(_Strict):
    method: Literal["camera"] = "camera"
    noise_sigma: Range = (0.0, 2.0)
    blur_sigma: Range = (0.0, 0.6)
    jpeg_quality: IntRange | None = None  # null = off

    @field_validator("noise_sigma", "blur_sigma")
    @classmethod
    def _non_negative(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] < 0:
            raise ValueError("σ 범위는 0 이상이어야 합니다")
        return v

    @field_validator("jpeg_quality")
    @classmethod
    def _quality_bounds(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        if v is not None and (v[0] < 1 or v[1] > 100):
            raise ValueError("jpeg_quality 범위는 [1, 100] 안이어야 합니다")
        return v


DegradeConfig = Annotated[NoneDegradeConfig | CameraDegradeConfig, Field(discriminator="method")]


class GtMaskConfig(_Strict):
    policy: Literal["source", "diff", "union"] = "union"
    diff_threshold: int = Field(default=12, ge=0, le=255)
    dilate_px: int = Field(default=2, ge=0)


class PipelineConfig(_Strict):
    preset: str | None = None  # 출처 기록용 — 병합은 로드 시 끝난다
    source: SourceConfig = Field(default_factory=BankSourceConfig)
    geometry: GeometryConfig = Field(default_factory=AffineGeometryConfig)
    placement: PlacementConfig = Field(default_factory=SampledPlacementConfig)
    blend: BlendConfig = Field(default_factory=PoissonBlendConfig)
    harmonize: HarmonizeConfig = Field(default_factory=StatsHarmonizeConfig)
    degrade: DegradeConfig = Field(default_factory=CameraDegradeConfig)
    gtmask: GtMaskConfig = Field(default_factory=GtMaskConfig)


# 스테이지 키 = 파이프라인 순서. registry·pipeline·CLI가 이 튜플을 공유한다. (roi는 placement 안의 하위 설정)
STAGE_KEYS: tuple[str, ...] = (
    "source",
    "geometry",
    "placement",
    "blend",
    "harmonize",
    "degrade",
    "gtmask",
)


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


@runtime_checkable
class BankLike(Protocol):
    """``validate_against``가 필요로 하는 은행의 최소 인터페이스 (실물은 bank-import 단위에서)."""

    @property
    def classes(self) -> Sequence[str]: ...

    def counts(self) -> Mapping[str, int]: ...


class Recipe(_Strict):
    version: Literal[1] = 1
    name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    inputs: Inputs
    output: Output
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @model_validator(mode="after")
    def _cross_checks(self) -> Recipe:
        ratio = self.output.class_ratio
        classes = self.pipeline.source.classes
        if ratio is not None and classes is not None:
            missing = sorted(set(ratio) - set(classes))
            if missing:
                raise ValueError(f"class_ratio의 클래스가 source.classes에 없습니다: {missing}")
        return self

    # --- 직렬화 ---

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        """필드 순서 고정 덤프 — 파이프라인 해시 입력. 같은 레시피는 언제 덤프해도 같은 문자열."""
        return yaml.safe_dump(
            self.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=None
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Recipe:
        """프리셋 병합 후 검증. ``pipeline.preset``이 있으면 패키지 프리셋을 아래에 깐다."""
        merged = apply_preset(dict(data))
        return cls.model_validate(merged)

    @classmethod
    def from_yaml(cls, text: str) -> Recipe:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("레시피 YAML의 최상위는 매핑이어야 합니다")
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path, **overrides: Any) -> Recipe:
        """파일 로드 + CLI 오버라이드(``seed=``, ``count=``, ``out=``)."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"레시피 YAML의 최상위는 매핑이어야 합니다: {path}")
        return cls.from_dict(apply_overrides(data, **overrides))

    # --- 은행 대조 (런타임 검증) ---

    def effective_classes(self, bank: BankLike) -> list[str]:
        """실제로 뽑을 클래스 목록: source.classes → class_ratio 키 → 은행 전체."""
        if self.pipeline.source.classes is not None:
            return list(self.pipeline.source.classes)
        if self.output.class_ratio is not None:
            return list(self.output.class_ratio)
        return list(bank.classes)

    def class_probabilities(self, bank: BankLike) -> dict[str, float]:
        classes = self.effective_classes(bank)
        ratio = self.output.class_ratio
        if ratio is None:
            return {c: 1.0 / len(classes) for c in classes}
        probs = {c: ratio.get(c, 0.0) for c in classes}
        total = sum(probs.values())
        if total <= 0:
            raise ValueError("선택된 클래스의 class_ratio 합이 0입니다")
        return {c: p / total for c, p in probs.items()}

    def with_method(self, stage: str, method: str) -> Recipe:
        """스테이지 method만 바꾼 새 레시피 — 그 스테이지 블록은 새 method의 **기본값만** 남긴다(discriminated union이라
        다른 method의 키가 섞이면 검증 오류). ``preview --compare-methods``·GUI 카드 콤보가 같은 규칙을 쓴다."""
        d = self.to_dict()
        set_method_in_dict(d, stage, method)
        return Recipe.from_dict(d)

    def validate_against(self, bank: BankLike) -> list[str]:
        """은행과 대조. 치명적이면 ``ValueError``, 아니면 경고 문자열 목록을 돌려준다."""
        warnings: list[str] = []
        bank_classes = set(bank.classes)
        counts = bank.counts()
        classes = self.effective_classes(bank)
        missing = sorted(set(classes) - bank_classes)
        if missing:
            raise ValueError(f"은행에 없는 클래스입니다: {missing} (은행: {sorted(bank_classes)})")
        empty = [c for c in classes if counts.get(c, 0) == 0]
        if empty:
            raise ValueError(f"소스가 0개인 클래스입니다: {empty}")
        threshold = self.pipeline.source.min_sources_warn
        for c in classes:
            n = counts.get(c, 0)
            if n < threshold:
                warnings.append(
                    f"클래스 '{c}' 소스가 {n}개 — {threshold}개 미만이면 모델이 특정 소스를 외울 수 있습니다"
                )
        if self.output.class_ratio is not None:
            unused = sorted(set(self.output.class_ratio) - set(classes))
            if unused:
                warnings.append(f"class_ratio에 있지만 쓰이지 않는 클래스: {unused}")
        return warnings


def set_method_in_dict(d: dict[str, Any], stage: str, method: str) -> None:
    """레시피 dict의 스테이지 블록을 ``{method}``(gtmask는 ``{policy}``, roi는 ``placement.roi``)로 갈아 끼운다."""
    key = "policy" if stage == "gtmask" else "method"
    pipe = d.setdefault("pipeline", {})
    if stage == "roi":
        pipe.setdefault("placement", {})["roi"] = {"method": method}
    else:
        pipe[stage] = {key: method}


# ---------------------------------------------------------------------------
# 프리셋
# ---------------------------------------------------------------------------

_PRESET_PACKAGE = "anograft.presets"


def preset_names() -> list[str]:
    files = importlib.resources.files(_PRESET_PACKAGE)
    return sorted(p.name[: -len(".yaml")] for p in files.iterdir() if p.name.endswith(".yaml"))


def load_preset(name: str) -> dict[str, Any]:
    """프리셋 = ``pipeline`` 아래 기본값 딕셔너리. 파일의 최상위 키는 ``pipeline`` 하나."""
    files = importlib.resources.files(_PRESET_PACKAGE)
    res = files / f"{name}.yaml"
    if not res.is_file():
        raise KeyError(f"프리셋이 없습니다: {name!r} (사용 가능: {', '.join(preset_names())})")
    data = yaml.safe_load(res.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "pipeline" not in data:
        raise ValueError(f"프리셋 파일 형식 오류: {name}.yaml — 최상위 'pipeline' 키가 필요")
    pipe = dict(data["pipeline"])
    pipe["preset"] = name
    return pipe


def merge_method_aware(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """딕셔너리 깊은 병합. 단, 양쪽 모두 ``method``가 있고 값이 다르면 ``over`` 블록을 통째로 쓴다
    (프리셋의 다른 method 키가 섞여 extra=forbid에 걸리는 것을 막는다)."""
    out: dict[str, Any] = dict(base)
    for k, v in over.items():
        bv = out.get(k)
        if isinstance(v, Mapping) and isinstance(bv, Mapping):
            bm, om = bv.get("method"), v.get("method")
            if bm is not None and om is not None and bm != om:
                out[k] = dict(v)
            else:
                out[k] = merge_method_aware(bv, v)
        else:
            out[k] = v
    return out


def apply_preset(data: dict[str, Any]) -> dict[str, Any]:
    pipe = data.get("pipeline")
    if not isinstance(pipe, Mapping):
        return data
    name = pipe.get("preset")
    if not name:
        return data
    merged = merge_method_aware(load_preset(str(name)), pipe)
    merged["preset"] = str(name)
    out = dict(data)
    out["pipeline"] = merged
    return out


def apply_overrides(
    data: dict[str, Any],
    *,
    seed: int | None = None,
    count: int | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """CLI 오버라이드 — 검증 전 딕셔너리 단계에서 적용한다."""
    d = dict(data)
    if seed is not None:
        d["seed"] = seed
    if count is not None or out is not None:
        o = dict(d.get("output") or {})
        if count is not None:
            o["count"] = count
        if out is not None:
            o["root"] = str(out)
        d["output"] = o
    return d


def format_validation_error(err: ValidationError) -> str:
    """사용자용 한 줄 오류 목록 — 어느 키가 왜 틀렸는지."""
    lines = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"]) or "<root>"
        lines.append(f"  {loc}: {e['msg']}")
    return "레시피 검증 실패:\n" + "\n".join(lines)


def init_recipe_dict(
    preset: str = "poisson-graft",
    *,
    name: str | None = None,
    bank: str = "./bank/mine",
    targets: str = "./normals",
    out: str = "./out/run-01",
    seed: int = 20260913,
    count: int = 100,
) -> dict[str, Any]:
    """``recipe init``용 — 프리셋을 완전히 펼친 레시피 딕셔너리(사용자가 모든 손잡이를 본다)."""
    data = {
        "version": 1,
        "name": name or preset,
        "seed": seed,
        "inputs": {"bank": bank, "targets": targets, "um_per_px": None},
        "output": {
            "root": out,
            "count": count,
            "class_ratio": None,
            "defects_per_image": [1, 1],
            "include_normals": True,
            "copy_mode": "copy",
            "writer": {"format": "yolo", "seg": False, "names_from": "bank"},
        },
        "pipeline": {"preset": preset},
    }
    return Recipe.from_dict(data).to_dict()

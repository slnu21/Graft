"""레시피 스키마 (설계 §4) — YAML(사람이 읽고 diff) + pydantic v2(GUI·CLI 공용 검증).

규칙:
- ``extra="forbid"`` — 오타 키는 에러. 조용히 무시되는 옵션을 만들지 않는다.
- 스테이지 설정은 ``method``로 갈리는 discriminated union. 다른 method의 키를 주면 에러.
- ``[lo, hi]`` 범위는 lo <= hi. 비율·strength는 0..1.
- 프리셋은 ``pipeline`` 아래 기본값 딕셔너리(``anograft/presets/*.yaml``). 병합 순서:
  코드 기본값 → 프리셋 → 사용자 YAML → CLI 오버라이드. 사용자가 스테이지의 ``method``를 바꾸면
  그 스테이지는 사용자 블록을 통째로 쓴다(프리셋의 다른 method 키가 섞이지 않게).
- ``to_yaml()``은 필드 순서 고정(sort_keys=False). 파이프라인 해시의 입력은 ``hash_yaml()``(output.root·count 제외).
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Protocol, runtime_checkable

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
    bank: Path | None = (
        None  # source.method == bank 면 필수 (Recipe._cross_checks). self-cut·perlin은 은행 없이 동작
    )
    targets: Path  # 폴더 또는 경로 목록 .txt
    um_per_px: float | None = Field(default=None, gt=0)

    @field_serializer("bank", "targets")
    def _ser_paths(self, p: Path | None) -> str | None:
        return None if p is None else _posix(p)

    def bank_key(self) -> str:
        """GUI·prepare 비교용 문자열 — 은행이 없으면 빈 문자열."""
        return "" if self.bank is None else self.bank.as_posix()


class YoloWriterConfig(_Strict):
    format: Literal["yolo"] = "yolo"
    seg: bool = False  # true면 박스 대신 폴리곤 (YOLO-seg)
    names_from: Literal["bank"] = "bank"


class PairsWriterConfig(_Strict):
    format: Literal["pairs"] = "pairs"


class MvtecWriterConfig(_Strict):
    """정본 위에 ``<layout_dir>/<category>/{train/good, test/good, test/<class>, ground_truth/<class>}`` 를 추가로(anomalib).
    정상 이미지는 ``test_normal_ratio`` 비율로 test/good(``split_rng(seed)``, 결정적). 이미지당 클래스 하나 — 섞이면 면적 최대."""

    format: Literal["mvtec"] = "mvtec"
    category: str = Field(default="graft", min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    test_normal_ratio: Unit = 0.2
    layout_dir: str = Field(default="mvtec", min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")


class CocoWriterConfig(_Strict):
    """정본 위에 ``annotations.json``(COCO instances — 폴리곤 segmentation·bbox·area, categories = 은행 classes, id 1-based)."""

    format: Literal["coco"] = "coco"
    description: str = "anograft synthetic defects"
    supercategory: str = "defect"
    segmentation: Literal["polygon", "rle"] = (
        "polygon"  # rle = 비압축 RLE(열 우선 run 길이, iscrowd 0) — 조각난 마스크도 무손실
    )


WriterConfig = Annotated[
    YoloWriterConfig | PairsWriterConfig | MvtecWriterConfig | CocoWriterConfig,
    Field(discriminator="format"),
]


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


class TagFilter(_Strict):
    """은행 소스를 태그로 거른다(임포트 ``--tags`` 로 붙인 제품명 등). ``include`` 중 **하나라도** 가진 소스만(비면 전부),
    ``exclude`` 중 하나라도 가지면 제외. 둘 다 비면 필터 없음. 클래스 목록·확률은 그대로고 클래스 **안의 풀**만 줄어든다."""

    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self.include or self.exclude)

    def accepts(self, tags: Sequence[str]) -> bool:
        have = set(tags)
        if self.include and not have.intersection(self.include):
            return False
        return not have.intersection(self.exclude)

    def describe(self) -> str:
        parts = []
        if self.include:
            parts.append(f"include={list(self.include)}")
        if self.exclude:
            parts.append(f"exclude={list(self.exclude)}")
        return " ".join(parts) or "(없음)"


class BankSourceConfig(_Strict):
    method: Literal["bank"] = "bank"
    classes: list[str] | None = None  # null = class_ratio 키 (그것도 null이면 은행 전체)
    tags: TagFilter = Field(default_factory=TagFilter)  # null 도 허용(= 필터 없음)
    min_sources_warn: int = Field(default=10, ge=0)

    @field_validator("tags", mode="before")
    @classmethod
    def _tags_none(cls, v: Any) -> Any:
        return {} if v is None else v


class ColorJitterConfig(_Strict):
    """CutPaste식 색 지터 — 각 값은 최대 변화폭(0 = 끔). brightness·contrast·saturation은 배율 ``1±v``, hue는 ``±v·180°``."""

    brightness: float = Field(default=0.1, ge=0.0, le=1.0)
    contrast: float = Field(default=0.1, ge=0.0, le=1.0)
    saturation: float = Field(default=0.1, ge=0.0, le=1.0)
    hue: float = Field(default=0.1, ge=0.0, le=0.5)

    @property
    def enabled(self) -> bool:
        return any(v > 0 for v in (self.brightness, self.contrast, self.saturation, self.hue))


class SelfCutSourceConfig(_Strict):
    """CutPaste(Li 2021) — 대상 이미지 자신에서 패치를 잘라 붙인다. 은행 불필요.

    ``shape``: ``rect``(면적비·종횡비) · ``scar``(가는 긴 띠, px 치수; 회전은 geometry 몫) · ``mixed``(결함마다 50/50)."""

    method: Literal["self-cut"] = "self-cut"
    cls: str = Field(default="cutpaste", min_length=1)  # 사이드카·YOLO 클래스 이름
    shape: Literal["rect", "scar", "mixed"] = "mixed"
    area_ratio: Range = (0.02, 0.15)  # rect: 패치 면적 / 이미지 면적
    aspect: Range = (0.3, 3.3)  # rect: w/h (로그균등)
    scar_width_px: IntRange = (2, 16)
    scar_length_px: IntRange = (20, 120)
    margin_px: int = Field(default=8, ge=0)  # 크롭 여유 (poisson 팽창·페더가 잘리지 않게)
    max_tries: int = Field(default=20, ge=1)  # ROI 안에서 자를 자리 찾기
    jitter: ColorJitterConfig = Field(default_factory=ColorJitterConfig)

    @field_validator("area_ratio")
    @classmethod
    def _area_ratio_unit(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] <= 0 or v[1] > 1:
            raise ValueError("area_ratio는 (0, 1] 안이어야 합니다")
        return v

    @field_validator("aspect")
    @classmethod
    def _aspect_positive(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] <= 0:
            raise ValueError("aspect 범위는 양수여야 합니다")
        return v

    @field_validator("scar_width_px", "scar_length_px")
    @classmethod
    def _scar_positive(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] < 1:
            raise ValueError("스카 치수는 1px 이상이어야 합니다")
        return v


class PerlinSourceConfig(_Strict):
    """DRAEM(Zavrtanik 2021) — 펄린 노이즈 임계 마스크 + 텍스처. 은행 불필요.

    ``texture``: ``self``(대상 자신의 다른 창을 증강) · ``dir``(``texture_dir``의 이미지, 예: 사용자가 내려받은 DTD —
    재배포하지 않고 읽기만; 비어 있으면 self로 폴백 + 경고)."""

    method: Literal["perlin-texture"] = "perlin-texture"
    cls: str = Field(default="anomaly", min_length=1)
    texture: Literal["self", "dir"] = "self"
    texture_dir: Path | None = None
    size_ratio: Range = (0.2, 0.5)  # 창 한 변 / min(H, W)
    scale_range: IntRange = (0, 5)  # 펄린 해상도 지수 k: res = 2^k (DRAEM 0..5)
    threshold: float = Field(default=0.5, ge=-1.0, le=1.0)
    rotate: Range = (-90.0, 90.0)
    min_area_px: int = Field(default=16, ge=1)  # 임계 마스크가 이보다 작으면 그 결함 skip
    augment: bool = True  # DRAEM식 텍스처 증강(3종 무작위)
    max_tries: int = Field(default=5, ge=1)  # 마스크 면적 부족 시 재생성

    @field_serializer("texture_dir")
    def _ser_dir(self, p: Path | None) -> str | None:
        return None if p is None else _posix(p)

    @field_validator("size_ratio")
    @classmethod
    def _size_ratio_unit(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] <= 0 or v[1] > 1:
            raise ValueError("size_ratio는 (0, 1] 안이어야 합니다")
        return v

    @field_validator("scale_range")
    @classmethod
    def _scale_range_bounds(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] < 0 or v[1] > 8:
            raise ValueError("scale_range는 [0, 8] 안이어야 합니다 (res = 2^k)")
        return v

    @model_validator(mode="after")
    def _dir_requires_path(self) -> PerlinSourceConfig:
        if self.texture == "dir" and self.texture_dir is None:
            raise ValueError("texture: dir 이면 texture_dir 이 필요합니다")
        return self


SourceConfig = Annotated[
    BankSourceConfig | SelfCutSourceConfig | PerlinSourceConfig, Field(discriminator="method")
]
BANKLESS_SOURCES: frozenset[str] = frozenset({"self-cut", "perlin-texture"})


class ElasticConfig(_Strict):
    alpha: float = Field(default=0.0, ge=0.0)  # 0 = off
    sigma: float = Field(default=4.0, gt=0.0)


def _check_scale(v: tuple[float, float]) -> tuple[float, float]:
    if v[0] <= 0:
        raise ValueError("scale 범위는 양수여야 합니다")
    return v


def _check_rotate(v: tuple[float, float]) -> tuple[float, float]:
    if v[0] < -360 or v[1] > 360:
        raise ValueError("rotate 범위는 [-360, 360] 안이어야 합니다")
    return v


FlipMode = Literal["none", "horizontal", "vertical", "both"]
FLIP_MODES: tuple[str, ...] = ("none", "horizontal", "vertical", "both")


def _norm_flip(v: Any) -> Any:
    """``flip`` 하위 호환 — ``true`` = both(좌우·상하 각각 50 %), ``false`` = none. 문자열은 그대로 검증에."""
    if isinstance(v, bool):
        return "both" if v else "none"
    return v


class GeometryOverride(_Strict):
    """클래스별 기하 오버라이드(0.7.3) — 준 것만 덮어쓴다. 한 은행에 스크래치(±180 무방)와 찍힘(조명 의존 → ±15·flip 끔)이
    섞여 있을 때 레시피 하나로. rng 소비는 flip 여부 외엔 같다(범위만 다름)."""

    scale: Range | None = None
    rotate: Range | None = None
    flip: FlipMode | None = None

    @field_validator("flip", mode="before")
    @classmethod
    def _flip_compat(cls, v: Any) -> Any:
        return _norm_flip(v)

    @field_validator("scale")
    @classmethod
    def _positive_scale(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        return None if v is None else _check_scale(v)

    @field_validator("rotate")
    @classmethod
    def _rotate_bounds(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        return None if v is None else _check_rotate(v)


@dataclass(frozen=True)
class EffectiveGeometry:
    scale: tuple[float, float]
    rotate: tuple[float, float]
    flip: str  # FlipMode
    overridden: bool

    @property
    def flip_h(self) -> bool:
        return self.flip in ("horizontal", "both")

    @property
    def flip_v(self) -> bool:
        return self.flip in ("vertical", "both")


class AffineGeometryConfig(_Strict):
    method: Literal["affine"] = "affine"
    scale: Range = (0.8, 1.25)  # 물리 축척 × 이 배율
    rotate: Range = (-180.0, 180.0)  # deg
    flip: FlipMode = "both"  # none · horizontal · vertical · both(좌우·상하 각각 50 %). YAML 의 true/false 도 받는다(0.7.5 전 호환)
    elastic: ElasticConfig = Field(default_factory=ElasticConfig)
    per_class: dict[str, GeometryOverride] = Field(
        default_factory=dict
    )  # 클래스 → 오버라이드(카드 편집기엔 안 나옴 — YAML 로). 은행에 없는 클래스는 validate_against 경고

    @field_validator("flip", mode="before")
    @classmethod
    def _flip_compat(cls, v: Any) -> Any:
        return _norm_flip(v)

    @field_validator("scale")
    @classmethod
    def _positive_scale(cls, v: tuple[float, float]) -> tuple[float, float]:
        return _check_scale(v)

    @field_validator("rotate")
    @classmethod
    def _rotate_bounds(cls, v: tuple[float, float]) -> tuple[float, float]:
        return _check_rotate(v)

    def for_class(self, cls: str | None) -> EffectiveGeometry:
        """클래스에 적용되는 (scale, rotate, flip) — ``per_class`` 에 있으면 준 필드만 덮어쓴다."""
        o = self.per_class.get(cls) if cls is not None else None
        if o is None:
            return EffectiveGeometry(tuple(self.scale), tuple(self.rotate), self.flip, False)
        return EffectiveGeometry(
            tuple(o.scale) if o.scale is not None else tuple(self.scale),
            tuple(o.rotate) if o.rotate is not None else tuple(self.rotate),
            o.flip if o.flip is not None else self.flip,
            True,
        )


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


class GrabCutRoiConfig(_Strict):
    """GrabCut 전경 = ROI (v0.4). ``init: rect``는 테두리 ``rect_margin`` 비율 바깥을 확정 배경으로 두고 안쪽에서 전경을 찾는다
    (물체가 중앙, 배경이 테두리 — Otsu ``auto``와 같은 전제). ``init: otsu``는 Otsu 전경/배경을 '아마도' 라벨로 넣어 다듬는다.
    ``work_px``(긴 변, 0 = 원본)로 줄여 풀고 NEAREST로 되돌린다 — 4K 원본 GrabCut은 수십 초."""

    method: Literal["grabcut"] = "grabcut"
    init: Literal["rect", "otsu"] = "rect"
    invert: Literal["auto", "yes", "no"] = "auto"  # init: otsu 일 때만
    rect_margin: float = Field(default=0.03, ge=0.0, lt=0.5)
    iters: int = Field(default=5, ge=1)
    work_px: int = Field(default=1024, ge=0)
    erode_px: int = Field(default=8, ge=0)


class AnnulusRoiConfig(_Strict):
    """링(annulus) ROI (v0.6, KNOWN-ISSUES #2 #4) — 원형 부품에서 결함이 생기는 **가공 링 면만** 허용한다. ``otsu``/``grabcut`` 은
    물체 vs 배경만 가르므로 중앙 리세스에도 결함이 떨어졌다. ``center``/``radius`` 가 null 이면 Otsu 전경의 최소외접원으로
    대상마다 자동 검출(촬영마다 부품이 움직여도 링이 따라감). ``units: ratio`` 면 ``r_inner``·``r_outer`` 는 그 반경의 배율
    (토크스 소켓 실측: 319~510 px / 바깥 570 px ≈ 0.56~0.9), ``px`` 면 절대값. ``erode_px`` 는 안·바깥 경계 모두에서 깎는다."""

    method: Literal["annulus"] = "annulus"
    center: tuple[float, float] | None = None  # (cx, cy) px. null = 자동
    radius: float | None = Field(
        default=None, gt=0.0
    )  # 비율의 기준 반경(px). null = 자동(최소외접원)
    r_inner: float = Field(default=0.55, ge=0.0)
    r_outer: float = Field(default=0.9, gt=0.0)
    units: Literal["ratio", "px"] = "ratio"
    invert: Literal["auto", "yes", "no"] = "auto"  # 자동 검출의 Otsu 극성
    erode_px: int = Field(default=4, ge=0)

    @model_validator(mode="after")
    def _inner_lt_outer(self) -> AnnulusRoiConfig:
        if self.r_inner >= self.r_outer:
            raise ValueError(
                f"r_inner({self.r_inner:g}) 는 r_outer({self.r_outer:g}) 보다 작아야 합니다"
            )
        return self


RoiConfig = Annotated[
    OtsuRoiConfig | NoneRoiConfig | MaskDirRoiConfig | GrabCutRoiConfig | AnnulusRoiConfig,
    Field(discriminator="method"),
]
ROI_METHODS: tuple[str, ...] = ("otsu", "none", "mask_dir", "grabcut", "annulus")  # union 순서


class ShrinkConfig(_Strict):
    factor: float = Field(default=0.8, gt=0.0, lt=1.0)
    rounds: int = Field(default=3, ge=0)


class _PlacementBase(_Strict):
    """모든 placement method 의 공통 설정 — ``roi``는 여기 하위(이미지당 1회 실행되는 별도 스테이지)."""

    roi: RoiConfig = Field(default_factory=OtsuRoiConfig)
    margin_px: int = Field(default=8, ge=0)
    max_tries: int = Field(default=50, ge=1)
    shrink_on_fail: ShrinkConfig = Field(default_factory=ShrinkConfig)


class SampledPlacementConfig(_PlacementBase):
    method: Literal["sampled"] = "sampled"
    distribution: Literal["uniform", "edge", "center"] = "uniform"


class StructureAwarePlacementConfig(_PlacementBase):
    """구조 정합 배치(v0.4). 위치 = 그래디언트 크기 가중(``prefer`` edges/flat/uniform, ``strength`` 지수, ``smooth_px`` 평활),
    방향 = 후보 자리의 구조 텐서 지배 방향에 패치 주축을 맞춘다(``align`` along = 결·에지 방향, across = 그에 수직).
    일관성 < ``min_coherence``(자리에 방향이 없음) 또는 패치 이방성 < ``min_anisotropy``(둥근 결함)면 정렬하지 않고
    geometry 가 준 방향을 유지한다. ``jitter_deg``는 정렬각에 더하는 ±균등 잡음. 시도마다 rng 2회(자리·지터).
    ``max_align_deg``(0.7.3) = 정렬 회전의 상한 — 이보다 큰 회전이 필요한 자리는 정렬하지 않는다(조명 의존 결함은
    결 정렬보다 하이라이트 방향이 우선 → dent-graft 30). null = 제한 없음(정렬은 (-90, 90]). rng 소비는 같다."""

    method: Literal["structure-aware"] = "structure-aware"
    prefer: Literal["edges", "flat", "uniform"] = "edges"
    strength: float = Field(default=1.0, ge=0.0)
    smooth_px: float = Field(default=3.0, ge=0.0)
    align: Literal["along", "across", "none"] = "along"
    min_coherence: Unit = 0.2
    min_anisotropy: Unit = 0.1
    jitter_deg: float = Field(default=10.0, ge=0.0, le=90.0)
    max_align_deg: float | None = Field(default=None, ge=0.0, le=90.0)


PlacementConfig = Annotated[
    SampledPlacementConfig | StructureAwarePlacementConfig, Field(discriminator="method")
]


class PasteBlendConfig(_Strict):
    method: Literal["paste"] = "paste"


class AlphaBlendConfig(_Strict):
    method: Literal["alpha"] = "alpha"
    feather_px: int = Field(default=3, ge=0)
    # DRAEM β — 결함마다 uniform(lo, hi) 불투명도. null(기본)이면 rng를 소비하지 않는다(기존 프리셋 스트림 불변).
    opacity: Range | None = None

    @field_validator("opacity")
    @classmethod
    def _opacity_unit(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        if v is not None and (v[0] < 0 or v[1] > 1):
            raise ValueError("opacity 범위는 [0, 1] 안이어야 합니다")
        return v


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
    """카메라 재현. v0.4 추가 옵션(``motion_blur_px``·``vignette``·``gamma``)은 **null = off 이고 off 면 rng 를 소비하지
    않는다** — v0.1 레시피의 난수 스트림·골든이 그대로다. 적용 순서 = 모션 블러 → 가우시안 블러 → 비네팅 → 노이즈 → 감마 → JPEG."""

    method: Literal["camera"] = "camera"
    noise_sigma: Range = (0.0, 2.0)
    blur_sigma: Range = (0.0, 0.6)
    jpeg_quality: IntRange | None = None  # null = off
    motion_blur_px: Range | None = None  # 선형 모션 블러 커널 길이(px). null = off, < 1 이면 생략
    motion_angle: Range = (
        0.0,
        180.0,
    )  # 모션 방향(deg, 축이라 180 주기). motion_blur_px 가 있을 때만 소비
    vignette: Range | None = (
        None  # 모서리 감광 강도 [0, 1] — 1 이면 모서리가 완전히 검다. null = off
    )
    gamma: Range | None = None  # 톤 커브 지수(1 = 항등, < 1 밝게). null = off

    @field_validator("noise_sigma", "blur_sigma", "motion_blur_px")
    @classmethod
    def _non_negative(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        if v is not None and v[0] < 0:
            raise ValueError("범위는 0 이상이어야 합니다")
        return v

    @field_validator("vignette")
    @classmethod
    def _vignette_unit(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        if v is not None and (v[0] < 0 or v[1] > 1):
            raise ValueError("vignette 범위는 [0, 1] 안이어야 합니다")
        return v

    @field_validator("gamma")
    @classmethod
    def _gamma_positive(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        if v is not None and v[0] <= 0:
            raise ValueError("gamma 범위는 양수여야 합니다")
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

    @model_validator(mode="before")
    @classmethod
    def _default_placement_method(cls, data: Any) -> Any:
        """v0.1~v0.3 레시피는 placement 가 method 하나뿐이라 ``method:`` 를 생략할 수 있었다 — 생략이면 ``sampled``
        (v0.4 에서 union 이 되면서 태그가 필수가 됐다. 다른 스테이지는 처음부터 union 이라 해당 없음)."""
        if isinstance(data, dict):
            pl = data.get("placement")
            if isinstance(pl, dict) and "method" not in pl:
                data = dict(data)
                data["placement"] = {"method": "sampled", **pl}
        return data


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
        src = self.pipeline.source
        if src.method == "bank" and self.inputs.bank is None:
            raise ValueError(
                "inputs.bank 가 필요합니다 (source.method: bank). 은행 없이 쓰려면 self-cut·perlin-texture"
            )
        ratio = self.output.class_ratio
        if src.method == "bank":
            classes = src.classes
            if ratio is not None and classes is not None:
                missing = sorted(set(ratio) - set(classes))
                if missing:
                    raise ValueError(f"class_ratio의 클래스가 source.classes에 없습니다: {missing}")
        elif ratio is not None and set(ratio) - {src.cls}:
            raise ValueError(
                f"class_ratio의 클래스가 source.cls('{src.cls}')와 다릅니다: {sorted(set(ratio) - {src.cls})}"
            )
        return self

    @property
    def bankless(self) -> bool:
        """은행 없이 동작하는 소스(self-cut·perlin-texture)인가."""
        return self.pipeline.source.method in BANKLESS_SOURCES

    # --- 직렬화 ---

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        """필드 순서 고정 덤프(``recipe.resolved.yaml``). 같은 레시피는 언제 덤프해도 같은 문자열."""
        return yaml.safe_dump(
            self.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=None
        )

    # 합성 결과(이미지 i의 픽셀·GT)에 영향이 없는 필드 — 파이프라인 해시에서 뺀다.
    # root = 어디에 쓰는가, count = 몇 장 뽑는가(이미지 i는 index로만 정해진다). 나머지 output 키
    # (class_ratio·defects_per_image)는 뽑기에 영향이 있으므로 남긴다.
    HASH_EXCLUDE: ClassVar[tuple[tuple[str, ...], ...]] = (("output", "root"), ("output", "count"))

    def hash_yaml(self) -> str:
        """``pipeline_hash`` 입력 — ``to_yaml()``에서 ``HASH_EXCLUDE`` 키를 뺀 덤프.
        ``--out``·``--count``만 바꾼 두 실행의 사이드카 해시가 같아진다(데브로그 07 결정)."""
        data = self.to_dict()
        for path in self.HASH_EXCLUDE:
            node: Any = data
            for key in path[:-1]:
                node = node.get(key, {}) if isinstance(node, dict) else {}
            if isinstance(node, dict):
                node.pop(path[-1], None)
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=None)

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
        """파일 로드 + CLI 오버라이드(``seed=``, ``count=``, ``out=``). 상대 입력 경로는 ``resolve_recipe_paths`` 규칙
        (cwd 우선, 없으면 레시피 파일 기준)."""
        return cls.load_with_notes(path, **overrides)[0]

    @classmethod
    def load_with_notes(cls, path: str | Path, **overrides: Any) -> tuple[Recipe, list[str]]:
        """``load`` + 레시피 파일 기준으로 다시 해석한 경로 목록(사용자에게 알릴 것)."""
        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"레시피 YAML의 최상위는 매핑이어야 합니다: {path}")
        data, notes = resolve_recipe_paths(data, p.parent)
        return cls.from_dict(apply_overrides(data, **overrides)), notes

    # --- 은행 대조 (런타임 검증) ---

    def effective_classes(self, bank: BankLike) -> list[str]:
        """실제로 뽑을 클래스 목록: (비-bank 소스면 ``[cls]``) → source.classes → class_ratio 키 → 은행 전체."""
        if self.bankless:
            return [self.pipeline.source.cls]
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
        """은행과 대조. 치명적이면 ``ValueError``, 아니면 경고 문자열 목록을 돌려준다. 비-bank 소스는 대조할 게 없다."""
        warnings: list[str] = []
        if self.bankless:
            if self.inputs.bank is not None:
                warnings.append(
                    f"source.method '{self.pipeline.source.method}'는 은행을 쓰지 않습니다 — inputs.bank는 무시됩니다"
                )
            return warnings
        bank_classes = set(bank.classes)
        counts = dict(bank.counts())
        classes = self.effective_classes(bank)
        missing = sorted(set(classes) - bank_classes)
        if missing:
            raise ValueError(f"은행에 없는 클래스입니다: {missing} (은행: {sorted(bank_classes)})")
        empty = [c for c in classes if counts.get(c, 0) == 0]
        if empty:
            raise ValueError(f"소스가 0개인 클래스입니다: {empty}")
        tags = self.pipeline.source.tags
        by_class = getattr(bank, "by_class", None)
        if tags.active and by_class is not None:
            # 태그 필터 후 풀 크기로 다시 센다 — 전부 0 이면 치명, 일부 0 이면 그 클래스만 건너뛴다(스테이지가 skip)
            counts = {c: sum(1 for s in by_class(c) if tags.accepts(s.tags)) for c in classes}
            if all(n == 0 for n in counts.values()):
                raise ValueError(f"태그 필터({tags.describe()}) 후 남는 소스가 없습니다")
            for c in classes:
                if counts[c] == 0:
                    warnings.append(
                        f"클래스 '{c}' 는 태그 필터({tags.describe()}) 후 소스 0개 — 그 클래스 결함은 건너뜁니다"
                    )
        unknown = sorted(set(self.pipeline.geometry.per_class) - bank_classes)
        if unknown:
            warnings.append(
                f"geometry.per_class 에 은행에 없는 클래스가 있습니다: {unknown} (은행: {sorted(bank_classes)}) — 무시됩니다"
            )
        threshold = self.pipeline.source.min_sources_warn
        for c in classes:
            n = counts.get(c, 0)
            if 0 < n < threshold:
                warnings.append(
                    f"클래스 '{c}' 소스가 {n}개 — {threshold}개 미만이면 모델이 특정 소스를 외울 수 있습니다"
                )
        if self.output.class_ratio is not None:
            unused = sorted(set(self.output.class_ratio) - set(classes))
            if unused:
                warnings.append(f"class_ratio에 있지만 쓰이지 않는 클래스: {unused}")
        return warnings


def set_method_in_dict(d: dict[str, Any], stage: str, method: str) -> None:
    """레시피 dict의 스테이지 블록을 ``{method}``(gtmask는 ``{policy}``, roi는 ``placement.roi``)로 갈아 끼운다.
    placement 를 바꿀 때는 하위 ``roi`` 블록을 유지한다(모든 placement method 가 공유하는 설정)."""
    key = "policy" if stage == "gtmask" else "method"
    pipe = d.setdefault("pipeline", {})
    if stage == "roi":
        pipe.setdefault("placement", {})["roi"] = {"method": method}
    elif stage == "placement":
        old = pipe.get("placement")
        block: dict[str, Any] = {key: method}
        if isinstance(old, dict) and "roi" in old:
            block["roi"] = old["roi"]
        pipe[stage] = block
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


# 레시피 안의 입력 경로 — (키 경로, 이 값이 있을 때만 조건) · output.root 는 아직 없는 폴더라 존재 검사가 무의미해 제외(cwd 기준)
_INPUT_PATH_KEYS: tuple[tuple[tuple[str, ...], tuple[str, ...] | None], ...] = (
    (("inputs", "bank"), None),
    (("inputs", "targets"), None),
    (("pipeline", "source", "texture_dir"), None),
    (("pipeline", "placement", "roi", "path"), None),
)


def resolve_recipe_paths(
    data: dict[str, Any], base: str | Path
) -> tuple[dict[str, Any], list[str]]:
    """상대 입력 경로를 **cwd 우선, 없으면 레시피 파일 기준**으로 해석한다(KNOWN-ISSUES #9 보완 방향). 반환 ``(새 dict, 노트)``.

    - cwd 에 있으면 그대로(하위 호환 — repo 루트에서 `recipes/x.yaml` 을 돌리는 기본 사용법은 변하지 않는다).
    - cwd 에 없고 ``base/<경로>`` 가 있으면 그것으로 바꾸고 노트 한 줄. 둘 다 없으면 그대로 두어 원래 오류가 난다.
    - 절대경로·null 은 건드리지 않는다. 검증 전 dict 단계라 pydantic 가 그대로 검증한다.
    """
    base = Path(base)
    out: dict[str, Any] = dict(data)
    notes: list[str] = []
    for keys, _cond in _INPUT_PATH_KEYS:
        node: Any = out
        parents: list[dict[str, Any]] = []
        ok = True
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node or not isinstance(node[k], dict):
                ok = False
                break
            parents.append(node)
            node = node[k]
        if not ok or not isinstance(node, dict):
            continue
        value = node.get(keys[-1])
        if not isinstance(value, str) or not value.strip():
            continue
        rel = Path(value)
        if rel.is_absolute() or rel.exists():
            continue
        candidate = base / rel
        if not candidate.exists():
            continue
        # 경로 사슬을 복사해 원본 dict 를 바꾸지 않는다
        cur = out
        for k in keys[:-1]:
            cur[k] = dict(cur[k])
            cur = cur[k]
        cur[keys[-1]] = candidate.as_posix()
        notes.append(f"{'.'.join(keys)}: {value} → {candidate.as_posix()} (레시피 파일 기준)")
    return out, notes


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
    roi: str | None = None,
    um_per_px: float | None = None,
    dent_classes: Sequence[str] | Mapping[str, Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """``recipe init``용 — 프리셋을 완전히 펼친 레시피 딕셔너리(사용자가 모든 손잡이를 본다).
    ``dent_classes`` 는 조명 의존 클래스 — ``geometry.per_class`` 에 ``DENT_OVERRIDE``(±15°, flip none)를 넣는다(프리셋은 그대로).
    매핑을 주면(클래스 → 오버라이드 dict) 그 값을 쓴다(예: 조명이 위/아래면 ``flip: horizontal`` — ``dent_override_for``).
    ``roi`` 는 프리셋의 ROI method 만 갈아 끼운다(예: dent-graft + annulus — 원형 부품의 찍힘). 그 method 의
    기본값으로 펼쳐지므로 반경 비율 등은 파일에서 조정. ``um_per_px`` 는 대상 피치(축척 정합)."""
    data: dict[str, Any] = {
        "version": 1,
        "name": name or preset,
        "seed": seed,
        "inputs": {"bank": bank, "targets": targets, "um_per_px": um_per_px},
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
    pipe = apply_preset(dict(data))["pipeline"]
    if str(pipe.get("source", {}).get("method", "bank")) in BANKLESS_SOURCES:
        data["inputs"]["bank"] = None  # 은행 없이 동작하는 프리셋
    if roi is not None:
        if roi not in ROI_METHODS:
            raise KeyError(f"ROI method 가 없습니다: {roi!r} (선택: {', '.join(ROI_METHODS)})")
        set_method_in_dict(data, "roi", roi)
    if dent_classes:
        geo = data["pipeline"].setdefault("geometry", {"method": "affine"})
        if isinstance(dent_classes, Mapping):
            overrides = {c: dict(o) for c, o in dent_classes.items()}
        else:
            overrides = {c: dict(DENT_OVERRIDE) for c in dent_classes}
        geo.setdefault("per_class", {}).update(overrides)
    return Recipe.from_dict(data).to_dict()


def dent_override_for(light_dir_deg: float | None) -> dict[str, Any]:
    """조명 의존 클래스의 오버라이드 — 회전 ±15° + 방향에 안전한 flip(위/아래 조명 → horizontal, 옆 → vertical, 모르면 none)."""
    from anograft.core.appearance import safe_flip

    return {"rotate": list(DENT_OVERRIDE["rotate"]), "flip": safe_flip(light_dir_deg)}


DENT_OVERRIDE: dict[str, Any] = {
    "rotate": [-15.0, 15.0],
    "flip": "none",
}  # 조명 의존 클래스의 기하(dent-graft 와 같음). 조명이 위/아래에서 오면 "horizontal" 도 안전(경고가 방향으로 판단)

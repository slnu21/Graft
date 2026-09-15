"""샘플 → 은행 → 레시피 한 번에(Qt 없음). GUI 상단 '샘플 데이터' 버튼과 ``anograft sample --bank`` 가 같은 함수를 부른다.

데이터가 하나도 없는 사용자가 클릭 한 번(또는 명령 한 줄)으로 은행·정상 목록·레시피까지 얻어 스튜디오에서 바로 미리보기.
``ring`` 은 원형 부품이라 레시피가 ``dent-graft`` + ``annulus`` ROI(찍힘·링 면), ``plate`` 는 ``poisson-graft``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from anograft.bank.importers.yolo import import_yolo
from anograft.core import recipe as R
from anograft.samples.yolo import SHAPES, generate

PRESET_FOR_SHAPE = {"plate": ("poisson-graft", None), "ring": ("dent-graft", "annulus")}


@dataclass(frozen=True)
class Quickstart:
    root: Path  # 샘플 세트 (images/·labels/·data.yaml)
    bank: Path  # 은행
    normals: Path  # 정상 목록 .txt (inputs.targets)
    recipe: Path  # 펼친 레시피 YAML
    preset: str
    n_sources: int
    low_confidence: int

    def line(self) -> str:
        return (
            f"샘플 {self.root.as_posix()} → 은행 {self.bank.as_posix()} (소스 {self.n_sources}, "
            f"저신뢰 {self.low_confidence}) · 정상 {self.normals.name} · 레시피 {self.recipe.name} ({self.preset})"
        )


def quickstart(
    root: str | Path,
    *,
    shape: str = "plate",
    seed: int = 7,
    n_normal: int = 12,
    n_defect: int = 10,
    size: tuple[int, int] = (640, 480),
    count: int = 24,
) -> Quickstart:
    """``<root>/`` 에 샘플 세트, ``<root>/bank`` 에 은행, ``<root>/normals.txt``, ``<root>/recipe.yaml``(출력 ``<root>/out``).
    이미 있으면 덮어쓴다(같은 시드 → 같은 바이트)."""
    if shape not in SHAPES:
        raise ValueError(f"shape 는 {'/'.join(SHAPES)} 중 하나 ({shape!r})")
    root = Path(root)
    generate(root, n_normal=n_normal, n_defect=n_defect, size=size, seed=seed, shape=shape)
    bank, normals = root / "bank", root / "normals.txt"
    res = import_yolo(
        root / "images", root / "labels", root / "data.yaml", bank, list_normals=normals
    )
    preset, roi = PRESET_FOR_SHAPE[shape]
    data = R.init_recipe_dict(
        preset,
        name=f"sample-{shape}",
        bank=bank.as_posix(),
        targets=normals.as_posix(),
        out=(root / "out").as_posix(),
        count=count,
        roi=roi,
    )
    src = data["pipeline"]["source"]
    src["min_sources_warn"] = 3  # 샘플은 클래스당 몇 장뿐 — 경고로 화면을 덮지 않게
    per_class = res.stats.per_class()
    if any(
        per_class.get(c, 0) == 0 for c in res.classes
    ):  # 작은 샘플은 클래스가 비기도 — 있는 클래스만 뽑는다
        src["classes"] = [c for c in res.classes if per_class.get(c, 0)]
    recipe = root / "recipe.yaml"
    recipe.write_text(
        f"# anograft quickstart --shape {shape}\n"
        + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=None),
        encoding="utf-8",
    )
    return Quickstart(root, bank, normals, recipe, preset, len(res.stats.added), res.low_confidence)

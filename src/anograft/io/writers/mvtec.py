"""MVTec 형식 writer (설계 §8.2b, v0.4) — 정본 위에 anomalib 이 그대로 읽는 ``MVTec AD`` 레이아웃을 **추가로** 쓴다.

::

    <root>/
      images/ · masks/ · meta/ · manifest.csv · recipe.resolved.yaml     # 정본(pairs) — 항상
      mvtec/<category>/
        train/good/n_<stem>.png            # 정상 중 train 분할 (복사)
        test/good/n_<stem>.png             # 정상 중 test 분할 — ``test_normal_ratio``, ``split_rng(seed)`` 로 결정적
        test/<class>/000000.png            # 합성 결함 (정본 images/ 사본)
        ground_truth/<class>/000000_mask.png

- **이미지당 단일 클래스**: 폴더는 클래스 하나뿐이라, 인스턴스가 여러 클래스면 **면적이 가장 큰 인스턴스의 클래스**로 두고
  사이드카 ``writer.mixed = true`` + 경고. 클래스가 섞이지 않게 하려면 ``source.single_class_per_image: true``(첫 결함이 뽑은
  클래스를 그 이미지의 나머지가 따름 — 설계의 '이미지당 1회 추첨', v0.8.x) 또는 ``defects_per_image: [1, 1]``.
- 정상 분할은 ``write_normal`` 호출 순서(= ``prep.targets`` 정렬)와 ``seed`` 로만 정해진다 — 워커 수와 무관.
- anomalib: ``MVTecAD(root="<root>/mvtec", category="<category>")`` 또는 ``Folder`` 데이터모듈. ``include_normals: false``
  면 train/good 이 비어 학습이 안 되므로 경고.
- 정본 PNG 를 다시 인코딩하지 않고 복사한다(바이트 동일).
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from anograft.core.recipe import MvtecWriterConfig, Recipe
from anograft.core.seeds import split_rng
from anograft.core.types import GraftResult, Instance
from anograft.io.writers.base import WriterSummary
from anograft.io.writers.pairs import PairsWriter, index_name

GOOD = "good"


def image_class(instances: Sequence[Instance]) -> tuple[str | None, bool]:
    """``(대표 클래스, 섞였는가)`` — 면적 최대 인스턴스의 클래스. 인스턴스가 없으면 ``(None, False)``."""
    if not instances:
        return None, False
    best = max(instances, key=lambda i: (i.area_px, -i.defect_index))
    mixed = len({i.cls for i in instances}) > 1
    return best.cls, mixed


class MvtecWriter(PairsWriter):
    format: ClassVar[str] = "mvtec"

    def __init__(self, cfg: MvtecWriterConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MvtecWriterConfig()
        self._split_rng: np.random.Generator | None = None
        self._pending: dict[int, dict[str, Any]] = {}
        self._pending_normal: str = "train"
        self.n_train_good = 0
        self.n_test_good = 0

    # ------------------------------------------------------------------

    def begin(
        self,
        root: Path,
        recipe: Recipe,
        pipeline_hash: str,
        classes: Sequence[str],
        *,
        bank_fingerprint: str = "",
    ) -> None:
        super().begin(root, recipe, pipeline_hash, classes, bank_fingerprint=bank_fingerprint)
        self._split_rng = split_rng(recipe.seed)
        self.n_train_good = self.n_test_good = 0
        for d in ("train/good", "test/good"):
            (self.category_dir / d).mkdir(parents=True, exist_ok=True)

    @property
    def category_dir(self) -> Path:
        assert self.root is not None
        return self.root / self.cfg.layout_dir / self.cfg.category

    # ------------------------------------------------------------------

    def write_normal(self, path: Path) -> None:
        assert self._split_rng is not None
        is_test = float(self._split_rng.random()) < self.cfg.test_normal_ratio
        self._pending_normal = "test" if is_test else "train"
        name = self._write_normal(path)
        dst = self.category_dir / self._pending_normal / GOOD / f"{name}.png"
        shutil.copyfile(self._paths(name)[0], dst)
        if is_test:
            self.n_test_good += 1
        else:
            self.n_train_good += 1

    def write_synthetic(self, result: GraftResult) -> None:
        if result.status != "ok":
            super().write_synthetic(result)
            return
        cls, mixed = image_class(result.instances)
        name = index_name(result.index)
        cls_dir = cls if cls is not None else "anomaly"
        img_rel = f"{self.cfg.layout_dir}/{self.cfg.category}/test/{cls_dir}/{name}.png"
        mask_rel = (
            f"{self.cfg.layout_dir}/{self.cfg.category}/ground_truth/{cls_dir}/{name}_mask.png"
        )
        entry: dict[str, Any] = {
            "format": self.format,
            "category": self.cfg.category,
            "split": "test",
            "class": cls_dir,
            "mixed": mixed,
            "image": img_rel,
            "mask": mask_rel,
        }
        self._pending[result.index] = entry
        if mixed:
            classes = sorted({i.cls for i in result.instances})
            w = (
                f"mvtec: 인덱스 {result.index} 에 클래스 {classes} 가 섞임 — 면적 최대 '{cls}' 폴더에 둠"
                " (source.single_class_per_image: true 면 이미지당 한 클래스)"
            )
            sidecar = dict(result.sidecar)
            sidecar["warnings"] = [*sidecar.get("warnings", []), w]
            result = replace(result, sidecar=sidecar, warnings=(*result.warnings, w))
            assert self.summary is not None
            self.summary.warnings.append(w)
        super().write_synthetic(result)  # 정본 + 사이드카(writer 항목) + manifest 행
        img_p, mask_p, _ = self._paths(name)
        assert self.root is not None
        dst_img, dst_mask = self.root / img_rel, self.root / mask_rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_mask.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(img_p, dst_img)
        shutil.copyfile(mask_p, dst_mask)
        self._pending.pop(result.index, None)

    # PairsWriter 훅
    def writer_entry(self, result: GraftResult) -> dict[str, Any]:
        return dict(self._pending.get(result.index, {"format": self.format}))

    def label_entry(self, result: GraftResult) -> str:
        return str(self._pending.get(result.index, {}).get("image", ""))

    def normal_label_entry(self, name: str) -> str:
        return f"{self.cfg.layout_dir}/{self.cfg.category}/{self._pending_normal}/{GOOD}/{name}.png"

    def finish(self) -> WriterSummary:
        summary = super().finish()
        summary.files["mvtec"] = f"{self.cfg.layout_dir}/{self.cfg.category}/"
        if self.n_train_good == 0:
            summary.warnings.append(
                "mvtec: train/good 이 비었습니다 — include_normals: true 로 정상 이미지를 함께 내보내야 anomalib 학습이 됩니다"
            )
        elif self.n_test_good == 0:
            summary.warnings.append(
                "mvtec: test/good 이 비었습니다 — test_normal_ratio 를 올리거나 정상 이미지를 더 주세요"
            )
        return summary

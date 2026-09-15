"""COCO writer (v0.7) — 정본 위에 ``annotations.json``(COCO instances 형식). Detectron2·mmdetection·YOLO(convert) 가 그대로 읽는다.

::

    <root>/
      images/000000.png · masks/ · meta/ · manifest.csv · recipe.resolved.yaml   # 정본(pairs)
      annotations.json             # {"info", "licenses": [], "images": [...], "annotations": [...], "categories": [...]}

- ``images``: 합성 이미지 + (``include_normals`` 면) 정상 이미지 — id = 등장 순서(1-based), ``file_name`` 은 ``images/`` 기준 상대.
  정상 이미지는 annotation 이 없다(COCO 의 배경 이미지 규약).
- ``categories``: 은행 ``classes`` 순서, **id = class_id + 1**(COCO 는 1-based; 사이드카·YOLO 의 0-based 와의 대응을 ``info`` 에 적는다).
- ``annotations``: 인스턴스 GT 마스크마다 하나 — ``segmentation`` = 외곽 윤곽 폴리곤 목록(``[[x1,y1,…], …]``, 픽셀 좌표 소수 1자리,
  ``approxPolyDP ε 0.5``, 점 < 3 조각 제외), ``bbox`` = ``[x, y, w, h]``(``Instance.bbox``), ``area`` = 마스크 픽셀 수(폴리곤 면적이
  아니라 GT 면적 — 학습 프레임워크는 대개 area 로 small/medium/large 만 가른다), ``iscrowd 0``. 구멍(내부 윤곽)은 폴리곤으로
  표현하지 않는다(RLE 를 쓰지 않는 이유: 순수 JSON, pycocotools 의존 0).
- 면적 0 인스턴스는 annotation 을 만들지 않고 사이드카 ``warnings`` 에 남긴다. 사이드카 ``writer`` = ``{format, image_id, annotation_ids}``,
  manifest ``label`` = ``annotations.json#<image_id>``.
- 결정성: 같은 결과 → 같은 JSON 바이트(정렬·소수 자리 고정, ``info.date`` 없음 — 재현 비교를 위해 시각을 넣지 않는다).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np

from anograft import __version__
from anograft.core.recipe import CocoWriterConfig, Recipe
from anograft.core.types import GraftResult, Instance
from anograft.io import imgio
from anograft.io.writers.base import WriterSummary
from anograft.io.writers.pairs import IMAGES_DIR, PairsWriter, index_name

ANNOTATIONS_FILE = "annotations.json"
POLY_EPS_PX = 0.5


def instance_polygons(mask: np.ndarray) -> list[list[float]]:
    """외곽 윤곽 → COCO ``segmentation`` 폴리곤 목록(픽셀, 소수 1자리). 점 3개 미만 조각은 버린다."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[float]] = []
    for c in contours:
        poly = cv2.approxPolyDP(c, POLY_EPS_PX, True).reshape(-1, 2)
        if len(poly) < 3:
            continue
        polys.append([round(float(v), 1) for v in poly.reshape(-1).tolist()])
    return polys


def annotations_for(
    instances: Sequence[Instance], image_id: int, next_id: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """인스턴스 → annotation dict 목록(+경고). id 는 ``next_id`` 부터 연속."""
    out: list[dict[str, Any]] = []
    warnings: list[str] = []
    aid = next_id
    for k, inst in enumerate(instances):
        if inst.area_px <= 0 or not np.any(inst.mask):
            warnings.append(f"coco: 인스턴스 {k}({inst.cls}) 면적 0 — annotation 없음")
            continue
        polys = instance_polygons(inst.mask)
        if not polys:
            warnings.append(f"coco: 인스턴스 {k}({inst.cls}) 폴리곤 점 < 3 — annotation 없음")
            continue
        x, y, w, h = inst.bbox
        out.append(
            {
                "id": aid,
                "image_id": image_id,
                "category_id": int(inst.class_id) + 1,
                "segmentation": polys,
                "area": int(inst.area_px),
                "bbox": [int(x), int(y), int(w), int(h)],
                "iscrowd": 0,
            }
        )
        aid += 1
    return out, warnings


class CocoWriter(PairsWriter):
    format: ClassVar[str] = "coco"

    def __init__(self, cfg: CocoWriterConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CocoWriterConfig()
        self.images: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self._pending: dict[
            int, tuple[int, list[int]]
        ] = {}  # result.index → (image_id, annotation ids)

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
        self.images.clear()
        self.annotations.clear()
        self._pending.clear()

    def _add_image(self, file_name: str, h: int, w: int) -> int:
        image_id = len(self.images) + 1
        self.images.append(
            {"id": image_id, "file_name": file_name, "width": int(w), "height": int(h)}
        )
        return image_id

    def write_normal(self, path: Path) -> None:
        name = self._write_normal(path)
        assert self.root is not None
        image, _gray = imgio.read_image(
            self.root / IMAGES_DIR / f"{name}.png"
        )  # 정본이 쓴 파일의 크기
        h, w = image.shape[:2]
        self._add_image(f"{name}.png", h, w)

    def write_synthetic(self, result: GraftResult) -> None:
        if result.status != "ok":
            super().write_synthetic(result)
            return
        name = index_name(result.index)
        h, w = result.image.shape[:2]
        image_id = self._add_image(f"{name}.png", h, w)
        anns, warns = annotations_for(result.instances, image_id, len(self.annotations) + 1)
        self.annotations.extend(anns)
        self._pending[result.index] = (image_id, [a["id"] for a in anns])
        if warns:
            sidecar = dict(result.sidecar)
            sidecar["warnings"] = [*sidecar.get("warnings", []), *warns]
            result = replace(result, sidecar=sidecar, warnings=(*result.warnings, *warns))
        super().write_synthetic(result)
        self._pending.pop(result.index, None)

    # PairsWriter 훅
    def writer_entry(self, result: GraftResult) -> dict[str, Any]:
        image_id, ids = self._pending.get(result.index, (0, []))
        return {"format": self.format, "image_id": image_id, "annotation_ids": ids}

    def label_entry(self, result: GraftResult) -> str:
        image_id, _ids = self._pending.get(result.index, (0, []))
        return f"{ANNOTATIONS_FILE}#{image_id}" if image_id else ""

    def normal_label_entry(self, name: str) -> str:
        return f"{ANNOTATIONS_FILE}#{len(self.images) + 1}"

    def document(self) -> dict[str, Any]:
        return {
            "info": {
                "description": self.cfg.description,
                "version": __version__,
                "contributor": "anograft",
                "pipeline_hash": self.pipeline_hash,
                "category_id": "class_id + 1 (사이드카·YOLO 는 0-based)",
                "image_dir": IMAGES_DIR,
            },
            "licenses": [],
            "images": list(self.images),
            "annotations": list(self.annotations),
            "categories": [
                {"id": i + 1, "name": c, "supercategory": self.cfg.supercategory}
                for i, c in enumerate(self.classes)
            ],
        }

    def finish(self) -> WriterSummary:
        assert self.root is not None
        (self.root / ANNOTATIONS_FILE).write_text(
            json.dumps(self.document(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        summary = super().finish()
        summary.files["annotations"] = ANNOTATIONS_FILE
        return summary

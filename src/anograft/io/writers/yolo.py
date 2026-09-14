"""YOLO writer (설계 §8.2) — 정본 위에 ``labels/*.txt`` + ``data.yaml``. **기존 YOLO 학습셋의 ``images/``·``labels/``에
그대로 복사해 넣을 수 있는 형태**가 목표(들어온 형식으로 되돌린다).

::

    <root>/
      images/000000.png · masks/ · meta/ · manifest.csv · recipe.resolved.yaml   # 정본(pairs)
      labels/000000.txt            # 결함 인스턴스마다 한 줄. 박스 `c cx cy w h` / seg `c x1 y1 … xn yn` (정규화, 소수 6자리)
      labels/n_<stem>.txt          # 정상 이미지는 빈 파일 (YOLO의 배경 이미지 규약)
      data.yaml                    # names: [은행 classes 순서], nc, path: ., train: images

- 박스 = **인스턴스 GT 마스크의 bbox**(``Instance.bbox`` = ``cv2.boundingRect``; 정책·팽창 적용 뒤라 소스 박스보다 조금 클 수 있다).
- ``seg``: ``cv2.findContours(RETR_EXTERNAL)`` → ``approxPolyDP(ε=0.5px)`` → 점 ≥ 3인 조각마다 한 줄(같은 클래스).
- 면적 0 인스턴스(방어)는 줄을 쓰지 않고 사이드카 ``warnings``에 남긴다.
- **왕복 불변식**: ``images/``+``labels/``를 ``bank import-yolo --mask-from rect``로 읽으면 박스가 사이드카 bbox와 일치(테스트).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import yaml

from anograft.core.recipe import Recipe, YoloWriterConfig
from anograft.core.types import GraftResult, Instance
from anograft.io.writers.base import WriterSummary
from anograft.io.writers.pairs import IMAGES_DIR, PairsWriter, index_name

LABELS_DIR = "labels"
DATA_YAML = "data.yaml"
POLY_EPS_PX = 0.5


def _fmt(v: float) -> str:
    return f"{v:.6f}"


def box_line(class_id: int, bbox: tuple[int, int, int, int], shape: tuple[int, ...]) -> str:
    """``c cx cy w h`` 정규화 — bbox는 (x, y, w, h) 픽셀. 픽셀 중심 규약: 박스 [x, x+w) → 중심 x + w/2."""
    h, w = shape[:2]
    x, y, bw, bh = bbox
    return " ".join(
        [
            str(class_id),
            _fmt((x + bw / 2.0) / w),
            _fmt((y + bh / 2.0) / h),
            _fmt(bw / w),
            _fmt(bh / h),
        ]
    )


def polygon_lines(class_id: int, mask: np.ndarray) -> list[str]:
    """외곽 윤곽마다 ``c x1 y1 … xn yn``(정규화). 점 3개 미만 조각은 버린다."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []
    for c in contours:
        poly = cv2.approxPolyDP(c, POLY_EPS_PX, True).reshape(-1, 2)
        if len(poly) < 3:
            continue
        coords = [f"{_fmt(px / w)} {_fmt(py / h)}" for px, py in poly.tolist()]
        lines.append(f"{class_id} " + " ".join(coords))
    return lines


def label_lines(
    instances: Sequence[Instance], shape: tuple[int, ...], *, seg: bool
) -> tuple[list[str], list[str]]:
    """반환 ``(줄 목록, 경고)``."""
    lines: list[str] = []
    warnings: list[str] = []
    for k, inst in enumerate(instances):
        if inst.area_px <= 0 or not np.any(inst.mask):
            warnings.append(f"yolo: 인스턴스 {k}({inst.cls}) 면적 0 — 라벨 줄 없음")
            continue
        if seg:
            polys = polygon_lines(inst.class_id, inst.mask)
            if not polys:
                warnings.append(f"yolo: 인스턴스 {k}({inst.cls}) 폴리곤 점 < 3 — 라벨 줄 없음")
            lines += polys
        else:
            lines.append(box_line(inst.class_id, inst.bbox, shape))
    return lines, warnings


class YoloWriter(PairsWriter):
    format: ClassVar[str] = "yolo"

    def __init__(self, cfg: YoloWriterConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or YoloWriterConfig()
        self._pending: dict[int, tuple[list[str], str]] = {}

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
        (self.root / LABELS_DIR).mkdir(parents=True, exist_ok=True)  # type: ignore[operator]

    def _label_path(self, name: str) -> Path:
        assert self.root is not None
        return self.root / LABELS_DIR / f"{name}.txt"

    def write_normal(self, path: Path) -> None:
        name = self._write_normal(path)
        self._label_path(name).write_text("", encoding="utf-8")  # 배경 이미지 = 빈 라벨

    def write_synthetic(self, result: GraftResult) -> None:
        if result.status != "ok":
            super().write_synthetic(result)
            return
        lines, warns = label_lines(result.instances, result.image.shape, seg=self.cfg.seg)
        name = index_name(result.index)
        self._pending[result.index] = (lines, f"{LABELS_DIR}/{name}.txt")
        if warns:
            sidecar = dict(result.sidecar)
            sidecar["warnings"] = [*sidecar.get("warnings", []), *warns]
            result = replace(result, sidecar=sidecar, warnings=(*result.warnings, *warns))
        super().write_synthetic(result)  # 정본 + 사이드카(writer 항목·경고 포함) + manifest 행
        self._label_path(name).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        self._pending.pop(result.index, None)

    # PairsWriter 훅 — write_synthetic 안에서 불린다
    def writer_entry(self, result: GraftResult) -> dict[str, Any]:
        lines, rel = self._pending.get(result.index, ([], ""))
        return {"format": self.format, "seg": self.cfg.seg, "label": rel, "n_lines": len(lines)}

    def label_entry(self, result: GraftResult) -> str:
        return self._pending.get(result.index, ([], ""))[1]

    def normal_label_entry(self, name: str) -> str:
        return f"{LABELS_DIR}/{name}.txt"

    def finish(self) -> WriterSummary:
        assert self.root is not None
        data = {
            "path": ".",
            "train": IMAGES_DIR,
            "val": IMAGES_DIR,
            "nc": len(self.classes),
            "names": list(self.classes),
        }
        (self.root / DATA_YAML).write_text(
            "# anograft — names 는 은행 classes 순서(= 사이드카 class_id). 기존 학습셋과 합칠 때 id 대응의 기준\n"
            + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=None),
            encoding="utf-8",
        )
        summary = super().finish()
        summary.files["data.yaml"] = DATA_YAML
        summary.files["labels"] = LABELS_DIR + "/"
        return summary

"""정본 레이아웃 writer (설계 §8.1) — 모든 writer의 공통 부분. 다른 writer는 이 위에 형식 파일을 덧붙인다.

::

    <root>/
      images/000000.png …      # 합성 이미지 (+ include_normals면 정상 이미지 n_<stem>.png)
      masks/000000.png         # 전체 GT 마스크 0/255 (정상 이미지는 전부 0)
      meta/000000.json         # 사이드카 §8.3 (정상 이미지는 {"normal": true, target: …})
      manifest.csv             # §8.4
      recipe.resolved.yaml     # 헤더 주석에 pipeline_hash · bank_fingerprint

- 번호는 전체 인덱스 ``i`` 기준 6자리. skipped 인덱스는 파일이 없고 manifest에만 행이 남는다.
- 정상 이미지 stem이 겹치면(하위 폴더) ``n_<stem>-2`` 식으로 피한다.
- 파일 기록은 ``imgio``(imencode + write_bytes) — 한글 경로.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from anograft import __version__
from anograft.core.recipe import Recipe
from anograft.core.types import GraftResult
from anograft.io import imgio
from anograft.io.manifest import MANIFEST_FILE, row_from_sidecar, write_manifest
from anograft.io.writers.base import WriterSummary, sidecar_with_header

RESOLVED_RECIPE_FILE = "recipe.resolved.yaml"
IMAGES_DIR, MASKS_DIR, META_DIR = "images", "masks", "meta"


def index_name(index: int) -> str:
    return f"{index:06d}"


class PairsWriter:
    format: ClassVar[str] = "pairs"

    def __init__(self) -> None:
        self.root: Path | None = None
        self.recipe: Recipe | None = None
        self.pipeline_hash = ""
        self.classes: list[str] = []
        self.rows: list[dict[str, Any]] = []
        self.summary: WriterSummary | None = None
        self._normal_names: set[str] = set()

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
        self.root = Path(root)
        self.recipe = recipe
        self.pipeline_hash = pipeline_hash
        self.classes = list(classes)
        self.rows = []
        self.summary = WriterSummary(root=self.root)
        for d in (IMAGES_DIR, MASKS_DIR, META_DIR):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        header = (
            f"# anograft {__version__} pipeline_hash={pipeline_hash} bank_fingerprint={bank_fingerprint}\n"
            "# 해시 입력은 아래 본문(이 주석 제외) + 패키지 버전 + 은행 지문\n"
        )
        (self.root / RESOLVED_RECIPE_FILE).write_text(header + recipe.to_yaml(), encoding="utf-8")
        self.summary.files["recipe"] = RESOLVED_RECIPE_FILE

    def _paths(self, name: str) -> tuple[Path, Path, Path]:
        assert self.root is not None
        return (
            self.root / IMAGES_DIR / f"{name}.png",
            self.root / MASKS_DIR / f"{name}.png",
            self.root / META_DIR / f"{name}.json",
        )

    def _rel(self, p: Path) -> str:
        assert self.root is not None
        return p.relative_to(self.root).as_posix()

    def _normal_name(self, stem: str) -> str:
        name, n = f"n_{stem}", 1
        while name in self._normal_names:
            n += 1
            name = f"n_{stem}-{n}"
        self._normal_names.add(name)
        return name

    # ------------------------------------------------------------------

    def write_normal(self, path: Path) -> None:
        """정상 이미지 = 이미지(복사/하드링크) + 빈 마스크 + 최소 사이드카. 세 쌍 불변식은 정상 이미지에도 적용."""
        assert self.root is not None and self.recipe is not None and self.summary is not None
        name = self._normal_name(Path(path).stem)
        img_p, mask_p, meta_p = self._paths(name)
        src = Path(path)
        mode = self.recipe.output.copy_mode
        linked = False
        if mode == "hardlink" and src.suffix.lower() == ".png":
            try:
                if img_p.exists():
                    img_p.unlink()
                os.link(src, img_p)
                linked = True
            except OSError as e:  # 다른 드라이브·권한 → 복사로 폴백
                self.summary.warnings.append(f"hardlink 실패 {src.name}: {e} — 복사")
        if not linked:
            if src.suffix.lower() == ".png":
                shutil.copyfile(src, img_p)
            else:
                image, gray = imgio.read_image(src)
                imgio.write_image(img_p, image[:, :, 0] if gray else image)
        image, gray = imgio.read_image(img_p)
        h, w = image.shape[:2]
        imgio.write_image(mask_p, np.zeros((h, w), dtype=np.uint8))
        sidecar = sidecar_with_header(
            {
                "normal": True,
                "target": {"file": src.as_posix(), "shape": [h, w, 1 if gray else 3], "gray": gray},
                "warnings": [],
            },
            version=__version__,
            pipeline_hash=self.pipeline_hash,
            writer={"format": self.format, "copy_mode": mode, "hardlink": linked},
        )
        meta_p.write_text(json.dumps(sidecar, ensure_ascii=False, indent=1), encoding="utf-8")
        self.rows.append(
            row_from_sidecar(
                "",
                "normal",
                sidecar,
                image=self._rel(img_p),
                mask=self._rel(mask_p),
                meta=self._rel(meta_p),
            )
        )
        self.summary.n_normals += 1

    def write_synthetic(self, result: GraftResult) -> None:
        assert self.root is not None and self.summary is not None
        if result.status != "ok":
            self.rows.append(
                row_from_sidecar(result.index, "skipped", result.sidecar, reason=result.reason)
            )
            self.summary.n_skipped += 1
            return
        name = index_name(result.index)
        img_p, mask_p, meta_p = self._paths(name)
        imgio.write_image(img_p, result.image)
        imgio.write_image(mask_p, result.gt_mask)
        sidecar = sidecar_with_header(
            result.sidecar,
            version=__version__,
            pipeline_hash=self.pipeline_hash,
            writer=self.writer_entry(result),
        )
        meta_p.write_text(json.dumps(sidecar, ensure_ascii=False, indent=1), encoding="utf-8")
        row = row_from_sidecar(
            result.index,
            "ok",
            sidecar,
            image=self._rel(img_p),
            mask=self._rel(mask_p),
            meta=self._rel(meta_p),
            label=self.label_entry(result),
        )
        self.rows.append(row)
        self.summary.n_ok += 1
        self.summary.n_fallback += int(row["fallback"])
        for inst in result.instances:
            self.summary.per_class[inst.cls] = self.summary.per_class.get(inst.cls, 0) + 1

    # 형식 writer가 덮어쓰는 훅 — 정본 writer는 형식 파일이 없다
    def writer_entry(self, result: GraftResult) -> dict[str, Any]:
        return {"format": self.format}

    def label_entry(self, result: GraftResult) -> str:
        return ""

    def finish(self) -> WriterSummary:
        assert self.root is not None and self.summary is not None
        write_manifest(self.root / MANIFEST_FILE, self.rows)
        self.summary.files["manifest"] = MANIFEST_FILE
        return self.summary

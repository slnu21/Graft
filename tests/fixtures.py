"""런타임 합성 픽스처 (설계 §11) — MVTec 다운로드 없이 스테이지·파이프라인을 시험한다.

- ``disk_target(size, gray)``: 어두운 배경 + 밝은 원판(물체는 중앙, 배경은 테두리 — Otsu ``auto``의 전제).
- ``line_defect(length, width)``: 밝은 선 결함 크롭 + 마스크 (``DefectSource``).
- ``context(...)``: 스테이지 단위 테스트용 Context 생성.
- ``memory_bank(...)`` + ``pipeline_deps(recipe, bank)``: 디스크 없는 실물 ``Bank`` — 통합 테스트는 실제 ``BankSource``로 돈다.
- ``blob_image`` · ``fake_yolo_dataset(root)``: 어두운 판 위 밝은 얼룩 + YOLO 박스/폴리곤 라벨 — 임포터·CLI e2e 픽스처.
- ``blob_mask`` · ``fake_pairs_dataset(root)``: 이미지 + 마스크 PNG 쌍(suffix/동일 stem/CSV 세 매칭) — import-pairs 픽스처.
- ``fake_mvtec_tree(root)``: MVTec AD 카테고리 폴더 흉내(test/<defect> 2종 + ground_truth + train/good 4장 + test/good).
- ``fake_visa_tree(root)``: VisA 카테고리 폴더 흉내(Data/Images/{Normal,Anomaly} JPG + Data/Masks/Anomaly 0/1 라벨맵 + image_anno.csv).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from anograft.bank import Bank
from anograft.core.recipe import Recipe
from anograft.core.types import Context, DefectSource, PlacedDefect, TargetImage
from anograft.io import imgio


def load_tool(name: str) -> types.ModuleType:
    """``tools/<name>.py`` 를 모듈로 로드(패키지 밖 스크립트). ``sys.modules`` 에 먼저 등록해야 ``from __future__ import
    annotations`` 아래의 dataclass 가 애노테이션을 풀 수 있다."""
    path = Path(__file__).resolve().parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def disk_image(size: int = 128, *, invert: bool = False, radius: int | None = None) -> np.ndarray:
    """``size×size×3`` — 배경 40, 중앙 원판 200. ``invert``면 밝은 배경·어두운 원판."""
    img = np.full((size, size, 3), 40, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius or size // 3, (200, 200, 200), -1)
    return (255 - img) if invert else img


def disk_target(
    size: int = 128,
    gray: bool = False,
    *,
    invert: bool = False,
    um_per_px: float | None = None,
    name: str = "target.png",
) -> TargetImage:
    return TargetImage(
        path=Path(name), image=disk_image(size, invert=invert), gray=gray, um_per_px=um_per_px
    )


def line_defect(
    length: int = 16,
    width: int = 3,
    *,
    margin: int = 4,
    cls: str = "scratch",
    um_per_px: float | None = None,
) -> DefectSource:
    """가로 밝은 선 + 마스크. 크롭 = 선 bbox + ``margin``(은행 포맷과 같은 형태)."""
    h, w = width + 2 * margin, length + 2 * margin
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    img[margin : margin + width, margin : margin + length] = 230
    mask[margin : margin + width, margin : margin + length] = 255
    return DefectSource(
        id=f"{cls}/000", cls=cls, image=img, mask=mask, um_per_px=um_per_px, origin="fixture"
    )


def context(
    target: TargetImage | None = None,
    *,
    seed: int = 0,
    source: DefectSource | None = None,
    roi: np.ndarray | None = None,
    patch: np.ndarray | None = None,
    patch_mask: np.ndarray | None = None,
    placed: tuple[PlacedDefect, ...] = (),
) -> Context:
    t = target or disk_target()
    return Context(
        rng=np.random.default_rng(seed),
        target=t,
        composite=t.image,
        roi=roi,
        source=source,
        patch=patch,
        patch_mask=patch_mask,
        placed=placed,
    )


# ---------------------------------------------------------------------------
# 실물 은행 (디스크 없음) — 파이프라인 통합 테스트
# ---------------------------------------------------------------------------


def memory_bank(sources: Iterable[DefectSource] | None = None) -> Bank:
    """기본: scratch(선 18×4) + dent(선 10×6). ``Bank.from_sources`` — classes 순서 = id."""
    srcs = (
        list(sources)
        if sources is not None
        else [line_defect(18, 4), line_defect(10, 6, cls="dent")]
    )
    # 같은 id("<cls>/000")가 겹치지 않게 클래스별로 번호를 매긴다
    fixed: list[DefectSource] = []
    seen: dict[str, int] = {}
    for s in srcs:
        n = seen.get(s.cls, 0)
        seen[s.cls] = n + 1
        fixed.append(
            DefectSource(
                f"{s.cls}/{n:03d}",
                s.cls,
                s.image,
                s.mask,
                s.um_per_px,
                s.tags,
                s.origin,
                s.mask_origin,
            )
        )
    return Bank.from_sources(fixed, classes=sorted(seen))


def pipeline_deps(recipe: Recipe, bank: Bank) -> dict[str, Any]:
    """``runner.build_deps``와 같은 모양(core 테스트가 io를 끌어오지 않게 여기서 만든다).
    비-bank 소스(self-cut·perlin)면 class_ids = ``{cls: 0}`` — runner.prepared_classes 와 같은 규칙."""
    classes = recipe.effective_classes(bank) if recipe.bankless else list(bank.classes)
    return {
        "bank": bank,
        "class_ids": {c: i for i, c in enumerate(classes)},
        "class_probs": recipe.class_probabilities(bank),
        "read_mask": imgio.read_mask,
    }


# ---------------------------------------------------------------------------
# YOLO 데이터셋 픽스처 — 임포터·CLI e2e
# ---------------------------------------------------------------------------


def blob_image(
    size: int = 96, blobs: Iterable[tuple[int, int, int]] = (), *, gray: bool = False
) -> np.ndarray:
    """어두운 판(60) 위 밝은 얼룩(220) — ``(cx, cy, r)``. 배경 테두리 20px은 더 어둡게(20)해 Otsu ROI가 판을 고르게 한다."""
    img = np.full((size, size, 3), 20, dtype=np.uint8)
    cv2.rectangle(img, (14, 14), (size - 15, size - 15), (60, 60, 60), -1)
    for cx, cy, r in blobs:
        cv2.circle(img, (cx, cy), r, (220, 220, 220), -1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if gray else img


def yolo_box(cx: int, cy: int, r: int, size: int, slack: int = 1) -> str:
    x0, y0, x1, y1 = cx - r - slack, cy - r - slack, cx + r + slack, cy + r + slack
    return f"{(x0 + x1) / 2 / size:.6f} {(y0 + y1) / 2 / size:.6f} {(x1 - x0) / size:.6f} {(y1 - y0) / size:.6f}"


def fake_yolo_dataset(
    root: Path, *, size: int = 96, names: tuple[str, ...] = ("spot", "crack")
) -> dict[str, Any]:
    """``images/``·``labels/``·``data.yaml``. 결함 3장(박스 4개 + 폴리곤 1개), 정상 2장(빈 파일 1 · 파일 없음 1), 하위 폴더 1장.

    반환: ``{"images": Path, "labels": Path, "names": Path, "normals": [Path], "boxes": {stem: [(cid, cx, cy, r)]}}``.
    """
    images, labels = root / "images", root / "labels"
    (images / "sub").mkdir(parents=True)
    (labels / "sub").mkdir(parents=True)
    boxes: dict[str, list[tuple[int, int, int, int]]] = {
        "d0": [(0, 40, 40, 8)],
        "d1": [(0, 30, 60, 6), (1, 66, 30, 7)],
        "sub/d2": [(1, 48, 48, 9)],
    }
    for stem, bl in boxes.items():
        imgio.write_image(
            images / f"{stem}.png", blob_image(size, [(cx, cy, r) for _c, cx, cy, r in bl])
        )
        lines = [f"{c} {yolo_box(cx, cy, r, size)}" for c, cx, cy, r in bl]
        if stem == "d1":  # 폴리곤 한 줄 추가: 삼각형 (class 1)
            tri = [(20, 20), (34, 20), (27, 34)]
            lines.append("1 " + " ".join(f"{x / size:.6f} {y / size:.6f}" for x, y in tri))
        (labels / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    normals: list[Path] = []
    imgio.write_image(images / "n0.png", blob_image(size))
    (labels / "n0.txt").write_text("", encoding="utf-8")  # 빈 라벨 파일
    normals.append(images / "n0.png")
    imgio.write_image(images / "n1.png", blob_image(size, gray=True)[:, :])  # 라벨 파일 없음 + 흑백
    normals.append(images / "n1.png")
    names_path = root / "data.yaml"
    names_path.write_text(yaml.safe_dump({"names": list(names)}), encoding="utf-8")
    return {
        "images": images,
        "labels": labels,
        "names": names_path,
        "normals": normals,
        "boxes": boxes,
    }


# ---------------------------------------------------------------------------
# 마스크 PNG 쌍 · MVTec AD 트리 픽스처 — import-pairs · datasets
# ---------------------------------------------------------------------------


def blob_mask(size: int, blobs: Iterable[tuple[int, int, int]]) -> np.ndarray:
    """``(cx, cy, r)`` 원판들의 0/255 마스크 — ``blob_image``와 같은 좌표 규약."""
    m = np.zeros((size, size), dtype=np.uint8)
    for cx, cy, r in blobs:
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def fake_pairs_dataset(root: Path, *, size: int = 96) -> dict[str, Any]:
    """``images/<class>/x.png`` + ``masks/<class>/x_mask.png``(suffix) · ``y.png``(동일 stem) · 마스크 없는 ``z.png`` +
    ``pairs.csv``(image,mask,class 두 줄). 반환: ``{"images", "masks", "csv", "blobs": {stem: [(cx,cy,r)]}, "n_missing": 1}``.
    """
    images, masks = root / "images", root / "masks"
    blobs: dict[str, list[tuple[int, int, int]]] = {
        "spot/a": [(40, 40, 8)],
        "spot/b": [(30, 60, 6), (66, 30, 7)],  # 성분 2개 → 소스 2개
        "crack/c": [(48, 48, 9)],
    }
    for stem, bl in blobs.items():
        (images / Path(stem).parent).mkdir(parents=True, exist_ok=True)
        (masks / Path(stem).parent).mkdir(parents=True, exist_ok=True)
        imgio.write_image(images / f"{stem}.png", blob_image(size, bl))
        mask_name = f"{stem}_mask.png" if stem != "spot/b" else f"{stem}.png"  # b 는 동일 stem 매칭
        imgio.write_image(masks / mask_name, blob_mask(size, bl))
    imgio.write_image(images / "crack" / "z.png", blob_image(size, [(20, 20, 5)]))  # 마스크 없음
    csv_path = root / "pairs.csv"
    csv_path.write_text(
        "\n".join(
            [
                "image,mask,class",
                "images/spot/a.png,masks/spot/a_mask.png,spot",
                "images/crack/c.png,masks/crack/c_mask.png,crack",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"images": images, "masks": masks, "csv": csv_path, "blobs": blobs, "n_missing": 1}


MVTEC_DEFECTS: dict[str, list[tuple[int, int, int]]] = {
    "scratch": [(40, 40, 8)],
    "hole": [(56, 30, 6), (30, 60, 5)],
}


def fake_mvtec_tree(root: Path, *, category: str = "metal_nut", size: int = 96) -> Path:
    """``<root>/<category>/{train/good ×4, test/good ×2, test/<defect>/<idx>.png ×2, ground_truth/<defect>/<idx>_mask.png}``.
    ``test/hole/001.png``은 마스크가 **없다**(어댑터가 건너뛰고 경고). 반환: 카테고리 폴더."""
    cat = root / category
    for i in range(4):
        imgio.write_image(cat / "train" / "good" / f"{i:03d}.png", blob_image(size))
    for i in range(2):
        imgio.write_image(cat / "test" / "good" / f"{i:03d}.png", blob_image(size))
    for defect, bl in MVTEC_DEFECTS.items():
        for i in range(2):
            shifted = [(cx + i * 3, cy, r) for cx, cy, r in bl]
            imgio.write_image(cat / "test" / defect / f"{i:03d}.png", blob_image(size, shifted))
            if defect == "hole" and i == 1:
                continue  # 마스크 누락 케이스
            imgio.write_image(
                cat / "ground_truth" / defect / f"{i:03d}_mask.png", blob_mask(size, shifted)
            )
    return cat


def fake_visa_tree(
    root: Path, *, category: str = "candle", size: int = 96, anno: bool = True
) -> Path:
    """``<root>/<category>/Data/Images/{Normal ×3, Anomaly ×3}.JPG + Data/Masks/Anomaly/<idx>.png + image_anno.csv``.
    ``Anomaly/002`` 는 마스크가 **없다**(건너뛰고 경고). 마스크는 **0/1 라벨맵**(VisA 사본 케이스 — 어댑터가 threshold 0 으로 읽는다).
    ``anno=True`` 면 CSV 에 정상 3 + 결함 3 행(마지막은 존재하지 않는 파일 → 경고). 반환: 카테고리 폴더."""
    cat = root / category
    blobs = MVTEC_DEFECTS["scratch"]
    rows: list[tuple[str, str, str]] = []
    for i in range(3):
        name = f"{i:04d}.JPG"
        imgio.write_image(cat / "Data" / "Images" / "Normal" / name, blob_image(size))
        rows.append((f"{category}/Data/Images/Normal/{name}", "normal", ""))
    for i in range(3):
        name = f"{i:03d}.JPG"
        shifted = [(cx + i * 3, cy, r) for cx, cy, r in blobs]
        imgio.write_image(cat / "Data" / "Images" / "Anomaly" / name, blob_image(size, shifted))
        rows.append(
            (
                f"{category}/Data/Images/Anomaly/{name}",
                "anomaly",
                f"{category}/Data/Masks/Anomaly/{i:03d}.png",
            )
        )
        if i == 2:
            continue  # 마스크 누락 케이스
        m = (blob_mask(size, shifted) > 0).astype(np.uint8)  # 0/1 라벨맵
        imgio.write_image(cat / "Data" / "Masks" / "Anomaly" / f"{i:03d}.png", m)
    if anno:
        rows.append(
            (
                f"{category}/Data/Images/Anomaly/999.JPG",
                "anomaly",
                f"{category}/Data/Masks/Anomaly/999.png",
            )
        )
        lines = ["image,label,mask"] + [",".join(r) for r in rows]
        (cat / "image_anno.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cat

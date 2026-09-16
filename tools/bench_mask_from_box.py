"""박스→마스크 추정 벤치 — GT 마스크가 있는 데이터(MVTec AD 카테고리 · pairs.csv)에서 GT 로 박스를 만들어
``mask_from_box`` 4방법의 IoU 와 ``mask_confidence`` 의 실패 검출력을 잰다(KNOWN-ISSUES #3 · 결정 대기 grabcut/otsu · 저신뢰 임계).

    python tools/bench_mask_from_box.py samples/mvtec/metal_nut samples/mvtec/screw samples/magnetic-tile/pairs.csv --out out/bench/mask-bench.md

- 입력: ``<root>``(``ground_truth/`` 가 있으면 MVTec 레이아웃) 또는 ``pairs.csv``(image,mask,class — CSV 파일 기준 경로).
- GT 성분(연결 성분, ``--min-area``) 하나 = 인스턴스 하나. 박스 = 성분 bbox 를 ``--box-pad``(기본 10 %) 만큼 느슨하게(사람 박스 흉내).
- 방법별 IoU(GT 성분 vs 추정, 이미지 전체) · 사슬(``grabcut`` 시작, 폴백 포함)의 실제 방법 · confidence. **실패 = 사슬 IoU < 0.3**.
- 출력: 데이터셋×클래스 표(markdown) + JSON. **결정은 하지 않는다** — 표만.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anograft.bank.mask_from_box import LOW_CONFIDENCE, mask_confidence, mask_from_box  # noqa: E402
from anograft.core.seeds import stable_seed  # noqa: E402
from anograft.io import imgio  # noqa: E402

METHODS = ("grabcut", "otsu", "ellipse", "rect")
FAIL_IOU = 0.3


@dataclass(frozen=True)
class Record:
    dataset: str
    cls: str
    image: str
    k: int
    box: tuple[int, int, int, int]
    iou: dict[str, float]  # 방법 → IoU
    chain_used: str
    chain_iou: float
    confidence: float
    flags: tuple[str, ...]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """두 0/255(또는 bool) 마스크의 IoU. 둘 다 비면 1.0."""
    aa, bb = a.astype(bool), b.astype(bool)
    union = int(np.count_nonzero(aa | bb))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(aa & bb)) / union


def pad_box(
    box: tuple[int, int, int, int], pad: float, shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """bbox(x, y, w, h) 를 각 변 ``pad`` 비율(변 길이 기준, 최소 1 px)만큼 넓혀 이미지 안으로 자른다."""
    x, y, w, h = box
    dx, dy = max(1, round(w * pad)), max(1, round(h * pad))
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(shape[1], x + w + dx), min(shape[0], y + h + dy)
    return x0, y0, x1 - x0, y1 - y0


def components(
    mask: np.ndarray, min_area: int
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """GT 마스크의 연결 성분(면적 ≥ min_area) → [(성분 마스크 0/255, bbox)] — 면적 큰 순."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        out.append(((labels == i).astype(np.uint8) * 255, (x, y, w, h), area))
    out.sort(key=lambda t: -t[2])
    return [(m, b) for m, b, _ in out]


def bench_instance(
    image: np.ndarray, gt: np.ndarray, box: tuple[int, int, int, int], *, seed: int, margin: int = 6
) -> tuple[dict[str, float], str, float, float, tuple[str, ...]]:
    ious: dict[str, float] = {}
    for m in METHODS:
        est, _ = mask_from_box(image, box, m, margin, seed=seed)
        ious[m] = round(iou(est, gt), 4)
    chain, used = mask_from_box(image, box, "grabcut", margin, seed=seed)
    conf = mask_confidence(image, chain, box, margin=margin)
    return ious, used, round(iou(chain, gt), 4), round(conf.score, 3), conf.flags


def iter_mvtec(root: Path):
    """MVTec 레이아웃 → (dataset, class, image_path, mask_path)."""
    for cls_dir in sorted((root / "ground_truth").iterdir()):
        if not cls_dir.is_dir():
            continue
        for mp in sorted(cls_dir.glob("*_mask.png")):
            ip = root / "test" / cls_dir.name / (mp.name[: -len("_mask.png")] + ".png")
            if ip.is_file():
                yield root.name, cls_dir.name, ip, mp


def iter_pairs_csv(path: Path):
    base = path.parent
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            yield base.name, row["class"], base / row["image"], base / row["mask"]


def run(inputs: list[Path], *, box_pad: float, min_area: int, limit: int | None) -> list[Record]:
    records: list[Record] = []
    for src in inputs:
        items = iter_pairs_csv(src) if src.suffix.lower() == ".csv" else iter_mvtec(src)
        per_class: Counter[str] = Counter()
        for ds, cls, ip, mp in items:
            if limit is not None and per_class[cls] >= limit:
                continue
            per_class[cls] += 1
            image, _ = imgio.read_image(ip)
            gt = imgio.read_mask(mp)
            for k, (comp, bbox) in enumerate(components(gt, min_area)):
                box = pad_box(bbox, box_pad, image.shape)
                seed = stable_seed(f"{ds}/{cls}/{ip.stem}-{k}")
                ious, used, ciou, conf, flags = bench_instance(image, comp, box, seed=seed)
                records.append(Record(ds, cls, ip.name, k, box, ious, used, ciou, conf, flags))
    return records


def summarize(records: list[Record]) -> list[dict]:
    """(dataset, class) 별 + 전체 요약 행: n · 방법별 IoU 중앙값 · 사슬 방법 분포 · 사슬 IoU 중앙값 · 실패율 · 저신뢰 비율 ·
    저신뢰→실패 정밀도/재현율."""
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        groups[(r.dataset, r.cls)].append(r)
    groups[("(전체)", "")] = list(records)
    rows = []
    for (ds, cls), rs in groups.items():
        if not rs:
            continue
        fail = [r.chain_iou < FAIL_IOU for r in rs]
        low = [r.confidence < LOW_CONFIDENCE for r in rs]
        tp = sum(f and lo for f, lo in zip(fail, low, strict=True))
        rows.append(
            {
                "dataset": ds,
                "class": cls,
                "n": len(rs),
                **{f"iou_{m}": round(statistics.median(r.iou[m] for r in rs), 3) for m in METHODS},
                "chain_used": dict(Counter(r.chain_used for r in rs)),
                "iou_chain": round(statistics.median(r.chain_iou for r in rs), 3),
                "fail_rate": round(sum(fail) / len(rs), 3),
                "lowconf_rate": round(sum(low) / len(rs), 3),
                "lowconf_precision": round(tp / sum(low), 3) if sum(low) else None,
                "lowconf_recall": round(tp / sum(fail), 3) if sum(fail) else None,
                "flags": dict(Counter(fl for r in rs for fl in r.flags)),
            }
        )
    return rows


def to_markdown(rows: list[dict], *, box_pad: float) -> str:
    head = (
        f"| 데이터셋 | 클래스 | n | grabcut | otsu | ellipse | rect | 사슬 실제 방법 | 사슬 IoU | 실패율(<{FAIL_IOU}) | 저신뢰율(<{LOW_CONFIDENCE}) | 저신뢰→실패 정밀도/재현율 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for r in rows:
        used = " ".join(
            f"{k} {v}" for k, v in sorted(r["chain_used"].items(), key=lambda kv: -kv[1])
        )
        pr = "—" if r["lowconf_precision"] is None else f"{r['lowconf_precision']:.2f}"
        rc = "—" if r["lowconf_recall"] is None else f"{r['lowconf_recall']:.2f}"
        body += (
            f"| {r['dataset']} | {r['class']} | {r['n']} | {r['iou_grabcut']:.3f} | {r['iou_otsu']:.3f} | {r['iou_ellipse']:.3f} | "
            f"{r['iou_rect']:.3f} | {used} | {r['iou_chain']:.3f} | {r['fail_rate']:.2f} | {r['lowconf_rate']:.2f} | {pr} / {rc} |\n"
        )
    note = (
        f"\n박스 = GT 성분 bbox 를 각 변 {box_pad:.0%} 느슨하게. IoU 는 클래스 중앙값. 사슬 = `mask_from_box(method=grabcut)` 폴백 포함. "
        "실패 = 사슬 IoU < 0.3. 저신뢰 = `mask_confidence.score < LOW_CONFIDENCE`. 결정은 하지 않는다 — KNOWN-ISSUES 결정 대기 항목의 근거 표.\n"
    )
    return head + body + note


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("inputs", nargs="+", type=Path, help="MVTec 카테고리 루트 또는 pairs.csv")
    ap.add_argument(
        "--out", type=Path, default=None, help="markdown 표 경로(.md; 같은 이름 .json 도 씀)"
    )
    ap.add_argument("--box-pad", type=float, default=0.1)
    ap.add_argument("--min-area", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="클래스당 이미지 수 상한(빠른 확인용)")
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    records = run(a.inputs, box_pad=a.box_pad, min_area=a.min_area, limit=a.limit)
    rows = summarize(records)
    md = to_markdown(rows, box_pad=a.box_pad)
    print(md)
    print(f"인스턴스 {len(records)} · {time.perf_counter() - t0:.1f} s")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(md, encoding="utf-8", newline="\n")
        a.out.with_suffix(".json").write_text(
            json.dumps(
                {"rows": rows, "records": [asdict(r) for r in records]},
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
            newline="\n",
        )
        print(f"→ {a.out} · {a.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

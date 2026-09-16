"""합성 유/무 YOLO mAP 비교 — MVTec 카테고리(GT 마스크)로 "합성 데이터가 검출력을 올리는가"의 첫 숫자.

    <train-venv>/python tools/train_mvtec_map.py samples/mvtec/metal_nut --anograft .venv/Scripts/python.exe \\
        --classes bent color scratch --k 8 --count 200 --epochs 30 --out out/train-map

절차(전부 로컬):
1. MVTec test/<cls> 결함 이미지 → GT 마스크 성분 bbox → YOLO 박스 라벨. 클래스마다 ``--k`` 장은 **real-train**, 나머지는 **real-val**(홀드아웃).
   test/good 은 val 에 음성으로(절반), 나머지 절반은 train 음성으로.
2. 은행은 **real-train 의 박스만으로**(``anograft bank import-yolo`` — 사용자 흐름과 같음, val 누수 없음) → 프리셋별로 ``count`` 장 합성(대상 = train/good).
3. 학습 A = real-train 만 · B = real-train + 합성(프리셋별) — 같은 모델·epoch·imgsz. 평가는 real-val 에서 mAP50 / mAP50-95.
4. 결과 표(markdown) — ``BENCHMARKS.md`` 에 옮긴다. 이 스크립트는 ultralytics 가 있는 **별도 venv** 에서 돌린다(코어 의존성 아님).

순수 부분(``boxes_from_mask``·``split_per_class``·``yolo_line``)은 ``tests/test_tools_train_map.py`` 가 고정한다.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def boxes_from_mask(
    mask: np.ndarray, *, min_area: int = 16
) -> list[tuple[float, float, float, float]]:
    """GT 마스크 → 성분별 YOLO 정규화 박스 (cx, cy, w, h). 면적 큰 순."""
    h, w = mask.shape[:2]
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        out.append((area, ((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h)))
    out.sort(key=lambda t: -t[0])
    return [b for _, b in out]


def yolo_line(cls_id: int, box: tuple[float, float, float, float]) -> str:
    return f"{cls_id} " + " ".join(f"{v:.6f}" for v in box)


def split_per_class(
    items: dict[str, list[Path]], k: int, seed: int
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    """클래스마다 ``k`` 장 train, 나머지 val — 결정적(seed)."""
    rng = random.Random(seed)
    train, val = [], []
    for cls in sorted(items):
        paths = sorted(items[cls])
        rng.shuffle(paths)
        train += [(cls, p) for p in paths[:k]]
        val += [(cls, p) for p in paths[k:]]
    return train, val


def write_yolo(root: Path, split: str, image: Path, lines: list[str], stem: str) -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image, root / "images" / split / f"{stem}{image.suffix}")
    (root / "labels" / split / f"{stem}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )


def build_real_sets(mvtec: Path, classes: list[str], k: int, seed: int, out: Path) -> dict:
    """real-train / real-val YOLO 셋 + train 전용 임포트 폴더(images/·labels/·data.yaml)."""
    items = {c: sorted((mvtec / "test" / c).glob("*.png")) for c in classes}
    train, val = split_per_class(items, k, seed)
    real = out / "real"
    imp = out / "import-train"
    for d in (real, imp):
        if d.exists():
            shutil.rmtree(d)
    (imp / "images").mkdir(parents=True)
    (imp / "labels").mkdir(parents=True)
    ids = {c: i for i, c in enumerate(classes)}
    counts = {"train": 0, "val": 0, "train_neg": 0, "val_neg": 0}
    for split, rows in (("train", train), ("val", val)):
        for cls, img in rows:
            mask = cv2.imread(
                str(mvtec / "ground_truth" / cls / f"{img.stem}_mask.png"), cv2.IMREAD_GRAYSCALE
            )
            lines = [yolo_line(ids[cls], b) for b in boxes_from_mask(mask)]
            stem = f"{cls}_{img.stem}"
            write_yolo(real, split, img, lines, stem)
            counts[split] += 1
            if split == "train":
                shutil.copyfile(img, imp / "images" / f"{stem}.png")
                (imp / "labels" / f"{stem}.txt").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
    goods = sorted((mvtec / "test" / "good").glob("*.png"))
    half = len(goods) // 2
    for i, g in enumerate(goods):
        split = "val" if i < half else "train"
        write_yolo(real, split, g, [], f"good_{g.stem}")
        counts[f"{split}_neg"] += 1
    names = {i: c for c, i in ids.items()}
    (real / "data.yaml").write_text(
        f"path: {real.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames: {json.dumps(names)}\n",
        encoding="utf-8",
    )
    (imp / "data.yaml").write_text(f"names: {json.dumps([c for c in classes])}\n", encoding="utf-8")
    return {"counts": counts, "classes": classes}


def run(cmd: list[str]) -> None:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def synthesize(
    anograft: str,
    mvtec: Path,
    out: Path,
    preset: str,
    count: int,
    seed: int,
    roi: str,
    mask_from: str,
) -> Path:
    bank = out / f"bank-{mask_from}"
    if not (bank / "bank.yaml").exists():
        run(
            [
                anograft,
                "-m",
                "anograft",
                "bank",
                "import-yolo",
                "--images",
                str(out / "import-train" / "images"),
                "--labels",
                str(out / "import-train" / "labels"),
                "--names",
                str(out / "import-train" / "data.yaml"),
                "--out",
                str(bank),
                "--mask-from",
                mask_from,
            ]
        )
    syn = out / f"syn-{preset}-{mask_from}"
    recipe = out / f"{preset}-{mask_from}.yaml"
    args = [
        anograft,
        "-m",
        "anograft",
        "recipe",
        "init",
        "--preset",
        preset,
        "--bank",
        str(bank),
        "--targets",
        str(mvtec / "train" / "good"),
        "--out",
        str(syn),
        "--count",
        str(count),
        "--seed",
        str(seed),
        "--write",
        str(recipe),
    ]
    if roi:
        args += ["--roi", roi]
    run(args)
    run([anograft, "-m", "anograft", "run", str(recipe), "--workers", "4"])
    return syn


def make_train_set(real: Path, syn_roots: list[Path], out: Path) -> Path:
    """train = real-train (+ 합성 images/labels) · val = real-val. 합성의 정상 이미지(빈 라벨)도 포함."""
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        shutil.copytree(real / "images" / split, out / "images" / split)
        shutil.copytree(real / "labels" / split, out / "labels" / split)
    for j, syn in enumerate(syn_roots):
        for img in sorted((syn / "images").iterdir()):
            lab = syn / "labels" / f"{img.stem}.txt"
            shutil.copyfile(img, out / "images" / "train" / f"syn{j}_{img.name}")
            (out / "labels" / "train" / f"syn{j}_{img.stem}.txt").write_text(
                lab.read_text(encoding="utf-8") if lab.exists() else "", encoding="utf-8"
            )
    data = (
        (real / "data.yaml")
        .read_text(encoding="utf-8")
        .replace(real.resolve().as_posix(), out.resolve().as_posix())
    )
    (out / "data.yaml").write_text(data, encoding="utf-8")
    return out


def train_eval(
    data_yaml: Path, *, model: str, epochs: int, imgsz: int, project: Path, name: str, seed: int
) -> dict:
    from ultralytics import YOLO

    t0 = time.perf_counter()
    m = YOLO(model)
    res = m.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        project=str(project.resolve()),  # 상대 경로면 ultralytics 가 runs/detect/ 를 앞에 붙인다
        name=name,
        exist_ok=True,
        seed=seed,
        deterministic=True,
        workers=0,
        verbose=False,
        plots=False,
        val=True,
        device="cpu",
        batch=16,
    )
    save_dir = Path(getattr(res, "save_dir", project.resolve() / name))
    best = save_dir / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(
        data=str(data_yaml),
        imgsz=imgsz,
        split="val",
        device="cpu",
        plots=False,
        verbose=False,
        workers=0,
    )
    return {
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "per_class_map50": {
            metrics.names[i]: round(float(v), 4)
            for i, v in zip(metrics.box.ap_class_index, metrics.box.ap50, strict=False)
        },
        "minutes": round((time.perf_counter() - t0) / 60, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mvtec", type=Path)
    ap.add_argument("--anograft", required=True, help="anograft 가 설치된 venv 의 python.exe")
    ap.add_argument("--classes", nargs="+", default=["bent", "color", "scratch"])
    ap.add_argument("--k", type=int, default=8, help="클래스당 real-train 장수")
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--presets", nargs="+", default=["poisson-graft", "dent-graft"])
    ap.add_argument("--roi", default="annulus")
    ap.add_argument("--mask-from", default="grabcut")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("out/train-map"))
    ap.add_argument("--skip-train", action="store_true", help="데이터만 만들고 학습은 생략")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    a.anograft = str(Path(a.anograft).resolve())  # CreateProcess 는 상대 경로를 못 찾는다
    info = build_real_sets(a.mvtec, a.classes, a.k, a.seed, a.out)
    print("real sets:", info, flush=True)
    syn_roots = [
        synthesize(a.anograft, a.mvtec, a.out, p, a.count, a.seed, a.roi, a.mask_from)
        for p in a.presets
    ]
    sets = {"A real-train only": make_train_set(a.out / "real", [], a.out / "set-A")}
    for p, syn in zip(a.presets, syn_roots, strict=True):
        sets[f"B +{p}"] = make_train_set(a.out / "real", [syn], a.out / f"set-B-{p}")
    if a.skip_train:
        print("데이터만 만들었습니다:", {k: str(v) for k, v in sets.items()})
        return 0
    results = {}
    for name, root in sets.items():
        print(f"== train {name}", flush=True)
        results[name] = train_eval(
            root / "data.yaml",
            model=a.model,
            epochs=a.epochs,
            imgsz=a.imgsz,
            project=a.out / "runs",
            name=name.split(" ")[0] + "-" + name.split(" ")[-1].strip("+"),
            seed=a.seed,
        )
        print(name, results[name], flush=True)
    rows = ["| 학습셋 | mAP50 | mAP50-95 | 클래스별 mAP50 | 분 |", "|---|---|---|---|---|"]
    for name, r in results.items():
        pc = " · ".join(f"{c} {v:.2f}" for c, v in r["per_class_map50"].items())
        rows.append(f"| {name} | {r['map50']:.3f} | {r['map50_95']:.3f} | {pc} | {r['minutes']} |")
    md = (
        "\n".join(rows)
        + f"\n\n{a.mvtec.name} · classes {a.classes} · real-train {a.k}/class · real-val {info['counts']['val']}(+good {info['counts']['val_neg']}) · 합성 {a.count}/프리셋(roi {a.roi}, mask_from {a.mask_from}) · {a.model} imgsz {a.imgsz} epochs {a.epochs} seed {a.seed} · CPU\n"
    )
    (a.out / "result.md").write_text(md, encoding="utf-8")
    (a.out / "result.json").write_text(
        json.dumps(
            {
                "info": info,
                "results": results,
                "args": vars(a) | {"mvtec": str(a.mvtec), "out": str(a.out)},
            },
            ensure_ascii=False,
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

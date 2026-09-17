"""합성 유/무 YOLO mAP 비교 — GT 마스크가 있는 공개 데이터(MVTec 카테고리 · pairs.csv)로 "합성 데이터가 검출력을 올리는가"의 숫자.

    <train-venv>/python tools/train_mvtec_map.py samples/mvtec/metal_nut --anograft .venv/Scripts/python.exe \\
        --classes bent color scratch --k 8 --count 200 --epochs 40 --out out/train-map
    <train-venv>/python tools/train_mvtec_map.py samples/magnetic-tile/pairs.csv --anograft .venv/Scripts/python.exe \\
        --classes blowhole break crack --roi none --mask-from grabcut hybrid --train-seeds 7 8 9 --out out/train-map-mt

절차(전부 로컬):
1. 결함 이미지 → GT 마스크 성분 bbox → YOLO 박스 라벨. 클래스마다 ``--k`` 장은 **real-train**, 나머지는 **real-val**(홀드아웃).
   정상(MVTec ``test/good`` · pairs 는 ``normals.txt`` 앞 ``--n-good`` 장)은 반은 val 음성, 반은 train 음성. 합성 대상은
   MVTec ``train/good`` · pairs 는 그다음 ``--n-targets`` 장(``targets.txt``).
2. 은행은 **real-train 의 박스만으로**(``anograft bank import-yolo`` — 사용자 흐름과 같음, val 누수 없음) → ``--mask-from`` 마다
   은행 하나 → 프리셋별로 ``count`` 장 합성.
3. 학습 A = real-train 만 · B = real-train + 합성(프리셋 × 은행) — 같은 모델·epoch·imgsz. 평가는 real-val 에서 mAP50 / mAP50-95.
   ``--train-seeds`` 로 학습 시드만 여러 개(분할·합성은 고정) → 표에 시드별 + 평균 Δ. 끝난 (셋, 시드) 는 ``results/`` 에 캐시되어
   같은 ``--out`` 으로 다시 부르면 건너뛴다(프리셋·은행을 나중에 보태는 흐름).
4. 결과 표(markdown) — ``BENCHMARKS.md`` 에 옮긴다. 이 스크립트는 ultralytics 가 있는 **별도 venv** 에서 돌린다(코어 의존성 아님).

**한 ``--out`` 폴더 = 한 분할**(``real/split.json``) — 다른 ``--split-seed``·``--k``·``--classes`` 면 새 폴더.
순수 부분(``boxes_from_mask``·``split_per_class``·``parse_pairs_csv``·``split_normals``·``summarize``)은
``tests/test_tools_train_map.py`` 가 고정한다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


def boxes_from_mask(
    mask: np.ndarray, *, min_area: int = 16, threshold: int = 127
) -> list[tuple[float, float, float, float]]:
    """GT 마스크 → 성분별 YOLO 정규화 박스 (cx, cy, w, h). 면적 큰 순. ``> threshold`` 로 이진화(JPEG 링 잡음 제거)."""
    h, w = mask.shape[:2]
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > threshold).astype(np.uint8), connectivity=8
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


@dataclass
class DatasetSpec:
    """벤치 입력 한 벌 — 결함 이미지(클래스별)·GT 마스크·정상(음성 분할용)·합성 대상. 레이아웃 차이는 여기서 끝난다."""

    name: str
    items: dict[str, list[Path]]
    masks: dict[Path, Path]
    goods: list[Path]
    targets: list[Path]
    extra: dict[str, int] = field(default_factory=dict)


def load_mvtec(root: Path, classes: list[str]) -> DatasetSpec:
    """MVTec 레이아웃: ``test/<cls>/*.png`` + ``ground_truth/<cls>/<stem>_mask.png`` · 정상 ``test/good`` · 대상 ``train/good``."""
    items = {c: sorted((root / "test" / c).glob("*.png")) for c in classes}
    masks = {
        p: root / "ground_truth" / c / f"{p.stem}_mask.png" for c, ps in items.items() for p in ps
    }
    return DatasetSpec(
        root.name,
        items,
        masks,
        sorted((root / "test" / "good").glob("*.png")),
        sorted((root / "train" / "good").glob("*.png")),
    )


def parse_pairs_csv(text: str) -> list[tuple[str, str, str]]:
    """``image,mask,class`` CSV 본문 → [(image, mask, class)] (헤더 필수, 빈 줄 무시)."""
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        if not row.get("image"):
            continue
        rows.append((row["image"], row["mask"], row["class"]))
    return rows


def split_normals(
    normals: list[Path], n_good: int, n_targets: int
) -> tuple[list[Path], list[Path]]:
    """정렬된 정상 목록 → (음성 분할용 ``n_good``, 합성 대상 ``n_targets``) — 겹치지 않게 앞에서부터."""
    goods = normals[:n_good]
    targets = normals[n_good : n_good + n_targets]
    if len(targets) < n_targets:
        raise SystemExit(
            f"정상 이미지가 모자랍니다: {len(normals)}장 < n-good {n_good} + n-targets {n_targets}"
        )
    return goods, targets


def load_pairs(
    csv_path: Path, classes: list[str], normals_txt: Path, n_good: int, n_targets: int
) -> DatasetSpec:
    """``pairs.csv``(image,mask,class — CSV 파일 기준 경로) + ``normals.txt``(목록 파일 기준 경로)."""
    base = csv_path.parent
    items: dict[str, list[Path]] = {c: [] for c in classes}
    masks: dict[Path, Path] = {}
    for image, mask, cls in parse_pairs_csv(csv_path.read_text(encoding="utf-8")):
        if cls in items:
            items[cls].append(base / image)
            masks[base / image] = base / mask
    missing = [c for c in classes if not items[c]]
    if missing:
        raise SystemExit(f"pairs.csv 에 없는 클래스: {missing}")
    lines = [
        ln.strip()
        for ln in normals_txt.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    normals = sorted(normals_txt.parent / ln for ln in lines)
    goods, targets = split_normals(normals, n_good, n_targets)
    return DatasetSpec(
        base.name, items, masks, goods, targets, {"normals": len(normals), "n_good": n_good}
    )


def write_yolo(root: Path, split: str, image: Path, lines: list[str], stem: str) -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image, root / "images" / split / f"{stem}{image.suffix}")
    (root / "labels" / split / f"{stem}.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )


def build_real_sets(
    spec: DatasetSpec | Path, classes: list[str], k: int, seed: int, out: Path
) -> dict:
    """real-train / real-val YOLO 셋 + train 전용 임포트 폴더(images/·labels/·data.yaml) + ``targets.txt``.

    ``real/split.json`` 이 있고 같은 분할이면 그대로 두고(은행·합성 캐시 유지), 다르면 거부한다(한 폴더 = 한 분할).
    """
    if isinstance(spec, Path):
        spec = load_mvtec(spec, classes)
    real = out / "real"
    imp = out / "import-train"
    marker = real / "split.json"
    key = {"split_seed": seed, "k": k, "classes": classes, "dataset": spec.name}
    if marker.is_file():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if {kk: old.get(kk) for kk in key} != key:
            raise SystemExit(
                f"{out} 에는 다른 분할이 있습니다({old}) — 새 --out 폴더를 쓰세요 (한 폴더 = 한 분할)"
            )
        return json.loads((out / "real-info.json").read_text(encoding="utf-8"))
    train, val = split_per_class(spec.items, k, seed)
    for d in (real, imp):
        if d.exists():
            shutil.rmtree(d)
    (imp / "images").mkdir(parents=True)
    (imp / "labels").mkdir(parents=True)
    ids = {c: i for i, c in enumerate(classes)}
    counts = {"train": 0, "val": 0, "train_neg": 0, "val_neg": 0}
    for split, rows in (("train", train), ("val", val)):
        for cls, img in rows:
            mask = cv2.imread(str(spec.masks[img]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise SystemExit(f"GT 마스크를 읽지 못했습니다: {spec.masks[img]}")
            lines = [yolo_line(ids[cls], b) for b in boxes_from_mask(mask)]
            stem = f"{cls}_{img.stem}"
            write_yolo(real, split, img, lines, stem)
            counts[split] += 1
            if split == "train":
                shutil.copyfile(img, imp / "images" / f"{stem}{img.suffix}")
                (imp / "labels" / f"{stem}.txt").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
    half = len(spec.goods) // 2
    for i, g in enumerate(spec.goods):
        split = "val" if i < half else "train"
        write_yolo(real, split, g, [], f"good_{g.stem}")
        counts[f"{split}_neg"] += 1
    names = {i: c for c, i in ids.items()}
    (real / "data.yaml").write_text(
        f"path: {real.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames: {json.dumps(names)}\n",
        encoding="utf-8",
    )
    (imp / "data.yaml").write_text(f"names: {json.dumps([c for c in classes])}\n", encoding="utf-8")
    (out / "targets.txt").write_text(
        "\n".join(p.resolve().as_posix() for p in spec.targets) + "\n", encoding="utf-8"
    )
    info = {
        "counts": counts,
        "classes": classes,
        "dataset": spec.name,
        "targets": len(spec.targets),
        **spec.extra,
    }
    (out / "real-info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    marker.write_text(json.dumps(key, ensure_ascii=False), encoding="utf-8")
    return info


def run(cmd: list[str]) -> None:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def recipe_matches(recipe: Path, seed: int, count: int) -> bool:
    """``recipe init --write`` 결과가 같은 seed·count 인지(줄 끝 CRLF 무관 — Windows 에서 CRLF 로 써진다)."""
    if not recipe.is_file():
        return False
    lines = {ln.strip() for ln in recipe.read_text(encoding="utf-8").splitlines()}
    return f"seed: {seed}" in lines and f"count: {count}" in lines


def synthesize(
    anograft: str,
    out: Path,
    preset: str,
    count: int,
    seed: int,
    roi: str,
    mask_from: str,
) -> Path:
    """은행(``bank-<mask_from>``, 없을 때만) → 레시피 → ``run``. 같은 seed·count 의 합성이 이미 있으면 건너뛴다."""
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
    if (syn / "manifest.csv").is_file() and recipe_matches(recipe, seed, count):
        print(f"  (합성 재사용: {syn})", flush=True)
        return syn
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
        str(out / "targets.txt"),
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


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def set_short_name(name: str) -> str:
    """표 이름 → 캐시·runs 폴더 이름. ``A real-train only`` → ``A-real-train`` · ``B +poisson-graft (hybrid)`` → ``B-poisson-graft-hybrid``."""
    words = name.split(" ")
    short = words[0] + "-" + words[1].strip("+")
    if "(" in name:
        short += "-" + name.split("(")[1].rstrip(")")
    return short


def summarize(results: dict[str, dict[int, dict]], seeds: list[int], baseline: str) -> str:
    """{셋: {학습 시드: metrics}} → markdown 표. 시드별 mAP50(Δ vs baseline) · 평균 mAP50(평균 Δ) · 평균 mAP50-95 · 클래스별 평균."""
    head = (
        ["학습셋"]
        + [f"s{s}" for s in seeds]
        + ["mAP50 평균(Δ)", "mAP50-95 평균", "클래스별 mAP50 평균", "분/학습"]
    )
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    base = results.get(baseline, {})
    for name, per_seed in results.items():
        cells = []
        for s in seeds:
            r = per_seed.get(s)
            if r is None:
                cells.append("—")
                continue
            b = base.get(s)
            delta = f" ({r['map50'] - b['map50']:+.2f})" if b and name != baseline else ""
            cells.append(f"{r['map50']:.3f}{delta}")
        have = [per_seed[s] for s in seeds if s in per_seed]
        if not have:
            rows.append(f"| {name} | " + " | ".join(cells) + " | — | — | — | — |")
            continue
        m50 = _mean([r["map50"] for r in have])
        both = [
            (per_seed[s]["map50"], base[s]["map50"]) for s in seeds if s in per_seed and s in base
        ]
        md = f" ({_mean([a - b for a, b in both]):+.2f})" if both and name != baseline else ""
        classes = sorted({c for r in have for c in r["per_class_map50"]})
        pc = " · ".join(
            f"{c} {_mean([r['per_class_map50'].get(c, 0.0) for r in have]):.2f}" for c in classes
        )
        rows.append(
            f"| {name} | "
            + " | ".join(cells)
            + f" | **{m50:.3f}**{md} | {_mean([r['map50_95'] for r in have]):.3f} | {pc} | {_mean([r['minutes'] for r in have]):.1f} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset", type=Path, help="MVTec 카테고리 루트 또는 pairs.csv")
    ap.add_argument("--anograft", required=True, help="anograft 가 설치된 venv 의 python.exe")
    ap.add_argument("--classes", nargs="+", default=["bent", "color", "scratch"])
    ap.add_argument("--k", type=int, default=8, help="클래스당 real-train 장수")
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--presets", nargs="+", default=["poisson-graft", "dent-graft"])
    ap.add_argument("--roi", default="annulus")
    ap.add_argument(
        "--mask-from",
        nargs="+",
        default=["grabcut"],
        help="은행 박스→마스크 방법(여럿이면 은행마다 B 셋)",
    )
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--seed", type=int, default=7, help="합성 시드(+ 분할·학습 시드의 기본값)")
    ap.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="real-train/val 분할 시드(기본 = --seed). 분할은 고정하고 학습 시드만 바꿔 분산을 볼 때",
    )
    ap.add_argument(
        "--train-seeds",
        nargs="+",
        type=int,
        default=None,
        help="학습 시드 목록(기본 = --seed 하나). 분할·합성은 고정, 학습만 반복 → 시드별 + 평균 Δ 표",
    )
    ap.add_argument(
        "--normals",
        type=Path,
        default=None,
        help="pairs.csv 일 때 정상 목록(기본 <csv 폴더>/normals.txt)",
    )
    ap.add_argument(
        "--n-good", type=int, default=40, help="pairs.csv 일 때 음성 분할용 정상 장수(반반)"
    )
    ap.add_argument(
        "--n-targets", type=int, default=220, help="pairs.csv 일 때 합성 대상 정상 장수"
    )
    ap.add_argument("--out", type=Path, default=Path("out/train-map"))
    ap.add_argument("--skip-train", action="store_true", help="데이터만 만들고 학습은 생략")
    ap.add_argument("--force", action="store_true", help="results/ 캐시를 무시하고 다시 학습")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    a.anograft = str(Path(a.anograft).resolve())  # CreateProcess 는 상대 경로를 못 찾는다
    split_seed = a.seed if a.split_seed is None else a.split_seed
    train_seeds = a.train_seeds or [a.seed]
    if a.dataset.suffix.lower() == ".csv":
        normals = a.normals or a.dataset.parent / "normals.txt"
        spec = load_pairs(a.dataset, a.classes, normals, a.n_good, a.n_targets)
    else:
        spec = load_mvtec(a.dataset, a.classes)
    info = build_real_sets(spec, a.classes, a.k, split_seed, a.out)
    info["split_seed"] = split_seed
    print("real sets:", info, flush=True)
    baseline = "A real-train only"
    sets = {baseline: make_train_set(a.out / "real", [], a.out / "set-A")}
    for mf in a.mask_from:
        for p in a.presets:
            syn = synthesize(a.anograft, a.out, p, a.count, a.seed, a.roi, mf)
            sets[f"B +{p} ({mf})"] = make_train_set(
                a.out / "real", [syn], a.out / f"set-B-{p}-{mf}"
            )
    if a.skip_train:
        print("데이터만 만들었습니다:", {k: str(v) for k, v in sets.items()})
        return 0
    cache = a.out / "results"
    cache.mkdir(exist_ok=True)
    results: dict[str, dict[int, dict]] = {}
    for name, root in sets.items():
        short = set_short_name(name)
        for s in train_seeds:
            entry = cache / f"{short}-s{s}.json"
            if entry.is_file() and not a.force:
                results.setdefault(name, {})[s] = json.loads(entry.read_text(encoding="utf-8"))
                print(f"== cached {name} s{s}: {results[name][s]}", flush=True)
                continue
            print(f"== train {name} s{s}", flush=True)
            r = train_eval(
                root / "data.yaml",
                model=a.model,
                epochs=a.epochs,
                imgsz=a.imgsz,
                project=a.out / "runs",
                name=f"{short}-s{s}",
                seed=s,
            )
            entry.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
            results.setdefault(name, {})[s] = r
            print(name, s, r, flush=True)
    md = (
        summarize(results, train_seeds, baseline)
        + f"\n\n{spec.name} · classes {a.classes} · real-train {a.k}/class · real-val {info['counts']['val']}(+good {info['counts']['val_neg']}) · 합성 {a.count}/프리셋(roi {a.roi}, mask_from {a.mask_from}, 대상 {info['targets']}) · {a.model} imgsz {a.imgsz} epochs {a.epochs} · split-seed {split_seed} · 합성 seed {a.seed} · 학습 seeds {train_seeds} · CPU\n"
    )
    (a.out / "result.md").write_text(md, encoding="utf-8")
    (a.out / "result.json").write_text(
        json.dumps(
            {
                "info": info,
                "results": {k: {str(s): r for s, r in v.items()} for k, v in results.items()},
                "args": vars(a) | {"dataset": str(a.dataset), "out": str(a.out)},
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

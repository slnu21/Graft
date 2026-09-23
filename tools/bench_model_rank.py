"""(A) 모델 순위 상관 — **합성셋으로 모델을 고를 수 있는가**(설계 `v1.x-training-loop.md` §5).

주장하려는 것은 "합성이 mAP 를 올린다"(그건 `train_mvtec_map.py` 의 §2)가 아니라 **"실제 결함이 거의 없어도
합성셋만으로 모델·백본을 고를 수 있다"** 이다. 현장의 진짜 문제는 결함 샘플이 없다는 것이라, 이 주장이 더 세다.

    .venv/Scripts/python.exe tools/bench_model_rank.py samples/mvtec/metal_nut --anograft .venv/Scripts/python.exe \\
        --models patchcore padim --n-train 60 --n-targets 40 --count 40 --out out/rank-metal-nut

절차(전부 로컬):
1. ``train/good`` 을 **학습 정상**(앞 ``--n-train`` 장)과 **합성 대상 정상**(그다음 ``--n-targets`` 장)으로 **가른다**.
   겹치면 배경을 외운 모델이 그 이미지의 합성본에서 비현실적으로 깨끗한 이상맵을 내 낙관 편향이 생긴다(§5 함정).
2. 평가 루트 둘을 만든다 — 둘 다 같은 ``train/good``(학습 정상) 을 쓰고 ``test/`` 만 다르다.
   - **real**: MVTec ``test/<결함>`` + ``test/good`` + ``ground_truth/``(원본 GT).
   - **syn**: Graft 합성(대상 = 합성 대상 정상, writer ``mvtec``) + 남은 정상 일부를 ``test/good`` 으로.
3. 모델마다 두 루트에서 ``anograft trainer fit anomalib --spec {"model": …}`` → 지표(이미지 AUROC · 픽셀 AUPRO).
   **학습기 계약을 그대로 쓴다**(T1·T3) — anomalib 은 ``trains_on: normal_only`` 라 합성 결함이 학습셋에 들어가지 않는다.
4. 두 순위의 **Spearman ρ** → ``BENCHMARKS.md`` §3.

순수 부분(``ranks``·``spearman``·``split_normals``·``rank_table``)은 ``tests/test_tools_model_rank.py`` 가 고정한다.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

# --------------------------------------------------------------------- 순수 부분


def ranks(values: Sequence[float]) -> list[float]:
    """큰 값이 1등인 순위(동점은 평균 순위). Spearman 의 입력."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based 평균
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """순위 상관. 표본이 2개 미만이거나 한쪽이 전부 동점이면 ``nan``(의미가 없다)."""
    if len(xs) != len(ys):
        raise ValueError("두 계열의 길이가 달라 순위를 맞출 수 없습니다")
    n = len(xs)
    if n < 2:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def split_normals(
    paths: Sequence[Path], n_train: int, n_targets: int
) -> tuple[list[Path], list[Path]]:
    """정상 목록 → (학습 정상, 합성 대상 정상). **겹치지 않는다**(§5 낙관 편향 방지).

    목록은 부르는 쪽이 정렬해서 준다(재현성). 모자라면 있는 만큼만 — 두 몫이 겹치는 것보다 낫다.
    """
    train = list(paths[:n_train])
    targets = list(paths[n_train : n_train + n_targets])
    return train, targets


def rank_table(results: dict[str, dict[str, float]], metric: str) -> str:
    """모델 × (real·syn) 지표 → markdown 표 + Spearman ρ 한 줄."""
    models = sorted(results)
    real = [results[m].get(f"real/{metric}", float("nan")) for m in models]
    syn = [results[m].get(f"syn/{metric}", float("nan")) for m in models]
    rr, rs = ranks(real), ranks(syn)
    lines = [
        f"| 모델 | 실제 {metric} (순위) | 합성 {metric} (순위) |",
        "|---|---|---|",
    ]
    for i, m in enumerate(models):
        lines.append(f"| {m} | {real[i]:.3f} ({rr[i]:.0f}) | {syn[i]:.3f} ({rs[i]:.0f}) |")
    rho = spearman(real, syn)
    lines.append("")
    lines.append(
        f"Spearman ρ(실제, 합성) = **{rho:.3f}** — 1.0 이면 합성셋만 보고 고른 순위가 실제와 같다."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------- 실행 부분


def run(cmd: list[str]) -> None:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def defect_classes(dataset: Path) -> list[str]:
    """MVTec 카테고리의 결함 폴더 이름(= 클래스). ``good`` 은 뺀다."""
    return sorted(d.name for d in (dataset / "test").iterdir() if d.is_dir() and d.name != "good")


def copy_images(paths: Sequence[Path], dst: Path, prefix: str = "") -> int:
    dst.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copyfile(p, dst / f"{prefix}{p.name}")
    return len(paths)


def build_real_root(
    dataset: Path, classes: list[str], train_normals: Sequence[Path], k: int, out: Path
) -> dict:
    """실제 평가 루트 — ``train/good`` = 학습 정상 · ``test/`` = MVTec 원본(은행에 쓴 k 장 제외).

    은행에 넣은 결함은 **평가에서 뺀다**(그 이미지의 조각이 합성셋에 들어가므로 남겨 두면 누수).
    """
    if out.exists():
        shutil.rmtree(out)
    copy_images(train_normals, out / "train" / "good")
    copy_images(sorted((dataset / "test" / "good").glob("*.png")), out / "test" / "good")
    counts: dict[str, int] = {}
    for cls in classes:
        imgs = sorted((dataset / "test" / cls).glob("*.png"))
        held = imgs[k:]  # 앞 k 장은 은행(합성 재료)으로 갔다
        copy_images(held, out / "test" / cls)
        masks = [dataset / "ground_truth" / cls / f"{p.stem}_mask.png" for p in held]
        copy_images([m for m in masks if m.is_file()], out / "ground_truth" / cls)
        counts[cls] = len(held)
    return counts


def build_bank(anograft: str, dataset: Path, classes: list[str], k: int, out: Path) -> Path:
    """은행은 **결함 k 장/클래스**만으로 — "결함 몇 장이면 되는가"가 이 실험의 주장이다."""
    stage = out / "bank-src"
    if stage.exists():
        shutil.rmtree(stage)
    for cls in classes:
        imgs = sorted((dataset / "test" / cls).glob("*.png"))[:k]
        copy_images(imgs, stage / "test" / cls)
        copy_images(
            [dataset / "ground_truth" / cls / f"{p.stem}_mask.png" for p in imgs],
            stage / "ground_truth" / cls,
        )
    (stage / "train" / "good").mkdir(parents=True, exist_ok=True)
    bank = out / "bank"
    if bank.exists():
        shutil.rmtree(bank)
    # 표준셋은 **진짜 GT 마스크**를 주므로 박스→마스크 추정(`--mask-from`)이 없다 — 이 실험의 변수는
    # '결함 몇 장이면 되는가'이지 마스크 추정 품질이 아니다(그건 BENCHMARKS §1).
    run([
        anograft, "-m", "anograft", "bank", "import-dataset", "mvtec-ad", str(stage),
        "--out", str(bank),
    ])  # fmt: skip
    return bank


def build_syn_root(
    anograft: str,
    bank: Path,
    targets: Sequence[Path],
    train_normals: Sequence[Path],
    category: str,
    a: argparse.Namespace,
    out: Path,
) -> dict:
    """합성 평가 루트 — 대상은 **학습 정상과 겹치지 않는** 정상(§5 낙관 편향)."""
    listing = out / "targets.txt"
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text("\n".join(str(p.resolve()) for p in targets), encoding="utf-8")

    recipe = out / "recipe.yaml"
    run([
        anograft, "-m", "anograft", "recipe", "init", "--preset", a.preset,
        "--bank", str(bank), "--targets", str(listing), "--roi", a.roi,
        "--seed", str(a.seed), "--count", str(a.count), "--write", str(recipe),
    ])  # fmt: skip
    # writer 는 CLI 플래그가 없다 → 레시피의 writer 줄을 갈아 끼운다(mvtec = anomalib 이 그대로 읽는 레이아웃).
    # test_normal_ratio 1.0 = 합성 대상 중 결함이 안 붙은 정상은 전부 test/good 으로(학습 정상은 따로 덮어쓴다).
    text, n = re.subn(
        r"^(\s*)writer:.*$",
        lambda m: (
            f"{m.group(1)}writer: {{format: mvtec, category: {category}, test_normal_ratio: 1.0}}"
        ),
        recipe.read_text(encoding="utf-8"),
        count=1,
        flags=re.MULTILINE,
    )
    if n != 1 or "format: mvtec" not in text:
        raise SystemExit(
            "레시피에 mvtec writer 를 넣지 못했습니다 — recipe init 출력 형식을 확인하세요"
        )
    recipe.write_text(text, encoding="utf-8")

    syn = out / "syn"
    if syn.exists():
        shutil.rmtree(syn)
    run([anograft, "-m", "anograft", "run", str(recipe), "--out", str(syn)])

    made = syn / "mvtec" / category
    dst = out.parent / "eval-syn" / category
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(made, dst)
    # train/good 은 **학습 정상으로 덮어쓴다** — 합성 대상이 학습에 들어가면 안 된다(§5 함정)
    shutil.rmtree(dst / "train" / "good", ignore_errors=True)
    copy_images(train_normals, dst / "train" / "good")
    return {
        "test": {d.name: len(list(d.glob("*.png"))) for d in sorted((dst / "test").iterdir())},
        "train_good": len(list((dst / "train" / "good").glob("*.png"))),
    }


def build_eval_roots(a: argparse.Namespace) -> dict:
    """실제·합성 평가 루트 둘을 만든다. **``train/good`` 은 두 루트가 똑같다** — 다른 건 ``test/`` 뿐이다."""
    category = a.dataset.name
    classes = a.classes or defect_classes(a.dataset)
    normals = sorted((a.dataset / "train" / "good").glob("*.png"))
    train_normals, targets = split_normals(normals, a.n_train, a.n_targets)
    if not train_normals or not targets:
        raise SystemExit(
            f"정상이 모자랍니다: {len(normals)}장 (학습 {a.n_train} + 대상 {a.n_targets})"
        )

    real_counts = build_real_root(
        a.dataset, classes, train_normals, a.k, a.out / "eval-real" / category
    )
    bank = build_bank(a.anograft, a.dataset, classes, a.k, a.out)
    syn_counts = build_syn_root(
        a.anograft, bank, targets, train_normals, category, a, a.out / "work"
    )
    return {
        "category": category,
        "classes": classes,
        "k_per_class": a.k,
        "train_normals": len(train_normals),
        "targets": len(targets),
        "real_test": real_counts,
        "syn": syn_counts,
    }


def fit_metrics(
    anograft: str,
    trainer: str,
    dataset: Path,
    out: Path,
    seed: int,
    spec: dict,
    trainers_file: Path | None,
) -> dict[str, float]:
    """``anograft trainer fit`` 한 번 → 평평한 지표 맵(계약 §1.1)."""
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / "spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    cmd = [
        anograft, "-m", "anograft", "trainer", "fit", trainer,
        "--dataset", str(dataset), "--out", str(out), "--seed", str(seed),
        "--spec", str(spec_path), "--json",
    ]  # fmt: skip
    if trainers_file:
        cmd += ["--file", str(trainers_file)]
    print("  $", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8", check=True)
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    return {k: float(v) for k, v in (payload.get("metrics") or {}).items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "dataset", type=Path, help="MVTec 카테고리 루트(train/good · test/ · ground_truth/)"
    )
    ap.add_argument("--anograft", required=True, help="anograft 가 설치된 venv 의 python.exe")
    ap.add_argument("--trainer", default="anomalib", help="trainers.yaml 의 학습기 이름")
    ap.add_argument("--trainers-file", type=Path, default=None)
    ap.add_argument(
        "--models", nargs="+", default=["patchcore", "padim"], help="비교할 모델(spec.model)"
    )
    ap.add_argument("--n-train", type=int, default=60, help="anomalib 학습 정상 장수")
    ap.add_argument(
        "--n-targets", type=int, default=40, help="합성 대상 정상 장수(학습 정상과 분리)"
    )
    ap.add_argument("--count", type=int, default=40, help="합성 장수")
    ap.add_argument(
        "--k", type=int, default=5, help="은행에 넣을 결함 장수/클래스 (평가에서는 제외)"
    )
    ap.add_argument("--preset", default="poisson-graft")
    ap.add_argument("--roi", default="annulus")
    ap.add_argument(
        "--classes", nargs="+", default=None, help="은행 클래스(기본: 데이터셋의 결함 폴더 전부)"
    )
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--metric", default="image_AUROC", help="순위에 쓸 지표 키")
    ap.add_argument("--out", type=Path, default=Path("out/rank"))
    a = ap.parse_args(argv)
    a.anograft = str(Path(a.anograft).resolve())
    a.out.mkdir(parents=True, exist_ok=True)

    info = build_eval_roots(a)
    print("eval roots:", json.dumps(info, ensure_ascii=False, default=str), flush=True)

    results: dict[str, dict[str, float]] = {}
    for model in a.models:
        for kind in ("real", "syn"):
            root = a.out / f"eval-{kind}" / info["category"]  # 어댑터는 **카테고리 폴더**를 받는다
            m = fit_metrics(
                a.anograft,
                a.trainer,
                root,
                a.out / "runs" / f"{model}-{kind}",
                a.seed,
                {"model": model, "image_size": a.image_size},
                a.trainers_file,
            )
            print(f"== {model} {kind}: {m}", flush=True)
            results.setdefault(model, {}).update({f"{kind}/{k}": v for k, v in m.items()})

    md = rank_table(results, a.metric)
    (a.out / "result.md").write_text(md + "\n", encoding="utf-8")
    (a.out / "result.json").write_text(
        json.dumps({"info": info, "results": results}, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8",
    )
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

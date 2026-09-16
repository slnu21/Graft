"""공개 데이터셋 받기 — 표준 라이브러리만(`datasets`·`huggingface_hub` 불요). 로컬 사본만, 재배포 금지.

    python tools/fetch_public_datasets.py metal_nut screw grid      # MVTec AD 카테고리(HF 미러, 원본 폴더 구조)
    python tools/fetch_public_datasets.py dtd                        # DTD 결함류 15 카테고리 → samples/dtd (perlin-texture)
    python tools/fetch_public_datasets.py magnetic-tile              # GitHub abin24 (git clone) + pairs.csv
    python tools/fetch_public_datasets.py visa                       # 공식 S3 tar 1.9 GB → samples/visa (CC BY 4.0)
    python tools/fetch_public_datasets.py metal_nut --import         # 받은 뒤 bank/<name> 까지 (anograft 가 설치된 venv 에서)

라이선스: MVTec AD **CC BY-NC-SA 4.0**(HF 카드가 MIT 라고 적혀 있어도 안의 license.txt 가 원본) · DTD 연구용 · Magnetic Tile 논문 인용 ·
VisA CC BY 4.0. 전부 `samples/`(gitignore) 에만 둔다. 이미 있는 파일은 건너뛴다(재실행 안전).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"

MVTEC_REPO = "foersben/mvtec-ad"  # 원본 구조 15 카테고리 (results/ 는 받지 않음)
MVTEC_CATEGORIES = [
    "bottle",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "cable",
    "capsule",
    "zipper",
]
DTD_REPO = "cansa/Describable-Textures-Dataset-DTD"  # imagefolder images/<category>/
DTD_DEFECT_LIKE = [
    "blotchy",
    "bumpy",
    "cracked",
    "crystalline",
    "fibrous",
    "flecked",
    "pitted",
    "porous",
    "potholed",
    "scaly",
    "smeared",
    "sprinkled",
    "stained",
    "veined",
    "wrinkled",
]
MT_GIT = "https://github.com/abin24/Magnetic-tile-defect-datasets."
VISA_TAR = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"


def hf_files(repo: str) -> list[str]:
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}", timeout=30) as r:
        meta = json.load(r)
    return [s["rfilename"] for s in meta["siblings"] if not s["rfilename"].startswith(".")]


def select_prefixed(files: list[str], prefixes: tuple[str, ...]) -> list[str]:
    """``prefixes`` 로 시작하는 파일만(비면 전부). 순수 함수 — 테스트."""
    return [f for f in files if not prefixes or f.startswith(prefixes)]


def fetch_hf(repo: str, out: Path, prefixes: tuple[str, ...]) -> int:
    files = select_prefixed(hf_files(repo), prefixes)
    raw = f"https://huggingface.co/datasets/{repo}/resolve/main/"
    print(f"{repo}: {len(files)} files -> {out}  prefixes={prefixes or '(all)'}", flush=True)

    def one(path: str) -> tuple[str, int, str]:
        dst = out / path
        if dst.exists() and dst.stat().st_size > 0:
            return path, dst.stat().st_size, "skip"
        dst.parent.mkdir(parents=True, exist_ok=True)
        err = ""
        for _ in range(3):
            try:
                with urllib.request.urlopen(raw + path, timeout=60) as r:
                    data = r.read()
                dst.write_bytes(data)
                return path, len(data), "ok"
            except Exception as e:
                err = str(e)
        return path, 0, f"FAIL {err}"

    total, fails = 0, []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (path, n, st) in enumerate(ex.map(one, files), 1):
            total += n
            if st.startswith("FAIL"):
                fails.append(path)
            if i % 100 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  {total / 1e6:.1f} MB", flush=True)
    for p in fails:
        print("  FAIL", p)
    return len(fails)


def mt_pairs_csv(root: Path) -> tuple[int, int]:
    """Magnetic Tile ``MT_<Class>/Imgs/<stem>.jpg + <stem>.png`` → ``pairs.csv``(image,mask,class; MT_Free 는 정상 = 대상 풀).
    반환 (쌍 수, 정상 수). 순수 함수에 가깝다(파일 존재만 본다) — 테스트."""
    rows = ["image,mask,class"]
    normals = 0
    for d in sorted(root.glob("MT_*")):
        cls = d.name[3:].lower()
        imgs = sorted((d / "Imgs").glob("*.jpg"))
        if cls == "free":
            normals = len(imgs)
            continue
        for jpg in imgs:
            png = jpg.with_suffix(".png")
            if png.exists():
                rows.append(
                    f"{jpg.relative_to(root).as_posix()},{png.relative_to(root).as_posix()},{cls}"
                )
    (root / "pairs.csv").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    (root / "normals.txt").write_text(
        "\n".join(
            (root / "MT_Free" / "Imgs" / p.name).relative_to(root).as_posix()
            for p in sorted((root / "MT_Free" / "Imgs").glob("*.jpg"))
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return len(rows) - 1, normals


def fetch_magnetic_tile(out: Path) -> None:
    if not (out / "MT_Blowhole").is_dir():
        subprocess.run(["git", "clone", "--depth", "1", "-q", MT_GIT, str(out)], check=True)
    n, free = mt_pairs_csv(out)
    print(
        f"magnetic-tile: {n} pairs, {free} normals -> {out / 'pairs.csv'} · {out / 'normals.txt'}"
    )


def fetch_visa(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tar = out / "VisA_20220922.tar"
    if not (out / "pcb1").is_dir():
        if not tar.exists():
            print("visa: 1.9 GB 받는 중 …", flush=True)
            urllib.request.urlretrieve(VISA_TAR, tar)
        with tarfile.open(tar) as t:
            t.extractall(out, filter="data")
        tar.unlink()
    print(f"visa: {sorted(p.name for p in out.iterdir() if p.is_dir())}")


def run_import(args: list[str]) -> None:
    print("  $ anograft", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", "anograft", *args], check=True, cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("names", nargs="+", help="MVTec 카테고리 이름 · dtd · magnetic-tile · visa")
    ap.add_argument(
        "--import", dest="do_import", action="store_true", help="받은 뒤 bank/<name> 임포트"
    )
    ap.add_argument("--out", type=Path, default=SAMPLES, help="samples 루트 (기본 repo/samples)")
    a = ap.parse_args(argv)
    fails = 0
    for name in a.names:
        if name in MVTEC_CATEGORIES:
            fails += fetch_hf(MVTEC_REPO, a.out / "mvtec", (f"{name}/",))
            if a.do_import:
                run_import(
                    [
                        "bank",
                        "import-dataset",
                        "mvtec-ad",
                        str(a.out / "mvtec" / name),
                        "--out",
                        f"bank/{name}",
                    ]
                )
        elif name == "dtd":
            fails += fetch_hf(
                DTD_REPO, a.out / "dtd", tuple(f"images/{c}/" for c in DTD_DEFECT_LIKE)
            )
            if a.do_import:
                run_import(
                    [
                        "dataset",
                        "textures",
                        "dtd",
                        str(a.out / "dtd"),
                        "--out",
                        str(a.out / "dtd" / "textures.txt"),
                    ]
                )
        elif name == "magnetic-tile":
            fetch_magnetic_tile(a.out / "magnetic-tile")
            if a.do_import:
                run_import(
                    [
                        "bank",
                        "import-pairs",
                        "--csv",
                        str(a.out / "magnetic-tile" / "pairs.csv"),
                        "--out",
                        "bank/magnetic-tile",
                        "--tags",
                        "magnetic-tile",
                    ]
                )
        elif name == "visa":
            fetch_visa(a.out / "visa")
            if a.do_import:
                run_import(
                    [
                        "bank",
                        "import-dataset",
                        "visa",
                        str(a.out / "visa" / "pcb1"),
                        "--out",
                        "bank/visa-pcb1",
                    ]
                )
        else:
            print(
                f"모르는 이름: {name} (MVTec: {', '.join(MVTEC_CATEGORIES)} · dtd · magnetic-tile · visa)"
            )
            return 2
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

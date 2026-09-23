#!/usr/bin/env python3
"""더미 학습기 어댑터 — 계약(`docs/design/v1.x-training-loop.md` §1)의 **실증이자 테스트 도구**.

학습을 하지 않는다. 대신 계약대로 JSON 을 내고, ``predict`` 에서 **Graft 가 그대로 임포트할 수 있는 형식**
(``masks/<stem>.png`` + ``scores/<stem>.json``)을 만든다. 이것만으로 루프 전체가 torch 없이 단위 테스트된다 —
**더미로 루프가 안 돌면 설계가 틀린 것이다**(§1.6).

`anograft` 를 import 하지 않는다. **stdlib 만** 쓴다(PNG 도 직접 인코딩) — 어댑터가 어떤 언어·어떤 환경에서도
붙는다는 계약의 증거다. 결과는 파일 이름에서 파생돼 **완전히 결정적**이다.

사용::

    python adapters/noop.py info
    python adapters/noop.py fit --dataset <p> --out <model_dir> --seed 7 [--spec spec.json]
    python adapters/noop.py predict --model <model> --images list.txt --out <pred_dir> [--spec spec.json]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

NAME = "noop"
VERSION = "1.0"
FALLBACK_SIZE = (256, 256)


# ---------------------------------------------------------------- PNG (stdlib 만)


def read_png_size(path: Path) -> tuple[int, int]:
    """PNG 헤더에서 (width, height). PNG 가 아니면 기본 크기."""
    try:
        with path.open("rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return FALLBACK_SIZE
            f.read(4)  # IHDR 길이
            if f.read(4) != b"IHDR":
                return FALLBACK_SIZE
            w, h = struct.unpack(">II", f.read(8))
    except OSError:
        return FALLBACK_SIZE
    return (w, h) if w > 0 and h > 0 else FALLBACK_SIZE


def write_gray_png(path: Path, width: int, height: int, rows: list[bytearray]) -> None:
    """8bit 그레이스케일 PNG 를 직접 인코딩한다(의존성 0)."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(r) for r in rows)  # 필터 타입 0
    blob = b"\x89PNG\r\n\x1a\n"
    blob += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    blob += chunk(b"IDAT", zlib.compress(raw, 6))
    blob += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)


# ---------------------------------------------------------------- 결정적 가짜 예측


def fake_box(stem: str, width: int, height: int) -> tuple[int, int, int, int]:
    """파일 이름에서 박스를 파생한다 — 같은 입력이면 언제나 같은 결과."""
    h = zlib.crc32(stem.encode("utf-8"))
    bw = max(8, int(width * (0.08 + (h % 7) / 100)))
    bh = max(8, int(height * (0.08 + ((h >> 3) % 7) / 100)))
    x = int((h >> 6) % max(1, width - bw))
    y = int((h >> 14) % max(1, height - bh))
    return x, y, bw, bh


def fake_score(stem: str) -> float:
    return round(((zlib.crc32(b"score:" + stem.encode("utf-8")) % 1000) / 1000.0), 3)


def ellipse_mask_rows(width: int, height: int, box: tuple[int, int, int, int]) -> list[bytearray]:
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    rx, ry = max(1.0, bw / 2), max(1.0, bh / 2)
    rows: list[bytearray] = []
    for j in range(height):
        row = bytearray(width)
        dy = (j - cy) / ry
        if -1.0 <= dy <= 1.0:
            span = rx * (1.0 - dy * dy) ** 0.5
            lo, hi = max(0, int(cx - span)), min(width, int(cx + span) + 1)
            for i in range(lo, hi):
                row[i] = 255
        rows.append(row)
    return rows


# ---------------------------------------------------------------- verbs


def cmd_info(_: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "name": NAME,
                "version": VERSION,
                "capabilities": ["score", "mask", "box"],
                "dataset_format": "pairs",
                "trains_on": "labeled",
                "deterministic": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spec: dict[str, object] = {}
    if args.spec:
        try:
            spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            spec = {}

    # 학습은 하지 않는다. 데이터셋 크기와 시드에서 '지표'를 결정적으로 만든다.
    n_files = sum(1 for _ in dataset.rglob("*")) if dataset.exists() else 0
    base = 0.30 + min(0.40, n_files / 2000.0)
    bias = float(spec.get("bias", 0.0) or 0.0)  # 테스트에서 challenger 를 만들 때 쓴다
    seed_jitter = (int(args.seed) % 7) / 1000.0

    model = out / "model.json"
    model.write_text(
        json.dumps(
            {"trained_on": str(dataset), "files": n_files, "seed": args.seed, "spec": spec},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[noop] 학습하지 않음 — 파일 {n_files}개 · seed {args.seed}", file=sys.stderr)
    print(
        json.dumps(
            {
                "model": str(model),
                "metrics": {"mAP50": round(base + bias + seed_jitter, 4), "files": n_files},
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    out = Path(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(parents=True, exist_ok=True)

    listing = Path(args.images)
    try:
        images = [
            Path(line.strip())
            for line in listing.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    except OSError as exc:
        print(json.dumps({"error": f"이미지 목록을 읽을 수 없습니다: {exc}"}, ensure_ascii=False))
        return 2

    count = 0
    for img in images:
        stem = img.stem
        width, height = read_png_size(img)
        box = fake_box(stem, width, height)
        write_gray_png(
            out / "masks" / f"{stem}.png", width, height, ellipse_mask_rows(width, height, box)
        )
        (out / "scores" / f"{stem}.json").write_text(
            json.dumps(
                {
                    "image_score": fake_score(stem),
                    "instances": [
                        {
                            "bbox": [box[0], box[1], box[2], box[3]],
                            "score": fake_score(stem),
                            "class": "anomaly",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        count += 1

    print(f"[noop] 가짜 예측 {count}장", file=sys.stderr)
    print(json.dumps({"predictions": str(out), "count": count}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="noop", description="더미 학습기 어댑터 (계약 검증용)")
    sub = ap.add_subparsers(dest="verb", required=True)

    sub.add_parser("info")

    f = sub.add_parser("fit")
    f.add_argument("--dataset", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--spec")

    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--spec")  # 계약: predict 도 --spec 을 받는다(쓰지 않아도 거부하면 안 된다)

    args = ap.parse_args(argv)
    return {"info": cmd_info, "fit": cmd_fit, "predict": cmd_predict}[args.verb](args)


if __name__ == "__main__":
    raise SystemExit(main())

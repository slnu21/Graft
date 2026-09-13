"""샘플 YOLO 데이터셋 생성기 — 보유 데이터가 없을 때 ``import-yolo → run → preview`` 를 끝까지 돌려 보기 위한 합성 세트.

브러시드 메탈 판(밝은 부품, 어두운 배경 — Otsu ROI의 전제)에 세 클래스 결함을 그리고 **YOLO 박스** 라벨을 만든다:
``scratch``(가는 밝은/어두운 선) · ``pit``(어두운 움푹 자국) · ``stain``(반투명 갈색 얼룩). 라벨 박스는 결함 마스크 bbox에
라벨러 흉내로 0~2px 여유를 준다(박스 ≠ 결함 경계 — 박스→마스크 추정이 실제로 일을 하게).

    python tools/make_sample_yolo.py --out samples/metal --n-normal 12 --n-defect 10 --seed 7

출력::

    <out>/images/*.png   <out>/labels/*.txt (정상은 빈 파일 — 일부는 파일 자체가 없음)   <out>/data.yaml

이 스크립트는 저장소 도구이지 패키지가 아니다(전역 난수 금지 규칙은 ``src/``에만 적용). 난수는 ``default_rng(seed)`` 하나.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

NAMES = ["scratch", "pit", "stain"]


def brushed_plate(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """어두운 배경 + 중앙 둥근 사각 금속판(가로 브러시 결). BGR uint8."""
    bg = np.full((h, w, 3), 28, dtype=np.uint8)
    bg = np.clip(bg.astype(np.int16) + rng.normal(0, 3, (h, w, 1)), 0, 255).astype(np.uint8)
    # 판 크기·위치 약간씩 다르게 (ROI 추정을 시험)
    pw, ph = int(w * rng.uniform(0.62, 0.8)), int(h * rng.uniform(0.62, 0.8))
    x0, y0 = (w - pw) // 2 + int(rng.integers(-15, 16)), (h - ph) // 2 + int(rng.integers(-15, 16))
    base = rng.uniform(135, 170)
    tint = np.array([base * rng.uniform(0.98, 1.06), base, base * rng.uniform(0.94, 1.0)])
    noise = rng.normal(0, 1, (ph, pw)).astype(np.float32)
    streak = cv2.GaussianBlur(noise, (0, 0), sigmaX=18, sigmaY=0.8) * 55  # 가로 결
    grain = cv2.GaussianBlur(rng.normal(0, 1, (ph, pw)).astype(np.float32), (0, 0), 0.7) * 6
    yy, xx = np.mgrid[0:ph, 0:pw]
    vignette = 1.0 - 0.18 * (((xx - pw / 2) / (pw / 2)) ** 2 + ((yy - ph / 2) / (ph / 2)) ** 2)
    plate = tint[None, None, :] * vignette[..., None] + (streak + grain)[..., None]
    plate = np.clip(plate, 0, 255).astype(np.uint8)
    mask = np.zeros((ph, pw), dtype=np.uint8)
    r = min(pw, ph) // 8
    cv2.rectangle(mask, (r, 0), (pw - r - 1, ph - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (pw - 1, ph - r - 1), 255, -1)
    for cx, cy in ((r, r), (pw - r - 1, r), (r, ph - r - 1), (pw - r - 1, ph - r - 1)):
        cv2.circle(mask, (cx, cy), r, 255, -1)
    out = bg.copy()
    roi = out[y0 : y0 + ph, x0 : x0 + pw]
    roi[mask > 0] = plate[mask > 0]
    # 판 가장자리 살짝 어두운 모따기
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    roi[edge] = (roi[edge].astype(np.int16) * 0.75).astype(np.uint8)
    return out


def draw_scratch(
    rng: np.random.Generator, img: np.ndarray, plate_box: tuple[int, int, int, int]
) -> np.ndarray:
    x0, y0, x1, y1 = plate_box
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    length = int(rng.uniform(40, 130))
    ang = rng.uniform(0, np.pi)
    cx, cy = rng.uniform(x0 + 40, x1 - 40), rng.uniform(y0 + 40, y1 - 40)
    pts = []
    for t in np.linspace(-0.5, 0.5, 6):
        jitter = rng.normal(0, 2.0)
        px = cx + np.cos(ang) * t * length - np.sin(ang) * jitter
        py = cy + np.sin(ang) * t * length + np.cos(ang) * jitter
        pts.append((int(px), int(py)))
    thickness = int(rng.integers(1, 4))
    bright = rng.random() < 0.5
    color = (235, 235, 240) if bright else (60, 62, 66)
    cv2.polylines(img, [np.array(pts, np.int32)], False, color, thickness, cv2.LINE_AA)
    cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, thickness + 1)
    return mask


def draw_pit(
    rng: np.random.Generator, img: np.ndarray, plate_box: tuple[int, int, int, int]
) -> np.ndarray:
    x0, y0, x1, y1 = plate_box
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = int(rng.uniform(x0 + 30, x1 - 30)), int(rng.uniform(y0 + 30, y1 - 30))
    rx, ry = int(rng.integers(5, 14)), int(rng.integers(5, 14))
    ang = float(rng.uniform(0, 180))
    blob = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(blob, (cx, cy), (rx, ry), ang, 0, 360, 255, -1)
    # 울퉁불퉁하게: 작은 원 몇 개 더
    for _ in range(int(rng.integers(2, 5))):
        cv2.circle(
            blob,
            (cx + int(rng.integers(-rx, rx + 1)), cy + int(rng.integers(-ry, ry + 1))),
            int(rng.integers(2, 6)),
            255,
            -1,
        )
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 3)
    depth = np.clip(dist / max(1.0, dist.max()), 0, 1)
    dark = (1.0 - 0.7 * depth)[..., None]
    region = blob > 0
    img[region] = np.clip(img[region].astype(np.float32) * dark[region], 0, 255).astype(np.uint8)
    # 아래쪽 가장자리에 하이라이트 (움푹 팬 느낌)
    rim = cv2.morphologyEx(blob, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    rim[:cy] = False
    img[rim] = np.clip(img[rim].astype(np.int16) + 45, 0, 255).astype(np.uint8)
    mask[region] = 255
    return mask


def draw_stain(
    rng: np.random.Generator, img: np.ndarray, plate_box: tuple[int, int, int, int]
) -> np.ndarray:
    x0, y0, x1, y1 = plate_box
    h, w = img.shape[:2]
    cx, cy = int(rng.uniform(x0 + 40, x1 - 40)), int(rng.uniform(y0 + 40, y1 - 40))
    blob = np.zeros((h, w), dtype=np.float32)
    for _ in range(int(rng.integers(3, 7))):
        cv2.circle(
            blob,
            (cx + int(rng.integers(-14, 15)), cy + int(rng.integers(-14, 15))),
            int(rng.integers(8, 20)),
            1.0,
            -1,
        )
    blob = cv2.GaussianBlur(blob, (0, 0), 3)
    alpha = np.clip(blob, 0, 1) * rng.uniform(0.35, 0.6)
    color = np.array([40, 75, 120], dtype=np.float32)  # BGR 갈색
    img[:] = np.clip(img * (1 - alpha[..., None]) + color * alpha[..., None], 0, 255).astype(
        np.uint8
    )
    return ((alpha > 0.08).astype(np.uint8)) * 255


DRAW = {"scratch": draw_scratch, "pit": draw_pit, "stain": draw_stain}


def plate_box_of(img: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 90)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def yolo_line(cid: int, mask: np.ndarray, rng: np.random.Generator) -> str | None:
    if not np.any(mask):
        return None
    h, w = mask.shape
    x, y, bw, bh = cv2.boundingRect(mask)
    sl = int(rng.integers(0, 3))  # 라벨러 여유
    x0, y0 = max(0, x - sl), max(0, y - sl)
    x1, y1 = min(w, x + bw + sl), min(h, y + bh + sl)
    cx, cy = (x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h
    return f"{cid} {cx:.6f} {cy:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-normal", type=int, default=12)
    ap.add_argument("--n-defect", type=int, default=10)
    ap.add_argument("--size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--gray-every", type=int, default=0, help="N>0이면 N장마다 한 장을 흑백(1ch)으로 저장"
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    w, h = args.size
    total = args.n_normal + args.n_defect
    n_lines = 0
    for i in range(total):
        img = brushed_plate(rng, w, h)
        lines: list[str] = []
        if i >= args.n_normal:
            box = plate_box_of(img)
            for _ in range(int(rng.integers(1, 4))):
                cid = int(rng.integers(len(NAMES)))
                m = DRAW[NAMES[cid]](rng, img, box)
                ln = yolo_line(cid, m, rng)
                if ln:
                    lines.append(ln)
        stem = f"plate_{i:03d}"
        gray = args.gray_every > 0 and i % args.gray_every == args.gray_every - 1
        to_write = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if gray else img
        ok, buf = cv2.imencode(".png", to_write)
        assert ok
        (out / "images" / f"{stem}.png").write_bytes(buf.tobytes())
        # 정상 이미지: 짝수는 빈 라벨 파일, 홀수는 파일 없음 — 두 경우 모두 "정상"으로 인식돼야 한다
        if lines or i % 2 == 0:
            (out / "labels" / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
        n_lines += len(lines)
    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": ".", "train": "images", "val": "images", "names": NAMES}, sort_keys=False
        ),
        encoding="utf-8",
    )
    print(
        f"{out}: 이미지 {total}장 (정상 {args.n_normal}, 결함 {args.n_defect}, 박스 {n_lines}개) · names {NAMES}"
    )


if __name__ == "__main__":
    main()

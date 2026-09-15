"""학습 스모크 — anograft 출력(YOLO writer)을 기존 YOLO 학습셋에 합쳐 1 epoch 이 "먹는지" 확인한다.

    python tools/train_smoke.py --synthetic out/sample --base samples/metal --out train/merged            # 합치고 1 epoch
    python tools/train_smoke.py --synthetic out/sample --out train/merged --dry-run                        # 합치기만

- ``--synthetic``: ``anograft run`` 출력 루트(``images/``·``labels/``·``data.yaml``). 정상 이미지의 빈 라벨도 그대로 간다.
- ``--base``(선택): 기존 학습셋(같은 레이아웃). **클래스 이름·순서가 같아야** 합친다(다르면 오류 — id 가 어긋난 라벨은 조용한 독).
- 합친 셋: ``<out>/images/{train,val}``·``labels/{train,val}``·``data.yaml``. train = base + synthetic, val = base(실데이터) —
  base 가 없으면 val = synthetic 앞 20%(스모크용, 성능 평가 아님). 파일은 ``syn_``/``base_`` 접두어로 stem 충돌을 피한다.
- 학습은 **ultralytics** 가 있을 때만(``pip install ultralytics`` — 코어 의존성이 아니다, 확인 게이트). 없으면 안내 후 종료 코드 2.
  mAP 비교는 로드맵(v1 이후) — 여기선 "학습 루프가 도는가"만 본다.

순수 로직(``merge_yolo_sets``)은 ``tests/test_train_smoke.py`` 가 고정한다. 2026-09-14 기준 실제 1 epoch 실행은 미검증
(ultralytics 미설치 — 사용자 결정).
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MergeSummary:
    out: Path
    names: tuple[str, ...]
    n_train: int
    n_val: int
    n_labels_empty: int

    def line(self) -> str:
        return (
            f"{self.out}: train {self.n_train}장 · val {self.n_val}장 · 빈 라벨 {self.n_labels_empty} · "
            f"names {list(self.names)}"
        )


def read_names(data_yaml: Path) -> list[str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):  # {0: a, 1: b} 형식
        names = [names[k] for k in sorted(names)]
    if not isinstance(names, list) or not names:
        raise ValueError(f"{data_yaml}: names 가 없습니다")
    return [str(n) for n in names]


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        raise ValueError(f"images 폴더가 없습니다: {images_dir}")
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _copy_pair(
    image: Path, labels_dir: Path, prefix: str, dst_images: Path, dst_labels: Path
) -> bool:
    """이미지 + (있으면) 라벨 복사. 라벨이 없으면 빈 파일(= 정상). 반환: 라벨이 비었는가."""
    stem = f"{prefix}{image.stem}"
    shutil.copy2(image, dst_images / f"{stem}{image.suffix.lower()}")
    src_label = labels_dir / f"{image.stem}.txt"
    text = src_label.read_text(encoding="utf-8") if src_label.exists() else ""
    (dst_labels / f"{stem}.txt").write_text(text, encoding="utf-8")
    return text.strip() == ""


def merge_yolo_sets(
    synthetic: Path, out: Path, *, base: Path | None = None, val_ratio: float = 0.2
) -> MergeSummary:
    """anograft 출력 + (선택) 기존 셋 → ultralytics 가 바로 읽는 ``<out>/data.yaml``."""
    synthetic, out = Path(synthetic), Path(out)
    names = read_names(synthetic / "data.yaml")
    if base is not None:
        base = Path(base)
        base_names = read_names(base / "data.yaml")
        if base_names != names:
            raise ValueError(
                f"클래스 이름/순서가 다릅니다 — synthetic {names} vs base {base_names}. "
                "레시피 output.writer.names_from 또는 은행 클래스 순서를 맞추세요"
            )
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    n_train = n_val = n_empty = 0
    syn_images = list_images(synthetic / "images")
    if not syn_images:
        raise ValueError(f"합성 이미지가 없습니다: {synthetic / 'images'}")
    if base is not None:
        for img in list_images(base / "images"):
            n_empty += _copy_pair(
                img, base / "labels", "base_", out / "images/train", out / "labels/train"
            )
            n_train += 1
            _copy_pair(img, base / "labels", "base_", out / "images/val", out / "labels/val")
            n_val += 1
        val_syn: set[Path] = set()
    else:
        k = max(1, round(len(syn_images) * val_ratio))
        val_syn = set(syn_images[:k])
    for img in syn_images:
        n_empty += _copy_pair(
            img, synthetic / "labels", "syn_", out / "images/train", out / "labels/train"
        )
        n_train += 1
        if img in val_syn:
            _copy_pair(img, synthetic / "labels", "syn_", out / "images/val", out / "labels/val")
            n_val += 1
    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "nc": len(names),
                "names": names,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return MergeSummary(out, tuple(names), n_train, n_val, n_empty)


def train(data_yaml: Path, *, model: str, epochs: int, imgsz: int, project: Path) -> int:
    try:
        from ultralytics import YOLO  # 선택 의존성 — 지연 import
    except ImportError:
        print(
            "ultralytics 가 없습니다. 학습 스모크를 돌리려면:  pip install ultralytics  "
            "(코어 의존성이 아닙니다 — 별도 venv 권장)",
            file=sys.stderr,
        )
        return 2
    results = YOLO(model).train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        device="cpu",
        workers=0,
        batch=4,
        project=str(project),
        name="smoke",
        exist_ok=True,
        plots=False,
        verbose=False,
    )
    print(f"학습 완료: {getattr(results, 'save_dir', project)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])  # type: ignore[union-attr]
    ap.add_argument("--synthetic", required=True, help="anograft run 출력 루트")
    ap.add_argument("--base", default=None, help="기존 YOLO 학습셋(images/·labels/·data.yaml)")
    ap.add_argument("--out", required=True, help="합친 셋 출력 폴더")
    ap.add_argument("--dry-run", action="store_true", help="합치기만 하고 학습은 생략")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args(argv)
    for stream in (
        sys.stdout,
        sys.stderr,
    ):  # cp1252/cp949 콘솔에서 한글 요약이 CLI를 죽이지 않게(anograft.cli 와 같은 규칙)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="replace")
    try:
        summary = merge_yolo_sets(
            Path(args.synthetic), Path(args.out), base=Path(args.base) if args.base else None
        )
    except (ValueError, OSError) as e:  # 잘못된 인자·없는 폴더/파일
        print(f"합치기 실패: {e}", file=sys.stderr)
        return 1
    print(summary.line())
    if args.dry_run:
        return 0
    return train(
        Path(args.out) / "data.yaml",
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=Path(args.out) / "runs",
    )


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""YOLO(ultralytics) 학습기 어댑터 — 계약(`docs/design/v1.x-training-loop.md` §1)의 **첫 실제 구현**.

``tools/train_mvtec_map.py`` 가 in-process 로 하던 학습·평가가 여기로 옮겨 왔다(T3). 그래서 기존 mAP
벤치가 **계약 검증을 겸한다** — 벤치가 돌면 계약이 도는 것이다.

`anograft` 를 import 하지 않는다. **ultralytics 가 있는 별도 venv**(예 ``Workspace/.venvs/graft-train``)에서
실행되고, 코어는 순수 wheel 만 유지한다. 등록은 레시피 밖 ``trainers.yaml``::

    trainers:
      yolo:
        command: ["C:/…/.venvs/graft-train/Scripts/python.exe", "adapters/yolo.py"]
        spec: {model: yolov8n.pt, epochs: 40, imgsz: 640}
        timeout: 43200

사용::

    python adapters/yolo.py info
    python adapters/yolo.py fit --dataset <yolo 데이터셋|data.yaml> --out <model_dir> --seed 7 [--spec spec.json]
    python adapters/yolo.py predict --model <best.pt> --images list.txt --out <pred_dir>

규약 셋:

- **stdout 은 결과 JSON 한 줄뿐.** ultralytics 는 진행을 stdout 에 찍으므로 학습 동안 stdout 을 stderr 로
  돌린다(계약: stderr = 로그). 안 그러면 호출자의 "마지막 JSON 줄" 파싱이 진행 표에 걸린다.
- **``--spec`` 은 불투명**(model·epochs·imgsz·batch·device·workers·conf). Graft 는 해석하지 않고 넘기기만 한다.
- ``predict`` 출력은 **Graft 가 이미 임포트하는 형식**(``scores/<stem>.json`` + seg 모델이면 ``masks/<stem>.png``)
  이라 새 포맷 없이 루프가 닫힌다(§1.4).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

NAME = "yolo"

# --spec 기본값. 벤치(tools/train_mvtec_map.py)가 쓰던 값과 같다.
DEFAULTS: dict[str, Any] = {
    "model": "yolov8n.pt",
    "epochs": 30,
    "imgsz": 320,
    "batch": 16,
    "device": "cpu",
    "workers": 0,
    "conf": 0.25,
}


def load_spec(path: str | None) -> dict[str, Any]:
    """``--spec`` JSON 을 기본값 위에 얹는다. 읽을 수 없으면 기본값(계약은 fail-soft)."""
    spec = dict(DEFAULTS)
    if not path:
        return spec
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[yolo] spec 을 읽지 못해 기본값으로 갑니다: {exc}", file=sys.stderr)
        return spec
    if isinstance(raw, dict):
        spec.update(raw)
    return spec


def data_yaml_of(dataset: Path) -> Path:
    """데이터셋 인자 → ``data.yaml``. 폴더면 그 안의 ``data.yaml`` 을 쓴다."""
    if dataset.is_dir():
        return dataset / "data.yaml"
    return dataset


def is_seg(model: str) -> bool:
    """세그멘테이션 가중치인가 — ``predict`` 가 마스크를 낼 수 있는지 가른다."""
    return "-seg" in Path(model).stem.lower()


def flat_metrics(metrics: Any) -> dict[str, float]:
    """ultralytics 결과 → **평평한 float 맵**. 클래스별은 ``mAP50/<class>``(계약 §1.1)."""
    out: dict[str, float] = {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
    }
    names = getattr(metrics, "names", {}) or {}
    for i, v in zip(metrics.box.ap_class_index, metrics.box.ap50, strict=False):
        out[f"mAP50/{names.get(int(i), int(i))}"] = round(float(v), 4)
    return out


# ---------------------------------------------------------------- verbs


def cmd_info(_: argparse.Namespace) -> int:
    try:
        import ultralytics
    except ImportError as exc:  # 이 venv 에 학습 환경이 없다 → trainer list 가 사유를 보여 준다
        print(f"ultralytics 를 import 할 수 없습니다: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "name": NAME,
                "version": getattr(ultralytics, "__version__", ""),
                # 마스크는 seg 가중치일 때만 나온다 → 선언은 보수적으로(못 하는 걸 약속하지 않는다)
                "capabilities": ["score", "box"],
                "dataset_format": "yolo",
                "trains_on": "labeled",
                # 같은 seed·같은 데이터면 재현된다(BENCHMARKS §2: seed 7 소수 셋째 자리까지).
                # 시드끼리의 요동(±0.04)은 별개라 루프의 승급 판정은 그 노이즈를 쓴다.
                "deterministic": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    spec = load_spec(args.spec)
    data = data_yaml_of(Path(args.dataset))
    if not data.is_file():
        print(f"data.yaml 이 없습니다: {data}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    # ultralytics 는 진행을 stdout 에 찍는다 → 계약상 stdout 은 결과 JSON 전용이므로 stderr 로 돌린다.
    with contextlib.redirect_stdout(sys.stderr):
        model = YOLO(str(spec["model"]))
        res = model.train(
            data=str(data),
            epochs=int(spec["epochs"]),
            imgsz=int(spec["imgsz"]),
            project=str(out.resolve()),  # 상대 경로면 ultralytics 가 runs/detect/ 를 앞에 붙인다
            name="train",
            exist_ok=True,
            seed=int(args.seed),
            deterministic=True,
            workers=int(spec["workers"]),
            verbose=False,
            plots=False,
            val=True,
            device=str(spec["device"]),
            batch=int(spec["batch"]),
        )
        save_dir = Path(getattr(res, "save_dir", out.resolve() / "train"))
        best = save_dir / "weights" / "best.pt"
        metrics = YOLO(str(best)).val(
            data=str(data),
            imgsz=int(spec["imgsz"]),
            split="val",
            device=str(spec["device"]),
            plots=False,
            verbose=False,
            workers=int(spec["workers"]),
        )

    payload = flat_metrics(metrics)
    payload["minutes"] = round((time.perf_counter() - t0) / 60, 1)
    print(json.dumps({"model": str(best), "metrics": payload}, ensure_ascii=False))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    spec = load_spec(args.spec)
    out = Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    seg = is_seg(args.model)
    if seg:
        (out / "masks").mkdir(parents=True, exist_ok=True)

    listing = Path(args.images)
    try:
        images = [
            Path(line.strip())
            for line in listing.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
    except OSError as exc:
        print(f"이미지 목록을 읽을 수 없습니다: {exc}", file=sys.stderr)
        return 2

    count = 0
    with contextlib.redirect_stdout(sys.stderr):
        model = YOLO(str(args.model))
        for img in images:
            res = model.predict(
                source=str(img),
                imgsz=int(spec["imgsz"]),
                conf=float(spec["conf"]),
                device=str(spec["device"]),
                verbose=False,
            )[0]
            names = res.names or {}
            instances = []
            for box in res.boxes or []:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                instances.append(
                    {
                        "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                        "score": round(float(box.conf[0]), 4),
                        "class": str(names.get(int(box.cls[0]), int(box.cls[0]))),
                    }
                )
            (out / "scores" / f"{img.stem}.json").write_text(
                json.dumps(
                    {
                        # 이미지 점수 = 가장 확신하는 인스턴스(지도 모델의 관례). 없으면 0.
                        "image_score": max((i["score"] for i in instances), default=0.0),
                        "instances": instances,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            if seg and getattr(res, "masks", None) is not None:
                write_mask_png(out / "masks" / f"{img.stem}.png", res)
            count += 1

    print(
        f"[yolo] 예측 {count}장 · 마스크 {'예' if seg else '아니오(detect 가중치)'}",
        file=sys.stderr,
    )
    print(json.dumps({"predictions": str(out), "count": count}, ensure_ascii=False))
    return 0


def write_mask_png(path: Path, res: Any) -> None:
    """seg 결과의 인스턴스 마스크를 합쳐 0/255 PNG 로 — ``bank import-pairs`` 가 그대로 받는 형식."""
    import cv2  # ultralytics 가 이미 의존한다(이 어댑터의 venv 안)
    import numpy as np

    h, w = res.orig_shape
    acc = np.zeros((h, w), dtype=np.uint8)
    for m in res.masks.data.cpu().numpy():
        resized = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        acc[resized > 0.5] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", acc)[1].tobytes())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="yolo", description="YOLO(ultralytics) 학습기 어댑터")
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
    p.add_argument("--spec")

    args = ap.parse_args(argv)
    try:
        return {"info": cmd_info, "fit": cmd_fit, "predict": cmd_predict}[args.verb](args)
    except ImportError as exc:  # 학습 환경이 없다 — 계약은 종료 코드로 말한다(fail-soft)
        print(f"ultralytics 를 import 할 수 없습니다: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

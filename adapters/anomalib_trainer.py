#!/usr/bin/env python3
"""anomalib 학습기 어댑터 — **비지도**(정상만 학습) 계열을 계약에 붙인다(설계 §1·§5, T2).

`adapters/yolo.py` 와 결정적으로 다른 점 하나: **`trains_on: "normal_only"`**. PatchCore·PaDiM·FastFlow 는
정상 이미지만으로 학습하므로 **합성 결함을 학습셋에 넣으면 오염**이다. 루프는 이 선언을 보고 합성을
평가에만 쓴다(§6.4). 선언이 없으면 그 오염이 조용히 일어난다.

`anograft` 를 import 하지 않는다. anomalib 이 있는 별도 venv 에서 돈다(코어는 순수 wheel 유지)::

    trainers:
      anomalib:                      # 등록 이름은 anomalib — 파일 이름과 무관하다
        command: ["C:/…/.venvs/graft-train/Scripts/python.exe", "adapters/anomalib_trainer.py"]
        spec: {model: patchcore, image_size: 256}
        timeout: 43200

**파일 이름이 ``anomalib.py`` 면 안 된다.** 스크립트로 돌면 그 폴더가 ``sys.path`` 앞에 붙어 *자기 자신*이
진짜 패키지를 가린다(``No module named 'anomalib.engine'; 'anomalib' is not a package``). 어댑터 파일은
자기가 import 하는 패키지와 이름이 겹치면 안 된다.

verb::

    python adapters/anomalib_trainer.py info
    python adapters/anomalib_trainer.py fit --dataset <mvtec 레이아웃 카테고리 폴더> --out <dir> --seed 7 [--spec s.json]
    python adapters/anomalib_trainer.py predict --model <ckpt> --images list.txt --out <pred_dir> [--spec s.json]

``--dataset`` 은 **카테고리 폴더**(`<root>/<category>/{train/good, test/…, ground_truth/…}`)를 가리킨다 —
Graft 의 `mvtec` writer 출력이 그대로 들어온다. 루트·카테고리는 여기서 갈라 `MVTecAD` 에 넘긴다.

규약 둘(T3 에서 정한 것 그대로):

- **stdout 은 결과 JSON 한 줄뿐.** lightning 이 진행 막대를 stdout 에 찍으므로 전 구간을 stderr 로 돌린다.
- ``fit`` 의 ``metrics`` 는 **평평한 float 맵**(`image_AUROC` · `pixel_AUROC` …) — anomalib 의 지표 이름을
  그대로 쓰되 `/` 는 `_` 로 바꾼다(`/` 는 계약에서 '클래스별' 구분자다).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

NAME = "anomalib"

DEFAULTS: dict[str, Any] = {
    "model": "patchcore",
    "image_size": 256,
    "batch_size": 8,
    "num_workers": 0,  # Windows spawn — 0 이 안전하고 CPU 에선 차이도 작다
    "accelerator": "cpu",
    "max_epochs": 1,  # PatchCore·PaDiM 은 1 에폭 메모리 적재형
    "threshold": 0.5,  # predict 에서 이상맵 → 0/1 마스크
}

# 이름 → 클래스. 비지도(정상만 학습) 계열만 둔다 — 그게 이 어댑터의 선언(trains_on)이다.
MODELS = {
    "patchcore": ("Patchcore", {"coreset_sampling_ratio": 0.1}),
    "padim": ("Padim", {}),
    "fastflow": ("Fastflow", {}),
    "cfa": ("Cfa", {}),
    "dfkde": ("Dfkde", {}),
    "dfm": ("Dfm", {}),
}


def load_spec(path: str | None) -> dict[str, Any]:
    spec = dict(DEFAULTS)
    if not path:
        return spec
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[anomalib] spec 을 읽지 못해 기본값으로 갑니다: {exc}", file=sys.stderr)
        return spec
    if isinstance(raw, dict):
        spec.update(raw)
    return spec


def flat_metrics(raw: Any) -> dict[str, float]:
    """``engine.test`` 결과(리스트 of dict) → 평평한 float 맵. ``/`` 는 계약의 클래스 구분자라 ``_`` 로."""
    out: dict[str, float] = {}
    items = raw if isinstance(raw, list) else [raw]
    for entry in items:
        if not isinstance(entry, dict):
            continue
        for k, v in entry.items():
            try:
                out[str(k).replace("/", "_")] = round(float(v), 4)
            except (TypeError, ValueError):
                continue
    return out


def build_model(spec: dict[str, Any]) -> Any:
    """spec 의 ``model`` 이름 → anomalib 모델. 모르는 이름은 **사유를 주고 거절**한다.

    이름 검사는 import 보다 **먼저** — 오타는 anomalib 이 없는 환경에서도 걸려야 한다.
    """
    name = str(spec["model"]).lower()
    if name not in MODELS:
        raise SystemExit(
            f"모르는 model 입니다: {name} (가능: {', '.join(sorted(MODELS))}) — "
            "지도 학습 모델은 이 어댑터가 아니라 adapters/yolo.py 입니다"
        )
    import anomalib.models as M

    cls_name, kwargs = MODELS[name]
    cls = getattr(M, cls_name)
    size = int(spec["image_size"])
    return cls(pre_processor=cls.configure_pre_processor(image_size=(size, size)), **kwargs)


def build_datamodule(dataset: Path, spec: dict[str, Any]) -> Any:
    """``<root>/<category>`` 를 갈라 ``MVTecAD`` 로 — Graft `mvtec` writer 출력이 그대로 들어온다."""
    from anomalib.data import MVTecAD

    return MVTecAD(
        root=str(dataset.parent),
        category=dataset.name,
        train_batch_size=int(spec["batch_size"]),
        eval_batch_size=int(spec["batch_size"]),
        num_workers=int(spec["num_workers"]),
    )


def original_size(path: Path) -> tuple[int, int] | None:
    """원본 이미지의 ``(h, w)``. 한글 경로에서도 안전하게(``imread`` 는 Windows 에서 조용히 실패한다)."""
    import cv2
    import numpy as np

    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return None if img is None else (int(img.shape[0]), int(img.shape[1]))


# ---------------------------------------------------------------- verbs


def cmd_info(_: argparse.Namespace) -> int:
    try:
        import anomalib
    except ImportError as exc:
        print(f"anomalib 을 import 할 수 없습니다: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "name": NAME,
                "version": getattr(anomalib, "__version__", ""),
                "capabilities": ["score", "mask"],
                "dataset_format": "mvtec",
                # 이 한 줄이 이 어댑터의 안전장치다 — 합성 결함은 학습이 아니라 평가로만 간다(§6.4).
                "trains_on": "normal_only",
                # 백본 초기화·coreset 샘플링에 무작위가 있다 → 루프는 시드 여러 개로 평균 낸다.
                "deterministic": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    dataset = Path(args.dataset)
    if not (dataset / "train" / "good").is_dir():
        print(
            f"train/good 이 없습니다: {dataset} (mvtec 레이아웃의 카테고리 폴더를 주세요)",
            file=sys.stderr,
        )
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with contextlib.redirect_stdout(sys.stderr):  # lightning 진행 막대가 stdout 에 나간다
        from anomalib.engine import Engine
        from lightning.pytorch import seed_everything

        seed_everything(int(args.seed), workers=True)
        model = build_model(spec)
        datamodule = build_datamodule(dataset, spec)
        engine = Engine(
            default_root_dir=str(out.resolve()),
            accelerator=str(spec["accelerator"]),
            devices=1,
            max_epochs=int(spec["max_epochs"]),
            logger=False,
            enable_progress_bar=False,
        )
        engine.fit(model=model, datamodule=datamodule)
        results = engine.test(model=model, datamodule=datamodule)
        ckpt = engine.trainer.checkpoint_callback
        model_path = getattr(ckpt, "best_model_path", "") or ""
        if not model_path:  # 체크포인트 콜백이 꺼져 있으면 직접 저장한다(모델 참조는 불투명 문자열)
            model_path = str(out / "model.ckpt")
            engine.trainer.save_checkpoint(model_path)

    metrics = flat_metrics(results)
    print(json.dumps({"model": str(model_path), "metrics": metrics}, ensure_ascii=False))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
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
        print(f"이미지 목록을 읽을 수 없습니다: {exc}", file=sys.stderr)
        return 2

    count = 0
    with contextlib.redirect_stdout(sys.stderr):
        import cv2
        import numpy as np
        from anomalib.engine import Engine

        model = build_model(spec)
        engine = Engine(
            default_root_dir=str(out.resolve()),
            accelerator=str(spec["accelerator"]),
            devices=1,
            logger=False,
            enable_progress_bar=False,
        )
        thr = float(spec["threshold"])
        for img in images:
            preds = engine.predict(model=model, data_path=str(img), ckpt_path=str(args.model))
            if not preds:
                continue
            item = preds[0]
            score = float(np.asarray(getattr(item, "pred_score", 0.0)).reshape(-1)[0])
            amap = getattr(item, "anomaly_map", None)
            if amap is not None:
                arr = np.asarray(amap).squeeze().astype(np.float32)
                # 이상맵은 **모델 입력 해상도**(spec.image_size)로 나온다 → 원본 크기로 되돌린다.
                # 안 되돌리면 `bank import-pairs` 가 이미지와 크기가 다른 마스크를 받는다(루프가 안 닫힌다).
                hw = original_size(img)
                if hw and arr.shape[:2] != hw:
                    arr = cv2.resize(arr, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
                mask = (arr >= thr).astype(
                    np.uint8
                ) * 255  # 연속 맵을 키운 뒤 이진화(회색값이 안 생긴다)
                (out / "masks" / f"{img.stem}.png").write_bytes(
                    cv2.imencode(".png", mask)[1].tobytes()
                )
            (out / "scores" / f"{img.stem}.json").write_text(
                json.dumps({"image_score": round(score, 4), "instances": []}, ensure_ascii=False),
                encoding="utf-8",
            )
            count += 1

    print(f"[anomalib] 예측 {count}장 (이상맵 임계 {spec['threshold']})", file=sys.stderr)
    print(json.dumps({"predictions": str(out), "count": count}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="anomalib", description="anomalib 학습기 어댑터(비지도)")
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
    p.add_argument("--spec")  # 계약: predict 도 --spec 을 받는다

    args = ap.parse_args(argv)
    try:
        return {"info": cmd_info, "fit": cmd_fit, "predict": cmd_predict}[args.verb](args)
    except ImportError as exc:
        print(f"anomalib 환경이 아닙니다: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

<div align="center">

# Graft

**라벨링한 실제 결함을 정상 이미지에 이식해, 정답 마스크와 재현 정보가 붙은 학습용 이상 데이터셋을 만듭니다.**

패키지·CLI 이름: `anograft` · 전 과정 오프라인·로컬 · CPU만으로 동작

</div>

> **상태: v0.1 개발 중.** CLI 코어(7단계 파이프라인 + 레시피 + YOLO 입출력)부터 만들고 GUI는 그 위에 올립니다.

## 왜

이상 탐지·결함 검출 모델은 결함 이미지가 늘 부족합니다. 합성 기법(CutPaste · DRAEM · NSA · 확산 인페인팅)은 이미 여럿 나와 있지만 논문 저장소마다 흩어져 있고, 그 사이의 실무가 비어 있습니다 — 결함을 **모아 두는 곳**, **어디에 붙일지**의 통제, **정답 마스크 정책**, **재현성**, 학습 파이프라인이 **바로 먹는 출력 형식**.

Graft는 알고리즘을 새로 만드는 도구가 아니라 그 사이를 메우는 도구입니다.

- **결함 은행** — 보유 YOLO 라벨(박스·폴리곤)이나 마스크 PNG에서 결함을 모읍니다. 박스만 있으면 마스크를 추정합니다(GrabCut 등). 데이터가 없으면 표준 산업 데이터셋(MVTec AD 등)을 로컬 사본에서 읽어 시험합니다.
- **7단계 파이프라인** `소스 → 기하 → 배치 → 블렌딩 → 조화 → 열화 → 정답 마스크` — 알고리즘은 각 단계의 `method`로 고릅니다(블렌딩: paste · alpha · Poisson · multiband, 조화: stats · Reinhard · 히스토그램 매칭 …). 프리셋으로 시작하고 필요할 때만 펼칩니다.
- **배치 허용 영역(ROI)** 이 기본값 — 배경에 붙은 결함은 학습에 해롭습니다.
- **재현** — 레시피(YAML) + 시드가 같으면 워커 수와 무관하게 바이트 단위로 같은 데이터셋. 이미지마다 사이드카 JSON(소스 id · 변환 · 좌표 · 시드 · 파이프라인 해시).
- **출력** — 정본은 이미지 + GT 마스크 + 사이드카. 그 위에 writer가 학습 형식을 덧붙입니다(v0.1: YOLO `labels/*.txt` + `data.yaml` — 기존 학습셋에 그대로 합칠 수 있음).

## 시작하기

요구사항은 **Python 3.10 이상** 하나입니다. 의존성은 순수 wheel(numpy · opencv-python-headless · pydantic · pyyaml)이라 컴파일러·GPU 없이 설치됩니다.

```powershell
git clone https://github.com/slnu21/Graft.git
cd Graft
.\bootstrap.ps1            # Windows  (Linux/macOS: ./bootstrap.sh)
.venv\Scripts\Activate.ps1
anograft --help
```

### 5분 시작 (초안 — v0.1 릴리스에서 다듬음)

보유 데이터가 없어도 됩니다. 샘플 YOLO 세트(브러시드 메탈 + scratch/pit/stain)로 끝까지 한 바퀴:

```powershell
python tools/make_sample_yolo.py --out samples/metal                     # 샘플 이미지 + YOLO 라벨 (cp949 콘솔이면 PYTHONIOENCODING=utf-8)
anograft bank import-yolo --images samples/metal/images --labels samples/metal/labels `
    --names samples/metal/data.yaml --out bank/sample --list-normals samples/metal/normals.txt
anograft bank ls bank/sample                                             # 클래스별 소스 수 · 마스크 출처(정확/추정)
anograft bank preview bank/sample --out out/bank-preview.png             # 추정 마스크를 눈으로 (amber = 추정, ellipse = 과라벨)
anograft run recipes/sample-poisson.yaml --workers 4                     # → out/sample/{images,masks,meta,labels,data.yaml,manifest.csv}
anograft preview recipes/sample-poisson.yaml --index 0 --compare-methods blend --out out/compare.png
```

**보유 YOLO 라벨**이 있으면 `import-yolo`에 그 폴더를, **마스크 PNG 쌍**이면 `bank import-pairs --images … --masks … --class scratch`, **표준셋**이면 `anograft dataset info mvtec-ad` → 로컬 사본을 `bank import-dataset mvtec-ad <root>/<category> --out bank/<category>`. 출력 `images/`·`labels/`·`data.yaml`은 기존 YOLO 학습셋에 그대로 합쳐집니다(같은 `names` 순서 확인). GUI: `pip install -e ".[gui]"` 후 `python -m anograft.gui recipes/sample-poisson.yaml`.

## 개발

```powershell
pip install -e ".[dev]"
pytest
ruff check . ; ruff format --check .
```

구조·규약은 `CLAUDE.md`, 진행은 `docs/TASKS.md`, v0.1 설계는 `docs/design/v0.1-core.md` (docs는 로컬 컨텍스트).

## 라이선스

MIT © 2026 slnu21

외부 데이터셋(MVTec AD 등)·모델 가중치는 각자의 라이선스를 따르며, Graft는 재배포하지 않고 로컬 사본을 읽기만 합니다.

---

<div align="center">

## English

</div>

**Graft real, labeled defects onto normal images to build training datasets for anomaly detection — with ground-truth masks and full reproducibility metadata.** Package/CLI: `anograft`. Fully offline, CPU-only.

> **Status: v0.1 in development.** CLI core first (7-stage pipeline + recipes + YOLO in/out); GUI comes on top.

## Why

Synthetic-defect methods (CutPaste, DRAEM, NSA, diffusion inpainting) exist, but they live in scattered paper repos and the practical glue is missing: a place to **collect** defects, control over **where** they land, a **ground-truth mask policy**, **reproducibility**, and output your training pipeline can **consume directly**. Graft fills that gap instead of inventing another algorithm.

- **Defect bank** — import from your YOLO labels (boxes/polygons) or mask PNGs; boxes get a pixel mask estimated (GrabCut etc.). No data? Point it at a local copy of a standard industrial dataset (MVTec AD, …).
- **7-stage pipeline** `source → geometry → placement → blend → harmonize → degrade → gt-mask` — pick algorithms per stage via `method` (blend: paste · alpha · Poisson · multiband; harmonize: stats · Reinhard · histogram matching …). Start from a preset, unfold only what you need.
- **Placement ROI is on by default** — defects pasted onto background hurt training.
- **Reproducible** — same recipe (YAML) + seed ⇒ byte-identical dataset regardless of worker count. Per-image sidecar JSON (source id, transform, coordinates, seed, pipeline hash).
- **Output** — canonical image + GT mask + sidecar, plus a writer layer for training formats (v0.1: YOLO `labels/*.txt` + `data.yaml`, mergeable into your existing set).

## Getting started

Only requirement: **Python ≥ 3.10**. All dependencies are pure wheels — no compiler, no GPU.

```sh
git clone https://github.com/slnu21/Graft.git
cd Graft
./bootstrap.sh              # Windows: .\bootstrap.ps1
source .venv/bin/activate
anograft --help
```

## Development

```sh
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

## License

MIT © 2026 slnu21. External datasets and model weights keep their own licenses; Graft never redistributes them.

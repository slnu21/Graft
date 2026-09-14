<div align="center">

# Graft

**라벨링한 실제 결함을 정상 이미지에 이식해, 정답 마스크와 재현 정보가 붙은 학습용 이상 데이터셋을 만듭니다.**

패키지·CLI 이름: `anograft` · 전 과정 오프라인·로컬 · CPU만으로 동작 · Windows zip은 파이썬 없이 실행

[English below ↓](#english)

</div>

![원본 | 합성 | GT](assets/preview.png)

> **상태: v0.1.0** — CLI 코어(7단계 파이프라인 · 레시피 · 결함 은행 · YOLO 입출력 · 프리셋 4종)와 GUI 스튜디오 탭. 샘플 데이터로 엔드투엔드 검증. **보유 실데이터·표준셋(MVTec AD)·학습 1 epoch은 아직 미검증**(데이터가 생기면 아래 3줄로 확인).

## 왜

이상 탐지·결함 검출 모델은 결함 이미지가 늘 부족합니다. 합성 기법(CutPaste · DRAEM · NSA · 확산 인페인팅)은 이미 여럿 나와 있지만 논문 저장소마다 흩어져 있고, 그 사이의 실무가 비어 있습니다 — 결함을 **모아 두는 곳**, **어디에 붙일지**의 통제, **정답 마스크 정책**, **재현성**, 학습 파이프라인이 **바로 먹는 출력 형식**.

Graft는 알고리즘을 새로 만드는 도구가 아니라 그 사이를 메우는 도구입니다.

- **결함 은행** — 보유 YOLO 라벨(박스·폴리곤)이나 마스크 PNG에서 결함을 모읍니다. 박스만 있으면 마스크를 추정합니다(GrabCut 등, 출처를 `mask_origin`으로 끌고 다님). 데이터가 없으면 표준 산업 데이터셋(MVTec AD)을 로컬 사본에서 읽습니다.
- **7단계 파이프라인** `소스 → 기하 → 배치 → 블렌딩 → 조화 → 열화 → 정답 마스크` — 알고리즘은 각 단계의 `method`로 고릅니다(블렌딩: paste · alpha · Poisson · multiband, 조화: stats · Reinhard · 히스토그램 매칭). 프리셋으로 시작하고 필요할 때만 펼칩니다.
- **배치 허용 영역(ROI)** 이 기본값 — 배경에 붙은 결함은 학습에 해롭습니다.
- **재현** — 레시피(YAML) + 시드가 같으면 워커 수와 무관하게 바이트 단위로 같은 데이터셋. 이미지마다 사이드카 JSON(소스 id · 변환 · 좌표 · 시드 · 파이프라인 해시).
- **출력** — 정본은 이미지 + GT 마스크 + 사이드카 + `manifest.csv`. 그 위에 writer가 학습 형식을 덧붙입니다(v0.1: YOLO `labels/*.txt` + `data.yaml` — 기존 학습셋에 그대로 합침).

## 설치

| 방법 | 언제 | 명령 |
|---|---|---|
| **Windows zip** (권장) | 파이썬 없는 PC, 현장 | [Releases](https://github.com/slnu21/Graft/releases)에서 `anograft-<ver>-win64.zip` → 풀기 → 그 폴더에서 `.\anograft.exe`(CLI) · `.\anograft-gui.exe`(GUI). 설치·관리자 권한 없음 |
| **pip / uv** | 파이썬 3.10+ 이 있는 PC, Linux/macOS | `pip install "anograft[gui] @ git+https://github.com/slnu21/Graft.git"` 또는 `uv tool install "anograft[gui] @ git+https://github.com/slnu21/Graft.git"`(uv가 파이썬까지 받아 줌). `[gui]`를 빼면 CLI만(순수 wheel 4개). PyPI 등록은 예정 |
| **소스** | 개발 | 아래 [개발](#개발) |

의존성은 numpy · opencv-python-headless · pydantic · pyyaml(전부 순수 wheel — 컴파일러·GPU 불필요) + GUI는 PySide6(LGPL, 동적 링크).

## 5분 시작

보유 데이터가 없어도 됩니다. 샘플 YOLO 세트(브러시드 메탈 + scratch/pit/stain)로 끝까지 한 바퀴. zip을 푼 폴더(또는 repo 루트)에서:

```powershell
# zip 이면 anograft → .\anograft.exe
anograft sample --out samples/metal                                       # 샘플 이미지 22장 + YOLO 박스 라벨
anograft bank import-yolo --images samples/metal/images --labels samples/metal/labels `
    --names samples/metal/data.yaml --out bank/sample --list-normals samples/metal/normals.txt
anograft bank ls bank/sample                                              # 클래스별 소스 수 · 마스크 출처(정확/추정)
anograft bank preview bank/sample --out out/bank-preview.png              # 추정 마스크를 눈으로 (amber = 추정, ellipse = 과라벨)
anograft run recipes/sample-poisson.yaml --workers 4                      # → out/sample/{images,masks,meta,labels,data.yaml,manifest.csv}
anograft preview recipes/sample-poisson.yaml --index 0 --compare-methods blend --out out/compare.png
anograft-gui recipes/sample-poisson.yaml                                  # GUI (zip: .\anograft-gui.exe · pip: python -m anograft.gui)
```

`anograft methods`가 스테이지별 선택지와 가용 여부를, `anograft recipe init --preset <이름> --write my.yaml`이 프리셋을 펼친 레시피를 줍니다. 레시피 상대경로는 **현재 폴더 기준**입니다.

![GUI 스튜디오](assets/gui-studio.png)

## 보유 데이터로

| 가진 것 | 명령 |
|---|---|
| YOLO 라벨(`images/`·`labels/*.txt`·`data.yaml`) | `anograft bank import-yolo --images … --labels … --names data.yaml --out bank/mine --list-normals normals.txt` (빈 라벨 이미지 = 정상 후보) |
| 이미지 + 마스크 PNG 쌍 | `anograft bank import-pairs --images … --masks … --class scratch --out bank/mine` (`--class-from-dir` · `--csv`) |
| MVTec AD 로컬 사본 | `anograft dataset info mvtec-ad` → `anograft bank import-dataset mvtec-ad <root>/metal_nut --out bank/metal_nut` (내려받지 않음, CC BY-NC-SA) |

그다음은 `recipe init` → `inputs.bank`·`inputs.targets`(정상 이미지 폴더 또는 목록) 수정 → `run`. 출력 `images/`·`labels/`·`data.yaml`은 기존 YOLO 학습셋에 그대로 합쳐집니다(같은 `names` 순서). `python tools/train_smoke.py --synthetic out/sample --base <기존셋> --out train/merged`가 합쳐서 `ultralytics` 1 epoch을 돌립니다(ultralytics는 별도 설치, `--dry-run`은 합치기만).

`bank ls`의 `est`·origins 열에서 `ellipse` 폴백 비율이 높으면(가늘고 희미한 스크래치) `--mask-from otsu`나 `--min-box`를 조정하세요 — 박스는 결함 경계가 아닙니다.

## 프리셋

같은 시드·같은 대상에 프리셋 4종(`preview --compare-methods blend|harmonize`로 스테이지별 비교도 가능):

![프리셋 4종](assets/presets-4x.png)

| 프리셋 | 계열 | 블렌딩 · 조화 | 쓰임 |
|---|---|---|---|
| `poisson-graft` (기본) | NSA | Poisson(normal) · stats 0.3 | 대부분의 결함. 얼룩처럼 그래디언트가 약한 결함도 살린다 |
| `multiband-graft` | 라플라시안 피라미드 | multiband · histmatch 0.3 | 텍스처 보존이 좋고 경계 halo가 덜함 |
| `alpha-paste` | 페더 합성 | alpha(feather 2) · Reinhard 0.5 | 빠름, 경계 색 정합 |
| `hard-paste` | CutPaste | paste · 없음 | 가장 거친 대조군(학습 실험용) |

기본값은 샘플 은행에서 "결함이 옅어지는 정도"(hard-paste 대비 마스크 안 L1 비율)를 재서 정했습니다 — 세 조화 방법 모두 정의상 결함 톤을 대상 쪽으로 당기므로 strength를 낮게 뒀습니다. 실데이터 학습 mAP 근거는 아직 없습니다(로드맵).

**재현 보증 범위**: 같은 OS · 같은 OpenCV 부버전에서 바이트 동일. 다른 환경에서는 `cv2.seamlessClone` 내부 솔버 차이로 픽셀 단위 차이가 있을 수 있습니다. zip은 OpenCV를 함께 실으므로 zip끼리는 동일합니다.

## 개발

```powershell
git clone https://github.com/slnu21/Graft.git ; cd Graft
.\bootstrap.ps1 -Gui             # venv + pip install -e ".[dev,gui]"   (Linux/macOS: ./bootstrap.sh)
.venv\Scripts\Activate.ps1
pytest ; ruff check . ; ruff format --check .
.\bootstrap.ps1 -Build ; .\tools\build_zip.ps1   # Windows zip (PyInstaller) → dist/anograft-<ver>-win64.zip
```

구조·규약은 `CLAUDE.md`, 진행은 `docs/TASKS.md`, v0.1 설계는 `docs/design/v0.1-core.md`(docs는 로컬 컨텍스트). 새 알고리즘 = 스테이지 클래스 1 + 레지스트리 1줄 + 프리셋 YAML 1장.

## 라이선스

MIT © 2026 slnu21 — `LICENSE`. 함께 배포되는 구성 요소(PySide6/Qt LGPL 등)는 `THIRD-PARTY-NOTICES.md`, 오프라인·로컬 원칙은 `PRIVACY.md`.

외부 데이터셋(MVTec AD 등)·모델 가중치는 각자의 라이선스를 따르며, Graft는 재배포하지 않고 로컬 사본을 읽기만 합니다.

---

<div align="center">

## English

</div>

**Graft real, labeled defects onto normal images to build training datasets for anomaly detection — with ground-truth masks and full reproducibility metadata.** Package/CLI: `anograft`. Fully offline, CPU-only; the Windows zip needs no Python.

> **Status: v0.1.0** — CLI core (7-stage pipeline, recipes, defect bank, YOLO in/out, 4 presets) plus the GUI Studio tab. Verified end to end on the bundled sample set. **Not yet verified on real customer data, MVTec AD, or a training epoch** — three commands once you have data (below).

### Why

Synthetic-defect methods (CutPaste, DRAEM, NSA, diffusion inpainting) exist, but they live in scattered paper repos and the practical glue is missing: a place to **collect** defects, control over **where** they land, a **ground-truth mask policy**, **reproducibility**, and output your training pipeline can **consume directly**. Graft fills that gap instead of inventing another algorithm.

- **Defect bank** — import from your YOLO labels (boxes/polygons) or mask PNGs; boxes get a pixel mask estimated (GrabCut etc., provenance kept as `mask_origin`). No data? Point it at a local copy of MVTec AD.
- **7-stage pipeline** `source → geometry → placement → blend → harmonize → degrade → gt-mask` — pick algorithms per stage via `method` (blend: paste · alpha · Poisson · multiband; harmonize: stats · Reinhard · histogram matching). Start from a preset, unfold only what you need.
- **Placement ROI is on by default** — defects pasted onto background hurt training.
- **Reproducible** — same recipe (YAML) + seed ⇒ byte-identical dataset regardless of worker count. Per-image sidecar JSON (source id, transform, coordinates, seed, pipeline hash).
- **Output** — canonical image + GT mask + sidecar + `manifest.csv`, plus a writer layer for training formats (v0.1: YOLO `labels/*.txt` + `data.yaml`, mergeable into your existing set).

### Install

| Method | When | Command |
|---|---|---|
| **Windows zip** (recommended) | No Python, shop-floor PCs | Grab `anograft-<ver>-win64.zip` from [Releases](https://github.com/slnu21/Graft/releases), unzip, run `.\anograft.exe` (CLI) / `.\anograft-gui.exe` (GUI) from that folder. No installer, no admin rights |
| **pip / uv** | Python ≥ 3.10, Linux/macOS | `pip install "anograft[gui] @ git+https://github.com/slnu21/Graft.git"` or `uv tool install "anograft[gui] @ git+https://github.com/slnu21/Graft.git"` (uv fetches Python for you). Drop `[gui]` for CLI-only (four pure wheels). PyPI listing planned |
| **Source** | Development | see [Development](#development) |

Dependencies: numpy · opencv-python-headless · pydantic · pyyaml (all pure wheels — no compiler, no GPU); the GUI adds PySide6 (LGPL, dynamically linked).

### Five-minute start

No data needed — the bundled sample set (brushed metal + scratch/pit/stain) runs the whole loop. From the unzipped folder (or the repo root):

```powershell
# zip: anograft → .\anograft.exe
anograft sample --out samples/metal
anograft bank import-yolo --images samples/metal/images --labels samples/metal/labels `
    --names samples/metal/data.yaml --out bank/sample --list-normals samples/metal/normals.txt
anograft bank ls bank/sample
anograft bank preview bank/sample --out out/bank-preview.png              # eyeball estimated masks (amber = estimated)
anograft run recipes/sample-poisson.yaml --workers 4                      # → out/sample/{images,masks,meta,labels,data.yaml,manifest.csv}
anograft preview recipes/sample-poisson.yaml --index 0 --compare-methods blend --out out/compare.png
anograft-gui recipes/sample-poisson.yaml                                  # GUI (zip: .\anograft-gui.exe · pip: python -m anograft.gui)
```

`anograft methods` lists per-stage choices and availability; `anograft recipe init --preset <name> --write my.yaml` expands a preset. Relative paths in recipes resolve against the **current directory**.

### Your own data

| You have | Command |
|---|---|
| YOLO labels (`images/`, `labels/*.txt`, `data.yaml`) | `anograft bank import-yolo --images … --labels … --names data.yaml --out bank/mine --list-normals normals.txt` (images with empty labels become normal candidates) |
| Image + mask PNG pairs | `anograft bank import-pairs --images … --masks … --class scratch --out bank/mine` (`--class-from-dir`, `--csv`) |
| Local MVTec AD copy | `anograft dataset info mvtec-ad` → `anograft bank import-dataset mvtec-ad <root>/metal_nut --out bank/metal_nut` (never downloaded; CC BY-NC-SA) |

Then `recipe init` → edit `inputs.bank` / `inputs.targets` (normal-image folder or list) → `run`. The output `images/`, `labels/`, `data.yaml` merge straight into an existing YOLO set (same `names` order). `python tools/train_smoke.py --synthetic out/sample --base <your set> --out train/merged` merges and runs one `ultralytics` epoch (install ultralytics separately; `--dry-run` only merges).

If `bank ls` shows many `ellipse` fallbacks (thin, faint scratches), try `--mask-from otsu` or adjust `--min-box` — a box is not a defect boundary.

### Presets

Same seed and target across the four presets (`preview --compare-methods blend|harmonize` compares within a stage):

![four presets](assets/presets-4x.png)

| Preset | Family | Blend · harmonize | Use |
|---|---|---|---|
| `poisson-graft` (default) | NSA | Poisson (normal) · stats 0.3 | Most defects; keeps low-gradient stains alive |
| `multiband-graft` | Laplacian pyramid | multiband · histmatch 0.3 | Best texture preservation, fewer halos |
| `alpha-paste` | Feathered paste | alpha (feather 2) · Reinhard 0.5 | Fast, colour-matched edges |
| `hard-paste` | CutPaste | paste · none | Crudest baseline for training experiments |

Defaults were chosen by measuring how much each method fades the defect on the sample bank (in-mask L1 relative to hard-paste); all three harmonize methods pull defect tone toward the target by construction, so strengths are kept low. No real-data mAP evidence yet (roadmap).

**Reproducibility scope**: byte-identical on the same OS and OpenCV minor version. Other environments may differ by a few pixels (`cv2.seamlessClone` solver). The zip ships its own OpenCV, so zip-to-zip results match.

### Development

```sh
git clone https://github.com/slnu21/Graft.git && cd Graft
./bootstrap.sh                   # venv + pip install -e ".[dev]"   (Windows: .\bootstrap.ps1 -Gui)
source .venv/bin/activate
pytest && ruff check . && ruff format --check .
# Windows zip: .\bootstrap.ps1 -Build ; .\tools\build_zip.ps1  →  dist/anograft-<ver>-win64.zip
```

A new algorithm = one stage class + one registry line + one preset YAML.

### License

MIT © 2026 slnu21 — `LICENSE`. Bundled components (PySide6/Qt LGPL, …) in `THIRD-PARTY-NOTICES.md`; the offline/local promise in `PRIVACY.md`. External datasets and model weights keep their own licenses; Graft never redistributes them.

<div align="center">

# Graft

**라벨링한 실제 결함을 정상 이미지에 이식해, 정답 마스크와 재현 정보가 붙은 학습용 이상 데이터셋을 만듭니다.**

패키지·CLI 이름: `anograft` · 전 과정 오프라인·로컬 · CPU만으로 동작 · Windows zip은 파이썬 없이 실행

[English below ↓](#english)

</div>

![원본 | 합성 | GT](assets/preview.png)

> **상태: v0.9.0** — CLI 코어(7단계 파이프라인 · 레시피 · 결함 보관함(bank) · YOLO/MVTec/COCO 출력 · 프리셋 10종 · 보관함 없이 도는 self-cut/perlin · 구조 정합 배치 · GrabCut·annulus ROI · VisA·DTD 어댑터)와 GUI **① 결함 표시 · ② 결함 보관함 · ③ 미리보기 · ④ 일괄 생성 · ⑤ 검수** 5탭(흐름 순서, 상태 배지·'다음 →'·시작 체크리스트 · ⚙ 라이트 테마·글자 크기) — 결함 사진만 있으면 GUI 만으로 마스크 그리기 → 보관함 정리 → 미리보기 → 일괄 생성 → 검수·정리된 데이터셋까지. 실데이터 1차 적용(경면 금속 원형 부품)에서 나온 [알려진 문제 10건](KNOWN-ISSUES.md)을 v0.6 에서 보정. **2차 실데이터 적용은 아직 미검증**(데이터가 생기면 아래 3줄로 확인). 공개 데이터(MVTec metal_nut)에선 합성 유/무 YOLO mAP50 **0.31 → 0.38**(grabcut 보관함) / **→ 0.45**(hybrid 보관함) — 같은 분할·학습 시드 3개 평균(`BENCHMARKS.md`; Magnetic Tile 은 한 프리셋으론 break +0.23 · blowhole −0.30 으로 갈려 합계 −0.02 였지만, blowhole 만 `relative-paste` 로 갈라 `dataset merge` 하니 **0.46 → 0.61(+0.15, 3/3)** — 결함 성격별 프리셋). 표준셋 실제 사본(MVTec metal_nut·screw·grid·bottle·hazelnut·carpet · Magnetic Tile · VisA · DTD)으로는 v0.8 에서 리허설·벤치 완료 — `TESTING.md` · `KNOWN-ISSUES.md` 결정 근거 절.

## 왜

이상 탐지·결함 검출 모델은 결함 이미지가 늘 부족합니다. 합성 기법(CutPaste · DRAEM · NSA · 확산 인페인팅)은 이미 여럿 나와 있지만 논문 저장소마다 흩어져 있고, 그 사이의 실무가 비어 있습니다 — 결함을 **모아 두는 곳**, **어디에 붙일지**의 통제, **정답 마스크 정책**, **재현성**, 학습 파이프라인이 **바로 먹는 출력 형식**.

Graft는 알고리즘을 새로 만드는 도구가 아니라 그 사이를 메우는 도구입니다.

- **결함 보관함(bank)** — 보유 YOLO 라벨(박스·폴리곤)이나 마스크 PNG에서 결함 조각을 모읍니다. 박스만 있으면 마스크를 추정합니다(GrabCut 등, 출처를 `mask_origin`으로 끌고 다님). 데이터가 없으면 표준 산업 데이터셋(MVTec AD)을 로컬 사본에서 읽습니다.
- **7단계 파이프라인** `결함 고르기 → 크기·회전 → 위치 정하기 → 붙이기 → 색·밝기 맞추기 → 카메라 효과 → 정답 영역`(source → geometry → placement → blend → harmonize → degrade → gtmask) — 알고리즘은 각 단계의 `method`로 고릅니다(붙이기: paste · alpha · Poisson · multiband, 색·밝기 맞추기: stats · Reinhard · 히스토그램 매칭 · relative(노출 보정), 위치: sampled · structure-aware, 붙일 수 있는 영역(ROI): otsu · grabcut · none · mask_dir). 프리셋으로 시작하고 필요할 때만 펼칩니다(미리보기 탭 **프리셋 고르기…** 갤러리가 각 프리셋을 지금 바탕 이미지에 적용한 썸네일과 이럴 때/피할 때를 함께 보여 줍니다, v0.9).
- **붙일 수 있는 영역(ROI)** 이 기본값 — 배경에 붙은 결함은 학습에 해롭습니다.
- **재현** — 레시피(YAML) + 시드가 같으면 워커 수와 무관하게 바이트 단위로 같은 데이터셋. 이미지마다 사이드카 JSON(결함 조각 id · 변환 · 좌표 · 시드 · 파이프라인 해시).
- **출력** — 정본은 이미지 + GT 마스크 + 사이드카 + `manifest.csv`. 그 위에 writer가 학습 형식을 덧붙입니다(YOLO `labels/*.txt` + `data.yaml` — 기존 학습셋에 그대로 합침 · `mvtec` — anomalib 이 읽는 `mvtec/<category>/{train,test,ground_truth}` 레이아웃, v0.4 · `coco` — `annotations.json`(instances: 폴리곤 segmentation(`segmentation: rle` 로 비압축 RLE — 조각·구멍 무손실)·bbox·area, categories = 보관함 classes) — Detectron2·mmdetection 용, v0.7).

## 설치

| 방법 | 언제 | 명령 |
|---|---|---|
| **Windows zip** (권장) | 파이썬 없는 PC, 현장 | [Releases](https://github.com/slnu21/Graft/releases)에서 `anograft-<ver>-win64.zip` → 풀기 → 그 폴더에서 `.\anograft.exe`(CLI) · `.\anograft-gui.exe`(GUI). 설치·관리자 권한 없음 |
| **pip / uv** | 파이썬 3.10+ 이 있는 PC, Linux/macOS | `pip install "anograft[gui] @ git+https://github.com/slnu21/Graft.git"` 또는 `uv tool install "anograft[gui] @ git+https://github.com/slnu21/Graft.git"`(uv가 파이썬까지 받아 줌). `[gui]`를 빼면 CLI만(순수 wheel 4개). PyPI 등록은 예정 |
| **소스** | 개발 | 아래 [개발](#개발) |

의존성은 numpy · opencv-python-headless · pydantic · pyyaml(전부 순수 wheel — 컴파일러·GPU 불필요) + GUI는 PySide6(LGPL, 동적 링크).

## 5분 시작

보유 데이터가 없어도 됩니다. 샘플 YOLO 세트(브러시드 메탈 + scratch/pit/stain)로 끝까지 한 바퀴. **GUI 라면 상단 '샘플 데이터…' 버튼 하나**(판/원형 → 샘플·보관함·레시피를 만들어 미리보기 탭에 엶), CLI 는 `anograft sample --out samples/metal --quickstart` 한 줄(아래는 단계별). zip을 푼 폴더(또는 repo 루트)에서:

```powershell
# zip 이면 anograft → .\anograft.exe
anograft sample --out samples/metal                                       # 샘플 이미지 22장 + YOLO 박스 라벨 (--shape ring: 원형 부품 — annulus ROI·dent-graft 연습)
anograft bank import-yolo --images samples/metal/images --labels samples/metal/labels `
    --names samples/metal/data.yaml --out bank/sample --list-normals samples/metal/normals.txt
anograft bank ls bank/sample                                              # 클래스별 조각 수 · 마스크 출처(정확/추정) · 신뢰도 낮음 · µm/px (--json 스크립트용)
anograft doctor                                                           # 환경 진단(버전·Qt·스레드·프리셋) — 문제 보고 첫 줄
anograft bank preview bank/sample --out out/bank-preview.png              # 추정 마스크를 눈으로 (amber = 추정, ellipse = 과라벨, 화살표 = 밝은 쪽 — 한 방향이면 조명 의존)
anograft run recipes/sample-poisson.yaml --workers 4                      # → out/sample/{images,masks,meta,labels,data.yaml,manifest.csv}
anograft preview recipes/sample-poisson.yaml --index 0 --compare-methods blend --out out/compare.png
anograft-gui recipes/sample-poisson.yaml                                  # GUI (zip: .\anograft-gui.exe · pip: python -m anograft.gui). 인자 없이 켜면 최근 레시피 복원, 없으면 시작 안내
```

`anograft methods`가 스테이지별 선택지와 가용 여부를, `anograft recipe init --preset <이름> --write my.yaml`이 프리셋을 펼친 레시피를 줍니다(`--roi annulus` 로 ROI 만 바꾸고 `--um-per-px` 로 바탕 이미지의 픽셀 크기를 — 예: `--preset dent-graft --roi annulus` 는 원형 부품의 찍힘). 레시피 상대경로는 **현재 폴더 기준**입니다.

![GUI 미리보기 탭](assets/gui-studio.png)

미리보기 탭 오른쪽 **파이프라인 카드**에서 단계별 method 를 고르고 **모든 파라미터를 바로 편집**합니다(레시피 스키마에서 자동 생성 — 범위·on/off·선택지·경로, `null` 은 체크 해제). 라벨은 한국어(단위 포함)이고 툴팁에 설명·영어 이름·YAML 키·범위가 있습니다(v0.9). 프리셋 값과 다른 행은 teal 라벨 + ↺(되돌리기), 자주 안 쓰는 값은 **고급 옵션** 아래 접혀 있고, 세기 같은 값은 슬라이더(더블클릭 = 되돌리기)로 끕니다. 같은 설명을 CLI 에서: `anograft explain geometry.scale` · `explain blend:poisson` · `explain preset:dent-graft`, 전체 표는 [`PARAMS.md`](PARAMS.md). 잘못된 값은 그 카드에 빨간 줄로 막히고, 결과는 레시피 저장에 그대로 반영됩니다 (v0.6).

![GUI 결함 표시 탭](assets/gui-label.png)

![GUI 일괄 생성 탭](assets/gui-batch.png)

**결함 보관함 탭**(v0.7)에서 조각을 마스크 신뢰도순으로 보고 신뢰도 낮은 것(빨간 테두리)을 골라 다듬거나 지웁니다. **검수 탭**(v0.7)에서 합성 결과를 A/R 로 판정하고, 왼쪽 히스토그램으로 합성(teal) vs 실제(amber) 분포(크기 · 밝기 차 · 거칠기 · 선명도 · **밝은 쪽 방향**)를 비교한 뒤 반려를 뺀 정리된 데이터셋을 내보냅니다. 아래 화면은 MVTec metal_nut(공개 데이터, `recipes/public-metal-nut-dent.yaml`) 검수 — 분포를 **클래스 콤보**(v0.8)로 `bent` 만 보면 실제 35장(amber)의 조명 방향이 +90° 쪽으로 모이고 합성 9장(teal)이 그 안에 들어온다. `현장 측정값…` 로 실제 계열을 현장 측정값으로 바꿀 수 있다.

![GUI 결함 보관함 탭](assets/gui-bank.png)

![GUI 검수 탭](assets/gui-review.png)

## 보유 데이터로

| 가진 것 | 명령 |
|---|---|
| YOLO 라벨(`images/`·`labels/*.txt`·`data.yaml`) | `anograft bank import-yolo --images … --labels … --names data.yaml --out bank/mine --list-normals normals.txt` (빈 라벨 이미지 = 정상 후보) |
| 이미지 + 마스크 PNG 쌍 | `anograft bank import-pairs --images … --masks … --class scratch --out bank/mine` (`--class-from-dir` · `--csv`) |
| MVTec AD 로컬 사본 | `anograft dataset info mvtec-ad` → `anograft bank import-dataset mvtec-ad <root>/metal_nut --out bank/metal_nut` (내려받지 않음, CC BY-NC-SA) |
| VisA 로컬 사본 | `anograft dataset info visa` → `anograft bank import-dataset visa <VisA>/candle --out bank/candle` (결함 유형 세분이 없어 클래스 `anomaly` 하나, CC BY-NC-SA) |
| DTD 텍스처(보관함 없이 DRAEM 식) | `anograft dataset info dtd` → `anograft dataset textures dtd <dtd> --out textures.txt [--categories cracked,stained] [--limit 300]` → 레시피 `source: {method: perlin-texture, texture: dir, texture_dir: textures.txt}`. 결함이 아니라 보관함에는 넣지 않는다(연구 목적 라이선스, 로컬 사본만) (v0.7) |
| **결함 사진만**(라벨 없음) | GUI **결함 표시 탭** — 폴더 열기 → 사진 선택 → 브러시/다각형/자동 선택(상자를 끌면 GrabCut) → 클래스 입력 → **보관함에 저장**(Ctrl+S). 라벨링 도구가 따로 필요 없다 (v0.5). 그다음 스튜디오에서 미리보기 → **배치로 보내기** → 배치 탭 **생성 시작** — CLI 없이 끝까지 |
| 배치가 될지 미리 | `anograft run <recipe> --dry-run` — 배분·경고에 더해 **ROI 최대 폭 vs 클래스별 패치 폭**(가능/빠듯/불가): 좁은 링에 큰 패치면 돌리기 전에 안다 |
| 합성 결과 검수 | GUI **검수 탭**(v0.7) — 출력 폴더를 열어 썸네일로 보고 A/R 로 채택·반려(`review.csv`), 합성 vs 실제(보관함) 분포 히스토그램(면적·긴 변·밝기 차·거칠기·선명도·**밝은 쪽 방향** — 회전이 하이라이트를 뒤집으면 클래스별 R 로 드러나 `dent-graft` 를 권함), **정리본 내보내기**(반려 제외), **리포트**(HTML 한 장). CLI `anograft dataset prune <out> --out <pruned>` · `dataset report <out>` · 정리본 여러 개를 한 학습셋으로 `dataset merge a b --out c` |
| 보관함 정리 | GUI **결함 보관함 탭**(v0.7) — 조각 그리드(신뢰도 낮음 빨간 테두리) · 필터/정렬 · 삭제 · **결함 표시 탭에서 다듬기**(마스크를 고쳐 같은 id 에 덮어쓰기). `bank ls`/`bank preview` 의 GUI 판 |
| YOLO 라벨을 GUI 에서 다듬기 | 결함 표시 탭에서 `images/` 를 열면 옆의 `labels/<stem>.txt`(+`data.yaml`)를 찾아 **YOLO 초안**으로 마스크를 미리 채운다(박스 → GrabCut 추정, 폴리곤 → 채움, 목록에 `▸`). 손보고 저장하면 `manual:mixed`, 그대로 저장하면 임포터와 같은 `yolo-box:*` (v0.6) |
| 결함이 생겨도 되는 면만 지정(ROI) | 결함 표시 탭 **어디에 저장 → 붙일 수 있는 영역**: 바탕(정상) 이미지 폴더를 열고 허용 영역을 칠해 `<mask_dir>/<stem>.png` 로 저장 → 레시피 `placement.roi: {method: mask_dir, path: <mask_dir>}`. 원형 부품은 파일 없이 `annulus` ROI (v0.6) |

그다음은 `recipe init` → `inputs.bank`·`inputs.targets`(정상 이미지 폴더 또는 목록) 수정 → `run`. 출력 `images/`·`labels/`·`data.yaml`은 기존 YOLO 학습셋에 그대로 합쳐집니다(같은 `names` 순서). `python tools/train_smoke.py --synthetic out/sample --base <기존셋> --out train/merged`가 합쳐서 `ultralytics` 1 epoch을 돌립니다(ultralytics는 별도 설치, `--dry-run`은 합치기만).

`bank ls`의 `est`·origins 열에서 `ellipse` 폴백 비율이 높으면(가늘고 희미한 스크래치) `--mask-from otsu`나 `--min-box`를 조정하세요 — 박스는 결함 경계가 아닙니다. **`--mask-from hybrid`**(v0.8)는 grabcut 사슬 결과가 저신뢰면 내접 타원으로 대체합니다 — 공개 데이터 1180 인스턴스 벤치에서 IoU 중앙값 0.37 → 0.50, 실패율 0.43 → 0.21(`KNOWN-ISSUES.md` 결정 근거 절; 기본값은 그대로 `grabcut`). 박스 추정 마스크에는 **타당성 점수**(`confidence` 0..1 — 마스크 안/밖 대비·박스 테두리 접촉·조각 수·포화)가 붙고, 0.5 미만은 `bank ls` `lowconf` 열·`bank preview` **빨간 테두리**·`run` 경고로 드러납니다(경면 금속처럼 면적은 그럴듯한데 엉뚱한 곳을 잡는 경우). 저신뢰 소스는 라벨 탭에서 YOLO 초안으로 열어 다듬으세요. `bank ls` 의 `lightR` 열은 클래스별 **조명 일관성**(실제 소스들의 하이라이트가 같은 방향인가, 1 = 전부 같은 방향) — 0.5 이상이면 찍힘·덴트류라 회전 ±180/flip 이 하이라이트를 뒤집으므로 `run` 이 경고하고 `dent-graft` 를 권합니다.

**여러 제품의 결함을 한 보관함에** — 제품별로 따로 만든 보관함은 `anograft bank merge bank/A bank/B --out bank/all --rename 찍힘=dent [--dedupe]` 로 합치고(v0.7.x; `--dedupe` 는 내용이 같은 소스를 한 번만), 임포트마다 `--tags prodA,lot3` 로 표시해 두고, 레시피 `pipeline.source.tags: {include: [prodA, prodC], exclude: [old]}` 로 골라 씁니다(클래스는 그대로, 클래스 안의 풀만 줄어듦 · `run --dry-run` 이 필터 후 소스 수를 보여줌). **다른 카메라/배율의 결함**은 크기가 틀어지므로 임포트 `--um-per-px` 와 레시피 `inputs.um_per_px` 를 둘 다 지정하세요 — 한쪽이라도 없으면 축척 정합이 꺼진 채(`factor 1.0`) 돌아가고, `bank ls` 의 `no_um` 열과 `run`/GUI 상태바가 이를 경고합니다.

**라벨링한 결함이 하나도 없다면** — 정상 이미지만으로 `self-cut`(CutPaste) · `perlin-texture`(DRAEM) 프리셋이 돕니다: `anograft recipe init --preset self-cut --targets <정상 폴더> --write r.yaml`(`inputs.bank: null`) → `run`. 클래스는 `cutpaste`/`anomaly` 하나(이상 탐지 이진 학습용). `preview --compare-methods source`로 세 소스를 나란히.

## 프리셋

같은 시드·같은 대상에 프리셋 4종(`preview --compare-methods blend|harmonize`로 스테이지별 비교도 가능):

![프리셋 4종](assets/presets-4x.png)

| 프리셋 | 계열 | 블렌딩 · 조화 | 쓰임 |
|---|---|---|---|
| `poisson-graft` (기본) | NSA | Poisson(normal) · stats 0.3 | 대부분의 결함. 얼룩처럼 그래디언트가 약한 결함도 살린다 |
| `multiband-graft` | 라플라시안 피라미드 | multiband · histmatch 0.3 | 텍스처 보존이 좋고 경계 halo가 덜함 |
| `alpha-paste` | 페더 합성 | alpha(feather 2) · Reinhard 0.5 | 빠름, 경계 색 정합 |
| `hard-paste` | CutPaste(보관함) | paste · 없음 | 가장 거친 대조군(학습 실험용) |
| `relative-paste` | 노출 보정 paste | paste · **relative 1.0**(소스 링 → 대상 링 오프셋, 결함의 상대 대비 보존) · camera 열화 | **대비가 곧 신호인 결함**(블로우홀·검은 구멍·핏). poisson·stats 계열은 정의상 결함 톤을 옅게 만들고(MT blowhole 합성 대비 −17 vs 실제 −48 → mAP 하락), hard-paste 그대로는 노출이 다른 대상에서 **배경보다 밝은 구멍**이 된다(합성 대비 중앙값 0). relative 는 −37. MT 에서 blowhole 만 이 프리셋으로 갈라 merge → mAP50 +0.15(blowhole 0.34 → 0.81) (v0.8.2) |
| `self-cut` | CutPaste·Scar | 소스 = 대상 자신의 사각/스카 패치 + 색 지터 · paste | **보관함 불필요** — 정상 이미지만으로 시작 |
| `perlin-texture` | DRAEM | 소스 = 펄린 노이즈 마스크 + 텍스처(대상 자신 증강 또는 `texture_dir`) · alpha β 0.4~1 | **보관함 불필요** — 불규칙한 이상 영역 |
| `structure-aware-graft` | 구조 정합 배치 | poisson-graft + 배치 `structure-aware`(그래디언트 큰 곳 선호 · 결·에지 방향에 정렬) · ROI `grabcut` | 스크래치가 결을 따르고 칩이 모서리에 생기는 부품. 무광·그림자로 Otsu가 안 갈리는 대상 |
| `annulus-graft` | 링 ROI | poisson-graft + ROI `annulus`(중심·반경을 대상마다 자동 검출, `r_inner`/`r_outer` 비율) | **원형 부품의 가공 링 면에만** 결함을 놓는다 — otsu/grabcut 은 물체 전체를 허용해 중앙 리세스에도 떨어졌다. 촬영마다 부품이 움직여도 링이 따라간다 (v0.6) |
| `dent-graft` | 조명 의존 결함 | poisson NORMAL + 회전 **±15°**·flip none(조명이 위/아래면 `horizontal` 로 두 배)·축척 0.9~1.1 · `structure-aware`(위치 균등, 방향은 결·접선 정렬, jitter 5°, **정렬 상한 30°** — 그 이상 돌려야 하는 자리는 정렬 안 함) · 조화 0.2 | **찍힘·덴트·눌림** — 3D 변형이라 보이는 모양이 곧 조명 효과. ±180° 로 돌리면 음영/하이라이트가 뒤집혀 물리적으로 불가능한 그림이 된다. 스크래치·얼룩은 다른 프리셋(±180 유지) (v0.6) |

**결함 성격별 프리셋**(v0.8.2) — 한 보관함의 클래스마다 맞는 블렌딩이 다르면(구멍은 `relative-paste`, 깨짐·얼룩은 `poisson-graft`) `recipe init --classes` 로 프리셋을 클래스 부분집합에만 적용해 따로 돌리고 `dataset merge --dedupe-normals` 로 한 학습셋을 만듭니다: `anograft recipe init --preset relative-paste --classes blowhole --write a.yaml` · `… --preset poisson-graft --classes break crack --write b.yaml` → `run a.yaml` · `run b.yaml` → `dataset merge out/a out/b --out out/ab --dedupe-normals`(같은 보관함이라 클래스 id 일치, 정상은 한 벌만). 검수 탭 **대비 히스토그램**(합성 vs 보관함 실제)이 학습 전에 어느 클래스가 옅어졌는지 보여 주고, 합성 중앙값이 실제의 절반 미만이면 제목·리포트·`dataset report` 에 **대비 힌트**("이 클래스만 `relative-paste` 로 갈라 보라")가 붙습니다(v0.8.3). `tools/train_mvtec_map.py --class-presets blowhole=relative-paste break=poisson-graft …` 가 같은 흐름을 mAP 비교까지 자동으로(`BENCHMARKS.md` §2 Magnetic Tile).

한 보관함에 스크래치(±180° 무방)와 찍힘(조명 의존)이 **섞여 있으면** 프리셋을 둘로 나누지 말고 `geometry.per_class` 로 그 클래스만 좁힙니다 — `per_class: {찍힘: {rotate: [-15, 15], flip: false}}` (준 필드만 덮어씀, 나머지 클래스는 그대로; `bank ls` 의 lightR ≥ 0.5 인 클래스가 후보, `run` 경고가 이 문법을 알려줍니다). `anograft recipe init --bank bank/mine --auto-dent` 가 그 클래스를 찾아 써 줍니다(`--dent-class 찍힘` 으로 직접도). 스튜디오에선 기하 카드 아래 **클래스별 표**(적용 · 회전 · flip)로 편집합니다.

기하 스테이지의 작은 손잡이(v0.8.1, 전부 기본 off·기존 결과 불변): `geometry.tps: {points: 3, jitter: 0.06}` 은 제어점 격자를 흔들어 패치를 **휘고 늘리는** thin-plate spline(elastic 이 국소 잔물결이면 tps 는 전역 휘어짐) · `source.redraw_on_empty`(기본 2)는 축소 뒤 마스크가 사라진 아주 작은 소스를 그 자리에서 **다시 뽑아** 이미지가 skipped 되지 않게 · `source.single_class_per_image: true` 는 한 이미지의 결함을 첫 결함의 클래스로 묶어 MVTec writer 의 클래스 섞임을 없앱니다.

기본값은 샘플 보관함에서 "결함이 옅어지는 정도"(hard-paste 대비 마스크 안 L1 비율)를 재서 정했습니다 — stats·Reinhard·histmatch 세 조화 방법은 내부를 대상 링에 맞추므로 정의상 결함 톤을 (1−strength) 배로 옅게 만들어 strength 를 낮게 뒀고, 그래도 대비가 신호인 결함엔 `relative`(소스 자기 배경 → 대상 배경 오프셋만)를 씁니다. 실데이터 학습 mAP 근거는 아직 없습니다(공개 데이터 근거는 `BENCHMARKS.md`).

### 조명 의존 결함 한 바퀴 (찍힘·덴트)

찍힘은 모양이 곧 조명 효과라 ±180° 로 돌리면 하이라이트가 반대쪽에 붙습니다. 도구가 그걸 **재고 · 미리 잡고 · 골라내는** 순서:

| 단계 | 어디서 | 무엇 |
|---|---|---|
| 보관함 | `anograft bank ls` · 결함 보관함 탭 요약 · `bank preview` 화살표 | 클래스별 **lightR**(실제 소스들의 하이라이트 방향 일관성, 1 = 전부 같은 쪽). `*` = 유의(R ≥ 0.5 이고 n·R² ≥ 2.9 — 클래스당 몇 장이면 우연히도 크므로) |
| 레시피 | `recipe init --bank … --auto-dent` · `--dent-class 찍힘` | 그 클래스만 `geometry.per_class: {찍힘: {rotate: [-15, 15], flip: horizontal}}`(조명이 위/아래에서 오면 좌우 뒤집기는 안전 — 옆이면 `vertical`, 모르면 `none`) — 스크래치는 그대로 ±180° |
| 실행 전 | `run --dry-run` · `recipe check` · 미리보기 탭 크기·회전 카드 ⚠ | 유의한 클래스를 rotate 폭 > 90° 또는 flip 으로 돌리면 경고. 카드의 **▶ 조명 클래스만 ±15°·flip 끔** 이 한 번에 고침 |
| 미리보기 | 미리보기 탭 시드 변형 카드 **↯** | 인스턴스의 하이라이트가 실제 방향과 90° 넘게 다르면 표시 |
| 검수 | 검수 탭 분포 **밝은 쪽 방향** · 필터 **빛 방향이 뒤집힌 듯함** · `run --report` | 합성 vs 실제 각도 분포와 클래스별 R, 뒤집힌 이미지 목록. `dataset prune --drop-flipped` 로 제외 |

샘플로 보면: `anograft sample --out s --quickstart` 는 pit 을 조명 의존으로 찾아 per_class 를 써 줍니다. 같은 보관함을 `--preset poisson-graft` 로 그대로 돌리면 검수 탭 필터가 40장 중 33장을 골라냅니다(위 스크린샷).

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

**공개 데이터로 확인·재기**(실데이터 없을 때): `python tools/fetch_public_datasets.py metal_nut magnetic-tile --import`(표준 라이브러리만, MVTec 은 CC BY-NC-SA 로컬 개발용) → `recipes/public-*.yaml` 로 한 바퀴. 받아서 확인할 항목은 **`TESTING.md`**, 잰 숫자(박스→마스크 IoU 1180 인스턴스 · 합성 유/무 YOLO mAP)는 **`BENCHMARKS.md`**, 결정 대기 항목의 근거는 `KNOWN-ISSUES.md`. `tools/bench_mask_from_box.py`·`tools/train_mvtec_map.py`(별도 venv ultralytics) 로 재현.

## 라이선스

MIT © 2026 slnu21 — `LICENSE`. 함께 배포되는 구성 요소(PySide6/Qt LGPL 등)는 `THIRD-PARTY-NOTICES.md`, 오프라인·로컬 원칙은 `PRIVACY.md`.

외부 데이터셋(MVTec AD 등)·모델 가중치는 각자의 라이선스를 따르며, Graft는 재배포하지 않고 로컬 사본을 읽기만 합니다.

---

<div align="center">

## English

</div>

**Graft real, labeled defects onto normal images to build training datasets for anomaly detection — with ground-truth masks and full reproducibility metadata.** Package/CLI: `anograft`. Fully offline, CPU-only; the Windows zip needs no Python.

> **Status: v0.9.0** — CLI core (7-stage pipeline, recipes, defect bank, YOLO/MVTec/COCO output, 10 presets, bank-free self-cut/perlin sources, structure-aware placement, GrabCut/annulus ROI, VisA/DTD adapters) plus all five GUI tabs **① Label · ② Bank · ③ Studio · ④ Batch · ⑤ Review** (v0.9: Korean-first labels with English/YAML keys in tooltips, parameter help, preset gallery, workflow badges, light theme) — with only defect photos you can label → tidy the bank → preview → generate → review and export a pruned set entirely in the GUI. The [10 known issues](KNOWN-ISSUES.md) from a first real-data application (mirror-finish round metal part) are addressed in v0.6. **A second real-data pass is still unverified** — three commands once you have data (below). On public data (MVTec metal_nut) synthetic augmentation lifted YOLO mAP50 **0.31 → 0.38** (grabcut bank) / **→ 0.45** (hybrid bank), mean of three training seeds on a fixed split (`BENCHMARKS.md`; on Magnetic Tile one preset diverged per class — break +0.23, blowhole −0.30, net −0.02 — but splitting only blowhole onto `relative-paste` and merging with `dataset merge` gives **0.46 → 0.61 (+0.15, 3/3)** — presets per defect type). Real public copies (MVTec metal_nut/screw/grid/bottle/hazelnut/carpet, Magnetic Tile, VisA, DTD) were rehearsed and benchmarked in v0.8 — see `TESTING.md` and the evidence section of `KNOWN-ISSUES.md`.

### Why

Synthetic-defect methods (CutPaste, DRAEM, NSA, diffusion inpainting) exist, but they live in scattered paper repos and the practical glue is missing: a place to **collect** defects, control over **where** they land, a **ground-truth mask policy**, **reproducibility**, and output your training pipeline can **consume directly**. Graft fills that gap instead of inventing another algorithm.

- **Defect bank** — import from your YOLO labels (boxes/polygons) or mask PNGs; boxes get a pixel mask estimated (GrabCut etc., provenance kept as `mask_origin`). No data? Point it at a local copy of MVTec AD.
- **7-stage pipeline** `source → geometry → placement → blend → harmonize → degrade → gt-mask` — pick algorithms per stage via `method` (blend: paste · alpha · Poisson · multiband; harmonize: stats · Reinhard · histogram matching; placement: sampled · structure-aware; ROI: otsu · grabcut · none · mask_dir). Start from a preset, unfold only what you need.
- **Placement ROI is on by default** — defects pasted onto background hurt training.
- **Reproducible** — same recipe (YAML) + seed ⇒ byte-identical dataset regardless of worker count. Per-image sidecar JSON (source id, transform, coordinates, seed, pipeline hash).
- **Output** — canonical image + GT mask + sidecar + `manifest.csv`, plus a writer layer for training formats (YOLO `labels/*.txt` + `data.yaml`, mergeable into your existing set; `mvtec` — the `mvtec/<category>/{train,test,ground_truth}` layout anomalib reads, v0.4; `coco` — `annotations.json` (instances: polygon segmentation — or lossless uncompressed RLE with `segmentation: rle`, bbox, area, categories = bank classes) for Detectron2/mmdetection, v0.7).

### Install

| Method | When | Command |
|---|---|---|
| **Windows zip** (recommended) | No Python, shop-floor PCs | Grab `anograft-<ver>-win64.zip` from [Releases](https://github.com/slnu21/Graft/releases), unzip, run `.\anograft.exe` (CLI) / `.\anograft-gui.exe` (GUI) from that folder. No installer, no admin rights |
| **pip / uv** | Python ≥ 3.10, Linux/macOS | `pip install "anograft[gui] @ git+https://github.com/slnu21/Graft.git"` or `uv tool install "anograft[gui] @ git+https://github.com/slnu21/Graft.git"` (uv fetches Python for you). Drop `[gui]` for CLI-only (four pure wheels). PyPI listing planned |
| **Source** | Development | see [Development](#development) |

Dependencies: numpy · opencv-python-headless · pydantic · pyyaml (all pure wheels — no compiler, no GPU); the GUI adds PySide6 (LGPL, dynamically linked).

### Five-minute start

No data needed — the bundled sample set (brushed metal + scratch/pit/stain) runs the whole loop. **In the GUI it is one click**: the **Sample…** button at the top (plate / ring → sample set, bank and recipe, opened in Studio); on the CLI `anograft sample --out samples/metal --quickstart` does the same (step by step below). From the unzipped folder (or the repo root):

```powershell
# zip: anograft → .\anograft.exe
anograft sample --out samples/metal                                       # --shape ring: a round part for the annulus ROI / dent-graft path
anograft bank import-yolo --images samples/metal/images --labels samples/metal/labels `
    --names samples/metal/data.yaml --out bank/sample --list-normals samples/metal/normals.txt
anograft bank ls bank/sample
anograft bank preview bank/sample --out out/bank-preview.png              # eyeball estimated masks (amber = estimated)
anograft run recipes/sample-poisson.yaml --workers 4                      # → out/sample/{images,masks,meta,labels,data.yaml,manifest.csv}
anograft preview recipes/sample-poisson.yaml --index 0 --compare-methods blend --out out/compare.png
anograft-gui recipes/sample-poisson.yaml                                  # GUI (zip: .\anograft-gui.exe · pip: python -m anograft.gui)
```

`anograft methods` lists per-stage choices and availability; `anograft recipe init --preset <name> --write my.yaml` expands a preset (`--roi annulus` swaps only the ROI, `--um-per-px` sets the target pitch — e.g. `--preset dent-graft --roi annulus` for dents on a round part). Relative paths in recipes resolve against the **current directory**.

In the Studio, the **pipeline cards** on the right let you pick each stage's method and **edit every parameter in place** (generated from the recipe schema — ranges, on/off, choices, paths; `null` = unchecked). Invalid values are rejected with a red line on that card, and edits go straight into the saved recipe (v0.6).

### Your own data

| You have | Command |
|---|---|
| YOLO labels (`images/`, `labels/*.txt`, `data.yaml`) | `anograft bank import-yolo --images … --labels … --names data.yaml --out bank/mine --list-normals normals.txt` (images with empty labels become normal candidates) |
| Image + mask PNG pairs | `anograft bank import-pairs --images … --masks … --class scratch --out bank/mine` (`--class-from-dir`, `--csv`) |
| Local MVTec AD copy | `anograft dataset info mvtec-ad` → `anograft bank import-dataset mvtec-ad <root>/metal_nut --out bank/metal_nut` (never downloaded; CC BY-NC-SA) |
| Local VisA copy | `anograft dataset info visa` → `anograft bank import-dataset visa <VisA>/candle --out bank/candle` (no defect-type split, so one class `anomaly`; CC BY-NC-SA) |
| DTD textures (DRAEM-style, no bank) | `anograft dataset info dtd` → `anograft dataset textures dtd <dtd> --out textures.txt [--categories cracked,stained] [--limit 300]` → recipe `source: {method: perlin-texture, texture: dir, texture_dir: textures.txt}`. Not defects, so never imported into a bank (research-only licence, local copy only) (v0.7) |
| **Only defect photos** (no labels) | GUI **Label tab** — open folder → pick a photo → brush / polygon / auto-select (drag a box → GrabCut) → type the class → **Save to bank** (Ctrl+S). No separate labelling tool needed (v0.5). Then preview in Studio → **Send to batch** → **Run** in the Batch tab — no CLI required |
| Will it place? | `anograft run <recipe> --dry-run` — besides allocation and warnings, **ROI max width vs per-class patch size** (ok / tight / impossible): know before running that a big patch will not fit a narrow ring |
| Review results | GUI **Review tab** (v0.7) — open an output folder, accept/reject thumbnails with A/R (`review.csv`), compare synthetic vs real (bank) histograms (area, length, contrast, texture, sharpness, **lighting direction** — when rotation flips the highlights, the per-class concentration R exposes it and suggests `dent-graft`), **export a pruned copy** without the rejects, and a one-page **HTML report**. CLI `anograft dataset prune <out> --out <pruned>` · `dataset report <out>` · combine several pruned outputs into one training set with `dataset merge a b --out c` |
| Tidy the bank | GUI **Bank tab** (v0.7) — source grid (red border = low confidence), filter/sort, delete, **Refine in Label** (fix the mask and overwrite the same id). The GUI counterpart of `bank ls` / `bank preview` |
| Refine YOLO labels in the GUI | Open `images/` in the Label tab: the neighbouring `labels/<stem>.txt` (+`data.yaml`) is loaded as a **YOLO draft** that pre-fills the mask (box → GrabCut estimate, polygon → fill; `▸` in the list). Edited masks save as `manual:mixed`, untouched ones as `yolo-box:*` like the importer (v0.6) |
| Restrict where defects may go (ROI) | Label tab **Save as → ROI mask**: open the normal-image folder, paint the allowed surface, save to `<mask_dir>/<stem>.png` → recipe `placement.roi: {method: mask_dir, path: <mask_dir>}`. Round parts need no files — use the `annulus` ROI (v0.6) |

Then `recipe init` → edit `inputs.bank` / `inputs.targets` (normal-image folder or list) → `run`. The output `images/`, `labels/`, `data.yaml` merge straight into an existing YOLO set (same `names` order). `python tools/train_smoke.py --synthetic out/sample --base <your set> --out train/merged` merges and runs one `ultralytics` epoch (install ultralytics separately; `--dry-run` only merges).

If `bank ls` shows many `ellipse` fallbacks (thin, faint scratches), try `--mask-from otsu` or adjust `--min-box` — a box is not a defect boundary. **`--mask-from hybrid`** (v0.8) replaces a low-confidence grabcut-chain result with the inscribed ellipse — on a 1,180-instance public-data benchmark the median IoU went 0.37 → 0.50 and the failure rate 0.43 → 0.21 (see the evidence section in `KNOWN-ISSUES.md`; the default stays `grabcut`). Every box-estimated mask carries a **plausibility score** (`confidence` 0..1 — inside/outside contrast, box-edge contact, fragment count, saturation); below 0.5 it shows up in the `lowconf` column of `bank ls`, as a **red border** in `bank preview`, and as a `run` warning (mirror-like metal where a plausible-sized but wrong region gets picked). Refine such sources in the Label tab via the YOLO draft. The `lightR` column of `bank ls` is the per-class **lighting consistency** of the real sources (1 = every highlight points the same way); at 0.5 or above the class behaves like a dent, so ±180° rotation or flips would invert the highlights — `run` warns and suggests `dent-graft`.

**Several products in one bank** — merge per-product banks with `anograft bank merge bank/A bank/B --out bank/all --rename dent_ko=dent [--dedupe]` (v0.7.x; `--dedupe` keeps content-identical sources once), tag each import (`--tags prodA,lot3`) and select in the recipe with `pipeline.source.tags: {include: [prodA, prodC], exclude: [old]}` (classes stay the same, only the pool inside each class shrinks; `run --dry-run` shows the filtered counts). **Defects shot with another camera/magnification** come out the wrong size unless both `--um-per-px` at import and `inputs.um_per_px` in the recipe are set — with either missing, physical scaling is silently off (`factor 1.0`); the `no_um` column of `bank ls` and the `run`/GUI status bar warn about it.

**No labelled defects at all?** The `self-cut` (CutPaste) and `perlin-texture` (DRAEM) presets run on normal images alone: `anograft recipe init --preset self-cut --targets <normals> --write r.yaml` (`inputs.bank: null`) → `run`. Single class (`cutpaste` / `anomaly`) for binary anomaly training; `preview --compare-methods source` shows the three sources side by side.

### Presets

Same seed and target across the four presets (`preview --compare-methods blend|harmonize` compares within a stage):

![four presets](assets/presets-4x.png)

| Preset | Family | Blend · harmonize | Use |
|---|---|---|---|
| `poisson-graft` (default) | NSA | Poisson (normal) · stats 0.3 | Most defects; keeps low-gradient stains alive |
| `multiband-graft` | Laplacian pyramid | multiband · histmatch 0.3 | Best texture preservation, fewer halos |
| `alpha-paste` | Feathered paste | alpha (feather 2) · Reinhard 0.5 | Fast, colour-matched edges |
| `hard-paste` | CutPaste (bank) | paste · none | Crudest baseline for training experiments |
| `relative-paste` | Exposure-compensated paste | paste · **relative 1.0** (offset source ring → target ring, keeps the defect's relative contrast) · camera degrade | **Defects whose contrast is the signal** (blowholes, dark holes, pits). Poisson/stats fade the defect by construction (MT blowhole synthetic contrast −17 vs real −48 → mAP drop), while plain hard-paste on targets with varying exposure yields **holes brighter than the background** (median contrast 0). relative gives −37. On MT, splitting only blowhole onto this preset and merging → mAP50 +0.15 (blowhole 0.34 → 0.81) (v0.8.2) |
| `self-cut` | CutPaste · Scar | source = rect/scar patch cut from the target itself + colour jitter · paste | **No bank needed** — start from normal images only |
| `perlin-texture` | DRAEM | source = Perlin-noise mask + texture (augmented self-window or `texture_dir`) · alpha β 0.4–1 | **No bank needed** — irregular anomaly regions |
| `structure-aware-graft` | Structure-aware placement | poisson-graft + `structure-aware` placement (prefers high-gradient spots · aligns to grain/edge direction) · `grabcut` ROI | Parts where scratches follow the grain and chips sit on edges; matte/shadowed parts Otsu cannot segment |
| `annulus-graft` | Ring ROI | poisson-graft + `annulus` ROI (centre/radius auto-detected per target, `r_inner`/`r_outer` as ratios) | **Round parts whose defects only occur on the machined ring** — otsu/grabcut allow the whole object and dropped defects into the central recess. The ring follows the part as it shifts between shots (v0.6) |
| `dent-graft` | Lighting-dependent defects | poisson NORMAL + rotation **±15°**, flip none (`horizontal` doubles the data when the light comes from above/below), scale 0.9–1.1 · `structure-aware` (uniform position, axis aligned to grain/tangent, 5° jitter, **alignment capped at 30°** — spots needing more are left unaligned) · harmonize 0.2 | **Dents, dings, indentations** — 3D deformations whose appearance *is* the lighting. Rotating ±180° flips shadow/highlight against a light that did not move, which the eye catches first. Scratches and stains keep ±180° in the other presets (v0.6) |

**Per-defect-type presets** (v0.8.2) — when classes in one bank want different blending (holes → `relative-paste`, breaks/stains → `poisson-graft`), apply a preset to a subset of classes with `recipe init --classes`, run each, and merge with `dataset merge --dedupe-normals`: `anograft recipe init --preset relative-paste --classes blowhole --write a.yaml` · `… --preset poisson-graft --classes break crack --write b.yaml` → `run a.yaml` · `run b.yaml` → `dataset merge out/a out/b --out out/ab --dedupe-normals` (same bank → same class ids; normals kept once). The review tab's **contrast histogram** (synthetic vs bank real) shows before training which class got washed out, and when a class's synthetic median contrast drops below half of the real one a **contrast hint** ("split this class onto `relative-paste`") appears in the title, the report and `dataset report` (v0.8.3). `tools/train_mvtec_map.py --class-presets blowhole=relative-paste break=poisson-graft …` automates the same flow through to the mAP comparison (`BENCHMARKS.md` §2 Magnetic Tile).

When one bank **mixes** scratches (±180° is fine) and dents (lighting-dependent), do not split the recipe — narrow only that class with `geometry.per_class`: `per_class: {dent: {rotate: [-15, 15], flip: false}}` (only the given fields override; other classes are untouched; classes with `lightR` ≥ 0.5 in `bank ls` are the candidates, and the `run` warning spells out the syntax). `anograft recipe init --bank bank/mine --auto-dent` finds those classes and writes the override for you (`--dent-class dent` to name them yourself); in the Studio, edit it in the **per-class table** under the geometry card (enable · rotation · flip).

Small geometry knobs (v0.8.1, all off by default — existing outputs unchanged): `geometry.tps: {points: 3, jitter: 0.06}` bends and stretches the patch with a thin-plate spline over a jittered control grid (elastic is local ripple, tps is global bending) · `source.redraw_on_empty` (default 2) redraws a source whose mask vanished after scaling instead of skipping the image · `source.single_class_per_image: true` keeps one class per image so the MVTec writer never sees mixed classes.

Defaults were chosen by measuring how much each method fades the defect on the sample bank (in-mask L1 relative to hard-paste); stats/Reinhard/histmatch match the inside to the target ring and therefore fade the defect to (1−strength) by construction, so strengths are kept low — and for defects whose contrast is the signal, `relative` (only an offset from the source's own surroundings to the target's) keeps it. No real-data mAP evidence yet (public-data evidence in `BENCHMARKS.md`).

### The lighting-dependent loop (dents, dings)

A dent's appearance *is* the lighting, so rotating it ±180° puts the highlight on the wrong side. The tool **measures, warns before, and picks out after**:

| Step | Where | What |
|---|---|---|
| Bank | `anograft bank ls` · Bank-tab summary · `bank preview` arrows | Per-class **lightR** (how consistently the real sources' highlights point one way; 1 = all the same). `*` = significant (R ≥ 0.5 and n·R² ≥ 2.9 — a handful of sources can be high by chance) |
| Recipe | `recipe init --bank … --auto-dent` · `--dent-class dent` | Only that class gets `geometry.per_class: {dent: {rotate: [-15, 15], flip: horizontal}}` (a horizontal flip is safe when the light comes from above/below; `vertical` for side light, `none` if unknown) — scratches keep ±180° |
| Before running | `run --dry-run` · `recipe check` · Studio geometry card ⚠ | Warns when a significant class is rotated more than 90° or flipped. The card's **▶ dent classes only: ±15°, no flip** fixes it in one click |
| Preview | Studio variant card **↯** | Shown when an instance's highlight differs from the real direction by more than 90° |
| Review | Review-tab **lighting** histogram · filter **flipped lighting** · `run --report` | Synthetic vs real angle distribution, per-class R, list of flipped images. `dataset prune --drop-flipped` removes them |

On the sample: `anograft sample --out s --quickstart` detects pit as lighting-dependent and writes the per_class override. Run the same bank with `--preset poisson-graft` instead and the review filter picks out most of the images. The screenshot above is the Review tab on MVTec metal_nut (`recipes/public-metal-nut-dent.yaml`): with the **per-class combo** (v0.8) set to `bent`, the 35 real sources (amber) cluster around +90° and the 9 synthetic ones (teal) fall inside; `real CSV…` swaps the real series for field measurements.

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

**Public data, when you have none of your own**: `python tools/fetch_public_datasets.py metal_nut magnetic-tile --import` (stdlib only; MVTec AD is CC BY-NC-SA — local development use) and run `recipes/public-*.yaml`. What to check is in **`TESTING.md`**, the measured numbers (box→mask IoU over 1,180 instances, synthetic-vs-none YOLO mAP) in **`BENCHMARKS.md`**, and the evidence for pending defaults in `KNOWN-ISSUES.md`. Reproduce with `tools/bench_mask_from_box.py` and `tools/train_mvtec_map.py` (separate venv with ultralytics).

### License

MIT © 2026 slnu21 — `LICENSE`. Bundled components (PySide6/Qt LGPL, …) in `THIRD-PARTY-NOTICES.md`; the offline/local promise in `PRIVACY.md`. External datasets and model weights keep their own licenses; Graft never redistributes them.

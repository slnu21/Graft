# PARAMS — 레시피 파라미터 도움말 / Recipe parameter reference

> `anograft explain --markdown` 이 만든다(원천 `src/anograft/core/help.py` · 프리셋 `meta:`). 손으로 고치지 말고 문안을 고친 뒤 재생성. GUI 카드 툴팁 · `anograft explain <stage>.<field>` 와 같은 글. 기본값은 스키마 기본(프리셋은 다를 수 있음 — 프리셋 열).

## 프리셋 / Presets

| 프리셋 | 제목 | 한 줄 | 이럴 때 | 피할 때 | 근거 |
|---|---|---|---|---|---|
| `alpha-paste` | 가장자리 부드럽게 붙이기 | 페더 2px 알파 합성 + Reinhard(Lab 3채널) 색 맞춤 0.5 + 카메라 효과 | 경계만 감추면 되는 텍스처성 결함 · 색조 차이가 큰 조각 | 가는 결함(페더에 먹힘) · 그래디언트 정합이 필요한 경계 | 샘플 보관함 대비 유지율 0.26(hard-paste 1.0 기준) — 데브로그 2026-09-14-07 |
| `annulus-graft` | 링 면에만 붙이기 | poisson-graft + 붙일 수 있는 영역을 링(안쪽 0.55 ~ 바깥 0.9 반지름)으로 — 중심·반지름은 바탕마다 자동 검출 | 원형 부품의 가공 링 면(가운데 리세스에는 결함이 생기지 않는 부품) | 원형이 아닌 부품 | 합성 토크스 링 적중 16/16(poisson-graft 0/4) — 데브로그 2026-09-15-09 |
| `dent-graft` | 찍힘·덴트 (빛 방향 유지) | poisson-graft 에서 회전 ±15° · 뒤집기 없음 · 축척 0.9~1.1 · 결 정렬 상한 30° — 음영/하이라이트 방향을 지킵니다 | 빛 방향이 정해진 결함 — 찍힘·덴트·눌림·돌기(보관함 lightR 유의 클래스) | 회전해도 되는 텍스처성 결함(스크래치·얼룩) — 회전 다양성이 줄어 mAP 손해 | metal_nut bent 는 찍힘이 아니라 이득 없음(0.41 vs poisson 0.45) — 보관함의 lightR 로 판단 — BENCHMARKS §2 |
| `hard-paste` | 그대로 붙이기 (대조군) | 마스크 안 픽셀 치환, 색·밝기 맞춤·카메라 효과 없음 — 가장 거친 기준선 | 학습 실험의 대조군 | 노출이 제각각인 바탕 — 절반이 배경보다 밝은 구멍이 됩니다(→ relative-paste) | MT blowhole 합성 대비 중앙값 0(실제 −48) — BENCHMARKS §2 |
| `multiband-graft` | 다중 대역 붙이기 | 라플라시안 피라미드로 굵은 톤은 넓게 잔결은 좁게 섞고 히스토그램 맞춤 0.3 | 큰 얼룩처럼 톤 전이가 넓은 결함 · Poisson 의 halo 가 거슬릴 때 | 얇은 결함(단계 수가 두께에 맞춰 자동으로 줄어 효과가 작음) | 샘플 보관함 대비 유지율 0.50 — 데브로그 2026-09-14-07 |
| `perlin-texture` | 노이즈 텍스처 결함 | 펄린 노이즈 마스크 + 텍스처(바탕 자신 또는 DTD 폴더)를 알파 0.4~1 로 겹칩니다(DRAEM). 보관함 불필요 | 불규칙한 이상 영역을 흉내낼 때 · 재구성 계열 모델 학습 | 실제 결함 모양을 배워야 하는 검출기 |  |
| `poisson-graft` | 경계 자연스럽게 붙이기 | 보관함 조각을 Poisson 으로 붙이고 평균·분산을 살짝 맞춘 뒤 카메라 효과 — 기본 | 얼룩·스크래치·핏 등 표면 결함 대부분 | 대비가 곧 신호인 구멍·검은 점(→ relative-paste) · 빛 방향이 정해진 찍힘(→ dent-graft) | metal_nut mAP50 0.31 → 0.38(grabcut 보관함) / 0.45(hybrid 보관함), 학습 시드 3개 평균 — BENCHMARKS §2 |
| `relative-paste` | 대비 지키며 붙이기 | 그대로 붙인 뒤 조각 주변과 바탕 주변의 밝기 차만큼만 옮깁니다(노출 보정) — 결함의 상대 대비 보존 | 대비가 곧 신호인 결함 — 블로우홀·검은 구멍·핏 | 경계 정합이 곧 사실감인 큰 결손·깨짐(→ poisson-graft) | Magnetic Tile blowhole 만 이 프리셋으로 갈라 merge → mAP50 0.460 → 0.605(+0.15, 3/3) — BENCHMARKS §2 |
| `self-cut` | 정상 부위 잘라 붙이기 | 바탕 이미지 자신에서 네모/띠 조각을 잘라 색을 흔들어 붙입니다(CutPaste). 보관함 불필요 | 라벨링한 결함이 하나도 없을 때의 자기지도 기준선 | 실제 결함 조각이 생기면 poisson-graft 등으로 |  |
| `structure-aware-graft` | 표면 결을 따라 붙이기 | poisson-graft + 결·에지가 있는 곳 선호, 조각을 결 방향에 정렬 + GrabCut 영역 | 스크래치가 결을 따르고 칩이 모서리에 생기는 부품(무광·헤어라인) | 평탄한 면의 얼룩(→ prefer: flat 또는 poisson-graft) · 빛 방향이 정해진 결함(정렬 회전이 조명을 돌림) |  |

## 입력 · 출력 / Inputs · Output

| 키 | 라벨 | 단위 | 설명 | EN | 기본 |
|---|---|---|---|---|---|
| `inputs.bank` | 결함 보관함 |  | 결함 조각을 모아 둔 폴더(bank.yaml). 비우면 보관함 없이 도는 방법(정상 부위 잘라 붙이기 · 노이즈 텍스처)만 쓸 수 있습니다 | Bank folder | `null` |
| `inputs.targets` | 바탕 이미지 |  | 결함을 붙일 정상 이미지 폴더, 또는 경로 목록 .txt | Target (normal) images | `(필수)` |
| `inputs.um_per_px` | 바탕 픽셀 크기 | µm/px | 바탕 이미지 1픽셀이 몇 µm 인지. 조각에도 값이 있으면 실제 크기로 자동 축척합니다. 끄면 픽셀 크기 그대로 | Pixel pitch of targets | `null` |
| `output.root` | 출력 폴더 |  | 이미지·마스크·메타·manifest.csv 가 생기는 폴더 | Output root | `(필수)` |
| `output.count` | 만들 장수 | 장 | 합성 이미지 수(정상 이미지는 따로 함께 나갑니다) | Number of synthetic images | `(필수)` |
| `output.class_ratio` | 클래스 비율 |  | 클래스별 추첨 확률. 비우면 보관함 클래스를 고르게 | Class draw ratio | `null` |
| `output.defects_per_image` | 이미지당 결함 수 | 개 | 한 이미지에 붙일 결함 수 범위 | Defects per image | `[1, 1]` |
| `output.include_normals` | 정상 이미지 포함 |  | 켜면 바탕 이미지도 정상(결함 없음)으로 함께 내보냅니다 | Include normal images | `true` |
| `output.copy_mode` | 정상 이미지 복사 방식 |  | copy = 복사 · hardlink = 하드링크(같은 드라이브, 용량 절약) | Copy mode for normals | `copy` |
| `output.writer[yolo].seg` | 다각형 라벨 |  | 켜면 상자 대신 다각형(YOLO-seg)을 씁니다 | Polygon (YOLO-seg) labels | `false` |
| `output.writer[yolo].names_from` | 클래스 이름 출처 |  | bank = 보관함의 클래스 순서(id)를 그대로 | Class names from | `bank` |
| `output.writer[mvtec].category` | 카테고리 이름 |  | mvtec 폴더 구조의 카테고리 이름 | Category name | `graft` |
| `output.writer[mvtec].test_normal_ratio` | test 정상 비율 |  | 정상 이미지 중 test/good 으로 보낼 비율(나머지는 train/good) | Ratio of normals in test | `0.2` |
| `output.writer[mvtec].layout_dir` | 레이아웃 폴더 |  | 출력 폴더 아래 mvtec 구조를 만들 하위 폴더 이름 | Layout subfolder | `mvtec` |
| `output.writer[coco].description` | 설명 |  | annotations.json 의 info.description | Dataset description | `anograft synthetic defects` |
| `output.writer[coco].supercategory` | 상위 카테고리 |  | 모든 클래스의 supercategory 이름 | Supercategory | `defect` |
| `output.writer[coco].segmentation` | 세그멘테이션 표현 |  | polygon = 다각형 · rle = 비압축 RLE(조각·구멍 무손실) | Segmentation encoding | `polygon` |

## 결함 고르기 / Source — `pipeline.source`

어떤 결함 조각을 쓸지 — 보관함 · 정상 부위 잘라 붙이기 · 노이즈 텍스처

### `bank` — 보관함

결함 보관함의 조각을 클래스 확률대로 뽑습니다 · **이럴 때:** 라벨링한 결함이 있을 때(기본) · EN: Bank

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `classes` | 쓸 클래스 |  | 보관함의 이 클래스만 씁니다. 비우면 전부(또는 출력 클래스 비율의 키) | Classes to use | `null` |  |  |
| `tags.include` | 포함 태그 |  | 이 태그가 있는 조각만 씁니다(클래스 목록·확률·id 는 그대로) | Include tags | `[]` |  | ✓ |
| `tags.exclude` | 제외 태그 |  | 이 태그가 있는 조각은 뺍니다 | Exclude tags | `[]` |  | ✓ |
| `min_sources_warn` | 조각 부족 경고 기준 | 개 | 클래스의 조각 수가 이보다 적으면 경고합니다. 적은 조각을 반복하면 모델이 그 한 장을 외웁니다 | Warn below this many sources | `10` | alpha-paste `10` · annulus-graft `10` · dent-graft `10` · hard-paste `10` · multiband-graft `10` · poisson-graft `10` · relative-paste `10` · structure-aware-graft `10` | ✓ |
| `redraw_on_empty` | 빈 마스크 재추첨 | 회 | 크기 변환 뒤 마스크가 사라진 아주 작은 조각을 이만큼 다시 뽑습니다. 실패했을 때만 난수를 더 씁니다 | Redraws when mask vanishes | `2` |  | ✓ |
| `single_class_per_image` | 이미지당 한 클래스 |  | 켜면 한 이미지의 결함이 모두 첫 결함의 클래스가 됩니다(MVTec 형식용) | One class per image | `false` |  | ✓ |

### `self-cut` — 정상 부위 잘라 붙이기

바탕 이미지 자신에서 네모/띠 조각을 잘라 결함처럼 붙입니다(CutPaste) · **이럴 때:** 라벨링한 결함이 하나도 없을 때. 보관함 불필요 · EN: Self-cut (CutPaste)

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `cls` | 클래스 이름 |  | 출력 라벨에 쓸 이름 | Class name | `cutpaste` | self-cut `cutpaste` |  |
| `shape` | 조각 모양 |  | rect = 네모 조각 · scar = 가는 긴 띠 · mixed = 결함마다 반반 | Patch shape | `mixed` | self-cut `mixed` |  |
| `area_ratio` | 조각 면적 비율 |  | 네모 조각의 면적 / 이미지 면적 범위. 올리면 큰 결함 | Patch area ratio | `[0.02, 0.15]` | self-cut `[0.02, 0.15]` |  |
| `aspect` | 가로세로 비 |  | 네모 조각의 가로/세로 범위(로그 균등) | Aspect ratio | `[0.3, 3.3]` | self-cut `[0.3, 3.3]` | ✓ |
| `scar_width_px` | 띠 굵기 | px | 띠 모양의 굵기 범위 | Scar width | `[2, 16]` | self-cut `[2, 16]` | ✓ |
| `scar_length_px` | 띠 길이 | px | 띠 모양의 길이 범위 | Scar length | `[20, 120]` | self-cut `[20, 120]` | ✓ |
| `margin_px` | 잘라낼 여유 | px | 조각 둘레를 이만큼 더 잘라 Poisson 팽창·페더가 잘리지 않게 합니다 | Crop margin | `8` | self-cut `8` | ✓ |
| `max_tries` | 자리 찾기 시도 | 회 | 붙일 수 있는 영역 안에서 잘라낼 자리를 찾는 최대 횟수 | Max tries | `20` | self-cut `20` | ✓ |
| `jitter.brightness` | 밝기 흔들기 |  | 잘라낸 조각의 밝기를 이만큼 무작위로 바꿔 배경과 구분되게 합니다. 0 = 그대로 | Brightness jitter | `0.1` | self-cut `0.1` | ✓ |
| `jitter.contrast` | 대비 흔들기 |  | 조각의 대비를 이만큼 무작위로. 0 = 그대로 | Contrast jitter | `0.1` | self-cut `0.1` | ✓ |
| `jitter.saturation` | 채도 흔들기 |  | 조각의 채도를 이만큼 무작위로. 0 = 그대로 | Saturation jitter | `0.1` | self-cut `0.1` | ✓ |
| `jitter.hue` | 색상 흔들기 |  | 조각의 색상을 이만큼 무작위로 돌립니다. 0 = 그대로 | Hue jitter | `0.1` | self-cut `0.1` | ✓ |

### `perlin-texture` — 노이즈 텍스처

펄린 노이즈로 불규칙한 마스크를 만들고 텍스처를 채웁니다(DRAEM) · **이럴 때:** 불규칙한 이상 영역을 흉내낼 때. 보관함 불필요 · EN: Perlin texture (DRAEM)

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `cls` | 클래스 이름 |  | 출력 라벨에 쓸 이름 | Class name | `anomaly` | perlin-texture `anomaly` |  |
| `texture` | 텍스처 출처 |  | self = 바탕 이미지의 다른 부분 · dir = 텍스처 폴더(DTD 등, 읽기만) | Texture source | `self` | perlin-texture `self` |  |
| `texture_dir` | 텍스처 폴더 |  | dir 일 때 읽을 폴더 또는 목록 .txt | Texture folder | `null` |  |  |
| `size_ratio` | 결함 창 크기 비율 |  | 결함이 생길 창 한 변 / 이미지 짧은 변 범위 | Window size ratio | `[0.2, 0.5]` | perlin-texture `[0.2, 0.5]` |  |
| `scale_range` | 노이즈 굵기 지수 |  | 펄린 노이즈 해상도 2^k 의 k 범위. 올리면 잘게, 내리면 큼직하게 | Perlin scale exponent | `[0, 5]` | perlin-texture `[0, 5]` | ✓ |
| `threshold` | 노이즈 임계 |  | 노이즈 값이 이보다 큰 곳이 결함이 됩니다. 올리면 결함이 작고 드물어집니다 | Noise threshold | `0.5` | perlin-texture `0.5` |  |
| `rotate` | 텍스처 회전 | ° | 텍스처를 돌리는 각도 범위 | Texture rotation | `[-90, 90]` | perlin-texture `[-90, 90]` | ✓ |
| `min_area_px` | 최소 면적 | px | 이보다 작은 결함은 만들지 않고 건너뜁니다 | Minimum area | `16` | perlin-texture `16` | ✓ |
| `augment` | 텍스처 증강 |  | 켜면 텍스처에 무작위 증강 3종을 적용합니다(DRAEM 방식) | Augment texture | `true` | perlin-texture `true` | ✓ |
| `max_tries` | 재생성 시도 | 회 | 결함 면적이 부족할 때 다시 만드는 최대 횟수 | Max tries | `5` | perlin-texture `5` | ✓ |

## 크기·회전 / Geometry — `pipeline.geometry`

조각의 크기·회전·뒤집기·휘어짐

### `affine` — 크기·회전·뒤집기

조각을 배율·각도·뒤집기로 바꾸고, 잔물결·휘어짐을 더할 수 있습니다 · EN: Affine

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `scale` | 크기 배율 | 배 | 조각을 붙일 때 곱하는 배율 범위(픽셀 크기 자동 축척 뒤). 넓히면 크기가 다양해지고 1 근처로 좁히면 실제 크기를 지킵니다 | Scale range | `[0.8, 1.25]` | alpha-paste `[0.8, 1.25]` · annulus-graft `[0.8, 1.25]` · dent-graft `[0.9, 1.1]` · hard-paste `[0.8, 1.25]` · multiband-graft `[0.8, 1.25]` · perlin-texture `[1, 1]` · poisson-graft `[0.8, 1.25]` · relative-paste `[0.8, 1.25]` · self-cut `[1, 1]` · structure-aware-graft `[0.8, 1.25]` |  |
| `rotate` | 회전 | ° | 돌리는 각도 범위. 빛 방향이 정해진 결함(찍힘)은 ±15° 안으로 — 뒤집히면 음영이 물리적으로 틀립니다 | Rotation range | `[-180, 180]` | alpha-paste `[-180, 180]` · annulus-graft `[-180, 180]` · dent-graft `[-15, 15]` · hard-paste `[-180, 180]` · multiband-graft `[-180, 180]` · perlin-texture `[0, 0]` · poisson-graft `[-180, 180]` · relative-paste `[-180, 180]` · self-cut `[-45, 45]` · structure-aware-graft `[-180, 180]` |  |
| `flip` | 뒤집기 |  | 없음 · 좌우 · 상하 · 둘 다. 빛 방향이 정해진 결함은 상하 뒤집기가 조명을 뒤집습니다 | Flip | `both` | alpha-paste `both` · annulus-graft `both` · dent-graft `none` · hard-paste `both` · multiband-graft `both` · perlin-texture `none` · poisson-graft `both` · relative-paste `both` · self-cut `both` · structure-aware-graft `both` |  |
| `elastic.alpha` | 잔물결 세기 |  | 조각을 국소적으로 흔드는 세기. 0 = 끔 | Elastic strength | `0` | alpha-paste `0` · annulus-graft `0` · dent-graft `0` · hard-paste `0` · multiband-graft `0` · perlin-texture `0` · poisson-graft `0` · relative-paste `0` · self-cut `0` · structure-aware-graft `0` | ✓ |
| `elastic.sigma` | 잔물결 굵기 | px | 흔들림의 파장. 크면 완만하게 | Elastic sigma | `4` | alpha-paste `4` · annulus-graft `4` · dent-graft `4` · hard-paste `4` · multiband-graft `4` · perlin-texture `4` · poisson-graft `4` · relative-paste `4` · self-cut `4` · structure-aware-graft `4` | ✓ |
| `tps.points` | 휘어짐 제어점 | 개 | 전역 휘어짐(thin-plate spline)에 쓰는 제어점 격자 한 변의 수 | TPS grid points | `3` |  | ✓ |
| `tps.jitter` | 휘어짐 세기 |  | 제어점을 짧은 변의 이 비율만큼 흔들어 조각을 휩니다. 0 = 끔 | TPS jitter | `0` |  | ✓ |
| `per_class` | 클래스별 예외 |  | 클래스마다 크기·회전·뒤집기를 따로 둡니다(찍힘만 ±15° 등). 카드의 표에서 편집 | Per-class overrides | `{}` |  | ✓ |

## 위치 정하기 / Placement — `pipeline.placement`

바탕 이미지의 어디에 놓을지(붙일 수 있는 영역 안에서)

### `sampled` — 무작위 자리

허용 영역 안에서 자리를 무작위로 뽑습니다(고르게 · 가장자리 · 가운데) · **이럴 때:** 기본 · EN: Sampled

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `margin_px` | 가장자리 여유 | px | 붙일 수 있는 영역 경계에서 이만큼 안쪽에만 놓습니다. Poisson 은 경계에 닿으면 실패하므로 0 은 피하세요 | Edge margin | `8` | alpha-paste `8` · annulus-graft `8` · hard-paste `8` · multiband-graft `8` · perlin-texture `8` · poisson-graft `8` · relative-paste `8` · self-cut `8` |  |
| `max_tries` | 자리 찾기 시도 | 회 | 겹치지 않는 자리를 찾는 최대 횟수. 넘으면 축소 시도 뒤 건너뜁니다 | Max placement tries | `50` | alpha-paste `50` · annulus-graft `50` · hard-paste `50` · multiband-graft `50` · perlin-texture `50` · poisson-graft `50` · relative-paste `50` · self-cut `50` | ✓ |
| `shrink_on_fail.factor` | 실패 시 축소 배율 | 배 | 자리를 못 찾으면 조각을 이 배율로 줄여 다시 시도합니다 | Shrink factor on failure | `0.8` | alpha-paste `0.8` · annulus-graft `0.8` · hard-paste `0.8` · multiband-graft `0.8` · perlin-texture `0.8` · poisson-graft `0.8` · relative-paste `0.8` · self-cut `0.8` | ✓ |
| `shrink_on_fail.rounds` | 축소 반복 | 회 | 축소를 반복하는 최대 횟수. 0 = 축소 안 함 | Shrink rounds | `3` | alpha-paste `3` · annulus-graft `3` · hard-paste `3` · multiband-graft `3` · perlin-texture `3` · poisson-graft `3` · relative-paste `3` · self-cut `3` | ✓ |
| `distribution` | 위치 분포 |  | uniform = 영역 안 고르게 · edge = 가장자리 쪽 · center = 가운데 쪽 | Position distribution | `uniform` | alpha-paste `uniform` · annulus-graft `uniform` · hard-paste `uniform` · multiband-graft `uniform` · perlin-texture `uniform` · poisson-graft `uniform` · relative-paste `uniform` · self-cut `uniform` |  |

### `structure-aware` — 표면 결을 따라

결·에지가 있는 곳을 선호하고 조각을 결 방향에 맞춰 돌립니다 · **이럴 때:** 스크래치가 결을 따르고 칩이 모서리에 생기는 부품 · EN: Structure-aware

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `margin_px` | 가장자리 여유 | px | 붙일 수 있는 영역 경계에서 이만큼 안쪽에만 놓습니다. Poisson 은 경계에 닿으면 실패하므로 0 은 피하세요 | Edge margin | `8` | dent-graft `8` · structure-aware-graft `8` |  |
| `max_tries` | 자리 찾기 시도 | 회 | 겹치지 않는 자리를 찾는 최대 횟수. 넘으면 축소 시도 뒤 건너뜁니다 | Max placement tries | `50` | dent-graft `50` · structure-aware-graft `50` | ✓ |
| `shrink_on_fail.factor` | 실패 시 축소 배율 | 배 | 자리를 못 찾으면 조각을 이 배율로 줄여 다시 시도합니다 | Shrink factor on failure | `0.8` | dent-graft `0.8` · structure-aware-graft `0.8` | ✓ |
| `shrink_on_fail.rounds` | 축소 반복 | 회 | 축소를 반복하는 최대 횟수. 0 = 축소 안 함 | Shrink rounds | `3` | dent-graft `3` · structure-aware-graft `3` | ✓ |
| `prefer` | 선호 표면 |  | edges = 결·에지가 있는 곳 · flat = 평탄한 곳 · uniform = 가리지 않음 | Preferred surface | `edges` | dent-graft `uniform` · structure-aware-graft `edges` |  |
| `strength` | 선호 세기 |  | 선호 표면 가중의 지수. 0 = 무작위와 같음 | Preference strength | `1` | dent-graft `1` · structure-aware-graft `1` | ✓ |
| `smooth_px` | 표면 평활 | px | 결 세기를 이만큼 뭉개서 봅니다. 크면 넓은 결만 | Smoothing | `3` | dent-graft `3` · structure-aware-graft `3` | ✓ |
| `align` | 결 방향 정렬 |  | along = 결 방향으로 · across = 결에 수직 · none = 회전 유지 | Align to structure | `along` | dent-graft `along` · structure-aware-graft `along` |  |
| `min_coherence` | 정렬 최소 일관성 |  | 자리의 결 방향이 이보다 흐리면 정렬하지 않습니다 | Min coherence | `0.2` | dent-graft `0.2` · structure-aware-graft `0.2` | ✓ |
| `min_anisotropy` | 정렬 최소 길쭉함 |  | 조각이 이보다 둥글면 정렬하지 않습니다 | Min anisotropy | `0.1` | dent-graft `0.1` · structure-aware-graft `0.1` | ✓ |
| `jitter_deg` | 정렬 흔들림 | ° | 정렬 각도에 더하는 무작위 ±각 | Alignment jitter | `10` | dent-graft `5` · structure-aware-graft `10` | ✓ |
| `max_align_deg` | 정렬 회전 상한 | ° | 이보다 큰 회전이 필요한 자리는 정렬하지 않습니다(찍힘은 30). 끄면 제한 없음 | Max alignment rotation | `null` | dent-graft `30` | ✓ |

## 붙이기 / Blend — `pipeline.blend`

조각을 바탕에 어떻게 붙일지(그대로 · 가장자리 부드럽게 · Poisson · 다중 대역)

### `paste` — 그대로 붙이기

마스크 안 픽셀을 그대로 덮어씁니다. 가장 거친 대조군 · **이럴 때:** 학습 실험의 대조군 · 노출 보정(relative)과 함께 · EN: Paste

(조정할 값 없음)

### `alpha` — 가장자리 부드럽게

마스크 경계를 페더로 서서히 섞고, 결함마다 불투명도를 줄 수 있습니다 · **이럴 때:** 경계만 감추면 되는 텍스처성 결함 · EN: Alpha

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `feather_px` | 가장자리 부드럽게 | px | 조각 경계를 이만큼 서서히 섞습니다 | Feather | `3` | alpha-paste `2` · perlin-texture `1` |  |
| `opacity` | 불투명도 |  | 결함마다 이 범위의 불투명도로 겹칩니다. 끄면 완전 불투명 | Opacity range | `null` | perlin-texture `[0.4, 1]` | ✓ |

### `poisson` — 경계 자연스럽게(Poisson)

경계의 그래디언트를 맞춰 이음새 없이 붙입니다. 결함 톤이 바탕에 끌려가 옅어질 수 있습니다 · **이럴 때:** 얼룩·스크래치 등 대부분(기본). 대비가 신호인 구멍엔 relative-paste · EN: Poisson

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `poisson_mode` | Poisson 방식 |  | normal = 조각 그래디언트 그대로(얼룩처럼 약한 결함) · mixed = 배경과 강한 쪽 선택(스크래치·핏) | Poisson mode | `mixed` | annulus-graft `normal` · dent-graft `normal` · poisson-graft `normal` · structure-aware-graft `normal` |  |
| `mask_dilate_px` | 풀이 영역 여유 | px | Poisson 이 다시 계산하는 영역을 마스크보다 이만큼 넓힙니다. 5 미만이면 가는 스크래치가 통째로 사라집니다 | Solve-region dilation | `5` | annulus-graft `5` · dent-graft `5` · poisson-graft `5` · structure-aware-graft `5` | ✓ |
| `feather_px` | 대체 처리 페더 | px | Poisson 이 실패해 알파로 대체될 때의 경계 부드럽기 | Fallback feather | `3` | annulus-graft `3` · dent-graft `3` · poisson-graft `3` · structure-aware-graft `3` | ✓ |

### `multiband` — 다중 대역

라플라시안 피라미드로 굵은 톤은 넓게, 잔결은 좁게 섞습니다 · **이럴 때:** 큰 얼룩처럼 톤 전이가 넓은 결함 · EN: Multiband

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `levels` | 피라미드 단계 |  | 다중 대역 혼합의 단계 수(상한). 조각 두께에 맞춰 자동으로 줄어듭니다 | Pyramid levels | `4` | multiband-graft `4` |  |

## 색·밝기 맞추기 / Harmonize — `pipeline.harmonize`

붙인 결함의 색·밝기를 주변에 맞추기

### `none` — 안 함

색·밝기를 맞추지 않습니다 · **이럴 때:** hard-paste 대조군 · EN: None

(조정할 값 없음)

### `stats` — 평균·분산 맞춤

결함 안쪽 L 채널의 평균·표준편차를 주변 링에 맞춥니다. 세기만큼 대비가 옅어집니다 · **이럴 때:** 기본(세기 0.3) · EN: Stats

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `strength` | 맞추는 세기 |  | 결함 안쪽을 주변 링에 맞추는 정도. 올릴수록 자연스럽지만 결함 대비가 (1−세기) 배로 옅어집니다 | Strength | `0.5` | annulus-graft `0.3` · dent-graft `0.2` · poisson-graft `0.3` · structure-aware-graft `0.3` |  |
| `ring_px` | 주변 링 폭 | px | 비교 기준이 되는 결함 둘레 띠의 폭 | Ring width | `12` | annulus-graft `12` · dent-graft `12` · poisson-graft `12` · structure-aware-graft `12` | ✓ |

### `reinhard` — Reinhard

Lab 세 채널의 평균·표준편차를 주변에 맞춥니다(색까지) · **이럴 때:** 색조 차이가 큰 조각 · EN: Reinhard

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `strength` | 맞추는 세기 |  | Lab 세 채널을 주변에 맞추는 정도. 올릴수록 대비가 옅어집니다 | Strength | `0.5` | alpha-paste `0.5` |  |
| `ring_px` | 주변 링 폭 | px | 비교 기준이 되는 결함 둘레 띠의 폭 | Ring width | `12` | alpha-paste `12` | ✓ |

### `histmatch` — 히스토그램 맞춤

결함 안쪽 밝기 분포를 주변 링 분포에 맞춥니다 · **이럴 때:** 밝기 분포 모양까지 맞춰야 할 때 · EN: Histogram match

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `strength` | 맞추는 세기 |  | 밝기 분포를 주변에 맞추는 정도. 올릴수록 대비가 옅어집니다 | Strength | `0.5` | multiband-graft `0.3` |  |
| `ring_px` | 주변 링 폭 | px | 비교 기준이 되는 결함 둘레 띠의 폭 | Ring width | `12` | multiband-graft `12` | ✓ |

### `relative` — 노출만 맞춤

조각 주변과 바탕 주변의 밝기 차만큼 결함을 옮깁니다 — 결함의 상대 대비는 그대로 · **이럴 때:** 대비가 곧 신호인 결함(구멍·핏). paste 뒤에만 · EN: Relative (exposure offset)

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `strength` | 노출 보정 세기 |  | 조각 주변과 바탕 주변의 밝기 차만큼 결함을 옮깁니다. 1 = 전부. 결함의 상대 대비는 지킵니다 | Exposure-offset strength | `1` | relative-paste `1` |  |
| `ring_px` | 주변 링 폭 | px | 밝기 차를 재는 결함 둘레 띠의 폭 | Ring width | `12` | relative-paste `12` | ✓ |
| `gain` | 대비 배율도 맞춤 |  | 켜면 표준편차 비로 대비 배율까지 맞춥니다(링이 작아 잡음 — 보통 끔) | Match gain too | `false` | relative-paste `false` | ✓ |

## 카메라 효과 / Degrade — `pipeline.degrade`

노이즈·흐림·JPEG 등 카메라 느낌 입히기

### `none` — 안 함

카메라 효과를 넣지 않습니다 · EN: None

(조정할 값 없음)

### `camera` — 카메라 재현

노이즈 · 흐림 · JPEG · 움직임 흐림 · 가장자리 어둡게 · 감마를 무작위 범위로 · **이럴 때:** 기본 · EN: Camera

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `noise_sigma` | 노이즈 | gray | 가우시안 노이즈 표준편차 범위. 올리면 거칠게 | Noise sigma | `[0, 2]` | alpha-paste `[0, 2]` · annulus-graft `[0, 2]` · dent-graft `[0, 2]` · multiband-graft `[0, 2]` · poisson-graft `[0, 2]` · relative-paste `[0, 2]` · structure-aware-graft `[0, 2]` |  |
| `blur_sigma` | 흐림 | px | 가우시안 블러 σ 범위. 올리면 초점이 흐려집니다 | Blur sigma | `[0, 0.6]` | alpha-paste `[0, 0.6]` · annulus-graft `[0, 0.6]` · dent-graft `[0, 0.6]` · multiband-graft `[0, 0.6]` · poisson-graft `[0, 0.6]` · relative-paste `[0, 0.6]` · structure-aware-graft `[0, 0.6]` |  |
| `jpeg_quality` | JPEG 화질 |  | 이 범위의 화질로 다시 압축합니다. 끄면 압축 없음 | JPEG quality | `null` |  | ✓ |
| `motion_blur_px` | 움직임 흐림 | px | 직선 모션 블러 길이 범위. 끄면 없음 | Motion blur length | `null` |  | ✓ |
| `motion_angle` | 움직임 방향 | ° | 모션 블러 방향 범위(0~180) | Motion blur angle | `[0, 180]` |  | ✓ |
| `vignette` | 가장자리 어둡게 |  | 모서리 감광 세기 범위(1 = 완전히 검게). 끄면 없음 | Vignette | `null` |  | ✓ |
| `gamma` | 감마 |  | 톤 커브 지수 범위(1 = 그대로, 1 미만 = 밝게). 끄면 없음 | Gamma | `null` |  | ✓ |

## 정답 영역 / GT mask — `pipeline.gtmask`

학습용 정답 영역(마스크)을 무엇으로 삼을지

### `source` — 원본 마스크

조각의 마스크를 그대로 정답으로 씁니다 · **이럴 때:** paste 처럼 마스크 밖이 안 바뀌는 붙이기 · EN: Source mask

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `diff_threshold` | 변화 감지 임계 | gray | 바탕과 이만큼 이상 달라진 픽셀을 결함으로 봅니다. 내리면 넓게 잡힙니다 | Diff threshold | `12` | hard-paste `12` · perlin-texture `12` · relative-paste `12` · self-cut `12` | ✓ |
| `dilate_px` | 정답 영역 팽창 | px | 정답 영역을 이만큼 넓혀 경계 오차를 흡수합니다 | GT dilation | `2` | hard-paste `0` · perlin-texture `0` · relative-paste `0` · self-cut `0` | ✓ |

### `diff` — 바뀐 픽셀

바탕과 임계 이상 달라진 픽셀만 정답으로 삼습니다 · **이럴 때:** Poisson 이 지운 저대비 부분을 라벨에서 빼고 싶을 때 · EN: Diff

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `diff_threshold` | 변화 감지 임계 | gray | 바탕과 이만큼 이상 달라진 픽셀을 결함으로 봅니다. 내리면 넓게 잡힙니다 | Diff threshold | `12` |  | ✓ |
| `dilate_px` | 정답 영역 팽창 | px | 정답 영역을 이만큼 넓혀 경계 오차를 흡수합니다 | GT dilation | `2` |  | ✓ |

### `union` — 합집합

원본 마스크와 바뀐 픽셀을 합치고 팽창합니다 — 붙이기 오차를 흡수(기본) · **이럴 때:** 기본 · EN: Union

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `diff_threshold` | 변화 감지 임계 | gray | 바탕과 이만큼 이상 달라진 픽셀을 결함으로 봅니다. 내리면 넓게 잡힙니다 | Diff threshold | `12` | alpha-paste `12` · annulus-graft `12` · dent-graft `12` · multiband-graft `12` · poisson-graft `12` · structure-aware-graft `12` | ✓ |
| `dilate_px` | 정답 영역 팽창 | px | 정답 영역을 이만큼 넓혀 경계 오차를 흡수합니다 | GT dilation | `2` | alpha-paste `2` · annulus-graft `2` · dent-graft `2` · multiband-graft `2` · poisson-graft `2` · structure-aware-graft `2` | ✓ |

## 붙일 수 있는 영역 / ROI — `pipeline.placement.roi`

바탕 이미지에서 결함을 놓아도 되는 영역(배경·리세스 제외)

### `otsu` — 밝기로 자동 분리

밝기 임계(Otsu)로 물체와 배경을 가르고 물체 안만 허용합니다 · **이럴 때:** 물체가 배경과 밝기로 뚜렷이 갈릴 때(기본) · EN: Otsu

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `invert` | 밝기 반전 |  | auto = 물체가 배경보다 밝은지 자동 · yes/no 로 고정 | Invert polarity | `auto` | alpha-paste `auto` · hard-paste `auto` · multiband-graft `auto` · perlin-texture `auto` · poisson-graft `auto` · relative-paste `auto` · self-cut `auto` | ✓ |
| `erode_px` | 경계 깎기 | px | 영역 가장자리를 이만큼 안쪽으로 줄여 경계에 걸치지 않게 합니다 | Erode edge | `8` | alpha-paste `8` · dent-graft `8` · hard-paste `8` · multiband-graft `8` · perlin-texture `8` · poisson-graft `8` · relative-paste `8` · self-cut `8` |  |

### `none` — 제한 없음

이미지 어디든 놓습니다 · **이럴 때:** 배경이 없는 타일·표면 사진 · EN: None

(조정할 값 없음)

### `mask_dir` — 마스크 폴더

바탕 이미지와 같은 이름의 마스크 PNG(흰색 = 허용)를 읽습니다 · **이럴 때:** 결함 표시 탭에서 영역을 직접 칠했을 때 · EN: Mask folder

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `path` | 마스크 폴더 |  | 바탕 이미지와 같은 이름의 마스크 PNG 폴더(흰색 = 허용) | Mask folder | `(필수)` |  |  |

### `grabcut` — 전경 자동 분리

GrabCut 으로 전경(물체)을 찾아 그 안만 허용합니다. 느리지만 밝기 분리보다 정확 · **이럴 때:** 밝기만으로는 물체/배경이 안 갈릴 때 · EN: GrabCut

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `init` | 초기화 |  | rect = 테두리를 배경으로 두고 안쪽에서 전경 찾기 · otsu = 밝기 분리 결과를 다듬기 | Initialization | `rect` | structure-aware-graft `rect` | ✓ |
| `invert` | 밝기 반전 |  | 초기화가 otsu 일 때의 극성. auto = 자동 | Invert polarity | `auto` |  | ✓ |
| `rect_margin` | 테두리 배경 비율 |  | rect 초기화에서 확정 배경으로 둘 테두리 두께(이미지 비율) | Border margin ratio | `0.03` | structure-aware-graft `0.03` | ✓ |
| `iters` | 반복 | 회 | GrabCut 반복 횟수. 올리면 정확하지만 느립니다 | Iterations | `5` | structure-aware-graft `5` | ✓ |
| `work_px` | 계산 해상도 | px | 긴 변을 이만큼 줄여 계산합니다(4K 원본은 수십 초). 0 = 원본 | Working resolution | `1024` | structure-aware-graft `1024` | ✓ |
| `erode_px` | 경계 깎기 | px | 영역 가장자리를 이만큼 안쪽으로 줄입니다 | Erode edge | `8` | structure-aware-graft `8` |  |

### `annulus` — 링(도넛) 영역

원형 부품의 중심·반지름을 찾아 안쪽~바깥 반지름 사이 링만 허용합니다 · **이럴 때:** 원형 부품의 가공 링 면(리세스 제외) · EN: Annulus

| 키 | 라벨 | 단위 | 설명 | EN | 기본 | 프리셋 | 고급 |
|---|---|---|---|---|---|---|---|
| `center` | 링 중심 | px | 끄면 바탕마다 자동 검출(가장 큰 물체의 최소 외접원) | Ring center | `null` |  | ✓ |
| `radius` | 기준 반지름 | px | 비율의 기준. 끄면 자동 검출 | Reference radius | `null` |  | ✓ |
| `r_inner` | 안쪽 반지름 |  | 기준 반지름의 배율(단위 ratio) 또는 px. 이 안쪽은 허용하지 않습니다 | Inner radius | `0.55` | annulus-graft `0.55` |  |
| `r_outer` | 바깥 반지름 |  | 기준 반지름의 배율(단위 ratio) 또는 px. 이 바깥은 허용하지 않습니다 | Outer radius | `0.9` | annulus-graft `0.9` |  |
| `units` | 반지름 단위 |  | ratio = 기준 반지름의 배율 · px = 절대값 | Radius units | `ratio` | annulus-graft `ratio` | ✓ |
| `invert` | 밝기 반전 |  | 자동 검출에 쓰는 밝기 분리의 극성. auto = 자동 | Invert polarity | `auto` | annulus-graft `auto` | ✓ |
| `erode_px` | 경계 깎기 | px | 링의 안·바깥 경계를 이만큼 깎습니다 | Erode edge | `4` | annulus-graft `4` |  |

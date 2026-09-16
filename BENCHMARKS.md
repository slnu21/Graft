# BENCHMARKS — 공개 데이터로 잰 것 (결정 아님, 근거)

> 사용자 실데이터 없이 **GT 가 있는 공개 데이터**로 도구의 두 질문에 첫 숫자를 붙인다: (1) 박스→마스크 추정이 얼마나 맞나, (2) 합성 데이터가 검출력을 올리나. 재현 명령은 각 절에. 라이선스: MVTec AD CC BY-NC-SA(로컬 개발용) · Magnetic Tile 논문 인용.

## 1. 박스→마스크 추정 (`tools/bench_mask_from_box.py`, 2026-09-16)

1180 인스턴스(MVTec metal_nut·screw·grid·bottle·hazelnut·carpet + Magnetic Tile), 박스 = GT 성분 bbox 각 변 10 % 느슨. IoU 는 클래스 중앙값, 실패 = 사슬 IoU < 0.3, 저신뢰 = `confidence < 0.5`.

| 데이터셋 | 사슬(grabcut→otsu→ellipse) IoU / 실패율 | ellipse | 하이브리드(저신뢰면 ellipse = `--mask-from hybrid`) |
|---|---|---|---|
| metal_nut(경면 금속) | 0.641 / 0.31 | 0.503 / 0.14 | 0.522 / 0.15 |
| screw(흑백) | 0.287 / 0.51 | 0.497 / 0.05 | 0.488 / 0.21 |
| grid(흑백 텍스처) | 0.253 / 0.58 | 0.431 / 0.14 | 0.431 / 0.19 |
| bottle | 0.295 / 0.51 | 0.377 / 0.22 | 0.358 / 0.28 |
| hazelnut | 0.556 / 0.21 | 0.413 / 0.27 | 0.487 / 0.27 |
| carpet(텍스처) | 0.215 / 0.79 | 0.525 / 0.24 | 0.511 / 0.25 |
| magnetic-tile(산업 흑백) | 0.444 / 0.36 | 0.554 / 0.15 | 0.658 / 0.20 |
| **전체** | **0.367 / 0.43** | **0.487 / 0.16** | **0.502 / 0.21** (오라클 best-of-4 0.564 / 0.12) |

읽는 법은 `KNOWN-ISSUES.md` "결정 근거" 절. 재현: `python tools/bench_mask_from_box.py samples/mvtec/metal_nut … samples/magnetic-tile/pairs.csv --out bench.md`(전체 ≈ 75 분, `--limit 12` ≈ 10 분).

## 2. 합성 유/무 YOLO mAP (`tools/train_mvtec_map.py`, 2026-09-17)

MVTec **metal_nut**, 클래스 bent·color·scratch(flip 은 부품 전체 이상이라 제외). real-train = 클래스당 8장(박스 라벨), real-val = 나머지 46장 + good 11장(홀드아웃, 학습·은행에 안 들어감). 은행 = real-train 24장의 박스만(`import-yolo`, grabcut 사슬) → 소스 38. 합성 = 프리셋별 200장(대상 train/good 220장, 정상 220장도 빈 라벨로 학습셋에). yolov8n · imgsz 320 · 40 epochs · CPU · seed 7.

| 학습셋 | mAP50 | mAP50-95 | 클래스별 mAP50 (bent · color · scratch) | 학습 시간 |
|---|---|---|---|---|
| **A** real-train 24장(+정상 11)만 | 0.267 | 0.168 | 0.36 · 0.28 · 0.17 | 3.6 분 |
| **B** A + `poisson-graft` 합성 179장(+정상 220) | **0.371** (+0.10) | 0.177 | 0.36 · **0.40** · **0.35** | 24.5 분 |
| **B** A + `dent-graft` 합성 188장(+정상 220) | 0.329 (+0.06) | 0.175 | **0.40** · 0.38 · 0.21 | 22.8 분 |

같은 절차에서 은행만 **`--mask-from hybrid`**(저신뢰 사슬 결과 → 내접 타원)로 바꾸면(`out/train-map-hybrid`, A 는 동일 0.267 로 재현):

| 학습셋 | mAP50 | mAP50-95 | 클래스별 mAP50 (bent · color · scratch) |
|---|---|---|---|
| **B** A + `poisson-graft` 합성(은행 grabcut) | 0.371 | 0.177 | 0.36 · 0.40 · 0.35 |
| **B** A + `poisson-graft` 합성(은행 **hybrid**) | **0.462** (+0.20 vs A) | **0.238** | **0.46 · 0.47 · 0.46** |

### 분산 — 분할·시드 3개 (같은 절차, `--seed` 7·8·9 = 분할도 함께 바뀜)

| 카테고리 | seed(분할) | A 실제만 | B +poisson (은행 grabcut) | B +poisson (은행 **hybrid**) |
|---|---|---|---|---|
| metal_nut | 7 | 0.267 | 0.371 (+0.10) | 0.462 (+0.19) |
| metal_nut | 8 | 0.404 | 0.332 (−0.07) | 0.454 (+0.05) |
| metal_nut | 9 | 0.185 | 0.469 (+0.28) | 0.461 (+0.28) |
| **metal_nut 평균 Δ** | | | **+0.10** (−0.07 ~ +0.28) | **+0.17** (+0.05 ~ +0.28) |
| screw(흑백 5 클래스, ROI otsu) | 7 | 0.292 | 0.291 (±0.00) | 0.255 (−0.04) |

- **실제 24장뿐인 기준선(A)은 분할에 따라 0.18~0.40 으로 요동**한다 — 어느 8장이 train 에 들어가느냐가 곧 성능. 그래서 비교는 **같은 분할 안의 Δ** 로만 읽는다.
- **hybrid 은행은 세 분할 모두 +**(0.45~0.46 으로 수렴 — 합성 179장이 학습셋을 지배해 분할 의존이 준다). **grabcut 은행은 한 분할에서 −0.07** — 엉뚱한 마스크(§1 실패율 0.31)가 붙은 합성이 실제 24장의 신호를 흐린 경우. 마스크 품질이 나쁘면 합성이 해가 될 수 있다는 것도 첫 숫자.
- **screw 에선 합성이 도움이 안 됐다**(grabcut ±0, hybrid −0.04, seed 7 하나). 가늘고 희미한 스크래치·나사산 결함은 §1 에서도 IoU 가 낮은 클래스(thread_side 0.23, thread_top 0.21)라 마스크가 결함 경계를 못 잡고, `manipulated_front`·`thread_side` 는 실제 24장으로도 0.0x — 흑백·저대비 결함엔 박스 추정보다 **라벨 탭에서 마스크를 그려 은행을 만드는 흐름**이 맞다는 신호. 카테고리 한 번·시드 하나라 단정은 금물.
- 학습 시드만 바꾸고 분할을 고정하려면 `--split-seed`(v0.8.1+).

읽는 법:
- **합성이 mAP50 을 0.27 → 0.37 로 올렸다**(같은 홀드아웃 57장). 가장 큰 이득은 **scratch**(0.17 → 0.35)와 color(0.28 → 0.40) — 실제 24장으로는 못 배우던 가늘고 긴 결함을 합성 179장이 보탠다. `dent-graft`(±15°·flip 없음)는 **bent** 에서 가장 좋고(0.36 → 0.40) scratch 에선 poisson(±180°)보다 못하다 — 회전 범위가 결함 성격을 따라가야 한다는 것(`geometry.per_class` 의 근거).
- **마스크 품질이 곧 검출력이다**: 같은 합성 절차에서 은행 마스크만 hybrid 로 바꿔 mAP50 0.37 → 0.46(§1 의 IoU 0.37 → 0.50 과 같은 방향). 실데이터 2차에서 `hybrid` 기본 전환(`KNOWN-ISSUES.md` 결정 대기)을 지지하는 두 번째 근거 — 단, 아래 분산 주의.
- 한 번의 시드·한 카테고리·CPU 40 epoch 이라 **분산이 크다**(같은 A 를 시드만 바꿔도 ±0.05 는 움직인다) — 결정에 쓰려면 시드 3개 이상, 실데이터에서 같은 절차(`TESTING.md` §3). 실제 NG 24장뿐인 소표본에서 합성 200장이 mAP50 을 올리면 도구의 전제가 살아 있는 것이고, 내리면 합성 결함이 실제 분포와 다르다는 뜻(검수 탭 분포·조명 R 로 원인을 좁힌다).
- 재현: `<train-venv>/python tools/train_mvtec_map.py samples/mvtec/metal_nut --anograft .venv/Scripts/python.exe --k 8 --count 200 --presets poisson-graft dent-graft --epochs 40 --out out/train-map`(≈ 40 분).

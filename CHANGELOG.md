# Changelog

[Keep a Changelog](https://keepachangelog.com/) + [SemVer](https://semver.org/). 상세는 `docs/개발로그/`(로컬).

## [Unreleased]

### Added
- **은행 없이 동작하는 소스 2종**(v0.4): `self-cut`(CutPaste·Scar — 대상 자신의 사각/스카 패치 + 색 지터) · `perlin-texture`(DRAEM — 펄린 노이즈 임계 마스크 + 텍스처(대상 자신 증강 또는 `texture_dir`)). 프리셋 `self-cut` · `perlin-texture`, `inputs.bank: null` 허용(bank 소스만 은행 필수), `recipe init`이 자동으로 `bank: null`. `anograft methods` source 3종, `preview --compare-methods source`.
- `blend.alpha.opacity`(DRAEM β) — 설정 시에만 결함마다 rng 1회(기존 프리셋 결과 불변).
- `core/perlin.py`(펄린 노이즈·마스크), 골든 +4(self-cut·perlin-texture × gray/color).
- **구조 정합 배치** `placement: structure-aware`(v0.4): 위치 = 그래디언트 크기 가중(`prefer` edges/flat/uniform · `strength` · `smooth_px`), 방향 = 후보 자리의 구조 텐서 지배 방향에 패치 주축 정렬(`align` along/across/none · `min_coherence` · `min_anisotropy` · `jitter_deg`). 시도마다 rng 2회. 사이드카 `placement`에 `aligned`·`angle_deg`·`orientation_deg`·`coherence`·`patch_axis_deg`·`anisotropy`.
- **GrabCut ROI** `roi: grabcut`(`init` rect/otsu · `rect_margin` · `iters` · `work_px` 작업 해상도 · `erode_px`) — 시드는 대상 파일 이름의 crc32(폴더를 옮겨도 같은 ROI), 실패 시 Otsu 로 대체(경고). 프리셋 `structure-aware-graft`(poisson-graft + structure-aware + grabcut). 골든 +2. `core/structure.py`(그래디언트 크기·창 구조 텐서·마스크 주축).
- `stable_seed`가 `core/seeds.py`로 이동(`bank.mask_from_box`에서 재export).
- **VisA 어댑터**(`dataset info visa` · `bank import-dataset visa <VisA>/<category>`, v0.4): `Data/Images/{Normal,Anomaly}` + `Data/Masks/Anomaly` + `image_anno.csv`(있으면 정본, 없으면 폴더). 결함 유형 세분이 없어 클래스 `anomaly` 하나(카테고리는 id·tags). 마스크는 0 이 아니면 결함(`PairRecord.mask_threshold`, `imgio.read_mask(threshold=)`) — 0/1 라벨맵 사본 대비. 로컬 사본 읽기만(CC BY-NC-SA).
- **`mvtec` writer**(`output.writer: {format: mvtec, category, test_normal_ratio, layout_dir}`, v0.4): 정본 위에 anomalib 이 읽는 `mvtec/<category>/{train/good, test/good, test/<class>, ground_truth/<class>/*_mask.png}` 를 추가로 쓴다(정본 PNG 사본, 바이트 동일). 정상은 `split_rng(seed)` 로 train/test 분할(결정적), 이미지당 클래스 하나(섞이면 면적 최대 + `writer.mixed` + 경고). `include_normals: false` 면 경고.
- **열화 추가**(`degrade: camera`, v0.4): `motion_blur_px`(+`motion_angle`) 선형 모션 블러 · `vignette` 모서리 감광 · `gamma` 톤 커브 — 전부 `null` = off 이고 off 면 rng 를 소비하지 않는다(기존 레시피·골든 불변). 적용 순서 모션 → 가우시안 → 비네팅 → 노이즈 → 감마 → JPEG. 사이드카 `degrade`에 실제 값.

### Changed
- `placement` 블록의 `method:` 생략은 `sampled`로 해석(v0.4 에서 union 이 되며 태그가 필수가 됐지만 v0.1~v0.3 레시피 호환). `with_method("placement", …)`는 하위 `roi` 블록을 유지.

## [0.1.0] - 2026-09-14

첫 릴리스 — CLI 코어 + GUI 스튜디오 탭 + Windows zip. **샘플 데이터로만 검증**(보유 실데이터·MVTec AD·학습 1 epoch 미검증 — 데이터가 생기면 README "보유 데이터로" 3줄).

### Added
- **결함 은행**: `bank import-yolo`(박스·폴리곤·빈 라벨, 박스→마스크 추정 grabcut/otsu/ellipse/rect + 폴백 사슬, `mask_origin` 기록, `--list-normals`) · `import-pairs`(마스크 PNG 쌍, suffix/stem/CSV 매칭) · `import-dataset mvtec-ad`(로컬 사본 읽기만) · `bank ls`(클래스별 소스 수·정확/추정) · `bank preview`(추정 마스크 그리드, amber = 추정).
- **7단계 파이프라인** `source → geometry(affine·elastic) → placement(sampled, ROI otsu/none/mask_dir) → blend(paste·alpha·poisson·multiband) → harmonize(none·stats·reinhard·histmatch) → degrade(none·camera) → gtmask(source·diff·union)`. `Stage.apply(Context) -> Context`, 레지스트리 `(stage, method)` + 가용/사유, `anograft methods`.
- **프리셋 4종** `poisson-graft`(기본) · `hard-paste` · `alpha-paste` · `multiband-graft` — 레시피 YAML(pydantic v2, method별 discriminated union, `extra=forbid`) + `recipe init/check`.
- **재현성**: `image_rng(seed, index)` 파생 시드, 대상도 rng로 선택, `pipeline_hash`(resolved 레시피 − `output.root`/`count` + 버전 + 은행 지문). `run --workers N`(spawn 풀, 워커는 레시피 YAML만 받아 `prepare`, 메인만 기록) — workers 0/N 바이트 동일(테스트 고정).
- **출력**: 정본 `images/`·`masks/`·`meta/*.json`(사이드카) + `manifest.csv` + `recipe.resolved.yaml`(`pairs` writer, 항상) + `yolo` writer(`labels/*.txt`·`data.yaml`, seg 폴리곤 옵션, `import-yolo` 왕복 테스트).
- `preview --index i` 3패널 · `--compare-methods blend|harmonize`(결함 주변 크롭 확대, `--full`).
- **GUI**(`[gui]` extra, PySide6): 스튜디오 탭 — 대상 레일·프리셋/시드·A/B 와이프 캔버스(GT/ROI 오버레이·줌)·시드 변형 6장·파이프라인 7카드(method 콤보 + 단계 썸네일)·레시피 열기/저장. 미리보기는 긴 변 1024 축소본. 다른 탭은 자리만.
- **샘플 데이터** `anograft sample --out DIR`(브러시드 메탈 + scratch/pit/stain, YOLO 박스) — 데이터 없이 5분 시작.
- **Windows zip**(`tools/build_zip.ps1`, PyInstaller onedir): `anograft.exe`(CLI) + `anograft-gui.exe` + `recipes/` + 라이선스 문서, 파이썬 불필요. 프리셋 동봉, spawn 워커 `freeze_support`. 72 MB(풀면 182 MB).
- `tools/train_smoke.py`: 출력을 기존 YOLO 셋과 합쳐(`syn_`/`base_` 접두어, 클래스 순서 검사) `ultralytics` 1 epoch(선택 설치, `--dry-run`).
- `LICENSE`(MIT) · `THIRD-PARTY-NOTICES.md`(PySide6/Qt LGPL 동적 링크, OpenCV Apache-2.0, PyInstaller 부트로더 예외, MVTec 비재배포) · `PRIVACY.md`(오프라인·로컬).
- CI: Windows·Linux × py3.10·3.12 ruff+pytest+wheel; `release.yml` 태그 → PyPI(Trusted Publishing) + Windows zip → GitHub Release 자산.

### Changed
- 프리셋 기본값(샘플 은행 실측, hard-paste 대비 마스크 안 L1 유지율): `poisson-graft` MIXED → **NORMAL**(얼룩 0.19 → 0.34) · stats 0.5 → **0.3**; `alpha-paste` feather 3 → **2** · reinhard 0.7 → **0.5**; `multiband-graft` histmatch 0.5 → **0.3**. 골든 6장 갱신.
- `pipeline_hash`가 `output.root`·`count`를 제외 — `--out`만 바꾼 두 실행의 사이드카 해시가 같다.
- CLI 콘솔 인코딩 fail-soft(`errors="replace"`) — cp949 콘솔에서 기호 때문에 죽지 않음.

### Known limitations
- 보유 실데이터·MVTec AD 실제 사본·학습 1 epoch 미검증(샘플로만). `diff` 임계 12의 텍스처 과검출, grabcut vs otsu 기본값, YOLO 박스 `dilate_px` 반영 여부는 실데이터에서 결정.
- GUI는 스튜디오 탭만. 재현은 같은 OS·OpenCV 부버전 범위(`seamlessClone` 솔버).
- `release.yml`의 Windows zip 잡은 첫 push 전이라 CI에서 미검증(로컬 `tools/build_zip.ps1`와 같은 절차).

[Unreleased]: https://github.com/slnu21/Graft/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/slnu21/Graft/releases/tag/v0.1.0

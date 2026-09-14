# Changelog

[Keep a Changelog](https://keepachangelog.com/) + [SemVer](https://semver.org/). 상세는 `docs/개발로그/`(로컬).

## [Unreleased]

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

# Changelog

[Keep a Changelog](https://keepachangelog.com/) + [SemVer](https://semver.org/). 상세는 `docs/개발로그/`(로컬).

## [Unreleased]

### Added
- **검수 리포트**: 검수 탭 **리포트** 버튼 / `anograft dataset report <root> [--out x.html] [--no-bank]` → HTML 한 장(의존성 0, 인라인 SVG) — 합성/정상/skipped · 채택/반려/미검수/검수율/폴백 타일, 합성 vs 실제 면적·긴 변 히스토그램, 클래스별 인스턴스(채택+미검수), 반려 목록(index·클래스·메모), skipped 사유, 레시피·시드·프리셋·pipeline_hash. 기본 `<root>/review-report.html`.
- **`anograft bank merge <a> <b> … --out <c> [--rename old=new] [--tags a,b]`** — 여러 은행을 하나로. 세 파일을 그대로 복사(크롭·마스크·추정 점수 보존), 클래스 이름 기준 병합(대상 순서 유지), id 충돌 `-dup<n>`, 은행 기본 `um_per_px` 를 소스 메타에 실체화, `merged_from` 기록. 대상이 은행이면 이어 쓴다.

### Changed
- **레시피 상대경로**(KNOWN-ISSUES #9 보완): `inputs.bank`·`inputs.targets`·`source.texture_dir`·`roi.path` 가 상대경로면 **cwd 우선, 없으면 레시피 파일 기준**으로 해석한다(`resolve_recipe_paths`, 하위 호환 — repo 루트에서 `recipes/x.yaml` 을 돌리는 방식은 그대로). CLI 는 `경로: …` 로, 스튜디오는 상태바로 알린다. GUI 최근 레시피 복원·검수 탭 실제 분포가 다른 폴더에서 열어도 동작. `output.root` 는 여전히 cwd 기준.
- **ROI 캐시**(대상당 ROI 1회): 스튜디오 변형 k 개·카드 파라미터 편집·CLI `run` 의 대상 재추첨에서 같은 대상의 ROI(grabcut 은 1400² 에서 수 초~수십 초)를 다시 풀지 않는다 — `Pipeline.roi_cache`(LRU 16, 키 = 대상 경로·크기·ROI 설정), `runner.prepare` 가 켜고 `reprepare` 가 이어 받는다. 결과 바이트·사이드카 불변(ROI 스테이지는 rng 를 쓰지 않는다). 1400² grabcut 변형 2.74 s → 0.20 s.

## [0.7.0] - 2026-09-16

v0.7 — **GUI 5탭 전부 실물**: 은행 탭 · 검수 탭 · COCO writer · DTD 텍스처. 흐름 = 라벨 → 은행 → 스튜디오 → 배치 → 검수 → 정리본.

### Fixed
- v0.6.0 CI ubuntu 실패 1건 — 스튜디오 카드 경로 필드 `coerce` 가 POSIX 에서 백슬래시를 `/` 로 바꾸지 않던 문제.

### Added
- **DTD 어댑터 + `dataset textures`**(v0.7): `dataset info dtd`(연구 목적 라이선스·구조·결함처럼 보이는 카테고리 15) · `anograft dataset textures dtd <dtd> --out textures.txt [--categories a,b|*] [--limit n --seed s]` 가 `perlin-texture` 용 목록 파일을 만든다(은행에 넣지 않는다 — `import-dataset dtd` 는 거부). `texture_dir` 이 폴더뿐 아니라 **`.txt` 목록**(목록 파일 기준 경로)도 받는다.
- **검수 탭**(v0.7, GUI): `anograft run` 출력 폴더를 열어 합성 결과를 썸네일(GT 윤곽)로 보고 **채택(A)/반려(R)/보류(U)** — `review.csv` 자동 저장, 다중 선택, 메모, "판정 후 다음으로". 필터(미검수·채택·반려·폴백·skipped·클래스). 상세(이미지+GT, 클래스·소스·면적·blend·폴백·경고). **분포 히스토그램** — 합성 인스턴스(반려 제외) vs 은행 실제 소스의 면적/긴 변을 같은 로그 구간에(합성 `#00A188` · 실제 `#C8841C`). **정리본 내보내기** = 반려를 뺀 사본(정상 유지, skipped 행 제거, manifest 재작성, `annotations.json` 필터, mvtec 사본 포함). 배치가 끝나면 검수 탭 경로가 그 출력을 가리킨다.
- CLI **`anograft dataset prune <root> --out <dir> [--review review.csv] [--drop-unreviewed]`** — 검수 탭과 같은 정리본(`io/prune.py`).
- **`coco` writer**(v0.7): `output.writer: {format: coco}` → 정본 위에 `annotations.json`(COCO instances — images 는 합성+정상, categories 는 은행 classes 순서로 id 1-based, annotations 는 인스턴스 GT 마스크의 외곽 폴리곤 `segmentation`·`bbox`·`area`(픽셀 수)·`iscrowd 0`). 순수 JSON(pycocotools 불필요), 같은 결과 → 같은 바이트(`info` 에 시각 없음). 사이드카 `writer.image_id/annotation_ids`, manifest `label = annotations.json#<image_id>`. 배치 탭 출력 형식에 `coco`.
- **은행 탭**(v0.7, GUI): 은행을 열어 소스를 타일 그리드로 보고(마스크 윤곽, 추정 amber, 저신뢰 빨간 테두리) 클래스·태그·저신뢰만·추정만·검색으로 거르고 id/신뢰도/면적/클래스로 정렬. 상세(크롭+마스크 오버레이 확대, 메타·flags). **삭제**(세 파일, `bank.yaml` classes 유지 — class id 불변, imports 이력) · **라벨 탭에서 다듬기**(크롭+현재 마스크를 라벨 탭 은행 소스 편집 모드로 → 저장하면 같은 id 에 덮어쓰기, `mask_origin: manual:<tool>`, 점수 제거). 은행이 바뀌면 같은 은행을 쓰는 스튜디오가 다시 준비되고, 라벨 탭 저장은 은행 탭을 새로고침한다. 레시피를 열면 그 은행을 자동으로 연다. `BankWriter.delete/replace_mask`.

## [0.6.0] - 2026-09-15

v0.6 — **실데이터 보정**(`KNOWN-ISSUES.md`, 경면 금속 토크스 소켓 1차 적용에서 나온 10건 전부). 프리셋 9종 · 골든 18장. 2차 실데이터 적용은 아직.

### Fixed
- **`mask_dir` ROI 가 미리보기 축소에서 깨지던 문제**(KNOWN-ISSUES #1): `roi_from_mask` 가 대상과 크기가 다른 마스크를 `INTER_NEAREST` 로 맞춘다(같은 비율만 — 비율이 다르면 "다른 이미지의 마스크" 로 실패). 사이드카 `roi.resized_from`. 스튜디오 1024·768·512 축소에서 원본 크기 ROI 마스크가 그대로 동작.
- **skipped 사유가 결과("ROI 없음")가 아니라 원인("roi: mask_dir 로드 실패 …")을 가리킨다** — `Pipeline` 이 `roi:` 경고를 우선(`skip_reason`). CLI manifest·GUI 공통.
- `recipe init --write recipes/new.yaml` 이 없는 상위 폴더를 만든다(KNOWN-ISSUES #10).

### Added
- **ROI `annulus`**(KNOWN-ISSUES #2 #4) + 프리셋 **`annulus-graft`**: 원형 부품의 **가공 링 면만** 허용 — `otsu`/`grabcut` 은 물체 vs 배경만 갈라 결함이 안 생기는 중앙 리세스에도 배치됐다. 중심·반경은 대상마다 Otsu 전경의 최소외접원으로 자동 검출(`center`/`radius: null`)하거나 고정, `r_inner`/`r_outer` 는 검출 반경의 비율(`units: ratio`, 토크스 소켓 실측 ≈ 0.56~0.9) 또는 px. `erode_px` 는 안·바깥 경계 모두에서 깎는다. 검출 실패는 이미지 중심으로 대체 + 경고. 합성 토크스 축소판 8장(중심 ±60 px 이동)에서 링 적중 **16/16**(poisson-graft otsu 12/16), 중심 오차 0 px. `core/roi.py::detect_disk/annulus_mask/roi_annulus`, 골든 +2(16장).
- **배치 실패 진단**: `max_tries` 소진 시 사유에 `ROI 최대 폭 N px vs 패치 W×H px(축소 후 …)` 를 붙이고, 짧은 변이 폭을 넘으면 조치 힌트(geometry.scale · shrink_on_fail · ROI). 사이드카 `placement.roi_max_width_px`·`patch_bbox_px`·`patch_bbox_last_px`.
- **스튜디오 카드 파라미터 편집기**(`stage-params`): 7단계 카드마다 레시피 스키마(pydantic)에서 위젯을 자동 생성 — 정수/실수 스핀, `[lo, hi]` 범위, on/off, 선택지, 경로(폴더 버튼), 목록, `X | None` 은 체크박스로 켬/끔. 값은 `StudioSession.set_stage_field` 재검증을 거쳐 미리보기 재계산·레시피 저장에 반영, 잘못된 값은 대화상자 없이 **카드 빨간 줄** + 되돌림. 배치 카드에 **ROI 하위 스테이지**(method 콤보 + 폼 — mask_dir 를 고르면 `path` 자리표시). 새 method 를 스키마에 추가하면 GUI 수정 없이 편집기가 생긴다(`gui/studio/params.py` · `param_form.py`, `registry.config_class`).
- **`source.tags` 필터**(KNOWN-ISSUES #7): `pipeline.source.tags: {include: [..], exclude: [..]}` — 임포트 `--tags` 로 붙인 제품명 등으로 은행 소스를 고른다(include 중 하나라도 · exclude 중 하나라도 제외). 클래스·확률은 그대로고 클래스 안의 풀만 줄어듦, 순서 유지(재현성). 필터 후 0개 클래스는 경고 + 건너뜀, 전부 0 이면 prepare 실패. `run --dry-run` 에 `source.tags` 행 + 필터 후 소스 수, 사이드카 `source.tags`. 스튜디오 소스 카드에 `tags.include`/`tags.exclude` 목록 필드가 자동으로 생김.
- **µm/px 축척 정합 표면화**(KNOWN-ISSUES #6): `bank ls` 에 `no_um` 열(피치 없는 소스 수)과 태그별 수 · `runner.prepare` 가 소스/대상 피치가 빠져 정합이 (일부) 꺼진 상태를 한 줄로 경고 — CLI stderr·스튜디오 상태바·배치 로그에 그대로.
- **라벨 탭 YOLO 초안**(KNOWN-ISSUES #8): 이미지를 열 때 옆의 `labels/<stem>.txt`(`images`↔`labels` 미러 · 같은 폴더 · `<folder>/labels/`)와 `data.yaml`/`classes.txt` names 를 찾아 박스는 GrabCut/Otsu/타원/사각 추정, 폴리곤은 채움으로 마스크를 **미리 채운다**(클래스 콤보 동기, 클래스별 채우기, "열 때 자동 채우기"). 목록에 라벨 있는 이미지 `▸`. 손대지 않은 초안은 임포터와 같은 `yolo-box:<method>`/`yolo-polygon`(= `bank ls` est), 손대면 `manual:mixed`.
- **라벨 탭 ROI 모드**(KNOWN-ISSUES #4 임의 형상): "저장 대상 → ROI 마스크" 로 바꾸면 정상 이미지에 허용 영역을 칠해 `<mask_dir>/<stem>.png`(`placement.roi: mask_dir` 형식, 원본 크기·크롭 없음)로 저장하고, 이미지를 열 때 같은 이름의 ROI 를 불러온다. `LabelSession.save_roi_png`.
- **GUI 시작 안내**(KNOWN-ISSUES #9): 레시피 인자 없이 켜면 최근 레시피(열기·저장 때 기억, `QSettings`)를 복원하고, 없으면 스튜디오 캔버스에 ko/en 4단계 안내(라벨 → 입력 → 프리셋 → 배치)와 상태바 문구.
- **박스→마스크 타당성 점수**(KNOWN-ISSUES #3): `mask_from_box.mask_confidence` — 마스크 안 평균 vs 박스 바깥 링 평균의 분리도(링 σ 단위) · 박스 테두리 접촉 비율 · 성분 수 · 포화 비율 → `confidence` 0..1 + `flags`(low-contrast · box-edge · fragmented · saturated · area-out). 면적 비율만 보던 폴백 사슬을 보완하되 채택은 바꾸지 않는다(재현성). `import-yolo`·라벨 탭 초안 저장이 은행 메타 `confidence`/`flags` 에 기록, `bank ls` `lowconf` 열 + 참고 줄, `bank preview` 점수 표기 + 0.5 미만 **빨간 테두리**, `import-yolo` 요약에 저신뢰 수, `run`/스튜디오가 저신뢰 추정 마스크를 경고(추정의 절반을 넘으면 "은행을 먼저 손보세요"로 강화). 샘플 은행: ellipse 폴백 1건이 0.15 로 잡힘.
- **프리셋 `dent-graft`**(KNOWN-ISSUES #5, 프리셋 9종): 조명 의존 결함(찍힘·덴트·눌림)용 — 회전 **±15°**·flip 끔·축척 0.9~1.1(3D 변형은 보이는 모양이 곧 조명 효과라 ±180° 회전은 음영/하이라이트를 뒤집는다) · `structure-aware`(위치 균등 `prefer: uniform`, 긴 축을 결·접선에 정렬, jitter 5°) · poisson NORMAL · 조화 0.2(음영 깊이 = 결함 신호). 기존 프리셋의 ±180 은 텍스처성 결함용이라 그대로(골든·게시 레시피 재현성). 골든 +2(18장).
- 스튜디오 파이프라인 카드에 **fail-soft 경고 표시**(`⚠ <stage>: …`, ROI 경고는 배치 카드에) · 변형 카드의 잘린 사유는 툴팁에 원문 · 상태바에 "경고 n건 더".

## [0.5.0] - 2026-09-15

v0.5 — **여기서부터 GUI 만으로 단독 사용**: 결함 사진 라벨링(라벨 탭) → 미리보기(스튜디오) → 데이터셋 생성(배치 탭). 여전히 샘플 데이터로만 검증.

### Added
- **라벨 탭**(v0.5, GUI): 결함 사진을 열어 브러시·지우개·폴리곤·자동 선택(박스 → GrabCut/Otsu/타원/사각)으로 마스크를 만들고 팽창·침식·되돌리기(40단계) 뒤 **은행에 저장** — 임포터와 같은 경로(`ImportRecord` → `BankWriter`, 성분마다 소스, `mask_origin: manual:<tool>`), 클래스는 은행 클래스에 이름 기준 병합. 클래스·픽셀 피치·태그·마스크 통계(면적·성분·길이·대비). 스튜디오가 같은 은행을 쓰면 저장 즉시 다시 준비. 단축키 B/E/P/A/H · [ ] · Ctrl+Z/Y · Enter/Esc · Ctrl+S · PageUp/Down. **라벨링 도구 없이 정상 사진 + 결함 사진만으로 은행을 만들 수 있다.**
- **배치 탭**(v0.5, GUI): 스튜디오 "배치로 보내기"가 현재 레시피(저장 안 해도)를 배치 탭으로 넘기고, 출력 폴더·장수·시드·워커·출력 형식(yolo/pairs/mvtec)을 고쳐 **생성 시작** — 진행률 바·로그(첫 줄에 같은 CLI 명령)·중지(그때까지의 파일·manifest 유지)·출력 폴더 열기. 결과는 `anograft run` 과 바이트 동일. `runner.run(should_stop=)` · `RunSummary.cancelled/done`.
- `imgio.read_mask(threshold=)` · `PairRecord.mask_threshold`(VisA).

## [0.4.0] - 2026-09-15

v0.4 — CPU 알고리즘 확장. **여전히 샘플 데이터로만 검증**(실데이터·MVTec/VisA 실제 사본·anomalib/ultralytics 학습 미검증). 프리셋 7종, `anograft methods` 24 선택지 전부 ok.

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

[Unreleased]: https://github.com/slnu21/Graft/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/slnu21/Graft/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/slnu21/Graft/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/slnu21/Graft/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/slnu21/Graft/compare/v0.1.0...v0.4.0
[0.1.0]: https://github.com/slnu21/Graft/releases/tag/v0.1.0

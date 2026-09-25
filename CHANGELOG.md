# Changelog

[Keep a Changelog](https://keepachangelog.com/) + [SemVer](https://semver.org/). 상세는 `docs/개발로그/`(로컬).

## [Unreleased]

### Added
- **라운드 원장 + 트리거 정책(학습 루프 T14)** — "라운드 3의 모델은 무슨 보관함으로 학습됐나 · 지난주에 왜 안 돌았나"에 답한다. `<out>/rounds.jsonl` 은 **더하기만 하는** 기록(라운드 시작 · 단계 완료 · 끝: 지표·승급·champion·`pipeline_hash`·보관함 스냅샷 · **실패 사유** · 안 돈 사유)이고, `loop.state.json` 에는 **움직이는 포인터 둘**(지금 라운드·champion)만 남는다 — 롤백은 포인터를 되돌리는 일이고 감사는 지워지지 않는 줄이어야 하기 때문입니다. 라운드마다 **보관함 스냅샷**(복사가 아니라 id+해시 목록)을 `round-NNN/bank.snapshot.json` 에 남깁니다. **언제 돌지는 `loop.yaml` 의 `trigger`** 가 정합니다(새 조각 N개 · 새 이미지 N장 · 최소 간격 · 최대 간격(양과 무관하게 도는 드리프트 감시)). 기본값은 **제한 없음**이라 지금까지와 똑같이 돌고, 켜면 안 돈 사유가 `tick.json` 에 남고 **사유가 바뀔 때만** 원장에도 남습니다(5분마다 부르는 스케줄러가 원장을 덮지 않게). `loop tick --force` 는 기준을 무시하고 돕니다(잠금은 그대로 지킵니다). `loop status` 가 "지금 돌 때인가 · 마지막 tick · 최근 실패"를 함께 보여 줍니다.
- **유입 커서 + `anograft loop tick`(학습 루프 T13)** — **스케줄러가 부르는 진입점**입니다. Windows 작업 스케줄러·cron 이 인자 없이 `loop tick` 을 부르면 한 라운드를 멱등하게 진행하고 끝납니다(Graft 는 상주하지 않습니다 — 고객사엔 이미 스케줄러가 있습니다). **같은 이미지를 두 번 스코어링하지 않습니다**: 처리 이력을 `<out>/processed.jsonl` 에 더해 가고(경로·크기·mtime·내용 해시 앞 8) 신원은 두 겹이라 이름을 바꿔 복사한 사진도 같은 이미지로 봅니다 — "마지막 처리 시각" 한 값으로 두면 늦게 도착한 파일·되돌린 파일·시계 역행에 조용히 구멍이 납니다. 다시 보게 하려면 `--since <라운드>`(커서를 지우지 않고 읽을 때 걸러내므로 이력은 남습니다). 새로 들어온 것이 없으면 스코어링을 건너뛰고 **합성만 도는 라운드**가 됩니다. 이름(stem)이 겹치는 사진은 예측이 덮어쓰므로 **버리지 않고 다음 라운드로 미룹니다**(경고와 함께).
- **동시 실행 잠금**(T13) — 앞 tick 의 학습이 세 시간째 도는 중에 다음 tick 이 들어와도 라운드 폴더가 섞이지 않습니다(`<out>/loop.lock`). 살아 있는지는 pid 가 아니라 **갱신 나이**로 보고(도는 쪽이 단계마다 갱신) 하루가 지난 잠금만 사유를 알리고 가져갑니다. `loop tick` 은 다른 실행이 돌고 있으면 **오류가 아니라 "다음에"**(종료 코드 0)로 답하고, 사람이 부른 `loop run` 에게는 오류입니다. `loop status` 가 누가 돌고 있는지와 커서 건수를 함께 보여 줍니다.
- **라운드 한 바퀴(`anograft loop run` · `loop status`, 학습 루프 T10)** — 현장 이미지 스코어링 → 검토 대기 → **[사람 판정]** → 보관함 편입 → 합성 → 학습·평가 → 승급 판정까지 한 명령으로 엮는다. **멱등하다** — 사람이 판정할 차례면 무엇을 해야 하는지 알려 주고 멈추고, 다시 부르면 **그 다음 단계부터** 이어 간다(학습이 실패해도 합성을 다시 하지 않는다). 첫 라운드처럼 모델이 없으면 수집을 건너뛰고 합성부터 도는다. 설정은 레시피 밖 **`loop.yaml`**(예시 `loop.example.yaml`) — 레시피는 공유되는 물건이라 현장 경로·평가셋 같은 개인 환경은 넣지 않는다. 평가는 동결 평가셋을 학습 데이터셋의 val 로 넣고 학습기가 돌려주는 지표를 쓴다. 정상만 학습하는 학습기(비지도)는 데이터를 만들기 전에 거부한다(합성 결함을 먹이면 오염).
- **검토 대기 큐(`anograft loop queue` · `loop accept`, 학습 루프 T5)** — 학습기 예측(`trainer predict` 출력)에서 **사람이 볼 것만 추려** 한 폴더로 내고, 거기서 채택한 것만 결함 보관함으로 돌려보낸다. 넣는 순서는 **불일치 → 경계 → 확신 → 무작위**(예측 폴더를 둘 주면 두 모델이 어긋난 것이 먼저 온다 — 오류가 독립이라 정보량이 가장 많다). **새 포맷을 만들지 않았다** — 큐 폴더는 `manifest.csv` + 빈 `review.csv` 구조라 **검수 화면이 그대로 열고**(화면 변경 0) 판정도 원래 쓰던 채택/반려 그대로다. 되돌릴 때는 모델 마스크를 **추정으로 표시**하고(`mask_origin: pred:<학습기>`) 평가셋(`holdout.txt`)은 보관함이 거부한다.

### Changed
- 라운드 이력이 `loop.state.json` 에서 **`rounds.jsonl`(append-only)** 로 옮겨졌습니다 — `loop status` 의 이력·JSON 의 `history` 는 그대로 나옵니다(출처만 바뀜).
- 검토 대기를 고를 때 **몫 하나가 비면 경계(임계값 근처) 순서로 채운다** — 임계값을 넘긴 검출이 하나도 없는 초기 라운드에서 큐가 요청한 장수보다 작게 나오던 것을 고쳤습니다.
- 추정 마스크 판정을 한 함수로(`bank.is_estimated` — `yolo-box:` · `pred:`) — `bank ls` 의 `est`, 보관함 타일, `bank preview` 테두리가 같은 답을 낸다.

## [0.10.1] - 2026-09-25

**화면 사이 손잡이** — 웹 다섯 화면이 서로 인계된다. v0.10.0 을 직접 써 보고 나온 지적("매 화면마다 뭘 열어야 하는가")의 답이고, 정체는 새 기능이 아니라 기존 창에 있던 인계가 웹으로 옮겨오지 않은 것이었다. 합성 알고리즘·레시피 스키마·출력·골든 불변, 런타임 의존성 4개 불변.

### Added
- **미리보기 → 일괄 생성 `일괄 생성으로 보내기`** — 레시피를 저장하고 일괄 생성 화면이 그대로 이어받는다(설정·CLI 한 줄까지). **저장까지 한 번에** 하는 이유는 화면이 보여 주는 `anograft run <레시피>` 가 실제로 돌아가야 하기 때문이다. 저장만 하고 싶으면 `레시피만 저장` 이 그대로 있다. 일괄 생성이 돌고 있는 중에는 거절한다(설정이 발밑에서 바뀌면 진행 중인 실행을 설명하는 화면이 거짓말을 한다).
- **결함 표시 → 결함 보관함 반영** — 조각을 저장하면 보관함 화면이 **열어 보기만 해도 새 조각을 보고 있다**. 다른 보관함을 보고 있었으면 건드리지 않고, 화면의 `결함 보관함에서 보기 →` 가 사람의 뜻으로 옮겨 준다.
- **결함 보관함 → 결함 표시 `다듬기`** — 조각 타일·상세에서 누르면 그 크롭과 지금 마스크를 결함 표시가 열고, 저장하면 **같은 조각의 마스크를 덮어쓴다**(새 조각이 생기지 않는다). 기존 창의 "라벨 탭에서 다듬기" 와 같은 경로를 지나므로 `mask_origin` 은 `manual:<도구>` 가 되고 박스 추정 신뢰도는 지워진다.
- **열기 칸 최근 경로** — 다섯 화면의 경로 칸에 최근에 연 것이 뜬다. 저장 위치는 `~/.anograft/recent.json` 하나(`ANOGRAFT_HOME` 으로 옮길 수 있다). 종류는 화면이 아니라 **칸의 뜻**으로 나뉘어서(보관함 · 결함 사진 · 레시피 · 출력 폴더) 미리보기에서 저장한 레시피가 일괄 생성 목록에 그대로 뜬다. 목록을 못 읽어도 칸은 평소처럼 쓴다.

### Changed
- 웹 결함 표시의 `전부 지우기` 가 기존 창과 **같은 메서드**를 쓴다 — 예전에는 웹에만 있는 도구 이름이 기록에 남아 `mask_origin` 이 창과 갈릴 수 있었다.
- 흑백 판정(`core.channels.is_gray`)과 다듬기 도구 이름(`LabelSession.edit_tool`)을 공용 함수로 올렸다 — 두 UI 가 같은 값을 쓰게.

## [0.10.0] - 2026-09-25

**웹 UI 릴리스** — `anograft serve` 한 줄로 브라우저에서 다섯 화면(결함 보관함 · 결함 표시 · 미리보기 · 일괄 생성 · 검수)이 전부 돈다. 기존 창(PySide6)은 그대로 유지하고 **둘을 함께 낸다**. 합성 알고리즘·레시피 스키마·출력·골든 불변, **런타임 의존성 4개 불변**(서버는 stdlib `http.server`, node 는 빌드 때만). 루프 선행 작업(학습기 계약·평가셋 누수 방지·보관함 스냅샷)도 함께.

### Added
- **웹 UI(`anograft serve`, U2~U6)** — 127.0.0.1:8000 고정. **결함 보관함**(타일·필터·삭제 + 여는 순간 평가셋 누수 보고) · **결함 표시**(붓·지우개·다각형·자동 선택 — 마스크 연산은 전부 서버가 해 Qt 와 결과 동일, 획이 끝날 때 원본 픽셀 좌표를 한 번 전송) · **미리보기**(원본|합성 A/B 와이프 · 정답/허용 영역 오버레이 · 바탕 목록 · 프리셋 갤러리 · **7단계 카드 편집**(라벨·툴팁·제약은 `core/help.py`·스키마에서 서버가 실어 준다) · 시드 변형 그리드 · `per_class` 표 · 레시피 저장) · **일괄 생성**(설정·시작/중지·진행률·로그·요약 → 검수로 넘기기) · **검수**(그리드 + 키보드 판정 · 분포 · 대비/조명 힌트 · 정리된 데이터셋·리포트). 프론트는 **API 를 호출만** 하고(합성·판정 로직 0) 비즈니스 로직은 계속 `core`/`runner` 에 있다. 쓰기는 POST + 전용 헤더(CSRF) + `Host` 검사. 번들은 저장소에 없고 CI 가 만들어 wheel·zip 에 싣는다 — 없으면 `serve` 가 죽지 않고 안내 페이지를 보여 준다(fail-soft).
- **평가셋(holdout) 누수 방지 + 보관함 스냅샷(T4)** — `<bank>/holdout.txt` 의 stem 은 `BankWriter.add` 한 지점에서 거부되고(임포터·결함 표시 저장 모두) 몇 건을 왜 뺐는지 요약에 찍힌다. 나중에 만든 목록은 `bank ls`·`bank verify` 가 `violations` 로 알린다(자동 삭제 없음). `bank snapshot`/`verify --snapshot` 은 **복사가 아니라 id+내용 해시 목록**이라 폴더를 옮겨도 같고(`pipeline_hash` 가 깨지지 않는다) 마스크를 다듬으면 `changed` 로 잡힌다.
- **학습기 계약(`anograft trainer list|info|fit|predict`, 루프 T1·T3)** — 학습기는 ABC 가 아니라 **프로세스 경계 + JSON**(torch 가 코어 venv 로 들어오지 않게). 등록은 레시피 밖 `trainers.yaml`. `adapters/noop.py`(stdlib 만) · `adapters/yolo.py`(ultralytics 검출).
- **anomalib 어댑터(비지도) + (A) 모델 순위 상관**(학습 루프 T2) — `adapters/anomalib_trainer.py`(PatchCore·PaDiM·FastFlow·Cfa·Dfkde·Dfm, 별도 venv): **`trains_on: normal_only`** 를 선언해 **합성 결함이 학습셋에 들어가지 않게** 한다(평가에만). 데이터는 `mvtec` writer 출력의 카테고리 폴더를 그대로 받고, `predict` 는 이상맵 임계 마스크 + 이미지 점수. `tools/bench_model_rank.py` 는 설계 §5 (A) 실험 — 학습 정상과 합성 대상 정상을 **코드가** 가르고(겹치면 낙관 편향), 결함 **k 장/클래스**만 은행에 넣어(그 k 장은 실제 평가에서 제외) 실제·합성 두 순위의 **Spearman ρ** 를 낸다 → `BENCHMARKS.md` §3.
- **YOLO 학습기 어댑터 + `anograft trainer fit|predict`**(학습 루프 T3) — `adapters/yolo.py`(ultralytics 검출: `fit` = train+val, `predict` = `scores/<stem>.json`, seg 가중치면 `masks/` 까지)를 `trainers.yaml` 에 등록하면 CLI 한 줄로 학습·예측이 나간다. 어댑터는 **별도 venv**(코어는 순수 wheel 그대로)이고 `anograft` 를 import 하지 않는다. 지표는 **평평한 float 맵**(`mAP50` · `mAP50-95` · 클래스별 `mAP50/<class>` · `minutes`), `--spec` 은 불투명하게 등록부 spec 위에 얕게 병합, 어댑터 로그(stderr)는 **줄 단위 실시간**으로 흐른다(학습은 길다).
- **`tools/train_mvtec_map.py` 가 계약 위로**(T3) — 이제 ultralytics 를 import 하지 않고 `anograft trainer fit --json` 으로 학습을 시킨다(`--trainer` 기본 `yolo` · `--trainers-file`). 표·`results/` 캐시 형식은 그대로라 옛 결과가 계속 읽힌다. **벤치를 돌리는 것이 곧 계약 검증**이고 루프(T10)가 같은 경로를 쓴다.

## [0.9.0] - 2026-09-19

**사용성 릴리스** — 기능은 v0.8.2 그대로, 화면·문구·설명만 바꿨다(알고리즘·스키마 값·출력·골든 불변). 한국어 단독 라벨(영어·YAML 키는 툴팁) · 7단계 동사형 이름 · 카드 폼 한글 라벨+설명 툴팁 · `anograft explain`/`PARAMS.md` · 프리셋 대비 바뀜 ↺·고급 접기·슬라이더 · 프리셋 갤러리(현재 바탕 썸네일) · 탭 흐름 순서+배지+다음 → · 시작 체크리스트 · 경고/힌트 한 컴포넌트 · 대비 힌트 · 라이트 테마·글자 크기. 근거 `docs/ux/2026-09-19-usability-review.md`(로컬).

### Changed
- **GUI 용어·라벨(v0.9 사용성 ①)** — 화면 문구를 한국어 단독으로, 영어 이름·YAML 키는 툴팁 둘째 줄로. 탭 `결함 표시 · 결함 보관함 · 미리보기 · 일괄 생성 · 검수`, 7단계 `결함 고르기 · 크기·회전 · 위치 정하기 · 붙이기 · 색·밝기 맞추기 · 카메라 효과 · 정답 영역`(카드 제목 툴팁에 영어·`pipeline.<stage>`). 은행→결함 보관함 · 소스→결함 조각 · 대상→바탕 이미지 · 폴백→대체 처리 · skipped→건너뜀 · 정리본→정리된 데이터셋 · 실측 CSV→현장 측정값 · 저신뢰→마스크 신뢰도 낮음 · "배치" 두 뜻 분리(Batch = 일괄 생성 · Placement = 위치 정하기). 위젯 툴팁 +40(일괄 생성·보관함·퀵스타트 전부). README ko 사용자 문구 동일 갱신. CLI 서브커맨드·YAML 키·사이드카·핵심 경고 문장은 그대로(용어 사전 `docs/ux/2026-09-19-usability-review.md` §3).

### Added
- **테마·글자 크기(v0.9 사용성 ⑦)** — 상단 ⚙ 설정: **라이트/다크** 테마, **글자 크기 100/115/130 %**(4K·고배율). 저장하면 다음 실행 때 적용(`ui/theme`·`ui/font_scale`). 다크 보조 텍스트 색을 `#697683 → #87939F` 로 올려 배경·패널 어디서나 4.5:1 이상(두 테마의 본문·보조 텍스트 대비를 테스트가 검사).
- **경고·힌트 한 컴포넌트(v0.9 사용성 ⑥)** — `gui/notice.py` `Notice`(아이콘 · 문장 · 행동 버튼, level warn/hint/error): 카드 경고+고치기 버튼(썸네일 아래 전체 폭), 카드 재검증 오류, 검수 탭 대비·**조명 힌트**(회전이 빛 방향을 뒤집음 → dent-graft, 이제 제목이 아니라 라벨) 가 같은 모양. 검수 분포 패널 220 → 300 px. **핵심 경고 문장 3종을 '무엇이 · 왜 → 이렇게' 로**(축척 정합 · 마스크 신뢰도 · 빛 방향; 용어 사전 적용, CLI 도 같은 문장). 일괄 생성 로그에 같은 경고가 3번 찍히던 것 → 1번(prepare 경고는 실행 시작 때 한 번, 완료 후엔 새 경고만).
- **작업 흐름 표시(v0.9 사용성 ⑤)** — 탭을 흐름 순서로(`① 결함 표시 → ② 결함 보관함 → ③ 미리보기 → ④ 일괄 생성 → ⑤ 검수`) 번호를 붙이고 **상태 배지**(보관함 조각 수 · 준비됨 · 마지막 생성 장수 · 미검수 수)를 탭 이름에. 탭 바 오른쪽 **다음: … →** 버튼(툴팁 = 이 탭에서 끝낼 것). 레시피가 없을 때 미리보기 캔버스 자리에 **시작 체크리스트**(샘플 데이터 → 결함 표시 → 보관함 → 레시피 열기 → 일괄 생성 → 검수, 된 단계는 ✓, 첫 미완 단계 버튼 강조). 문구·판정은 `gui/workflow.py`(Qt 없음).
- **프리셋 갤러리(v0.9 사용성 ④)** — 미리보기 탭 프리셋 콤보 옆 **고르기…**: 프리셋 10개를 카드(제목 · 한 줄 · 이럴 때 · 피할 때 · 단계별 method · 근거)로, **썸네일은 지금 고른 바탕 이미지에 각 프리셋을 적용한 것**(`gallery.render_preset_thumbs`, 같은 시드 v1, 10장 ≈ 0.3 s). 더블클릭 또는 '이 프리셋으로' → 콤보·카드·미리보기가 따라온다. 돌 수 없는 프리셋(보관함 없이 bank 프리셋 등)은 회색 칸.
- **카드 폼 v2(v0.9 사용성 ③)** — 프리셋 값과 다른 행은 라벨 teal + **↺ 되돌리기**, 카드 헤더 **바뀜 n · ↺(전부 프리셋 값으로)** · **고급 옵션 (n)** 접기(프리셋을 고른 뒤 보통 안 만지는 값 — 펼침은 기억) · 세기·임계처럼 폭 ≤ 10 인 실수 값은 **슬라이더 + 스핀**(더블클릭 = 되돌리기) · 색·밝기 맞추기/카메라 효과 카드 헤더 **켬/끔** 토글(끄면 method none, 켜면 마지막 method). 기준값 = 레시피 프리셋의 같은 method 블록(없으면 스키마 기본, `params.baseline_config`, Qt 없음). method 콤보·헤더가 카드 최소 폭을 넘기지 않게 정리(패널 352 px).
- **파라미터 도움말 한 원천(v0.9 사용성 ②)** — `core/help.py`: 스테이지·method·필드(≈110) 라벨·설명(무엇 → 올리면/내리면)·단위·영어·고급 여부, 프리셋 YAML `meta:`(제목·한 줄·이럴 때·피할 때·근거). 소비자: 카드 폼 **한국어 라벨 + 단위**, 3줄 툴팁(라벨 · YAML 키 / 설명 / 영어 · 형식·범위) · method 콤보 한국어 표시명 + 요약 툴팁 · 프리셋 콤보 툴팁 · **`anograft explain`**(`geometry.scale` · `placement.roi.erode_px` · `blend:poisson` · `preset:dent-graft` · `inputs` · `output`, 프리셋별 값까지) · `methods` 요약 열 · **`PARAMS.md`**(`explain --markdown` 으로 생성, 테스트가 최신인지 검사) · `tests/test_param_help.py` 완전성(도움말 없는 필드·method·프리셋이 생기면 실패).
- **대비 힌트** — 검수 탭 대비 히스토그램 제목 · 리포트 문단 · `dataset report` stderr: 클래스별 합성 대비 중앙값이 실제(은행·실측 CSV)의 **50% 미만**(또는 극성 반대)이면 "이 클래스만 `relative-paste` 로 갈라(`recipe init --classes <cls>`) `dataset merge`" — MT blowhole −17 vs −48 이 학습 전에 보였던 신호(BENCHMARKS §2, 갈라서 +0.15)의 자동화. `appearance.contrast_hints`(순수, rng 0): 양쪽 n ≥ 5 · |실제| ≥ 8 · 중앙값 차 ≥ 12(MT crack −4.5 vs −13.7 은 블렌딩에 무차별이라 제외). MT 40장: poisson → blowhole 13% 힌트 · relative-paste → 힌트 없음. 기본값·출력 불변(진단만).

## [0.8.2] - 2026-09-19

**결함 성격별 프리셋** — `harmonize.relative`(노출 보정) + 프리셋 `relative-paste` · `recipe init --classes`(프리셋을 클래스 부분집합에만) → `run` × n → `dataset merge --dedupe-normals`. Magnetic Tile 에서 blowhole 만 갈라 mAP50 0.460 → 0.605(+0.15, 학습 시드 3/3). 기본값·기존 프리셋·골든·은행 결과 불변(새 프리셋·옵션만 추가).

### Added
- **`harmonize.relative`**(+ 프리셋 **`relative-paste`**) — 노출 보정: 소스 패치의 링(결함 주변) 평균을 대상 링 평균에 맞추는 **오프셋**을 마스크 안에 더한다(`gain: true` 면 σ 비로 배율도). stats·reinhard·histmatch 는 내부를 링에 맞춰 정의상 결함 대비를 (1−strength) 배로 줄이지만, 이건 결함의 *상대* 대비를 두고 소스·대상의 노출 차이만 없앤다. 대비가 곧 신호인 결함(블로우홀·검은 구멍)용 — Magnetic Tile blowhole 합성 대비 중앙값: poisson+stats 0.3 −17 · hard-paste 0(노출이 다른 타일에서 절반이 배경보다 밝은 구멍) · **relative-paste −37**(실제 −48). `paste` 뒤에 쓰는 것(poisson 뒤엔 이중 보정). L 채널만, 마스크 밖 불변, 패치/소스 링 없으면 skipped. 골든 +2(기존 불변).
- `anograft recipe init --classes A B` → `source.classes` — 프리셋을 은행의 일부 클래스에만. **결함 성격별 프리셋** 흐름: `init --classes` 로 레시피 둘 → `run` 둘 → `dataset merge --dedupe-normals`(같은 은행 = 같은 클래스 id, 정상 한 벌). 은행 없는 프리셋(self-cut·perlin)엔 거부.
- `tools/train_mvtec_map.py --class-presets blowhole=relative-paste break=poisson-graft crack=poisson-graft` — 클래스 그룹별로 따로 합성(`count` 클래스 수 비례)해 merge 한 B 셋 · `--presets <preset>+dent:<cls>`(한 레시피 안 `geometry.per_class`) · 표 이름 → 폴더 이름 안전화. `.gitignore /*.pt`.

### Changed
- `tools/train_mvtec_map.py` — `--train-seeds 7 8 9`(분할·합성 고정, 학습 시드만 반복 → 시드별 + 평균 Δ 표) · `--mask-from grabcut hybrid`(은행마다 B 셋) · **`pairs.csv` 입력**(Magnetic Tile: `normals.txt` 앞 `--n-good` 은 음성, 그다음 `--n-targets` 는 합성 대상) · `results/` 캐시와 합성 재사용(프리셋·은행을 나중에 보태도 끝난 학습은 안 돌림) · 한 `--out` = 한 분할. GT 마스크 이진화 `> 127`(MT 의 JPEG 링 잡음 — MVTec 0/255 는 불변).
- `BENCHMARKS.md` §2 — 분할 고정(split-seed 7) 학습 시드 3개 · dent-graft 분산 · Magnetic Tile 행 · **결함 성격별 프리셋 절**(MT: blowhole 만 relative-paste 로 갈라 merge → mAP50 0.460 → 0.605, +0.15 3/3 · metal_nut: per_class/split ≈ poisson, bent 는 찍힘 아님).

## [0.8.1] - 2026-09-17

v0.8.0 뒤 같은 밤 — 작은 기하·소스 손잡이(전부 기본 off) · `dataset merge --dedupe-normals` · 샘플 데이터 대화상자 · **`BENCHMARKS.md`**(박스→마스크 IoU · 합성 유/무 YOLO mAP 0.27 → 0.37) · `tools/train_mvtec_map.py`. 기존 레시피·골든·은행 결과 불변.

### Added
- `source.redraw_on_empty`(기본 2) — 기하 변환 뒤 마스크가 비면(< 4 px, 아주 작은 소스 × 축소) 그 자리에서 소스를 다시 뽑는다. 실패했을 때만 rng 를 더 쓰므로 성공 경로·골든·기존 출력 불변. 사이드카 `source.redraws` · 경고 `source: … 소스 재추첨 k/n`. 소스 카드 폼에 자동 노출.
- 레시피를 **다른 폴더에서 열어** 입력 경로가 레시피 파일 기준으로 폴백하면 `output.root`(상대)도 레시피 파일 기준으로 — 출력이 cwd 에 흩어지지 않게. repo 루트에서 쓰는 기본 사용법(cwd)은 그대로, `--out` 은 명시값 우선.
- `run --roi-cache N` — 대상당 ROI 캐시 항목 수(LRU, 기본 16 · 0 = 끔; `runner.prepare(roi_cache_size=)`). 결과와 무관, 4K 대상 수백 장에서 메모리를 아낄 때. 워커도 같은 값.
- `source.single_class_per_image`(기본 false) — 한 이미지의 결함은 첫 결함이 뽑은 클래스로(설계의 '이미지당 1회 추첨'). mvtec writer 의 `mixed` 경고가 이 옵션을 가리킨다. 켜면 2번째 결함부터 클래스 추첨 rng 0회(첫 결함·기본값은 불변).
- GUI 상단 **'샘플 데이터' 옵션 대화상자**(`gui/quickstart_dialog.py`) — 모양·폴더·정상/결함 장수·크기·합성 장수·시드를 한 번에(종전엔 모양+폴더만). `MainWindow.make_sample(root, shape, **opts)`.
- `tools/train_mvtec_map.py` — MVTec GT 마스크로 real-train(클래스당 k)/real-val 을 나누고, **real-train 의 박스만으로** 은행을 만들어 합성한 뒤 YOLO(ultralytics, 별도 venv) 합성 유/무 mAP 를 같은 홀드아웃에서 비교. 결과는 `BENCHMARKS.md`.
- `geometry.tps: {points: 3, jitter: 0.0}` — **thin-plate spline 휘어짐**(설계 v0.2 열의 마지막 미구현 method). `points×points` 제어점을 `jitter × 짧은 변` 만큼 흔들어 전역으로 휘고 늘린다(elastic 은 국소 잔물결). headless OpenCV 에 TPS 가 없어 numpy 로(`core/tps.py`). 기본 0 = off = rng 0회 → 골든 불변. 사이드카 `geometry.tps.max_shift_px`. 기하 카드 폼에 자동 노출.
- `dataset merge --dedupe-normals` — 같은 대상(manifest `target`)의 정상 이미지는 첫 루트 것만(같은 정상 폴더로 돌린 출력 여러 개를 합칠 때 정상이 n배로 불지 않게). `merge.json` `normals_dropped`.
- `bank preview` 타일 id 라벨이 타일 밖으로 넘치던 것 — 클래스 접두를 떼고 앞을 잘라 번호(꼬리)가 보이게.

## [0.8.0] - 2026-09-17

공개 데이터(MVTec·Magnetic Tile)로 KNOWN-ISSUES 2차 절차를 리허설하며 나온 것들 — 진단 v2(짧은 변·정렬·축척) · `dataset merge` · 검수 클래스별 분포·실측 CSV · `--mask-from hybrid`(옵션) · 박스→마스크 벤치와 결정 근거 표 · 공개 데이터 받기 도구. 기본값·프리셋·골든 불변. 확인 절차는 `TESTING.md`.

### Added
- 검수 리포트에 **기하 한 줄**(scale·rotate·flip·per_class) — 조명 히스토그램을 어떤 회전/flip 으로 만든 결과인지 리포트만 봐도 알 수 있게.
- `tools/fetch_public_datasets.py` — MVTec AD 카테고리(HF 미러, 원본 폴더 구조) · DTD 결함류 15 · Magnetic Tile(pairs.csv·normals.txt) · VisA 를 표준 라이브러리만으로 받고 `--import` 로 은행까지. 공개 데이터 리허설 레시피 `recipes/public-*.yaml` 3종.
- `run --dry-run`·배치 로그·스튜디오 소스 카드에 **`source:` 비국소 클래스 경고** — 패치 긴 변이 대상 짧은 변의 50 % 이상인 클래스(MVTec `flip`, MT `uneven`)는 결함이 아니라 부품 전체 이상일 수 있으니 `source.classes` 로 제외하라고. ROI 를 넓혀도 답이 아닌 경우를 `placement:` 경고와 구분.
- prepare `targets:` 경고 — 대상 폴더에 같은 이름·다른 확장자 쌍(이미지 옆 마스크 PNG)이 있으면 마스크도 대상으로 뽑힌다고(.txt 목록 권고).
- **`anograft dataset merge a b … --out c`** — `run` 출력 폴더 여러 개(다른 프리셋·시드, 정리본 권장)를 한 학습셋으로: 파일 이름에 `d<k>_` 접두어(`--prefix`), 합성·skipped 행 index 0부터 재부여(사이드카 `index` 갱신 + `merged_from: {root, index}`), `review.csv` 이어 붙임, yolo `labels/`·mvtec 레이아웃 사본도 접두어, coco `annotations.json` 은 id 오프셋으로 병합, `merge.json` 요약. 같은 writer 형식·같은 클래스 이름(`data.yaml names`)이어야 한다(아니면 거부 — YOLO class id 가 어긋나므로).
- 검수 탭 분포에 **클래스 콤보** — 외형 지표(대비·질감·선명도·조명 방향)를 한 클래스만 합성 vs 실제로. 조명 방향은 클래스마다 달라 전체 분포는 섞여 보이므로 클래스별로 보고, 제목에 그 클래스의 R(n). 리포트에도 실제 방향이 유의한 클래스별 조명 방향 히스토그램 격자(`ReportData.hist_lighting_class`). `ReviewSession.class_options/distribution_by_class/lighting_r_for/lighting_histograms_by_class`.
- `tools/bench_mask_from_box.py` — GT 마스크가 있는 데이터(MVTec 카테고리·pairs.csv)에서 박스→마스크 추정 4방법의 IoU 와 `mask_confidence` 의 실패 검출력을 재는 벤치(markdown + JSON). 공개 데이터 1180 인스턴스 결과와 읽는 법은 `KNOWN-ISSUES.md` "결정 근거" 절.
- 검수 '실제' 분포를 **실측 CSV** 로 — 검수 탭 `실측 CSV…` 버튼 · `dataset report --real-csv x.csv`. 열은 `class`(선택) + `area/length/contrast/texture/sharpness/lighting` 중 있는 것(숫자, 빈 칸 건너뜀); CSV 에 있는 열은 CSV 가, 없는 열은 은행이 실제 값을 댄다. 레시피 `source.classes` 필터는 CSV 에도 적용. 리포트 머리 `실제 = 실측 x.csv`. 현장 실측(현미경 µm→px)과 합성 분포를 견줄 때.
- `bank import-yolo --mask-from hybrid`(라벨 탭 자동 선택에도) — grabcut 사슬 결과의 `mask_confidence` 가 0.5 미만이면 내접 타원으로(`mask_origin: yolo-box:ellipse`). 벤치(1180 인스턴스): 사슬 0.37/0.43 → 하이브리드 0.50/0.21. **기본은 그대로 `grabcut`**(은행 재현성) — 전환은 실데이터 2차 뒤. `tools/bench_mask_from_box.py` 에 `hybrid` 열.

### Fixed
- 합성 사이드카 `target.file` 이 Windows 에서 역슬래시였던 것 → manifest·정상 사이드카와 같은 posix.

### Changed
- **배치 가능성 진단 v2**(`run --dry-run` fit 행 · 배치 로그 · 스튜디오 배치 카드): 폭에 걸리는 건 **짧은 변**(`minAreaRect`)이다 — 가늘고 긴 스크래치(긴 변 214 px)가 링 폭 100 px 에 13/14 들어가는데 종전엔 "불가"라고 했다. 불가 = 짧은 변이 shrink 뒤에도 폭 초과 · 빠듯 = 짧은 변이 폭의 80 % 초과, 또는 긴 변이 폭을 넘는데 정렬(structure-aware `align`)이 없어 회전에 달림 · 그 외 가능. 행에 짧은/긴 변·근거를 함께. **µm/px 축척**(소스·대상 피치가 모두 있을 때 `physical_scale`)도 패치 크기에 곱한다(`× 축척 f`).

## [0.7.7] - 2026-09-16

v0.7.6 의 퀵스타트 멱등 수정 + COCO RLE · `bank merge --dedupe` · dry-run 기하 행.

### Fixed
- 퀵스타트(GUI '샘플 데이터' · `sample --quickstart`)를 같은 폴더에 두 번 돌리면 은행이 두 배로 쌓여(`-dup`) 조명 유의성까지 왜곡되던 것 — 퀵스타트 은행은 늘 새로 만든다(멱등).

### Added
- `bank merge --dedupe` — 같은 클래스에서 이미지·마스크 내용(sha256)이 같은 소스는 한 번만(같은 원본을 두 은행에 임포트한 경우). 내용이 다르면 id 가 같아도 종전처럼 `-dup`.
- COCO writer `segmentation: rle` — 비압축 RLE(열 우선, pycocotools 규약, iscrowd 0). 폴리곤은 외곽 윤곽만이라 조각·구멍이 있는 마스크는 RLE 가 무손실. `mask_to_rle/rle_to_mask`.
- `run --dry-run` 에 `geometry` 요약 행(scale·rotate·flip)과 `per_class <cls>` 행 · `bank ls --json` 에 `light_dir`.

## [0.7.6] - 2026-09-16

v0.7.5 후속 — flip 모드(none/horizontal/vertical/both, 호환) · per_class 편집 표 · 변형 카드 ↯ · 안전한 flip 자동 선택 · train_smoke 다중 출력. 기존 레시피·골든 불변.

### Added
- 조명 의존 클래스의 자동 오버라이드(`--auto-dent` · 퀵스타트 · 기하 카드 ▶)가 **방향에 안전한 flip 을 고른다**(`recipe.dent_override_for(light_dir)`: 위/아래 조명 → `horizontal`, 옆 → `vertical`, 모르면 `none`) — 찍힘도 좌우 뒤집기로 데이터 두 배. `--dent-class` 로 이름만 주면 `none`.
- **`geometry.flip` 가 4가지** — `none · horizontal · vertical · both`(YAML `true/false` 는 both/none 으로 그대로 읽힘, rng 소비 both 2·h/v 1·none 0 → 기존 레시피·골든 불변). 조명 경고는 **방향으로 판단**(`flip_breaks_lighting`): 위/아래에서 오는 조명이면 `horizontal` 은 안전(찍힘도 좌우 뒤집기로 두 배). `per_class` 표·카드 폼(콤보)·`DENT_OVERRIDE`(`none`)·프리셋 YAML 어휘 갱신.
- `tools/train_smoke.py --synthetic a --synthetic b …` — 출력 여러 개(다른 프리셋·시드)를 한 학습셋으로(접두어 `syn<k>_`, 클래스 순서가 다르면 오류).
- 스튜디오 기하 카드에 **`per_class` 편집 표** — 은행 클래스마다 적용 체크 · 회전 lo/hi · flip(기본/켬/끔). 디바운스 뒤 재검증(카드 폼과 같은 규칙), scale 오버라이드는 YAML 값 보존. ▶ 버튼·`--auto-dent` 결과가 표에 그대로 보인다.
- 검수 탭 **표시된 것 전부 반려** — 현재 필터(예: 조명 뒤집힘 의심·폴백)의 합성 결과를 한 번에.
- `bank ls` `lightR` 열에 `*`(유의한 조명 의존) — 0.66 과 1.00* 을 한눈에.
- 스튜디오 변형 카드에 **↯ 조명 뒤집힘 의심** — 은행에서 방향이 유의한 클래스(`Bank.real_lighting_direction`, `ClassSummary.light_dir`)의 실제 평균 방향과 90° 넘게 벗어난 인스턴스가 있으면 캡션 ↯ + 툴팁(인스턴스·조치). 검수 탭 필터를 미리보기에서 미리. `core.appearance.lighting_stats/flipped_instances`.

### Changed
- 프리셋 YAML flip 어휘 `both`/`none`(값 동일) · `DENT_OVERRIDE.flip = none`, 자동 경로는 방향에 따라 `horizontal`/`vertical`.
- CI wheel e2e 에 ring 퀵스타트 경로(16-54).

## [0.7.5] - 2026-09-16

v0.7.4 의 조명 의존 판정 수정(소표본 오판). 다른 변경 없음.

### Fixed
- **조명 의존 판정에 유의성** — R ≥ 0.5 에 더해 n·R² ≥ 2.9(Rayleigh, p ≈ 0.05)를 요구(`core.appearance.is_directional`, `ClassSummary.directional`, `bank ls --json directional`). 무작위 각도의 R 은 ≈ 1/√n 이라 클래스당 5장짜리 은행에서 scratch·stain 까지 '조명 의존'으로 잡혀 per_class 가 불필요하게 들어가던 것(0.7.4 스모크에서 발견). n=3 은 R ≥ 0.98, n=5 는 0.76, n ≥ 12 는 0.5. 모든 소비처(경고·auto-dent·카드 버튼·은행 탭·퀵스타트·검수 뒤집힘·리포트)가 같은 판정을 쓴다.

## [0.7.4] - 2026-09-16

v0.7.3 후속 — 클래스별 기하(`geometry.per_class`) · 검수 정밀화(조명 뒤집힘 의심 필터 · 실제 분포 클래스 필터) · 조명 지표를 은행·라벨·검수 어디서나. 레시피 스키마는 옵션 추가만.

### Added
- **`geometry.per_class`** — 클래스별 기하 오버라이드(`{cls: {scale?, rotate?, flip?}}`, 준 필드만 덮어씀). 한 은행에 스크래치(±180)와 찍힘(±15·flip 끔)이 섞여 있을 때 레시피 하나로. 오버라이드 없는 클래스는 바이트 동일 · 사이드카 `geometry.per_class: true` · 은행에 없는 클래스는 `validate_against` 경고 · `lighting_warning`/배치 진단이 클래스별 범위로 판단(경고문이 문법을 안내). 카드 편집기엔 안 나옴(YAML).
- 퀵스타트(GUI '샘플 데이터' · `sample --quickstart`)가 은행의 조명 의존 클래스(샘플 pit)를 `geometry.per_class` 로 써 준다 — 처음 만든 레시피부터 조명 경고 없음. 검수 히스토그램 제목에 '실제 = <클래스>'.
- 검수 '실제' 분포는 **레시피가 뽑은 클래스만**(`source.classes` → `class_ratio` 키 → 전부) — pit 만 합성한 출력을 은행의 스크래치와 비교하지 않는다(`ReviewSession.real_classes/real_sources`).
- `dataset prune --drop-flipped` — 조명 뒤집힘 의심(검수 탭 필터와 같은 집합, 은행 필요)도 제외. `prune_dataset(drop_indices=)`.
- 빈 상태 안내 ⓪ "데이터가 하나도 없으면 상단 '샘플 데이터'".
- 스튜디오 기하 카드: 조명 경고가 있으면 **▶ 조명 클래스만 ±15°·flip 끔** 버튼(한 번에 `per_class` 적용, 경고가 사라짐) · 카드 아래 `per_class` 정보 한 줄(폼엔 없는 dict 를 읽기 전용으로).
- `recipe init --dent-class <cls>`(반복) · `--auto-dent`(은행의 lightR ≥ 0.5 클래스를 자동으로) — `geometry.per_class` 에 ±15°·flip 끔을 써 준다. 헤더 주석에 기록.
- 검수 탭 필터 **조명 뒤집힘 의심 flipped lighting** — 실제 클래스 방향이 뚜렷할 때(R ≥ 0.5, n ≥ 3) 그 평균 방향에서 90° 넘게 벗어난 인스턴스가 있는 이미지. 리포트에 목록 한 줄. `core/appearance.circular_mean/angle_diff`.
- 배치 탭 로그에 prepare 경고(축척·저신뢰·조명)와 **배치 가능성 진단** 한 줄(ROI 최대 폭 vs 패치 폭 · 빠듯/불가면 경고) — 종전엔 run 중 경고만 보였다.
- `bank preview`·은행 탭 타일 오른쪽 위에 **조명 방향 화살표**(밝은 쪽) — 클래스 안에서 화살표가 한 방향이면 조명 의존 결함(샘플 pit 7개 전부 ↓).
- 라벨 탭 통계에 **조명 방향**(각도 + 8방향 낱말) — 라벨링하면서 이 결함이 조명 의존인지(어느 쪽 림이 밝은지) 바로 본다.

### Changed
- `Bank.summary()` 캐시(조명 R 계산이 들어가 카드 편집마다 `reprepare` → `lighting_warning` 이 다시 재던 것) — Bank 는 로드 뒤 불변.
- README 검수 탭 스크린샷 교체(조명 방향 히스토그램 · 뒤집힘 의심 필터 33/40).

### Fixed
- CI ubuntu(PySide6 없음)에서 깨지던 테스트 — Qt 없는 테스트 파일이 탭을 import(16-41). 릴리스 zip 과 무관.

## [0.7.3] - 2026-09-16

v0.7.2 후속 — 진단·시작 패치(조명 방향 지표·경고 · 배치 가능성 진단 · GUI 샘플 한 클릭 · ring 샘플 · `max_align_deg`). 레시피 스키마는 옵션 추가만.

### Added
- 검수 탭 분포에 **질감 texture**(마스크 안 Sobel 그래디언트 평균)·**선명도 sharpness**(라플라시안 분산) — 대비와 함께 외형 지표 3종(전부 선형 구간, 이미지·마스크를 한 번 읽어 같이 캐시).
- 검수 탭 분포에 **조명 방향 lighting**(둘레 2 px 링에서 밝은 쪽 각도, −180..180° 고정 구간) + **조명 일관성 R**(전체·클래스별, 1 = 하이라이트가 모두 같은 방향) — KI #5 의 검수 근거. 어느 클래스든 실제 R ≥ 0.5 인데 합성 R < 0.3 이면 리포트·히스토그램 제목에 "회전이 조명을 뒤집음 → dent-graft". 샘플 pit: 실제 0.99 · poisson-graft(±180°) 0.12 · dent-graft(±15°) 0.51.
- **조명 의존 클래스를 은행에서 미리 잡는다** — `bank ls` 에 `lightR` 열(클래스별 실제 소스의 조명 일관성 R, n ≥ 3)·`--json` `light_r/light_n`, `run`/스튜디오 `prepare` 경고 `lighting_warning`(R ≥ 0.5 클래스를 rotate 폭 > 90° 또는 flip 으로 합성하면 "하이라이트 방향이 뒤집힙니다 → dent-graft"). 외형 지표 4종은 `core/appearance.py` 로(검수 탭·은행·runner 가 같은 정의).
- 스튜디오: prepare 경고 중 스테이지 접두가 있는 것(`geometry:` 조명 경고)은 **해당 카드 ⚠** 로 — 변형을 바꿔도 남고, 회전을 조이거나 flip 을 끄면 사라진다. `reprepare`(카드 편집)도 축척·저신뢰·조명 경고를 다시 계산(전엔 카드 편집 뒤 상태바에서 사라졌음).
- 은행 탭 요약 한 줄에 **조명 의존 클래스**(`lightR` ≥ 0.5) 와 dent-graft 안내.
- 스튜디오 배치 카드에도 **배치 가능성**: 미리보기 ROI(축소본 ÷ 배율)로 잰 허용 폭 vs 클래스별 패치 폭이 빠듯/불가면 ⚠(annulus r_inner/r_outer·erode_px·scale 을 만지면 바로 바뀐다).
- `anograft run --report` — 끝나면 출력 폴더에 검수 리포트 HTML(= `dataset report`)을 바로. 요약에 조명 일관성 R.
- **`run --dry-run` 배치 가능성 진단**(`runner.fit_diagnostic`) — 처음 3장의 ROI 를 실제로 풀어(캐시, rng 0회) 허용 영역 최대 폭(내접원 지름, 테두리 여유 제외)을 재고 클래스별 패치 긴 변 중앙값 × `geometry.scale` 상한과 견준다: `roi width` 행 + `fit <class>` 행(가능 · 빠듯(폭의 80 % 초과) · 불가(shrink_on_fail 뒤에도 폭 초과)) + `placement:` 경고. KNOWN-ISSUES 부록의 "ROI 폭 대비 패치 크기 사전 진단".
- **GUI 상단 '샘플 데이터 Sample…' 버튼** — 판/원형을 고르고 폴더를 고르면 샘플 → 은행 → 레시피(`samples/quickstart.py`, Qt 없음)를 만들어 스튜디오·은행 탭에 연다(원형은 dent-graft + annulus). CLI 는 `anograft sample --quickstart`(`<out>/bank`·`normals.txt`·`recipe.yaml`). 작은 샘플에서 빈 클래스는 `source.classes` 로 제외.
- **`anograft sample --shape ring`** — 원형 부품 샘플(원주 방향 브러시 결의 가공 링 면 + 어두운 리세스, 결함은 링 면 안에만). Otsu 전경이 링이라 `annulus` ROI 의 자동 중심·`dent-graft` 를 데이터 없이 연습(`recipe init --preset dent-graft --roi annulus`). `plate`(기본)는 종전과 바이트 동일.
- `anograft recipe init --roi <method> --um-per-px <피치>` — 프리셋의 ROI 만 교체(예: `--preset dent-graft --roi annulus`)·대상 피치. 헤더 주석에 남는다.
- `anograft recipe check` 가 `run`/GUI 와 같은 prepare 경고(대조·축척·저신뢰·조명)를 낸다(`runner.prepare_warnings` 공용).
- `structure-aware` 배치 옵션 **`max_align_deg`**(null = 종전) — 결·접선 정렬에 이보다 큰 회전이 필요한 자리는 정렬하지 않는다(사이드카 `align_capped` 에 필요했던 각). rng 소비 동일.

### Changed
- **프리셋 `dent-graft` 에 `max_align_deg: 30`** — 종전엔 geometry 를 ±15° 로 묶어도 structure-aware 정렬이 (−90, 90] 까지 돌려 하이라이트 방향을 깼다(원형 부품의 접선 정렬은 한 바퀴). 골든 `dent-graft-{color,gray}` 갱신(267/4096 px). 다른 프리셋은 변화 없음.

## [0.7.2] - 2026-09-16

v0.7.1 후속 — 흐름 다듬기(배치→검수 · 저신뢰 차례로 다듬기 · 대비 분포 · 스튜디오 µm/px · doctor). 포맷·스키마 변경 없음.

### Added
- 스튜디오 변형 카드에 **저신뢰 소스 표시**(캡션 ⚠ + 툴팁에 소스 id) — 미리보기가 그럴듯해도 은행 마스크가 헐거우면 알 수 있게(KI #3).
- **`anograft doctor [--json]`** — 환경 진단 한 장(버전·OpenCV 스레드·pydantic·Qt/PySide6·프리셋·불가 method·frozen·stdout 인코딩). 다른 PC 에서 문제를 보고할 때 첫 줄. **`bank ls --json`** — 스크립트용(클래스별 행 + 저신뢰 id + 태그).
- **저신뢰 소스 차례로 다듬기**: 은행 탭 **저신뢰 전부 차례로 다듬기** 버튼 → 라벨 탭 편집 모드에 **다음 저신뢰 소스** 버튼 — 신뢰도 오름차순으로 하나씩 열어 다듬고 저장(저장한 소스는 목록에서 빠짐), 끝나면 은행 탭으로. KI #3 "추정 초안을 사람이 다듬는" 흐름의 반복 경로.
- 검수 탭 분포에 **대비 contrast**(마스크 안 평균 그레이 − 8 px 링 평균, 선형 구간) — 합성 결함이 실제(은행 소스)보다 옅어졌는지(조화가 톤을 빨아들였는지) 한눈에. 리포트에도 세 번째 히스토그램. v1.0 "실제 결함 분포 비교"의 첫 외형 지표.
- 스튜디오 입력 패널에 **대상 µm/px** 스핀(`inputs.um_per_px`, 0 = 모름) — 은행 소스 피치와 함께 축척 정합을 GUI 에서 켠다(KNOWN-ISSUES #6 의 GUI 쪽).
- 배치 탭 **검수 탭에서 열기** 버튼(완료 뒤) → 검수 탭이 그 출력을 열고 전환 · 검수 탭 **다음 미검수 (N)** 버튼/단축키(끝이면 처음부터).

## [0.7.1] - 2026-09-16

v0.7.0 후속 패치 — ROI 캐시(스튜디오 속도) · 레시피 상대경로 폴백 · `bank merge` · 검수 리포트 · 자동 선택 점수 · CI wheel e2e. 포맷·스키마 변경 없음.

### Added
- 라벨 탭 **자동 선택 뒤 타당성 점수**: 상태줄·결과 칸에 `confidence 0.xx`, 0.5 미만이면 `⚠ 저신뢰(flags) — 다듬거나 다른 방법으로`(임포터·초안과 같은 `mask_confidence`).
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

[Unreleased]: https://github.com/slnu21/Graft/compare/v0.10.0...HEAD
[0.10.1]: https://github.com/slnu21/Graft/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/slnu21/Graft/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/slnu21/Graft/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/slnu21/Graft/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/slnu21/Graft/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/slnu21/Graft/compare/v0.7.7...v0.8.0
[0.7.7]: https://github.com/slnu21/Graft/compare/v0.7.6...v0.7.7
[0.7.6]: https://github.com/slnu21/Graft/compare/v0.7.5...v0.7.6
[0.7.5]: https://github.com/slnu21/Graft/compare/v0.7.4...v0.7.5
[0.7.4]: https://github.com/slnu21/Graft/compare/v0.7.3...v0.7.4
[0.7.3]: https://github.com/slnu21/Graft/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/slnu21/Graft/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/slnu21/Graft/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/slnu21/Graft/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/slnu21/Graft/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/slnu21/Graft/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/slnu21/Graft/compare/v0.1.0...v0.4.0
[0.1.0]: https://github.com/slnu21/Graft/releases/tag/v0.1.0

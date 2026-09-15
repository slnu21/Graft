# 알려진 문제와 보완 방향

v0.5.0 을 실제 산업 데이터에 처음 적용하며 드러난 것들. 각 항목은 **증상 → 원인(코드 위치) → 보완 방향** 순.

> **v0.6.0(2026-09-15)에서 10건 전부 처리** — 각 항목 위의 "해결" 메모와 데브로그 참조. 실제 데이터로의 2차 적용(해결 확인)은 아직이며, #3 의 추정 알고리즘 자체(SAM)는 v1.0.

> 적용 대상: 경면 금속 원형 부품(토크스 소켓), 결함 = 찍힘, 1400×1400 8-bit RGB, NG 10장(YOLO 박스 라벨) + OK 10장.
> 결함은 가공 링 면에만 발생(사용자 확인). 이 조건에서 측정한 수치를 근거로 쓴다.

| # | 문제 | 심각도 | 성격 | 상태 |
|---|---|---|---|---|
| 1 | `mask_dir` ROI 가 미리보기 축소에서 조용히 깨진다 | 높음 | 버그 | ✅ v0.6 `roi-mask-resize` |
| 2 | ROI 가 "부품 전체"라 결함이 안 생기는 면에도 배치된다 | 높음 | 설계 공백 | ✅ v0.6 `roi-annulus` |
| 3 | 박스→마스크 자동 추정이 저대비 경면 표면에서 실패한다 | 높음 | 알고리즘 한계 | ✅ v0.6 `mask-confidence`(실패를 **잡아내는** 지표 + 라벨 탭 다듬기) · 추정 자체의 개선(SAM)은 v1.0 |
| 4 | `mask_dir` 는 대상마다 파일이 필요해 실무 부담이 크다 | 중간 | 설계 공백 | ✅ v0.6 `roi-annulus`(원형) · ✅ `label-yolo-roi`(라벨 탭 ROI 모드) |
| 5 | 회전 기본값 ±180° 가 조명 의존 결함에 물리적으로 맞지 않는다 | 중간 | 기본값 | ✅ v0.6 `preset-rotate`(`dent-graft` 신설, 기존 기본값 유지) |
| 6 | µm/px 축척 정합이 사실상 꺼진 채로 돌아간다 | 중간 | 기본값 | ✅ v0.6 `source-tags-scale` |
| 7 | 은행 소스의 태그를 선택에 쓸 수 없다 | 중간 | 기능 누락 | ✅ v0.6 `source-tags-scale` |
| 8 | GUI 가 기존 YOLO 라벨을 읽지 못한다 | 낮음 | 기능 누락 | ✅ v0.6 `label-yolo-roi` |
| 9 | GUI 를 레시피 인자 없이 켜면 빈 화면 | 낮음 | UX | ✅ v0.6 `label-yolo-roi` |
| 10 | `recipe init --write` 가 상위 폴더를 만들지 않는다 | 낮음 | 버그 | ✅ v0.6 `roi-mask-resize` |

---

## 1. `mask_dir` ROI 가 미리보기 축소에서 조용히 깨진다

> **해결(v0.6, 2026-09-15 `roi-mask-resize`)** — `roi_from_mask` 가 같은 비율의 마스크를 `INTER_NEAREST` 로 대상 크기에 맞춤(사이드카 `roi.resized_from`), 비율이 다르면 실패. skipped 사유는 `roi:` 경고를 우선해 원인이 보이고, 스튜디오 배치 카드·변형 카드 툴팁·상태바에 그 경고가 표시된다.

**증상** — 스튜디오 변형 카드가 전부 `배치 실패 → ROI 없음`. 같은 레시피가 CLI `run` 에서는 정상 동작(37/40 생성).

**원인** — 스튜디오는 대상을 `long_side`(기본 1024)로 **축소한 뒤 합성**한다(`gui/studio/jobs.py` · `preview_target`). `mask_dir` 마스크는 원본 크기(1400)이므로:

```python
# core/roi.py · roi_from_mask
if mask.shape[:2] != tuple(shape[:2]):
    raise ValueError(f"ROI 마스크 크기 {mask.shape[:2]} 가 대상 {tuple(shape[:2])} 와 다릅니다")
```

`core/stages/roi.py` 의 `MaskDirRoi.apply` 가 이 예외를 fail-soft 로 삼켜 `roi=None` 으로 만들고, `core/stages/placement.py:153` 이 `"ROI 없음"` 으로 실패한다.

**왜 나쁜가** — 사용자에게 원인이 전혀 보이지 않는다. "미리보기 해상도"라는 무관해 보이는 설정이 기능을 죽이고, 표시되는 메시지는 진짜 원인을 가린다. 축소 3종(1024·768·512) 모두 같은 증상이고 `원본 해상도`만 동작한다.

**보완 방향**
- (권장) `roi_from_mask` 에서 크기가 다르면 `INTER_NEAREST` 로 리사이즈. ROI 는 대략적 허용 영역이라 픽셀 정밀도가 불필요하고, 이 제약을 유지할 실익이 없다.
- 또는 `mask_dir` 사용 시 스튜디오 축소를 강제로 원본 고정.
- 최소한 fail-soft 경고를 GUI 표면에 노출 — 지금은 파이프라인 로그에만 남는다.

---

## 2. ROI 가 "부품 전체"라 결함이 안 생기는 면에도 배치된다

> **해결(v0.6, 2026-09-15 `roi-annulus`)** — ROI 메서드 `annulus` + 프리셋 `annulus-graft`. 중심·반경을 대상마다 자동 검출하고 `r_inner`/`r_outer` 비율로 링을 지정(실측 0.56~0.9). 합성 토크스 축소판(중심 ±60 px 이동) 8장 16결함 → 링 적중 16/16(otsu 12/16). 아래 부록의 "ROI 폭 대비 패치 크기" 진단도 같이(배치 실패 사유에 붙음). 스튜디오 ROI 오버레이는 이미 있으므로 배치 카드의 r_inner/r_outer 를 링에 맞추며 확인한다.

**증상** — 실제 찍힘 10개는 전부 가공 링 면(이미지 중심 기준 반경 **319~510px**)에 있는데, 기본 `otsu` ROI 로 합성하면 결함이 중앙 육각 리세스에 떨어진다.

측정값(합성 결함의 중심 반경, 정상 범위 319~510):

| 설정 | 적중 | 빗나간 값 |
|---|---|---|
| `poisson-graft` (otsu ROI · sampled) | **0 / 4** | 68 · 82 · 126 · 202 — 전부 중앙 리세스 |
| `structure-aware-graft` (grabcut ROI · erode 8) | 3 / 4 | 570 — 부품 바깥 경계 |
| 같은 설정 · `erode_px: 70` | 4 / 6 | 275 · 306 — 중앙 리세스 |
| `mask_dir` ROI (링 마스크) | **9 / 9** | — (407~443 로 수렴) |

**원인** — `otsu` · `grabcut` 은 **물체 vs 배경**만 가른다. 물체 **내부**에서 검사 대상 면과 그렇지 않은 면을 구분하는 수단이 파이프라인에 없다. `erode_px` 는 ROI 바깥 테두리에서만 깎기 때문에 중앙 홈을 제외하지 못하고, 오히려 배치를 중앙으로 민다.

`structure-aware` 는 그래디언트가 큰 곳을 선호해 거친 가공면을 간접적으로 고르지만(3/4), **보장이 아니다.**

**왜 중요한가** — 실제로 결함이 발생하지 않는 위치에 정답 라벨이 붙은 학습 데이터가 만들어진다. 합성 데이터의 가치를 정면으로 훼손한다.

**보완 방향**
- **파라메트릭 ROI 메서드** 추가 — 예: `roi: {method: annulus, r_inner: …, r_outer: …, center: auto}`. 원형 부품 계열에 재사용되고, 중심을 대상마다 검출하므로 위치 변동에 자동 대응한다. 등록 비용은 `스테이지 클래스 1 + 레지스트리 1줄 + 프리셋 1장`.
- ROI 편집 UI. 편집 엔진은 이미 있다(4번 참조).
- 스튜디오의 `배치 허용 영역 ROI` 오버레이를 편집 중 실시간으로 겹쳐 보이게.

---

## 3. 박스→마스크 자동 추정이 저대비 경면 표면에서 실패한다

> **해결(지표, v0.6 2026-09-15 `mask-confidence`)** — 추정 알고리즘은 그대로지만 **실패가 보이게** 했다. `mask_confidence` 가 마스크 안/밖 대비 분리도·박스 테두리 접촉·조각 수·포화로 0..1 점수와 flags 를 매기고, `import-yolo` 가 메타에 남긴다. 이 데이터의 실패 모드(면적은 정상, 대비 없음)는 `low-contrast` 로 잡힌다. `bank ls` `lowconf` 열 · `bank preview` 빨간 테두리 · `run`/스튜디오 경고(추정의 절반 넘으면 강화). 저신뢰 소스는 라벨 탭 YOLO 초안으로 열어 다듬는다(`label-yolo-roi`). **SAM 플러그인은 v1.0**.

**증상** — YOLO 박스 10개를 `bank import-yolo` 로 넣으면 `grabcut 4 · otsu 6 · ellipse 0` 으로 지표상 정상이지만, 실제 마스크는 **결함이 아니라 밝은 금속 텍스처**를 잡는다. 이 은행으로 합성하면 결함이 거의 보이지 않는다(밝은 금속을 밝은 금속 위에 이식하는 꼴).

**원인** — `bank/mask_from_box.py` 의 폴백 사슬은 **면적 비율만** 검증한다:

```python
AREA_RATIO_MIN = 0.05
AREA_RATIO_MAX = 0.95
```

면적이 박스의 5~95% 안에 들면 채택하고 모양의 타당성은 보지 않는다. 경면 금속처럼 하이라이트가 포화되고(측정: 가로 프로파일에 255 다수) 결함 대비가 낮은 표면에서는 "그럴듯한 면적의 엉뚱한 영역"이 쉽게 만들어진다.

**파급** — 다른 제품(A~N)의 결함을 신규 제품(X)에 이식하는 설계 의도에서는 더 치명적이다. 헐거운 마스크는 결함뿐 아니라 **A 제품의 표면 자체를 X 에 이식**한다. 모델이 결함이 아니라 이질적 패치를 학습한다.

**보완 방향**
- **SAM 플러그인** — `pyproject.toml` 에 이미 자리가 있다(`torch = []`, 주석: *"v1.0 선택 플러그인(SAM·확산). 없으면 해당 프리셋이 비활성 + 이유 노출"*). 저대비 표면 이상에 GrabCut/Otsu 보다 훨씬 강하고, 라벨 작업량을 직접 줄인다.
- `bank ls` / `bank preview` 에 **추정 신뢰도** 지표 추가. 지금은 `ellipse` 폴백 비율만 경고 신호인데, 이 데이터에서는 폴백이 0인데도 마스크가 전부 틀렸다 — 현재 지표로는 잡히지 않는 실패 모드다.
- 추정 마스크(`mask_origin` 이 `yolo-box:*`) 비율이 높은 은행으로 `run` 할 때 경고 강화.

---

## 4. `mask_dir` 는 대상마다 파일이 필요해 실무 부담이 크다

> **해결(원형 부품, v0.6 `roi-annulus`)** — 파라메트릭 ROI 가 중심을 대상마다 검출하므로 파일이 필요 없다. **임의 형상(v0.6 `label-yolo-roi`)** — 라벨 탭 "저장 대상 → ROI 마스크": 정상 이미지에 허용 영역을 칠해 `<mask_dir>/<stem>.png` 로 저장(아래에서 말한 대로 `LabelSession` 편집 엔진 + 크롭 없는 저장 함수 `save_roi_png` 하나). 열 때 같은 이름 ROI 를 불러오므로 한 장을 다듬어 복사·보정하는 흐름도 된다.

**증상** — ROI 마스크는 `<path>/<대상 stem>.png` 로 **대상 1장당 1개**가 필요하고 크기가 정확히 같아야 한다. 촬영마다 부품 위치가 **40~70px** 이동하므로 고정 마스크 1장을 복사해 돌려쓸 수 없다(10장에 겹쳐 확인 — 여러 장에서 안쪽 경계가 중앙 홈을 침범하거나 바깥 경계가 부품을 벗어남).

**원인** — 대상과 마스크를 **파일명으로만** 잇는 설계라 정합·보정 단계가 없다.

**보완 방향**
- 2번의 파라메트릭 ROI 가 이 문제도 같이 해결한다(중심을 대상마다 검출).
- 또는 기준 마스크 1장 + 템플릿 매칭 정합.
- ROI 편집 UI — **편집 엔진은 이미 있다.** `gui/label/session.py` 의 `LabelSession` 은 Qt 비의존 · 원본 크기 배열 연산으로 `stroke`(브러시/지우개) · `fill_polygon` · `auto_select`(GrabCut) · `dilate`/`erode` · `undo`/`redo` 를 전부 갖췄다. ROI 편집에 필요한 것과 동일하다. 다른 점은 저장뿐 — `save_to_bank` 는 `BankWriter.add` 를 타며 **크롭해서** 은행에 넣는데, ROI 는 원본 크기 그대로 PNG 로 나가야 한다. 크롭·메타 없이 `self.mask` 를 그대로 쓰는 저장 함수 하나면 된다.

---

## 5. 회전 기본값 ±180° 가 조명 의존 결함에 물리적으로 맞지 않는다

> **해결(v0.6, 2026-09-15 `preset-rotate`)** — 프리셋 **`dent-graft`** 신설: 회전 ±15° · flip 끔 · 축척 0.9~1.1 · `structure-aware`(위치 균등, 긴 축을 결·접선에 정렬 — 아래 세 번째 보완 방향) · poisson NORMAL · 조화 0.2. 기존 프리셋 기본값은 텍스처성 결함(스크래치·얼룩)용이라 유지(사용자 결정 — 골든·게시 레시피 재현성). "조명 방향 고정" 옵션(회전 시 음영 보정)은 만들지 않았다 — 복사·붙여넣기로는 원리적으로 불가, v1.0 확산 인페인팅의 몫.
>
> **근거(0.7.2 뒤, `review-lighting`)** — 검수 탭 분포 **조명 방향**(둘레 2 px 링에서 밝은 쪽 각도) + **조명 일관성 R**(클래스별). 샘플 pit 98 인스턴스(`tools/make_sample_yolo.py` 은행, 시드 11): 실제 소스 R **0.99** · `poisson-graft`(±180°) **0.12**(무작위) · `dent-graft`(±15°, 정렬 끔) **0.51**. 두 조건 다 성립하면(실제 ≥ 0.5 **이고 n·R² ≥ 2.9 로 유의** · 합성 < 0.3) 리포트가 클래스 이름과 함께 dent-graft 를 권한다(소표본은 R 잡음 바닥이 1/√n 이라 유의성 없이는 오판). dent-graft 가 1.0 이 못 되는 건 GT `union`+`dilate_px 2` 가 하이라이트 림을 마스크 안으로 삼켜 링이 림 바깥을 재기 때문 — 실데이터에선 같은 지표로 실제 vs 합성을 보면 된다. **클래스별 오버라이드**(`geometry-per-class`, 0.7.3 뒤): 한 은행에 스크래치와 찍힘이 섞여 있으면 `geometry.per_class: {찍힘: {rotate: [-15, 15], flip: false}}` 로 그 클래스만 좁힌다(프리셋을 둘로 나눌 필요 없음). **정렬 상한**(`align-cap`, 0.7.3): dent-graft 의 structure-aware 정렬이 geometry 의 ±15° 를 무시하고 (−90, 90] 까지 돌리던 구멍 — `max_align_deg: 30` 으로 막음(원형 부품의 접선 정렬은 방향이 한 바퀴 돈다). **은행 단계에서도**(`lighting-warning`): `bank ls` 의 `lightR` 열(`*` = 유의)이 클래스별 실제 R 을 보여주고, 유의한 클래스를 rotate 폭 > 90° 또는 flip 으로 돌리는 레시피는 `run`/`recipe check`/스튜디오(기하 카드 ⚠ + **▶ 한 번에 per_class**)가 prepare 시점에 경고한다. **미리보기·검수에서도**: 변형 카드 **↯**, 검수 탭 필터 **조명 뒤집힘 의심**(실제 클래스 방향에서 > 90°), `dataset prune --drop-flipped`. 전체 흐름은 README "조명 의존 결함 한 바퀴"(2차 적용에서 찍힘 클래스에 `*` 가 뜨는지, 뒤집힘 필터가 dent-graft/per_class 에서 0 에 가까운지 확인).

**증상** — 합성 결과가 부자연스럽다. 음영과 하이라이트의 관계가 주변과 어긋난다.

**원인** — `geometry.affine` 의 기본 `rotate: [-180.0, 180.0]`. 찍힘은 **3D 변형**이라 보이는 모양이 곧 조명 효과다. 패치를 180° 돌리면 음영/하이라이트의 위아래가 뒤집히는데 조명은 돌지 않으므로 물리적으로 불가능한 그림이 된다. 사람 눈이 가장 먼저 잡아내는 종류의 위화감이다.

기본값이 텍스처성 결함(스크래치·얼룩)을 전제로 정해져 있다.

**보완 방향**
- 프리셋별 `rotate` 기본값 재검토 — 조명 의존 결함용 프리셋은 좁은 범위.
- "조명 방향 고정" 옵션(회전 시 음영을 보정하거나 회전을 제한).
- 원형 부품은 접선 방향 정렬이 물리적으로 타당하다 — `structure-aware` 가 국소 구조 방향에 주축을 맞추므로 근사치로 쓸 수 있다. 문서화 가치가 있다.

> 이와 별개로, 복사·붙여넣기 방식은 조명 의존 결함을 원리적으로 재현할 수 없다. 근본 해결은 **확산 인페인팅**(그 자리의 곡률·조명 맥락으로 새로 렌더링)이며 `torch` extra 가 그 자리다. 다만 GPU 의존·비결정성이 현재의 "레시피+시드 같으면 바이트 동일" 보장과 충돌하고, 생성 영역과 실제 변화 영역이 달라 **GT 마스크 정확도**라는 새 문제를 만든다(`gtmask` 의 `diff`/`union` 정책 의존도가 커진다). 선택 플러그인으로 두는 현재 설계가 타당하다.

---

## 6. µm/px 축척 정합이 사실상 꺼진 채로 돌아간다

> **해결(v0.6, 2026-09-15 `source-tags-scale`)** — `runner.prepare` 가 소스·대상 피치 상태를 보고 "축척 정합 꺼짐/일부 꺼짐" 한 줄을 경고(CLI stderr · 스튜디오 상태바 · 배치 로그). `bank ls` 헤더에 `미지정 n/N`, 클래스별 `no_um` 열. 기본값 자체는 바꾸지 않았다(피치는 사용자만 안다).

**증상** — 서로 다른 카메라/배율로 찍은 결함이 대상에서 물리적으로 틀린 크기가 된다.

**원인** — `core/scale.py` 의 `physical_scale` 은 소스·대상 **양쪽 모두** 피치가 있어야 동작하고, 한쪽이라도 없으면 `factor 1.0` + 사유로 조용히 no-op 한다. 그런데 `bank import-*` 의 `--um-per-px` 도, 레시피 `inputs.um_per_px` 도 기본이 미지정이다. 결과적으로 기본 경로에서는 축척 정합이 **항상 꺼져 있다.**

A~N 제품 결함을 X 에 이식하는 설계 의도에서는 이 기능이 핵심인데, 켜야 켜진다는 사실이 드러나지 않는다.

**보완 방향**
- 은행에 피치 없는 소스가 있고 대상에도 없을 때 `run` 에서 경고(현재는 로그에만).
- `bank ls` 에 `um_per_px` 미지정 소스 수 표시.

---

## 7. 은행 소스의 태그를 선택에 쓸 수 없다

> **해결(v0.6, 2026-09-15 `source-tags-scale`)** — `pipeline.source.tags: {include, exclude}`. `BankSource` 가 클래스별 풀을 태그로 거른 뒤(순서 유지) 뽑고, `validate_against` 가 필터 후 수로 경고/실패를 판단한다. `bank ls` 가 태그별 수를 보여 준다.

**증상** — `--tags` 로 제품명 등을 붙여 저장할 수 있고 `<class>/<id>.json` 에도 남지만, 합성할 때 "A·C·F 제품 결함만 사용" 같은 통제가 불가능하다.

**원인** — `core/recipe.py` 의 `BankSourceConfig` 는 `classes` 와 `min_sources_warn` 만 갖는다. `tags` 는 `core/types.py:29` 의 필드 선언 외에 `core/` 어디에서도 쓰이지 않는다.

**왜 중요한가** — 여러 제품의 결함을 한 은행에 누적하는 것이 설계 의도인데, 누적 후 골라 쓸 수단이 없다. 클래스명에 인코딩하면(`scratch-A`) 출력 `data.yaml` 의 클래스 수가 제품 수만큼 불어난다.

**보완 방향** — `source.tags` 필터(포함/제외) 추가. 저장·로드 경로는 이미 있으므로 선택 단계만 붙이면 된다.

---

## 8. GUI 가 기존 YOLO 라벨을 읽지 못한다

> **해결(v0.6, 2026-09-15 `label-yolo-roi`)** — 이미지를 열 때 `find_yolo_label`(images↔labels 미러 · 같은 폴더 · labels/)로 라벨을 찾아 `YoloDraft` 로 파싱, 박스는 `mask_from_box` 추정·폴리곤은 채움으로 **초안 프리필**(클래스별, 자동/수동). 손대지 않은 초안은 `yolo-box:*`(est) 로 저장돼 3번의 지표와 이어진다.

**증상** — 라벨 탭에서 `images/` 를 열면 이미지 목록은 뜨지만, 옆에 있는 `labels/*.txt` 의 기존 박스는 표시되지 않는다. 이미 라벨이 있어도 GUI 에서 재활용할 수 없고 처음부터 그려야 한다.

**원인** — `gui/label/tab.py` 의 `open_folder` 는 `imgio.list_images(d)` 만 호출한다. 라벨 탭 전체에 `labels` · `.txt` · `yolo` 참조가 없다.

**보완 방향** — 기존 박스/폴리곤을 초안으로 로드(박스는 `auto_select` 결과를 미리 채워 두는 식). 3번의 자동 추정 품질 문제와 함께 보면, "추정 초안을 사람이 다듬는" 흐름이 자연스럽다.

---

## 9. GUI 를 레시피 인자 없이 켜면 빈 화면

> **해결(v0.6, 2026-09-15 `label-yolo-roi`)** — 최근 레시피 복원(`QSettings` slnu21/Graft) + 없으면 스튜디오 캔버스 ko/en 4단계 안내. 상대경로는 v0.7.x `recipe-relative-paths` 에서 **cwd 우선, 없으면 레시피 파일 기준**으로 해석(입력 경로 4개, `output.root` 제외).

**증상** — `anograft-gui.exe` 를 그냥 실행하면 아무것도 로드되지 않아 무엇을 해야 할지 알 수 없다.

**원인** — `gui/app.py:156` — `run_app` 은 `recipe` 인자가 있을 때만 스튜디오와 라벨 탭에 경로를 넣는다.

**보완 방향** — 최근 레시피 기억, 또는 빈 상태에 다음 행동을 안내하는 문구.

> 관련: 레시피의 상대경로는 **현재 작업 디렉터리** 기준이라, GUI 를 다른 위치에서 실행하면 조용히 깨진다. GUI 사용을 전제하면 절대경로를 권하거나, 레시피 파일 기준 상대경로를 지원하는 편이 안전하다.

---

## 10. `recipe init --write` 가 상위 폴더를 만들지 않는다

> **해결(v0.6, 2026-09-15 `roi-mask-resize`)** — `parent.mkdir(parents=True, exist_ok=True)`.

**증상**

```
FileNotFoundError: [Errno 2] No such file or directory: 'recipes\\gbs1005.yaml'
```

**원인** — `cli.py` 의 `cmd_recipe_init` 이 `Path.write_text` 를 바로 호출한다. `session.save` 등 다른 쓰기 경로는 `parent.mkdir(parents=True, exist_ok=True)` 를 한다.

**보완 방향** — 동일하게 `parent.mkdir`.

---

## 부록 — 이번 적용에서 확인된 것

동작이 확인된 부분도 함께 남긴다.

- YOLO 임포트, 은행 누적, 재현성(같은 명령 → 마스크 10개 해시 동일), 한글 클래스명(UTF-8 폴더·`bank.yaml` 정상), YOLO writer 출력(`nc: 1` · `names: [찍힘]`), 워커 병렬 `run`(ok 37 · skipped 3 · 오류 0) 모두 정상.
- 스킵 3건은 전부 `placement: 배치 실패 — max_tries 소진`. 링 폭보다 큰 결함 패치가 들어갈 자리를 못 찾은 경우로, 좁은 ROI 를 쓰면 예상되는 동작이다. 다만 **ROI 폭 대비 패치 크기**를 사전에 알려 주는 진단이 있으면 좋겠다. → **0.7.3 `run --dry-run`** 의 `roi width`·`fit <class>` 행(가능/빠듯/불가)과 `placement:` 경고.
- GrabCut ROI 는 1400×1400 에서 미리보기 1장당 수십 초가 걸려 대화형 사용에 부담이 있다(`work_px` 조정 여지).

---

## 2차 적용 절차 (v0.7.5 기준 — 다른 PC 에서 그대로 따라 하기)

> 목적: 위 10건이 실제 토크스 소켓 데이터에서 통했는지 확인하고, 아래 **결정 대기 항목**을 확정한다. 결과는 이 파일에 "2차 결과" 절로 덧붙여 PR 로.

```powershell
# 0. 환경 — 문제 보고 첫 줄
anograft doctor --json > doctor.json

# 1. 은행 (기존 은행이 있으면 bank ls 만; 새로 넣으면 --tags 로 제품명·--um-per-px 로 피치)
anograft bank import-yolo --images ng/images --labels ng/labels --names ng/data.yaml --out bank/torx --tags torx,lot1 --um-per-px <피치>
anograft bank ls bank/torx --json > bank-ls.json      # lowconf 열 · no_um 열 — 저신뢰 비율이 #3 의 답
anograft bank preview bank/torx --out bank-preview.png  # 빨간 테두리 = 저신뢰. 실제로 엉뚱한 마스크와 일치하는가?

# 2. 저신뢰 다듬기 (GUI) — 은행 탭 → "저신뢰 전부 차례로 다듬기" → 라벨 탭에서 다듬고 Ctrl+S → 다음 저신뢰
#    또는 기존 YOLO 라벨을 라벨 탭에서 열면 초안이 자동으로 채워진다(images/ 옆 labels/).

# 3. 링 ROI 레시피 (#2 #4) — annulus-graft, r_inner/r_outer 를 스튜디오 ROI 오버레이에 맞춰 조정(실측 0.56~0.9)
anograft recipe init --preset annulus-graft --bank bank/torx --targets ok/ --out out/torx-annulus --count 40 --write recipes/torx-annulus.yaml
#    찍힘이면 dent-graft 도 (#5, 0.7.3): --preset dent-graft --roi annulus  (피치를 알면 --um-per-px <피치>)
#    스크래치·찍힘이 섞인 은행이면 프리셋은 그대로 두고 --auto-dent (lightR ≥ 0.5 클래스만 ±15·flip 끔 = geometry.per_class)
#    (데이터 없이 이 단계를 먼저 연습: anograft sample --out samples/ring --shape ring → import → 위 init)
anograft run recipes/torx-annulus.yaml --dry-run            # roi width(링 폭) vs fit <class>(패치 폭) — 불가/빠듯이면 geometry.scale·erode_px 먼저
anograft run recipes/torx-annulus.yaml --workers 4 --report # stderr 경고: 축척 정합 · 저신뢰 · skipped 사유(ROI 폭 vs 패치) · --report 는 리포트 HTML 까지

# 4. 검수 (GUI 검수 탭 또는) — 반려하고 정리본 + 리포트
anograft dataset report out/torx-annulus                    # 합성 vs 실제 면적·긴 변·대비 히스토그램
anograft dataset prune out/torx-annulus --out out/torx-pruned [--drop-flipped]   # 조명 뒤집힘 의심도 빼려면

# 5. 학습(선택, ultralytics 별도 설치) — 합성 유/무 mAP
python tools/train_smoke.py --synthetic out/torx-pruned --base <기존 YOLO 셋> --out train/merged
```

**확인할 것 (항목 ↔ 근거)**

| # | 확인 | 어디서 |
|---|---|---|
| 1 | 스튜디오 1024 축소에서 `mask_dir` ROI 가 동작 | 스튜디오 배치 카드 ⚠ 없음, ROI 오버레이 |
| 2 · 4 | annulus 링 안에만 배치(중심 반경 319~510) | 사이드카 `placement.center` 반경 · `roi.center_source: detected` |
| 3 | 저신뢰 비율이 실제 실패 마스크와 맞는가 | `bank ls --json` `low_confidence_ids` vs `bank preview` 눈 확인 |
| 5 | dent-graft 가 자연스러운가(음영 방향) | `bank ls` `lightR`(찍힘 클래스 ≥ 0.5 인가) · poisson-graft 로 `run --dry-run` 하면 조명 경고가 뜨는가 · 스튜디오 변형 6개 vs poisson-graft · 검수 탭 분포 **조명 방향** 의 클래스별 R(합성 vs 실제) — `dataset report` 의 조명 일관성 줄 |
| 6 | 축척 경고가 사라지는가(피치 넣은 뒤) | `run` stderr · 상태바 |
| 7 | `source.tags` 로 제품 필터 | `run --dry-run` `source.tags` 행 |
| 8 · 9 | 라벨 탭 초안 · 최근 레시피 복원 | GUI |
| 10 | `recipe init --write recipes/x.yaml` | 폴더 자동 생성 |

**결정 대기 항목(결과로 확정)**: `gtmask.diff_threshold` 12 · 박스→마스크 기본 `grabcut` vs `otsu` · YOLO 박스 GT `dilate_px` · `structure-aware prefer` edges/uniform · `grabcut work_px` 1024→512 · 저신뢰 경고 강화 임계 50 % · `dent-graft` 조화 0.2. 뒤집히는 게 있으면 0.7.3(프리셋 기본값 변경은 골든 갱신 + 데브로그 사유).

### 2차 결과 (채울 것 — 다른 PC 에서 돌린 뒤 이 절을 PR 로)

> 아래 표의 빈칸을 채우고, 뒤집힌 결정은 "결정" 절에 한 줄씩. 숫자는 명령 출력을 그대로(`bank ls --json` · `run --dry-run` · `review-report.html`).

| 항목 | 결과 | 근거(명령·파일) |
|---|---|---|
| 환경 | `anograft doctor` 버전/OpenCV/Qt: | `doctor.json` |
| 은행 | 소스 n · 저신뢰 n/est · lightR(클래스별): | `bank-ls.json` |
| 1 mask_dir ROI(1024 축소) | ✅/❌ — | 스튜디오 캡처 |
| 2 · 4 annulus | 배치 중심 반경 min~max / 링 r_inner~r_outer: | 사이드카 `placement.center` |
| 3 저신뢰 | 저신뢰 n 중 실제 실패 마스크 n(정밀도 n/n): | `bank preview` 눈 확인 |
| 5 조명 | 실제 R(클래스) / 합성 R(poisson-graft · dent-graft) / 뒤집힘 의심 n/N: | `review-report.html` 조명 줄 |
| 6 축척 | 피치 지정 후 경고 사라짐 ✅/❌: | `run` stderr |
| 7 tags | dry-run `source.tags` 행: | `run --dry-run` |
| 8 · 9 GUI | 초안 로드 ✅/❌ · 최근 레시피 ✅/❌: | GUI |
| 10 init | 폴더 자동 생성 ✅/❌: | `recipe init --write` |
| 배치 가능성 | `roi width` / `fit <class>`(가능·빠듯·불가) vs 실제 skipped n: | `run --dry-run` · manifest |
| 검수 | 채택/반려/미검수 · 반려 사유 상위 3: | `review.csv` |

**결정** (뒤집힌 것만): `diff_threshold` 12 → ? · grabcut/otsu → ? · `dilate_px` → ? · `prefer` → ? · `work_px` → ? · 저신뢰 임계 50 % → ? · `dent-graft` 조화 0.2 → ? · `max_align_deg` 30 → ? · 조명 임계 0.5/0.3 → ?

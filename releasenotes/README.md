# Release notes / 릴리스 노트

최신순. 1 버전 = 1 파일. GitHub Release 본문은 각 파일을 그대로 쓴다.

- [v0.7.6](v0.7.6.md) — 2026-09-16 · flip 모드(none/horizontal/vertical/both, true/false 호환) · 안전한 flip 자동 선택 · 기하 카드 per_class 편집 표 · 변형 카드 ↯ · `bank ls` lightR* · 검수 일괄 반려 · train_smoke 다중 출력
- [v0.7.5](v0.7.5.md) — 2026-09-16 · 조명 의존 판정에 Rayleigh 유의성(n·R² ≥ 2.9) — 소표본 은행에서 스크래치까지 잡히던 오판 수정
- [v0.7.4](v0.7.4.md) — 2026-09-16 · 클래스별 기하·검수 정밀화: `geometry.per_class`(`recipe init --auto-dent` · 기하 카드 버튼) · 검수 필터 '조명 뒤집힘 의심'(`prune --drop-flipped`) · 실제 분포 = 레시피 클래스 · 조명 화살표(bank preview)·라벨 통계 · 배치 로그 진단
- [v0.7.3](v0.7.3.md) — 2026-09-16 · 진단·시작: 조명 방향 분포·일관성 R · `bank ls` lightR·조명 경고 · `run --dry-run` 배치 가능성 · GUI '샘플 데이터' 한 클릭 · `sample --shape ring` · `recipe init --roi` · `max_align_deg`(dent-graft 30)
- [v0.7.2](v0.7.2.md) — 2026-09-16 · 흐름 다듬기: 배치→검수 버튼 · 다음 미검수 · 저신뢰 차례로 다듬기 · 대비 분포 · 스튜디오 µm/px · `doctor` · `bank ls --json`
- [v0.7.1](v0.7.1.md) — 2026-09-16 · 패치: 스튜디오 ROI 캐시(2.7→0.2 s) · 레시피 상대경로 폴백 · `bank merge` · 검수 리포트 HTML · 자동 선택 confidence · CI wheel e2e
- [v0.7.0](v0.7.0.md) — 2026-09-16 · GUI 5탭 전부 실물: 은행 탭(보기·삭제·다듬기) · 검수 탭(채택/반려·실제 vs 합성 분포·정리본) · COCO writer · DTD 텍스처. 실데이터 2차·학습 미검증
- [v0.6.0](v0.6.0.md) — 2026-09-15 · 실데이터 보정: annulus ROI · dent-graft 프리셋 · 박스→마스크 타당성 점수 · 라벨 탭 YOLO 초안/ROI 모드 · source.tags · 축척 경고 · 카드 파라미터 편집기 · GUI 시작 안내. KNOWN-ISSUES 10건. 2차 실데이터 미검증
- [v0.5.0](v0.5.0.md) — 2026-09-15 · GUI 만으로 단독 사용: 라벨 탭(브러시·폴리곤·자동 선택 → 은행 저장) + 배치 탭(레시피 실행·진행률·중지). 실데이터·학습 미검증
- [v0.4.0](v0.4.0.md) — 2026-09-15 · CPU 알고리즘 확장: 은행 없는 self-cut/perlin · 구조 정합 배치 · GrabCut ROI · 카메라 열화 3종 · VisA 어댑터 · mvtec writer. 프리셋 7종. 실데이터·학습 미검증
- [v0.1.0](v0.1.0.md) — 2026-09-14 · 첫 릴리스: 결함 은행(YOLO·마스크 쌍·MVTec AD) · 7단계 파이프라인 + 프리셋 4종 · 재현 가능한 출력(YOLO writer) · GUI 스튜디오 · Windows zip · 샘플 데이터. 실데이터·학습 미검증

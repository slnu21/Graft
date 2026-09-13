# Changelog

[Keep a Changelog](https://keepachangelog.com/) + [SemVer](https://semver.org/).

## [Unreleased]

### Added
- 초기 스캐폴드 — anograft 패키지, 순수 wheel 의존성 4개, bootstrap·lock·CI(win/linux).
- 코어 기반: 값 객체(Context 등 frozen dataclass), 시드 계약(seeds), µm/px 축척(scale), 채널 승격/복원(channels), 한글 경로 안전 이미지 IO(io.imgio).
- 레시피 스키마(pydantic v2, method별 discriminated union, extra=forbid) + 프리셋 4종(poisson-graft · hard-paste · alpha-paste · multiband-graft) + method-aware 병합.
- 스테이지 레지스트리((stage, method) → 클래스, 가용/사유) + 파이프라인 골격(결함 루프 · trace).
- CLI: nograft methods, nograft recipe init|check.

[Unreleased]: https://github.com/slnu21/Graft/commits/main

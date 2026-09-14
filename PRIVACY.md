# 개인정보 처리 방침 / Privacy Policy — Graft (`anograft`)

**Graft는 전 과정 오프라인·로컬입니다.** 네트워크 연결을 열지 않고, 어떤 데이터도 외부로 보내지 않으며, 원격 측정(telemetry)·크래시 리포트·자동 업데이트 확인이 없습니다.

- **읽는 것**: 사용자가 명시한 폴더의 이미지·라벨·마스크·레시피(YAML)·은행(`bank.yaml` + PNG)만.
- **쓰는 것**: 사용자가 지정한 출력 폴더(`output.root`)에 합성 이미지·GT 마스크·사이드카 JSON·`manifest.csv`·`recipe.resolved.yaml`(+ writer 형식 파일). 사이드카에는 **경로·시드·파이프라인 파라미터**가 기록됩니다 — 재현을 위한 것이며, 절대 경로가 들어갈 수 있으니 공유 전 확인하세요.
- **보관하지 않는 것**: 계정·식별자·사용 기록. 설정 파일도 만들지 않습니다(GUI 창 상태 등은 Qt 기본 동작 범위 안에서 OS가 관리).
- **표준 데이터셋·모델**: 내려받지도, 포함하지도 않습니다. `dataset info`는 라이선스·URL 텍스트만 보여 줍니다.

---

**Graft is fully offline and local.** It opens no network connections, sends nothing anywhere, and has no telemetry, crash reporting, or update checks.

- **Reads**: only the images, labels, masks, recipes (YAML) and banks (`bank.yaml` + PNG) in folders you point it at.
- **Writes**: synthetic images, GT masks, sidecar JSON, `manifest.csv`, `recipe.resolved.yaml` (plus writer-format files) into the output folder you choose. Sidecars record **paths, seeds and pipeline parameters** for reproducibility — they may contain absolute paths, so review before sharing.
- **Keeps nothing else**: no accounts, identifiers or usage logs; no config files are created.
- **Datasets & models**: never downloaded or bundled; `dataset info` only prints license/URL text.

문의 / Contact: https://github.com/slnu21/Graft/issues

# Third-Party Notices / 제3자 고지

Graft(`anograft`)는 MIT 라이선스입니다(`LICENSE`). 아래는 함께 배포되거나(zip) pip 설치 시 함께 설치되는 구성 요소와 그 라이선스입니다. 각 구성 요소의 전문 라이선스는 해당 프로젝트 저장소 및 zip의 `_internal/<패키지>-<버전>.dist-info/` 안의 LICENSE 파일을 따릅니다.

Graft (`anograft`) is MIT-licensed (`LICENSE`). The components below are bundled in the Windows zip and/or installed alongside via pip. Full license texts live in each project and in `_internal/<package>-<version>.dist-info/` inside the zip.

## 런타임 의존성 / Runtime dependencies

| 구성 요소 Component | 라이선스 License | 비고 Notes |
|---|---|---|
| Python | PSF License | zip에 인터프리터(`python3xx.dll`)가 포함됨 / interpreter bundled in the zip |
| NumPy | BSD-3-Clause | `numpy.libs/`의 OpenBLAS(BSD-3-Clause) 포함 / includes OpenBLAS |
| OpenCV (`opencv-python-headless`) | Apache-2.0 | 비디오 코덱(FFmpeg) DLL은 zip에서 제외 / FFmpeg DLL is not shipped |
| pydantic · pydantic-core | MIT | |
| PyYAML | MIT | |
| PySide6 · shiboken6 (Qt for Python) | **LGPL-3.0** (또는 GPL/상용) | GUI(`anograft-gui.exe`)만 사용. **동적 링크** — Qt DLL은 `_internal/PySide6/`에 그대로 두어 사용자가 교체할 수 있습니다. Qt는 The Qt Company Ltd.의 상표. / GUI only; dynamically linked, DLLs replaceable by the user. |
| Qt 6 (PySide6에 포함) | LGPL-3.0 | 위와 동일 / same as above |
| Microsoft Visual C++ Runtime (`VCRUNTIME140*.dll`, `MSVCP140*.dll`) | Microsoft 재배포 허용 라이선스 / Microsoft redistributable license | PySide6·OpenCV wheel에 동봉 / shipped with the wheels |

## 빌드 도구 (배포물에 코드가 포함되지 않음) / Build tooling (no code shipped)

| 구성 요소 | 라이선스 | 비고 |
|---|---|---|
| PyInstaller | GPL-2.0 with **bootloader exception** | 부트로더가 exe에 결합되지만 예외 조항에 따라 Graft의 MIT 라이선스에 영향 없음 / bootloader exception keeps Graft MIT |
| pytest · ruff | MIT | 개발 전용 / dev only |
| Node.js · npm | MIT (Node는 MIT + 동봉 구성 요소 각자의 라이선스) | 웹 UI 빌드 전용 — **런타임에는 필요 없습니다**(`pip install anograft` 불변) / builds the web UI only |

## 웹 UI (npm) / Web UI (npm)

아래 절은 `tools/collect_npm_notices.mjs` 가 `web/package-lock.json` 에서 만들어 넣습니다 —
의존성이 바뀌면 CI 가 `--check` 로 낡음을 잡습니다. 갱신은 `cd web && npm run notices -- --write`.

<!-- npm:begin — tools/collect_npm_notices.mjs 가 만든다. 손으로 고치지 말 것 -->

### 웹 UI 번들에 포함 / Bundled in the web UI

`anograft serve` 가 내보내는 화면 파일에 코드가 들어갑니다. 폰트·아이콘은 받지 않습니다(시스템 폰트 + 인라인 SVG).

| 구성 요소 Component | 버전 Version | 라이선스 License |
|---|---|---|
| react | 19.3.0 | MIT |
| react-dom | 19.3.0 | MIT |
| scheduler | 0.28.0 | MIT |

### 웹 빌드 도구 (배포물에 코드가 포함되지 않음) / Web build tooling (no code shipped)

Vite·TypeScript·Babel·vitest 등 **빌드 때만** 쓰는 것들입니다 — 총 150개 · MIT 139 · ISC 6 · Apache-2.0 3 · BSD-3-Clause 1 · CC-BY-4.0 1.
플랫폼별 선택 의존성까지 포함한 목록이라 실제로 설치되는 것은 이보다 적습니다.
전체 목록은 `node tools/collect_npm_notices.mjs` 로 뽑을 수 있고, 버전은 `web/package-lock.json` 이 고정합니다.

<!-- npm:end -->

## 데이터셋·모델 (재배포하지 않음) / Datasets & models (never redistributed)

Graft는 어떤 데이터셋·모델 가중치도 포함하거나 내려받지 않습니다. 사용자가 로컬에 둔 사본을 읽기만 합니다.

- **MVTec AD** — CC BY-NC-SA 4.0 (비상업). `anograft dataset info mvtec-ad`가 라이선스·URL을 안내합니다.
- 향후 어댑터(VisA 등)·플러그인(SAM·확산 가중치)도 같은 원칙 — 다운로드 링크와 동의만 중개합니다.

Graft ships and downloads no dataset or model weights; importers only read local copies you obtained yourself under their own terms.

# TESTING — 받아서 확인할 것 (v0.8.0)

> 자율 세션(2026-09-16 저녁 ~ 09-17 새벽)이 만든 것을 **다른 PC 에서 그대로 따라 하며** 확인하는 절차. 각 항목은 *무엇을 → 기대 결과 → 어긋나면*. 실데이터가 있으면 `KNOWN-ISSUES.md` "2차 적용 절차"를 먼저, 없으면 아래 공개 데이터로.
> English summary at the end.

## 0. 설치 · 환경

```powershell
# A) 파이썬 없이 — 릴리스 zip 을 풀고
.\anograft.exe doctor                      # 버전 0.8.0 · gui ok 인지
# B) 소스 —
.\bootstrap.ps1 -Gui                       # = venv + pip install -e ".[dev,gui]"
anograft doctor --json > doctor.json       # 문제 보고 첫 줄
pytest -q                                  # 전부 통과(≈ 660) — 실패하면 그 파일 이름을 보고에
```

## 1. 공개 데이터 받기 (실데이터 없을 때, 4.1 GB 전부는 불필요)

```powershell
python tools/fetch_public_datasets.py metal_nut --import        # ≈ 170 MB → samples/mvtec/metal_nut · bank/metal_nut
python tools/fetch_public_datasets.py magnetic-tile --import    # ≈ 110 MB (git clone) → pairs.csv · normals.txt · bank/magnetic-tile
python tools/fetch_public_datasets.py screw --import            # (선택) 흑백 경로
```

- 기대: `bank ls bank/metal_nut` → bent 35 · color 30 · flip 23(**0.52\***) · scratch 44, 저신뢰 0.
- 라이선스: MVTec CC BY-NC-SA(로컬 개발용) — `samples/` 는 gitignore, 재배포 금지.

## 2. 이번 릴리스에서 새로 확인할 것

| # | 무엇을 | 기대 결과 | 어긋나면 |
|---|---|---|---|
| 1 | `anograft run recipes/public-metal-nut-dent.yaml --dry-run` | `fit bent … 짧은 변 96 · 긴 변 153px(× scale 1.1) vs 폭 100px → 빠듯 (짧은 변이 폭의 80% 초과)` 처럼 **짧은 변·긴 변·근거**가 한 줄에. `source:` 경고 **없음**(레시피가 flip 을 `classes` 로 뺐음) | 행 형식·판정 캡처 |
| 2 | 같은 레시피에서 `classes: [bent, color, scratch]` 줄을 지우고 dry-run | `경고: source: 클래스 flip(대상 짧은 변의 98%) … source.classes 로 제외 검토` | 경고 문구 캡처 |
| 3 | `anograft run recipes/public-metal-nut-dent.yaml --workers 4 --report` | `완료: ok 38 · skipped 2`(±2) · `out/public-metal-nut-dent/review-report.html` 이 열림 | manifest.csv 의 skipped reason |
| 4 | MT 레시피의 `targets:` 를 `samples/magnetic-tile/MT_Free/Imgs` 로 바꿔 dry-run | `경고: targets: 같은 이름의 파일이 확장자만 다르게 952쌍 … .txt 목록으로` · targets 1904장. `normals.txt` 로 되돌리면 경고 없음·952장 | |
| 5 | `anograft run recipes/public-mt-structure.yaml --workers 4` 뒤 두 출력 병합: `anograft dataset merge out/public-metal-nut-dent out/public-mt-structure --out out/merged` | **거부**: `병합 실패: 클래스 이름(data.yaml names)이 다릅니다` (의도된 동작 — 다른 은행) | |
| 6 | 같은 레시피를 `--seed 7 --out out/mn2` 로 한 번 더 돌린 뒤 `dataset merge out/public-metal-nut-dent out/mn2 --out out/mn-merged` | `병합 → … 합성 ≈76 · 정상 440 · 파일 ≈2000`, `merge.json`, `images/d0_…`·`d1_…`, manifest index 0..N 연속. `dataset report out/mn-merged` 동작 | manifest·merge.json |
| 7 | GUI `python -m anograft.gui` → 검수 탭 → `out/public-metal-nut-dent` 열기 → 분포 `조명 방향` → **클래스 콤보**에서 `bent` | 제목 `… · 클래스 bent · R 합성 x (n) / 실제 0.42 (n 35)`. 면적/긴 변에선 콤보 비활성 | 스크린샷 |
| 8 | 검수 탭 `실측 CSV…` 로 아래 파일 → 분포 `면적` | 제목 `… · 실제 = 실측 real.csv`, 실제 계열 3개. `은행으로` 누르면 원래대로 | |
| 9 | `anograft dataset report out/public-metal-nut-dent --real-csv real.csv` | stderr `실측 CSV real.csv: 3행 (area)`, HTML 머리 `실제 = 실측 real.csv` | |
| 10 | (YOLO 라벨이 있는 데이터에서) `bank import-yolo … --mask-from hybrid --out bank/h` 와 기본(grabcut) 두 번 → `bank ls`·`bank preview` 비교 | hybrid 은행은 origins 에 `ellipse` 비율↑ · `lowconf` ↓ · preview 에서 엉뚱한 조각 대신 타원. **기본값은 그대로**(재현성) | 두 preview PNG |
| 11 | `python tools/bench_mask_from_box.py samples/mvtec/metal_nut --limit 5` | 표에 `grabcut otsu ellipse rect hybrid` 5열 · 30 인스턴스 ≈ 15 s | |
| 12 | 재현성: `run … --workers 0` 과 `--workers 2` 를 같은 `output.root` 로 | 이미지·마스크·사이드카 바이트 동일(`test_cli` 가 자동화 — 손으로는 `Get-FileHash` 몇 개) | |

`real.csv` 예시(항목 8·9):

```
class,area,length
bent,8000,140
color,6500,150
scratch,10000,220
```

## 3. 결정 대기 항목에 근거를 보태려면

- `KNOWN-ISSUES.md` "결정 근거 — 공개 데이터 벤치" 절의 표를 **실데이터**로 재현: `python tools/bench_mask_from_box.py <GT 마스크가 있는 폴더 또는 pairs.csv> --out bench.md`. GT 마스크가 없으면(YOLO 박스만) 라벨 탭에서 10~20장만 마스크를 그려 `pairs.csv` 로.
- 조명 임계(0.5/0.3·Rayleigh 2.9·링 2 px·뒤집힘 90°): `bank ls --json` 의 `light_r`·`light_n`·`directional` 와 검수 탭 **클래스 콤보 조명 방향** 히스토그램을 결정 표 "5 조명" 행에.

## 4. 보고 형식

`doctor.json` + 위 표의 # 와 실제 출력(명령 그대로 복사) + 스크린샷(검수 탭·스튜디오 카드) + 뒤집힌 결정이 있으면 `KNOWN-ISSUES.md` "2차 결과" 표에 한 줄. 이슈는 `docs/TASKS.md` 다음 세션 절 또는 GitHub Issue.

---

## English (summary)

1. **Install**: unzip the release and run `anograft.exe doctor` (expect 0.8.0, `gui ok`), or `.\bootstrap.ps1 -Gui` + `pytest -q` (all green).
2. **Data**: `python tools/fetch_public_datasets.py metal_nut magnetic-tile --import` (MVTec is CC BY-NC-SA — local dev only).
3. **Check** (table above): dry-run `fit` rows now show short/long side + reason; `source:` warning for whole-part classes; `targets:` warning when mask PNGs sit next to images; `dataset merge` (refuses different class lists; merges same-bank outputs with `d<k>_` prefixes and re-indexed manifest); review tab **per-class combo** and **measured CSV** button; `dataset report --real-csv`; `--mask-from hybrid` (opt-in; default unchanged); `tools/bench_mask_from_box.py`.
4. **Report**: `doctor.json`, the exact command output, screenshots, and any reversed decision in `KNOWN-ISSUES.md`.

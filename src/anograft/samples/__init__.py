"""샘플 데이터 생성기 — 보유 데이터가 없을 때 ``sample → import-yolo → run → preview``를 끝까지 돌려 보기 위한 합성 세트.

패키지 안에 두는 이유(2026-09-14, #759): 배포 정본이 PyInstaller zip(파이썬 없는 PC)이라 ``tools/`` 스크립트로는
"5분 시작"을 못 한다 → ``anograft sample --out DIR`` 서브커맨드. ``tools/make_sample_yolo.py``는 이 패키지를 부르는 셔틀.
"""

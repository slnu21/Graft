"""셔틀 — 본체는 ``anograft.samples.yolo``(2026-09-14 패키지로 이동: zip 배포에서 ``anograft sample``로 쓰기 위해).

python tools/make_sample_yolo.py --out samples/metal --n-normal 12 --n-defect 10 --seed 7
anograft sample --out samples/metal            # 같은 인자
"""

from __future__ import annotations

from anograft.samples.yolo import main

if __name__ == "__main__":
    raise SystemExit(main())

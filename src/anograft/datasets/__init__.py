"""표준 산업 데이터셋 어댑터 — 파일 1개 = 데이터셋 1개 (mvtec_ad · visa …). 내려받지 않고 읽기만.

이 패키지를 import하면 어댑터가 ``REGISTRY``에 등록된다(``anograft dataset info`` · ``bank import-dataset``).
"""

from anograft.datasets import mvtec_ad as mvtec_ad  # mvtec-ad 등록
from anograft.datasets import visa as visa  # visa 등록
from anograft.datasets.base import (
    REGISTRY,
    DatasetAdapter,
    DatasetError,
    DatasetInfo,
    adapter_names,
    get_adapter,
    info_lines,
)

__all__ = [
    "REGISTRY",
    "DatasetAdapter",
    "DatasetError",
    "DatasetInfo",
    "adapter_names",
    "get_adapter",
    "info_lines",
    "mvtec_ad",
    "visa",
]

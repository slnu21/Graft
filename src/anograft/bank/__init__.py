"""결함 은행 — 디스크 포맷(설계 §3.1) 로더와 임포터.

- ``bank.Bank``: 로드·클래스 순서(=id)·클래스별 소스·지문. 파일 IO는 ``io/imgio``만 쓴다.
- ``mask_from_box``: YOLO 박스 → 픽셀 마스크 추정 (순수 함수).
- ``importers/``: yolo(·pairs·dataset)가 레코드를 만들고 ``common.BankWriter`` 하나가 은행을 쓴다.
"""

from anograft.bank.bank import BANK_FILE, Bank, ClassSummary

__all__ = ["BANK_FILE", "Bank", "ClassSummary"]

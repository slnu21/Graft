"""1단계 소스 선택 — 방법이 여럿이라 패키지. import 한 줄이 곧 레지스트리 등록.

- ``bank``: 결함 은행에서 클래스 → 소스 (v0.1)
- ``self-cut``: 대상 자신에서 사각/스카 패치 (CutPaste, v0.4) — 은행 불필요
- ``perlin-texture``: 펄린 노이즈 마스크 + 텍스처 (DRAEM, v0.4) — 은행 불필요

세 방법 모두 출력은 ``ctx.source: DefectSource``(image HxWx3 · mask HxW 0/255 · cls) 하나 — 이후 스테이지는 소스가 어디서
왔는지 모른다. 실패는 ``source=None`` + 경고(fail-soft), 로그 ``source = {method, ..., skipped?}``.
"""

from anograft.core.stages.source import bank as bank  # bank
from anograft.core.stages.source import perlin as perlin  # perlin-texture
from anograft.core.stages.source import selfcut as selfcut  # self-cut
from anograft.core.stages.source.bank import BankSource as BankSource
from anograft.core.stages.source.perlin import PerlinSource as PerlinSource
from anograft.core.stages.source.selfcut import SelfCutSource as SelfCutSource

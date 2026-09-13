"""4단계 blend — 방법이 여럿인 스테이지는 패키지. import되는 순간 등록된다.

v0.1: ``paste`` · ``alpha`` · ``poisson``(이 단위) · ``multiband``(stages-more-methods). 공통 창 클리핑·페더는 ``common.py``.
"""

from anograft.core.stages.blend import alpha as alpha  # alpha
from anograft.core.stages.blend import paste as paste  # paste
from anograft.core.stages.blend import poisson as poisson  # poisson (+alpha 폴백)

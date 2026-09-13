"""스테이지 구현 모듈. import되는 순간 ``registry.register``가 일어난다.

작업 단위별로 채운다: stages-geometry-placement(geometry·roi·placement) → stages-poisson-gt(blend/·harmonize·degrade·gtmask)
→ first-run(source) → stages-more-methods(multiband·reinhard·histmatch). 여기 import 한 줄이 곧 "등록" — 등록되면
``anograft methods``에서 "ok".
"""

from anograft.core.stages import blend as blend  # paste · alpha · poisson
from anograft.core.stages import degrade as degrade  # none · camera
from anograft.core.stages import geometry as geometry  # affine
from anograft.core.stages import gtmask as gtmask  # source · diff · union
from anograft.core.stages import harmonize as harmonize  # none · stats
from anograft.core.stages import placement as placement  # sampled
from anograft.core.stages import roi as roi  # otsu · none · mask_dir
from anograft.core.stages import source as source  # bank

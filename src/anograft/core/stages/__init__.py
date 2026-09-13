"""스테이지 구현 모듈. import되는 순간 ``registry.register``가 일어난다.

작업 단위별로 채운다: stages-geometry-placement(geometry·roi·placement) → stages-blend-harmonize(blend/·harmonize)
→ stages-degrade-gt(degrade·gtmask) → bank-import(source). 여기 import 한 줄이 곧 "등록".
"""

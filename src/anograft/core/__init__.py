"""anograft.core — 순수 합성 코어. Qt·경로·파일 IO를 모른다. numpy/opencv만 안다.

불변식:
1. Qt를 import하지 않는다.
2. 모든 무작위성은 ``Context.rng``(``seeds.py`` 파생)를 통과한다.
3. 선택 의존성은 지연 import. 없으면 해당 method를 비활성 + 이유 노출(fail-soft).
4. 스테이지는 ``Context``를 바꾸지 않고 ``dataclasses.replace()``로 새 것을 돌려준다.
"""

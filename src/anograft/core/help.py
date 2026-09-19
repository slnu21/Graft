"""파라미터·method·스테이지 도움말 — **한 원천**(v0.9 사용성 ②). Qt·파일 IO 없음.

카드 폼(라벨·툴팁·고급 접기) · ``anograft explain`` · ``PARAMS.md`` 생성 · 탭 툴팁이 전부 여기를 읽는다. 문안 규칙(Blender 툴팁 HIG +
Toss UX writing, ``docs/ux/2026-09-19-usability-review.md`` §4.1):

- ``label`` = 쉬운 우리말 한 단어~구(YAML 키를 되풀이하지 않는다). 단위는 ``unit`` 에 따로(px · ° · 배 · gray · 회 · 개).
- ``desc`` = 1~2줄: 무엇 → 올리면/내리면(또는 켜면). 기본값은 쓰지 않는다(바뀐다 — 되돌리기 버튼·``explain`` 이 보여 준다). ≤ 200자.
- ``en`` = 영어 한 줄(툴팁 둘째 줄·PARAMS.md 영어 열).
- ``advanced`` = 카드에서 "고급 옵션"으로 접히는 필드(프리셋을 고른 뒤 보통 만지지 않는 것).

키 = ``"<설정 클래스 이름>.<필드 점 경로>"``(``AffineGeometryConfig.scale`` · ``SelfCutSourceConfig.jitter.brightness``). 상속 필드는
``_PlacementBase.margin_px`` 처럼 정의한 클래스 이름으로 두고 ``field_help()`` 가 MRO 를 따라 찾는다. **모든 필드·method·프리셋에 항목이
있어야 한다** — ``tests/test_param_help.py`` 가 스키마·레지스트리와 대조한다(빠지면 CI 실패).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FieldHelp:
    label: str
    desc: str
    unit: str = ""
    en: str = ""
    advanced: bool = False


@dataclass(frozen=True)
class MethodHelp:
    label: str  # 콤보 표시명(한국어)
    summary: str  # 한 줄 — 무엇을 어떻게
    when: str = ""  # 이럴 때 / 피할 때
    en: str = ""


@dataclass(frozen=True)
class StageHelp:
    label: str  # 카드 제목
    desc: str  # 한 줄
    en: str  # 영어 이름


STAGE_HELP: dict[str, StageHelp] = {
    "source": StageHelp(
        "결함 고르기",
        "어떤 결함 조각을 쓸지 — 보관함 · 정상 부위 잘라 붙이기 · 노이즈 텍스처",
        "Source",
    ),
    "geometry": StageHelp("크기·회전", "조각의 크기·회전·뒤집기·휘어짐", "Geometry"),
    "placement": StageHelp(
        "위치 정하기", "바탕 이미지의 어디에 놓을지(붙일 수 있는 영역 안에서)", "Placement"
    ),
    "roi": StageHelp(
        "붙일 수 있는 영역", "바탕 이미지에서 결함을 놓아도 되는 영역(배경·리세스 제외)", "ROI"
    ),
    "blend": StageHelp(
        "붙이기",
        "조각을 바탕에 어떻게 붙일지(그대로 · 가장자리 부드럽게 · Poisson · 다중 대역)",
        "Blend",
    ),
    "harmonize": StageHelp("색·밝기 맞추기", "붙인 결함의 색·밝기를 주변에 맞추기", "Harmonize"),
    "degrade": StageHelp("카메라 효과", "노이즈·흐림·JPEG 등 카메라 느낌 입히기", "Degrade"),
    "gtmask": StageHelp("정답 영역", "학습용 정답 영역(마스크)을 무엇으로 삼을지", "GT mask"),
}

METHOD_HELP: dict[tuple[str, str], MethodHelp] = {
    ("source", "bank"): MethodHelp(
        "보관함",
        "결함 보관함의 조각을 클래스 확률대로 뽑습니다",
        "라벨링한 결함이 있을 때(기본)",
        "Bank",
    ),
    ("source", "self-cut"): MethodHelp(
        "정상 부위 잘라 붙이기",
        "바탕 이미지 자신에서 네모/띠 조각을 잘라 결함처럼 붙입니다(CutPaste)",
        "라벨링한 결함이 하나도 없을 때. 보관함 불필요",
        "Self-cut (CutPaste)",
    ),
    ("source", "perlin-texture"): MethodHelp(
        "노이즈 텍스처",
        "펄린 노이즈로 불규칙한 마스크를 만들고 텍스처를 채웁니다(DRAEM)",
        "불규칙한 이상 영역을 흉내낼 때. 보관함 불필요",
        "Perlin texture (DRAEM)",
    ),
    ("geometry", "affine"): MethodHelp(
        "크기·회전·뒤집기",
        "조각을 배율·각도·뒤집기로 바꾸고, 잔물결·휘어짐을 더할 수 있습니다",
        "",
        "Affine",
    ),
    ("roi", "otsu"): MethodHelp(
        "밝기로 자동 분리",
        "밝기 임계(Otsu)로 물체와 배경을 가르고 물체 안만 허용합니다",
        "물체가 배경과 밝기로 뚜렷이 갈릴 때(기본)",
        "Otsu",
    ),
    ("roi", "none"): MethodHelp(
        "제한 없음", "이미지 어디든 놓습니다", "배경이 없는 타일·표면 사진", "None"
    ),
    ("roi", "mask_dir"): MethodHelp(
        "마스크 폴더",
        "바탕 이미지와 같은 이름의 마스크 PNG(흰색 = 허용)를 읽습니다",
        "결함 표시 탭에서 영역을 직접 칠했을 때",
        "Mask folder",
    ),
    ("roi", "grabcut"): MethodHelp(
        "전경 자동 분리",
        "GrabCut 으로 전경(물체)을 찾아 그 안만 허용합니다. 느리지만 밝기 분리보다 정확",
        "밝기만으로는 물체/배경이 안 갈릴 때",
        "GrabCut",
    ),
    ("roi", "annulus"): MethodHelp(
        "링(도넛) 영역",
        "원형 부품의 중심·반지름을 찾아 안쪽~바깥 반지름 사이 링만 허용합니다",
        "원형 부품의 가공 링 면(리세스 제외)",
        "Annulus",
    ),
    ("placement", "sampled"): MethodHelp(
        "무작위 자리",
        "허용 영역 안에서 자리를 무작위로 뽑습니다(고르게 · 가장자리 · 가운데)",
        "기본",
        "Sampled",
    ),
    ("placement", "structure-aware"): MethodHelp(
        "표면 결을 따라",
        "결·에지가 있는 곳을 선호하고 조각을 결 방향에 맞춰 돌립니다",
        "스크래치가 결을 따르고 칩이 모서리에 생기는 부품",
        "Structure-aware",
    ),
    ("blend", "paste"): MethodHelp(
        "그대로 붙이기",
        "마스크 안 픽셀을 그대로 덮어씁니다. 가장 거친 대조군",
        "학습 실험의 대조군 · 노출 보정(relative)과 함께",
        "Paste",
    ),
    ("blend", "alpha"): MethodHelp(
        "가장자리 부드럽게",
        "마스크 경계를 페더로 서서히 섞고, 결함마다 불투명도를 줄 수 있습니다",
        "경계만 감추면 되는 텍스처성 결함",
        "Alpha",
    ),
    ("blend", "poisson"): MethodHelp(
        "경계 자연스럽게(Poisson)",
        "경계의 그래디언트를 맞춰 이음새 없이 붙입니다. 결함 톤이 바탕에 끌려가 옅어질 수 있습니다",
        "얼룩·스크래치 등 대부분(기본). 대비가 신호인 구멍엔 relative-paste",
        "Poisson",
    ),
    ("blend", "multiband"): MethodHelp(
        "다중 대역",
        "라플라시안 피라미드로 굵은 톤은 넓게, 잔결은 좁게 섞습니다",
        "큰 얼룩처럼 톤 전이가 넓은 결함",
        "Multiband",
    ),
    ("harmonize", "none"): MethodHelp(
        "안 함", "색·밝기를 맞추지 않습니다", "hard-paste 대조군", "None"
    ),
    ("harmonize", "stats"): MethodHelp(
        "평균·분산 맞춤",
        "결함 안쪽 L 채널의 평균·표준편차를 주변 링에 맞춥니다. 세기만큼 대비가 옅어집니다",
        "기본(세기 0.3)",
        "Stats",
    ),
    ("harmonize", "reinhard"): MethodHelp(
        "Reinhard",
        "Lab 세 채널의 평균·표준편차를 주변에 맞춥니다(색까지)",
        "색조 차이가 큰 조각",
        "Reinhard",
    ),
    ("harmonize", "histmatch"): MethodHelp(
        "히스토그램 맞춤",
        "결함 안쪽 밝기 분포를 주변 링 분포에 맞춥니다",
        "밝기 분포 모양까지 맞춰야 할 때",
        "Histogram match",
    ),
    ("harmonize", "relative"): MethodHelp(
        "노출만 맞춤",
        "조각 주변과 바탕 주변의 밝기 차만큼 결함을 옮깁니다 — 결함의 상대 대비는 그대로",
        "대비가 곧 신호인 결함(구멍·핏). paste 뒤에만",
        "Relative (exposure offset)",
    ),
    ("degrade", "none"): MethodHelp("안 함", "카메라 효과를 넣지 않습니다", "", "None"),
    ("degrade", "camera"): MethodHelp(
        "카메라 재현",
        "노이즈 · 흐림 · JPEG · 움직임 흐림 · 가장자리 어둡게 · 감마를 무작위 범위로",
        "기본",
        "Camera",
    ),
    ("gtmask", "source"): MethodHelp(
        "원본 마스크",
        "조각의 마스크를 그대로 정답으로 씁니다",
        "paste 처럼 마스크 밖이 안 바뀌는 붙이기",
        "Source mask",
    ),
    ("gtmask", "diff"): MethodHelp(
        "바뀐 픽셀",
        "바탕과 임계 이상 달라진 픽셀만 정답으로 삼습니다",
        "Poisson 이 지운 저대비 부분을 라벨에서 빼고 싶을 때",
        "Diff",
    ),
    ("gtmask", "union"): MethodHelp(
        "합집합",
        "원본 마스크와 바뀐 픽셀을 합치고 팽창합니다 — 붙이기 오차를 흡수(기본)",
        "기본",
        "Union",
    ),
}

# 필드 도움말 — 키 "<Config 클래스>.<점 경로>"
FIELD_HELP: dict[str, FieldHelp] = {
    # ---- inputs / output
    "Inputs.bank": FieldHelp(
        "결함 보관함",
        "결함 조각을 모아 둔 폴더(bank.yaml). 비우면 보관함 없이 도는 방법(정상 부위 잘라 붙이기 · 노이즈 텍스처)만 쓸 수 있습니다",
        "",
        "Bank folder",
    ),
    "Inputs.targets": FieldHelp(
        "바탕 이미지",
        "결함을 붙일 정상 이미지 폴더, 또는 경로 목록 .txt",
        "",
        "Target (normal) images",
    ),
    "Inputs.um_per_px": FieldHelp(
        "바탕 픽셀 크기",
        "바탕 이미지 1픽셀이 몇 µm 인지. 조각에도 값이 있으면 실제 크기로 자동 축척합니다. 끄면 픽셀 크기 그대로",
        "µm/px",
        "Pixel pitch of targets",
    ),
    "Output.root": FieldHelp(
        "출력 폴더", "이미지·마스크·메타·manifest.csv 가 생기는 폴더", "", "Output root"
    ),
    "Output.count": FieldHelp(
        "만들 장수",
        "합성 이미지 수(정상 이미지는 따로 함께 나갑니다)",
        "장",
        "Number of synthetic images",
    ),
    "Output.class_ratio": FieldHelp(
        "클래스 비율",
        "클래스별 추첨 확률. 비우면 보관함 클래스를 고르게",
        "",
        "Class draw ratio",
        True,
    ),
    "Output.defects_per_image": FieldHelp(
        "이미지당 결함 수", "한 이미지에 붙일 결함 수 범위", "개", "Defects per image"
    ),
    "Output.include_normals": FieldHelp(
        "정상 이미지 포함",
        "켜면 바탕 이미지도 정상(결함 없음)으로 함께 내보냅니다",
        "",
        "Include normal images",
    ),
    "Output.copy_mode": FieldHelp(
        "정상 이미지 복사 방식",
        "copy = 복사 · hardlink = 하드링크(같은 드라이브, 용량 절약)",
        "",
        "Copy mode for normals",
        True,
    ),
    "Output.writer": FieldHelp(
        "학습 형식",
        "학습 프레임워크가 읽는 형식. 이미지·마스크·메타는 어느 형식이든 항상 함께",
        "",
        "Writer",
    ),
    "YoloWriterConfig.format": FieldHelp("형식", "yolo = labels/*.txt + data.yaml", "", "Format"),
    "YoloWriterConfig.seg": FieldHelp(
        "다각형 라벨", "켜면 상자 대신 다각형(YOLO-seg)을 씁니다", "", "Polygon (YOLO-seg) labels"
    ),
    "YoloWriterConfig.names_from": FieldHelp(
        "클래스 이름 출처", "bank = 보관함의 클래스 순서(id)를 그대로", "", "Class names from", True
    ),
    "PairsWriterConfig.format": FieldHelp(
        "형식", "pairs = 이미지 + 마스크 + 메타만(정본)", "", "Format"
    ),
    "MvtecWriterConfig.format": FieldHelp(
        "형식", "mvtec = anomalib 이 읽는 <category>/{train,test,ground_truth}", "", "Format"
    ),
    "MvtecWriterConfig.category": FieldHelp(
        "카테고리 이름", "mvtec 폴더 구조의 카테고리 이름", "", "Category name"
    ),
    "MvtecWriterConfig.test_normal_ratio": FieldHelp(
        "test 정상 비율",
        "정상 이미지 중 test/good 으로 보낼 비율(나머지는 train/good)",
        "",
        "Ratio of normals in test",
        True,
    ),
    "MvtecWriterConfig.layout_dir": FieldHelp(
        "레이아웃 폴더",
        "출력 폴더 아래 mvtec 구조를 만들 하위 폴더 이름",
        "",
        "Layout subfolder",
        True,
    ),
    "CocoWriterConfig.format": FieldHelp(
        "형식", "coco = annotations.json(폴리곤/RLE · bbox · area)", "", "Format"
    ),
    "CocoWriterConfig.description": FieldHelp(
        "설명", "annotations.json 의 info.description", "", "Dataset description", True
    ),
    "CocoWriterConfig.supercategory": FieldHelp(
        "상위 카테고리", "모든 클래스의 supercategory 이름", "", "Supercategory", True
    ),
    "CocoWriterConfig.segmentation": FieldHelp(
        "세그멘테이션 표현",
        "polygon = 다각형 · rle = 비압축 RLE(조각·구멍 무손실)",
        "",
        "Segmentation encoding",
        True,
    ),
    # ---- source: bank
    "BankSourceConfig.classes": FieldHelp(
        "쓸 클래스",
        "보관함의 이 클래스만 씁니다. 비우면 전부(또는 출력 클래스 비율의 키)",
        "",
        "Classes to use",
    ),
    "TagFilter.include": FieldHelp(
        "포함 태그",
        "이 태그가 있는 조각만 씁니다(클래스 목록·확률·id 는 그대로)",
        "",
        "Include tags",
        True,
    ),
    "TagFilter.exclude": FieldHelp(
        "제외 태그", "이 태그가 있는 조각은 뺍니다", "", "Exclude tags", True
    ),
    "BankSourceConfig.min_sources_warn": FieldHelp(
        "조각 부족 경고 기준",
        "클래스의 조각 수가 이보다 적으면 경고합니다. 적은 조각을 반복하면 모델이 그 한 장을 외웁니다",
        "개",
        "Warn below this many sources",
        True,
    ),
    "BankSourceConfig.redraw_on_empty": FieldHelp(
        "빈 마스크 재추첨",
        "크기 변환 뒤 마스크가 사라진 아주 작은 조각을 이만큼 다시 뽑습니다. 실패했을 때만 난수를 더 씁니다",
        "회",
        "Redraws when mask vanishes",
        True,
    ),
    "BankSourceConfig.single_class_per_image": FieldHelp(
        "이미지당 한 클래스",
        "켜면 한 이미지의 결함이 모두 첫 결함의 클래스가 됩니다(MVTec 형식용)",
        "",
        "One class per image",
        True,
    ),
    # ---- source: self-cut
    "SelfCutSourceConfig.cls": FieldHelp("클래스 이름", "출력 라벨에 쓸 이름", "", "Class name"),
    "SelfCutSourceConfig.shape": FieldHelp(
        "조각 모양",
        "rect = 네모 조각 · scar = 가는 긴 띠 · mixed = 결함마다 반반",
        "",
        "Patch shape",
    ),
    "SelfCutSourceConfig.area_ratio": FieldHelp(
        "조각 면적 비율",
        "네모 조각의 면적 / 이미지 면적 범위. 올리면 큰 결함",
        "",
        "Patch area ratio",
    ),
    "SelfCutSourceConfig.aspect": FieldHelp(
        "가로세로 비", "네모 조각의 가로/세로 범위(로그 균등)", "", "Aspect ratio", True
    ),
    "SelfCutSourceConfig.scar_width_px": FieldHelp(
        "띠 굵기", "띠 모양의 굵기 범위", "px", "Scar width", True
    ),
    "SelfCutSourceConfig.scar_length_px": FieldHelp(
        "띠 길이", "띠 모양의 길이 범위", "px", "Scar length", True
    ),
    "SelfCutSourceConfig.margin_px": FieldHelp(
        "잘라낼 여유",
        "조각 둘레를 이만큼 더 잘라 Poisson 팽창·페더가 잘리지 않게 합니다",
        "px",
        "Crop margin",
        True,
    ),
    "SelfCutSourceConfig.max_tries": FieldHelp(
        "자리 찾기 시도",
        "붙일 수 있는 영역 안에서 잘라낼 자리를 찾는 최대 횟수",
        "회",
        "Max tries",
        True,
    ),
    "ColorJitterConfig.brightness": FieldHelp(
        "밝기 흔들기",
        "잘라낸 조각의 밝기를 이만큼 무작위로 바꿔 배경과 구분되게 합니다. 0 = 그대로",
        "",
        "Brightness jitter",
        True,
    ),
    "ColorJitterConfig.contrast": FieldHelp(
        "대비 흔들기", "조각의 대비를 이만큼 무작위로. 0 = 그대로", "", "Contrast jitter", True
    ),
    "ColorJitterConfig.saturation": FieldHelp(
        "채도 흔들기", "조각의 채도를 이만큼 무작위로. 0 = 그대로", "", "Saturation jitter", True
    ),
    "ColorJitterConfig.hue": FieldHelp(
        "색상 흔들기", "조각의 색상을 이만큼 무작위로 돌립니다. 0 = 그대로", "", "Hue jitter", True
    ),
    # ---- source: perlin
    "PerlinSourceConfig.cls": FieldHelp("클래스 이름", "출력 라벨에 쓸 이름", "", "Class name"),
    "PerlinSourceConfig.texture": FieldHelp(
        "텍스처 출처",
        "self = 바탕 이미지의 다른 부분 · dir = 텍스처 폴더(DTD 등, 읽기만)",
        "",
        "Texture source",
    ),
    "PerlinSourceConfig.texture_dir": FieldHelp(
        "텍스처 폴더", "dir 일 때 읽을 폴더 또는 목록 .txt", "", "Texture folder"
    ),
    "PerlinSourceConfig.size_ratio": FieldHelp(
        "결함 창 크기 비율", "결함이 생길 창 한 변 / 이미지 짧은 변 범위", "", "Window size ratio"
    ),
    "PerlinSourceConfig.scale_range": FieldHelp(
        "노이즈 굵기 지수",
        "펄린 노이즈 해상도 2^k 의 k 범위. 올리면 잘게, 내리면 큼직하게",
        "",
        "Perlin scale exponent",
        True,
    ),
    "PerlinSourceConfig.threshold": FieldHelp(
        "노이즈 임계",
        "노이즈 값이 이보다 큰 곳이 결함이 됩니다. 올리면 결함이 작고 드물어집니다",
        "",
        "Noise threshold",
    ),
    "PerlinSourceConfig.rotate": FieldHelp(
        "텍스처 회전", "텍스처를 돌리는 각도 범위", "°", "Texture rotation", True
    ),
    "PerlinSourceConfig.min_area_px": FieldHelp(
        "최소 면적", "이보다 작은 결함은 만들지 않고 건너뜁니다", "px", "Minimum area", True
    ),
    "PerlinSourceConfig.augment": FieldHelp(
        "텍스처 증강",
        "켜면 텍스처에 무작위 증강 3종을 적용합니다(DRAEM 방식)",
        "",
        "Augment texture",
        True,
    ),
    "PerlinSourceConfig.max_tries": FieldHelp(
        "재생성 시도", "결함 면적이 부족할 때 다시 만드는 최대 횟수", "회", "Max tries", True
    ),
    # ---- geometry
    "AffineGeometryConfig.scale": FieldHelp(
        "크기 배율",
        "조각을 붙일 때 곱하는 배율 범위(픽셀 크기 자동 축척 뒤). 넓히면 크기가 다양해지고 1 근처로 좁히면 실제 크기를 지킵니다",
        "배",
        "Scale range",
    ),
    "AffineGeometryConfig.rotate": FieldHelp(
        "회전",
        "돌리는 각도 범위. 빛 방향이 정해진 결함(찍힘)은 ±15° 안으로 — 뒤집히면 음영이 물리적으로 틀립니다",
        "°",
        "Rotation range",
    ),
    "AffineGeometryConfig.flip": FieldHelp(
        "뒤집기",
        "없음 · 좌우 · 상하 · 둘 다. 빛 방향이 정해진 결함은 상하 뒤집기가 조명을 뒤집습니다",
        "",
        "Flip",
    ),
    "ElasticConfig.alpha": FieldHelp(
        "잔물결 세기", "조각을 국소적으로 흔드는 세기. 0 = 끔", "", "Elastic strength", True
    ),
    "ElasticConfig.sigma": FieldHelp(
        "잔물결 굵기", "흔들림의 파장. 크면 완만하게", "px", "Elastic sigma", True
    ),
    "TpsConfig.points": FieldHelp(
        "휘어짐 제어점",
        "전역 휘어짐(thin-plate spline)에 쓰는 제어점 격자 한 변의 수",
        "개",
        "TPS grid points",
        True,
    ),
    "TpsConfig.jitter": FieldHelp(
        "휘어짐 세기",
        "제어점을 짧은 변의 이 비율만큼 흔들어 조각을 휩니다. 0 = 끔",
        "",
        "TPS jitter",
        True,
    ),
    "AffineGeometryConfig.per_class": FieldHelp(
        "클래스별 예외",
        "클래스마다 크기·회전·뒤집기를 따로 둡니다(찍힘만 ±15° 등). 카드의 표에서 편집",
        "",
        "Per-class overrides",
        True,
    ),
    "GeometryOverride.scale": FieldHelp(
        "크기 배율", "이 클래스만의 배율 범위", "배", "Scale range"
    ),
    "GeometryOverride.rotate": FieldHelp("회전", "이 클래스만의 각도 범위", "°", "Rotation range"),
    "GeometryOverride.flip": FieldHelp("뒤집기", "이 클래스만의 뒤집기", "", "Flip"),
    # ---- placement (공통 = _PlacementBase)
    "_PlacementBase.margin_px": FieldHelp(
        "가장자리 여유",
        "붙일 수 있는 영역 경계에서 이만큼 안쪽에만 놓습니다. Poisson 은 경계에 닿으면 실패하므로 0 은 피하세요",
        "px",
        "Edge margin",
    ),
    "_PlacementBase.max_tries": FieldHelp(
        "자리 찾기 시도",
        "겹치지 않는 자리를 찾는 최대 횟수. 넘으면 축소 시도 뒤 건너뜁니다",
        "회",
        "Max placement tries",
        True,
    ),
    "ShrinkConfig.factor": FieldHelp(
        "실패 시 축소 배율",
        "자리를 못 찾으면 조각을 이 배율로 줄여 다시 시도합니다",
        "배",
        "Shrink factor on failure",
        True,
    ),
    "ShrinkConfig.rounds": FieldHelp(
        "축소 반복", "축소를 반복하는 최대 횟수. 0 = 축소 안 함", "회", "Shrink rounds", True
    ),
    "SampledPlacementConfig.distribution": FieldHelp(
        "위치 분포",
        "uniform = 영역 안 고르게 · edge = 가장자리 쪽 · center = 가운데 쪽",
        "",
        "Position distribution",
    ),
    "StructureAwarePlacementConfig.prefer": FieldHelp(
        "선호 표면",
        "edges = 결·에지가 있는 곳 · flat = 평탄한 곳 · uniform = 가리지 않음",
        "",
        "Preferred surface",
    ),
    "StructureAwarePlacementConfig.strength": FieldHelp(
        "선호 세기", "선호 표면 가중의 지수. 0 = 무작위와 같음", "", "Preference strength", True
    ),
    "StructureAwarePlacementConfig.smooth_px": FieldHelp(
        "표면 평활", "결 세기를 이만큼 뭉개서 봅니다. 크면 넓은 결만", "px", "Smoothing", True
    ),
    "StructureAwarePlacementConfig.align": FieldHelp(
        "결 방향 정렬",
        "along = 결 방향으로 · across = 결에 수직 · none = 회전 유지",
        "",
        "Align to structure",
    ),
    "StructureAwarePlacementConfig.min_coherence": FieldHelp(
        "정렬 최소 일관성",
        "자리의 결 방향이 이보다 흐리면 정렬하지 않습니다",
        "",
        "Min coherence",
        True,
    ),
    "StructureAwarePlacementConfig.min_anisotropy": FieldHelp(
        "정렬 최소 길쭉함", "조각이 이보다 둥글면 정렬하지 않습니다", "", "Min anisotropy", True
    ),
    "StructureAwarePlacementConfig.jitter_deg": FieldHelp(
        "정렬 흔들림", "정렬 각도에 더하는 무작위 ±각", "°", "Alignment jitter", True
    ),
    "StructureAwarePlacementConfig.max_align_deg": FieldHelp(
        "정렬 회전 상한",
        "이보다 큰 회전이 필요한 자리는 정렬하지 않습니다(찍힘은 30). 끄면 제한 없음",
        "°",
        "Max alignment rotation",
        True,
    ),
    # ---- roi
    "OtsuRoiConfig.invert": FieldHelp(
        "밝기 반전",
        "auto = 물체가 배경보다 밝은지 자동 · yes/no 로 고정",
        "",
        "Invert polarity",
        True,
    ),
    "OtsuRoiConfig.erode_px": FieldHelp(
        "경계 깎기",
        "영역 가장자리를 이만큼 안쪽으로 줄여 경계에 걸치지 않게 합니다",
        "px",
        "Erode edge",
    ),
    "MaskDirRoiConfig.path": FieldHelp(
        "마스크 폴더", "바탕 이미지와 같은 이름의 마스크 PNG 폴더(흰색 = 허용)", "", "Mask folder"
    ),
    "GrabCutRoiConfig.init": FieldHelp(
        "초기화",
        "rect = 테두리를 배경으로 두고 안쪽에서 전경 찾기 · otsu = 밝기 분리 결과를 다듬기",
        "",
        "Initialization",
        True,
    ),
    "GrabCutRoiConfig.invert": FieldHelp(
        "밝기 반전", "초기화가 otsu 일 때의 극성. auto = 자동", "", "Invert polarity", True
    ),
    "GrabCutRoiConfig.rect_margin": FieldHelp(
        "테두리 배경 비율",
        "rect 초기화에서 확정 배경으로 둘 테두리 두께(이미지 비율)",
        "",
        "Border margin ratio",
        True,
    ),
    "GrabCutRoiConfig.iters": FieldHelp(
        "반복", "GrabCut 반복 횟수. 올리면 정확하지만 느립니다", "회", "Iterations", True
    ),
    "GrabCutRoiConfig.work_px": FieldHelp(
        "계산 해상도",
        "긴 변을 이만큼 줄여 계산합니다(4K 원본은 수십 초). 0 = 원본",
        "px",
        "Working resolution",
        True,
    ),
    "GrabCutRoiConfig.erode_px": FieldHelp(
        "경계 깎기", "영역 가장자리를 이만큼 안쪽으로 줄입니다", "px", "Erode edge"
    ),
    "AnnulusRoiConfig.center": FieldHelp(
        "링 중심", "끄면 바탕마다 자동 검출(가장 큰 물체의 최소 외접원)", "px", "Ring center", True
    ),
    "AnnulusRoiConfig.radius": FieldHelp(
        "기준 반지름", "비율의 기준. 끄면 자동 검출", "px", "Reference radius", True
    ),
    "AnnulusRoiConfig.r_inner": FieldHelp(
        "안쪽 반지름",
        "기준 반지름의 배율(단위 ratio) 또는 px. 이 안쪽은 허용하지 않습니다",
        "",
        "Inner radius",
    ),
    "AnnulusRoiConfig.r_outer": FieldHelp(
        "바깥 반지름",
        "기준 반지름의 배율(단위 ratio) 또는 px. 이 바깥은 허용하지 않습니다",
        "",
        "Outer radius",
    ),
    "AnnulusRoiConfig.units": FieldHelp(
        "반지름 단위", "ratio = 기준 반지름의 배율 · px = 절대값", "", "Radius units", True
    ),
    "AnnulusRoiConfig.invert": FieldHelp(
        "밝기 반전", "자동 검출에 쓰는 밝기 분리의 극성. auto = 자동", "", "Invert polarity", True
    ),
    "AnnulusRoiConfig.erode_px": FieldHelp(
        "경계 깎기", "링의 안·바깥 경계를 이만큼 깎습니다", "px", "Erode edge"
    ),
    # ---- blend
    "AlphaBlendConfig.feather_px": FieldHelp(
        "가장자리 부드럽게", "조각 경계를 이만큼 서서히 섞습니다", "px", "Feather"
    ),
    "AlphaBlendConfig.opacity": FieldHelp(
        "불투명도",
        "결함마다 이 범위의 불투명도로 겹칩니다. 끄면 완전 불투명",
        "",
        "Opacity range",
        True,
    ),
    "PoissonBlendConfig.poisson_mode": FieldHelp(
        "Poisson 방식",
        "normal = 조각 그래디언트 그대로(얼룩처럼 약한 결함) · mixed = 배경과 강한 쪽 선택(스크래치·핏)",
        "",
        "Poisson mode",
    ),
    "PoissonBlendConfig.mask_dilate_px": FieldHelp(
        "풀이 영역 여유",
        "Poisson 이 다시 계산하는 영역을 마스크보다 이만큼 넓힙니다. 5 미만이면 가는 스크래치가 통째로 사라집니다",
        "px",
        "Solve-region dilation",
        True,
    ),
    "PoissonBlendConfig.feather_px": FieldHelp(
        "대체 처리 페더",
        "Poisson 이 실패해 알파로 대체될 때의 경계 부드럽기",
        "px",
        "Fallback feather",
        True,
    ),
    "MultibandBlendConfig.levels": FieldHelp(
        "피라미드 단계",
        "다중 대역 혼합의 단계 수(상한). 조각 두께에 맞춰 자동으로 줄어듭니다",
        "",
        "Pyramid levels",
    ),
    # ---- harmonize
    "StatsHarmonizeConfig.strength": FieldHelp(
        "맞추는 세기",
        "결함 안쪽을 주변 링에 맞추는 정도. 올릴수록 자연스럽지만 결함 대비가 (1−세기) 배로 옅어집니다",
        "",
        "Strength",
    ),
    "StatsHarmonizeConfig.ring_px": FieldHelp(
        "주변 링 폭", "비교 기준이 되는 결함 둘레 띠의 폭", "px", "Ring width", True
    ),
    "ReinhardHarmonizeConfig.strength": FieldHelp(
        "맞추는 세기",
        "Lab 세 채널을 주변에 맞추는 정도. 올릴수록 대비가 옅어집니다",
        "",
        "Strength",
    ),
    "ReinhardHarmonizeConfig.ring_px": FieldHelp(
        "주변 링 폭", "비교 기준이 되는 결함 둘레 띠의 폭", "px", "Ring width", True
    ),
    "HistmatchHarmonizeConfig.strength": FieldHelp(
        "맞추는 세기", "밝기 분포를 주변에 맞추는 정도. 올릴수록 대비가 옅어집니다", "", "Strength"
    ),
    "HistmatchHarmonizeConfig.ring_px": FieldHelp(
        "주변 링 폭", "비교 기준이 되는 결함 둘레 띠의 폭", "px", "Ring width", True
    ),
    "RelativeHarmonizeConfig.strength": FieldHelp(
        "노출 보정 세기",
        "조각 주변과 바탕 주변의 밝기 차만큼 결함을 옮깁니다. 1 = 전부. 결함의 상대 대비는 지킵니다",
        "",
        "Exposure-offset strength",
    ),
    "RelativeHarmonizeConfig.ring_px": FieldHelp(
        "주변 링 폭", "밝기 차를 재는 결함 둘레 띠의 폭", "px", "Ring width", True
    ),
    "RelativeHarmonizeConfig.gain": FieldHelp(
        "대비 배율도 맞춤",
        "켜면 표준편차 비로 대비 배율까지 맞춥니다(링이 작아 잡음 — 보통 끔)",
        "",
        "Match gain too",
        True,
    ),
    # ---- degrade
    "CameraDegradeConfig.noise_sigma": FieldHelp(
        "노이즈", "가우시안 노이즈 표준편차 범위. 올리면 거칠게", "gray", "Noise sigma"
    ),
    "CameraDegradeConfig.blur_sigma": FieldHelp(
        "흐림", "가우시안 블러 σ 범위. 올리면 초점이 흐려집니다", "px", "Blur sigma"
    ),
    "CameraDegradeConfig.jpeg_quality": FieldHelp(
        "JPEG 화질", "이 범위의 화질로 다시 압축합니다. 끄면 압축 없음", "", "JPEG quality", True
    ),
    "CameraDegradeConfig.motion_blur_px": FieldHelp(
        "움직임 흐림", "직선 모션 블러 길이 범위. 끄면 없음", "px", "Motion blur length", True
    ),
    "CameraDegradeConfig.motion_angle": FieldHelp(
        "움직임 방향", "모션 블러 방향 범위(0~180)", "°", "Motion blur angle", True
    ),
    "CameraDegradeConfig.vignette": FieldHelp(
        "가장자리 어둡게", "모서리 감광 세기 범위(1 = 완전히 검게). 끄면 없음", "", "Vignette", True
    ),
    "CameraDegradeConfig.gamma": FieldHelp(
        "감마", "톤 커브 지수 범위(1 = 그대로, 1 미만 = 밝게). 끄면 없음", "", "Gamma", True
    ),
    # ---- gtmask
    "GtMaskConfig.policy": FieldHelp(
        "정답 영역 기준",
        "source = 원본 마스크 그대로 · diff = 실제로 바뀐 픽셀 · union = 둘을 합침(붙이기 오차 흡수)",
        "",
        "GT policy",
    ),
    "GtMaskConfig.diff_threshold": FieldHelp(
        "변화 감지 임계",
        "바탕과 이만큼 이상 달라진 픽셀을 결함으로 봅니다. 내리면 넓게 잡힙니다",
        "gray",
        "Diff threshold",
        True,
    ),
    "GtMaskConfig.dilate_px": FieldHelp(
        "정답 영역 팽창",
        "정답 영역을 이만큼 넓혀 경계 오차를 흡수합니다",
        "px",
        "GT dilation",
        True,
    ),
    # ---- recipe 상단
    "Recipe.name": FieldHelp("레시피 이름", "출력 메타·리포트에 남는 이름", "", "Recipe name"),
    "Recipe.seed": FieldHelp(
        "난수 시드", "같은 시드·같은 레시피면 항상 같은 결과", "", "Random seed"
    ),
    "Recipe.version": FieldHelp(
        "레시피 버전", "레시피 형식 버전(1)", "", "Recipe format version", True
    ),
    "PipelineConfig.preset": FieldHelp(
        "프리셋", "출처 기록용 — 병합은 레시피를 읽을 때 끝납니다", "", "Preset name", True
    ),
}


def field_help(cls: type, path: str) -> FieldHelp | None:
    """``cls``(설정 모델)와 점 경로로 도움말을 찾는다 — 중첩 모델은 마지막 모델의 클래스 이름으로, 상속 필드는 MRO 로."""
    for c in cls.__mro__:
        h = FIELD_HELP.get(f"{c.__name__}.{path}")
        if h is not None:
            return h
    return None


def method_help(stage: str, method: str) -> MethodHelp | None:
    return METHOD_HELP.get((stage, method))


def stage_help(stage: str) -> StageHelp | None:
    return STAGE_HELP.get(stage)


def describe_field(cls: type, path: str) -> dict[str, Any]:
    """``explain``·PARAMS.md 용 — 한 필드의 도움말을 dict 로(없으면 이름만)."""
    h = field_help(cls, path)
    if h is None:
        return {"key": path, "label": path, "desc": "", "unit": "", "en": "", "advanced": False}
    return {
        "key": path,
        "label": h.label,
        "desc": h.desc,
        "unit": h.unit,
        "en": h.en,
        "advanced": h.advanced,
    }

"""작업 흐름(v0.9 사용성 ⑤) — 탭 순서·번호·상태 배지·다음 단계·시작 체크리스트의 **Qt 없는** 계산.

흐름 = ① 결함 표시 → ② 결함 보관함 → ③ 미리보기 → ④ 일괄 생성 → ⑤ 검수. ``WorkflowState`` 는 각 탭 세션에서 읽은 사실(보관함
조각 수 · 레시피 열림 · 준비됨 · 마지막 일괄 생성 결과 · 검수 열림)이고, 여기서 탭 라벨(번호 + 배지)·"다음" 안내·체크리스트 항목을
만든다. ``MainWindow`` 가 상태를 채우고 위젯에 뿌린다.
"""

from __future__ import annotations

from dataclasses import dataclass

STEPS: tuple[tuple[str, str], ...] = (  # (탭 키, 한국어 이름) — 흐름 순서
    ("label", "결함 표시"),
    ("bank", "결함 보관함"),
    ("studio", "미리보기"),
    ("batch", "일괄 생성"),
    ("review", "검수"),
)
CIRCLED = "①②③④⑤"


@dataclass(frozen=True)
class WorkflowState:
    bank_pieces: int | None = None  # 열린 보관함(또는 준비된 세션)의 조각 수
    recipe_open: bool = False  # 미리보기 탭에 레시피가 있는가
    prepared: bool = False  # 보관함·바탕 이미지가 읽혀 미리보기가 도는가
    targets: int = 0
    batch_ok: int | None = None  # 마지막 일괄 생성 성공 장수
    review_open: bool = False
    review_unreviewed: int | None = None
    label_images: int = 0  # 결함 표시 탭에 연 이미지 수


def tab_label(key: str, state: WorkflowState) -> str:
    """탭 라벨 = 번호 + 이름 + 배지(상태가 있을 때만)."""
    idx = [k for k, _n in STEPS].index(key)
    name = STEPS[idx][1]
    badge = ""
    if key == "label" and state.label_images:
        badge = f"{state.label_images}장"
    elif key == "bank" and state.bank_pieces is not None:
        badge = f"{state.bank_pieces}"
    elif key == "studio":
        badge = "준비됨" if state.prepared else ("레시피" if state.recipe_open else "")
    elif key == "batch" and state.batch_ok is not None:
        badge = f"{state.batch_ok}장"
    elif key == "review" and state.review_open:
        badge = f"미검수 {state.review_unreviewed}" if state.review_unreviewed else "완료"
    return f"{CIRCLED[idx]} {name}" + (f" · {badge}" if badge else "")


def next_step(key: str) -> tuple[str, str] | None:
    """다음 탭 (키, 이름). 마지막이면 None."""
    keys = [k for k, _n in STEPS]
    i = keys.index(key)
    return STEPS[i + 1] if i + 1 < len(STEPS) else None


NEXT_HINT: dict[str, str] = {  # 다음 단계로 가기 전에 이 탭에서 끝낼 것
    "label": "결함 사진에 마스크를 그려 '보관함에 저장'(Ctrl+S) 하면 다음은 보관함에서 조각을 확인합니다",
    "bank": "조각과 마스크 신뢰도를 확인·다듬고 나면 미리보기에서 프리셋을 고릅니다",
    "studio": "프리셋·값을 정했으면 '일괄 생성으로 보내기' 로 데이터셋을 만듭니다",
    "batch": "생성이 끝나면 '검수 탭에서 열기' 로 결과를 채택/반려합니다",
    "review": "반려를 뺀 데이터셋을 내보내면 끝 — 학습 형식 폴더를 학습 코드에 넘기세요",
}


@dataclass(frozen=True)
class ChecklistItem:
    key: str  # 이동할 탭 키
    text: str
    done: bool
    action: str = ""  # 버튼 라벨(빈 문자열 = 탭으로 이동만)


def checklist(state: WorkflowState) -> list[ChecklistItem]:
    """시작 체크리스트(빈 상태) — 위에서부터 순서대로, 상태로 ✓."""
    has_bank = bool(state.bank_pieces)
    return [
        ChecklistItem(
            "sample",
            "데이터가 하나도 없으면 상단 '샘플 데이터…' 로 샘플 이미지·보관함·레시피를 한 번에 만듭니다",
            has_bank or state.recipe_open,
            "샘플 데이터…",
        ),
        ChecklistItem(
            "label", "결함 사진에 마스크를 그려 결함 보관함에 저장합니다", has_bank, "결함 표시 탭"
        ),
        ChecklistItem("bank", "보관함의 조각·마스크 신뢰도를 확인합니다", has_bank, "보관함 탭"),
        ChecklistItem(
            "studio",
            "왼쪽 '입력'에 보관함 폴더와 바탕(정상) 이미지 폴더를 넣고 '열기' — 프리셋을 고르고 바탕을 클릭하면 미리보기",
            state.prepared,
            "레시피 열기",
        ),
        ChecklistItem(
            "batch",
            "'일괄 생성으로 보내기' → '생성 시작'",
            state.batch_ok is not None,
            "일괄 생성 탭",
        ),
        ChecklistItem(
            "review",
            "결과를 채택/반려하고 반려를 뺀 데이터셋을 내보냅니다",
            state.review_open,
            "검수 탭",
        ),
    ]

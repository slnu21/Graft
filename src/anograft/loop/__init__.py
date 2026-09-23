"""학습 루프 계층 (설계 `docs/design/v1.x-training-loop.md`).

**코어는 그대로 데이터 계층으로 남고, 여기는 그 위의 얇은 오케스트레이터다**(§0). 학습기는 Python ABC 가
아니라 **프로세스 경계 계약**(`contract`)으로 붙고, 등록은 레시피 밖 로컬 ``trainers.yaml``(`registry`)에 둔다.
루프가 "무엇을 사람에게 보내고 무엇을 자동으로 받고 언제 승급하는가"는 전부 순수 함수(`policy`)다.

어댑터가 하나도 없어도 `anograft run` 은 그대로 돌아간다 — 이게 이 계층의 시험대다.
"""

from anograft.loop.contract import (
    FitResult,
    PredictResult,
    TrainerError,
    TrainerInfo,
    call,
    fit,
    last_json_line,
    merge_spec,
    parse_fit,
    parse_info,
    parse_predict,
    predict,
)
from anograft.loop.policy import (
    Agreement,
    PromotionVerdict,
    ReviewMix,
    ReviewPick,
    RoundPlan,
    agreement,
    gate,
    iou,
    partition_by_gate,
    round_plan,
    select_for_review,
    should_promote,
    synthetic_ratio,
)
from anograft.loop.registry import (
    TrainerSpec,
    TrainerStatus,
    find_trainers_file,
    load_trainers,
    probe,
    probe_all,
)

__all__ = [
    "Agreement",
    "FitResult",
    "PredictResult",
    "PromotionVerdict",
    "ReviewMix",
    "ReviewPick",
    "RoundPlan",
    "TrainerError",
    "TrainerInfo",
    "TrainerSpec",
    "TrainerStatus",
    "agreement",
    "call",
    "find_trainers_file",
    "fit",
    "gate",
    "iou",
    "last_json_line",
    "load_trainers",
    "merge_spec",
    "parse_fit",
    "parse_info",
    "parse_predict",
    "partition_by_gate",
    "predict",
    "probe",
    "probe_all",
    "round_plan",
    "select_for_review",
    "should_promote",
    "synthetic_ratio",
]

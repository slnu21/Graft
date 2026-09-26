"""라운드 오케스트레이션 (설계 `v1.x-training-loop.md` §2 루프 규약, 작업 단위 T10).

한 라운드는 이렇게 돈다 — 그리고 **사람 앞에서 멈춘다**::

    predict → queue → [사람이 검수 화면에서 판정] → accept → synth → train → judge

`anograft loop run` 은 **멱등**하다: 라운드 폴더에 남은 상태를 읽고 **다음 단계부터** 이어 간다. 사람이
판정할 차례면 무엇을 해야 하는지 알려 주고 종료한다(데몬을 만들지 않는다 — 설계 §2b.1). 학습이 3시간 뒤
실패해도 다음 실행이 합성을 다시 하지 않는다("부분 진행 재사용").

**부트스트랩 라운드**(champion 모델이 없거나 현장 이미지가 없을 때)는 수집 넷을 건너뛰고 `synth` 부터 돈다 —
첫 라운드에는 스코어링할 모델이 없다. 그게 Graft 의 자리다(설계 §2b.5(4)).

평가는 별도 predict 가 아니라 **학습 데이터셋의 `val` = 동결 평가셋**이고, `fit` 이 돌려주는 평평한 지표
맵에서 `promote.metric` 을 읽는다 — `tools/train_mvtec_map.py` 가 이미 쓰는 그 경로다(계약 §1.1).

상태는 셋이다(T14 에서 **이력이 원장으로 승격**됐다 — `loop/ledger.py`):

- ``<out>/loop.state.json`` — **움직이는 포인터 둘**뿐이다(지금 라운드 번호 · champion). 롤백이 포인터를
  되돌리는 일이라 이 파일은 덮어쓴다.
- ``<out>/round-NNN/round.json`` — 이 라운드의 `done` 목록(= **다음 단계를 고르는 진실**)과 산출 요약.
- ``<out>/rounds.jsonl`` — append-only 원장(시작·단계·끝·실패·안 돎). 감사·조회는 여기서 나오고,
  **다음 단계를 여기서 재생(replay)해 고르지 않는다** — 한 줄이 깨질 때마다 루프가 멈추는 구조가 된다.

여기는 `core` 위의 얇은 오케스트레이터다 — 합성은 `runner`, 학습은 `loop.contract`, 큐는 `loop.queue`,
승급 판정은 `loop.policy` 가 하고 이 모듈은 **순서와 상태**만 들고 있다.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anograft.loop import ledger as L
from anograft.loop.config import SUPPORTED_FORMATS, LoopConfigError, ResolvedLoop
from anograft.loop.ingest import (
    PROCESSED_FILE,
    FileStamp,
    append_processed,
    plan_ingest,
    read_processed,
    stamp_all,
)
from anograft.loop.lock import DEFAULT_STALE_S, LOCK_FILE, LockInfo, RoundLock, read_lock

Log = Callable[[str], None]
#: 확인이 끝난 학습기 — ``(등록 스펙, 실행 폴더, info 선언)``. 라운드당 한 번만 만든다.
Trainer = tuple[Any, Path, Any]

#: 한 라운드의 단계. 순서가 곧 계약이다(`round.json` 의 `done` 이 이 이름들을 담는다).
PHASES: tuple[str, ...] = ("predict", "queue", "review", "accept", "synth", "train", "judge")
#: 사람에게 보낼 것을 모으는 앞 넷 — 모델이 없으면 통째로 건너뛴다.
COLLECT_PHASES: tuple[str, ...] = ("predict", "queue", "review", "accept")

STATE_FILE = "loop.state.json"
ROUND_FILE = "round.json"
FIELD_LIST = "field.txt"

PHASE_LABEL: dict[str, str] = {
    "predict": "현장 이미지 스코어링",
    "queue": "검토 대기 고르기",
    "review": "사람 판정 대기",
    "accept": "채택분 보관함 편입",
    "synth": "합성",
    "train": "학습·평가",
    "judge": "승급 판정",
}


class LoopError(RuntimeError):
    """라운드를 더 진행할 수 없다 — 사유를 그대로 사람에게 보여 준다(fail-soft, 앱을 죽이지 않는다)."""


# --------------------------------------------------------------------------------------
# 1. 순수 — 어떤 단계를 지나고, 지금 어디인가
# --------------------------------------------------------------------------------------


def round_phases(*, has_champion: bool, has_field: bool) -> tuple[str, ...]:
    """이 라운드가 지날 단계.

    스코어링할 **모델이 없거나**(첫 라운드) 스코어링할 **현장 이미지가 없으면** 수집 넷은 의미가 없다 —
    합성부터 돈다. 이것이 부트스트랩 라운드다.
    """
    rest = tuple(p for p in PHASES if p not in COLLECT_PHASES)
    return PHASES if (has_champion and has_field) else rest


def next_phase(phases: Sequence[str], done: Sequence[str]) -> str | None:
    """아직 안 끝난 첫 단계. 전부 끝났으면 ``None``(= 라운드 완료)."""
    finished = set(done)
    for p in phases:
        if p not in finished:
            return p
    return None


def round_name(number: int) -> str:
    return f"round-{number:03d}"


def review_line(judged: int, total: int) -> str:
    """사람 판정 진행을 한 줄로 — 멈춘 이유는 언제나 숫자와 함께 보여 준다."""
    return f"검토 대기 {total}장 중 {judged}장 판정됨"


# --------------------------------------------------------------------------------------
# 2. 상태 — 파일 둘
# --------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class Champion:
    """지금 배포된 것으로 치는 모델. 롤백은 이 포인터를 옛 라운드로 되돌리는 것뿐이다(설계 §2b.4)."""

    round: int
    model: str
    metric: float
    metric_name: str = "mAP50"

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "model": self.model,
            "metric": self.metric,
            "metric_name": self.metric_name,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Champion:
        return cls(
            round=int(d.get("round", 0)),
            model=str(d.get("model", "")),
            metric=float(d.get("metric", 0.0)),
            metric_name=str(d.get("metric_name", "mAP50")),
        )


@dataclass
class RoundRecord:
    """``round-NNN/round.json`` — 이 라운드가 어디까지 갔고 무엇을 냈는가."""

    number: int
    phases: tuple[str, ...]
    done: list[str] = field(default_factory=list)
    bootstrap: bool = False
    started: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return next_phase(self.phases, self.done) is None

    def mark(self, phase: str, data: Mapping[str, Any] | None = None) -> None:
        if phase not in self.done:
            self.done.append(phase)
        if data:
            self.data[phase] = dict(data)
        self.updated = _now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.number,
            "phases": list(self.phases),
            "done": list(self.done),
            "bootstrap": self.bootstrap,
            "started": self.started,
            "updated": self.updated,
            "data": self.data,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RoundRecord:
        return cls(
            number=int(d.get("round", 1)),
            phases=tuple(str(p) for p in d.get("phases", PHASES)),
            done=[str(p) for p in d.get("done", [])],
            bootstrap=bool(d.get("bootstrap", False)),
            started=str(d.get("started", "")),
            updated=str(d.get("updated", "")),
            data=dict(d.get("data", {})),
            warnings=[str(w) for w in d.get("warnings", [])],
        )


@dataclass
class LoopState:
    """``loop.state.json`` — **움직이는 포인터 둘**(지금 라운드 번호 · champion).

    이력은 여기 없다(T14) — `rounds.jsonl` 이 든다. 덮어쓰는 파일과 더하는 파일을 나누는 이유는 롤백이
    "champion 포인터를 옛 라운드로 되돌리는 일"이고 감사는 "지워지지 않는 줄"이어야 하기 때문이다.
    """

    round: int = 0
    champion: Champion | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "champion": self.champion.to_dict() if self.champion else None,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LoopState:
        champ = d.get("champion")
        return cls(
            round=int(d.get("round", 0)),
            champion=Champion.from_dict(champ) if isinstance(champ, Mapping) else None,
        )


def load_state(out: Path) -> LoopState:
    p = Path(out) / STATE_FILE
    if not p.is_file():
        return LoopState()
    try:
        return LoopState.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise LoopError(f"{p} 를 읽을 수 없습니다: {exc}") from exc


def save_state(out: Path, state: LoopState) -> Path:
    p = Path(out) / STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load_record(round_dir: Path) -> RoundRecord | None:
    p = Path(round_dir) / ROUND_FILE
    if not p.is_file():
        return None
    try:
        return RoundRecord.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise LoopError(f"{p} 를 읽을 수 없습니다: {exc}") from exc


def save_record(round_dir: Path, record: RoundRecord) -> Path:
    p = Path(round_dir) / ROUND_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    return p


# --------------------------------------------------------------------------------------
# 3. 학습 데이터셋 조립 — train = 실제 + 합성 · val = 동결 평가셋
# --------------------------------------------------------------------------------------


def _copy_pairs(
    images: Path,
    second: Path | None,
    dst_images: Path,
    dst_second: Path,
    *,
    prefix: str = "",
    empty_when_missing: bool = False,
) -> tuple[int, int]:
    """이미지와 짝(라벨 ``.txt`` 또는 마스크 ``.png``)을 복사한다. ``(복사한 장수, 짝이 없던 수)``.

    YOLO 는 **짝이 없는 이미지 = 배경(정상)** 이므로 빈 라벨을 만들어 준다(``empty_when_missing``).
    마스크는 만들어 주지 않는다 — 없는 GT 를 지어내면 라벨 노이즈가 된다.
    """
    from anograft.io import imgio

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_second.mkdir(parents=True, exist_ok=True)
    n = missing = 0
    for img in imgio.list_images(images):
        name = f"{prefix}{img.name}"
        shutil.copyfile(img, dst_images / name)
        n += 1
        if second is None:
            missing += 1
            continue
        label = second / f"{img.stem}.txt"
        mask_candidates = [second / f"{img.stem}{ext}" for ext in sorted(imgio.IMAGE_SUFFIXES)]
        if empty_when_missing:
            src = label if label.is_file() else None
            dst = dst_second / f"{prefix}{img.stem}.txt"
            if src is None:
                dst.write_text("", encoding="utf-8")
                missing += 1
            else:
                shutil.copyfile(src, dst)
        else:
            src = next((m for m in mask_candidates if m.is_file()), None)
            if src is None:
                missing += 1
                continue
            shutil.copyfile(src, dst_second / f"{prefix}{img.stem}{src.suffix}")
    return n, missing


def dataset_names(synth: Path | None, fallback: Sequence[str] = ()) -> list[str]:
    """클래스 이름 순서 — 합성 출력의 ``data.yaml``(= 은행 순서)이 정본. 없으면 ``fallback``."""
    if synth is not None:
        p = Path(synth) / "data.yaml"
        if p.is_file():
            import yaml

            try:
                doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                doc = {}
            names = doc.get("names")
            if isinstance(names, Sequence) and not isinstance(names, str):
                return [str(x) for x in names]
    return [str(x) for x in fallback]


def check_label_classes(labels: Path | None, n_classes: int) -> list[str]:
    """평가셋 라벨의 class id 가 은행 클래스 수 안에 드는가 — **어긋나면 평가가 조용히 무의미해진다**."""
    if labels is None or n_classes <= 0 or not Path(labels).is_dir():
        return []
    worst = -1
    for p in sorted(Path(labels).glob("*.txt")):
        for line in p.read_text(encoding="utf-8").splitlines():
            head = line.strip().split(" ")[:1]
            if not head or not head[0]:
                continue
            try:
                worst = max(worst, int(float(head[0])))
            except ValueError:
                continue
    if worst >= n_classes:
        return [
            f"평가셋 라벨의 class id 최대 {worst} 가 클래스 수 {n_classes} 를 넘습니다 — "
            "평가셋 라벨이 보관함 클래스 순서(data.yaml names)와 같은지 확인하세요"
        ]
    return []


def assemble_dataset(
    fmt: str,
    out: Path,
    *,
    synth: Path | None,
    train_base: Mapping[str, Path | None] | None,
    eval_split: Mapping[str, Path | None],
    names: Sequence[str],
) -> tuple[Path, list[str]]:
    """학습기에 넘길 데이터셋을 만든다 — **train = 실제 학습분 + 이번 라운드 합성 · val = 동결 평가셋**.

    ``yolo`` 는 표준 레이아웃(``images/{train,val}`` + ``labels/{train,val}`` + ``data.yaml``),
    ``pairs`` 는 ``{train,val}/{images,masks}``. 합성은 ``syn_`` 접두로 들어가 섞여도 구분된다.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise LoopError(
            f"학습 데이터셋 형식 {fmt!r} 은 아직 루프가 조립하지 않습니다 (지원: {', '.join(SUPPORTED_FORMATS)})"
        )
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)  # 조립은 언제나 처음부터 — 옛 라운드 파일이 섞이면 평가가 거짓말을 한다
    warnings: list[str] = []

    if fmt == "yolo":
        img_tr, lab_tr = out / "images" / "train", out / "labels" / "train"
        img_va, lab_va = out / "images" / "val", out / "labels" / "val"
        n_real = n_syn = 0
        if train_base is not None:
            n_real, missing = _copy_pairs(
                train_base["images"],
                train_base.get("labels"),
                img_tr,
                lab_tr,
                empty_when_missing=True,
            )
            if missing:
                warnings.append(f"실제 학습분 {missing}장에 라벨이 없어 빈 라벨(배경)로 넣었습니다")
        if synth is not None:
            n_syn, _ = _copy_pairs(
                synth / "images",
                synth / "labels",
                img_tr,
                lab_tr,
                prefix="syn_",
                empty_when_missing=True,
            )
        n_val, missing_val = _copy_pairs(
            eval_split["images"],
            eval_split.get("labels"),
            img_va,
            lab_va,
            empty_when_missing=True,
        )
        if missing_val:
            warnings.append(f"평가셋 {missing_val}장에 라벨이 없어 빈 라벨(배경)로 넣었습니다")
        if n_val == 0:
            raise LoopError("평가셋이 0장입니다 — 동결 평가셋 없이는 라운드를 비교할 수 없습니다")
        warnings += check_label_classes(eval_split.get("labels"), len(names))
        import yaml

        (out / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": out.resolve().as_posix(),
                    "train": "images/train",
                    "val": "images/val",
                    "nc": len(names),
                    "names": list(names),
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        if n_real + n_syn == 0:
            raise LoopError("학습 이미지가 0장입니다 (실제 학습분도 합성도 없습니다)")
        return out, warnings

    # pairs — 정본 레이아웃 그대로 train/val 로 나눠 준다
    n_train = 0
    if train_base is not None:
        n_train, _ = _copy_pairs(
            train_base["images"],
            train_base.get("masks"),
            out / "train" / "images",
            out / "train" / "masks",
        )
    if synth is not None:
        n_syn, _ = _copy_pairs(
            synth / "images",
            synth / "masks",
            out / "train" / "images",
            out / "train" / "masks",
            prefix="syn_",
        )
        n_train += n_syn
    n_val, _ = _copy_pairs(
        eval_split["images"], eval_split.get("masks"), out / "val" / "images", out / "val" / "masks"
    )
    if n_val == 0:
        raise LoopError("평가셋이 0장입니다 — 동결 평가셋 없이는 라운드를 비교할 수 없습니다")
    if n_train == 0:
        raise LoopError("학습 이미지가 0장입니다 (실제 학습분도 합성도 없습니다)")
    return out, warnings


# --------------------------------------------------------------------------------------
# 4. 한 라운드 돌리기
# --------------------------------------------------------------------------------------


@dataclass
class RoundResult:
    record: RoundRecord
    state: LoopState
    round_dir: Path
    waiting_for_human: bool = False
    message: str = ""
    warnings: list[str] = field(default_factory=list)


def _trainer_of(loop: ResolvedLoop) -> Trainer:
    """등록부에서 학습기를 찾아 **선언까지 확인**한다 — 못 받을 데이터를 만들기 전에 막는다."""
    from anograft.loop.registry import find_trainers_file, load_trainers, probe, resolve_cwd

    path = find_trainers_file(loop.trainers_file, cwd=loop.base)
    if path is None:
        raise LoopError(
            "trainers.yaml 이 없습니다 — 학습기를 등록해야 루프가 학습을 시킬 수 있습니다"
            " (./trainers.yaml · ~/.anograft/trainers.yaml · loop.yaml 의 trainers_file)"
        )
    trainers = load_trainers(path)
    name = loop.config.trainer
    spec = trainers.get(name)
    if spec is None:
        known = ", ".join(sorted(trainers)) or "(없음)"
        raise LoopError(f"학습기 {name!r} 이 {path} 에 없습니다 — 등록된 이름: {known}")
    status = probe(name, spec, path.parent)
    if not status.available or status.info is None:
        raise LoopError(f"학습기 {name!r} 를 쓸 수 없습니다 — {status.reason}")
    info = status.info
    if info.normal_only:
        raise LoopError(
            f"학습기 {name!r} 는 정상만 학습합니다(trains_on: normal_only) — 합성 결함을 학습셋에 "
            "넣으면 오염입니다. 비지도 경로는 아직 루프가 돌리지 않습니다(설계 §5)"
        )
    if info.dataset_format not in SUPPORTED_FORMATS:
        raise LoopError(
            f"학습기 {name!r} 는 {info.dataset_format} 형식을 받습니다 — 루프의 데이터셋 조립은 "
            f"아직 {', '.join(SUPPORTED_FORMATS)} 만 합니다"
        )
    return spec, resolve_cwd(spec, path.parent), info


def cursor_path(loop: ResolvedLoop) -> Path:
    """유입 커서 파일 — ``<out>/processed.jsonl`` (`loop.state.json` 과 나란히, T13)."""
    return loop.out / PROCESSED_FILE


def ledger_path(loop: ResolvedLoop) -> Path:
    """라운드 원장 — ``<out>/rounds.jsonl`` (T14)."""
    return loop.out / L.ROUNDS_FILE


def tick_path(loop: ResolvedLoop) -> Path:
    """마지막 tick 기록 — ``<out>/tick.json`` (T14)."""
    return loop.out / L.TICK_FILE


def log_event(
    loop: ResolvedLoop, event: str, *, log: Log | None = None, round_no: int = 0, **fields: Any
) -> None:
    """원장에 한 줄. **원장 때문에 라운드를 죽이지 않는다** — 못 적으면 경고만 남긴다(fail-soft)."""
    try:
        L.append(ledger_path(loop), event, round_no=round_no, **fields)
    except OSError as exc:
        if log:
            log(f"경고: 원장에 적지 못했습니다({exc})")


def _field_list(
    loop: ResolvedLoop, round_dir: Path, log: Log, *, since: int | None = None
) -> tuple[Path, list[Path], list[FileStamp]]:
    """이 라운드가 볼 현장 이미지 목록 ``field.txt`` — **한 번 정해지면 얼린다**.

    얼리는 이유가 T13 의 핵심이다: 유입 커서는 predict 가 끝나는 순간 움직이므로, 뒤따르는 queue 단계가
    목록을 **다시 계산하면 방금 스코어링한 것이 전부 "이미 처리"로 빠져 큐가 빈다.** 라운드가 무엇을
    보는가는 라운드가 열릴 때 정해지고, 그 뒤로는 파일이 답한다.

    돌려주는 세 번째 값은 **커서에 적을 도장**이다(새로 고른 경우에만 채워진다 — 얼린 목록을 다시 읽은
    경우엔 호출부가 필요할 때 직접 찍는다).
    """
    from anograft.io import imgio
    from anograft.io.targets import TargetsError, list_targets
    from anograft.loop.queue import index_images

    listing = round_dir / FIELD_LIST
    if listing.is_file():
        if since is not None:  # 조용한 무효과를 만들지 않는다
            log(
                f"이 라운드는 볼 목록이 이미 정해져 있습니다 — --since {since} 는 다음 라운드부터 듭니다"
            )
        return listing, imgio.read_path_list(listing), []

    field_root = loop.field
    if field_root is None:
        raise LoopError("loop.yaml 에 field(현장 이미지)가 없습니다")
    try:
        candidates = list_targets(field_root)
    except TargetsError as exc:
        raise LoopError(str(exc)) from exc

    log_cursor = read_processed(cursor_path(loop))
    if since is not None:
        before = len(log_cursor)
        log_cursor = log_cursor.before_round(since)
        log(
            f"--since {since} — 라운드 {since} 이후 이력 {before - len(log_cursor)}건을 다시 봅니다"
        )
    for w in log_cursor.warnings:
        log(f"경고: {w}")
    plan = plan_ingest(candidates, log_cursor)
    for w in plan.warnings:
        log(f"경고: {w}")
    log(f"유입: {plan.line()}")
    paths = plan.paths
    _, warns = index_images(paths)
    for w in warns:
        log(f"경고: {w}")
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text("\n".join(str(p) for p in paths), encoding="utf-8", newline="\n")
    return listing, paths, plan.fresh


def _phase_predict(
    loop: ResolvedLoop,
    state: LoopState,
    round_dir: Path,
    number: int,
    log: Log,
    trainer: Trainer,
    *,
    since: int | None = None,
) -> dict:
    from anograft.loop.contract import DEFAULT_TIMEOUT_S, TrainerError, merge_spec, predict

    spec, cwd, _info = trainer
    assert state.champion is not None  # 부트스트랩이면 이 단계에 오지 않는다
    listing, paths, stamps = _field_list(loop, round_dir, log, since=since)
    if not paths:
        log("새로 들어온 이미지가 없습니다 — 이번 라운드는 합성만 돕니다")
        return {"count": 0, "scored": 0, "note": "새 이미지 없음"}
    log(f"현장 이미지 {len(paths)}장 스코어링 — 모델 {state.champion.model}")
    try:
        result = predict(
            spec.command,
            model=state.champion.model,
            images=listing,
            out=round_dir / "pred",
            spec=merge_spec(spec.spec, loop.config.spec),
            cwd=cwd,
            timeout=spec.timeout or DEFAULT_TIMEOUT_S,
            on_log=log,
        )
    except TrainerError as exc:
        raise LoopError(f"예측 실패: {exc}") from exc

    # 스코어링이 끝난 **뒤에** 커서를 움직인다 — 여기서 움직여야 라운드가 뒤에서 실패해도 같은 이미지를
    # 두 번 보지 않고, 예측이 실패하면 커서도 그대로다(다음 tick 이 다시 본다).
    if not stamps:  # 얼린 목록을 이어받은 경우
        stamps, warns = stamp_all(paths)
        for w in warns:
            log(f"경고: {w}")
    try:
        written = append_processed(cursor_path(loop), stamps, round_no=number)
    except OSError as exc:
        log(f"경고: 유입 커서를 적지 못했습니다({exc}) — 다음 라운드가 같은 이미지를 다시 봅니다")
        written = 0
    return {
        "count": result.count,
        "scored": len(paths),
        "predictions": str(result.predictions),
        "cursor": written,
    }


def _phase_queue(loop: ResolvedLoop, round_dir: Path, number: int, log: Log) -> dict:
    from anograft.core.seeds import split_rng
    from anograft.loop.policy import ReviewMix
    from anograft.loop.queue import QueueError, build_queue, index_images, read_predictions

    review = loop.config.review
    if not (round_dir / "pred").is_dir():
        # 새로 들어온 이미지가 없어 스코어링을 안 했다(T13) — 고를 것도 없다. 합성만 도는 라운드가 된다.
        log("예측이 없습니다 — 검토 대기도 비웁니다")
        return {"count": 0, "reasons": {}, "note": "예측 없음"}
    try:
        preds = read_predictions(round_dir / "pred")
    except QueueError as exc:
        raise LoopError(str(exc)) from exc
    for w in preds.warnings:
        log(f"경고: {w}")
    _, paths, _ = _field_list(loop, round_dir, log)
    images, _ = index_images(paths)

    from anograft.loop.queue import select_queue

    items = select_queue(
        preds,
        None,
        threshold=review.threshold,
        n=review.n,
        mix=ReviewMix(*review.mix),
        iou_thresh=review.iou,
        rng=split_rng(loop.config.seed),
    )
    if not items:
        return {"count": 0, "reasons": {}}
    items = _with_novelty(loop, items, preds, images, log)
    queue_dir = round_dir / "queue"
    if queue_dir.exists():
        shutil.rmtree(queue_dir)
    summary = build_queue(
        items,
        preds,
        images,
        queue_dir,
        threshold=review.threshold,
        trainer=loop.config.trainer,
        round_no=number,
        novelty_threshold=review.novelty_threshold,
    )
    for w in summary.warnings:
        log(f"경고: {w}")
    if summary.novel:
        log(
            f"처음 보는 형상 {len(summary.novel)}장을 맨 앞에 두었습니다 — "
            "채택하면 이름 없이 미분류로 들어갑니다"
        )
    return {
        "count": len(summary.written),
        "reasons": summary.reasons(),
        "queue": str(queue_dir),
        "without_mask": len(summary.without_mask),
        "novel": len(summary.novel),
    }


def _with_novelty(
    loop: ResolvedLoop,
    items: Sequence[Any],
    preds: Any,
    images: Mapping[str, Path],
    log: Log,
) -> list[Any]:
    """**처음 보는 형상**을 재서 맨 앞으로(T15). 보관함을 못 읽거나 임계가 0 이면 그대로 둔다(fail-soft)."""
    from anograft.loop.queue import novelty_scores, order_by_novelty

    threshold = loop.config.review.novelty_threshold
    if threshold <= 0:
        return list(items)
    from anograft.bank import Bank
    from anograft.bank.bank import BankError

    try:
        refs = Bank.load(loop.bank).novelty_refs()
    except (BankError, OSError, ValueError) as exc:
        log(f"경고: 보관함을 읽지 못해 처음 보는 형상을 재지 않습니다 — {exc}")
        return list(items)
    scores = novelty_scores(items, preds, images, refs)
    if not scores:
        return list(items)
    return order_by_novelty(items, scores, threshold=threshold)


def _review_progress(round_dir: Path) -> tuple[int, int]:
    """``(판정된 수, 판정 대상 수)``. 큐가 비었으면 ``(0, 0)``."""
    from anograft.io.manifest import MANIFEST_FILE, read_manifest
    from anograft.io.prune import REVIEW_FILE, read_review

    queue_dir = round_dir / "queue"
    if not (queue_dir / MANIFEST_FILE).is_file():
        return 0, 0
    rows = [r for r in read_manifest(queue_dir / MANIFEST_FILE) if r.get("status") == "ok"]
    review = read_review(queue_dir / REVIEW_FILE)
    judged = sum(1 for r in rows if review.get(str(r.get("index", "")), ("", ""))[0])
    return judged, len(rows)


def _phase_accept(loop: ResolvedLoop, round_dir: Path, number: int, log: Log) -> dict:
    from anograft.loop.queue import QueueError, accept_to_bank

    queue_dir = round_dir / "queue"
    from anograft.io.manifest import MANIFEST_FILE

    if not (queue_dir / MANIFEST_FILE).is_file():
        return {"accepted": 0, "imported": 0}
    review = loop.config.review
    try:
        summary = accept_to_bank(
            queue_dir,
            loop.bank,
            cls=review.accept_class,
            round_no=number,
            keep_whole=review.keep_whole,
            novelty_threshold=review.novelty_threshold,
            log=log,
        )
    except (QueueError, OSError, ValueError) as exc:
        raise LoopError(f"보관함 편입 실패: {exc}") from exc
    for w in summary.warnings:
        log(f"경고: {w}")
    return {
        "accepted": summary.accepted,
        "imported": summary.imported,
        "per_class": summary.per_class,
        "held_out": len(summary.held_out),
        "unsorted": len(summary.unsorted),
    }


def _phase_synth(loop: ResolvedLoop, round_dir: Path, fmt: str, log: Log, workers: int) -> dict:
    """레시피를 이번 라운드용으로 덮어써 돌린다 — 은행·출력·형식·시드는 루프가 정한다."""
    from anograft import runner
    from anograft.core import recipe as R

    path = loop.recipe
    if not path.is_file():
        raise LoopError(f"레시피가 없습니다: {path}")
    try:
        base = R.Recipe.load(path)
    except (ValueError, OSError) as exc:
        raise LoopError(f"레시피를 읽을 수 없습니다: {exc}") from exc

    # 이 라운드가 **어떤 보관함으로** 만들었는지 남긴다 — 복사가 아니라 id+해시 목록(설계 §6.2).
    snapshot_file = round_dir / "bank.snapshot.json"
    bank_sources = 0
    try:
        from anograft.bank import snapshot as bank_snapshot
        from anograft.bank.bank import BankError

        snap = bank_snapshot.take(loop.bank)
        bank_snapshot.write(snap, snapshot_file)
        bank_sources = len(snap.sources)
    except (BankError, OSError, ValueError) as exc:
        log(f"경고: 보관함 스냅샷을 남기지 못했습니다({exc})")

    out = round_dir / "synth"
    if out.exists():
        shutil.rmtree(out)
    d: dict[str, Any] = base.to_dict()
    d.setdefault("inputs", {})["bank"] = loop.bank.as_posix()
    d.setdefault("output", {})["root"] = out.as_posix()
    if loop.config.synth_count is not None:
        d["output"]["count"] = int(loop.config.synth_count)
    d["output"]["writer"] = {"format": fmt}
    d["seed"] = int(loop.config.seed)
    try:
        recipe = R.Recipe.from_dict(d)
    except Exception as exc:  # pydantic ValidationError
        raise LoopError(f"라운드용 레시피 검증 실패: {exc}") from exc

    try:
        prep = runner.prepare(recipe)
        summary = runner.run(prep, workers=workers, warn=log)
    except runner.PrepareError as exc:
        raise LoopError(f"합성 준비 실패: {exc}") from exc
    if summary.all_skipped:
        raise LoopError("합성이 전부 건너뛰어졌습니다 — dry-run 으로 배치 가능성을 확인하세요")
    return {
        "root": str(out),
        "count": summary.count,
        "ok": summary.writer.n_ok,
        "pipeline_hash": prep.pipeline_hash,
        "bank_fingerprint": prep.bank.fingerprint(),
        "bank_snapshot": str(snapshot_file) if snapshot_file.is_file() else "",
        "bank_sources": bank_sources,
    }


def _phase_train(loop: ResolvedLoop, round_dir: Path, fmt: str, log: Log, trainer: Trainer) -> dict:
    from anograft.loop.contract import DEFAULT_TIMEOUT_S, TrainerError, fit, merge_spec

    spec, cwd, _info = trainer
    synth = round_dir / "synth"
    names = dataset_names(synth if synth.is_dir() else None)
    if not names:
        from anograft.bank import Bank
        from anograft.bank.bank import BankError

        try:
            names = list(Bank.load(loop.bank).classes)
        except BankError as exc:
            raise LoopError(f"클래스 이름을 알 수 없습니다: {exc}") from exc

    dataset, warns = assemble_dataset(
        fmt,
        round_dir / "dataset",
        synth=synth if synth.is_dir() else None,
        train_base=loop.split(loop.config.train_base) if loop.config.train_base else None,
        eval_split=loop.split(loop.config.eval),
        names=names,
    )
    for w in warns:
        log(f"경고: {w}")
    log(f"데이터셋 조립 완료 → {dataset}")
    try:
        result = fit(
            spec.command,
            dataset=dataset,
            out=round_dir / "model",
            seed=loop.config.seed,
            spec=merge_spec(spec.spec, loop.config.spec),
            cwd=cwd,
            timeout=spec.timeout or DEFAULT_TIMEOUT_S,
            on_log=log,
        )
    except TrainerError as exc:
        raise LoopError(f"학습 실패: {exc}") from exc
    metrics = {k: float(v) for k, v in result.metrics.items()}
    return {
        "dataset": str(dataset),
        "model": result.model,
        "metrics": metrics,
        "warnings": warns,
    }


def _phase_judge(loop: ResolvedLoop, state: LoopState, record: RoundRecord, log: Log) -> dict:
    from anograft.loop.policy import should_promote

    train = record.data.get("train") or {}
    metrics = train.get("metrics") or {}
    name = loop.config.promote.metric
    if name not in metrics:
        raise LoopError(
            f"학습 지표에 {name!r} 이 없습니다 (받은 것: {', '.join(sorted(metrics)) or '없음'}) — "
            "loop.yaml 의 promote.metric 을 어댑터가 내는 이름으로 맞추세요"
        )
    challenger = float(metrics[name])
    model = str(train.get("model") or "")

    if state.champion is None:
        state.champion = Champion(
            round=record.number, model=model, metric=challenger, metric_name=name
        )
        log(f"기준선이 없어 이 모델을 champion 으로 둡니다 ({name} {challenger:.4f})")
        return {"promote": True, "reason": "기준선 없음 — 첫 모델", "metric": challenger}

    if baseline_reset_after(loop, state.champion.round):
        state.champion = Champion(
            round=record.number, model=model, metric=challenger, metric_name=name
        )
        log(
            f"기준선 재설정 뒤 첫 라운드 — 점수를 견주지 않고 champion 을 세웁니다 ({name} {challenger:.4f})"
        )
        return {
            "promote": True,
            "reason": "기준선 재설정 뒤 첫 라운드 — 점수를 견주지 않습니다",
            "metric": challenger,
            "baseline_reset": True,
        }

    verdict = should_promote(
        fixed_champion=state.champion.metric,
        fixed_challenger=challenger,
        noise=loop.config.promote.noise,
    )
    log(f"승급 판정: {'승급' if verdict.promote else '유지'} — {verdict.reason}")
    if verdict.promote:
        state.champion = Champion(
            round=record.number, model=model, metric=challenger, metric_name=name
        )
    return {"promote": verdict.promote, "reason": verdict.reason, "metric": challenger}


def run_round(
    loop: ResolvedLoop,
    *,
    on_log: Log | None = None,
    accept_partial: bool = False,
    workers: int = 0,
    since: int | None = None,
    use_lock: bool = True,
    stale_after_s: float = DEFAULT_STALE_S,
) -> RoundResult:
    """다음 단계부터 라운드를 진행한다. 사람이 판정할 차례면 **멈추고 무엇을 할지 알려 준다**.

    잠금(T13)은 기본으로 잡는다 — 스케줄러가 `loop tick` 을 부르기 시작하면 앞 실행의 학습이 세 시간째
    도는 중에 다음 실행이 들어올 수 있다. 못 잡으면 `LockBusyError` 가 그대로 올라간다(오류가 아니라 상태다).
    """
    log: Log = on_log or (lambda _m: None)
    out = loop.out
    out.mkdir(parents=True, exist_ok=True)
    if not use_lock:
        return _run_round(
            loop, log, accept_partial=accept_partial, workers=workers, since=since, lock=None
        )
    with RoundLock(out / LOCK_FILE, stale_after_s=stale_after_s, note="시작", on_log=log) as lock:
        return _run_round(
            loop, log, accept_partial=accept_partial, workers=workers, since=since, lock=lock
        )


def _run_round(
    loop: ResolvedLoop,
    log: Log,
    *,
    accept_partial: bool,
    workers: int,
    since: int | None,
    lock: RoundLock | None,
) -> RoundResult:
    out = loop.out
    # 새 클래스가 생겼으면 **라운드를 열기도 전에** 선다(T15) — 빈 라운드 폴더·원장 줄을 남기지 않는다
    check_new_classes(loop)
    state = load_state(out)

    # 진행 중인 라운드가 있으면 이어서, 없으면 다음 라운드를 연다
    current = state.round if state.round else 0
    record = load_record(out / round_name(current)) if current else None
    if record is None or record.finished:
        # 새 라운드를 **열기 전에만** 자동 정지를 본다(T12) — 진행 중인 라운드를 세우면 사람이 판정해 둔
        # 큐가 썩는다(트리거의 첫 규칙과 같은 이유). 여기서 서므로 빈 라운드 폴더·원장 줄도 남지 않는다.
        check_breaker(loop)
        current += 1
        has_champion = state.champion is not None
        phases = round_phases(has_champion=has_champion, has_field=loop.field is not None)
        record = RoundRecord(
            number=current, phases=phases, bootstrap=not has_champion or loop.field is None
        )
        state.round = current
        save_state(out, state)
        log_event(
            loop,
            L.EVENT_ROUND_START,
            log=log,
            round_no=record.number,
            phases=list(record.phases),
            bootstrap=record.bootstrap,
        )
    round_dir = out / round_name(record.number)
    round_dir.mkdir(parents=True, exist_ok=True)
    save_record(round_dir, record)

    # 학습기 확인은 **라운드당 한 번**이다 — 무거운 어댑터는 `info` 하나가 torch 를 import 한다.
    # 겸사겸사 fail-fast: 못 쓸 학습기면 현장 이미지를 스코어링하기 전에 선다.
    trainer = _trainer_of(loop)
    fmt = trainer[2].dataset_format

    log(f"{round_name(record.number)} — {'부트스트랩 ' if record.bootstrap else ''}시작")
    while True:
        phase = next_phase(record.phases, record.done)
        if phase is None:
            break
        try:
            waiting = _run_phase(
                loop,
                state,
                record,
                round_dir,
                phase,
                log,
                trainer=trainer,
                fmt=fmt,
                accept_partial=accept_partial,
                workers=workers,
                since=since,
                lock=lock,
            )
        except LoopError as exc:
            # 실패 사유는 원장에 남는다(설계 §2b.4) — 세 시간 뒤 실패를 아침에 읽을 사람이 있다.
            log_event(
                loop, L.EVENT_FAILED, log=log, round_no=record.number, phase=phase, reason=str(exc)
            )
            save_record(round_dir, record)
            save_state(out, state)
            raise
        if waiting is not None:
            save_record(round_dir, record)
            save_state(out, state)
            return waiting
        save_record(round_dir, record)
        save_state(out, state)
        log_event(
            loop,
            L.EVENT_PHASE,
            log=log,
            round_no=record.number,
            phase=phase,
            data=record.data.get(phase, {}),
        )

    judge = record.data.get("judge") or {}
    train = record.data.get("train") or {}
    synth = record.data.get("synth") or {}
    accept = record.data.get("accept") or {}
    log_event(
        loop,
        L.EVENT_ROUND_END,
        log=log,
        round_no=record.number,
        metric=judge.get("metric"),
        metric_name=loop.config.promote.metric,
        promoted=bool(judge.get("promote")),
        reason=judge.get("reason", ""),
        model=train.get("model", ""),
        champion=state.champion.to_dict() if state.champion else None,
        pipeline_hash=synth.get("pipeline_hash", ""),
        bank_fingerprint=synth.get("bank_fingerprint", ""),
        bank_snapshot=synth.get("bank_snapshot", ""),
        bank_sources=int(synth.get("bank_sources", 0) or 0),
        bootstrap=record.bootstrap,
        intake=int(accept.get("imported", 0) or 0),
        # 자동 정지(T12)가 다음 라운드에 볼 사실 — 클래스 목록·분포·사람 수정률의 분모/분자
        **bank_facts(loop),
    )
    message = (
        f"{round_name(record.number)} 완료 — "
        f"{loop.config.promote.metric} {judge.get('metric', float('nan')):.4f} · "
        f"{'승급' if judge.get('promote') else '유지'}({judge.get('reason', '')})"
    )
    return RoundResult(
        record=record, state=state, round_dir=round_dir, message=message, warnings=record.warnings
    )


def _run_phase(
    loop: ResolvedLoop,
    state: LoopState,
    record: RoundRecord,
    round_dir: Path,
    phase: str,
    log: Log,
    *,
    trainer: Trainer,
    fmt: str,
    accept_partial: bool,
    workers: int,
    since: int | None,
    lock: RoundLock | None,
) -> RoundResult | None:
    """한 단계. 사람을 기다려야 하면 그 자리에서 `RoundResult` 를 돌려준다(그게 멈춤 신호다)."""
    log(f"· {phase} ({PHASE_LABEL.get(phase, phase)})")
    if lock is not None:  # 하트비트 — 긴 학습이 버려진 잠금으로 보이지 않게
        lock.touch(f"{round_name(record.number)} {PHASE_LABEL.get(phase, phase)}")
    if phase == "predict":
        record.mark(
            phase,
            _phase_predict(loop, state, round_dir, record.number, log, trainer, since=since),
        )
    elif phase == "queue":
        record.mark(phase, _phase_queue(loop, round_dir, record.number, log))
    elif phase == "review":
        judged, total = _review_progress(round_dir)
        if total == 0:
            record.mark(phase, {"judged": 0, "total": 0, "note": "큐가 비었습니다"})
        elif judged >= total or (accept_partial and judged > 0):
            record.mark(phase, {"judged": judged, "total": total})
        else:  # 저장은 부르는 쪽이 한다(단계 하나가 상태 파일을 두 번 쓰지 않게)
            return RoundResult(
                record=record,
                state=state,
                round_dir=round_dir,
                waiting_for_human=True,
                message=(
                    f"{review_line(judged, total)} — 검수 화면에서 판정한 뒤 다시 `loop run` 하세요"
                    f" (폴더 {(round_dir / 'queue').as_posix()})"
                ),
            )
    elif phase == "accept":
        record.mark(phase, _phase_accept(loop, round_dir, record.number, log))
    elif phase == "synth":
        record.mark(phase, _phase_synth(loop, round_dir, fmt, log, workers))
    elif phase == "train":
        record.mark(phase, _phase_train(loop, round_dir, fmt, log, trainer))
    elif phase == "judge":
        record.mark(phase, _phase_judge(loop, state, record, log))
    return None


# --------------------------------------------------------------------------------------
# 4a. 클래스 신설 — 자동화 금지, 사람 결정 이벤트 (설계 §2b.5(3)·§6.7, T15)
# --------------------------------------------------------------------------------------


def bank_classes(loop: ResolvedLoop) -> list[str]:
    """보관함의 **이름 있는** 클래스(미분류 제외). 읽을 수 없으면 빈 목록(fail-soft)."""
    from anograft.bank import Bank
    from anograft.bank.bank import BankError

    try:
        return list(Bank.load(loop.bank).usable_classes)
    except (BankError, OSError, ValueError):
        return []


def bank_facts(loop: ResolvedLoop) -> dict[str, Any]:
    """원장 ``round_end`` 에 남길 보관함 사실 — **다음 라운드의 자동 정지가 이것만 보고 판정한다**(T12).

    - ``bank_classes`` — 이름 있는 클래스 목록(클래스 신설 판정, T15).
    - ``bank_per_class`` — 클래스별 조각 수(분포 변화 감시). 미분류도 센다 — 새 유형이 쏟아지는 것도
      "로트·공정이 바뀌었나"를 물을 이유다.
    - ``drafted``/``corrected`` — **사람 수정률**의 분모/분자(설계 §2 규약 4). 분모는 현장에서 들어온
      조각(`origin:field` 태그 = 편입 당시 전부 ``pred:*``)이고, 분자는 지금 ``manual:*`` 인 것 =
      사람이 결함 표시 화면에서 다듬은 것이다. 라운드 안에서는 잴 수 없으므로(사람은 라운드가 끝난 뒤에
      다듬는다) **누계를 적어 두고 다음 라운드가 차이를 본다**.

    보관함을 못 읽으면 빈 사실을 돌려준다(fail-soft — 원장 때문에 라운드를 죽이지 않는다).
    """
    from anograft.bank import Bank
    from anograft.bank.bank import BankError, is_manual
    from anograft.loop.queue import FIELD_TAG

    empty: dict[str, Any] = {
        "bank_classes": [],
        "bank_per_class": {},
        "drafted": 0,
        "corrected": 0,
    }
    try:
        bank = Bank.load(loop.bank)
    except (BankError, OSError, ValueError):
        return empty
    from_field = [s for s in bank.sources() if FIELD_TAG in s.tags]
    return {
        "bank_classes": list(bank.usable_classes),
        "bank_per_class": bank.counts(),
        "drafted": len(from_field),
        "corrected": sum(1 for s in from_field if is_manual(s.mask_origin)),
    }


def new_classes(loop: ResolvedLoop, led: Any | None = None) -> list[str]:
    """마지막으로 끝난 라운드가 쓴 목록에 없는 **새 클래스 이름**.

    사람이 `baseline_reset` 을 찍어 두었으면(그 라운드 이후로) 빈 목록 — 이미 결정이 내려진 것이다.
    옛 원장처럼 `bank_classes` 가 적혀 있지 않으면 판정하지 않는다(모르면 막지 않는다).
    """
    led = led if led is not None else L.read(ledger_path(loop))
    end = led.last_end
    if end is None:
        return []
    known = [str(c) for c in (end.get("bank_classes") or [])]
    if not known:
        return []
    reset = led.last(L.EVENT_BASELINE_RESET)
    if reset is not None and int(reset.get("after_round", 0) or 0) >= end.round:
        return []
    return [c for c in bank_classes(loop) if c not in known]


def check_new_classes(loop: ResolvedLoop) -> None:
    """새 클래스가 생겼으면 **라운드를 멈춘다** — 회로 차단기가 아니라 사람 결정 이벤트다(설계 §2b.5(3)).

    왜 멈춰야 하나: `classes` 순서 = class id = 출력 `data.yaml` 순서라 새 클래스는 **기존 모델과 비호환**
    이고, 무엇보다 **평가셋에 그 클래스의 정답이 없으면 잘 잡을수록 헛검출로 집계되어 점수가 떨어진다**
    — 라운드 Δ 비교가 그 지점에서 끊긴다. 자동으로 넘기면 루프가 조용히 거짓말을 하기 시작한다.
    """
    added = new_classes(loop)
    if not added:
        return
    names = ", ".join(added)
    raise LoopError(
        f"새 클래스가 생겼습니다: {names} — 라운드를 멈춥니다(사람 결정이 필요합니다).\n"
        "  평가셋에 이 클래스의 정답이 없으면 모델이 잘 잡을수록 헛검출로 집계되어 점수가 떨어집니다"
        " — 라운드 비교가 끊깁니다.\n"
        "  ① 평가셋에 이 클래스를 넣어 다시 만들고"
        " ② `anograft loop baseline-reset --note ...` 로 기준선을 재설정하세요.\n"
        "  그 지점 앞뒤로는 점수를 견주지 않습니다(원장에 남습니다)."
    )


def baseline_reset(loop: ResolvedLoop, *, note: str = "", log: Log | None = None) -> dict[str, Any]:
    """**기준선 재설정** — 원장에 "이 지점 앞뒤로 점수를 견주지 않는다"를 남긴다.

    자동으로 부르지 않는다. 사람이 평가셋을 다시 만든 뒤 한 번 찍는 것이고, 그 다음 라운드는 Δ 비교 없이
    champion 을 새로 세운다(`_phase_judge`).
    """
    led = L.read(ledger_path(loop))
    end = led.last_end
    after = end.round if end is not None else load_state(loop.out).round
    classes = bank_classes(loop)
    log_event(
        loop,
        L.EVENT_BASELINE_RESET,
        log=log,
        round_no=after,
        after_round=after,
        classes=classes,
        note=note,
    )
    return {"after_round": after, "classes": classes, "note": note}


def baseline_reset_after(loop: ResolvedLoop, round_no: int) -> bool:
    """``round_no`` 라운드 뒤에 기준선 재설정이 있었나 — 있으면 그 이전 점수와 견주지 않는다."""
    reset = L.read(ledger_path(loop)).last(L.EVENT_BASELINE_RESET)
    return reset is not None and int(reset.get("after_round", 0) or 0) >= round_no


# --------------------------------------------------------------------------------------
# 4b. 지금 돌 때인가 (`loop tick` — 설계 §2b.3, T14)
# --------------------------------------------------------------------------------------


def _hours_since(when: str) -> float | None:
    """ISO 시각 → 지금까지 몇 시간. 못 읽으면 ``None``(모르면 막지 않는다)."""
    if not when:
        return None
    try:
        then = datetime.fromisoformat(when)
    except ValueError:
        return None
    now = datetime.now(then.tzinfo) if then.tzinfo else datetime.now()
    return max(0.0, (now - then).total_seconds() / 3600.0)


def _bank_source_count(loop: ResolvedLoop) -> int:
    from anograft.bank import Bank
    from anograft.bank.bank import BankError

    try:
        return len(Bank.load(loop.bank).sources())
    except (BankError, OSError, ValueError):
        return 0


def _new_image_count(loop: ResolvedLoop) -> int:
    """아직 스코어링하지 않은 현장 이미지 — **해시를 읽지 않는 어림수**(`ingest.quick_new_count`)."""
    if loop.field is None:
        return 0
    from anograft.io.targets import TargetsError, list_targets
    from anograft.loop.ingest import quick_new_count

    try:
        candidates = list_targets(loop.field)
    except (TargetsError, OSError):
        return 0
    return quick_new_count(candidates, read_processed(cursor_path(loop)))


def trigger_state(loop: ResolvedLoop) -> Any:
    """판정에 필요한 사실을 파일에서 모은다 — `policy.TriggerState`.

    "새 조각"은 **마지막 라운드가 적어 둔 보관함 조각 수와의 차이**다(원장 `round_end.bank_sources`).
    보관함에 시각이 없으므로 이 방법이 사람이 손으로 넣은 조각까지 같이 센다 — 그게 맞다(사람이 라벨을
    스무 장 넣었으면 그건 돌 이유다).
    """
    from anograft.loop.policy import TriggerState

    out = loop.out
    state = load_state(out) if out.exists() else LoopState()
    record = load_record(out / round_name(state.round)) if state.round else None
    end = L.read(ledger_path(loop)).last_end
    previous = int(end.get("bank_sources", 0) or 0) if end is not None else 0
    return TriggerState(
        has_round=bool(state.round) or end is not None,
        in_progress=record is not None and not record.finished,
        new_labels=max(0, _bank_source_count(loop) - previous),
        new_images=_new_image_count(loop),
        hours_since=_hours_since(end.at) if end is not None else None,
    )


def check_trigger(loop: ResolvedLoop) -> tuple[Any, Any]:
    """``(결정, 사실)`` — `loop tick` 과 `loop status` 가 같은 답을 보게."""
    from anograft.loop.policy import should_start_round

    facts = trigger_state(loop)
    return should_start_round(facts, loop.config.trigger.policy()), facts


def note_tick(
    loop: ResolvedLoop, *, ran: bool, reason: str, round_no: int = 0, log: Log | None = None
) -> None:
    """마지막 tick 을 적고, **안 돈 사유가 바뀔 때만** 원장에 남긴다.

    스케줄러가 5분마다 부르는 것을 원장에 다 적으면 원장이 tick 으로 덮인다. 그래도 "조용히 안 도는 루프가
    제일 나쁘다"는 규율은 지켜야 하므로, 사유가 이어지는 동안은 `tick.json` 만 갱신한다.
    """
    previous = L.read_tick(tick_path(loop))
    if not ran and (previous is None or previous.ran or previous.reason != reason):
        log_event(loop, L.EVENT_SKIPPED, log=log, reason=reason)
    try:
        L.write_tick(tick_path(loop), L.Tick(ran=ran, reason=reason, round=round_no))
    except OSError as exc:
        if log:
            log(f"경고: tick 기록을 남기지 못했습니다({exc})")


# --------------------------------------------------------------------------------------
# 4c. 자동 정지 — 망가진 루프는 망가진 채로 계속 돈다 (설계 §6.5, T12)
# --------------------------------------------------------------------------------------


def breaker_history(loop: ResolvedLoop, led: Any | None = None) -> list[Any]:
    """자동 정지가 볼 라운드들 — 원장 ``round_end`` 를 `policy.RoundOutcome` 으로.

    **재설정 지점 뒤만 본다**: 기준선 재설정(T15)은 점수 비교를 끊고, 자동 정지 해제(T12)는 "무엇을
    바꿨다"는 선언이다. 둘 중 뒤에 있는 것 이후의 라운드만 견준다 — 조건이 바뀐 앞 구간과 견주면
    해제가 아무것도 사 주지 않는다.
    """
    from anograft.loop.policy import RoundOutcome

    led = led if led is not None else L.read(ledger_path(loop))
    after = 0
    for event in (L.EVENT_BASELINE_RESET, L.EVENT_BREAKER_RESET):
        mark = led.last(event)
        if mark is not None:
            after = max(after, int(mark.get("after_round", 0) or 0))
    out: list[Any] = []
    for e in led.of(L.EVENT_ROUND_END):
        if e.round <= after:
            continue
        metric = e.get("metric")
        out.append(
            RoundOutcome(
                round=e.round,
                promoted=bool(e.get("promoted")),
                metric=float(metric) if isinstance(metric, (int, float)) else None,
                intake=int(e.get("intake", 0) or 0),
                drafted=int(e.get("drafted", 0) or 0),
                corrected=int(e.get("corrected", 0) or 0),
                per_class={str(k): int(v) for k, v in (e.get("bank_per_class") or {}).items()},
                bootstrap=bool(e.get("bootstrap")),
            )
        )
    return out


def breaker_verdict(loop: ResolvedLoop, led: Any | None = None) -> Any:
    """``(멈출까?, 왜)`` — `loop run`·`loop tick`·`loop status` 가 **같은 답**을 보게 한 곳에서 판정한다."""
    from anograft.loop.policy import circuit_break

    return circuit_break(breaker_history(loop, led), loop.config.breaker.policy())


def check_breaker(loop: ResolvedLoop) -> None:
    """자동 정지가 걸렸으면 **라운드를 열지 않는다**(설계 §6.5).

    멈추는 것은 실패가 아니라 **"데이터 말고 다른 걸 바꿀 때"라는 신호**다 — 조명·해상도·모델. 그래서
    되돌리는 길은 `--force` 같은 무시 스위치가 아니라 `loop breaker-reset --note …` 하나다(사람이 무엇을
    바꿨는지 적고, 그 지점이 원장에 남는다).
    """
    verdict = breaker_verdict(loop)
    if not verdict.tripped:
        return
    raise LoopError(
        f"자동 정지: {verdict.reason}\n"
        "  라운드를 열지 않았습니다 — 멈춘 것은 실패가 아니라 데이터 말고 다른 것을 바꿀 때라는"
        " 신호입니다(조명·해상도·모델·평가셋).\n"
        '  무엇을 바꿨는지 적고 다시 돌리세요: anograft loop breaker-reset --note "..."\n'
        "  (원장 rounds.jsonl 에 남고, 그 앞 라운드는 다음 판정에서 빠집니다)"
    )


def breaker_reset(loop: ResolvedLoop, *, note: str = "", log: Log | None = None) -> dict[str, Any]:
    """**자동 정지 해제** — 무엇을 바꿨는지 적고 다시 돌린다. 자동으로 부르는 곳은 없다.

    `baseline_reset` 과 같은 모양이다(원장에 지점 하나). 걸려 있던 사유를 함께 적어 둔다 — 나중에
    "라운드 7에서 왜 멈췄고 무엇을 바꿔서 풀었나"에 답하는 것이 이 줄이다.
    """
    verdict = breaker_verdict(loop)
    led = L.read(ledger_path(loop))
    end = led.last_end
    after = end.round if end is not None else load_state(loop.out).round
    log_event(
        loop,
        L.EVENT_BREAKER_RESET,
        log=log,
        round_no=after,
        after_round=after,
        tripped=bool(verdict.tripped),
        reason=verdict.reason,
        kinds=list(verdict.kinds),
        note=note,
    )
    return {
        "after_round": after,
        "tripped": bool(verdict.tripped),
        "reason": verdict.reason,
        "kinds": list(verdict.kinds),
        "note": note,
    }


# --------------------------------------------------------------------------------------
# 5. 지금 어디인가 (`loop status`)
# --------------------------------------------------------------------------------------


@dataclass
class LoopStatus:
    out: Path
    state: LoopState
    record: RoundRecord | None
    judged: int = 0
    total: int = 0
    #: 지금 다른 실행이 잡고 있는 잠금(T13). 있으면 "돌고 있는 중"이다.
    lock: LockInfo | None = None
    #: 유입 커서에 적힌 처리 이력 건수(T13).
    processed: int = 0
    #: 최근 라운드 이력 — **원장에서** 온다(T14).
    history: list[dict[str, Any]] = field(default_factory=list)
    #: 마지막 tick(돌았나 · 안 돌았으면 왜, T14).
    tick: Any = None
    #: 지금 돌 때인가 + 그 사유(T14). `loop tick` 이 보는 것과 같은 답이다.
    trigger: Any = None
    #: **자동 정지**(T12) — 걸려 있으면 트리거보다 먼저 답한다(사람이 무엇을 바꿔야 한다).
    breaker: Any = None
    #: 최근 실패(원장 `failed`, T14).
    failures: list[Any] = field(default_factory=list)

    @property
    def next(self) -> str | None:
        if self.record is None:
            return "predict/synth (첫 라운드)"
        return next_phase(self.record.phases, self.record.done)

    def lines(self) -> list[str]:
        out: list[str] = []
        champ = self.state.champion
        out.append(
            f"champion: 라운드 {champ.round} · {champ.metric_name} {champ.metric:.4f} · {champ.model}"
            if champ
            else "champion: 없음 (아직 승급한 모델이 없습니다)"
        )
        if self.lock is not None:
            out.append(f"지금 돌고 있습니다 — {self.lock.text()}")
        if self.processed:
            out.append(f"유입 커서: 이미 처리한 이미지 {self.processed}장")
        if self.record is None:
            out.append("라운드: 아직 없음 — `anograft loop run` 이 첫 라운드를 엽니다")
            return out
        phase = self.next
        state = "완료" if phase is None else f"다음 단계 {phase} ({PHASE_LABEL.get(phase, phase)})"
        out.append(
            f"{round_name(self.record.number)}: {state}"
            + (" · 부트스트랩" if self.record.bootstrap else "")
        )
        if phase == "review" and self.total:
            out.append("  " + review_line(self.judged, self.total))
        if self.breaker is not None and self.breaker.tripped:
            out.append(f"자동 정지: {self.breaker.reason}")
            out.append('  풀려면: anograft loop breaker-reset --note "무엇을 바꿨는지"')
        elif self.breaker is not None and self.breaker.reason:
            out.append(f"자동 정지 감시: {self.breaker.reason}")
        if self.trigger is not None:
            if self.breaker is not None and self.breaker.tripped:
                # 자동 정지가 걸린 채로 "지금 돌 때입니다" 를 같이 찍으면 화면이 자기모순이 된다
                out.append(f"트리거 판정(참고): {self.trigger.reason}")
            else:
                head = "지금 돌 때입니다" if self.trigger.start else "지금은 돌지 않습니다"
                out.append(f"트리거: {head} — {self.trigger.reason}")
        if self.tick is not None:
            out.append(self.tick.line())
        for e in self.failures:
            out.append(f"  실패(라운드 {e.round} {e.phase}): {e.reason}")
        for h in self.history:
            metric = h.get("metric")
            metric_s = f"{metric:.4f}" if isinstance(metric, (int, float)) else "-"
            out.append(
                f"  라운드 {h.get('round')}: {metric_s} · "
                f"{'승급' if h.get('promoted') else '유지'} ({h.get('reason', '')})"
            )
        return out


def status(loop: ResolvedLoop) -> LoopStatus:
    out = loop.out
    state = load_state(out) if out.exists() else LoopState()
    record = load_record(out / round_name(state.round)) if state.round else None
    judged = total = 0
    if record is not None:
        judged, total = _review_progress(out / round_name(record.number))
    led = L.read(ledger_path(loop))
    decision, _facts = check_trigger(loop)
    return LoopStatus(
        out=out,
        state=state,
        record=record,
        judged=judged,
        total=total,
        lock=read_lock(out / LOCK_FILE),
        processed=len(read_processed(cursor_path(loop))),
        history=led.history(),
        tick=L.read_tick(tick_path(loop)),
        trigger=decision,
        breaker=breaker_verdict(loop, led),
        failures=led.failures(limit=2),
    )


__all__ = [
    "COLLECT_PHASES",
    "PHASES",
    "PHASE_LABEL",
    "Champion",
    "LoopConfigError",
    "LoopError",
    "LoopState",
    "LoopStatus",
    "RoundRecord",
    "RoundResult",
    "assemble_dataset",
    "bank_classes",
    "bank_facts",
    "baseline_reset",
    "baseline_reset_after",
    "breaker_history",
    "breaker_reset",
    "breaker_verdict",
    "check_breaker",
    "check_label_classes",
    "check_new_classes",
    "check_trigger",
    "cursor_path",
    "dataset_names",
    "ledger_path",
    "load_record",
    "load_state",
    "log_event",
    "new_classes",
    "next_phase",
    "note_tick",
    "round_name",
    "round_phases",
    "run_round",
    "save_record",
    "save_state",
    "status",
    "tick_path",
    "trigger_state",
]

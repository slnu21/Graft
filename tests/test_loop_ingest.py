"""유입 커서 + 동시 실행 잠금 (설계 §2b.1·2b.2, 작업 단위 T13).

여기가 못 박는 것:

1. **커서는 mtime 단일 값이 아니다** — 처리 이력(집합)이고, 신원은 `(경로·크기·mtime)` 빠른 길과
   `(크기·내용 해시)` 진짜 신원 두 겹이다. 이름을 바꿔 복사해 와도 두 번 스코어링하지 않는다.
2. **라운드가 보는 목록은 얼린다** — 커서는 predict 뒤에 움직이므로, queue 단계가 목록을 다시 계산하면
   방금 스코어링한 것이 전부 "이미 처리"로 빠져 큐가 빈다.
3. **이름이 겹치면 미룬다**(버리지 않는다) — 예측은 stem 키라 덮어쓴다.
4. **잠금은 나이로 판정한다** — Windows 에서 pid 로 살아 있는지 볼 수 없다. 도는 쪽이 단계마다 갱신한다.
5. **`loop tick` 은 스케줄러의 말로 답한다** — 다른 실행이 돌고 있으면 오류가 아니라 종료 코드 0.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from anograft.cli import EXIT_OK, EXIT_RECIPE_ERROR, main
from anograft.io import imgio
from anograft.io.manifest import MANIFEST_FILE, read_manifest
from anograft.io.prune import REVIEW_FILE, write_review
from anograft.loop import ingest
from anograft.loop.config import load_loop_config
from anograft.loop.ingest import (
    PROCESSED_FILE,
    FileStamp,
    ProcessedEntry,
    ProcessedLog,
    append_processed,
    dedupe_stems,
    plan_ingest,
    read_processed,
    stamp,
    unprocessed,
)
from anograft.loop.lock import LOCK_FILE, LockBusyError, RoundLock, read_lock
from anograft.loop.round import PHASES, round_name, run_round, status
from tests.fixtures import blob_image, loop_workspace


def _image(path: Path, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    imgio.write_image(path, blob_image(48, [(20 + seed, 24, 5)]))
    return path


# --------------------------------------------------------------------- 신원 · 순수


def test_stamp_identity_is_content_not_name(tmp_path: Path) -> None:
    """이름을 바꿔 복사한 파일은 **같은 이미지**다 — 두 번 스코어링하면 라운드가 거짓말을 한다."""
    a = _image(tmp_path / "a.png")
    b = tmp_path / "copy_of_a.png"
    b.write_bytes(a.read_bytes())

    sa, sb = stamp(a), stamp(b)
    assert sa.digest and len(sa.digest) == ingest.DIGEST_CHARS
    assert sa.content_key == sb.content_key  # 내용은 같다
    assert sa.stat_key != sb.stat_key  # 빠른 길에서는 다른 파일


def test_unprocessed_skips_by_stat_and_by_content(tmp_path: Path) -> None:
    a = stamp(_image(tmp_path / "a.png"))
    b = stamp(_image(tmp_path / "b.png", seed=3))
    copy = FileStamp(path="/elsewhere/a.png", size=a.size, mtime=0.0, digest=a.digest)
    log = ProcessedLog(entries=[ProcessedEntry(stamp=a, round=1)])

    assert [s.path for s in unprocessed([a, b, copy], log)] == [b.path]
    assert ingest.is_seen(copy, log)  # 경로가 달라도 내용이 같으면 본 것
    assert not ingest.is_seen(b, log)


def test_processed_log_round_trip_and_broken_lines_are_skipped(tmp_path: Path) -> None:
    cursor = tmp_path / PROCESSED_FILE
    first = stamp(_image(tmp_path / "a.png"))
    assert append_processed(cursor, [first], round_no=2) == 1
    assert append_processed(cursor, [], round_no=2) == 0  # 적을 것이 없으면 파일을 건드리지 않는다

    with open(cursor, "a", encoding="utf-8", newline="\n") as fh:
        fh.write("{망가진 줄}\n")
        fh.write(json.dumps({"size": 1}) + "\n")  # path 가 없다

    log = read_processed(cursor)
    assert len(log) == 1 and log.entries[0].round == 2
    assert len(log.warnings) == 2  # 건너뛴 두 줄을 **말해 준다**
    assert ingest.is_seen(first, log)
    assert read_processed(tmp_path / "없는파일.jsonl").entries == []


def test_before_round_reprocesses_that_round_again(tmp_path: Path) -> None:
    """``--since n`` — 커서를 지우지 않고 읽을 때 걸러내므로 감사 이력은 남는다."""
    a = stamp(_image(tmp_path / "a.png"))
    b = stamp(_image(tmp_path / "b.png", seed=4))
    log = ProcessedLog(entries=[ProcessedEntry(stamp=a, round=1), ProcessedEntry(stamp=b, round=3)])
    assert log.last_round == 3
    rolled = log.before_round(3)
    assert len(rolled) == 1 and ingest.is_seen(a, rolled) and not ingest.is_seen(b, rolled)
    assert len(log) == 2  # 원본은 그대로


def test_dedupe_stems_defers_instead_of_dropping() -> None:
    """예측은 ``scores/<stem>.json`` 이라 이름이 겹치면 조용히 덮어쓴다 — 미루고 경고한다."""
    a = FileStamp(path="/x/000.png", size=1, mtime=0.0, digest="aa")
    b = FileStamp(path="/y/000.png", size=2, mtime=0.0, digest="bb")
    c = FileStamp(path="/y/001.png", size=3, mtime=0.0, digest="cc")
    kept, deferred = dedupe_stems([a, b, c])
    assert [s.path for s in kept] == ["/x/000.png", "/y/001.png"]
    assert [s.path for s in deferred] == ["/y/000.png"]


# --------------------------------------------------------------------- 계획 (파일 IO)


def test_plan_ingest_hashes_only_unknown_files(tmp_path: Path, monkeypatch) -> None:
    """아는 파일을 매번 해시하면 4K 수천 장 폴더에서 tick 이 몇 분씩 걸린다 — 빠른 길이 먼저다."""
    paths = [_image(tmp_path / f"f{i}.png", seed=i) for i in range(3)]
    calls: list[str] = []
    real = ingest.content_digest

    def counting(path, **kw):
        calls.append(str(path))
        return real(path, **kw)

    monkeypatch.setattr(ingest, "content_digest", counting)

    first = plan_ingest(paths, ProcessedLog())
    assert len(first.fresh) == 3 and first.seen == 0 and len(calls) == 3

    cursor = tmp_path / PROCESSED_FILE
    append_processed(cursor, first.fresh, round_no=1)
    calls.clear()
    again = plan_ingest(paths, read_processed(cursor))
    assert again.empty and again.seen == 3
    assert calls == []  # 하나도 다시 읽지 않았다


def test_plan_ingest_sees_changed_content_again(tmp_path: Path) -> None:
    p = _image(tmp_path / "f0.png")
    cursor = tmp_path / PROCESSED_FILE
    append_processed(cursor, plan_ingest([p], ProcessedLog()).fresh, round_no=1)
    assert plan_ingest([p], read_processed(cursor)).empty

    time.sleep(0.01)
    _image(p, seed=9)  # 같은 경로, 다른 내용(다시 찍은 사진)
    assert len(plan_ingest([p], read_processed(cursor)).fresh) == 1


def test_plan_ingest_is_fail_soft_about_unreadable_files(tmp_path: Path) -> None:
    ok = _image(tmp_path / "ok.png")
    plan = plan_ingest([ok, tmp_path / "없는파일.png"], ProcessedLog())
    assert [s.path for s in plan.fresh] == [ingest.normalize(ok)]
    assert any("없는파일" in w for w in plan.warnings)
    assert "새 이미지 1장" in plan.line()


def test_plan_ingest_reports_deferred_stems(tmp_path: Path) -> None:
    a = _image(tmp_path / "lot-a" / "000.png")
    b = _image(tmp_path / "lot-b" / "000.png", seed=6)
    plan = plan_ingest([a, b], ProcessedLog())
    assert len(plan.fresh) == 1 and len(plan.deferred) == 1
    assert any("이름이 겹쳐" in w for w in plan.warnings)
    assert "다음 라운드로 1장" in plan.line()


# --------------------------------------------------------------------- 잠금


def test_lock_blocks_a_second_runner(tmp_path: Path) -> None:
    path = tmp_path / LOCK_FILE
    with RoundLock(path, note="합성") as held:
        assert held.held and path.is_file()
        info = read_lock(path)
        assert info is not None and info.note == "합성" and info.pid > 0
        assert "방금 갱신" in info.text()
        with pytest.raises(LockBusyError) as exc:
            RoundLock(path).acquire()
        assert exc.value.info is not None and "합성" in str(exc.value)
    assert not path.exists()  # 놓으면 사라진다
    assert read_lock(path) is None


def test_stale_lock_is_taken_over_with_a_warning(tmp_path: Path) -> None:
    """죽은 프로세스가 남긴 잠금 때문에 라인이 멈추면 안 된다 — 다만 **조용히** 가져가지 않는다."""
    path = tmp_path / LOCK_FILE
    RoundLock(path, note="죽은 실행").acquire()  # 놓지 않고 버린다
    logs: list[str] = []
    with RoundLock(path, stale_after_s=0.0, note="새 실행", on_log=logs.append):
        assert read_lock(path).note == "새 실행"  # type: ignore[union-attr]
    assert any("버려진 잠금" in m for m in logs)


def test_touch_keeps_a_long_phase_from_looking_abandoned(tmp_path: Path) -> None:
    path = tmp_path / LOCK_FILE
    with RoundLock(path, note="시작") as lock:
        lock.touch("학습·평가")
        info = read_lock(path)
        assert info is not None and info.note == "학습·평가"


def test_corrupt_lock_is_still_respected_while_fresh(tmp_path: Path) -> None:
    path = tmp_path / LOCK_FILE
    path.write_text("이건 JSON 이 아니다", encoding="utf-8")
    info = read_lock(path)
    assert info is not None and info.pid == 0 and "알 수 없는 실행" in info.text()
    with pytest.raises(LockBusyError):
        RoundLock(path).acquire()


# --------------------------------------------------------------------- 라운드와 함께


@pytest.fixture
def loop_ws(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, Path]:
    ws = loop_workspace(tmp_path)
    capsys.readouterr()
    return ws


def _judge_all(queue_dir: Path) -> int:
    rows = [r for r in read_manifest(queue_dir / MANIFEST_FILE) if r.get("status") == "ok"]
    write_review(queue_dir / REVIEW_FILE, {r["index"]: ("accept", "") for r in rows})
    return len(rows)


def test_round_scores_each_image_once_and_the_list_is_frozen(loop_ws: dict[str, Path]) -> None:
    loop = load_loop_config(loop_ws["loop"])
    out = loop_ws["out"]
    run_round(loop)  # 1라운드 = 부트스트랩(스코어링할 모델이 없다)
    assert not (out / PROCESSED_FILE).exists()  # 스코어링을 안 했으니 커서도 안 움직인다

    second = run_round(loop)  # 2라운드 — 현장 이미지를 보고 사람 앞에서 선다
    scored = second.record.data["predict"]["scored"]
    assert scored > 0 and len(read_processed(out / PROCESSED_FILE)) == scored
    field_txt = (second.round_dir / "field.txt").read_text(encoding="utf-8")
    queue_rows = len(
        [
            r
            for r in read_manifest(second.round_dir / "queue" / MANIFEST_FILE)
            if r["status"] == "ok"
        ]
    )
    assert queue_rows > 0  # 커서가 방금 적힌 뒤에도 큐가 비지 않았다(목록을 얼렸다)

    again = run_round(loop)  # 멈춘 자리에서 다시 불러도 목록은 그대로
    assert again.waiting_for_human
    assert (again.round_dir / "field.txt").read_text(encoding="utf-8") == field_txt

    _judge_all(second.round_dir / "queue")
    assert run_round(loop).record.done == list(PHASES)

    # 3라운드 — 새로 들어온 것이 없으니 스코어링을 건너뛰고 **합성만** 돈다
    third = run_round(loop)
    assert third.record.number == 3 and not third.waiting_for_human
    assert third.record.data["predict"]["scored"] == 0
    assert third.record.data["queue"]["count"] == 0
    assert third.record.data["train"]["metrics"]["mAP50"] > 0
    assert len(read_processed(out / PROCESSED_FILE)) == scored  # 커서는 그대로

    # 새 사진 한 장이 들어오면 그것만 본다
    _image(Path(loop_ws["field"]) / "new-lot.png", seed=7)
    fourth = run_round(loop)
    assert fourth.record.data["predict"]["scored"] == 1
    assert len(read_processed(out / PROCESSED_FILE)) == scored + 1


def test_since_makes_a_round_look_at_them_again(loop_ws: dict[str, Path]) -> None:
    loop = load_loop_config(loop_ws["loop"])
    run_round(loop)
    second = run_round(loop)
    scored = second.record.data["predict"]["scored"]
    _judge_all(second.round_dir / "queue")
    run_round(loop)

    # 3라운드를 --since 2 로 — 라운드 2 에서 본 것을 다시 스코어링한다(어댑터 교체·버그 수정)
    third = run_round(loop, since=2)
    assert third.record.data["predict"]["scored"] == scored
    # 이력은 지우지 않는다 — 두 번 적혔다(감사가 끊기지 않는다)
    assert len(read_processed(loop_ws["out"] / PROCESSED_FILE)) == scored * 2


def test_status_shows_the_cursor_and_who_is_running(loop_ws: dict[str, Path]) -> None:
    loop = load_loop_config(loop_ws["loop"])
    run_round(loop)
    run_round(loop)
    st = status(loop)
    assert st.processed > 0 and st.lock is None
    assert any("유입 커서" in line for line in st.lines())

    with RoundLock(loop_ws["out"] / LOCK_FILE, note=f"{round_name(2)} 합성"):
        busy = status(loop)
        assert busy.lock is not None
        assert any("지금 돌고 있습니다" in line for line in busy.lines())


# --------------------------------------------------------------------- CLI


def test_cli_loop_tick_runs_a_round(
    loop_ws: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["loop", "tick", "--config", str(loop_ws["loop"]), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["ran"] is True and payload["busy"] is False and payload["round"] == 1


def test_cli_loop_tick_is_quiet_while_another_run_holds_the_lock(
    loop_ws: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """스케줄러에게 "지금은 아니다"는 **정상**이다 — 5분마다 실패 알림이 오면 아무도 안 본다."""
    out = Path(loop_ws["out"])
    out.mkdir(parents=True, exist_ok=True)
    with RoundLock(out / LOCK_FILE, note="라운드 1 학습·평가"):
        assert main(["loop", "tick", "--config", str(loop_ws["loop"]), "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert payload["ran"] is False and payload["busy"] is True
        assert "학습·평가" in payload["reason"]

        # 사람이 직접 부른 `loop run` 에게는 오류다("내가 시킨 일이 안 됐다")
        assert main(["loop", "run", "--config", str(loop_ws["loop"])]) == EXIT_RECIPE_ERROR
        assert "다른 실행이 돌고 있습니다" in capsys.readouterr().err

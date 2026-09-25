"""동시 실행 잠금 — 스케줄러가 부르기 시작하면 필요해진다 (작업 단위 T13, T10 이 남긴 몫).

`loop run` 은 사람이 한 번에 하나만 돌리므로 잠금이 없어도 됐다. `loop tick` 부터는 **Windows 작업
스케줄러·cron 이 부르므로** 앞 tick 의 학습이 세 시간째 돌고 있는데 다음 tick 이 들어올 수 있다. 그때
둘이 같은 라운드 폴더에 쓰면 상태 파일과 데이터셋이 섞인다.

잠금은 파일 하나다 — ``<out>/loop.lock``. 만드는 순간이 원자적이어야 하므로 `os.O_CREAT | os.O_EXCL`
로 연다(mkdir 도 같은 성질이지만 내용을 담을 수 없다). 새 의존성 0.

**살아 있는지는 pid 로 보지 않는다.** Windows 에서 `os.kill(pid, 0)` 은 시그널 0 을 지원하지 않아
**프로세스를 죽인다**(플랫폼별 분기를 두는 것보다 시간이 낫다). 그래서 판정은 **나이**다 — 기본
24시간(학습기 기본 타임아웃 12시간의 두 배)이 지난 잠금만 버려진 것으로 보고 가져간다. 도는 쪽은 단계마다
`touch()` 로 나이를 되돌리므로, 살아 있는 라운드를 빼앗는 일은 없다.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: 라운드 출력 루트에 놓이는 잠금 파일 이름.
LOCK_FILE = "loop.lock"
#: 이만큼 갱신이 없으면 버려진 잠금으로 본다. 학습기 기본 타임아웃(12h)의 두 배.
DEFAULT_STALE_S = 24 * 3600.0


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class LockInfo:
    """잠금을 잡은 쪽이 남긴 쪽지 — 사람에게 "누가 돌고 있나"를 말해 주는 것이 전부다."""

    pid: int = 0
    host: str = ""
    started: str = ""
    note: str = ""
    age_s: float = 0.0

    def text(self) -> str:
        """사람이 읽는 한 줄 — 잠금 때문에 안 돌았다면 **누가 잡고 있는지**를 언제나 함께 보여 준다."""
        who = f"pid {self.pid}" if self.pid else "알 수 없는 실행"
        if self.host:
            who += f" @ {self.host}"
        parts = [who]
        if self.note:
            parts.append(self.note)
        minutes = int(self.age_s // 60)
        parts.append(f"{minutes}분 전 갱신" if minutes else "방금 갱신")
        return " · ".join(parts)

    def to_json(self) -> dict[str, Any]:
        return {"pid": self.pid, "host": self.host, "started": self.started, "note": self.note}


class LockBusyError(RuntimeError):
    """다른 실행이 잡고 있다. **오류가 아니라 상태**다 — 스케줄러에게는 "다음에 오라"가 정상 응답이다."""

    def __init__(self, message: str, info: LockInfo | None = None) -> None:
        super().__init__(message)
        self.info = info


def read_lock(path: str | Path) -> LockInfo | None:
    """잠금 쪽지를 읽는다. 없으면 ``None``, 내용이 깨졌으면 나이만 담아 돌려준다(fail-soft)."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return None
    age = max(0.0, time.time() - st.st_mtime)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return LockInfo(age_s=age)
    if not isinstance(data, dict):
        return LockInfo(age_s=age)
    return LockInfo(
        pid=int(data.get("pid", 0) or 0),
        host=str(data.get("host", "") or ""),
        started=str(data.get("started", "") or ""),
        note=str(data.get("note", "") or ""),
        age_s=age,
    )


class RoundLock:
    """``with RoundLock(out / LOCK_FILE) as lock:`` — 못 잡으면 `LockBusyError`.

    `note` 는 지금 무엇을 하는 중인지다(단계 이름) — `loop status` 가 그것을 그대로 보여 준다.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        stale_after_s: float = DEFAULT_STALE_S,
        note: str = "",
        on_log: Any = None,
    ) -> None:
        self.path = Path(path)
        self.stale_after_s = float(stale_after_s)
        self.note = note
        self._held = False
        self._log = on_log or (lambda _m: None)

    # ---------------------------------------------------------------- 잡기·놓기

    def _write(self, note: str) -> None:
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started": _now(),
            "note": note,
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )

    def _try_create(self) -> bool:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.close(fd)
        return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._try_create():
            info = read_lock(self.path)
            age = info.age_s if info else 0.0
            # `<` 이다 — mtime 해상도 때문에 갓 만든 잠금의 나이가 정확히 0.0 으로 나오므로
            # `stale_after_s=0` 은 "무조건 가져간다"는 뜻이어야 한다(테스트가 그 값을 쓴다).
            if info is not None and age < self.stale_after_s:
                raise LockBusyError(
                    f"다른 실행이 돌고 있습니다 — {info.text()}. 끝나면 다시 부르세요"
                    f" (잠금 {self.path.as_posix()})",
                    info,
                )
            # 나이가 지난 잠금 = 죽은 프로세스가 남긴 것. 알리고 가져간다(조용히 넘어가지 않는다).
            self._log(
                f"경고: 버려진 잠금을 가져갑니다 — {int(age / 60)}분간 갱신이 없었습니다"
                f" ({self.path.as_posix()})"
            )
            try:
                self.path.unlink()
            except OSError as exc:
                raise LockBusyError(f"잠금을 치울 수 없습니다: {exc}", info) from exc
            if not self._try_create():  # 경합 — 그 사이 다른 쪽이 잡았다
                raise LockBusyError("다른 실행이 방금 잠금을 잡았습니다", read_lock(self.path))
        self._held = True
        self._write(self.note)

    def touch(self, note: str | None = None) -> None:
        """하트비트 — 단계마다 부른다. 이걸 빼먹으면 긴 학습이 버려진 잠금으로 보인다."""
        if not self._held:
            return
        if note is not None:
            self.note = note
        with contextlib.suppress(OSError):  # 잠금 갱신 실패로 라운드를 죽이지 않는다
            self._write(self.note)

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        with contextlib.suppress(OSError):
            self.path.unlink()

    @property
    def held(self) -> bool:
        return self._held

    def __enter__(self) -> RoundLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


__all__ = [
    "DEFAULT_STALE_S",
    "LOCK_FILE",
    "LockBusyError",
    "LockInfo",
    "RoundLock",
    "read_lock",
]

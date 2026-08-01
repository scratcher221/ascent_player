"""Parse skill-training watchdog / thread-BC cycle logs for the training monitor."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_LOG_NAME_RE = re.compile(
    r"skill_training_watchdog_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.log$"
)

_HOURS_PS_RE = re.compile(
    r"run_thread_bc_with_watchdog\.py(?:\s+\S+)*\s+--hours\s+([0-9.]+)"
)

_KEY_LINE_MARKERS = (
    "WATCHDOG_",
    "THREAD_BC_CYCLE",
    "CYCLE_ROUND",
    "PHASE_",
    "COLLECT_DONE",
    "CEILING_SUMMARY",
    "EVAL_",
    "TRAIN_LOOP",
    "OFFLINE_BC",
    "SKILL_BC",
    "SEED_BEST",
    "TARGET_MET",
    "WATCHDOG_STALL",
    "Traceback",
    "Error",
    "error",
)


@dataclass(slots=True)
class SessionStats:
    hours: float = 8.0
    session_start: datetime | None = None
    cycle_round: int | None = None
    cycle_remaining_h: float | None = None
    episode: int | None = None
    episode_score: float | None = None
    episode_reward: float | None = None
    epsilon: float | None = None
    collect_recent_avg: float | None = None
    collect_replay: int | None = None
    ceiling_mean: float | None = None
    eval_aligned_mean: float | None = None
    eval_greedy_mean: float | None = None
    watchdog_events: int | None = None
    watchdog_last: str | None = None
    watchdog_trainer_pid: int | None = None
    last_phase: str | None = None
    last_train_marker: str | None = None
    error_count: int = 0
    key_lines: list[str] = field(default_factory=list)
    log_bytes: int = 0


def session_start_from_log_path(path: Path) -> datetime | None:
    m = _LOG_NAME_RE.search(path.name)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups())
    return datetime(y, mo, d, h, mi, s)


def parse_watchdog_hours_from_ps() -> float | None:
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "args="],
            text=True,
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        if "run_thread_bc_with_watchdog.py" not in line:
            continue
        m = _HOURS_PS_RE.search(line)
        if m:
            return float(m.group(1))
    return None


def watchdog_running() -> bool:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "scripts/run_thread_bc_with_watchdog.py"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return False
    return bool(out.strip())


def trainer_pids() -> list[int]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "scripts/run_thread_bc_cycle.py"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def find_active_watchdog_log(log_dir: Path) -> Path | None:
    if not log_dir.is_dir():
        return None
    candidates = sorted(
        log_dir.glob("skill_training_watchdog_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    pids = set(trainer_pids())
    if pids:
        for path in candidates[:5]:
            try:
                tail = path.read_bytes()[-12000:].decode("utf-8", errors="replace")
            except OSError:
                continue
            for pid in pids:
                if f"WATCHDOG_PROGRESS pid={pid} " in tail or f"pid={pid}" in tail:
                    return path
    return candidates[0]


def _remember_key_line(stats: SessionStats, line: str, *, max_lines: int = 24) -> None:
    stripped = line.strip()
    if not stripped:
        return
    if not any(m in stripped for m in _KEY_LINE_MARKERS):
        if not stripped.startswith("episode="):
            return
    if stats.key_lines and stats.key_lines[-1] == stripped:
        return
    stats.key_lines.append(stripped)
    if len(stats.key_lines) > max_lines:
        stats.key_lines = stats.key_lines[-max_lines:]


def apply_log_lines(stats: SessionStats, lines: list[str]) -> SessionStats:
    for line in lines:
        _apply_line(stats, line)
    return stats


def _apply_line(stats: SessionStats, line: str) -> None:
    stripped = line.strip()
    if not stripped:
        return

    if "Traceback" in stripped or re.search(r"\bError\b", stripped):
        stats.error_count += 1

    m = re.search(r"THREAD_BC_CYCLE_START\b.*\bhours=([0-9.]+)", stripped)
    if m:
        stats.hours = float(m.group(1))

    m = re.search(r"CYCLE_ROUND id=(\d+) remaining_h=([0-9.]+)", stripped)
    if m:
        stats.cycle_round = int(m.group(1))
        stats.cycle_remaining_h = float(m.group(2))

    m = re.search(
        r"episode=(\d+) reward=([-0-9.]+) score=([0-9.]+) epsilon=([0-9.]+)",
        stripped,
    )
    if m:
        stats.episode = int(m.group(1))
        stats.episode_reward = float(m.group(2))
        stats.episode_score = float(m.group(3))
        stats.epsilon = float(m.group(4))

    m = re.search(r"COLLECT_DONE recent_avg=([0-9.]+) replay=(\d+)", stripped)
    if m:
        stats.collect_recent_avg = float(m.group(1))
        stats.collect_replay = int(m.group(2))

    m = re.search(r"CEILING_SUMMARY mean=([0-9.]+)", stripped)
    if m:
        stats.ceiling_mean = float(m.group(1))

    m = re.search(r"EVAL_ALIGNED mean=([0-9.]+)", stripped)
    if m:
        stats.eval_aligned_mean = float(m.group(1))

    m = re.search(r"EVAL_GREEDY mean=([0-9.]+)", stripped)
    if m:
        stats.eval_greedy_mean = float(m.group(1))

    m = re.search(
        r"WATCHDOG_PROGRESS pid=(\d+) events=(\d+) last='([^']*)'",
        stripped,
    )
    if m:
        stats.watchdog_trainer_pid = int(m.group(1))
        stats.watchdog_events = int(m.group(2))
        stats.watchdog_last = m.group(3)

    if stripped.startswith("PHASE_"):
        stats.last_phase = stripped.split()[0]

    if "TRAIN_LOOP" in stripped or re.search(r"(?:SKILL_BC|OFFLINE_BC) step=", stripped):
        stats.last_train_marker = stripped[:120]

    _remember_key_line(stats, stripped)


def parse_log_tail(text: str, *, base: SessionStats | None = None) -> SessionStats:
    stats = base if base is not None else SessionStats()
    apply_log_lines(stats, text.splitlines())
    return stats


@dataclass(slots=True)
class LogTailReader:
    """Incrementally read new log bytes and maintain parsed session stats."""

    path: Path
    offset: int = 0
    stats: SessionStats = field(default_factory=SessionStats)
    bootstrap_bytes: int = 200_000

    def __post_init__(self) -> None:
        if self.stats.session_start is None:
            self.stats.session_start = session_start_from_log_path(self.path)

    def refresh(self) -> SessionStats:
        if not self.path.exists():
            return self.stats
        size = self.path.stat().st_size
        self.stats.log_bytes = size
        if self.offset == 0 and size > 0:
            start = max(0, size - self.bootstrap_bytes)
            with self.path.open("rb") as handle:
                handle.seek(start)
                chunk = handle.read().decode("utf-8", errors="replace")
            if start > 0:
                chunk = chunk.split("\n", 1)[-1]
            parse_log_tail(chunk, base=self.stats)
            self.offset = size
            return self.stats
        if size < self.offset:
            self.offset = 0
            self.stats = SessionStats(session_start=session_start_from_log_path(self.path))
        if size == self.offset:
            return self.stats
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read().decode("utf-8", errors="replace")
        self.offset = size
        if chunk:
            parse_log_tail(chunk, base=self.stats)
        return self.stats


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"

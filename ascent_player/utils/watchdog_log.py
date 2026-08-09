"""Parse skill-training / v2-climb watchdog logs for the training monitor."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_LOG_NAME_RE = re.compile(
    r"(?:skill_training_watchdog|v2_climb_watchdog|v2_climb_ladder)_"
    r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.log$"
)

_HOURS_PS_RE = re.compile(
    r"run_(?:thread_bc|v2_climb)_with_watchdog\.py(?:\s+\S+)*\s+--hours\s+([0-9.]+)"
)
_CLIMB_HOURS_PS_RE = re.compile(
    r"run_v2_climb_ladder\.py(?:\s+\S+)*\s+--hours\s+([0-9.]+)"
)

_WATCHDOG_SCRIPTS = (
    "scripts/run_v2_climb_with_watchdog.py",
    "scripts/run_thread_bc_with_watchdog.py",
)
_TRAINER_SCRIPTS = (
    "scripts/run_v2_climb_ladder.py",
    "scripts/run_thread_bc_cycle.py",
)

_KEY_LINE_MARKERS = (
    "WATCHDOG_",
    "THREAD_BC_CYCLE",
    "CYCLE_ROUND",
    "CLIMB_",
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
    "BROWSER_PLAYING",
    "BROWSER_RESET done",
    "SEED_THREAD",
    "Traceback",
    "Error",
    "error",
)


@dataclass(slots=True)
class SessionStats:
    hours: float = 8.0
    session_start: datetime | None = None
    session_kind: str = "thread_bc"  # thread_bc | v2_climb
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
    climb_probe_mean: float | None = None
    climb_best_mean: float | None = None
    climb_bc_loss: float | None = None
    climb_bc_step: int | None = None
    climb_bc_steps: int | None = None
    model_variant: str | None = None
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


def _pgrep_pids(pattern: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
    except subprocess.CalledProcessError:
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


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
        if "run_v2_climb_with_watchdog.py" in line or "run_thread_bc_with_watchdog.py" in line:
            m = _HOURS_PS_RE.search(line)
            if m:
                return float(m.group(1))
        if "run_v2_climb_ladder.py" in line and "with_watchdog" not in line:
            m = _CLIMB_HOURS_PS_RE.search(line)
            if m:
                return float(m.group(1))
    return None


def watchdog_running() -> bool:
    return any(_pgrep_pids(script) for script in _WATCHDOG_SCRIPTS)


def trainer_pids() -> list[int]:
    pids: list[int] = []
    for script in _TRAINER_SCRIPTS:
        pids.extend(_pgrep_pids(script))
    return pids


def active_session_kind() -> str:
    if _pgrep_pids("scripts/run_v2_climb_with_watchdog.py") or _pgrep_pids(
        "scripts/run_v2_climb_ladder.py"
    ):
        return "v2_climb"
    return "thread_bc"


def find_active_watchdog_log(log_dir: Path) -> Path | None:
    if not log_dir.is_dir():
        return None

    # Prefer pointer written by the v2 climb watchdog.
    for pointer in (
        log_dir / "v2_climb_ladder_latest.path",
        log_dir / "v2_climb_watchdog_supervisor.path",
    ):
        if pointer.exists():
            try:
                rel = pointer.read_text(encoding="utf-8").strip().splitlines()[0]
            except OSError:
                rel = ""
            if rel:
                candidate = Path(rel)
                if not candidate.is_absolute():
                    candidate = (log_dir.parent / candidate).resolve()
                    if not candidate.exists():
                        candidate = (log_dir / Path(rel).name).resolve()
                if candidate.exists():
                    # Prefer trainer log over supervisor when both exist.
                    if "supervisor" in candidate.name:
                        sibling = log_dir / candidate.name.replace(
                            "_supervisor", ""
                        ).replace("supervisor_", "")
                        # Map v2_climb_watchdog_supervisor_STAMP.log → v2_climb_watchdog_STAMP.log
                        stamp = re.search(r"(\d{8}_\d{6})", candidate.name)
                        if stamp:
                            climb_log = log_dir / f"v2_climb_watchdog_{stamp.group(1)}.log"
                            if climb_log.exists():
                                return climb_log
                    return candidate

    patterns = (
        "v2_climb_watchdog_*.log",
        "v2_climb_ladder_*.log",
        "skill_training_watchdog_*.log",
    )
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(log_dir.glob(pattern))
    candidates = [
        p
        for p in candidates
        if "supervisor" not in p.name and p.is_file()
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    pids = set(trainer_pids())
    if pids:
        for path in candidates[:8]:
            try:
                tail = path.read_bytes()[-16000:].decode("utf-8", errors="replace")
            except OSError:
                continue
            for pid in pids:
                if f"WATCHDOG_PROGRESS pid={pid} " in tail or f"pid={pid}" in tail:
                    return path
            if "CLIMB_START" in tail or "CLIMB_ROUND" in tail:
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
        stats.session_kind = "thread_bc"

    m = re.search(
        r"CLIMB_START\b.*\bvariant=(\S+).*?\bhours=([0-9.]+)",
        stripped,
    )
    if m:
        stats.model_variant = m.group(1)
        stats.hours = float(m.group(2))
        stats.session_kind = "v2_climb"
    elif stripped.startswith("CLIMB_START"):
        stats.session_kind = "v2_climb"
        m = re.search(r"hours=([0-9.]+)", stripped)
        if m:
            stats.hours = float(m.group(1))
        m = re.search(r"variant=(\S+)", stripped)
        if m:
            stats.model_variant = m.group(1)
        m = re.search(r"best=([0-9.]+)", stripped)
        if m:
            stats.climb_best_mean = float(m.group(1))

    m = re.search(r"WATCHDOG_START\b.*\bremaining_h=([0-9.]+)", stripped)
    if m:
        stats.hours = float(m.group(1))
        if "variant=impala" in stripped or "v2_climb" in stripped:
            stats.session_kind = "v2_climb"

    m = re.search(r"CYCLE_ROUND id=(\d+) remaining_h=([0-9.]+)", stripped)
    if m:
        stats.cycle_round = int(m.group(1))
        stats.cycle_remaining_h = float(m.group(2))

    m = re.search(r"CLIMB_ROUND\s+(\d+)\s+remaining_s=([0-9.]+)", stripped)
    if m:
        stats.cycle_round = int(m.group(1))
        stats.cycle_remaining_h = float(m.group(2)) / 3600.0
        stats.session_kind = "v2_climb"
        stats.last_phase = f"CLIMB_ROUND {m.group(1)}"

    m = re.search(
        r"episode=(\d+) reward=([-0-9.]+) score=([0-9.]+) epsilon=([0-9.]+)",
        stripped,
    )
    if m:
        stats.episode = int(m.group(1))
        stats.episode_reward = float(m.group(2))
        stats.episode_score = float(m.group(3))
        stats.epsilon = float(m.group(4))

    # Browser collect step lines (no episode counter while gate blocks replay).
    m = re.search(r"^step=\d+\s+action=\S+.*\bscore=([0-9.]+)", stripped)
    if m:
        stats.episode_score = float(m.group(1))
        stats.last_train_marker = stripped[:120]

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

    m = re.search(r"CLIMB_PROBE_GREEDY mean=([0-9.]+)", stripped)
    if m:
        stats.climb_probe_mean = float(m.group(1))
        stats.eval_greedy_mean = float(m.group(1))

    m = re.search(r"CLIMB_BEST_UPDATE mean=([0-9.]+)", stripped)
    if m:
        stats.climb_best_mean = float(m.group(1))

    m = re.search(r"CLIMB_BC step=(\d+)/(\d+) loss=([0-9.]+)", stripped)
    if m:
        stats.climb_bc_step = int(m.group(1))
        stats.climb_bc_steps = int(m.group(2))
        stats.climb_bc_loss = float(m.group(3))
        stats.last_phase = "CLIMB_BC"
        stats.last_train_marker = stripped[:120]

    m = re.search(r"CLIMB_TD step=(\d+)/(\d+)", stripped)
    if m:
        stats.last_phase = "CLIMB_TD"
        stats.last_train_marker = stripped[:120]

    if stripped.startswith("CLIMB_POLICY_COLLECT"):
        stats.last_phase = "CLIMB_POLICY_COLLECT"
        stats.last_train_marker = stripped[:120]

    if stripped.startswith("CLIMB_RELIABILITY"):
        stats.last_phase = "CLIMB_RELIABILITY"
        m = re.search(r"mean=([0-9.]+)", stripped)
        if m:
            stats.eval_greedy_mean = float(m.group(1))

    m = re.search(r"MODEL_BUILD variant=(\S+)", stripped)
    if m:
        stats.model_variant = m.group(1)

    m = re.search(
        r"WATCHDOG_PROGRESS pid=(\d+) events=(\d+) last='([^']*)'",
        stripped,
    )
    if m:
        stats.watchdog_trainer_pid = int(m.group(1))
        stats.watchdog_events = int(m.group(2))
        stats.watchdog_last = m.group(3)

    if stripped.startswith("PHASE_") or stripped.startswith("CLIMB_"):
        token = stripped.split()[0]
        if token not in {"CLIMB_BC", "CLIMB_TD"} or stats.last_phase is None:
            if token.startswith("PHASE_") or token in {
                "CLIMB_POLICY_COLLECT",
                "CLIMB_PROBE_GREEDY",
                "CLIMB_RELIABILITY",
                "CLIMB_ROUND",
            }:
                stats.last_phase = token

    if "TRAIN_LOOP" in stripped or re.search(
        r"(?:SKILL_BC|OFFLINE_BC|CLIMB_BC|CLIMB_TD) step=", stripped
    ):
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

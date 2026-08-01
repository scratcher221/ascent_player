"""Live stats for scripts/run_reliability_matrix.py sessions."""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

_CELL_START = re.compile(
    r"MATRIX_CELL_START policy=(\S+) watch=([0-9.]+) skill=(True|False)"
)
_CELL_DONE = re.compile(
    r"MATRIX_CELL_DONE policy=(\S+) watch=([0-9.]+) skill=(True|False) "
    r"mean=([0-9.]+) min=([0-9.]+) rc=(-?\d+) elapsed_s=([0-9.]+)"
)
_EVAL_PROGRESS = re.compile(
    r"BROWSER_EVAL_PROGRESS (\d+)/(\d+) last_score=([0-9.]+)"
)
_MATRIX_DONE = re.compile(r"MATRIX_DONE summary=(\S+) failed=(\d+)")

DEFAULT_CELLS_TOTAL = 11
DEFAULT_MINUTES_PER_CELL = 22.0


@dataclass(slots=True)
class MatrixSessionStats:
    log_path: Path
    log_bytes: int = 0
    cells_total: int = DEFAULT_CELLS_TOTAL
    cells_complete: int = 0
    cells_failed: int = 0
    current_policy: str | None = None
    current_watch: float | None = None
    current_skill: bool | None = None
    eval_episode: int | None = None
    eval_episodes: int | None = None
    last_score: float | None = None
    matrix_running: bool = False
    benchmark_running: bool = False
    matrix_pid: int | None = None
    benchmark_pid: int | None = None
    last_cell_mean: float | None = None
    key_lines: list[str] = field(default_factory=list)
    error_count: int = 0
    session_start_ts: float | None = None


def _pgrep(pattern: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
    except subprocess.CalledProcessError:
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _cell_complete(path: Path, episodes: int = 100) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    scores = payload.get("scores") or []
    return len(scores) >= episodes and float(payload.get("mean", 0)) > 0


def count_complete_cells(cell_dir: Path, *, episodes: int = 100) -> int:
    if not cell_dir.is_dir():
        return 0
    return sum(1 for p in cell_dir.glob("*.json") if _cell_complete(p, episodes))


def refresh_matrix_stats(
    log_path: Path,
    *,
    cell_dir: Path | None = None,
    bootstrap_bytes: int = 120_000,
) -> MatrixSessionStats:
    import time

    cell_dir = cell_dir or (log_path.parent / "matrix_cells")
    stats = MatrixSessionStats(log_path=log_path)
    stats.cells_complete = count_complete_cells(cell_dir)

    pids = _pgrep("scripts/run_reliability_matrix.py")
    stats.matrix_running = bool(pids)
    stats.matrix_pid = pids[0] if pids else None
    bench = _pgrep("scripts/run_browser_reliability_100.py")
    stats.benchmark_running = bool(bench)
    stats.benchmark_pid = bench[0] if bench else None
    if stats.benchmark_pid:
        try:
            args = subprocess.check_output(
                ["ps", "-o", "args=", "-p", str(stats.benchmark_pid)],
                text=True,
            )
            pm = re.search(r"--policy\s+(\S+)", args)
            wm = re.search(r"--thread-watch-prior\s+([0-9.]+)", args)
            if pm:
                stats.current_policy = pm.group(1)
            if wm:
                stats.current_watch = float(wm.group(1))
            stats.current_skill = "--skill-exec" in args
        except (OSError, subprocess.CalledProcessError):
            pass

    if not log_path.is_file():
        return stats

    size = log_path.stat().st_size
    stats.log_bytes = size
    read_from = max(0, size - bootstrap_bytes)
    with log_path.open("rb") as handle:
        handle.seek(read_from)
        text = handle.read().decode("utf-8", errors="replace")

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "Traceback" in stripped or "MATRIX_CELL_FAIL" in stripped:
            stats.error_count += 1
        m = _CELL_START.search(stripped)
        if m:
            stats.current_policy = m.group(1)
            stats.current_watch = float(m.group(2))
            stats.current_skill = m.group(3) == "True"
        m = _CELL_DONE.search(stripped)
        if m:
            stats.last_cell_mean = float(m.group(4))
            if int(m.group(6)) != 0:
                stats.cells_failed += 1
        m = _EVAL_PROGRESS.search(stripped)
        if m:
            stats.eval_episode = int(m.group(1))
            stats.eval_episodes = int(m.group(2))
            stats.last_score = float(m.group(3))
        m = _MATRIX_DONE.search(stripped)
        if m:
            stats.cells_failed = int(m.group(2))

        if any(
            k in stripped
            for k in (
                "MATRIX_",
                "BROWSER_EVAL_PROGRESS",
                "BROWSER_100:",
                "BROWSER_RELIABILITY_DONE",
                "Traceback",
                "MATRIX_CELL_FAIL",
            )
        ):
            if not stats.key_lines or stats.key_lines[-1] != stripped:
                stats.key_lines.append(stripped)
    if len(stats.key_lines) > 24:
        stats.key_lines = stats.key_lines[-24:]

    if stats.matrix_pid:
        try:
            out = subprocess.check_output(
                ["ps", "-o", "etimes=", "-p", str(stats.matrix_pid)],
                text=True,
            ).strip()
            if out.isdigit():
                stats.session_start_ts = time.time() - int(out)
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
    if stats.session_start_ts is None and log_path.exists():
        stats.session_start_ts = log_path.stat().st_mtime - max(60.0, size / 400.0)

    return stats


def estimate_remaining_seconds(stats: MatrixSessionStats) -> float:
    remaining_cells = max(
        0,
        stats.cells_total - stats.cells_complete - (1 if stats.benchmark_running else 0),
    )
    cell_seconds = DEFAULT_MINUTES_PER_CELL * 60.0
    in_cell = 0.0
    if stats.benchmark_running and stats.eval_episode and stats.eval_episodes:
        frac = stats.eval_episode / max(1, stats.eval_episodes)
        in_cell = cell_seconds * (1.0 - frac)
    elif stats.benchmark_running:
        in_cell = cell_seconds * 0.5
    return in_cell + remaining_cells * cell_seconds

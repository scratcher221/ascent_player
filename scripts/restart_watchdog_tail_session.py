#!/usr/bin/env python3
"""Restart skill watchdog for remaining wall time with improved cycle flags."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ascent_player.utils.watchdog_log import session_start_from_log_path  # noqa: E402


def _pgrep(pattern: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
    except subprocess.CalledProcessError:
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _stop_watchdog() -> None:
    for pid in _pgrep("scripts/run_thread_bc_with_watchdog.py"):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    for pat in ("scripts/run_thread_bc_cycle.py", "ms-playwright/chromium"):
        for pid in _pgrep(pat):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    time.sleep(2)


def main() -> int:
    anchor = ROOT / "logs" / "skill_training_watchdog_20260725_152812.log"
    started = session_start_from_log_path(anchor)
    if started is None:
        remaining_h = 2.0
    else:
        end = started + timedelta(hours=8.0)
        remaining_h = max(0.25, (end - datetime.now()).total_seconds() / 3600.0)

    floor_path = ROOT / "logs" / "thread_watch_floor.txt"
    if not floor_path.exists():
        floor_path.write_text("1080.0000\n", encoding="utf-8")
        print(f"Wrote default thread_watch_floor=1080", flush=True)

    extra = [
        "--skip-ceiling",
        "--skip-skill-bootstrap",
        "--bc-steps",
        "750",
        "--bc-lr",
        "2.5e-6",
        "--thread-watch-prior",
        "0.58",
        "--collect-minutes",
        "22",
        "--eval-episodes",
        "10",
    ]

    _stop_watchdog()

    log_path = ROOT / "logs" / f"skill_training_watchdog_{time.strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [
        str(ROOT / ".venv/bin/python"),
        "-u",
        str(ROOT / "scripts/run_thread_bc_with_watchdog.py"),
        "--hours",
        f"{remaining_h:.3f}",
        "--run-seed",
        "424242",
        "--collect-gate",
        "1000",
        "--elite-gate",
        "2000",
        "--target-mean",
        "2000",
        "--target-min",
        "1000",
        "--skill-bootstrap-minutes",
        "15",
        "--collect-minutes",
        "22",
        "--stall-seconds",
        "120",
        *extra,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    print(f"TAIL_RESTART remaining_h={remaining_h:.3f} log={log_path.name}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    print(f"TAIL_RESTART watchdog_pid={proc.pid}", flush=True)

    subprocess.run(["pkill", "-f", "scripts/training_monitor.py"], check=False)
    mon_cmd = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/training_monitor.py"),
        "--log",
        str(log_path),
        "--hours",
        f"{remaining_h:.3f}",
    ]
    subprocess.Popen(
        mon_cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

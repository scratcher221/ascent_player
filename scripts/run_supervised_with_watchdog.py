#!/usr/bin/env python3
"""Run supervised browser finetune with a log/browser-activity watchdog.

Restarts the trainer if no step/episode progress appears for too long.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pgrep(pattern: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
    except subprocess.CalledProcessError:
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _kill_trainer() -> None:
    for pat in (
        "scripts/run_browser_frame_finetune.py",
        "ms-playwright/chromium-1223",
    ):
        for pid in _pgrep(pat):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    time.sleep(2)
    for pat in (
        "scripts/run_browser_frame_finetune.py",
        "ms-playwright/chromium-1223",
    ):
        for pid in _pgrep(pat):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _progress_stamp(log_path: Path) -> tuple[int, str]:
    if not log_path.exists():
        return 0, ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = list(
        re.finditer(
            r"(?:^|\n)(?:step=\d+|episode=\d+|BROWSER_ALIVE|BROWSER_RESET done|LR_CHECK|SEED_THREAD|TRANSFER_EVAL)",
            text,
        )
    )
    last = matches[-1].group(0).strip() if matches else ""
    return len(matches), last


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--run-seed", type=int, default=424242)
    parser.add_argument("--stall-seconds", type=float, default=90.0)
    parser.add_argument("--check-every", type=float, default=20.0)
    args = parser.parse_args()

    deadline = time.time() + max(600.0, args.hours * 3600.0)
    restart = 0
    while time.time() < deadline:
        restart += 1
        remaining = max(0.15, (deadline - time.time()) / 3600.0)
        log_path = ROOT / "logs" / f"supervised_2h_watchdog_{time.strftime('%Y%m%d_%H%M%S')}.log"
        cmd = [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / "scripts/run_browser_frame_finetune.py"),
            "--hours",
            f"{remaining:.3f}",
            "--run-seed",
            str(args.run_seed),
            "--continue-latest",
        ]
        print(
            f"WATCHDOG_START restart={restart} remaining_h={remaining:.3f} log={log_path.name}",
            flush=True,
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        last_count = 0
        last_change = time.time()
        last_msg = ""
        while time.time() < deadline:
            ret = proc.poll()
            count, msg = _progress_stamp(log_path)
            if count > last_count or (msg and msg != last_msg):
                last_count = count
                last_msg = msg
                last_change = time.time()
                print(
                    f"WATCHDOG_PROGRESS pid={proc.pid} events={count} last={msg!r}",
                    flush=True,
                )
            stalled = time.time() - last_change
            if stalled >= args.stall_seconds:
                print(
                    f"WATCHDOG_STALL seconds={stalled:.0f} — killing trainer",
                    flush=True,
                )
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                _kill_trainer()
                break
            if ret is not None:
                print(f"WATCHDOG_EXIT code={ret}", flush=True)
                break
            time.sleep(args.check_every)
        else:
            # deadline
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            _kill_trainer()
            break
        # brief pause before restart
        time.sleep(3)

    print("WATCHDOG_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

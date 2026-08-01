#!/usr/bin/env python3
"""Run thread-BC skill training with log/progress watchdog (restart on stall)."""
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


def _kill_training() -> None:
    for pat in (
        "scripts/run_thread_bc_cycle.py",
        "ms-playwright/chromium",
    ):
        for pid in _pgrep(pat):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    time.sleep(2)
    for pat in (
        "scripts/run_thread_bc_cycle.py",
        "ms-playwright/chromium",
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
    patterns = (
        r"BROWSER_PLAYING",
        r"TRAIN_LOOP first_step",
        r"step=\d+",
        r"episode=\d+",
        r"BROWSER_ALIVE",
        r"BROWSER_RESET done",
        r"COLLECT_DONE",
        r"CEILING_SUMMARY",
        r"OFFLINE_BC",
        r"WATCHDOG_STALL",
    )
    matches = list(
        re.finditer(rf"(?:^|\n)(?:{'|'.join(patterns)})", text)
    )
    last = matches[-1].group(0).strip() if matches else ""
    return len(matches), last


def _menu_stuck(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
    if "BROWSER_PLAYING" in tail or "TRAIN_LOOP first_step" in tail:
        return False
    resets = tail.count("BROWSER_RESET done")
    if resets < 2:
        return False
    if re.search(r"step=\d+", tail):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--run-seed", type=int, default=424242)
    parser.add_argument("--collect-gate", type=float, default=1000.0)
    parser.add_argument("--elite-gate", type=float, default=2000.0)
    parser.add_argument("--target-mean", type=float, default=2000.0)
    parser.add_argument("--target-min", type=float, default=1000.0)
    parser.add_argument("--stall-seconds", type=float, default=120.0)
    parser.add_argument("--check-every", type=float, default=25.0)
    parser.add_argument("--skill-bootstrap-minutes", type=float, default=15.0)
    parser.add_argument("--collect-minutes", type=float, default=20.0)
    args, extra = parser.parse_known_args()

    deadline = time.time() + max(600.0, args.hours * 3600.0)
    restart = 0
    while time.time() < deadline:
        restart += 1
        remaining_h = max(0.15, (deadline - time.time()) / 3600.0)
        log_path = (
            ROOT
            / "logs"
            / f"skill_training_watchdog_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        cmd = [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / "scripts/run_thread_bc_cycle.py"),
            "--hours",
            f"{remaining_h:.3f}",
            "--run-seed",
            str(args.run_seed),
            "--collect-gate",
            str(args.collect_gate),
            "--elite-gate",
            str(args.elite_gate),
            "--target-mean",
            str(args.target_mean),
            "--target-min",
            str(args.target_min),
            "--skill-bootstrap-minutes",
            str(args.skill_bootstrap_minutes),
            "--collect-minutes",
            str(args.collect_minutes),
            *extra,
        ]
        print(
            f"WATCHDOG_START restart={restart} remaining_h={remaining_h:.2f} "
            f"log={log_path.name}",
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
            if stalled >= args.stall_seconds or _menu_stuck(log_path):
                reason = (
                    "menu_stuck"
                    if _menu_stuck(log_path)
                    else f"stall_{stalled:.0f}s"
                )
                print(
                    f"WATCHDOG_STALL reason={reason} — killing trainer",
                    flush=True,
                )
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                _kill_training()
                break
            if ret is not None:
                print(f"WATCHDOG_EXIT code={ret}", flush=True)
                if ret != 0:
                    _kill_training()
                break
            time.sleep(args.check_every)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            _kill_training()
            break
        time.sleep(5)

    print("WATCHDOG_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

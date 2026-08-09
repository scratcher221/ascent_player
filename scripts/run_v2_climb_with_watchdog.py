#!/usr/bin/env python3
"""Run Impala-mid v2 climb ladder with log/progress watchdog (restart on stall)."""
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
        "scripts/run_v2_climb_ladder.py",
        "ms-playwright/chromium",
        "chrome_crashpad_handler",
    ):
        for pid in _pgrep(pat):
            # Never kill this watchdog process.
            if pat.endswith("run_v2_climb_ladder.py") and "with_watchdog" in (
                open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(
                    "utf-8", "replace"
                )
                if Path(f"/proc/{pid}/cmdline").exists()
                else ""
            ):
                continue
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    time.sleep(2)
    for pat in (
        "scripts/run_v2_climb_ladder.py",
        "ms-playwright/chromium",
    ):
        for pid in _pgrep(pat):
            cmdline = ""
            try:
                cmdline = (
                    open(f"/proc/{pid}/cmdline", "rb")
                    .read()
                    .replace(b"\0", b" ")
                    .decode("utf-8", "replace")
                )
            except OSError:
                pass
            if "with_watchdog" in cmdline:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _progress_stamp(log_path: Path) -> tuple[int, str]:
    if not log_path.exists():
        return 0, ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = (
        r"CLIMB_ROUND",
        r"CLIMB_BC step=",
        r"CLIMB_TD step=",
        r"CLIMB_TD_FAIL",
        r"CLIMB_POLICY_COLLECT",
        r"CLIMB_PROBE",
        r"CLIMB_RELIABILITY",
        r"CLIMB_BEST_UPDATE",
        r"BROWSER_PLAYING",
        r"TRAIN_LOOP first_step",
        r"BROWSER_RESET done",
        r"COLLECT_DONE",
        r"episode=\d+",
        r"OPTIMIZER_REBUILD",
        r"WATCHDOG_STALL",
        r"MODEL_BUILD variant=impala_mid",
    )
    matches = list(re.finditer(rf"(?:^|\n)(?:{'|'.join(patterns)})", text))
    last = matches[-1].group(0).strip() if matches else ""
    return len(matches), last


def _menu_stuck(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
    if "TEACHER_WARMSTART begin" in tail and "TEACHER_WARMSTART done" not in tail:
        # Warmstart should be disabled for climb; treat as hard stall.
        return "TRAIN_LOOP first_step" not in tail and "BROWSER_PLAYING" in tail
    if "TRAIN_LOOP first_step" in tail or re.search(r"episode=\d+", tail):
        return False
    if "BROWSER_PLAYING" in tail and "BROWSER_RESET done" in tail:
        # Playing started but no train step yet — allow short grace via stall timer.
        return False
    resets = tail.count("BROWSER_RESET done")
    if resets < 2:
        return False
    if re.search(r"step=\d+", tail):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--run-seed", type=int, default=424242)
    parser.add_argument("--target-mean", type=float, default=2000.0)
    parser.add_argument("--stall-seconds", type=float, default=180.0)
    parser.add_argument("--check-every", type=float, default=20.0)
    parser.add_argument("--bc-steps", type=int, default=800)
    parser.add_argument("--td-steps", type=int, default=400)
    parser.add_argument(
        "--browser-bc-steps",
        type=int,
        default=0,
        help="Forwarded: BC on fresh browser replay after collect.",
    )
    parser.add_argument("--collect-minutes", type=float, default=12.0)
    parser.add_argument("--probe-episodes", type=int, default=8)
    parser.add_argument("--reliability-episodes", type=int, default=100)
    parser.add_argument("--reliability-every", type=int, default=4)
    parser.add_argument(
        "--bootstrap-keep-floor",
        type=float,
        default=350.0,
        help="Forwarded to climb ladder for early Impala keep gate.",
    )
    args, extra = parser.parse_known_args()

    deadline = time.time() + max(600.0, args.hours * 3600.0)
    restart = 0
    while time.time() < deadline:
        restart += 1
        remaining_h = max(0.15, (deadline - time.time()) / 3600.0)
        log_path = (
            ROOT
            / "logs"
            / f"v2_climb_watchdog_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        (ROOT / "logs" / "v2_climb_ladder_latest.path").write_text(
            str(log_path.relative_to(ROOT)) + "\n",
            encoding="utf-8",
        )
        cmd = [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / "scripts/run_v2_climb_ladder.py"),
            "--hours",
            f"{remaining_h:.3f}",
            "--run-seed",
            str(args.run_seed),
            "--target-mean",
            str(args.target_mean),
            "--bc-steps",
            str(args.bc_steps),
            "--td-steps",
            str(args.td_steps),
            "--browser-bc-steps",
            str(args.browser_bc_steps),
            "--collect-minutes",
            str(args.collect_minutes),
            "--probe-episodes",
            str(args.probe_episodes),
            "--reliability-episodes",
            str(args.reliability_episodes),
            "--reliability-every",
            str(args.reliability_every),
            "--bootstrap-keep-floor",
            str(args.bootstrap_keep_floor),
            *extra,
        ]
        print(
            f"WATCHDOG_START restart={restart} remaining_h={remaining_h:.2f} "
            f"log={log_path.name} variant=impala_mid",
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
        (ROOT / "logs" / "v2_climb_ladder.pid").write_text(
            f"{proc.pid}\n", encoding="utf-8"
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
            teacher_stuck = False
            if log_path.exists():
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
                if (
                    "TEACHER_WARMSTART begin" in tail
                    and "TEACHER_WARMSTART done" not in tail
                    and stalled >= min(90.0, args.stall_seconds)
                ):
                    teacher_stuck = True
            if stalled >= args.stall_seconds or _menu_stuck(log_path) or teacher_stuck:
                reason = (
                    "teacher_warmstart_stuck"
                    if teacher_stuck
                    else (
                        "menu_stuck"
                        if _menu_stuck(log_path)
                        else f"stall_{stalled:.0f}s"
                    )
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

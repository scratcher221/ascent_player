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


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Terminate only the trainer session started by this watchdog."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
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
    sys.path.insert(0, str(ROOT))
    from climb_args import add_climb_args

    parser = argparse.ArgumentParser(description=__doc__)
    add_climb_args(parser)
    parser.add_argument("--stall-seconds", type=float, default=180.0)
    parser.add_argument("--check-every", type=float, default=20.0)
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
            "--eval-seed",
            str(args.eval_seed),
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
            "--collect-eps",
            str(args.collect_eps),
            "--probe-episodes",
            str(args.probe_episodes),
            "--confirm-episodes",
            str(args.confirm_episodes),
            "--reliability-episodes",
            str(args.reliability_episodes),
            "--reliability-every",
            str(args.reliability_every),
            "--bootstrap-keep-floor",
            str(args.bootstrap_keep_floor),
            "--bc-lr",
            str(args.bc_lr),
            "--td-lr",
            str(args.td_lr),
            *(["--lock-run-seed"] if args.lock_run_seed else []),
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
                start_new_session=True,
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
                    f"WATCHDOG_STALL reason={reason} — killing trainer session",
                    flush=True,
                )
                _kill_process_group(proc)
                break
            if ret is not None:
                print(f"WATCHDOG_EXIT code={ret}", flush=True)
                if ret != 0:
                    _kill_process_group(proc)
                break
            time.sleep(args.check_every)
        else:
            _kill_process_group(proc)
            break
        time.sleep(5)

    print("WATCHDOG_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

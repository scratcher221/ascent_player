#!/usr/bin/env python3
"""Wait for watchdog training to finish, analyze, benchmark, start next 8h run.

Example (attach to current session, run in background):
  cd "/home/david/Projekte/AI Projects/Ascent Player"
  PYTHONPATH=. nohup .venv/bin/python -u scripts/run_post_session_chain.py \\
    --wait-log logs/skill_training_watchdog_20260725_152812.log \\
    > logs/post_session_chain.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ascent_player.utils.session_analysis import (  # noqa: E402
    analyze_session_logs,
    analyze_watchdog_log,
    watchdog_logs_since,
    write_report,
)
from ascent_player.utils.watchdog_log import (  # noqa: E402
    session_start_from_log_path,
    watchdog_running,
)


def _wait_for_session(log_path: Path | None, poll_s: float) -> None:
    label = log_path if log_path else "any_watchdog"
    print(f"CHAIN_WAIT log={label}", flush=True)
    while watchdog_running():
        time.sleep(poll_s)
    for _ in range(12):
        if log_path is None or not log_path.exists():
            break
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        if "WATCHDOG_DONE" in tail or "THREAD_BC_CYCLE_END" in tail:
            break
        time.sleep(10)
    print("CHAIN_WAIT done", flush=True)


def _analysis_for_session(
    log_dir: Path,
    anchor_log: Path,
    chain_started: float,
) -> SessionAnalysis:
    anchor_start = session_start_from_log_path(anchor_log)
    since_ts = (
        anchor_start.timestamp()
        if anchor_start is not None
        else min(chain_started, anchor_log.stat().st_mtime)
    )
    paths = watchdog_logs_since(log_dir, since_ts)
    if not paths:
        paths = [anchor_log]
    return analyze_session_logs(paths)


def _run_benchmark(
    *,
    episodes: int,
    run_seed: int,
    report_path: Path,
) -> int:
    cmd = [
        str(ROOT / ".venv/bin/python"),
        "-u",
        str(ROOT / "scripts/run_browser_reliability_100.py"),
        "--policy",
        "thread_bc",
        "--episodes",
        str(episodes),
        "--run-seed",
        str(run_seed),
        "--thread-watch-prior",
        "0.58",
        "--skill-exec",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    print(f"CHAIN_BENCHMARK {' '.join(cmd)}", flush=True)
    with report_path.open("a", encoding="utf-8") as log:
        log.write(f"\n--- benchmark episodes={episodes} ---\n")
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(proc.returncode)


def _start_watchdog(
    *,
    hours: float,
    run_seed: int,
    elite_gate: float,
    target_mean: float,
    target_min: float,
    extra_cycle_args: list[str],
) -> tuple[int, Path]:
    log_path = ROOT / "logs" / f"skill_training_watchdog_{time.strftime('%Y%m%d_%H%M%S')}.log"
    cmd = [
        str(ROOT / ".venv/bin/python"),
        "-u",
        str(ROOT / "scripts/run_thread_bc_with_watchdog.py"),
        "--hours",
        str(hours),
        "--run-seed",
        str(run_seed),
        "--collect-gate",
        "1000",
        "--elite-gate",
        str(elite_gate),
        "--target-mean",
        str(target_mean),
        "--target-min",
        str(target_min),
        "--skill-bootstrap-minutes",
        "15",
        "--collect-minutes",
        "22",
        "--stall-seconds",
        "120",
        *extra_cycle_args,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    print(f"CHAIN_START {' '.join(cmd)} log={log_path.name}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return proc.pid, log_path


def _start_monitor(log_path: Path, hours: float) -> int | None:
    cmd = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/training_monitor.py"),
        "--log",
        str(log_path),
        "--hours",
        str(hours),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("DISPLAY", os.environ.get("DISPLAY", ":0"))
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return proc.pid
    except OSError as exc:
        print(f"CHAIN_MONITOR_SKIP {exc}", flush=True)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wait-log",
        type=Path,
        default=None,
        help="Watchdog log to wait on (default: active log)",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--benchmark-episodes", type=int, default=25)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--run-seed", type=int, default=424242)
    parser.add_argument("--elite-gate", type=float, default=2000.0)
    parser.add_argument("--target-mean", type=float, default=2000.0)
    parser.add_argument("--target-min", type=float, default=1000.0)
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Only analyze log and restart training",
    )
    parser.add_argument(
        "--skip-restart",
        action="store_true",
        help="Analyze (+ benchmark) only",
    )
    parser.add_argument(
        "--skip-follow-up-wait",
        action="store_true",
        help="After starting the next 8h run, exit instead of waiting for it to finish",
    )
    args = parser.parse_args()
    chain_started = time.time()

    log_path = args.wait_log
    if log_path is None:
        from ascent_player.utils.watchdog_log import find_active_watchdog_log

        found = find_active_watchdog_log(ROOT / "logs")
        if found is None:
            print("No active watchdog log", file=sys.stderr)
            return 1
        log_path = found
    elif not log_path.is_absolute():
        log_path = (ROOT / log_path).resolve()

    _wait_for_session(log_path, args.poll_seconds)

    log_dir = ROOT / "logs"
    analysis = _analysis_for_session(log_dir, log_path, chain_started)
    report_json = ROOT / "logs" / f"post_session_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_report(analysis, report_json)
    print(f"CHAIN_ANALYSIS report={report_json}", flush=True)
    for line in analysis.recommendations:
        print(f"CHAIN_NOTE {line}", flush=True)

    bench_log = ROOT / "logs" / f"post_session_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.log"
    if not args.skip_benchmark:
        rc = _run_benchmark(
            episodes=args.benchmark_episodes,
            run_seed=args.run_seed,
            report_path=bench_log,
        )
        print(f"CHAIN_BENCHMARK_DONE rc={rc} log={bench_log}", flush=True)

    if args.skip_restart:
        return 0

    # Stop old monitor if still running (optional).
    subprocess.run(
        ["pkill", "-f", "scripts/training_monitor.py"],
        check=False,
    )

    wd_pid, new_log = _start_watchdog(
        hours=args.hours,
        run_seed=args.run_seed,
        elite_gate=args.elite_gate,
        target_mean=args.target_mean,
        target_min=args.target_min,
        extra_cycle_args=analysis.watchdog_extra_args,
    )
    mon_pid = _start_monitor(new_log, args.hours)
    print(
        f"CHAIN_RESTART watchdog_pid={wd_pid} log={new_log} "
        f"monitor_pid={mon_pid} extra={analysis.watchdog_extra_args}",
        flush=True,
    )

    if args.skip_follow_up_wait:
        print("CHAIN_EXIT after restart (skip follow-up wait)", flush=True)
        return 0

    follow_started = time.time()
    _wait_for_session(new_log, args.poll_seconds)
    follow_analysis = _analysis_for_session(log_dir, new_log, follow_started)
    follow_report = (
        ROOT / "logs" / f"post_session_followup_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    write_report(follow_analysis, follow_report)
    print(f"CHAIN_FOLLOWUP_COMPLETE report={follow_report}", flush=True)
    for line in follow_analysis.recommendations:
        print(f"CHAIN_FOLLOWUP_NOTE {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

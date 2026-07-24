#!/usr/bin/env python3
"""Durable 9h supervision helper for seed-thread continue runs.

Checks watchdog/trainer health, summarizes scores/evals, and auto-restarts
if the durable training tree dies. Does not change hyperparameters by itself.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "logs" / "supervise_9h_status.txt"
REPORT = ROOT / "logs" / "supervise_9h_monitor.log"
PID_FILE = ROOT / "logs" / "watchdog_supervisor.pid"
HOURS = 9.0
CHECK_EVERY = 20 * 60  # 20 minutes


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with REPORT.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _alive(pid: int) -> bool:
    return pid > 0 and os.path.exists(f"/proc/{pid}")


def _find_procs() -> tuple[int | None, int | None]:
    wd = None
    tr = None
    if PID_FILE.exists():
        try:
            wd = int(PID_FILE.read_text().strip())
        except ValueError:
            wd = None
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read()
        except OSError:
            continue
        if b"scripts/run_supervised_with_watchdog.py" in cmd:
            wd = int(pid)
        if b"scripts/run_browser_frame_finetune.py" in cmd:
            tr = int(pid)
    return wd, tr


def _latest_train_log() -> Path | None:
    logs = sorted(
        (ROOT / "logs").glob("supervised_2h_watchdog_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return logs[0] if logs else None


def _summarize(log: Path) -> str:
    text = log.read_text(encoding="utf-8", errors="replace")
    eps = [
        float(m.group(1))
        for m in re.finditer(
            r"episode=\d+ reward=[-\d.]+ score=([\d.]+)", text
        )
    ]
    evals = re.findall(
        r"TRANSFER_EVAL_RESULT mean=([\d.]+) min=([\d.]+) max=([\d.]+)", text
    )
    threads = len(re.findall(r"SEED_THREAD ", text))
    skips = len(re.findall(r"REPLAY_GATE skip", text))
    losses = [float(m.group(1)) for m in re.finditer(r"loss=([0-9.]+)", text)]
    nones = len(re.findall(r"loss=None", text))
    parts = [f"log={log.name}", f"eps={len(eps)}", f"thread_logs={threads}", f"gate_skips={skips}"]
    if eps:
        recent = eps[-30:]
        parts.append(
            f"score_mean_all={sum(eps)/len(eps):.1f} "
            f"last30={sum(recent)/len(recent):.1f} "
            f"max={max(eps):.0f}"
        )
    if evals:
        m, mn, mx = evals[-1]
        parts.append(f"last_eval mean={m} min={mn} max={mx} n_eval={len(evals)}")
    parts.append(f"loss_n={len(losses)} loss_none={nones}")
    if losses:
        parts.append(f"loss_last={losses[-1]:.3f}")
    return " | ".join(parts)


def _restart() -> int:
    import shutil

    shutil.copy2(
        ROOT / "checkpoints" / "seed_map_best.keras",
        ROOT / "checkpoints" / "dqn_latest.keras",
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    super_log = ROOT / "logs" / f"watchdog_supervisor_nohup_{stamp}.log"
    cmd = [
        str(ROOT / ".venv/bin/python"),
        "-u",
        str(ROOT / "scripts/run_supervised_with_watchdog.py"),
        "--hours",
        "9",
        "--run-seed",
        "424242",
        "--stall-seconds",
        "120",
        "--check-every",
        "15",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    with super_log.open("wb") as out:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    PID_FILE.write_text(f"{proc.pid}\n")
    _log(f"RESTARTED pid={proc.pid} log={super_log.name}")
    return proc.pid


def main() -> int:
    deadline = time.time() + HOURS * 3600.0
    _log(f"MONITOR_START hours={HOURS} check_every_s={CHECK_EVERY}")
    while time.time() < deadline:
        wd, tr = _find_procs()
        train = _latest_train_log()
        if not _alive(wd or 0):
            _log(f"WATCHDOG_DEAD wd={wd} trainer={tr} — restarting")
            _restart()
        elif tr is None:
            _log(f"TRAINER_MISSING wd={wd} — waiting one cycle")
        else:
            summary = _summarize(train) if train else "no_train_log"
            _log(f"OK wd={wd} tr={tr} {summary}")
        # Sleep in chunks so we can exit near deadline.
        left = deadline - time.time()
        if left <= 0:
            break
        time.sleep(min(CHECK_EVERY, left))
    _log("MONITOR_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

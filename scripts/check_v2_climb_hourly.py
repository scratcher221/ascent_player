#!/usr/bin/env python3
"""Hourly productivity check for the Impala-mid v2 climb session.

Exits 0 always; prints a short verdict and appends JSON to
logs/v2_climb_hourly_checks.jsonl for the overnight agent loop.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
OUT = LOG_DIR / "v2_climb_hourly_checks.jsonl"
LATEST_PATH = LOG_DIR / "v2_climb_ladder_latest.path"
SUP_PATH = LOG_DIR / "v2_climb_watchdog_supervisor.path"
V2_BEST = LOG_DIR / "v2_probe_best_mean.txt"
JSONL = LOG_DIR / "v2_climb_ladder.jsonl"


def _pgrep(pattern: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
    except subprocess.CalledProcessError:
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _read_path(pointer: Path) -> Path | None:
    if not pointer.exists():
        return None
    rel = pointer.read_text(encoding="utf-8").strip().splitlines()[0]
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    return path if path.exists() else None


def _tail_text(path: Path, nbytes: int = 120_000) -> str:
    data = path.read_bytes()
    if len(data) > nbytes:
        data = data[-nbytes:]
    return data.decode("utf-8", errors="replace")


def main() -> int:
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    trainer = _pgrep("scripts/run_v2_climb_ladder.py")
    watchdog = _pgrep("scripts/run_v2_climb_with_watchdog.py")
    log = _read_path(LATEST_PATH)
    sup = _read_path(SUP_PATH)

    v2_best = 0.0
    if V2_BEST.exists():
        try:
            v2_best = float(V2_BEST.read_text(encoding="utf-8").strip())
        except ValueError:
            pass

    probes: list[float] = []
    reverts = 0
    keeps = 0
    rounds = 0
    last_probe = None
    last_bc_loss = None
    last_event_ts = None
    now_dt = datetime.now()
    if JSONL.exists():
        for line in JSONL.read_text(encoding="utf-8").splitlines()[-80:]:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = str(row.get("ts") or "")
            try:
                row_dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                continue
            # Overnight productivity: only the last 4 hours of events.
            if (now_dt - row_dt).total_seconds() > 4 * 3600:
                continue
            if row.get("event") == "probe":
                probes.append(float(row.get("probe", 0)))
                keeps += 1
                last_probe = float(row.get("probe", 0))
                last_event_ts = row.get("ts")
            elif row.get("event") == "revert":
                reverts += 1
                probes.append(float(row.get("probe", 0)))
                last_probe = float(row.get("probe", 0))
                last_event_ts = row.get("ts")
            if "bc_loss" in row and row["bc_loss"] is not None:
                last_bc_loss = float(row["bc_loss"])
            if "round" in row:
                rounds = max(rounds, int(row["round"]))

    log_age_s = None
    markers: dict[str, int] = {}
    recent_tail = ""
    if log is not None:
        log_age_s = time.time() - log.stat().st_mtime
        text = _tail_text(log)
        recent_tail = "\n".join(text.splitlines()[-12:])
        for key in (
            "CLIMB_PROBE_GREEDY",
            "CLIMB_V2_BEST",
            "CLIMB_REVERT",
            "CLIMB_BC step=",
            "CLIMB_COLLECT_SKIP",
            "CLIMB_POLICY_COLLECT",
            "CLIMB_TD_FAIL",
            "Traceback",
            "OPTIMIZER_REBUILD",
        ):
            markers[key] = text.count(key)
        m = re.findall(r"CLIMB_PROBE_GREEDY mean=([0-9.]+)", text)
        if m:
            last_probe = float(m[-1])
        m = re.findall(r"CLIMB_BC step=\d+/\d+ loss=([0-9.]+)", text)
        if m:
            last_bc_loss = float(m[-1])
        m = re.findall(r"CLIMB_ROUND\s+(\d+)", text)
        if m:
            rounds = max(rounds, int(m[-1]))

    stalled = bool(log_age_s is not None and log_age_s > 900 and trainer)
    dead = not trainer or not watchdog
    productive = False
    issues: list[str] = []
    actions: list[str] = []

    if dead:
        issues.append("trainer_or_watchdog_not_running")
        actions.append("restart_9h_watchdog")
    if stalled:
        issues.append(f"log_stale_{log_age_s:.0f}s")
        actions.append("inspect_browser_or_restart")
    if markers.get("Traceback", 0) or markers.get("CLIMB_TD_FAIL", 0):
        issues.append("errors_in_log_tail")
    if reverts > keeps + 2 and rounds >= 3:
        issues.append("revert_loop")
        actions.append("raise_bootstrap_keep_or_offline_bc")
    if last_probe is not None and last_probe < 300:
        issues.append("probe_collapsed")
        actions.append("keep_offline_bc_reduce_lr")
    if v2_best >= 539.7 and keeps > 0:
        productive = True
    if markers.get("CLIMB_PROBE_GREEDY", 0) > 0 and not dead:
        productive = True
    if markers.get("CLIMB_BC step=", 0) > 0 and log_age_s is not None and log_age_s < 300:
        productive = True

    verdict = "OK" if (not dead and not stalled and not issues) else (
        "NEEDS_ACTION" if dead or stalled or "revert_loop" in issues else "WATCH"
    )
    if productive and verdict == "WATCH" and not dead:
        verdict = "OK_PROGRESS"

    report = {
        "ts": now,
        "verdict": verdict,
        "productive": productive,
        "trainer_pids": trainer,
        "watchdog_pids": watchdog,
        "log": str(log) if log else None,
        "supervisor": str(sup) if sup else None,
        "log_age_s": log_age_s,
        "v2_best": v2_best,
        "last_probe": last_probe,
        "last_bc_loss": last_bc_loss,
        "rounds": rounds,
        "keeps": keeps,
        "reverts": reverts,
        "markers": markers,
        "issues": issues,
        "suggested_actions": actions,
        "last_event_ts": last_event_ts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True) + "\n")

    age_txt = f"{log_age_s:.0f}" if log_age_s is not None else "-1"
    print(
        f"HOURLY_CHECK verdict={verdict} productive={int(productive)} "
        f"v2_best={v2_best:.1f} last_probe={last_probe} "
        f"rounds={rounds} keeps={keeps} reverts={reverts} "
        f"log_age_s={age_txt} "
        f"issues={issues}",
        flush=True,
    )
    if recent_tail.strip():
        print("--- log tail ---", flush=True)
        print(recent_tail[-1500:], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

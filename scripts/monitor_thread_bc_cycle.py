#!/usr/bin/env python3
"""Lightweight monitor for the 7h thread-BC cycle (logs heartbeats + key events)."""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path


KEYS = (
    "THREAD_BC_CYCLE_START",
    "THREAD_CEILING",
    "CEILING_SUMMARY",
    "CYCLE_ROUND",
    "PHASE_",
    "COLLECT_DONE",
    "ELITE_REPLAY",
    "OFFLINE_BC",
    "EVAL_GREEDY",
    "EVAL_ALIGNED",
    "SEED_BEST_UPDATE",
    "NO_PROMOTE",
    "STARVE_ADJUST",
    "TARGET_MET",
    "THREAD_BC_CYCLE_END",
    "BROWSER_ALIVE",
    "TRANSFER_EVAL",
    "error",
    "Error",
    "Traceback",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=7.0)
    parser.add_argument("--every", type=float, default=600.0)
    args = parser.parse_args()

    deadline = time.time() + max(300.0, args.hours * 3600.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    with args.out.open("a", encoding="utf-8") as out:
        out.write(f"MONITOR_START log={args.log} hours={args.hours}\n")
        out.flush()
        while time.time() < deadline:
            if args.log.exists():
                data = args.log.read_text(errors="ignore")
                chunk = data[offset:]
                offset = len(data)
                for line in chunk.splitlines():
                    if any(k in line for k in KEYS):
                        out.write(line + "\n")
                # Heartbeat summary
                means = re.findall(r"EVAL_ALIGNED mean=([0-9.]+)", data)
                greedy = re.findall(r"EVAL_GREEDY mean=([0-9.]+)", data)
                ceil = re.findall(r"CEILING_SUMMARY mean=([0-9.]+)", data)
                updates = re.findall(r"SEED_BEST_UPDATE mean=([0-9.]+)", data)
                out.write(
                    f"HEARTBEAT t={time.strftime('%H:%M:%S')} "
                    f"ceil={ceil[-1] if ceil else '-'} "
                    f"aligned={means[-1] if means else '-'} "
                    f"greedy={greedy[-1] if greedy else '-'} "
                    f"promotions={len(updates)} "
                    f"log_bytes={len(data)}\n"
                )
                out.flush()
            else:
                out.write(f"WAITING_FOR_LOG {args.log}\n")
                out.flush()
            time.sleep(max(30.0, float(args.every)))
        out.write("MONITOR_DONE\n")
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

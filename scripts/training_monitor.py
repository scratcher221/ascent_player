#!/usr/bin/env python3
"""Live UI for watchdog thread-BC training (elapsed time + log stats).

Example:
  PYTHONPATH=. .venv/bin/python scripts/training_monitor.py
  PYTHONPATH=. .venv/bin/python scripts/training_monitor.py \\
      --log logs/skill_training_watchdog_20260725_152812.log --hours 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Watchdog/cycle log (default: newest active skill_training_watchdog_*.log)",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Session budget in hours (default: from running watchdog --hours)",
    )
    parser.add_argument("--refresh-ms", type=int, default=1500)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Monitor run_reliability_matrix.py (auto if matrix process running)",
    )
    parser.add_argument(
        "--cells-total",
        type=int,
        default=11,
        help="Total matrix cells (full Phase-A grid)",
    )
    args = parser.parse_args()

    log_dir = ROOT / "logs"
    import subprocess

    matrix_running = bool(
        subprocess.run(
            ["pgrep", "-f", "scripts/run_reliability_matrix.py"],
            capture_output=True,
        ).stdout.strip()
    )
    use_matrix = args.matrix or (
        args.log is None and matrix_running and not subprocess.run(
            ["pgrep", "-f", "scripts/run_thread_bc_with_watchdog.py"],
            capture_output=True,
        ).stdout.strip()
    )

    if use_matrix:
        log_path = args.log or (log_dir / "reliability_matrix_run.log")
        if not log_path.is_absolute():
            log_path = (ROOT / log_path).resolve()
        if not log_path.exists():
            print(f"Log not found: {log_path}", file=sys.stderr)
            return 1
        from ascent_player.ui.training_monitor import MatrixMonitorWindow

        app = QApplication(sys.argv)
        window = MatrixMonitorWindow(
            log_path=log_path,
            cells_total=args.cells_total,
            refresh_ms=args.refresh_ms,
        )
        window.show()
        return app.exec()

    log_path = args.log
    if log_path is None:
        from ascent_player.utils.watchdog_log import find_active_watchdog_log

        log_path = find_active_watchdog_log(log_dir)
        if log_path is None:
            print("No skill_training_watchdog_*.log found under logs/", file=sys.stderr)
            return 1
    elif not log_path.is_absolute():
        log_path = (ROOT / log_path).resolve()

    if not log_path.exists():
        print(f"Log not found: {log_path}", file=sys.stderr)
        return 1

    from ascent_player.ui.training_monitor import TrainingMonitorWindow

    app = QApplication(sys.argv)
    window = TrainingMonitorWindow(
        log_path=log_path,
        hours=args.hours,
        refresh_ms=args.refresh_ms,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

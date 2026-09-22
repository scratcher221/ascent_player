#!/usr/bin/env python3
"""Live UI for watchdog / v2-climb training (elapsed, remaining, log stats).

Example:
  PYTHONPATH=. .venv/bin/python scripts/training_monitor.py
  PYTHONPATH=. .venv/bin/python scripts/training_monitor.py \\
      --log logs/v2_climb_watchdog_20260801_165247.log --hours 6
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
        help="Trainer/watchdog log (default: newest active climb or thread-BC log)",
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

    def _running(pattern: str) -> bool:
        return bool(
            subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
            ).stdout.strip()
        )

    matrix_running = _running("scripts/run_reliability_matrix.py")
    climb_or_thread = _running("scripts/run_v2_climb_with_watchdog.py") or _running(
        "scripts/run_thread_bc_with_watchdog.py"
    )
    use_matrix = args.matrix or (args.log is None and matrix_running and not climb_or_thread)

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
        from PyQt6.QtCore import QTimer
        from ascent_player.env.desktop_workspace import pin_current_process_to_workspace

        QTimer.singleShot(400, pin_current_process_to_workspace)
        return app.exec()

    log_path = args.log
    if log_path is None:
        from ascent_player.utils.watchdog_log import find_active_watchdog_log

        log_path = find_active_watchdog_log(log_dir)
        if log_path is None:
            print(
                "No active training log found under logs/ "
                "(expected v2_climb_watchdog_*.log or skill_training_watchdog_*.log)",
                file=sys.stderr,
            )
            return 1
    elif not log_path.is_absolute():
        log_path = (ROOT / log_path).resolve()

    if not log_path.exists():
        print(f"Log not found: {log_path}", file=sys.stderr)
        return 1

    from ascent_player.ui.training_monitor import TrainingMonitorWindow
    from ascent_player.utils.watchdog_log import parse_watchdog_hours_from_ps

    hours = args.hours if args.hours is not None else parse_watchdog_hours_from_ps()
    print(f"Monitoring {log_path}" + (f" ({hours:g}h)" if hours else ""), flush=True)

    app = QApplication(sys.argv)
    window = TrainingMonitorWindow(
        log_path=log_path,
        hours=hours,
        refresh_ms=args.refresh_ms,
    )
    window.show()
    from PyQt6.QtCore import QTimer
    from ascent_player.env.desktop_workspace import pin_current_process_to_workspace

    QTimer.singleShot(400, pin_current_process_to_workspace)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

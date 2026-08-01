"""PyQt window for live watchdog / thread-BC training session stats."""
from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from ascent_player.ui.widgets import APP_STYLESHEET
from ascent_player.utils.watchdog_log import (
    LogTailReader,
    SessionStats,
    format_duration,
    parse_watchdog_hours_from_ps,
    watchdog_running,
)


def _fmt_opt(value: float | int | None, *, digits: int = 1) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


class TrainingMonitorWindow(QMainWindow):
    def __init__(
        self,
        log_path: Path,
        hours: float | None = None,
        refresh_ms: int = 1500,
    ) -> None:
        super().__init__()
        self._log_path = log_path
        self._hours = hours if hours is not None else parse_watchdog_hours_from_ps() or 8.0
        self._reader = LogTailReader(path=log_path)
        self._reader.stats.hours = self._hours

        self.setWindowTitle("Ascent — Training Monitor")
        self.setMinimumSize(520, 520)
        self.setStyleSheet(APP_STYLESHEET)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self._log_label = QLabel()
        self._log_label.setWordWrap(True)
        self._log_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._log_label)

        time_box = QGroupBox("Session time")
        time_form = QFormLayout(time_box)
        self._elapsed_lbl = QLabel("—")
        self._remaining_lbl = QLabel("—")
        self._deadline_lbl = QLabel("—")
        self._watchdog_lbl = QLabel("—")
        time_form.addRow("Elapsed", self._elapsed_lbl)
        time_form.addRow("Remaining", self._remaining_lbl)
        time_form.addRow(f"Budget ({self._hours:g}h)", self._deadline_lbl)
        time_form.addRow("Watchdog", self._watchdog_lbl)
        layout.addWidget(time_box)

        stats_box = QGroupBox("Training stats")
        stats_form = QFormLayout(stats_box)
        self._cycle_lbl = QLabel("—")
        self._episode_lbl = QLabel("—")
        self._collect_lbl = QLabel("—")
        self._eval_lbl = QLabel("—")
        self._phase_lbl = QLabel("—")
        self._wd_prog_lbl = QLabel("—")
        self._errors_lbl = QLabel("0")
        stats_form.addRow("Cycle", self._cycle_lbl)
        stats_form.addRow("Episode", self._episode_lbl)
        stats_form.addRow("Collect", self._collect_lbl)
        stats_form.addRow("Eval / ceiling", self._eval_lbl)
        stats_form.addRow("Phase", self._phase_lbl)
        stats_form.addRow("Watchdog progress", self._wd_prog_lbl)
        stats_form.addRow("Errors flagged", self._errors_lbl)
        layout.addWidget(stats_box)

        events_box = QGroupBox("Recent log events")
        events_layout = QVBoxLayout(events_box)
        self._events = QPlainTextEdit()
        self._events.setReadOnly(True)
        self._events.setMaximumBlockCount(200)
        self._events.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        events_layout.addWidget(self._events)
        layout.addWidget(events_box, stretch=1)

        foot = QHBoxLayout()
        self._status = QLabel("")
        foot.addWidget(self._status)
        foot.addStretch()
        layout.addLayout(foot)

        self._timer = QTimer(self)
        self._timer.setInterval(refresh_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._tick()

    def _tick(self) -> None:
        stats = self._reader.refresh()
        self._render(stats)

    def _render(self, stats: SessionStats) -> None:
        self._log_label.setText(f"Log: {self._log_path}  ({stats.log_bytes:,} bytes)")

        start = stats.session_start
        if start is not None:
            elapsed = time.time() - start.timestamp()
            budget = self._hours * 3600.0
            remaining = max(0.0, budget - elapsed)
            self._elapsed_lbl.setText(format_duration(elapsed))
            self._remaining_lbl.setText(format_duration(remaining))
            self._deadline_lbl.setText(
                f"started {start.strftime('%Y-%m-%d %H:%M:%S')} local"
            )
        else:
            self._elapsed_lbl.setText("—")
            self._remaining_lbl.setText("—")
            self._deadline_lbl.setText("unknown start (name not timestamped)")

        running = watchdog_running()
        pid = stats.watchdog_trainer_pid
        self._watchdog_lbl.setText(
            f"{'running' if running else 'not detected'}"
            + (f"  trainer pid {pid}" if pid else "")
        )

        cycle = _fmt_opt(stats.cycle_round, digits=0)
        if stats.cycle_remaining_h is not None:
            cycle += f"  (cycle remaining {stats.cycle_remaining_h:.2f}h)"
        self._cycle_lbl.setText(cycle)

        if stats.episode is not None:
            self._episode_lbl.setText(
                f"#{stats.episode}  score={_fmt_opt(stats.episode_score, digits=0)}  "
                f"reward={_fmt_opt(stats.episode_reward)}  ε={_fmt_opt(stats.epsilon, digits=3)}"
            )
        else:
            self._episode_lbl.setText("—")

        if stats.collect_recent_avg is not None:
            self._collect_lbl.setText(
                f"recent_avg={stats.collect_recent_avg:.0f}  replay={stats.collect_replay or 0}"
            )
        else:
            self._collect_lbl.setText("—")

        parts: list[str] = []
        if stats.ceiling_mean is not None:
            parts.append(f"ceiling={stats.ceiling_mean:.0f}")
        if stats.eval_aligned_mean is not None:
            parts.append(f"aligned={stats.eval_aligned_mean:.0f}")
        if stats.eval_greedy_mean is not None:
            parts.append(f"greedy={stats.eval_greedy_mean:.0f}")
        self._eval_lbl.setText("  ".join(parts) if parts else "—")

        phase = stats.last_phase or "—"
        if stats.last_train_marker:
            phase += f"\n{stats.last_train_marker}"
        self._phase_lbl.setText(phase)

        if stats.watchdog_events is not None:
            last = stats.watchdog_last or ""
            self._wd_prog_lbl.setText(f"events={stats.watchdog_events}  last={last!r}")
        else:
            self._wd_prog_lbl.setText("—")

        self._errors_lbl.setText(str(stats.error_count))

        self._events.setPlainText("\n".join(stats.key_lines))
        self._events.verticalScrollBar().setValue(
            self._events.verticalScrollBar().maximum()
        )

        self._status.setText(f"Updated {time.strftime('%H:%M:%S')}")


class MatrixMonitorWindow(QMainWindow):
    """Live view of run_reliability_matrix.py progress."""

    def __init__(
        self,
        log_path: Path,
        *,
        cells_total: int = 11,
        refresh_ms: int = 1500,
    ) -> None:
        super().__init__()
        self._log_path = log_path
        self._cells_total = cells_total
        self._cell_dir = log_path.parent / "matrix_cells"

        self.setWindowTitle("Ascent — Reliability Matrix Monitor")
        self.setMinimumSize(520, 520)
        self.setStyleSheet(APP_STYLESHEET)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self._log_label = QLabel()
        self._log_label.setWordWrap(True)
        layout.addWidget(self._log_label)

        time_box = QGroupBox("Session time")
        time_form = QFormLayout(time_box)
        self._elapsed_lbl = QLabel("—")
        self._remaining_lbl = QLabel("—")
        self._cells_lbl = QLabel("—")
        self._process_lbl = QLabel("—")
        time_form.addRow("Elapsed", self._elapsed_lbl)
        time_form.addRow("Est. remaining", self._remaining_lbl)
        time_form.addRow("Cells", self._cells_lbl)
        time_form.addRow("Processes", self._process_lbl)
        layout.addWidget(time_box)

        stats_box = QGroupBox("Current benchmark")
        stats_form = QFormLayout(stats_box)
        self._cell_lbl = QLabel("—")
        self._eval_lbl = QLabel("—")
        self._last_mean_lbl = QLabel("—")
        self._errors_lbl = QLabel("0")
        stats_form.addRow("Cell", self._cell_lbl)
        stats_form.addRow("Eval progress", self._eval_lbl)
        stats_form.addRow("Last cell mean", self._last_mean_lbl)
        stats_form.addRow("Errors flagged", self._errors_lbl)
        layout.addWidget(stats_box)

        events_box = QGroupBox("Recent log events")
        events_layout = QVBoxLayout(events_box)
        self._events = QPlainTextEdit()
        self._events.setReadOnly(True)
        self._events.setMaximumBlockCount(200)
        self._events.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        events_layout.addWidget(self._events)
        layout.addWidget(events_box, stretch=1)

        self._status = QLabel("")
        layout.addWidget(self._status)

        self._timer = QTimer(self)
        self._timer.setInterval(refresh_ms)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._tick()

    def _tick(self) -> None:
        from ascent_player.utils.matrix_monitor import (
            estimate_remaining_seconds,
            refresh_matrix_stats,
        )

        stats = refresh_matrix_stats(self._log_path, cell_dir=self._cell_dir)
        stats.cells_total = self._cells_total
        self._render(stats, estimate_remaining_seconds(stats))

    def _render(self, stats, remaining_s: float) -> None:
        self._log_label.setText(
            f"Log: {self._log_path}  ({stats.log_bytes:,} bytes)"
        )
        if stats.session_start_ts:
            elapsed = time.time() - stats.session_start_ts
            self._elapsed_lbl.setText(format_duration(elapsed))
        else:
            self._elapsed_lbl.setText("—")
        self._remaining_lbl.setText(format_duration(remaining_s))

        in_progress = 1 if stats.benchmark_running else 0
        self._cells_lbl.setText(
            f"{stats.cells_complete}/{stats.cells_total} done"
            f"  (+{in_progress} running, {stats.cells_failed} failed)"
        )
        parts = []
        if stats.matrix_running:
            parts.append(f"matrix pid {stats.matrix_pid}")
        if stats.benchmark_running:
            parts.append(f"benchmark pid {stats.benchmark_pid}")
        self._process_lbl.setText("  ".join(parts) if parts else "not running")

        if stats.current_policy:
            skill = "on" if stats.current_skill else "off"
            self._cell_lbl.setText(
                f"{stats.current_policy}  watch={stats.current_watch:.2f}  skill={skill}"
            )
        else:
            self._cell_lbl.setText("—")

        if stats.eval_episode and stats.eval_episodes:
            self._eval_lbl.setText(
                f"{stats.eval_episode}/{stats.eval_episodes} ep  "
                f"last_score={stats.last_score:.0f}"
                if stats.last_score is not None
                else f"{stats.eval_episode}/{stats.eval_episodes} ep"
            )
        elif stats.benchmark_running:
            self._eval_lbl.setText("running (no progress line yet)")
        else:
            self._eval_lbl.setText("—")

        self._last_mean_lbl.setText(
            f"{stats.last_cell_mean:.1f}" if stats.last_cell_mean else "—"
        )
        self._errors_lbl.setText(str(stats.error_count))
        self._events.setPlainText("\n".join(stats.key_lines))
        self._events.verticalScrollBar().setValue(
            self._events.verticalScrollBar().maximum()
        )
        self._status.setText(f"Updated {time.strftime('%H:%M:%S')}")

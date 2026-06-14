from __future__ import annotations

import asyncio
from dataclasses import dataclass

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.demo.recorder import DemoRecorder
from ascent_player.demo.ingest import ingest_demonstrations
from ascent_player.env.browser_backend import BrowserBackend, BrowserStatus
from ascent_player.env.browser_discovery import discover_ascent_tab, list_chromium_windows
from ascent_player.env.game_env import ACTION_LABELS, AscentGameEnv
from ascent_player.ui.widgets import (
    BrowserPanel,
    EpisodeChart,
    EpisodePoint,
    HyperparameterPanel,
    PreviewWidget,
)
from ascent_player.utils.preprocessing import qimage_bytes_from_frame


@dataclass(slots=True)
class WorkerMetrics:
    episode: int
    episode_reward: float
    episode_score: float
    epsilon: float
    action: str
    replay_size: int
    total_steps: int
    loss: float | None
    train_ms: float | None
    device: str
    boost_level: float
    can_boost: bool


class RecordingWorker(QThread):
    frame_ready = pyqtSignal(bytes)
    status_ready = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    error_ready = pyqtSignal(str)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.running = True

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            self.error_ready.emit(str(exc))

    async def _run_async(self) -> None:
        backend = BrowserBackend(self.config.browser)
        env = AscentGameEnv(self.config, backend)
        recorder = DemoRecorder(self.config, backend, env)
        try:
            self.status_ready.emit("Preparing demo recording...")
            await recorder.prepare()
            self.status_ready.emit(
                "Recording: play in the browser with A / D / Space. Click Stop when done."
            )
            while self.running:
                frame, action, done = await recorder.capture_step()
                self.frame_ready.emit(qimage_bytes_from_frame(frame))
                self.status_ready.emit(
                    f"Recording demo | action={ACTION_LABELS[action]} | frames={len(recorder.transitions)}"
                )
                await backend.wait_ms(env._step_ms())
                if done:
                    self.status_ready.emit("Run ended — restarting for more recording...")
                    recorder.reward_tracker.reset()
                    recorder._last_state = None
                    recorder._last_action = None
                    await env.reset()
            path = await recorder.stop_and_save()
            self.finished_ok.emit(f"Saved demonstration: {path} ({len(recorder.transitions)} transitions)")
        finally:
            await env.close()


class TrainingWorker(QThread):
    frame_ready = pyqtSignal(bytes)
    status_ready = pyqtSignal(str)
    metrics_ready = pyqtSignal(object)
    episode_ready = pyqtSignal(object)
    error_ready = pyqtSignal(str)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.running = True
        self.paused = False
        self.watch_mode = config.training.watch_mode
        self.save_requested = False
        self.load_requested = False
        self._preview_stride = 2

    def stop(self) -> None:
        self.running = False

    def set_paused(self, paused: bool) -> None:
        self.paused = paused

    def set_watch_mode(self, enabled: bool) -> None:
        self.watch_mode = enabled
        self.config.training.watch_mode = enabled

    def request_save(self) -> None:
        self.save_requested = True

    def request_load(self) -> None:
        self.load_requested = True

    def run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            self.error_ready.emit(str(exc))

    async def _run_async(self) -> None:
        backend = BrowserBackend(self.config.browser)
        env = AscentGameEnv(self.config, backend)
        agent = DQNAgent(self.config)
        episode = 0
        try:
            self.status_ready.emit("Connecting browser...")
            status = await backend.connect_auto()
            self.status_ready.emit(_format_browser_status(status))
            if not status.connected:
                return

            if self.config.demo.use_demos_on_start:
                result = ingest_demonstrations(agent, self.config)
                if result.transitions_added or result.transitions_skipped:
                    self.status_ready.emit(result.status_message)

            state = await env.reset()
            episode_reward = 0.0
            episode_score = 0.0
            while self.running:
                if self.paused:
                    await asyncio.sleep(0.1)
                    continue
                if self.load_requested:
                    loaded = agent.load()
                    self.status_ready.emit(
                        "Loaded checkpoint" if loaded else "No checkpoint found"
                    )
                    self.load_requested = False
                if self.save_requested:
                    path = agent.save()
                    self.status_ready.emit(f"Saved checkpoint: {path}")
                    self.save_requested = False

                action = agent.act(
                    state,
                    training=not self.watch_mode,
                    can_boost=env.can_boost,
                )
                result = await env.step(action)
                agent.remember(
                    state,
                    action,
                    result.reward,
                    result.state,
                    result.done,
                )
                metrics = agent.maybe_train()
                state = result.state
                episode_reward += result.reward
                episode_score = float(result.frame_state.score or episode_score)

                if metrics.total_steps % self._preview_stride == 0:
                    self.frame_ready.emit(qimage_bytes_from_frame(result.raw_frame))
                self.metrics_ready.emit(
                    WorkerMetrics(
                        episode=episode,
                        episode_reward=episode_reward,
                        episode_score=episode_score,
                        epsilon=agent.epsilon,
                        action=ACTION_LABELS.get(action, str(action)),
                        replay_size=metrics.replay_size,
                        total_steps=metrics.total_steps,
                        loss=metrics.loss,
                        train_ms=metrics.train_ms,
                        device=agent.device_message,
                        boost_level=env.boost_level,
                        can_boost=env.can_boost,
                    )
                )

                if result.done:
                    self.episode_ready.emit(
                        EpisodePoint(
                            episode=episode,
                            reward=episode_reward,
                            score=episode_score,
                        )
                    )
                    agent.end_episode()
                    if episode > 0 and episode % 10 == 0:
                        agent.save()
                    episode += 1
                    episode_reward = 0.0
                    episode_score = 0.0
                    state = await env.reset()
        finally:
            await env.close()


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.worker: TrainingWorker | None = None
        self.recording_worker: RecordingWorker | None = None

        self.setWindowTitle("Ascent Neural Network Player")
        self.resize(1180, 760)

        self.browser_panel = BrowserPanel()
        self.preview = PreviewWidget()
        self.params = HyperparameterPanel()
        self.chart = EpisodeChart()
        self.status = QLabel("Ready")

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.save_button = QPushButton("Save checkpoint")
        self.load_button = QPushButton("Load checkpoint")
        self.record_button = QPushButton("Record demo")
        self.stop_record_button = QPushButton("Stop recording")
        self.stop_record_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.load_button.setEnabled(False)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.addWidget(self.browser_panel)

        body = QHBoxLayout()
        body.addWidget(self.preview, stretch=3)
        side = QVBoxLayout()
        side.addWidget(self.params)
        side.addWidget(self.chart)
        side.addWidget(self.start_button)
        side.addWidget(self.pause_button)
        side.addWidget(self.save_button)
        side.addWidget(self.load_button)
        side.addWidget(self.record_button)
        side.addWidget(self.stop_record_button)
        side.addStretch()
        body.addLayout(side, stretch=1)
        root_layout.addLayout(body)
        root_layout.addWidget(self.status)
        self.setCentralWidget(root)

        self.start_button.clicked.connect(self.start_training)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.browser_panel.rescan_requested.connect(self.rescan)
        self.browser_panel.connect_requested.connect(self.attach_cdp)
        self.browser_panel.launch_requested.connect(self.force_launch)
        self.browser_panel.windows_requested.connect(self.refresh_windows)
        self.params.changed.connect(self.apply_config_from_ui)
        self.save_button.clicked.connect(self.save_checkpoint)
        self.load_button.clicked.connect(self.load_checkpoint)
        self.record_button.clicked.connect(self.start_recording)
        self.stop_record_button.clicked.connect(self.stop_recording)

        self.rescan_timer = QTimer(self)
        self.rescan_timer.timeout.connect(self.rescan)
        self.rescan_timer.start(self.config.browser.rescan_seconds * 1000)
        QTimer.singleShot(0, self.rescan)

    def apply_config_from_ui(self) -> None:
        self.config.training.learning_rate = self.params.learning_rate.value()
        self.config.training.gamma = self.params.gamma.value()
        self.config.training.epsilon_decay = self.params.epsilon_decay.value()
        self.config.training.min_replay_size = self.params.min_replay.value()
        self.config.training.frame_skip = self.params.frame_skip.value()
        self.config.training.batch_size_cpu = self.params.batch_size.value()
        self.config.training.batch_size_gpu = self.params.batch_size.value()
        self.config.training.train_every_cpu = self.params.train_every.value()
        self.config.training.train_every_gpu = self.params.train_every.value()
        self.config.training.device_mode = DeviceMode(self.params.device.currentText())
        self.config.browser.auto_launch_on_miss = self.browser_panel.auto_launch.isChecked()
        self.config.demo.use_demos_on_start = self.params.use_demos.isChecked()
        mode = self.params.mode.currentText()
        self.config.training.watch_mode = mode == "watch"
        if self.worker is not None:
            self.worker.set_watch_mode(mode == "watch")
            self.worker.set_paused(mode == "paused")

    def start_recording(self) -> None:
        if self.worker is not None or self.recording_worker is not None:
            return
        self.apply_config_from_ui()
        self.recording_worker = RecordingWorker(self.config)
        self.recording_worker.frame_ready.connect(self.preview.set_png)
        self.recording_worker.status_ready.connect(self.status.setText)
        self.recording_worker.finished_ok.connect(self.recording_finished)
        self.recording_worker.error_ready.connect(self.show_error)
        self.recording_worker.finished.connect(self.recording_worker_finished)
        self.recording_worker.start()
        self.record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        self.start_button.setEnabled(False)

    def stop_recording(self) -> None:
        if self.recording_worker is not None:
            self.recording_worker.stop()

    def recording_finished(self, message: str) -> None:
        self.status.setText(message)

    def recording_worker_finished(self) -> None:
        self.recording_worker = None
        self.record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        self.start_button.setEnabled(True)

    def start_training(self) -> None:
        if self.worker is not None or self.recording_worker is not None:
            return
        self.apply_config_from_ui()
        self.worker = TrainingWorker(self.config)
        self.worker.frame_ready.connect(self.preview.set_png)
        self.worker.status_ready.connect(self.browser_panel.set_status)
        self.worker.status_ready.connect(self.status.setText)
        self.worker.metrics_ready.connect(self.update_metrics)
        self.worker.episode_ready.connect(self.chart.add_point)
        self.worker.error_ready.connect(self.show_error)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.load_button.setEnabled(True)

    def toggle_pause(self) -> None:
        if self.worker is None:
            return
        pause = self.pause_button.text() == "Pause"
        self.worker.set_paused(pause)
        self.pause_button.setText("Resume" if pause else "Pause")

    def rescan(self) -> None:
        if self.worker is not None or self.recording_worker is not None:
            return
        self.browser_panel.set_status("Scanning for Ascent tab...")
        try:
            tab = asyncio.run(discover_ascent_tab(self.config.browser))
        except Exception as exc:
            self.browser_panel.set_status(f"Scan failed: {exc}")
            return
        if tab is None:
            self.browser_panel.set_status("No CDP Ascent tab found")
        else:
            self.browser_panel.set_status(
                f"Found Ascent tab: {tab.title or tab.url} on port {tab.port}"
            )

    def attach_cdp(self, cdp_url: str) -> None:
        self.config.browser.manual_cdp_url = cdp_url
        self.start_training()

    def force_launch(self) -> None:
        self.config.browser.manual_cdp_url = None
        self.config.browser.auto_launch_on_miss = True
        self.start_training()

    def refresh_windows(self) -> None:
        windows = list_chromium_windows()
        labels = [
            f"{window.title} (pid {window.pid or '-'})"
            for window in windows
        ]
        self.browser_panel.set_windows(labels)

    def save_checkpoint(self) -> None:
        if self.worker is not None:
            self.worker.request_save()

    def load_checkpoint(self) -> None:
        if self.worker is not None:
            self.worker.request_load()

    def update_metrics(self, metrics: WorkerMetrics) -> None:
        loss = "-" if metrics.loss is None else f"{metrics.loss:.4f}"
        train_ms = "-" if metrics.train_ms is None else f"{metrics.train_ms:.1f}ms"
        self.status.setText(
            " | ".join(
                [
                    f"ep {metrics.episode}",
                    f"reward {metrics.episode_reward:.1f}",
                    f"score {metrics.episode_score:.1f}",
                    f"eps {metrics.epsilon:.3f}",
                    f"action {metrics.action}",
                    f"boost {metrics.boost_level:.0%}"
                    + ("" if metrics.can_boost else " (empty)"),
                    f"replay {metrics.replay_size}",
                    f"steps {metrics.total_steps}",
                    f"loss {loss}",
                    f"train {train_ms}",
                    metrics.device,
                ]
            )
        )

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ascent player error", message)

    def worker_finished(self) -> None:
        self.worker = None
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.load_button.setEnabled(False)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.recording_worker is not None:
            self.recording_worker.stop()
            self.recording_worker.wait(5_000)
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(5_000)
        event.accept()


def _format_browser_status(status: BrowserStatus) -> str:
    if not status.connected:
        return status.message
    label = status.title or status.url
    return f"{status.message}: {label}"

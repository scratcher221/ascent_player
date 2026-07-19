from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ascent_player.agent.dqn import AgentMetrics, DQNAgent
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.demo.recorder import DemoRecorder
from ascent_player.demo.ingest import ingest_demonstrations
from ascent_player.env.browser_backend import BrowserBackend, BrowserStatus
from ascent_player.env.browser_discovery import discover_ascent_tab
from ascent_player.env.game_env import ACTION_LABELS, AscentGameEnv
from ascent_player.ui.widgets import (
    APP_STYLESHEET,
    BrowserPanel,
    EpisodeChart,
    EpisodePoint,
    HyperparameterPanel,
    PreviewWidget,
    ProgressPanel,
    SessionControls,
)
from ascent_player.utils.preprocessing import qimage_bytes_from_frame
from ascent_player.utils.training_log import BrowserStepContext, TrainingLogger


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
    loop_hz: float
    session_message: str
    baseline_reward: float | None
    baseline_score: float | None
    vs_baseline_pct: float | None
    best_reward: float
    best_score: float
    recent_avg_reward: float | None
    score_velocity: float
    autosave_message: str


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
                "Recording: play in the browser with A / D / Space. "
                "Click Stop recording when done."
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
                    recorder.on_episode_end()
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
    session_ready = pyqtSignal(object)
    error_ready = pyqtSignal(str)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.running = True
        self.paused = False
        self.watch_mode = config.training.watch_mode
        self.save_requested = False
        self.load_requested = False
        self.load_best_requested = False
        self._preview_stride = 4
        self._metrics_stride = 2
        self._train_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dqn-train")
        self._train_future: Future | None = None
        self._loop_hz = 0.0
        self._executor_closed = False

    def shutdown(self) -> None:
        if self._executor_closed:
            return
        self._executor_closed = True
        if self._train_future is not None and self._train_future.done():
            try:
                self._train_future.result()
            except Exception:
                pass
        self._train_executor.shutdown(wait=False, cancel_futures=True)

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

    def request_load_best(self) -> None:
        self.load_best_requested = True

    def sync_agent_hyperparameters(self, agent: DQNAgent) -> None:
        agent.set_train_every(
            self.config.training.train_every_gpu
            if agent.device_info.training_device.startswith("/GPU")
            else self.config.training.train_every_cpu
        )
        agent.set_batch_size(
            self.config.training.batch_size_gpu
            if agent.device_info.training_device.startswith("/GPU")
            else self.config.training.batch_size_cpu
        )
        agent.set_learning_rate(self.config.training.learning_rate)

    def _collect_train_metrics(self, agent: DQNAgent) -> None:
        if self._train_future is None or not self._train_future.done():
            return
        try:
            self._train_future.result()
        except Exception:
            pass
        finally:
            self._train_future = None

    def _schedule_training(self, agent: DQNAgent) -> AgentMetrics:
        if self.config.training.watch_mode:
            return agent.metrics
        if not self.config.training.async_training:
            return agent.maybe_train()
        if self._train_future is not None and not self._train_future.done():
            return agent.metrics
        self._train_future = self._train_executor.submit(agent.maybe_train)
        return agent.metrics

    def run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            self.error_ready.emit(str(exc))

    async def _run_async(self) -> None:
        backend = BrowserBackend(self.config.browser)
        env = AscentGameEnv(self.config, backend)
        agent = DQNAgent(self.config)
        load_result = agent.try_autoload()
        logger = TrainingLogger(self.config, "browser")
        logger.log_session_start(
            agent,
            message=load_result.message,
            extra={"ui_mode": not self.config.training.watch_mode},
        )
        session_label = (
            "Resumed from checkpoint" if load_result.loaded else "Fresh training session"
        )
        ui_result = type(load_result)(
            load_result.loaded,
            session_label,
            load_result.progress,
        )
        self.session_ready.emit(ui_result)
        episode = agent.progress.episodes_completed
        autosave_message = f"Log: {logger.path.name}"
        try:
            self.status_ready.emit(
                f"Connecting browser… · {session_label} · log {logger.path.name}"
            )
            status = await backend.connect_auto()
            self.status_ready.emit(_format_browser_status(status))
            if not status.connected:
                logger.log_note(f"browser_connect_failed={status.message}")
                self.error_ready.emit(
                    f"Browser connection failed: {status.message}"
                )
                logger.close(agent)
                return

            if (
                self.config.demo.use_demos_on_start
                and not self.watch_mode
                and not self.config.training.watch_mode
            ):
                result = ingest_demonstrations(agent, self.config)
                if result.transitions_added or result.transitions_skipped:
                    self.status_ready.emit(result.status_message)
                    logger.log_note(f"demo_ingest={result.status_message}")
            elif self.watch_mode or self.config.training.watch_mode:
                # Watch must keep loaded weights intact — demo BC would overwrite them.
                agent.epsilon = 0.0
                agent.metrics.epsilon = 0.0
                agent.progress.epsilon = 0.0
                logger.log_note("watch_mode=skip_demo_ingest")

            state = await env.reset()
            episode_reward = 0.0
            episode_score = 0.0
            episode_max_score = 0.0
            prev_step_score = 0.0
            score_velocity = 0.0
            self.sync_agent_hyperparameters(agent)
            while self.running:
                if self.paused:
                    await asyncio.sleep(0.1)
                    continue
                self.sync_agent_hyperparameters(agent)
                self._collect_train_metrics(agent)
                if self.load_requested:
                    loaded = agent.load()
                    if loaded and self.watch_mode:
                        agent.epsilon = 0.0
                        agent.metrics.epsilon = 0.0
                        agent.progress.epsilon = 0.0
                    self.status_ready.emit(
                        "Loaded checkpoint" if loaded else "No checkpoint found"
                    )
                    logger.log_note(
                        f"checkpoint_load={'ok' if loaded else 'missing'}"
                    )
                    self.load_requested = False
                if self.load_best_requested:
                    best = agent.resolve_best_checkpoint()
                    loaded = bool(best and agent.load(best))
                    if loaded and best is not None:
                        if self.watch_mode:
                            agent.epsilon = 0.0
                            agent.metrics.epsilon = 0.0
                            agent.progress.epsilon = 0.0
                        agent.save(self.config.training.playable_checkpoint_path)
                        agent.save(self.config.training.checkpoint_path)
                        self.status_ready.emit(
                            f"Loaded best model ({best.name}, "
                            f"best score {agent.progress.best_score:.0f})"
                        )
                        logger.log_note(f"checkpoint_load_best=ok path={best}")
                    else:
                        self.status_ready.emit("No best model checkpoint found")
                        logger.log_note("checkpoint_load_best=missing")
                    self.load_best_requested = False
                if self.save_requested:
                    path = agent.save()
                    autosave_message = f"Autosave: saved {path.name}"
                    logger.log_note(f"checkpoint_save={path}")
                    self.save_requested = False

                step_started = time.perf_counter()
                action = agent.act(
                    state,
                    training=not self.watch_mode,
                    can_boost=env.can_boost,
                    boost_level=env.boost_level,
                    frame_state=env._last_frame_state,
                )
                result = await env.step(action)
                step_ms = (time.perf_counter() - step_started) * 1000.0
                if step_ms > 0:
                    instant_hz = 1000.0 / step_ms
                    self._loop_hz = (0.85 * self._loop_hz) + (0.15 * instant_hz)
                agent.remember(
                    state,
                    action,
                    result.reward,
                    result.state,
                    result.done,
                )
                metrics = self._schedule_training(agent)
                self._collect_train_metrics(agent)
                episode_reward += result.reward
                if result.frame_state.score is not None:
                    episode_score = float(result.frame_state.score)
                    episode_max_score = max(episode_max_score, episode_score)
                    score_velocity = episode_score - prev_step_score
                    prev_step_score = episode_score
                logger.record_browser_step(
                    action,
                    result.reward,
                    result.frame_state,
                    agent,
                    can_boost=env.can_boost,
                    boost_level=env.boost_level,
                    done=result.done,
                    context=BrowserStepContext(
                        step_ms=step_ms,
                        loop_hz=self._loop_hz,
                        score_velocity=score_velocity,
                        episode_reward=episode_reward,
                        total_steps=agent.metrics.total_steps,
                    ),
                    train_loss=agent.metrics.loss,
                    train_ms=agent.metrics.train_ms,
                )
                logger.maybe_flush(agent, agent.metrics.total_steps)
                state = result.state

                if metrics.total_steps % self._preview_stride == 0:
                    self.frame_ready.emit(qimage_bytes_from_frame(result.raw_frame))
                if metrics.total_steps % self._metrics_stride == 0 or result.done:
                    progress = agent.progress
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
                            loop_hz=self._loop_hz,
                            session_message=session_label,
                            baseline_reward=progress.baseline_reward,
                            baseline_score=progress.baseline_score,
                            vs_baseline_pct=progress.reward_vs_baseline_pct(),
                            best_reward=(
                                progress.best_reward
                                if progress.best_reward != float("-inf")
                                else 0.0
                            ),
                            best_score=progress.best_score,
                            recent_avg_reward=progress.recent_avg_reward,
                            score_velocity=score_velocity,
                            autosave_message=autosave_message,
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
                    agent.record_episode(episode_reward, episode_max_score)
                    logger.log_episode_end(
                        agent,
                        episode,
                        episode_reward,
                        episode_max_score,
                    )
                    agent.end_episode()
                    if not self.watch_mode and agent.maybe_autosave(force=True):
                        autosave_message = (
                            f"Autosave: episode {agent.progress.episodes_completed} saved"
                        )
                    episode += 1
                    episode_reward = 0.0
                    episode_score = 0.0
                    episode_max_score = 0.0
                    prev_step_score = 0.0
                    score_velocity = 0.0
                    state = await env.reset()
                elif not self.watch_mode and agent.maybe_autosave():
                    autosave_message = f"Autosave: step {metrics.total_steps:,} saved"
        finally:
            self._collect_train_metrics(agent)
            try:
                # Never overwrite trained checkpoints from Watch (eval-only) sessions.
                if not self.watch_mode:
                    path = agent.save()
                    autosave_message = f"Autosave: final save {path.name}"
                    self.status_ready.emit(autosave_message)
            except Exception:
                pass
            logger.close(agent)
            self.status_ready.emit(f"Training log saved: {logger.path}")
            await env.close()
            self.shutdown()


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.worker: TrainingWorker | None = None
        self.recording_worker: RecordingWorker | None = None

        self.setWindowTitle("Ascent Neural Network Player")
        self.resize(1280, 820)
        self.setStyleSheet(APP_STYLESHEET)

        self.browser_panel = BrowserPanel()
        self.preview = PreviewWidget()
        self.session = SessionControls()
        self.params = HyperparameterPanel()
        self.progress_panel = ProgressPanel()
        self.chart = EpisodeChart()
        self.status = QLabel("Ready — hover ⓘ icons for explanations")
        self.status.setObjectName("statusBar")
        self.status.setWordWrap(True)

        # Compatibility aliases used by older call sites / mental model.
        self.start_button = self.session.start_button
        self.pause_button = self.session.pause_button
        self.save_button = self.session.save_button
        self.load_button = self.session.load_button
        self.record_button = self.session.record_button
        self.stop_record_button = self.session.stop_record_button

        side_host = QWidget()
        side = QVBoxLayout(side_host)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(10)
        side.addWidget(self.session)
        side.addWidget(self.progress_panel)
        side.addWidget(self.params)
        side.addWidget(self.chart)
        side.addStretch(1)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setWidget(side_host)
        side_scroll.setMinimumWidth(340)
        side_scroll.setMaximumWidth(420)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self.preview, stretch=3)
        body.addWidget(side_scroll, stretch=1)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 0)
        root_layout.setSpacing(10)
        root_layout.addWidget(self.browser_panel)
        root_layout.addLayout(body, stretch=1)
        root_layout.addWidget(self.status)
        self.setCentralWidget(root)

        self.session.start_clicked.connect(self.start_training)
        self.session.stop_clicked.connect(self.stop_training)
        self.session.pause_clicked.connect(self.toggle_pause)
        self.session.save_clicked.connect(self.save_checkpoint)
        self.session.load_clicked.connect(self.load_checkpoint)
        self.session.load_best_clicked.connect(self.load_best_checkpoint)
        self.session.record_clicked.connect(self.start_recording)
        self.session.stop_record_clicked.connect(self.stop_recording)
        self.session.changed.connect(self.apply_config_from_ui)
        self.browser_panel.rescan_requested.connect(self.rescan)
        self.browser_panel.connect_requested.connect(self.attach_cdp)
        self.browser_panel.launch_requested.connect(self.force_launch)
        self.params.changed.connect(self.apply_config_from_ui)

        if config.training.watch_mode:
            self.session.set_mode_value("watch")
        self.params.set_watch_mode(self.session.mode_value() == "watch")
        self.apply_config_from_ui()

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
        self.config.demo.use_demos_on_start = self.session.use_demos.isChecked()
        mode = self.session.mode_value()
        self.config.training.watch_mode = mode == "watch"
        self.params.set_watch_mode(mode == "watch")
        if self.worker is not None:
            self.worker.set_watch_mode(mode == "watch")

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
        self.session.set_recording()
        self.browser_panel.set_busy(True)

    def stop_recording(self) -> None:
        if self.recording_worker is not None:
            self.recording_worker.stop()

    def recording_finished(self, message: str) -> None:
        self.status.setText(message)

    def recording_worker_finished(self) -> None:
        self.recording_worker = None
        self.session.set_idle()
        self.browser_panel.set_busy(False)
        self.preview.show_empty_state()

    def start_training(self) -> None:
        if self.worker is not None or self.recording_worker is not None:
            return
        self.apply_config_from_ui()
        self.worker = TrainingWorker(self.config)
        self.worker.frame_ready.connect(self.preview.set_png)
        self.worker.status_ready.connect(self.browser_panel.set_status)
        self.worker.status_ready.connect(self.status.setText)
        self.worker.metrics_ready.connect(self.update_metrics)
        self.worker.session_ready.connect(self.on_session_ready)
        self.worker.episode_ready.connect(self.chart.add_point)
        self.worker.error_ready.connect(self.on_worker_error)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()
        self.session.set_training()
        self.browser_panel.set_busy(True)
        # Connection is settled once training starts — collapse setup to free space.
        self.browser_panel.setChecked(False)

    def stop_training(self) -> None:
        if self.worker is None:
            return
        self.status.setText("Stopping session…")
        self.worker.stop()

    def toggle_pause(self) -> None:
        if self.worker is None:
            return
        pause = self.pause_button.text() == "Pause"
        self.worker.set_paused(pause)
        self.session.set_paused(pause)

    def rescan(self) -> None:
        if self.worker is not None or self.recording_worker is not None:
            return
        self.browser_panel.set_status("Scanning for Ascent tab…")
        try:
            tab = asyncio.run(discover_ascent_tab(self.config.browser))
        except Exception as exc:
            self.browser_panel.set_status(f"Scan failed: {exc}")
            return
        if tab is None:
            self.browser_panel.set_status(
                "No Ascent tab found — enable auto-launch, or open Advanced to force a new browser"
            )
        else:
            self.browser_panel.set_status(
                f"Ready: found “{tab.title or tab.url}” on port {tab.port}"
            )

    def attach_cdp(self, cdp_url: str) -> None:
        self.config.browser.manual_cdp_url = cdp_url
        self.start_training()

    def force_launch(self) -> None:
        self.config.browser.manual_cdp_url = None
        self.config.browser.auto_launch_on_miss = True
        self.browser_panel.auto_launch.setChecked(True)
        self.start_training()

    def on_session_ready(self, load_result) -> None:
        self.progress_panel.set_session(load_result.message)
        self.status.setText(load_result.message)
        if load_result.progress is not None:
            progress = load_result.progress
            self.progress_panel.set_baseline(
                progress.baseline_reward,
                progress.baseline_score,
            )
            self.progress_panel.set_best_ever(
                progress.best_score,
                progress.best_reward if progress.best_reward != float("-inf") else 0.0,
            )

    def save_checkpoint(self) -> None:
        if self.worker is not None:
            self.worker.request_save()

    def load_checkpoint(self) -> None:
        if self.worker is not None:
            self.worker.request_load()

    def load_best_checkpoint(self) -> None:
        if self.worker is not None:
            self.worker.request_load_best()
            return
        # Idle: promote strongest weights into playable + dqn_latest for next Start.
        agent = DQNAgent(self.config)
        path = agent.promote_playable_checkpoint()
        if path is None:
            self.status.setText("No best model checkpoint found")
            QMessageBox.warning(
                self,
                "Load best model",
                "No best_playable / sim_best_eval checkpoint was found.",
            )
            return
        self.status.setText(
            f"Best model promoted to {path.name} and dqn_latest "
            f"(best score {agent.progress.best_score:.0f}). "
            "Start Train or Watch to use it."
        )
        self.progress_panel.set_session(
            f"Best model ready ({path.name}, score {agent.progress.best_score:.0f})"
        )
        self.progress_panel.set_best_ever(
            agent.progress.best_score,
            agent.progress.best_reward
            if agent.progress.best_reward != float("-inf")
            else 0.0,
        )

    def update_metrics(self, metrics: WorkerMetrics) -> None:
        train_ms = "-" if metrics.train_ms is None else f"{metrics.train_ms:.1f}ms"
        boost_label = f"boost {metrics.boost_level:.0%}"
        if not metrics.can_boost:
            boost_label += " (depleted)"
        self.progress_panel.set_session(metrics.session_message)
        self.progress_panel.set_live(
            metrics.episode,
            metrics.episode_reward,
            metrics.episode_score,
            metrics.epsilon,
            metrics.loss,
        )
        self.progress_panel.set_baseline(metrics.baseline_reward, metrics.baseline_score)
        self.progress_panel.set_best_ever(metrics.best_score, metrics.best_reward)
        self.progress_panel.set_comparison(
            metrics.recent_avg_reward,
            metrics.vs_baseline_pct,
            metrics.best_reward,
        )
        self.progress_panel.set_score_velocity(metrics.score_velocity)
        self.progress_panel.set_autosave(metrics.autosave_message)
        self.status.setText(
            " · ".join(
                [
                    f"ep {metrics.episode}",
                    f"score {metrics.episode_score:.1f}",
                    f"action {metrics.action}",
                    boost_label,
                    f"replay {metrics.replay_size}",
                    f"steps {metrics.total_steps}",
                    f"train {train_ms}",
                    f"loop {metrics.loop_hz:.1f}Hz",
                    metrics.device,
                ]
            )
        )

    def on_worker_error(self, message: str) -> None:
        self.browser_panel.setChecked(True)
        self.browser_panel.set_status(message)
        self.show_error(message)

    def show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ascent player error", message)

    def worker_finished(self) -> None:
        self.worker = None
        self.session.set_idle()
        self.browser_panel.set_busy(False)
        self.browser_panel.setChecked(True)
        self.preview.show_empty_state()
        if self.status.text().startswith("Stopping"):
            self.status.setText("Session stopped")

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

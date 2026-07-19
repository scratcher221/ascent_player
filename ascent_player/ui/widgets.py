from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWhatsThis,
    QWidget,
)

TOOLTIPS = {
    "mode": (
        "Train: the agent explores and updates its neural network.\n"
        "Watch: the agent plays using the loaded checkpoint with no learning.\n"
        "Use Watch (after Load best model) to verify skill — Train scores can look weak due to exploration."
    ),
    "device": (
        "Where TensorFlow runs the network.\n"
        "GPU — prefer NVIDIA (fastest).\n"
        "Auto — use GPU if available, otherwise CPU.\n"
        "CPU — force CPU (slower, more compatible)."
    ),
    "learning_rate": (
        "How large each weight update is during training.\n"
        "Higher learns faster but can become unstable.\n"
        "Lower is steadier but slower to improve."
    ),
    "gamma": (
        "Discount factor for future rewards (0–1).\n"
        "Closer to 1 values long-term score more;\n"
        "lower values focus on immediate rewards."
    ),
    "epsilon_decay": (
        "How quickly random exploration (epsilon) shrinks each episode.\n"
        "Closer to 1 = explore longer.\n"
        "Lower = switch to the learned policy sooner."
    ),
    "batch_size": (
        "Number of past experiences sampled for each training update.\n"
        "Larger batches are more stable but use more memory/GPU."
    ),
    "train_every": (
        "Run a network update every N environment steps.\n"
        "1 = train every step (heavier).\n"
        "Higher values free more time for playing the game."
    ),
    "min_replay": (
        "Minimum transitions in the replay buffer before training starts.\n"
        "Prevents the network from learning from too little data."
    ),
    "frame_skip": (
        "Repeat each chosen action for N game frames.\n"
        "Higher speeds up wall-clock training but makes control coarser."
    ),
    "use_demos": (
        "On Train start, load recorded human demonstrations into the replay buffer.\n"
        "Ignored in Watch mode — Watch never runs demo BC so loaded weights stay intact."
    ),
    "auto_launch": (
        "If no Ascent game tab is found on a CDP port, launch Chromium\n"
        "with remote debugging and open the local game automatically."
    ),
    "cdp_url": (
        "Chrome DevTools Protocol endpoint for an already-running Chromium.\n"
        "Default http://localhost:9222."
    ),
    "connect_status": (
        "Whether an Ascent game tab is visible over CDP.\n"
        "Rescan refreshes this without starting a session."
    ),
    "session": "How the current run started (fresh session or resumed checkpoint).",
    "baseline": (
        "Average reward/score from the first few episodes.\n"
        "Used as a reference to judge later improvement."
    ),
    "best_ever": "Highest game score (and best reward) seen in this training progress.",
    "comparison": (
        "Recent average reward compared with the baseline.\n"
        "Positive % means the agent is beating its early performance."
    ),
    "score_velocity": (
        "Change in in-game score over the last step.\n"
        "Rising velocity usually means the agent is climbing / scoring."
    ),
    "autosave": "Last automatic or manual checkpoint save message.",
    "episode": "Current episode number and reward/score for this run.",
    "epsilon": "Exploration rate — chance of taking a random action (higher = more random).",
    "loss": "Latest training loss from the DQN update (lower is typically better).",
    "preview": (
        "Live view of the game canvas captured from the browser.\n"
        "Appears after a session connects."
    ),
    "start": "Connect to the game if needed and begin the selected mode (Train or Watch).",
    "stop": "End the current training or watch session and return to idle.",
    "pause": "Temporarily halt the loop without closing the browser.",
    "resume": "Continue the paused training or watch session.",
    "save_ckpt": "Write the current network weights to the checkpoint file now.",
    "load_ckpt": "Reload weights from the default checkpoint (dqn_latest) into the running agent.",
    "load_best": (
        "Load the strongest saved policy.\n"
        "Prefers browser_best (Watch-adapted) when present, else best_playable / sim_best_eval.\n"
        "Copies into dqn_latest for the next Train/Watch session.\n"
        "Tip: use Watch mode to verify — Train still explores."
    ),
    "record": (
        "Record your play in the browser (A / D / Space) as a demonstration\n"
        "that can be loaded on future training starts."
    ),
    "stop_record": "Stop recording and save the demonstration to disk.",
    "rescan": "Look again for an Ascent tab on known CDP ports (does not start a session).",
    "attach": (
        "Start using the CDP URL below instead of auto-discovery.\n"
        "Use when Chromium is already running with remote debugging."
    ),
    "launch": (
        "Ignore any manual CDP URL, launch a fresh Chromium instance,\n"
        "open the game, and start."
    ),
}


def _apply_tip(widget: QWidget, tip_key: str) -> None:
    tip = TOOLTIPS[tip_key]
    widget.setToolTip(tip)
    widget.setWhatsThis(tip)


def _info_label(text: str, tip_key: str) -> QWidget:
    """Label with a focusable ⓘ button that shows an explanation."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    name = QLabel(text)
    tip = TOOLTIPS[tip_key]
    name.setToolTip(tip)

    info = QToolButton()
    info.setText("ⓘ")
    info.setObjectName("infoHint")
    info.setAutoRaise(True)
    info.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    info.setAccessibleName(f"About {text}")
    info.setToolTip(tip)
    info.setWhatsThis(tip)
    info.clicked.connect(
        lambda _=False, t=tip, w=info: QWhatsThis.showText(w.mapToGlobal(w.rect().center()), t, w)
    )
    layout.addWidget(name)
    layout.addWidget(info)
    layout.addStretch(1)
    return row


class PreviewWidget(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self._last_png: bytes | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName("previewPane")
        _apply_tip(self, "preview")
        self.show_empty_state()

    def show_empty_state(self) -> None:
        self._last_png = None
        self.clear()
        self.setText(
            "Game preview\n\n"
            "Press Start training to connect and run, or Record demo to capture your play.\n"
            "The live canvas appears here once the browser session is running."
        )

    def set_png(self, data: bytes) -> None:
        self._last_png = data
        self._render_png()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._last_png is not None:
            self._render_png()

    def _render_png(self) -> None:
        if self._last_png is None:
            return
        image = QImage.fromData(self._last_png, "PNG")
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class ProgressPanel(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Progress")
        self.session = QLabel("Not started")
        self.episode = QLabel("—")
        self.epsilon = QLabel("—")
        self.loss = QLabel("—")
        self.baseline = QLabel("Measuring…")
        self.best_ever = QLabel("—")
        self.comparison = QLabel("—")
        self.score_velocity = QLabel("—")
        self.autosave = QLabel("—")

        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        form.addRow(_info_label("Session", "session"), self.session)
        form.addRow(_info_label("Episode", "episode"), self.episode)
        form.addRow(_info_label("Epsilon", "epsilon"), self.epsilon)
        form.addRow(_info_label("Loss", "loss"), self.loss)
        form.addRow(_info_label("Baseline", "baseline"), self.baseline)
        form.addRow(_info_label("Best score", "best_ever"), self.best_ever)
        form.addRow(_info_label("Vs baseline", "comparison"), self.comparison)
        form.addRow(_info_label("Score velocity", "score_velocity"), self.score_velocity)
        form.addRow(_info_label("Autosave", "autosave"), self.autosave)

        for key, widget in (
            ("session", self.session),
            ("episode", self.episode),
            ("epsilon", self.epsilon),
            ("loss", self.loss),
            ("baseline", self.baseline),
            ("best_ever", self.best_ever),
            ("comparison", self.comparison),
            ("score_velocity", self.score_velocity),
            ("autosave", self.autosave),
        ):
            _apply_tip(widget, key)
            widget.setWordWrap(True)
            widget.setObjectName("metricValue")

    def set_session(self, text: str) -> None:
        cleaned = text.removeprefix("Session: ").strip()
        first_line = cleaned.splitlines()[0].strip() if cleaned else "Not started"
        if len(first_line) > 80:
            first_line = first_line[:77] + "…"
        self.session.setText(first_line or "Not started")

    def set_live(
        self,
        episode: int,
        episode_reward: float,
        episode_score: float,
        epsilon: float,
        loss: float | None,
    ) -> None:
        self.episode.setText(
            f"{episode} · reward {episode_reward:.1f} · score {episode_score:.1f}"
        )
        self.epsilon.setText(f"{epsilon:.3f}")
        self.loss.setText("—" if loss is None else f"{loss:.4f}")

    def set_baseline(self, reward: float | None, score: float | None) -> None:
        if reward is None or score is None:
            self.baseline.setText("Measuring first episodes…")
            return
        self.baseline.setText(f"reward {reward:.1f} · score {score:.1f}")

    def set_best_ever(self, best_score: float, best_reward: float) -> None:
        if best_score < 0 or (
            best_score == 0 and (best_reward <= 0 or best_reward == float("-inf"))
        ):
            self.best_ever.setText("None yet")
            return
        reward_text = ""
        if best_reward > 0 and best_reward != float("inf"):
            reward_text = f" · reward {best_reward:.1f}"
        self.best_ever.setText(f"{best_score:.0f}{reward_text}")

    def set_comparison(
        self,
        current_reward: float | None,
        vs_baseline_pct: float | None,
        best_reward: float,
    ) -> None:
        del best_reward
        if current_reward is None:
            self.comparison.setText("—")
            return
        if vs_baseline_pct is None:
            self.comparison.setText(f"recent avg {current_reward:.1f}")
            return
        sign = "+" if vs_baseline_pct >= 0 else ""
        self.comparison.setText(
            f"recent {current_reward:.1f} · {sign}{vs_baseline_pct:.0f}% vs baseline"
        )

    def set_score_velocity(self, velocity: float) -> None:
        if velocity > 0:
            self.score_velocity.setText(f"+{velocity:.1f} / step")
        elif velocity < 0:
            self.score_velocity.setText(f"{velocity:.1f} / step")
        else:
            self.score_velocity.setText("Flat")

    def set_autosave(self, text: str) -> None:
        self.autosave.setText(text or "—")


class BrowserPanel(QGroupBox):
    """Connection status + rescan up front; CDP attach/launch under Advanced."""

    rescan_requested = pyqtSignal()
    connect_requested = pyqtSignal(str)
    launch_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__("Browser setup")
        self.setCheckable(True)
        self.setChecked(True)
        self.setToolTip("Collapse after connecting — Start handles the usual flow.")

        self.hint = QLabel("Tip: hover or click ⓘ next to any label for an explanation.")
        self.hint.setObjectName("hintLabel")
        self.hint.setWordWrap(True)

        self.status = QLabel("Scanning for Ascent tab…")
        self.status.setObjectName("connectionStatus")
        self.status.setWordWrap(True)
        _apply_tip(self.status, "connect_status")

        self.auto_launch = QCheckBox("Auto-launch browser if game tab is missing")
        self.auto_launch.setChecked(True)
        _apply_tip(self.auto_launch, "auto_launch")

        self.rescan = QPushButton("Rescan")
        _apply_tip(self.rescan, "rescan")

        top_actions = QHBoxLayout()
        top_actions.addWidget(self.rescan)
        top_actions.addStretch(1)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced connection options")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.setAutoRaise(True)
        self.advanced_toggle.setToolTip(
            "Manual CDP URL, attach to an existing browser, or force a fresh launch."
        )

        self.advanced_frame = QFrame()
        self.advanced_frame.setObjectName("advancedFrame")
        advanced_layout = QVBoxLayout(self.advanced_frame)
        advanced_layout.setContentsMargins(0, 4, 0, 0)
        advanced_layout.setSpacing(8)

        cdp_row = QHBoxLayout()
        cdp_row.addWidget(_info_label("CDP URL", "cdp_url"))
        self.cdp_url = QComboBox()
        self.cdp_url.setEditable(True)
        self.cdp_url.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.cdp_url.addItem("http://localhost:9222")
        _apply_tip(self.cdp_url, "cdp_url")
        cdp_row.addWidget(self.cdp_url, stretch=1)

        self.attach = QPushButton("Use this CDP & start")
        self.launch = QPushButton("Force new browser & start")
        self.attach.setObjectName("secondaryButton")
        self.launch.setObjectName("secondaryButton")
        _apply_tip(self.attach, "attach")
        _apply_tip(self.launch, "launch")

        adv_actions = QHBoxLayout()
        adv_actions.setSpacing(8)
        adv_actions.addWidget(self.attach)
        adv_actions.addWidget(self.launch)

        advanced_layout.addLayout(cdp_row)
        advanced_layout.addLayout(adv_actions)
        self.advanced_frame.setVisible(False)

        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)
        body_layout.addWidget(self.hint)
        body_layout.addWidget(self.status)
        body_layout.addWidget(self.auto_launch)
        body_layout.addLayout(top_actions)
        body_layout.addWidget(self.advanced_toggle)
        body_layout.addWidget(self.advanced_frame)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(self._body)

        self.rescan.clicked.connect(self.rescan_requested.emit)
        self.attach.clicked.connect(
            lambda: self.connect_requested.emit(self.cdp_url.currentText().strip())
        )
        self.launch.clicked.connect(self.launch_requested.emit)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.toggled.connect(self._on_group_toggled)

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_frame.setVisible(checked)
        self.advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def _on_group_toggled(self, checked: bool) -> None:
        self._body.setVisible(checked)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def set_busy(self, busy: bool) -> None:
        for widget in (
            self.rescan,
            self.attach,
            self.launch,
            self.cdp_url,
            self.auto_launch,
        ):
            widget.setEnabled(not busy)


class SessionControls(QGroupBox):
    """Primary run controls: mode, start/stop/pause, demos, checkpoints."""

    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    save_clicked = pyqtSignal()
    load_clicked = pyqtSignal()
    load_best_clicked = pyqtSignal()
    record_clicked = pyqtSignal()
    stop_record_clicked = pyqtSignal()
    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__("Session")
        self.mode = QComboBox()
        self.mode.addItem("Train", "train")
        self.mode.addItem("Watch", "watch")
        _apply_tip(self.mode, "mode")

        self.start_button = QPushButton("Start training")
        self.stop_button = QPushButton("Stop")
        self.pause_button = QPushButton("Pause")
        self.save_button = QPushButton("Save checkpoint")
        self.load_button = QPushButton("Load checkpoint")
        self.load_best_button = QPushButton("Load best model")
        self.record_button = QPushButton("Record demo")
        self.stop_record_button = QPushButton("Stop recording")
        self.use_demos = QCheckBox("Load demonstrations on start")
        self.use_demos.setChecked(True)

        self.start_button.setObjectName("primaryButton")
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.load_best_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)

        _apply_tip(self.start_button, "start")
        _apply_tip(self.stop_button, "stop")
        _apply_tip(self.pause_button, "pause")
        _apply_tip(self.save_button, "save_ckpt")
        _apply_tip(self.load_button, "load_ckpt")
        _apply_tip(self.load_best_button, "load_best")
        _apply_tip(self.record_button, "record")
        _apply_tip(self.stop_record_button, "stop_record")
        _apply_tip(self.use_demos, "use_demos")

        mode_row = QHBoxLayout()
        mode_row.addWidget(_info_label("Mode", "mode"))
        mode_row.addWidget(self.mode, stretch=1)

        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        run_row.addWidget(self.start_button, stretch=2)
        run_row.addWidget(self.pause_button, stretch=1)
        run_row.addWidget(self.stop_button, stretch=1)

        demo_row = QHBoxLayout()
        demo_row.setSpacing(8)
        demo_row.addWidget(self.record_button, stretch=1)
        demo_row.addWidget(self.stop_record_button, stretch=1)

        ckpt_row = QHBoxLayout()
        ckpt_row.setSpacing(8)
        ckpt_row.addWidget(self.save_button, stretch=1)
        ckpt_row.addWidget(self.load_button, stretch=1)

        best_row = QHBoxLayout()
        best_row.setSpacing(8)
        best_row.addWidget(self.load_best_button, stretch=1)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addLayout(mode_row)
        layout.addLayout(run_row)
        layout.addWidget(self.use_demos)
        layout.addLayout(demo_row)
        layout.addLayout(ckpt_row)
        layout.addLayout(best_row)

        self.mode.currentIndexChanged.connect(self._on_mode_changed)
        self.use_demos.toggled.connect(self.changed.emit)
        self.start_button.clicked.connect(self.start_clicked.emit)
        self.stop_button.clicked.connect(self.stop_clicked.emit)
        self.pause_button.clicked.connect(self.pause_clicked.emit)
        self.save_button.clicked.connect(self.save_clicked.emit)
        self.load_button.clicked.connect(self.load_clicked.emit)
        self.load_best_button.clicked.connect(self.load_best_clicked.emit)
        self.record_button.clicked.connect(self.record_clicked.emit)
        self.stop_record_button.clicked.connect(self.stop_record_clicked.emit)
        self._refresh_start_label()

    def mode_value(self) -> str:
        data = self.mode.currentData()
        return str(data) if data is not None else "train"

    def set_mode_value(self, value: str) -> None:
        index = self.mode.findData(value)
        if index >= 0:
            self.mode.setCurrentIndex(index)

    def _on_mode_changed(self) -> None:
        self._refresh_start_label()
        self.changed.emit()

    def _refresh_start_label(self) -> None:
        if self.mode_value() == "watch":
            self.start_button.setText("Start watching")
        else:
            self.start_button.setText("Start training")

    def set_idle(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        _apply_tip(self.pause_button, "pause")
        self.save_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.load_best_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        self.mode.setEnabled(True)
        self.use_demos.setEnabled(True)
        self._refresh_start_label()

    def set_training(self) -> None:
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.load_best_button.setEnabled(True)
        self.record_button.setEnabled(False)
        self.stop_record_button.setEnabled(False)
        self.mode.setEnabled(False)
        self.use_demos.setEnabled(False)

    def set_recording(self) -> None:
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.load_best_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        self.mode.setEnabled(False)
        self.use_demos.setEnabled(False)

    def set_paused(self, paused: bool) -> None:
        self.pause_button.setText("Resume" if paused else "Pause")
        _apply_tip(self.pause_button, "resume" if paused else "pause")


class HyperparameterPanel(QGroupBox):
    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__("Learning parameters")
        self.setCheckable(True)
        self.setChecked(True)
        self.setToolTip("Collapse to hide; values stay applied while training.")

        self.device = QComboBox()
        self.device.addItems(["gpu", "auto", "cpu"])
        self.device.setCurrentText("gpu")
        _apply_tip(self.device, "device")

        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(6)
        self.learning_rate.setRange(0.000001, 0.01)
        self.learning_rate.setSingleStep(0.00001)
        self.learning_rate.setValue(0.0005)
        _apply_tip(self.learning_rate, "learning_rate")

        self.gamma = QDoubleSpinBox()
        self.gamma.setDecimals(3)
        self.gamma.setRange(0.8, 0.999)
        self.gamma.setSingleStep(0.001)
        self.gamma.setValue(0.99)
        _apply_tip(self.gamma, "gamma")

        self.epsilon_decay = QDoubleSpinBox()
        self.epsilon_decay.setDecimals(4)
        self.epsilon_decay.setRange(0.90, 0.9999)
        self.epsilon_decay.setSingleStep(0.0005)
        self.epsilon_decay.setValue(0.992)
        _apply_tip(self.epsilon_decay, "epsilon_decay")

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 512)
        self.batch_size.setValue(32)
        _apply_tip(self.batch_size, "batch_size")

        self.train_every = QSpinBox()
        self.train_every.setRange(1, 60)
        self.train_every.setValue(4)
        _apply_tip(self.train_every, "train_every")

        self.min_replay = QSpinBox()
        self.min_replay.setRange(1, 100_000)
        self.min_replay.setValue(400)
        _apply_tip(self.min_replay, "min_replay")

        self.frame_skip = QSpinBox()
        self.frame_skip.setRange(1, 20)
        self.frame_skip.setValue(1)
        _apply_tip(self.frame_skip, "frame_skip")

        self._body = QWidget()
        form = QFormLayout(self._body)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        form.addRow(_info_label("Device", "device"), self.device)
        form.addRow(_info_label("Learning rate", "learning_rate"), self.learning_rate)
        form.addRow(_info_label("Gamma", "gamma"), self.gamma)
        form.addRow(_info_label("Epsilon decay", "epsilon_decay"), self.epsilon_decay)
        form.addRow(_info_label("Batch size", "batch_size"), self.batch_size)
        form.addRow(_info_label("Train every", "train_every"), self.train_every)
        form.addRow(_info_label("Min replay", "min_replay"), self.min_replay)
        form.addRow(_info_label("Frame skip", "frame_skip"), self.frame_skip)

        self._train_only = (
            self.learning_rate,
            self.gamma,
            self.epsilon_decay,
            self.batch_size,
            self.train_every,
            self.min_replay,
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self._body)

        for widget in (
            self.device,
            self.learning_rate,
            self.gamma,
            self.epsilon_decay,
            self.batch_size,
            self.train_every,
            self.min_replay,
            self.frame_skip,
        ):
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.changed.emit)
            else:
                widget.currentTextChanged.connect(self.changed.emit)

        self.toggled.connect(self._on_group_toggled)

    def _on_group_toggled(self, checked: bool) -> None:
        self._body.setVisible(checked)

    def set_watch_mode(self, watch: bool) -> None:
        for widget in self._train_only:
            widget.setEnabled(not watch)
        self.setTitle("Inference settings" if watch else "Learning parameters")


@dataclass(slots=True)
class EpisodePoint:
    episode: int
    reward: float
    score: float


class EpisodeChart(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Episode history")
        self.points: deque[EpisodePoint] = deque(maxlen=200)
        self.setToolTip(
            "Reward and in-game score for each finished episode.\n"
            "Rising curves usually mean the agent is improving."
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.empty_label = QLabel(
            "Episode reward and score will appear here after the first run ends."
        )
        self.empty_label.setObjectName("hintLabel")
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.empty_label)

        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self.figure = Figure(figsize=(5, 2.0), tight_layout=True)
            self.figure.patch.set_facecolor("#1a1f24")
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setStyleSheet("background: #1a1f24;")
            self.canvas.setVisible(False)
            layout.addWidget(self.canvas)
            self.label = None
        except Exception:
            self.figure = None
            self.canvas = None
            self.label = QLabel("Episode chart unavailable (matplotlib missing)")
            self.label.setVisible(False)
            layout.addWidget(self.label)

    def add_point(self, point: EpisodePoint) -> None:
        self.points.append(point)
        self.redraw()

    def redraw(self) -> None:
        has_points = bool(self.points)
        self.empty_label.setVisible(not has_points)
        if self.figure is None or self.canvas is None:
            if self.label is not None:
                self.label.setVisible(has_points)
                if has_points:
                    last = self.points[-1]
                    self.label.setText(
                        f"Episode {last.episode}: reward {last.reward:.1f}, "
                        f"score {last.score:.1f}"
                    )
            return
        self.canvas.setVisible(has_points)
        if not has_points:
            return
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor("#12161a")
        episodes = [point.episode for point in self.points]
        rewards = [point.reward for point in self.points]
        scores = [point.score for point in self.points]
        axis.plot(episodes, rewards, color="#5ec8ff", label="reward", linewidth=1.5)
        axis.plot(episodes, scores, color="#7dffa3", label="score", linewidth=1.5)
        axis.legend(
            loc="upper left",
            facecolor="#1a1f24",
            edgecolor="#2a333c",
            labelcolor="#d7e0e8",
        )
        axis.set_xlabel("episode", color="#9aabba")
        axis.tick_params(colors="#9aabba")
        for spine in axis.spines.values():
            spine.set_color("#2a333c")
        self.canvas.draw_idle()


APP_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0f1317;
    color: #d7e0e8;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2a333c;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
    background-color: #151a1f;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #9ec9ff;
}
QLabel#previewPane {
    background: #080b0e;
    color: #8a9aaa;
    border: 1px solid #2a333c;
    border-radius: 10px;
    padding: 24px;
    font-size: 14px;
}
QLabel#connectionStatus {
    color: #b8c7d4;
}
QLabel#metricValue {
    color: #e8f0f6;
}
QLabel#hintLabel {
    color: #7a8b9a;
    font-weight: 400;
    font-size: 12px;
}
QLabel#statusBar {
    background: #12171c;
    border-top: 1px solid #2a333c;
    padding: 8px 10px;
    color: #a8b6c3;
}
QToolButton#infoHint {
    color: #6ea8e8;
    font-size: 12px;
    padding: 0 2px;
    border: none;
    background: transparent;
}
QToolButton#infoHint:hover, QToolButton#infoHint:focus {
    color: #9ec9ff;
}
QPushButton {
    background-color: #24303a;
    border: 1px solid #364552;
    border-radius: 6px;
    padding: 8px 12px;
    min-height: 18px;
}
QPushButton:hover {
    background-color: #2d3b47;
}
QPushButton:pressed {
    background-color: #1c262e;
}
QPushButton:disabled {
    color: #6a7885;
    background-color: #1a2229;
    border-color: #2a333c;
}
QPushButton#primaryButton {
    background-color: #1f6feb;
    border-color: #388bfd;
    color: #ffffff;
    font-weight: 600;
}
QPushButton#primaryButton:hover {
    background-color: #2b7af0;
}
QPushButton#primaryButton:disabled {
    background-color: #243447;
    border-color: #2a333c;
    color: #7a8a9a;
}
QPushButton#secondaryButton {
    background-color: #1c2833;
    border-color: #3d5163;
    color: #c5d4e0;
}
QPushButton#secondaryButton:hover {
    background-color: #243442;
}
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #0f1419;
    border: 1px solid #364552;
    border-radius: 5px;
    padding: 4px 6px;
    min-height: 22px;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #4a90d9;
}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    color: #6a7885;
    background-color: #1a2229;
}
QCheckBox {
    spacing: 8px;
}
QToolButton {
    color: #8ab4e8;
    padding: 2px 0;
}
QFrame#advancedFrame {
    background: transparent;
}
QScrollArea {
    border: none;
    background: transparent;
}
QToolTip {
    background-color: #1c2430;
    color: #e8f0f6;
    border: 1px solid #3d4f63;
    padding: 6px 8px;
    font-size: 12px;
}
"""

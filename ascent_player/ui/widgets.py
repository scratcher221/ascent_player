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
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PreviewWidget(QLabel):
    def __init__(self) -> None:
        super().__init__("No frame yet")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(560, 315)
        self.setStyleSheet("background: #050909; color: #bdfef5; border: 1px solid #355;")

    def set_png(self, data: bytes) -> None:
        image = QImage.fromData(data, "PNG")
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class BrowserPanel(QGroupBox):
    rescan_requested = pyqtSignal()
    connect_requested = pyqtSignal(str)
    launch_requested = pyqtSignal()
    windows_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__("Browser connection")
        self.status = QLabel("Scanning...")
        self.cdp_url = QComboBox()
        self.cdp_url.setEditable(True)
        self.cdp_url.addItem("http://localhost:9222")
        self.auto_launch = QCheckBox("Auto-launch if not found")
        self.auto_launch.setChecked(True)
        self.rescan = QPushButton("Rescan now")
        self.connect = QPushButton("Attach CDP URL")
        self.launch = QPushButton("Force launch new")
        self.open_game = QPushButton("Open game")
        self.window_list = QComboBox()
        self.window_list.addItem("No windows scanned")
        self.refresh_windows = QPushButton("Refresh window list")

        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.auto_launch)
        layout.addWidget(self.cdp_url)
        layout.addWidget(self.connect)
        layout.addWidget(self.rescan)
        layout.addWidget(self.window_list)
        layout.addWidget(self.refresh_windows)
        layout.addWidget(self.launch)
        layout.addWidget(self.open_game)

        self.rescan.clicked.connect(self.rescan_requested.emit)
        self.connect.clicked.connect(
            lambda: self.connect_requested.emit(self.cdp_url.currentText().strip())
        )
        self.launch.clicked.connect(self.launch_requested.emit)
        self.open_game.clicked.connect(self.launch_requested.emit)
        self.refresh_windows.clicked.connect(self.windows_requested.emit)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def set_windows(self, labels: list[str]) -> None:
        self.window_list.clear()
        if not labels:
            self.window_list.addItem("No Chromium windows found")
            return
        self.window_list.addItems(labels)


class HyperparameterPanel(QGroupBox):
    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__("DQN parameters")
        self.mode = QComboBox()
        self.mode.addItems(["train", "watch", "paused"])
        self.device = QComboBox()
        self.device.addItems(["auto", "gpu", "cpu"])

        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(6)
        self.learning_rate.setRange(0.000001, 0.01)
        self.learning_rate.setSingleStep(0.00001)
        self.learning_rate.setValue(0.0001)

        self.gamma = QDoubleSpinBox()
        self.gamma.setDecimals(3)
        self.gamma.setRange(0.8, 0.999)
        self.gamma.setSingleStep(0.001)
        self.gamma.setValue(0.99)

        self.epsilon_decay = QDoubleSpinBox()
        self.epsilon_decay.setDecimals(4)
        self.epsilon_decay.setRange(0.90, 0.9999)
        self.epsilon_decay.setSingleStep(0.0005)
        self.epsilon_decay.setValue(0.995)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 512)
        self.batch_size.setValue(32)

        self.train_every = QSpinBox()
        self.train_every.setRange(1, 60)
        self.train_every.setValue(4)

        self.min_replay = QSpinBox()
        self.min_replay.setRange(1, 100_000)
        self.min_replay.setValue(1_000)

        self.frame_skip = QSpinBox()
        self.frame_skip.setRange(1, 20)
        self.frame_skip.setValue(4)

        layout = QFormLayout(self)
        layout.addRow("Mode", self.mode)
        layout.addRow("Device", self.device)
        layout.addRow("Learning rate", self.learning_rate)
        layout.addRow("Gamma", self.gamma)
        layout.addRow("Epsilon decay", self.epsilon_decay)
        layout.addRow("Batch size", self.batch_size)
        layout.addRow("Train every", self.train_every)
        layout.addRow("Min replay", self.min_replay)
        layout.addRow("Frame skip", self.frame_skip)

        for widget in (
            self.mode,
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


@dataclass(slots=True)
class EpisodePoint:
    episode: int
    reward: float
    score: float


class EpisodeChart(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.points: deque[EpisodePoint] = deque(maxlen=200)
        layout = QVBoxLayout(self)
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self.figure = Figure(figsize=(5, 2.2), tight_layout=True)
            self.canvas = FigureCanvasQTAgg(self.figure)
            layout.addWidget(self.canvas)
            self.label = None
        except Exception:
            self.figure = None
            self.canvas = None
            self.label = QLabel("Episode chart unavailable")
            layout.addWidget(self.label)

    def add_point(self, point: EpisodePoint) -> None:
        self.points.append(point)
        self.redraw()

    def redraw(self) -> None:
        if self.figure is None or self.canvas is None:
            if self.label is not None and self.points:
                last = self.points[-1]
                self.label.setText(
                    f"Episode {last.episode}: reward {last.reward:.1f}, score {last.score:.1f}"
                )
            return
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        episodes = [point.episode for point in self.points]
        rewards = [point.reward for point in self.points]
        scores = [point.score for point in self.points]
        axis.plot(episodes, rewards, label="reward")
        axis.plot(episodes, scores, label="score")
        axis.legend(loc="upper left")
        axis.set_xlabel("episode")
        self.canvas.draw_idle()

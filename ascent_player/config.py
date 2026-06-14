from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


ASCENT_URL = "https://ascent.xrd.workers.dev/"
ASCENT_HOST = "ascent.xrd.workers.dev"


class DeviceMode(str, Enum):
    AUTO = "auto"
    GPU = "gpu"
    CPU = "cpu"


class RunMode(str, Enum):
    TRAIN = "train"
    WATCH = "watch"
    PAUSED = "paused"


@dataclass(slots=True)
class BrowserConfig:
    ascent_url: str = ASCENT_URL
    host_match: str = ASCENT_HOST
    cdp_ports: tuple[int, ...] = tuple(range(9222, 9230))
    cdp_timeout_seconds: float = 0.25
    manual_cdp_url: str | None = None
    auto_launch_on_miss: bool = True
    chromium_path: str | None = None
    viewport_width: int = 1280
    viewport_height: int = 720
    rescan_seconds: int = 5
    canvas_selector: str = "#gameCanvas"
    # Read canvas pixels via JS instead of Playwright element screenshots.
    # Element screenshots scroll into view and cause visible flicker.
    use_js_canvas_capture: bool = True
    dom_poll_interval: int = 12


@dataclass(slots=True)
class ObservationConfig:
    width: int = 84
    height: int = 84
    frame_stack: int = 4


@dataclass(slots=True)
class RewardConfig:
    survival: float = 0.1
    score_gain: float = 1.0
    altitude_gain: float = 0.5
    death: float = -50.0
    idle_penalty: float = -0.05
    idle_steps: int = 30


@dataclass(slots=True)
class TrainingConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    replay_buffer_size: int = 50_000
    batch_size_cpu: int = 32
    batch_size_gpu: int = 64
    min_replay_size: int = 1_000
    target_sync_interval: int = 1_000
    frame_skip: int = 4
    train_every_cpu: int = 4
    train_every_gpu: int = 1
    checkpoint_path: Path = Path("checkpoints/dqn_latest.keras")
    device_mode: DeviceMode = DeviceMode.AUTO
    watch_mode: bool = False


@dataclass(slots=True)
class AppConfig:
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def action_count(self) -> int:
        return 6

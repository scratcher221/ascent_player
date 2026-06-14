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
    include_boost_channel: bool = True

    @property
    def channel_count(self) -> int:
        return self.frame_stack + (1 if self.include_boost_channel else 0)


@dataclass(slots=True)
class RewardConfig:
    survival: float = 0.25
    score_gain: float = 1.0
    altitude_gain: float = 0.8
    death: float = -50.0
    early_death_penalty: float = -25.0
    early_death_steps: int = 80
    falling_penalty: float = -0.35
    idle_penalty: float = -0.03
    idle_steps: int = 40
    wasted_jump_penalty: float = -0.35
    boost_gain: float = 2.0
    boost_spent: float = 0.0
    low_boost_penalty: float = -0.05
    boost_jump_threshold: float = 0.06
    survival_step_bonus: float = 0.002


@dataclass(slots=True)
class DemoConfig:
    save_dir: Path = Path("demonstrations")
    replay_multiplier: int = 3
    pretrain_steps: int = 300
    use_demos_on_start: bool = True
    # Keep at least this much RAM free for the OS while loading demos.
    os_memory_reserve_mb: int = 2048
    # Hard cap on unique demo transitions; None = derive from available memory.
    max_transitions: int | None = None


@dataclass(slots=True)
class TrainingConfig:
    learning_rate: float = 5e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.992
    replay_buffer_size: int = 50_000
    batch_size_cpu: int = 32
    batch_size_gpu: int = 64
    min_replay_size: int = 400
    target_sync_interval: int = 800
    frame_skip: int = 4
    train_every_cpu: int = 2
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
    demo: DemoConfig = field(default_factory=DemoConfig)

    @property
    def action_count(self) -> int:
        return 6

from __future__ import annotations

import argparse
import sys

# Configure NVIDIA pip library paths before TensorFlow can be imported anywhere.
from ascent_player.utils.gpu_env import bootstrap_gpu_environment

bootstrap_gpu_environment()

from ascent_player.config import AppConfig, DeviceMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ascent Neural Network Player")
    parser.add_argument("--cdp", help="Attach to a specific Chrome DevTools URL")
    parser.add_argument(
        "--no-auto-launch",
        action="store_true",
        help="Only auto-detect an existing Ascent tab; do not launch Chromium",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Run in inference/watch mode with epsilon set to zero",
    )
    parser.add_argument(
        "--device",
        choices=[mode.value for mode in DeviceMode],
        default=DeviceMode.GPU.value,
        help="Compute device preference for TensorFlow",
    )
    parser.add_argument(
        "--chromium-path",
        help="Executable path used when the app launches Chromium",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run a minimal training loop without the PyQt interface",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Use the headless physics simulator instead of the browser",
    )
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        default=0,
        help="Run fast simulator pretraining for N environment steps",
    )
    parser.add_argument(
        "--transfer-from-sim",
        action="store_true",
        help="Load the simulator checkpoint and fine-tune in the browser",
    )
    parser.add_argument(
        "--calibrate-sim",
        action="store_true",
        help="Run a short random-policy calibration report for the simulator",
    )
    parser.add_argument(
        "--sim-envs",
        type=int,
        default=0,
        help="Parallel simulator envs for pretrain (0 = auto from CPU count)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig()
    config.browser.manual_cdp_url = args.cdp
    config.browser.auto_launch_on_miss = not args.no_auto_launch
    config.browser.chromium_path = args.chromium_path
    config.training.device_mode = DeviceMode(args.device)
    config.training.watch_mode = args.watch
    config.training.sim_mode = args.sim
    config.training.sim_pretrain_steps = max(0, int(args.pretrain_steps))
    config.training.transfer_from_sim = args.transfer_from_sim
    config.training.sim_pretrain_envs = max(0, int(args.sim_envs))
    if args.watch:
        config.training.epsilon_start = 0.0
        config.training.epsilon_end = 0.0
    if args.transfer_from_sim and not args.sim:
        config.training.frame_skip = max(config.training.frame_skip, 2)
    return config


def run_ui(config: AppConfig) -> int:
    from PyQt6.QtWidgets import QApplication

    from ascent_player.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow(config)
    window.show()
    return app.exec()


def run_no_ui(config: AppConfig) -> int:
    import asyncio

    from ascent_player.training import (
        run_sim_calibration,
        run_sim_pretrain,
        run_training_no_ui,
    )

    if config.training.sim_pretrain_steps > 0:
        run_sim_pretrain(config, config.training.sim_pretrain_steps)
        return 0
    if config.training.sim_mode and "--calibrate-sim" in sys.argv:
        asyncio.run(run_sim_calibration(config))
        return 0
    asyncio.run(run_training_no_ui(config))
    return 0


def main() -> int:
    args = parse_args()
    config = build_config(args)
    if args.calibrate_sim:
        import asyncio

        from ascent_player.training import run_sim_calibration

        config.training.sim_mode = True
        asyncio.run(run_sim_calibration(config))
        return 0
    if args.pretrain_steps > 0:
        config.training.sim_mode = True
        from ascent_player.training import run_sim_pretrain

        run_sim_pretrain(config, args.pretrain_steps)
        return 0
    if args.no_ui:
        return run_no_ui(config)
    return run_ui(config)


if __name__ == "__main__":
    raise SystemExit(main())

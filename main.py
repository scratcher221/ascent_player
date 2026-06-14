from __future__ import annotations

import argparse
import sys

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
        default=DeviceMode.AUTO.value,
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
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig()
    config.browser.manual_cdp_url = args.cdp
    config.browser.auto_launch_on_miss = not args.no_auto_launch
    config.browser.chromium_path = args.chromium_path
    config.training.device_mode = DeviceMode(args.device)
    config.training.watch_mode = args.watch
    if args.watch:
        config.training.epsilon_start = 0.0
        config.training.epsilon_end = 0.0
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

    from ascent_player.training import run_training_no_ui

    asyncio.run(run_training_no_ui(config))
    return 0


def main() -> int:
    args = parse_args()
    config = build_config(args)
    if args.no_ui:
        return run_no_ui(config)
    return run_ui(config)


if __name__ == "__main__":
    raise SystemExit(main())

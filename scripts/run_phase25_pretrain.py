#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from ascent_player.config import AppConfig, DeviceMode
from ascent_player.training import run_sim_pretrain


def main() -> int:
    config = AppConfig()
    config.training.device_mode = DeviceMode.GPU
    config.training.sim_pretrain_envs = 12
    config.mechanics_curriculum.teacher_episodes = 24
    config.training.sim_eval_every_steps = 20_000
    config.training.sim_eval_episodes = 5
    config.training.consistency_eval_episodes = 10
    config.training.sim_min_best_score = 10_000
    config.training.sim_eval_max_steps = 10_000
    config.demo.max_transitions = 6_000
    config.demo.pretrain_steps = 400
    print("CONFIG_READY", flush=True)
    run_sim_pretrain(config, 800_000)
    print("PRETRAIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

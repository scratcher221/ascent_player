#!/usr/bin/env python3
"""Aligned sim pretrain — rendered observations for browser transfer.

Uses browser-like `render_sim_frame` + preprocess (sim_fast_observations=False),
resumes from the best available checkpoint, and writes under checkpoints/aligned_*.

Does NOT touch checkpoints/browser_best.keras.

Usage:
  PYTHONPATH=. python -u scripts/run_aligned_sim_pretrain.py
  PYTHONPATH=. python -u scripts/run_aligned_sim_pretrain.py --steps 100000
  PYTHONPATH=. python -u scripts/run_aligned_sim_pretrain.py --steps 200000 --envs 4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.training import run_sim_pretrain


def main() -> int:
    parser = argparse.ArgumentParser(description="Rendered-obs sim pretrain for transfer")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument(
        "--from-browser-best",
        action="store_true",
        help="Seed weights from browser_best.keras (read-only; never overwritten)",
    )
    parser.add_argument(
        "--promote-sim-pretrained",
        action="store_true",
        help="Also copy aligned checkpoint to sim_pretrained.keras (logged clearly)",
    )
    args = parser.parse_args()

    config = AppConfig()
    config.training.device_mode = DeviceMode.GPU
    config.training.sim_mode = True
    config.training.sim_fast_observations = False  # rendered ≈ browser preprocess
    config.training.sim_pretrain_envs = max(1, args.envs)
    config.training.sim_resume_from_best = True
    config.training.sim_resume_epsilon = 0.20
    config.training.sim_warmstart_teacher = True
    config.training.sim_warmstart_demos = False
    config.training.sim_eval_every_steps = 20_000
    config.training.sim_eval_episodes = 6
    config.training.learning_rate = 1.5e-4
    config.training.sim_min_best_score = 0  # don't force extra steps past --steps
    config.mechanics_curriculum.teacher_episodes = 60
    config.training.rule_prior_start = 0.40
    config.training.rule_prior_end = 0.08
    config.training.rule_prior_steps = 150_000

    aligned_dir = Path("checkpoints")
    aligned_dir.mkdir(parents=True, exist_ok=True)
    aligned_latest = aligned_dir / "aligned_sim_latest.keras"
    aligned_best = aligned_dir / "aligned_sim_best_eval.keras"
    config.training.sim_checkpoint_path = aligned_latest
    config.training.sim_best_eval_checkpoint_path = aligned_best
    # Prefer aligned checkpoints over playable/browser for this script.
    config.training.playable_checkpoint_path = aligned_best

    agent = DQNAgent(config)
    src = None
    if aligned_best.exists():
        src = aligned_best
    elif aligned_latest.exists():
        src = aligned_latest
    else:
        src = agent.resolve_best_checkpoint()
    browser_best = config.training.browser_best_checkpoint_path
    if args.from_browser_best and browser_best.exists():
        src = browser_best
    src_name = src.name if src is not None else "none"
    print(
        f"ALIGNED_PRETRAIN_START steps={args.steps} envs={args.envs} "
        f"fast_obs={config.training.sim_fast_observations} resume_from={src_name}",
        flush=True,
    )
    print(
        "NOTE: Writing checkpoints/aligned_sim_*.keras only; "
        "browser_best.keras will not be overwritten.",
        flush=True,
    )

    start_steps = 0
    if src is not None and agent.load(src):
        start_steps = int(agent.progress.total_steps)
        # Seed aligned path so run_sim_pretrain resume logic can find it if needed.
        agent.save(aligned_latest)
        print(
            f"ALIGNED_SEEDED_FROM {src.name} steps={start_steps} "
            f"best={agent.progress.best_score:.0f}",
            flush=True,
        )

    target = start_steps + max(1, args.steps)
    run_sim_pretrain(config, target)

    done = DQNAgent(config)
    if done.load(aligned_latest):
        done.save(aligned_best)
        print(
            f"ALIGNED_PRETRAIN_DONE latest={aligned_latest} best_eval={aligned_best} "
            f"steps={done.progress.total_steps} best_score={done.progress.best_score:.0f}",
            flush=True,
        )
        if args.promote_sim_pretrained:
            dest = Path("checkpoints/sim_pretrained.keras")
            print(
                f"NOTE: Promoting aligned weights -> {dest} "
                f"(explicit --promote-sim-pretrained)",
                flush=True,
            )
            done.save(dest)
    else:
        print("ALIGNED_PRETRAIN_WARN no checkpoint written", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

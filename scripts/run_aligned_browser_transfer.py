#!/usr/bin/env python3
"""After aligned sim pretrain: transfer to browser aiming for consistent 5k Watch.

Usage (after aligned_sim_best_eval exists):
  PYTHONPATH=. python -u scripts/run_aligned_browser_transfer.py --hours 4
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ascent_player.agent.dqn import DQNAgent


def _load_transfer_mod():
    path = ROOT / "scripts" / "run_browser_transfer.py"
    spec = importlib.util.spec_from_file_location("run_browser_transfer", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--session-minutes", type=float, default=10.0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--target-mean", type=float, default=5000.0)
    parser.add_argument("--target-min", type=float, default=1500.0)
    args = parser.parse_args()

    transfer = _load_transfer_mod()
    aligned = Path("checkpoints/aligned_sim_best_eval.keras")
    aligned_latest = Path("checkpoints/aligned_sim_latest.keras")
    src = aligned if aligned.exists() else aligned_latest
    if not src.exists():
        print("MISSING_ALIGNED_CHECKPOINT", flush=True)
        return 1

    config = transfer._build_config()
    config.training.transfer_from_sim = True
    config.training.transfer_learning_rate = 3e-5
    config.training.transfer_epsilon_start = 0.18
    config.training.browser_epsilon_cap = 0.18
    config.training.learning_rate = 3e-5
    config.training.sim_best_eval_checkpoint_path = src
    config.training.sim_checkpoint_path = aligned_latest

    agent = DQNAgent(config)
    assert agent.load(src)
    # Keep a dedicated copy for the transfer loader without wiping aligned files.
    print(f"ALIGNED_TRANSFER_SEED {src} best={agent.progress.best_score}", flush=True)

    deadline = time.time() + max(600.0, args.hours * 3600.0)
    print(
        f"ALIGNED_BROWSER_TRANSFER_START hours={args.hours} "
        f"target_mean>={args.target_mean} target_min>={args.target_min}",
        flush=True,
    )
    return asyncio.run(
        transfer._transfer_loop(
            config,
            deadline=deadline,
            session_seconds=max(180, int(args.session_minutes * 60)),
            eval_episodes=max(5, args.eval_episodes),
            target_mean=args.target_mean,
            target_min=args.target_min,
            start_from_browser_best=False,
        )
    )


if __name__ == "__main__":
    sys.exit(main())

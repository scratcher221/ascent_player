#!/usr/bin/env python3
"""Sweep Watch-time rule prior on frozen seed_map_best (no weight updates).

Online TD/BC both eroded seed Q-values; this searches inference blends instead.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ascent_player.agent.checkpoint import checkpoint_exists
from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.training import run_eval_watch

SEED_BEST = Path("checkpoints/seed_map_best.keras")
SEED_MEAN = Path("logs/seed_map_best_mean.txt")
LATEST = Path("checkpoints/dqn_latest.keras")


def _load_seed(config: AppConfig) -> None:
    agent = DQNAgent(config)
    src = SEED_BEST if checkpoint_exists(SEED_BEST) else LATEST
    assert agent.load(src), f"failed to load {src}"
    agent.save(LATEST)


async def eval_prior(config: AppConfig, prior: float, episodes: int) -> dict:
    config.training.watch_rule_prior = float(prior)
    _load_seed(config)
    print(f"WATCH_SWEEP prior={prior:.2f} eps={episodes}", flush=True)
    stats = await run_eval_watch(config, max_episodes=episodes)
    mean = float(stats.get("recent_avg", 0.0))
    emin = float(stats.get("recent_min", 0.0))
    emax = float(stats.get("recent_max", 0.0))
    print(
        f"WATCH_RESULT prior={prior:.2f} mean={mean:.1f} min={emin:.1f} max={emax:.1f}",
        flush=True,
    )
    return {"prior": prior, "mean": mean, "min": emin, "max": emax}


async def main_async(args: argparse.Namespace) -> int:
    deadline = time.time() + max(300.0, args.hours * 3600.0)
    target_mean = float(args.target_mean)
    target_min = float(args.target_min)

    config = AppConfig()
    config.training.sim_mode = False
    config.training.device_mode = DeviceMode.AUTO
    config.training.frame_skip = 1
    config.training.transfer_frame_skip = 1
    config.training.log_decision_every = 1
    config.browser.run_seed = int(args.run_seed)
    config.browser.lock_run_seed = True

    priors = [float(x) for x in args.priors.split(",") if x.strip()]
    best = {"prior": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    if SEED_MEAN.exists():
        try:
            best["mean"] = float(SEED_MEAN.read_text().strip())
        except ValueError:
            pass

    print(
        f"WATCH_SWEEP_START seed={args.run_seed} hours={args.hours} "
        f"target_mean>={target_mean} priors={priors} floor_mean={best['mean']:.1f}",
        flush=True,
    )

    round_id = 0
    while time.time() < deadline:
        round_id += 1
        remaining = deadline - time.time()
        if remaining < 180:
            break
        # Coarse then refine around winners.
        if round_id == 1:
            grid = priors
            episodes = max(6, args.eval_episodes)
        else:
            center = best["prior"]
            grid = sorted(
                {
                    max(0.0, min(0.85, center + d))
                    for d in (-0.12, -0.06, 0.0, 0.06, 0.12, 0.18)
                }
            )
            episodes = max(8, args.eval_episodes + 2)

        print(
            f"WATCH_SWEEP_ROUND id={round_id} remaining_s={remaining:.0f} "
            f"grid={grid} episodes={episodes}",
            flush=True,
        )
        for prior in grid:
            if time.time() + 120 > deadline:
                break
            stats = await eval_prior(config, prior, episodes)
            if stats["mean"] > best["mean"] and stats["min"] >= target_min * 0.4:
                best = stats
                SEED_MEAN.write_text(f"{best['mean']:.4f}\n", encoding="utf-8")
                print(
                    f"WATCH_BEST prior={best['prior']:.2f} mean={best['mean']:.1f} "
                    f"min={best['min']:.1f}",
                    flush=True,
                )
            if stats["mean"] >= target_mean and stats["min"] >= target_min:
                print(
                    f"SEED_TARGET_MET mean={stats['mean']:.1f} min={stats['min']:.1f} "
                    f"prior={prior:.2f}",
                    flush=True,
                )
                return 0

    print(
        f"WATCH_SWEEP_END best_prior={best['prior']:.2f} best_mean={best['mean']:.1f}",
        flush=True,
    )
    return 0 if best["mean"] >= target_mean else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=2.0)
    p.add_argument("--run-seed", type=int, default=424242)
    p.add_argument("--target-mean", type=float, default=2000.0)
    p.add_argument("--target-min", type=float, default=900.0)
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument(
        "--priors",
        type=str,
        default="0.0,0.12,0.22,0.32,0.42,0.55,0.70",
    )
    return asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

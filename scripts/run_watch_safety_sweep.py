#!/usr/bin/env python3
"""Compare pure greedy vs Watch safety-override on frozen seed_map_best."""
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
LATEST = Path("checkpoints/dqn_latest.keras")
SEED_MEAN = Path("logs/seed_map_best_mean.txt")


def _sync_seed(config: AppConfig) -> None:
    agent = DQNAgent(config)
    src = SEED_BEST if checkpoint_exists(SEED_BEST) else LATEST
    assert agent.load(src)
    agent.save(LATEST)


async def one_eval(config: AppConfig, *, safety: bool, episodes: int, tag: str) -> dict:
    config.training.watch_rule_prior = 0.0
    config.training.watch_safety_override = bool(safety)
    _sync_seed(config)
    print(f"SAFETY_EVAL tag={tag} safety={safety} eps={episodes}", flush=True)
    stats = await run_eval_watch(config, max_episodes=episodes)
    out = {
        "tag": tag,
        "safety": safety,
        "mean": float(stats.get("recent_avg", 0.0)),
        "min": float(stats.get("recent_min", 0.0)),
        "max": float(stats.get("recent_max", 0.0)),
    }
    print(
        f"SAFETY_RESULT tag={tag} safety={safety} "
        f"mean={out['mean']:.1f} min={out['min']:.1f} max={out['max']:.1f}",
        flush=True,
    )
    return out


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

    best = {"mean": 0.0, "min": 0.0, "max": 0.0, "safety": False, "tag": "none"}
    if SEED_MEAN.exists():
        try:
            best["mean"] = float(SEED_MEAN.read_text().strip())
        except ValueError:
            pass

    print(
        f"SAFETY_SWEEP_START seed={args.run_seed} hours={args.hours} "
        f"target>={target_mean} floor={best['mean']:.1f}",
        flush=True,
    )

    round_id = 0
    modes = [False, True]
    while time.time() < deadline:
        round_id += 1
        remaining = deadline - time.time()
        if remaining < 200:
            break
        episodes = max(8, args.eval_episodes + (2 if round_id > 1 else 0))
        print(
            f"SAFETY_ROUND id={round_id} remaining_s={remaining:.0f} episodes={episodes}",
            flush=True,
        )
        for safety in modes:
            if time.time() + 150 > deadline:
                break
            tag = f"r{round_id}_{'safe' if safety else 'greedy'}"
            stats = await one_eval(config, safety=safety, episodes=episodes, tag=tag)
            if stats["mean"] > best["mean"] and stats["min"] >= target_min * 0.4:
                best = stats
                SEED_MEAN.write_text(f"{best['mean']:.4f}\n", encoding="utf-8")
                print(
                    f"SAFETY_BEST mean={best['mean']:.1f} min={best['min']:.1f} "
                    f"safety={best['safety']}",
                    flush=True,
                )
            if stats["mean"] >= target_mean and stats["min"] >= target_min:
                print(
                    f"SEED_TARGET_MET mean={stats['mean']:.1f} min={stats['min']:.1f} "
                    f"safety={safety}",
                    flush=True,
                )
                return 0

    print(
        f"SAFETY_SWEEP_END best_mean={best['mean']:.1f} safety={best.get('safety')}",
        flush=True,
    )
    return 0 if best["mean"] >= target_mean else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=1.5)
    p.add_argument("--run-seed", type=int, default=424242)
    p.add_argument("--target-mean", type=float, default=2000.0)
    p.add_argument("--target-min", type=float, default=900.0)
    p.add_argument("--eval-episodes", type=int, default=10)
    return asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

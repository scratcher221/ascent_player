#!/usr/bin/env python3
"""One-shot: collect skill-labeled browser demos and BC the skill head."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ascent_player.agent.checkpoint import checkpoint_exists
from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig, DeviceMode
from scripts.run_thread_bc_cycle import (  # noqa: E402
    BROWSER_REPLAY,
    LATEST,
    SEED_BEST,
    THREAD_BC,
    _build_config,
    _enable_skill_teacher,
    _save_thread_bc,
    phase_collect,
)


async def main_async(args: argparse.Namespace) -> int:
    config = _build_config(int(args.run_seed), elite_gate=float(args.elite_gate))
    _enable_skill_teacher(config, 0.95)
    config.training.browser_replay_path = Path(args.replay)
    await phase_collect(
        config,
        seconds=max(120, int(args.minutes * 60)),
        eps=0.02,
        thread_prior=0.04,
        skip_replay_load=True,
    )
    agent = DQNAgent(config)
    src = THREAD_BC if checkpoint_exists(THREAD_BC) else SEED_BEST
    assert agent.load(src), f"load failed {src}"
    agent.replay.clear()
    replay_path = Path(args.replay)
    if replay_path.exists():
        agent.replay.load_pickle(
            replay_path,
            max_items=config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )
    loss = agent.train_skill_head(steps=int(args.bc_steps), lr=float(args.bc_lr))
    _save_thread_bc(agent)
    agent.save(LATEST)
    print(
        f"SKILL_BOOTSTRAP_DONE replay={len(agent.replay)} "
        f"replay_path={replay_path} loss={loss}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-seed", type=int, default=424242)
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--bc-steps", type=int, default=400)
    parser.add_argument("--bc-lr", type=float, default=2e-5)
    parser.add_argument("--elite-gate", type=float, default=1600.0)
    parser.add_argument(
        "--replay",
        type=Path,
        default=BROWSER_REPLAY,
        help="Replay pickle containing skill-labeled transitions.",
    )
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

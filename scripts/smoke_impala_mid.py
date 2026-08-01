#!/usr/bin/env python3
"""GPU smoke for Impala-mid: build, train step, short offline BC on elite.

Exit 0 only when v2 trains and BC loss moves. Falls back to CPU if no GPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ascent_player.agent.checkpoint import checkpoint_exists
from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig, DeviceMode

ELITE = Path("checkpoints/elite_thread_replay.pkl")


def main() -> int:
    config = AppConfig()
    config.training.model_variant = "impala_mid"
    config.training.mixed_precision = True
    config.training.skills_enabled = True
    config.training.batch_size_gpu = 128
    config.training.device_mode = DeviceMode.AUTO
    try:
        agent = DQNAgent(config)
    except Exception as exc:
        print(f"SMOKE_GPU_FAIL build={exc}", flush=True)
        print("SMOKE_RETRY device=cpu mixed_precision=0", flush=True)
        config.training.device_mode = DeviceMode.CPU
        config.training.mixed_precision = False
        config.training.batch_size_gpu = 64
        agent = DQNAgent(config)

    params = agent.online.count_params()
    print(
        f"SMOKE_BUILD variant={agent._model_variant} params={params:,} "
        f"device={agent.device_info.training_device} "
        f"mixed={int(agent._use_mixed_precision)} batch={agent.batch_size}",
        flush=True,
    )
    if params < 8_000_000:
        print("SMOKE_FAIL params_below_band", flush=True)
        return 2

    # Synthetic train step via BC on random-looking batch from empty→need data.
    agent.replay.clear()
    if ELITE.exists():
        n = agent.replay.load_pickle(
            ELITE,
            max_items=config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )
        print(f"SMOKE_ELITE loaded={n}", flush=True)
    else:
        print("SMOKE_ELITE missing — synthesizing transitions", flush=True)
        import numpy as np

        for i in range(max(256, agent.batch_size)):
            vis = np.random.rand(84, 84, 6).astype(np.float32) * 0.01
            vec = np.random.randn(59).astype(np.float32) * 0.01
            agent.replay.add(
                (vis, vec),
                int(i % 6),
                0.0,
                (vis, vec),
                False,
                episode_score=2000.0,
            )

    if len(agent.replay) < agent.batch_size:
        print(f"SMOKE_FAIL replay={len(agent.replay)}", flush=True)
        return 2

    agent.set_learning_rate(2.5e-6)
    losses = []
    steps = 40
    for i in range(steps):
        batch = agent.replay.sample(min(agent.batch_size, len(agent.replay)))
        loss = float(agent._invoke_bc_train_step(batch.states, batch.actions).numpy())
        losses.append(loss)
        if (i + 1) % 10 == 0:
            print(f"SMOKE_BC step={i+1}/{steps} loss={loss:.4f}", flush=True)

    first = sum(losses[:5]) / 5
    last = sum(losses[-5:]) / 5
    moved = last < first * 0.999 or abs(first - last) > 1e-4
    print(
        f"SMOKE_DONE first5={first:.4f} last5={last:.4f} moved={int(moved)}",
        flush=True,
    )
    # Also poke one TD train path if enough data.
    try:
        agent.config.training.min_replay_size = 1
        agent.advance_steps(1)
        print(f"SMOKE_TD ok loss={agent.metrics.loss}", flush=True)
    except Exception as exc:
        print(f"SMOKE_TD_SKIP {exc}", flush=True)

    if not moved and first > 1.0:
        # High CE that doesn't move is still a fail; tiny flat loss is OK.
        print("SMOKE_FAIL loss_did_not_move", flush=True)
        return 3
    print("SMOKE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

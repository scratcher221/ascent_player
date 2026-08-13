#!/usr/bin/env python3
"""Supervised seed climb: collect (no TD) → offline TD → ε=0 Watch eval.

Avoids the online TD collapse seen when fine-tuning seed_map_best in-session.
Loops until Watch mean >= target or wall-clock deadline.

Usage:
  PYTHONPATH=. python -u scripts/run_seed_climb_supervisor.py --hours 4 --run-seed 424242
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
from ascent_player.agent.offline_bc import offline_bc_frozen_trunk
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.training import run_eval_watch, run_training_no_ui
from ascent_player.utils.policy_floors import read_seed_floor, write_seed_floor
from ascent_player.utils.run_profile import (
    collect_only_fields,
    override_attrs,
    override_training,
)


SEED_BEST = Path("checkpoints/seed_map_best.keras")
LATEST = Path("checkpoints/dqn_latest.keras")


def _build_config(run_seed: int) -> AppConfig:
    config = AppConfig()
    config.training.sim_mode = False
    config.training.device_mode = DeviceMode.AUTO
    config.training.frame_skip = 1
    config.training.transfer_frame_skip = 1
    config.training.mixed_sim_replay_ratio = 0.0
    config.training.log_decision_every = 1
    config.training.reason_aux_weight = 0.12
    config.training.replay_min_episode_score = 1200.0
    config.browser.run_seed = int(run_seed)
    config.browser.lock_run_seed = True
    # Landing / combo focused (already in MechanicsRewardConfig defaults).
    config.mechanics_reward.steer_gain = 0.24
    config.mechanics_reward.wrong_way_penalty = -0.30
    config.mechanics_reward.aligned_bonus = 0.05
    config.mechanics_reward.platform_land = 1.7
    config.mechanics_reward.combo_gain = 0.45
    return config


def _load_seed_agent(config: AppConfig) -> DQNAgent:
    agent = DQNAgent(config)
    src = SEED_BEST if checkpoint_exists(SEED_BEST) else LATEST
    assert agent.load(src), f"failed to load {src}"
    agent.save(LATEST)
    return agent


def _read_best_mean() -> float:
    return read_seed_floor(default=0.0)


def offline_td_train(agent: DQNAgent, *, steps: int, lr: float) -> float | None:
    """Run pure TD updates from the current browser replay (no env interaction)."""
    if len(agent.replay) < max(64, agent.batch_size):
        print(f"OFFLINE_SKIP replay={len(agent.replay)} too small", flush=True)
        return None
    agent.set_learning_rate(lr)
    # Bypass min_replay gate used by maybe_train.
    prev_min = agent.config.training.min_replay_size
    agent.config.training.min_replay_size = 0
    prev_every = agent.train_every
    agent.train_every = 1
    last_loss = None
    try:
        for i in range(max(1, steps)):
            batch = agent.sample_training_batch()
            with agent.tf.device(agent.device_info.training_device):
                loss = agent.train_batch(batch)
            last_loss = float(loss)
            if (i + 1) % max(1, steps // 5) == 0:
                print(
                    f"OFFLINE_TD step={i+1}/{steps} loss={last_loss:.4f} "
                    f"replay={len(agent.replay)}",
                    flush=True,
                )
            if (i + 1) % agent.config.training.target_sync_interval == 0:
                agent.sync_target_network(hard=True)
            else:
                agent.sync_target_network(hard=False)
    finally:
        agent.config.training.min_replay_size = prev_min
        agent.train_every = prev_every
    agent.save(LATEST)
    return last_loss


def offline_bc_train(agent: DQNAgent, *, steps: int, lr: float) -> float | None:
    return offline_bc_frozen_trunk(
        agent, steps=steps, lr=lr, save_path=LATEST, log_prefix="OFFLINE_BC"
    )


async def collect_rollouts(
    config: AppConfig,
    *,
    seconds: int,
    eps: float,
    rule: float,
    skip_replay_load: bool,
) -> dict:
    with override_training(
        config,
        **collect_only_fields(
            eps=eps, skip_replay_load=skip_replay_load, rule_prior=rule
        ),
        learning_rate=1e-5,
    ), override_attrs(config.demo, use_demos_on_start=False):
        print(
            f"COLLECT start s={seconds} eps={eps} rule={rule} "
            f"gate>={config.training.replay_min_episode_score} "
            f"skip_load={skip_replay_load}",
            flush=True,
        )
        return await run_training_no_ui(
            config,
            max_seconds=seconds,
            ingest_demos=False,
        )


async def main_async(args: argparse.Namespace) -> int:
    deadline = time.time() + max(600.0, args.hours * 3600.0)
    target_mean = float(args.target_mean)
    target_min = float(args.target_min)
    run_seed = int(args.run_seed)

    config = _build_config(run_seed)
    agent = _load_seed_agent(config)
    best_mean = max(_read_best_mean(), 0.0)
    if best_mean <= 0:
        best_mean = 1213.1
        write_seed_floor(best_mean)

    print(
        f"SEED_CLIMB_START seed={run_seed} hours={args.hours} "
        f"best_mean={best_mean:.1f} target_mean>={target_mean} "
        f"target_min>={target_min} replay={len(agent.replay)}",
        flush=True,
    )

    round_id = 0
    # Fixed gentle BC — ramping steps on near-miss previously hurt round 3.
    collect_eps = 0.04
    collect_rule = 0.12
    offline_steps = 500
    offline_lr = 3e-6
    use_bc = True
    collect_seconds = int(args.collect_minutes * 60)
    eval_episodes = max(8, args.eval_episodes)
    fresh_collect = True  # first round: empty buffer (drop any poisoned pickle)

    while time.time() < deadline:
        round_id += 1
        remaining = deadline - time.time()
        if remaining < 240:
            print("SEED_CLIMB_TIME_LOW", flush=True)
            break

        # Always start collect from current seed best weights.
        agent = _load_seed_agent(config)
        agent.epsilon = collect_eps
        agent.metrics.epsilon = collect_eps
        agent.progress.epsilon = collect_eps
        agent.save(LATEST)

        sess = min(collect_seconds, max(180, int(remaining - 180)))
        print(
            f"SEED_CLIMB_ROUND id={round_id} collect_s={sess} remaining_s={remaining:.0f} "
            f"eps={collect_eps} rule={collect_rule} offline_steps={offline_steps} "
            f"lr={offline_lr:.2e} mode={'BC' if use_bc else 'TD'}",
            flush=True,
        )

        collect_stats = await collect_rollouts(
            config,
            seconds=sess,
            eps=collect_eps,
            rule=collect_rule,
            skip_replay_load=fresh_collect,
        )
        replay_path = Path("checkpoints/browser_replay.pkl")
        replay_n = 0
        if replay_path.exists():
            # Count via quick load into a throwaway agent buffer size from stats if present.
            replay_n = int(collect_stats.get("replay_size", 0) or 0)
        print(
            f"COLLECT_DONE recent_avg={collect_stats.get('recent_avg', 0):.0f} "
            f"replay_size≈{replay_n} file={'yes' if replay_path.exists() else 'no'}",
            flush=True,
        )
        fresh_collect = False

        # Reload agent + replay for offline TD from seed best (not collapsed play weights).
        agent = _load_seed_agent(config)
        agent.replay.clear()
        loaded = agent.replay.load_pickle(
            config.training.browser_replay_path,
            max_items=config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )
        print(f"OFFLINE_LOAD replay={loaded}", flush=True)
        if loaded < 200:
            print("OFFLINE_WAIT need more gated transitions", flush=True)
            # Loosen gate slightly if starving.
            config.training.replay_min_episode_score = max(
                900.0, config.training.replay_min_episode_score - 50.0
            )
            collect_rule = min(0.35, collect_rule + 0.03)
            continue

        if use_bc:
            loss = offline_bc_train(agent, steps=offline_steps, lr=offline_lr)
            print(f"OFFLINE_BC_DONE loss={loss}", flush=True)
        else:
            loss = offline_td_train(agent, steps=offline_steps, lr=offline_lr)
            print(f"OFFLINE_TD_DONE loss={loss}", flush=True)

        config.training.force_save_browser_replay = False
        config.training.skip_browser_replay_load = True
        print(f"EVAL watch_eps={eval_episodes}", flush=True)
        eval_stats = await run_eval_watch(config, max_episodes=eval_episodes)
        mean = float(eval_stats.get("recent_avg", 0.0))
        emin = float(eval_stats.get("recent_min", 0.0))
        emax = float(eval_stats.get("recent_max", 0.0))
        print(
            f"EVAL_RESULT mean={mean:.1f} min={emin:.1f} max={emax:.1f} "
            f"seed_best={best_mean:.1f}",
            flush=True,
        )

        improved = mean > best_mean and emin >= target_min * 0.40
        if improved:
            best_mean = mean
            agent = DQNAgent(config)
            assert agent.load(LATEST)
            agent.save(SEED_BEST)
            agent.save(LATEST)
            write_seed_floor(best_mean)
            print(
                f"SEED_BEST_UPDATE mean={mean:.1f} min={emin:.1f} -> {SEED_BEST.name}",
                flush=True,
            )
            # Hold gentle BC; allow one small step-up only on real gains.
            offline_steps = min(900, offline_steps + 50)
            collect_eps = max(0.03, collect_eps * 0.97)
            collect_rule = max(0.08, collect_rule * 0.97)
        else:
            print(
                f"SEED_REGRESS mean={mean:.1f} min={emin:.1f} — keep {SEED_BEST.name}",
                flush=True,
            )
            agent = _load_seed_agent(config)
            fresh_collect = True
            if Path("checkpoints/browser_replay.pkl").exists():
                Path("checkpoints/browser_replay.pkl").unlink()
                print("CLEARED_REPLAY after regress", flush=True)
            # Keep BC intensity capped; explore a bit more for better gated demos.
            offline_steps = 500
            offline_lr = 3e-6
            if mean < best_mean * 0.90:
                collect_rule = min(0.25, collect_rule + 0.02)
                collect_eps = min(0.07, collect_eps + 0.01)
                config.training.replay_min_episode_score = max(
                    950.0, config.training.replay_min_episode_score - 25.0
                )
            else:
                collect_eps = min(0.08, collect_eps + 0.005)
                # Prefer rarer elite demos when close.
                config.training.replay_min_episode_score = min(
                    1200.0, config.training.replay_min_episode_score + 25.0
                )

        if mean >= target_mean and emin >= target_min:
            print(
                f"SEED_TARGET_MET mean={mean:.1f} min={emin:.1f}",
                flush=True,
            )
            return 0

    print(f"SEED_CLIMB_END best_mean={best_mean:.1f}", flush=True)
    return 0 if best_mean >= target_mean else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed climb supervisor (collect→offline→eval)")
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--run-seed", type=int, default=424242)
    parser.add_argument("--target-mean", type=float, default=2000.0)
    parser.add_argument("--target-min", type=float, default=900.0)
    parser.add_argument("--collect-minutes", type=float, default=12.0)
    parser.add_argument("--eval-episodes", type=int, default=8)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

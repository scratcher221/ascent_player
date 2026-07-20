#!/usr/bin/env python3
"""Reliable sim → browser transfer.

Pipeline:
  1) Visual bridge — short sim fine-tune with rendered frames (not stick-figure obs)
  2) Seed mixed sim_replay with rendered rollouts
  3) Browser fine-tune with low LR / moderate ε / delayed demos
  4) Promote browser_best + playable only from ε=0 Watch evals

Usage:
  PYTHONPATH=. python -u scripts/run_browser_transfer.py
  PYTHONPATH=. python -u scripts/run_browser_transfer.py --hours 2 --skip-bridge
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from ascent_player.agent.checkpoint import checkpoint_exists
from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.training import run_eval_watch, run_sim_pretrain, run_training_no_ui
from ascent_player.utils.skill_ledger import append_skill_ledger


def _build_config() -> AppConfig:
    config = AppConfig()
    config.training.device_mode = DeviceMode.GPU
    config.training.sim_mode = False
    config.training.transfer_from_sim = True
    config.training.transfer_learning_rate = 5e-5
    config.training.transfer_epsilon_start = 0.28
    config.training.transfer_epsilon_restart = 0.20
    config.training.transfer_demo_delay_episodes = 30
    config.training.transfer_frame_skip = 2
    config.training.frame_skip = 2
    config.training.browser_epsilon_cap = 0.28
    config.training.browser_epsilon_floor = 0.05
    config.training.mixed_sim_replay_ratio = 0.25
    config.training.learning_rate = 5e-5
    config.demo.use_demos_on_start = True
    config.browser.auto_launch_on_miss = True
    return config


def _run_visual_bridge(config: AppConfig, steps: int) -> None:
    if steps <= 0:
        print("BRIDGE_SKIP steps=0", flush=True)
        return
    print(f"BRIDGE_START rendered_sim_steps={steps}", flush=True)
    bridge = AppConfig()
    bridge.training = config.training
    bridge.observation = config.observation
    bridge.mechanics_reward = config.mechanics_reward
    bridge.mechanics_curriculum = config.mechanics_curriculum
    bridge.training.sim_mode = True
    bridge.training.sim_fast_observations = False  # rendered frames ≈ browser preprocess
    bridge.training.sim_jpeg_augment = True
    bridge.training.sim_jpeg_quality = 0.82
    # Match browser/transfer decision rate (bulk sim stays FS=1 elsewhere).
    bridge.training.frame_skip = max(1, int(config.training.transfer_frame_skip))
    bridge.training.sim_keep_frame_skip = True
    bridge.training.sim_resume_from_best = True
    bridge.training.sim_resume_epsilon = 0.10
    bridge.training.sim_pretrain_envs = 4  # rendering is heavier
    bridge.training.sim_warmstart_teacher = False
    bridge.training.sim_warmstart_demos = False
    bridge.training.sim_eval_every_steps = max(
        10_000, steps // 2
    )  # ε=0 gate before overwriting
    bridge.training.learning_rate = 5e-5

    agent = DQNAgent(bridge)
    # Prefer explicit sim_best_eval / playable from caller over generic resolve.
    src = None
    for candidate in (
        bridge.training.sim_best_eval_checkpoint_path,
        bridge.training.playable_checkpoint_path,
        agent.resolve_best_checkpoint(),
    ):
        if candidate is not None and checkpoint_exists(candidate):
            src = candidate
            break
    if src is None or not agent.load(src):
        print("BRIDGE_FAIL no baseline", flush=True)
        return
    current = int(agent.progress.total_steps)
    target = current + steps
    print(
        f"BRIDGE_FROM {src.name} steps={current} -> {target} "
        f"frame_skip={bridge.training.frame_skip}",
        flush=True,
    )
    run_sim_pretrain(bridge, target)
    # Keep sim_best_eval + playable updated with bridged weights.
    done = DQNAgent(bridge)
    if done.load(bridge.training.sim_checkpoint_path):
        done.save(bridge.training.sim_best_eval_checkpoint_path)
        done.promote_playable_checkpoint(bridge.training.sim_checkpoint_path)
        print(
            f"BRIDGE_DONE best={done.progress.best_score:.0f} "
            f"steps={done.progress.total_steps}",
            flush=True,
        )


async def _transfer_loop(
    config: AppConfig,
    *,
    deadline: float,
    session_seconds: int,
    eval_episodes: int,
    target_mean: float,
    target_min: float,
    start_from_browser_best: bool = False,
    ledger_mode: str = "transfer_watch",
    keep_weights_on_regress: bool = False,
) -> int:
    best_eval_mean = -1.0
    round_id = 0
    lr = float(config.training.transfer_learning_rate)
    eps = float(config.training.transfer_epsilon_start)

    # Never overwrite a stronger existing browser_best on a fresh restart.
    browser_best = config.training.browser_best_checkpoint_path
    if start_from_browser_best or browser_best.exists() or browser_best.with_name(
        f"{browser_best.stem}.weights.h5"
    ).exists():
        import os

        prior = os.environ.get("ASCENT_BROWSER_BEST_MEAN")
        if prior:
            best_eval_mean = float(prior)
        else:
            # Floor from the first successful transfer eval (~770) so weaker
            # restarts cannot clobber browser_best.
            best_eval_mean = 750.0 if start_from_browser_best else best_eval_mean
        print(f"TRANSFER_BEST_FLOOR mean>={best_eval_mean:.1f}", flush=True)

    while time.time() < deadline:
        round_id += 1
        remaining = deadline - time.time()
        if remaining < 180:
            print("TRANSFER_TIME_LOW", flush=True)
            break
        sess = min(session_seconds, max(180, int(remaining - 120)))
        print(
            f"TRANSFER_ROUND id={round_id} session_s={sess} remaining_s={remaining:.0f} "
            f"lr={lr:.2e} eps={eps:.3f}",
            flush=True,
        )

        # Round 1: either fresh sim transfer or continue from browser_best.
        if round_id == 1 and start_from_browser_best:
            config.training.transfer_from_sim = False
            config.training.prefer_best_checkpoint = True
            # Force load browser_best into dqn_latest before the session.
            seed = DQNAgent(config)
            browser_path = config.training.browser_best_checkpoint_path
            if seed.load(browser_path):
                seed.set_learning_rate(lr)
                seed.epsilon = eps
                seed.metrics.epsilon = eps
                seed.progress.epsilon = eps
                seed.save(config.training.checkpoint_path)
                print(f"SEEDED_FROM_BROWSER_BEST {browser_path.name}", flush=True)
        elif round_id == 1:
            config.training.transfer_from_sim = True
            config.training.prefer_best_checkpoint = True
        else:
            # Continue fine-tune from latest, but restore browser_best if we regressed.
            config.training.transfer_from_sim = False
            config.training.prefer_best_checkpoint = False

        config.training.learning_rate = lr
        config.training.transfer_learning_rate = lr
        config.training.transfer_epsilon_start = eps
        config.training.browser_epsilon_cap = max(0.08, eps)

        stats = await run_training_no_ui(
            config,
            max_seconds=sess,
            ingest_demos=(round_id == 1 and not start_from_browser_best),
        )
        if stats.get("error"):
            print(f"TRANSFER_ERROR {stats['error']}", flush=True)
            await asyncio.sleep(5)
            continue

        print(
            f"TRANSFER_TRAIN recent_avg={stats.get('recent_avg', 0):.0f} "
            f"best={stats.get('best_score', 0):.0f} "
            f"min={stats.get('recent_min', 0):.0f}",
            flush=True,
        )

        print(f"TRANSFER_EVAL watch_eps={eval_episodes}", flush=True)
        eval_stats = await run_eval_watch(config, max_episodes=eval_episodes)
        mean = float(eval_stats.get("recent_avg", 0.0))
        emin = float(eval_stats.get("recent_min", 0.0))
        emax = float(eval_stats.get("recent_max", 0.0))
        print(
            f"TRANSFER_EVAL_RESULT mean={mean:.1f} min={emin:.1f} max={emax:.1f}",
            flush=True,
        )
        append_skill_ledger(
            config.training.skill_ledger_path,
            mode=ledger_mode,
            mean=mean,
            min_score=emin,
            max_score=emax,
            steps=int(stats.get("total_steps", 0) or 0),
            checkpoint=str(config.training.checkpoint_path),
        )

        # Promote only when mean improves and min clears the floor (mean ∧ min).
        promote_min_floor = float(config.training.sim_eval_promote_min) * 0.6
        if (
            not keep_weights_on_regress
            and mean > best_eval_mean
            and emin >= promote_min_floor
        ):
            best_eval_mean = mean
            agent = DQNAgent(config)
            if agent.load(config.training.checkpoint_path):
                path = agent.promote_browser_best()
                print(
                    f"PERSISTED_BROWSER_BEST mean={mean:.0f} min={emin:.0f} -> {path.name}",
                    flush=True,
                )
        elif keep_weights_on_regress:
            # Fixed-seed / curriculum: keep dqn_latest so map learning compounds.
            # Do NOT write browser_best (protects the random-map elite).
            if mean > best_eval_mean:
                best_eval_mean = mean
                # Persist working weights only.
                agent = DQNAgent(config)
                if agent.load(config.training.checkpoint_path):
                    agent.save(config.training.checkpoint_path)
                print(
                    f"TRANSFER_SEED_BEST mean={mean:.1f} min={emin:.1f} "
                    f"(kept weights, no browser_best write)",
                    flush=True,
                )
                # Reward progress: hold LR/ε so learning stays aggressive.
                print(f"TRANSFER_HOLD lr={lr:.2e} eps={eps:.3f}", flush=True)
            else:
                print(
                    f"TRANSFER_KEEP mean={mean:.1f} min={emin:.1f} "
                    f"seed_best={best_eval_mean:.1f} (no restore)",
                    flush=True,
                )
                # Only cool after a flat/worse eval.
                lr = max(2e-5, lr * 0.95)
                eps = max(0.10, eps * 0.97)
                print(f"TRANSFER_COOL lr={lr:.2e} eps={eps:.3f}", flush=True)
        else:
            # Regression: roll back to browser_best and cool exploration/LR.
            print(
                f"TRANSFER_REGRESS mean={mean:.1f} min={emin:.1f} "
                f"< best={best_eval_mean:.1f} — restoring browser_best",
                flush=True,
            )
            restore = DQNAgent(config)
            if restore.load(config.training.browser_best_checkpoint_path):
                restore.save(config.training.checkpoint_path)
                restore.promote_playable_checkpoint(
                    config.training.browser_best_checkpoint_path
                )
            lr = max(1e-5, lr * 0.7)
            # Softer cool: keep exploration usable longer.
            eps = max(0.08, eps * 0.85)
            print(f"TRANSFER_COOL lr={lr:.2e} eps={eps:.3f}", flush=True)

        if mean >= target_mean and emin >= target_min:
            print(
                f"TRANSFER_MET mean={mean:.1f} min={emin:.1f} "
                f"elapsed_target_ok=True",
                flush=True,
            )
            return 0

        config.training.transfer_from_sim = False

    print(f"TRANSFER_END best_eval_mean={best_eval_mean:.1f}", flush=True)
    return 0 if best_eval_mean >= target_mean else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--session-minutes", type=float, default=12.0)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--target-mean", type=float, default=1500.0)
    parser.add_argument("--target-min", type=float, default=500.0)
    parser.add_argument("--skip-bridge", action="store_true")
    parser.add_argument("--bridge-steps", type=int, default=-1)
    parser.add_argument(
        "--from-browser-best",
        action="store_true",
        help="Skip sim reload; continue fine-tuning from browser_best.keras",
    )
    args = parser.parse_args()

    config = _build_config()
    if args.from_browser_best:
        args.skip_bridge = True
        # Gentler continuation settings.
        config.training.transfer_learning_rate = 2e-5
        config.training.transfer_epsilon_start = 0.12
        config.training.browser_epsilon_cap = 0.12
        config.training.learning_rate = 2e-5

    bridge_steps = (
        0
        if args.skip_bridge
        else (
            args.bridge_steps
            if args.bridge_steps >= 0
            else config.training.transfer_visual_bridge_steps
        )
    )

    print(
        f"BROWSER_TRANSFER_START hours={args.hours} "
        f"bridge_steps={bridge_steps} "
        f"from_browser_best={args.from_browser_best} "
        f"target_mean>={args.target_mean} target_min>={args.target_min}",
        flush=True,
    )
    started = time.time()
    deadline = started + max(300.0, args.hours * 3600.0)

    if bridge_steps > 0:
        _run_visual_bridge(config, bridge_steps)

    return asyncio.run(
        _transfer_loop(
            config,
            deadline=deadline,
            session_seconds=max(180, int(args.session_minutes * 60)),
            eval_episodes=max(3, args.eval_episodes),
            target_mean=args.target_mean,
            target_min=args.target_min,
            start_from_browser_best=args.from_browser_best,
        )
    )


if __name__ == "__main__":
    sys.exit(main())

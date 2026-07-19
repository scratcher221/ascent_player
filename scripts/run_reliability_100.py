#!/usr/bin/env python3
"""Train until 100 consecutive ε=0 evals have mean≥5k and min≥1k, or 1 hour elapses."""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.evaluation import (
    evaluate_sim_greedy_sync,
    format_skill_metrics,
)
from ascent_player.training import meets_reliability_target, run_sim_pretrain


DEADLINE_SECONDS = 60 * 60
EVAL_PARALLEL = 12
CHUNK_STEPS = 40_000


def _persist(agent: DQNAgent, config: AppConfig, label: str) -> None:
    agent.save(config.training.sim_best_eval_checkpoint_path)
    agent.save(config.training.playable_checkpoint_path)
    agent.save(config.training.checkpoint_path)
    agent.save_sim_checkpoint()
    print(f"PERSISTED[{label}] playable+latest+best_eval+sim_pretrained", flush=True)


def _eval100(agent: DQNAgent, config: AppConfig, *, fail_fast: bool) -> object:
    return evaluate_sim_greedy_sync(
        agent,
        config,
        episodes=config.training.reliability_eval_episodes,
        max_steps=config.training.sim_eval_max_steps,
        parallel_envs=EVAL_PARALLEL,
        fail_below_score=(
            config.training.reliability_min_score if fail_fast else None
        ),
    )


def main() -> int:
    started = time.time()
    deadline = started + DEADLINE_SECONDS

    config = AppConfig()
    config.training.device_mode = DeviceMode.GPU
    config.training.sim_mode = True
    config.training.sim_pretrain_envs = 12
    config.training.sim_resume_from_best = True
    config.training.sim_warmstart_teacher = False
    config.training.sim_warmstart_demos = False
    config.training.sim_eval_every_steps = 20_000
    config.training.sim_eval_episodes = 8
    config.training.sim_eval_max_steps = 12_000
    config.training.reliability_eval_episodes = 100
    config.training.reliability_mean_score = 5000.0
    config.training.reliability_min_score = 1000.0
    # Mid-train gated probes stay short; full 100-ep checks happen in this script.
    config.training.consistency_eval_episodes = 12
    config.training.consistency_mean_score = 5000.0
    config.training.consistency_min_score = 1000.0
    config.training.sim_min_best_score = 5000
    # Disable mid-chunk gated/consistency evals — this script owns the 100-ep checks.
    config.training.sim_eval_every_steps = 0
    config.training.sim_eval_episodes = 6
    config.training.sim_epsilon_end = 0.03
    config.training.sim_resume_epsilon = 0.08
    config.training.sim_epsilon_anneal_steps = 50_000
    config.training.learning_rate = 5e-5
    config.training.rule_prior_start = 0.10
    config.training.rule_prior_end = 0.03
    config.training.rule_prior_steps = 50_000
    config.mechanics_curriculum.start_height_prob = 0.0
    config.mechanics_curriculum.teacher_episodes = 0

    print(
        f"RELIABILITY_100 deadline_s={DEADLINE_SECONDS} "
        f"mean>={config.training.reliability_mean_score} "
        f"min>={config.training.reliability_min_score}",
        flush=True,
    )

    agent = DQNAgent(config)
    best = agent.resolve_best_checkpoint()
    if best is None or not agent.load(best):
        print("FAILED_BASELINE_LOAD", flush=True)
        return 1
    agent.promote_playable_checkpoint(best)
    print(
        f"BASELINE {best.name} best={agent.progress.best_score:.0f} "
        f"steps={agent.progress.total_steps}",
        flush=True,
    )

    round_id = 0
    best_mean = -1.0
    while time.time() < deadline:
        round_id += 1
        remaining = deadline - time.time()
        print(f"\n=== ROUND {round_id} remaining_s={remaining:.0f} ===", flush=True)

        # Full 100-ep probe (fail-fast on <1k to save wall time when already failing).
        metrics = _eval100(agent, config, fail_fast=True)
        print(format_skill_metrics(f"probe100_r{round_id}", metrics), flush=True)
        complete = metrics.episodes >= config.training.reliability_eval_episodes
        met = complete and meets_reliability_target(
            metrics.mean_score, metrics.min_score, config
        )
        print(
            f"probe_complete={complete} reliability_met={met} "
            f"mean={metrics.mean_score:.1f} min={metrics.min_score:.1f}",
            flush=True,
        )
        if metrics.mean_score > best_mean and complete:
            best_mean = metrics.mean_score
            _persist(agent, config, f"best_mean_{best_mean:.0f}")
        if met:
            # Re-run without fail-fast to confirm a clean consecutive window.
            confirm = _eval100(agent, config, fail_fast=False)
            print(format_skill_metrics("CONFIRM_100", confirm), flush=True)
            if (
                confirm.episodes >= config.training.reliability_eval_episodes
                and meets_reliability_target(
                    confirm.mean_score, confirm.min_score, config
                )
            ):
                _persist(agent, config, "reliability_met")
                print(
                    f"RELIABILITY_MET elapsed_s={time.time() - started:.0f}",
                    flush=True,
                )
                return 0
            print("CONFIRM_FAILED — continuing", flush=True)

        remaining = deadline - time.time()
        if remaining < 180:
            print("TIME_BUDGET_LOW — skipping further train chunk", flush=True)
            break

        # Train additional steps beyond the resumed counter (not an absolute target).
        current_steps = int(agent.progress.total_steps)
        chunk = min(CHUNK_STEPS, int(remaining * 150))
        chunk = max(8_000, chunk)
        target_steps = current_steps + chunk
        print(
            f"TRAIN_CHUNK +{chunk} steps (from {current_steps} -> {target_steps})",
            flush=True,
        )
        run_sim_pretrain(config, target_steps)

        agent = DQNAgent(config)
        # Prefer the checkpoint just written by pretrain over a stale playable slot.
        reload_path = config.training.sim_checkpoint_path
        if not agent.load(reload_path):
            reload_path = agent.resolve_best_checkpoint()
            if reload_path is None or not agent.load(reload_path):
                print("RELOAD_FAILED", flush=True)
                return 1
        agent.promote_playable_checkpoint(reload_path)
        print(
            f"RELOADED {reload_path.name} best={agent.progress.best_score:.0f} "
            f"steps={agent.progress.total_steps}",
            flush=True,
        )

    # Final full window if time remains.
    remaining = deadline - time.time()
    if remaining > 60:
        final = _eval100(agent, config, fail_fast=False)
        print(format_skill_metrics("FINAL_100", final), flush=True)
        met = (
            final.episodes >= config.training.reliability_eval_episodes
            and meets_reliability_target(final.mean_score, final.min_score, config)
        )
        print(f"final_reliability_met={met}", flush=True)
        if met or final.mean_score >= best_mean:
            _persist(agent, config, "final")
        return 0 if met else 2

    print(f"DEADLINE_HIT elapsed_s={time.time() - started:.0f}", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())

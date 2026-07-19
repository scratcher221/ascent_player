#!/usr/bin/env python3
"""4-hour reliability training worker for supervised loops.

Target (ε=0 consecutive evals):
  - 100 episodes
  - mean score >= 5000
  - every episode >= 1000 (min >= 1000)

Designed to be spawned by a supervisor agent. Emits parseable markers:
  SUPERVISED_START / ROUND / PROBE / TRAIN / PERSISTED / SUPERVISED_MET / SUPERVISED_END

Usage:
  PYTHONPATH=. python -u scripts/run_supervised_4h.py
  PYTHONPATH=. python -u scripts/run_supervised_4h.py --hours 4 --baseline auto
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from ascent_player.agent.checkpoint import checkpoint_quality, prefer_checkpoint
from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.evaluation import (
    evaluate_sim_greedy_sync,
    format_skill_metrics,
)
from ascent_player.training import meets_reliability_target, run_sim_pretrain


EVAL_PARALLEL = 12
CHUNK_STEPS = 50_000
PROBE_FAIL_FAST = True


def _persist(agent: DQNAgent, config: AppConfig, label: str) -> None:
    agent.save(config.training.sim_best_eval_checkpoint_path)
    agent.save(config.training.playable_checkpoint_path)
    agent.save(config.training.checkpoint_path)
    agent.save_sim_checkpoint()
    print(f"PERSISTED[{label}] playable+latest+best_eval+sim_pretrained", flush=True)


def _eval100(agent: DQNAgent, config: AppConfig, *, fail_fast: bool):
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


def _pick_baseline(config: AppConfig, preference: str) -> Path:
    training = config.training
    if preference == "best_eval":
        return training.sim_best_eval_checkpoint_path
    if preference == "playable":
        return training.playable_checkpoint_path
    if preference == "sim":
        return training.sim_checkpoint_path
    # auto: highest (best_score, steps), but prefer dedicated best_eval when tied
    # on score within 5% if its greedy history is known-strong.
    return prefer_checkpoint(
        training.playable_checkpoint_path,
        training.sim_checkpoint_path,
        training.sim_best_eval_checkpoint_path,
        training.checkpoint_path,
    ) or training.sim_best_eval_checkpoint_path


def _build_config() -> AppConfig:
    config = AppConfig()
    config.training.device_mode = DeviceMode.GPU
    config.training.sim_mode = True
    config.training.sim_pretrain_envs = 12
    config.training.sim_resume_from_best = True
    config.training.sim_warmstart_teacher = False
    config.training.sim_warmstart_demos = False
    # Supervisor owns full 100-ep probes; disable mid-chunk gated evals.
    config.training.sim_eval_every_steps = 0
    config.training.sim_eval_max_steps = 12_000
    config.training.reliability_eval_episodes = 100
    config.training.reliability_mean_score = 5000.0
    config.training.reliability_min_score = 1000.0
    config.training.consistency_eval_episodes = 12
    config.training.consistency_mean_score = 5000.0
    config.training.consistency_min_score = 1000.0
    config.training.sim_min_best_score = 5000
    config.training.sim_epsilon_end = 0.03
    config.training.sim_resume_epsilon = 0.08
    config.training.sim_epsilon_anneal_steps = 80_000
    config.training.learning_rate = 5e-5
    config.training.rule_prior_start = 0.10
    config.training.rule_prior_end = 0.03
    config.training.rule_prior_steps = 80_000
    config.mechanics_curriculum.start_height_prob = 0.0
    config.mechanics_curriculum.teacher_episodes = 0
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument(
        "--baseline",
        choices=("auto", "best_eval", "playable", "sim"),
        default="auto",
    )
    parser.add_argument("--chunk-steps", type=int, default=CHUNK_STEPS)
    args = parser.parse_args()

    deadline_s = max(60.0, args.hours * 3600.0)
    started = time.time()
    deadline = started + deadline_s
    config = _build_config()

    print(
        f"SUPERVISED_START deadline_s={deadline_s:.0f} "
        f"mean>={config.training.reliability_mean_score} "
        f"min>={config.training.reliability_min_score} "
        f"episodes={config.training.reliability_eval_episodes} "
        f"baseline={args.baseline}",
        flush=True,
    )

    baseline = _pick_baseline(config, args.baseline)
    quality = checkpoint_quality(baseline)
    print(f"BASELINE_CANDIDATE path={baseline} quality={quality}", flush=True)

    agent = DQNAgent(config)
    if not agent.load(baseline):
        print("FAILED_BASELINE_LOAD", flush=True)
        return 1
    agent.promote_playable_checkpoint(baseline)
    print(
        f"BASELINE_LOADED path={baseline.name} "
        f"best={agent.progress.best_score:.0f} steps={agent.progress.total_steps} "
        f"eps={agent.epsilon:.3f}",
        flush=True,
    )

    round_id = 0
    best_complete_mean = -1.0
    best_complete_min = -1.0

    while time.time() < deadline:
        round_id += 1
        remaining = deadline - time.time()
        print(f"ROUND id={round_id} remaining_s={remaining:.0f}", flush=True)

        metrics = _eval100(agent, config, fail_fast=PROBE_FAIL_FAST)
        print(format_skill_metrics(f"PROBE_r{round_id}", metrics), flush=True)
        complete = metrics.episodes >= config.training.reliability_eval_episodes
        met = complete and meets_reliability_target(
            metrics.mean_score, metrics.min_score, config
        )
        print(
            f"PROBE_RESULT round={round_id} complete={complete} met={met} "
            f"mean={metrics.mean_score:.1f} min={metrics.min_score:.1f} "
            f"max={metrics.max_score:.1f} episodes={metrics.episodes}",
            flush=True,
        )

        if complete and metrics.mean_score > best_complete_mean:
            best_complete_mean = metrics.mean_score
            best_complete_min = metrics.min_score
            _persist(agent, config, f"best_probe_mean_{best_complete_mean:.0f}")

        if met:
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
                    f"SUPERVISED_MET mean={confirm.mean_score:.1f} "
                    f"min={confirm.min_score:.1f} "
                    f"elapsed_s={time.time() - started:.0f}",
                    flush=True,
                )
                print("SUPERVISED_END status=met", flush=True)
                return 0
            print("CONFIRM_FAILED continuing", flush=True)

        remaining = deadline - time.time()
        if remaining < 240:
            print("TIME_BUDGET_LOW skip_train", flush=True)
            break

        current_steps = int(agent.progress.total_steps)
        chunk = min(args.chunk_steps, max(10_000, int(remaining * 120)))
        target_steps = current_steps + chunk
        print(
            f"TRAIN start_steps={current_steps} target_steps={target_steps} "
            f"chunk={chunk}",
            flush=True,
        )
        run_sim_pretrain(config, target_steps)

        agent = DQNAgent(config)
        reload_path = config.training.sim_checkpoint_path
        if not agent.load(reload_path):
            print("RELOAD_FAILED", flush=True)
            return 1
        agent.promote_playable_checkpoint(reload_path)
        print(
            f"TRAIN_DONE path={reload_path.name} "
            f"best={agent.progress.best_score:.0f} "
            f"steps={agent.progress.total_steps}",
            flush=True,
        )

    remaining = deadline - time.time()
    if remaining > 90:
        final = _eval100(agent, config, fail_fast=False)
        print(format_skill_metrics("FINAL_100", final), flush=True)
        met = (
            final.episodes >= config.training.reliability_eval_episodes
            and meets_reliability_target(final.mean_score, final.min_score, config)
        )
        print(
            f"FINAL_RESULT met={met} mean={final.mean_score:.1f} "
            f"min={final.min_score:.1f}",
            flush=True,
        )
        if met or (
            final.episodes >= config.training.reliability_eval_episodes
            and final.mean_score >= best_complete_mean
        ):
            _persist(agent, config, "final")
        print(
            f"SUPERVISED_END status={'met' if met else 'deadline'} "
            f"best_complete_mean={best_complete_mean:.1f} "
            f"best_complete_min={best_complete_min:.1f}",
            flush=True,
        )
        return 0 if met else 2

    print(
        f"SUPERVISED_END status=deadline "
        f"best_complete_mean={best_complete_mean:.1f} "
        f"best_complete_min={best_complete_min:.1f}",
        flush=True,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())

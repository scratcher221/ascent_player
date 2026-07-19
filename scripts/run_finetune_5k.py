#!/usr/bin/env python3
"""Fine-tune from the best known checkpoint toward mean≥5k over 10 ε=0 evals."""
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
    score_gates,
)
from ascent_player.training import meets_consistency_target, run_sim_pretrain


DEADLINE_SECONDS = 2 * 60 * 60
TARGET_MEAN = 5000.0
# User goal is mean≥5k over 10 runs; do not require a high floor for success.
TARGET_MIN = 0.0
EVAL_EPISODES = 10


def _persist_best(agent: DQNAgent, config: AppConfig, label: str) -> None:
    agent.save(config.training.sim_best_eval_checkpoint_path)
    agent.save(config.training.playable_checkpoint_path)
    agent.save(config.training.checkpoint_path)
    agent.save_sim_checkpoint()
    print(
        f"PERSISTED[{label}] -> "
        f"{config.training.playable_checkpoint_path.name}, "
        f"{config.training.checkpoint_path.name}, "
        f"{config.training.sim_best_eval_checkpoint_path.name}",
        flush=True,
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
    config.training.sim_eval_every_steps = 15_000
    config.training.sim_eval_episodes = 8
    config.training.consistency_eval_episodes = EVAL_EPISODES
    config.training.consistency_mean_score = TARGET_MEAN
    config.training.consistency_min_score = TARGET_MIN
    config.training.sim_min_best_score = int(TARGET_MEAN)
    config.training.sim_eval_max_steps = 12_000
    config.training.sim_epsilon_end = 0.03
    config.training.sim_epsilon_anneal_steps = 80_000
    config.training.sim_resume_epsilon = 0.15
    config.training.learning_rate = 8e-5
    config.training.rule_prior_start = 0.18
    config.training.rule_prior_end = 0.05
    config.training.rule_prior_steps = 80_000
    config.mechanics_curriculum.start_height_prob = 0.08
    config.mechanics_curriculum.teacher_episodes = 0

    print(
        f"FINETUNE_5K deadline_s={DEADLINE_SECONDS} "
        f"target_mean>={TARGET_MEAN} target_min>={TARGET_MIN} "
        f"eval_eps={EVAL_EPISODES}",
        flush=True,
    )

    probe = DQNAgent(config)
    best = probe.resolve_best_checkpoint()
    if best is None or not probe.load(best):
        print("FAILED_TO_LOAD_BASELINE", flush=True)
        return 1
    probe.promote_playable_checkpoint(best)
    print(
        f"BASELINE path={best.name} best={probe.progress.best_score:.0f} "
        f"steps={probe.progress.total_steps}",
        flush=True,
    )

    preflight = evaluate_sim_greedy_sync(
        probe, config, episodes=EVAL_EPISODES, max_steps=12_000
    )
    print(format_skill_metrics("preflight_10ep", preflight), flush=True)
    print(
        f"preflight_gates="
        f"{','.join(score_gates(preflight.mean_score, preflight.min_score, config)) or 'none'}",
        flush=True,
    )
    if meets_consistency_target(preflight.mean_score, preflight.min_score, config):
        _persist_best(probe, config, "preflight")
        print("TARGET_ALREADY_MET", flush=True)
        return 0

    remaining = max(60.0, deadline - time.time())
    # ~200 sps → budget steps from remaining wall time (leave 12 min for final eval).
    train_budget_s = max(120.0, remaining - 720.0)
    steps = int(min(600_000, max(80_000, train_budget_s * 180)))
    print(f"TRAIN_BUDGET_S={train_budget_s:.0f} steps={steps}", flush=True)

    run_sim_pretrain(config, steps)

    final_agent = DQNAgent(config)
    final_path = final_agent.resolve_best_checkpoint()
    if final_path is None or not final_agent.load(final_path):
        print("FAILED_FINAL_LOAD", flush=True)
        return 1

    final = evaluate_sim_greedy_sync(
        final_agent,
        config,
        episodes=EVAL_EPISODES,
        max_steps=config.training.sim_eval_max_steps,
    )
    print(format_skill_metrics("FINAL_10EP", final), flush=True)
    met = meets_consistency_target(final.mean_score, final.min_score, config)
    print(f"target_met={met} elapsed_s={time.time() - started:.0f}", flush=True)
    if met or final.mean_score >= preflight.mean_score:
        _persist_best(final_agent, config, "final")
    return 0 if met else 2


if __name__ == "__main__":
    sys.exit(main())

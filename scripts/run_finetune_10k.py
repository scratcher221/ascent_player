#!/usr/bin/env python3
"""Fine-tune from best-eval checkpoint toward consistent 10k."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from ascent_player.agent import dqn as dqn_mod
from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.evaluation import (
    evaluate_sim_greedy_sync,
    format_skill_metrics,
    score_gates,
)
from ascent_player import training as tr
from ascent_player.training import meets_consistency_target, run_sim_pretrain


def main() -> int:
    config = AppConfig()
    config.training.device_mode = DeviceMode.GPU
    config.training.sim_mode = True
    config.training.sim_pretrain_envs = 12
    config.training.sim_warmstart_teacher = False
    config.training.sim_warmstart_demos = False
    config.training.sim_eval_every_steps = 12_000
    config.training.sim_eval_episodes = 8
    config.training.consistency_eval_episodes = 12
    config.training.sim_min_best_score = 10_000
    config.training.sim_eval_max_steps = 12_000
    config.training.sim_epsilon_end = 0.05
    config.training.sim_epsilon_anneal_steps = 60_000
    config.training.learning_rate = 1e-4
    config.training.rule_prior_start = 0.20
    config.training.rule_prior_end = 0.05
    config.training.rule_prior_steps = 60_000
    config.mechanics_curriculum.start_height_prob = 0.10
    config.mechanics_curriculum.teacher_episodes = 0

    best_path = config.training.sim_best_eval_checkpoint_path
    if not best_path.exists() and not DQNAgent.weights_sidecar_path(best_path).exists():
        best_path = config.training.sim_checkpoint_path

    print(f"FINETUNE_FROM={best_path}", flush=True)

    probe = DQNAgent(config)
    ok = probe.load(best_path)
    print(
        f"loaded={ok} best={probe.progress.best_score} steps={probe.progress.total_steps}",
        flush=True,
    )
    if not ok:
        print("FAILED_TO_LOAD_BEST", flush=True)
        return 1

    metrics = evaluate_sim_greedy_sync(probe, config, episodes=12, max_steps=12_000)
    print(format_skill_metrics("preflight", metrics), flush=True)
    print(f"preflight_gates={','.join(score_gates(metrics.mean_score, metrics.min_score, config)) or 'none'}", flush=True)
    if meets_consistency_target(metrics.mean_score, metrics.min_score, config):
        print("CONSISTENCY_ALREADY_MET", flush=True)
        probe.save(config.training.sim_checkpoint_path)
        return 0

    probe._epsilon_anneal_start = 0.18
    probe.epsilon = 0.18
    probe.set_learning_rate(config.training.learning_rate)
    probe.save_sim_checkpoint()
    anneal_start = 0.18
    seed_path = config.training.sim_checkpoint_path
    lr = config.training.learning_rate
    RealAgent = dqn_mod.DQNAgent

    class SeededAgent(RealAgent):
        def apply_sim_pretrain_profile(self) -> None:
            super().apply_sim_pretrain_profile()
            if self.load(seed_path):
                self._epsilon_anneal_start = anneal_start
                self.epsilon = anneal_start
                self.metrics.epsilon = anneal_start
                self.progress.epsilon = anneal_start
                self.set_learning_rate(lr)
                print(
                    f"seeded steps={self.progress.total_steps} "
                    f"best={self.progress.best_score:.0f} eps={self.epsilon:.3f}",
                    flush=True,
                )

    dqn_mod.DQNAgent = SeededAgent
    tr.DQNAgent = SeededAgent
    try:
        run_sim_pretrain(config, 200_000)
    finally:
        dqn_mod.DQNAgent = RealAgent
        tr.DQNAgent = RealAgent

    final_agent = RealAgent(config)
    if not final_agent.load(config.training.sim_best_eval_checkpoint_path):
        final_agent.load(config.training.sim_checkpoint_path)
    final = evaluate_sim_greedy_sync(
        final_agent,
        config,
        episodes=config.training.consistency_eval_episodes,
        max_steps=config.training.sim_eval_max_steps,
    )
    print(format_skill_metrics("FINAL_CONSISTENCY", final), flush=True)
    met = meets_consistency_target(final.mean_score, final.min_score, config)
    print(f"consistency_met={met}", flush=True)
    return 0 if met else 2


if __name__ == "__main__":
    sys.exit(main())

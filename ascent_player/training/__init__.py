"""Training entrypoints (sim pretrain, browser collect, eval watch)."""
from __future__ import annotations

from ascent_player.training.browser_runner import (
    create_env,
    run_eval_watch,
    run_training_no_ui,
)
from ascent_player.training.curriculum import (
    apply_curriculum,
    curriculum_start_height,
    episode_score_for_progress,
    meets_consistency_target,
    meets_reliability_target,
    warmstart_from_teacher,
    warmstart_from_teacher_sync,
)
from ascent_player.training.sim_runner import run_sim_calibration, run_sim_pretrain

__all__ = [
    "apply_curriculum",
    "create_env",
    "curriculum_start_height",
    "episode_score_for_progress",
    "meets_consistency_target",
    "meets_reliability_target",
    "run_eval_watch",
    "run_sim_calibration",
    "run_sim_pretrain",
    "run_training_no_ui",
    "warmstart_from_teacher",
    "warmstart_from_teacher_sync",
]

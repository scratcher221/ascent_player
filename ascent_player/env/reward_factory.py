from __future__ import annotations

from ascent_player.config import AppConfig
from ascent_player.env.mechanics_rewards import MechanicsRewardTracker


def create_reward_tracker(config: AppConfig) -> MechanicsRewardTracker:
    return MechanicsRewardTracker(config.mechanics_reward)

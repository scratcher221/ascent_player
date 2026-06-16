from __future__ import annotations

from ascent_player.config import AppConfig
from ascent_player.env.mechanics_rewards import MechanicsRewardTracker
from ascent_player.env.rewards import RewardTracker


def create_reward_tracker(config: AppConfig):
    if config.mechanics_curriculum.use_mechanics_rewards:
        return MechanicsRewardTracker(config.mechanics_reward)
    return RewardTracker(config.reward)

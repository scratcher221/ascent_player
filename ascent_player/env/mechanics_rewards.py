"""Reward shaping aligned with ASCENT game mechanics (score-rules.js)."""

from __future__ import annotations

from dataclasses import dataclass

from ascent_player.config import MechanicsRewardConfig
from ascent_player.env.score_rules import combo_multiplier, live_style
from ascent_player.env.state_detector import JUMP_ACTIONS, FrameState

LEFT_ACTIONS = frozenset({1, 4})
RIGHT_ACTIONS = frozenset({2, 5})


@dataclass(slots=True)
class MechanicsRewardTracker:
    config: MechanicsRewardConfig
    curriculum_stage: str = "M0"
    last_state: FrameState | None = None
    episode_steps: int = 0
    milestones_hit: set[int] | None = None
    last_steer_dir: int = 0
    persist_steer_dir: int = 0
    persist_steer_steps: int = 0

    def set_curriculum_stage(self, stage: str) -> None:
        valid = {f"M{i}" for i in range(7)}
        self.curriculum_stage = stage if stage in valid else "M0"

    def reset(self) -> None:
        self.last_state = None
        self.episode_steps = 0
        self.milestones_hit = set()
        self.last_steer_dir = 0
        self.persist_steer_dir = 0
        self.persist_steer_steps = 0

    def compute(self, state: FrameState, action: int) -> float:
        self.episode_steps += 1
        reward = self.config.survival
        previous = self.last_state

        if previous is not None:
            reward += self._height_reward(previous, state)
            if self._stage_at_least("M1"):
                reward += self._landing_reward(previous, state)
                reward += self._falling_penalty(previous, state)
            if self._stage_at_least("M2"):
                reward += self._steer_reward(previous, state, action)
            if self._stage_at_least("M3"):
                reward += self._boost_economy_reward(previous, state, action)
            if self._stage_at_least("M4"):
                reward += self._combo_reward(previous, state)
            if self._stage_at_least("M5"):
                reward += self._booster_reward(previous, state)
            if self._stage_at_least("M6"):
                reward += self._score_reward(previous, state)
                reward += self._milestone_reward(state)

        if state.game_over:
            reward += self.config.death
            if self.episode_steps < self.config.early_death_steps:
                reward += self.config.early_death_penalty

        self.last_state = state
        return float(
            max(
                -self.config.reward_clip,
                min(self.config.reward_clip, reward),
            )
        )

    def _stage_at_least(self, stage: str) -> bool:
        return int(self.curriculum_stage[1:]) >= int(stage[1:])

    def _height_reward(self, previous: FrameState, state: FrameState) -> float:
        delta = state.height - previous.height
        if delta <= 0:
            return 0.0
        return delta * self.config.height_gain / 5.0

    def _landing_reward(self, previous: FrameState, state: FrameState) -> float:
        if state.bounces <= previous.bounces:
            return 0.0
        return self.config.platform_land

    def _falling_penalty(self, previous: FrameState, state: FrameState) -> float:
        if state.orb_vy is None or previous.orb_vy is None:
            return 0.0
        if state.orb_vy < -0.35 and state.orb_y is not None and state.orb_y > 0.72:
            return self.config.falling_penalty
        return 0.0

    def _steer_reward(self, previous: FrameState, state: FrameState, action: int) -> float:
        dx = state.nearest_platform_dx
        if dx is None:
            return 0.0
        reward = 0.0
        gain = self.config.steer_gain
        if dx < -0.012:
            if action in LEFT_ACTIONS:
                reward += gain * min(abs(dx) * 6.0, 2.0)
            elif action in RIGHT_ACTIONS:
                reward += self.config.wrong_way_penalty
        elif dx > 0.012:
            if action in RIGHT_ACTIONS:
                reward += gain * min(abs(dx) * 6.0, 2.0)
            elif action in LEFT_ACTIONS:
                reward += self.config.wrong_way_penalty
        else:
            reward += self.config.aligned_bonus
        return reward

    def _boost_economy_reward(
        self,
        previous: FrameState,
        state: FrameState,
        action: int,
    ) -> float:
        reward = 0.0
        jumped = action in JUMP_ACTIONS
        if jumped and previous.can_boost and not state.can_boost:
            reward += self.config.boost_spent
            if (previous.orb_vy or 0) > 0.15:
                reward += self.config.wasted_boost_penalty
        if jumped and not previous.can_boost:
            reward += self.config.empty_boost_penalty
        vy = state.orb_vy or 0.0
        gap = state.nearest_platform_dy or 0.0
        if jumped and vy < -0.2 and gap > 0.15:
            reward += self.config.timed_boost_bonus
        return reward

    def _combo_reward(self, previous: FrameState, state: FrameState) -> float:
        reward = 0.0
        if state.combo > previous.combo:
            scale = combo_multiplier(state.combo)
            reward += self.config.combo_gain * scale
        elif state.combo < previous.combo and previous.combo >= 3:
            lost_style = live_style(bonus=previous.bonus, combo=previous.combo)
            reward += self.config.combo_break_penalty * min(lost_style / 400.0, 1.0)
        return reward

    def _booster_reward(self, previous: FrameState, state: FrameState) -> float:
        if state.bonus <= previous.bonus and state.booster_type is None:
            return 0.0
        if state.booster_type == "drag" or (
            previous.booster_type != "drag" and state.combo < previous.combo
        ):
            return self.config.drag_penalty
        if state.bonus > previous.bonus:
            return self.config.booster_collect
        return 0.0

    def _score_reward(self, previous: FrameState, state: FrameState) -> float:
        if state.score is None or previous.score is None:
            return 0.0
        delta = state.score - previous.score
        if delta <= 0:
            return 0.0
        return delta * self.config.score_gain

    def _milestone_reward(self, state: FrameState) -> float:
        if state.score is None or self.milestones_hit is None:
            return 0.0
        reward = 0.0
        for milestone in self.config.milestone_scores:
            if state.score >= milestone and milestone not in self.milestones_hit:
                self.milestones_hit.add(milestone)
                reward += self.config.milestone_bonus
        return reward

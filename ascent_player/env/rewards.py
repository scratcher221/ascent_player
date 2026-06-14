from __future__ import annotations

from dataclasses import dataclass

from ascent_player.config import RewardConfig
from ascent_player.env.state_detector import JUMP_ACTIONS, FrameState


@dataclass(slots=True)
class RewardTracker:
    config: RewardConfig
    last_state: FrameState | None = None
    idle_steps: int = 0
    episode_steps: int = 0
    stagnant_steps: int = 0
    last_score: int | None = None
    milestones_hit: set[int] | None = None
    recent_boost_spend: int = 0

    def reset(self) -> None:
        self.last_state = None
        self.idle_steps = 0
        self.episode_steps = 0
        self.stagnant_steps = 0
        self.last_score = None
        self.milestones_hit = set()
        self.recent_boost_spend = 0

    def compute(self, state: FrameState, action: int) -> float:
        self.episode_steps += 1
        reward = self.config.survival
        reward += min(
            self.episode_steps * self.config.survival_step_bonus,
            self.config.survival * 2.0,
        )
        previous = self.last_state

        if previous is not None:
            reward += self._score_reward(previous, state)
            reward += self._altitude_reward(previous, state)
            reward += self._falling_penalty(previous, state)
            reward += self._idle_penalty(previous, state, action)
            reward += self._boost_reward(previous, state, action)
            reward += self._platform_reward(previous, state, action)
            reward += self._combo_streak_reward(previous, state)
            reward += self._stagnation_penalty(previous, state)
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

    def _score_reward(self, previous: FrameState, state: FrameState) -> float:
        if previous.score is None or state.score is None:
            return 0.0
        delta = max(0, state.score - previous.score)
        if delta > 0:
            self.stagnant_steps = 0
            self.last_score = state.score
        scale = max(1.0, state.score_multiplier)
        return delta * self.config.score_gain * scale

    def _altitude_reward(self, previous: FrameState, state: FrameState) -> float:
        if previous.orb_y is None or state.orb_y is None:
            return 0.0
        altitude_gain = previous.orb_y - state.orb_y
        if altitude_gain <= 0:
            return 0.0
        score_confirmed = (
            previous.score is not None
            and state.score is not None
            and state.score > previous.score
        )
        scale = 1.0 if score_confirmed else 0.2
        return max(0.0, altitude_gain / 100.0) * self.config.altitude_gain * scale

    def _falling_penalty(self, previous: FrameState, state: FrameState) -> float:
        if previous.orb_y is None or state.orb_y is None:
            return 0.0
        fall = state.orb_y - previous.orb_y
        if fall <= 6:
            return 0.0
        weight = 1.0 + state.streak * 0.15
        return (fall / 100.0) * self.config.falling_penalty * weight

    def _idle_penalty(
        self,
        previous: FrameState,
        state: FrameState,
        action: int,
    ) -> float:
        horizontal_action = action in (1, 2, 4, 5)
        falling = (
            previous.orb_y is not None
            and state.orb_y is not None
            and state.orb_y > previous.orb_y + 4
        )
        if horizontal_action or not falling:
            self.idle_steps = 0
            return 0.0
        self.idle_steps += 1
        if self.idle_steps > self.config.idle_steps:
            return self.config.idle_penalty
        return 0.0

    def _boost_reward(
        self,
        previous: FrameState,
        state: FrameState,
        action: int,
    ) -> float:
        reward = 0.0
        boost_delta = state.boost_level - previous.boost_level
        if boost_delta > 0.03:
            reward += boost_delta * self.config.boost_gain

        if action in JUMP_ACTIONS:
            if not previous.can_boost:
                reward += self.config.empty_boost_jump_penalty
            elif previous.boost_level < self.config.boost_jump_threshold:
                reward += self.config.wasted_jump_penalty
            elif boost_delta < -0.04:
                gained_score = (
                    previous.score is not None
                    and state.score is not None
                    and state.score > previous.score
                )
                gained_altitude = (
                    previous.orb_y is not None
                    and state.orb_y is not None
                    and state.orb_y < previous.orb_y - 8
                )
                landed = state.platform_landed
                if not gained_score and not gained_altitude and not landed:
                    reward += self.config.boost_spent
                self.recent_boost_spend = 8

        if self.recent_boost_spend > 0:
            self.recent_boost_spend -= 1

        if state.boost_level < 0.05:
            reward += self.config.low_boost_penalty

        return reward

    def _combo_streak_reward(self, previous: FrameState, state: FrameState) -> float:
        reward = 0.0
        landed = state.platform_landed or state.combo > previous.combo
        if landed:
            reward += self.config.platform_land
            reward += self.config.combo_gain * max(1, state.combo)

        if state.combo > previous.combo:
            reward += self.config.combo_gain * (state.combo - previous.combo) * 0.5
        elif state.combo < previous.combo:
            reward += self.config.combo_break_penalty

        if state.streak > previous.streak:
            delta = state.streak - previous.streak
            reward += self.config.streak_level_bonus * (state.streak ** 1.4) * delta

        multiplier_delta = state.score_multiplier - previous.score_multiplier
        if multiplier_delta > 0.01:
            reward += multiplier_delta * self.config.multiplier_gain

        return reward

    def _platform_reward(
        self,
        previous: FrameState,
        state: FrameState,
        action: int,
    ) -> float:
        if state.nearest_platform_dx is None:
            return 0.0

        falling = (
            previous.orb_y is not None
            and state.orb_y is not None
            and state.orb_y > previous.orb_y + 3
        )
        platform_below = (
            state.nearest_platform_dy is not None and state.nearest_platform_dy > 0.02
        )
        if not falling and not platform_below:
            return 0.0

        reward = 0.0
        dx = state.nearest_platform_dx
        steer_scale = self.config.platform_fall_weight if falling else 1.0
        steer_scale *= 1.0 + state.streak * 0.2

        if dx < -0.02 and action in (1, 4):
            reward += self.config.platform_align * min(abs(dx) * 4.0, 1.5) * steer_scale
        elif dx > 0.02 and action in (2, 5):
            reward += self.config.platform_align * min(abs(dx) * 4.0, 1.5) * steer_scale
        elif falling:
            wrong_dir = (dx < -0.04 and action in (2, 5)) or (dx > 0.04 and action in (1, 4))
            if wrong_dir:
                reward += self.config.off_platform_penalty * min(abs(dx) * 3.0, 1.0)

        if previous.nearest_platform_dx is not None:
            prev_dist = abs(previous.nearest_platform_dx)
            curr_dist = abs(state.nearest_platform_dx)
            if curr_dist < prev_dist:
                reward += (
                    (prev_dist - curr_dist)
                    * self.config.platform_align
                    * steer_scale
                    * 2.0
                )

        if (
            previous.nearest_platform_dy is not None
            and state.nearest_platform_dy is not None
            and state.nearest_platform_dy > 0
        ):
            dy_gain = previous.nearest_platform_dy - state.nearest_platform_dy
            if dy_gain > 0:
                reward += dy_gain * self.config.platform_align * steer_scale

        return reward

    def _stagnation_penalty(self, previous: FrameState, state: FrameState) -> float:
        if previous.score is None or state.score is None:
            return 0.0
        if state.score > previous.score:
            self.stagnant_steps = 0
            return 0.0
        self.stagnant_steps += 1
        if self.stagnant_steps < self.config.score_stagnation_steps:
            return 0.0
        return self.config.score_stagnation_penalty

    def _milestone_reward(self, state: FrameState) -> float:
        if state.score is None or self.milestones_hit is None:
            return 0.0
        reward = 0.0
        for milestone in self.config.milestone_scores:
            if milestone in self.milestones_hit:
                continue
            if state.score >= milestone:
                self.milestones_hit.add(milestone)
                reward += self.config.milestone_bonus * max(1.0, state.score_multiplier)
        return reward

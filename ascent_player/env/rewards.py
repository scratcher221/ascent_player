from __future__ import annotations

from dataclasses import dataclass

from ascent_player.config import RewardConfig
from ascent_player.env.state_detector import JUMP_ACTIONS, FrameState

LEFT_ACTIONS = frozenset({1, 4})
RIGHT_ACTIONS = frozenset({2, 5})


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
    last_steer_dir: int = 0
    persist_steer_dir: int = 0
    persist_steer_steps: int = 0
    curriculum_stage: str = "A"

    def set_curriculum_stage(self, stage: str) -> None:
        self.curriculum_stage = stage if stage in {"A", "B", "C"} else "A"

    def reset(self) -> None:
        self.last_state = None
        self.idle_steps = 0
        self.episode_steps = 0
        self.stagnant_steps = 0
        self.last_score = None
        self.milestones_hit = set()
        self.recent_boost_spend = 0
        self.last_steer_dir = 0
        self.persist_steer_dir = 0
        self.persist_steer_steps = 0

    def compute(self, state: FrameState, action: int) -> float:
        self.episode_steps += 1
        reward = self.config.survival
        previous = self.last_state

        if previous is not None:
            reward += self._target_steering_reward(previous, state, action)
            if self.curriculum_stage != "A":
                reward += self._combo_streak_reward(previous, state)
            if self.curriculum_stage == "C":
                reward += self._score_reward(previous, state)
                reward += self._milestone_reward(state)
            elif self.curriculum_stage == "B" and state.combo >= 3:
                reward += self._score_reward(previous, state)
            reward += self._altitude_reward(previous, state)
            if self.curriculum_stage != "A":
                reward += self._stagnation_penalty(previous, state)
            reward += self._falling_penalty(previous, state)
            reward += self._boost_reward(previous, state, action)

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

    def _target_dx(self, state: FrameState) -> float | None:
        if state.target_dx is not None:
            return state.target_dx
        return state.nearest_platform_dx

    def _target_dy(self, state: FrameState) -> float | None:
        if state.target_dy is not None:
            return state.target_dy
        return state.nearest_platform_dy

    def _target_steering_reward(
        self,
        previous: FrameState,
        state: FrameState,
        action: int,
    ) -> float:
        dx = self._target_dx(state)
        if dx is None:
            return 0.0

        reward = 0.0
        prev_dx = self._target_dx(previous)
        if prev_dx is not None:
            prev_dist = abs(prev_dx)
            curr_dist = abs(dx)
            if curr_dist < prev_dist:
                reward += (
                    (prev_dist - curr_dist)
                    * self.config.target_approach_gain
                    * 4.0
                )

        steer_gain = self.config.target_steer_gain
        if self.curriculum_stage == "A":
            steer_gain *= 1.15
        elif self.curriculum_stage == "C":
            steer_gain *= 0.92

        if state.target_kind == "yellow_orb" and self.curriculum_stage != "A":
            steer_gain *= 1.15
        elif state.target_kind == "green_booster" and self.curriculum_stage != "A":
            steer_gain *= 1.1

        if dx < -0.012:
            if action in LEFT_ACTIONS:
                reward += steer_gain * min(abs(dx) * 6.0, 2.0)
                reward += self._steer_flip_penalty(-1, abs_dx=abs(dx))
                reward += self._persistence_bonus(-1)
            elif action in RIGHT_ACTIONS:
                reward += self.config.target_wrong_way_penalty * min(abs(dx) * 4.0, 1.5)
                self._set_steer_dir(1)
                self._reset_persistence()
            elif action == 0:
                reward += self.config.target_idle_penalty * min(abs(dx) * 3.0, 1.0)
                self._reset_persistence()
        elif dx > 0.012:
            if action in RIGHT_ACTIONS:
                reward += steer_gain * min(abs(dx) * 6.0, 2.0)
                reward += self._steer_flip_penalty(1, abs_dx=abs(dx))
                reward += self._persistence_bonus(1)
            elif action in LEFT_ACTIONS:
                reward += self.config.target_wrong_way_penalty * min(abs(dx) * 4.0, 1.5)
                self._set_steer_dir(-1)
                self._reset_persistence()
            elif action == 0:
                reward += self.config.target_idle_penalty * min(abs(dx) * 3.0, 1.0)
                self._reset_persistence()
        else:
            if action in (0, 3):
                reward += self.config.target_aligned_bonus
            self._reset_persistence()

        dy = self._target_dy(state)
        if dy is not None and dy > 0.02 and action in JUMP_ACTIONS:
            reward += self.config.target_aligned_bonus * 0.5

        return reward

    def _steer_flip_penalty(self, direction: int, *, abs_dx: float = 0.0) -> float:
        penalty = 0.0
        if (
            self.last_steer_dir != 0
            and direction != 0
            and self.last_steer_dir != direction
        ):
            scale = min(max(abs_dx, 0.05) * 4.0, 1.5)
            penalty = self.config.direction_flip_penalty * scale
        self._set_steer_dir(direction)
        return penalty

    def _persistence_bonus(self, direction: int) -> float:
        if direction == 0:
            return 0.0
        if self.persist_steer_dir == direction:
            self.persist_steer_steps += 1
        else:
            self.persist_steer_dir = direction
            self.persist_steer_steps = 1
        if self.persist_steer_steps >= self.config.direction_persistence_steps:
            return self.config.direction_persistence_bonus
        return 0.0

    def _reset_persistence(self) -> None:
        self.persist_steer_dir = 0
        self.persist_steer_steps = 0

    def _set_steer_dir(self, direction: int) -> None:
        if direction != 0:
            self.last_steer_dir = direction

    def _score_reward(self, previous: FrameState, state: FrameState) -> float:
        if self.curriculum_stage == "A":
            return 0.0
        if previous.score is None or state.score is None:
            return 0.0
        if self.curriculum_stage == "B" and state.combo < 3:
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
        scale = 1.0 if score_confirmed else 0.15
        return max(0.0, altitude_gain / 100.0) * self.config.altitude_gain * scale

    def _falling_penalty(self, previous: FrameState, state: FrameState) -> float:
        if previous.orb_y is None or state.orb_y is None:
            return 0.0
        fall = state.orb_y - previous.orb_y
        if fall <= 6:
            return 0.0
        dx = self._target_dx(state)
        misaligned = dx is not None and abs(dx) > 0.08
        weight = 1.2 if misaligned else 1.0
        return (fall / 100.0) * self.config.falling_penalty * weight

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
            if state.combo >= 3 or self.curriculum_stage == "C":
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

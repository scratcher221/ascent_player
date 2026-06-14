from __future__ import annotations

from dataclasses import dataclass

from ascent_player.config import RewardConfig
from ascent_player.env.state_detector import FrameState


@dataclass(slots=True)
class RewardTracker:
    config: RewardConfig
    last_state: FrameState | None = None
    idle_steps: int = 0

    def reset(self) -> None:
        self.last_state = None
        self.idle_steps = 0

    def compute(self, state: FrameState, action: int) -> float:
        reward = self.config.survival
        previous = self.last_state

        if previous is not None:
            reward += self._score_reward(previous, state)
            reward += self._altitude_reward(previous, state)
            reward += self._idle_penalty(previous, state, action)

        if state.game_over:
            reward += self.config.death

        self.last_state = state
        return float(reward)

    def _score_reward(self, previous: FrameState, state: FrameState) -> float:
        if previous.score is None or state.score is None:
            return 0.0
        return max(0, state.score - previous.score) * self.config.score_gain

    def _altitude_reward(self, previous: FrameState, state: FrameState) -> float:
        if previous.orb_y is None or state.orb_y is None:
            return 0.0
        # In image coordinates, smaller y means higher on screen.
        altitude_gain = previous.orb_y - state.orb_y
        return max(0.0, altitude_gain / 100.0) * self.config.altitude_gain

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
            and state.orb_y > previous.orb_y
        )
        if horizontal_action or not falling:
            self.idle_steps = 0
            return 0.0
        self.idle_steps += 1
        if self.idle_steps > self.config.idle_steps:
            return self.config.idle_penalty
        return 0.0

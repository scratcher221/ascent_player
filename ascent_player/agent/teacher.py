"""Rule-based teacher for behavior-cloning warm-start."""

from __future__ import annotations

from ascent_player.env.state_detector import FrameState, mask_jump_action

LEFT = 1
RIGHT = 2
JUMP = 3
LEFT_JUMP = 4
RIGHT_JUMP = 5


class RuleTeacher:
    def __init__(
        self,
        *,
        steer_threshold: float = 0.012,
        fall_vy_threshold: float = -0.2,
        gap_threshold: float = 0.15,
        min_energy_to_boost: float = 0.3,
    ) -> None:
        self.steer_threshold = steer_threshold
        self.fall_vy_threshold = fall_vy_threshold
        self.gap_threshold = gap_threshold
        self.min_energy_to_boost = min_energy_to_boost

    def act(self, state: FrameState) -> int:
        dx = state.nearest_platform_dx
        dy = state.nearest_platform_dy
        vy = state.orb_vy or 0.0
        action = 0

        if dx is not None:
            if dx < -self.steer_threshold:
                action = LEFT
            elif dx > self.steer_threshold:
                action = RIGHT

        should_boost = (
            state.can_boost
            and state.boost_level >= self.min_energy_to_boost
            and vy < self.fall_vy_threshold
            and (dy is None or dy > self.gap_threshold)
        )

        if state.booster_type == "drag" and state.combo > 3:
            should_boost = False

        if should_boost:
            if action == LEFT:
                action = LEFT_JUMP
            elif action == RIGHT:
                action = RIGHT_JUMP
            else:
                action = JUMP

        return mask_jump_action(action, state.can_boost)


def generate_teacher_episode(env, teacher: RuleTeacher, max_steps: int = 600) -> list[tuple]:
    """Sync helper for sim env — returns (state, action, reward, next_state, done) tuples."""
    import asyncio

    async def _run() -> list[tuple]:
        transitions: list[tuple] = []
        state = await env.reset()
        for _ in range(max_steps):
            fs = env._last_frame_state
            if fs is None:
                break
            action = teacher.act(fs)
            result = await env.step(action)
            transitions.append(
                (state, action, result.reward, result.state, result.done)
            )
            state = result.state
            if result.done:
                break
        return transitions

    return asyncio.get_event_loop().run_until_complete(_run())

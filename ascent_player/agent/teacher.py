"""Rule-based policy for warm-start, exploration prior, and baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from ascent_player.env.state_detector import FrameState, mask_jump_action

LEFT = 1
RIGHT = 2
JUMP = 3
LEFT_JUMP = 4
RIGHT_JUMP = 5


@dataclass(slots=True)
class RulePolicy:
    steer_threshold: float = 0.012
    fall_vy_threshold: float = -0.2
    gap_threshold: float = 0.15
    min_energy_to_boost: float = 0.14
    recovery_dx_threshold: float = 0.18

    def target_dx(self, state: FrameState) -> float | None:
        # Survival: always aim at the pad below when falling.
        if state.falling or state.landing_window or state.miss_risk:
            if state.nearest_platform_dx is not None:
                return state.nearest_platform_dx
        if state.target_dx is not None:
            return state.target_dx
        if state.nearest_platform_above_dx is not None and state.rising:
            return state.nearest_platform_above_dx
        return state.nearest_platform_dx

    def target_dy(self, state: FrameState) -> float | None:
        if state.falling or state.landing_window or state.miss_risk:
            if state.nearest_platform_dy is not None:
                return state.nearest_platform_dy
        if state.target_dy is not None:
            return state.target_dy
        if state.nearest_platform_above_dy is not None and state.rising:
            return state.nearest_platform_above_dy
        return state.nearest_platform_dy

    def act(self, state: FrameState) -> int:
        dx = self.target_dx(state)
        dy = self.target_dy(state)
        vy = state.orb_vy or 0.0
        if abs(vy) > 1.5:
            vy = vy / 1500.0
        action = 0

        if state.danger_sell and dx is not None:
            if dx < -self.steer_threshold:
                action = LEFT
            elif dx > self.steer_threshold:
                action = RIGHT
        elif dx is not None:
            threshold = self.recovery_dx_threshold if state.miss_risk else self.steer_threshold
            if dx < -threshold:
                action = LEFT
            elif dx > threshold:
                action = RIGHT

        should_boost = (
            state.can_boost
            and state.boost_level >= self.min_energy_to_boost
            and (
                state.boost_useful is True
                or (
                    vy < self.fall_vy_threshold
                    and (dy is None or dy > self.gap_threshold)
                )
            )
        )

        if state.booster_type == "drag" and state.combo > 3:
            should_boost = False
        if state.danger_sell and not state.miss_risk:
            should_boost = False

        if should_boost:
            if action == LEFT:
                action = LEFT_JUMP
            elif action == RIGHT:
                action = RIGHT_JUMP
            else:
                action = JUMP

        return mask_jump_action(action, state.can_boost)


RuleTeacher = RulePolicy


def generate_teacher_episode(env, teacher: RulePolicy, max_steps: int = 600) -> list[tuple]:
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

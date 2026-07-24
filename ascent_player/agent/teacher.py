"""Rule-based policy for warm-start, exploration prior, and baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ascent_player.env.state_detector import FrameState, mask_jump_action

LEFT = 1
RIGHT = 2
JUMP = 3
LEFT_JUMP = 4
RIGHT_JUMP = 5


def _norm_orb_x(state: FrameState) -> float | None:
    """Return orb x in ~[0, 1] when available."""
    x = state.orb_x
    if x is None:
        return None
    if x > 1.5:
        # Pixel-space from some hooks — treat as unavailable for thread lock.
        return None
    return float(max(0.0, min(1.0, x)))


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
        # Climb bias: while rising, prefer the pad above over lateral distractors.
        if state.rising and state.nearest_platform_above_dx is not None:
            return state.nearest_platform_above_dx
        if state.target_dx is not None:
            return state.target_dx
        return state.nearest_platform_dx

    def target_dy(self, state: FrameState) -> float | None:
        if state.falling or state.landing_window or state.miss_risk:
            if state.nearest_platform_dy is not None:
                return state.nearest_platform_dy
        if state.rising and state.nearest_platform_above_dy is not None:
            return state.nearest_platform_above_dy
        if state.target_dy is not None:
            return state.target_dy
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

        gap = dy if dy is not None else 0.0
        should_boost = (
            state.can_boost
            and state.boost_level >= self.min_energy_to_boost
            and (
                state.boost_useful is True
                or (
                    vy < self.fall_vy_threshold
                    and (dy is None or dy > self.gap_threshold)
                )
                # Climb assist: spend boost to reach the next pad above.
                or (
                    state.rising
                    and gap > 0.08
                    and state.boost_level >= 0.22
                    and abs(dx or 0.0) < 0.08
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


@dataclass(slots=True)
class SeedThreadPolicy(RulePolicy):
    """Bias toward a stable climb "thread" (preferred lateral corridor) on a fixed seed.

    On successful landings, lock an EMA of orb x. While climbing, prefer the next
    landable pad above when it stays near that corridor; never override survival
    landings or force early fuel dumps into hazards/drag.
    """

    thread_corridor: float = 0.18
    thread_blend: float = 0.45
    thread_ema: float = 0.7
    lock_on_land: bool = True
    # Soften boost when aligned on-thread early (avoids dump farming).
    suppress_aligned_boost_energy: float = 0.55
    thread_x: float | None = None
    landings_locked: int = 0
    _prev_bounces: int = field(default=-1, repr=False)

    @staticmethod
    def _safe_climb_target(state: FrameState) -> tuple[float | None, float | None]:
        """Climb through the vertically next pad; scored targets may be below."""
        if state.nearest_platform_above_dx is not None:
            return state.nearest_platform_above_dx, state.nearest_platform_above_dy
        if state.target_kind == "platform" and state.target_dx is not None:
            return state.target_dx, state.target_dy
        return None, None

    def reset(self) -> None:
        self.thread_x = None
        self.landings_locked = 0
        self._prev_bounces = -1

    def observe(self, state: FrameState) -> None:
        """Update thread lock when a platform landing succeeds."""
        if not self.lock_on_land:
            return
        bounces = int(state.bounces or 0)
        landed = bounces > self._prev_bounces and (
            bool(state.platform_landed) or self._prev_bounces >= 0
        )
        self._prev_bounces = bounces
        if not landed:
            return
        orb_x = _norm_orb_x(state)
        if orb_x is None:
            # Fall back to inferred pad x from relative dx.
            if state.nearest_platform_dx is None:
                return
            # Assume orb near 0.5 if absolute x missing.
            orb_x = 0.5 + float(state.nearest_platform_dx)
            orb_x = max(0.0, min(1.0, orb_x))
        if self.thread_x is None:
            self.thread_x = orb_x
        else:
            a = float(self.thread_ema)
            self.thread_x = a * self.thread_x + (1.0 - a) * orb_x
        self.landings_locked += 1

    def landing_available(self, state: FrameState) -> bool:
        if state.falling or state.landing_window or state.miss_risk:
            return state.nearest_platform_dx is not None
        if state.rising and state.nearest_platform_above_dx is not None:
            return True
        return state.target_dx is not None or state.nearest_platform_dx is not None

    def thread_dx(self, state: FrameState) -> float | None:
        if self.thread_x is None:
            return None
        orb_x = _norm_orb_x(state)
        if orb_x is None:
            return None
        return self.thread_x - orb_x

    def target_dx(self, state: FrameState) -> float | None:
        # Survival always wins — never miss a pad to stay on-thread.
        if state.falling or state.landing_window or state.miss_risk:
            pad = state.nearest_platform_dx
            if pad is None:
                return None
            tdx = self.thread_dx(state)
            if tdx is None:
                return pad
            if abs(pad) < self.thread_corridor and abs(tdx) > 0.04:
                return (1.0 - self.thread_blend) * pad + self.thread_blend * tdx
            return pad

        above, _ = self._safe_climb_target(state)
        tdx = self.thread_dx(state)
        if state.rising and above is not None:
            if tdx is None:
                return above
            if abs(above) < 0.10 and abs(tdx) > 0.06:
                return (1.0 - self.thread_blend) * above + self.thread_blend * tdx
            if abs(tdx) > self.thread_corridor and abs(above) > 0.12:
                return 0.55 * above + 0.45 * tdx
            return above

        base = RulePolicy.target_dx(self, state)
        if base is None:
            return tdx
        if tdx is None:
            return base
        if abs(tdx) > 0.08 and abs(base) < 0.05:
            return (1.0 - self.thread_blend) * base + self.thread_blend * tdx
        return base

    def act(self, state: FrameState) -> int:
        self.observe(state)
        dx = self.target_dx(state)
        _, climb_dy = self._safe_climb_target(state)
        dy = (
            state.nearest_platform_dy
            if state.falling or state.landing_window or state.miss_risk
            else climb_dy
        )
        if dy is None:
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
            threshold = (
                self.recovery_dx_threshold if state.miss_risk else self.steer_threshold
            )
            if dx < -threshold:
                action = LEFT
            elif dx > threshold:
                action = RIGHT

        gap = dy if dy is not None else 0.0
        should_boost = (
            state.can_boost
            and state.boost_level >= self.min_energy_to_boost
            and (
                state.boost_useful is True
                or (
                vy < self.fall_vy_threshold
                and (dy is None or dy > self.gap_threshold)
                )
                or (
                    state.rising
                    and gap > 0.08
                    and state.boost_level >= 0.22
                    and abs(dx or 0.0) < 0.08
                )
            )
        )

        # Don't dump fuel when already on-thread and not in a real gap.
        tdx = self.thread_dx(state)
        if (
            should_boost
            and not state.miss_risk
            and not state.falling
            and tdx is not None
            and abs(tdx) < 0.06
            and gap < 0.12
            and state.boost_level >= self.suppress_aligned_boost_energy
            and state.boost_useful is not True
        ):
            should_boost = False

        if state.booster_type == "drag" and state.combo > 3:
            should_boost = False
        if state.danger_sell and not state.miss_risk:
            should_boost = False
        if state.hazard_dx is not None and abs(state.hazard_dx) < 0.12:
            # Near hazard: only boost for survival miss_risk.
            if not state.miss_risk:
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

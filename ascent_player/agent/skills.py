"""Composable play skills: deterministic controllers + priority router.

Each skill is a short motor routine (land, climb, surge, stream, hazard).
The router picks the highest-priority applicable skill; FREE_PLAY leaves
control to the DQN Q-head.
"""

from __future__ import annotations

from dataclasses import dataclass

from ascent_player.agent.teacher import (
    LEFT,
    LEFT_JUMP,
    RIGHT,
    RIGHT_JUMP,
    JUMP,
    RulePolicy,
)
from ascent_player.env.entity_labels import target_kind_slot
from ascent_player.env.state_detector import FrameState, mask_jump_action

FREE_PLAY = "free_play"
LAND_BELOW = "land_below"
CLIMB_ABOVE = "climb_above"
INTERCEPT_SURGE = "intercept_surge"
RIDE_STREAM = "ride_stream"
AVOID_DRAG = "avoid_drag"
DODGE_HAZARD = "dodge_hazard"
HOLD_ALIGN = "hold_align"

SKILL_IDS: tuple[str, ...] = (
    FREE_PLAY,
    LAND_BELOW,
    CLIMB_ABOVE,
    INTERCEPT_SURGE,
    RIDE_STREAM,
    AVOID_DRAG,
    DODGE_HAZARD,
    HOLD_ALIGN,
)

_SKILL_INDEX = {name: i for i, name in enumerate(SKILL_IDS)}

# Router priority (first match wins).
_SKILL_PRIORITY: tuple[str, ...] = (
    DODGE_HAZARD,
    LAND_BELOW,
    AVOID_DRAG,
    INTERCEPT_SURGE,
    RIDE_STREAM,
    CLIMB_ABOVE,
    HOLD_ALIGN,
)


def skill_count() -> int:
    return len(SKILL_IDS)


def skill_to_index(skill: str | None) -> int:
    if skill is None:
        return _SKILL_INDEX[FREE_PLAY]
    return _SKILL_INDEX.get(skill, _SKILL_INDEX[FREE_PLAY])


def index_to_skill(index: int) -> str:
    if 0 <= index < len(SKILL_IDS):
        return SKILL_IDS[index]
    return FREE_PLAY


def _steer_toward(dx: float | None, *, threshold: float) -> int:
    if dx is None:
        return 0
    if dx < -threshold:
        return LEFT
    if dx > threshold:
        return RIGHT
    return 0


def _steer_away(dx: float | None, *, threshold: float) -> int:
    if dx is None:
        return 0
    if dx < -threshold:
        return RIGHT
    if dx > threshold:
        return LEFT
    return 0


def _apply_boost(
    action: int,
    state: FrameState,
    *,
    steer_dx: float | None,
    gap_dy: float | None,
    vy: float,
    policy: RulePolicy,
) -> int:
    gap = gap_dy if gap_dy is not None else 0.0
    should_boost = (
        state.can_boost
        and state.boost_level >= policy.min_energy_to_boost
        and (
            state.boost_useful is True
            or (
                vy < policy.fall_vy_threshold
                and (gap_dy is None or gap > policy.gap_threshold)
            )
            or (
                state.rising
                and gap > 0.08
                and state.boost_level >= 0.22
                and abs(steer_dx or 0.0) < 0.10
            )
        )
    )
    if state.booster_type == "drag" and state.combo > 3:
        should_boost = False
    if state.danger_sell and not state.miss_risk:
        should_boost = False
    if should_boost:
        if action == LEFT:
            return LEFT_JUMP
        if action == RIGHT:
            return RIGHT_JUMP
        return JUMP
    return action


@dataclass(slots=True)
class SkillControllers:
    """Focused routines derived from RulePolicy thresholds."""

    policy: RulePolicy

    def applicable(self, skill: str, state: FrameState) -> bool:
        if skill == FREE_PLAY:
            return True
        if skill == DODGE_HAZARD:
            return state.hazard_dx is not None and abs(state.hazard_dx) < 0.35
        if skill == LAND_BELOW:
            return bool(
                state.falling
                or state.landing_window
                or state.miss_risk
                or (
                    state.airborne
                    and not state.rising
                    and (state.orb_vy or 0.0) < -0.05
                    and (state.nearest_platform_dy or 0.0) > 0.10
                    and state.nearest_platform_dx is not None
                )
            )
        if skill == AVOID_DRAG:
            dx = state.booster_dx if state.booster_type == "drag" else None
            if dx is None and target_kind_slot(state.target_kind) == "booster_drag":
                dx = state.target_dx
            return dx is not None and abs(dx) < 0.40
        if skill == INTERCEPT_SURGE:
            if target_kind_slot(state.target_kind) != "booster_surge":
                return False
            return state.target_dx is not None and abs(state.target_dx) < 0.55
        if skill == RIDE_STREAM:
            if target_kind_slot(state.target_kind) == "booster_stream":
                return state.target_dx is not None and abs(state.target_dx) < 0.60
            return state.booster_type == "stream" and state.booster_dx is not None
        if skill == CLIMB_ABOVE:
            return bool(
                state.rising
                and state.nearest_platform_above_dx is not None
                and (state.nearest_platform_above_dy or 0.0) > 0.02
            )
        if skill == HOLD_ALIGN:
            aim = state.target_dx or state.nearest_platform_dx
            return aim is not None and abs(aim) < 0.04 and not state.falling
        return False

    def act(self, skill: str, state: FrameState) -> int:
        if skill == FREE_PLAY:
            return 0
        p = self.policy
        vy = state.orb_vy or 0.0
        if abs(vy) > 1.5:
            vy = vy / 1500.0

        if skill == DODGE_HAZARD:
            action = _steer_away(state.hazard_dx, threshold=p.steer_threshold)
            return mask_jump_action(action, state.can_boost)

        if skill == LAND_BELOW:
            dx = state.nearest_platform_dx
            threshold = p.recovery_dx_threshold if state.miss_risk else p.steer_threshold
            action = _steer_toward(dx, threshold=threshold)
            if state.danger_sell and dx is not None:
                action = _steer_toward(dx, threshold=p.steer_threshold)
            action = _apply_boost(
                action,
                state,
                steer_dx=dx,
                gap_dy=state.nearest_platform_dy,
                vy=vy,
                policy=p,
            )
            return mask_jump_action(action, state.can_boost)

        if skill == AVOID_DRAG:
            dx = state.booster_dx if state.booster_type == "drag" else state.target_dx
            action = _steer_away(dx, threshold=p.steer_threshold)
            return mask_jump_action(action, state.can_boost)

        if skill == INTERCEPT_SURGE:
            dx = state.target_dx
            action = _steer_toward(dx, threshold=p.steer_threshold * 0.85)
            action = _apply_boost(
                action,
                state,
                steer_dx=dx,
                gap_dy=state.target_dy,
                vy=vy,
                policy=p,
            )
            return mask_jump_action(action, state.can_boost)

        if skill == RIDE_STREAM:
            dx = state.target_dx if state.target_dx is not None else state.booster_dx
            action = _steer_toward(dx, threshold=p.steer_threshold * 0.7)
            if state.boost_useful is True and state.can_boost:
                action = _apply_boost(
                    action,
                    state,
                    steer_dx=dx,
                    gap_dy=state.target_dy,
                    vy=vy,
                    policy=p,
                )
            return mask_jump_action(action, state.can_boost)

        if skill == CLIMB_ABOVE:
            dx = state.nearest_platform_above_dx
            action = _steer_toward(dx, threshold=p.steer_threshold)
            action = _apply_boost(
                action,
                state,
                steer_dx=dx,
                gap_dy=state.nearest_platform_above_dy,
                vy=vy,
                policy=p,
            )
            return mask_jump_action(action, state.can_boost)

        if skill == HOLD_ALIGN:
            return mask_jump_action(0, state.can_boost)

        return mask_jump_action(0, state.can_boost)


class SkillRouter:
    """Pick the active skill for the current frame (teacher labels)."""

    def __init__(self, controllers: SkillControllers | None = None) -> None:
        self.controllers = controllers or SkillControllers(RulePolicy())

    def propose(self, state: FrameState) -> str:
        for skill in _SKILL_PRIORITY:
            if self.controllers.applicable(skill, state):
                return skill
        return FREE_PLAY

    def act(self, state: FrameState, skill: str | None = None) -> tuple[int, str]:
        chosen = skill or self.propose(state)
        if chosen == FREE_PLAY:
            return 0, FREE_PLAY
        if not self.controllers.applicable(chosen, state):
            chosen = self.propose(state)
        if chosen == FREE_PLAY:
            return 0, FREE_PLAY
        return self.controllers.act(chosen, state), chosen

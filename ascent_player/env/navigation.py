"""Shared navigation helpers for browser and sim FrameState."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ascent_player.env.state_detector import FrameState

PLATFORM_TYPES = ("neutral", "buy", "sell")


def platform_type_onehot(platform_type: str | None) -> tuple[float, float, float]:
    if platform_type == "buy":
        return 1.0, 0.0, 0.0
    if platform_type == "sell":
        return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0


def enrich_navigation(state: "FrameState") -> "FrameState":
    """Fill derived navigation fields when platform targets are known."""
    vx = state.orb_vx or 0.0
    vy = state.orb_vy or 0.0
    # Prefer agent-hook phase when present; otherwise derive from velocity.
    if not state.falling and not state.rising:
        state.rising = vy > 20.0
        state.falling = vy < -20.0
    else:
        # Keep hook phase but refresh if velocity strongly disagrees.
        if vy < -20.0:
            state.falling = True
            state.rising = False
        elif vy > 20.0:
            state.rising = True
            state.falling = False
    state.airborne = abs(vy) > 20.0 or (state.nearest_platform_dy or 1.0) > 0.08
    state.landing_window = state.falling and 0.05 < (state.nearest_platform_dy or 0.0) < 0.35

    # While falling, lock aim to the pad below — never above/booster.
    if state.falling or state.landing_window:
        if state.nearest_platform_dx is not None:
            state.target_dx = state.nearest_platform_dx
            state.target_dy = state.nearest_platform_dy
            state.target_kind = "platform"

    target_dx = state.target_dx
    if target_dx is None:
        target_dx = state.nearest_platform_above_dx
    if target_dx is None:
        target_dx = state.nearest_platform_dx
    if target_dx is not None:
        state.target_dx = target_dx

    target_dy = state.target_dy
    if target_dy is None:
        target_dy = state.nearest_platform_above_dy
    if target_dy is None:
        target_dy = state.nearest_platform_dy
    if target_dy is not None:
        state.target_dy = target_dy

    if target_dx is not None:
        state.horizontal_error = target_dx
    elif state.orb_x is not None:
        state.horizontal_error = 0.0

    wear = max(state.platform_wear, state.nearest_platform_above_wear)
    state.danger_worn = wear >= 0.85
    state.danger_sell = state.target_platform_type == "sell" or state.nearest_platform_above_type == "sell"
    below_dx = state.nearest_platform_dx if state.nearest_platform_dx is not None else target_dx
    state.miss_risk = (
        state.falling
        and below_dx is not None
        and abs(below_dx) > 0.18
        and (state.nearest_platform_dy or 0.0) < 0.4
    )

    gap = state.nearest_platform_dy or 0.0
    energy_ok = state.can_boost and state.boost_level >= 0.14
    falling_fast = vy < -120.0
    wide_gap = gap > 0.18
    if state.boost_useful is None:
        state.boost_useful = bool(
            energy_ok and (falling_fast or wide_gap or state.miss_risk)
        )
    return state


def score_platform(
    dx: float,
    dy: float,
    *,
    platform_type: str = "neutral",
    wear: float = 0.0,
    prefer_above: bool,
) -> float:
    type_penalty = 0.35 if platform_type == "sell" else 0.0
    wear_penalty = max(0.0, wear - 0.7) * 0.5
    horizontal = abs(dx) * 2.0
    vertical = abs(dy) if prefer_above else -dy * 0.5
    return horizontal + vertical + type_penalty + wear_penalty

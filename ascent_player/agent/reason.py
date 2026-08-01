"""Heuristic action reasons for aux-head training and decision logs."""

from __future__ import annotations

from ascent_player.env.entity_labels import target_kind_slot
from ascent_player.env.state_detector import FrameState, JUMP_ACTIONS

LEFT_ACTIONS = frozenset({1, 4})
RIGHT_ACTIONS = frozenset({2, 5})
NOOP = 0

TOWARD_PLATFORM_BELOW = "toward_platform_below"
TOWARD_PLATFORM_ABOVE = "toward_platform_above"
TOWARD_BOOSTER_SURGE = "toward_booster_surge"
TOWARD_BOOSTER_STREAM = "toward_booster_stream"
AVOID_BOOSTER_DRAG = "avoid_booster_drag"
TOWARD_VOID_PORTAL = "toward_void_portal"
DODGE_HAZARD = "dodge_hazard"
CLOSE_GAP_BOOST = "close_gap_boost"
HOLD_ALIGN = "hold_align"
EXPLORE = "explore"
RULE_PRIOR = "rule_prior"

REASON_IDS: tuple[str, ...] = (
    TOWARD_PLATFORM_BELOW,
    TOWARD_PLATFORM_ABOVE,
    TOWARD_BOOSTER_SURGE,
    TOWARD_BOOSTER_STREAM,
    AVOID_BOOSTER_DRAG,
    TOWARD_VOID_PORTAL,
    DODGE_HAZARD,
    CLOSE_GAP_BOOST,
    HOLD_ALIGN,
    EXPLORE,
    RULE_PRIOR,
)

REASON_LABELS: dict[str, str] = {
    TOWARD_PLATFORM_BELOW: "move toward platform below",
    TOWARD_PLATFORM_ABOVE: "move toward platform above",
    TOWARD_BOOSTER_SURGE: "move toward yellow orb (surge)",
    TOWARD_BOOSTER_STREAM: "move toward boost highway (stream)",
    AVOID_BOOSTER_DRAG: "move away from drag wall",
    TOWARD_VOID_PORTAL: "move toward void portal",
    DODGE_HAZARD: "dodge anomaly hazard",
    CLOSE_GAP_BOOST: "boost to close gap",
    HOLD_ALIGN: "hold / stay aligned",
    EXPLORE: "explore / unknown",
    RULE_PRIOR: "rule prior override",
}

_REASON_INDEX = {name: i for i, name in enumerate(REASON_IDS)}


def reason_to_index(reason: str | None) -> int:
    if reason is None:
        return _REASON_INDEX[EXPLORE]
    return _REASON_INDEX.get(reason, _REASON_INDEX[EXPLORE])


def index_to_reason(index: int) -> str:
    if 0 <= index < len(REASON_IDS):
        return REASON_IDS[index]
    return EXPLORE


def reason_count() -> int:
    return len(REASON_IDS)


def _steer_matches(action: int, dx: float | None, *, threshold: float = 0.012) -> bool:
    if dx is None:
        return False
    if dx < -threshold:
        return action in LEFT_ACTIONS
    if dx > threshold:
        return action in RIGHT_ACTIONS
    return action == NOOP or action == 3


def _steer_opposes(action: int, dx: float | None, *, threshold: float = 0.012) -> bool:
    if dx is None:
        return False
    if dx < -threshold:
        return action in RIGHT_ACTIONS
    if dx > threshold:
        return action in LEFT_ACTIONS
    return False


def assign_reason(
    frame_state: FrameState | None,
    action: int,
    *,
    source: str = "greedy",
) -> str:
    """Return a discrete reason ID for (state, action, decision source)."""
    if source == "explore":
        return EXPLORE
    # Thread prior shares the rule-prior aux class so checkpoint heads stay sized.
    if source in ("rule", "thread", "skill"):
        return RULE_PRIOR
    if frame_state is None:
        return EXPLORE

    if action in JUMP_ACTIONS and (
        frame_state.boost_useful
        or (
            frame_state.falling
            and (frame_state.nearest_platform_dy or 0.0) > 0.15
            and frame_state.can_boost
        )
    ):
        return CLOSE_GAP_BOOST

    if frame_state.falling or frame_state.landing_window or frame_state.miss_risk:
        plat_dx = frame_state.nearest_platform_dx
        if _steer_matches(action, plat_dx):
            return TOWARD_PLATFORM_BELOW

    portal_dx = frame_state.portal_dx
    if portal_dx is not None and _steer_matches(action, portal_dx):
        return TOWARD_VOID_PORTAL

    hazard_dx = frame_state.hazard_dx
    if hazard_dx is not None and _steer_opposes(action, hazard_dx):
        return DODGE_HAZARD

    slot = target_kind_slot(frame_state.target_kind)
    target_dx = frame_state.target_dx
    if slot == "booster_surge" and _steer_matches(action, target_dx):
        return TOWARD_BOOSTER_SURGE
    if slot == "booster_stream" and _steer_matches(action, target_dx):
        return TOWARD_BOOSTER_STREAM
    if (
        slot == "booster_drag" or frame_state.booster_type == "drag"
    ) and _steer_opposes(action, frame_state.booster_dx or target_dx):
        return AVOID_BOOSTER_DRAG
    if slot == "platform" and not (frame_state.falling or frame_state.landing_window):
        if _steer_matches(action, frame_state.nearest_platform_above_dx):
            return TOWARD_PLATFORM_ABOVE
        if _steer_matches(action, frame_state.nearest_platform_dx or target_dx):
            return TOWARD_PLATFORM_BELOW

    aim_dx = target_dx if target_dx is not None else frame_state.nearest_platform_dx
    if action == NOOP and (aim_dx is None or abs(aim_dx) < 0.05):
        return HOLD_ALIGN
    if action == 3 and (aim_dx is None or abs(aim_dx) < 0.05):
        return HOLD_ALIGN

    return EXPLORE


def wrong_vs_target(frame_state: FrameState | None, action: int) -> bool:
    """True when steering clearly opposes the survival / target aim."""
    if frame_state is None:
        return False
    if frame_state.falling or frame_state.landing_window or frame_state.miss_risk:
        return _steer_opposes(action, frame_state.nearest_platform_dx, threshold=0.08)
    return _steer_opposes(action, frame_state.target_dx, threshold=0.08)

"""Climb promote / milestone policy (pure decisions)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Nature-floor recovery: offline TD after BC often undoes the clone below this.
# Defer offline TD until online-FT gate; BC-only climbs past Nature more reliably.
TD_RECOVERY_DEFER_MEAN = 1400.0
RELIABILITY_WARN_RATIO = 0.70


class PromoteAction(str, Enum):
    PROMOTE = "promote"
    HOLD = "hold"
    REVERT = "revert"


@dataclass(frozen=True)
class PromoteDecision:
    action: PromoteAction
    improved: bool
    collapsed: bool


def decide_promote(
    *,
    probe_mean: float,
    v2_best: float,
    bootstrap_floor: float,
) -> PromoteDecision:
    improved = probe_mean > v2_best + 1.0
    collapsed = v2_best > 0 and probe_mean < max(
        bootstrap_floor * 0.5, v2_best * 0.55
    )
    if v2_best <= 0:
        action = (
            PromoteAction.PROMOTE
            if probe_mean >= bootstrap_floor
            else PromoteAction.REVERT
        )
    elif improved:
        action = PromoteAction.PROMOTE
    elif collapsed:
        action = PromoteAction.REVERT
    else:
        action = PromoteAction.HOLD
    return PromoteDecision(action=action, improved=improved, collapsed=collapsed)


def elite_gate_for_score(v2_best: float, post_2k_gate: float = 3000.0) -> float | None:
    """Return elite min score to apply, or None if unchanged."""
    if v2_best >= 3000.0:
        return max(float(post_2k_gate), 4000.0)
    if v2_best >= 2000.0:
        return float(post_2k_gate)
    return None


def aux_enabled_for_score(v2_best: float) -> bool:
    return v2_best >= 2000.0


def human_cap(room: int, max_items: int, human_share_cap: float) -> int:
    share_cap = int(max_items * min(1.0, max(0.0, human_share_cap)))
    return max(0, min(room, share_cap))


def should_defer_offline_td(
    v2_best: float, *, gate: float = TD_RECOVERY_DEFER_MEAN
) -> bool:
    """True while recovering below the Impala TD-safe floor."""
    return float(v2_best) < float(gate)


def offline_td_min_score(*, best_mean: float, v2_best: float) -> float:
    """Replay score gate for offline TD after browser collect."""
    if best_mean >= 1500.0:
        return 2000.0
    if v2_best >= 500.0:
        return 600.0
    return 400.0


def reliability_below_floor(
    rel_mean: float,
    v2_best: float,
    *,
    ratio: float = RELIABILITY_WARN_RATIO,
    slack: float = 1.0,
) -> bool:
    """True when reliability mean is suspiciously below the promoted floor."""
    if v2_best <= 0:
        return False
    return float(rel_mean) + float(slack) < float(v2_best) * float(ratio)

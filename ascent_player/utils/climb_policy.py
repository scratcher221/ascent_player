"""Climb promote / milestone policy (pure decisions)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Nature-floor recovery: offline TD after BC often undoes the clone below this.
# Defer offline TD until online-FT gate; BC-only climbs past Nature more reliably.
TD_RECOVERY_DEFER_MEAN = 1400.0
RELIABILITY_WARN_RATIO = 0.70
# Reject confirm that falls this far below the probe (last-10 style spikes).
PROMOTE_CONFIRM_SLACK = 150.0
TAIL_ELITE_GATE = 1400.0
COLLECT_GATE_LOW = 1000.0
COLLECT_GATE_HIGH = 1400.0
COLLECT_GATE_RAISE_MEAN = 1200.0
ELITE_BC_MIN_EPISODES = 32
STALL_FT_GATE = 1200.0
TEACHER_DISTILL_MIN_SCORE = 2000.0
MIN_OVERNIGHT_HOURS = 8.0
DEFAULT_COLLECT_PRIOR = 0.20
STARVE_COLLECT_PRIOR = 0.35
SESSION_ABORT_DELTA = 50.0
SESSION_ABORT_MAX = 1600.0
CLIMB_RUNGS = (1400.0, 1600.0, 1800.0, 2000.0)
CLIMB_SESSIONS = Path("logs/v2_climb_sessions.jsonl")
COLLECT_PRIOR_PATH = Path("logs/v2_collect_prior.txt")
PHASE2_STALL_PATH = Path("logs/v2_phase2_bc_stall.txt")


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


def elite_gate_for_score(v2_best: float, post_2k_gate: float = 3000.0) -> float:
    """Elite min score for the current floor.

    Impala greedy has not produced 2000-class play, so harvest the real
    1400–1550 tail until a true 2k floor exists.
    """
    if v2_best >= 3000.0:
        return max(float(post_2k_gate), 4000.0)
    if v2_best >= 2000.0:
        return float(post_2k_gate)
    return float(TAIL_ELITE_GATE)


def collect_replay_gate(true_mean: float) -> float:
    """Collect persist filter: 1000, then 1400 once mean ≥1200."""
    if float(true_mean) >= COLLECT_GATE_RAISE_MEAN:
        return float(COLLECT_GATE_HIGH)
    return float(COLLECT_GATE_LOW)


def browser_bc_gate(true_mean: float, elite_episodes: int) -> float:
    """Post-collect BC filter.

    Once elite has ≥32 episodes, clone the 1400 tail even if true_mean is
    still below 1200. Otherwise follow collect_replay_gate.
    """
    if int(elite_episodes) >= int(ELITE_BC_MIN_EPISODES):
        return float(COLLECT_GATE_HIGH)
    return collect_replay_gate(true_mean)


def bc_score_gate(
    true_mean: float,
    kept_at_primary: int,
    *,
    elite_episodes: int = 0,
    min_kept: int = 64,
) -> float:
    """Effective BC filter after a too-thin primary keep.

    Fall back to 1000 only when the elite+browser mix still has <64
    transitions after the primary gate.
    """
    primary = browser_bc_gate(true_mean, elite_episodes)
    if int(kept_at_primary) >= int(min_kept):
        return primary
    return float(COLLECT_GATE_LOW)


def should_skip_leftover_hybrid_bc(
    *, collecting: bool, collect_minutes: float
) -> bool:
    """True when the collect window closed — do not refit the same pickle."""
    return (not collecting) and float(collect_minutes) > 0


def current_rung(mean: float, rungs: tuple[float, ...] = CLIMB_RUNGS) -> float:
    reached = 0.0
    for target in rungs:
        if float(mean) >= float(target):
            reached = float(target)
        else:
            break
    return reached


def should_run_online_td(
    *,
    collecting: bool,
    true_mean: float,
    ft_gate: float,
) -> bool:
    """Online collect TD: at the FT gate, or immediately after a Phase-2 stall."""
    if not collecting:
        return False
    if PHASE2_STALL_PATH.exists():
        return True
    return float(true_mean) >= float(ft_gate)


def online_td_gate(default: float = TD_RECOVERY_DEFER_MEAN) -> float:
    """Online FT bar. After a stalled Phase-2 overnight, allow TD from 1200."""
    if PHASE2_STALL_PATH.exists():
        return float(STALL_FT_GATE)
    return float(default)


def confirm_supports_probe(
    probe_mean: float,
    confirm_mean: float,
    *,
    slack: float = PROMOTE_CONFIRM_SLACK,
) -> bool:
    """True when the full-N confirm is not a collapse vs the probe."""
    return float(confirm_mean) + 1e-6 >= float(probe_mean) - float(slack)


def eval_session_stats(scores: list[float]) -> dict[str, float]:
    """Full-N mean plus last-10 window. Empty scores yield zeros."""
    if not scores:
        return {
            "episode_mean": 0.0,
            "episode_min": 0.0,
            "episode_max": 0.0,
            "n_episodes": 0.0,
            "recent_avg": 0.0,
            "recent_min": 0.0,
            "recent_max": 0.0,
        }
    recent = scores[-10:]
    return {
        "episode_mean": float(sum(scores) / len(scores)),
        "episode_min": float(min(scores)),
        "episode_max": float(max(scores)),
        "n_episodes": float(len(scores)),
        "recent_avg": float(sum(recent) / len(recent)),
        "recent_min": float(min(recent)),
        "recent_max": float(max(recent)),
    }


def collect_thread_prior(default: float = DEFAULT_COLLECT_PRIOR) -> float:
    if not COLLECT_PRIOR_PATH.exists():
        return float(default)
    try:
        return float(COLLECT_PRIOR_PATH.read_text(encoding="utf-8").strip())
    except ValueError:
        return float(default)


def record_climb_session(
    *,
    start_true_mean: float,
    end_true_mean: float,
    max_score: float,
    hours: float,
    greedy_max: float | None = None,
) -> str | None:
    """Append session stats. Returns 'abort' after two stalled overnights.

    Stall uses greedy_max (not assisted collect peaks). A single overnight
    stall (≥8h, delta<50, greedy max<1600) writes PHASE2_STALL_PATH so
    the next climb may run online TD from 1200.
    """
    CLIMB_SESSIONS.parent.mkdir(parents=True, exist_ok=True)
    delta = float(end_true_mean) - float(start_true_mean)
    greedy = float(max_score if greedy_max is None else greedy_max)
    row = {
        "start_true_mean": float(start_true_mean),
        "end_true_mean": float(end_true_mean),
        "delta": delta,
        "max_score": float(max_score),
        "greedy_max": greedy,
        "hours": float(hours),
    }
    with CLIMB_SESSIONS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    if (
        float(hours) >= MIN_OVERNIGHT_HOURS
        and delta < SESSION_ABORT_DELTA
        and greedy < SESSION_ABORT_MAX
    ):
        PHASE2_STALL_PATH.parent.mkdir(parents=True, exist_ok=True)
        PHASE2_STALL_PATH.write_text("1\n", encoding="utf-8")
        print(
            f"CLIMB_PHASE2_STALL delta={delta:.1f} greedy_max={greedy:.0f} "
            f"— next online TD gate={STALL_FT_GATE:.0f}",
            flush=True,
        )
    stalled = _last_sessions_stalled(n=2)
    if stalled:
        COLLECT_PRIOR_PATH.write_text(f"{STARVE_COLLECT_PRIOR:.2f}\n", encoding="utf-8")
        return "abort"
    return None


def _last_sessions_stalled(n: int = 2) -> bool:
    if not CLIMB_SESSIONS.exists():
        return False
    lines = [
        line
        for line in CLIMB_SESSIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < n:
        return False
    rows = []
    for line in lines[-n:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            return False
    return all(
        float(row.get("delta", 0.0)) < SESSION_ABORT_DELTA
        and float(row.get("greedy_max", row.get("max_score", 0.0)))
        < SESSION_ABORT_MAX
        for row in rows
    )


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

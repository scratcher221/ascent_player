"""Browser replay persistence policy (extracted from training finally-block)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ascent_player.agent.dqn import DQNAgent
    from ascent_player.config import AppConfig


@dataclass(frozen=True)
class PersistDecision:
    action: str  # save | skip
    reason: str
    saved: int = 0


def persist_browser_replay(
    agent: "DQNAgent",
    config: "AppConfig",
    *,
    recent_avg: float,
) -> PersistDecision:
    """Decide whether to write browser_replay.pkl at session end."""
    if config.training.sim_mode:
        return PersistDecision("skip", "sim_mode")
    force_save = bool(getattr(config.training, "force_save_browser_replay", False))
    replay_n = len(agent.replay)
    if config.training.watch_mode and not force_save:
        print(f"SKIP_REPLAY_SAVE watch_mode (recent_avg={recent_avg:.0f})", flush=True)
        return PersistDecision("skip", "watch_mode")
    if force_save and replay_n <= 0:
        existing = config.training.browser_replay_path
        existing_n = existing.stat().st_size if existing.exists() else 0
        print(
            f"SKIP_REPLAY_SAVE force_save but replay empty "
            f"(keep existing={existing_n}B recent_avg={recent_avg:.0f})",
            flush=True,
        )
        return PersistDecision("skip", "empty_force_save")
    if force_save or recent_avg >= 850.0:
        saved = agent.replay.save_pickle(
            config.training.browser_replay_path,
            max_items=config.training.browser_replay_max_items,
        )
        if saved:
            print(
                f"Saved {saved} browser replay transitions -> "
                f"{config.training.browser_replay_path} "
                f"(recent_avg={recent_avg:.0f}"
                f"{', forced' if force_save else ''})",
                flush=True,
            )
            return PersistDecision("save", "ok", saved=int(saved))
        print(
            f"SKIP_REPLAY_SAVE save_pickle returned 0 (recent_avg={recent_avg:.0f})",
            flush=True,
        )
        return PersistDecision("skip", "save_zero")
    print(
        f"SKIP_REPLAY_SAVE recent_avg={recent_avg:.0f} < 850 "
        f"(avoid poisoning seed continue)",
        flush=True,
    )
    return PersistDecision("skip", "below_threshold")

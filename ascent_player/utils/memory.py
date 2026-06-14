from __future__ import annotations

import psutil

from ascent_player.config import AppConfig


def available_memory_bytes() -> int:
    return int(psutil.virtual_memory().available)


def observation_state_bytes(config: AppConfig) -> int:
    channels = config.observation.channel_count
    return (
        config.observation.height
        * config.observation.width
        * channels
        * 4  # float32
    )


def replay_transition_bytes(config: AppConfig, multiplier: int | None = None) -> int:
    """Bytes consumed in replay for one unique demo transition."""
    mult = multiplier if multiplier is not None else config.demo.replay_multiplier
    state_bytes = observation_state_bytes(config)
    return state_bytes * 2 * max(1, mult)


def demo_memory_budget_bytes(config: AppConfig) -> int:
    reserve = config.demo.os_memory_reserve_mb * 1024 * 1024
    return max(0, available_memory_bytes() - reserve)


def memory_headroom_ok(config: AppConfig) -> bool:
    return available_memory_bytes() >= config.demo.os_memory_reserve_mb * 1024 * 1024


def max_demo_transitions(config: AppConfig) -> int:
    """Upper bound on unique demo transitions to load without starving the OS."""
    if config.demo.max_transitions is not None:
        return max(0, config.demo.max_transitions)

    budget = demo_memory_budget_bytes(config)
    per_transition = replay_transition_bytes(config)
    if per_transition <= 0:
        return 0

    from_memory = int(budget / per_transition)
    replay_cap = config.training.replay_buffer_size // max(
        1, config.demo.replay_multiplier
    )
    return max(0, min(from_memory, replay_cap))

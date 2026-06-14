from __future__ import annotations

import numpy as np

from ascent_player.config import RewardConfig

JUMP_ACTIONS = frozenset({3, 4, 5})
BOOST_CHANNEL_INDEX = -2


def boost_level_from_state(state: np.ndarray) -> float:
    if state.ndim != 3 or state.shape[-1] < 5:
        return 1.0
    return float(np.clip(state[..., BOOST_CHANNEL_INDEX].mean(), 0.0, 1.0))


def can_boost_from_state(
    state: np.ndarray,
    *,
    min_energy: float = 14.0,
) -> bool:
    return boost_level_from_state(state) * 100.0 >= min_energy


def valid_action_mask(
    state: np.ndarray,
    action_count: int,
    *,
    min_energy: float = 14.0,
) -> np.ndarray:
    mask = np.zeros(action_count, dtype=np.float32)
    if can_boost_from_state(state, min_energy=min_energy):
        mask[:] = 1.0
    else:
        mask[:3] = 1.0
    return mask


def masked_argmax_q(
    q_values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    masked = np.where(mask > 0.0, q_values, -np.inf)
    return np.argmax(masked, axis=-1)


def masked_max_q(
    q_values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    masked = np.where(mask > 0.0, q_values, -np.inf)
    return np.max(masked, axis=-1)

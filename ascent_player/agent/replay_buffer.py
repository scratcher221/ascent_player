from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
import threading

import numpy as np


@dataclass(slots=True)
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: deque[tuple] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    @staticmethod
    def _copy_state(state):
        if isinstance(state, tuple):
            return (state[0].copy(), state[1].copy())
        return state.copy()

    def add(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
    ) -> None:
        with self._lock:
            self._items.append(
                (
                    self._copy_state(state),
                    action,
                    reward,
                    self._copy_state(next_state),
                    done,
                )
            )

    def add_many(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        with self._lock:
            for idx in range(len(actions)):
                self._items.append(
                    (
                        self._copy_state(states[idx]),
                        int(actions[idx]),
                        float(rewards[idx]),
                        self._copy_state(next_states[idx]),
                        bool(dones[idx]),
                    )
                )

    def sample(self, batch_size: int) -> TransitionBatch:
        with self._lock:
            batch = random.sample(self._items, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        if batch and isinstance(states[0], tuple):
            return TransitionBatch(
                states=np.asarray(states, dtype=object),
                actions=np.asarray(actions, dtype=np.int32),
                rewards=np.asarray(rewards, dtype=np.float32),
                next_states=np.asarray(next_states, dtype=object),
                dones=np.asarray(dones, dtype=np.float32),
            )
        return TransitionBatch(
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int32),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_states=np.asarray(next_states, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
        )

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def trim_to(self, max_size: int) -> int:
        """Drop oldest transitions until at most *max_size* remain."""
        max_size = max(1, max_size)
        removed = 0
        with self._lock:
            while len(self._items) > max_size:
                self._items.popleft()
                removed += 1
        return removed

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

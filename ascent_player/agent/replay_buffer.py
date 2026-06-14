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
        self._items: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )
        self._lock = threading.Lock()

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        with self._lock:
            self._items.append((state.copy(), action, reward, next_state.copy(), done))

    def sample(self, batch_size: int) -> TransitionBatch:
        with self._lock:
            batch = random.sample(self._items, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        return TransitionBatch(
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int32),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_states=np.asarray(next_states, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

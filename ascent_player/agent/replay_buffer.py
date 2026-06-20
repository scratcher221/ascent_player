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
    indices: np.ndarray | None = None
    weights: np.ndarray | None = None


def _is_hybrid_state(state) -> bool:
    if not isinstance(state, (tuple, list)) or len(state) != 2:
        return False
    visual, vector = state
    return isinstance(visual, np.ndarray) and isinstance(vector, np.ndarray)


def _pack_hybrid_states(states) -> np.ndarray:
    packed = np.empty(len(states), dtype=object)
    packed[:] = list(states)
    return packed


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        *,
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> None:
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.beta = beta
        self._items: deque[tuple] = deque(maxlen=capacity)
        self._priorities: deque[float] = deque(maxlen=capacity)
        self._max_priority = 1.0
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
        *,
        priority: float | None = None,
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
            if self.prioritized:
                self._priorities.append(priority if priority is not None else self._max_priority)

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
                if self.prioritized:
                    self._priorities.append(self._max_priority)

    def sample(self, batch_size: int) -> TransitionBatch:
        with self._lock:
            if not self.prioritized or len(self._priorities) != len(self._items):
                batch = random.sample(self._items, batch_size)
                indices = None
                weights = None
            else:
                priorities = np.asarray(self._priorities, dtype=np.float64)
                scaled = np.power(priorities + 1e-6, self.alpha)
                probs = scaled / scaled.sum()
                indices = np.random.choice(len(self._items), batch_size, replace=False, p=probs)
                batch = [self._items[int(i)] for i in indices]
                weights = np.power(len(self._items) * probs[indices], -self.beta)
                weights = weights / weights.max()
                weights = weights.astype(np.float32)
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        if batch and _is_hybrid_state(states[0]):
            return TransitionBatch(
                states=_pack_hybrid_states(states),
                actions=np.asarray(actions, dtype=np.int32),
                rewards=np.asarray(rewards, dtype=np.float32),
                next_states=_pack_hybrid_states(next_states),
                dones=np.asarray(dones, dtype=np.float32),
                indices=indices,
                weights=weights,
            )
        return TransitionBatch(
            states=np.asarray(states, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int32),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_states=np.asarray(next_states, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
            indices=indices,
            weights=weights,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        if not self.prioritized or indices is None:
            return
        with self._lock:
            for index, error in zip(indices, td_errors, strict=True):
                priority = float(abs(error) + 1e-5)
                self._priorities[int(index)] = priority
                self._max_priority = max(self._max_priority, priority)

    def extend_from(self, other: ReplayBuffer, *, max_items: int | None = None) -> None:
        with self._lock, other._lock:
            items = list(other._items)
            if max_items is not None:
                items = items[-max_items:]
            for item in items:
                self._items.append(item)
                if self.prioritized:
                    self._priorities.append(self._max_priority)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._priorities.clear()
            self._max_priority = 1.0

    def trim_to(self, max_size: int) -> int:
        max_size = max(1, max_size)
        removed = 0
        with self._lock:
            while len(self._items) > max_size:
                self._items.popleft()
                if self.prioritized and self._priorities:
                    self._priorities.popleft()
                removed += 1
        return removed

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

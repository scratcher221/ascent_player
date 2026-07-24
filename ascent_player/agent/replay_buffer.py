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
    discounts: np.ndarray
    indices: np.ndarray | None = None
    weights: np.ndarray | None = None
    sim_indices: np.ndarray | None = None
    reasons: np.ndarray | None = None


def _is_hybrid_state(state) -> bool:
    if not isinstance(state, (tuple, list)) or len(state) != 2:
        return False
    visual, vector = state
    return isinstance(visual, np.ndarray) and isinstance(vector, np.ndarray)


def _pack_hybrid_states(states) -> np.ndarray:
    packed = np.empty(len(states), dtype=object)
    packed[:] = list(states)
    return packed


def _normalize_item(item: tuple) -> tuple:
    """Normalize to (state, action, reward, next, done, discount, reason)."""
    if len(item) == 5:
        state, action, reward, next_state, done = item
        return (state, action, reward, next_state, done, 1.0, -1)
    if len(item) == 6:
        state, action, reward, next_state, done, discount = item
        return (state, action, reward, next_state, done, float(discount), -1)
    if len(item) >= 7:
        return (
            item[0],
            item[1],
            item[2],
            item[3],
            item[4],
            float(item[5]),
            int(item[6]),
        )
    raise ValueError(f"unexpected replay item length {len(item)}")


def _adapt_vector(state, vector_dim: int):
    """Pad/truncate hybrid vector to ``vector_dim``; pass through non-hybrid."""
    if not _is_hybrid_state(state):
        return state
    visual, vector = state
    vec = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vec.shape[0] == vector_dim:
        return (visual, vec)
    out = np.zeros(vector_dim, dtype=np.float32)
    n = min(vec.shape[0], vector_dim)
    out[:n] = vec[:n]
    return (visual, out)


def _adapt_item_vector_dim(item: tuple, vector_dim: int) -> tuple:
    state, action, reward, next_state, done, discount, reason = _normalize_item(item)
    return (
        _adapt_vector(state, vector_dim),
        action,
        reward,
        _adapt_vector(next_state, vector_dim),
        done,
        discount,
        reason,
    )


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

    def set_beta(self, beta: float) -> None:
        self.beta = float(np.clip(beta, 0.0, 1.0))

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
        discount: float = 1.0,
        priority: float | None = None,
        reason: int = -1,
    ) -> None:
        with self._lock:
            self._items.append(
                (
                    self._copy_state(state),
                    action,
                    reward,
                    self._copy_state(next_state),
                    done,
                    float(discount),
                    int(reason),
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
        *,
        discounts: np.ndarray | None = None,
        reasons: np.ndarray | None = None,
    ) -> None:
        with self._lock:
            for idx in range(len(actions)):
                discount = 1.0 if discounts is None else float(discounts[idx])
                reason = -1 if reasons is None else int(reasons[idx])
                self._items.append(
                    (
                        self._copy_state(states[idx]),
                        int(actions[idx]),
                        float(rewards[idx]),
                        self._copy_state(next_states[idx]),
                        bool(dones[idx]),
                        discount,
                        reason,
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
        normalized = [_normalize_item(item) for item in batch]
        # Defensive: pad/truncate hybrid vectors if buffer has mixed dims.
        if batch and _is_hybrid_state(normalized[0][0]):
            dims = {
                int(np.asarray(item[0][1]).reshape(-1).shape[0])
                for item in normalized
                if _is_hybrid_state(item[0])
            }
            if len(dims) > 1:
                target_dim = max(dims)
                normalized = [
                    _adapt_item_vector_dim(item, target_dim) for item in normalized
                ]
        states, actions, rewards, next_states, dones, discounts, reasons = zip(
            *normalized, strict=True
        )
        common = dict(
            actions=np.asarray(actions, dtype=np.int32),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
            discounts=np.asarray(discounts, dtype=np.float32),
            indices=indices,
            weights=weights,
            reasons=np.asarray(reasons, dtype=np.int32),
        )
        if batch and _is_hybrid_state(states[0]):
            return TransitionBatch(
                states=_pack_hybrid_states(states),
                next_states=_pack_hybrid_states(next_states),
                **common,
            )
        return TransitionBatch(
            states=np.asarray(states, dtype=np.float32),
            next_states=np.asarray(next_states, dtype=np.float32),
            **common,
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
                self._items.append(_normalize_item(item))
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

    def compact_diverse_episodes(
        self,
        *,
        max_episodes: int = 8,
        max_transitions_per_episode: int = 128,
        signature_bins: int = 24,
    ) -> dict[str, int]:
        """Keep a compact, action-diverse sample of complete episodes.

        Browser frame transitions are large (~hundreds of KiB each), so keeping
        every frame from every accepted climb quickly creates multi-GB replay
        files.  BC only needs representative state/action pairs.  This method:

        * splits replay at terminal transitions,
        * prefers episodes with distinct coarse action traces,
        * uniformly samples each selected episode while retaining its terminal,
        * caps both episode count and transitions per episode.

        The replay must already have been score-gated by the collector; score is
        intentionally not inferred from observation-vector layout here.
        """
        max_episodes = max(1, int(max_episodes))
        max_per = max(2, int(max_transitions_per_episode))
        bins = max(4, int(signature_bins))

        with self._lock:
            items = [_normalize_item(item) for item in self._items]

        episodes: list[list[tuple]] = []
        current: list[tuple] = []
        for item in items:
            current.append(item)
            if bool(item[4]):
                episodes.append(current)
                current = []
        # A saved gated episode should be terminal, but retain a trailing chunk
        # defensively rather than silently losing valid collected transitions.
        if current:
            episodes.append(current)

        def signature(episode: list[tuple]) -> tuple[int, ...]:
            if len(episode) <= bins:
                sampled = episode
            else:
                indices = np.linspace(0, len(episode) - 1, bins, dtype=np.int32)
                sampled = [episode[int(index)] for index in indices]
            length_bucket = min(31, len(episode) // 64)
            return (length_bucket, *(int(item[1]) for item in sampled))

        selected: list[list[tuple]] = []
        seen: set[tuple[int, ...]] = set()
        # Prefer recent episodes because they reflect the current policy/rule.
        for episode in reversed(episodes):
            key = signature(episode)
            if key in seen:
                continue
            seen.add(key)
            selected.append(episode)
            if len(selected) >= max_episodes:
                break
        selected.reverse()

        compacted: list[tuple] = []
        for episode in selected:
            if len(episode) <= max_per:
                sampled = list(episode)
            else:
                indices = np.linspace(0, len(episode) - 1, max_per, dtype=np.int32)
                # Preserve order and avoid duplicate rounded indices.
                sampled = [episode[index] for index in dict.fromkeys(map(int, indices))]
                if bool(episode[-1][4]) and sampled[-1] is not episode[-1]:
                    sampled[-1] = episode[-1]
            compacted.extend(sampled)

        with self._lock:
            self._items.clear()
            self._priorities.clear()
            for item in compacted:
                self._items.append(item)
                if self.prioritized:
                    self._priorities.append(self._max_priority)

        return {
            "input_episodes": len(episodes),
            "selected_episodes": len(selected),
            "input_transitions": len(items),
            "output_transitions": len(compacted),
        }

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def save_pickle(self, path, *, max_items: int | None = None) -> int:
        """Persist recent transitions for cross-session compounding."""
        import pickle
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            items = [_normalize_item(item) for item in self._items]
        if max_items is not None:
            items = items[-max(1, int(max_items)) :]
        with path.open("wb") as handle:
            pickle.dump(items, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return len(items)

    def load_pickle(
        self,
        path,
        *,
        max_items: int | None = None,
        vector_dim: int | None = None,
    ) -> int:
        """Reload transitions saved by save_pickle. Returns count loaded.

        If ``vector_dim`` is set, hybrid vectors are padded/truncated to match
        (legacy 43-dim replay → current 59-dim).
        """
        import pickle
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            return 0
        with path.open("rb") as handle:
            items = pickle.load(handle)
        if not isinstance(items, list):
            return 0
        if max_items is not None:
            items = items[-max(1, int(max_items)) :]
        loaded = 0
        adapted = 0
        with self._lock:
            for item in items:
                if vector_dim is not None:
                    before = _normalize_item(item)
                    item = _adapt_item_vector_dim(item, vector_dim)
                    if _is_hybrid_state(before[0]):
                        old_n = int(np.asarray(before[0][1]).reshape(-1).shape[0])
                        if old_n != vector_dim:
                            adapted += 1
                else:
                    item = _normalize_item(item)
                self._items.append(item)
                if self.prioritized:
                    self._priorities.append(self._max_priority)
                loaded += 1
        if adapted:
            print(
                f"REPLAY_VECTOR_ADAPT adapted={adapted}/{loaded} → dim={vector_dim}",
                flush=True,
            )
        return loaded

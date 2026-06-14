from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

from ascent_player.config import AppConfig


@dataclass(slots=True)
class DemoTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    score: float | None = None
    episode_id: int | None = None


@dataclass(slots=True)
class DemoArrays:
    path: Path
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    scores: np.ndarray | None = None
    episode_ids: np.ndarray | None = None
    peak_score: float = 0.0
    _payload: np.lib.npyio.NpzFile | None = None

    def __len__(self) -> int:
        return int(len(self.actions))

    def close(self) -> None:
        if self._payload is not None:
            self._payload.close()
            self._payload = None


def demonstrations_dir(config: AppConfig) -> Path:
    return config.demo.save_dir


def list_demo_files(config: AppConfig) -> list[Path]:
    folder = demonstrations_dir(config)
    if not folder.exists():
        return []
    return sorted(folder.glob("*.npz"))


def demo_transition_count(path: Path) -> int:
    with np.load(path, mmap_mode="r") as payload:
        return int(len(payload["actions"]))


def demo_peak_score(path: Path) -> float:
    with np.load(path, mmap_mode="r") as payload:
        if "peak_score" in payload:
            peak = float(payload["peak_score"])
            if peak > 0:
                return peak
        if "scores" in payload:
            return float(np.max(payload["scores"]))
    return 0.0


def demo_has_score_metadata(path: Path) -> bool:
    with np.load(path, mmap_mode="r") as payload:
        if "peak_score" in payload and float(payload["peak_score"]) > 0:
            return True
        return "scores" in payload


@contextmanager
def open_demo(path: Path, *, mmap: bool = True) -> Iterator[DemoArrays]:
    mode = "r" if mmap else None
    payload = np.load(path, mmap_mode=mode)
    try:
        peak_score = float(payload["peak_score"]) if "peak_score" in payload else 0.0
        scores = payload["scores"] if "scores" in payload else None
        episode_ids = payload["episode_ids"] if "episode_ids" in payload else None
        if peak_score <= 0 and scores is not None:
            peak_score = float(np.max(scores))
        yield DemoArrays(
            path=path,
            states=payload["states"],
            actions=payload["actions"],
            rewards=payload["rewards"],
            next_states=payload["next_states"],
            dones=payload["dones"],
            scores=scores,
            episode_ids=episode_ids,
            peak_score=peak_score,
            _payload=payload,
        )
    finally:
        payload.close()


def save_demo(path: Path, transitions: list[DemoTransition]) -> Path:
    if not transitions:
        raise ValueError("Cannot save an empty demonstration.")
    path.parent.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(
        [item.score if item.score is not None else 0.0 for item in transitions],
        dtype=np.float32,
    )
    episode_ids = np.asarray(
        [item.episode_id if item.episode_id is not None else 0 for item in transitions],
        dtype=np.int32,
    )
    peak_score = float(np.max(scores)) if len(scores) else 0.0
    np.savez_compressed(
        path,
        states=np.asarray([item.state for item in transitions], dtype=np.float32),
        actions=np.asarray([item.action for item in transitions], dtype=np.int32),
        rewards=np.asarray([item.reward for item in transitions], dtype=np.float32),
        next_states=np.asarray([item.next_state for item in transitions], dtype=np.float32),
        dones=np.asarray([item.done for item in transitions], dtype=np.float32),
        scores=scores,
        episode_ids=episode_ids,
        peak_score=np.asarray(peak_score, dtype=np.float32),
    )
    return path


def load_demo(path: Path) -> list[DemoTransition]:
    with open_demo(path, mmap=False) as demo:
        transitions: list[DemoTransition] = []
        for idx in range(len(demo)):
            transitions.append(
                DemoTransition(
                    state=np.asarray(demo.states[idx], dtype=np.float32),
                    action=int(demo.actions[idx]),
                    reward=float(demo.rewards[idx]),
                    next_state=np.asarray(demo.next_states[idx], dtype=np.float32),
                    done=bool(demo.dones[idx]),
                    score=float(demo.scores[idx]) if demo.scores is not None else None,
                    episode_id=int(demo.episode_ids[idx])
                    if demo.episode_ids is not None
                    else None,
                )
            )
        return transitions


def load_all_demos(config: AppConfig) -> list[DemoTransition]:
    transitions: list[DemoTransition] = []
    for path in list_demo_files(config):
        transitions.extend(load_demo(path))
    return transitions


def new_demo_path(config: AppConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return demonstrations_dir(config) / f"demo_{stamp}.npz"

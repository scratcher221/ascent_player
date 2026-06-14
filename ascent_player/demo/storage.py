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


@dataclass(slots=True)
class DemoArrays:
    path: Path
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
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


@contextmanager
def open_demo(path: Path, *, mmap: bool = True) -> Iterator[DemoArrays]:
    mode = "r" if mmap else None
    payload = np.load(path, mmap_mode=mode)
    try:
        yield DemoArrays(
            path=path,
            states=payload["states"],
            actions=payload["actions"],
            rewards=payload["rewards"],
            next_states=payload["next_states"],
            dones=payload["dones"],
            _payload=payload,
        )
    finally:
        payload.close()


def save_demo(path: Path, transitions: list[DemoTransition]) -> Path:
    if not transitions:
        raise ValueError("Cannot save an empty demonstration.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        states=np.asarray([item.state for item in transitions], dtype=np.float32),
        actions=np.asarray([item.action for item in transitions], dtype=np.int32),
        rewards=np.asarray([item.reward for item in transitions], dtype=np.float32),
        next_states=np.asarray([item.next_state for item in transitions], dtype=np.float32),
        dones=np.asarray([item.done for item in transitions], dtype=np.float32),
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

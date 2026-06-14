from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ascent_player.config import AppConfig


@dataclass(slots=True)
class DemoTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


def demonstrations_dir(config: AppConfig) -> Path:
    return config.demo.save_dir


def list_demo_files(config: AppConfig) -> list[Path]:
    folder = demonstrations_dir(config)
    if not folder.exists():
        return []
    return sorted(folder.glob("*.npz"))


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
    payload = np.load(path)
    transitions: list[DemoTransition] = []
    for idx in range(len(payload["actions"])):
        transitions.append(
            DemoTransition(
                state=payload["states"][idx],
                action=int(payload["actions"][idx]),
                reward=float(payload["rewards"][idx]),
                next_state=payload["next_states"][idx],
                done=bool(payload["dones"][idx]),
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

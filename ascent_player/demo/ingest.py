from __future__ import annotations

from dataclasses import dataclass
import gc

import numpy as np

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig
from ascent_player.demo.storage import demo_transition_count, list_demo_files, open_demo
from ascent_player.utils.memory import max_demo_transitions, memory_headroom_ok


@dataclass(slots=True)
class DemoIngestResult:
    transitions_added: int
    transitions_skipped: int
    unique_transitions: int
    bc_loss: float | None = None

    @property
    def status_message(self) -> str:
        parts = [f"Loaded {self.transitions_added} demo transitions"]
        if self.bc_loss is not None:
            parts.append(f"BC loss {self.bc_loss:.4f}")
        if self.transitions_skipped:
            parts.append(
                f"skipped {self.transitions_skipped} (memory limit — "
                f"kept {self.unique_transitions})"
            )
        return " | ".join(parts)


def _subsample_indices(total: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    if limit >= total:
        return np.arange(total, dtype=np.int64)
    return np.sort(rng.choice(total, size=limit, replace=False))


def ingest_demonstrations(agent: DQNAgent, config: AppConfig) -> DemoIngestResult:
    paths = list_demo_files(config)
    if not paths:
        return DemoIngestResult(0, 0, 0)

    transition_cap = max_demo_transitions(config)
    total_available = sum(demo_transition_count(path) for path in paths)
    if transition_cap <= 0:
        return DemoIngestResult(0, total_available, 0)

    rng = np.random.default_rng(0)
    unique_loaded = 0
    skipped = 0
    added_total = 0
    multiplier = config.demo.replay_multiplier

    for path in paths:
        file_count = demo_transition_count(path)
        if unique_loaded >= transition_cap or not memory_headroom_ok(config):
            skipped += file_count
            continue

        remaining = transition_cap - unique_loaded
        take = min(file_count, remaining)

        with open_demo(path) as demo:
            indices = _subsample_indices(len(demo), take, rng)
            added = agent.absorb_demonstration_arrays(
                demo.states,
                demo.actions,
                demo.rewards,
                demo.next_states,
                demo.dones,
                multiplier=multiplier,
                indices=indices,
            )
            unique_loaded += len(indices)
            added_total += added
            skipped += file_count - len(indices)

        gc.collect()

    bc_loss = agent.pretrain_from_replay() if unique_loaded else None
    return DemoIngestResult(
        transitions_added=added_total,
        transitions_skipped=skipped,
        unique_transitions=unique_loaded,
        bc_loss=bc_loss,
    )

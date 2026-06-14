from __future__ import annotations

from dataclasses import dataclass
import gc

import numpy as np

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig
from ascent_player.demo.storage import (
    demo_has_score_metadata,
    demo_peak_score,
    demo_transition_count,
    list_demo_files,
    open_demo,
)
from ascent_player.env.state_detector import mask_jump_action
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


def _episode_peak_scores(demo) -> dict[int, float]:
    if demo.episode_ids is None or demo.scores is None:
        return {0: demo.peak_score}
    peaks: dict[int, float] = {}
    for episode_id, score in zip(demo.episode_ids, demo.scores, strict=True):
        episode = int(episode_id)
        peaks[episode] = max(peaks.get(episode, 0.0), float(score))
    return peaks


def _transition_weight(
    demo,
    index: int,
    *,
    min_episode_score: float,
    high_score_weight: float,
) -> float:
    if demo.scores is None:
        return 1.0
    score = float(demo.scores[index])
    if score < min_episode_score:
        return 0.0
    if score >= 1500:
        return high_score_weight
    if score >= 1000:
        return max(1.0, high_score_weight * 0.6)
    return 1.0


def _weighted_subsample(
    demo,
    limit: int,
    rng: np.random.Generator,
    *,
    min_episode_score: float,
    high_score_weight: float,
) -> np.ndarray:
    weights = np.asarray(
        [
            _transition_weight(
                demo,
                idx,
                min_episode_score=min_episode_score,
                high_score_weight=high_score_weight,
            )
            for idx in range(len(demo))
        ],
        dtype=np.float64,
    )
    valid = np.flatnonzero(weights > 0.0)
    if valid.size == 0:
        return np.array([], dtype=np.int64)
    if valid.size <= limit:
        return valid.astype(np.int64)
    probabilities = weights[valid] / weights[valid].sum()
    chosen = rng.choice(valid, size=limit, replace=False, p=probabilities)
    return np.sort(chosen.astype(np.int64))


def _mask_demo_actions(demo, indices: np.ndarray) -> np.ndarray:
    actions = np.asarray(demo.actions[indices], dtype=np.int32).copy()
    if demo.states.ndim != 4 or demo.states.shape[-1] < 5:
        return actions
    for offset, idx in enumerate(indices):
        boost_level = float(np.clip(demo.states[idx, ..., -2].mean(), 0.0, 1.0))
        can_boost = boost_level * 100.0 >= 14.0
        actions[offset] = mask_jump_action(int(actions[offset]), can_boost)
    return actions


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

    ranked_paths = sorted(
        paths,
        key=lambda path: demo_peak_score(path),
        reverse=True,
    )

    for path in ranked_paths:
        has_scores = demo_has_score_metadata(path)
        peak = demo_peak_score(path)
        if has_scores and peak < config.demo.min_episode_score:
            skipped += demo_transition_count(path)
            continue
        min_transition_score = (
            config.demo.min_episode_score if has_scores else 0.0
        )
        file_count = demo_transition_count(path)
        if unique_loaded >= transition_cap or not memory_headroom_ok(config):
            skipped += file_count
            continue

        remaining = transition_cap - unique_loaded
        take = min(file_count, remaining)

        with open_demo(path) as demo:
            indices = _weighted_subsample(
                demo,
                take,
                rng,
                min_episode_score=min_transition_score,
                high_score_weight=config.demo.high_score_weight,
            )
            if len(indices) == 0:
                skipped += file_count
                continue
            masked_actions = _mask_demo_actions(demo, indices)
            added = agent.absorb_demonstration_arrays(
                demo.states,
                masked_actions,
                demo.rewards,
                demo.next_states,
                demo.dones,
                multiplier=multiplier,
                indices=indices,
                target_buffer=agent.demo_replay,
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

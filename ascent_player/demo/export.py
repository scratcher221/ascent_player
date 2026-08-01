from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ascent_player.agent.replay_buffer import ReplayBuffer
from ascent_player.config import AppConfig
from ascent_player.demo.storage import open_demo


@dataclass(slots=True)
class ReplayExportResult:
    replay_path: Path
    demos_seen: int
    demos_loaded: int
    transitions_added: int
    transitions_skipped: int
    skill_labeled: int
    vector_ready: int

    @property
    def status_message(self) -> str:
        return (
            f"Exported {self.transitions_added} transitions from {self.demos_loaded}/"
            f"{self.demos_seen} demos to {self.replay_path} "
            f"(skill_labeled={self.skill_labeled}, vector_ready={self.vector_ready}, "
            f"skipped={self.transitions_skipped})"
        )


def export_demos_to_replay(
    config: AppConfig,
    *,
    demo_paths: list[Path],
    replay_path: Path,
    append: bool = False,
    max_items: int | None = None,
) -> ReplayExportResult:
    capacity = max(
        int(config.training.browser_replay_max_items),
        int(max_items or 0),
        1,
    )
    replay = ReplayBuffer(capacity)
    if append and replay_path.exists():
        replay.load_pickle(
            replay_path,
            max_items=max_items or config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )

    demos_loaded = 0
    transitions_added = 0
    transitions_skipped = 0
    skill_labeled = 0
    vector_ready = 0

    for path in demo_paths:
        with open_demo(path) as demo:
            if demo.state_vectors is None or demo.next_state_vectors is None:
                transitions_skipped += len(demo)
                continue
            demos_loaded += 1
            for idx in range(len(demo)):
                state = (
                    demo.states[idx],
                    demo.state_vectors[idx],
                )
                next_state = (
                    demo.next_states[idx],
                    demo.next_state_vectors[idx],
                )
                skill = int(demo.skills[idx]) if demo.skills is not None else -1
                score = float(demo.scores[idx]) if demo.scores is not None else -1.0
                episode_id = int(demo.episode_ids[idx]) if demo.episode_ids is not None else -1
                replay.add(
                    state,
                    int(demo.actions[idx]),
                    float(demo.rewards[idx]),
                    next_state,
                    bool(demo.dones[idx]),
                    discount=config.training.gamma,
                    skill=skill,
                    episode_score=score,
                    episode_id=episode_id,
                )
                transitions_added += 1
                vector_ready += 1
                if skill >= 0:
                    skill_labeled += 1

    replay.save_pickle(
        replay_path,
        max_items=max_items or config.training.browser_replay_max_items,
    )
    return ReplayExportResult(
        replay_path=replay_path,
        demos_seen=len(demo_paths),
        demos_loaded=demos_loaded,
        transitions_added=transitions_added,
        transitions_skipped=transitions_skipped,
        skill_labeled=skill_labeled,
        vector_ready=vector_ready,
    )

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass(slots=True)
class TrainingProgress:
    episodes_completed: int = 0
    total_steps: int = 0
    epsilon: float = 1.0
    baseline_episodes: int = 5
    baseline_reward: float | None = None
    baseline_score: float | None = None
    best_reward: float = float("-inf")
    best_score: float = 0.0
    recent_rewards: list[float] = field(default_factory=list)
    recent_scores: list[float] = field(default_factory=list)
    last_saved_at: str | None = None
    started_fresh: bool = True

    @property
    def has_baseline(self) -> bool:
        return self.baseline_reward is not None

    @property
    def recent_avg_reward(self) -> float | None:
        if not self.recent_rewards:
            return None
        return float(sum(self.recent_rewards) / len(self.recent_rewards))

    @property
    def recent_avg_score(self) -> float | None:
        if not self.recent_scores:
            return None
        return float(sum(self.recent_scores) / len(self.recent_scores))

    def reward_vs_baseline_pct(self) -> float | None:
        if self.baseline_reward is None or self.baseline_reward == 0:
            return None
        current = self.recent_avg_reward
        if current is None:
            return None
        return ((current - self.baseline_reward) / abs(self.baseline_reward)) * 100.0


@dataclass(slots=True)
class LoadResult:
    loaded: bool
    message: str
    progress: TrainingProgress | None = None


def meta_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.meta.json")


def save_progress(checkpoint_path: Path, progress: TrainingProgress) -> None:
    progress.last_saved_at = datetime.now(UTC).isoformat()
    meta_path(checkpoint_path).write_text(
        json.dumps(asdict(progress), indent=2),
        encoding="utf-8",
    )


def load_progress(checkpoint_path: Path) -> TrainingProgress | None:
    path = meta_path(checkpoint_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return TrainingProgress(
            episodes_completed=int(payload.get("episodes_completed", 0)),
            total_steps=int(payload.get("total_steps", 0)),
            epsilon=float(payload.get("epsilon", 1.0)),
            baseline_episodes=int(payload.get("baseline_episodes", 5)),
            baseline_reward=payload.get("baseline_reward"),
            baseline_score=payload.get("baseline_score"),
            best_reward=float(payload.get("best_reward", float("-inf"))),
            best_score=float(payload.get("best_score", 0.0)),
            recent_rewards=[float(v) for v in payload.get("recent_rewards", [])],
            recent_scores=[float(v) for v in payload.get("recent_scores", [])],
            last_saved_at=payload.get("last_saved_at"),
            started_fresh=bool(payload.get("started_fresh", True)),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None

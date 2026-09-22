"""Elite high-score replay store (gate, compact, persist)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ascent_player.config import AppConfig

DEFAULT_REPLAY = Path("checkpoints/elite_thread_replay.pkl")
DEFAULT_META = Path("checkpoints/elite_thread_replay.json")
DEFAULT_BROWSER = Path("checkpoints/browser_replay.pkl")
DEFAULT_MIN_SCORE = 1400.0
DEFAULT_MAX_EPISODES = 64
DEFAULT_MAX_PER_EPISODE = 128


@dataclass
class EliteReplayStore:
    replay_path: Path = DEFAULT_REPLAY
    meta_path: Path = DEFAULT_META
    browser_path: Path = DEFAULT_BROWSER
    min_score: float = DEFAULT_MIN_SCORE
    max_episodes: int = DEFAULT_MAX_EPISODES
    max_per_episode: int = DEFAULT_MAX_PER_EPISODE

    def selected_episodes(self) -> int:
        if not self.meta_path.exists():
            return 0
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return 0
        try:
            return int(meta.get("selected_episodes") or 0)
        except (TypeError, ValueError):
            return 0

    def compatible(self) -> bool:
        if not self.replay_path.exists() or not self.meta_path.exists():
            return False
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False
        return float(meta.get("min_episode_score", 0.0)) >= float(self.min_score)

    def reset_if_incompatible(self) -> None:
        if not self.replay_path.exists() or self.compatible():
            return
        existing = -1.0
        if self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                existing = float(meta.get("min_episode_score", 0.0))
            except (OSError, ValueError, TypeError):
                existing = -1.0
        # Raising the harvest gate must not delete a lower-gated tail. Persist
        # still merges and compact prefers high scores.
        if existing >= 0.0:
            print(
                f"ELITE_GATE_KEEP existing>={existing:.0f} "
                f"requested>={self.min_score:.0f} — not wiping",
                flush=True,
            )
            return
        size = self.replay_path.stat().st_size
        self.replay_path.unlink()
        self.meta_path.unlink(missing_ok=True)
        print(
            f"ELITE_REPLAY_REBUILD removed_incompatible_bytes={size} "
            f"required_score>={self.min_score:.0f}",
            flush=True,
        )

    def persist_from_browser(self, config: AppConfig) -> int:
        """Merge collect replay, keep only ≥min_score episodes, then compact."""
        from ascent_player.agent.dqn import DQNAgent
        self.reset_if_incompatible()
        if not self.browser_path.exists():
            if not self.replay_path.exists():
                return 0
            probe = DQNAgent(config)
            probe.replay.clear()
            return int(
                probe.replay.load_pickle(
                    self.replay_path,
                    max_items=config.training.browser_replay_max_items,
                    vector_dim=config.observation.vector_dim,
                )
            )
        self.replay_path.parent.mkdir(parents=True, exist_ok=True)
        merged = DQNAgent(config)
        merged.replay.clear()
        before = 0
        if self.replay_path.exists():
            before = merged.replay.load_pickle(
                self.replay_path,
                max_items=config.training.browser_replay_max_items,
                vector_dim=config.observation.vector_dim,
            )
        fresh = DQNAgent(config)
        fresh.replay.clear()
        browser_n = fresh.replay.load_pickle(
            self.browser_path,
            max_items=config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )
        if browser_n > 0:
            kept = fresh.replay.filter_min_episode_score(self.min_score)
            print(
                f"ELITE_FILTER browser={browser_n} kept>={self.min_score:.0f} -> {kept}",
                flush=True,
            )
            if kept > 0:
                merged.replay.extend_from(fresh.replay)
        compact = merged.replay.compact_diverse_episodes(
            max_episodes=self.max_episodes,
            max_transitions_per_episode=self.max_per_episode,
        )
        saved = merged.replay.save_pickle(
            self.replay_path,
            max_items=self.max_episodes * self.max_per_episode,
        )
        self.meta_path.write_text(
            json.dumps(
                {
                    "min_episode_score": self.min_score,
                    "max_episodes": self.max_episodes,
                    "max_transitions_per_episode": self.max_per_episode,
                    **compact,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"ELITE_REPLAY merge before={before} browser={browser_n} "
            f"episodes={compact['selected_episodes']} saved={saved} "
            f"gate>={self.min_score:.0f} -> {self.replay_path.name}",
            flush=True,
        )
        return int(saved)

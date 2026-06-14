from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from ascent_player.config import AppConfig
from ascent_player.training import (
    run_sim_calibration,
    run_sim_pretrain,
    run_training_no_ui,
)

EPISODE_SCORE_RE = re.compile(r"score=([0-9.]+)")


def parse_episode_scores(log_path: Path, last_n: int = 10) -> list[float]:
    if not log_path.exists():
        return []
    scores: list[float] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "EPISODE END" not in line:
            continue
        match = EPISODE_SCORE_RE.search(line)
        if match:
            scores.append(float(match.group(1)))
    return scores[-last_n:]


def meets_target(stats: dict[str, float], target: int) -> bool:
    recent = stats.get("recent_avg", 0.0)
    recent_min = stats.get("recent_min", 0.0)
    return recent >= target and recent_min >= target * 0.85


async def run_pipeline(
    config: AppConfig,
    *,
    sim_steps: int | None = None,
    finetune_seconds: int | None = None,
    max_cycles: int = 8,
) -> dict[str, float]:
    target = config.training.target_score
    finetune = finetune_seconds or config.training.finetune_max_seconds
    steps = sim_steps or config.training.sim_pretrain_steps or 300_000
    best_stats: dict[str, float] = {"best_score": 0.0, "recent_avg": 0.0}

    print("=== Phase A: sim calibration ===")
    await run_sim_calibration(config, episodes=12)

    for cycle in range(1, max_cycles + 1):
        print(f"\n=== Cycle {cycle}/{max_cycles}: sim pretrain ({steps} steps) ===")
        config.training.sim_mode = True
        run_sim_pretrain(config, steps)

        print(f"\n=== Cycle {cycle}: browser fine-tune ({finetune}s) ===")
        config.training.sim_mode = False
        config.training.transfer_from_sim = True
        config.training.frame_skip = config.training.transfer_frame_skip
        stats = await run_training_no_ui(config, max_seconds=finetune)
        best_stats = stats
        print(
            f"cycle={cycle} best={stats['best_score']:.0f} "
            f"recent_avg={stats['recent_avg']:.0f} "
            f"recent_min={stats['recent_min']:.0f} "
            f"recent_max={stats['recent_max']:.0f}"
        )
        if meets_target(stats, target):
            print(f"Target met: recent_avg >= {target}")
            return stats

        log_scores = parse_episode_scores(Path(stats["log_path"]))
        if log_scores and max(log_scores) < 800:
            steps = min(int(steps * 1.5), 1_000_000)
            print(f"Low browser scores — extending sim pretrain to {steps} steps")
        elif stats["best_score"] < 1500:
            config.training.transfer_epsilon_start = min(
                0.7,
                config.training.transfer_epsilon_start + 0.05,
            )
            print(
                f"Bumping transfer epsilon to {config.training.transfer_epsilon_start:.2f}"
            )

    print(f"Pipeline finished without consistent {target}+ (best recent_avg={best_stats['recent_avg']:.0f})")
    return best_stats


def main() -> int:
    from ascent_player.utils.gpu_env import bootstrap_gpu_environment

    bootstrap_gpu_environment()
    from ascent_player.config import AppConfig

    config = AppConfig()
    config.training.sim_pretrain_steps = 300_000
    config.training.transfer_from_sim = True
    asyncio.run(run_pipeline(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

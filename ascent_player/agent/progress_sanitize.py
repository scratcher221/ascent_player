from __future__ import annotations

from pathlib import Path

from ascent_player.agent.checkpoint import TrainingProgress, load_progress, save_progress

# Browser scores above this without recent confirmation are treated as corrupt.
BROWSER_SCORE_SANITY_CAP = 5000.0
BROWSER_EPSILON_CAP = 0.35


def sanitize_browser_progress(
    progress: TrainingProgress,
    *,
    score_cap: float = BROWSER_SCORE_SANITY_CAP,
    epsilon_cap: float = BROWSER_EPSILON_CAP,
) -> tuple[TrainingProgress, list[str]]:
    """Reset stale sim-era metadata so browser training metrics stay honest."""
    notes: list[str] = []
    recent_max = max(progress.recent_scores) if progress.recent_scores else 0.0

    if progress.best_score > score_cap and recent_max < score_cap * 0.4:
        notes.append(
            f"best_score {progress.best_score:.0f}→{recent_max:.0f} (sanity cap)"
        )
        progress.best_score = max(recent_max, 0.0)

    if progress.baseline_score is not None and progress.baseline_score > score_cap:
        if recent_max < 2000:
            notes.append(
                f"baseline_score {progress.baseline_score:.0f} cleared (sim artifact)"
            )
            progress.baseline_score = None
            progress.baseline_reward = None

    if progress.epsilon > epsilon_cap:
        notes.append(f"epsilon {progress.epsilon:.3f}→{epsilon_cap:.2f} (browser cap)")
        progress.epsilon = epsilon_cap

    return progress, notes


def patch_checkpoint_epsilon(
    checkpoint_path: Path,
    *,
    multiply: float,
    cap: float = BROWSER_EPSILON_CAP,
    floor: float = 0.05,
) -> float | None:
    """Decay exploration stored in checkpoint metadata (used between overnight runs)."""
    progress = load_progress(checkpoint_path)
    if progress is None:
        return None
    progress.epsilon = max(floor, min(cap, progress.epsilon * multiply))
    save_progress(checkpoint_path, progress)
    return progress.epsilon

"""Deployable score floors for thread-BC vs seed-map-best promotion."""
from __future__ import annotations

from pathlib import Path

SEED_MEAN = Path("logs/seed_map_best_mean.txt")
THREAD_BC_BEST = Path("logs/thread_bc_best_mean.txt")


def read_seed_floor(default: float = 1250.7) -> float:
    if not SEED_MEAN.exists():
        return float(default)
    try:
        return float(SEED_MEAN.read_text(encoding="utf-8").strip())
    except ValueError:
        return float(default)


def read_thread_bc_best(default: float = 0.0) -> float:
    if not THREAD_BC_BEST.exists():
        return float(default)
    try:
        return float(THREAD_BC_BEST.read_text(encoding="utf-8").strip())
    except ValueError:
        return float(default)


def write_thread_bc_best(mean: float) -> bool:
    """Persist if this is a new best for thread_bc deploy tracking."""
    mean = float(mean)
    current = read_thread_bc_best(default=0.0)
    if mean <= current:
        return False
    THREAD_BC_BEST.parent.mkdir(parents=True, exist_ok=True)
    THREAD_BC_BEST.write_text(f"{mean:.4f}\n", encoding="utf-8")
    return True


def thread_bc_promotion_floor() -> float:
    """Greedy confirm must beat this floor during thread-BC cycles."""
    return read_thread_bc_best(default=0.0)


def record_thread_bc_eval(mean: float) -> bool:
    return write_thread_bc_best(mean)


def apply_reliability_baseline(mean: float) -> bool:
    """Update thread_bc best from a full aggregated reliability run."""
    return write_thread_bc_best(mean)

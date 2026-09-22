"""Deployable score floors for thread-BC, seed-map-best, and v2 climb."""
from __future__ import annotations

from pathlib import Path

SEED_MEAN = Path("logs/seed_map_best_mean.txt")
THREAD_BC_BEST = Path("logs/thread_bc_best_mean.txt")
V2_PROBE_BEST = Path("logs/v2_probe_best_mean.txt")
V2_FULL_N_BEST = Path("logs/v2_full_n_best_mean.txt")
V2_FULL_N_LAST = Path("logs/v2_full_n_last_mean.txt")


def _read_float(path: Path, default: float) -> float:
    if not path.exists():
        return float(default)
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return float(default)


def _write_float(path: Path, mean: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{float(mean):.4f}\n", encoding="utf-8")


def read_seed_floor(default: float = 1250.7) -> float:
    return _read_float(SEED_MEAN, default)


def write_seed_floor(mean: float) -> None:
    """Unconditionally persist seed-map best mean (promotion path)."""
    _write_float(SEED_MEAN, mean)


def read_thread_bc_best(default: float = 0.0) -> float:
    return _read_float(THREAD_BC_BEST, default)


def write_thread_bc_best(mean: float) -> bool:
    """Persist if this is a new best for thread_bc deploy tracking."""
    mean = float(mean)
    current = read_thread_bc_best(default=0.0)
    if mean <= current:
        return False
    _write_float(THREAD_BC_BEST, mean)
    return True


def read_v2_probe_best(default: float = 0.0) -> float:
    return _read_float(V2_PROBE_BEST, default)


def write_v2_probe_best(mean: float) -> None:
    """Persist Impala climb greedy floor (caller decides when to raise)."""
    _write_float(V2_PROBE_BEST, mean)


def read_v2_full_n_best(default: float = 0.0) -> float:
    return _read_float(V2_FULL_N_BEST, default)


def read_v2_full_n_last(default: float = 0.0) -> float:
    return _read_float(V2_FULL_N_LAST, default)


def write_v2_full_n_last(mean: float) -> None:
    """Last full-N greedy mean (may fall). Used for TD / collect gates."""
    _write_float(V2_FULL_N_LAST, mean)


def write_v2_full_n_best(mean: float) -> bool:
    """Raise-only full-N floor. Returns True when the file changed."""
    mean = float(mean)
    current = read_v2_full_n_best(default=0.0)
    if mean <= current:
        return False
    _write_float(V2_FULL_N_BEST, mean)
    return True


def seed_v2_full_n_best(mean: float) -> None:
    """First baseline: set even when below the stale last-10 probe floor."""
    if read_v2_full_n_best(default=0.0) <= 0:
        _write_float(V2_FULL_N_BEST, float(mean))
    write_v2_full_n_last(mean)


def live_v2_floor() -> float:
    """Promote / TD bar: full-N best if seeded, else 0 (do not use last-10)."""
    return read_v2_full_n_best(default=0.0)


def thread_bc_promotion_floor() -> float:
    """Greedy confirm must beat this floor during thread-BC cycles."""
    return read_thread_bc_best(default=0.0)


def record_thread_bc_eval(mean: float) -> bool:
    return write_thread_bc_best(mean)


def apply_reliability_baseline(mean: float) -> bool:
    """Update thread_bc best from a full aggregated reliability run."""
    return write_thread_bc_best(mean)

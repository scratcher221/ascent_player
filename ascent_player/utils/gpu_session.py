from __future__ import annotations

import gc


def release_gpu_between_runs() -> None:
    """Lightweight cleanup between training sessions (full reset via subprocess)."""
    gc.collect()

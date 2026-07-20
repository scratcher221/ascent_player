"""Append ε=0 eval rows for cross-run skill tracking."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


LEDGER_FIELDS = (
    "timestamp",
    "mode",
    "mean",
    "min",
    "max",
    "steps",
    "checkpoint",
)


def append_skill_ledger(
    path: Path,
    *,
    mode: str,
    mean: float,
    min_score: float,
    max_score: float,
    steps: int = 0,
    checkpoint: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mode": mode,
                "mean": f"{float(mean):.2f}",
                "min": f"{float(min_score):.2f}",
                "max": f"{float(max_score):.2f}",
                "steps": int(steps),
                "checkpoint": checkpoint,
            }
        )

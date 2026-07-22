#!/usr/bin/env python3
"""Summarize a decisions CSV from fixed-seed / browser training."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def _resolve_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    latest = Path("logs/decisions_latest.path")
    if latest.exists():
        return Path(latest.read_text(encoding="utf-8").strip())
    candidates = sorted(Path("logs").glob("decisions_*.csv"))
    if not candidates:
        raise SystemExit("No decisions CSV found under logs/")
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize action-reason decision log")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="CSV path (default: logs/decisions_latest.path)",
    )
    args = parser.parse_args()
    path = _resolve_path(args.path)
    if not path.exists():
        raise SystemExit(f"Missing {path}")

    reasons: Counter[str] = Counter()
    preds: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    falling_reasons: Counter[str] = Counter()
    n = 0
    agree = 0
    wrong = 0
    falling_n = 0
    falling_wrong = 0

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n += 1
            reason = row.get("reason") or ""
            pred = row.get("reason_pred") or ""
            reasons[reason] += 1
            preds[pred] += 1
            sources[row.get("source") or ""] += 1
            if row.get("agree") == "1":
                agree += 1
            if row.get("wrong_vs_target") == "1":
                wrong += 1
            # Heuristic: teacher toward_platform_below ≈ falling survival steer
            if reason == "toward_platform_below":
                falling_n += 1
                falling_reasons[reason] += 1
                if row.get("wrong_vs_target") == "1":
                    falling_wrong += 1
            elif row.get("plat_dx") and reason:
                # Count other reasons while plat_dx present as possible falling context
                pass

    print(f"file={path}")
    print(f"rows={n}")
    if n == 0:
        return 0
    print(f"agree_rate={agree / n:.3f} ({agree}/{n})")
    print(f"wrong_vs_target_rate={wrong / n:.3f} ({wrong}/{n})")
    print("reasons:")
    for name, count in reasons.most_common():
        print(f"  {name}: {count} ({count / n:.1%})")
    print("reason_pred:")
    for name, count in preds.most_common(8):
        print(f"  {name}: {count} ({count / n:.1%})")
    print("sources:")
    for name, count in sources.most_common():
        print(f"  {name}: {count} ({count / n:.1%})")
    below = reasons.get("toward_platform_below", 0)
    print(f"toward_platform_below_share={below / n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

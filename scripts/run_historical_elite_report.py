#!/usr/bin/env python3
"""Summarize historical >=threshold browser episodes and whether they are reusable."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _scan_decision_file(path: Path, min_score: float) -> list[dict[str, object]]:
    episodes: dict[str, float] = {}
    steps: dict[str, int] = defaultdict(int)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            episode = row.get("episode", "")
            score_raw = row.get("score", "")
            if not episode:
                continue
            steps[episode] += 1
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                continue
            episodes[episode] = max(episodes.get(episode, float("-inf")), score)
    rows = []
    for episode, max_score in episodes.items():
        if max_score < min_score:
            continue
        rows.append(
            {
                "file": str(path),
                "episode": int(episode),
                "max_score": float(max_score),
                "steps": int(steps[episode]),
                "importable": False,
                "reason": "decision logs do not retain state tensors / next states",
            }
        )
    rows.sort(key=lambda item: (item["max_score"], item["steps"]), reverse=True)
    return rows


def _scan_replay_pickle(path: Path, min_score: float) -> list[dict[str, object]]:
    if not path.exists():
        return []
    import pickle

    with path.open("rb") as handle:
        items = pickle.load(handle)
    grouped: dict[int, dict[str, float]] = {}
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) < 9:
            continue
        episode_score = float(item[8])
        episode_id = int(item[9]) if len(item) >= 10 else -1
        if episode_score < min_score or episode_id < 0:
            continue
        info = grouped.setdefault(episode_id, {"score": episode_score, "steps": 0.0})
        info["score"] = max(info["score"], episode_score)
        info["steps"] += 1.0
    rows = []
    for episode_id, info in sorted(grouped.items(), key=lambda kv: kv[1]["score"], reverse=True):
        rows.append(
            {
                "file": str(path),
                "episode": int(episode_id),
                "max_score": float(info["score"]),
                "steps": int(info["steps"]),
                "importable": True,
                "reason": "replay contains stamped episode_score and episode_id",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Report historical >=threshold elite episodes.")
    parser.add_argument("--min-score", type=float, default=2000.0)
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--replay", type=Path, default=ROOT / "checkpoints" / "browser_replay.pkl")
    parser.add_argument("--elite", type=Path, default=ROOT / "checkpoints" / "elite_thread_replay.pkl")
    parser.add_argument("--out", type=Path, default=ROOT / "logs" / "historical_elite_report.json")
    args = parser.parse_args()

    decision_candidates: list[dict[str, object]] = []
    for path in sorted(args.logs_dir.glob("decisions_*.csv")):
        decision_candidates.extend(_scan_decision_file(path, args.min_score))
    replay_candidates = _scan_replay_pickle(args.replay, args.min_score)
    elite_candidates = _scan_replay_pickle(args.elite, args.min_score)

    report = {
        "min_score": float(args.min_score),
        "decision_candidates": decision_candidates[:200],
        "decision_candidate_count": len(decision_candidates),
        "importable_from_browser_replay": replay_candidates,
        "importable_from_elite_replay": elite_candidates,
        "summary": {
            "recoverable_episode_count": len(replay_candidates) + len(elite_candidates),
            "decision_only_unrecoverable_count": len(decision_candidates),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"HISTORICAL_ELITE_REPORT min_score={args.min_score:.0f} "
        f"decision_only={len(decision_candidates)} "
        f"recoverable={len(replay_candidates) + len(elite_candidates)} "
        f"-> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

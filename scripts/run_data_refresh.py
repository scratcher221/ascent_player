#!/usr/bin/env python3
"""Rebuild hybrid BC data: human demos + teacher distill + diverse elite.

Pillar 3 / Step 4 of the autonomous overhaul:
  1. Export seeded_human demos (59-d vectors) → hybrid_human_replay.pkl
  2. Distill teacher labels from elite (or short browser teacher collect)
  3. Re-compact elite_thread_replay.pkl with score + action diversity (cap 64)

Usage:
  PYTHONPATH=. python -u scripts/run_data_refresh.py
  PYTHONPATH=. python -u scripts/run_data_refresh.py --teacher-episodes 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.demo.export import export_demos_to_replay
from ascent_player.training import run_training_no_ui
from ascent_player.utils.climb_policy import human_cap
from ascent_player.utils.run_profile import (
    collect_only_fields,
    override_attrs,
    override_training,
)

ELITE_REPLAY = Path("checkpoints/elite_thread_replay.pkl")
ELITE_META = Path("checkpoints/elite_thread_replay.json")
HUMAN_REPLAY = Path("checkpoints/hybrid_human_replay.pkl")
TEACHER_REPLAY = Path("checkpoints/teacher_distill_replay.pkl")
HYBRID_BC = Path("checkpoints/hybrid_bc_mix.pkl")
HUMAN_DIR = Path("demonstrations/seeded_human")
DEFAULT_ELITE_GATE = 2000.0
DEFAULT_MAX_EPISODES = 64
DEFAULT_MAX_PER = 128


def _build_config(run_seed: int) -> AppConfig:
    config = AppConfig()
    config.training.sim_mode = False
    config.training.device_mode = DeviceMode.AUTO
    config.training.model_variant = "impala_mid"
    config.training.mixed_precision = True
    config.training.frame_skip = 1
    config.training.transfer_frame_skip = 1
    config.browser.run_seed = int(run_seed)
    config.browser.lock_run_seed = True
    return config


def export_human_hybrid(config: AppConfig) -> dict[str, int | str]:
    demos = sorted(HUMAN_DIR.glob("demo_*.npz"))
    if not demos:
        print(f"HUMAN_EXPORT_SKIP no demos in {HUMAN_DIR}", flush=True)
        return {"demos": 0, "transitions": 0}
    result = export_demos_to_replay(
        config,
        demo_paths=demos,
        replay_path=HUMAN_REPLAY,
        append=False,
        max_items=config.training.browser_replay_max_items,
    )
    print(result.status_message, flush=True)
    return {
        "demos": result.demos_loaded,
        "transitions": result.transitions_added,
        "vector_ready": result.vector_ready,
        "path": str(HUMAN_REPLAY),
    }


def rebuild_elite(
    config: AppConfig,
    *,
    min_score: float,
    max_episodes: int,
    max_per: int,
) -> dict[str, int]:
    if not ELITE_REPLAY.exists():
        print(f"ELITE_REBUILD_SKIP missing {ELITE_REPLAY}", flush=True)
        return {"saved": 0}
    backup = ELITE_REPLAY.with_suffix(".pkl.bak_diversity")
    if not backup.exists():
        shutil.copy2(ELITE_REPLAY, backup)
        print(f"ELITE_BACKUP {backup}", flush=True)

    agent = DQNAgent(config)
    agent.replay.clear()
    loaded = agent.replay.load_pickle(
        ELITE_REPLAY,
        max_items=config.training.browser_replay_max_items,
        vector_dim=config.observation.vector_dim,
    )
    kept = agent.replay.filter_min_episode_score(min_score)
    compact = agent.replay.compact_diverse_episodes(
        max_episodes=max_episodes,
        max_transitions_per_episode=max_per,
    )
    saved = agent.replay.save_pickle(
        ELITE_REPLAY,
        max_items=max_episodes * max_per,
    )
    meta = {
        "min_episode_score": min_score,
        "max_episodes": max_episodes,
        "max_transitions_per_episode": max_per,
        "loaded": loaded,
        "kept_after_gate": kept,
        **compact,
        "saved": saved,
        "refreshed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    ELITE_META.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"ELITE_REBUILD loaded={loaded} kept>={min_score:.0f}->{kept} "
        f"episodes={compact['selected_episodes']} saved={saved}",
        flush=True,
    )
    return meta


def distill_teacher_from_elite(config: AppConfig) -> dict[str, int | str]:
    """Copy elite transitions as teacher-labeled BC targets (already high-prior)."""
    if not ELITE_REPLAY.exists():
        print("TEACHER_DISTILL_SKIP no elite store", flush=True)
        return {"saved": 0}
    agent = DQNAgent(config)
    agent.replay.clear()
    n = agent.replay.load_pickle(
        ELITE_REPLAY,
        max_items=config.training.browser_replay_max_items,
        vector_dim=config.observation.vector_dim,
    )
    saved = agent.replay.save_pickle(
        TEACHER_REPLAY,
        max_items=config.training.browser_replay_max_items,
    )
    print(
        f"TEACHER_DISTILL from_elite loaded={n} saved={saved} -> {TEACHER_REPLAY}",
        flush=True,
    )
    return {"source": "elite", "loaded": n, "saved": saved, "path": str(TEACHER_REPLAY)}


async def distill_teacher_browser(
    config: AppConfig,
    *,
    episodes_budget_seconds: int,
    thread_prior: float,
) -> dict[str, int | str]:
    """Short teacher-only collect (prior high, no TD) into teacher_distill_replay.pkl."""
    agent = DQNAgent(config)
    agent.epsilon = 0.02
    agent.save(Path("checkpoints/dqn_latest.keras"))
    print(
        f"TEACHER_COLLECT s={episodes_budget_seconds} prior={thread_prior:.2f}",
        flush=True,
    )
    with override_training(
        config,
        **collect_only_fields(
            eps=0.02,
            skip_replay_load=True,
            thread_prior=thread_prior,
            rule_prior=0.05,
        ),
        replay_min_episode_score=800.0,
        skill_exec_at_watch=False,
        seed_thread_watch_prior=0.0,
        browser_epsilon_floor=0.02,
    ), override_attrs(config.demo, use_demos_on_start=False):
        await run_training_no_ui(
            config,
            max_seconds=episodes_budget_seconds,
            ingest_demos=False,
        )
    browser = Path("checkpoints/browser_replay.pkl")
    if not browser.exists():
        print("TEACHER_DISTILL_SKIP browser_replay missing after collect", flush=True)
        return {"saved": 0}
    agent.replay.clear()
    n = agent.replay.load_pickle(
        browser,
        max_items=config.training.browser_replay_max_items,
        vector_dim=config.observation.vector_dim,
    )
    kept = agent.replay.filter_min_episode_score(800.0)
    compact = agent.replay.compact_diverse_episodes(
        max_episodes=100,
        max_transitions_per_episode=128,
    )
    saved = agent.replay.save_pickle(
        TEACHER_REPLAY,
        max_items=config.training.browser_replay_max_items,
    )
    print(
        f"TEACHER_DISTILL browser loaded={n} kept={kept} "
        f"episodes={compact['selected_episodes']} saved={saved}",
        flush=True,
    )
    return {
        "source": "browser",
        "loaded": n,
        "kept": kept,
        "saved": saved,
        "path": str(TEACHER_REPLAY),
    }


def merge_hybrid_bc_mix(
    config: AppConfig,
    *,
    human_share_cap: float = 0.25,
) -> dict[str, int | str]:
    """Combine elite + human + teacher into one offline BC pickle.

    Prefer elite/teacher quality: score-gate, diversity-compact, then append a
    capped human sample so the pickle cap does not drop climbs.
    """
    max_items = max(20_000, int(config.training.browser_replay_max_items))
    agent = DQNAgent(config)
    agent.replay.clear()
    parts: dict[str, int] = {"elite": 0, "human": 0, "teacher": 0}
    human_share_cap = min(1.0, max(0.0, float(human_share_cap)))

    for label, path in (("elite", ELITE_REPLAY), ("teacher", TEACHER_REPLAY)):
        if not path.exists():
            continue
        aux = DQNAgent(config)
        aux.replay.clear()
        n = aux.replay.load_pickle(
            path,
            max_items=max_items,
            vector_dim=config.observation.vector_dim,
        )
        if n > 0:
            agent.replay.extend_from(aux.replay)
            parts[label] = int(n)

    # Compact privileged teacher/elite first so human cannot evict them.
    if len(agent.replay) > 0:
        agent.replay.compact_diverse_episodes(
            max_episodes=64,
            max_transitions_per_episode=128,
        )
    privileged = len(agent.replay)

    if HUMAN_REPLAY.exists():
        aux = DQNAgent(config)
        aux.replay.clear()
        n = aux.replay.load_pickle(
            HUMAN_REPLAY,
            max_items=max_items,
            vector_dim=config.observation.vector_dim,
        )
        room = max(0, max_items - privileged)
        human_cap_n = human_cap(room, max_items, human_share_cap)
        if n > 0 and human_cap_n > 0:
            if n > human_cap_n:
                aux.replay.trim_to(human_cap_n)
            agent.replay.extend_from(aux.replay, max_items=human_cap_n)
            parts["human"] = min(int(n), human_cap_n)
        else:
            parts["human"] = 0
    saved = agent.replay.save_pickle(HYBRID_BC, max_items=max_items)
    print(
        f"HYBRID_BC_MIX elite={parts['elite']} human={parts['human']} "
        f"teacher={parts['teacher']} privileged={privileged} "
        f"human_share_cap={human_share_cap:.2f} saved={saved} -> {HYBRID_BC}",
        flush=True,
    )
    return {
        **parts,
        "saved": saved,
        "path": str(HYBRID_BC),
        "human_share_cap": human_share_cap,
    }


async def main_async(args: argparse.Namespace) -> int:
    config = _build_config(int(args.run_seed))
    summary: dict[str, object] = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    summary["human"] = export_human_hybrid(config)
    summary["elite"] = rebuild_elite(
        config,
        min_score=float(args.elite_gate),
        max_episodes=int(args.elite_max_episodes),
        max_per=int(args.max_transitions_per_episode),
    )

    if int(args.teacher_seconds) > 0:
        summary["teacher"] = await distill_teacher_browser(
            config,
            episodes_budget_seconds=int(args.teacher_seconds),
            thread_prior=float(args.teacher_prior),
        )
    else:
        summary["teacher"] = distill_teacher_from_elite(config)

    summary["hybrid_mix"] = merge_hybrid_bc_mix(
        config, human_share_cap=float(args.human_share_cap)
    )
    out = Path("logs/data_refresh_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"DATA_REFRESH_DONE {out}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid data refresh for v2 BC")
    parser.add_argument("--run-seed", type=int, default=424242)
    parser.add_argument("--elite-gate", type=float, default=DEFAULT_ELITE_GATE)
    parser.add_argument("--elite-max-episodes", type=int, default=DEFAULT_MAX_EPISODES)
    parser.add_argument(
        "--max-transitions-per-episode",
        type=int,
        default=DEFAULT_MAX_PER,
    )
    parser.add_argument(
        "--teacher-seconds",
        type=int,
        default=0,
        help="If >0, run browser teacher collect for this many seconds; "
        "else copy elite as distill labels.",
    )
    parser.add_argument("--teacher-prior", type=float, default=0.90)
    parser.add_argument(
        "--human-share-cap",
        type=float,
        default=0.25,
        help="Max fraction of hybrid_bc_mix reserved for human demos (elite/teacher first).",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

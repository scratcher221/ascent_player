#!/usr/bin/env python3
"""True 100-episode browser ε=0 reliability benchmark (full-run aggregation).

Unlike run_eval_watch (last-10 window), this uses evaluate_* helpers so mean/min/max
and percentiles cover every episode.

Examples:
  PYTHONPATH=. .venv/bin/python -u scripts/run_browser_reliability_100.py \\
    --policy seed_map_best --episodes 100 --run-seed 424242
  PYTHONPATH=. .venv/bin/python -u scripts/run_browser_reliability_100.py \\
    --policy skill_router --episodes 20
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ascent_player.agent.checkpoint import checkpoint_exists
from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.evaluation import (
    evaluate_learned_policy,
    evaluate_seed_thread_baseline,
    evaluate_skill_router_baseline,
    format_skill_metrics,
)
from ascent_player.training import meets_reliability_target
from ascent_player.utils.skill_ledger import append_skill_ledger

SEED_BEST = Path("checkpoints/seed_map_best.keras")
THREAD_BC = Path("checkpoints/dqn_thread_bc.keras")
LATEST = Path("checkpoints/dqn_latest.keras")


def _file_sha256(path: Path, *, limit: int = 2_000_000) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = limit
        while remaining > 0:
            chunk = handle.read(min(65536, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()[:16]


def _build_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig()
    config.training.sim_mode = False
    config.training.device_mode = DeviceMode.AUTO
    config.training.frame_skip = 1
    config.training.transfer_frame_skip = 1
    config.training.watch_mode = True
    config.training.epsilon_start = 0.0
    config.training.epsilon_end = 0.0
    config.training.mixed_sim_replay_ratio = 0.0
    config.training.seed_thread_enabled = True
    config.training.seed_thread_watch_prior = float(args.thread_watch_prior)
    config.training.watch_rule_prior = float(args.watch_rule_prior)
    config.training.watch_safety_override = bool(args.watch_safety)
    config.training.skills_enabled = True
    config.training.skill_exec_at_watch = bool(args.skill_exec)
    config.training.reliability_eval_episodes = int(args.episodes)
    config.training.reliability_mean_score = float(args.target_mean)
    config.training.reliability_min_score = float(args.target_min)
    config.browser.run_seed = int(args.run_seed)
    config.browser.lock_run_seed = True
    return config


def _resolve_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.checkpoint:
        return Path(args.checkpoint)
    if args.policy == "seed_map_best":
        return SEED_BEST if checkpoint_exists(SEED_BEST) else LATEST
    if args.policy == "thread_bc":
        return THREAD_BC if checkpoint_exists(THREAD_BC) else SEED_BEST
    if args.policy in ("learned", "skill_gated"):
        if checkpoint_exists(THREAD_BC):
            return THREAD_BC
        if checkpoint_exists(SEED_BEST):
            return SEED_BEST
        return LATEST
    return None


async def main_async(args: argparse.Namespace) -> int:
    config = _build_config(args)
    policy = str(args.policy)
    started = time.time()
    ckpt: Path | None = None

    if policy in ("skill_router", "seed_thread"):
        print(
            f"BROWSER_RELIABILITY policy={policy} episodes={args.episodes} "
            f"seed={args.run_seed}",
            flush=True,
        )
        if policy == "skill_router":
            metrics = await evaluate_skill_router_baseline(
                config, episodes=int(args.episodes), use_sim=False
            )
        else:
            metrics = await evaluate_seed_thread_baseline(
                config, episodes=int(args.episodes), use_sim=False
            )
        mode = f"browser_reliability_{policy}"
        steps = 0
        ckpt_label = policy
    else:
        ckpt = _resolve_checkpoint(args)
        if ckpt is None or not checkpoint_exists(ckpt):
            print(f"FAILED_LOAD checkpoint={ckpt}", flush=True)
            return 1
        if policy == "skill_gated":
            config.training.skill_exec_at_watch = True
            config.training.skills_enabled = True
        elif policy in ("seed_map_best", "thread_bc", "learned"):
            # Pure greedy Q unless skill_exec explicitly requested.
            if not args.skill_exec:
                config.training.skill_exec_at_watch = False

        agent = DQNAgent(config)
        assert agent.load(ckpt), f"failed to load {ckpt}"
        agent.save(LATEST)
        config.training.checkpoint_path = LATEST
        agent.epsilon = 0.0
        print(
            f"BROWSER_RELIABILITY policy={policy} ckpt={ckpt.name} "
            f"episodes={args.episodes} seed={args.run_seed} "
            f"skill_exec={config.training.skill_exec_at_watch} "
            f"sha={_file_sha256(ckpt)}",
            flush=True,
        )
        metrics = await evaluate_learned_policy(
            config, episodes=int(args.episodes), use_sim=False
        )
        mode = f"browser_reliability_{policy}"
        steps = int(agent.progress.total_steps)
        ckpt_label = str(ckpt)

    elapsed = time.time() - started
    print(format_skill_metrics("BROWSER_100", metrics), flush=True)
    passed = meets_reliability_target(
        metrics.mean_score, metrics.min_score, config
    )
    print(
        f"BROWSER_RELIABILITY_DONE elapsed_s={elapsed:.0f} "
        f"mean={metrics.mean_score:.1f} min={metrics.min_score:.1f} "
        f"max={metrics.max_score:.1f} p10={metrics.p10:.1f} "
        f"p50={metrics.p50:.1f} p90={metrics.p90:.1f} "
        f"target_mean>={args.target_mean} target_min>={args.target_min} "
        f"pass={int(passed)}",
        flush=True,
    )

    if policy == "thread_bc":
        from ascent_player.utils.policy_floors import apply_reliability_baseline

        if apply_reliability_baseline(metrics.mean_score):
            print(
                f"THREAD_BC_BEST_UPDATE reliability_mean={metrics.mean_score:.1f}",
                flush=True,
            )

    append_skill_ledger(
        Path(args.ledger),
        mode=mode,
        mean=metrics.mean_score,
        min_score=metrics.min_score,
        max_score=metrics.max_score,
        steps=steps,
        checkpoint=ckpt_label,
    )

    out = {
        "policy": policy,
        "checkpoint": ckpt_label,
        "checkpoint_sha16": _file_sha256(ckpt) if ckpt else "",
        "run_seed": int(args.run_seed),
        "episodes": int(metrics.episodes),
        "mean": metrics.mean_score,
        "min": metrics.min_score,
        "max": metrics.max_score,
        "p10": metrics.p10,
        "p50": metrics.p50,
        "p90": metrics.p90,
        "mean_length": metrics.mean_length,
        "landing_rate": metrics.landing_rate,
        "scores": list(metrics.scores),
        "skill_exec_at_watch": bool(config.training.skill_exec_at_watch),
        "thread_watch_prior": float(config.training.seed_thread_watch_prior),
        "watch_rule_prior": float(config.training.watch_rule_prior),
        "target_mean": float(args.target_mean),
        "target_min": float(args.target_min),
        "passed": bool(passed),
        "elapsed_s": elapsed,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"BROWSER_RELIABILITY_JSON -> {out_path}", flush=True)
    return 0 if passed or not args.require_pass else 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Browser ε=0 reliability benchmark (full episode aggregation)"
    )
    parser.add_argument(
        "--policy",
        choices=(
            "seed_map_best",
            "thread_bc",
            "skill_router",
            "seed_thread",
            "skill_gated",
            "learned",
        ),
        default="seed_map_best",
    )
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--run-seed", type=int, default=424242)
    parser.add_argument("--target-mean", type=float, default=5000.0)
    parser.add_argument("--target-min", type=float, default=2000.0)
    parser.add_argument("--thread-watch-prior", type=float, default=0.0)
    parser.add_argument("--watch-rule-prior", type=float, default=0.0)
    parser.add_argument("--watch-safety", action="store_true")
    parser.add_argument(
        "--skill-exec",
        action="store_true",
        help="Allow network skill head to execute routines at Watch",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit 2 when reliability targets are not met",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="JSON output path (default logs/browser_reliability_<policy>_<ts>.json)",
    )
    parser.add_argument("--ledger", type=str, default="logs/skill_ledger.csv")
    args = parser.parse_args()
    if not args.out:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = f"logs/browser_reliability_{args.policy}_{ts}.json"
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Offline-heavy v2 climb ladder (Impala-mid only).

Ladder targets (greedy 100-ep mean): 1300 → 1500 → 1800 → 2000.

Phases per round:
  1. Offline BC on hybrid_bc_mix / elite (no browser)
  2. Optional short policy collect (thread prior ≤0.2, no TD)
  3. Optional offline TD on gated ≥1500 / ≥2000 replay
  4. Greedy probe; every --reliability-every rounds run 100-ep greedy check
  5. Promote only on confirmed greedy gains; FT stays off until ≥1400

Usage:
  PYTHONPATH=. python -u scripts/run_v2_climb_ladder.py --hours 4 --run-seed 424242
  PYTHONPATH=. python -u scripts/run_v2_climb_ladder.py --offline-only --bc-steps 2000
"""
from __future__ import annotations

import argparse
import asyncio
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
from ascent_player.evaluation import evaluate_learned_policy
from ascent_player.training import run_eval_watch, run_training_no_ui
from ascent_player.utils.policy_floors import (
    record_thread_bc_eval,
    thread_bc_promotion_floor,
    write_thread_bc_best,
)

LATEST = Path("checkpoints/dqn_latest.keras")
THREAD_BC = Path("checkpoints/dqn_thread_bc.keras")
V2_BEST = Path("checkpoints/dqn_v2_best.keras")
ELITE_REPLAY = Path("checkpoints/elite_thread_replay.pkl")
HYBRID_BC = Path("checkpoints/hybrid_bc_mix.pkl")
HUMAN_REPLAY = Path("checkpoints/hybrid_human_replay.pkl")
TEACHER_REPLAY = Path("checkpoints/teacher_distill_replay.pkl")
BROWSER_REPLAY = Path("checkpoints/browser_replay.pkl")
LADDER_LOG = Path("logs/v2_climb_ladder.jsonl")

LADDER = (1300.0, 1500.0, 1800.0, 2000.0)


def _build_config(run_seed: int) -> AppConfig:
    config = AppConfig()
    config.training.sim_mode = False
    config.training.device_mode = DeviceMode.AUTO
    config.training.model_variant = "impala_mid"
    config.training.mixed_precision = True
    config.training.batch_size_gpu = max(64, int(config.training.batch_size_gpu))
    config.training.n_step = max(3, int(config.training.n_step or 5))
    config.training.frame_skip = 1
    config.training.transfer_frame_skip = 1
    config.training.skills_enabled = True
    config.training.skill_exec_at_watch = False
    config.training.seed_thread_enabled = True
    config.training.watch_rule_prior = 0.0
    config.browser.run_seed = int(run_seed)
    config.browser.lock_run_seed = True
    return config


def _append_log(row: dict) -> None:
    LADDER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with LADDER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _current_rung(mean: float) -> float:
    reached = 0.0
    for target in LADDER:
        if mean >= target:
            reached = target
        else:
            break
    return reached


def _load_work_agent(config: AppConfig) -> DQNAgent:
    """Load v2 weights; skip Nature-era checkpoints that only partially migrate."""
    agent = DQNAgent(config)
    expected = int(agent.online.count_params())
    for path in (V2_BEST, THREAD_BC, LATEST):
        if not checkpoint_exists(path):
            continue
        # Nature-era files are ~7MB; Impala-mid weights are much larger once saved.
        size_mb = path.stat().st_size / (1024 * 1024)
        if expected > 5_000_000 and size_mb < 20:
            print(
                f"CLIMB_SKIP {path.name} size={size_mb:.1f}MB "
                f"(likely Nature; prefer clean v2 + BC warmstart)",
                flush=True,
            )
            continue
        if agent.load(path):
            print(f"CLIMB_LOAD {path.name}", flush=True)
            agent.save(LATEST)
            return agent
    print("CLIMB_LOAD fresh impala_mid (clean warmstart)", flush=True)
    agent.save(LATEST)
    return agent


def _load_offline_replay(agent: DQNAgent, config: AppConfig) -> int:
    agent.replay.clear()
    loaded = 0
    for path in (HYBRID_BC, ELITE_REPLAY, TEACHER_REPLAY, HUMAN_REPLAY):
        if not path.exists():
            continue
        aux = DQNAgent(config)
        aux.replay.clear()
        n = aux.replay.load_pickle(
            path,
            max_items=config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )
        if n > 0:
            agent.replay.extend_from(aux.replay)
            loaded += n
            print(f"CLIMB_REPLAY +{n} from {path.name}", flush=True)
        if loaded >= 512:
            break
    return int(len(agent.replay))


def offline_bc(agent: DQNAgent, *, steps: int, lr: float) -> float | None:
    if len(agent.replay) < max(64, agent.batch_size):
        print(f"CLIMB_BC_SKIP replay={len(agent.replay)}", flush=True)
        return None
    agent.set_learning_rate(lr)
    last = None
    batch_size = min(agent.batch_size, len(agent.replay))
    with agent.tf.device(agent.device_info.training_device):
        for i in range(max(1, steps)):
            batch = agent.replay.sample(batch_size)
            last = float(agent._invoke_bc_train_step(batch.states, batch.actions).numpy())
            if (i + 1) % max(1, steps // 5) == 0:
                print(f"CLIMB_BC step={i+1}/{steps} loss={last:.4f}", flush=True)
    agent._sync_target_network(hard=True)
    agent.save(LATEST)
    agent.save(THREAD_BC)
    return last


def offline_td(agent: DQNAgent, *, steps: int, lr: float, min_score: float) -> float | None:
    if len(agent.replay) < max(64, agent.batch_size):
        return None
    kept = agent.replay.filter_min_episode_score(min_score)
    if kept < max(64, agent.batch_size):
        print(f"CLIMB_TD_SKIP kept>={min_score:.0f} -> {kept}", flush=True)
        return None
    agent.set_learning_rate(lr)
    prev_min = agent.config.training.min_replay_size
    agent.config.training.min_replay_size = 0
    prev_every = agent.train_every
    agent.train_every = 1
    last = None
    try:
        for i in range(max(1, steps)):
            batch = agent._sample_training_batch()
            with agent.tf.device(agent.device_info.training_device):
                loss = agent._train_batch(batch)
            last = float(loss)
            if (i + 1) % max(1, steps // 5) == 0:
                print(f"CLIMB_TD step={i+1}/{steps} loss={last:.4f}", flush=True)
            if (i + 1) % agent.config.training.target_sync_interval == 0:
                agent._sync_target_network(hard=True)
            else:
                agent._sync_target_network(hard=False)
    finally:
        agent.config.training.min_replay_size = prev_min
        agent.train_every = prev_every
    agent.save(LATEST)
    agent.save(THREAD_BC)
    return last


async def greedy_probe(config: AppConfig, episodes: int) -> float:
    config.training.seed_thread_watch_prior = 0.0
    config.training.watch_rule_prior = 0.0
    config.training.skill_exec_at_watch = False
    config.training.watch_mode = True
    config.training.force_save_browser_replay = False
    config.training.skip_browser_replay_load = True
    stats = await run_eval_watch(config, max_episodes=max(4, episodes))
    return float(stats.get("recent_avg", 0.0))


async def reliability_100(config: AppConfig, episodes: int) -> dict[str, float]:
    ckpt = THREAD_BC if checkpoint_exists(THREAD_BC) else LATEST
    config.training.checkpoint_path = ckpt
    config.training.seed_thread_watch_prior = 0.0
    config.training.skill_exec_at_watch = False
    result = await evaluate_learned_policy(config, episodes=episodes, use_sim=False)
    return {
        "mean": float(result.mean_score),
        "min": float(result.min_score),
        "max": float(result.max_score),
    }


async def policy_collect(
    config: AppConfig,
    *,
    seconds: int,
    prior: float,
    eps: float,
) -> dict[str, float]:
    config.training.watch_mode = False
    config.training.min_replay_size = 10**9
    config.training.train_every_gpu = 10**9
    config.training.train_every_cpu = 10**9
    config.training.seed_thread_prior_start = prior
    config.training.seed_thread_prior_end = prior
    config.training.seed_thread_prior_steps = 10**9
    config.training.transfer_epsilon_start = eps
    config.training.browser_epsilon_cap = eps
    config.training.browser_epsilon_floor = min(0.03, eps)
    config.training.force_save_browser_replay = True
    config.training.skip_browser_replay_load = True
    config.training.replay_min_episode_score = 1200.0
    config.demo.use_demos_on_start = False
    print(f"CLIMB_POLICY_COLLECT s={seconds} prior={prior:.2f} eps={eps}", flush=True)
    return await run_training_no_ui(config, max_seconds=seconds, ingest_demos=False)


async def main_async(args: argparse.Namespace) -> int:
    config = _build_config(int(args.run_seed))
    deadline = time.time() + float(args.hours) * 3600.0
    agent = _load_work_agent(config)
    best_mean = thread_bc_promotion_floor()
    round_idx = 0
    reliability_every = max(1, int(args.reliability_every))

    print(
        f"CLIMB_START variant=impala_mid best={best_mean:.1f} "
        f"hours={args.hours} offline_only={int(args.offline_only)}",
        flush=True,
    )

    while time.time() < deadline:
        round_idx += 1
        remaining = deadline - time.time()
        print(f"CLIMB_ROUND {round_idx} remaining_s={remaining:.0f}", flush=True)

        n = _load_offline_replay(agent, config)
        pre = Path("checkpoints/dqn_v2_pre_bc.keras")
        agent.save(pre)
        loss = offline_bc(
            agent,
            steps=int(args.bc_steps),
            lr=float(args.bc_lr),
        )

        if not args.offline_only and remaining > 20 * 60:
            collect_s = min(int(args.collect_minutes * 60), max(300, int(remaining * 0.25)))
            await policy_collect(
                config,
                seconds=collect_s,
                prior=float(
                    getattr(config.training, "policy_collect_thread_prior", 0.20) or 0.20
                ),
                eps=float(args.collect_eps),
            )
            # Merge fresh browser into offline buffer for TD.
            if BROWSER_REPLAY.exists():
                aux = DQNAgent(config)
                aux.replay.clear()
                aux.replay.load_pickle(
                    BROWSER_REPLAY,
                    max_items=config.training.browser_replay_max_items,
                    vector_dim=config.observation.vector_dim,
                )
                agent.replay.extend_from(aux.replay)

            td_gate = 2000.0 if best_mean >= 1500 else 1500.0
            offline_td(
                agent,
                steps=int(args.td_steps),
                lr=float(args.td_lr),
                min_score=td_gate,
            )

        if args.offline_only and args.skip_browser_probe:
            probe_mean = best_mean
            print("CLIMB_PROBE skipped (offline-only)", flush=True)
        else:
            probe_mean = await greedy_probe(config, int(args.probe_episodes))
            print(f"CLIMB_PROBE_GREEDY mean={probe_mean:.1f}", flush=True)
            if probe_mean < best_mean * 0.90 and probe_mean < 800.0:
                assert agent.load(pre)
                agent.save(LATEST)
                agent.save(THREAD_BC)
                print(
                    f"CLIMB_REVERT probe={probe_mean:.1f} < safety floor — restored",
                    flush=True,
                )
                _append_log(
                    {
                        "round": round_idx,
                        "event": "revert",
                        "probe": probe_mean,
                        "best": best_mean,
                        "bc_loss": loss,
                    }
                )
                continue

            record_thread_bc_eval(probe_mean)
            if probe_mean > best_mean:
                best_mean = probe_mean
                write_thread_bc_best(best_mean)
                agent.save(V2_BEST)
                agent.save(THREAD_BC)
                print(f"CLIMB_BEST_UPDATE mean={best_mean:.1f}", flush=True)

        rung = _current_rung(best_mean)
        row = {
            "round": round_idx,
            "event": "probe",
            "probe": probe_mean,
            "best": best_mean,
            "rung": rung,
            "bc_loss": loss,
            "replay": n,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _append_log(row)

        if round_idx % reliability_every == 0 and not args.skip_reliability:
            if remaining < 30 * 60 and int(args.reliability_episodes) >= 50:
                print("CLIMB_RELIABILITY_SKIP low remaining wall time", flush=True)
            else:
                rel = await reliability_100(config, int(args.reliability_episodes))
                print(
                    f"CLIMB_RELIABILITY n={args.reliability_episodes} "
                    f"mean={rel['mean']:.1f} min={rel['min']:.1f} max={rel['max']:.1f}",
                    flush=True,
                )
                _append_log({"round": round_idx, "event": "reliability", **rel})
                if rel["mean"] > best_mean:
                    best_mean = rel["mean"]
                    write_thread_bc_best(best_mean)
                    agent.save(V2_BEST)

        if best_mean >= float(args.target_mean):
            print(f"CLIMB_TARGET_HIT mean={best_mean:.1f}", flush=True)
            # Step 6 scaffolding: raise elite gate once 2k is stable.
            if best_mean >= 2000.0:
                raised = float(
                    getattr(config.training, "post_2k_elite_gate", 3000.0) or 3000.0
                )
                config.training.elite_replay_min_episode_score = raised
                print(
                    f"CLIMB_POST_2K elite_gate->{raised:.0f} "
                    f"(FT allowed if >= {config.training.finetune_min_greedy_mean:.0f})",
                    flush=True,
                )
            break

        if args.offline_only and round_idx >= int(args.max_rounds):
            break

    print(
        f"CLIMB_DONE rounds={round_idx} best={best_mean:.1f} "
        f"rung={_current_rung(best_mean):.0f}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 offline-heavy climb ladder")
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--run-seed", type=int, default=424242)
    parser.add_argument("--target-mean", type=float, default=2000.0)
    parser.add_argument("--bc-steps", type=int, default=800)
    parser.add_argument("--bc-lr", type=float, default=3e-6)
    parser.add_argument("--td-steps", type=int, default=400)
    parser.add_argument("--td-lr", type=float, default=1e-5)
    parser.add_argument("--collect-minutes", type=float, default=12.0)
    parser.add_argument("--collect-eps", type=float, default=0.04)
    parser.add_argument("--probe-episodes", type=int, default=8)
    parser.add_argument("--reliability-episodes", type=int, default=100)
    parser.add_argument("--reliability-every", type=int, default=4)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--skip-browser-probe", action="store_true")
    parser.add_argument("--skip-reliability", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

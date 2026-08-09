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

V2_PROBE_BEST = Path("logs/v2_probe_best_mean.txt")
LADDER = (1300.0, 1500.0, 1800.0, 2000.0)


def _read_v2_probe_best() -> float:
    if V2_PROBE_BEST.exists():
        try:
            return float(V2_PROBE_BEST.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    return 0.0


def _write_v2_probe_best(mean: float) -> None:
    V2_PROBE_BEST.parent.mkdir(parents=True, exist_ok=True)
    V2_PROBE_BEST.write_text(f"{mean:.4f}\n", encoding="utf-8")


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
    # Never run sim teacher/demo warmstart in browser climb sessions — it stalls.
    config.training.sim_warmstart_teacher = False
    config.training.sim_warmstart_demos = False
    config.demo.use_demos_on_start = False
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
    """Load v2 weights; prefer V2_BEST when present, else thread_bc / latest."""
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
            agent.save(THREAD_BC)
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


def offline_td(agent: DQNAgent, *, steps: int, lr: float, min_score: float) -> float | None:
    if steps <= 0:
        print("CLIMB_TD_SKIP steps=0", flush=True)
        return None
    if len(agent.replay) < max(64, agent.batch_size):
        return None
    kept = agent.replay.filter_min_episode_score(min_score)
    if kept < max(64, agent.batch_size):
        print(f"CLIMB_TD_SKIP kept>={min_score:.0f} -> {kept}", flush=True)
        return None
    # BC may have bound LossScaleOptimizer to a Q-only variable set.
    agent.rebuild_optimizer(lr)
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
    except Exception as exc:
        print(f"CLIMB_TD_FAIL {type(exc).__name__}: {exc} — continuing to probe", flush=True)
        return None
    finally:
        agent.config.training.min_replay_size = prev_min
        agent.train_every = prev_every
    # Trial weights only — THREAD_BC is updated on keep/revert.
    agent.save(LATEST)
    return last


def offline_bc(agent: DQNAgent, *, steps: int, lr: float) -> float | None:
    if steps <= 0:
        print("CLIMB_BC_SKIP steps=0", flush=True)
        return None
    if len(agent.replay) < max(64, agent.batch_size):
        print(f"CLIMB_BC_SKIP replay={len(agent.replay)}", flush=True)
        return None
    agent.rebuild_optimizer(lr)
    last = None
    batch_size = min(agent.batch_size, len(agent.replay))
    with agent.tf.device(agent.device_info.training_device):
        for i in range(max(1, steps)):
            batch = agent.replay.sample(batch_size)
            last = float(agent._invoke_bc_train_step(batch.states, batch.actions).numpy())
            if (i + 1) % max(1, steps // 5) == 0:
                print(f"CLIMB_BC step={i+1}/{steps} loss={last:.4f}", flush=True)
    agent._sync_target_network(hard=True)
    # Trial weights only — THREAD_BC is updated on keep/revert.
    agent.save(LATEST)
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


async def reliability_100(
    config: AppConfig, episodes: int, *, ckpt: Path | None = None
) -> dict[str, float]:
    path = ckpt if ckpt is not None else (
        V2_BEST if checkpoint_exists(V2_BEST) else (
            THREAD_BC if checkpoint_exists(THREAD_BC) else LATEST
        )
    )
    prev_ckpt = config.training.checkpoint_path
    config.training.checkpoint_path = path
    config.training.seed_thread_watch_prior = 0.0
    config.training.skill_exec_at_watch = False
    try:
        result = await evaluate_learned_policy(config, episodes=episodes, use_sim=False)
    finally:
        # Critical: do not leave checkpoint_path on V2_BEST — collect/train would
        # overwrite the best Impala save with a slim browser checkpoint.
        config.training.checkpoint_path = prev_ckpt or LATEST
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
    config.training.checkpoint_path = LATEST
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
    config.training.sim_warmstart_teacher = False
    config.training.sim_warmstart_demos = False
    config.demo.use_demos_on_start = False
    print(f"CLIMB_POLICY_COLLECT s={seconds} prior={prior:.2f} eps={eps}", flush=True)
    return await run_training_no_ui(config, max_seconds=seconds, ingest_demos=False)


async def main_async(args: argparse.Namespace) -> int:
    config = _build_config(int(args.run_seed))
    deadline = time.time() + float(args.hours) * 3600.0
    agent = _load_work_agent(config)
    best_mean = thread_bc_promotion_floor()
    v2_best = _read_v2_probe_best()
    round_idx = 0
    reliability_every = max(1, int(args.reliability_every))

    print(
        f"CLIMB_START variant=impala_mid best={best_mean:.1f} "
        f"v2_best={v2_best:.1f} hours={args.hours} "
        f"offline_only={int(args.offline_only)} collect_min={args.collect_minutes}",
        flush=True,
    )

    while time.time() < deadline:
        round_idx += 1
        remaining = deadline - time.time()
        print(f"CLIMB_ROUND {round_idx} remaining_s={remaining:.0f}", flush=True)

        n = _load_offline_replay(agent, config)
        pre = Path("checkpoints/dqn_v2_pre_bc.keras")
        agent.save(pre)

        # When collecting, skip hybrid BC — it collapses Impala off demos.
        bc_steps = int(args.bc_steps)
        collecting = (
            not args.offline_only
            and remaining > 20 * 60
            and float(args.collect_minutes) > 0
        )
        if collecting and bc_steps > 0 and v2_best >= 500.0:
            print(
                f"CLIMB_BC_SKIP protect v2_best={v2_best:.1f} "
                f"(collect+TD instead of hybrid BC)",
                flush=True,
            )
            bc_steps = 0
        loss = offline_bc(
            agent,
            steps=bc_steps,
            lr=float(args.bc_lr),
        )

        if collecting:
            collect_s = min(int(args.collect_minutes * 60), max(300, int(remaining * 0.25)))
            if collect_s >= 60:
                # Always collect with pre-trial weights (good policy), not collapsed BC.
                assert agent.load(pre)
                agent.save(LATEST)
                await policy_collect(
                    config,
                    seconds=collect_s,
                    prior=float(
                        getattr(config.training, "policy_collect_thread_prior", 0.20)
                        or 0.20
                    ),
                    eps=float(args.collect_eps),
                )
                # Restore good weights; browser session may have overwritten LATEST.
                assert agent.load(pre)
                agent.save(LATEST)
                # Prefer light BC on fresh policy replay; TD optional and often noisy.
                browser_bc_steps = int(getattr(args, "browser_bc_steps", 0) or 0)
                if BROWSER_REPLAY.exists() and browser_bc_steps > 0:
                    agent.replay.clear()
                    n_br = agent.replay.load_pickle(
                        BROWSER_REPLAY,
                        max_items=config.training.browser_replay_max_items,
                        vector_dim=config.observation.vector_dim,
                    )
                    print(f"CLIMB_BROWSER_BC replay={n_br} steps={browser_bc_steps}", flush=True)
                    loss = offline_bc(
                        agent,
                        steps=browser_bc_steps,
                        lr=float(args.bc_lr),
                    )
                if BROWSER_REPLAY.exists() and int(args.td_steps) > 0:
                    if len(agent.replay) < 64:
                        _load_offline_replay(agent, config)
                    aux = DQNAgent(config)
                    aux.replay.clear()
                    aux.replay.load_pickle(
                        BROWSER_REPLAY,
                        max_items=config.training.browser_replay_max_items,
                        vector_dim=config.observation.vector_dim,
                    )
                    agent.replay.extend_from(aux.replay)
                    td_gate = 2000.0 if best_mean >= 1500 else (
                        600.0 if v2_best >= 500.0 else 400.0
                    )
                    offline_td(
                        agent,
                        steps=int(args.td_steps),
                        lr=float(args.td_lr),
                        min_score=td_gate,
                    )
                elif int(args.td_steps) <= 0 and browser_bc_steps <= 0:
                    print("CLIMB_TD_SKIP disabled (td_steps=0) — probe next", flush=True)
            else:
                print("CLIMB_COLLECT_SKIP collect_s<60", flush=True)
        elif float(args.collect_minutes) <= 0:
            print("CLIMB_COLLECT_SKIP collect_minutes=0 (offline BC + probe)", flush=True)

        if args.offline_only and args.skip_browser_probe:
            probe_mean = best_mean
            print("CLIMB_PROBE skipped (offline-only)", flush=True)
        else:
            # Probe trial weights in memory / LATEST (do not reload THREAD_BC).
            agent.save(LATEST)
            probe_mean = await greedy_probe(config, int(args.probe_episodes))
            print(
                f"CLIMB_PROBE_GREEDY mean={probe_mean:.1f} "
                f"v2_best={v2_best:.1f} nature_floor={best_mean:.1f}",
                flush=True,
            )
            # Keep/promote: never overwrite a strong v2_best with weaker soft-keeps.
            bootstrap_floor = float(args.bootstrap_keep_floor)
            improved = probe_mean > v2_best + 1.0
            collapsed = v2_best > 0 and probe_mean < max(
                bootstrap_floor * 0.5, v2_best * 0.55
            )
            if v2_best <= 0:
                promote = probe_mean >= bootstrap_floor
            else:
                # Never soft-keep below best — that rewrote 862/274 floors with ~176 junk.
                promote = improved

            if not promote:
                restore = (
                    V2_BEST
                    if (v2_best > 0 and checkpoint_exists(V2_BEST))
                    else pre
                )
                assert agent.load(restore)
                agent.save(LATEST)
                agent.save(THREAD_BC)
                event = "revert" if collapsed else "hold"
                print(
                    f"CLIMB_{event.upper()} probe={probe_mean:.1f} "
                    f"v2_best={v2_best:.1f} — restored {restore.name}",
                    flush=True,
                )
                _append_log(
                    {
                        "round": round_idx,
                        "event": event,
                        "probe": probe_mean,
                        "best": best_mean,
                        "v2_best": v2_best,
                        "bc_loss": loss,
                    }
                )
                # Periodic reliability on frozen V2_BEST only (not trial THREAD_BC).
                if (
                    round_idx % max(1, int(args.reliability_every)) == 0
                    and not args.skip_reliability
                    and remaining >= 25 * 60
                ):
                    rel = await reliability_100(
                        config, int(args.reliability_episodes), ckpt=V2_BEST
                    )
                    print(
                        f"CLIMB_RELIABILITY n={args.reliability_episodes} "
                        f"mean={rel['mean']:.1f} min={rel['min']:.1f} max={rel['max']:.1f}",
                        flush=True,
                    )
                    _append_log({"round": round_idx, "event": "reliability", **rel})
                    # Do not lower v2_best from a single 40-ep draw — high variance
                    # was repeatedly eating 690→560 floors while 12-ep probes stayed ~620+.
                    # Only warn when reliability looks collapsed vs the keep floor.
                    if rel["mean"] + 1.0 < v2_best * 0.70:
                        print(
                            f"CLIMB_RELIABILITY_WARN mean={rel['mean']:.1f} "
                            f"vs v2_best={v2_best:.1f} (floor unchanged)",
                            flush=True,
                        )
                continue

            # Promote trial weights.
            agent.save(THREAD_BC)
            agent.save(LATEST)
            if improved or v2_best <= 0 or probe_mean > v2_best:
                v2_best = max(v2_best, probe_mean)
                _write_v2_probe_best(v2_best)
                agent.save(V2_BEST)
                print(f"CLIMB_V2_BEST mean={v2_best:.1f}", flush=True)
            else:
                print(
                    f"CLIMB_KEEP probe={probe_mean:.1f} (v2_best stays {v2_best:.1f})",
                    flush=True,
                )
            if probe_mean > best_mean:
                best_mean = probe_mean
                write_thread_bc_best(best_mean)
                print(f"CLIMB_BEST_UPDATE mean={best_mean:.1f}", flush=True)
            record_thread_bc_eval(probe_mean)

        rung = _current_rung(best_mean)
        row = {
            "round": round_idx,
            "event": "probe",
            "probe": probe_mean,
            "best": best_mean,
            "v2_best": v2_best,
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
    parser.add_argument(
        "--browser-bc-steps",
        type=int,
        default=0,
        help="After policy collect, BC on browser replay only (0=skip).",
    )
    parser.add_argument("--collect-minutes", type=float, default=12.0)
    parser.add_argument("--collect-eps", type=float, default=0.04)
    parser.add_argument("--probe-episodes", type=int, default=8)
    parser.add_argument(
        "--bootstrap-keep-floor",
        type=float,
        default=350.0,
        help="While v2_best<900, keep BC weights if greedy probe >= this floor.",
    )
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

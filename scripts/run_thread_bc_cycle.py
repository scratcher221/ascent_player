#!/usr/bin/env python3
"""7h thread-BC cycle: ceiling → elite collect → offline BC → aligned Watch → light FT.

Implements the post-overnight action plan:
  1. Pure SeedThreadPolicy ceiling measurement
  2. Elite (≥1500) collect under strong seed-thread prior (no online TD)
  3. Offline BC onto side checkpoint dqn_thread_bc (not wiped on regress)
  4. Dual Watch eval (greedy + thread-aligned); promote only on real gains
  5. Light online fine-tune from the side checkpoint for remaining wall time

Usage:
  PYTHONPATH=. python -u scripts/run_thread_bc_cycle.py --hours 7 --run-seed 424242
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
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
    evaluate_seed_thread_baseline,
    format_skill_metrics,
)
from ascent_player.training import run_eval_watch, run_training_no_ui

SEED_BEST = Path("checkpoints/seed_map_best.keras")
SEED_MEAN = Path("logs/seed_map_best_mean.txt")
LATEST = Path("checkpoints/dqn_latest.keras")
THREAD_BC = Path("checkpoints/dqn_thread_bc.keras")
ELITE_REPLAY = Path("checkpoints/elite_thread_replay.pkl")
BROWSER_REPLAY = Path("checkpoints/browser_replay.pkl")


def offline_bc_train(agent: DQNAgent, *, steps: int, lr: float) -> float | None:
    """Behavior-clone actions from gated high-score replay (no TD bootstrap)."""
    if len(agent.replay) < max(64, agent.batch_size):
        print(f"OFFLINE_BC_SKIP replay={len(agent.replay)} too small", flush=True)
        return None
    agent.set_learning_rate(lr)
    frozen = []
    for layer in agent.online.layers:
        name = (layer.name or "").lower()
        if any(k in name for k in ("conv", "separable", "depthwise")) and layer.trainable:
            layer.trainable = False
            frozen.append(layer.name)
    if frozen:
        print(f"OFFLINE_BC_FREEZE layers={frozen}", flush=True)
    last_loss = None
    batch_size = min(agent.batch_size, len(agent.replay))
    try:
        with agent.tf.device(agent.device_info.training_device):
            for i in range(max(1, steps)):
                batch = agent.replay.sample(batch_size)
                loss = float(
                    agent._invoke_bc_train_step(batch.states, batch.actions).numpy()
                )
                last_loss = loss
                if (i + 1) % max(1, steps // 5) == 0:
                    print(
                        f"OFFLINE_BC step={i+1}/{steps} loss={last_loss:.4f} "
                        f"replay={len(agent.replay)}",
                        flush=True,
                    )
        agent._sync_target_network(hard=True)
        agent.save(LATEST)
    finally:
        for layer in agent.online.layers:
            if layer.name in frozen:
                layer.trainable = True
    return last_loss


def _read_best_mean() -> float:
    if SEED_MEAN.exists():
        try:
            return float(SEED_MEAN.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    return 1250.7


def _write_best_mean(mean: float) -> None:
    SEED_MEAN.parent.mkdir(parents=True, exist_ok=True)
    SEED_MEAN.write_text(f"{mean:.4f}\n", encoding="utf-8")


def _build_config(run_seed: int, *, elite_gate: float) -> AppConfig:
    config = AppConfig()
    config.training.sim_mode = False
    config.training.device_mode = DeviceMode.AUTO
    config.training.frame_skip = 1
    config.training.transfer_frame_skip = 1
    config.training.mixed_sim_replay_ratio = 0.0
    config.training.log_decision_every = 1
    config.training.reason_aux_weight = 0.12
    config.training.replay_min_episode_score = float(elite_gate)
    config.training.thread_bc_checkpoint_path = THREAD_BC
    config.browser.run_seed = int(run_seed)
    config.browser.lock_run_seed = True
    config.training.seed_thread_enabled = True
    config.training.seed_thread_corridor = 0.16
    config.training.seed_thread_only_when_landing = True
    config.mechanics_reward.steer_gain = 0.24
    config.mechanics_reward.wrong_way_penalty = -0.30
    config.mechanics_reward.aligned_bonus = 0.05
    config.mechanics_reward.platform_land = 1.7
    config.mechanics_reward.combo_gain = 0.45
    return config


def _load_work_agent(config: AppConfig, *, prefer_thread_bc: bool) -> DQNAgent:
    agent = DQNAgent(config)
    if prefer_thread_bc and checkpoint_exists(THREAD_BC):
        src = THREAD_BC
    elif checkpoint_exists(SEED_BEST):
        src = SEED_BEST
    else:
        src = LATEST
    assert agent.load(src), f"failed to load {src}"
    agent.save(LATEST)
    print(f"LOAD_WEIGHTS src={src.name}", flush=True)
    return agent


def _save_thread_bc(agent: DQNAgent) -> None:
    agent.save(THREAD_BC)
    agent.save(LATEST)
    print(f"SAVE_THREAD_BC -> {THREAD_BC.name}", flush=True)


def _promote_seed_best(agent: DQNAgent, mean: float) -> None:
    agent.save(SEED_BEST)
    agent.save(THREAD_BC)
    agent.save(LATEST)
    _write_best_mean(mean)
    print(f"SEED_BEST_UPDATE mean={mean:.1f} -> {SEED_BEST.name}", flush=True)


def _persist_elite_replay(config: AppConfig) -> int:
    """Merge gated browser replay into the durable elite pickle."""
    if not BROWSER_REPLAY.exists():
        if not ELITE_REPLAY.exists():
            return 0
        # Report existing elite size without merging.
        probe = DQNAgent(config)
        probe.replay.clear()
        n = probe.replay.load_pickle(
            ELITE_REPLAY,
            max_items=config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )
        return int(n)
    ELITE_REPLAY.parent.mkdir(parents=True, exist_ok=True)
    merged = DQNAgent(config)
    merged.replay.clear()
    before = 0
    if ELITE_REPLAY.exists():
        before = merged.replay.load_pickle(
            ELITE_REPLAY,
            max_items=config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )
    fresh = DQNAgent(config)
    fresh.replay.clear()
    browser_n = fresh.replay.load_pickle(
        BROWSER_REPLAY,
        max_items=config.training.browser_replay_max_items,
        vector_dim=config.observation.vector_dim,
    )
    if browser_n > 0:
        merged.replay.extend_from(fresh.replay)
    saved = merged.replay.save_pickle(
        ELITE_REPLAY,
        max_items=config.training.browser_replay_max_items,
    )
    print(
        f"ELITE_REPLAY merge before={before} browser={browser_n} "
        f"saved={saved} -> {ELITE_REPLAY.name}",
        flush=True,
    )
    return int(saved)


def _enable_thread_priors(
    config: AppConfig,
    *,
    train_prior: float,
    watch_prior: float,
) -> None:
    config.training.seed_thread_enabled = True
    config.training.seed_thread_prior_start = float(train_prior)
    config.training.seed_thread_prior_end = float(train_prior)
    config.training.seed_thread_prior_steps = 10**9
    config.training.seed_thread_watch_prior = float(watch_prior)
    config.training.seed_thread_corridor = 0.16
    config.training.seed_thread_only_when_landing = True


async def phase_ceiling(config: AppConfig, *, episodes: int) -> dict[str, float]:
    print(f"PHASE_CEILING episodes={episodes}", flush=True)
    metrics = await evaluate_seed_thread_baseline(
        config, episodes=episodes, use_sim=False
    )
    print(format_skill_metrics("THREAD_CEILING", metrics), flush=True)
    return {
        "mean": float(metrics.mean_score),
        "min": float(metrics.min_score),
        "max": float(metrics.max_score),
    }


async def phase_collect(
    config: AppConfig,
    *,
    seconds: int,
    eps: float,
    thread_prior: float,
    skip_replay_load: bool,
) -> dict[str, float]:
    _enable_thread_priors(config, train_prior=thread_prior, watch_prior=0.0)
    config.training.watch_mode = False
    config.training.min_replay_size = 10**9
    config.training.train_every_gpu = 10**9
    config.training.train_every_cpu = 10**9
    config.training.transfer_epsilon_start = eps
    config.training.browser_epsilon_cap = eps
    config.training.browser_epsilon_floor = min(0.03, eps)
    config.training.rule_prior_start = 0.05
    config.training.rule_prior_end = 0.05
    config.training.rule_prior_steps = 10**9
    config.training.learning_rate = 1e-5
    config.training.force_save_browser_replay = True
    config.training.skip_browser_replay_load = skip_replay_load
    config.training.sim_warmstart_teacher = False
    config.training.sim_warmstart_demos = False
    config.demo.use_demos_on_start = False
    # Ensure latest weights are the work agent (thread_bc or seed).
    print(
        f"PHASE_COLLECT s={seconds} eps={eps} thread_prior={thread_prior} "
        f"gate>={config.training.replay_min_episode_score} "
        f"skip_load={skip_replay_load}",
        flush=True,
    )
    return await run_training_no_ui(
        config,
        max_seconds=seconds,
        ingest_demos=False,
    )


async def phase_bc(
    config: AppConfig,
    *,
    steps: int,
    lr: float,
    prefer_thread_bc: bool,
    eval_probe_episodes: int = 4,
) -> float | None:
    """BC from elite demos onto a *copy* of seed/side weights; revert if greedy collapses."""
    # Always BC from the frozen floor so thin elites can't compound destruction.
    agent = DQNAgent(config)
    assert agent.load(SEED_BEST if checkpoint_exists(SEED_BEST) else LATEST)
    agent.save(LATEST)
    print(f"PHASE_BC base=seed_map_best steps={steps} lr={lr:.2e}", flush=True)

    agent.replay.clear()
    if ELITE_REPLAY.exists():
        agent.replay.load_pickle(
            ELITE_REPLAY,
            max_items=config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )
    loaded = len(agent.replay)
    print(f"PHASE_BC load replay={loaded}", flush=True)
    # Need a meaningful elite buffer — one lucky episode is not enough.
    if loaded < 1500:
        print("PHASE_BC_SKIP need >=1500 elite transitions", flush=True)
        return None

    pre_path = Path("checkpoints/dqn_thread_bc_pre_bc.keras")
    agent.save(pre_path)

    loss = offline_bc_train(agent, steps=steps, lr=lr)
    _save_thread_bc(agent)
    print(f"PHASE_BC_DONE loss={loss}", flush=True)

    # Quick greedy probe; revert if BC destroyed the floor policy.
    config.training.seed_thread_enabled = True
    config.training.seed_thread_watch_prior = 0.0
    config.training.watch_rule_prior = 0.0
    config.training.force_save_browser_replay = False
    config.training.skip_browser_replay_load = True
    probe = await run_eval_watch(config, max_episodes=max(3, eval_probe_episodes))
    mean = float(probe.get("recent_avg", 0.0))
    floor = _read_best_mean()
    print(f"BC_PROBE_GREEDY mean={mean:.1f} floor={floor:.1f}", flush=True)
    if mean < floor * 0.85:
        agent = DQNAgent(config)
        assert agent.load(pre_path)
        _save_thread_bc(agent)
        print(
            f"BC_REVERT mean={mean:.1f} < 0.85*floor — restored pre-BC weights",
            flush=True,
        )
        return None
    # If the probe clears the floor, immediately run a longer greedy Watch for
    # promotion — short probes have beaten the floor then failed an 8-ep eval.
    if mean > floor:
        confirm = await run_eval_watch(config, max_episodes=12)
        cmean = float(confirm.get("recent_avg", 0.0))
        cmin = float(confirm.get("recent_min", 0.0))
        cmax = float(confirm.get("recent_max", 0.0))
        print(
            f"BC_CONFIRM_GREEDY mean={cmean:.1f} min={cmin:.1f} max={cmax:.1f}",
            flush=True,
        )
        if cmean > floor and cmin >= 0.35 * 900.0:
            agent = DQNAgent(config)
            assert agent.load(THREAD_BC if checkpoint_exists(THREAD_BC) else LATEST)
            _promote_seed_best(agent, cmean)
            print(
                f"PROMOTE_VIA bc_confirm mean={cmean:.1f}",
                flush=True,
            )
    return loss


async def phase_dual_eval(
    config: AppConfig,
    *,
    episodes: int,
    thread_watch_prior: float,
) -> dict[str, dict[str, float]]:
    """Run greedy Watch then thread-aligned Watch from dqn_thread_bc/latest."""
    # Sync latest from side ckpt if present.
    if checkpoint_exists(THREAD_BC):
        agent = DQNAgent(config)
        assert agent.load(THREAD_BC)
        agent.save(LATEST)

    results: dict[str, dict[str, float]] = {}

    # A: pure greedy (no thread prior)
    config.training.seed_thread_enabled = True
    config.training.seed_thread_watch_prior = 0.0
    config.training.watch_rule_prior = 0.0
    config.training.force_save_browser_replay = False
    config.training.skip_browser_replay_load = True
    print(f"PHASE_EVAL greedy episodes={episodes}", flush=True)
    greedy = await run_eval_watch(config, max_episodes=episodes)
    results["greedy"] = {
        "mean": float(greedy.get("recent_avg", 0.0)),
        "min": float(greedy.get("recent_min", 0.0)),
        "max": float(greedy.get("recent_max", 0.0)),
    }
    print(
        f"EVAL_GREEDY mean={results['greedy']['mean']:.1f} "
        f"min={results['greedy']['min']:.1f} max={results['greedy']['max']:.1f}",
        flush=True,
    )

    # B: thread-aligned
    if checkpoint_exists(THREAD_BC):
        agent = DQNAgent(config)
        assert agent.load(THREAD_BC)
        agent.save(LATEST)
    config.training.seed_thread_watch_prior = float(thread_watch_prior)
    print(
        f"PHASE_EVAL thread_aligned prior={thread_watch_prior} episodes={episodes}",
        flush=True,
    )
    aligned = await run_eval_watch(config, max_episodes=episodes)
    results["aligned"] = {
        "mean": float(aligned.get("recent_avg", 0.0)),
        "min": float(aligned.get("recent_min", 0.0)),
        "max": float(aligned.get("recent_max", 0.0)),
    }
    print(
        f"EVAL_ALIGNED mean={results['aligned']['mean']:.1f} "
        f"min={results['aligned']['min']:.1f} max={results['aligned']['max']:.1f}",
        flush=True,
    )
    return results


async def phase_finetune(
    config: AppConfig,
    *,
    seconds: int,
    lr: float,
    eps: float,
    thread_prior: float,
    gate: float,
) -> dict[str, float]:
    """Light online TD from dqn_thread_bc; does not wipe side ckpt on its own."""
    agent = _load_work_agent(config, prefer_thread_bc=True)
    agent.epsilon = eps
    agent.metrics.epsilon = eps
    agent.progress.epsilon = eps
    agent.save(LATEST)

    _enable_thread_priors(config, train_prior=thread_prior, watch_prior=thread_prior)
    config.training.watch_mode = False
    config.training.min_replay_size = 1000
    config.training.train_every_gpu = 4
    config.training.train_every_cpu = 4
    config.training.transfer_epsilon_start = eps
    config.training.browser_epsilon_cap = eps
    config.training.browser_epsilon_floor = min(0.03, eps)
    config.training.rule_prior_start = 0.05
    config.training.rule_prior_end = 0.02
    config.training.rule_prior_steps = 40_000
    config.training.learning_rate = lr
    config.training.replay_min_episode_score = float(gate)
    config.training.force_save_browser_replay = True
    config.training.skip_browser_replay_load = False
    # Prefer elite buffer if present.
    if ELITE_REPLAY.exists():
        shutil.copy2(ELITE_REPLAY, BROWSER_REPLAY)
    config.training.sim_warmstart_teacher = False
    config.training.sim_warmstart_demos = False
    config.demo.use_demos_on_start = False
    print(
        f"PHASE_FINETUNE s={seconds} lr={lr:.2e} eps={eps} "
        f"thread={thread_prior} gate>={gate}",
        flush=True,
    )
    stats = await run_training_no_ui(
        config,
        max_seconds=seconds,
        ingest_demos=False,
    )
    # Persist work weights to side ckpt (never touch seed_map_best here).
    agent = DQNAgent(config)
    if agent.load(LATEST):
        _save_thread_bc(agent)
    _persist_elite_replay(config)
    return stats


async def main_async(args: argparse.Namespace) -> int:
    deadline = time.time() + max(600.0, float(args.hours) * 3600.0)
    run_seed = int(args.run_seed)
    elite_gate = float(args.elite_gate)
    floor = _read_best_mean()
    target_mean = float(args.target_mean)
    target_min = float(args.target_min)

    config = _build_config(run_seed, elite_gate=elite_gate)
    # Bootstrap side ckpt from seed best if missing.
    if not checkpoint_exists(THREAD_BC) and checkpoint_exists(SEED_BEST):
        agent = DQNAgent(config)
        assert agent.load(SEED_BEST)
        _save_thread_bc(agent)

    print(
        f"THREAD_BC_CYCLE_START seed={run_seed} hours={args.hours} "
        f"floor={floor:.1f} elite_gate={elite_gate} "
        f"target_mean>={target_mean} target_min>={target_min}",
        flush=True,
    )

    # ---- Phase 0: ceiling ----
    if not bool(args.skip_ceiling):
        ceiling = await phase_ceiling(
            config, episodes=max(6, int(args.ceiling_episodes))
        )
        print(
            f"CEILING_SUMMARY mean={ceiling['mean']:.1f} "
            f"min={ceiling['min']:.1f} max={ceiling['max']:.1f} floor={floor:.1f}",
            flush=True,
        )
    else:
        print("CEILING_SKIPPED", flush=True)

    round_id = 0
    thread_prior = float(args.thread_prior)
    thread_watch = float(args.thread_watch_prior)
    bc_steps = int(args.bc_steps)
    bc_lr = float(args.bc_lr)
    collect_eps = float(args.collect_eps)
    prefer_thread_bc = checkpoint_exists(THREAD_BC)
    first_collect = True

    while time.time() < deadline:
        round_id += 1
        remaining = deadline - time.time()
        if remaining < 300:
            print("TIME_LOW stopping", flush=True)
            break

        print(
            f"CYCLE_ROUND id={round_id} remaining_h={remaining/3600:.2f} "
            f"gate={config.training.replay_min_episode_score} "
            f"thread_prior={thread_prior} watch={thread_watch}",
            flush=True,
        )

        # Reserve time: collect / bc / dual eval / optional FT slice
        # First rounds prioritize collect+BC; later rounds add FT.
        collect_budget = min(
            int(args.collect_minutes * 60),
            max(300, int(remaining * 0.40)),
        )

        # Ensure collect plays from work weights.
        agent = _load_work_agent(config, prefer_thread_bc=prefer_thread_bc)
        agent.epsilon = collect_eps
        agent.metrics.epsilon = collect_eps
        agent.progress.epsilon = collect_eps
        agent.save(LATEST)

        collect_stats = await phase_collect(
            config,
            seconds=collect_budget,
            eps=collect_eps,
            thread_prior=thread_prior,
            # Never reload the shared browser_replay.pkl during collect — it has
            # mixed historical gates and poisoned prior FT data. Elite pickle is
            # the only durable demo store for this cycle.
            skip_replay_load=True,
        )
        first_collect = False
        print(
            f"COLLECT_DONE recent_avg={collect_stats.get('recent_avg', 0):.0f} "
            f"replay={collect_stats.get('replay_size', 0)}",
            flush=True,
        )
        elite_n = _persist_elite_replay(config)
        collect_replay = int(collect_stats.get("replay_size", 0) or 0)
        if collect_replay < 200 and (elite_n < 200):
            old = config.training.replay_min_episode_score
            config.training.replay_min_episode_score = max(1200.0, old - 100.0)
            thread_prior = min(0.85, thread_prior + 0.05)
            collect_eps = min(0.08, collect_eps + 0.01)
            print(
                f"STARVE_ADJUST gate {old:.0f}->{config.training.replay_min_episode_score:.0f} "
                f"thread_prior={thread_prior:.2f} eps={collect_eps:.3f}",
                flush=True,
            )
            continue

        loss = await phase_bc(
            config,
            steps=bc_steps,
            lr=bc_lr,
            prefer_thread_bc=prefer_thread_bc,
            eval_probe_episodes=6,
        )
        prefer_thread_bc = True
        if loss is None:
            # Accumulate more diverse elites; gently loosen gate toward 1400.
            config.training.replay_min_episode_score = max(
                1400.0, config.training.replay_min_episode_score - 25.0
            )
            continue

        remaining = deadline - time.time()
        if remaining < 240:
            break

        evals = await phase_dual_eval(
            config,
            episodes=max(6, int(args.eval_episodes)),
            thread_watch_prior=thread_watch,
        )
        aligned = evals["aligned"]
        greedy = evals["greedy"]
        floor = _read_best_mean()

        # Promote on the stronger of greedy / aligned, but never if both lag the floor.
        # Pure thread ceiling is often < floor; forcing a high watch_prior can *hurt*
        # a partially BC'd net (round-1: greedy 1035 > aligned 876).
        best_mean = max(aligned["mean"], greedy["mean"])
        best_min = (
            aligned["min"] if aligned["mean"] >= greedy["mean"] else greedy["min"]
        )
        promote_src = "aligned" if aligned["mean"] >= greedy["mean"] else "greedy"
        improved = best_mean > floor and best_min >= target_min * 0.35
        # If greedy collapsed far below floor while aligned looks good, require
        # greedy not to be catastrophic before promoting a hybrid policy.
        if promote_src == "aligned" and greedy["mean"] < floor * 0.70:
            improved = False
        if improved:
            agent = DQNAgent(config)
            assert agent.load(THREAD_BC if checkpoint_exists(THREAD_BC) else LATEST)
            _promote_seed_best(agent, best_mean)
            floor = best_mean
            print(
                f"PROMOTE_VIA {promote_src} mean={best_mean:.1f} "
                f"greedy={greedy['mean']:.1f} aligned={aligned['mean']:.1f}",
                flush=True,
            )
            bc_steps = min(900, bc_steps + 50)
            collect_eps = max(0.03, collect_eps * 0.97)
            # Tighten gate after a win.
            config.training.replay_min_episode_score = min(
                1500.0, config.training.replay_min_episode_score + 50.0
            )
            # Anneal watch prior toward greedy skill as BC sticks.
            thread_watch = max(0.25, thread_watch * 0.9)
        else:
            print(
                f"NO_PROMOTE aligned={aligned['mean']:.1f} greedy={greedy['mean']:.1f} "
                f"floor={floor:.1f} — keep side ckpt + elite replay",
                flush=True,
            )
            # Do NOT clear elite replay. Prefer more elites + milder watch prior.
            if best_mean < floor * 0.90:
                thread_prior = min(0.85, thread_prior + 0.03)
                collect_eps = min(0.08, collect_eps + 0.01)
                thread_watch = max(0.30, thread_watch * 0.92)
                # Loosen gate slightly when starving for ≥1500 elites.
                config.training.replay_min_episode_score = max(
                    1300.0, config.training.replay_min_episode_score - 50.0
                )
            else:
                # Near miss: prefer rarer elites; keep watch prior moderate.
                config.training.replay_min_episode_score = min(
                    1500.0, config.training.replay_min_episode_score + 25.0
                )
                thread_watch = max(0.35, min(thread_watch, 0.45))

        if best_mean >= target_mean and best_min >= target_min:
            print(
                f"TARGET_MET mean={best_mean:.1f} min={best_min:.1f}",
                flush=True,
            )
            return 0

        # Light fine-tune only when BC/eval is already near the floor — otherwise
        # online TD has repeatedly collapsed greedy Watch (seen: 1035 → 732).
        remaining = deadline - time.time()
        near_floor = best_mean >= floor * 0.85
        if (
            remaining >= 45 * 60
            and near_floor
            and float(args.finetune_minutes) > 0
        ):
            # Snapshot side ckpt before FT; restore if FT regresses greedy.
            pre_ft = Path("checkpoints/dqn_thread_bc_pre_ft.keras")
            if checkpoint_exists(THREAD_BC):
                shutil.copy2(THREAD_BC, pre_ft)
            ft_seconds = min(
                int(args.finetune_minutes * 60), max(600, int(remaining * 0.35))
            )
            await phase_finetune(
                config,
                seconds=ft_seconds,
                lr=float(args.ft_lr),
                eps=max(0.04, collect_eps),
                thread_prior=max(0.35, thread_prior * 0.85),
                gate=max(1200.0, config.training.replay_min_episode_score - 100.0),
            )
            # Re-eval after FT (aligned only if time is short).
            remaining = deadline - time.time()
            if remaining >= 200:
                evals = await phase_dual_eval(
                    config,
                    episodes=max(6, int(args.eval_episodes)),
                    thread_watch_prior=thread_watch,
                )
                aligned = evals["aligned"]
                greedy = evals["greedy"]
                floor = _read_best_mean()
                best_mean = max(aligned["mean"], greedy["mean"])
                best_min = (
                    aligned["min"]
                    if aligned["mean"] >= greedy["mean"]
                    else greedy["min"]
                )
                improved = best_mean > floor and best_min >= target_min * 0.35
                if aligned["mean"] >= greedy["mean"] and greedy["mean"] < floor * 0.70:
                    improved = False
                if improved:
                    agent = DQNAgent(config)
                    assert agent.load(
                        THREAD_BC if checkpoint_exists(THREAD_BC) else LATEST
                    )
                    _promote_seed_best(agent, best_mean)
                    floor = best_mean
                    print(
                        f"PROMOTE_VIA post_ft mean={best_mean:.1f} "
                        f"greedy={greedy['mean']:.1f} aligned={aligned['mean']:.1f}",
                        flush=True,
                    )
                elif pre_ft.exists() and greedy["mean"] < floor * 0.85:
                    agent = DQNAgent(config)
                    assert agent.load(pre_ft)
                    _save_thread_bc(agent)
                    print(
                        f"FT_REVERT greedy={greedy['mean']:.1f} — restored pre-FT side ckpt",
                        flush=True,
                    )
        elif remaining >= 45 * 60 and not near_floor:
            print(
                f"FT_SKIP best_mean={best_mean:.1f} < 0.90*floor={floor*0.90:.1f} "
                f"— more elite BC first",
                flush=True,
            )

    print(
        f"THREAD_BC_CYCLE_END floor={_read_best_mean():.1f} "
        f"thread_bc={'yes' if checkpoint_exists(THREAD_BC) else 'no'} "
        f"elite={'yes' if ELITE_REPLAY.exists() else 'no'}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Thread-BC 7h supervised cycle")
    parser.add_argument("--hours", type=float, default=7.0)
    parser.add_argument("--run-seed", type=int, default=424242)
    parser.add_argument("--target-mean", type=float, default=2000.0)
    parser.add_argument("--target-min", type=float, default=900.0)
    parser.add_argument("--elite-gate", type=float, default=1500.0)
    parser.add_argument("--thread-prior", type=float, default=0.75)
    parser.add_argument("--thread-watch-prior", type=float, default=0.55)
    parser.add_argument("--collect-minutes", type=float, default=18.0)
    parser.add_argument("--collect-eps", type=float, default=0.04)
    parser.add_argument("--bc-steps", type=int, default=500)
    parser.add_argument("--bc-lr", type=float, default=3e-6)
    parser.add_argument("--finetune-minutes", type=float, default=40.0)
    parser.add_argument("--ft-lr", type=float, default=2e-5)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--ceiling-episodes", type=int, default=8)
    parser.add_argument(
        "--skip-ceiling",
        action="store_true",
        help="Skip pure-thread ceiling phase (use when resuming mid-window).",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

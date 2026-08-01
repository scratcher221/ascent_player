#!/usr/bin/env python3
"""v2 thread-BC cycle: teacher/policy collect → offline BC → greedy probe → gated FT.

Objective realignment (Impala-mid default):
  1. Teacher collect (high prior, labels) + policy collect (prior ≤0.2)
  2. Offline BC onto dqn_thread_bc; keep/revert on pure greedy probe
  3. Aligned Watch is diagnostic only
  4. Online FT only when greedy floor ≥ finetune_min_greedy_mean (default 1400)
  5. Skill exec off in Watch until accuracy recovers

Usage:
  PYTHONPATH=. python -u scripts/run_thread_bc_cycle.py --hours 7 --run-seed 424242
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

from ascent_player.agent.checkpoint import checkpoint_exists
from ascent_player.agent.dqn import DQNAgent
from ascent_player.agent.seed_curriculum import (
    FineTuneReadiness,
    confirmed_promotion,
)
from ascent_player.utils.policy_floors import (
    read_seed_floor,
    record_thread_bc_eval,
    thread_bc_promotion_floor,
    write_thread_bc_best,
)
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.evaluation import (
    evaluate_seed_thread_baseline,
    evaluate_skill_router_baseline,
    format_skill_metrics,
)
from ascent_player.training import run_eval_watch, run_training_no_ui

SEED_BEST = Path("checkpoints/seed_map_best.keras")
SEED_MEAN = Path("logs/seed_map_best_mean.txt")
THREAD_WATCH_FLOOR = Path("logs/thread_watch_floor.txt")
LATEST = Path("checkpoints/dqn_latest.keras")
THREAD_BC = Path("checkpoints/dqn_thread_bc.keras")
ELITE_REPLAY = Path("checkpoints/elite_thread_replay.pkl")
ELITE_META = Path("checkpoints/elite_thread_replay.json")
BROWSER_REPLAY = Path("checkpoints/browser_replay.pkl")
MIN_ELITE_SCORE = 2000.0
DEFAULT_COLLECT_GATE = 1200.0
MAX_ELITE_EPISODES = 64
MAX_TRANSITIONS_PER_ELITE = 128
DEFAULT_BC_SECONDARY_GATE = 1800.0


def offline_bc_train(agent: DQNAgent, *, steps: int, lr: float) -> float | None:
    """Behavior-clone actions from gated high-score replay (no TD bootstrap)."""
    if len(agent.replay) < max(64, agent.batch_size):
        print(f"OFFLINE_BC_SKIP replay={len(agent.replay)} too small", flush=True)
        return None
    agent.set_learning_rate(lr)
    frozen = []
    for layer in agent.online.layers:
        name = (layer.name or "").lower()
        # Freeze vision trunk and skill head so motor BC does not wipe routing.
        if (
            any(k in name for k in ("conv", "separable", "depthwise", "skill_logits"))
            and layer.trainable
        ):
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
    return read_seed_floor()


def _write_best_mean(mean: float) -> None:
    SEED_MEAN.parent.mkdir(parents=True, exist_ok=True)
    SEED_MEAN.write_text(f"{mean:.4f}\n", encoding="utf-8")


def _read_thread_floor() -> float:
    if THREAD_WATCH_FLOOR.exists():
        try:
            return float(THREAD_WATCH_FLOOR.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    return _read_best_mean()


def _write_thread_floor(mean: float) -> None:
    THREAD_WATCH_FLOOR.parent.mkdir(parents=True, exist_ok=True)
    THREAD_WATCH_FLOOR.write_text(f"{mean:.4f}\n", encoding="utf-8")


def _build_config(
    run_seed: int,
    *,
    elite_gate: float,
    collect_gate: float = DEFAULT_COLLECT_GATE,
    elite_max_episodes: int = MAX_ELITE_EPISODES,
) -> AppConfig:
    config = AppConfig()
    config.training.sim_mode = False
    config.training.device_mode = DeviceMode.AUTO
    config.training.frame_skip = 1
    config.training.transfer_frame_skip = 1
    config.training.mixed_sim_replay_ratio = 0.0
    config.training.log_decision_every = 1
    config.training.reason_aux_weight = 0.12
    # Dual gates: collect volume vs elite quality.
    config.training.replay_min_episode_score = max(0.0, float(collect_gate))
    config.training.elite_replay_min_episode_score = max(
        MIN_ELITE_SCORE, float(elite_gate)
    )
    config.training.elite_max_episodes = max(8, int(elite_max_episodes))
    config.training.elite_max_transitions_per_episode = MAX_TRANSITIONS_PER_ELITE
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
    config.training.skills_enabled = True
    # Watch skill execution remains gated; default off until accuracy recovers.
    config.training.skill_exec_at_watch = False
    config.training.skill_exec_min_accuracy = 0.85
    config.training.skill_aux_weight = 0.12
    config.training.model_variant = str(
        getattr(config.training, "model_variant", "impala_mid") or "impala_mid"
    )
    config.training.mixed_precision = True
    config.training.batch_size_gpu = max(64, int(config.training.batch_size_gpu))
    config.training.n_step = max(3, int(getattr(config.training, "n_step", 5) or 5))
    config.training.skill_teacher_prior_start = 0.55
    config.training.skill_teacher_prior_end = 0.10
    config.training.skill_teacher_prior_steps = 100_000
    config.training.skill_confidence_margin = 0.10
    # Skills own motor control; thread is diagnostic / light bias only.
    config.training.seed_thread_prior_start = 0.12
    config.training.seed_thread_prior_end = 0.04
    config.training.seed_thread_watch_prior = 0.0
    config.training.watch_rule_prior = 0.0
    return config


def _checkpoint_likely_compatible(path: Path, expected_params: int) -> bool:
    """Heuristic: Nature-era .keras ~7MB; Impala-mid weights are much larger."""
    if not path.exists():
        return False
    size_mb = path.stat().st_size / (1024 * 1024)
    if expected_params > 5_000_000 and size_mb < 20:
        return False
    return True


def _load_work_agent(config: AppConfig, *, prefer_thread_bc: bool) -> DQNAgent:
    agent = DQNAgent(config)
    expected = int(agent.online.count_params())
    candidates: list[Path] = []
    if prefer_thread_bc:
        candidates.append(THREAD_BC)
    candidates.extend([LATEST, SEED_BEST])
    src = None
    for path in candidates:
        if checkpoint_exists(path) and _checkpoint_likely_compatible(path, expected):
            src = path
            break
    if src is None:
        print(
            "LOAD_WEIGHTS fresh impala (no compatible checkpoint; BC warmstart)",
            flush=True,
        )
        agent.save(LATEST)
        return agent
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
    write_thread_bc_best(mean)
    print(f"SEED_BEST_UPDATE mean={mean:.1f} -> {SEED_BEST.name}", flush=True)


def _load_bc_replay_mix(
    agent: DQNAgent,
    config: AppConfig,
    *,
    secondary_gate: float,
    elite_gate: float,
) -> int:
    """Load hybrid BC mix (preferred) or elite + optional mid-tier browser."""
    agent.replay.clear()
    hybrid = Path("checkpoints/hybrid_bc_mix.pkl")
    if hybrid.exists():
        n = agent.replay.load_pickle(
            hybrid,
            max_items=config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )
        print(f"PHASE_BC replay hybrid_mix={n} total={len(agent.replay)}", flush=True)
        if n >= 512:
            return len(agent.replay)
    elite_n = 0
    if ELITE_REPLAY.exists():
        elite_n = agent.replay.load_pickle(
            ELITE_REPLAY,
            max_items=config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )
    for extra in (
        Path("checkpoints/hybrid_human_replay.pkl"),
        Path("checkpoints/teacher_distill_replay.pkl"),
    ):
        if not extra.exists():
            continue
        aux = DQNAgent(config)
        aux.replay.clear()
        n = aux.replay.load_pickle(
            extra,
            max_items=config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )
        if n > 0:
            agent.replay.extend_from(aux.replay)
            print(f"PHASE_BC +{n} from {extra.name}", flush=True)
    secondary_n = 0
    sec_gate = float(secondary_gate)
    if sec_gate > 0 and sec_gate < elite_gate and BROWSER_REPLAY.exists():
        aux = DQNAgent(config)
        aux.replay.clear()
        loaded = aux.replay.load_pickle(
            BROWSER_REPLAY,
            max_items=config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )
        if loaded > 0:
            kept = aux.replay.filter_episode_score_range(sec_gate, elite_gate)
            cap = min(900, max(128, elite_n // 3))
            if kept > cap:
                aux.replay.trim_to(cap)
            if kept > 0:
                before = len(agent.replay)
                agent.replay.extend_from(aux.replay, max_items=cap)
                secondary_n = len(agent.replay) - before
    print(
        f"PHASE_BC replay elite={elite_n} secondary={secondary_n} "
        f"total={len(agent.replay)} sec_gate>={sec_gate:.0f}",
        flush=True,
    )
    return len(agent.replay)


def _elite_store_compatible(min_score: float) -> bool:
    if not ELITE_REPLAY.exists() or not ELITE_META.exists():
        return False
    try:
        meta = json.loads(ELITE_META.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return float(meta.get("min_episode_score", 0.0)) >= float(min_score)


def _reset_incompatible_elite_store(min_score: float) -> None:
    """Discard legacy/mixed-gate replay rather than silently training on it."""
    if not ELITE_REPLAY.exists():
        return
    if _elite_store_compatible(min_score):
        return
    size = ELITE_REPLAY.stat().st_size
    ELITE_REPLAY.unlink()
    ELITE_META.unlink(missing_ok=True)
    print(
        f"ELITE_REPLAY_REBUILD removed_incompatible_bytes={size} "
        f"required_score>={min_score:.0f}",
        flush=True,
    )


def _persist_elite_replay(config: AppConfig) -> int:
    """Merge collect replay, keep only ≥elite_gate episodes, then compact."""
    min_score = max(
        MIN_ELITE_SCORE,
        float(
            getattr(config.training, "elite_replay_min_episode_score", 0.0)
            or config.training.replay_min_episode_score
            or 0.0
        ),
    )
    max_eps = int(
        getattr(config.training, "elite_max_episodes", MAX_ELITE_EPISODES)
        or MAX_ELITE_EPISODES
    )
    max_per = int(
        getattr(
            config.training,
            "elite_max_transitions_per_episode",
            MAX_TRANSITIONS_PER_ELITE,
        )
        or MAX_TRANSITIONS_PER_ELITE
    )
    _reset_incompatible_elite_store(min_score)
    if not BROWSER_REPLAY.exists():
        if not ELITE_REPLAY.exists():
            return 0
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
        # Drop collect-only episodes below the elite gate (uses episode_score stamps).
        kept = fresh.replay.filter_min_episode_score(min_score)
        print(
            f"ELITE_FILTER browser={browser_n} kept>={min_score:.0f} -> {kept}",
            flush=True,
        )
        if kept > 0:
            merged.replay.extend_from(fresh.replay)
    compact = merged.replay.compact_diverse_episodes(
        max_episodes=max_eps,
        max_transitions_per_episode=max_per,
    )
    saved = merged.replay.save_pickle(
        ELITE_REPLAY,
        max_items=max_eps * max_per,
    )
    ELITE_META.write_text(
        json.dumps(
            {
                "min_episode_score": min_score,
                "max_episodes": max_eps,
                "max_transitions_per_episode": max_per,
                **compact,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"ELITE_REPLAY merge before={before} browser={browser_n} "
        f"episodes={compact['selected_episodes']} saved={saved} "
        f"gate>={min_score:.0f} -> {ELITE_REPLAY.name}",
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
    skill_metrics = await evaluate_skill_router_baseline(
        config, episodes=episodes, use_sim=False
    )
    print(format_skill_metrics("SKILL_CEILING", skill_metrics), flush=True)
    metrics = await evaluate_seed_thread_baseline(
        config, episodes=episodes, use_sim=False
    )
    print(format_skill_metrics("THREAD_CEILING", metrics), flush=True)
    return {
        "mean": float(skill_metrics.mean_score),
        "min": float(skill_metrics.min_score),
        "max": float(skill_metrics.max_score),
        "thread_mean": float(metrics.mean_score),
    }


async def confirm_greedy_promotion(
    config: AppConfig,
    *,
    floor: float,
    target_min: float,
    episodes: int = 16,
    label: str,
) -> dict[str, float]:
    """Run the only eval allowed to promote a checkpoint."""
    if checkpoint_exists(THREAD_BC):
        agent = DQNAgent(config)
        assert agent.load(THREAD_BC)
        agent.save(LATEST)
    config.training.seed_thread_enabled = True
    config.training.seed_thread_watch_prior = 0.0
    config.training.watch_rule_prior = 0.0
    config.training.force_save_browser_replay = False
    config.training.skip_browser_replay_load = True
    stats = await run_eval_watch(config, max_episodes=max(12, int(episodes)))
    result = {
        "mean": float(stats.get("recent_avg", 0.0)),
        "min": float(stats.get("recent_min", 0.0)),
        "max": float(stats.get("recent_max", 0.0)),
    }
    passed = confirmed_promotion(
        mean_score=result["mean"],
        min_score=result["min"],
        floor_mean=floor,
        target_min=target_min,
    )
    print(
        f"GREEDY_CONFIRM label={label} episodes={max(12, int(episodes))} "
        f"mean={result['mean']:.1f} min={result['min']:.1f} "
        f"max={result['max']:.1f} thread_floor={floor:.1f} "
        f"seed_floor={read_seed_floor():.1f} pass={int(passed)}",
        flush=True,
    )
    if passed:
        agent = DQNAgent(config)
        assert agent.load(THREAD_BC if checkpoint_exists(THREAD_BC) else LATEST)
        record_thread_bc_eval(result["mean"])
        if result["mean"] > read_seed_floor():
            _promote_seed_best(agent, result["mean"])
        else:
            _save_thread_bc(agent)
            print(
                f"THREAD_BC_BEST_UPDATE mean={result['mean']:.1f} "
                f"(seed floor {read_seed_floor():.1f} unchanged)",
                flush=True,
            )
        print(f"PROMOTE_VIA greedy_confirm:{label}", flush=True)
    return {**result, "passed": float(passed)}


def _enable_skill_teacher(config: AppConfig, prior: float) -> None:
    config.training.skills_enabled = True
    config.training.skill_teacher_prior_start = float(prior)
    config.training.skill_teacher_prior_end = float(prior)
    config.training.skill_teacher_prior_steps = 10**9
    config.training.seed_thread_prior_start = 0.05
    config.training.seed_thread_prior_end = 0.05


async def phase_skill_bootstrap(
    config: AppConfig,
    *,
    minutes: float,
    bc_steps: int,
    bc_lr: float = 2e-5,
) -> dict[str, float]:
    """Label transitions with the skill router, then BC the skill head."""
    seconds = max(120, int(minutes * 60))
    _enable_skill_teacher(config, 0.92)
    print(
        f"PHASE_SKILL_BOOTSTRAP collect_s={seconds} bc_steps={bc_steps}",
        flush=True,
    )
    await phase_collect(
        config,
        seconds=seconds,
        eps=0.02,
        thread_prior=0.05,
        skip_replay_load=True,
    )
    agent = DQNAgent(config)
    src = THREAD_BC if checkpoint_exists(THREAD_BC) else SEED_BEST
    assert agent.load(src), f"failed to load {src}"
    agent.replay.clear()
    if BROWSER_REPLAY.exists():
        agent.replay.load_pickle(
            BROWSER_REPLAY,
            max_items=config.training.browser_replay_max_items,
            vector_dim=config.observation.vector_dim,
        )
    n = len(agent.replay)
    print(f"PHASE_SKILL_BOOTSTRAP replay={n}", flush=True)
    if n < 256:
        print("PHASE_SKILL_BOOTSTRAP_SKIP replay too small", flush=True)
        return {"replay": float(n), "loss": -1.0}
    loss = agent.train_skill_head(steps=bc_steps, lr=bc_lr)
    if loss is not None:
        agent.evaluate_skill_accuracy()
    _save_thread_bc(agent)
    agent.save(LATEST)
    print(f"PHASE_SKILL_BOOTSTRAP_DONE loss={loss}", flush=True)
    return {"replay": float(n), "loss": float(loss or -1.0)}


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
        f"collect_gate>={config.training.replay_min_episode_score} "
        f"elite_gate>={config.training.elite_replay_min_episode_score} "
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
    target_min: float = 900.0,
    thread_watch_prior: float = 0.55,
    collect_recent_avg: float = 0.0,
    secondary_gate: float = DEFAULT_BC_SECONDARY_GATE,
    elite_gate: float = MIN_ELITE_SCORE,
) -> float | None:
    """BC from elite demos onto side checkpoint; revert if greedy collapses."""
    agent = DQNAgent(config)
    expected = int(agent.online.count_params())
    order = (
        (THREAD_BC, LATEST, SEED_BEST)
        if prefer_thread_bc
        else (SEED_BEST, THREAD_BC, LATEST)
    )
    base = next(
        (
            path
            for path in order
            if checkpoint_exists(path)
            and _checkpoint_likely_compatible(path, expected)
        ),
        None,
    )
    if base is None:
        print(f"PHASE_BC base=fresh steps={steps} lr={lr:.2e}", flush=True)
        agent.save(LATEST)
    else:
        assert agent.load(base), f"failed to load {base}"
        agent.save(LATEST)
        print(f"PHASE_BC base={base.name} steps={steps} lr={lr:.2e}", flush=True)

    agent.replay.clear()
    loaded = _load_bc_replay_mix(
        agent,
        config,
        secondary_gate=secondary_gate,
        elite_gate=elite_gate,
    )
    print(f"PHASE_BC load replay={loaded}", flush=True)
    # Need a meaningful elite buffer — one lucky episode is not enough.
    if loaded < 512:
        print("PHASE_BC_SKIP need >=512 compact elite transitions", flush=True)
        return None

    pre_path = Path("checkpoints/dqn_thread_bc_pre_bc.keras")
    agent.save(pre_path)

    loss = offline_bc_train(agent, steps=steps, lr=lr)
    # Refresh skill head on the same elite labels after motor BC (skill logits frozen above).
    skill_loss = agent.train_skill_head(steps=max(80, steps // 2), lr=max(lr, 1e-5))
    if skill_loss is not None:
        agent.evaluate_skill_accuracy()
    _save_thread_bc(agent)
    print(f"PHASE_BC_DONE loss={loss} skill_loss={skill_loss}", flush=True)

    # Primary keep gate: pure greedy. Aligned Watch is diagnostic only.
    config.training.seed_thread_enabled = True
    config.training.watch_rule_prior = 0.0
    config.training.skill_exec_at_watch = False
    config.training.force_save_browser_replay = False
    config.training.skip_browser_replay_load = True
    eps_n = max(4, eval_probe_episodes)
    config.training.seed_thread_watch_prior = 0.0
    greedy_probe = await run_eval_watch(config, max_episodes=eps_n)
    greedy_mean = float(greedy_probe.get("recent_avg", 0.0))
    config.training.seed_thread_watch_prior = float(thread_watch_prior)
    aligned_probe = await run_eval_watch(config, max_episodes=max(3, eps_n // 2))
    aligned_mean = float(aligned_probe.get("recent_avg", 0.0))
    promote_floor = max(850.0, thread_bc_promotion_floor())
    seed_floor = _read_best_mean()
    collect_ref = max(0.0, float(collect_recent_avg))
    greedy_min = max(
        promote_floor * 0.90,
        collect_ref * 0.75 if collect_ref > 0 else 0.0,
        800.0,
    )
    print(
        f"BC_PROBE_GREEDY mean={greedy_mean:.1f} greedy_min={greedy_min:.1f} "
        f"thread_bc_best={promote_floor:.1f} seed_floor={seed_floor:.1f}",
        flush=True,
    )
    print(
        f"BC_PROBE_ALIGNED mean={aligned_mean:.1f} "
        f"watch_prior={thread_watch_prior:.2f} (diagnostic)",
        flush=True,
    )
    if greedy_mean < greedy_min:
        agent = DQNAgent(config)
        assert agent.load(pre_path)
        _save_thread_bc(agent)
        print(
            f"BC_REVERT greedy={greedy_mean:.1f} < greedy_min={greedy_min:.1f} "
            f"— restored pre-BC weights",
            flush=True,
        )
        return None
    if greedy_mean > promote_floor:
        await confirm_greedy_promotion(
            config,
            floor=promote_floor,
            target_min=target_min,
            episodes=16,
            label="bc_probe",
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
    config.training.skill_exec_at_watch = False
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
    elite_gate = max(MIN_ELITE_SCORE, float(args.elite_gate))
    collect_gate = max(0.0, float(args.collect_gate))
    floor = _read_best_mean()
    target_mean = float(args.target_mean)
    target_min = float(args.target_min)

    config = _build_config(
        run_seed,
        elite_gate=elite_gate,
        collect_gate=collect_gate,
        elite_max_episodes=int(getattr(args, "elite_max_episodes", MAX_ELITE_EPISODES)),
    )
    bc_secondary_gate = float(getattr(args, "bc_secondary_gate", DEFAULT_BC_SECONDARY_GATE))
    _reset_incompatible_elite_store(elite_gate)
    # Bootstrap side ckpt from seed best if missing.
    if not checkpoint_exists(THREAD_BC) and checkpoint_exists(SEED_BEST):
        agent = DQNAgent(config)
        assert agent.load(SEED_BEST)
        _save_thread_bc(agent)

    print(
        f"THREAD_BC_CYCLE_START seed={run_seed} hours={args.hours} "
        f"seed_floor={floor:.1f} thread_bc_best={thread_bc_promotion_floor():.1f} "
        f"collect_gate={collect_gate:.0f} elite_gate={elite_gate} "
        f"target_mean>={target_mean} target_min>={target_min}",
        flush=True,
    )

    # ---- Phase 0: ceiling ----
    if not bool(args.skip_ceiling):
        ceiling = await phase_ceiling(
            config, episodes=max(6, int(args.ceiling_episodes))
        )
        thread_mean = float(ceiling.get("thread_mean", 0.0))
        if thread_mean > 0:
            _write_thread_floor(thread_mean)
        print(
            f"CEILING_SUMMARY mean={ceiling['mean']:.1f} "
            f"min={ceiling['min']:.1f} max={ceiling['max']:.1f} "
            f"thread_mean={thread_mean:.1f} floor={floor:.1f}",
            flush=True,
        )
    else:
        print("CEILING_SKIPPED", flush=True)

    if getattr(config.training, "skills_enabled", False) and not bool(
        getattr(args, "skip_skill_bootstrap", False)
    ):
        await phase_skill_bootstrap(
            config,
            minutes=float(getattr(args, "skill_bootstrap_minutes", 10.0)),
            bc_steps=int(getattr(args, "skill_bc_steps", 400)),
        )

    round_id = 0
    thread_prior = float(args.thread_prior)
    thread_watch = float(args.thread_watch_prior)
    bc_steps = int(args.bc_steps)
    bc_lr = float(args.bc_lr)
    collect_eps = float(args.collect_eps)
    prefer_thread_bc = checkpoint_exists(THREAD_BC)
    finetune_readiness = FineTuneReadiness(floor_ratio=0.95, required_rounds=2)

    while time.time() < deadline:
        round_id += 1
        remaining = deadline - time.time()
        if remaining < 300:
            print("TIME_LOW stopping", flush=True)
            break

        print(
            f"CYCLE_ROUND id={round_id} remaining_h={remaining/3600:.2f} "
            f"collect_gate={config.training.replay_min_episode_score} "
            f"elite_gate={config.training.elite_replay_min_episode_score} "
            f"thread_prior={thread_prior} watch={thread_watch}",
            flush=True,
        )

        # Reserve time: collect / bc / dual eval / optional FT slice
        # First rounds prioritize collect+BC; later rounds add FT.
        collect_budget = min(
            int(args.collect_minutes * 60),
            max(300, int(remaining * 0.40)),
        )

        # Split wall time: teacher collect (labels) then low-prior policy collect.
        teacher_prior = float(
            getattr(config.training, "teacher_collect_thread_prior", 0.75) or 0.75
        )
        policy_prior_cap = float(
            getattr(config.training, "policy_collect_thread_prior", 0.20) or 0.20
        )
        teacher_budget = max(120, int(collect_budget * 0.35))
        policy_budget = max(180, collect_budget - teacher_budget)
        policy_prior = min(float(thread_prior), policy_prior_cap)

        # Ensure collect plays from work weights.
        agent = _load_work_agent(config, prefer_thread_bc=prefer_thread_bc)
        agent.epsilon = collect_eps
        agent.metrics.epsilon = collect_eps
        agent.progress.epsilon = collect_eps
        agent.save(LATEST)

        print(
            f"PHASE_COLLECT_SPLIT teacher_s={teacher_budget} prior={teacher_prior:.2f} "
            f"policy_s={policy_budget} prior={policy_prior:.2f}",
            flush=True,
        )
        await phase_collect(
            config,
            seconds=teacher_budget,
            eps=min(0.03, collect_eps),
            thread_prior=teacher_prior,
            skip_replay_load=True,
        )
        # Teacher data is for distill labels; policy collect is the BC/self-play buffer.
        collect_stats = await phase_collect(
            config,
            seconds=policy_budget,
            eps=collect_eps,
            thread_prior=policy_prior,
            skip_replay_load=True,
        )
        print(
            f"COLLECT_DONE recent_avg={collect_stats.get('recent_avg', 0):.0f} "
            f"replay={collect_stats.get('replay_size', 0)}",
            flush=True,
        )
        elite_n = _persist_elite_replay(config)
        collect_replay = int(collect_stats.get("replay_size", 0) or 0)
        if collect_replay < 200 and (elite_n < 200):
            thread_prior = min(0.85, thread_prior + 0.05)
            collect_eps = min(0.08, collect_eps + 0.01)
            print(
                f"STARVE_ADJUST collect_gate={config.training.replay_min_episode_score:.0f} "
                f"elite_gate={config.training.elite_replay_min_episode_score:.0f} "
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
            target_min=target_min,
            thread_watch_prior=thread_watch,
            collect_recent_avg=float(collect_stats.get("recent_avg", 0) or 0),
            secondary_gate=bc_secondary_gate,
            elite_gate=elite_gate,
        )
        prefer_thread_bc = True
        if loss is None:
            # Accumulate more diverse ≥1600 episodes; never dilute the buffer.
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
        best_mean = greedy["mean"]
        best_min = greedy["min"]
        promote_floor = max(850.0, thread_bc_promotion_floor())
        seed_floor = _read_best_mean()
        ft_ready = finetune_readiness.observe(best_mean, promote_floor)
        record_thread_bc_eval(greedy["mean"])
        promoted = False
        if greedy["mean"] > promote_floor:
            confirmation = await confirm_greedy_promotion(
                config,
                floor=promote_floor,
                target_min=target_min,
                episodes=max(16, int(args.eval_episodes)),
                label=f"round_{round_id}",
            )
            promoted = bool(confirmation["passed"])
        if promoted:
            promote_floor = max(promote_floor, thread_bc_promotion_floor())
            seed_floor = _read_best_mean()
            bc_steps = min(900, bc_steps + 50)
            collect_eps = max(0.03, collect_eps * 0.97)
            finetune_readiness.consecutive_rounds = 0
            thread_watch = max(0.25, thread_watch * 0.9)
        else:
            print(
                f"NO_PROMOTE aligned={aligned['mean']:.1f} greedy={greedy['mean']:.1f} "
                f"thread_floor={promote_floor:.1f} seed_floor={seed_floor:.1f} "
                f"— keep side ckpt + elite replay",
                flush=True,
            )
            if best_mean < promote_floor * 0.90:
                thread_prior = min(0.85, thread_prior + 0.03)
                collect_eps = min(0.08, collect_eps + 0.01)
                thread_watch = max(0.30, thread_watch * 0.92)
            else:
                thread_watch = max(0.35, min(thread_watch, 0.45))

        if promoted and seed_floor >= target_mean:
            print(
                f"TARGET_MET confirmed_mean={seed_floor:.1f}",
                flush=True,
            )
            return 0

        remaining = deadline - time.time()
        ft_min = float(
            getattr(config.training, "finetune_min_greedy_mean", 1400.0) or 1400.0
        )
        allow_ft = best_mean >= ft_min and thread_bc_promotion_floor() >= ft_min
        if (
            remaining >= 45 * 60
            and ft_ready
            and allow_ft
            and float(args.finetune_minutes) > 0
        ):
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
                gate=max(MIN_ELITE_SCORE, config.training.replay_min_episode_score),
            )
            remaining = deadline - time.time()
            if remaining >= 200:
                evals = await phase_dual_eval(
                    config,
                    episodes=max(6, int(args.eval_episodes)),
                    thread_watch_prior=thread_watch,
                )
                aligned = evals["aligned"]
                greedy = evals["greedy"]
                promote_floor = max(850.0, thread_bc_promotion_floor())
                if greedy["mean"] > promote_floor:
                    confirmation = await confirm_greedy_promotion(
                        config,
                        floor=promote_floor,
                        target_min=target_min,
                        episodes=max(16, int(args.eval_episodes)),
                        label=f"post_ft_round_{round_id}",
                    )
                    if confirmation["passed"]:
                        seed_floor = _read_best_mean()
                if pre_ft.exists() and greedy["mean"] < promote_floor * 0.95:
                    agent = DQNAgent(config)
                    assert agent.load(pre_ft)
                    _save_thread_bc(agent)
                    print(
                        f"FT_REVERT greedy={greedy['mean']:.1f} — restored pre-FT side ckpt",
                        flush=True,
                    )
        elif remaining >= 45 * 60 and (not ft_ready or not allow_ft):
            reason = (
                f"greedy={best_mean:.1f}<ft_min={ft_min:.1f}"
                if not allow_ft
                else (
                    f"streak={finetune_readiness.consecutive_rounds}/"
                    f"{finetune_readiness.required_rounds} at >=0.95*thread_floor "
                    f"({promote_floor * 0.95:.1f})"
                )
            )
            print(
                f"FT_SKIP {reason} — more elite BC first",
                flush=True,
            )

    print(
        f"THREAD_BC_CYCLE_END seed_floor={_read_best_mean():.1f} "
        f"thread_bc_best={thread_bc_promotion_floor():.1f} "
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
    parser.add_argument("--target-min", type=float, default=2000.0)
    parser.add_argument(
        "--collect-gate",
        type=float,
        default=DEFAULT_COLLECT_GATE,
        help="Episode score gate for normal collect replay volume.",
    )
    parser.add_argument(
        "--elite-gate",
        type=float,
        default=MIN_ELITE_SCORE,
        help="Episode score gate for elite BC (values below 2000 are clamped).",
    )
    parser.add_argument("--thread-prior", type=float, default=0.75)
    parser.add_argument("--thread-watch-prior", type=float, default=0.55)
    parser.add_argument("--collect-minutes", type=float, default=18.0)
    parser.add_argument("--collect-eps", type=float, default=0.04)
    parser.add_argument("--bc-steps", type=int, default=500)
    parser.add_argument("--bc-lr", type=float, default=3e-6)
    parser.add_argument(
        "--bc-secondary-gate",
        type=float,
        default=DEFAULT_BC_SECONDARY_GATE,
        help="Mix browser episodes in [gate, elite_gate) into offline BC.",
    )
    parser.add_argument(
        "--elite-max-episodes",
        type=int,
        default=MAX_ELITE_EPISODES,
        help="Max diverse elite episodes in elite_thread_replay.pkl.",
    )
    parser.add_argument(
        "--finetune-minutes",
        type=float,
        default=0.0,
        help="Online FT minutes (0 until greedy ≥ finetune_min_greedy_mean).",
    )
    parser.add_argument("--ft-lr", type=float, default=2e-5)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--ceiling-episodes", type=int, default=8)
    parser.add_argument(
        "--skip-ceiling",
        action="store_true",
        help="Skip pure-thread ceiling phase (use when resuming mid-window).",
    )
    parser.add_argument(
        "--skip-skill-bootstrap",
        action="store_true",
        help="Skip skill-router demo collect + skill-head BC at cycle start.",
    )
    parser.add_argument("--skill-bootstrap-minutes", type=float, default=10.0)
    parser.add_argument("--skill-bc-steps", type=int, default=400)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

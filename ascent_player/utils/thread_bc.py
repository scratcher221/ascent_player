"""Thread-BC cycle state, round plans, and phase helpers."""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from ascent_player.agent.checkpoint import checkpoint_exists
from ascent_player.agent.checkpoint_guard import (
    is_likely_nature_checkpoint,
    save_guarded,
)
from ascent_player.agent.dqn import DQNAgent
from ascent_player.agent.elite_replay import (
    DEFAULT_BROWSER,
    DEFAULT_MAX_EPISODES,
    DEFAULT_MAX_PER_EPISODE,
    DEFAULT_META,
    DEFAULT_MIN_SCORE,
    DEFAULT_REPLAY,
    EliteReplayStore,
)
from ascent_player.agent.offline_bc import offline_bc_frozen_trunk
from ascent_player.agent.seed_curriculum import confirmed_promotion
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.evaluation import (
    evaluate_seed_thread_baseline,
    evaluate_skill_router_baseline,
    format_skill_metrics,
)
from ascent_player.training import run_eval_watch, run_training_no_ui
from ascent_player.utils.policy_floors import (
    read_seed_floor,
    record_thread_bc_eval,
    thread_bc_promotion_floor,
    write_seed_floor,
    write_thread_bc_best,
)
from ascent_player.utils.run_profile import collect_only_fields, override_attrs, override_training

SEED_BEST = Path("checkpoints/seed_map_best.keras")
THREAD_WATCH_FLOOR = Path("logs/thread_watch_floor.txt")
LATEST = Path("checkpoints/dqn_latest.keras")
THREAD_BC = Path("checkpoints/dqn_thread_bc.keras")
ELITE_REPLAY = DEFAULT_REPLAY
ELITE_META = DEFAULT_META
BROWSER_REPLAY = DEFAULT_BROWSER
MIN_ELITE_SCORE = DEFAULT_MIN_SCORE
DEFAULT_COLLECT_GATE = 1200.0
MAX_ELITE_EPISODES = DEFAULT_MAX_EPISODES
MAX_TRANSITIONS_PER_ELITE = DEFAULT_MAX_PER_EPISODE
DEFAULT_BC_SECONDARY_GATE = 1800.0
HYBRID_BC = Path("checkpoints/hybrid_bc_mix.pkl")
HYBRID_HUMAN = Path("checkpoints/hybrid_human_replay.pkl")
TEACHER_REPLAY = Path("checkpoints/teacher_distill_replay.pkl")


@dataclass
class CycleState:
    deadline: float
    elite_gate: float
    collect_gate: float
    target_mean: float
    target_min: float
    bc_secondary_gate: float = DEFAULT_BC_SECONDARY_GATE
    round_id: int = 0
    thread_prior: float = 0.75
    thread_watch: float = 0.55
    bc_steps: int = 500
    collect_eps: float = 0.04
    prefer_thread_bc: bool = False

    @property
    def remaining(self) -> float:
        return self.deadline - time.time()


@dataclass(frozen=True)
class CollectSplit:
    teacher_seconds: int
    policy_seconds: int
    teacher_prior: float
    policy_prior: float


@dataclass(frozen=True)
class FineTuneDecision:
    allowed: bool
    reason: str
    seconds: int = 0


def plan_collect_split(
    state: CycleState,
    config: AppConfig,
    collect_minutes: float,
) -> CollectSplit:
    collect_budget = min(
        int(collect_minutes * 60),
        max(300, int(state.remaining * 0.40)),
    )
    teacher_prior = float(
        getattr(config.training, "teacher_collect_thread_prior", 0.75) or 0.75
    )
    policy_prior_cap = float(
        getattr(config.training, "policy_collect_thread_prior", 0.20) or 0.20
    )
    teacher_budget = max(120, int(collect_budget * 0.35))
    policy_budget = max(180, collect_budget - teacher_budget)
    return CollectSplit(
        teacher_seconds=teacher_budget,
        policy_seconds=policy_budget,
        teacher_prior=teacher_prior,
        policy_prior=min(float(state.thread_prior), policy_prior_cap),
    )


def plan_finetune(
    state: CycleState,
    *,
    best_mean: float,
    ft_ready: bool,
    ft_minutes: float,
    ft_min: float,
) -> FineTuneDecision:
    remaining = state.remaining
    allow_ft = best_mean >= ft_min and thread_bc_promotion_floor() >= ft_min
    if remaining < 45 * 60:
        return FineTuneDecision(False, "low_remaining")
    if ft_minutes <= 0:
        return FineTuneDecision(False, "ft_minutes=0")
    if not allow_ft:
        return FineTuneDecision(
            False, f"greedy={best_mean:.1f}<ft_min={ft_min:.1f}"
        )
    if not ft_ready:
        return FineTuneDecision(False, "not_ft_ready")
    seconds = min(int(ft_minutes * 60), max(600, int(remaining * 0.35)))
    return FineTuneDecision(True, "ok", seconds=seconds)


def write_thread_floor(mean: float) -> None:
    THREAD_WATCH_FLOOR.parent.mkdir(parents=True, exist_ok=True)
    THREAD_WATCH_FLOOR.write_text(f"{mean:.4f}\n", encoding="utf-8")


def build_config(
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
    config.training.seed_thread_prior_start = 0.12
    config.training.seed_thread_prior_end = 0.04
    config.training.seed_thread_watch_prior = 0.0
    config.training.watch_rule_prior = 0.0
    return config


def elite_store_for(config: AppConfig, min_score: float) -> EliteReplayStore:
    return EliteReplayStore(
        min_score=max(
            MIN_ELITE_SCORE,
            float(min_score),
        ),
        max_episodes=int(
            getattr(config.training, "elite_max_episodes", MAX_ELITE_EPISODES)
            or MAX_ELITE_EPISODES
        ),
        max_per_episode=int(
            getattr(
                config.training,
                "elite_max_transitions_per_episode",
                MAX_TRANSITIONS_PER_ELITE,
            )
            or MAX_TRANSITIONS_PER_ELITE
        ),
    )


def checkpoint_likely_compatible(path: Path, expected_params: int) -> bool:
    if not path.exists():
        return False
    if expected_params > 5_000_000 and is_likely_nature_checkpoint(path):
        return False
    return True


def load_work_agent(config: AppConfig, *, prefer_thread_bc: bool) -> DQNAgent:
    agent = DQNAgent(config)
    expected = int(agent.online.count_params())
    candidates: list[Path] = []
    if prefer_thread_bc:
        candidates.append(THREAD_BC)
    candidates.extend([LATEST, SEED_BEST])
    src = None
    for path in candidates:
        if checkpoint_exists(path) and checkpoint_likely_compatible(path, expected):
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


def save_thread_bc(agent: DQNAgent) -> None:
    result = save_guarded(agent, THREAD_BC)
    agent.save(LATEST)
    if result.rejected:
        print(
            f"SAVE_THREAD_BC rejected size={result.size_mb:.1f}MB — kept previous",
            flush=True,
        )
        return
    print(f"SAVE_THREAD_BC -> {THREAD_BC.name}", flush=True)


def promote_seed_best(agent: DQNAgent, mean: float) -> None:
    seed_res = save_guarded(agent, SEED_BEST)
    thread_res = save_guarded(agent, THREAD_BC)
    agent.save(LATEST)
    if seed_res.rejected or thread_res.rejected:
        print(
            f"SEED_BEST_HOLD save_rejected seed={seed_res.rejected} "
            f"thread={thread_res.rejected} mean={mean:.1f}",
            flush=True,
        )
        return
    write_seed_floor(mean)
    write_thread_bc_best(mean)
    print(f"SEED_BEST_UPDATE mean={mean:.1f} -> {SEED_BEST.name}", flush=True)


def load_bc_replay_mix(
    agent: DQNAgent,
    config: AppConfig,
    *,
    secondary_gate: float,
    elite_gate: float,
) -> int:
    agent.replay.clear()
    if HYBRID_BC.exists():
        n = agent.replay.load_pickle(
            HYBRID_BC,
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
    for extra in (HYBRID_HUMAN, TEACHER_REPLAY):
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
    if checkpoint_exists(THREAD_BC):
        agent = DQNAgent(config)
        assert agent.load(THREAD_BC)
        agent.save(LATEST)
    with override_training(
        config,
        seed_thread_enabled=True,
        seed_thread_watch_prior=0.0,
        watch_rule_prior=0.0,
        force_save_browser_replay=False,
        skip_browser_replay_load=True,
    ):
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
            promote_seed_best(agent, result["mean"])
        else:
            save_thread_bc(agent)
            print(
                f"THREAD_BC_BEST_UPDATE mean={result['mean']:.1f} "
                f"(seed floor {read_seed_floor():.1f} unchanged)",
                flush=True,
            )
        print(f"PROMOTE_VIA greedy_confirm:{label}", flush=True)
    return {**result, "passed": float(passed)}


async def phase_skill_bootstrap(
    config: AppConfig,
    *,
    minutes: float,
    bc_steps: int,
    bc_lr: float = 2e-5,
) -> dict[str, float]:
    seconds = max(120, int(minutes * 60))
    print(
        f"PHASE_SKILL_BOOTSTRAP collect_s={seconds} bc_steps={bc_steps}",
        flush=True,
    )
    with override_training(
        config,
        skills_enabled=True,
        skill_teacher_prior_start=0.92,
        skill_teacher_prior_end=0.92,
        seed_thread_prior_start=0.05,
        seed_thread_prior_end=0.05,
    ):
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
    save_thread_bc(agent)
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
    print(
        f"PHASE_COLLECT s={seconds} eps={eps} thread_prior={thread_prior} "
        f"collect_gate>={config.training.replay_min_episode_score} "
        f"elite_gate>={config.training.elite_replay_min_episode_score} "
        f"skip_load={skip_replay_load}",
        flush=True,
    )
    with override_training(
        config,
        **collect_only_fields(
            eps=eps,
            skip_replay_load=skip_replay_load,
            thread_prior=thread_prior,
            rule_prior=0.05,
        ),
        learning_rate=1e-5,
        seed_thread_watch_prior=0.0,
        seed_thread_corridor=0.16,
        seed_thread_only_when_landing=True,
    ), override_attrs(config.demo, use_demos_on_start=False):
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
            and checkpoint_likely_compatible(path, expected)
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

    loaded = load_bc_replay_mix(
        agent,
        config,
        secondary_gate=secondary_gate,
        elite_gate=elite_gate,
    )
    print(f"PHASE_BC load replay={loaded}", flush=True)
    if loaded < 512:
        print("PHASE_BC_SKIP need >=512 compact elite transitions", flush=True)
        return None

    pre_path = Path("checkpoints/dqn_thread_bc_pre_bc.keras")
    agent.save(pre_path)
    loss = offline_bc_frozen_trunk(
        agent,
        steps=steps,
        lr=lr,
        save_path=LATEST,
        log_prefix="OFFLINE_BC",
    )
    skill_loss = agent.train_skill_head(steps=max(80, steps // 2), lr=max(lr, 1e-5))
    if skill_loss is not None:
        agent.evaluate_skill_accuracy()
    save_thread_bc(agent)
    print(f"PHASE_BC_DONE loss={loss} skill_loss={skill_loss}", flush=True)

    eps_n = max(4, eval_probe_episodes)
    with override_training(
        config,
        seed_thread_enabled=True,
        watch_rule_prior=0.0,
        skill_exec_at_watch=False,
        force_save_browser_replay=False,
        skip_browser_replay_load=True,
        seed_thread_watch_prior=0.0,
    ):
        greedy_probe = await run_eval_watch(config, max_episodes=eps_n)
        greedy_mean = float(greedy_probe.get("recent_avg", 0.0))
        config.training.seed_thread_watch_prior = float(thread_watch_prior)
        aligned_probe = await run_eval_watch(config, max_episodes=max(3, eps_n // 2))
        aligned_mean = float(aligned_probe.get("recent_avg", 0.0))
    promote_floor = max(850.0, thread_bc_promotion_floor())
    seed_floor = read_seed_floor()
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
        save_thread_bc(agent)
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
    if checkpoint_exists(THREAD_BC):
        agent = DQNAgent(config)
        assert agent.load(THREAD_BC)
        agent.save(LATEST)
    results: dict[str, dict[str, float]] = {}
    with override_training(
        config,
        seed_thread_enabled=True,
        seed_thread_watch_prior=0.0,
        watch_rule_prior=0.0,
        skill_exec_at_watch=False,
        force_save_browser_replay=False,
        skip_browser_replay_load=True,
    ):
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
    agent = load_work_agent(config, prefer_thread_bc=True)
    agent.epsilon = eps
    agent.metrics.epsilon = eps
    agent.progress.epsilon = eps
    agent.save(LATEST)
    if ELITE_REPLAY.exists():
        shutil.copy2(ELITE_REPLAY, BROWSER_REPLAY)
    print(
        f"PHASE_FINETUNE s={seconds} lr={lr:.2e} eps={eps} "
        f"thread={thread_prior} gate>={gate}",
        flush=True,
    )
    with override_training(
        config,
        watch_mode=False,
        disable_td=False,
        min_replay_size=1000,
        train_every_gpu=4,
        train_every_cpu=4,
        transfer_epsilon_start=eps,
        browser_epsilon_cap=eps,
        browser_epsilon_floor=min(0.03, eps),
        rule_prior_start=0.05,
        rule_prior_end=0.02,
        rule_prior_steps=40_000,
        learning_rate=lr,
        replay_min_episode_score=float(gate),
        force_save_browser_replay=True,
        skip_browser_replay_load=False,
        sim_warmstart_teacher=False,
        sim_warmstart_demos=False,
        seed_thread_enabled=True,
        seed_thread_prior_start=float(thread_prior),
        seed_thread_prior_end=float(thread_prior),
        seed_thread_watch_prior=float(thread_prior),
        seed_thread_corridor=0.16,
        seed_thread_only_when_landing=True,
    ), override_attrs(config.demo, use_demos_on_start=False):
        stats = await run_training_no_ui(
            config,
            max_seconds=seconds,
            ingest_demos=False,
        )
    agent = DQNAgent(config)
    if agent.load(LATEST):
        save_thread_bc(agent)
    elite_store_for(config, float(gate)).persist_from_browser(config)
    return stats

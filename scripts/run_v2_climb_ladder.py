#!/usr/bin/env python3
"""Offline-heavy v2 climb ladder (Impala-mid only).

Ladder targets (greedy 100-ep mean): 1400 → 1600 → 1800 → 2000.

Phases per round:
  1. Offline BC on gated browser/elite (no browser) — skip hybrid while collecting
  2. Optional short policy collect (locked seed, thread prior ≤0.2, no TD below 1400)
  3. Frozen-trunk BC on ≥1000 / ≥1400 replay; offline TD only after true mean ≥1400
  4. Full-N greedy probe (20) + 40-ep confirm; reliability every N rounds
  5. Promote only on confirmed full-N gains; online FT off until true mean ≥1400

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
from contextlib import nullcontext

os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ascent_player.agent.checkpoint import checkpoint_exists
from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.evaluation import evaluate_learned_policy
from ascent_player.training import run_eval_watch, run_training_no_ui
from ascent_player.agent.checkpoint_guard import (
    is_impala_sized_checkpoint,
    is_likely_nature_checkpoint,
    restore_work_checkpoints_from_best,
    save_guarded,
)
from ascent_player.agent.offline_bc import offline_bc_frozen_trunk
from ascent_player.utils.policy_floors import (
    live_v2_floor,
    read_v2_full_n_last,
    read_v2_probe_best,
    seed_v2_full_n_best,
    thread_bc_promotion_floor,
    write_v2_full_n_last,
)
from ascent_player.utils.climb_phases import (
    harvest_elite_after_collect,
    resolve_probe_and_persist,
    run_browser_bc_and_td,
    run_reliability,
)
from ascent_player.utils.climb_policy import (
    collect_replay_gate,
    collect_thread_prior,
    current_rung,
    elite_gate_for_score,
    online_td_gate,
    record_climb_session,
    should_defer_offline_td,
    should_run_online_td,
    should_skip_leftover_hybrid_bc,
    PHASE2_STALL_PATH,
)
from ascent_player.utils.run_profile import (
    collect_only_fields,
    locked_eval_seed,
    override_attrs,
    override_training,
)
from climb_args import add_climb_args

LATEST = Path("checkpoints/dqn_latest.keras")
THREAD_BC = Path("checkpoints/dqn_thread_bc.keras")
V2_BEST = Path("checkpoints/dqn_v2_best.keras")
ELITE_REPLAY = Path("checkpoints/elite_thread_replay.pkl")
HYBRID_BC = Path("checkpoints/hybrid_bc_mix.pkl")
TEACHER_REPLAY = Path("checkpoints/teacher_distill_replay.pkl")
BROWSER_REPLAY = Path("checkpoints/browser_replay.pkl")
LADDER_LOG = Path("logs/v2_climb_ladder.jsonl")


def _build_config(run_seed: int, *, lock_run_seed: bool = True) -> AppConfig:
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
    # Aux CE distracts Q climb until ≥2k; re-enable later in-loop.
    config.training.reason_aux_weight = 0.0
    config.training.skill_aux_weight = 0.0
    config.training.seed_thread_enabled = True
    config.training.watch_rule_prior = 0.0
    # Never run sim teacher/demo warmstart in browser climb sessions — it stalls.
    config.training.sim_warmstart_teacher = False
    config.training.sim_warmstart_demos = False
    config.demo.use_demos_on_start = False
    if lock_run_seed:
        config.browser.run_seed = int(run_seed)
        config.browser.lock_run_seed = True
    else:
        # Random maps for collect/train; probes re-lock eval seed separately.
        config.browser.run_seed = None
        config.browser.lock_run_seed = False
    return config


def _append_log(row: dict) -> None:
    LADDER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with LADDER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _current_rung(mean: float) -> float:
    return current_rung(mean)


def _load_work_agent(config: AppConfig) -> DQNAgent:
    """Load v2 weights; prefer V2_BEST when present, else thread_bc / latest."""
    restored = restore_work_checkpoints_from_best(
        V2_BEST, (THREAD_BC, LATEST)
    )
    if restored:
        print(f"CLIMB_RESTORE_WORK from {V2_BEST.name} -> {restored}", flush=True)
    agent = DQNAgent(config)
    expected = int(agent.online.count_params())
    for path in (V2_BEST, THREAD_BC, LATEST):
        if not checkpoint_exists(path):
            continue
        if expected > 5_000_000 and is_likely_nature_checkpoint(path):
            print(
                f"CLIMB_SKIP {path.name} size={path.stat().st_size / (1024*1024):.1f}MB "
                f"(likely Nature; prefer clean v2 + BC warmstart)",
                flush=True,
            )
            continue
        if agent.load(path):
            print(f"CLIMB_LOAD {path.name}", flush=True)
            agent.save(LATEST)
            if not is_impala_sized_checkpoint(THREAD_BC):
                save_guarded(agent, THREAD_BC)
            return agent
    print("CLIMB_LOAD fresh impala_mid (clean warmstart)", flush=True)
    agent.save(LATEST)
    return agent


def _load_offline_replay(agent: DQNAgent, config: AppConfig) -> int:
    """Prefer Impala-native browser → elite → teacher; hybrid last and capped."""
    agent.replay.clear()
    loaded = 0
    sources = (
        (BROWSER_REPLAY, "browser_replay.pkl"),
        (ELITE_REPLAY, "elite_thread_replay.pkl"),
        (TEACHER_REPLAY, "teacher_distill_replay.pkl"),
        (HYBRID_BC, "hybrid_bc_mix.pkl"),
    )
    for path, label in sources:
        if not path.exists() or path.stat().st_size < 64:
            continue
        if label == "hybrid_bc_mix.pkl" and loaded >= 512:
            break
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
            print(f"CLIMB_REPLAY +{n} from {label}", flush=True)
        if loaded >= 2048 and label != "hybrid_bc_mix.pkl":
            break
        if loaded >= 512 and label == "hybrid_bc_mix.pkl":
            break
    return int(len(agent.replay))


async def greedy_probe(
    config: AppConfig, episodes: int, *, eval_seed: int | None = None
) -> float:
    seed_cm = (
        locked_eval_seed(config, int(eval_seed))
        if eval_seed is not None
        else nullcontext()
    )
    with seed_cm, override_training(
        config,
        seed_thread_watch_prior=0.0,
        watch_rule_prior=0.0,
        skill_exec_at_watch=False,
        watch_mode=True,
        force_save_browser_replay=False,
        skip_browser_replay_load=True,
        checkpoint_path=LATEST,
    ):
        stats = await run_eval_watch(config, max_episodes=max(4, episodes))
        full = float(stats.get("episode_mean") or stats.get("recent_avg") or 0.0)
        greedy_probe.last_max = float(stats.get("episode_max") or 0.0)
        print(
            f"CLIMB_PROBE_WINDOW n={int(stats.get('n_episodes') or episodes)} "
            f"full_n={full:.1f} last10={float(stats.get('recent_avg') or 0.0):.1f} "
            f"min={float(stats.get('episode_min') or 0.0):.1f} "
            f"max={float(stats.get('episode_max') or 0.0):.1f}",
            flush=True,
        )
        return full


async def reliability_100(
    config: AppConfig,
    episodes: int,
    *,
    ckpt: Path | None = None,
    eval_seed: int | None = None,
) -> dict[str, float]:
    path = ckpt if ckpt is not None else (
        V2_BEST if checkpoint_exists(V2_BEST) else (
            THREAD_BC if checkpoint_exists(THREAD_BC) else LATEST
        )
    )
    seed_cm = (
        locked_eval_seed(config, int(eval_seed))
        if eval_seed is not None
        else nullcontext()
    )
    with seed_cm, override_training(
        config,
        checkpoint_path=path,
        seed_thread_watch_prior=0.0,
        skill_exec_at_watch=False,
    ):
        result = await evaluate_learned_policy(config, episodes=episodes, use_sim=False)
    mean = float(result.mean_score)
    reliability_100.last_max = float(result.max_score)
    write_v2_full_n_last(mean)
    return {
        "mean": mean,
        "min": float(result.min_score),
        "max": float(result.max_score),
        "p10": float(result.p10),
        "p50": float(result.p50),
        "p90": float(result.p90),
    }


async def policy_collect(
    config: AppConfig,
    *,
    seconds: int,
    prior: float,
    eps: float,
    online_td: bool = False,
    transfer_lr: float = 3e-5,
) -> dict[str, float]:
    fields = collect_only_fields(eps=eps, skip_replay_load=True, thread_prior=prior)
    fields.update(
        checkpoint_path=LATEST,
        replay_min_episode_score=float(
            getattr(config.training, "replay_min_episode_score", 1000.0) or 1000.0
        ),
        elite_replay_min_episode_score=float(
            getattr(config.training, "elite_replay_min_episode_score", 1400.0)
            or 1400.0
        ),
    )
    if online_td:
        fields.update(
            disable_td=False,
            min_replay_size=256,
            train_every_gpu=4,
            train_every_cpu=4,
            transfer_learning_rate=float(transfer_lr),
            learning_rate=float(transfer_lr),
        )
        print(
            f"CLIMB_POLICY_COLLECT_ONLINE_TD s={seconds} prior={prior:.2f} "
            f"eps={eps} lr={transfer_lr:.1e}",
            flush=True,
        )
    else:
        print(f"CLIMB_POLICY_COLLECT s={seconds} prior={prior:.2f} eps={eps}", flush=True)
    with override_training(config, **fields), override_attrs(
        config.demo, use_demos_on_start=False
    ):
        return await run_training_no_ui(
            config, max_seconds=seconds, ingest_demos=False
        )


async def main_async(args: argparse.Namespace) -> int:
    config = _build_config(
        int(args.run_seed),
        lock_run_seed=bool(getattr(args, "lock_run_seed", True)),
    )
    eval_seed = int(getattr(args, "eval_seed", args.run_seed) or args.run_seed)
    deadline = time.time() + float(args.hours) * 3600.0
    agent = _load_work_agent(config)
    best_mean = thread_bc_promotion_floor()
    stale_probe = read_v2_probe_best()
    v2_best = live_v2_floor()
    if v2_best <= 0:
        # Do not promote against the last-10 artifact; wait for a full-N seed.
        print(
            f"CLIMB_FULL_N_UNSEEDED last10_artifact={stale_probe:.1f} "
            f"— promote bar 0 until baseline/confirm writes full-N",
            flush=True,
        )
    true_mean = read_v2_full_n_last(default=v2_best)
    start_true_mean = true_mean
    session_max = 0.0
    session_greedy_max = 0.0
    round_idx = 0
    reliability_every = max(1, int(args.reliability_every))
    collect_prior = collect_thread_prior(
        float(getattr(config.training, "policy_collect_thread_prior", 0.20) or 0.20)
    )
    config.training.elite_replay_min_episode_score = elite_gate_for_score(v2_best)
    config.training.replay_min_episode_score = collect_replay_gate(true_mean)

    print(
        f"CLIMB_START variant=impala_mid best={best_mean:.1f} "
        f"v2_best={v2_best:.1f} last10_artifact={stale_probe:.1f} "
        f"true_mean={true_mean:.1f} hours={args.hours} "
        f"offline_only={int(args.offline_only)} collect_min={args.collect_minutes} "
        f"lock_seed={int(config.browser.lock_run_seed)} eval_seed={eval_seed} "
        f"collect_gate={config.training.replay_min_episode_score:.0f} "
        f"elite_gate={config.training.elite_replay_min_episode_score:.0f} "
        f"collect_prior={collect_prior:.2f} "
        f"wait_energy={int(config.mechanics_reward.wait_for_energy_enabled)} "
        f"ft_gate={online_td_gate(float(getattr(config.training, 'finetune_min_greedy_mean', 1400.0) or 1400.0)):.0f} "
        f"phase2_stall={int(PHASE2_STALL_PATH.exists())}",
        flush=True,
    )

    if v2_best <= 0 and checkpoint_exists(V2_BEST) and not args.offline_only:
        print("CLIMB_SEED_FULL_N n=40 on V2_BEST (last-10 floor ignored)", flush=True)
        rel = await reliability_100(
            config, 40, ckpt=V2_BEST, eval_seed=eval_seed
        )
        seed_v2_full_n_best(rel["mean"])
        v2_best = rel["mean"]
        true_mean = rel["mean"]
        start_true_mean = true_mean
        session_max = max(session_max, float(rel.get("max") or 0.0))
        config.training.replay_min_episode_score = collect_replay_gate(true_mean)
        config.training.elite_replay_min_episode_score = elite_gate_for_score(v2_best)
        print(
            f"CLIMB_FULL_N_SEEDED mean={v2_best:.1f} min={rel['min']:.1f} "
            f"max={rel['max']:.1f}",
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
        ft_gate = online_td_gate(
            float(
                getattr(config.training, "finetune_min_greedy_mean", 1400.0)
                or 1400.0
            )
        )
        true_mean = read_v2_full_n_last(default=v2_best)
        config.training.replay_min_episode_score = collect_replay_gate(true_mean)
        config.training.elite_replay_min_episode_score = elite_gate_for_score(
            max(v2_best, true_mean)
        )
        online_td = should_run_online_td(
            collecting=collecting, true_mean=true_mean, ft_gate=ft_gate
        )
        if collecting and bc_steps > 0:
            print(
                f"CLIMB_BC_SKIP pre-collect hybrid "
                f"(v2_best={v2_best:.1f}; browser/elite BC after collect)",
                flush=True,
            )
            bc_steps = 0
        elif should_skip_leftover_hybrid_bc(
            collecting=collecting, collect_minutes=float(args.collect_minutes)
        ):
            if bc_steps > 0:
                print(
                    "CLIMB_BC_SKIP leftover hybrid (collect window closed)",
                    flush=True,
                )
            bc_steps = 0
        if online_td:
            print(
                f"CLIMB_ONLINE_TD gate={ft_gate:.0f} true_mean={true_mean:.1f} "
                f"stall={int(PHASE2_STALL_PATH.exists())}",
                flush=True,
            )
        loss = offline_bc_frozen_trunk(
            agent,
            steps=bc_steps,
            lr=float(args.bc_lr),
            save_path=LATEST,
            log_prefix="CLIMB_BC",
        )

        if collecting:
            collect_s = min(int(args.collect_minutes * 60), max(300, int(remaining * 0.25)))
            if collect_s >= 60:
                assert agent.load(pre)
                agent.save(LATEST)
                collect_stats = await policy_collect(
                    config,
                    seconds=collect_s,
                    prior=collect_prior,
                    eps=float(args.collect_eps),
                    online_td=online_td,
                    transfer_lr=float(args.td_lr) if online_td else 3e-5,
                )
                session_max = max(
                    session_max, float(collect_stats.get("best_score") or 0.0)
                )
                harvest_elite_after_collect(
                    config,
                    true_mean=true_mean,
                    collect_best=float(collect_stats.get("best_score") or 0.0),
                )
                if not online_td:
                    assert agent.load(pre)
                    agent.save(LATEST)
                else:
                    print("CLIMB_KEEP_ONLINE_TD_WEIGHTS for probe", flush=True)
                browser_bc_steps = int(getattr(args, "browser_bc_steps", 0) or 0)
                if online_td:
                    if browser_bc_steps > 0:
                        print(
                            "CLIMB_BROWSER_BC_SKIP after online TD "
                            "(keep on-policy weights)",
                            flush=True,
                        )
                    browser_bc_steps = 0
                elif browser_bc_steps <= 0:
                    browser_bc_steps = max(400, int(args.bc_steps) // 2 or 400)
                td_steps = int(args.td_steps)
                if should_defer_offline_td(true_mean, gate=ft_gate) and not online_td:
                    td_steps = 0
                loss = run_browser_bc_and_td(
                    agent,
                    config,
                    browser_replay=BROWSER_REPLAY,
                    elite_replay=ELITE_REPLAY,
                    latest=LATEST,
                    browser_bc_steps=browser_bc_steps,
                    bc_lr=float(args.bc_lr),
                    td_steps=td_steps,
                    td_lr=float(args.td_lr),
                    online_td=online_td,
                    v2_best=true_mean,
                    best_mean=max(best_mean, true_mean),
                    load_offline_replay=_load_offline_replay,
                )
            else:
                print("CLIMB_COLLECT_SKIP collect_s<60", flush=True)
        elif float(args.collect_minutes) <= 0:
            print("CLIMB_COLLECT_SKIP collect_minutes=0 (offline BC + probe)", flush=True)

        if args.offline_only and args.skip_browser_probe:
            probe_mean = best_mean
            print("CLIMB_PROBE skipped (offline-only)", flush=True)
            promoted = False
        else:
            agent.save(LATEST)
            probe_mean = await greedy_probe(
                config, int(args.probe_episodes), eval_seed=eval_seed
            )
            print(
                f"CLIMB_PROBE_GREEDY mean={probe_mean:.1f} "
                f"v2_best={v2_best:.1f} nature_floor={best_mean:.1f}",
                flush=True,
            )
            session_max = max(session_max, probe_mean)
            session_greedy_max = max(
                session_greedy_max,
                float(getattr(greedy_probe, "last_max", 0.0) or 0.0),
                float(probe_mean),
            )
            v2_best, best_mean, promoted = await resolve_probe_and_persist(
                agent=agent,
                config=config,
                probe_mean=probe_mean,
                v2_best=v2_best,
                best_mean=best_mean,
                bootstrap_floor=float(args.bootstrap_keep_floor),
                confirm_eps=int(getattr(args, "confirm_episodes", 0) or 0),
                probe_eps=int(args.probe_episodes),
                greedy_probe_fn=greedy_probe,
                eval_seed=eval_seed,
                pre=pre,
                latest=LATEST,
                thread_bc=THREAD_BC,
                v2_best_path=V2_BEST,
                round_idx=round_idx,
                bc_loss=loss,
                append_log=_append_log,
            )
            if not promoted:
                if (
                    round_idx % max(1, int(args.reliability_every)) == 0
                    and not args.skip_reliability
                    and remaining >= 25 * 60
                ):
                    await run_reliability(
                        reliability_fn=reliability_100,
                        config=config,
                        episodes=int(args.reliability_episodes),
                        eval_seed=eval_seed,
                        round_idx=round_idx,
                        append_log=_append_log,
                        ckpt=V2_BEST,
                        warn_vs_v2=v2_best,
                    )
                    session_greedy_max = max(
                        session_greedy_max,
                        float(getattr(reliability_100, "last_max", 0.0) or 0.0),
                    )
                continue

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
                updated = await run_reliability(
                    reliability_fn=reliability_100,
                    config=config,
                    episodes=int(args.reliability_episodes),
                    eval_seed=eval_seed,
                    round_idx=round_idx,
                    append_log=_append_log,
                    raise_nature_floor=True,
                    agent=agent,
                    v2_best_path=V2_BEST,
                    best_mean=best_mean,
                )
                if updated is not None:
                    best_mean = updated
                session_greedy_max = max(
                    session_greedy_max,
                    float(getattr(reliability_100, "last_max", 0.0) or 0.0),
                )

        if best_mean >= float(args.target_mean) or v2_best >= float(args.target_mean):
            hit = max(best_mean, v2_best)
            print(f"CLIMB_TARGET_HIT mean={hit:.1f}", flush=True)
            if hit >= 2000.0:
                raised = float(
                    getattr(config.training, "post_2k_elite_gate", 3000.0) or 3000.0
                )
                config.training.elite_replay_min_episode_score = raised
                print(
                    f"CLIMB_POST_2K elite_gate->{raised:.0f} "
                    f"(FT allowed if >= {config.training.finetune_min_greedy_mean:.0f})",
                    flush=True,
                )
                print("CLIMB_CANONICAL_100 begin", flush=True)
                canon = await reliability_100(
                    config,
                    100,
                    ckpt=V2_BEST if checkpoint_exists(V2_BEST) else LATEST,
                    eval_seed=eval_seed,
                )
                print(
                    f"CLIMB_CANONICAL_100 mean={canon['mean']:.1f} "
                    f"min={canon['min']:.1f} max={canon['max']:.1f} "
                    f"p10={canon.get('p10', 0):.1f} p50={canon.get('p50', 0):.1f} "
                    f"p90={canon.get('p90', 0):.1f}",
                    flush=True,
                )
                _append_log({"round": round_idx, "event": "canonical_100", **canon})
                session_greedy_max = max(
                    session_greedy_max, float(canon.get("max") or 0.0)
                )
            break

        if args.offline_only and round_idx >= int(args.max_rounds):
            break

    end_true = read_v2_full_n_last(default=true_mean)
    abort = record_climb_session(
        start_true_mean=start_true_mean,
        end_true_mean=end_true,
        max_score=session_max,
        hours=float(args.hours),
        greedy_max=session_greedy_max,
    )
    if abort == "abort":
        print(
            "CLIMB_TAIL_BC_ABORT two sessions with delta<50 and max<1600 "
            f"— next collect prior={collect_thread_prior():.2f} "
            "(record seed-424242 teacher/human 2k traces)",
            flush=True,
        )
    print(
        f"CLIMB_DONE rounds={round_idx} best={best_mean:.1f} "
        f"v2_best={v2_best:.1f} true_mean={end_true:.1f} "
        f"rung={_current_rung(max(best_mean, v2_best)):.0f}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 offline-heavy climb ladder")
    add_climb_args(parser)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--skip-browser-probe", action="store_true")
    parser.add_argument("--skip-reliability", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

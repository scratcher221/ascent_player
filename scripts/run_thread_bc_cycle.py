#!/usr/bin/env python3
"""v2 thread-BC cycle: teacher/policy collect → offline BC → greedy probe → gated FT.

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
from ascent_player.agent.seed_curriculum import FineTuneReadiness
from ascent_player.utils.policy_floors import (
    read_seed_floor,
    record_thread_bc_eval,
    thread_bc_promotion_floor,
)
from ascent_player.utils.thread_bc import (
    DEFAULT_BC_SECONDARY_GATE,
    DEFAULT_COLLECT_GATE,
    ELITE_REPLAY,
    MAX_ELITE_EPISODES,
    MIN_ELITE_SCORE,
    SEED_BEST,
    THREAD_BC,
    CycleState,
    build_config,
    confirm_greedy_promotion,
    elite_store_for,
    load_work_agent,
    phase_bc,
    phase_ceiling,
    phase_collect,
    phase_dual_eval,
    phase_finetune,
    phase_skill_bootstrap,
    plan_collect_split,
    plan_finetune,
    save_thread_bc,
    write_thread_floor,
)


async def main_async(args: argparse.Namespace) -> int:
    state = CycleState(
        deadline=time.time() + max(600.0, float(args.hours) * 3600.0),
        elite_gate=max(MIN_ELITE_SCORE, float(args.elite_gate)),
        collect_gate=max(0.0, float(args.collect_gate)),
        target_mean=float(args.target_mean),
        target_min=float(args.target_min),
        bc_secondary_gate=float(
            getattr(args, "bc_secondary_gate", DEFAULT_BC_SECONDARY_GATE)
        ),
        thread_prior=float(args.thread_prior),
        thread_watch=float(args.thread_watch_prior),
        bc_steps=int(args.bc_steps),
        collect_eps=float(args.collect_eps),
        prefer_thread_bc=checkpoint_exists(THREAD_BC),
    )
    config = build_config(
        int(args.run_seed),
        elite_gate=state.elite_gate,
        collect_gate=state.collect_gate,
        elite_max_episodes=int(getattr(args, "elite_max_episodes", MAX_ELITE_EPISODES)),
    )
    store = elite_store_for(config, state.elite_gate)
    store.reset_if_incompatible()
    if not checkpoint_exists(THREAD_BC) and checkpoint_exists(SEED_BEST):
        agent = DQNAgent(config)
        assert agent.load(SEED_BEST)
        save_thread_bc(agent)

    floor = read_seed_floor()
    print(
        f"THREAD_BC_CYCLE_START seed={args.run_seed} hours={args.hours} "
        f"seed_floor={floor:.1f} thread_bc_best={thread_bc_promotion_floor():.1f} "
        f"collect_gate={state.collect_gate:.0f} elite_gate={state.elite_gate} "
        f"target_mean>={state.target_mean} target_min>={state.target_min}",
        flush=True,
    )

    if not bool(args.skip_ceiling):
        ceiling = await phase_ceiling(
            config, episodes=max(6, int(args.ceiling_episodes))
        )
        thread_mean = float(ceiling.get("thread_mean", 0.0))
        if thread_mean > 0:
            write_thread_floor(thread_mean)
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

    finetune_readiness = FineTuneReadiness(floor_ratio=0.95, required_rounds=2)
    while time.time() < state.deadline:
        state.round_id += 1
        if state.remaining < 300:
            print("TIME_LOW stopping", flush=True)
            break
        print(
            f"CYCLE_ROUND id={state.round_id} remaining_h={state.remaining/3600:.2f} "
            f"collect_gate={config.training.replay_min_episode_score} "
            f"elite_gate={config.training.elite_replay_min_episode_score} "
            f"thread_prior={state.thread_prior} watch={state.thread_watch}",
            flush=True,
        )
        split = plan_collect_split(state, config, float(args.collect_minutes))
        agent = load_work_agent(config, prefer_thread_bc=state.prefer_thread_bc)
        agent.epsilon = state.collect_eps
        agent.metrics.epsilon = state.collect_eps
        agent.progress.epsilon = state.collect_eps
        agent.save(Path("checkpoints/dqn_latest.keras"))
        print(
            f"PHASE_COLLECT_SPLIT teacher_s={split.teacher_seconds} "
            f"prior={split.teacher_prior:.2f} "
            f"policy_s={split.policy_seconds} prior={split.policy_prior:.2f}",
            flush=True,
        )
        await phase_collect(
            config,
            seconds=split.teacher_seconds,
            eps=min(0.03, state.collect_eps),
            thread_prior=split.teacher_prior,
            skip_replay_load=True,
        )
        collect_stats = await phase_collect(
            config,
            seconds=split.policy_seconds,
            eps=state.collect_eps,
            thread_prior=split.policy_prior,
            skip_replay_load=True,
        )
        print(
            f"COLLECT_DONE recent_avg={collect_stats.get('recent_avg', 0):.0f} "
            f"replay={collect_stats.get('replay_size', 0)}",
            flush=True,
        )
        elite_n = store.persist_from_browser(config)
        collect_replay = int(collect_stats.get("replay_size", 0) or 0)
        if collect_replay < 200 and elite_n < 200:
            state.thread_prior = min(0.85, state.thread_prior + 0.05)
            state.collect_eps = min(0.08, state.collect_eps + 0.01)
            print(
                f"STARVE_ADJUST collect_gate={config.training.replay_min_episode_score:.0f} "
                f"elite_gate={config.training.elite_replay_min_episode_score:.0f} "
                f"thread_prior={state.thread_prior:.2f} eps={state.collect_eps:.3f}",
                flush=True,
            )
            continue

        loss = await phase_bc(
            config,
            steps=state.bc_steps,
            lr=float(args.bc_lr),
            prefer_thread_bc=state.prefer_thread_bc,
            eval_probe_episodes=6,
            target_min=state.target_min,
            thread_watch_prior=state.thread_watch,
            collect_recent_avg=float(collect_stats.get("recent_avg", 0) or 0),
            secondary_gate=state.bc_secondary_gate,
            elite_gate=state.elite_gate,
        )
        state.prefer_thread_bc = True
        if loss is None:
            continue
        if state.remaining < 240:
            break

        evals = await phase_dual_eval(
            config,
            episodes=max(6, int(args.eval_episodes)),
            thread_watch_prior=state.thread_watch,
        )
        aligned = evals["aligned"]
        greedy = evals["greedy"]
        best_mean = greedy["mean"]
        promote_floor = max(850.0, thread_bc_promotion_floor())
        seed_floor = read_seed_floor()
        ft_ready = finetune_readiness.observe(best_mean, promote_floor)
        record_thread_bc_eval(greedy["mean"])
        promoted = False
        if greedy["mean"] > promote_floor:
            confirmation = await confirm_greedy_promotion(
                config,
                floor=promote_floor,
                target_min=state.target_min,
                episodes=max(16, int(args.eval_episodes)),
                label=f"round_{state.round_id}",
            )
            promoted = bool(confirmation["passed"])
        if promoted:
            promote_floor = max(promote_floor, thread_bc_promotion_floor())
            seed_floor = read_seed_floor()
            state.bc_steps = min(900, state.bc_steps + 50)
            state.collect_eps = max(0.03, state.collect_eps * 0.97)
            finetune_readiness.consecutive_rounds = 0
            state.thread_watch = max(0.25, state.thread_watch * 0.9)
        else:
            print(
                f"NO_PROMOTE aligned={aligned['mean']:.1f} greedy={greedy['mean']:.1f} "
                f"thread_floor={promote_floor:.1f} seed_floor={seed_floor:.1f} "
                f"— keep side ckpt + elite replay",
                flush=True,
            )
            if best_mean < promote_floor * 0.90:
                state.thread_prior = min(0.85, state.thread_prior + 0.03)
                state.collect_eps = min(0.08, state.collect_eps + 0.01)
                state.thread_watch = max(0.30, state.thread_watch * 0.92)
            else:
                state.thread_watch = max(0.35, min(state.thread_watch, 0.45))

        if promoted and seed_floor >= state.target_mean:
            print(f"TARGET_MET confirmed_mean={seed_floor:.1f}", flush=True)
            return 0

        ft_min = float(
            getattr(config.training, "finetune_min_greedy_mean", 1400.0) or 1400.0
        )
        ft = plan_finetune(
            state,
            best_mean=best_mean,
            ft_ready=ft_ready,
            ft_minutes=float(args.finetune_minutes),
            ft_min=ft_min,
        )
        if ft.allowed:
            pre_ft = Path("checkpoints/dqn_thread_bc_pre_ft.keras")
            if checkpoint_exists(THREAD_BC):
                shutil.copy2(THREAD_BC, pre_ft)
            await phase_finetune(
                config,
                seconds=ft.seconds,
                lr=float(args.ft_lr),
                eps=max(0.04, state.collect_eps),
                thread_prior=max(0.35, state.thread_prior * 0.85),
                gate=max(MIN_ELITE_SCORE, config.training.replay_min_episode_score),
            )
            if state.remaining >= 200:
                evals = await phase_dual_eval(
                    config,
                    episodes=max(6, int(args.eval_episodes)),
                    thread_watch_prior=state.thread_watch,
                )
                greedy = evals["greedy"]
                promote_floor = max(850.0, thread_bc_promotion_floor())
                if greedy["mean"] > promote_floor:
                    confirmation = await confirm_greedy_promotion(
                        config,
                        floor=promote_floor,
                        target_min=state.target_min,
                        episodes=max(16, int(args.eval_episodes)),
                        label=f"post_ft_round_{state.round_id}",
                    )
                    if confirmation["passed"]:
                        seed_floor = read_seed_floor()
                if pre_ft.exists() and greedy["mean"] < promote_floor * 0.95:
                    agent = DQNAgent(config)
                    assert agent.load(pre_ft)
                    save_thread_bc(agent)
                    print(
                        f"FT_REVERT greedy={greedy['mean']:.1f} — restored pre-FT side ckpt",
                        flush=True,
                    )
        elif state.remaining >= 45 * 60:
            print(f"FT_SKIP {ft.reason} — more elite BC first", flush=True)

    print(
        f"THREAD_BC_CYCLE_END seed_floor={read_seed_floor():.1f} "
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
    parser.add_argument("--collect-gate", type=float, default=DEFAULT_COLLECT_GATE)
    parser.add_argument("--elite-gate", type=float, default=MIN_ELITE_SCORE)
    parser.add_argument("--thread-prior", type=float, default=0.75)
    parser.add_argument("--thread-watch-prior", type=float, default=0.55)
    parser.add_argument("--collect-minutes", type=float, default=18.0)
    parser.add_argument("--collect-eps", type=float, default=0.04)
    parser.add_argument("--bc-steps", type=int, default=500)
    parser.add_argument("--bc-lr", type=float, default=3e-6)
    parser.add_argument("--bc-secondary-gate", type=float, default=DEFAULT_BC_SECONDARY_GATE)
    parser.add_argument("--elite-max-episodes", type=int, default=MAX_ELITE_EPISODES)
    parser.add_argument("--finetune-minutes", type=float, default=0.0)
    parser.add_argument("--ft-lr", type=float, default=2e-5)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--ceiling-episodes", type=int, default=8)
    parser.add_argument("--skip-ceiling", action="store_true")
    parser.add_argument("--skip-skill-bootstrap", action="store_true")
    parser.add_argument("--skill-bootstrap-minutes", type=float, default=10.0)
    parser.add_argument("--skill-bc-steps", type=int, default=400)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

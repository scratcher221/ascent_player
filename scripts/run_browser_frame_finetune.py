#!/usr/bin/env python3
"""Fine-tune on real browser #gameCanvas JPEG frames (no sim visual bridge).

Uses live Playwright capture → hybrid (frames + vectors) into browser_replay,
ε=0 Watch evals, and promote-on-mean∧min with regression restore.

Wall-time guidance (RTX 3070, frame_skip=2, ~8–12 decision Hz):
  1h  — smoke / few rounds
  2h  — first plausible promote if near current best
  4h  — default meaningful pass (replay turnover + ~12–20 evals)
  8h+ — overnight compounding; diminishing returns past ~6–8h at fixed LR/ε

Usage:
  PYTHONPATH=. python -u scripts/run_browser_frame_finetune.py --hours 4
  PYTHONPATH=. python -u scripts/run_browser_frame_finetune.py --hours 4 --from aligned
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ascent_player.agent.checkpoint import checkpoint_exists
from ascent_player.agent.dqn import DQNAgent


def _load_transfer_mod():
    path = ROOT / "scripts" / "run_browser_transfer.py"
    spec = importlib.util.spec_from_file_location("run_browser_transfer", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real-browser-frame fine-tune (skip rendered-sim bridge)"
    )
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--session-minutes", type=float, default=10.0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--target-mean", type=float, default=2000.0)
    parser.add_argument("--target-min", type=float, default=600.0)
    parser.add_argument(
        "--from",
        dest="seed_from",
        choices=("browser_best", "aligned"),
        default="browser_best",
        help="Weight seed: browser_best (default) or aligned_sim_best_eval",
    )
    parser.add_argument(
        "--mixed-sim-ratio",
        type=float,
        default=0.05,
        help="Fraction of each train batch from rendered sim (keep low)",
    )
    parser.add_argument(
        "--seed-sim-replay",
        action="store_true",
        help="Once-seed rendered sim_replay if empty (only useful with --mixed-sim-ratio>0)",
    )
    parser.add_argument(
        "--run-seed",
        type=int,
        default=None,
        help="Fixed game map seed (platforms/boosters). Omit for random layouts.",
    )
    args = parser.parse_args()

    transfer = _load_transfer_mod()
    config = transfer._build_config()
    config.training.sim_mode = False
    config.training.frame_skip = 2
    config.training.transfer_frame_skip = 2
    config.training.mixed_sim_replay_ratio = max(0.0, float(args.mixed_sim_ratio))
    config.training.browser_epsilon_floor = 0.08
    config.training.sim_pretrain_batch_size = 64
    # Larger replay cap so multi-hour real-frame runs keep more experience.
    config.training.browser_replay_max_items = max(
        20_000,
        int(config.training.browser_replay_max_items),
    )
    if args.run_seed is not None:
        config.browser.run_seed = int(args.run_seed)
        config.browser.lock_run_seed = True
        config.training.log_decision_every = 1
        print(f"FIXED_MAP_SEED {config.browser.run_seed}", flush=True)
        print("DECISION_LOG every=1 (full step logging)", flush=True)

    browser_best = config.training.browser_best_checkpoint_path
    aligned = Path("checkpoints/aligned_sim_best_eval.keras")
    start_from_browser_best = args.seed_from == "browser_best"

    if start_from_browser_best:
        if not checkpoint_exists(browser_best):
            print("MISSING_BROWSER_BEST", flush=True)
            return 1
        config.training.transfer_from_sim = False
        # Stronger learning on a fixed map; FS=1 for tighter steer timing.
        config.training.frame_skip = 1
        config.training.transfer_frame_skip = 1
        config.training.transfer_learning_rate = 4e-5
        config.training.transfer_epsilon_start = 0.18
        config.training.transfer_epsilon_restart = 0.15
        config.training.browser_epsilon_cap = 0.18
        config.training.browser_epsilon_floor = 0.10
        config.training.browser_epsilon_cap_after_gate_a = 0.18
        config.training.learning_rate = 4e-5
        # Mature checkpoint: keep some rule prior so corrected below-steer teaches fast.
        config.training.rule_prior_start = 0.25
        config.training.rule_prior_end = 0.12
        config.training.rule_prior_steps = 40_000
        config.demo.use_demos_on_start = False
        # Stronger horizontal steering pressure.
        config.mechanics_reward.steer_gain = 0.22
        config.mechanics_reward.wrong_way_penalty = -0.28
        config.mechanics_reward.aligned_bonus = 0.04
        agent = DQNAgent(config)
        assert agent.load(browser_best)
        agent.set_learning_rate(config.training.learning_rate)
        agent.epsilon = config.training.transfer_epsilon_start
        agent.metrics.epsilon = agent.epsilon
        agent.progress.epsilon = agent.epsilon
        agent.save(config.training.checkpoint_path)
        print(
            f"BROWSER_FRAME_FINETUNE_SEED {browser_best} "
            f"best={agent.progress.best_score}",
            flush=True,
        )
        if args.seed_sim_replay and config.training.mixed_sim_replay_ratio > 0:
            seeded = agent.seed_sim_replay_from_rendered()
            print(f"SEEDED_SIM_REPLAY transitions={seeded}", flush=True)
            agent.save(config.training.checkpoint_path)
    else:
        if not aligned.exists():
            print("MISSING_ALIGNED_CHECKPOINT", flush=True)
            return 1
        config.training.transfer_from_sim = True
        config.training.transfer_learning_rate = 2.5e-5
        config.training.transfer_epsilon_start = 0.15
        config.training.browser_epsilon_cap = 0.15
        config.training.learning_rate = 2.5e-5
        config.training.sim_best_eval_checkpoint_path = aligned
        config.training.playable_checkpoint_path = aligned
        agent = DQNAgent(config)
        assert agent.load(aligned)
        agent.prepare_transfer_from_sim()
        if args.seed_sim_replay or config.training.mixed_sim_replay_ratio > 0:
            seeded = agent.seed_sim_replay_from_rendered()
            print(f"SEEDED_SIM_REPLAY transitions={seeded}", flush=True)
        agent.save(config.training.checkpoint_path)
        print(
            f"BROWSER_FRAME_FINETUNE_SEED {aligned} "
            f"best={agent.progress.best_score}",
            flush=True,
        )

    # Floor from env or last promoted Watch mean so we never clobber a stronger browser_best.
    # Fixed-seed curriculum starts from a seed-relative floor so early map gains can promote.
    keep_weights = args.run_seed is not None
    if keep_weights:
        # Don't use the random-map 1408 floor — seed layouts score differently.
        os.environ.pop("ASCENT_BROWSER_BEST_MEAN", None)
        print(
            "SEED_CURRICULUM keep_weights_on_regress=1 "
            "(no browser_best restore; compounds dqn_latest)",
            flush=True,
        )
    elif "ASCENT_BROWSER_BEST_MEAN" not in os.environ and browser_best.exists():
        os.environ["ASCENT_BROWSER_BEST_MEAN"] = "1408.6"

    deadline = time.time() + max(600.0, args.hours * 3600.0)
    print(
        f"BROWSER_FRAME_FINETUNE_START hours={args.hours} "
        f"from={args.seed_from} mixed_sim={config.training.mixed_sim_replay_ratio:.3f} "
        f"run_seed={config.browser.run_seed} "
        f"eps={config.training.transfer_epsilon_start} "
        f"lr={config.training.learning_rate} "
        f"frame_skip={config.training.frame_skip} "
        f"steer_gain={config.mechanics_reward.steer_gain} "
        f"target_mean>={args.target_mean} target_min>={args.target_min} "
        f"(no visual bridge — real canvas JPEG only)",
        flush=True,
    )
    return asyncio.run(
        transfer._transfer_loop(
            config,
            deadline=deadline,
            session_seconds=max(180, int(args.session_minutes * 60)),
            eval_episodes=max(5, args.eval_episodes),
            target_mean=args.target_mean,
            target_min=args.target_min,
            start_from_browser_best=start_from_browser_best,
            ledger_mode="browser_frame_finetune",
            keep_weights_on_regress=keep_weights,
        )
    )


if __name__ == "__main__":
    sys.exit(main())

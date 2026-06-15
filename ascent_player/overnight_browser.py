#!/usr/bin/env python3
"""Unsupervised overnight browser training with adaptive session length.

Starts with short 2-minute probes, scales up to 1-hour sessions when metrics
improve, evaluates after every session, and stops after 9 hours total.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ascent_player.agent.progress_sanitize import patch_checkpoint_epsilon
from ascent_player.config import AppConfig
from ascent_player.training import run_eval_watch, run_sim_pretrain, run_training_no_ui

EPISODE_SCORE_RE = re.compile(r"max_score=([\d.]+)")
SUMMARY_LAST50_RE = re.compile(
    r"last_50_episodes score_mean=([\d.]+) score_max=([\d.]+) score_min=([\d.]+)"
)
LOOP_HZ_RE = re.compile(r"loop_hz_mean=([\d.]+)")
TRAIN_LOSS_RE = re.compile(r"train_loss last=([\d.]+) mean=([\d.]+)")

TOTAL_BUDGET_SECONDS = 9 * 3600
INITIAL_SESSION_SECONDS = 120
MAX_SESSION_SECONDS = 3600
MIN_SESSION_SECONDS = 120
TARGET_SCORE = 10_000
TARGET_CONSISTENCY_RATIO = 0.85
EVAL_EPISODES_SHORT = 3
EVAL_EPISODES_LONG = 10
EVAL_LONG_EVERY = 5
ROLLING_EVAL_WINDOW = 10
IMPROVE_ROLLING_EVAL = 35.0
IMPROVE_EVAL_MAX = 80.0
CONSECUTIVE_IMPROVE_TO_EXTEND = 2


@dataclass(slots=True)
class RunMetrics:
    run_index: int
    session_seconds: int
    training_best: float = 0.0
    training_recent_avg: float = 0.0
    training_recent_min: float = 0.0
    training_recent_max: float = 0.0
    log_last50_mean: float = 0.0
    log_last50_max: float = 0.0
    log_last50_min: float = 0.0
    log_episode_count: int = 0
    eval_mean: float = 0.0
    eval_min: float = 0.0
    eval_max: float = 0.0
    rolling_eval_mean: float = 0.0
    loop_hz_mean: float = 0.0
    train_loss_mean: float = 0.0
    demo_replay_size: float = 0.0
    epsilon: float = 0.0
    improved: bool = False
    duration_next: int = INITIAL_SESSION_SECONDS
    training_log: str = ""
    eval_log: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OvernightState:
    started_at: str
    total_elapsed_s: float = 0.0
    run_count: int = 0
    session_seconds: int = INITIAL_SESSION_SECONDS
    best_eval_mean: float = 0.0
    best_eval_max: float = 0.0
    best_rolling_eval_mean: float = 0.0
    best_training_recent_avg: float = 0.0
    plateau_runs: int = 0
    consecutive_improvements: int = 0
    demos_ingested: bool = False
    frame_skip: int = 4
    learning_rate: float = 2e-4
    target_steer_gain: float = 0.42
    steer_gain_stagnant_runs: int = 0
    target_met: bool = False
    eval_history: list[float] = field(default_factory=list)
    steer_gain_history: list[float] = field(default_factory=list)
    sim_refresh_count: int = 0
    runs: list[dict] = field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_log_file(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    episode_scores: list[float] = []
    for line in text.splitlines():
        if "EPISODE END" in line:
            match = EPISODE_SCORE_RE.search(line)
            if match:
                episode_scores.append(float(match.group(1)))

    last50_mean = last50_max = last50_min = 0.0
    for match in SUMMARY_LAST50_RE.finditer(text):
        last50_mean = float(match.group(1))
        last50_max = float(match.group(2))
        last50_min = float(match.group(3))

    loop_hz = 0.0
    for match in LOOP_HZ_RE.finditer(text):
        loop_hz = float(match.group(1))

    loss_mean = 0.0
    for match in TRAIN_LOSS_RE.finditer(text):
        loss_mean = float(match.group(2))

    recent = episode_scores[-10:] if episode_scores else []
    return {
        "episode_count": float(len(episode_scores)),
        "recent_avg": float(sum(recent) / len(recent)) if recent else 0.0,
        "recent_min": float(min(recent)) if recent else 0.0,
        "recent_max": float(max(recent)) if recent else 0.0,
        "last50_mean": last50_mean,
        "last50_max": last50_max,
        "last50_min": last50_min,
        "loop_hz": loop_hz,
        "loss_mean": loss_mean,
    }


def meets_target(eval_mean: float, eval_min: float) -> bool:
    return (
        eval_mean >= TARGET_SCORE
        and eval_min >= TARGET_SCORE * TARGET_CONSISTENCY_RATIO
    )


def rolling_mean(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    tail = values[-window:]
    return float(sum(tail) / len(tail))


def _apply_browser_profile(config: AppConfig, state: OvernightState) -> None:
    config.training.sim_mode = False
    config.training.transfer_from_sim = False
    config.training.watch_mode = False
    config.training.frame_skip = state.frame_skip
    config.training.target_score = TARGET_SCORE
    config.training.learning_rate = state.learning_rate
    config.reward.target_steer_gain = state.target_steer_gain
    config.reward.target_approach_gain = state.target_steer_gain * 0.83
    config.demo.use_demos_on_start = not state.demos_ingested
    config.demo.min_episode_score = 0.0

    rolling = rolling_mean(state.eval_history, ROLLING_EVAL_WINDOW)
    if rolling >= config.training.curriculum_stage_b_max:
        config.training.epsilon_end = 0.04
        config.reward.score_gain = 0.008
    elif rolling >= config.training.curriculum_stage_a_max:
        config.training.epsilon_end = 0.06
        config.reward.score_gain = 0.006
    else:
        config.training.epsilon_end = 0.08
        config.reward.score_gain = 0.004


def _score_improved(
    state: OvernightState,
    training: dict[str, float],
    log_stats: dict[str, float],
    eval_stats: dict[str, float],
    rolling_eval_mean: float,
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    eval_mean = float(eval_stats.get("recent_avg", 0.0))
    eval_max = float(eval_stats.get("recent_max", 0.0))
    train_recent = float(training.get("recent_avg", 0.0))
    log_mean = float(
        log_stats.get("last50_mean", 0.0) or log_stats.get("recent_avg", 0.0)
    )

    improved = False
    if rolling_eval_mean >= state.best_rolling_eval_mean + IMPROVE_ROLLING_EVAL:
        notes.append(
            f"rolling_eval +{rolling_eval_mean - state.best_rolling_eval_mean:.0f}"
        )
        improved = True
    if eval_max >= state.best_eval_max + IMPROVE_EVAL_MAX:
        notes.append(f"eval_max +{eval_max - state.best_eval_max:.0f}")
        improved = True
    if train_recent >= state.best_training_recent_avg + 60 and train_recent > 0:
        notes.append(f"train_recent +{train_recent - state.best_training_recent_avg:.0f}")
        improved = True
    if log_mean >= state.best_training_recent_avg + 50 and log_mean > 0:
        notes.append(f"log_mean {log_mean:.0f}")
        improved = True
    if eval_mean >= state.best_eval_mean + 40:
        notes.append(f"eval_mean +{eval_mean - state.best_eval_mean:.0f}")
        improved = True

    return improved, notes


def _next_session_seconds(
    current: int,
    improved: bool,
    consecutive_improvements: int,
    plateau_runs: int,
) -> int:
    if improved and consecutive_improvements >= CONSECUTIVE_IMPROVE_TO_EXTEND:
        bumped = int(current * 1.5)
        return max(MIN_SESSION_SECONDS, min(MAX_SESSION_SECONDS, bumped))
    if plateau_runs >= 3:
        return MIN_SESSION_SECONDS
    return current


def _tune_hyperparams(
    config: AppConfig,
    state: OvernightState,
    log_stats: dict[str, float],
    improved: bool,
    rolling_eval_mean: float,
) -> list[str]:
    notes: list[str] = []
    if improved:
        state.steer_gain_stagnant_runs = 0
        notes.append("keeping hyperparams (improving)")
        return notes

    loop_hz = float(log_stats.get("loop_hz", 0.0))
    loss_mean = float(log_stats.get("loss_mean", 0.0))

    new_eps = patch_checkpoint_epsilon(
        config.training.checkpoint_path,
        multiply=config.training.browser_plateau_epsilon_decay,
        cap=config.training.browser_epsilon_cap,
        floor=config.training.browser_epsilon_floor,
    )
    if new_eps is not None:
        notes.append(f"ε→{new_eps:.3f} (plateau decay)")

    if loop_hz > 0 and loop_hz < 11 and state.frame_skip < 6:
        state.frame_skip += 1
        config.training.frame_skip = state.frame_skip
        notes.append(f"frame_skip→{state.frame_skip} (loop {loop_hz:.1f}Hz)")
    elif state.frame_skip > 4 and loop_hz >= 13:
        state.frame_skip = max(4, state.frame_skip - 1)
        config.training.frame_skip = state.frame_skip
        notes.append(f"frame_skip→{state.frame_skip} (throughput)")

    if loss_mean > 0.8 and state.learning_rate > 2e-5:
        state.learning_rate *= 0.9
        config.training.learning_rate = state.learning_rate
        notes.append(f"lr→{state.learning_rate:.2e} (loss {loss_mean:.2f})")

    state.steer_gain_stagnant_runs += 1
    state.steer_gain_history.append(rolling_eval_mean)
    state.steer_gain_history = state.steer_gain_history[-5:]
    steer_improved = (
        len(state.steer_gain_history) >= 2
        and state.steer_gain_history[-1]
        > state.steer_gain_history[0] + 20
    )
    if steer_improved and state.steer_gain_stagnant_runs >= 3:
        state.target_steer_gain = min(0.65, state.target_steer_gain + 0.03)
        config.reward.target_steer_gain = state.target_steer_gain
        config.reward.target_approach_gain = state.target_steer_gain * 0.83
        state.steer_gain_stagnant_runs = 0
        notes.append(f"target_steer_gain→{state.target_steer_gain:.2f} (rolling gain)")

    config.reward.direction_flip_penalty = max(
        -0.30,
        config.reward.direction_flip_penalty - 0.005,
    )
    notes.append(f"flip_penalty→{config.reward.direction_flip_penalty:.2f}")

    return notes


def _write_run_summary(overnight_dir: Path, metrics: RunMetrics) -> Path:
    path = overnight_dir / f"run_{metrics.run_index:03d}_summary.txt"
    lines = [
        f"run={metrics.run_index}",
        f"time={_utc_now()}",
        f"session_seconds={metrics.session_seconds}",
        f"improved={metrics.improved}",
        f"duration_next={metrics.duration_next}",
        "",
        "[training]",
        f"  best={metrics.training_best:.0f}",
        f"  recent_avg={metrics.training_recent_avg:.0f}",
        f"  recent_min={metrics.training_recent_min:.0f}",
        f"  recent_max={metrics.training_recent_max:.0f}",
        f"  log_last50_mean={metrics.log_last50_mean:.0f}",
        f"  log_last50_max={metrics.log_last50_max:.0f}",
        f"  demo_replay={metrics.demo_replay_size:.0f}",
        f"  epsilon={metrics.epsilon:.3f}",
        f"  log={metrics.training_log}",
        "",
        "[eval ε=0]",
        f"  mean={metrics.eval_mean:.0f}",
        f"  min={metrics.eval_min:.0f}",
        f"  max={metrics.eval_max:.0f}",
        f"  rolling_mean={metrics.rolling_eval_mean:.0f}",
        f"  log={metrics.eval_log}",
        "",
        "[notes]",
    ]
    lines.extend(f"  - {note}" for note in metrics.notes)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _save_state(path: Path, state: OvernightState) -> None:
    path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def _load_state(path: Path) -> OvernightState | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in OvernightState.__dataclass_fields__.values()}
        filtered = {key: value for key, value in data.items() if key in known}
        return OvernightState(**filtered)
    except (json.JSONDecodeError, TypeError):
        return None


def _reexec_process() -> None:
    argv = [sys.executable, "-m", "ascent_player.overnight_browser", *sys.argv[1:]]
    os.execv(sys.executable, argv)


async def run_overnight_session(
    config: AppConfig,
    *,
    total_budget_s: float = TOTAL_BUDGET_SECONDS,
    initial_session_s: int = INITIAL_SESSION_SECONDS,
    overnight_dir: Path | None = None,
    dry_run: bool = False,
) -> OvernightState:
    log_root = config.training.log_dir
    overnight_dir = overnight_dir or (log_root / "overnight")
    overnight_dir.mkdir(parents=True, exist_ok=True)
    state_path = overnight_dir / "state.json"
    journal_path = overnight_dir / "journal.log"

    state = _load_state(state_path)
    if state is None:
        state = OvernightState(
            started_at=_utc_now(),
            session_seconds=initial_session_s,
            learning_rate=config.training.learning_rate,
            frame_skip=config.training.transfer_frame_skip,
        )

    budget_start = time.perf_counter()
    if state.total_elapsed_s > 0:
        budget_start -= state.total_elapsed_s

    def journal(line: str) -> None:
        msg = f"[{_utc_now()}] {line}"
        print(msg, flush=True)
        with journal_path.open("a", encoding="utf-8") as handle:
            handle.write(msg + "\n")

    journal(
        f"OVERNIGHT START budget={total_budget_s:.0f}s "
        f"target={TARGET_SCORE} session={state.session_seconds}s"
    )

    while state.total_elapsed_s < total_budget_s:
        remaining = total_budget_s - state.total_elapsed_s
        session_s = min(state.session_seconds, int(remaining))
        if session_s < 60:
            journal("Budget exhausted (<1 min left) — stopping.")
            break

        state.run_count += 1
        run_index = state.run_count
        journal(
            f"=== RUN {run_index} train {session_s}s "
            f"(elapsed {state.total_elapsed_s:.0f}s) ==="
        )

        if dry_run:
            journal("dry-run: skipping browser")
            break

        if (
            state.plateau_runs >= config.training.sim_refresh_plateau_runs
            and state.sim_refresh_count < 3
        ):
            journal(
                f"SIM REFRESH after {state.plateau_runs} plateau runs "
                f"({config.training.sim_refresh_steps} steps)"
            )
            refresh_config = AppConfig()
            refresh_config.training.sim_mode = True
            refresh_config.training.sim_pretrain_steps = config.training.sim_refresh_steps
            await asyncio.to_thread(run_sim_pretrain, refresh_config)
            state.sim_refresh_count += 1
            state.plateau_runs = 0
            journal("SIM REFRESH complete")

        _apply_browser_profile(config, state)
        reingest_demos = (
            state.demos_ingested
            and run_index % config.demo.demo_reingest_every_runs == 0
        )
        train_started = time.perf_counter()
        training = await run_training_no_ui(
            config,
            max_seconds=session_s,
            ingest_demos=not state.demos_ingested,
            force_demo_reingest=reingest_demos,
        )
        train_elapsed = time.perf_counter() - train_started
        state.total_elapsed_s += train_elapsed
        state.demos_ingested = True

        if training.get("error"):
            journal(f"TRAIN FAILED: {training['error']}")
            state.plateau_runs += 1
            _save_state(state_path, state)
            if "OOM" in str(training.get("error", "")).upper() or "memory" in str(
                training.get("error", "")
            ).lower():
                journal("OOM detected — restarting process")
                _save_state(state_path, state)
                _reexec_process()
            await asyncio.sleep(30)
            continue

        log_stats = parse_log_file(Path(str(training.get("log_path", ""))))
        journal(
            f"TRAIN done {train_elapsed:.0f}s "
            f"best={training.get('best_score', 0):.0f} "
            f"recent_avg={training.get('recent_avg', 0):.0f} "
            f"demo_replay={training.get('demo_replay_size', 0):.0f} "
            f"ε={training.get('epsilon', 0):.3f}"
        )

        eval_episodes = (
            EVAL_EPISODES_LONG if run_index % EVAL_LONG_EVERY == 0 else EVAL_EPISODES_SHORT
        )
        eval_started = time.perf_counter()
        eval_stats = await run_eval_watch(config, max_episodes=eval_episodes)
        eval_elapsed = time.perf_counter() - eval_started
        state.total_elapsed_s += eval_elapsed

        eval_mean = float(eval_stats.get("recent_avg", 0.0))
        state.eval_history.append(eval_mean)
        state.eval_history = state.eval_history[-ROLLING_EVAL_WINDOW:]
        rolling_eval_mean = rolling_mean(state.eval_history, ROLLING_EVAL_WINDOW)

        journal(
            f"EVAL done {eval_elapsed:.0f}s ({eval_episodes} ep) "
            f"mean={eval_mean:.0f} rolling={rolling_eval_mean:.0f}"
        )

        improved, improve_notes = _score_improved(
            state,
            training,
            log_stats,
            eval_stats,
            rolling_eval_mean,
        )
        tune_notes = _tune_hyperparams(
            config,
            state,
            log_stats,
            improved,
            rolling_eval_mean,
        )

        if improved:
            state.plateau_runs = 0
            state.consecutive_improvements += 1
            state.best_eval_mean = max(state.best_eval_mean, eval_mean)
            state.best_eval_max = max(
                state.best_eval_max,
                float(eval_stats.get("recent_max", 0.0)),
            )
            state.best_rolling_eval_mean = max(
                state.best_rolling_eval_mean,
                rolling_eval_mean,
            )
            state.best_training_recent_avg = max(
                state.best_training_recent_avg,
                float(training.get("recent_avg", 0.0)),
                float(log_stats.get("last50_mean", 0.0)),
            )
        else:
            state.plateau_runs += 1
            state.consecutive_improvements = 0

        next_session = _next_session_seconds(
            state.session_seconds,
            improved,
            state.consecutive_improvements,
            state.plateau_runs,
        )
        state.session_seconds = next_session

        metrics = RunMetrics(
            run_index=run_index,
            session_seconds=session_s,
            training_best=float(training.get("best_score", 0.0)),
            training_recent_avg=float(training.get("recent_avg", 0.0)),
            training_recent_min=float(training.get("recent_min", 0.0)),
            training_recent_max=float(training.get("recent_max", 0.0)),
            log_last50_mean=float(log_stats.get("last50_mean", 0.0)),
            log_last50_max=float(log_stats.get("last50_max", 0.0)),
            log_last50_min=float(log_stats.get("last50_min", 0.0)),
            log_episode_count=int(log_stats.get("episode_count", 0.0)),
            eval_mean=eval_mean,
            eval_min=float(eval_stats.get("recent_min", 0.0)),
            eval_max=float(eval_stats.get("recent_max", 0.0)),
            rolling_eval_mean=rolling_eval_mean,
            loop_hz_mean=float(log_stats.get("loop_hz", 0.0)),
            train_loss_mean=float(log_stats.get("loss_mean", 0.0)),
            demo_replay_size=float(training.get("demo_replay_size", 0.0)),
            epsilon=float(training.get("epsilon", 0.0)),
            improved=improved,
            duration_next=next_session,
            training_log=str(training.get("log_path", "")),
            eval_log=str(eval_stats.get("log_path", "")),
            notes=improve_notes + tune_notes,
        )
        summary_path = _write_run_summary(overnight_dir, metrics)
        state.runs.append(asdict(metrics))
        _save_state(state_path, state)

        journal(
            f"RUN {run_index} {'IMPROVED' if improved else 'plateau'} "
            f"next_session={next_session}s summary={summary_path.name}"
        )

        if meets_target(metrics.eval_mean, metrics.eval_min):
            state.target_met = True
            journal(
                f"TARGET MET eval_mean={metrics.eval_mean:.0f} "
                f"eval_min={metrics.eval_min:.0f}"
            )
            _save_state(state_path, state)
            break

        wall_elapsed = time.perf_counter() - budget_start
        if wall_elapsed >= total_budget_s:
            journal(f"9-hour budget reached ({wall_elapsed:.0f}s)")
            break

        if (
            run_index % config.training.gpu_restart_every_runs == 0
            and run_index > 0
        ):
            journal(f"GPU restart after run {run_index}")
            _save_state(state_path, state)
            _reexec_process()

        await asyncio.sleep(5)

    journal(
        f"OVERNIGHT END runs={state.run_count} "
        f"best_rolling_eval={state.best_rolling_eval_mean:.0f} "
        f"target_met={state.target_met} "
        f"elapsed={state.total_elapsed_s:.0f}s"
    )
    _save_state(state_path, state)
    return state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unsupervised overnight browser training (up to 9 hours)",
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=9.0,
        help="Total wall-clock budget in hours (default 9)",
    )
    parser.add_argument(
        "--initial-minutes",
        type=int,
        default=2,
        help="Initial training session length in minutes (default 2)",
    )
    parser.add_argument(
        "--max-session-minutes",
        type=int,
        default=60,
        help="Maximum training session length in minutes (default 60)",
    )
    parser.add_argument(
        "--target-score",
        type=int,
        default=TARGET_SCORE,
        help="Stop early when eval consistently reaches this score",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize state/journal without browser training",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from ascent_player.utils.gpu_env import bootstrap_gpu_environment

    bootstrap_gpu_environment()
    args = parse_args(argv)

    global TARGET_SCORE, TOTAL_BUDGET_SECONDS, INITIAL_SESSION_SECONDS, MAX_SESSION_SECONDS, MIN_SESSION_SECONDS
    TARGET_SCORE = int(args.target_score)
    TOTAL_BUDGET_SECONDS = int(args.budget_hours * 3600)
    INITIAL_SESSION_SECONDS = max(120, int(args.initial_minutes * 60))
    MIN_SESSION_SECONDS = INITIAL_SESSION_SECONDS
    MAX_SESSION_SECONDS = max(
        INITIAL_SESSION_SECONDS,
        int(args.max_session_minutes * 60),
    )

    config = AppConfig()
    asyncio.run(
        run_overnight_session(
            config,
            total_budget_s=float(TOTAL_BUDGET_SECONDS),
            initial_session_s=INITIAL_SESSION_SECONDS,
            dry_run=args.dry_run,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

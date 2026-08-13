from __future__ import annotations

from pathlib import Path

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig
from ascent_player.demo.ingest import ingest_demonstrations
from ascent_player.env.browser_backend import BrowserBackend
from ascent_player.env.game_env import AscentGameEnv
from ascent_player.env.sim_env import AscentSimEnv
from ascent_player.training.curriculum import (
    apply_curriculum,
    episode_score_for_progress,
    warmstart_from_teacher,
)
from ascent_player.utils.decision_log import DecisionLogger
from ascent_player.utils.gpu_session import release_gpu_between_runs
from ascent_player.utils.training_log import BrowserStepContext, TrainingLogger
from ascent_player.env.target_detector import TargetDetectionTracker
from ascent_player.agent.reason import wrong_vs_target
from ascent_player.env.game_env import ACTION_LABELS
from ascent_player.env.state_detector import FrameState
import os
import time
import numpy as np


def _empty_training_stats(log_path: Path, *, error: str = "") -> dict[str, float]:
    return {
        "best_score": 0.0,
        "recent_avg": 0.0,
        "recent_min": 0.0,
        "recent_max": 0.0,
        "episodes": 0.0,
        "log_path": str(log_path),
        "error": error,
    }


async def run_eval_watch(
    config: AppConfig,
    *,
    max_episodes: int = 5,
) -> dict[str, float]:
    """Run epsilon=0 evaluation episodes in the browser."""
    from dataclasses import replace

    eval_config = AppConfig()
    eval_config.browser = config.browser
    eval_config.observation = config.observation
    eval_config.reward = config.reward
    eval_config.demo = config.demo
    eval_config.mechanics_reward = config.mechanics_reward
    eval_config.training = replace(
        config.training,
        sim_mode=False,
        transfer_from_sim=False,
        watch_mode=True,
        epsilon_start=0.0,
        epsilon_end=0.0,
        frame_skip=max(1, min(config.training.transfer_frame_skip, config.training.frame_skip)),
        watch_rule_prior=float(
            getattr(config.training, "watch_rule_prior", 0.0) or 0.0
        ),
        watch_safety_override=bool(
            getattr(config.training, "watch_safety_override", False)
        ),
        skip_browser_replay_load=True,
        # Preserve seed-thread Watch prior so aligned evals match training skill.
        seed_thread_enabled=bool(
            getattr(config.training, "seed_thread_enabled", False)
        ),
        seed_thread_watch_prior=float(
            getattr(config.training, "seed_thread_watch_prior", 0.0) or 0.0
        ),
        seed_thread_corridor=float(
            getattr(config.training, "seed_thread_corridor", 0.18) or 0.18
        ),
        seed_thread_only_when_landing=bool(
            getattr(config.training, "seed_thread_only_when_landing", True)
        ),
    )
    return await run_training_no_ui(
        eval_config,
        max_episodes=max_episodes,
        ingest_demos=False,
    )


def create_env(config: AppConfig, backend: BrowserBackend | None = None):
    if config.training.sim_mode:
        return AscentSimEnv(config)
    if backend is None:
        backend = BrowserBackend(config.browser)
    return AscentGameEnv(config, backend)


async def run_training_no_ui(
    config: AppConfig,
    max_episodes: int | None = None,
    max_seconds: int | None = None,
    *,
    ingest_demos: bool | None = None,
    force_demo_reingest: bool = False,
) -> dict[str, float]:
    phase = "sim" if config.training.sim_mode else "browser"
    logger = TrainingLogger(config, phase)
    backend = None if config.training.sim_mode else BrowserBackend(config.browser)
    if config.training.transfer_from_sim and not config.training.sim_mode:
        config.training.frame_skip = max(
            config.training.frame_skip,
            config.training.transfer_frame_skip,
        )
    env = create_env(config, backend)
    agent = DQNAgent(config)
    load_result = agent.try_autoload()
    print(load_result.message)
    if not config.training.sim_mode:
        replay_path = config.training.browser_replay_path
        if getattr(config.training, "skip_browser_replay_load", False):
            print("SKIP_REPLAY_LOAD (fresh collect buffer)", flush=True)
            loaded = 0
        else:
            loaded = agent.replay.load_pickle(
                replay_path,
                max_items=config.training.browser_replay_max_items,
                vector_dim=config.observation.vector_dim
                if config.observation.include_vector_state
                else None,
            )
        if loaded:
            print(f"Loaded {loaded} browser replay transitions from {replay_path.name}")
            # Decay mixed sim replay as browser experience accumulates.
            fill = min(1.0, loaded / max(1, config.training.browser_replay_max_items))
            # Do not raise a caller-requested 0.0 floor back to 0.05.
            if float(config.training.mixed_sim_replay_ratio) > 0:
                config.training.mixed_sim_replay_ratio = max(
                    0.05,
                    float(config.training.mixed_sim_replay_ratio) * (1.0 - 0.6 * fill),
                )
            print(
                f"mixed_sim_replay_ratio→{config.training.mixed_sim_replay_ratio:.3f} "
                f"(browser_replay fill={fill:.2f})"
            )
    logger.log_session_start(
        agent,
        message=load_result.message,
        extra={
            "headless": True,
            "watch_mode": config.training.watch_mode,
            "frame_skip": config.training.frame_skip,
            "max_seconds": max_seconds,
        },
    )
    print(f"Training log: {logger.path}")
    demos_ingested = ingest_demos is False and not force_demo_reingest
    demo_policy = (
        config.demo.use_demos_on_start
        if ingest_demos is None
        else ingest_demos or force_demo_reingest
    )
    transfer_episodes = 0
    target_tracker = (
        TargetDetectionTracker(
            min_samples=config.training.target_detection_min_samples,
            min_special_rate=config.training.target_special_min_rate,
        )
        if not config.training.sim_mode
        else None
    )
    deadline = (
        time.perf_counter() + max_seconds if max_seconds and max_seconds > 0 else None
    )
    episode_steps = 0
    decision_logger: DecisionLogger | None = None
    if not config.training.sim_mode:
        every = max(1, int(getattr(config.training, "log_decision_every", 5)))
        decision_logger = DecisionLogger.create(
            Path("logs"),
            every=every,
            prefix="decisions",
        )
        print(
            f"Decision log: {decision_logger.path} (every={every})",
            flush=True,
        )
        logger.log_note(f"decision_log={decision_logger.path} every={every}")
    try:
        if (
            force_demo_reingest
            and demo_policy
            and not config.training.sim_mode
        ):
            demo_result = ingest_demonstrations(agent, config, replace_buffer=True)
            if demo_result.transitions_added or demo_result.transitions_skipped:
                print(demo_result.status_message)
                logger.log_note(f"demo_reingest={demo_result.status_message}")
            demos_ingested = True
        if not config.training.sim_mode:
            assert backend is not None
            status = await backend.connect_auto()
            if not status.connected:
                print(status.message)
                logger.log_note(f"browser_connect_failed={status.message}")
                logger.close(agent)
                return _empty_training_stats(logger.path, error="browser_connect_failed")
        state = await env.reset()
        apply_curriculum(config, agent, env)
        if (
            len(agent.replay) == 0
            and config.training.sim_warmstart_teacher
            and not config.training.watch_mode
        ):
            print("TEACHER_WARMSTART begin (can take a while)...", flush=True)
            teacher_added = await warmstart_from_teacher(agent, config)
            if teacher_added:
                print(f"Teacher warm-start added {teacher_added} sim transitions")
                logger.log_note(f"teacher_warmstart={teacher_added}")
            print("TEACHER_WARMSTART done", flush=True)
        print("TRAIN_LOOP begin", flush=True)
        episode = agent.progress.episodes_completed
        episodes_run = 0
        episode_reward = 0.0
        episode_max_score = 0.0
        prev_step_score = 0.0
        score_velocity = 0.0
        loop_hz = 0.0
        episode_buffer: list[tuple] = []
        replay_score_gate = float(
            getattr(config.training, "replay_min_episode_score", 0.0) or 0.0
        )
        elite_score_gate = float(
            getattr(config.training, "elite_replay_min_episode_score", 0.0) or 0.0
        )
        while max_episodes is None or episodes_run < max_episodes:
            if deadline is not None and time.perf_counter() >= deadline:
                print(f"Finetune time limit reached ({max_seconds}s)")
                logger.log_note(f"finetune_timeout={max_seconds}s")
                break
            step_started = time.perf_counter()
            if episodes_run == 0 and episode_steps == 0:
                print("TRAIN_LOOP first_act", flush=True)
            action = agent.act(
                state,
                training=not config.training.watch_mode,
                can_boost=env.can_boost,
                boost_level=env.boost_level,
                frame_state=env._last_frame_state,
            )
            if episodes_run == 0 and episode_steps == 0:
                print(f"TRAIN_LOOP first_step action={action}", flush=True)
            result = await env.step(action)
            if episodes_run == 0 and episode_steps == 0:
                print(
                    f"TRAIN_LOOP first_step_done score={result.frame_state.score}",
                    flush=True,
                )
            if (
                not config.training.sim_mode
                and episode_steps == 45
                and episode_max_score <= 0
            ):
                try:
                    hb = await env.backend.browser_heartbeat()
                    if isinstance(hb, dict) and (
                        hb.get("inMenu") or not hb.get("playing")
                    ):
                        print("BROWSER_MENU_RECOVERY begin", flush=True)
                        await env.backend.ensure_playing(timeout_seconds=15.0)
                except Exception as exc:
                    print(f"BROWSER_MENU_RECOVERY_FAIL {exc}", flush=True)
            step_ms = (time.perf_counter() - step_started) * 1000.0
            if step_ms > 0:
                instant_hz = 1000.0 / step_ms
                loop_hz = (0.85 * loop_hz) + (0.15 * instant_hz)
            if (
                not config.training.sim_mode
                and replay_score_gate > 0
                and not config.training.watch_mode
            ):
                episode_buffer.append(
                    (
                        state,
                        action,
                        result.reward,
                        result.state,
                        result.done,
                        int(getattr(agent, "last_reason_index", -1)),
                        int(getattr(agent, "last_skill_index", -1)),
                    )
                )
                metrics = agent.maybe_train() if len(agent.replay) >= config.training.min_replay_size else agent.metrics
            else:
                agent.remember(
                    state,
                    action,
                    result.reward,
                    result.state,
                    result.done,
                    sim=config.training.sim_mode,
                )
                metrics = agent.maybe_train()
            episode_reward += result.reward
            episode_steps += 1
            if result.frame_state.score is not None:
                episode_score = float(result.frame_state.score)
                episode_max_score = max(episode_max_score, float(result.frame_state.score))
                score_velocity = episode_score - prev_step_score
                prev_step_score = episode_score
            if config.training.sim_mode:
                logger.record_step(
                    action,
                    result.reward,
                    result.frame_state,
                    can_boost=env.can_boost,
                    boost_level=env.boost_level,
                    done=result.done,
                    train_loss=metrics.loss,
                    train_ms=metrics.train_ms,
                )
            else:
                logger.record_browser_step(
                    action,
                    result.reward,
                    result.frame_state,
                    agent,
                    can_boost=env.can_boost,
                    boost_level=env.boost_level,
                    done=result.done,
                    context=BrowserStepContext(
                        step_ms=step_ms,
                        loop_hz=loop_hz,
                        score_velocity=score_velocity,
                        episode_reward=episode_reward,
                        total_steps=metrics.total_steps,
                    ),
                    train_loss=metrics.loss,
                    train_ms=metrics.train_ms,
                )
                if decision_logger is not None:
                    decision_logger.maybe_log(
                        episode=episode,
                        step=episode_steps,
                        action=action,
                        reason=getattr(agent, "last_reason", ""),
                        reason_pred=getattr(agent, "last_reason_pred", ""),
                        source=getattr(agent, "last_action_source", ""),
                        score=result.frame_state.score,
                        eps=agent.epsilon,
                        frame_state=result.frame_state,
                        wrong_vs_target=wrong_vs_target(
                            result.frame_state, action
                        ),
                    )
            if target_tracker is not None:
                target_tracker.record(result.frame_state.target_kind)
                config.training.target_platform_only = target_tracker.use_platform_only
                if (
                    metrics.total_steps > 0
                    and metrics.total_steps % config.training.log_interval_steps == 0
                ):
                    logger.log_note(
                        f"target_detection {target_tracker.summary()} "
                        f"platform_only={config.training.target_platform_only}"
                    )
            logger.maybe_flush(agent, metrics.total_steps)
            state = result.state
            if metrics.total_steps % 25 == 0:
                print(
                    f"step={metrics.total_steps} action={ACTION_LABELS[action]} "
                    f"boost={env.boost_level:.0%} reward={episode_reward:.3f} "
                    f"score={episode_max_score:.0f} replay={metrics.replay_size} "
                    f"loss={metrics.loss}"
                )
            if (
                not config.training.sim_mode
                and metrics.total_steps > 0
                and metrics.total_steps % 500 == 0
            ):
                try:
                    lr_now = float(agent.online.optimizer.learning_rate.numpy())
                except Exception:
                    lr_now = float(config.training.learning_rate)
                print(
                    f"LR_CHECK step={metrics.total_steps} "
                    f"optimizer_lr={lr_now:.3e} "
                    f"config_lr={config.training.learning_rate:.3e} "
                    f"eps={agent.epsilon:.3f} loss={metrics.loss}",
                    flush=True,
                )
                try:
                    await env.backend.browser_heartbeat()
                except Exception as exc:
                    print(f"BROWSER_HEARTBEAT_FAIL {exc}", flush=True)
            if result.done:
                if episode_buffer:
                    if episode_max_score >= replay_score_gate:
                        ep_id = int(getattr(agent.progress, "episodes_completed", 0) or 0)
                        for (
                            buf_state,
                            buf_action,
                            buf_reward,
                            buf_next,
                            buf_done,
                            buf_reason,
                            buf_skill,
                        ) in episode_buffer:
                            agent.remember(
                                buf_state,
                                buf_action,
                                buf_reward,
                                buf_next,
                                buf_done,
                                sim=False,
                                reason=int(buf_reason),
                                skill=int(buf_skill),
                                episode_score=float(episode_max_score),
                                episode_id=ep_id,
                            )
                        if (
                            elite_score_gate > 0
                            and episode_max_score >= elite_score_gate
                        ):
                            print(
                                f"ELITE_STAGE score={episode_max_score:.0f} "
                                f">={elite_score_gate:.0f} "
                                f"steps={len(episode_buffer)}",
                                flush=True,
                            )
                    else:
                        print(
                            f"REPLAY_GATE skip episode score={episode_max_score:.0f} "
                            f"< {replay_score_gate:.0f} "
                            f"(held={len(episode_buffer)})",
                            flush=True,
                        )
                    episode_buffer.clear()
                agent.record_episode(episode_reward, episode_max_score)
                apply_curriculum(
                    config,
                    agent,
                    env,
                    episode_steps=episode_steps,
                    frame_state=result.frame_state,
                )
                logger.log_episode_end(
                    agent,
                    episode,
                    episode_reward,
                    episode_max_score,
                )
                print(
                    f"episode={episode} reward={episode_reward:.3f} "
                    f"score={episode_max_score:.0f} epsilon={agent.epsilon:.3f}"
                )
                agent.end_episode()
                agent.maybe_autosave(force=True)
                episode += 1
                episodes_run += 1
                if config.training.transfer_from_sim:
                    transfer_episodes += 1
                if (
                    demo_policy
                    and not demos_ingested
                    and not config.training.sim_mode
                    and (
                        not config.training.transfer_from_sim
                        or transfer_episodes
                        >= config.training.transfer_demo_delay_episodes
                    )
                ):
                    demo_result = ingest_demonstrations(
                        agent,
                        config,
                        replace_buffer=force_demo_reingest,
                    )
                    if demo_result.transitions_added or demo_result.transitions_skipped:
                        print(demo_result.status_message)
                        logger.log_note(f"demo_ingest={demo_result.status_message}")
                    demos_ingested = True
                episode_reward = 0.0
                episode_max_score = 0.0
                episode_steps = 0
                prev_step_score = 0.0
                score_velocity = 0.0
                state = await env.reset()
    finally:
        if decision_logger is not None:
            decision_logger.close()
        removed = agent.trim_replay_buffers()
        if removed:
            print(f"Trimmed {removed} replay transitions")
        if not config.training.sim_mode:
            recent = agent.progress.recent_scores[-10:]
            recent_avg = float(sum(recent) / len(recent)) if recent else 0.0
            from ascent_player.utils.browser_replay_persist import persist_browser_replay

            persist_browser_replay(agent, config, recent_avg=recent_avg)
        release_gpu_between_runs()
        agent.save()
        await env.close()
        logger.close(agent)
        print(f"Training log: {logger.path}")
    recent = agent.progress.recent_scores[-10:]
    return {
        "best_score": agent.progress.best_score,
        "recent_avg": float(sum(recent) / len(recent)) if recent else 0.0,
        "recent_min": float(min(recent)) if recent else 0.0,
        "recent_max": float(max(recent)) if recent else 0.0,
        "episodes": float(agent.progress.episodes_completed),
        "log_path": str(logger.path),
        "epsilon": agent.epsilon,
        "total_steps": float(agent.progress.total_steps),
        "demo_replay_size": float(len(agent.demo_replay)),
        "curriculum_stage": float(
            int(agent.curriculum_stage[1:])
            if str(agent.curriculum_stage).startswith("M")
            else 0.0
        ),
    }



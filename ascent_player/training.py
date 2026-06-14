from __future__ import annotations

import os
import random
import time

import numpy as np

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig
from ascent_player.demo.ingest import ingest_demonstrations
from ascent_player.env.browser_backend import BrowserBackend
from ascent_player.env.game_env import ACTION_LABELS, AscentGameEnv
from ascent_player.env.sim_env import AscentSimEnv, calibrate_sim_physics
from ascent_player.env.state_detector import FrameState
from ascent_player.utils.training_log import BrowserStepContext, TrainingLogger


def create_env(config: AppConfig, backend: BrowserBackend | None = None):
    if config.training.sim_mode:
        return AscentSimEnv(config)
    if backend is None:
        backend = BrowserBackend(config.browser)
    return AscentGameEnv(config, backend)


def _sim_env_count(config: AppConfig) -> int:
    if config.training.sim_pretrain_envs > 0:
        return config.training.sim_pretrain_envs
    cpus = os.cpu_count() or 8
    return max(4, min(16, cpus - 2))


def run_sim_pretrain(config: AppConfig, steps: int | None = None) -> None:
    """Fast vectorized simulator pretraining (sync, batched inference)."""
    config.training.sim_mode = True
    config.training.frame_skip = 1
    env_count = _sim_env_count(config)
    target_steps = steps or config.training.sim_pretrain_steps or 500_000

    envs = [
        AscentSimEnv(config, fast_mode=True, env_index=index)
        for index in range(env_count)
    ]
    agent = DQNAgent(config)
    agent.apply_sim_pretrain_profile()
    logger = TrainingLogger(config, "sim")

    states = np.stack([env.reset_sync() for env in envs])
    episode_rewards = [0.0] * env_count
    episode_max_scores = [0.0] * env_count
    episode_step_counts = [0] * env_count
    episode = 0
    started = time.perf_counter()
    last_report = 0

    print(
        f"Fast sim pretrain: {env_count} envs, batch={agent.batch_size}, "
        f"train_every={agent.train_every}, device={agent.device_message}"
    )
    logger.log_session_start(
        agent,
        extra={
            "target_steps": target_steps,
            "env_count": env_count,
            "fast_observations": config.training.sim_fast_observations,
        },
    )
    print(f"Training log: {logger.path}")

    try:
        while agent.metrics.total_steps < target_steps:
            can_boost = np.asarray([env.can_boost for env in envs], dtype=bool)
            boost_levels = np.asarray(
                [env.boost_level for env in envs],
                dtype=np.float32,
            )
            actions = agent.act_batch(
                states,
                training=True,
                can_boost=can_boost,
                boost_levels=boost_levels,
            )

            prev_states = states.copy()
            rewards = np.zeros(env_count, dtype=np.float32)
            next_states = np.empty_like(states)
            dones = np.zeros(env_count, dtype=np.float32)
            scores: list[float | None] = []

            for index, env in enumerate(envs):
                result = env.step_sync(int(actions[index]))
                rewards[index] = result.reward
                next_states[index] = result.state
                dones[index] = float(result.done)
                scores.append(
                    float(result.frame_state.score)
                    if result.frame_state.score is not None
                    else None
                )
                episode_rewards[index] += result.reward
                if result.frame_state.score is not None:
                    episode_max_scores[index] = max(
                        episode_max_scores[index],
                        float(result.frame_state.score),
                    )
                episode_step_counts[index] += 1
                if result.done:
                    ep_reward = episode_rewards[index]
                    ep_score = episode_max_scores[index]
                    agent.record_episode(ep_reward, ep_score)
                    agent.end_episode()
                    logger.log_episode_end(
                        agent,
                        episode,
                        ep_reward,
                        ep_score,
                        env_id=index,
                        episode_steps=episode_step_counts[index],
                    )
                    if episode > 0 and episode % 25 == 0:
                        agent.save_sim_checkpoint()
                    episode += 1
                    episode_rewards[index] = 0.0
                    episode_max_scores[index] = 0.0
                    episode_step_counts[index] = 0
                    next_states[index] = env.reset_sync()

            agent.remember_batch(
                prev_states,
                actions,
                rewards,
                next_states,
                dones,
            )
            states = next_states
            metrics = agent.advance_steps(env_count)
            for index in range(env_count):
                logger.record_step(
                    int(actions[index]),
                    float(rewards[index]),
                    FrameState(
                        score=int(scores[index]) if scores[index] is not None else None,
                        boost_level=float(boost_levels[index]),
                        can_boost=bool(can_boost[index]),
                        game_over=bool(dones[index]),
                    ),
                    can_boost=bool(can_boost[index]),
                    boost_level=float(boost_levels[index]),
                    done=bool(dones[index]),
                    train_loss=metrics.loss if index == 0 else None,
                    train_ms=metrics.train_ms if index == 0 else None,
                )
            logger.maybe_flush(agent, metrics.total_steps)

            if metrics.total_steps // 5000 > last_report:
                last_report = metrics.total_steps // 5000
                elapsed = time.perf_counter() - started
                sps = metrics.total_steps / max(elapsed, 1e-6)
                print(
                    f"sim step={metrics.total_steps}/{target_steps} "
                    f"sps={sps:.0f} loss={metrics.loss} "
                    f"eps={agent.epsilon:.3f} replay={metrics.replay_size}"
                )
    finally:
        agent.save_sim_checkpoint()
        elapsed = time.perf_counter() - started
        sps = agent.metrics.total_steps / max(elapsed, 1e-6)
        logger.close(agent)
        print(
            f"Saved sim checkpoint to {config.training.sim_checkpoint_path} "
            f"({agent.metrics.total_steps} steps in {elapsed:.1f}s, {sps:.0f} sps)"
        )
        print(f"Training log: {logger.path}")


async def run_training_no_ui(
    config: AppConfig,
    max_episodes: int | None = None,
) -> None:
    phase = "sim" if config.training.sim_mode else "browser"
    logger = TrainingLogger(config, phase)
    backend = None if config.training.sim_mode else BrowserBackend(config.browser)
    env = create_env(config, backend)
    agent = DQNAgent(config)
    load_result = agent.try_autoload()
    print(load_result.message)
    logger.log_session_start(
        agent,
        message=load_result.message,
        extra={"headless": True, "watch_mode": config.training.watch_mode},
    )
    print(f"Training log: {logger.path}")
    if config.demo.use_demos_on_start and not config.training.sim_mode:
        result = ingest_demonstrations(agent, config)
        if result.transitions_added or result.transitions_skipped:
            print(result.status_message)
            logger.log_note(f"demo_ingest={result.status_message}")
    episode_steps = 0
    try:
        if not config.training.sim_mode:
            assert backend is not None
            status = await backend.connect_auto()
            if not status.connected:
                print(status.message)
                logger.log_note(f"browser_connect_failed={status.message}")
                logger.close(agent)
                return
        state = await env.reset()
        episode = agent.progress.episodes_completed
        episode_reward = 0.0
        episode_max_score = 0.0
        prev_step_score = 0.0
        score_velocity = 0.0
        loop_hz = 0.0
        while max_episodes is None or episode < max_episodes:
            step_started = time.perf_counter()
            action = agent.act(
                state,
                training=not config.training.watch_mode,
                can_boost=env.can_boost,
                boost_level=env.boost_level,
            )
            result = await env.step(action)
            step_ms = (time.perf_counter() - step_started) * 1000.0
            if step_ms > 0:
                instant_hz = 1000.0 / step_ms
                loop_hz = (0.85 * loop_hz) + (0.15 * instant_hz)
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
            logger.maybe_flush(agent, metrics.total_steps)
            state = result.state
            if metrics.total_steps % 25 == 0:
                print(
                    f"step={metrics.total_steps} action={ACTION_LABELS[action]} "
                    f"boost={env.boost_level:.0%} reward={episode_reward:.3f} "
                    f"score={episode_max_score:.0f} replay={metrics.replay_size} "
                    f"loss={metrics.loss}"
                )
            if result.done:
                agent.record_episode(episode_reward, episode_max_score)
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
                episode_reward = 0.0
                episode_max_score = 0.0
                episode_steps = 0
                prev_step_score = 0.0
                score_velocity = 0.0
                state = await env.reset()
    finally:
        agent.save()
        await env.close()
        logger.close(agent)
        print(f"Training log: {logger.path}")


async def run_random_smoke(config: AppConfig, steps: int = 100) -> None:
    if config.training.sim_mode:
        env = AscentSimEnv(config)
        backend = None
    else:
        backend = BrowserBackend(config.browser)
        env = AscentGameEnv(config, backend)
    try:
        if backend is not None:
            status = await backend.connect_auto()
            if not status.connected:
                print(status.message)
                return
        state = await env.reset()
        print(f"initial_state_shape={state.shape}")
        for idx in range(steps):
            action = random.randrange(config.action_count)
            result = await env.step(action)
            print(
                f"step={idx} action={ACTION_LABELS[action]} "
                f"reward={result.reward:.3f} done={result.done}"
            )
            if result.done:
                await env.reset()
    finally:
        await env.close()


async def run_sim_calibration(config: AppConfig, episodes: int = 10) -> None:
    stats = await calibrate_sim_physics(config, episodes=episodes)
    print(
        "sim calibration:",
        f"episodes={int(stats['episodes'])}",
        f"mean_score={stats['mean_score']:.1f}",
        f"mean_length={stats['mean_length']:.1f}",
    )

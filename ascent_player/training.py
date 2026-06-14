from __future__ import annotations

import random
import time

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig
from ascent_player.demo.ingest import ingest_demonstrations
from ascent_player.env.browser_backend import BrowserBackend
from ascent_player.env.game_env import ACTION_LABELS, AscentGameEnv
from ascent_player.env.sim_env import AscentSimEnv, calibrate_sim_physics


def create_env(config: AppConfig, backend: BrowserBackend | None = None):
    if config.training.sim_mode:
        return AscentSimEnv(config)
    if backend is None:
        backend = BrowserBackend(config.browser)
    return AscentGameEnv(config, backend)


async def run_sim_pretrain(config: AppConfig, steps: int | None = None) -> None:
    config.training.sim_mode = True
    env = AscentSimEnv(config)
    agent = DQNAgent(config)
    target_steps = steps or config.training.sim_pretrain_steps or 500_000
    episode = 0
    episode_reward = 0.0
    episode_max_score = 0.0
    started = time.perf_counter()
    try:
        state = await env.reset()
        while agent.metrics.total_steps < target_steps:
            action = agent.act(
                state,
                training=True,
                can_boost=env.can_boost,
                boost_level=env.boost_level,
            )
            result = await env.step(action)
            agent.remember(
                state,
                action,
                result.reward,
                result.state,
                result.done,
                sim=False,
            )
            metrics = agent.maybe_train()
            state = result.state
            episode_reward += result.reward
            if result.frame_state.score is not None:
                episode_max_score = max(episode_max_score, float(result.frame_state.score))

            if metrics.total_steps % 5000 == 0 and metrics.total_steps > 0:
                elapsed = time.perf_counter() - started
                sps = metrics.total_steps / max(elapsed, 1e-6)
                print(
                    f"sim step={metrics.total_steps}/{target_steps} "
                    f"sps={sps:.0f} loss={metrics.loss} "
                    f"eps={agent.epsilon:.3f} replay={metrics.replay_size}"
                )

            if result.done:
                agent.record_episode(episode_reward, episode_max_score)
                agent.end_episode()
                if episode > 0 and episode % 25 == 0:
                    agent.save_sim_checkpoint()
                episode += 1
                episode_reward = 0.0
                episode_max_score = 0.0
                state = await env.reset()
    finally:
        agent.save_sim_checkpoint()
        await env.close()
        print(f"Saved sim checkpoint to {config.training.sim_checkpoint_path}")


async def run_training_no_ui(
    config: AppConfig,
    max_episodes: int | None = None,
) -> None:
    backend = None if config.training.sim_mode else BrowserBackend(config.browser)
    env = create_env(config, backend)
    agent = DQNAgent(config)
    load_result = agent.try_autoload()
    print(load_result.message)
    if config.demo.use_demos_on_start and not config.training.sim_mode:
        result = ingest_demonstrations(agent, config)
        if result.transitions_added or result.transitions_skipped:
            print(result.status_message)
    try:
        if not config.training.sim_mode:
            assert backend is not None
            status = await backend.connect_auto()
            if not status.connected:
                print(status.message)
                return
        state = await env.reset()
        episode = agent.progress.episodes_completed
        episode_reward = 0.0
        episode_max_score = 0.0
        while max_episodes is None or episode < max_episodes:
            action = agent.act(
                state,
                training=not config.training.watch_mode,
                can_boost=env.can_boost,
                boost_level=env.boost_level,
            )
            result = await env.step(action)
            agent.remember(
                state,
                action,
                result.reward,
                result.state,
                result.done,
                sim=config.training.sim_mode,
            )
            metrics = agent.maybe_train()
            state = result.state
            episode_reward += result.reward
            if result.frame_state.score is not None:
                episode_max_score = max(
                    episode_max_score,
                    float(result.frame_state.score),
                )
            if metrics.total_steps % 25 == 0:
                print(
                    f"step={metrics.total_steps} action={ACTION_LABELS[action]} "
                    f"boost={env.boost_level:.0%} reward={episode_reward:.3f} "
                    f"score={episode_max_score:.0f} replay={metrics.replay_size} "
                    f"loss={metrics.loss}"
                )
            if result.done:
                agent.record_episode(episode_reward, episode_max_score)
                print(
                    f"episode={episode} reward={episode_reward:.3f} "
                    f"score={episode_max_score:.0f} epsilon={agent.epsilon:.3f}"
                )
                agent.end_episode()
                agent.maybe_autosave(force=True)
                episode += 1
                episode_reward = 0.0
                episode_max_score = 0.0
                state = await env.reset()
    finally:
        agent.save()
        await env.close()


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

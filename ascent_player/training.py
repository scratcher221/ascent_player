from __future__ import annotations

import random

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig
from ascent_player.env.browser_backend import BrowserBackend
from ascent_player.env.game_env import ACTION_LABELS, AscentGameEnv


async def run_training_no_ui(config: AppConfig, max_episodes: int | None = None) -> None:
    backend = BrowserBackend(config.browser)
    env = AscentGameEnv(config, backend)
    agent = DQNAgent(config)
    try:
        status = await backend.connect_auto()
        if not status.connected:
            print(status.message)
            return
        state = await env.reset()
        episode = 0
        episode_reward = 0.0
        while max_episodes is None or episode < max_episodes:
            action = agent.act(
                state,
                training=not config.training.watch_mode,
                can_boost=env.can_boost,
            )
            result = await env.step(action)
            agent.remember(state, action, result.reward, result.state, result.done)
            metrics = agent.maybe_train()
            state = result.state
            episode_reward += result.reward
            if metrics.total_steps % 25 == 0:
                print(
                    f"step={metrics.total_steps} action={ACTION_LABELS[action]} "
                    f"boost={env.boost_level:.0%} reward={episode_reward:.1f} "
                    f"replay={metrics.replay_size} loss={metrics.loss}"
                )
            if result.done:
                print(
                    f"episode={episode} reward={episode_reward:.1f} "
                    f"epsilon={agent.epsilon:.3f}"
                )
                agent.end_episode()
                episode += 1
                episode_reward = 0.0
                state = await env.reset()
    finally:
        agent.save()
        await env.close()


async def run_random_smoke(config: AppConfig, steps: int = 100) -> None:
    backend = BrowserBackend(config.browser)
    env = AscentGameEnv(config, backend)
    try:
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
                f"reward={result.reward:.2f} done={result.done}"
            )
            if result.done:
                await env.reset()
    finally:
        await env.close()

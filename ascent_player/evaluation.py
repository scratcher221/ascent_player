"""Evaluate learned and rule-based policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ascent_player.agent.dqn import DQNAgent
from ascent_player.agent.teacher import RulePolicy
from ascent_player.config import AppConfig
from ascent_player.env.game_env import AscentGameEnv
from ascent_player.env.sim_env import AscentSimEnv
from ascent_player.env.state_detector import FrameState


@dataclass(slots=True)
class SkillMetrics:
    episodes: int
    mean_score: float
    max_score: float
    mean_length: float
    landing_rate: float
    mean_platform_dx: float
    meaningful_boost_rate: float
    wasted_boost_rate: float
    min_score: float = 0.0


def _episode_skill(
    *,
    steps: int,
    bounces: int,
    platform_dx_sum: float,
    platform_samples: int,
    meaningful_boosts: int,
    wasted_boosts: int,
    jump_actions: int,
    max_score: float,
) -> dict[str, float]:
    return {
        "steps": float(steps),
        "score": max_score,
        "landing_rate": bounces / max(steps, 1),
        "mean_platform_dx": platform_dx_sum / max(platform_samples, 1),
        "meaningful_boost_rate": meaningful_boosts / max(jump_actions, 1),
        "wasted_boost_rate": wasted_boosts / max(jump_actions, 1),
    }


async def evaluate_rule_baseline(
    config: AppConfig,
    *,
    episodes: int = 10,
    use_sim: bool = False,
) -> SkillMetrics:
    return await _evaluate_policy(
        config,
        episodes=episodes,
        use_sim=use_sim,
        agent=None,
        rule_only=True,
    )


async def evaluate_learned_policy(
    config: AppConfig,
    *,
    episodes: int = 10,
    use_sim: bool = False,
) -> SkillMetrics:
    if use_sim:
        config.training.sim_mode = True
        config.training.checkpoint_path = config.training.sim_checkpoint_path
    agent = DQNAgent(config)
    loaded = agent.load(config.training.checkpoint_path)
    if not loaded:
        detail = agent._last_load_error or "missing"
        print(f"Warning: could not load checkpoint ({detail}); evaluating untrained weights")
    agent.epsilon = 0.0
    return await _evaluate_policy(
        config,
        episodes=episodes,
        use_sim=use_sim,
        agent=agent,
        rule_only=False,
    )


async def _evaluate_policy(
    config: AppConfig,
    *,
    episodes: int,
    use_sim: bool,
    agent: DQNAgent | None,
    rule_only: bool,
) -> SkillMetrics:
    rule = RulePolicy()
    if use_sim:
        env = AscentSimEnv(config, fast_mode=True)
        backend = None
    else:
        from ascent_player.env.browser_backend import BrowserBackend

        backend = BrowserBackend(config.browser)
        env = AscentGameEnv(config, backend)
        status = await backend.connect_auto()
        if not status.connected:
            raise RuntimeError(status.message)

    episode_stats: list[dict[str, float]] = []
    try:
        for _ in range(episodes):
            state = await env.reset()
            steps = 0
            bounces = 0
            prev_bounces = 0
            platform_dx_sum = 0.0
            platform_samples = 0
            meaningful_boosts = 0
            wasted_boosts = 0
            jump_actions = 0
            max_score = 0.0
            done = False
            while not done and steps < 1200:
                frame_state = env._last_frame_state
                if frame_state is None:
                    break
                if rule_only:
                    action = rule.act(frame_state)
                else:
                    assert agent is not None
                    action = agent.act(
                        state,
                        training=False,
                        can_boost=env.can_boost,
                        boost_level=env.boost_level,
                        frame_state=frame_state,
                    )
                if action in (3, 4, 5):
                    jump_actions += 1
                    if frame_state.boost_useful:
                        meaningful_boosts += 1
                    elif not env.can_boost:
                        wasted_boosts += 1
                if frame_state.nearest_platform_dx is not None:
                    platform_dx_sum += abs(frame_state.nearest_platform_dx)
                    platform_samples += 1
                result = await env.step(action)
                state = result.state
                steps += 1
                if result.frame_state.bounces > prev_bounces:
                    bounces += 1
                    prev_bounces = result.frame_state.bounces
                if result.frame_state.score is not None:
                    max_score = max(max_score, float(result.frame_state.score))
                done = result.done
            episode_stats.append(
                _episode_skill(
                    steps=steps,
                    bounces=bounces,
                    platform_dx_sum=platform_dx_sum,
                    platform_samples=platform_samples,
                    meaningful_boosts=meaningful_boosts,
                    wasted_boosts=wasted_boosts,
                    jump_actions=jump_actions,
                    max_score=max_score,
                )
            )
    finally:
        await env.close()
        if backend is not None:
            await backend.stop()

    if not episode_stats:
        return SkillMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return SkillMetrics(
        episodes=len(episode_stats),
        mean_score=float(np.mean([item["score"] for item in episode_stats])),
        max_score=float(max(item["score"] for item in episode_stats)),
        mean_length=float(np.mean([item["steps"] for item in episode_stats])),
        landing_rate=float(np.mean([item["landing_rate"] for item in episode_stats])),
        mean_platform_dx=float(np.mean([item["mean_platform_dx"] for item in episode_stats])),
        meaningful_boost_rate=float(
            np.mean([item["meaningful_boost_rate"] for item in episode_stats])
        ),
        wasted_boost_rate=float(np.mean([item["wasted_boost_rate"] for item in episode_stats])),
        min_score=float(min(item["score"] for item in episode_stats)),
    )


def format_skill_metrics(label: str, metrics: SkillMetrics) -> str:
    return (
        f"{label}: episodes={metrics.episodes} "
        f"mean_score={metrics.mean_score:.1f} max_score={metrics.max_score:.1f} "
        f"min_score={metrics.min_score:.1f} "
        f"mean_length={metrics.mean_length:.0f} landing_rate={metrics.landing_rate:.3f} "
        f"mean_platform_dx={metrics.mean_platform_dx:.3f} "
        f"meaningful_boost={metrics.meaningful_boost_rate:.2f} "
        f"wasted_boost={metrics.wasted_boost_rate:.2f}"
    )


def score_gates(mean_score: float, min_score: float, config: AppConfig) -> list[str]:
    """Return which Phase-0 success gates the eval currently meets."""
    training = config.training
    passed: list[str] = []
    if mean_score >= training.gate_a_score:
        passed.append("A")
    if mean_score >= training.gate_b_score:
        passed.append("B")
    if mean_score >= training.gate_c_score:
        passed.append("C")
    if (
        mean_score >= training.gate_d_score
        and min_score >= training.gate_d_min_score
    ):
        passed.append("D")
    return passed


def evaluate_sim_greedy_sync(
    agent: DQNAgent,
    config: AppConfig,
    *,
    episodes: int | None = None,
    max_steps: int | None = None,
) -> SkillMetrics:
    """ε=0 sim eval using sync env API (safe to call mid-pretrain)."""
    episodes = episodes or config.training.sim_eval_episodes
    max_steps = max_steps or config.training.sim_eval_max_steps
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0
    episode_stats: list[dict[str, float]] = []
    try:
        for ep_index in range(max(1, episodes)):
            env = AscentSimEnv(config, fast_mode=True, env_index=10_000 + ep_index)
            try:
                state = env.reset_sync()
                steps = 0
                bounces = 0
                prev_bounces = 0
                platform_dx_sum = 0.0
                platform_samples = 0
                meaningful_boosts = 0
                wasted_boosts = 0
                jump_actions = 0
                max_score = 0.0
                done = False
                while not done and steps < max_steps:
                    frame_state = env._last_frame_state
                    if frame_state is None:
                        break
                    action = agent.act(
                        state,
                        training=False,
                        can_boost=env.can_boost,
                        boost_level=env.boost_level,
                        frame_state=frame_state,
                    )
                    if action in (3, 4, 5):
                        jump_actions += 1
                        if frame_state.boost_useful:
                            meaningful_boosts += 1
                        elif not env.can_boost:
                            wasted_boosts += 1
                    if frame_state.nearest_platform_dx is not None:
                        platform_dx_sum += abs(frame_state.nearest_platform_dx)
                        platform_samples += 1
                    result = env.step_sync(action)
                    state = result.state
                    steps += 1
                    if result.frame_state.bounces > prev_bounces:
                        bounces += 1
                        prev_bounces = result.frame_state.bounces
                    if result.frame_state.score is not None:
                        max_score = max(max_score, float(result.frame_state.score))
                    done = result.done
                episode_stats.append(
                    _episode_skill(
                        steps=steps,
                        bounces=bounces,
                        platform_dx_sum=platform_dx_sum,
                        platform_samples=platform_samples,
                        meaningful_boosts=meaningful_boosts,
                        wasted_boosts=wasted_boosts,
                        jump_actions=jump_actions,
                        max_score=max_score,
                    )
                )
            finally:
                pass
    finally:
        agent.epsilon = saved_epsilon

    if not episode_stats:
        return SkillMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return SkillMetrics(
        episodes=len(episode_stats),
        mean_score=float(np.mean([item["score"] for item in episode_stats])),
        max_score=float(max(item["score"] for item in episode_stats)),
        mean_length=float(np.mean([item["steps"] for item in episode_stats])),
        landing_rate=float(np.mean([item["landing_rate"] for item in episode_stats])),
        mean_platform_dx=float(np.mean([item["mean_platform_dx"] for item in episode_stats])),
        meaningful_boost_rate=float(
            np.mean([item["meaningful_boost_rate"] for item in episode_stats])
        ),
        wasted_boost_rate=float(np.mean([item["wasted_boost_rate"] for item in episode_stats])),
        min_score=float(min(item["score"] for item in episode_stats)),
    )

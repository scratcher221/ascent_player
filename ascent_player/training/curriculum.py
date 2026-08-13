from __future__ import annotations

import random

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig
from ascent_player.env.sim_env import AscentSimEnv
from ascent_player.env.state_detector import FrameState
from ascent_player.mechanics_curriculum import CurriculumMetrics, mechanics_stage_from_metrics


_curriculum_metrics = CurriculumMetrics()


def apply_curriculum(config: AppConfig, agent: DQNAgent, env, *, episode_steps: int = 0, frame_state: FrameState | None = None) -> str:
    if frame_state is not None:
        _curriculum_metrics.record_episode(
            steps=episode_steps,
            bounces=frame_state.bounces,
            height=frame_state.height,
            combo=frame_state.combo,
            score=float(frame_state.score or 0),
        )
    stage = mechanics_stage_from_metrics(_curriculum_metrics, config)
    agent.curriculum_stage = stage
    if hasattr(env, "reward_tracker"):
        env.reward_tracker.set_curriculum_stage(stage)
    return stage


async def warmstart_from_teacher(agent: DQNAgent, config: AppConfig) -> int:
    from ascent_player.agent.teacher import RuleTeacher

    env = AscentSimEnv(config, fast_mode=True)
    teacher = RuleTeacher()
    added = 0
    try:
        for _ in range(config.mechanics_curriculum.teacher_episodes):
            state = await env.reset()
            for _ in range(600):
                frame_state = env._last_frame_state
                if frame_state is None:
                    break
                action = teacher.act(frame_state)
                result = await env.step(action)
                agent.remember(state, action, result.reward, result.state, result.done, sim=True)
                state = result.state
                added += 1
                if result.done:
                    break
    finally:
        await env.close()
    if added:
        # Promote teacher transitions into the main replay used by sim pretrain.
        agent.replay.extend_from(agent.sim_replay)
        agent.pretrain_from_replay(steps=min(config.demo.pretrain_steps, added // 4))
        agent.epsilon = config.mechanics_curriculum.teacher_warmstart_epsilon
        agent.metrics.epsilon = agent.epsilon
        agent.progress.epsilon = agent.epsilon
    return added


def warmstart_from_teacher_sync(agent: DQNAgent, config: AppConfig) -> int:
    from ascent_player.agent.teacher import RuleTeacher

    env = AscentSimEnv(config, fast_mode=True)
    teacher = RuleTeacher()
    added = 0
    try:
        for _ in range(config.mechanics_curriculum.teacher_episodes):
            state = env.reset_sync()
            for _ in range(800):
                frame_state = env._last_frame_state
                if frame_state is None:
                    break
                action = teacher.act(frame_state)
                result = env.step_sync(action)
                agent.remember(
                    state,
                    action,
                    result.reward,
                    result.state,
                    result.done,
                )
                state = result.state
                added += 1
                if result.done:
                    break
    finally:
        pass
    # Teacher transitions are hybrid-compatible; keep them in the main replay.
    if added:
        agent.pretrain_from_replay(steps=min(config.demo.pretrain_steps, max(50, added // 4)))
        # Note: teacher used remember() into self.replay already.
        agent.epsilon = config.mechanics_curriculum.teacher_warmstart_epsilon
        agent._epsilon_anneal_start = agent.epsilon
        agent.metrics.epsilon = agent.epsilon
        agent.progress.epsilon = agent.epsilon
    return added


def curriculum_start_height(agent: DQNAgent, config: AppConfig) -> float:
    """Phase 4: sometimes start mid-climb once the agent can survive there."""
    import random

    mc = config.mechanics_curriculum
    best = float(agent.progress.best_score)
    # Prefer bottom-starts until eval-quality scores are real (not start-height inflated).
    if best < mc.start_height_gate_b:
        return 0.0
    if random.random() > mc.start_height_prob:
        return 0.0
    if best >= mc.start_height_gate_c:
        return float(random.uniform(0.0, mc.start_height_max_c))
    return float(random.uniform(0.0, mc.start_height_max_b))


def episode_score_for_progress(ep_score: float, start_height: float, config: AppConfig) -> float:
    """Optionally strip free score from mid-climb starts so curriculum stays honest."""
    if config.mechanics_curriculum.start_height_score_credit or start_height <= 0:
        return float(ep_score)
    free = float(start_height) / 5.0
    return float(max(0.0, ep_score - free))


def meets_consistency_target(mean_score: float, min_score: float, config: AppConfig) -> bool:
    return (
        mean_score >= config.training.consistency_mean_score
        and min_score >= config.training.consistency_min_score
    )


def meets_reliability_target(mean_score: float, min_score: float, config: AppConfig) -> bool:
    return (
        mean_score >= config.training.reliability_mean_score
        and min_score >= config.training.reliability_min_score
    )



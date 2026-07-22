from __future__ import annotations

import os
import time

import numpy as np

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig
from ascent_player.demo.ingest import ingest_demonstrations
from ascent_player.env.browser_backend import BrowserBackend
from ascent_player.env.game_env import ACTION_LABELS, AscentGameEnv
from ascent_player.env.sim_env import AscentSimEnv, calibrate_sim_physics
from ascent_player.env.state_detector import FrameState
from pathlib import Path
from ascent_player.utils.gpu_session import release_gpu_between_runs
from ascent_player.utils.decision_log import DecisionLogger
from ascent_player.utils.training_log import BrowserStepContext, TrainingLogger
from ascent_player.env.target_detector import TargetDetectionTracker
from ascent_player.agent.reason import wrong_vs_target


from ascent_player.mechanics_curriculum import CurriculumMetrics, mechanics_stage_from_metrics
from ascent_player.evaluation import (
    evaluate_sim_greedy_sync,
    format_skill_metrics,
    score_gates,
)


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
    eval_config.training = replace(
        config.training,
        sim_mode=False,
        transfer_from_sim=False,
        watch_mode=True,
        epsilon_start=0.0,
        epsilon_end=0.0,
        frame_skip=max(1, min(config.training.transfer_frame_skip, config.training.frame_skip)),
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


def _sim_env_count(config: AppConfig) -> int:
    if config.training.sim_pretrain_envs > 0:
        return config.training.sim_pretrain_envs
    cpus = os.cpu_count() or 8
    if config.training.sim_fast_observations:
        return max(4, min(16, cpus - 2))
    # Rendered frames are heavier — keep parallelism modest for transfer quality.
    return max(2, min(6, max(2, cpus // 4)))


def run_sim_pretrain(config: AppConfig, steps: int | None = None) -> None:
    """Fast vectorized simulator pretraining (sync, batched inference)."""
    config.training.sim_mode = True
    # Bulk pretrain stays FS=1 for speed; visual bridge keeps transfer_frame_skip.
    if not config.training.sim_keep_frame_skip:
        config.training.frame_skip = 1
    env_count = _sim_env_count(config)
    target_steps = steps or config.training.sim_pretrain_steps or 500_000
    max_steps = int(target_steps * config.training.sim_max_steps_multiplier)
    min_best = config.training.sim_min_best_score

    envs = [
        AscentSimEnv(
            config,
            fast_mode=None,  # honor config.training.sim_fast_observations
            env_index=index,
        )
        for index in range(env_count)
    ]
    agent = DQNAgent(config)
    agent.apply_sim_pretrain_profile()
    logger = TrainingLogger(config, "sim")

    resumed = False
    if config.training.sim_resume_from_best:
        resume_path = agent.resolve_best_checkpoint()
        if resume_path is not None and agent.load(resume_path):
            resumed = True
            resume_eps = max(
                float(config.training.sim_resume_epsilon),
                float(config.training.sim_epsilon_end),
            )
            # Resume explores from a low but non-zero ε (do not restart at 1.0).
            agent.epsilon = resume_eps
            agent._epsilon_anneal_start = agent.epsilon
            agent.metrics.epsilon = agent.epsilon
            agent.progress.epsilon = agent.epsilon
            msg = (
                f"Resumed sim pretrain from {resume_path.name} "
                f"(steps={agent.progress.total_steps}, "
                f"best={agent.progress.best_score:.0f}, ε={agent.epsilon:.3f})"
            )
            print(msg, flush=True)
            logger.log_note(msg)

    if (
        not resumed
        and config.training.sim_warmstart_demos
        and config.demo.use_demos_on_start
    ):
        from ascent_player.demo.ingest import ingest_demonstrations

        demo_result = ingest_demonstrations(agent, config, replace_buffer=False)
        if demo_result.transitions_added:
            print(demo_result.status_message)
            logger.log_note(f"demo_warmstart={demo_result.status_message}")
            # Keep demos in demo_replay for BC only — do not mix into main replay
            # (demo tensors are often visual-only / shape-incompatible with hybrid states).
            agent.pretrain_from_replay(
                steps=min(
                    config.demo.pretrain_steps,
                    max(100, demo_result.transitions_added // 20),
                )
            )

    if config.training.sim_warmstart_teacher:
        # Also allow on resume — aligned-physics restarts often need a fresh teacher buffer.
        teacher_added = warmstart_from_teacher_sync(agent, config)
        if teacher_added:
            print(f"Teacher warm-start added {teacher_added} transitions")
            logger.log_note(f"teacher_warmstart={teacher_added}")

    states = [env.reset_sync() for env in envs]
    hybrid_states = config.observation.include_vector_state
    episode_rewards = [0.0] * env_count
    episode_max_scores = [0.0] * env_count
    episode_step_counts = [0] * env_count
    episode_start_heights = [0.0] * env_count
    episode = 0
    started = time.perf_counter()
    last_report = 0
    eval_every = max(0, config.training.sim_eval_every_steps)
    last_eval = (
        agent.metrics.total_steps // eval_every
        if resumed and eval_every > 0
        else 0
    )
    best_eval_mean = 0.0
    best_eval_min = 0.0
    consistency_met = False

    obs_mode = (
        "fast_stick"
        if config.training.sim_fast_observations
        else "rendered_browserlike"
    )
    print(
        f"Sim pretrain ({obs_mode}): {env_count} envs, batch={agent.batch_size}, "
        f"train_every={agent.train_every}, device={agent.device_message}"
    )
    logger.log_session_start(
        agent,
        extra={
            "target_steps": target_steps,
            "env_count": env_count,
            "fast_observations": config.training.sim_fast_observations,
            "obs_mode": obs_mode,
        },
    )
    print(f"Training log: {logger.path}")

    try:
        while agent.metrics.total_steps < target_steps or (
            agent.progress.best_score < min_best
            and agent.metrics.total_steps < max_steps
        ):
            if consistency_met:
                break
            can_boost = np.asarray([env.can_boost for env in envs], dtype=bool)
            boost_levels = np.asarray(
                [env.boost_level for env in envs],
                dtype=np.float32,
            )
            frame_states = [env._last_frame_state for env in envs]
            actions = agent.act_batch(
                states,
                training=True,
                can_boost=can_boost,
                boost_levels=boost_levels,
                frame_states=frame_states,
            )

            prev_states = list(states) if hybrid_states else states.copy()
            rewards = np.zeros(env_count, dtype=np.float32)
            next_states = [None] * env_count if hybrid_states else np.empty_like(states)
            dones = np.zeros(env_count, dtype=np.float32)
            scores: list[float | None] = []
            result_frames: list[FrameState | None] = [None] * env_count

            for index, env in enumerate(envs):
                result = env.step_sync(int(actions[index]))
                rewards[index] = result.reward
                result_frames[index] = result.frame_state
                if hybrid_states:
                    next_states[index] = result.state
                else:
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
                    ep_score = episode_score_for_progress(
                        episode_max_scores[index],
                        episode_start_heights[index],
                        config,
                    )
                    agent.record_episode(ep_reward, ep_score, sim_pretrain=True)
                    agent.end_episode(sim_pretrain=True)
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
                    start_h = curriculum_start_height(agent, config)
                    episode_start_heights[index] = start_h
                    next_states[index] = env.reset_sync(start_height=start_h)

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
                frame = result_frames[index]
                logger.record_step(
                    int(actions[index]),
                    float(rewards[index]),
                    frame
                    if frame is not None
                    else FrameState(
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
                    f"(max={max_steps}) sps={sps:.0f} loss={metrics.loss} "
                    f"eps={agent.epsilon:.3f} best={agent.progress.best_score:.0f} "
                    f"replay={metrics.replay_size}"
                )

            eval_every = max(0, config.training.sim_eval_every_steps)
            if (
                eval_every > 0
                and metrics.total_steps >= eval_every
                and metrics.total_steps // eval_every > last_eval
            ):
                last_eval = metrics.total_steps // eval_every
                eval_metrics = evaluate_sim_greedy_sync(agent, config)
                gates = score_gates(
                    eval_metrics.mean_score,
                    eval_metrics.min_score,
                    config,
                )
                gate_label = ",".join(gates) if gates else "none"
                promote_min_floor = float(config.training.sim_eval_promote_min)
                min_ok = (
                    eval_metrics.min_score >= best_eval_min
                    if best_eval_mean > 0
                    else eval_metrics.min_score >= 0
                )
                floor_ok = (
                    best_eval_mean <= 0
                    or eval_metrics.min_score >= promote_min_floor
                    or eval_metrics.min_score >= best_eval_min
                )
                if (
                    eval_metrics.mean_score > best_eval_mean
                    and min_ok
                    and floor_ok
                ):
                    best_eval_mean = eval_metrics.mean_score
                    best_eval_min = float(eval_metrics.min_score)
                    agent.save(config.training.sim_best_eval_checkpoint_path)
                    agent.save(config.training.playable_checkpoint_path)
                    agent.save(config.training.checkpoint_path)
                    logger.log_note(
                        f"best_eval_ckpt mean={best_eval_mean:.0f} "
                        f"min={best_eval_min:.0f} "
                        f"-> {config.training.sim_best_eval_checkpoint_path} "
                        f"(also playable + dqn_latest)"
                    )
                msg = (
                    f"GATED EVAL ε=0 @ step={metrics.total_steps}: "
                    f"{format_skill_metrics('sim', eval_metrics)} "
                    f"gates={gate_label} best_eval_mean={best_eval_mean:.0f}"
                )
                print(msg)
                logger.log_note(msg)
                if "A" in gates:
                    agent.save_sim_checkpoint()
                # Probe once mean reaches Gate B (or Gate C), whichever fits the target.
                probe_threshold = min(
                    config.training.gate_c_score,
                    max(
                        config.training.gate_b_score,
                        config.training.consistency_mean_score * 0.7,
                    ),
                )
                if eval_metrics.mean_score >= probe_threshold:
                    consistency = evaluate_sim_greedy_sync(
                        agent,
                        config,
                        episodes=config.training.consistency_eval_episodes,
                        max_steps=config.training.sim_eval_max_steps,
                    )
                    cmsg = (
                        f"CONSISTENCY ε=0: {format_skill_metrics('sim', consistency)}"
                    )
                    print(cmsg)
                    logger.log_note(cmsg)
                    if meets_consistency_target(
                        consistency.mean_score,
                        consistency.min_score,
                        config,
                    ):
                        consistency_met = True
                        agent.save_sim_checkpoint()
                        agent.save(config.training.sim_best_eval_checkpoint_path)
                        agent.save(config.training.playable_checkpoint_path)
                        agent.save(config.training.checkpoint_path)
                        print("Consistency target met — stopping pretrain early.")
                        logger.log_note("consistency_target_met=true")
    finally:
        if agent.progress.best_score >= min_best:
            agent.save_sim_checkpoint()
            gate_msg = "passed"
        else:
            agent.save_sim_checkpoint()
            gate_msg = f"below gate ({agent.progress.best_score:.0f} < {min_best})"
        elapsed = time.perf_counter() - started
        sps = agent.metrics.total_steps / max(elapsed, 1e-6)
        logger.close(agent)
        print(
            f"Saved sim checkpoint to {config.training.sim_checkpoint_path} "
            f"({agent.metrics.total_steps} steps in {elapsed:.1f}s, {sps:.0f} sps, "
            f"best_score={agent.progress.best_score:.0f}, gate={gate_msg}, "
            f"best_eval_mean={best_eval_mean:.0f}, consistency={consistency_met})"
        )
        print(f"Training log: {logger.path}")


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
        loaded = agent.replay.load_pickle(
            replay_path,
            max_items=config.training.browser_replay_max_items,
        )
        if loaded:
            print(f"Loaded {loaded} browser replay transitions from {replay_path.name}")
            # Decay mixed sim replay as browser experience accumulates.
            fill = min(1.0, loaded / max(1, config.training.browser_replay_max_items))
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
        if len(agent.replay) == 0:
            teacher_added = await warmstart_from_teacher(agent, config)
            if teacher_added:
                print(f"Teacher warm-start added {teacher_added} sim transitions")
                logger.log_note(f"teacher_warmstart={teacher_added}")
        episode = agent.progress.episodes_completed
        episodes_run = 0
        episode_reward = 0.0
        episode_max_score = 0.0
        prev_step_score = 0.0
        score_velocity = 0.0
        loop_hz = 0.0
        while max_episodes is None or episodes_run < max_episodes:
            if deadline is not None and time.perf_counter() >= deadline:
                print(f"Finetune time limit reached ({max_seconds}s)")
                logger.log_note(f"finetune_timeout={max_seconds}s")
                break
            step_started = time.perf_counter()
            action = agent.act(
                state,
                training=not config.training.watch_mode,
                can_boost=env.can_boost,
                boost_level=env.boost_level,
                frame_state=env._last_frame_state,
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
            if result.done:
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
            saved = agent.replay.save_pickle(
                config.training.browser_replay_path,
                max_items=config.training.browser_replay_max_items,
            )
            if saved:
                print(
                    f"Saved {saved} browser replay transitions -> "
                    f"{config.training.browser_replay_path}"
                )
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


async def run_sim_calibration(config: AppConfig, episodes: int = 10) -> None:
    stats = await calibrate_sim_physics(config, episodes=episodes)
    print(
        "sim calibration:",
        f"episodes={int(stats['episodes'])}",
        f"random_mean={stats['mean_score']:.1f}",
        f"climb_mean={stats['climb_mean_score']:.1f}",
        f"max={stats['max_score']:.1f}",
        f"p90={stats['p90_score']:.1f}",
        f"mean_length={stats['mean_length']:.1f}",
    )

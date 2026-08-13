from __future__ import annotations

import time

import numpy as np

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig
from ascent_player.env.sim_env import AscentSimEnv, calibrate_sim_physics
from ascent_player.env.state_detector import FrameState
from ascent_player.evaluation import evaluate_sim_greedy_sync, score_gates
from ascent_player.training.curriculum import (
    curriculum_start_height,
    episode_score_for_progress,
    meets_consistency_target,
    warmstart_from_teacher_sync,
)
from ascent_player.utils.training_log import TrainingLogger


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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time

import numpy as np

from ascent_player.agent.checkpoint import (
    checkpoint_exists,
    prefer_checkpoint,
    LoadResult,
    TrainingProgress,
    load_progress,
    save_progress,
)
from ascent_player.agent.progress_sanitize import sanitize_browser_progress
from ascent_player.agent.model import build_q_network
from ascent_player.agent.replay_buffer import ReplayBuffer, TransitionBatch
from ascent_player.agent.teacher import RulePolicy
from ascent_player.config import AppConfig
from ascent_player.env.state_detector import FrameState
from ascent_player.utils.device import (
    DeviceInfo,
    benchmark_inference_device,
    import_tensorflow,
    resolve_device,
)


@dataclass(slots=True)
class AgentMetrics:
    loss: float | None = None
    train_ms: float | None = None
    epsilon: float = 1.0
    replay_size: int = 0
    total_steps: int = 0


class DQNAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.tf = import_tensorflow(config.training.device_mode)
        self.device_info: DeviceInfo = resolve_device(config.training.device_mode)
        self.batch_size = (
            config.training.batch_size_gpu
            if self.device_info.training_device.startswith("/GPU")
            else config.training.batch_size_cpu
        )
        self.train_every = (
            config.training.train_every_gpu
            if self.device_info.training_device.startswith("/GPU")
            else config.training.train_every_cpu
        )
        self.epsilon = config.training.epsilon_start
        self.metrics = AgentMetrics(epsilon=self.epsilon)
        self.replay = ReplayBuffer(
            config.training.replay_buffer_size,
            prioritized=config.training.use_prioritized_replay,
        )
        self.demo_replay = ReplayBuffer(config.training.replay_buffer_size)
        self.sim_replay = ReplayBuffer(
            config.training.replay_buffer_size,
            prioritized=config.training.use_prioritized_replay,
        )
        self.rule_policy = RulePolicy()
        self._n_step_queue: list[tuple] = []
        self._n_step_queues: list[list[tuple]] = []
        self._last_load_error: str | None = None
        self._epsilon_anneal_start: float = config.training.epsilon_start
        self.progress = TrainingProgress(
            baseline_episodes=config.training.baseline_episodes,
            epsilon=config.training.epsilon_start,
        )
        self._baseline_samples: list[tuple[float, float]] = []
        self._last_autosave_steps = 0
        self._episodes_since_best = 0
        self._sim_pretrain_mode = False
        self.curriculum_stage = "M0"
        input_shape = (
            config.observation.height,
            config.observation.width,
            config.observation.channel_count,
        )
        self._vector_dim = (
            config.observation.vector_dim
            if config.observation.include_vector_state
            else 0
        )

        with self.tf.device(self.device_info.training_device):
            self.online = build_q_network(
                input_shape,
                config.action_count,
                config.training.learning_rate,
                vector_dim=self._vector_dim,
                dueling=config.training.dueling_dqn,
            )
            self.target = build_q_network(
                input_shape,
                config.action_count,
                config.training.learning_rate,
                vector_dim=self._vector_dim,
                dueling=config.training.dueling_dqn,
            )
            self.target.set_weights(self.online.get_weights())

        sample = self._sample_model_input(np.zeros(input_shape, dtype=np.float32))
        self.device_info.inference_device = benchmark_inference_device(
            self.online,
            sample,
            self.device_info,
        )
        self._batch_predict = self._build_batch_predict()

    def _sample_model_input(self, visual: np.ndarray) -> np.ndarray | list[np.ndarray]:
        if self._vector_dim > 0:
            return [
                visual,
                np.zeros(self._vector_dim, dtype=np.float32),
            ]
        return visual

    def _to_model_batch(self, states) -> np.ndarray | list[np.ndarray]:
        if self._vector_dim <= 0:
            arr = np.asarray(states, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[None, ...]
            return arr

        def zero_vectors(batch_size: int) -> np.ndarray:
            return np.zeros((batch_size, self._vector_dim), dtype=np.float32)

        def split_hybrid_item(item):
            if isinstance(item, (tuple, list)) and len(item) == 2:
                return item[0], item[1]
            if isinstance(item, np.ndarray) and item.shape == (2,):
                return item[0], item[1]
            return item, None

        if isinstance(states, np.ndarray):
            if states.dtype == object and states.ndim == 2 and states.shape[1] == 2:
                visuals = np.stack(states[:, 0], axis=0).astype(np.float32)
                vectors = np.stack(states[:, 1], axis=0).astype(np.float32)
                return [visuals, vectors]
            if states.dtype == object:
                items = list(states)
            elif states.ndim == 4:
                visuals = np.asarray(states, dtype=np.float32)
                return [visuals, zero_vectors(visuals.shape[0])]
            elif states.ndim == 3:
                visuals = np.asarray(states, dtype=np.float32)[None, ...]
                return [visuals, zero_vectors(1)]
            else:
                items = list(states)
        elif (
            isinstance(states, (tuple, list))
            and len(states) == 2
            and isinstance(states[0], np.ndarray)
            and states[0].ndim == 3
            and isinstance(states[1], np.ndarray)
            and states[1].ndim == 1
        ):
            return [
                np.asarray(states[0], dtype=np.float32)[None, ...],
                np.asarray(states[1], dtype=np.float32)[None, ...],
            ]
        else:
            items = list(states)

        if isinstance(items[0], (tuple, list)):
            visuals = np.stack([item[0] for item in items], axis=0).astype(np.float32)
            vectors = np.stack([item[1] for item in items], axis=0).astype(np.float32)
        else:
            split = [split_hybrid_item(item) for item in items]
            if split[0][1] is not None:
                visuals = np.stack([item[0] for item in split], axis=0).astype(np.float32)
                vectors = np.stack([item[1] for item in split], axis=0).astype(np.float32)
            else:
                visuals = np.stack(items, axis=0).astype(np.float32)
                vectors = zero_vectors(visuals.shape[0])
        return [visuals, vectors]

    def _predict_q_values(self, state) -> np.ndarray:
        with self.tf.device(self.device_info.inference_device):
            if self._vector_dim > 0:
                visual, vector = state
                q_values = self.online(
                    [
                        self.tf.convert_to_tensor(visual[None, ...], dtype=self.tf.float32),
                        self.tf.convert_to_tensor(vector[None, ...], dtype=self.tf.float32),
                    ],
                    training=False,
                )[0].numpy()
            else:
                q_values = self.online(
                    self.tf.convert_to_tensor(state[None, ...], dtype=self.tf.float32),
                    training=False,
                )[0].numpy()
        return q_values

    def apply_sim_pretrain_profile(self) -> None:
        training = self.config.training
        self._sim_pretrain_mode = True
        self.train_every = max(1, training.sim_pretrain_train_every)
        self.batch_size = max(self.batch_size, training.sim_pretrain_batch_size)
        self.config.training.min_replay_size = min(
            self.config.training.min_replay_size,
            training.sim_pretrain_min_replay,
        )
        self.epsilon = training.epsilon_start
        self._epsilon_anneal_start = training.epsilon_start
        self.metrics.epsilon = self.epsilon
        self.progress.epsilon = self.epsilon

    def _build_batch_predict(self):
        agent = self

        @self.tf.function(reduce_retracing=True)
        def batch_predict(states):
            return agent.online(states, training=False)

        return batch_predict

    def rule_prior_probability(self) -> float:
        training = self.config.training
        if training.rule_prior_steps <= 0:
            return 0.0
        progress = min(1.0, self.metrics.total_steps / training.rule_prior_steps)
        return max(
            training.rule_prior_end,
            training.rule_prior_start
            - (training.rule_prior_start - training.rule_prior_end) * progress,
        )

    def act(
        self,
        state: np.ndarray,
        training: bool = True,
        can_boost: bool = True,
        boost_level: float = 1.0,
        frame_state: FrameState | None = None,
    ) -> int:
        valid = self._valid_actions(can_boost, boost_level)
        if training and frame_state is not None:
            prior = self.rule_prior_probability()
            if prior > 0.0 and random.random() < prior:
                return self.rule_policy.act(frame_state)
        if training and random.random() < self.epsilon:
            return random.choice(valid)
        q_values = self._predict_q_values(state)
        masked = np.full(self.config.action_count, -np.inf, dtype=np.float32)
        for action in valid:
            masked[action] = q_values[action]
        return int(np.argmax(masked))

    def act_batch(
        self,
        states,
        *,
        training: bool = True,
        can_boost: np.ndarray | list[bool] | None = None,
        boost_levels: np.ndarray | list[float] | None = None,
        frame_states: list[FrameState | None] | None = None,
    ) -> np.ndarray:
        if isinstance(states, list):
            batch_size = len(states)
        else:
            batch_size = len(states)
        if can_boost is None:
            can_boost = np.ones(batch_size, dtype=bool)
        if boost_levels is None:
            boost_levels = np.ones(batch_size, dtype=np.float32)

        actions = np.zeros(batch_size, dtype=np.int32)
        decided = np.zeros(batch_size, dtype=bool)

        # Phase 3: rule prior in vectorized pretrain.
        if training and frame_states is not None:
            prior = self.rule_prior_probability()
            if prior > 0.0:
                for index in range(batch_size):
                    fs = frame_states[index]
                    if fs is not None and random.random() < prior:
                        actions[index] = self.rule_policy.act(fs)
                        decided[index] = True

        explore_mask = np.zeros(batch_size, dtype=bool)
        if training and self.epsilon > 0.0:
            explore_mask = (~decided) & (np.random.random(batch_size) < self.epsilon)

        greedy_indices = np.flatnonzero((~decided) & (~explore_mask))
        if len(greedy_indices) > 0:
            if isinstance(states, list):
                batch_states = [states[int(i)] for i in greedy_indices]
            else:
                batch_states = states[greedy_indices]
            batch_input = self._to_model_batch(batch_states)
            with self.tf.device(self.device_info.inference_device):
                if self._vector_dim > 0:
                    q_values = self.online(
                        [
                            self.tf.convert_to_tensor(batch_input[0], dtype=self.tf.float32),
                            self.tf.convert_to_tensor(batch_input[1], dtype=self.tf.float32),
                        ],
                        training=False,
                    ).numpy()
                else:
                    q_values = self._batch_predict(
                        self.tf.convert_to_tensor(batch_input, dtype=self.tf.float32)
                    ).numpy()
            for offset, index in enumerate(greedy_indices):
                valid = self._valid_actions(bool(can_boost[index]), float(boost_levels[index]))
                masked = np.full(self.config.action_count, -np.inf, dtype=np.float32)
                for action in valid:
                    masked[action] = q_values[offset, action]
                actions[index] = int(np.argmax(masked))

        bias = self.config.training.smart_explore_boost_bias
        for index in np.flatnonzero(explore_mask):
            fs = frame_states[index] if frame_states is not None else None
            valid = self._valid_actions(bool(can_boost[index]), float(boost_levels[index]))
            if (
                fs is not None
                and bias > 0
                and bool(can_boost[index])
                and (fs.boost_useful or (fs.orb_vy is not None and fs.orb_vy < -0.2))
                and random.random() < bias
            ):
                jump_valid = [a for a in valid if a >= 3]
                if jump_valid:
                    actions[index] = random.choice(jump_valid)
                    continue
            actions[index] = random.choice(valid)
        return actions

    def anneal_epsilon_by_steps(self) -> float:
        """Phase 3: step-based ε schedule for sim pretrain."""
        if self.config.training.watch_mode:
            self.epsilon = 0.0
            self.metrics.epsilon = 0.0
            self.progress.epsilon = 0.0
            return self.epsilon
        if not self._sim_pretrain_mode:
            return self.epsilon
        training = self.config.training
        anneal = max(1, training.sim_epsilon_anneal_steps)
        progress = min(1.0, self.metrics.total_steps / anneal)
        start = self._epsilon_anneal_start
        end = training.sim_epsilon_end
        self.epsilon = start - (start - end) * progress
        self.metrics.epsilon = self.epsilon
        self.progress.epsilon = self.epsilon
        return self.epsilon

    @staticmethod
    def _valid_actions(can_boost: bool, boost_level: float = 1.0) -> list[int]:
        if can_boost and boost_level * 100.0 >= 14.0:
            return list(range(6))
        return [0, 1, 2]

    def absorb_demonstrations(self, transitions, multiplier: int = 1) -> int:
        added = 0
        for _ in range(max(1, multiplier)):
            for transition in transitions:
                self.demo_replay.add(
                    transition.state,
                    transition.action,
                    transition.reward,
                    transition.next_state,
                    transition.done,
                    discount=self.config.training.gamma,
                )
                added += 1
        self.metrics.replay_size = len(self.replay)
        return added

    def absorb_demonstration_arrays(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        *,
        multiplier: int = 1,
        indices: np.ndarray | None = None,
        target_buffer: ReplayBuffer | None = None,
    ) -> int:
        expected_channels = self.config.observation.channel_count
        if (
            states.ndim != 4
            or states.shape[-1] != expected_channels
            or next_states.shape[-1] != expected_channels
        ):
            return 0
        buffer = target_buffer or self.demo_replay
        if indices is None:
            indices = np.arange(len(actions), dtype=np.int64)
            action_lookup = actions
            use_positional_actions = False
        else:
            indices = np.asarray(indices)
            # ingest may pass already-subsampled masked actions aligned to indices.
            use_positional_actions = len(actions) == len(indices)
            action_lookup = actions
        added = 0
        for _ in range(max(1, multiplier)):
            for offset, idx in enumerate(indices):
                action = (
                    int(action_lookup[offset])
                    if use_positional_actions
                    else int(action_lookup[int(idx)])
                )
                buffer.add(
                    states[int(idx)],
                    action,
                    float(rewards[int(idx)]),
                    next_states[int(idx)],
                    bool(dones[int(idx)]),
                    discount=self.config.training.gamma,
                )
                added += 1
        self.metrics.replay_size = len(self.replay)
        return added

    def pretrain_from_replay(self, steps: int | None = None) -> float | None:
        if len(self.demo_replay) == 0:
            return None
        # Hybrid policies need real vector features. Demo buffers are visual-only;
        # BC with zeroed vectors collapses Q-values (often to 100% noop) and must
        # not run — especially before Watch mode.
        if self._vector_dim > 0:
            print(
                "Skipping demo BC pretrain: hybrid model cannot use visual-only demos"
            )
            return None
        total_steps = steps or self.config.demo.pretrain_steps
        batch_size = min(self.batch_size, len(self.demo_replay))
        last_loss = None
        with self.tf.device(self.device_info.training_device):
            for _ in range(total_steps):
                batch = self.demo_replay.sample(batch_size)
                last_loss = float(
                    self._invoke_bc_train_step(
                        batch.states,
                        batch.actions,
                    ).numpy()
                )
        return last_loss

    def pretrain_from_demonstrations(self, transitions, steps: int | None = None) -> float | None:
        if not transitions:
            return None
        total_steps = steps or self.config.demo.pretrain_steps
        batch_size = min(self.batch_size, len(transitions))
        last_loss = None
        with self.tf.device(self.device_info.training_device):
            for _ in range(total_steps):
                indices = np.random.randint(0, len(transitions), batch_size)
                batch_states = [transitions[i].state for i in indices]
                batch_actions = np.asarray(
                    [transitions[i].action for i in indices],
                    dtype=np.int32,
                )
                last_loss = float(
                    self._invoke_bc_train_step(batch_states, batch_actions).numpy()
                )
        return last_loss

    def remember(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
        *,
        sim: bool = False,
    ) -> None:
        buffer = self.sim_replay if sim else self.replay
        self._push_n_step(
            self._n_step_queue,
            buffer,
            state,
            action,
            reward,
            next_state,
            done,
        )
        self.metrics.replay_size = len(self.replay)

    def _push_n_step(
        self,
        queue: list[tuple],
        buffer: ReplayBuffer,
        state,
        action: int,
        reward: float,
        next_state,
        done: bool,
    ) -> None:
        n_step = max(1, self.config.training.n_step)
        queue.append((state, action, reward, next_state, done))
        while len(queue) >= n_step:
            self._flush_n_step_queue(queue, buffer)
        if done:
            while queue:
                self._flush_n_step_queue(queue, buffer)
            queue.clear()

    def _flush_n_step_queue(self, queue: list[tuple], buffer: ReplayBuffer) -> None:
        if not queue:
            return
        gamma = self.config.training.gamma
        accumulated = 0.0
        steps_used = 0
        final_next = queue[-1][3]
        final_done = False
        for index, (_, _, step_reward, step_next, step_done) in enumerate(queue):
            accumulated += (gamma ** index) * step_reward
            final_next = step_next
            final_done = step_done
            steps_used = index + 1
            if step_done:
                break
        first_state, first_action, _, _, _ = queue[0]
        buffer.add(
            first_state,
            first_action,
            accumulated,
            final_next,
            final_done,
            discount=gamma ** steps_used,
        )
        queue.pop(0)

    def _flush_one_n_step(self, buffer: ReplayBuffer) -> None:
        """Backward-compatible alias for single-env n-step flush."""
        self._flush_n_step_queue(self._n_step_queue, buffer)

    def remember_batch(
        self,
        states,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states,
        dones: np.ndarray,
        *,
        sim: bool = False,
    ) -> None:
        buffer = self.sim_replay if sim else self.replay
        env_count = len(actions)
        if len(self._n_step_queues) != env_count:
            for queue in self._n_step_queues:
                while queue:
                    self._flush_n_step_queue(queue, buffer)
            self._n_step_queues = [[] for _ in range(env_count)]
        for index in range(env_count):
            state = states[index]
            next_state = next_states[index]
            self._push_n_step(
                self._n_step_queues[index],
                buffer,
                state,
                int(actions[index]),
                float(rewards[index]),
                next_state,
                bool(dones[index]),
            )
        self.metrics.replay_size = len(self.replay)

    def advance_steps(self, count: int = 1) -> AgentMetrics:
        if count <= 0:
            return self.metrics
        self.metrics.total_steps += count
        if self.config.training.watch_mode:
            return self.metrics
        if len(self.replay) < self.config.training.min_replay_size:
            return self.metrics
        if self.metrics.total_steps % self.train_every != 0:
            return self.metrics

        batch = self._sample_training_batch()
        start = time.perf_counter()
        with self.tf.device(self.device_info.training_device):
            loss = self._train_batch(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.metrics.loss = float(loss)
        self.metrics.train_ms = elapsed_ms

        if self.metrics.total_steps % self.config.training.target_sync_interval == 0:
            self._sync_target_network(hard=True)
        else:
            self._sync_target_network(hard=False)
        if self._sim_pretrain_mode:
            self.anneal_epsilon_by_steps()
        return self.metrics

    def maybe_train(self) -> AgentMetrics:
        return self.advance_steps(1)

    def _anneal_per_beta(self) -> float:
        training = self.config.training
        steps = max(1, training.per_beta_anneal_steps)
        progress = min(1.0, self.metrics.total_steps / steps)
        beta = training.per_beta_start + (
            training.per_beta_end - training.per_beta_start
        ) * progress
        self.replay.set_beta(beta)
        self.sim_replay.set_beta(beta)
        return beta

    def _sample_training_batch(self) -> TransitionBatch:
        self._anneal_per_beta()
        batch_size = self.batch_size
        sim_ratio = self.config.training.mixed_sim_replay_ratio
        sim_count = 0
        if len(self.sim_replay) > 0 and sim_ratio > 0:
            sim_count = min(int(batch_size * sim_ratio), len(self.sim_replay))
        rl_count = batch_size - sim_count
        rl_batch: TransitionBatch | None = None
        sim_batch: TransitionBatch | None = None
        if rl_count > 0 and len(self.replay) >= rl_count:
            rl_batch = self.replay.sample(rl_count)
        if sim_count > 0:
            sim_batch = self.sim_replay.sample(sim_count)
        if rl_batch is None and sim_batch is None:
            return self.replay.sample(min(batch_size, len(self.replay)))
        if rl_batch is not None and sim_batch is None:
            return rl_batch
        if sim_batch is not None and rl_batch is None:
            return sim_batch
        assert rl_batch is not None and sim_batch is not None
        weights = None
        if rl_batch.weights is not None or sim_batch.weights is not None:
            rl_w = (
                rl_batch.weights
                if rl_batch.weights is not None
                else np.ones(len(rl_batch.actions), dtype=np.float32)
            )
            sim_w = (
                sim_batch.weights
                if sim_batch.weights is not None
                else np.ones(len(sim_batch.actions), dtype=np.float32)
            )
            weights = np.concatenate([rl_w, sim_w], axis=0)
            weights = weights / max(float(weights.max()), 1e-6)
        return TransitionBatch(
            states=np.concatenate([rl_batch.states, sim_batch.states], axis=0),
            actions=np.concatenate([rl_batch.actions, sim_batch.actions], axis=0),
            rewards=np.concatenate([rl_batch.rewards, sim_batch.rewards], axis=0),
            next_states=np.concatenate(
                [rl_batch.next_states, sim_batch.next_states],
                axis=0,
            ),
            dones=np.concatenate([rl_batch.dones, sim_batch.dones], axis=0),
            discounts=np.concatenate(
                [rl_batch.discounts, sim_batch.discounts],
                axis=0,
            ),
            indices=rl_batch.indices,
            weights=weights,
            sim_indices=sim_batch.indices,
        )

    def end_episode(self, *, sim_pretrain: bool | None = None) -> None:
        sim_mode = (
            self._sim_pretrain_mode if sim_pretrain is None else sim_pretrain
        )
        if self.config.training.watch_mode:
            self.epsilon = 0.0
        elif sim_mode:
            # Step-based anneal is authoritative during sim pretrain.
            self.anneal_epsilon_by_steps()
            self.metrics.epsilon = self.epsilon
            self.progress.epsilon = self.epsilon
            return
        else:
            decay = self.config.training.epsilon_decay
            epsilon_end = self.config.training.epsilon_end
            self.epsilon = max(epsilon_end, self.epsilon * decay)
        if not sim_mode and not self.config.training.watch_mode:
            self._cap_browser_epsilon()
        self.metrics.epsilon = self.epsilon
        self.progress.epsilon = self.epsilon

    def _cap_browser_epsilon(self) -> None:
        cap = self.config.training.browser_epsilon_cap
        floor = self.config.training.browser_epsilon_floor
        self.epsilon = max(floor, min(cap, self.epsilon))

    def _browser_training_active(self) -> bool:
        return (
            not self._sim_pretrain_mode
            and not self.config.training.sim_mode
            and not self.config.training.watch_mode
        )

    def trim_replay_buffers(self) -> int:
        trim_size = self.config.training.replay_trim_size
        removed = self.replay.trim_to(trim_size)
        removed += self.demo_replay.trim_to(trim_size)
        removed += self.sim_replay.trim_to(trim_size)
        return removed

    def patch_epsilon_on_plateau(self) -> float:
        """Decay exploration on browser plateau instead of restarting it."""
        if not self._browser_training_active():
            return self.epsilon
        decay = self.config.training.browser_plateau_epsilon_decay
        floor = self.config.training.browser_epsilon_floor
        self.epsilon = max(floor, self.epsilon * decay)
        self._cap_browser_epsilon()
        self.metrics.epsilon = self.epsilon
        self.progress.epsilon = self.epsilon
        return self.epsilon

    def record_episode(
        self,
        reward: float,
        score: float,
        *,
        sim_pretrain: bool | None = None,
    ) -> None:
        sim_mode = (
            self._sim_pretrain_mode if sim_pretrain is None else sim_pretrain
        )
        improved_score = score > self.progress.best_score
        self.progress.episodes_completed += 1
        self.progress.recent_rewards.append(reward)
        self.progress.recent_scores.append(score)
        if len(self.progress.recent_rewards) > 20:
            self.progress.recent_rewards = self.progress.recent_rewards[-20:]
            self.progress.recent_scores = self.progress.recent_scores[-20:]

        if self.progress.baseline_reward is None:
            self._baseline_samples.append((reward, score))
            if len(self._baseline_samples) >= self.progress.baseline_episodes:
                rewards = [item[0] for item in self._baseline_samples]
                scores = [item[1] for item in self._baseline_samples]
                self.progress.baseline_reward = float(sum(rewards) / len(rewards))
                self.progress.baseline_score = float(sum(scores) / len(scores))

        if reward > self.progress.best_reward:
            self.progress.best_reward = reward
        if improved_score:
            self.progress.best_score = score
            self._episodes_since_best = 0
        elif not sim_mode and not self.config.training.watch_mode:
            self._episodes_since_best += 1
            plateau = self.config.training.transfer_plateau_episodes
            if self._episodes_since_best >= plateau:
                if self._browser_training_active():
                    self.patch_epsilon_on_plateau()
                else:
                    restart = self.config.training.transfer_epsilon_restart
                    self.epsilon = max(self.epsilon, restart)
                    self.metrics.epsilon = self.epsilon
                    self.progress.epsilon = self.epsilon
                self._episodes_since_best = 0

    def maybe_autosave(self, *, force: bool = False) -> bool:
        steps = self.metrics.total_steps
        every_steps = max(1, self.config.training.autosave_every_steps)
        if not force and (steps - self._last_autosave_steps) < every_steps:
            return False
        self.save()
        self._last_autosave_steps = steps
        return True

    def prepare_transfer_from_sim(self) -> None:
        preserved = ReplayBuffer(self.config.training.replay_buffer_size)
        preserved.extend_from(self.sim_replay)
        self.replay.clear()
        self.demo_replay.clear()
        self.sim_replay.clear()
        self.sim_replay.extend_from(preserved, max_items=5000)
        self._n_step_queue.clear()
        self._n_step_queues = []
        self._sim_pretrain_mode = False
        self.set_learning_rate(self.config.training.transfer_learning_rate)
        self.epsilon = self.config.training.transfer_epsilon_start
        self._cap_browser_epsilon()
        self.metrics.epsilon = self.epsilon
        self.progress.epsilon = self.epsilon
        # Keep sim score as baseline reference; reset browser progress counters.
        sim_best = float(self.progress.best_score)
        if sim_best > 0:
            self.progress.baseline_score = sim_best
        self.progress.best_score = 0.0
        self.progress.best_reward = float("-inf")
        self.progress.recent_scores = []
        self.progress.recent_rewards = []
        self.progress.episodes_completed = 0
        self.metrics.total_steps = 0
        self._episodes_since_best = 0
        self._last_autosave_steps = 0
        self._epsilon_anneal_start = self.epsilon

    def seed_sim_replay_from_rendered(
        self,
        *,
        transitions: int | None = None,
        env_index: int = 42,
    ) -> int:
        """Fill sim_replay with rendered-sim (non-fast) rollouts for mixed transfer."""
        from ascent_player.env.sim_env import AscentSimEnv

        target = transitions or self.config.training.transfer_seed_sim_replay
        if target <= 0:
            return 0
        saved_eps = self.epsilon
        self.epsilon = 0.05
        env = AscentSimEnv(self.config, fast_mode=False, env_index=env_index)
        added = 0
        try:
            state = env.reset_sync()
            while added < target:
                frame_state = env._last_frame_state
                action = self.act(
                    state,
                    training=True,
                    can_boost=env.can_boost,
                    boost_level=env.boost_level,
                    frame_state=frame_state,
                )
                result = env.step_sync(int(action))
                self.remember(
                    state,
                    int(action),
                    float(result.reward),
                    result.state,
                    bool(result.done),
                    sim=True,
                )
                state = result.state
                added += 1
                if result.done:
                    state = env.reset_sync()
        finally:
            self.epsilon = saved_eps
            self.metrics.epsilon = saved_eps
            self.progress.epsilon = saved_eps
        return added

    def _promote_replay_for_mixed_transfer(self) -> None:
        if len(self.sim_replay) > 0:
            return
        added = self.seed_sim_replay_from_rendered()
        if added:
            print(f"Seeded sim_replay with {added} rendered-sim transitions")

    def resolve_best_checkpoint(self) -> Path | None:
        training = self.config.training
        # Browser-adapted weights beat sim-only playable for Watch / UI.
        if not training.sim_mode and checkpoint_exists(
            training.browser_best_checkpoint_path
        ):
            return training.browser_best_checkpoint_path
        for path in (
            training.playable_checkpoint_path,
            training.sim_best_eval_checkpoint_path,
        ):
            if checkpoint_exists(path):
                return path
        return prefer_checkpoint(
            training.sim_checkpoint_path,
            training.checkpoint_path,
        )

    def promote_browser_best(self) -> Path:
        """Persist current weights as the browser Watch baseline."""
        path = self.save(self.config.training.browser_best_checkpoint_path)
        self.save(self.config.training.playable_checkpoint_path)
        self.save(self.config.training.checkpoint_path)
        return path

    def promote_playable_checkpoint(self, source: Path | None = None) -> Path | None:
        """Copy the strongest (or given) weights into playable + UI checkpoint slots."""
        training = self.config.training
        source = source or self.resolve_best_checkpoint()
        if source is None or not self.load(source):
            return None
        playable = self.save(training.playable_checkpoint_path)
        self.save(training.checkpoint_path)
        return playable

    def try_autoload(self) -> LoadResult:
        target = self.config.training.checkpoint_path
        if self.config.training.transfer_from_sim:
            sim_path = prefer_checkpoint(
                self.config.training.sim_best_eval_checkpoint_path,
                self.config.training.sim_checkpoint_path,
                self.config.training.playable_checkpoint_path,
            ) or self.config.training.sim_checkpoint_path
            if checkpoint_exists(sim_path) and self.load(sim_path):
                self.prepare_transfer_from_sim()
                self._promote_replay_for_mixed_transfer()
                message = (
                    f"Loaded sim pretrain from {sim_path.name} — "
                    f"fine-tuning with ε={self.epsilon:.2f}, "
                    f"sim_replay={len(self.sim_replay)}"
                )
                return LoadResult(True, message, self.progress)
        if not self.config.training.auto_load_checkpoint:
            return LoadResult(False, "Auto-load disabled — starting from scratch.")

        preferred = target
        if self.config.training.overnight_prefer_latest and not self.config.training.sim_mode:
            # Overnight working weights: continue dqn_latest; elite is browser_best.
            if checkpoint_exists(target):
                preferred = target
            elif self.config.training.prefer_best_checkpoint:
                best = self.resolve_best_checkpoint()
                if best is not None:
                    preferred = best
        elif self.config.training.prefer_best_checkpoint:
            best = self.resolve_best_checkpoint()
            if best is not None:
                preferred = best

        if not checkpoint_exists(preferred):
            return LoadResult(False, "No checkpoint found — starting from scratch.")
        if not self.load(preferred):
            detail = self._last_load_error or "incompatible"
            return LoadResult(
                False,
                f"Checkpoint load failed ({detail}) — starting from scratch.",
            )
        # Keep UI default path aligned with the strongest weights we just loaded.
        # Overnight mode: do NOT overwrite dqn_latest with browser_best.
        if (
            preferred != target
            and not self.config.training.overnight_prefer_latest
        ):
            try:
                self.save(target)
                self.save(self.config.training.playable_checkpoint_path)
            except Exception as exc:
                print(f"Could not promote preferred checkpoint: {exc}")
        message = (
            f"Resumed from {preferred.name} — "
            f"{self.progress.episodes_completed} episodes, "
            f"{self.progress.total_steps:,} steps, ε={self.progress.epsilon:.3f}"
        )
        if self.progress.has_baseline:
            message += (
                f" | baseline reward {self.progress.baseline_reward:.1f}, "
                f"score {self.progress.baseline_score:.1f}"
            )
        message += f" | best score {self.progress.best_score:.0f}"
        return LoadResult(True, message, self.progress)

    @staticmethod
    def weights_sidecar_path(keras_path: Path) -> Path:
        return keras_path.with_name(f"{keras_path.stem}.weights.h5")

    def save(self, path: Path | None = None) -> Path:
        target = path or self.config.training.checkpoint_path
        target.parent.mkdir(parents=True, exist_ok=True)
        self.progress.total_steps = self.metrics.total_steps
        self.progress.epsilon = self.epsilon
        weights_path = self.weights_sidecar_path(target)
        with self.tf.device(self.device_info.training_device):
            self.online.save_weights(weights_path)
            try:
                self.online.save(target)
            except Exception as exc:
                print(f"Full-model save skipped ({exc}); weights saved to {weights_path.name}")
        # Meta always keyed off the .keras path so resume finds it next to weights.
        save_progress(target, self.progress)
        return target

    def save_sim_checkpoint(self) -> Path:
        return self.save(self.config.training.sim_checkpoint_path)

    def _apply_loaded_progress(self, target: Path) -> None:
        progress = load_progress(target)
        if progress is None:
            weights_meta = self.weights_sidecar_path(target)
            progress = load_progress(weights_meta)
        if progress is None:
            return
        if progress.best_score <= 0 and progress.recent_scores:
            progress.best_score = max(progress.recent_scores)
        if not self.config.training.sim_mode:
            progress, sanitize_notes = sanitize_browser_progress(
                progress,
                score_cap=self.config.training.score_sanity_cap,
                epsilon_cap=self.config.training.browser_epsilon_cap,
            )
            for note in sanitize_notes:
                print(f"Progress sanitize: {note}")
        elif progress.baseline_score is not None:
            progress.best_score = max(progress.best_score, progress.baseline_score)
        self.progress = progress
        self.epsilon = progress.epsilon
        if not self.config.training.sim_mode:
            self._cap_browser_epsilon()
            # Strong sim checkpoints should not re-open heavy exploration in the browser.
            if progress.best_score >= self.config.training.gate_b_score:
                play_eps = min(
                    self.epsilon,
                    self.config.training.browser_epsilon_cap_after_gate_a,
                    0.06,
                )
                if play_eps < self.epsilon:
                    print(
                        f"Progress sanitize: epsilon {self.epsilon:.3f}→{play_eps:.3f} "
                        f"(strong checkpoint)"
                    )
                self.epsilon = play_eps
        self.progress.epsilon = self.epsilon
        self.metrics.epsilon = self.epsilon
        self.metrics.total_steps = progress.total_steps
        self._last_autosave_steps = progress.total_steps
        self._baseline_samples = []

    def load(self, path: Path | None = None) -> bool:
        target = path or self.config.training.checkpoint_path
        weights_path = self.weights_sidecar_path(target)
        errors: list[str] = []

        with self.tf.device(self.device_info.training_device):
            if weights_path.exists():
                try:
                    self.online.load_weights(weights_path)
                    self.target.set_weights(self.online.get_weights())
                    self._apply_loaded_progress(target)
                    self._last_load_error = None
                    return True
                except Exception as exc:
                    errors.append(f"weights load: {exc}")

            if target.exists():
                try:
                    from ascent_player.agent.model import _advantage_center_layer

                    AdvantageCenter = _advantage_center_layer()
                    loaded = self.tf.keras.models.load_model(
                        target,
                        custom_objects={"AdvantageCenter": AdvantageCenter},
                        safe_mode=False,
                    )
                    if self._vector_dim > 0:
                        if len(loaded.inputs) != 2:
                            raise ValueError("expected hybrid visual+vector inputs")
                        visual_shape = tuple(loaded.inputs[0].shape[1:])
                        vector_shape = tuple(loaded.inputs[1].shape[1:])
                        if visual_shape != tuple(self.online.inputs[0].shape[1:]):
                            raise ValueError(
                                f"visual shape {visual_shape} != "
                                f"{tuple(self.online.inputs[0].shape[1:])}"
                            )
                        if vector_shape != (self._vector_dim,):
                            raise ValueError(
                                f"vector shape {vector_shape} != ({self._vector_dim},)"
                            )
                    elif tuple(loaded.input_shape[1:]) != tuple(self.online.input_shape[1:]):
                        raise ValueError("visual-only input shape mismatch")
                    self.online.set_weights(loaded.get_weights())
                    self.target.set_weights(loaded.get_weights())
                    # Migrate legacy checkpoints to weights sidecar for future loads.
                    try:
                        self.online.save_weights(weights_path)
                    except Exception:
                        pass
                    self._apply_loaded_progress(target)
                    self._last_load_error = None
                    return True
                except Exception as exc:
                    errors.append(f"model load: {exc}")

        self._last_load_error = "; ".join(errors) if errors else "checkpoint missing"
        print(f"Checkpoint load failed: {self._last_load_error}")
        return False

    def _train_batch(self, batch: TransitionBatch):
        states = self._to_model_batch(batch.states)
        next_states = self._to_model_batch(batch.next_states)
        actions = self.tf.convert_to_tensor(batch.actions, dtype=self.tf.int32)
        rewards = self.tf.convert_to_tensor(batch.rewards, dtype=self.tf.float32)
        dones = self.tf.convert_to_tensor(batch.dones, dtype=self.tf.float32)
        discounts = self.tf.convert_to_tensor(batch.discounts, dtype=self.tf.float32)
        weights_t = (
            self.tf.convert_to_tensor(batch.weights, dtype=self.tf.float32)
            if batch.weights is not None
            else None
        )
        if self._vector_dim > 0:
            loss, td_errors = self._train_step(
                self.tf.convert_to_tensor(states[0], dtype=self.tf.float32),
                self.tf.convert_to_tensor(states[1], dtype=self.tf.float32),
                actions,
                rewards,
                self.tf.convert_to_tensor(next_states[0], dtype=self.tf.float32),
                self.tf.convert_to_tensor(next_states[1], dtype=self.tf.float32),
                dones,
                discounts,
                weights_t,
            )
        else:
            dummy = self.tf.zeros((len(batch.actions), 1), dtype=self.tf.float32)
            loss, td_errors = self._train_step(
                self.tf.convert_to_tensor(states, dtype=self.tf.float32),
                dummy,
                actions,
                rewards,
                self.tf.convert_to_tensor(next_states, dtype=self.tf.float32),
                dummy,
                dones,
                discounts,
                weights_t,
            )
        self._last_td_errors = td_errors
        td_abs = np.abs(td_errors.numpy())
        if batch.indices is not None:
            rl_n = len(batch.indices)
            self.replay.update_priorities(batch.indices, td_abs[:rl_n])
            if batch.sim_indices is not None:
                self.sim_replay.update_priorities(batch.sim_indices, td_abs[rl_n:])
        elif batch.sim_indices is not None:
            self.sim_replay.update_priorities(batch.sim_indices, td_abs)
        if (
            len(self.demo_replay) > 0
            and self._vector_dim <= 0
            and (
                self.curriculum_stage not in {"M0", "M1", "M2"}
                or self.progress.best_score >= self.config.training.curriculum_stage_a_max
            )
            and self.metrics.total_steps % max(1, self.config.demo.hybrid_bc_every) == 0
        ):
            demo_batch = self.demo_replay.sample(
                min(self.batch_size, len(self.demo_replay))
            )
            bc_loss = self._invoke_bc_train_step(
                demo_batch.states,
                demo_batch.actions,
            )
            loss = loss + self.config.demo.bc_loss_weight * bc_loss
        return loss

    def _sync_target_network(self, *, hard: bool) -> None:
        if hard:
            self.target.set_weights(self.online.get_weights())
            return
        tau = self.config.training.soft_target_tau
        if tau <= 0:
            return
        online_weights = self.online.get_weights()
        target_weights = self.target.get_weights()
        blended = [
            tau * online + (1.0 - tau) * target
            for online, target in zip(online_weights, target_weights, strict=True)
        ]
        self.target.set_weights(blended)

    @property
    def device_message(self) -> str:
        return self.device_info.message

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = max(1, batch_size)

    def set_train_every(self, train_every: int) -> None:
        self.train_every = max(1, train_every)

    def set_learning_rate(self, learning_rate: float) -> None:
        self.config.training.learning_rate = learning_rate
        self.online.optimizer.learning_rate.assign(learning_rate)

    @property
    def _train_step(self):
        if not hasattr(self, "_compiled_train_step"):
            agent = self
            action_count = agent.config.action_count
            tf = agent.tf
            neg_inf = tf.constant(-1e9, dtype=tf.float32)

            @self.tf.function
            def train_step(
                state_visual,
                state_vector,
                actions,
                rewards,
                next_visual,
                next_vector,
                dones,
                discounts,
                weights,
            ):
                if agent._vector_dim > 0:
                    next_states_tensor = [next_visual, next_vector]
                    states_tensor = [state_visual, state_vector]
                    boost_levels = tf.reduce_mean(next_visual[..., -2], axis=[1, 2])
                else:
                    next_states_tensor = next_visual
                    states_tensor = state_visual
                    boost_levels = tf.reduce_mean(next_states_tensor[..., -2], axis=[1, 2])
                min_energy = agent.config.mechanics_reward.boost_min_energy
                can_boost = boost_levels * 100.0 >= min_energy
                action_idx = tf.range(action_count, dtype=tf.int32)
                jump_actions = action_idx >= 3
                allowed = tf.logical_or(
                    ~jump_actions,
                    tf.tile(can_boost[:, None], [1, action_count]),
                )
                mask = tf.cast(allowed, tf.float32)

                online_next_q = agent.online(next_states_tensor, training=False)
                target_next_q = agent.target(next_states_tensor, training=False)
                masked_online = tf.where(mask > 0.0, online_next_q, neg_inf)
                masked_target = tf.where(mask > 0.0, target_next_q, neg_inf)

                if agent.config.training.use_double_dqn:
                    best_actions = tf.argmax(masked_online, axis=1, output_type=tf.int32)
                    next_values = tf.reduce_sum(
                        tf.one_hot(best_actions, action_count) * target_next_q,
                        axis=1,
                    )
                else:
                    next_values = tf.reduce_max(masked_target, axis=1)

                targets = rewards + (1.0 - dones) * discounts * next_values

                with tf.GradientTape() as tape:
                    q_values = agent.online(states_tensor, training=True)
                    action_masks = tf.one_hot(actions, action_count)
                    selected_q = tf.reduce_sum(q_values * action_masks, axis=1)
                    td_errors = targets - selected_q
                    per_sample = tf.keras.losses.Huber(reduction="none")(targets, selected_q)
                    if weights is not None:
                        per_sample = per_sample * weights
                    loss = tf.reduce_mean(per_sample)

                agent._last_td_errors = td_errors

                gradients = tape.gradient(loss, agent.online.trainable_variables)
                clipped, _ = tf.clip_by_global_norm(
                    gradients,
                    agent.config.training.gradient_clip_norm,
                )
                gradient_pairs = [
                    (gradient, variable)
                    for gradient, variable in zip(
                        clipped,
                        agent.online.trainable_variables,
                        strict=True,
                    )
                    if gradient is not None
                ]
                agent.online.optimizer.apply_gradients(gradient_pairs)
                return loss, tf.abs(td_errors)

            self._compiled_train_step = train_step
        return self._compiled_train_step

    def _invoke_bc_train_step(self, states, actions):
        actions_t = self.tf.convert_to_tensor(actions, dtype=self.tf.int32)
        model_batch = self._to_model_batch(states)
        dummy = self.tf.zeros((self.tf.shape(actions_t)[0], 1), dtype=self.tf.float32)
        if self._vector_dim > 0:
            visual, vector = model_batch
            return self._bc_train_step(
                self.tf.convert_to_tensor(visual, dtype=self.tf.float32),
                self.tf.convert_to_tensor(vector, dtype=self.tf.float32),
                actions_t,
            )
        return self._bc_train_step(
            self.tf.convert_to_tensor(model_batch, dtype=self.tf.float32),
            dummy,
            actions_t,
        )

    @property
    def _bc_train_step(self):
        if not hasattr(self, "_compiled_bc_train_step"):
            agent = self
            tf = agent.tf

            @self.tf.function
            def bc_train_step(state_visual, state_vector, actions):
                if agent._vector_dim > 0:
                    model_in = [state_visual, state_vector]
                else:
                    model_in = state_visual
                with tf.GradientTape() as tape:
                    q_values = agent.online(model_in, training=True)
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    )(actions, q_values)
                gradients = tape.gradient(loss, agent.online.trainable_variables)
                clipped, _ = tf.clip_by_global_norm(
                    gradients,
                    agent.config.training.gradient_clip_norm,
                )
                gradient_pairs = [
                    (gradient, variable)
                    for gradient, variable in zip(
                        clipped,
                        agent.online.trainable_variables,
                        strict=True,
                    )
                    if gradient is not None
                ]
                agent.online.optimizer.apply_gradients(gradient_pairs)
                return loss

            self._compiled_bc_train_step = bc_train_step
        return self._compiled_bc_train_step

    def weight_norm(self) -> float:
        total = 0.0
        for weight in self.online.get_weights():
            total += float(np.sum(np.square(weight)))
        return float(np.sqrt(total))

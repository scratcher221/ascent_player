from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time

import numpy as np

from ascent_player.agent.checkpoint import (
    LoadResult,
    TrainingProgress,
    load_progress,
    save_progress,
)
from ascent_player.agent.model import build_q_network
from ascent_player.agent.replay_buffer import ReplayBuffer, TransitionBatch
from ascent_player.config import AppConfig
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
        self.replay = ReplayBuffer(config.training.replay_buffer_size)
        self.demo_replay = ReplayBuffer(config.training.replay_buffer_size)
        self.sim_replay = ReplayBuffer(config.training.replay_buffer_size)
        self.progress = TrainingProgress(
            baseline_episodes=config.training.baseline_episodes,
            epsilon=config.training.epsilon_start,
        )
        self._baseline_samples: list[tuple[float, float]] = []
        self._last_autosave_steps = 0
        self._episodes_since_best = 0
        self._sim_pretrain_mode = False
        input_shape = (
            config.observation.height,
            config.observation.width,
            config.observation.channel_count,
        )

        with self.tf.device(self.device_info.training_device):
            self.online = build_q_network(
                input_shape,
                config.action_count,
                config.training.learning_rate,
            )
            self.target = build_q_network(
                input_shape,
                config.action_count,
                config.training.learning_rate,
            )
            self.target.set_weights(self.online.get_weights())

        sample = np.zeros(input_shape, dtype=np.float32)
        self.device_info.inference_device = benchmark_inference_device(
            self.online,
            sample,
            self.device_info,
        )
        self._batch_predict = self._build_batch_predict()

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
        self.metrics.epsilon = self.epsilon
        self.progress.epsilon = self.epsilon

    def _build_batch_predict(self):
        agent = self

        @self.tf.function(reduce_retracing=True)
        def batch_predict(states):
            return agent.online(states, training=False)

        return batch_predict

    def act(
        self,
        state: np.ndarray,
        training: bool = True,
        can_boost: bool = True,
        boost_level: float = 1.0,
    ) -> int:
        valid = self._valid_actions(can_boost, boost_level)
        if training and random.random() < self.epsilon:
            return random.choice(valid)
        with self.tf.device(self.device_info.inference_device):
            q_values = self.online(
                self.tf.convert_to_tensor(state[None, ...], dtype=self.tf.float32),
                training=False,
            )[0].numpy()
        masked = np.full(self.config.action_count, -np.inf, dtype=np.float32)
        for action in valid:
            masked[action] = q_values[action]
        return int(np.argmax(masked))

    def act_batch(
        self,
        states: np.ndarray,
        *,
        training: bool = True,
        can_boost: np.ndarray | list[bool] | None = None,
        boost_levels: np.ndarray | list[float] | None = None,
    ) -> np.ndarray:
        batch_size = len(states)
        if can_boost is None:
            can_boost = np.ones(batch_size, dtype=bool)
        if boost_levels is None:
            boost_levels = np.ones(batch_size, dtype=np.float32)

        actions = np.zeros(batch_size, dtype=np.int32)
        explore_mask = np.zeros(batch_size, dtype=bool)
        if training and self.epsilon > 0.0:
            explore_mask = np.random.random(batch_size) < self.epsilon

        greedy_indices = np.flatnonzero(~explore_mask)
        if len(greedy_indices) > 0:
            with self.tf.device(self.device_info.inference_device):
                q_values = self._batch_predict(
                    self.tf.convert_to_tensor(states[greedy_indices], dtype=self.tf.float32)
                ).numpy()
            for offset, index in enumerate(greedy_indices):
                valid = self._valid_actions(bool(can_boost[index]), float(boost_levels[index]))
                masked = np.full(self.config.action_count, -np.inf, dtype=np.float32)
                for action in valid:
                    masked[action] = q_values[offset, action]
                actions[index] = int(np.argmax(masked))

        for index in np.flatnonzero(explore_mask):
            valid = self._valid_actions(bool(can_boost[index]), float(boost_levels[index]))
            actions[index] = random.choice(valid)
        return actions

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
        added = 0
        for _ in range(max(1, multiplier)):
            for idx in indices:
                buffer.add(
                    states[idx],
                    int(actions[idx]),
                    float(rewards[idx]),
                    next_states[idx],
                    bool(dones[idx]),
                )
                added += 1
        self.metrics.replay_size = len(self.replay)
        return added

    def pretrain_from_replay(self, steps: int | None = None) -> float | None:
        if len(self.demo_replay) == 0:
            return None
        total_steps = steps or self.config.demo.pretrain_steps
        batch_size = min(self.batch_size, len(self.demo_replay))
        last_loss = None
        with self.tf.device(self.device_info.training_device):
            for _ in range(total_steps):
                batch = self.demo_replay.sample(batch_size)
                last_loss = float(
                    self._bc_train_step(
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
                batch_states = np.asarray(
                    [transitions[i].state for i in indices],
                    dtype=np.float32,
                )
                batch_actions = np.asarray(
                    [transitions[i].action for i in indices],
                    dtype=np.int32,
                )
                last_loss = float(
                    self._bc_train_step(
                        batch_states,
                        batch_actions,
                    ).numpy()
                )
        return last_loss

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        *,
        sim: bool = False,
    ) -> None:
        buffer = self.sim_replay if sim else self.replay
        buffer.add(state, action, reward, next_state, done)
        self.metrics.replay_size = len(self.replay)

    def remember_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        *,
        sim: bool = False,
    ) -> None:
        buffer = self.sim_replay if sim else self.replay
        buffer.add_many(states, actions, rewards, next_states, dones)
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
        return self.metrics

    def maybe_train(self) -> AgentMetrics:
        return self.advance_steps(1)

    def _sample_training_batch(self) -> TransitionBatch:
        batch_size = self.batch_size
        sim_ratio = self.config.training.mixed_sim_replay_ratio
        sim_count = 0
        if len(self.sim_replay) > 0 and sim_ratio > 0:
            sim_count = min(int(batch_size * sim_ratio), len(self.sim_replay))
        rl_count = batch_size - sim_count
        parts: list[TransitionBatch] = []
        if rl_count > 0 and len(self.replay) >= rl_count:
            parts.append(self.replay.sample(rl_count))
        if sim_count > 0:
            parts.append(self.sim_replay.sample(sim_count))
        if not parts:
            return self.replay.sample(min(batch_size, len(self.replay)))
        if len(parts) == 1:
            return parts[0]
        return TransitionBatch(
            states=np.concatenate([part.states for part in parts], axis=0),
            actions=np.concatenate([part.actions for part in parts], axis=0),
            rewards=np.concatenate([part.rewards for part in parts], axis=0),
            next_states=np.concatenate([part.next_states for part in parts], axis=0),
            dones=np.concatenate([part.dones for part in parts], axis=0),
        )

    def end_episode(self, *, sim_pretrain: bool | None = None) -> None:
        sim_mode = (
            self._sim_pretrain_mode if sim_pretrain is None else sim_pretrain
        )
        if self.config.training.watch_mode:
            self.epsilon = 0.0
        else:
            decay = (
                self.config.training.sim_epsilon_decay
                if sim_mode
                else self.config.training.epsilon_decay
            )
            epsilon_end = (
                self.config.training.sim_epsilon_end
                if sim_mode
                else self.config.training.epsilon_end
            )
            self.epsilon = max(epsilon_end, self.epsilon * decay)
        self.metrics.epsilon = self.epsilon
        self.progress.epsilon = self.epsilon

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
        self.replay.clear()
        self.sim_replay.clear()
        self.demo_replay.clear()
        self._sim_pretrain_mode = False
        self.set_learning_rate(self.config.training.transfer_learning_rate)
        self.epsilon = self.config.training.transfer_epsilon_start
        self.metrics.epsilon = self.epsilon
        self.progress.epsilon = self.epsilon
        self.progress.best_score = 0.0
        self.progress.best_reward = float("-inf")
        self.progress.recent_scores = []
        self.progress.recent_rewards = []
        self.progress.episodes_completed = 0
        self.metrics.total_steps = 0
        self._episodes_since_best = 0
        self._last_autosave_steps = 0

    def _promote_replay_for_mixed_transfer(self) -> None:
        return

    def try_autoload(self) -> LoadResult:
        target = self.config.training.checkpoint_path
        if self.config.training.transfer_from_sim:
            sim_path = self.config.training.sim_checkpoint_path
            if sim_path.exists() and self.load(sim_path):
                self.prepare_transfer_from_sim()
                message = (
                    f"Loaded sim pretrain from {sim_path.name} — "
                    f"fine-tuning with ε={self.epsilon:.2f}"
                )
                return LoadResult(True, message, self.progress)
        if not self.config.training.auto_load_checkpoint:
            return LoadResult(False, "Auto-load disabled — starting from scratch.")
        if not target.exists():
            return LoadResult(False, "No checkpoint found — starting from scratch.")
        if not self.load():
            return LoadResult(
                False,
                "Checkpoint missing or incompatible — starting from scratch.",
            )
        message = (
            f"Resumed training — {self.progress.episodes_completed} episodes, "
            f"{self.progress.total_steps:,} steps, ε={self.progress.epsilon:.3f}"
        )
        if self.progress.has_baseline:
            message += (
                f" | baseline reward {self.progress.baseline_reward:.1f}, "
                f"score {self.progress.baseline_score:.1f}"
            )
        message += f" | best score {self.progress.best_score:.0f}"
        return LoadResult(True, message, self.progress)

    def save(self, path: Path | None = None) -> Path:
        target = path or self.config.training.checkpoint_path
        target.parent.mkdir(parents=True, exist_ok=True)
        self.progress.total_steps = self.metrics.total_steps
        self.progress.epsilon = self.epsilon
        self.online.save(target)
        save_progress(target, self.progress)
        return target

    def save_sim_checkpoint(self) -> Path:
        return self.save(self.config.training.sim_checkpoint_path)

    def load(self, path: Path | None = None) -> bool:
        target = path or self.config.training.checkpoint_path
        if not target.exists():
            return False
        try:
            with self.tf.device(self.device_info.training_device):
                loaded = self.tf.keras.models.load_model(target)
                if tuple(loaded.input_shape[1:]) != tuple(self.online.input_shape[1:]):
                    return False
                self.online.set_weights(loaded.get_weights())
                self.target.set_weights(loaded.get_weights())
        except Exception:
            return False

        progress = load_progress(target)
        if progress is not None:
            if progress.best_score <= 0 and progress.recent_scores:
                progress.best_score = max(progress.recent_scores)
            if progress.baseline_score is not None:
                progress.best_score = max(progress.best_score, progress.baseline_score)
            self.progress = progress
            self.epsilon = progress.epsilon
            self.metrics.epsilon = progress.epsilon
            self.metrics.total_steps = progress.total_steps
            self._last_autosave_steps = progress.total_steps
            self._baseline_samples = []
        return True

    def _train_batch(self, batch: TransitionBatch):
        states = self.tf.convert_to_tensor(batch.states, dtype=self.tf.float32)
        next_states = self.tf.convert_to_tensor(batch.next_states, dtype=self.tf.float32)
        actions = self.tf.convert_to_tensor(batch.actions, dtype=self.tf.int32)
        rewards = self.tf.convert_to_tensor(batch.rewards, dtype=self.tf.float32)
        dones = self.tf.convert_to_tensor(batch.dones, dtype=self.tf.float32)
        loss = self._train_step(states, actions, rewards, next_states, dones)
        if (
            len(self.demo_replay) > 0
            and self.progress.best_score >= 1000
            and self.metrics.total_steps % max(1, self.config.demo.hybrid_bc_every) == 0
        ):
            demo_batch = self.demo_replay.sample(
                min(self.batch_size, len(self.demo_replay))
            )
            bc_loss = self._bc_train_step(
                self.tf.convert_to_tensor(demo_batch.states, dtype=self.tf.float32),
                self.tf.convert_to_tensor(demo_batch.actions, dtype=self.tf.int32),
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
            def train_step(states, actions, rewards, next_states, dones):
                boost_levels = tf.reduce_mean(next_states[..., -2], axis=[1, 2])
                can_boost = boost_levels * 100.0 >= agent.config.reward.boost_min_energy
                action_idx = tf.range(action_count, dtype=tf.int32)
                jump_actions = action_idx >= 3
                allowed = tf.logical_or(
                    ~jump_actions,
                    tf.tile(can_boost[:, None], [1, action_count]),
                )
                mask = tf.cast(allowed, tf.float32)

                online_next_q = agent.online(next_states, training=False)
                target_next_q = agent.target(next_states, training=False)
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

                targets = rewards + (1.0 - dones) * agent.config.training.gamma * next_values

                with tf.GradientTape() as tape:
                    q_values = agent.online(states, training=True)
                    action_masks = tf.one_hot(actions, action_count)
                    selected_q = tf.reduce_sum(q_values * action_masks, axis=1)
                    loss = tf.keras.losses.Huber()(targets, selected_q)

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

            self._compiled_train_step = train_step
        return self._compiled_train_step

    @property
    def _bc_train_step(self):
        if not hasattr(self, "_compiled_bc_train_step"):
            agent = self
            tf = agent.tf

            @self.tf.function
            def bc_train_step(states, actions):
                with tf.GradientTape() as tape:
                    q_values = agent.online(states, training=True)
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

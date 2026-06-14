from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import time

import numpy as np

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

    def act(self, state: np.ndarray, training: bool = True, can_boost: bool = True) -> int:
        valid = self._valid_actions(can_boost)
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

    @staticmethod
    def _valid_actions(can_boost: bool) -> list[int]:
        if can_boost:
            return list(range(6))
        return [0, 1, 2]

    def absorb_demonstrations(self, transitions, multiplier: int = 1) -> int:
        added = 0
        for _ in range(max(1, multiplier)):
            for transition in transitions:
                self.replay.add(
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
    ) -> int:
        expected_channels = self.config.observation.channel_count
        if (
            states.ndim != 4
            or states.shape[-1] != expected_channels
            or next_states.shape[-1] != expected_channels
        ):
            return 0
        if indices is None:
            indices = np.arange(len(actions), dtype=np.int64)
        added = 0
        for _ in range(max(1, multiplier)):
            for idx in indices:
                self.replay.add(
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
        if len(self.replay) == 0:
            return None
        total_steps = steps or self.config.demo.pretrain_steps
        batch_size = min(self.batch_size, len(self.replay))
        last_loss = None
        with self.tf.device(self.device_info.training_device):
            for _ in range(total_steps):
                batch = self.replay.sample(batch_size)
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
    ) -> None:
        self.replay.add(state, action, reward, next_state, done)
        self.metrics.replay_size = len(self.replay)

    def maybe_train(self) -> AgentMetrics:
        self.metrics.total_steps += 1
        if self.config.training.watch_mode:
            return self.metrics
        if len(self.replay) < self.config.training.min_replay_size:
            return self.metrics
        if self.metrics.total_steps % self.train_every != 0:
            return self.metrics

        batch = self.replay.sample(self.batch_size)
        start = time.perf_counter()
        with self.tf.device(self.device_info.training_device):
            loss = self._train_batch(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.metrics.loss = float(loss)
        self.metrics.train_ms = elapsed_ms

        if self.metrics.total_steps % self.config.training.target_sync_interval == 0:
            self.target.set_weights(self.online.get_weights())
        return self.metrics

    def end_episode(self) -> None:
        if self.config.training.watch_mode:
            self.epsilon = 0.0
        else:
            self.epsilon = max(
                self.config.training.epsilon_end,
                self.epsilon * self.config.training.epsilon_decay,
            )
        self.metrics.epsilon = self.epsilon

    def save(self, path: Path | None = None) -> Path:
        target = path or self.config.training.checkpoint_path
        target.parent.mkdir(parents=True, exist_ok=True)
        self.online.save(target)
        return target

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
        return True

    def _train_batch(self, batch: TransitionBatch):
        states = self.tf.convert_to_tensor(batch.states, dtype=self.tf.float32)
        next_states = self.tf.convert_to_tensor(batch.next_states, dtype=self.tf.float32)
        actions = self.tf.convert_to_tensor(batch.actions, dtype=self.tf.int32)
        rewards = self.tf.convert_to_tensor(batch.rewards, dtype=self.tf.float32)
        dones = self.tf.convert_to_tensor(batch.dones, dtype=self.tf.float32)
        return self._train_step(states, actions, rewards, next_states, dones)

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

            @self.tf.function
            def train_step(states, actions, rewards, next_states, dones):
                next_q = self.target(next_states, training=False)
                max_next_q = self.tf.reduce_max(next_q, axis=1)
                targets = rewards + (1.0 - dones) * self.config.training.gamma * max_next_q

                with self.tf.GradientTape() as tape:
                    q_values = self.online(states, training=True)
                    action_masks = self.tf.one_hot(actions, self.config.action_count)
                    selected_q = self.tf.reduce_sum(q_values * action_masks, axis=1)
                    loss = self.tf.keras.losses.Huber()(targets, selected_q)

                gradients = tape.gradient(loss, self.online.trainable_variables)
                gradient_pairs = [
                    (gradient, variable)
                    for gradient, variable in zip(
                        gradients,
                        self.online.trainable_variables,
                        strict=True,
                    )
                    if gradient is not None
                ]
                self.online.optimizer.apply_gradients(gradient_pairs)
                return loss

            self._compiled_train_step = train_step
        return self._compiled_train_step

    @property
    def _bc_train_step(self):
        if not hasattr(self, "_compiled_bc_train_step"):

            @self.tf.function
            def bc_train_step(states, actions):
                q_values = self.online(states, training=True)
                action_masks = self.tf.one_hot(actions, self.config.action_count)
                selected_q = self.tf.reduce_sum(q_values * action_masks, axis=1)
                return self.tf.reduce_mean(self.tf.square(1.0 - selected_q))

            self._compiled_bc_train_step = bc_train_step
        return self._compiled_bc_train_step

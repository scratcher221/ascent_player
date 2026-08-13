from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ascent_player.agent.checkpoint import TrainingProgress
from ascent_player.agent.model import build_q_network
from ascent_player.agent.replay_buffer import ReplayBuffer
from ascent_player.agent.reason import (
    EXPLORE,
    index_to_reason,
    reason_count,
    reason_to_index,
)
from ascent_player.agent.teacher import RulePolicy, SeedThreadPolicy
from ascent_player.agent.skills import (
    FREE_PLAY,
    SkillControllers,
    SkillRouter,
    index_to_skill,
    skill_count,
    skill_to_index,
)
from ascent_player.agent.dqn_acting import ActingMixin
from ascent_player.agent.dqn_replay import ReplayControlMixin
from ascent_player.agent.dqn_io import CheckpointIOMixin
from ascent_player.agent.dqn_learn import LearningMixin
from ascent_player.config import AppConfig
from ascent_player.utils.device import (
    DeviceInfo,
    benchmark_inference_device,
    enable_mixed_precision,
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


class DQNAgent(ActingMixin, ReplayControlMixin, CheckpointIOMixin, LearningMixin):
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
        self.seed_thread = SeedThreadPolicy(
            thread_corridor=float(
                getattr(config.training, "seed_thread_corridor", 0.18) or 0.18
            ),
        )
        self._n_step_queue: list[tuple] = []
        self._thread_log_every = 25
        self._thread_acts = 0
        # Priors anneal from this origin so continue runs aren't stuck at end
        # just because checkpoint total_steps is already huge.
        self._prior_anneal_origin = 0
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
        self._reason_count = reason_count()
        self._skill_count = (
            skill_count() if getattr(config.training, "skills_enabled", False) else 0
        )
        self.skill_router = SkillRouter(SkillControllers(self.rule_policy))
        self.last_action_source = "greedy"
        self.last_reason = EXPLORE
        self.last_reason_pred = EXPLORE
        self.last_reason_index = reason_to_index(EXPLORE)
        self.last_skill = FREE_PLAY
        self.last_skill_index = skill_to_index(FREE_PLAY)
        self._last_skill_logits: np.ndarray | None = None
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
        self._model_variant = str(
            getattr(config.training, "model_variant", "impala_mid") or "impala_mid"
        ).strip().lower()
        self._use_mixed_precision = bool(
            getattr(config.training, "mixed_precision", True)
        ) and self.device_info.training_device.startswith("/GPU")
        if self._use_mixed_precision:
            self._use_mixed_precision = enable_mixed_precision(True)
        else:
            enable_mixed_precision(False)

        with self.tf.device(self.device_info.training_device):
            self.online = build_q_network(
                input_shape,
                config.action_count,
                config.training.learning_rate,
                vector_dim=self._vector_dim,
                dueling=config.training.dueling_dqn,
                reason_count=self._reason_count,
                skill_count=self._skill_count,
                model_variant=self._model_variant,
                use_mixed_precision=self._use_mixed_precision,
            )
            self.target = build_q_network(
                input_shape,
                config.action_count,
                config.training.learning_rate,
                vector_dim=self._vector_dim,
                dueling=config.training.dueling_dqn,
                reason_count=self._reason_count,
                skill_count=self._skill_count,
                model_variant=self._model_variant,
                use_mixed_precision=self._use_mixed_precision,
            )
            self.target.set_weights(self.online.get_weights())
        print(
            f"MODEL_BUILD variant={self._model_variant} "
            f"params={self.online.count_params():,} "
            f"mixed_precision={int(self._use_mixed_precision)} "
            f"batch={self.batch_size}",
            flush=True,
        )

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

    def _unpack_outputs(self, outputs):
        """Split multi-output model into (q_values, reason_logits|None, skill_logits|None)."""
        if isinstance(outputs, (list, tuple)):
            q_values = outputs[0]
            reason_logits = outputs[1] if len(outputs) > 1 else None
            skill_logits = outputs[2] if len(outputs) > 2 else None
            return q_values, reason_logits, skill_logits
        return outputs, None, None

    def _predict_q_values(self, state) -> np.ndarray:
        with self.tf.device(self.device_info.inference_device):
            if self._vector_dim > 0:
                visual, vector = state
                outputs = self.online(
                    [
                        self.tf.convert_to_tensor(visual[None, ...], dtype=self.tf.float32),
                        self.tf.convert_to_tensor(vector[None, ...], dtype=self.tf.float32),
                    ],
                    training=False,
                )
            else:
                outputs = self.online(
                    self.tf.convert_to_tensor(state[None, ...], dtype=self.tf.float32),
                    training=False,
                )
            q_values, reason_logits, skill_logits = self._unpack_outputs(outputs)
            q_np = q_values[0].numpy() if hasattr(q_values, "numpy") else np.asarray(q_values)[0]
            if reason_logits is not None:
                logits = reason_logits[0].numpy() if hasattr(reason_logits, "numpy") else np.asarray(reason_logits)[0]
                self.last_reason_pred = index_to_reason(int(np.argmax(logits)))
            if skill_logits is not None:
                slogits = (
                    skill_logits[0].numpy()
                    if hasattr(skill_logits, "numpy")
                    else np.asarray(skill_logits)[0]
                )
                self._last_skill_logits = slogits
                self.last_skill = index_to_skill(int(np.argmax(slogits)))
            return q_np

    def _build_batch_predict(self):
        agent = self

        @self.tf.function(reduce_retracing=True)
        def batch_predict(states):
            outputs = agent.online(states, training=False)
            q_values, _, _ = agent._unpack_outputs(outputs)
            return q_values

        return batch_predict

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


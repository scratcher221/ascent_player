from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
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
from ascent_player.agent.reason import (
    EXPLORE,
    assign_reason,
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
from ascent_player.config import AppConfig
from ascent_player.env.state_detector import FrameState
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

    def rule_prior_probability(self) -> float:
        training = self.config.training
        if training.rule_prior_steps <= 0:
            return 0.0
        origin = int(getattr(self, "_prior_anneal_origin", 0) or 0)
        local = max(0, int(self.metrics.total_steps) - origin)
        progress = min(1.0, local / training.rule_prior_steps)
        return max(
            training.rule_prior_end,
            training.rule_prior_start
            - (training.rule_prior_start - training.rule_prior_end) * progress,
        )

    def seed_thread_prior_probability(self) -> float:
        training = self.config.training
        if not getattr(training, "seed_thread_enabled", False):
            return 0.0
        steps = int(getattr(training, "seed_thread_prior_steps", 0) or 0)
        start = float(getattr(training, "seed_thread_prior_start", 0.0) or 0.0)
        end = float(getattr(training, "seed_thread_prior_end", 0.0) or 0.0)
        if steps <= 0:
            return max(0.0, end)
        origin = int(getattr(self, "_prior_anneal_origin", 0) or 0)
        local = max(0, int(self.metrics.total_steps) - origin)
        progress = min(1.0, local / steps)
        return max(end, start - (start - end) * progress)

    def skill_teacher_prior_probability(self) -> float:
        training = self.config.training
        if not getattr(training, "skills_enabled", False):
            return 0.0
        steps = int(getattr(training, "skill_teacher_prior_steps", 0) or 0)
        start = float(getattr(training, "skill_teacher_prior_start", 0.0) or 0.0)
        end = float(getattr(training, "skill_teacher_prior_end", 0.0) or 0.0)
        if steps <= 0:
            return max(0.0, end)
        origin = int(getattr(self, "_prior_anneal_origin", 0) or 0)
        local = max(0, int(self.metrics.total_steps) - origin)
        progress = min(1.0, local / steps)
        return max(end, start - (start - end) * progress)

    def _label_skill_from_router(self, frame_state: FrameState | None) -> None:
        if frame_state is None:
            return
        proposed = self.skill_router.propose(frame_state)
        self.last_skill = proposed
        self.last_skill_index = skill_to_index(proposed)

    def _skill_network_confident(self) -> bool:
        logits = self._last_skill_logits
        if logits is None or logits.size < 2:
            return False
        order = np.argsort(logits)[::-1]
        best = float(logits[order[0]])
        second = float(logits[order[1]])
        margin = float(
            getattr(self.config.training, "skill_confidence_margin", 0.12) or 0.12
        )
        return (best - second) >= margin

    def reset_prior_anneal_origin(self) -> None:
        """Call after loading a continue checkpoint so priors start strong again."""
        self._prior_anneal_origin = int(self.metrics.total_steps)

    def act(
        self,
        state: np.ndarray,
        training: bool = True,
        can_boost: bool = True,
        boost_level: float = 1.0,
        frame_state: FrameState | None = None,
    ) -> int:
        valid = self._valid_actions(can_boost, boost_level)
        source = "greedy"
        prior = 0.0
        if frame_state is not None:
            skills_on = bool(getattr(self.config.training, "skills_enabled", False))
            if skills_on:
                self._label_skill_from_router(frame_state)

            safety = bool(
                getattr(self.config.training, "watch_safety_override", False)
            )
            if (
                (not training)
                and safety
                and (
                    frame_state.miss_risk
                    or (
                        frame_state.falling
                        and (frame_state.nearest_platform_dy or 0.0) > 0.18
                        and frame_state.can_boost
                        and frame_state.boost_level >= 0.12
                    )
                )
            ):
                action = self.rule_policy.act(frame_state)
                source = "rule"
                self._set_action_meta(action, frame_state, source)
                return action

            # Always observe landings so the thread lock tracks the episode.
            if getattr(self.config.training, "seed_thread_enabled", False):
                self.seed_thread.observe(frame_state)

            if skills_on:
                skill_teacher = self.skill_teacher_prior_probability() if training else 0.0
                if skill_teacher > 0.0 and random.random() < skill_teacher:
                    action, skill = self.skill_router.act(frame_state)
                    if skill != FREE_PLAY:
                        source = "skill"
                        self.last_skill = skill
                        self.last_skill_index = skill_to_index(skill)
                        self._set_action_meta(action, frame_state, source)
                        return action

            if training:
                prior = self.rule_prior_probability()
                thread_prior = self.seed_thread_prior_probability()
            else:
                prior = float(
                    getattr(self.config.training, "watch_rule_prior", 0.0) or 0.0
                )
                thread_prior = float(
                    getattr(self.config.training, "seed_thread_watch_prior", 0.0) or 0.0
                )
                if skills_on and self.skill_exec_allowed():
                    self._predict_q_values(state)
                    pred_skill = self.last_skill
                    if (
                        pred_skill != FREE_PLAY
                        and self._skill_network_confident()
                        and self.skill_router.controllers.applicable(
                            pred_skill, frame_state
                        )
                    ):
                        action = self.skill_router.controllers.act(
                            pred_skill, frame_state
                        )
                        source = "skill"
                        self._set_action_meta(action, frame_state, source)
                        return action

            # Seed-thread first: strong path bias when a landing/climb target exists.
            if thread_prior > 0.0:
                only_landing = bool(
                    getattr(self.config.training, "seed_thread_only_when_landing", True)
                )
                available = self.seed_thread.landing_available(frame_state)
                locked = self.seed_thread.thread_x is not None
                if (available or locked) and (
                    (not only_landing) or available or locked
                ):
                    if random.random() < thread_prior:
                        action = self.seed_thread.act(frame_state)
                        source = "thread"
                        self._thread_acts += 1
                        if self._thread_acts % self._thread_log_every == 1:
                            print(
                                f"SEED_THREAD act={action} "
                                f"lock_x={self.seed_thread.thread_x} "
                                f"lands={self.seed_thread.landings_locked} "
                                f"prior={thread_prior:.3f}",
                                flush=True,
                            )
                        self._set_action_meta(action, frame_state, source)
                        return action

            if prior > 0.0 and random.random() < prior:
                action = self.rule_policy.act(frame_state)
                source = "rule"
                self._set_action_meta(action, frame_state, source)
                return action
        if training and random.random() < self.epsilon:
            action = random.choice(valid)
            source = "explore"
            self._set_action_meta(action, frame_state, source)
            return action
        q_values = self._predict_q_values(state)
        masked = np.full(self.config.action_count, -np.inf, dtype=np.float32)
        for action in valid:
            masked[action] = q_values[action]
        action = int(np.argmax(masked))
        self._set_action_meta(action, frame_state, source)
        return action

    def _set_action_meta(
        self,
        action: int,
        frame_state: FrameState | None,
        source: str,
    ) -> None:
        self.last_action_source = source
        teacher = assign_reason(frame_state, action, source=source)
        self.last_reason = teacher
        self.last_reason_index = reason_to_index(teacher)
        if source != "greedy":
            # Still refresh predicted reason for logging when not greedily acting.
            try:
                # last_reason_pred may be stale; leave previous greedy pred if any
                pass
            except Exception:
                self.last_reason_pred = EXPLORE
        if source in ("rule", "explore", "thread", "skill"):
            # Predicted head not queried; mark pred as teacher for agree stats on explore
            # only when we actually ran the network (greedy). For explore/rule, pred=explore/rule.
            if source == "explore":
                self.last_reason_pred = EXPLORE
            else:
                self.last_reason_pred = teacher

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

        # Phase 3: seed-thread + rule prior in vectorized pretrain.
        if training and frame_states is not None:
            thread_prior = self.seed_thread_prior_probability()
            prior = self.rule_prior_probability()
            only_landing = bool(
                getattr(self.config.training, "seed_thread_only_when_landing", True)
            )
            for index in range(batch_size):
                fs = frame_states[index]
                if fs is None:
                    continue
                if thread_prior > 0.0:
                    self.seed_thread.observe(fs)
                    available = self.seed_thread.landing_available(fs)
                    locked = self.seed_thread.thread_x is not None
                    if (
                        (available or locked)
                        and ((not only_landing) or available or locked)
                        and random.random() < thread_prior
                    ):
                        actions[index] = self.seed_thread.act(fs)
                        decided[index] = True
                        continue
                if prior > 0.0 and random.random() < prior:
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
                    outputs = self.online(
                        [
                            self.tf.convert_to_tensor(batch_input[0], dtype=self.tf.float32),
                            self.tf.convert_to_tensor(batch_input[1], dtype=self.tf.float32),
                        ],
                        training=False,
                    )
                    q_values, _, _ = self._unpack_outputs(outputs)
                    q_values = q_values.numpy()
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

    def _demo_replay_has_hybrid_vectors(self) -> bool:
        """True when demo_replay stores (visual, vector) states matching vector_dim."""
        if self._vector_dim <= 0 or len(self.demo_replay) == 0:
            return self._vector_dim <= 0 and len(self.demo_replay) > 0
        with self.demo_replay._lock:
            probe = self.demo_replay._items[0]
        state = probe[0] if isinstance(probe, (tuple, list)) else None
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            return False
        vector = np.asarray(state[1], dtype=np.float32).reshape(-1)
        return int(vector.shape[0]) == int(self._vector_dim)

    def pretrain_from_replay(self, steps: int | None = None) -> float | None:
        if len(self.demo_replay) == 0:
            return None
        # Hybrid policies need real vector features. Visual-only demos would be
        # zero-padded and collapse Q-values; stamped hybrid demos are OK.
        if self._vector_dim > 0 and not self._demo_replay_has_hybrid_vectors():
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
        reason: int | None = None,
        skill: int | None = None,
        episode_score: float = -1.0,
        episode_id: int = -1,
    ) -> None:
        buffer = self.sim_replay if sim else self.replay
        reason_idx = self.last_reason_index if reason is None else int(reason)
        skill_idx = self.last_skill_index if skill is None else int(skill)
        self._push_n_step(
            self._n_step_queue,
            buffer,
            state,
            action,
            reward,
            next_state,
            done,
            reason=reason_idx,
            skill=skill_idx,
            episode_score=float(episode_score),
            episode_id=int(episode_id),
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
        *,
        reason: int = -1,
        skill: int = -1,
        episode_score: float = -1.0,
        episode_id: int = -1,
    ) -> None:
        n_step = max(1, self.config.training.n_step)
        queue.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
                reason,
                skill,
                float(episode_score),
                int(episode_id),
            )
        )
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
        for index, (_, _, step_reward, step_next, step_done, _, _, _, _) in enumerate(
            queue
        ):
            accumulated += (gamma ** index) * step_reward
            final_next = step_next
            final_done = step_done
            steps_used = index + 1
            if step_done:
                break
        (
            first_state,
            first_action,
            _,
            _,
            _,
            first_reason,
            first_skill,
            first_ep_score,
            first_ep_id,
        ) = queue[0]
        # Prefer terminal transition's episode score when available.
        ep_score = float(queue[-1][7]) if len(queue[-1]) > 7 else float(first_ep_score)
        ep_id = int(queue[-1][8]) if len(queue[-1]) > 8 else int(first_ep_id)
        if ep_score < 0:
            ep_score = float(first_ep_score)
        if ep_id < 0:
            ep_id = int(first_ep_id)
        buffer.add(
            first_state,
            first_action,
            accumulated,
            final_next,
            final_done,
            discount=gamma ** steps_used,
            reason=int(first_reason),
            skill=int(first_skill),
            episode_score=ep_score,
            episode_id=ep_id,
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
        if getattr(self, "seed_thread", None) is not None:
            self.seed_thread.reset()
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
            # Strong checkpoints: keep exploration modest, but respect floor/cap
            # (hardcoding 0.06 fought fine-tune floors and stalled map learning).
            if progress.best_score >= self.config.training.gate_b_score:
                play_eps = min(
                    self.epsilon,
                    self.config.training.browser_epsilon_cap_after_gate_a,
                )
                play_eps = max(
                    self.config.training.browser_epsilon_floor,
                    min(play_eps, self.config.training.browser_epsilon_cap),
                )
                if abs(play_eps - self.epsilon) > 1e-6:
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
        # Continue / seed-thread sessions: anneal priors from load time, not lifetime steps.
        if getattr(self.config.training, "seed_thread_enabled", False) or (
            float(getattr(self.config.training, "rule_prior_start", 0.0) or 0.0) > 0
            and not self.config.training.sim_mode
        ):
            self.reset_prior_anneal_origin()

    def _merge_weight_arrays(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
        """Copy overlapping slices when shapes differ (channel/vector migrate)."""
        if tuple(src.shape) == tuple(dst.shape):
            return src
        if src.ndim != dst.ndim:
            return None
        merged = np.array(dst, copy=True, dtype=np.float32)
        src = np.asarray(src, dtype=np.float32)
        if src.ndim == 1:
            n = min(src.shape[0], dst.shape[0])
            merged[:n] = src[:n]
            return merged
        if src.ndim == 2:
            r = min(src.shape[0], dst.shape[0])
            c = min(src.shape[1], dst.shape[1])
            merged[:r, :c] = src[:r, :c]
            return merged
        if src.ndim == 4:
            # Conv2D kernel: H, W, in_channels, out_channels
            h = min(src.shape[0], dst.shape[0])
            w = min(src.shape[1], dst.shape[1])
            cin = min(src.shape[2], dst.shape[2])
            cout = min(src.shape[3], dst.shape[3])
            merged[:h, :w, :cin, :cout] = src[:h, :w, :cin, :cout]
            return merged
        return None

    def _try_set_layer_weights(self, dest, src_weights) -> bool:
        dst_weights = dest.get_weights()
        if not src_weights or not dst_weights or len(src_weights) != len(dst_weights):
            return False
        merged: list[np.ndarray] = []
        for src, dst in zip(src_weights, dst_weights, strict=True):
            piece = self._merge_weight_arrays(np.asarray(src), np.asarray(dst))
            if piece is None:
                return False
            merged.append(piece)
        try:
            dest.set_weights(merged)
            return True
        except Exception:
            return False

    def _copy_compatible_weights(self, loaded) -> int:
        """Copy layers by name, then by Conv2D/Dense order (legacy auto-names)."""
        copied = 0
        online_by_name = {layer.name: layer for layer in self.online.layers}
        used_dest: set[int] = set()

        for layer in loaded.layers:
            dest = online_by_name.get(layer.name)
            if dest is None:
                continue
            src_w = layer.get_weights()
            if not src_w:
                continue
            if self._try_set_layer_weights(dest, src_w):
                copied += 1
                used_dest.add(id(dest))

        def _ordered(model, cls_name: str) -> list:
            return [
                layer
                for layer in model.layers
                if layer.__class__.__name__ == cls_name and layer.get_weights()
            ]

        for cls_name in ("Conv2D", "Dense"):
            src_layers = _ordered(loaded, cls_name)
            dst_layers = _ordered(self.online, cls_name)
            for src, dest in zip(src_layers, dst_layers):
                if id(dest) in used_dest:
                    continue
                if self._try_set_layer_weights(dest, src.get_weights()):
                    copied += 1
                    used_dest.add(id(dest))

        self.target.set_weights(self.online.get_weights())
        return copied

    def load(self, path: Path | None = None) -> bool:
        target = Path(path) if path is not None else self.config.training.checkpoint_path
        weights_path = self.weights_sidecar_path(target)
        errors: list[str] = []

        with self.tf.device(self.device_info.training_device):
            # Prefer the full .keras model so architecture migrations (vector dim /
            # reason head) always re-run against the real checkpoint, not a stale
            # same-arch weights sidecar written after a partial migrate.
            if target.exists():
                try:
                    from ascent_player.agent.model import _advantage_center_layer

                    AdvantageCenter = _advantage_center_layer()
                    loaded = self.tf.keras.models.load_model(
                        target,
                        custom_objects={"AdvantageCenter": AdvantageCenter},
                        safe_mode=False,
                    )
                    migrated = False
                    try:
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
                            if len(loaded.outputs) < 2 and self._reason_count > 0:
                                raise ValueError("missing reason head")
                            if self._skill_count > 0 and len(loaded.outputs) < 3:
                                raise ValueError("missing skill head")
                        self.online.set_weights(loaded.get_weights())
                        self.target.set_weights(loaded.get_weights())
                    except Exception as shape_exc:
                        copied = self._copy_compatible_weights(loaded)
                        migrated = True
                        print(
                            f"VECTOR_DIM_MIGRATE / REASON_HEAD_INIT: "
                            f"copied {copied} layers from {target.name} ({shape_exc})",
                            flush=True,
                        )
                    try:
                        self.online.save_weights(weights_path)
                    except Exception:
                        pass
                    self._apply_loaded_progress(target)
                    self._last_load_error = None
                    if migrated:
                        print(
                            "REASON_HEAD_INIT: reason/skill logits randomly initialized",
                            flush=True,
                        )
                    return True
                except Exception as exc:
                    errors.append(f"model load: {exc}")

            if weights_path.exists():
                try:
                    self.online.load_weights(weights_path)
                    self.target.set_weights(self.online.get_weights())
                    self._apply_loaded_progress(target)
                    self._last_load_error = None
                    return True
                except Exception as exc:
                    errors.append(f"weights load: {exc}")

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
        reasons_t = self.tf.convert_to_tensor(
            batch.reasons if batch.reasons is not None else np.full(len(batch.actions), -1, dtype=np.int32),
            dtype=self.tf.int32,
        )
        skills_t = self.tf.convert_to_tensor(
            batch.skills if batch.skills is not None else np.full(len(batch.actions), -1, dtype=np.int32),
            dtype=self.tf.int32,
        )
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
                reasons_t,
                skills_t,
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
                reasons_t,
                skills_t,
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
        opt = self.online.optimizer
        # LossScaleOptimizer (mixed precision) wraps the inner Adam.
        inner = getattr(opt, "inner_optimizer", None) or getattr(opt, "_optimizer", None)
        target = inner if inner is not None else opt
        lr_var = getattr(target, "learning_rate", None)
        if lr_var is not None and hasattr(lr_var, "assign"):
            lr_var.assign(learning_rate)
        elif hasattr(opt, "learning_rate") and hasattr(opt.learning_rate, "assign"):
            opt.learning_rate.assign(learning_rate)

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
                reasons,
                skills,
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

                online_next_out = agent.online(next_states_tensor, training=False)
                target_next_out = agent.target(next_states_tensor, training=False)
                online_next_q, _, _ = agent._unpack_outputs(online_next_out)
                target_next_q, _, _ = agent._unpack_outputs(target_next_out)
                online_next_q = tf.cast(online_next_q, tf.float32)
                target_next_q = tf.cast(target_next_q, tf.float32)
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
                aux_weight = float(agent.config.training.reason_aux_weight)
                skill_aux_weight = float(
                    getattr(agent.config.training, "skill_aux_weight", 0.0) or 0.0
                )
                reason_count_t = int(agent._reason_count)
                skill_count_t = int(agent._skill_count)

                with tf.GradientTape() as tape:
                    online_out = agent.online(states_tensor, training=True)
                    q_values, reason_logits, skill_logits = agent._unpack_outputs(
                        online_out
                    )
                    q_values = tf.cast(q_values, tf.float32)
                    if reason_logits is not None:
                        reason_logits = tf.cast(reason_logits, tf.float32)
                    if skill_logits is not None:
                        skill_logits = tf.cast(skill_logits, tf.float32)
                    action_masks = tf.one_hot(actions, action_count)
                    selected_q = tf.reduce_sum(q_values * action_masks, axis=1)
                    td_errors = targets - selected_q
                    per_sample = tf.keras.losses.Huber(reduction="none")(targets, selected_q)
                    if weights is not None:
                        per_sample = per_sample * weights
                    loss = tf.reduce_mean(per_sample)
                    if reason_logits is not None and aux_weight > 0.0:
                        valid = reasons >= 0
                        # Clip invalid to 0 for one_hot, then mask.
                        safe_reasons = tf.clip_by_value(reasons, 0, reason_count_t - 1)
                        ce = tf.keras.losses.sparse_categorical_crossentropy(
                            safe_reasons,
                            reason_logits,
                            from_logits=True,
                        )
                        ce = tf.where(valid, ce, tf.zeros_like(ce))
                        denom = tf.maximum(tf.reduce_sum(tf.cast(valid, tf.float32)), 1.0)
                        loss = loss + aux_weight * (tf.reduce_sum(ce) / denom)
                    if (
                        skill_logits is not None
                        and skill_aux_weight > 0.0
                        and skill_count_t > 0
                    ):
                        valid_s = skills >= 0
                        safe_skills = tf.clip_by_value(skills, 0, skill_count_t - 1)
                        ce_s = tf.keras.losses.sparse_categorical_crossentropy(
                            safe_skills,
                            skill_logits,
                            from_logits=True,
                        )
                        ce_s = tf.where(valid_s, ce_s, tf.zeros_like(ce_s))
                        denom_s = tf.maximum(
                            tf.reduce_sum(tf.cast(valid_s, tf.float32)), 1.0
                        )
                        loss = loss + skill_aux_weight * (tf.reduce_sum(ce_s) / denom_s)

                agent._last_td_errors = td_errors

                opt = agent.online.optimizer
                train_loss = loss
                if hasattr(opt, "get_scaled_loss"):
                    train_loss = opt.get_scaled_loss(loss)
                gradients = tape.gradient(train_loss, agent.online.trainable_variables)
                if hasattr(opt, "get_unscaled_gradients"):
                    gradients = opt.get_unscaled_gradients(gradients)
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
                    outputs = agent.online(model_in, training=True)
                    q_values, _, _ = agent._unpack_outputs(outputs)
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    )(actions, q_values)
                opt = agent.online.optimizer
                train_loss = loss
                if hasattr(opt, "get_scaled_loss"):
                    train_loss = opt.get_scaled_loss(loss)
                gradients = tape.gradient(train_loss, agent.online.trainable_variables)
                if hasattr(opt, "get_unscaled_gradients"):
                    gradients = opt.get_unscaled_gradients(gradients)
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

    def train_skill_head(self, *, steps: int, lr: float) -> float | None:
        """Behavior-clone router skill labels stored in replay (skill head only)."""
        if self._skill_count <= 0 or len(self.replay) < max(64, self.batch_size):
            return None
        self.set_learning_rate(lr)
        trainable = []
        for layer in self.online.layers:
            if layer.name == "skill_logits" or "skill_logits" in (layer.name or ""):
                layer.trainable = True
                trainable.append(layer)
            else:
                layer.trainable = False
        if not trainable:
            for var in self.online.trainable_variables:
                if "skill_logits" in var.name:
                    trainable.append(var)
        last_loss = None
        batch_size = min(self.batch_size, len(self.replay))
        try:
            with self.tf.device(self.device_info.training_device):
                for i in range(max(1, steps)):
                    batch = self.replay.sample(batch_size)
                    if batch.skills is None:
                        return None
                    valid = batch.skills >= 0
                    if not np.any(valid):
                        return None
                    loss = float(
                        self._invoke_skill_bc_step(
                            batch.states, batch.skills
                        ).numpy()
                    )
                    last_loss = loss
                    if (i + 1) % max(1, steps // 5) == 0:
                        print(
                            f"SKILL_BC step={i + 1}/{steps} loss={loss:.4f}",
                            flush=True,
                        )
        finally:
            for layer in self.online.layers:
                layer.trainable = True
        return last_loss

    def evaluate_skill_accuracy(
        self,
        *,
        max_batches: int = 32,
        holdout_fraction: float = 0.25,
    ) -> dict[str, float]:
        """Held-out skill-head accuracy / CE against router labels in replay."""
        if self._skill_count <= 0 or len(self.replay) < max(64, self.batch_size):
            return {"accuracy": 0.0, "loss": -1.0, "n": 0.0, "valid": 0.0}
        n = len(self.replay)
        hold = max(self.batch_size, int(n * holdout_fraction))
        batch_size = min(self.batch_size, hold)
        correct = 0
        total = 0
        loss_sum = 0.0
        batches = 0
        with self.tf.device(self.device_info.inference_device):
            for _ in range(max(1, max_batches)):
                batch = self.replay.sample(batch_size)
                if batch.skills is None:
                    break
                skills = np.asarray(batch.skills, dtype=np.int32)
                valid = skills >= 0
                if not np.any(valid):
                    continue
                model_batch = self._to_model_batch(batch.states)
                if self._vector_dim > 0:
                    outputs = self.online(
                        [
                            self.tf.convert_to_tensor(model_batch[0], dtype=self.tf.float32),
                            self.tf.convert_to_tensor(model_batch[1], dtype=self.tf.float32),
                        ],
                        training=False,
                    )
                else:
                    outputs = self.online(
                        self.tf.convert_to_tensor(model_batch, dtype=self.tf.float32),
                        training=False,
                    )
                _, _, skill_logits = self._unpack_outputs(outputs)
                if skill_logits is None:
                    break
                logits = (
                    skill_logits.numpy()
                    if hasattr(skill_logits, "numpy")
                    else np.asarray(skill_logits)
                )
                pred = np.argmax(logits, axis=1).astype(np.int32)
                correct += int(np.sum((pred == skills) & valid))
                total += int(np.sum(valid))
                # Sparse CE for reporting.
                safe = np.clip(skills, 0, self._skill_count - 1)
                row = np.arange(len(safe))
                log_probs = logits - logits.max(axis=1, keepdims=True)
                log_probs = log_probs - np.log(
                    np.sum(np.exp(log_probs), axis=1, keepdims=True) + 1e-8
                )
                ce = -log_probs[row, safe]
                loss_sum += float(np.sum(ce[valid]))
                batches += 1
        accuracy = float(correct / max(1, total))
        loss = float(loss_sum / max(1, total)) if total else -1.0
        result = {
            "accuracy": accuracy,
            "loss": loss,
            "n": float(total),
            "valid": float(total),
            "batches": float(batches),
        }
        path = Path(
            getattr(
                self.config.training,
                "skill_accuracy_path",
                Path("checkpoints/skill_head_accuracy.json"),
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    **result,
                    "threshold": float(
                        getattr(self.config.training, "skill_exec_min_accuracy", 0.85)
                        or 0.85
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"SKILL_ACCURACY accuracy={accuracy:.3f} loss={loss:.4f} n={total} "
            f"-> {path.name}",
            flush=True,
        )
        return result

    def skill_exec_allowed(self) -> bool:
        """Watch may execute predicted skills only after accuracy clears threshold."""
        if not getattr(self.config.training, "skills_enabled", False):
            return False
        if not getattr(self.config.training, "skill_exec_at_watch", True):
            return False
        threshold = float(
            getattr(self.config.training, "skill_exec_min_accuracy", 0.85) or 0.85
        )
        path = Path(
            getattr(
                self.config.training,
                "skill_accuracy_path",
                Path("checkpoints/skill_head_accuracy.json"),
            )
        )
        if not path.exists():
            return False
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            return float(meta.get("accuracy", 0.0)) >= threshold
        except (OSError, ValueError, TypeError):
            return False

    @property
    def _skill_bc_train_step(self):
        if not hasattr(self, "_compiled_skill_bc_train_step"):
            agent = self
            tf = agent.tf
            skill_count_t = int(agent._skill_count)

            @self.tf.function
            def skill_bc_step(state_visual, state_vector, skills):
                if agent._vector_dim > 0:
                    model_in = [state_visual, state_vector]
                else:
                    model_in = state_visual
                with tf.GradientTape() as tape:
                    outputs = agent.online(model_in, training=True)
                    _, _, skill_logits = agent._unpack_outputs(outputs)
                    if skill_logits is None:
                        return tf.constant(0.0, dtype=tf.float32)
                    valid = skills >= 0
                    safe = tf.clip_by_value(skills, 0, skill_count_t - 1)
                    ce = tf.keras.losses.sparse_categorical_crossentropy(
                        safe, skill_logits, from_logits=True
                    )
                    ce = tf.where(valid, ce, tf.zeros_like(ce))
                    denom = tf.maximum(tf.reduce_sum(tf.cast(valid, tf.float32)), 1.0)
                    loss = tf.reduce_sum(ce) / denom
                vars_skill = [
                    v
                    for v in agent.online.trainable_variables
                    if "skill_logits" in v.name
                ]
                gradients = tape.gradient(loss, vars_skill)
                pairs = [
                    (g, v)
                    for g, v in zip(gradients, vars_skill, strict=True)
                    if g is not None
                ]
                if pairs:
                    agent.online.optimizer.apply_gradients(pairs)
                return loss

            self._compiled_skill_bc_train_step = skill_bc_step
        return self._compiled_skill_bc_train_step

    def _invoke_skill_bc_step(self, states, skills):
        skills_t = self.tf.convert_to_tensor(skills, dtype=self.tf.int32)
        model_batch = self._to_model_batch(states)
        dummy = self.tf.zeros((self.tf.shape(skills_t)[0], 1), dtype=self.tf.float32)
        if self._vector_dim > 0:
            visual, vector = model_batch
            return self._skill_bc_train_step(
                self.tf.convert_to_tensor(visual, dtype=self.tf.float32),
                self.tf.convert_to_tensor(vector, dtype=self.tf.float32),
                skills_t,
            )
        return self._skill_bc_train_step(
            self.tf.convert_to_tensor(model_batch, dtype=self.tf.float32),
            dummy,
            skills_t,
        )

    def weight_norm(self) -> float:
        total = 0.0
        for weight in self.online.get_weights():
            total += float(np.sum(np.square(weight)))
        return float(np.sqrt(total))

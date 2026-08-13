from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from ascent_player.agent.reason import EXPLORE, assign_reason, reason_to_index
from ascent_player.agent.skills import FREE_PLAY, skill_to_index
from ascent_player.env.state_detector import FrameState


class ActingMixin:
    """Action selection, teacher priors, and Watch skill-exec gate."""

    def _annealed_prior(self, start: float, end: float, steps: int) -> float:
        if steps <= 0:
            return max(0.0, float(end))
        if abs(float(start) - float(end)) < 1e-12:
            return float(start)
        origin = int(getattr(self, "_prior_anneal_origin", 0) or 0)
        local = max(0, int(self.metrics.total_steps) - origin)
        progress = min(1.0, local / float(steps))
        return max(float(end), float(start) - (float(start) - float(end)) * progress)

    def rule_prior_probability(self) -> float:
        training = self.config.training
        if training.rule_prior_steps <= 0:
            return 0.0
        return self._annealed_prior(
            training.rule_prior_start,
            training.rule_prior_end,
            training.rule_prior_steps,
        )

    def seed_thread_prior_probability(self) -> float:
        training = self.config.training
        if not getattr(training, "seed_thread_enabled", False):
            return 0.0
        return self._annealed_prior(
            float(getattr(training, "seed_thread_prior_start", 0.0) or 0.0),
            float(getattr(training, "seed_thread_prior_end", 0.0) or 0.0),
            int(getattr(training, "seed_thread_prior_steps", 0) or 0),
        )

    def skill_teacher_prior_probability(self) -> float:
        training = self.config.training
        if not getattr(training, "skills_enabled", False):
            return 0.0
        return self._annealed_prior(
            float(getattr(training, "skill_teacher_prior_start", 0.0) or 0.0),
            float(getattr(training, "skill_teacher_prior_end", 0.0) or 0.0),
            int(getattr(training, "skill_teacher_prior_steps", 0) or 0),
        )

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

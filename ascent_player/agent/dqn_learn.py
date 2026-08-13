from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ascent_player.agent.replay_buffer import TransitionBatch


class LearningMixin:
    """TD / BC train graphs, optimizer lifecycle, skill-head BC."""

    def advance_steps(self, count: int = 1):
        if count <= 0:
            return self.metrics
        self.metrics.total_steps += count
        if self.config.training.watch_mode or self.config.training.disable_td:
            return self.metrics
        if len(self.replay) < self.config.training.min_replay_size:
            return self.metrics
        if self.metrics.total_steps % self.train_every != 0:
            return self.metrics

        batch = self.sample_training_batch()
        start = time.perf_counter()
        with self.tf.device(self.device_info.training_device):
            loss = self.train_batch(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.metrics.loss = float(loss)
        self.metrics.train_ms = elapsed_ms

        if self.metrics.total_steps % self.config.training.target_sync_interval == 0:
            self.sync_target_network(hard=True)
        else:
            self.sync_target_network(hard=False)
        if self._sim_pretrain_mode:
            self.anneal_epsilon_by_steps()
        return self.metrics

    def maybe_train(self):
        return self.advance_steps(1)

    def train_batch(self, batch: TransitionBatch):
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
            bc_loss = self.bc_train_step(
                demo_batch.states,
                demo_batch.actions,
            )
            loss = loss + self.config.demo.bc_loss_weight * bc_loss
        return loss

    def _train_batch(self, batch: TransitionBatch):
        return self.train_batch(batch)

    def sync_target_network(self, *, hard: bool) -> None:
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

    def _sync_target_network(self, *, hard: bool) -> None:
        return self.sync_target_network(hard=hard)

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

    def rebuild_optimizer(self, learning_rate: float | None = None) -> None:
        """Recreate Adam (+ LossScale) after BC/load so TD can see all variables.

        Mixed-precision LossScaleOptimizer tracks the variable set from the first
        apply_gradients call. Offline BC only backprops into Q logits, so a later
        TD step that also touches reason/skill heads raises "Unknown variable".
        """
        lr = float(
            self.config.training.learning_rate
            if learning_rate is None
            else learning_rate
        )
        self.config.training.learning_rate = lr
        optimizer = self.tf.keras.optimizers.Adam(learning_rate=lr)
        if self._use_mixed_precision:
            optimizer = self.tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        self.online.optimizer = optimizer
        for attr in (
            "_compiled_train_step",
            "_compiled_bc_train_step",
            "_compiled_skill_bc_train_step",
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        print(
            f"OPTIMIZER_REBUILD lr={lr:.2e} mixed_precision={int(self._use_mixed_precision)}",
            flush=True,
        )

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

    def bc_train_step(self, states, actions):
        """Public one-batch behavior-clone update (returns TF scalar loss)."""
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

    def _invoke_bc_train_step(self, states, actions):
        """Backward-compatible alias for :meth:`bc_train_step`."""
        return self.bc_train_step(states, actions)

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

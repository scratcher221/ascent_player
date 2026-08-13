from __future__ import annotations

import time

import numpy as np

from ascent_player.agent.replay_buffer import ReplayBuffer, TransitionBatch


class ReplayControlMixin:
    """N-step remember, mixed-batch sampling, episode/epsilon bookkeeping."""

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
                    self.bc_train_step(
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
                    self.bc_train_step(batch_states, batch_actions).numpy()
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

    def sample_training_batch(self) -> TransitionBatch:
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

    def _sample_training_batch(self) -> TransitionBatch:
        return self.sample_training_batch()

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


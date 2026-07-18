from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ascent_player.agent.dqn import DQNAgent
from ascent_player.agent.replay_buffer import ReplayBuffer
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.evaluation import score_gates


class Phase01LearningFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = AppConfig()
        cls.config.observation.include_vector_state = False
        cls.config.training.device_mode = DeviceMode.CPU
        cls.config.training.min_replay_size = 16
        cls.config.training.batch_size_cpu = 8
        cls.config.training.train_every_cpu = 1
        cls.config.training.n_step = 3
        cls.config.training.use_prioritized_replay = True

    def test_n_step_stores_gamma_n_discount(self) -> None:
        agent = DQNAgent(self.config)
        gamma = self.config.training.gamma
        channels = self.config.observation.channel_count
        shape = (84, 84, channels)
        states = [np.zeros(shape, dtype=np.float32) for _ in range(4)]
        for index in range(3):
            agent.remember(
                states[index],
                1,
                1.0,
                states[index + 1],
                done=False,
            )
        self.assertEqual(len(agent.replay), 1)
        batch = agent.replay.sample(1)
        expected = gamma ** 3
        self.assertAlmostEqual(float(batch.discounts[0]), expected, places=5)
        # n-step return: 1 + g + g^2
        expected_return = 1.0 + gamma + gamma**2
        self.assertAlmostEqual(float(batch.rewards[0]), expected_return, places=5)

    def test_remember_batch_uses_per_env_n_step(self) -> None:
        agent = DQNAgent(self.config)
        channels = self.config.observation.channel_count
        shape = (84, 84, channels)
        env_count = 2
        for _ in range(3):
            states = [np.zeros(shape, dtype=np.float32) for _ in range(env_count)]
            next_states = [np.ones(shape, dtype=np.float32) for _ in range(env_count)]
            agent.remember_batch(
                states,
                np.zeros(env_count, dtype=np.int32),
                np.ones(env_count, dtype=np.float32),
                next_states,
                np.zeros(env_count, dtype=np.float32),
            )
        # Each env should have flushed one n-step transition.
        self.assertEqual(len(agent.replay), env_count)

    def test_checkpoint_weights_roundtrip(self) -> None:
        agent = DQNAgent(self.config)
        before = [w.copy() for w in agent.online.get_weights()]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.keras"
            agent.save(path)
            weights_path = DQNAgent.weights_sidecar_path(path)
            self.assertTrue(weights_path.exists())
            agent2 = DQNAgent(self.config)
            ok = agent2.load(path)
            self.assertTrue(ok, msg=agent2._last_load_error)
            after = agent2.online.get_weights()
            for left, right in zip(before, after, strict=True):
                np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-5)

    def test_mixed_batch_keeps_per_weights(self) -> None:
        agent = DQNAgent(self.config)
        channels = self.config.observation.channel_count
        shape = (84, 84, channels)
        for _ in range(32):
            state = np.random.rand(*shape).astype(np.float32)
            agent.replay.add(state, 0, 0.1, state, False, discount=0.99)
            agent.sim_replay.add(state, 1, 0.2, state, False, discount=0.99)
        agent.config.training.mixed_sim_replay_ratio = 0.5
        agent.batch_size = 8
        agent.metrics.total_steps = 50_000
        batch = agent._sample_training_batch()
        self.assertEqual(len(batch.actions), 8)
        self.assertIsNotNone(batch.weights)
        self.assertIsNotNone(batch.indices)
        self.assertIsNotNone(batch.sim_indices)
        self.assertEqual(len(batch.indices) + len(batch.sim_indices), 8)

    def test_per_beta_anneals(self) -> None:
        buffer = ReplayBuffer(100, prioritized=True, beta=0.4)
        self.assertAlmostEqual(buffer.beta, 0.4)
        buffer.set_beta(1.0)
        self.assertAlmostEqual(buffer.beta, 1.0)
        agent = DQNAgent(self.config)
        agent.metrics.total_steps = agent.config.training.per_beta_anneal_steps
        beta = agent._anneal_per_beta()
        self.assertAlmostEqual(beta, agent.config.training.per_beta_end, places=5)

    def test_score_gates(self) -> None:
        config = AppConfig()
        self.assertEqual(score_gates(500, 400, config), [])
        self.assertEqual(score_gates(950, 800, config), ["A"])
        self.assertIn("B", score_gates(2100, 1800, config))
        self.assertIn("D", score_gates(12000, 8000, config))
        self.assertNotIn("D", score_gates(12000, 5000, config))


if __name__ == "__main__":
    unittest.main()

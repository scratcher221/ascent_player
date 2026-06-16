from __future__ import annotations

import asyncio
import unittest

import numpy as np

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.env.sim_env import AscentSimEnv


class LearningSanityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = AppConfig()
        cls.config.observation.include_vector_state = False
        cls.config.training.device_mode = DeviceMode.CPU
        cls.config.training.min_replay_size = 32
        cls.config.training.batch_size_cpu = 16
        cls.config.training.train_every_cpu = 1
        cls.config.demo.pretrain_steps = 20

    def test_bc_pretrain_changes_weights(self) -> None:
        agent = DQNAgent(self.config)
        before = agent.weight_norm()
        channels = self.config.observation.channel_count
        visual = np.random.rand(32, 84, 84, channels).astype(np.float32)
        if self.config.observation.include_vector_state:
            states = [
                (visual[i], np.zeros(self.config.observation.vector_dim, dtype=np.float32))
                for i in range(32)
            ]
        else:
            states = visual
        actions = np.random.randint(0, 6, size=32, dtype=np.int32)
        rewards = np.zeros(32, dtype=np.float32)
        dones = np.zeros(32, dtype=np.float32)
        if isinstance(states, list):
            for i in range(32):
                agent.demo_replay.add(
                    states[i],
                    int(actions[i]),
                    float(rewards[i]),
                    states[i],
                    bool(dones[i]),
                )
        else:
            agent.absorb_demonstration_arrays(
                states,
                actions,
                rewards,
                states,
                dones,
                multiplier=1,
            )
        loss = agent.pretrain_from_replay(steps=30)
        after = agent.weight_norm()
        self.assertIsNotNone(loss)
        self.assertNotAlmostEqual(before, after, places=5)

    async def _fill_replay(self, agent: DQNAgent, steps: int) -> list[float]:
        env = AscentSimEnv(self.config)
        state = await env.reset()
        losses: list[float] = []
        try:
            for _ in range(steps):
                action = agent.act(
                    state,
                    training=True,
                    can_boost=env.can_boost,
                    boost_level=env.boost_level,
                )
                result = await env.step(action)
                agent.remember(
                    state,
                    action,
                    result.reward,
                    result.state,
                    result.done,
                    sim=False,
                )
                metrics = agent.maybe_train()
                if metrics.loss is not None:
                    losses.append(metrics.loss)
                state = result.state
                if result.done:
                    state = await env.reset()
        finally:
            await env.close()
        return losses

    def test_scaled_reward_training_loss_bounded(self) -> None:
        agent = DQNAgent(self.config)
        losses = asyncio.run(self._fill_replay(agent, 500))
        self.assertGreater(len(losses), 0)
        self.assertLess(max(losses), 100.0)

    def test_sim_observation_shape(self) -> None:
        async def check() -> None:
            env = AscentSimEnv(self.config)
            try:
                state = await env.reset()
                if isinstance(state, tuple):
                    visual, vector = state
                    self.assertEqual(
                        visual.shape,
                        (84, 84, self.config.observation.channel_count),
                    )
                    self.assertEqual(vector.shape[0], self.config.observation.vector_dim)
                else:
                    self.assertEqual(
                        state.shape,
                        (84, 84, self.config.observation.channel_count),
                    )
                result = await env.step(0)
                if isinstance(result.state, tuple):
                    visual, vector = result.state
                    self.assertEqual(visual.ndim, 3)
                    self.assertEqual(vector.shape[0], self.config.observation.vector_dim)
                else:
                    self.assertEqual(result.state.ndim, 3)
            finally:
                await env.close()

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()

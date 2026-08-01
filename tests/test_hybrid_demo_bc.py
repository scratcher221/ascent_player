from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ascent_player.agent.dqn import DQNAgent
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.demo.export import export_demos_to_replay
from ascent_player.demo.storage import DemoTransition, save_demo


class HybridDemoBcTests(unittest.TestCase):
    def test_demo_replay_detects_hybrid_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            demo_path = tmp_path / "hybrid.npz"
            replay_path = tmp_path / "hybrid.pkl"
            vector_dim = 59
            transitions = [
                DemoTransition(
                    state=np.zeros((84, 84, 6), dtype=np.float32),
                    action=1,
                    reward=1.0,
                    next_state=np.ones((84, 84, 6), dtype=np.float32),
                    done=True,
                    state_vector=np.zeros(vector_dim, dtype=np.float32),
                    next_state_vector=np.ones(vector_dim, dtype=np.float32),
                    score=2000.0,
                    episode_id=1,
                    skill=2,
                )
            ]
            save_demo(demo_path, transitions)

            config = AppConfig()
            config.training.device_mode = DeviceMode.CPU
            config.training.mixed_precision = False
            config.training.model_variant = "nature"
            config.training.batch_size_gpu = 32
            config.training.batch_size_cpu = 8
            config.observation.include_vector_state = True
            config.observation.vector_dim = vector_dim
            config.training.browser_replay_max_items = 16
            export_demos_to_replay(
                config,
                demo_paths=[demo_path],
                replay_path=replay_path,
            )

            agent = DQNAgent(config)
            agent.demo_replay.clear()
            n = agent.demo_replay.load_pickle(
                replay_path,
                max_items=16,
                vector_dim=vector_dim,
            )
            self.assertEqual(n, 1)
            self.assertTrue(agent._demo_replay_has_hybrid_vectors())


if __name__ == "__main__":
    unittest.main()

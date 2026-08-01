from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ascent_player.agent.replay_buffer import ReplayBuffer
from ascent_player.config import AppConfig
from ascent_player.demo.export import export_demos_to_replay
from ascent_player.demo.storage import DemoTransition, open_demo, save_demo


class HumanDemoExportTests(unittest.TestCase):
    def test_save_and_open_demo_with_vectors_and_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo_roundtrip.npz"
            transitions = [
                DemoTransition(
                    state=np.ones((4, 4, 6), dtype=np.float32),
                    action=2,
                    reward=1.25,
                    next_state=np.full((4, 4, 6), 2.0, dtype=np.float32),
                    done=False,
                    state_vector=np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    next_state_vector=np.array([0.4, 0.5, 0.6], dtype=np.float32),
                    score=900.0,
                    episode_id=3,
                    skill=5,
                )
            ]
            save_demo(path, transitions)

            with open_demo(path, mmap=False) as demo:
                self.assertIsNotNone(demo.state_vectors)
                self.assertIsNotNone(demo.next_state_vectors)
                self.assertIsNotNone(demo.skills)
                np.testing.assert_allclose(demo.state_vectors[0], [0.1, 0.2, 0.3])
                np.testing.assert_allclose(demo.next_state_vectors[0], [0.4, 0.5, 0.6])
                self.assertEqual(int(demo.skills[0]), 5)
                self.assertEqual(float(demo.scores[0]), 900.0)
                self.assertEqual(int(demo.episode_ids[0]), 3)

    def test_export_demo_to_replay_pickle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            demo_path = tmp_path / "seeded_demo.npz"
            replay_path = tmp_path / "human_seeded_replay.pkl"

            transitions = [
                DemoTransition(
                    state=np.zeros((4, 4, 6), dtype=np.float32),
                    action=1,
                    reward=0.5,
                    next_state=np.ones((4, 4, 6), dtype=np.float32),
                    done=False,
                    state_vector=np.array([0.0, 1.0], dtype=np.float32),
                    next_state_vector=np.array([1.0, 0.0], dtype=np.float32),
                    score=1200.0,
                    episode_id=0,
                    skill=2,
                ),
                DemoTransition(
                    state=np.ones((4, 4, 6), dtype=np.float32),
                    action=4,
                    reward=1.5,
                    next_state=np.full((4, 4, 6), 2.0, dtype=np.float32),
                    done=True,
                    state_vector=np.array([0.2, 0.8], dtype=np.float32),
                    next_state_vector=np.array([0.3, 0.7], dtype=np.float32),
                    score=1800.0,
                    episode_id=0,
                    skill=4,
                ),
            ]
            save_demo(demo_path, transitions)

            config = AppConfig()
            config.training.browser_replay_max_items = 32
            result = export_demos_to_replay(
                config,
                demo_paths=[demo_path],
                replay_path=replay_path,
            )

            self.assertEqual(result.transitions_added, 2)
            self.assertEqual(result.skill_labeled, 2)
            self.assertEqual(result.vector_ready, 2)

            replay = ReplayBuffer(8)
            loaded = replay.load_pickle(
                replay_path,
                max_items=8,
                vector_dim=config.observation.vector_dim,
            )
            self.assertEqual(loaded, 2)
            batch = replay.sample(2)
            self.assertEqual(batch.skills.shape[0], 2)
            self.assertTrue(all(int(skill) in {2, 4} for skill in batch.skills))


if __name__ == "__main__":
    unittest.main()

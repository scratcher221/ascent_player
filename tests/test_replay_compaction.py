from __future__ import annotations

import unittest

import numpy as np

from ascent_player.agent.replay_buffer import ReplayBuffer


def _add_episode(buffer: ReplayBuffer, actions: list[int]) -> None:
    for index, action in enumerate(actions):
        state = np.asarray([index, action], dtype=np.float32)
        buffer.add(
            state,
            action,
            0.0,
            state + 1,
            index == len(actions) - 1,
        )


class ReplayCompactionTests(unittest.TestCase):
    def test_caps_episodes_and_uniformly_samples_transitions(self) -> None:
        replay = ReplayBuffer(10_000)
        _add_episode(replay, [0] * 300)
        _add_episode(replay, [1] * 300)
        _add_episode(replay, [2] * 300)

        stats = replay.compact_diverse_episodes(
            max_episodes=2,
            max_transitions_per_episode=32,
        )

        self.assertEqual(stats["input_episodes"], 3)
        self.assertEqual(stats["selected_episodes"], 2)
        self.assertLessEqual(len(replay), 64)
        batch = replay.sample(len(replay))
        self.assertEqual(int(np.sum(batch.dones)), 2)
        self.assertEqual(set(batch.actions.tolist()), {1, 2})

    def test_drops_duplicate_action_traces(self) -> None:
        replay = ReplayBuffer(10_000)
        _add_episode(replay, [1, 2] * 50)
        _add_episode(replay, [1, 2] * 50)
        _add_episode(replay, [2, 1] * 50)

        stats = replay.compact_diverse_episodes(
            max_episodes=8,
            max_transitions_per_episode=128,
        )

        self.assertEqual(stats["selected_episodes"], 2)
        self.assertEqual(int(np.sum(replay.sample(len(replay)).dones)), 2)

    def test_episode_score_and_id_survive_sampling(self) -> None:
        replay = ReplayBuffer(128)
        for index in range(3):
            state = np.asarray([index], dtype=np.float32)
            replay.add(
                state,
                1,
                0.0,
                state,
                index == 2,
                episode_score=2345.0,
                episode_id=77,
            )
        batch = replay.sample(3)
        self.assertTrue(np.allclose(batch.episode_scores, 2345.0))
        self.assertEqual(set(batch.episode_ids.tolist()), {77})

    def test_filter_min_episode_score_drops_legacy_and_low_scores(self) -> None:
        replay = ReplayBuffer(128)
        state = np.asarray([0], dtype=np.float32)
        replay.add(state, 0, 0.0, state, True, episode_score=1500.0, episode_id=1)
        replay.add(state, 1, 0.0, state, True, episode_score=2500.0, episode_id=2)
        # Legacy tuple without score stamp should be removed by the filter.
        replay._items.append((state.copy(), 2, 0.0, state.copy(), True))
        kept = replay.filter_min_episode_score(2000.0)
        self.assertEqual(kept, 1)
        batch = replay.sample(1)
        self.assertEqual(int(batch.actions[0]), 1)


if __name__ == "__main__":
    unittest.main()

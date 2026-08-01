from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ascent_player.agent.replay_buffer import ReplayBuffer
from ascent_player.utils.policy_floors import (
    apply_reliability_baseline,
    read_thread_bc_best,
    thread_bc_promotion_floor,
)


class PolicyFloorsTests(unittest.TestCase):
    def test_thread_bc_best_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            import ascent_player.utils.policy_floors as pf

            pf.THREAD_BC_BEST = root / "thread_bc_best_mean.txt"
            self.assertEqual(read_thread_bc_best(), 0.0)
            apply_reliability_baseline(900.0)
            self.assertEqual(thread_bc_promotion_floor(), 900.0)
            apply_reliability_baseline(850.0)
            self.assertEqual(thread_bc_promotion_floor(), 900.0)
            apply_reliability_baseline(950.0)
            self.assertEqual(thread_bc_promotion_floor(), 950.0)


class ReplayRangeFilterTests(unittest.TestCase):
    def test_filter_episode_score_range(self) -> None:
        import numpy as np

        replay = ReplayBuffer(10)
        for score in (1700.0, 1900.0, 2100.0):
            state = np.zeros(4, dtype=np.float32)
            replay.add(
                state,
                0,
                0.0,
                state,
                True,
                episode_score=score,
            )
        replay.filter_episode_score_range(1800.0, 2000.0)
        self.assertEqual(len(replay), 1)


if __name__ == "__main__":
    unittest.main()

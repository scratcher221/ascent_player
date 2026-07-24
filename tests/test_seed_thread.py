from __future__ import annotations

import unittest

from ascent_player.agent.reason import RULE_PRIOR, assign_reason
from ascent_player.agent.teacher import SeedThreadPolicy
from ascent_player.env.state_detector import FrameState


class SeedThreadTests(unittest.TestCase):
    def test_locks_thread_on_landing(self) -> None:
        policy = SeedThreadPolicy()
        self.assertIsNone(policy.thread_x)
        policy.observe(
            FrameState(
                orb_x=0.42,
                bounces=1,
                platform_landed=True,
                nearest_platform_dx=0.0,
            )
        )
        self.assertIsNotNone(policy.thread_x)
        self.assertAlmostEqual(policy.thread_x or 0.0, 0.42, places=3)
        self.assertEqual(policy.landings_locked, 1)

    def test_rising_prefers_next_pad_with_thread_bias(self) -> None:
        policy = SeedThreadPolicy(thread_blend=0.5)
        policy.thread_x = 0.30
        # Orb to the right of thread; pad above nearly centered.
        state = FrameState(
            orb_x=0.50,
            rising=True,
            nearest_platform_above_dx=0.02,
            nearest_platform_above_dy=0.2,
            can_boost=True,
            boost_level=0.3,
        )
        dx = policy.target_dx(state)
        self.assertIsNotNone(dx)
        # Thread pull is leftward (0.30 - 0.50 = -0.20); blend should move dx left of pad.
        self.assertLess(dx or 0.0, 0.02)

    def test_survival_overrides_thread(self) -> None:
        policy = SeedThreadPolicy()
        policy.thread_x = 0.2
        state = FrameState(
            orb_x=0.5,
            falling=True,
            miss_risk=True,
            nearest_platform_dx=0.25,
            nearest_platform_dy=0.3,
            can_boost=True,
            boost_level=0.8,
        )
        action = policy.act(state)
        self.assertIn(action, {2, 5})  # right toward pad

    def test_reason_source_thread_maps_to_rule_prior(self) -> None:
        fs = FrameState(falling=True, nearest_platform_dx=-0.2)
        self.assertEqual(assign_reason(fs, 1, source="thread"), RULE_PRIOR)

    def test_suppresses_aligned_boost_dump(self) -> None:
        policy = SeedThreadPolicy(suppress_aligned_boost_energy=0.4)
        policy.thread_x = 0.5
        state = FrameState(
            orb_x=0.5,
            rising=True,
            nearest_platform_above_dx=0.0,
            nearest_platform_above_dy=0.05,
            can_boost=True,
            boost_level=0.9,
            boost_useful=False,
            orb_vy=0.1,
        )
        action = policy.act(state)
        self.assertNotIn(action, {3, 4, 5})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from ascent_player.env.sim_physics import SimBall, SimPhysicsConfig, SimPlatform, SimWorld
from ascent_player.env.sim_state_adapter import (
    build_frame_state_from_world,
    game_orb_vy,
    game_orb_y,
)


class SimStateAdapterTests(unittest.TestCase):
    def test_game_orb_y_flips_screen_coords(self) -> None:
        self.assertAlmostEqual(game_orb_y(0.0, 360.0), 1.0)
        self.assertAlmostEqual(game_orb_y(360.0, 360.0), 0.0)
        self.assertAlmostEqual(game_orb_y(180.0, 360.0), 0.5)

    def test_game_orb_vy_flips_physics_velocity(self) -> None:
        self.assertEqual(game_orb_vy(-640.0), 640.0)
        self.assertEqual(game_orb_vy(200.0), -200.0)

    def test_rising_ball_has_positive_game_vy(self) -> None:
        world = SimWorld(SimPhysicsConfig(width=640, height=360, seed=1))
        world.reset()
        world.ball.vy = -400.0
        state = build_frame_state_from_world(world)
        self.assertTrue(state.rising)
        self.assertFalse(state.falling)
        self.assertGreater(state.orb_vy or 0.0, 20.0)

    def test_platform_below_has_positive_dy(self) -> None:
        world = SimWorld(SimPhysicsConfig(width=640, height=360, seed=2))
        world.reset()
        ball = world.ball
        below = SimPlatform(cx=ball.x, cy=ball.y + 80, width=60, height=10)
        world.platforms = [below]
        state = build_frame_state_from_world(world)
        self.assertGreater(state.nearest_platform_dy or 0.0, 0.0)

    def test_platform_above_has_positive_above_dy(self) -> None:
        world = SimWorld(SimPhysicsConfig(width=640, height=360, seed=3))
        world.reset()
        ball = world.ball
        above = SimPlatform(cx=ball.x, cy=ball.y - 80, width=60, height=10)
        world.platforms = [above]
        state = build_frame_state_from_world(world)
        self.assertGreater(state.nearest_platform_above_dy or 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()

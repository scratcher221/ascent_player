"""Tests for skill router and controllers."""

from __future__ import annotations

import unittest

from ascent_player.agent.teacher import RulePolicy
from ascent_player.agent.skills import (
    CLIMB_ABOVE,
    DODGE_HAZARD,
    FREE_PLAY,
    INTERCEPT_SURGE,
    LAND_BELOW,
    SkillControllers,
    SkillRouter,
    skill_to_index,
)
from ascent_player.env.state_detector import FrameState


class SkillRouterTests(unittest.TestCase):
    def test_falling_selects_land_below(self) -> None:
        fs = FrameState(
            falling=True,
            nearest_platform_dx=-0.2,
            nearest_platform_dy=0.25,
            can_boost=True,
            boost_level=0.5,
            agent_hook_ok=True,
        )
        router = SkillRouter()
        self.assertEqual(router.propose(fs), LAND_BELOW)
        action, skill = router.act(fs)
        self.assertEqual(skill, LAND_BELOW)
        self.assertIn(action, (1, 3, 4))

    def test_hazard_beats_land(self) -> None:
        fs = FrameState(
            falling=True,
            nearest_platform_dx=0.1,
            hazard_dx=0.08,
            agent_hook_ok=True,
        )
        self.assertEqual(SkillRouter().propose(fs), DODGE_HAZARD)

    def test_surge_when_target_near(self) -> None:
        fs = FrameState(
            rising=True,
            target_kind="booster_surge",
            target_dx=0.15,
            can_boost=True,
            boost_level=0.4,
            agent_hook_ok=True,
        )
        self.assertEqual(SkillRouter().propose(fs), INTERCEPT_SURGE)

    def test_climb_above_when_rising(self) -> None:
        fs = FrameState(
            rising=True,
            nearest_platform_above_dx=0.12,
            nearest_platform_above_dy=0.03,
            can_boost=True,
            boost_level=0.35,
            agent_hook_ok=True,
        )
        self.assertEqual(SkillRouter().propose(fs), CLIMB_ABOVE)

    def test_rising_airborne_no_longer_defaults_to_land_below(self) -> None:
        fs = FrameState(
            rising=True,
            airborne=True,
            orb_vy=0.2,
            nearest_platform_dx=-0.05,
            nearest_platform_dy=0.2,
            nearest_platform_above_dx=0.1,
            nearest_platform_above_dy=0.05,
            can_boost=True,
            boost_level=0.3,
            agent_hook_ok=True,
        )
        self.assertEqual(SkillRouter().propose(fs), CLIMB_ABOVE)

    def test_free_play_on_calm_rise(self) -> None:
        fs = FrameState(rising=True, agent_hook_ok=True)
        self.assertEqual(SkillRouter().propose(fs), FREE_PLAY)

    def test_controllers_mask_jump_without_boost(self) -> None:
        fs = FrameState(
            falling=True,
            nearest_platform_dx=0.0,
            nearest_platform_dy=0.3,
            can_boost=False,
            boost_level=0.05,
            agent_hook_ok=True,
        )
        action = SkillControllers(RulePolicy()).act(LAND_BELOW, fs)
        self.assertNotIn(action, (3, 4, 5))

    def test_skill_indices_stable(self) -> None:
        self.assertEqual(skill_to_index(FREE_PLAY), 0)
        self.assertEqual(skill_to_index(LAND_BELOW), 1)


if __name__ == "__main__":
    unittest.main()

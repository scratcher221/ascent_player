from __future__ import annotations

import unittest

from ascent_player.env.mechanics_rewards import MechanicsRewardTracker
from ascent_player.env.score_rules import combo_multiplier, live_score, live_style
from ascent_player.env.state_detector import FrameState
from ascent_player.config import AppConfig, MechanicsRewardConfig


class ScoreRulesTests(unittest.TestCase):
    def test_combo_multiplier_matches_game(self) -> None:
        self.assertAlmostEqual(combo_multiplier(0), 1.0)
        self.assertAlmostEqual(combo_multiplier(48), 5.0)
        self.assertAlmostEqual(combo_multiplier(24), 3.0)

    def test_live_score_climb_and_style(self) -> None:
        score = live_score(height=500, bank_style=0, bonus=100, combo=12, tier_weight=1.0)
        self.assertGreater(score, 100)


class MechanicsRewardTests(unittest.TestCase):
    def test_landing_and_combo_rewards_positive(self) -> None:
        tracker = MechanicsRewardTracker(MechanicsRewardConfig())
        tracker.set_curriculum_stage("M4")
        prev = FrameState(
            height=100,
            bounces=2,
            combo=2,
            bonus=50,
            agent_hook_ok=True,
        )
        curr = FrameState(
            height=120,
            bounces=3,
            combo=3,
            bonus=60,
            agent_hook_ok=True,
        )
        tracker.last_state = prev
        reward = tracker.compute(curr, action=1)
        self.assertGreater(reward, 0.0)

    def test_death_penalty(self) -> None:
        tracker = MechanicsRewardTracker(MechanicsRewardConfig())
        tracker.last_state = FrameState(height=10, agent_hook_ok=True)
        reward = tracker.compute(FrameState(height=10, game_over=True, agent_hook_ok=True), 0)
        self.assertLess(reward, 0.0)

    def test_m0_wasted_boost_penalty(self) -> None:
        tracker = MechanicsRewardTracker(MechanicsRewardConfig())
        tracker.set_curriculum_stage("M0")
        prev = FrameState(
            can_boost=True,
            boost_level=0.8,
            orb_vy=200.0,
            agent_hook_ok=True,
        )
        curr = FrameState(
            can_boost=False,
            boost_level=0.0,
            orb_vy=250.0,
            agent_hook_ok=True,
        )
        tracker.last_state = prev
        reward = tracker.compute(curr, action=3)
        self.assertLess(reward, tracker.config.survival)

    def test_m0_meaningful_boost_bonus(self) -> None:
        tracker = MechanicsRewardTracker(MechanicsRewardConfig())
        tracker.set_curriculum_stage("M0")
        prev = FrameState(
            can_boost=True,
            boost_level=0.8,
            nearest_platform_dy=0.25,
            orb_vy=-0.3,
            agent_hook_ok=True,
        )
        curr = FrameState(
            can_boost=False,
            boost_level=0.2,
            nearest_platform_dy=0.25,
            orb_vy=-0.5,
            boost_useful=True,
            agent_hook_ok=True,
        )
        tracker.last_state = prev
        reward = tracker.compute(curr, action=3)
        self.assertGreater(reward, tracker.config.survival)


class VectorObsTests(unittest.TestCase):
    def test_vector_shape(self) -> None:
        from ascent_player.env.vector_obs import VECTOR_DIM, vector_from_frame_state

        vec = vector_from_frame_state(
            FrameState(
                orb_x=320,
                orb_y=0.5,
                combo=6,
                bonus=100,
                agent_hook_ok=True,
            )
        )
        self.assertEqual(vec.shape[0], VECTOR_DIM)


class PlatformMaskTests(unittest.TestCase):
    def test_agent_payload_mask_draws_platforms(self) -> None:
        from ascent_player.env.state_detector import platform_mask_from_agent_payload

        payload = {
            "cameraY": 100.0,
            "canvasW": 640,
            "canvasH": 360,
            "platforms": [
                {"x": 320, "worldY": 200.0, "width": 80, "type": "neutral"},
            ],
        }
        mask = platform_mask_from_agent_payload(payload, width=640, height=360)
        self.assertEqual(mask.shape, (360, 640))
        self.assertGreater(mask.sum(), 0)


class CurriculumTests(unittest.TestCase):
    def test_stage_progression(self) -> None:
        from ascent_player.mechanics_curriculum import CurriculumMetrics, mechanics_stage_from_metrics

        config = AppConfig()
        metrics = CurriculumMetrics()
        self.assertEqual(mechanics_stage_from_metrics(metrics, config), "M0")
        for _ in range(10):
            metrics.record_episode(
                steps=500,
                bounces=40,
                height=500,
                combo=6,
                score=900,
            )
        stage = mechanics_stage_from_metrics(metrics, config)
        self.assertIn(stage, {"M3", "M4", "M5", "M6"})


if __name__ == "__main__":
    unittest.main()

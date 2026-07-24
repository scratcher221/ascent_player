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

    def test_direction_flip_penalty_and_persistence_bonus(self) -> None:
        cfg = MechanicsRewardConfig(
            direction_flip_penalty=-0.06,
            direction_persistence_steps=3,
            direction_persistence_bonus=0.03,
            steer_gain=0.0,
            wrong_way_penalty=0.0,
            aligned_bonus=0.0,
            survival=0.0,
        )
        tracker = MechanicsRewardTracker(cfg)
        base = FrameState(
            height=100,
            rising=True,
            target_dx=0.05,
            agent_hook_ok=True,
        )
        tracker.last_state = base
        # Hold right twice (build streak), third step pays persistence.
        r1 = tracker.compute(base, action=2)
        r2 = tracker.compute(base, action=2)
        r3 = tracker.compute(base, action=2)
        self.assertEqual(r1, 0.0)
        self.assertEqual(r2, 0.0)
        self.assertAlmostEqual(r3, 0.03)
        # Flip to left while rising (not exempt) → flip penalty.
        r_flip = tracker.compute(base, action=1)
        self.assertAlmostEqual(r_flip, -0.06)
        # Same flip while falling is exempt.
        falling = FrameState(
            height=90,
            falling=True,
            nearest_platform_dx=0.05,
            agent_hook_ok=True,
        )
        tracker.last_steer_dir = -1
        tracker.persist_steer_dir = -1
        tracker.persist_steer_steps = 2
        tracker.last_state = falling
        r_exempt = tracker.compute(falling, action=2)
        self.assertGreaterEqual(r_exempt, 0.0)

    def test_early_boost_dump_scaled_before_first_landing(self) -> None:
        cfg = MechanicsRewardConfig(
            early_boost_dump_penalty=-0.45,
            early_boost_dump_steps=100,
            early_boost_dump_min_drop=0.08,
            boost_spam_penalty=0.0,
            wasted_boost_penalty=0.0,
            boost_spent=0.0,
            timed_boost_bonus=0.0,
            survival=0.0,
            steer_gain=0.0,
            wrong_way_penalty=0.0,
            aligned_bonus=0.0,
            height_gain=0.0,
            score_gain=0.0,
            falling_penalty=0.0,
        )
        tracker = MechanicsRewardTracker(cfg)
        prev = FrameState(
            height=50,
            bounces=0,
            rising=True,
            can_boost=True,
            boost_level=1.0,
            boost_useful=False,
            agent_hook_ok=True,
        )
        curr = FrameState(
            height=50,
            bounces=0,
            rising=True,
            can_boost=True,
            boost_level=0.5,
            boost_useful=False,
            agent_hook_ok=True,
        )
        tracker.last_state = prev
        reward = tracker.compute(curr, action=3)  # jump
        # 0.5 drop / 0.08 → scale capped at 3 → -0.45 * 3
        self.assertAlmostEqual(reward, -1.35, places=5)
        # After a landing, same dump must not apply.
        tracker.episode_steps = 10
        landed_prev = FrameState(
            height=100,
            bounces=1,
            rising=True,
            can_boost=True,
            boost_level=1.0,
            boost_useful=False,
            agent_hook_ok=True,
        )
        landed_curr = FrameState(
            height=130,
            bounces=1,
            rising=True,
            can_boost=True,
            boost_level=0.5,
            boost_useful=False,
            agent_hook_ok=True,
        )
        tracker.last_state = landed_prev
        after = tracker.compute(landed_curr, action=3)
        self.assertGreater(after, -1.0)

    def test_landing_beats_height_only_climb(self) -> None:
        """Platform hit must dominate a pure height/score rocket tick."""
        cfg = MechanicsRewardConfig()
        land = MechanicsRewardTracker(cfg)
        land.last_state = FrameState(
            height=200,
            bounces=1,
            combo=2,
            score=100,
            agent_hook_ok=True,
        )
        land_r = land.compute(
            FrameState(
                height=210,
                bounces=2,
                combo=3,
                score=110,
                agent_hook_ok=True,
            ),
            action=0,
        )

        climb = MechanicsRewardTracker(cfg)
        climb.last_state = FrameState(
            height=200,
            bounces=0,
            combo=0,
            score=100,
            rising=True,
            orb_vy=0.5,
            can_boost=True,
            boost_level=0.9,
            agent_hook_ok=True,
        )
        climb_r = climb.compute(
            FrameState(
                height=400,
                bounces=0,
                combo=0,
                score=140,
                rising=True,
                orb_vy=0.6,
                can_boost=True,
                boost_level=0.5,
                agent_hook_ok=True,
            ),
            action=3,  # jump spam while rising
        )
        self.assertGreater(land_r, climb_r)
        self.assertGreater(land_r, 1.0)

    def test_rising_boost_spam_penalized(self) -> None:
        tracker = MechanicsRewardTracker(MechanicsRewardConfig())
        tracker.last_state = FrameState(
            rising=True,
            orb_vy=0.4,
            can_boost=True,
            boost_level=0.8,
            bounces=0,
            agent_hook_ok=True,
        )
        reward = tracker.compute(
            FrameState(
                rising=True,
                orb_vy=0.5,
                can_boost=True,
                boost_level=0.5,
                bounces=0,
                boost_useful=False,
                agent_hook_ok=True,
            ),
            action=3,
        )
        self.assertLess(reward, tracker.config.survival)

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
            rising=True,
            agent_hook_ok=True,
        )
        curr = FrameState(
            can_boost=False,
            boost_level=0.0,
            orb_vy=250.0,
            rising=True,
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
            falling=True,
            agent_hook_ok=True,
        )
        curr = FrameState(
            can_boost=False,
            boost_level=0.2,
            nearest_platform_dy=0.25,
            orb_vy=-0.5,
            falling=True,
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

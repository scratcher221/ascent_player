from __future__ import annotations

import unittest

from ascent_player.agent.seed_curriculum import (
    FineTuneReadiness,
    confirmed_promotion,
)


class SeedCurriculumTests(unittest.TestCase):
    def test_promotion_requires_long_eval_mean_and_minimum(self) -> None:
        self.assertTrue(
            confirmed_promotion(
                mean_score=1300,
                min_score=920,
                floor_mean=1250.7,
                target_min=900,
            )
        )
        self.assertFalse(
            confirmed_promotion(
                mean_score=1300,
                min_score=600,
                floor_mean=1250.7,
                target_min=900,
            )
        )
        self.assertFalse(
            confirmed_promotion(
                mean_score=1249,
                min_score=900,
                floor_mean=1250.7,
                target_min=900,
            )
        )

    def test_finetune_requires_two_consecutive_near_floor_rounds(self) -> None:
        gate = FineTuneReadiness(floor_ratio=0.95, required_rounds=2)
        self.assertFalse(gate.observe(1200, 1250))
        self.assertTrue(gate.observe(1190, 1250))
        self.assertFalse(gate.observe(1100, 1250))
        self.assertEqual(gate.consecutive_rounds, 0)


if __name__ == "__main__":
    unittest.main()

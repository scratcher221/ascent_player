from __future__ import annotations

import asyncio
import unittest

from ascent_player.config import AppConfig
from ascent_player.evaluation import evaluate_rule_baseline, score_percentiles


class EvaluationTests(unittest.TestCase):
    def test_score_percentiles(self) -> None:
        pct = score_percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(pct["mean"], 3.0)
        self.assertEqual(pct["min"], 1.0)
        self.assertEqual(pct["max"], 5.0)
        self.assertAlmostEqual(pct["p50"], 3.0)

    def test_rule_baseline_sim_runs(self) -> None:
        async def run() -> None:
            metrics = await evaluate_rule_baseline(
                AppConfig(),
                episodes=2,
                use_sim=True,
            )
            self.assertEqual(metrics.episodes, 2)
            self.assertGreater(metrics.mean_length, 10.0)
            self.assertGreaterEqual(metrics.p90, metrics.p10)
            self.assertEqual(len(metrics.scores), 2)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()

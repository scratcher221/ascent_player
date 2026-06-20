from __future__ import annotations

import asyncio
import unittest

from ascent_player.config import AppConfig
from ascent_player.evaluation import evaluate_rule_baseline


class EvaluationTests(unittest.TestCase):
    def test_rule_baseline_sim_runs(self) -> None:
        async def run() -> None:
            metrics = await evaluate_rule_baseline(
                AppConfig(),
                episodes=2,
                use_sim=True,
            )
            self.assertEqual(metrics.episodes, 2)
            self.assertGreater(metrics.mean_length, 10.0)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()

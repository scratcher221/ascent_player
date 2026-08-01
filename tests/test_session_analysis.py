from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ascent_player.utils.session_analysis import analyze_watchdog_log


class SessionAnalysisTests(unittest.TestCase):
    def test_parses_collect_and_reverts(self) -> None:
        text = """
COLLECT_DONE recent_avg=1100 replay=0
BC_PROBE_GREEDY mean=700.0 floor=1250.7
BC_REVERT mean=700.0
COLLECT_DONE recent_avg=1200 replay=0
ELITE_REPLAY merge before=100 browser=100 episodes=8 saved=643 gate>=2000
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skill_training_watchdog_20260725_120000.log"
            path.write_text(text, encoding="utf-8")
            analysis = analyze_watchdog_log(path)
        self.assertEqual(analysis.collect_avgs, [1100.0, 1200.0])
        self.assertEqual(analysis.bc_reverts, 1)
        self.assertIn("--skip-ceiling", analysis.watchdog_extra_args)
        self.assertIn("--bc-steps", analysis.watchdog_extra_args)


if __name__ == "__main__":
    unittest.main()

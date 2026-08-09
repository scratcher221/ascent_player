from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from ascent_player.utils.watchdog_log import (
    apply_log_lines,
    parse_log_tail,
    session_start_from_log_path,
    SessionStats,
)


class WatchdogLogTests(unittest.TestCase):
    def test_session_start_from_filename(self) -> None:
        start = session_start_from_log_path(
            Path("logs/skill_training_watchdog_20260725_152812.log")
        )
        self.assertEqual(start, datetime(2026, 7, 25, 15, 28, 12))

    def test_parse_tail_extracts_key_fields(self) -> None:
        text = """
THREAD_BC_CYCLE_START seed=424242 hours=8.0 floor=1250.7
CYCLE_ROUND id=2 remaining_h=7.32 collect_gate=1000.0
episode=920 reward=-3.728 score=826 epsilon=0.040
COLLECT_DONE recent_avg=965 replay=643
WATCHDOG_PROGRESS pid=1807856 events=172 last='episode=903'
EVAL_ALIGNED mean=1234.5
PHASE_COLLECT s=1200
""".strip()
        stats = parse_log_tail(text)
        self.assertEqual(stats.hours, 8.0)
        self.assertEqual(stats.cycle_round, 2)
        self.assertAlmostEqual(stats.cycle_remaining_h or 0, 7.32)
        self.assertEqual(stats.episode, 920)
        self.assertEqual(stats.episode_score, 826.0)
        self.assertEqual(stats.collect_recent_avg, 965.0)
        self.assertEqual(stats.watchdog_events, 172)
        self.assertEqual(stats.eval_aligned_mean, 1234.5)
        self.assertEqual(stats.last_phase, "PHASE_COLLECT")

    def test_parse_v2_climb_fields(self) -> None:
        text = """
CLIMB_START variant=impala_mid best=1165.7 hours=6.0 offline_only=0
CLIMB_ROUND 1 remaining_s=21594
CLIMB_BC step=160/800 loss=0.8399
CLIMB_POLICY_COLLECT s=720 prior=0.20 eps=0.04
CLIMB_PROBE_GREEDY mean=1012.5
CLIMB_BEST_UPDATE mean=1012.5
""".strip()
        stats = parse_log_tail(text)
        self.assertEqual(stats.session_kind, "v2_climb")
        self.assertEqual(stats.model_variant, "impala_mid")
        self.assertEqual(stats.hours, 6.0)
        self.assertEqual(stats.cycle_round, 1)
        self.assertEqual(stats.climb_bc_step, 160)
        self.assertEqual(stats.climb_bc_steps, 800)
        self.assertAlmostEqual(stats.climb_bc_loss or 0, 0.8399)
        self.assertAlmostEqual(stats.climb_probe_mean or 0, 1012.5)
        self.assertAlmostEqual(stats.climb_best_mean or 0, 1012.5)

    def test_v2_climb_session_start_from_filename(self) -> None:
        start = session_start_from_log_path(
            Path("logs/v2_climb_watchdog_20260801_165247.log")
        )
        self.assertEqual(start, datetime(2026, 8, 1, 16, 52, 47))

    def test_incremental_apply(self) -> None:
        stats = SessionStats()
        apply_log_lines(stats, ["episode=1 reward=1.0 score=100 epsilon=0.1"])
        apply_log_lines(stats, ["episode=2 reward=2.0 score=200 epsilon=0.2"])
        self.assertEqual(stats.episode, 2)
        self.assertEqual(stats.episode_score, 200.0)


if __name__ == "__main__":
    unittest.main()

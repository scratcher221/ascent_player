"""Tests for Impala checkpoint guards and climb policy helpers."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ascent_player.agent.checkpoint_guard import (
    is_impala_sized_checkpoint,
    is_likely_nature_checkpoint,
    reject_undersized_impala_save,
    restore_work_checkpoints_from_best,
)
from ascent_player.utils.climb_policy import (
    PromoteAction,
    decide_promote,
    human_cap,
    offline_td_min_score,
    reliability_below_floor,
    should_defer_offline_td,
)
import ascent_player.utils.policy_floors as policy_floors


class ImpalaCkptTests(unittest.TestCase):
    def test_nature_vs_impala_size_heuristics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nature = root / "nature.keras"
            nature.write_bytes(b"x" * (7 * 1024 * 1024))
            impala = root / "impala.keras"
            impala.write_bytes(b"x" * (57 * 1024 * 1024))
            self.assertTrue(is_likely_nature_checkpoint(nature))
            self.assertFalse(is_likely_nature_checkpoint(impala))
            self.assertTrue(is_impala_sized_checkpoint(impala))
            self.assertFalse(is_impala_sized_checkpoint(nature))

    def test_reject_catastrophic_shrink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.keras"
            path.write_bytes(b"x" * (57 * 1024 * 1024))
            prev = 287 * 1024 * 1024
            self.assertTrue(reject_undersized_impala_save(path, previous_bytes=prev))
            self.assertFalse(
                reject_undersized_impala_save(path, previous_bytes=60 * 1024 * 1024)
            )

    def test_restore_only_undersized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = root / "best.keras"
            best.write_bytes(b"x" * (57 * 1024 * 1024))
            good = root / "good.keras"
            good.write_bytes(b"y" * (57 * 1024 * 1024))
            bad = root / "bad.keras"
            bad.write_bytes(b"z" * (7 * 1024 * 1024))
            restored = restore_work_checkpoints_from_best(best, (good, bad))
            self.assertEqual(restored, ["bad.keras"])
            self.assertEqual(bad.stat().st_size, best.stat().st_size)
            self.assertTrue(good.read_bytes().startswith(b"y"))

    def test_human_cap(self) -> None:
        self.assertEqual(human_cap(10_000, 20_000, 0.25), 5000)
        self.assertEqual(human_cap(1000, 20_000, 0.25), 1000)

    def test_decide_promote(self) -> None:
        self.assertIs(
            decide_promote(
                probe_mean=700, v2_best=690, bootstrap_floor=350
            ).action,
            PromoteAction.PROMOTE,
        )
        self.assertIs(
            decide_promote(
                probe_mean=600, v2_best=690, bootstrap_floor=350
            ).action,
            PromoteAction.HOLD,
        )
        self.assertIs(
            decide_promote(
                probe_mean=200, v2_best=690, bootstrap_floor=350
            ).action,
            PromoteAction.REVERT,
        )

    def test_td_policy_helpers(self) -> None:
        self.assertTrue(should_defer_offline_td(811.0))
        self.assertTrue(should_defer_offline_td(1230.0))
        self.assertFalse(should_defer_offline_td(1400.0))
        self.assertEqual(
            offline_td_min_score(best_mean=1600, v2_best=900), 2000.0
        )
        self.assertEqual(
            offline_td_min_score(best_mean=800, v2_best=600), 600.0
        )
        self.assertEqual(
            offline_td_min_score(best_mean=400, v2_best=200), 400.0
        )
        self.assertTrue(reliability_below_floor(400.0, 811.0))
        self.assertFalse(reliability_below_floor(700.0, 811.0))

    def test_v2_and_seed_floors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prev_v2 = policy_floors.V2_PROBE_BEST
            prev_seed = policy_floors.SEED_MEAN
            try:
                policy_floors.V2_PROBE_BEST = root / "v2.txt"
                policy_floors.SEED_MEAN = root / "seed.txt"
                self.assertEqual(policy_floors.read_v2_probe_best(), 0.0)
                policy_floors.write_v2_probe_best(811.1)
                self.assertAlmostEqual(policy_floors.read_v2_probe_best(), 811.1)
                policy_floors.write_seed_floor(1250.7)
                self.assertAlmostEqual(policy_floors.read_seed_floor(), 1250.7)
            finally:
                policy_floors.V2_PROBE_BEST = prev_v2
                policy_floors.SEED_MEAN = prev_seed


if __name__ == "__main__":
    unittest.main()

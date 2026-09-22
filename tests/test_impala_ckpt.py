"""Tests for Impala checkpoint guards and climb policy helpers."""
from __future__ import annotations

import json
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

    def test_full_n_eval_and_gates(self) -> None:
        from ascent_player.utils.climb_policy import (
            collect_replay_gate,
            confirm_supports_probe,
            elite_gate_for_score,
            eval_session_stats,
            record_climb_session,
            COLLECT_PRIOR_PATH,
            CLIMB_SESSIONS,
        )

        stats = eval_session_stats([900, 1100, 800, 1500, 1000])
        self.assertAlmostEqual(stats["episode_mean"], 1060.0)
        self.assertEqual(stats["n_episodes"], 5.0)
        self.assertAlmostEqual(stats["recent_avg"], 1060.0)
        empty = eval_session_stats([])
        self.assertEqual(empty["episode_mean"], 0.0)

        self.assertEqual(elite_gate_for_score(1230.0), 1400.0)
        self.assertEqual(elite_gate_for_score(2000.0), 3000.0)
        self.assertEqual(elite_gate_for_score(3100.0), 4000.0)
        self.assertEqual(collect_replay_gate(1100.0), 1000.0)
        self.assertEqual(collect_replay_gate(1200.0), 1400.0)
        from ascent_player.utils.climb_policy import (
            bc_score_gate,
            browser_bc_gate,
            current_rung,
            online_td_gate,
            should_run_online_td,
            should_skip_leftover_hybrid_bc,
            PHASE2_STALL_PATH,
        )

        # Assisted 2k collect must not change the harvest gate; only true_mean does.
        self.assertEqual(elite_gate_for_score(1239.6), 1400.0)
        self.assertEqual(browser_bc_gate(1136.0, 35), 1400.0)
        self.assertEqual(browser_bc_gate(1136.0, 10), 1000.0)
        self.assertEqual(bc_score_gate(1136.0, 5000, elite_episodes=35), 1400.0)
        self.assertEqual(bc_score_gate(1239.6, 5000), 1400.0)
        self.assertEqual(bc_score_gate(1239.6, 0), 1000.0)
        self.assertEqual(bc_score_gate(1100.0, 0), 1000.0)
        self.assertTrue(
            should_skip_leftover_hybrid_bc(collecting=False, collect_minutes=15.0)
        )
        self.assertFalse(
            should_skip_leftover_hybrid_bc(collecting=True, collect_minutes=15.0)
        )
        self.assertFalse(
            should_skip_leftover_hybrid_bc(collecting=False, collect_minutes=0.0)
        )
        self.assertEqual(current_rung(1399.0), 0.0)
        self.assertEqual(current_rung(1400.0), 1400.0)
        self.assertEqual(current_rung(1599.0), 1400.0)
        self.assertEqual(current_rung(1600.0), 1600.0)
        self.assertEqual(current_rung(1800.0), 1800.0)
        self.assertEqual(current_rung(2000.0), 2000.0)
        self.assertTrue(confirm_supports_probe(1241.5, 1100.0))
        self.assertFalse(confirm_supports_probe(1241.5, 988.0))

        from ascent_player.agent.elite_replay import EliteReplayStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            replay = root / "elite.pkl"
            meta = root / "elite.json"
            replay.write_bytes(b"placeholder")
            meta.write_text(
                json.dumps(
                    {"min_episode_score": 1400.0, "selected_episodes": 35}
                )
                + "\n",
                encoding="utf-8",
            )
            store = EliteReplayStore(
                replay_path=replay, meta_path=meta, min_score=2000.0
            )
            self.assertEqual(store.selected_episodes(), 35)
            self.assertFalse(store.compatible())
            store.reset_if_incompatible()
            self.assertTrue(replay.exists(), "raising the gate must not wipe elite")
            self.assertTrue(meta.exists())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prev_sessions = CLIMB_SESSIONS
            prev_prior = COLLECT_PRIOR_PATH
            prev_stall = PHASE2_STALL_PATH
            try:
                import ascent_player.utils.climb_policy as cp

                cp.CLIMB_SESSIONS = root / "sessions.jsonl"
                cp.COLLECT_PRIOR_PATH = root / "prior.txt"
                cp.PHASE2_STALL_PATH = root / "stall.txt"
                self.assertEqual(cp.online_td_gate(1400.0), 1400.0)
                self.assertFalse(
                    cp.should_run_online_td(
                        collecting=True, true_mean=1136.0, ft_gate=1400.0
                    )
                )
                self.assertIsNone(
                    cp.record_climb_session(
                        start_true_mean=1050,
                        end_true_mean=1060,
                        max_score=3186,
                        hours=8,
                        greedy_max=1500,
                    )
                )
                self.assertTrue(cp.PHASE2_STALL_PATH.exists())
                self.assertEqual(cp.online_td_gate(1400.0), 1200.0)
                self.assertTrue(
                    cp.should_run_online_td(
                        collecting=True, true_mean=1136.0, ft_gate=1200.0
                    )
                )
                self.assertEqual(
                    cp.record_climb_session(
                        start_true_mean=1060,
                        end_true_mean=1070,
                        max_score=1510,
                        hours=8,
                        greedy_max=1510,
                    ),
                    "abort",
                )
                self.assertAlmostEqual(cp.collect_thread_prior(), 0.35)
            finally:
                cp.CLIMB_SESSIONS = prev_sessions
                cp.COLLECT_PRIOR_PATH = prev_prior
                cp.PHASE2_STALL_PATH = prev_stall

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
            prev_full = policy_floors.V2_FULL_N_BEST
            prev_last = policy_floors.V2_FULL_N_LAST
            try:
                policy_floors.V2_PROBE_BEST = root / "v2.txt"
                policy_floors.SEED_MEAN = root / "seed.txt"
                policy_floors.V2_FULL_N_BEST = root / "full.txt"
                policy_floors.V2_FULL_N_LAST = root / "last.txt"
                self.assertEqual(policy_floors.read_v2_probe_best(), 0.0)
                policy_floors.write_v2_probe_best(811.1)
                self.assertAlmostEqual(policy_floors.read_v2_probe_best(), 811.1)
                policy_floors.write_seed_floor(1250.7)
                self.assertAlmostEqual(policy_floors.read_seed_floor(), 1250.7)
                self.assertEqual(policy_floors.live_v2_floor(), 0.0)
                policy_floors.seed_v2_full_n_best(1050.2)
                self.assertAlmostEqual(policy_floors.live_v2_floor(), 1050.2)
                self.assertAlmostEqual(policy_floors.read_v2_full_n_last(), 1050.2)
                self.assertFalse(policy_floors.write_v2_full_n_best(1000.0))
                self.assertTrue(policy_floors.write_v2_full_n_best(1100.0))
                self.assertAlmostEqual(policy_floors.live_v2_floor(), 1100.0)
            finally:
                policy_floors.V2_PROBE_BEST = prev_v2
                policy_floors.SEED_MEAN = prev_seed
                policy_floors.V2_FULL_N_BEST = prev_full
                policy_floors.V2_FULL_N_LAST = prev_last


if __name__ == "__main__":
    unittest.main()

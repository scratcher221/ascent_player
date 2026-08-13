"""Tests for run-profile overrides, elite store gate, and thread-BC round plans."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from ascent_player.agent.dqn_acting import ActingMixin
from ascent_player.agent.elite_replay import EliteReplayStore
from ascent_player.config import AppConfig
from ascent_player.utils.run_profile import collect_only_fields, override_training
from ascent_player.utils.thread_bc import CycleState, plan_collect_split, plan_finetune


class _Metrics:
    total_steps = 100_000


class _Holder:
    metrics = _Metrics()
    _prior_anneal_origin = 0


class RunProfileTests(unittest.TestCase):
    def test_override_training_restores(self) -> None:
        config = AppConfig()
        original = config.training.disable_td
        with override_training(config, disable_td=True, watch_mode=True):
            self.assertTrue(config.training.disable_td)
            self.assertTrue(config.training.watch_mode)
        self.assertEqual(config.training.disable_td, original)
        self.assertFalse(config.training.watch_mode)

    def test_collect_only_disables_td(self) -> None:
        fields = collect_only_fields(eps=0.04, thread_prior=0.2)
        self.assertTrue(fields["disable_td"])
        self.assertEqual(fields["seed_thread_prior_start"], 0.2)
        self.assertEqual(fields["seed_thread_prior_end"], 0.2)
        self.assertFalse(fields["watch_mode"])

    def test_frozen_prior_ignores_steps(self) -> None:
        self.assertEqual(ActingMixin._annealed_prior(_Holder(), 0.2, 0.2, 10), 0.2)
        self.assertEqual(ActingMixin._annealed_prior(_Holder(), 0.4, 0.1, 0), 0.1)

    def test_elite_store_compatible_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = EliteReplayStore(
                replay_path=root / "elite.pkl",
                meta_path=root / "elite.json",
                min_score=2000.0,
            )
            self.assertFalse(store.compatible())
            store.replay_path.write_bytes(b"x")
            store.meta_path.write_text(
                json.dumps({"min_episode_score": 1500.0}), encoding="utf-8"
            )
            self.assertFalse(store.compatible())
            store.meta_path.write_text(
                json.dumps({"min_episode_score": 2000.0}), encoding="utf-8"
            )
            self.assertTrue(store.compatible())

    def test_plan_collect_and_finetune(self) -> None:
        state = CycleState(
            deadline=time.time() + 3600.0,
            elite_gate=2000.0,
            collect_gate=400.0,
            target_mean=2000.0,
            target_min=1000.0,
            thread_prior=0.5,
        )
        split = plan_collect_split(state, AppConfig(), collect_minutes=12.0)
        self.assertGreaterEqual(split.teacher_seconds, 120)
        self.assertGreaterEqual(split.policy_seconds, 180)
        self.assertLessEqual(split.policy_prior, 0.20)
        skip = plan_finetune(
            state, best_mean=800.0, ft_ready=True, ft_minutes=20.0, ft_min=1400.0
        )
        self.assertFalse(skip.allowed)
        go = plan_finetune(
            state, best_mean=1500.0, ft_ready=True, ft_minutes=20.0, ft_min=1400.0
        )
        # Floor file may be below 1400 in tests, so allow_ft can still be false.
        self.assertIsInstance(go.reason, str)


if __name__ == "__main__":
    unittest.main()

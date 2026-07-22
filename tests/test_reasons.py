"""Tests for entity labels, action reasons, vector dims, and reason aux head."""

from __future__ import annotations

import unittest

import numpy as np

from ascent_player.agent.model import build_q_network
from ascent_player.agent.reason import (
    AVOID_BOOSTER_DRAG,
    EXPLORE,
    TOWARD_BOOSTER_SURGE,
    TOWARD_PLATFORM_BELOW,
    assign_reason,
    reason_count,
    reason_to_index,
)
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.env.state_detector import FrameState, merge_agent_state
from ascent_player.env.vector_obs import VECTOR_DIM, vector_from_frame_state


class AssignReasonTests(unittest.TestCase):
    def test_explore_source(self) -> None:
        fs = FrameState(falling=True, nearest_platform_dx=-0.3)
        self.assertEqual(assign_reason(fs, 1, source="explore"), EXPLORE)

    def test_falling_toward_platform_below(self) -> None:
        fs = FrameState(falling=True, nearest_platform_dx=-0.25, agent_hook_ok=True)
        self.assertEqual(assign_reason(fs, 1, source="greedy"), TOWARD_PLATFORM_BELOW)
        self.assertEqual(assign_reason(fs, 4, source="greedy"), TOWARD_PLATFORM_BELOW)

    def test_surge_chase(self) -> None:
        fs = FrameState(
            rising=True,
            target_kind="booster_surge",
            target_dx=0.2,
            agent_hook_ok=True,
        )
        self.assertEqual(assign_reason(fs, 2, source="greedy"), TOWARD_BOOSTER_SURGE)

    def test_drag_avoid(self) -> None:
        fs = FrameState(
            rising=True,
            booster_type="drag",
            booster_dx=0.3,
            target_kind="booster_drag",
            target_dx=0.3,
            agent_hook_ok=True,
        )
        # Drag to the right → steer left to avoid
        self.assertEqual(assign_reason(fs, 1, source="greedy"), AVOID_BOOSTER_DRAG)


class VectorAndMergeTests(unittest.TestCase):
    def test_vector_dim_is_59(self) -> None:
        self.assertEqual(VECTOR_DIM, 59)
        cfg = AppConfig()
        self.assertEqual(cfg.observation.vector_dim, 59)
        vec = vector_from_frame_state(
            FrameState(
                orb_x=0.5,
                orb_y=0.5,
                target_kind="booster_surge",
                anomaly_type="liquidityVoid",
                portal_dx=0.1,
                portal_dy=-0.2,
                hazard_dx=-0.15,
                hazard_dy=0.05,
                agent_hook_ok=True,
            )
        )
        self.assertEqual(vec.shape[0], 59)

    def test_merge_anomaly_portal_hazard(self) -> None:
        base = FrameState(orb_x=0.5, orb_y=0.5, falling=False, rising=True)
        merged = merge_agent_state(
            base,
            {
                "activeAnomaly": {
                    "type": "liquidityVoid",
                    "remaining": 4.0,
                    "portal": {"dx": 0.2, "dy": -0.4, "kind": "void_portal"},
                    "nearestHazard": {
                        "dx": -0.3,
                        "dy": 0.1,
                        "kind": "void_shard",
                    },
                }
            },
        )
        self.assertEqual(merged.anomaly_type, "liquidityVoid")
        self.assertAlmostEqual(merged.portal_dx or 0.0, 0.2)
        self.assertEqual(merged.target_kind, "void_portal")
        self.assertEqual(merged.hazard_kind, "void_shard")
        self.assertAlmostEqual(merged.hazard_dx or 0.0, -0.3)


class ReasonHeadTests(unittest.TestCase):
    def test_model_two_outputs(self) -> None:
        model = build_q_network(
            (84, 84, 4),
            action_count=6,
            learning_rate=1e-4,
            vector_dim=VECTOR_DIM,
            reason_count=reason_count(),
        )
        self.assertEqual(len(model.outputs), 2)
        visual = np.zeros((1, 84, 84, 4), dtype=np.float32)
        vector = np.zeros((1, VECTOR_DIM), dtype=np.float32)
        q, reason_logits = model.predict([visual, vector], verbose=0)
        self.assertEqual(q.shape, (1, 6))
        self.assertEqual(reason_logits.shape, (1, reason_count()))

    def test_aux_train_step_runs(self) -> None:
        config = AppConfig()
        config.training.device_mode = DeviceMode.CPU
        config.training.min_replay_size = 8
        config.training.batch_size_cpu = 4
        config.training.train_every_cpu = 1
        config.observation.include_vector_state = True
        from ascent_player.agent.dqn import DQNAgent

        agent = DQNAgent(config)
        channels = config.observation.channel_count
        for _ in range(16):
            visual = np.random.rand(84, 84, channels).astype(np.float32)
            vector = np.random.randn(VECTOR_DIM).astype(np.float32) * 0.01
            state = (visual, vector)
            action = int(np.random.randint(0, 6))
            agent.last_reason_index = reason_to_index(TOWARD_PLATFORM_BELOW)
            agent.remember(state, action, 0.1, state, False)
        metrics = agent.maybe_train()
        self.assertIsNotNone(metrics.loss)


if __name__ == "__main__":
    unittest.main()

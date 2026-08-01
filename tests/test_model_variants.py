from __future__ import annotations

import unittest

from ascent_player.agent.model import (
    MODEL_VARIANT_IMPALA_LARGE,
    MODEL_VARIANT_IMPALA_MID,
    MODEL_VARIANT_NATURE,
    build_q_network,
)
from ascent_player.config import AppConfig, DeviceMode
from ascent_player.agent.dqn import DQNAgent


class ModelVariantTests(unittest.TestCase):
    def test_nature_param_band(self) -> None:
        model = build_q_network(
            (84, 84, 6),
            6,
            1e-4,
            vector_dim=59,
            reason_count=11,
            skill_count=8,
            model_variant=MODEL_VARIANT_NATURE,
        )
        n = model.count_params()
        self.assertGreater(n, 1_500_000)
        self.assertLess(n, 3_000_000)

    def test_impala_mid_param_band(self) -> None:
        model = build_q_network(
            (84, 84, 6),
            6,
            1e-4,
            vector_dim=59,
            reason_count=11,
            skill_count=8,
            model_variant=MODEL_VARIANT_IMPALA_MID,
        )
        n = model.count_params()
        self.assertGreaterEqual(n, 8_000_000)
        self.assertLessEqual(n, 20_000_000)

    def test_impala_large_bigger_than_mid(self) -> None:
        mid = build_q_network(
            (84, 84, 6),
            6,
            1e-4,
            vector_dim=59,
            reason_count=11,
            skill_count=8,
            model_variant=MODEL_VARIANT_IMPALA_MID,
        )
        large = build_q_network(
            (84, 84, 6),
            6,
            1e-4,
            vector_dim=59,
            reason_count=11,
            skill_count=8,
            model_variant=MODEL_VARIANT_IMPALA_LARGE,
        )
        self.assertGreater(large.count_params(), mid.count_params())

    def test_forward_hybrid_impala_mid(self) -> None:
        import numpy as np
        import tensorflow as tf

        model = build_q_network(
            (84, 84, 6),
            6,
            1e-4,
            vector_dim=59,
            reason_count=11,
            skill_count=8,
            model_variant=MODEL_VARIANT_IMPALA_MID,
        )
        visual = np.zeros((2, 84, 84, 6), dtype=np.float32)
        vector = np.zeros((2, 59), dtype=np.float32)
        out = model([visual, vector], training=False)
        if isinstance(out, (list, tuple)):
            q = out[0]
        else:
            q = out
        self.assertEqual(tuple(q.shape), (2, 6))

    def test_agent_defaults_to_impala_mid(self) -> None:
        config = AppConfig()
        config.training.device_mode = DeviceMode.CPU
        config.training.mixed_precision = False
        config.training.skills_enabled = True
        agent = DQNAgent(config)
        self.assertEqual(agent._model_variant, MODEL_VARIANT_IMPALA_MID)
        self.assertGreaterEqual(agent.online.count_params(), 8_000_000)


if __name__ == "__main__":
    unittest.main()

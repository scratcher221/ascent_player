"""Navigation target selection while falling."""

from __future__ import annotations

import unittest

from ascent_player.env.state_detector import FrameState, merge_agent_state


class FallingTargetTests(unittest.TestCase):
    def test_falling_prefers_below_over_best_landing_above(self) -> None:
        state = FrameState()
        payload = {
            "state": "playing",
            "canvasW": 640,
            "canvasH": 360,
            "cameraY": 0,
            "orb": {"x": 320, "worldY": 200, "vx": 0, "vy": -200, "energy": 50},
            "nearestPlatformBelow": {"dx": 0.25, "dy": 0.2, "width": 0.1, "wear": 0, "type": "neutral"},
            "nearestPlatformAbove": {"dx": -0.3, "dy": 0.15, "width": 0.1, "wear": 0, "type": "neutral"},
            # Above looks "better" to pickTarget (|dx|*2+|dy|), but must be ignored while falling.
            "bestLandingPlatform": {"dx": -0.3, "dy": 0.15, "type": "neutral"},
            "orbPhase": {"rising": False, "falling": True, "landingWindow": True, "airborne": True},
            "nearestBooster": {"dx": -0.5, "dy": 0.05, "type": "surge"},
        }
        out = merge_agent_state(state, payload)
        self.assertTrue(out.falling)
        self.assertAlmostEqual(out.target_dx or 0.0, 0.25, places=3)
        self.assertEqual(out.target_kind, "platform")

    def test_rising_can_use_best_landing(self) -> None:
        state = FrameState()
        payload = {
            "state": "playing",
            "canvasW": 640,
            "canvasH": 360,
            "cameraY": 0,
            "orb": {"x": 320, "worldY": 200, "vx": 0, "vy": 200, "energy": 50},
            "nearestPlatformBelow": {"dx": 0.25, "dy": 0.2, "width": 0.1, "wear": 0, "type": "neutral"},
            "bestLandingPlatform": {"dx": -0.1, "dy": 0.1, "type": "buy"},
            "orbPhase": {"rising": True, "falling": False, "landingWindow": False, "airborne": True},
        }
        out = merge_agent_state(state, payload)
        self.assertTrue(out.rising)
        self.assertAlmostEqual(out.target_dx or 0.0, -0.1, places=3)

    def test_near_pad_blocks_booster_hijack(self) -> None:
        state = FrameState()
        payload = {
            "state": "playing",
            "canvasW": 640,
            "canvasH": 360,
            "cameraY": 0,
            "orb": {"x": 320, "worldY": 200, "vx": 0, "vy": 50, "energy": 50},
            "nearestPlatformBelow": {"dx": 0.2, "dy": 0.2, "width": 0.1, "wear": 0, "type": "neutral"},
            "bestLandingPlatform": {"dx": 0.2, "dy": 0.2, "type": "neutral"},
            "orbPhase": {"rising": True, "falling": False, "landingWindow": False, "airborne": True},
            "nearestBooster": {"dx": -0.4, "dy": 0.05, "type": "drag"},
        }
        out = merge_agent_state(state, payload)
        self.assertAlmostEqual(out.target_dx or 0.0, 0.2, places=3)
        self.assertEqual(out.target_kind, "platform")



if __name__ == "__main__":
    unittest.main()

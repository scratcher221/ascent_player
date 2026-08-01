from __future__ import annotations

import importlib.util
import unittest

import numpy as np

_HAS_CV2 = importlib.util.find_spec("cv2") is not None

if _HAS_CV2:
    from ascent_player.agent.skills import LAND_BELOW, skill_to_index
    from ascent_player.config import AppConfig
    from ascent_player.demo.recorder import DemoRecorder
    from ascent_player.env.browser_backend import HudSnapshot
    from ascent_player.utils.preprocessing import FrameStack
else:
    LAND_BELOW = "land_below"

    def skill_to_index(_: str) -> int:
        return -1

    class HudSnapshot:  # pragma: no cover - test helper for skipped environments
        def __init__(self, **_: object) -> None:
            pass

    class FrameStack:  # pragma: no cover - test helper for skipped environments
        def __init__(self, _: int) -> None:
            pass


class _FakePage:
    async def evaluate(self, script: str) -> dict[str, bool]:
        return {
            "left": False,
            "right": False,
            "space": False,
            "installed": True,
        }


class _FakeBackend:
    def __init__(self) -> None:
        self.page = _FakePage()
        self._frame = np.zeros((64, 64, 3), dtype=np.uint8)
        self._payload = {
            "state": "playing",
            "orb": {"energy": 40.0, "reserve": 0.0, "combo": 0},
            "nearestPlatformBelow": {"dx": -0.2, "dy": 0.25, "width": 50.0, "type": "safe"},
            "orbPhase": {
                "falling": True,
                "rising": False,
                "landingWindow": False,
                "airborne": True,
            },
            "canBoost": True,
        }

    async def capture_turn(self, *, include_hud: bool = True) -> tuple[np.ndarray, HudSnapshot]:
        return self._frame.copy(), HudSnapshot(score=123, energy=0.4, reserve=0.0, can_boost=True)

    async def read_agent_state(self) -> dict:
        return dict(self._payload)

    async def text_content(self) -> str:
        return "SCORE 123"


class _FakeEnv:
    def __init__(self, frame_stack: int) -> None:
        self.frame_stack = FrameStack(frame_stack)


@unittest.skipUnless(_HAS_CV2, "OpenCV is required for recorder tests")
class DemoRecorderTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_step_uses_agent_state_for_skill_labels(self) -> None:
        config = AppConfig()
        backend = _FakeBackend()
        env = _FakeEnv(config.observation.frame_stack)
        recorder = DemoRecorder(config=config, backend=backend, env=env)

        await recorder.capture_step()
        await recorder.capture_step()

        self.assertEqual(len(recorder.transitions), 1)
        self.assertEqual(recorder.transitions[0].skill, skill_to_index(LAND_BELOW))


if __name__ == "__main__":
    unittest.main()

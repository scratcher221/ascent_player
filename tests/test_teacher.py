from __future__ import annotations

import asyncio
import unittest

from ascent_player.agent.teacher import RuleTeacher
from ascent_player.config import AppConfig
from ascent_player.env.state_detector import FrameState


class TeacherTests(unittest.TestCase):
    def test_teacher_steers_toward_platform(self) -> None:
        teacher = RuleTeacher()
        state = FrameState(
            nearest_platform_dx=-0.2,
            nearest_platform_dy=0.3,
            orb_vy=-0.4,
            can_boost=True,
            boost_level=0.8,
            agent_hook_ok=True,
        )
        action = teacher.act(state)
        self.assertIn(action, {1, 4})

    def test_teacher_sim_episode_runs(self) -> None:
        from ascent_player.env.sim_env import AscentSimEnv

        async def run() -> int:
            env = AscentSimEnv(AppConfig(), fast_mode=True)
            teacher = RuleTeacher()
            steps = 0
            state = await env.reset()
            for _ in range(200):
                fs = env._last_frame_state
                if fs is None:
                    break
                result = await env.step(teacher.act(fs))
                steps += 1
                state = result.state
                if result.done:
                    break
            await env.close()
            return steps

        steps = asyncio.run(run())
        self.assertGreater(steps, 20)


if __name__ == "__main__":
    unittest.main()

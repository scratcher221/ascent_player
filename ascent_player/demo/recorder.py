from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ascent_player.config import AppConfig
from ascent_player.demo.keyboard_probe import install_keyboard_probe, keys_to_action, read_keyboard_state
from ascent_player.demo.storage import DemoTransition, new_demo_path, save_demo
from ascent_player.env.browser_backend import BrowserBackend
from ascent_player.env.game_env import AscentGameEnv
from ascent_player.env.rewards import RewardTracker
from ascent_player.env.state_detector import (
    detect_from_frame,
    merge_dom_state,
    platform_mask_from_state,
)


@dataclass(slots=True)
class DemoRecorder:
    config: AppConfig
    backend: BrowserBackend
    env: AscentGameEnv
    reward_tracker: RewardTracker = field(init=False)
    transitions: list[DemoTransition] = field(default_factory=list)
    recording: bool = False
    _last_state: np.ndarray | None = None
    _last_action: int | None = None

    def __post_init__(self) -> None:
        self.reward_tracker = RewardTracker(self.config.reward)

    async def prepare(self) -> None:
        status = await self.backend.connect_auto()
        if not status.connected:
            raise RuntimeError(status.message)
        assert self.backend.page is not None
        await install_keyboard_probe(self.backend.page)
        await self.env.reset()
        self.reward_tracker.reset()
        self.transitions.clear()
        self._last_state = None
        self._last_action = None

    async def capture_step(self) -> tuple[np.ndarray, int, bool]:
        frame = await self.backend.canvas_screenshot()
        key_state = await read_keyboard_state(self.backend.page)
        action = keys_to_action(key_state)
        frame_state = detect_from_frame(frame)
        hud_score = await self.backend.read_game_score()
        if hud_score is not None:
            frame_state.score = hud_score
        if len(self.transitions) % max(1, self.config.browser.dom_poll_interval) == 0:
            body_text = await self.backend.text_content()
            frame_state = merge_dom_state(frame_state, body_text)

        from ascent_player.utils.preprocessing import build_observation

        observation = build_observation(
            frame,
            self.env.frame_stack,
            self.config.observation,
            frame_state.boost_level,
            platform_mask_from_state(frame_state, frame),
        )
        done = frame_state.game_over

        if self._last_state is not None and self._last_action is not None:
            reward = self.reward_tracker.compute(frame_state, self._last_action)
            self.transitions.append(
                DemoTransition(
                    state=self._last_state,
                    action=self._last_action,
                    reward=reward,
                    next_state=observation,
                    done=done,
                )
            )

        self._last_state = observation
        self._last_action = action
        return frame, action, done

    async def stop_and_save(self) -> Path:
        self.recording = False
        if self._last_state is not None and self._last_action is not None:
            self.transitions.append(
                DemoTransition(
                    state=self._last_state,
                    action=self._last_action,
                    reward=0.0,
                    next_state=self._last_state,
                    done=True,
                )
            )
        path = new_demo_path(self.config)
        return save_demo(path, self.transitions)

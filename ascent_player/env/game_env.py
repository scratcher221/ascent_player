from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ascent_player.config import AppConfig
from ascent_player.env.browser_backend import BrowserBackend
from ascent_player.env.rewards import RewardTracker
from ascent_player.env.state_detector import FrameState, detect_from_frame, merge_dom_state
from ascent_player.utils.preprocessing import FrameStack, preprocess_frame


ACTION_LABELS = {
    0: "noop",
    1: "left",
    2: "right",
    3: "jump",
    4: "left+jump",
    5: "right+jump",
}


@dataclass(slots=True)
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    raw_frame: np.ndarray
    frame_state: FrameState


class AscentGameEnv:
    def __init__(self, config: AppConfig, backend: BrowserBackend) -> None:
        self.config = config
        self.backend = backend
        self.reward_tracker = RewardTracker(config.reward)
        self.frame_stack = FrameStack(config.observation.frame_stack)
        self.held_keys: set[str] = set()
        self.last_raw_frame: np.ndarray | None = None
        self.recent_states: deque[FrameState] = deque(maxlen=8)

    async def connect(self) -> None:
        await self.backend.connect_auto()

    async def reset(self) -> np.ndarray:
        self.reward_tracker.reset()
        self.frame_stack.clear()
        self.recent_states.clear()
        await self._release_all()
        await self.backend.force_open_game()
        await self._start_or_restart()
        frame = await self._capture_frame()
        processed = preprocess_frame(frame, self.config.observation)
        self.frame_stack.reset(processed)
        return self.frame_stack.array()

    async def step(self, action: int) -> StepResult:
        await self._apply_action(action)
        await self.backend.wait_ms(self._step_ms())
        frame = await self._capture_frame()
        frame_state = await self._detect_state(frame)
        self.recent_states.append(frame_state)
        processed = preprocess_frame(frame, self.config.observation)
        state = self.frame_stack.append(processed)
        reward = self.reward_tracker.compute(frame_state, action)
        done = frame_state.game_over
        if done:
            await self._release_all()
        return StepResult(
            state=state,
            reward=reward,
            done=done,
            raw_frame=frame,
            frame_state=frame_state,
        )

    async def close(self) -> None:
        await self._release_all()
        await self.backend.stop()

    async def _capture_frame(self) -> np.ndarray:
        frame = await self.backend.canvas_screenshot()
        self.last_raw_frame = frame
        return frame

    async def _detect_state(self, frame: np.ndarray) -> FrameState:
        state = detect_from_frame(frame)
        body_text = await self.backend.text_content()
        return merge_dom_state(state, body_text)

    async def _start_or_restart(self) -> None:
        body = (await self.backend.text_content()).upper()
        if "BACK TO EARTH" in body or "FELL" in body:
            clicked = await self.backend.click_text("ASCEND AGAIN", timeout=2_000)
            if clicked:
                await self.backend.wait_ms(500)

        body = (await self.backend.text_content()).upper()
        if "START THE ASCENT" in body:
            clicked = await self.backend.click_text("START THE ASCENT", timeout=2_000)
            if not clicked:
                await self.backend.press("Space")
            await self.backend.wait_ms(800)

        body = (await self.backend.text_content()).upper()
        if "PICK 1 ULTI" in body or "LAUNCH" in body:
            clicked = await self.backend.click_text("LAUNCH", timeout=2_000)
            if not clicked:
                await self.backend.press("Space")
            await self.backend.wait_ms(800)

        if not await self.backend.has_canvas():
            await self.backend.force_open_game()

    async def _apply_action(self, action: int) -> None:
        target_keys: set[str] = set()
        if action in (1, 4):
            target_keys.add("KeyA")
        if action in (2, 5):
            target_keys.add("KeyD")

        for key in list(self.held_keys - target_keys):
            await self.backend.key_up(key)
            self.held_keys.remove(key)
        for key in target_keys - self.held_keys:
            await self.backend.key_down(key)
            self.held_keys.add(key)

        if action in (3, 4, 5):
            await self.backend.press("Space")

    async def _release_all(self) -> None:
        for key in list(self.held_keys):
            try:
                await self.backend.key_up(key)
            finally:
                self.held_keys.discard(key)

    def _step_ms(self) -> int:
        frame_ms = 1000 / 60
        return max(16, int(frame_ms * self.config.training.frame_skip))

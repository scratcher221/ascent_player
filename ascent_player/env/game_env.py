from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ascent_player.config import AppConfig
from ascent_player.env.browser_backend import BrowserBackend, HudSnapshot
from ascent_player.env.reward_factory import create_reward_tracker
from ascent_player.env.state_detector import (
    FrameState,
    apply_hud_boost,
    detect_from_frame,
    mask_jump_action,
    merge_agent_state,
    merge_dom_state,
    platform_mask_from_agent_payload,
    platform_mask_from_state,
)
from ascent_player.env.vector_obs import attach_vector
from ascent_player.utils.preprocessing import (
    FrameStack,
    append_observation_channels,
    build_observation,
    preprocess_frame,
)


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
    state: np.ndarray | tuple[np.ndarray, np.ndarray]
    reward: float
    done: bool
    raw_frame: np.ndarray
    frame_state: FrameState


class AscentGameEnv:
    def __init__(self, config: AppConfig, backend: BrowserBackend) -> None:
        self.config = config
        self.backend = backend
        self.reward_tracker = create_reward_tracker(config)
        self.frame_stack = FrameStack(config.observation.frame_stack)
        self.held_keys: set[str] = set()
        self.last_raw_frame: np.ndarray | None = None
        self.recent_states: deque[FrameState] = deque(maxlen=8)
        self._step_count = 0
        self._last_frame_state: FrameState | None = None
        self._prev_bounces = 0

    @property
    def can_boost(self) -> bool:
        if self._last_frame_state is None:
            return True
        return self._last_frame_state.can_boost

    @property
    def boost_level(self) -> float:
        if self._last_frame_state is None:
            return 1.0
        return self._last_frame_state.boost_level

    def _finalize_observation(
        self,
        visual: np.ndarray,
        frame_state: FrameState,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if self.config.observation.include_vector_state:
            _, vector = attach_vector(visual, frame_state, self.config.observation)
            return (visual, vector)
        return visual

    async def connect(self) -> None:
        await self.backend.connect_auto()

    async def reset(self):
        self.reward_tracker.reset()
        self.frame_stack.clear()
        self.recent_states.clear()
        self._step_count = 0
        self._last_frame_state = None
        self._prev_bounces = 0
        await self._release_all()
        await self.backend.force_open_game()
        if self.config.browser.agent_mode:
            await self.backend.ensure_agent_mode()
            await self.backend.select_training_tier(
                self.config.mechanics_curriculum.training_tier_index
            )
        await self._start_or_restart()
        frame, hud = await self._capture_turn()
        frame_state = await self._detect_state(frame, hud)
        self._last_frame_state = frame_state
        self._prev_bounces = frame_state.bounces
        gray = preprocess_frame(frame, self.config.observation)
        self.frame_stack.reset(gray)
        visual = append_observation_channels(
            self.frame_stack.array(),
            self.config.observation,
            frame_state.boost_level,
            platform_mask_from_state(frame_state, frame),
        )
        return self._finalize_observation(visual, frame_state)

    async def step(self, action: int) -> StepResult:
        action = mask_jump_action(action, self.can_boost)
        await self._apply_action(action)

        frames = max(1, self.config.training.frame_skip)
        frame_ms = self._frame_ms()
        frame: np.ndarray | None = None
        hud = HudSnapshot()
        for index in range(frames):
            await self.backend.wait_ms(frame_ms)
            if index == frames - 1:
                frame, hud = await self._capture_turn()

        assert frame is not None
        frame_state = await self._detect_state(frame, hud)
        frame_state.platform_landed = frame_state.bounces > self._prev_bounces
        self._prev_bounces = frame_state.bounces
        self._last_frame_state = frame_state
        self.recent_states.append(frame_state)
        visual = build_observation(
            frame,
            self.frame_stack,
            self.config.observation,
            frame_state.boost_level,
            platform_mask_from_state(frame_state, frame),
        )
        state = self._finalize_observation(visual, frame_state)
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

    async def _capture_turn(self) -> tuple[np.ndarray, HudSnapshot]:
        frame, hud = await self.backend.capture_turn(include_hud=True)
        self.last_raw_frame = frame
        return frame, hud

    async def _detect_state(
        self,
        frame: np.ndarray,
        hud: HudSnapshot | None = None,
    ) -> FrameState:
        agent_payload = await self.backend.read_agent_state()
        if agent_payload:
            state = merge_agent_state(FrameState(), agent_payload)
            height, width = frame.shape[:2]
            state.platform_mask = platform_mask_from_agent_payload(
                agent_payload,
                width,
                height,
            )
        else:
            state = detect_from_frame(
                frame,
                platform_only=self.config.training.target_platform_only,
            )
        apply_hud_boost(
            state,
            hud.energy if hud is not None else None,
            hud.reserve if hud is not None else None,
            hud.can_boost if hud is not None else None,
            min_energy=self.config.mechanics_reward.boost_min_energy,
        )
        if hud is not None and hud.score is not None and not state.agent_hook_ok:
            state.score = hud.score
        if hud is not None and not state.agent_hook_ok:
            state.combo = hud.combo
            state.streak = hud.streak
            state.score_multiplier = hud.multiplier
            if hud.fell:
                state.game_over = True
            if hud.in_menu:
                state.in_menu = True
        interval = max(1, self.config.browser.dom_poll_interval)
        if self._step_count % interval == 0:
            body_text = await self.backend.text_content()
            state = merge_dom_state(state, body_text)
        self._step_count += 1
        return state

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
        if "PICK 1 ULTI" in body:
            clicked = await self.backend.click_text("LAUNCH", timeout=2_000)
            if not clicked:
                card = await self.backend.click_selector(".ulti-card", timeout=1_500)
                if not card:
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

        if action in (3, 4, 5) and self.can_boost:
            await self.backend.press("Space")

    async def _release_all(self) -> None:
        for key in list(self.held_keys):
            try:
                await self.backend.key_up(key)
            finally:
                self.held_keys.discard(key)

    def _frame_ms(self) -> int:
        fps = max(30.0, float(self.config.training.game_fps))
        return max(8, int(round(1000.0 / fps)))


def build_platform_mask_fallback(frame: np.ndarray, state: FrameState) -> np.ndarray:
    from ascent_player.env.platform_detector import build_platform_mask, detect_platforms

    if state.platform_mask is not None:
        return state.platform_mask
    return build_platform_mask(frame, detect_platforms(frame))

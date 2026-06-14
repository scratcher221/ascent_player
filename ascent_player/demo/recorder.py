from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ascent_player.config import AppConfig
from ascent_player.demo.keyboard_probe import install_keyboard_probe, keys_to_action, read_keyboard_state
from ascent_player.demo.storage import DemoTransition, new_demo_path, save_demo
from ascent_player.env.browser_backend import BrowserBackend
from ascent_player.env.game_env import AscentGameEnv
from ascent_player.env.rewards import RewardTracker
from ascent_player.env.state_detector import (
    apply_hud_boost,
    detect_from_frame,
    mask_jump_action,
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
    episode_id: int = 0
    _last_state: np.ndarray | None = None
    _last_action: int | None = None
    _last_score: float = 0.0

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
        self.episode_id = 0
        self._last_state = None
        self._last_action = None
        self._last_score = 0.0

    async def capture_step(self) -> tuple[np.ndarray, int, bool]:
        frame, hud = await self.backend.capture_turn(include_hud=True)
        key_state = await read_keyboard_state(self.backend.page)
        action = keys_to_action(key_state)
        frame_state = detect_from_frame(frame)
        apply_hud_boost(
            frame_state,
            hud.energy,
            hud.reserve,
            hud.can_boost,
            min_energy=self.config.reward.boost_min_energy,
        )
        if hud.score is not None:
            frame_state.score = hud.score
        if hud.fell:
            frame_state.game_over = True
        if hud.in_menu:
            frame_state.in_menu = True
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
        action = mask_jump_action(action, frame_state.can_boost)
        done = frame_state.game_over
        score = float(frame_state.score or 0.0)

        if self._last_state is not None and self._last_action is not None:
            reward = self.reward_tracker.compute(frame_state, self._last_action)
            self.transitions.append(
                DemoTransition(
                    state=self._last_state,
                    action=self._last_action,
                    reward=reward,
                    next_state=observation,
                    done=done,
                    score=self._last_score,
                    episode_id=self.episode_id,
                )
            )

        self._last_state = observation
        self._last_action = action
        self._last_score = score
        return frame, action, done

    def on_episode_end(self) -> None:
        self.episode_id += 1
        self.reward_tracker.reset()
        self._last_state = None
        self._last_action = None
        self._last_score = 0.0

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
                    score=self._last_score,
                    episode_id=self.episode_id,
                )
            )
        path = new_demo_path(self.config)
        return save_demo(path, self.transitions)

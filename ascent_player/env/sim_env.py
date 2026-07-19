from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ascent_player.config import AppConfig
from ascent_player.env.game_env import ACTION_LABELS, StepResult
from ascent_player.env.platform_detector import Platform
from ascent_player.env.fast_sim_obs import fast_build_observation
from ascent_player.env.sim_state_adapter import build_frame_state_from_world
from ascent_player.env.reward_factory import create_reward_tracker
from ascent_player.env.sim_physics import SimBooster, SimPlatform, SimPhysicsConfig, SimWorld
from ascent_player.env.state_detector import FrameState, mask_jump_action
from ascent_player.env.vector_obs import attach_vector
from ascent_player.utils.preprocessing import (
    FrameStack,
    append_observation_channels,
    build_observation,
    preprocess_frame,
    apply_jpeg_augment,
)


@dataclass(slots=True)
class SimRenderConfig:
    width: int = 640
    height: int = 360


def _to_detector_platform(platform: SimPlatform, camera_y: float) -> Platform:
    return Platform(
        cx=platform.cx,
        cy=platform.cy - camera_y,
        width=platform.width,
        height=platform.height,
        is_hazard=platform.is_hazard,
    )


def render_sim_frame(world: SimWorld) -> np.ndarray:
    """RGB frame tuned so grayscale preprocess ≈ browser Genesis canvas.

    Tier-1 colours from tiers.js / game.js:
      bg #000000–#070707, neutral platforms #999999, sell #ff4d6d,
      orb mid #f2f2f2 with soft white bloom (halo).
    """
    cfg = world.config
    h, w = cfg.height, cfg.width
    # Vertical void gradient (tiers.js GENESIS bg / bg1)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    tone = (7.0 + 5.0 * yy).astype(np.uint8)
    frame[:, :, 0] = tone
    frame[:, :, 1] = tone
    frame[:, :, 2] = tone

    camera_y = world.camera_y
    for platform in world.platforms:
        # wear fades like platformOpacity() in platform-rules.js
        wear = platform.receptions / max(platform.bounce_limit, 1)
        alpha = max(0.2, 1.0 - wear * 0.8)
        if platform.is_hazard:
            base = np.array([255, 77, 109], dtype=np.float32)  # #ff4d6d sell
        else:
            base = np.array([153, 153, 153], dtype=np.float32)  # #999999 platN
        color = tuple(int(c * alpha) for c in base)
        x1 = int(platform.cx - platform.width / 2)
        x2 = int(platform.cx + platform.width / 2)
        y1 = int(platform.cy - camera_y - platform.height / 2)
        y2 = int(platform.cy - camera_y + platform.height / 2)
        if y2 < 0 or y1 > h:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)
        # Top highlight strip (game drawPlatforms tier≥2; helps edge contrast in gray)
        if not platform.is_hazard and y1 >= 0:
            hi = tuple(min(255, int(c + 40 * alpha)) for c in color)
            cv2.rectangle(frame, (x1, y1), (x2, min(y1 + 2, y2)), hi, thickness=-1)

    for booster in world.boosters:
        if booster.used:
            continue
        # surge #ffdd88, stream green-cyan, drag red wall
        if booster.type == "surge":
            color = (255, 221, 136)
        elif booster.type == "stream":
            color = (80, 200, 160)
        else:
            color = (255, 77, 109)
        x1 = int(booster.cx - booster.width / 2)
        x2 = int(booster.cx + booster.width / 2)
        y1 = int(booster.cy - camera_y - booster.height / 2)
        y2 = int(booster.cy - camera_y + booster.height / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)

    orb_x = int(world.ball.x)
    orb_y = int(world.ball.y - camera_y)
    radius = max(2, int(cfg.orb_radius))
    # Soft bloom then bright body (Genesis orb mid/bloom → high gray after preprocess)
    cv2.circle(frame, (orb_x, orb_y), radius + 6, (48, 48, 48), thickness=-1)
    cv2.circle(frame, (orb_x, orb_y), radius + 3, (180, 180, 180), thickness=-1)
    cv2.circle(frame, (orb_x, orb_y), radius, (242, 242, 242), thickness=-1)
    # Specular hot-spot
    cv2.circle(
        frame,
        (orb_x - radius // 3, orb_y - radius // 3),
        max(1, radius // 4),
        (255, 255, 255),
        thickness=-1,
    )

    # Compact energy tick (browser HUD is separate; keep a thin cue for boost channel)
    bar_x = int(w * 0.02)
    bar_bottom = int(h * 0.90)
    bar_top = int(h * 0.62)
    bar_h = bar_bottom - bar_top
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 6, bar_bottom), (28, 28, 28), -1)
    fill_h = int(bar_h * world.boost_level)
    if fill_h > 0:
        color = (90, 200, 140) if world.boost_level > 0.3 else (90, 90, 200)
        cv2.rectangle(
            frame,
            (bar_x + 1, bar_bottom - fill_h),
            (bar_x + 5, bar_bottom - 1),
            color,
            -1,
        )
    return frame


def build_platform_mask_from_list(
    platforms: list[Platform],
    shape: tuple[int, int],
    camera_y: float,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for platform in platforms:
        if platform.is_hazard:
            continue
        half_w = max(2, int(platform.width / 2))
        half_h = max(1, int(platform.height / 2))
        x1 = max(0, int(platform.cx - half_w))
        x2 = min(width, int(platform.cx + half_w))
        y1 = max(0, int(platform.cy - camera_y - half_h))
        y2 = min(height, int(platform.cy - camera_y + half_h))
        mask[y1:y2, x1:x2] = 255
    return mask


def frame_state_from_world(world: SimWorld, frame_rgb: np.ndarray) -> FrameState:
    platforms = [_to_detector_platform(platform, world.camera_y) for platform in world.platforms]
    return build_frame_state_from_world(
        world,
        platform_mask=build_platform_mask_from_list(
            platforms,
            frame_rgb.shape[:2],
            0.0,
        ),
    )


class AscentSimEnv:
    def __init__(
        self,
        config: AppConfig,
        *,
        fast_mode: bool | None = None,
        env_index: int = 0,
    ) -> None:
        self.config = config
        self.fast_mode = (
            config.training.sim_fast_observations
            if fast_mode is None
            else fast_mode
        )
        self.reward_tracker = create_reward_tracker(config)
        self.frame_stack = FrameStack(config.observation.frame_stack)
        self.world = SimWorld(
            SimPhysicsConfig(
                width=640,
                height=360,
                seed=config.training.baseline_episodes + env_index * 7919,
            )
        )
        self.held_left = False
        self.held_right = False
        self._last_frame_state: FrameState | None = None
        self._current_state = None

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

    @property
    def current_state(self):
        if self._current_state is None:
            raise RuntimeError("Environment has not been reset.")
        return self._current_state

    async def connect(self) -> None:
        return None

    def reset_sync(self, start_height: float = 0.0):
        self.reward_tracker.reset()
        self.frame_stack.clear()
        self.held_left = False
        self.held_right = False
        self.world.reset(start_height=start_height)
        self._current_state = self._build_state()
        return self._current_state

    async def reset(self, start_height: float = 0.0):
        return self.reset_sync(start_height=start_height)

    def _finalize_observation(self, visual: np.ndarray, frame_state: FrameState):
        if self.config.observation.include_vector_state:
            _, vector = attach_vector(visual, frame_state, self.config.observation)
            return (visual, vector)
        return visual

    def _frame_state_fast(self) -> FrameState:
        return build_frame_state_from_world(self.world)

    def _render_frame(self) -> np.ndarray:
        frame = render_sim_frame(self.world)
        if self.config.training.sim_jpeg_augment:
            frame = apply_jpeg_augment(
                frame,
                quality=self.config.training.sim_jpeg_quality,
            )
        return frame

    def _build_state(self):
        if self.fast_mode:
            frame_state = self._frame_state_fast()
            self._last_frame_state = frame_state
            visual = fast_build_observation(
                self.world,
                self.frame_stack,
                self.config.observation,
            )
            return self._finalize_observation(visual, frame_state)

        frame = self._render_frame()
        frame_state = frame_state_from_world(self.world, frame)
        self._last_frame_state = frame_state
        gray = preprocess_frame(frame, self.config.observation)
        if not self.frame_stack.frames:
            self.frame_stack.reset(gray)
        else:
            self.frame_stack.append(gray)
        visual = append_observation_channels(
            self.frame_stack.array(),
            self.config.observation,
            frame_state.boost_level,
            frame_state.platform_mask,
        )
        return self._finalize_observation(visual, frame_state)

    def step_sync(self, action: int) -> StepResult:
        action = mask_jump_action(action, self.can_boost)
        self._apply_action(action)
        jump_once = action in (3, 4, 5) and self.can_boost
        dt = self._frame_seconds()
        frames = max(1, self.config.training.frame_skip)
        done = False
        for index in range(frames):
            done = self.world.step(
                move_left=self.held_left,
                move_right=self.held_right,
                jump=jump_once and index == 0,
                dt=dt,
            )
            if done:
                break

        frame = self._render_frame()
        frame_state = frame_state_from_world(self.world, frame)
        frame_state.game_over = done
        self._last_frame_state = frame_state
        if self.fast_mode:
            visual = fast_build_observation(
                self.world,
                self.frame_stack,
                self.config.observation,
            )
            state = self._finalize_observation(visual, frame_state)
            raw_frame = np.zeros((1, 1, 3), dtype=np.uint8)
        else:
            state = self._build_state()
            raw_frame = frame
        reward = self.reward_tracker.compute(frame_state, action)
        self._current_state = state
        return StepResult(
            state=state,
            reward=reward,
            done=done,
            raw_frame=raw_frame,
            frame_state=frame_state,
        )

    async def step(self, action: int) -> StepResult:
        return self.step_sync(action)

    async def close(self) -> None:
        return None

    def _apply_action(self, action: int) -> None:
        self.held_left = action in (1, 4)
        self.held_right = action in (2, 5)

    def _frame_seconds(self) -> float:
        fps = max(30.0, float(self.config.training.game_fps))
        return 1.0 / fps


async def calibrate_sim_physics(config: AppConfig, episodes: int = 10) -> dict[str, float]:
    import random

    env = AscentSimEnv(config, fast_mode=False)
    random_scores: list[float] = []
    random_lengths: list[int] = []
    climb_scores: list[float] = []

    for _ in range(episodes):
        steps = 0
        max_score = 0.0
        await env.reset()
        while steps < 800:
            action = random.randrange(config.action_count)
            result = await env.step(action)
            steps += 1
            if result.frame_state.score is not None:
                max_score = max(max_score, float(result.frame_state.score))
            if result.done or steps >= 800:
                random_scores.append(max_score)
                random_lengths.append(steps)
                break

    for _ in range(episodes):
        steps = 0
        max_score = 0.0
        await env.reset()
        while steps < 800:
            action = random.choice([0, 2, 3, 5, 2, 3, 5])
            result = await env.step(action)
            steps += 1
            if result.frame_state.score is not None:
                max_score = max(max_score, float(result.frame_state.score))
            if result.done or steps >= 800:
                climb_scores.append(max_score)
                break

    await env.close()
    all_scores = random_scores + climb_scores
    sorted_scores = sorted(all_scores)
    p90 = sorted_scores[int(len(sorted_scores) * 0.9)] if sorted_scores else 0.0
    return {
        "episodes": float(len(random_scores)),
        "mean_score": float(sum(random_scores) / len(random_scores))
        if random_scores
        else 0.0,
        "mean_length": float(sum(random_lengths) / len(random_lengths))
        if random_lengths
        else 0.0,
        "climb_mean_score": float(sum(climb_scores) / len(climb_scores))
        if climb_scores
        else 0.0,
        "max_score": float(max(all_scores)) if all_scores else 0.0,
        "p90_score": float(p90),
    }

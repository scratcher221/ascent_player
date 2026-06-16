from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ascent_player.config import AppConfig, RewardConfig
from ascent_player.env.game_env import ACTION_LABELS, StepResult
from ascent_player.env.platform_detector import Platform, nearest_safe_platform
from ascent_player.env.fast_sim_obs import fast_build_observation
from ascent_player.env.reward_factory import create_reward_tracker
from ascent_player.env.sim_physics import SimBooster, SimPlatform, SimPhysicsConfig, SimWorld
from ascent_player.env.state_detector import FrameState, mask_jump_action
from ascent_player.env.vector_obs import attach_vector
from ascent_player.utils.preprocessing import (
    FrameStack,
    append_observation_channels,
    build_observation,
    preprocess_frame,
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
    cfg = world.config
    frame = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)
    frame[:] = (5, 9, 9)

    camera_y = world.camera_y
    for platform in world.platforms:
        color = (180, 70, 70) if platform.is_hazard else (120, 120, 120)
        x1 = int(platform.cx - platform.width / 2)
        x2 = int(platform.cx + platform.width / 2)
        y1 = int(platform.cy - camera_y - platform.height / 2)
        y2 = int(platform.cy - camera_y + platform.height / 2)
        if y2 < 0 or y1 > cfg.height:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)

    for booster in world.boosters:
        if booster.used:
            continue
        color = (80, 220, 120) if booster.type == "surge" else (120, 220, 80) if booster.type == "stream" else (80, 80, 220)
        x1 = int(booster.cx - booster.width / 2)
        x2 = int(booster.cx + booster.width / 2)
        y1 = int(booster.cy - camera_y - booster.height / 2)
        y2 = int(booster.cy - camera_y + booster.height / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)

    orb_x = int(world.ball.x)
    orb_y = int(world.ball.y - camera_y)
    radius = int(cfg.orb_radius)
    cv2.circle(frame, (orb_x, orb_y), radius + 4, (0, 220, 220), thickness=-1)
    cv2.circle(frame, (orb_x, orb_y), radius, (0, 255, 255), thickness=-1)

    bar_x = int(cfg.width * 0.02)
    bar_bottom = int(cfg.height * 0.88)
    bar_top = int(cfg.height * 0.58)
    bar_h = bar_bottom - bar_top
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 8, bar_bottom), (40, 40, 40), -1)
    fill_h = int(bar_h * world.boost_level)
    if fill_h > 0:
        color = (80, 220, 120) if world.boost_level > 0.3 else (80, 80, 220)
        cv2.rectangle(
            frame,
            (bar_x + 1, bar_bottom - fill_h),
            (bar_x + 7, bar_bottom - 1),
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
    ball = world.ball
    screen_y = ball.y - world.camera_y
    platforms = [_to_detector_platform(platform, world.camera_y) for platform in world.platforms]
    platform_dx, platform_dy = nearest_safe_platform(
        ball.x,
        screen_y,
        platforms,
        frame_rgb.shape,
    )
    ndx, ndy, wear, pwidth = world.nearest_platform_below()
    booster_dx = booster_dy = None
    booster_type = None
    best_dist = float("inf")
    for booster in world.boosters:
        if booster.used:
            continue
        dx = (booster.cx - ball.x) / world.config.width
        dy = (booster.cy - world.camera_y - screen_y) / world.config.height
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            booster_dx = dx
            booster_dy = dy
            booster_type = booster.type

    return FrameState(
        orb_x=ball.x,
        orb_y=screen_y / max(world.config.height, 1.0),
        orb_vx=ball.vx / 575.0,
        orb_vy=ball.vy / 1500.0,
        score=world.score,
        boost_level=world.boost_level,
        can_boost=world.can_boost,
        nearest_platform_dx=ndx,
        nearest_platform_dy=ndy,
        nearest_platform_width=pwidth,
        platform_wear=wear,
        booster_dx=booster_dx,
        booster_dy=booster_dy,
        booster_type=booster_type,
        combo=world.combo,
        score_multiplier=world.score_multiplier,
        bonus=world.bonus,
        bank_style=world.bank_style,
        height=world.height,
        bounces=world.bounces,
        canvas_w=float(world.config.width),
        canvas_h=float(world.config.height),
        agent_hook_ok=True,
        platform_landed=world.platform_landed,
        platform_mask=build_platform_mask_from_list(
            platforms,
            frame_rgb.shape[:2],
            0.0,
        ),
        game_over=False,
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

    def reset_sync(self):
        self.reward_tracker.reset()
        self.frame_stack.clear()
        self.held_left = False
        self.held_right = False
        self.world.reset()
        self._current_state = self._build_state()
        return self._current_state

    async def reset(self):
        return self.reset_sync()

    def _finalize_observation(self, visual: np.ndarray, frame_state: FrameState):
        if self.config.observation.include_vector_state:
            _, vector = attach_vector(visual, frame_state, self.config.observation)
            return (visual, vector)
        return visual

    def _frame_state_fast(self) -> FrameState:
        ball = self.world.ball
        screen_y = ball.y - self.world.camera_y
        ndx, ndy, wear, pwidth = self.world.nearest_platform_below()
        return FrameState(
            orb_x=ball.x / self.world.config.width,
            orb_y=screen_y / max(self.world.config.height, 1.0),
            orb_vx=ball.vx / 575.0,
            orb_vy=ball.vy / 1500.0,
            score=self.world.score,
            boost_level=self.world.boost_level,
            can_boost=self.world.can_boost,
            nearest_platform_dx=ndx,
            nearest_platform_dy=ndy,
            nearest_platform_width=pwidth,
            platform_wear=wear,
            combo=self.world.combo,
            score_multiplier=self.world.score_multiplier,
            bonus=self.world.bonus,
            bank_style=self.world.bank_style,
            height=self.world.height,
            bounces=self.world.bounces,
            canvas_w=float(self.world.config.width),
            canvas_h=float(self.world.config.height),
            agent_hook_ok=True,
            platform_landed=self.world.platform_landed,
            booster_type=self.world.booster_collected,
            game_over=False,
        )

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

        frame = render_sim_frame(self.world)
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

        frame = render_sim_frame(self.world)
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

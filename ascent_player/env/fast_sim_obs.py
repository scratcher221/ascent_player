from __future__ import annotations

import numpy as np

from ascent_player.config import ObservationConfig
from ascent_player.env.sim_physics import SimWorld
from ascent_player.utils.preprocessing import FrameStack, append_observation_channels


def _obs_scale(world: SimWorld, config: ObservationConfig) -> tuple[float, float]:
    return config.width / world.config.width, config.height / world.config.height


def fast_platform_mask(world: SimWorld, config: ObservationConfig) -> np.ndarray:
    height, width = config.height, config.width
    mask = np.zeros((height, width), dtype=np.uint8)
    scale_x, scale_y = _obs_scale(world, config)
    camera_y = world.camera_y
    for platform in world.platforms:
        if platform.is_hazard:
            continue
        half_w = max(1, int(platform.width * scale_x / 2))
        half_h = max(1, int(platform.height * scale_y / 2))
        cx = int(platform.cx * scale_x)
        cy = int((platform.cy - camera_y) * scale_y)
        x1 = max(0, cx - half_w)
        x2 = min(width, cx + half_w)
        y1 = max(0, cy - half_h)
        y2 = min(height, cy + half_h)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def fast_gray_frame(world: SimWorld, config: ObservationConfig) -> np.ndarray:
    height, width = config.height, config.width
    frame = np.zeros((height, width), dtype=np.float32)
    scale_x, scale_y = _obs_scale(world, config)
    camera_y = world.camera_y

    for platform in world.platforms:
        color = 0.35 if platform.is_hazard else 0.55
        x1 = int((platform.cx - platform.width / 2) * scale_x)
        x2 = int((platform.cx + platform.width / 2) * scale_x)
        y1 = int((platform.cy - camera_y - platform.height / 2) * scale_y)
        y2 = int((platform.cy - camera_y + platform.height / 2) * scale_y)
        if y2 < 0 or y1 >= height:
            continue
        y1 = max(0, y1)
        y2 = min(height, y2)
        x1 = max(0, x1)
        x2 = min(width, x2)
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = color

    orb_x = int(world.ball.x * scale_x)
    orb_y = int((world.ball.y - camera_y) * scale_y)
    radius = max(1, int(world.config.orb_radius * scale_x))
    y1 = max(0, orb_y - radius)
    y2 = min(height, orb_y + radius + 1)
    x1 = max(0, orb_x - radius)
    x2 = min(width, orb_x + radius + 1)
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] = 1.0
    return frame


def fast_build_observation(
    world: SimWorld,
    frame_stack: FrameStack,
    config: ObservationConfig,
) -> np.ndarray:
    gray = fast_gray_frame(world, config)
    stacked = frame_stack.append(gray)
    platform_mask = fast_platform_mask(world, config)
    return append_observation_channels(
        stacked,
        config,
        world.boost_level,
        platform_mask,
    )

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class Platform:
    cx: float
    cy: float
    width: float
    height: float
    is_hazard: bool = False


def detect_platforms(frame_rgb: np.ndarray) -> list[Platform]:
    """Find horizontal platform bars in the gameplay area."""
    if frame_rgb.size == 0:
        return []

    height, width = frame_rgb.shape[:2]
    crop = frame_rgb[
        int(height * 0.1) : int(height * 0.86),
        int(width * 0.07) : int(width * 0.93),
    ]
    if crop.size == 0:
        return []

    offset_x = int(width * 0.07)
    offset_y = int(height * 0.1)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

    # Neutral gray platform bars.
    gray_mask = cv2.inRange(hsv, np.array([0, 0, 55]), np.array([180, 55, 175]))
    # Red hatched resistance bars.
    red_mask = cv2.inRange(hsv, np.array([0, 70, 60]), np.array([12, 255, 255]))
    red_mask |= cv2.inRange(hsv, np.array([165, 70, 60]), np.array([180, 255, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (28, 4))
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    platforms: list[Platform] = []
    for mask, is_hazard in ((gray_mask, False), (red_mask, True)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 28 or h < 2 or w / max(h, 1) < 4.0:
                continue
            platforms.append(
                Platform(
                    cx=float(offset_x + x + w / 2),
                    cy=float(offset_y + y + h / 2),
                    width=float(w),
                    height=float(h),
                    is_hazard=is_hazard,
                )
            )
    return platforms


def build_platform_mask(frame_rgb: np.ndarray, platforms: list[Platform] | None = None) -> np.ndarray:
    """Binary mask of safe platforms for CNN input."""
    if frame_rgb.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    height, width = frame_rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    items = platforms if platforms is not None else detect_platforms(frame_rgb)
    for platform in items:
        if platform.is_hazard:
            continue
        half_w = max(2, int(platform.width / 2))
        half_h = max(1, int(platform.height / 2))
        x1 = max(0, int(platform.cx - half_w))
        x2 = min(width, int(platform.cx + half_w))
        y1 = max(0, int(platform.cy - half_h))
        y2 = min(height, int(platform.cy + half_h))
        mask[y1:y2, x1:x2] = 255
    return mask


def nearest_safe_platform(
    orb_x: float | None,
    orb_y: float | None,
    platforms: list[Platform],
    frame_shape: tuple[int, int],
) -> tuple[float | None, float | None]:
    """Return normalized dx/dy to the best platform target below/ahead of the orb."""
    if orb_x is None or orb_y is None or not platforms:
        return None, None

    height, width = frame_shape[:2]
    safe = [platform for platform in platforms if not platform.is_hazard]
    if not safe:
        return None, None

    def sort_key(platform: Platform) -> tuple[float, float]:
        dy = platform.cy - orb_y
        dx = abs(platform.cx - orb_x)
        if dy >= -20:
            return (max(0.0, dy), dx)
        return (9999.0, dx)

    target = min(safe, key=sort_key)
    dx = (target.cx - orb_x) / max(width, 1)
    dy = (target.cy - orb_y) / max(height, 1)
    return float(np.clip(dx, -1.0, 1.0)), float(np.clip(dy, -1.0, 1.0))

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ascent_player.env.platform_detector import Platform, nearest_safe_platform


@dataclass(slots=True)
class ScreenTarget:
    cx: float
    cy: float
    kind: str  # platform, yellow_orb, green_booster
    width: float = 0.0
    height: float = 0.0


def detect_yellow_orbs(frame_rgb: np.ndarray) -> list[ScreenTarget]:
    """Small yellow collectible squares in the playfield."""
    if frame_rgb.size == 0:
        return []
    height, width = frame_rgb.shape[:2]
    crop = frame_rgb[
        int(height * 0.08) : int(height * 0.88),
        int(width * 0.05) : int(width * 0.95),
    ]
    if crop.size == 0:
        return []
    offset_x = int(width * 0.05)
    offset_y = int(height * 0.08)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([18, 90, 120]), np.array([38, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    targets: list[ScreenTarget] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 12 or area > 1200:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if max(w, h) / max(min(w, h), 1) > 2.8:
            continue
        targets.append(
            ScreenTarget(
                cx=float(offset_x + x + w / 2),
                cy=float(offset_y + y + h / 2),
                kind="yellow_orb",
                width=float(w),
                height=float(h),
            )
        )
    return targets


def detect_green_boosters(frame_rgb: np.ndarray) -> list[ScreenTarget]:
    """Green horizontal boost-arrow pads in the playfield (not the HUD energy bar)."""
    if frame_rgb.size == 0:
        return []
    height, width = frame_rgb.shape[:2]
    crop = frame_rgb[
        int(height * 0.12) : int(height * 0.82),
        int(width * 0.07) : int(width * 0.93),
    ]
    if crop.size == 0:
        return []
    offset_x = int(width * 0.07)
    offset_y = int(height * 0.12)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([42, 80, 90]), np.array([88, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    targets: list[ScreenTarget] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 20 or h < 3 or w / max(h, 1) < 3.5:
            continue
        targets.append(
            ScreenTarget(
                cx=float(offset_x + x + w / 2),
                cy=float(offset_y + y + h / 2),
                kind="green_booster",
                width=float(w),
                height=float(h),
            )
        )
    return targets


def _kind_weight(kind: str, *, falling: bool) -> float:
    if kind == "platform":
        return 0.95 if falling else 1.0
    if kind == "yellow_orb":
        return 0.82
    if kind == "green_booster":
        return 0.88
    return 1.0


def nearest_navigation_target(
    orb_x: float | None,
    orb_y: float | None,
    platforms: list[Platform],
    yellow_orbs: list[ScreenTarget],
    green_boosters: list[ScreenTarget],
    frame_shape: tuple[int, ...],
    *,
    falling: bool = False,
) -> tuple[float | None, float | None, str | None]:
    """Pick the best on-screen element to steer toward."""
    if orb_x is None or orb_y is None:
        return None, None, None

    height, width = frame_shape[:2]
    candidates: list[tuple[float, float, float, str]] = []

    for platform in platforms:
        if platform.is_hazard:
            continue
        dx_px = platform.cx - orb_x
        dy_px = platform.cy - orb_y
        if falling and dy_px < -40:
            continue
        dist = float(np.hypot(dx_px, dy_px))
        weighted = dist * _kind_weight("platform", falling=falling)
        candidates.append(
            (
                weighted,
                float(np.clip(dx_px / max(width, 1), -1.0, 1.0)),
                float(np.clip(dy_px / max(height, 1), -1.0, 1.0)),
                "platform",
            )
        )

    for target in yellow_orbs + green_boosters:
        dx_px = target.cx - orb_x
        dy_px = target.cy - orb_y
        dist = float(np.hypot(dx_px, dy_px))
        weighted = dist * _kind_weight(target.kind, falling=falling)
        candidates.append(
            (
                weighted,
                float(np.clip(dx_px / max(width, 1), -1.0, 1.0)),
                float(np.clip(dy_px / max(height, 1), -1.0, 1.0)),
                target.kind,
            )
        )

    if not candidates:
        pdx, pdy = nearest_safe_platform(orb_x, orb_y, platforms, frame_shape)
        if pdx is None:
            return None, None, None
        return pdx, pdy, "platform"

    _, dx, dy, kind = min(candidates, key=lambda item: item[0])
    return dx, dy, kind


def platform_targets_from_sim(
    platforms: list[Platform],
    orb_x: float,
    orb_y: float,
    frame_shape: tuple[int, int],
) -> tuple[float | None, float | None, str | None]:
    """Sim-only: platforms are the primary navigation targets."""
    dx, dy = nearest_safe_platform(orb_x, orb_y, platforms, frame_shape)
    if dx is None:
        return None, None, None
    return dx, dy, "platform"

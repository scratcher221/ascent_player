from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from ascent_player.env.platform_detector import (
    build_platform_mask,
    detect_platforms,
    nearest_safe_platform,
)


@dataclass(slots=True)
class FrameState:
    orb_x: float | None = None
    orb_y: float | None = None
    score: int | None = None
    boost_level: float = 1.0
    can_boost: bool = True
    nearest_platform_dx: float | None = None
    nearest_platform_dy: float | None = None
    platform_mask: np.ndarray | None = None
    combo: int = 0
    streak: int = 0
    score_multiplier: float = 1.0
    platform_landed: bool = False
    game_over: bool = False
    in_menu: bool = False


JUMP_ACTIONS = frozenset({3, 4, 5})


def can_boost_from_levels(
    energy: float,
    reserve: float,
    *,
    min_energy: float = 14.0,
) -> bool:
    """Match game logic: boost costs 14 energy (main + reserve pool)."""
    return (energy * 100.0 + reserve * 100.0) >= min_energy


def apply_hud_boost(
    state: FrameState,
    hud_energy: float | None,
    hud_reserve: float | None,
    hud_can_boost: bool | None,
    *,
    min_energy: float = 14.0,
) -> FrameState:
    if hud_energy is not None:
        reserve = hud_reserve or 0.0
        state.boost_level = min(1.0, hud_energy + reserve)
        state.can_boost = (
            hud_can_boost
            if hud_can_boost is not None
            else can_boost_from_levels(hud_energy, reserve, min_energy=min_energy)
        )
    return state


def mask_jump_action(action: int, can_boost: bool) -> int:
    if can_boost or action not in JUMP_ACTIONS:
        return action
    if action == 3:
        return 0
    if action == 4:
        return 1
    return 2


def detect_orb(frame_rgb: np.ndarray) -> tuple[float, float] | None:
    if frame_rgb.size == 0:
        return None
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # Cyan/green glow around the orb in the screenshots.
    lower = np.array([65, 80, 80], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        return None
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def parse_score_from_text(text: str) -> int | None:
    match = re.search(r"SCORE\s*(\d[\d\s_]*)", text, flags=re.IGNORECASE)
    if not match:
        return None
    digits = re.sub(r"\D", "", match.group(1))
    if not digits:
        return None
    return int(digits)


def estimate_score(frame_rgb: np.ndarray) -> int | None:
    """Visual fallback for score digits in the HUD row."""
    if frame_rgb.size == 0:
        return None
    height, width = frame_rgb.shape[:2]
    crop = frame_rgb[
        int(height * 0.01) : max(2, int(height * 0.12)),
        int(width * 0.08) : max(3, int(width * 0.28)),
    ]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    components, _, stats, _ = cv2.connectedComponentsWithStats(threshold)
    digit_areas = [
        int(stats[idx, cv2.CC_STAT_AREA])
        for idx in range(1, components)
        if 8 <= int(stats[idx, cv2.CC_STAT_AREA]) <= 800
    ]
    if not digit_areas:
        return None
    # Monotonic proxy when OCR is ambiguous; DOM parsing is preferred.
    return sum(digit_areas)


def detect_game_over_visual(frame_rgb: np.ndarray) -> bool:
    """Heuristic game-over overlay detection without DOM queries."""
    if frame_rgb.size == 0:
        return False
    height, width = frame_rgb.shape[:2]
    center = frame_rgb[
        int(height * 0.28) : int(height * 0.72),
        int(width * 0.18) : int(width * 0.82),
    ]
    gray = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)
    bright = gray > 185
    # Death screen shows large bright title text in the center.
    return float(np.mean(bright)) > 0.04 and float(np.std(gray)) > 35.0


def detect_boost_bar(frame_rgb: np.ndarray) -> tuple[float, bool]:
    """Estimate boost energy from the vertical bar at the bottom-left HUD."""
    if frame_rgb.size == 0:
        return 1.0, True

    height, width = frame_rgb.shape[:2]
    crop = frame_rgb[
        int(height * 0.52) : int(height * 0.9),
        int(width * 0.0) : max(2, int(width * 0.045)),
    ]
    if crop.size == 0:
        return 1.0, True

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, np.array([40, 70, 70]), np.array([95, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([12, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([165, 70, 70]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    bar_height = max(1, green_mask.shape[0])
    filled_rows = 0
    for row in range(bar_height - 1, -1, -1):
        if float(np.mean(green_mask[row])) > 18.0:
            filled_rows += 1
        elif filled_rows > 0:
            break

    boost_level = min(1.0, filled_rows / bar_height)
    if boost_level < 0.08:
        green_ratio = float(np.mean(green_mask > 0))
        boost_level = min(1.0, green_ratio * 4.0)

    red_ratio = float(np.mean(red_mask > 0))
    total_energy = boost_level + min(1.0, red_ratio * 0.15)
    can_boost = total_energy * 100.0 >= 14.0
    if boost_level < 0.12 and red_ratio > 0.12:
        can_boost = False
        boost_level = min(boost_level, 0.08)
    return min(1.0, total_energy), can_boost


def detect_from_frame(frame_rgb: np.ndarray) -> FrameState:
    orb = detect_orb(frame_rgb)
    score = estimate_score(frame_rgb)
    boost_level, can_boost = detect_boost_bar(frame_rgb)
    platforms = detect_platforms(frame_rgb)
    platform_dx = None
    platform_dy = None
    if orb is not None:
        platform_dx, platform_dy = nearest_safe_platform(
            orb[0],
            orb[1],
            platforms,
            frame_rgb.shape,
        )
    state = FrameState(
        score=score,
        boost_level=boost_level,
        can_boost=can_boost,
        nearest_platform_dx=platform_dx,
        nearest_platform_dy=platform_dy,
        game_over=detect_game_over_visual(frame_rgb),
    )
    if orb is not None:
        state.orb_x, state.orb_y = orb
    state.platform_mask = build_platform_mask(frame_rgb, platforms)
    return state


def platform_mask_from_state(frame_state: FrameState, frame_rgb: np.ndarray) -> np.ndarray:
    if frame_state.platform_mask is not None:
        return frame_state.platform_mask
    return build_platform_mask(frame_rgb)


def parse_combo_from_text(text: str) -> int | None:
    match = re.search(r"x(\d+)\s*STREAK", text, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def parse_multiplier_from_text(text: str) -> float | None:
    match = re.search(r"[×x]([\d.]+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def merge_dom_state(frame_state: FrameState, body_text: str) -> FrameState:
    text = body_text.upper()
    frame_state.game_over = "FELL" in text or "BACK TO EARTH" in text
    frame_state.in_menu = "START THE ASCENT" in text or "PICK 1 ULTI" in text
    parsed_score = parse_score_from_text(body_text)
    if parsed_score is not None:
        frame_state.score = parsed_score
    combo = parse_combo_from_text(body_text)
    if combo is not None:
        frame_state.combo = combo
        frame_state.streak = combo // 10
    multiplier = parse_multiplier_from_text(body_text)
    if multiplier is not None:
        frame_state.score_multiplier = multiplier
    return frame_state

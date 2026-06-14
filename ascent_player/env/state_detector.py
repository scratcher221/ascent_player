from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class FrameState:
    orb_x: float | None = None
    orb_y: float | None = None
    score: int | None = None
    game_over: bool = False
    in_menu: bool = False


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


def estimate_score(frame_rgb: np.ndarray) -> int | None:
    """Rough HUD score estimate.

    This intentionally avoids a heavy OCR dependency. It detects bright score
    glyph activity in the top-left HUD and returns a monotonic-ish proxy. The
    reward layer only uses positive deltas, so noisy readings are tolerable.
    """
    if frame_rgb.size == 0:
        return None
    height, width = frame_rgb.shape[:2]
    crop = frame_rgb[0 : max(1, int(height * 0.14)), 0 : max(1, int(width * 0.2))]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, threshold = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    components, _, stats, _ = cv2.connectedComponentsWithStats(threshold)
    total = 0
    for idx in range(1, components):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if 3 <= area <= 500:
            total += area
    return total


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


def detect_from_frame(frame_rgb: np.ndarray) -> FrameState:
    orb = detect_orb(frame_rgb)
    score = estimate_score(frame_rgb)
    state = FrameState(score=score, game_over=detect_game_over_visual(frame_rgb))
    if orb is not None:
        state.orb_x, state.orb_y = orb
    return state


def merge_dom_state(frame_state: FrameState, body_text: str) -> FrameState:
    text = body_text.upper()
    frame_state.game_over = "FELL" in text or "BACK TO EARTH" in text
    frame_state.in_menu = "START THE ASCENT" in text or "PICK 1 ULTI" in text
    return frame_state

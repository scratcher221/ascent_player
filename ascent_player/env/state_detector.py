from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from ascent_player.env.navigation import enrich_navigation, platform_type_onehot
from ascent_player.env.platform_detector import (
    build_platform_mask,
    detect_platforms,
    nearest_safe_platform,
)
from ascent_player.env.target_detector import (
    detect_green_boosters,
    detect_yellow_orbs,
    nearest_navigation_target,
)


@dataclass(slots=True)
class FrameState:
    orb_x: float | None = None
    orb_y: float | None = None
    orb_vx: float | None = None
    orb_vy: float | None = None
    score: int | None = None
    boost_level: float = 1.0
    can_boost: bool = True
    nearest_platform_dx: float | None = None
    nearest_platform_dy: float | None = None
    nearest_platform_width: float = 0.0
    platform_wear: float = 0.0
    nearest_platform_above_dx: float | None = None
    nearest_platform_above_dy: float | None = None
    nearest_platform_above_width: float = 0.0
    nearest_platform_above_wear: float = 0.0
    nearest_platform_above_type: str = "neutral"
    target_platform_type: str = "neutral"
    horizontal_error: float = 0.0
    rising: bool = False
    falling: bool = False
    airborne: bool = False
    landing_window: bool = False
    boost_useful: bool | None = None
    time_to_platform: float = 0.0
    danger_worn: bool = False
    danger_sell: bool = False
    miss_risk: bool = False
    platform_mask: np.ndarray | None = None
    target_dx: float | None = None
    target_dy: float | None = None
    target_kind: str | None = None
    booster_dx: float | None = None
    booster_dy: float | None = None
    booster_type: str | None = None
    combo: int = 0
    streak: int = 0
    score_multiplier: float = 1.0
    bonus: float = 0.0
    bank_style: float = 0.0
    height: float = 0.0
    bounces: int = 0
    storm_level: float = 0.0
    tier_index: int = 0
    canvas_w: float = 640.0
    canvas_h: float = 360.0
    agent_hook_ok: bool = False
    game_phase: str = ""
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


def detect_from_frame(
    frame_rgb: np.ndarray,
    *,
    platform_only: bool = False,
) -> FrameState:
    orb = detect_orb(frame_rgb)
    score = estimate_score(frame_rgb)
    boost_level, can_boost = detect_boost_bar(frame_rgb)
    platforms = detect_platforms(frame_rgb)
    yellow_orbs = detect_yellow_orbs(frame_rgb)
    green_boosters = detect_green_boosters(frame_rgb)
    platform_dx = None
    platform_dy = None
    target_dx = None
    target_dy = None
    target_kind = None
    if orb is not None:
        platform_dx, platform_dy = nearest_safe_platform(
            orb[0],
            orb[1],
            platforms,
            frame_rgb.shape,
        )
        target_dx, target_dy, target_kind = nearest_navigation_target(
            orb[0],
            orb[1],
            platforms,
            [] if platform_only else yellow_orbs,
            [] if platform_only else green_boosters,
            frame_rgb.shape,
        )
    state = FrameState(
        score=score,
        boost_level=boost_level,
        can_boost=can_boost,
        nearest_platform_dx=platform_dx,
        nearest_platform_dy=platform_dy,
        target_dx=target_dx,
        target_dy=target_dy,
        target_kind=target_kind,
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


def merge_agent_state(frame_state: FrameState, payload: dict | None) -> FrameState:
    if not isinstance(payload, dict):
        return frame_state
    frame_state.agent_hook_ok = True
    frame_state.game_phase = str(payload.get("state") or "")
    if frame_state.game_phase == "crashed":
        frame_state.game_over = True
    if frame_state.game_phase in {"ready", "paused"}:
        frame_state.in_menu = True

    canvas_w = float(payload.get("canvasW") or frame_state.canvas_w or 640.0)
    canvas_h = float(payload.get("canvasH") or frame_state.canvas_h or 360.0)
    frame_state.canvas_w = canvas_w
    frame_state.canvas_h = canvas_h
    camera_y = float(payload.get("cameraY") or 0.0)

    orb = payload.get("orb") or {}
    if orb:
        frame_state.orb_x = float(orb.get("x", 0.0))
        frame_state.orb_vx = float(orb.get("vx", 0.0))
        frame_state.orb_vy = float(orb.get("vy", 0.0))
        world_y = float(orb.get("worldY", 0.0))
        frame_state.orb_y = (world_y - camera_y) / max(canvas_h, 1.0)
        energy = float(orb.get("energy", 0.0))
        reserve = float(orb.get("reserve", 0.0))
        frame_state.boost_level = min(1.0, (energy + reserve) / 100.0)
        frame_state.can_boost = bool(payload.get("canBoost", energy + reserve >= 14))
        frame_state.combo = int(orb.get("combo", frame_state.combo))
        frame_state.bonus = float(orb.get("bonus", 0.0))
        frame_state.bank_style = float(orb.get("bankStyle", 0.0))
        frame_state.height = float(orb.get("height", 0.0))
        frame_state.bounces = int(orb.get("bounces", 0))

    if payload.get("score") is not None:
        frame_state.score = int(payload["score"])
    frame_state.tier_index = int(payload.get("tierIndex", frame_state.tier_index))
    frame_state.storm_level = float(payload.get("stormLevel", 0.0))

    nearest = payload.get("nearestPlatformBelow")
    if isinstance(nearest, dict):
        frame_state.nearest_platform_dx = float(nearest.get("dx", 0.0))
        frame_state.nearest_platform_dy = float(nearest.get("dy", 0.0))
        frame_state.nearest_platform_width = float(nearest.get("width", 0.0))
        frame_state.platform_wear = float(nearest.get("wear", 0.0))
        frame_state.target_platform_type = str(nearest.get("type") or "neutral")
        frame_state.target_dx = frame_state.nearest_platform_dx
        frame_state.target_dy = frame_state.nearest_platform_dy
        frame_state.target_kind = "platform"

    above = payload.get("nearestPlatformAbove")
    if isinstance(above, dict):
        frame_state.nearest_platform_above_dx = float(above.get("dx", 0.0))
        frame_state.nearest_platform_above_dy = float(above.get("dy", 0.0))
        frame_state.nearest_platform_above_width = float(above.get("width", 0.0))
        frame_state.nearest_platform_above_wear = float(above.get("wear", 0.0))
        frame_state.nearest_platform_above_type = str(above.get("type") or "neutral")

    landing = payload.get("bestLandingPlatform")
    if isinstance(landing, dict):
        frame_state.target_dx = float(landing.get("dx", 0.0))
        frame_state.target_dy = float(landing.get("dy", 0.0))
        frame_state.target_platform_type = str(landing.get("type") or "neutral")
        frame_state.target_kind = "platform"

    phase = payload.get("orbPhase")
    if isinstance(phase, dict):
        frame_state.rising = bool(phase.get("rising"))
        frame_state.falling = bool(phase.get("falling"))
        frame_state.landing_window = bool(phase.get("landingWindow"))
        frame_state.airborne = bool(phase.get("airborne"))

    if payload.get("timeToPlatform") is not None:
        frame_state.time_to_platform = float(payload["timeToPlatform"])
    if payload.get("boostUseful") is not None:
        frame_state.boost_useful = bool(payload["boostUseful"])

    danger = payload.get("danger")
    if isinstance(danger, dict):
        frame_state.danger_worn = bool(danger.get("worn"))
        frame_state.danger_sell = bool(danger.get("sell"))
        frame_state.miss_risk = bool(danger.get("missRisk"))

    booster = payload.get("nearestBooster")
    if isinstance(booster, dict):
        frame_state.booster_dx = float(booster.get("dx", 0.0))
        frame_state.booster_dy = float(booster.get("dy", 0.0))
        frame_state.booster_type = str(booster.get("type") or "")
        if frame_state.target_kind != "platform" or abs(frame_state.booster_dy or 1) < abs(
            frame_state.nearest_platform_dy or 1
        ):
            frame_state.target_dx = frame_state.booster_dx
            frame_state.target_dy = frame_state.booster_dy
            frame_state.target_kind = f"booster_{frame_state.booster_type}"

    return enrich_navigation(frame_state)


def platform_mask_from_agent_payload(
    payload: dict,
    width: int,
    height: int,
) -> np.ndarray:
    """Build a platform channel mask from exported JS platform geometry."""
    mask = np.zeros((height, width), dtype=np.uint8)
    platforms = payload.get("platforms") or []
    camera_y = float(payload.get("cameraY", 0))
    canvas_w = max(float(payload.get("canvasW") or width), 1.0)
    canvas_h = max(float(payload.get("canvasH") or height), 1.0)
    scale_x = width / canvas_w
    scale_y = height / canvas_h
    for plat in platforms:
        if not isinstance(plat, dict):
            continue
        world_y = float(plat.get("worldY", 0))
        screen_y = world_y - camera_y
        if screen_y < -30 or screen_y > canvas_h + 30:
            continue
        cx = float(plat.get("x", 0)) * scale_x
        pw = max(4.0, float(plat.get("width", 40)) * scale_x)
        sy = screen_y * scale_y
        x1 = max(0, int(cx - pw / 2))
        x2 = min(width, int(cx + pw / 2))
        y1 = max(0, int(sy - 5))
        y2 = min(height, int(sy + 5))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask

"""Map sim physics (y-down) to browser/game.js FrameState conventions (y-up vy).

game.js exports:
  - orb_y = (worldY - cameraY) / H  → 0 at bottom of view, 1 at top
  - rising = vy > 20, falling = vy < -20
  - nearestPlatformBelow: plat.worldY < orb.worldY, dy = (orb.worldY - plat.worldY) / H
  - pickTarget: abs(dx)*2 + abs(dy) + sell/wear penalties
  - timeToPlatform while falling toward below pad

Sim physics uses screen y-down (vy < 0 when rising). This module converts at the
observation boundary so hybrid vectors match the browser hook.
"""

from __future__ import annotations

from ascent_player.env.navigation import enrich_navigation
from ascent_player.env.sim_physics import SimWorld
from ascent_player.env.state_detector import FrameState


def game_orb_y(screen_y: float, height: float) -> float:
    """Browser hook: 0 = bottom of viewport, 1 = top."""
    h = max(float(height), 1.0)
    return 1.0 - float(screen_y) / h


def game_orb_vy(physics_vy: float) -> float:
    """Flip sim y-down velocity to game y-up velocity."""
    return -float(physics_vy)


def pick_target_like_game(
    *,
    below_dx: float,
    below_dy: float,
    below_wear: float,
    below_type: str,
    above_dx: float,
    above_dy: float,
    above_wear: float,
    above_type: str,
    has_below: bool,
    has_above: bool,
) -> tuple[float, float, str]:
    """Match game.js pickTarget scoring over above/below candidates."""
    best_dx = 0.0
    best_dy = 0.0
    best_type = "neutral"
    best_score = float("inf")
    if has_below and below_dy > 0:
        score = abs(below_dx) * 2.0 + abs(below_dy)
        if below_type == "sell":
            score += 0.35
        score += max(0.0, below_wear - 0.7) * 0.5
        if score < best_score:
            best_score = score
            best_dx, best_dy, best_type = below_dx, below_dy, below_type
    if has_above and above_dy > 0:
        score = abs(above_dx) * 2.0 + abs(above_dy)
        if above_type == "sell":
            score += 0.35
        score += max(0.0, above_wear - 0.7) * 0.5
        if score < best_score:
            best_score = score
            best_dx, best_dy, best_type = above_dx, above_dy, above_type
    if best_score == float("inf"):
        if has_above:
            return above_dx, above_dy, above_type
        return below_dx, below_dy, below_type
    return best_dx, best_dy, best_type


def time_to_platform_game(
    *,
    falling: bool,
    orb_vy: float,
    below_dy: float,
    height: float,
) -> float:
    """game.js timeToPlatform: dyPx / |vy| / H while falling."""
    if not falling or below_dy <= 0 or orb_vy >= -1.0:
        return 0.0
    h = max(float(height), 1.0)
    dy_px = float(below_dy) * h
    return float(min(1.0, max(0.0, dy_px / max(abs(orb_vy), 1.0) / h)))


def build_frame_state_from_world(
    world: SimWorld,
    *,
    platform_mask=None,
    booster_dx: float | None = None,
    booster_dy: float | None = None,
    booster_type: str | None = None,
) -> FrameState:
    """Build a browser-convention FrameState from sim physics."""
    ball = world.ball
    cfg = world.config
    h = float(cfg.height)
    w = float(cfg.width)
    screen_y = ball.y - world.camera_y

    ndx, ndy, wear, pwidth = world.nearest_platform_below()
    adx, ady, awear, awidth, atype = world.nearest_platform_above()
    has_below = ndy > 0 or pwidth > 0
    has_above = ady > 0 or awidth > 0
    target_dx, target_dy, target_type = pick_target_like_game(
        below_dx=ndx,
        below_dy=ndy,
        below_wear=wear,
        below_type="neutral",
        above_dx=adx,
        above_dy=ady,
        above_wear=awear,
        above_type=atype,
        has_below=has_below,
        has_above=has_above,
    )

    if booster_dx is None or booster_dy is None:
        booster_dx, booster_dy, booster_type = _nearest_booster(world, screen_y)

    orb_vy = game_orb_vy(ball.vy)
    state = FrameState(
        orb_x=ball.x / w,
        orb_y=game_orb_y(screen_y, h),
        orb_vx=ball.vx,
        orb_vy=orb_vy,
        score=world.score,
        boost_level=world.boost_level,
        can_boost=world.can_boost,
        nearest_platform_dx=ndx,
        nearest_platform_dy=ndy,
        nearest_platform_width=pwidth,
        platform_wear=wear,
        nearest_platform_above_dx=adx,
        nearest_platform_above_dy=ady,
        nearest_platform_above_width=awidth,
        nearest_platform_above_wear=awear,
        nearest_platform_above_type=atype,
        target_dx=target_dx,
        target_dy=target_dy,
        target_platform_type=target_type,
        booster_dx=booster_dx,
        booster_dy=booster_dy,
        booster_type=booster_type,
        combo=world.combo,
        score_multiplier=world.score_multiplier,
        bonus=world.bonus,
        bank_style=world.bank_style,
        height=world.height,
        bounces=world.bounces,
        canvas_w=w,
        canvas_h=h,
        agent_hook_ok=True,
        platform_landed=world.platform_landed,
        platform_mask=platform_mask,
        game_over=False,
    )
    state = enrich_navigation(state)
    state.time_to_platform = time_to_platform_game(
        falling=bool(state.falling),
        orb_vy=float(state.orb_vy or 0.0),
        below_dy=float(state.nearest_platform_dy or 0.0),
        height=h,
    )
    return state


def _nearest_booster(
    world: SimWorld, screen_y: float
) -> tuple[float | None, float | None, str | None]:
    ball = world.ball
    h = max(world.config.height, 1.0)
    w = max(world.config.width, 1.0)
    best_dist = float("inf")
    booster_dx = booster_dy = booster_type = None
    for booster in world.boosters:
        if booster.used:
            continue
        dx = (booster.cx - ball.x) / w
        # Booster dy in browser convention: positive = booster above orb on screen.
        booster_screen_y = booster.cy - world.camera_y
        dy = (screen_y - booster_screen_y) / h
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            booster_dx = dx
            booster_dy = dy
            booster_type = booster.type
    return booster_dx, booster_dy, booster_type

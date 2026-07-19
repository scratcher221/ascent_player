"""Map sim physics (y-down) to browser/game.js FrameState conventions (y-up vy).

game.js exports:
  - orb_y = (worldY - cameraY) / H  → 0 at bottom of view, 1 at top
  - rising = vy > 20, falling = vy < -20
  - nearestPlatformBelow: plat.worldY < orb.worldY, dy = (orb.worldY - plat.worldY) / H

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

    if booster_dx is None or booster_dy is None:
        booster_dx, booster_dy, booster_type = _nearest_booster(world, screen_y)

    state = FrameState(
        orb_x=ball.x / w,
        orb_y=game_orb_y(screen_y, h),
        orb_vx=ball.vx,
        orb_vy=game_orb_vy(ball.vy),
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
        target_dx=adx if ady < ndy or ndy == 0 else ndx,
        target_dy=ady if ady < ndy or ndy == 0 else ndy,
        target_platform_type=atype if ady < ndy or ndy == 0 else "neutral",
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
    return enrich_navigation(state)


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

from __future__ import annotations

from dataclasses import dataclass, field
import random

import numpy as np

from ascent_player.env.score_rules import combo_multiplier, live_score

# Constants mirrored from game/ASCENT — Ride the pump_files/game.js
GRAVITY = 820.0  # world-px/s²
JUMP_FORCE = 640.0  # auto-bounce impulse
BOOST_FORCE = 520.0  # SPACE boost impulse
BOOST_COST = 14.0  # energy per boost
ENERGY_REGEN = 5.0  # energy/s base (game.js ENERGY_REGEN=5)
MAX_ORB_VY = 1500.0
ORB_SPEED = 230.0  # base; effective speed scales with width like game.js
TIER_WEIGHT = 1.0
# seedPlatforms: step = maxJump * 0.68 with maxJump = JUMP²/(2g) ≈ 250px → step ≈ 170px
PLATFORM_SPACING_Y = (JUMP_FORCE * JUMP_FORCE) / (2.0 * GRAVITY) * 0.68
# game.js seedPlatforms near-center staircase (fraction of W)
_STAIR_OFFSETS = (
    0.0,
    -0.18,
    0.18,
    -0.12,
    0.12,
    0.0,
    -0.2,
    0.2,
    -0.08,
    0.08,
    0.0,
    -0.16,
    0.16,
    0.0,
    -0.1,
    0.1,
    0.0,
    0.0,
)
# booster-rules.js base weights at pressure≈0
_BOOSTER_WEIGHTS = (("surge", 1.0), ("stream", 0.8), ("drag", 0.55))


def orb_speed_eff(width: float, base: float = ORB_SPEED) -> float:
    """Match game.js resize(): ORB_SPEED_EFF for horizontal reachability."""
    w = max(float(width), 1.0)
    narrow = max(0.0, min(1.0, (800.0 - w) / 380.0))
    return float(min(base * (w / 420.0) * (1.0 + 0.15 * narrow), 575.0))


@dataclass(slots=True)
class SimPhysicsConfig:
    width: int = 640
    height: int = 360
    gravity: float = GRAVITY
    jump_force: float = JUMP_FORCE
    boost_force: float = BOOST_FORCE
    boost_cost: float = BOOST_COST
    energy_regen_per_sec: float = ENERGY_REGEN
    # game.js: ORB_R = max(18, round(gs*22)), gs=H/700 → ~18 at 360p
    orb_radius: float = 18.0
    # seedPlatforms: gs*(90+rand*70); at H=360, gs≈0.51 → ~46–82
    platform_min_width: float = 46.0
    platform_max_width: float = 82.0
    platform_height: float = 10.0  # PLAT_H floor
    platform_spacing_y: float = PLATFORM_SPACING_Y
    bounce_limit: int = 3
    death_margin_below_camera: float = 80.0
    bonus_per_landing: float = 130.0
    surge_boost: float = 200.0
    surge_energy: float = 34.0
    stream_energy_per_sec: float = 32.0
    stream_score_per_sec: float = 12.0
    stream_entry_impulse: float = 420.0
    stream_min_ascent: float = 240.0
    stream_accel: float = 900.0
    drag_slow: float = 0.48
    drag_energy: float = -18.0
    booster_spawn_interval: float = 2.6
    # Mild livePressure stand-in so regen isn't stuck at dead-market floor.
    # game: ENERGY_REGEN * (1 + max(0, pressure) * 0.45); ~0.35 ≈ light buy pressure.
    energy_pressure: float = 0.35
    seed: int | None = None

    @property
    def orb_speed(self) -> float:
        return orb_speed_eff(self.width)


@dataclass(slots=True)
class SimPlatform:
    cx: float
    cy: float
    width: float
    height: float
    receptions: int = 0
    bounce_limit: int = 3
    is_hazard: bool = False
    bounce: float = 1.0


@dataclass(slots=True)
class SimBooster:
    type: str
    cx: float
    cy: float
    width: float
    height: float
    used: bool = False
    length: float = 0.0  # stream vertical extent (world-px)
    inside: bool = False
    entered: bool = False


@dataclass(slots=True)
class SimBall:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    energy: float = 100.0
    reserve: float = 0.0


@dataclass(slots=True)
class SimWorld:
    config: SimPhysicsConfig
    ball: SimBall = field(init=False)
    platforms: list[SimPlatform] = field(default_factory=list)
    boosters: list[SimBooster] = field(default_factory=list)
    camera_y: float = 0.0
    origin_y: float = 0.0
    max_height: float = 0.0
    score: int = 0
    bank_style: float = 0.0
    bonus: float = 0.0
    combo: int = 0
    max_combo: int = 0
    bounces: int = 0
    platform_landed: bool = False
    booster_collected: str | None = None
    rng: random.Random = field(init=False)
    next_platform_y: float = 0.0
    booster_timer: float = 0.0
    _spawn_index: int = 0
    _last_platform_x: float = 0.0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.config.seed)
        self.reset()

    def reset(self, start_height: float = 0.0) -> None:
        cfg = self.config
        self.ball = SimBall(x=cfg.width * 0.5, y=cfg.height * 0.72)
        self.platforms = []
        self.boosters = []
        self.camera_y = 0.0
        self.origin_y = self.ball.y
        self.max_height = 0.0
        self.score = 0
        self.bank_style = 0.0
        self.bonus = 0.0
        self.combo = 0
        self.max_combo = 0
        self.bounces = 0
        self.platform_landed = False
        self.booster_collected = None
        self.booster_timer = 0.0
        self._spawn_index = 0
        self._last_platform_x = self.ball.x
        start_y = self.ball.y
        self.next_platform_y = start_y - 36.0
        # Guaranteed floor + near-center staircase (game.js seedPlatforms).
        for index in range(18):
            self._spawn_platform(staircase=True)
        floor = SimPlatform(
            cx=self.ball.x,
            cy=self.ball.y + 28.0,
            width=cfg.width * 0.62,
            height=cfg.platform_height,
            bounce_limit=99,
            bounce=1.0,
        )
        self.platforms.append(floor)
        if start_height > 0:
            self.origin_y = self.ball.y + float(start_height)
            self.max_height = float(start_height)
            self.camera_y = self.ball.y - cfg.height * 0.42
            while self.next_platform_y > self.camera_y - cfg.height * 2.0:
                self._spawn_platform(staircase=True)
            self._sync_score()

    @property
    def height(self) -> float:
        return max(0.0, self.origin_y - self.ball.y)

    @property
    def score_multiplier(self) -> float:
        return combo_multiplier(self.combo)

    def break_combo(self) -> None:
        from ascent_player.env.score_rules import live_style

        self.bank_style += live_style(bonus=self.bonus, combo=self.combo)
        self.bonus = 0.0
        self.combo = 0
        self._sync_score()

    def _sync_score(self) -> None:
        if self.height > self.max_height:
            self.max_height = self.height
        self.score = live_score(
            height=self.max_height,
            bank_style=self.bank_style,
            bonus=self.bonus,
            combo=self.combo,
            tier_weight=TIER_WEIGHT,
        )

    def _land_on_platform(self) -> None:
        self.platform_landed = True
        self.bounces += 1
        self.combo = min(self.combo + 1, 999)
        self.max_combo = max(self.max_combo, self.combo)
        self.bonus = min(4000.0, self.bonus + self.config.bonus_per_landing * 0.05)
        self._sync_score()

    def _spawn_platform(self, *, staircase: bool = False, bias_x: float | None = None) -> None:
        """Spawn next platform upward.

        Matches game.js seedPlatforms (near-center offsets) and keeps live
        spawns laterally chained so one bounce can still reach the next pad.
        """
        cfg = self.config
        width = self.rng.uniform(cfg.platform_min_width, cfg.platform_max_width)
        margin = max(22.0, cfg.width * 0.04)
        lo = margin + width / 2
        hi = cfg.width - margin - width / 2
        if staircase:
            offset = _STAIR_OFFSETS[self._spawn_index % len(_STAIR_OFFSETS)]
            x = float(np.clip(cfg.width * 0.5 + offset * cfg.width, lo, hi))
        elif bias_x is not None:
            spread = min(cfg.orb_speed * 0.55, cfg.width * 0.18)
            x = float(np.clip(bias_x + self.rng.uniform(-spread, spread), lo, hi))
        else:
            # Chain from previous pad; cap step to ~0.9s of steering at ORB_SPEED_EFF.
            max_step = min(cfg.orb_speed * 0.9, cfg.width * 0.42)
            x = float(
                np.clip(
                    self._last_platform_x + self.rng.uniform(-max_step, max_step),
                    lo,
                    hi,
                )
            )
        self.next_platform_y -= self.rng.uniform(
            cfg.platform_spacing_y * 0.85,
            cfg.platform_spacing_y * 1.15,
        )
        bounce = 0.95 + self.rng.random() * 0.18
        self.platforms.append(
            SimPlatform(
                cx=x,
                cy=self.next_platform_y,
                width=width,
                height=cfg.platform_height,
                bounce_limit=cfg.bounce_limit,
                bounce=bounce,
            )
        )
        self._last_platform_x = x
        self._spawn_index += 1

    def _pick_booster_type(self) -> str:
        total = sum(w for _, w in _BOOSTER_WEIGHTS)
        cursor = self.rng.random() * total
        for name, weight in _BOOSTER_WEIGHTS:
            cursor -= weight
            if cursor <= 0:
                return name
        return "surge"

    def _spawn_booster(self) -> None:
        cfg = self.config
        gs = max(0.5, min(3.0, cfg.height / 700.0))
        booster_type = self._pick_booster_type()
        # game.js mkBooster: gs-scaled sizes (readable at 360p via floor below)
        if booster_type == "stream":
            width = max(28.0, gs * 64.0)
            length = max(90.0, gs * 220.0)
            height = max(12.0, gs * 16.0)
        elif booster_type == "surge":
            width = max(22.0, gs * 30.0)
            length = 0.0
            height = max(12.0, gs * 16.0)
        else:
            width = max(48.0, gs * 150.0 * 0.85)  # narrow-screen drag shrink
            length = 0.0
            height = max(12.0, gs * 16.0)
        # Bias toward climb path (last platform / orb), not full-width roulette.
        anchor = 0.6 * self.ball.x + 0.4 * self._last_platform_x
        spread = cfg.width * 0.28
        x = float(
            np.clip(
                anchor + self.rng.uniform(-spread, spread),
                width / 2 + 20,
                cfg.width - width / 2 - 20,
            )
        )
        y = self.camera_y - self.rng.uniform(40, cfg.height * 0.55)
        self.boosters.append(
            SimBooster(
                type=booster_type,
                cx=x,
                cy=y,
                width=width,
                height=height,
                length=length,
            )
        )

    def _award_booster_combo(self) -> None:
        self.combo = min(self.combo + 1, 999)
        self.max_combo = max(self.max_combo, self.combo)

    def _collect_booster(self, booster: SimBooster) -> None:
        if booster.used and booster.type != "stream":
            return
        self.booster_collected = booster.type
        ball = self.ball
        if booster.type == "surge":
            booster.used = True
            # game: vy = max(vy, 60) + boost  (up is +vy there; here up is -vy)
            ball.vy = min(-60.0, ball.vy) - self.config.surge_boost
            ball.energy = min(100.0, ball.energy + self.config.surge_energy)
            self.bonus = min(4000.0, self.bonus + 130.0)
            self._award_booster_combo()
        elif booster.type == "stream":
            # Instant entry kick; sustained boost handled in _resolve_booster_collisions.
            ball.vy = min(-self.config.stream_min_ascent, ball.vy - self.config.stream_entry_impulse)
            if not booster.entered:
                booster.entered = True
                self.bonus = min(4000.0, self.bonus + 12.0)
                self._award_booster_combo()
        elif booster.type == "drag":
            booster.used = True
            ball.vy = ball.vy * self.config.drag_slow - 50.0
            ball.energy = max(0.0, ball.energy + self.config.drag_energy)
            self.break_combo()

    def step(
        self,
        *,
        move_left: bool,
        move_right: bool,
        jump: bool,
        dt: float,
    ) -> bool:
        cfg = self.config
        ball = self.ball
        self.platform_landed = False
        self.booster_collected = None
        speed = cfg.orb_speed

        target_vx = 0.0
        if move_left:
            target_vx -= speed
        if move_right:
            target_vx += speed
        ball.vx += (target_vx - ball.vx) * (1.0 - np.exp(-dt * 14.0))
        ball.x = float(np.clip(ball.x, cfg.orb_radius, cfg.width - cfg.orb_radius))

        total_energy = ball.energy + ball.reserve
        if jump and total_energy >= cfg.boost_cost:
            ball.vy -= cfg.boost_force
            remaining = cfg.boost_cost
            from_reserve = min(ball.reserve, remaining)
            ball.reserve -= from_reserve
            remaining -= from_reserve
            ball.energy = max(0.0, ball.energy - remaining)

        ball.vy += cfg.gravity * dt
        ball.vy = float(np.clip(ball.vy, -MAX_ORB_VY, MAX_ORB_VY))
        ball.x += ball.vx * dt
        ball.y += ball.vy * dt

        regen = cfg.energy_regen_per_sec * (1.0 + max(0.0, cfg.energy_pressure) * 0.45)
        ball.energy = min(100.0, ball.energy + regen * dt)

        self._resolve_platform_collisions()
        self._resolve_booster_collisions(dt)

        if ball.y < self.camera_y + cfg.height * 0.42:
            self.camera_y = ball.y - cfg.height * 0.42

        self.booster_timer += dt
        if self.booster_timer >= cfg.booster_spawn_interval:
            self.booster_timer = 0.0
            self._spawn_booster()

        self._sync_score()

        while self.next_platform_y > self.camera_y - cfg.height:
            self._spawn_platform()

        # Fill sparse climb corridor (game.js ensureNeutralCoverage) without
        # advancing next_platform_y — spawn into the band at random Y.
        self._ensure_neutral_coverage()

        self.platforms = [
            platform
            for platform in self.platforms
            if self.camera_y - cfg.height * 1.2
            < platform.cy
            < self.camera_y + cfg.height + 140.0
        ]
        self.boosters = [
            booster
            for booster in self.boosters
            if not booster.used
            and booster.cy + booster.length
            < self.camera_y + cfg.height + 80.0
            and booster.cy > self.camera_y - cfg.height * 1.2
        ]

        return ball.y > self.camera_y + cfg.height + cfg.death_margin_below_camera

    def _ensure_neutral_coverage(self) -> None:
        """Mirror game.js ensureNeutralCoverage — densify pads near the orb."""
        cfg = self.config
        min_ahead = 4
        # Band from slightly below orb up toward camera top (smaller cy = higher).
        band_lo = self.ball.y + cfg.height * 0.15
        band_hi = min(self.ball.y - 20.0, self.camera_y + cfg.height * 0.15)
        if band_hi >= band_lo - 10.0:
            return
        ahead = sum(1 for p in self.platforms if band_hi <= p.cy <= band_lo)
        needed = min(3, min_ahead - ahead)
        margin = max(22.0, cfg.width * 0.04)
        for _ in range(max(0, needed)):
            width = self.rng.uniform(cfg.platform_min_width, cfg.platform_max_width)
            lo = margin + width / 2
            hi = cfg.width - margin - width / 2
            max_step = min(cfg.orb_speed * 0.75, cfg.width * 0.35)
            x = float(
                np.clip(
                    self.ball.x + self.rng.uniform(-max_step, max_step),
                    lo,
                    hi,
                )
            )
            cy = self.rng.uniform(band_hi, band_lo)
            self.platforms.append(
                SimPlatform(
                    cx=x,
                    cy=cy,
                    width=width,
                    height=cfg.platform_height,
                    bounce_limit=cfg.bounce_limit,
                    bounce=0.95 + self.rng.random() * 0.15,
                )
            )

    def _resolve_platform_collisions(self) -> None:
        cfg = self.config
        ball = self.ball
        for platform in list(self.platforms):
            half_w = platform.width / 2
            within_x = abs(ball.x - platform.cx) <= half_w + cfg.orb_radius * 0.4
            top = platform.cy - platform.height / 2
            if not within_x or ball.vy <= 0:
                continue
            if ball.y + cfg.orb_radius >= top and ball.y + cfg.orb_radius <= top + 18.0:
                ball.y = top - cfg.orb_radius
                ball.vy = -cfg.jump_force * platform.bounce
                platform.receptions += 1
                self._land_on_platform()
                if platform.receptions >= platform.bounce_limit:
                    self.platforms.remove(platform)
                break

    def _resolve_booster_collisions(self, dt: float = 1.0 / 60.0) -> None:
        cfg = self.config
        ball = self.ball
        gs = max(0.5, min(3.0, cfg.height / 700.0))
        for booster in self.boosters:
            if booster.type == "surge" and not booster.used:
                # Soft magnet pull (game.js attractRadius = gs*90).
                dx = ball.x - booster.cx
                dy = ball.y - booster.cy
                dist = float(np.hypot(dx, dy))
                attract = gs * 90.0
                if 0.0 < dist < attract:
                    pull = min(abs(dx), 220.0 * dt)
                    booster.cx += float(np.sign(dx) * pull)

            if booster.type == "stream":
                inside_x = abs(ball.x - booster.cx) <= booster.width / 2
                # Stream extends upward (smaller y) from booster.cy.
                top = booster.cy - booster.length
                inside_y = top - cfg.orb_radius <= ball.y <= booster.cy + cfg.orb_radius
                if inside_x and inside_y:
                    just_entered = not booster.inside
                    booster.inside = True
                    # Continuous ascent like streamAscentVelocity (up = negative vy).
                    ascent = -ball.vy  # convert to game-style positive-up
                    if just_entered:
                        ascent = max(ascent, cfg.stream_entry_impulse)
                        self._collect_booster(booster)
                    ascent = max(cfg.stream_min_ascent, ascent + cfg.stream_accel * dt)
                    ball.vy = -min(MAX_ORB_VY, ascent)
                    ball.energy = min(100.0, ball.energy + cfg.stream_energy_per_sec * dt)
                    self.bonus = min(4000.0, self.bonus + cfg.stream_score_per_sec * dt)
                    self.booster_collected = "stream"
                else:
                    booster.inside = False
                continue

            if booster.used:
                continue
            # game.js: surge collectable any direction; drag only while rising.
            rising = ball.vy < 0
            if booster.type == "drag" and not rising:
                continue
            half_w = booster.width / 2
            half_h = booster.height / 2
            if (
                abs(ball.x - booster.cx) <= half_w + cfg.orb_radius * 0.35
                and abs(ball.y - booster.cy) <= half_h + cfg.orb_radius * 0.7
            ):
                self._collect_booster(booster)

    @property
    def boost_level(self) -> float:
        return min(1.0, (self.ball.energy + self.ball.reserve) / 100.0)

    @property
    def can_boost(self) -> bool:
        return self.ball.energy + self.ball.reserve >= self.config.boost_cost

    def nearest_platform_below(self) -> tuple[float, float, float, float]:
        ball = self.ball
        nearest: SimPlatform | None = None
        for platform in self.platforms:
            if platform.cy >= ball.y - 8:
                continue
            if nearest is None or platform.cy > nearest.cy:
                nearest = platform
        if nearest is None:
            return 0.0, 0.0, 0.0, 0.0
        dx = (nearest.cx - ball.x) / max(self.config.width, 1.0)
        dy = (ball.y - nearest.cy) / max(self.config.height, 1.0)
        wear = nearest.receptions / max(nearest.bounce_limit, 1)
        width = nearest.width / max(self.config.width, 1.0)
        return dx, dy, wear, width

    def nearest_platform_above(self) -> tuple[float, float, float, float, str]:
        ball = self.ball
        nearest: SimPlatform | None = None
        for platform in self.platforms:
            if platform.cy <= ball.y + 8:
                continue
            if nearest is None or platform.cy < nearest.cy:
                nearest = platform
        if nearest is None:
            return 0.0, 0.0, 0.0, 0.0, "neutral"
        dx = (nearest.cx - ball.x) / max(self.config.width, 1.0)
        dy = (nearest.cy - ball.y) / max(self.config.height, 1.0)
        wear = nearest.receptions / max(nearest.bounce_limit, 1)
        width = nearest.width / max(self.config.width, 1.0)
        return dx, dy, wear, width, "neutral"

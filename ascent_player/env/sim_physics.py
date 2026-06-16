from __future__ import annotations

from dataclasses import dataclass, field
import random

import numpy as np

from ascent_player.env.score_rules import combo_multiplier, live_score

GRAVITY = 820.0
JUMP_FORCE = 640.0
BOOST_FORCE = 520.0
BOOST_COST = 14.0
ENERGY_REGEN = 5.0
MAX_ORB_VY = 1500.0
ORB_SPEED = 230.0
TIER_WEIGHT = 1.0


@dataclass(slots=True)
class SimPhysicsConfig:
    width: int = 640
    height: int = 360
    gravity: float = GRAVITY
    jump_force: float = JUMP_FORCE
    boost_force: float = BOOST_FORCE
    boost_cost: float = BOOST_COST
    energy_regen_per_sec: float = ENERGY_REGEN
    orb_radius: float = 14.0
    platform_min_width: float = 90.0
    platform_max_width: float = 180.0
    platform_height: float = 10.0
    platform_spacing_y: float = 52.0
    bounce_limit: int = 3
    death_margin_below_camera: float = 80.0
    bonus_per_landing: float = 130.0
    surge_boost: float = 200.0
    surge_energy: float = 34.0
    stream_energy_per_sec: float = 32.0
    drag_slow: float = 0.48
    drag_energy: float = -18.0
    booster_spawn_interval: float = 2.6
    seed: int | None = None


@dataclass(slots=True)
class SimPlatform:
    cx: float
    cy: float
    width: float
    height: float
    receptions: int = 0
    bounce_limit: int = 3
    is_hazard: bool = False


@dataclass(slots=True)
class SimBooster:
    type: str
    cx: float
    cy: float
    width: float
    height: float
    used: bool = False


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

    def __post_init__(self) -> None:
        self.rng = random.Random(self.config.seed)
        self.reset()

    def reset(self) -> None:
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
        start_y = self.ball.y
        self.next_platform_y = start_y - 36.0
        for index in range(14):
            self._spawn_platform(bias_x=self.ball.x if index < 4 else None)
        floor = SimPlatform(
            cx=self.ball.x,
            cy=self.ball.y + 28.0,
            width=cfg.width * 0.62,
            height=cfg.platform_height,
            bounce_limit=99,
        )
        self.platforms.append(floor)

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

    def _spawn_platform(self, bias_x: float | None = None) -> None:
        cfg = self.config
        width = self.rng.uniform(cfg.platform_min_width, cfg.platform_max_width)
        margin = cfg.width * 0.07
        if bias_x is not None:
            spread = min(120.0, cfg.width * 0.18)
            x = float(
                np.clip(
                    bias_x + self.rng.uniform(-spread, spread),
                    margin + width / 2,
                    cfg.width - margin - width / 2,
                )
            )
        else:
            x = self.rng.uniform(margin + width / 2, cfg.width - margin - width / 2)
        self.next_platform_y -= self.rng.uniform(
            cfg.platform_spacing_y * 0.85,
            cfg.platform_spacing_y * 1.15,
        )
        self.platforms.append(
            SimPlatform(
                cx=x,
                cy=self.next_platform_y,
                width=width,
                height=cfg.platform_height,
                bounce_limit=cfg.bounce_limit,
            )
        )

    def _spawn_booster(self) -> None:
        cfg = self.config
        booster_type = self.rng.choice(["surge", "stream", "drag"])
        width = 64.0 if booster_type == "stream" else 30.0 if booster_type == "surge" else 120.0
        x = self.rng.uniform(width / 2 + 20, cfg.width - width / 2 - 20)
        y = self.camera_y - self.rng.uniform(40, cfg.height * 0.5)
        self.boosters.append(
            SimBooster(
                type=booster_type,
                cx=x,
                cy=y,
                width=width,
                height=16.0,
            )
        )

    def _collect_booster(self, booster: SimBooster) -> None:
        if booster.used:
            return
        booster.used = True
        self.booster_collected = booster.type
        ball = self.ball
        if booster.type == "surge":
            ball.vy = min(-200.0, ball.vy - self.config.surge_boost)
            ball.energy = min(100.0, ball.energy + self.config.surge_energy)
            self.bonus = min(4000.0, self.bonus + 130.0)
            self.combo = min(self.combo + 1, 999)
        elif booster.type == "stream":
            ball.vy = min(-240.0, ball.vy - 420.0)
            ball.energy = min(100.0, ball.energy + self.config.stream_energy_per_sec * 0.25)
            self.bonus = min(4000.0, self.bonus + 12.0)
        elif booster.type == "drag":
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

        target_vx = 0.0
        if move_left:
            target_vx -= ORB_SPEED
        if move_right:
            target_vx += ORB_SPEED
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

        ball.energy = min(100.0, ball.energy + cfg.energy_regen_per_sec * dt)

        self._resolve_platform_collisions()
        self._resolve_booster_collisions()

        if ball.y < self.camera_y + cfg.height * 0.42:
            self.camera_y = ball.y - cfg.height * 0.42

        self.booster_timer += dt
        if self.booster_timer >= cfg.booster_spawn_interval:
            self.booster_timer = 0.0
            self._spawn_booster()

        self._sync_score()

        while self.next_platform_y > self.camera_y - cfg.height:
            self._spawn_platform()

        self.platforms = [
            platform
            for platform in self.platforms
            if platform.cy < self.camera_y + cfg.height + 140.0
        ]
        self.boosters = [
            booster
            for booster in self.boosters
            if not booster.used and booster.cy < self.camera_y + cfg.height + 80.0
        ]

        return ball.y > self.camera_y + cfg.height + cfg.death_margin_below_camera

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
                ball.vy = -cfg.jump_force
                platform.receptions += 1
                self._land_on_platform()
                if platform.receptions >= platform.bounce_limit:
                    self.platforms.remove(platform)
                break

    def _resolve_booster_collisions(self) -> None:
        cfg = self.config
        ball = self.ball
        for booster in self.boosters:
            if booster.used:
                continue
            half_w = booster.width / 2
            half_h = booster.height / 2
            if (
                abs(ball.x - booster.cx) <= half_w + cfg.orb_radius
                and abs(ball.y - booster.cy) <= half_h + cfg.orb_radius
                and ball.vy > 0
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

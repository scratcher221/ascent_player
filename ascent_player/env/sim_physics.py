from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

import numpy as np

from ascent_player.env.platform_detector import Platform

STREAK_SCALE = 0.625


def combo_multiplier(combo: int) -> float:
    """Match ascent game.js comboMult()."""
    if combo <= 0:
        return 1.0
    return 1.0 + math.floor(combo / 3) * 0.5


@dataclass(slots=True)
class SimPhysicsConfig:
    width: int = 640
    height: int = 360
    gravity: float = 1180.0
    horizontal_accel: float = 2200.0
    horizontal_drag: float = 0.91
    max_horizontal_speed: float = 340.0
    boost_impulse: float = -480.0
    boost_cost: float = 14.0
    energy_regen_per_sec: float = 20.0
    orb_radius: float = 12.0
    platform_min_width: float = 100.0
    platform_max_width: float = 200.0
    platform_height: float = 10.0
    platform_spacing_y: float = 54.0
    hazard_chance: float = 0.06
    death_margin_below_camera: float = 80.0
    score_per_pixel: float = 0.48
    bonus_per_landing: float = 28.0
    seed: int | None = None


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
    platforms: list[Platform] = field(default_factory=list)
    camera_y: float = 0.0
    origin_y: float = 0.0
    max_altitude: float = 0.0
    score: int = 0
    bank_total: int = 0
    bonus_reservoir: float = 0.0
    combo: int = 0
    max_combo: int = 0
    streak: int = 0
    platform_landed: bool = False
    rng: random.Random = field(init=False)
    next_platform_y: float = 0.0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.config.seed)
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        self.ball = SimBall(x=cfg.width * 0.5, y=cfg.height * 0.72)
        self.platforms = []
        self.camera_y = 0.0
        self.origin_y = self.ball.y
        self.max_altitude = 0.0
        self.score = 0
        self.bank_total = 0
        self.bonus_reservoir = 0.0
        self.combo = 0
        self.max_combo = 0
        self.streak = 0
        self.platform_landed = False
        start_y = self.ball.y
        self.next_platform_y = start_y - 36.0
        for index in range(12):
            self._spawn_platform(bias_x=self.ball.x if index < 4 else None)
        floor = Platform(
            cx=self.ball.x,
            cy=self.ball.y + 28.0,
            width=cfg.width * 0.62,
            height=cfg.platform_height,
            is_hazard=False,
        )
        self.platforms.append(floor)

    @property
    def score_multiplier(self) -> float:
        return combo_multiplier(self.combo)

    def streak_score(self) -> int:
        return int(
            round(
                self.bonus_reservoir
                * self.score_multiplier
                * STREAK_SCALE
            )
        )

    def _sync_score(self) -> None:
        altitude = max(0.0, self.origin_y - self.ball.y)
        if altitude > self.max_altitude:
            self.max_altitude = altitude
        altitude_score = int(altitude * self.config.score_per_pixel)
        self.score = max(self.bank_total + self.streak_score(), altitude_score)

    def break_combo(self) -> None:
        self.bank_total += self.streak_score()
        self.bonus_reservoir = 0.0
        self.combo = 0
        self.streak = 0
        self._sync_score()

    def _land_on_platform(self) -> None:
        self.platform_landed = True
        self.combo = min(self.combo + 1, 999)
        self.max_combo = max(self.max_combo, self.combo)
        self.bonus_reservoir += self.config.bonus_per_landing
        new_streak = self.combo // 10
        if new_streak > self.streak:
            self.streak = new_streak
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
        is_hazard = self.rng.random() < cfg.hazard_chance
        self.platforms.append(
            Platform(
                cx=x,
                cy=self.next_platform_y,
                width=width,
                height=cfg.platform_height,
                is_hazard=is_hazard,
            )
        )

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

        if move_left:
            ball.vx -= cfg.horizontal_accel * dt
        if move_right:
            ball.vx += cfg.horizontal_accel * dt
        ball.vx = float(np.clip(ball.vx, -cfg.max_horizontal_speed, cfg.max_horizontal_speed))
        ball.vx *= cfg.horizontal_drag

        total_energy = ball.energy + ball.reserve
        if jump and total_energy >= cfg.boost_cost:
            ball.vy += cfg.boost_impulse
            remaining = cfg.boost_cost
            from_reserve = min(ball.reserve, remaining)
            ball.reserve -= from_reserve
            remaining -= from_reserve
            ball.energy = max(0.0, ball.energy - remaining)

        ball.vy += cfg.gravity * dt
        ball.x += ball.vx * dt
        ball.y += ball.vy * dt
        ball.x = float(np.clip(ball.x, cfg.orb_radius, cfg.width - cfg.orb_radius))

        ball.energy = min(100.0, ball.energy + cfg.energy_regen_per_sec * dt)

        self._resolve_platform_collisions()

        if ball.y < self.camera_y + cfg.height * 0.38:
            self.camera_y = ball.y - cfg.height * 0.38

        self._sync_score()

        while self.next_platform_y > self.camera_y - cfg.height:
            self._spawn_platform()

        self.platforms = [
            platform
            for platform in self.platforms
            if platform.cy < self.camera_y + cfg.height + 140.0
        ]

        fell = ball.y > self.camera_y + cfg.height + cfg.death_margin_below_camera
        return fell

    def _resolve_platform_collisions(self) -> None:
        cfg = self.config
        ball = self.ball
        for platform in self.platforms:
            half_w = platform.width / 2
            half_h = max(2.0, platform.height / 2)
            within_x = abs(ball.x - platform.cx) <= half_w + cfg.orb_radius * 0.4
            top = platform.cy - half_h
            if not within_x or ball.vy <= 0:
                continue
            if ball.y + cfg.orb_radius >= top and ball.y + cfg.orb_radius <= top + 18.0:
                if platform.is_hazard:
                    self.break_combo()
                    ball.y = self.camera_y + cfg.height + cfg.death_margin_below_camera + 5.0
                    return
                ball.y = top - cfg.orb_radius
                ball.vy = min(0.0, ball.vy * 0.12)
                self._land_on_platform()

    @property
    def boost_level(self) -> float:
        return min(1.0, (self.ball.energy + self.ball.reserve) / 100.0)

    @property
    def can_boost(self) -> bool:
        return self.ball.energy + self.ball.reserve >= self.config.boost_cost

from __future__ import annotations

from dataclasses import dataclass, field
import random

import numpy as np

from ascent_player.env.platform_detector import Platform


@dataclass(slots=True)
class SimPhysicsConfig:
    width: int = 640
    height: int = 360
    gravity: float = 1450.0
    horizontal_accel: float = 2200.0
    horizontal_drag: float = 0.88
    max_horizontal_speed: float = 340.0
    boost_impulse: float = -520.0
    boost_cost: float = 14.0
    energy_regen_per_sec: float = 18.0
    orb_radius: float = 12.0
    platform_min_width: float = 70.0
    platform_max_width: float = 180.0
    platform_height: float = 8.0
    platform_spacing_y: float = 72.0
    hazard_chance: float = 0.12
    death_margin_below_camera: float = 80.0
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
    max_height: float = 0.0
    score: int = 0
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
        self.max_height = 0.0
        self.score = 0
        start_y = self.ball.y
        self.next_platform_y = start_y - 40.0
        for _ in range(10):
            self._spawn_platform()
        floor = Platform(
            cx=self.ball.x,
            cy=self.ball.y + 28.0,
            width=cfg.width * 0.55,
            height=cfg.platform_height,
            is_hazard=False,
        )
        self.platforms.append(floor)

    def _spawn_platform(self) -> None:
        cfg = self.config
        width = self.rng.uniform(cfg.platform_min_width, cfg.platform_max_width)
        margin = cfg.width * 0.08
        x = self.rng.uniform(margin + width / 2, cfg.width - margin - width / 2)
        self.next_platform_y -= self.rng.uniform(
            cfg.platform_spacing_y * 0.75,
            cfg.platform_spacing_y * 1.35,
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

        if ball.y < self.camera_y + cfg.height * 0.35:
            self.camera_y = ball.y - cfg.height * 0.35

        world_height = max(0.0, (self.camera_y + cfg.height * 0.72) - ball.y)
        if world_height > self.max_height:
            self.max_height = world_height
            self.score = int(self.max_height)

        while self.next_platform_y > self.camera_y - cfg.height:
            self._spawn_platform()

        self.platforms = [
            platform
            for platform in self.platforms
            if platform.cy < self.camera_y + cfg.height + 120.0
        ]

        fell = ball.y > self.camera_y + cfg.height + cfg.death_margin_below_camera
        return fell

    def _resolve_platform_collisions(self) -> None:
        cfg = self.config
        ball = self.ball
        for platform in self.platforms:
            half_w = platform.width / 2
            half_h = max(2.0, platform.height / 2)
            within_x = abs(ball.x - platform.cx) <= half_w + cfg.orb_radius * 0.35
            top = platform.cy - half_h
            if not within_x or ball.vy <= 0:
                continue
            if ball.y + cfg.orb_radius >= top and ball.y + cfg.orb_radius <= top + 14.0:
                if platform.is_hazard:
                    ball.y = self.camera_y + cfg.height + cfg.death_margin_below_camera + 5.0
                    return
                ball.y = top - cfg.orb_radius
                ball.vy = min(0.0, ball.vy * 0.15)

    @property
    def boost_level(self) -> float:
        return min(1.0, (self.ball.energy + self.ball.reserve) / 100.0)

    @property
    def can_boost(self) -> bool:
        return self.ball.energy + self.ball.reserve >= self.config.boost_cost

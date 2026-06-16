"""Mechanics-first curriculum stages M0–M6."""

from __future__ import annotations

from dataclasses import dataclass, field

from ascent_player.config import AppConfig


@dataclass(slots=True)
class CurriculumMetrics:
    survival_steps: list[int] = field(default_factory=list)
    landing_rates: list[float] = field(default_factory=list)
    avg_heights: list[float] = field(default_factory=list)
    avg_combos: list[float] = field(default_factory=list)
    avg_scores: list[float] = field(default_factory=list)

    def record_episode(
        self,
        *,
        steps: int,
        bounces: int,
        height: float,
        combo: int,
        score: float,
    ) -> None:
        self.survival_steps.append(steps)
        rate = bounces / max(steps, 1)
        self.landing_rates.append(min(1.0, rate * 8.0))
        self.avg_heights.append(height)
        self.avg_combos.append(float(combo))
        self.avg_scores.append(score)

    def rolling(self, values: list[float], window: int = 10) -> float:
        if not values:
            return 0.0
        sample = values[-window:]
        return float(sum(sample) / len(sample))


def mechanics_stage_from_metrics(metrics: CurriculumMetrics, config: AppConfig) -> str:
    mc = config.mechanics_curriculum
    if not metrics.survival_steps:
        return "M0"
    survival = metrics.rolling([float(v) for v in metrics.survival_steps])
    landing = metrics.rolling(metrics.landing_rates)
    height = metrics.rolling(metrics.avg_heights)
    combo = metrics.rolling(metrics.avg_combos)
    score = metrics.rolling(metrics.avg_scores)

    if score >= mc.stage_m6_min_score:
        return "M6"
    if score >= mc.stage_m5_min_score:
        return "M5"
    if combo >= mc.stage_m4_min_combo:
        return "M4"
    if height >= mc.stage_m3_min_height:
        return "M3"
    if landing >= mc.stage_m2_min_landing_rate:
        return "M2"
    if survival >= mc.stage_m1_min_survival_steps:
        return "M1"
    return "M0"

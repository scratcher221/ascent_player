"""Small, testable guards for fixed-seed promotion and fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass


def confirmed_promotion(
    *,
    mean_score: float,
    min_score: float,
    floor_mean: float,
    target_min: float,
) -> bool:
    """Require a real long-eval mean win without a weak worst episode."""
    return (
        float(mean_score) > float(floor_mean)
        and float(min_score) >= float(target_min)
    )


@dataclass(slots=True)
class FineTuneReadiness:
    """Open the TD fine-tune gate only after consecutive near-floor rounds."""

    floor_ratio: float = 0.95
    required_rounds: int = 2
    consecutive_rounds: int = 0

    def observe(self, mean_score: float, floor_mean: float) -> bool:
        threshold = float(floor_mean) * float(self.floor_ratio)
        if float(mean_score) >= threshold:
            self.consecutive_rounds += 1
        else:
            self.consecutive_rounds = 0
        return self.ready

    @property
    def ready(self) -> bool:
        return self.consecutive_rounds >= max(1, int(self.required_rounds))

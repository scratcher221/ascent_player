"""Pure scoring math mirroring game/.../score-rules.js."""

from __future__ import annotations

STYLE_WEIGHT = 0.30
COMBO_CAP = 48
MAX_COMBO_MULT = 5.0


def combo_multiplier(combo: int) -> float:
    c = min(max(int(combo), 0), COMBO_CAP) if combo > 0 else 0
    if c <= 0:
        return 1.0
    return 1.0 + (c / COMBO_CAP) * (MAX_COMBO_MULT - 1.0)


def live_style(
    *,
    bonus: float = 0.0,
    combo: int = 0,
    anomaly_mult: float = 1.0,
) -> float:
    return STYLE_WEIGHT * float(bonus) * combo_multiplier(combo) * float(anomaly_mult)


def live_score(
    *,
    height: float = 0.0,
    bank_style: float = 0.0,
    bonus: float = 0.0,
    combo: int = 0,
    tier_weight: float = 1.0,
    anomaly_mult: float = 1.0,
) -> int:
    base = float(height) / 5.0
    style = float(bank_style) + live_style(
        bonus=bonus,
        combo=combo,
        anomaly_mult=anomaly_mult,
    )
    return round((base + style) * float(tier_weight))


def final_score_after_bank(previous_max: float, current: float) -> int:
    return int(max(float(previous_max), float(current)))

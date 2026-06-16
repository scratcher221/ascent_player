// Pure scoring math — the single source of truth for how a point is earned, kept side-effect-free so
// it can be unit-tested in isolation. game.js binds these to the live orb state; the worker mirrors the
// tier weights in its plausibility bounds. Invariant: the score the HUD shows every frame IS the final
// score — nothing hidden is added at crash. Tier is a prestige weight applied ONCE, uniformly.

export const STYLE_WEIGHT = 0.30; // style's share dial (boosters/combo). Tuned so a GOOD run lands ≈ 40%
                                  // style / 60% climb (upper end of "climb first"); casual runs lean more on
                                  // the climb, great runs a bit more on style (the share rises with skill —
                                  // structural). Provisional: confirm against the in-game base-vs-style log.
export const COMBO_CAP = 48;      // combo at which comboMultiplier saturates (→ MAX_COMBO_MULT).
export const MAX_COMBO_MULT = 5;  // bounded so a single streak can't dwarf the climb (was unbounded ×167).

// Bounded, linear combo multiplier: 1.0 at combo 0 → MAX_COMBO_MULT at/above COMBO_CAP.
export function comboMultiplier(combo) {
  const c = Number.isFinite(combo) && combo > 0 ? Math.min(combo, COMBO_CAP) : 0;
  return 1 + (c / COMBO_CAP) * (MAX_COMBO_MULT - 1);
}

// Live (unsecured) style from the reservoir, PRE-tier. comboMult and the anomaly multiplier amplify it,
// both bounded, so style stays a topping on the climb rather than the whole score.
export function liveStyle({ bonus = 0, combo = 0, anomalyMult = 1 }) {
  return STYLE_WEIGHT * (Number(bonus) || 0) * comboMultiplier(combo) * (Number(anomalyMult) || 1);
}

// The score shown every frame: (climb + banked style + live style) × tier weight, applied exactly once.
export function liveScore({ height = 0, bankStyle = 0, bonus = 0, combo = 0, tierWeight = 1, anomalyMult = 1 }) {
  const base = (Number(height) || 0) / 5;
  const style = (Number(bankStyle) || 0) + liveStyle({ bonus, combo, anomalyMult });
  return Math.round((base + style) * (Number(tierWeight) || 1));
}

export function finalScoreAfterBank(previousMaxScore, currentScoreAfterBank) {
  return Math.max(Number(previousMaxScore) || 0, Number(currentScoreAfterBank) || 0);
}

// Shop purchases boost by skin rarity, expressed as an equivalent XRD buy so the boost runs through
// impactStrength() with the same market thresholds as token trades and rescales if those change.
const SHOP_IMPACT_XRD_EQUIVALENT = Object.freeze({ RARE: 500, EPIC: 1000, LEGENDARY: 5000, MYTHIC: 10000 });

export function shopImpactXrdEquivalent(rarity) {
  return SHOP_IMPACT_XRD_EQUIVALENT[rarity] || 0;
}

export function applySellDebuff({
  velocity,
  currentStrength,
  currentUntil,
  now,
  impactStrength,
  velocityPct,
  durationMs,
  maxDurationMs
}) {
  const strength = Math.min(1,impactStrength/3);
  return {
    velocity: velocity*(1-velocityPct/100*strength),
    strength: Math.max(currentStrength,strength),
    until: Math.min(Math.max(currentUntil,now)+durationMs,now+maxDurationMs)
  };
}

export const DEFAULT_BASE_BOUNCE_LIMIT = 3;

export function normalizeBaseBounceLimit(value, fallback = DEFAULT_BASE_BOUNCE_LIMIT) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(1, Math.min(100, Math.floor(number))) : fallback;
}

export function neutralPlatform(platform, bounceLimit) {
  return {
    ...platform,
    type: "neutral",
    receptions: 0,
    bounceLimit: normalizeBaseBounceLimit(bounceLimit)
  };
}

export function receivePlatform(platform) {
  if (platform.type !== "neutral") return false;
  platform.receptions = (platform.receptions || 0) + 1;
  return platform.receptions >= platform.bounceLimit;
}

export function awardPlatformCombo(platform) {
  if (platform.comboAwarded) return false;
  platform.comboAwarded = true;
  return true;
}

export function platformOpacity(platform) {
  if (platform.type !== "neutral" || !platform.bounceLimit) return 1;
  return Math.max(0.2,1-(platform.receptions || 0)/platform.bounceLimit*0.8);
}

export function shouldSpawnLivePlatform(pressure) {
  return pressure > 0;
}

export function shouldSpawnNotableBuyPlatform(amount, firstThreshold) {
  return Number(amount) > Number(firstThreshold);
}

export function notableBuyPlatformStats(strength, scale = 1) {
  const boundedStrength = Math.max(1, Math.min(4, Number(strength) || 1));
  return {
    width: scale * Math.min(220, 95 + boundedStrength * 30),
    bounce: Math.min(2.4, 1.05 + boundedStrength * 0.22)
  };
}

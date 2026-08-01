export const ANOMALY_TYPES = ["liquidityVoid", "shortSqueeze", "darkPool"];

export const ANOMALIES = Object.freeze({
  liquidityVoid: Object.freeze({
    id: "liquidityVoid",
    label: "LIQUIDITY VOID",
    shortLabel: "VOID",
    durationSeconds: 12,
    color: "#bfe6ff",
    gravityMode: "zero",
    exitEnergy: 100,
    exitBonus: 180,
    scoreMult: 1
  }),
  shortSqueeze: Object.freeze({
    id: "shortSqueeze",
    label: "SHORT SQUEEZE",
    shortLabel: "SQUEEZE",
    durationSeconds: 5,
    color: "#ffb0d8",
    gravityMode: "hazardsOnly",
    exitBonus: 260,
    scoreMult: 2,
    hazardCount: 4
  }),
  darkPool: Object.freeze({
    id: "darkPool",
    label: "DARK POOL",
    shortLabel: "DARK",
    durationSeconds: 10,
    color: "#c0c8ff",
    gravityMode: "normal",
    exitBonus: 220,
    scoreMult: 1
  })
});

export function anomalyDefinition(type) {
  return ANOMALIES[type] || null;
}

export function canTriggerAnomaly(gameState) {
  return gameState === "playing";
}

export function anomalyDuration(type) {
  return anomalyDefinition(type)?.durationSeconds || 0;
}

export function anomalyScoreMultiplier(type) {
  return anomalyDefinition(type)?.scoreMult || 1;
}

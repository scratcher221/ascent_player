export const BOOSTER_TYPES = ["surge","stream","drag"];

export function boosterDifficulty(height,viewportHeight,progressionScreens) {
  const screens = Math.max(1,Number(progressionScreens) || 1);
  const viewport = Math.max(1,Number(viewportHeight) || 1);
  return Math.max(0,Math.min(1,(Number(height) || 0) / viewport / screens));
}

export function boosterSpawnInterval(boosters,height,viewportHeight) {
  const difficulty = boosterDifficulty(height,viewportHeight,boosters.progressionScreens);
  return boosters.startIntervalMs + (boosters.minIntervalMs - boosters.startIntervalMs) * difficulty;
}

export function boosterWeights(boosters,pressure,height,viewportHeight) {
  const difficulty = boosterDifficulty(height,viewportHeight,boosters.progressionScreens);
  const threshold = Math.max(0.000001,Number(boosters.pressureThreshold) || 0.6);
  const marketBias = Math.max(-1,Math.min(1,(Number(pressure) || 0) / threshold));
  return {
    surge: Math.max(0.01,boosters.surge.spawnWeight * (1 - Math.abs(marketBias) * 0.55)),
    stream: Math.max(0.01,boosters.stream.spawnWeight * (1 + Math.max(0,marketBias) * 1.4)),
    drag: Math.max(0.01,boosters.drag.spawnWeight * (1 + Math.max(0,-marketBias) * 2 + difficulty * boosters.drag.heightWeightBonus))
  };
}

export function pickBoosterType(weights,random = Math.random()) {
  const total = BOOSTER_TYPES.reduce((sum,type) => sum + Math.max(0,Number(weights[type]) || 0),0);
  if (!total) return "surge";
  let cursor = Math.max(0,Math.min(0.999999999,random)) * total;
  for (const type of BOOSTER_TYPES) {
    cursor -= Math.max(0,Number(weights[type]) || 0);
    if (cursor <= 0) return type;
  }
  return BOOSTER_TYPES.at(-1);
}

export function streamAscentVelocity(velocity,stream,dt,{ entered=false }={}) {
  const entryImpulse = Math.max(0,Number(stream.entryImpulse) || 0);
  const minAscentSpeed = Math.max(0,Number(stream.minAscentSpeed) || 0);
  const accel = Math.max(0,Number(stream.accel) || 0);
  const initial = entered ? Math.max(Number(velocity) || 0,entryImpulse) : Number(velocity) || 0;
  return Math.max(minAscentSpeed,initial + accel * Math.max(0,Number(dt) || 0));
}

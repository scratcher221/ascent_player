/* Injected into the live ASCENT bundle for Ascent Player training (regular climb only). */
const __ASCENT_AGENT_MODE__ = Boolean(window.CHART_TRIAL_CONFIG?.agentMode);

window.__ASCENT_SET_RUN_SEED__ = (seed, lock=true) => {
  window.CHART_TRIAL_CONFIG = window.CHART_TRIAL_CONFIG || {};
  if (seed == null || seed === "") {
    window.CHART_TRIAL_CONFIG.runSeed = null;
    window.CHART_TRIAL_CONFIG.lockRunSeed = false;
    window.__ASCENT_RUN_SEED__ = undefined;
    return null;
  }
  window.CHART_TRIAL_CONFIG.runSeed = seed;
  window.CHART_TRIAL_CONFIG.lockRunSeed = Boolean(lock);
  const n = Number(seed);
  if (Number.isFinite(n)) window.__ASCENT_RUN_SEED__ = n >>> 0;
  return window.__ASCENT_RUN_SEED__;
};

window.__ASCENT_GET_RUN_SEED__ = () =>
  window.__ASCENT_RUN_SEED__ ?? __ascentResolveLocalSeed__();

function __ascentExportAgentState__(game) {
  if (!__ASCENT_AGENT_MODE__) return;
  const orb = game?.orb;
  if (!orb?.worldY) return;
  const W = game.W;
  const H = game.H;
  const cameraY = game.cameraY;
  const platforms = game.platforms || [];
  const boosters = game.boosters || [];
  const viewTop = cameraY - 50;
  const viewBot = cameraY + H + 50;
  const visiblePlatforms = platforms
    .filter((p) => p.worldY >= viewTop && p.worldY <= viewBot)
    .slice(0, 24)
    .map((p) => ({
      x: p.x + p.width / 2,
      worldY: p.worldY,
      width: p.width,
      type: p.type || "neutral",
      receptions: p.receptions ?? 0,
      bounceLimit: p.bounceLimit ?? 3,
    }));
  const visibleBoosters = boosters
    .filter((b) => !b.used && b.worldY >= viewTop && b.worldY <= viewBot)
    .slice(0, 12)
    .map((b) => ({ type: b.type, x: b.x, worldY: b.worldY, w: b.w }));
  let nearest = null;
  for (const plat of platforms) {
    if (plat.worldY > orb.worldY - 8) continue;
    if (!nearest || plat.worldY > nearest.worldY) nearest = plat;
  }
  let nearestPlatformBelow = null;
  if (nearest) {
    nearestPlatformBelow = {
      dx: (nearest.x + nearest.width / 2 - orb.x) / Math.max(W, 1),
      dy: (orb.worldY - nearest.worldY) / Math.max(H, 1),
      width: nearest.width / Math.max(W, 1),
      wear: (nearest.receptions ?? 0) / Math.max(nearest.bounceLimit ?? 3, 1),
      type: nearest.type || "neutral",
    };
  }
  let nearestAbove = null;
  for (const plat of platforms) {
    if (plat.worldY <= orb.worldY + 8) continue;
    if (!nearestAbove || plat.worldY < nearestAbove.worldY) nearestAbove = plat;
  }
  let nearestPlatformAbove = null;
  if (nearestAbove) {
    nearestPlatformAbove = {
      dx: (nearestAbove.x + nearestAbove.width / 2 - orb.x) / Math.max(W, 1),
      dy: (nearestAbove.worldY - orb.worldY) / Math.max(H, 1),
      width: nearestAbove.width / Math.max(W, 1),
      wear: (nearestAbove.receptions ?? 0) / Math.max(nearestAbove.bounceLimit ?? 3, 1),
      type: nearestAbove.type || "neutral",
    };
  }
  const pickTarget = (candidates) => {
    let best = null;
    let bestScore = Infinity;
    for (const item of candidates) {
      if (!item) continue;
      const typePenalty = item.type === "sell" ? 0.35 : 0;
      const wearPenalty = Math.max(0, (item.wear ?? 0) - 0.7) * 0.5;
      const score = Math.abs(item.dx) * 2 + Math.abs(item.dy) + typePenalty + wearPenalty;
      if (score < bestScore) {
        bestScore = score;
        best = item;
      }
    }
    return best;
  };
  const vy = orb.vy ?? 0;
  const rising = vy > 20;
  const falling = vy < -20;
  const bestLandingPlatform = falling
    ? nearestPlatformBelow || nearestPlatformAbove
    : pickTarget([nearestPlatformAbove, nearestPlatformBelow]);
  const landingWindow =
    falling &&
    nearestPlatformBelow &&
    nearestPlatformBelow.dy > 0.05 &&
    nearestPlatformBelow.dy < 0.35;
  let timeToPlatform = 0;
  if (falling && nearestPlatformBelow && vy < -1) {
    const dyPx = orb.worldY - (nearest?.worldY ?? orb.worldY);
    timeToPlatform = Math.min(1, Math.max(0, dyPx / (Math.max(Math.abs(vy), 1) / H)));
  }
  const canBoostNow = (orb.energy ?? 0) + (orb.reserve ?? 0) >= 14;
  const gapBelow = nearestPlatformBelow?.dy ?? 0;
  const boostUseful =
    canBoostNow &&
    (vy < -120 || gapBelow > 0.18 || (falling && Math.abs(nearestPlatformBelow?.dx ?? 0) > 0.18));
  const danger = {
    worn: (nearestPlatformBelow?.wear ?? 0) >= 0.85 || (nearestPlatformAbove?.wear ?? 0) >= 0.85,
    sell: nearestPlatformBelow?.type === "sell" || nearestPlatformAbove?.type === "sell",
    missRisk: falling && Math.abs(nearestPlatformBelow?.dx ?? 0) > 0.18 && gapBelow < 0.4,
  };
  let nearestBooster = null;
  let bestDist = Infinity;
  for (const b of visibleBoosters) {
    const dx = b.x - orb.x;
    const dy = b.worldY - orb.worldY;
    const dist = dx * dx + dy * dy;
    if (dist < bestDist) {
      bestDist = dist;
      nearestBooster = {
        type: b.type,
        dx: dx / Math.max(W, 1),
        dy: dy / Math.max(H, 1),
      };
    }
  }
  const activeAnomaly = game.activeAnomaly;
  const anomalyHazards = game.anomalyHazards || [];
  let anomalyPortal = null;
  let nearestAnomalyHazard = null;
  if (activeAnomaly) {
    const portal = activeAnomaly.portal;
    if (portal) {
      anomalyPortal = {
        dx: (portal.x - orb.x) / Math.max(W, 1),
        dy: (portal.worldY - orb.worldY) / Math.max(H, 1),
        kind: "void_portal",
      };
    }
    let hazBest = Infinity;
    for (const hazard of anomalyHazards) {
      const dx = hazard.x - orb.x;
      const dy = hazard.worldY - orb.worldY;
      const dist = dx * dx + dy * dy;
      if (dist < hazBest) {
        hazBest = dist;
        nearestAnomalyHazard = {
          dx: dx / Math.max(W, 1),
          dy: dy / Math.max(H, 1),
          kind: hazard.kind === "voidShard" ? "void_shard" : "squeeze_spike",
        };
      }
    }
  }
  const runSeed =
    (typeof Re !== "undefined" && Re?.runSeed) ||
    window.__ASCENT_RUN_SEED__ ||
    __ascentResolveLocalSeed__() ||
    0;
  window.__ASCENT_RUN_SEED__ = runSeed >>> 0;
  const tierIndex = typeof be === "function" ? be() : typeof ie !== "undefined" ? ie : 0;
  const score = typeof co === "function" ? co(game) : 0;
  window.__ASCENT_AGENT__ = {
    state: game.state,
    runSeed: runSeed >>> 0,
    orb: {
      x: orb.x,
      worldY: orb.worldY,
      vx: orb.vx ?? 0,
      vy: orb.vy ?? 0,
      energy: orb.energy ?? 0,
      reserve: orb.reserve ?? 0,
      combo: orb.combo ?? 0,
      bonus: orb.bonus ?? 0,
      bankStyle: orb.bankStyle ?? 0,
      height: orb.height ?? 0,
      bounces: orb.bounces ?? 0,
    },
    cameraY,
    canvasW: W,
    canvasH: H,
    platforms: visiblePlatforms,
    boosters: visibleBoosters,
    activeAnomaly: activeAnomaly
      ? {
          type: activeAnomaly.type,
          remaining: activeAnomaly.remaining ?? 0,
          portal: anomalyPortal,
          nearestHazard: nearestAnomalyHazard,
        }
      : null,
    ultiCharges: [...(game.ultiCharges || [])],
    tierIndex,
    score,
    canBoost: canBoostNow,
    nearestPlatformBelow,
    nearestPlatformAbove,
    bestLandingPlatform,
    orbPhase: { rising, falling, landingWindow, airborne: Math.abs(vy) > 20 },
    timeToPlatform,
    boostUseful,
    danger,
    nearestBooster,
    stormLevel: Number(game.livePressure ?? 0),
  };
}

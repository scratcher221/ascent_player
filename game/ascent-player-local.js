/** Ascent Player local overrides (loaded after the game's config bundle). */
try {
  localStorage.setItem("ascent-cosmetics-reveal-dismissed-v4", "1");
} catch {
  /* ignore */
}

window.CHART_TRIAL_CONFIG = Object.assign(
  {},
  window.CHART_TRIAL_CONFIG || {},
  {
    // No live calendar feed locally — hide fallback candle chart (gray wicks).
    offlineMode: true,
    hideMarketChart: true,
    agentMode: true,
    devUnlockTiers: true,
    trainingTierIndex: 0,
    calendarApiBaseUrl: "https://ascent.xrd.workers.dev",
    runSeed: null,
    lockRunSeed: true,
  },
);

/** Ascent Player local overrides (loaded after the game's config bundle). */
window.CHART_TRIAL_CONFIG = Object.assign(
  {
    offlineMode: true,
    agentMode: true,
    devUnlockTiers: true,
    trainingTierIndex: 0,
    calendarApiBaseUrl: window.location.origin,
    runSeed: null,
    lockRunSeed: true,
  },
  window.CHART_TRIAL_CONFIG || {},
);

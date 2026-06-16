window.CHART_TRIAL_CONFIG = {
  offlineMode: true,
  // Same-origin routes are served by offline-fetch.js from bundled snapshots.
  calendarApiBaseUrl: window.location.origin,
  // Unlock every tier locally so training agents can reach all content.
  devUnlockTiers: true,
};

/* ── Radix / ASCENT constants (kept for shop UI labels; wallet is stubbed offline) ── */
const ASCENT_DAPP_DEFINITION = "account_rdx12yhd59pzstcekvfpeu7dzvln20q9ekyfcd7k34wzlkulynavgnaxgx";
const ASCENT_SHOP_COMPONENT  = "component_rdx1cz92wt309k6qrrpcs0fvz3u2wkmj0jmjgkml8qkd0aghdfq7fa5c9m";
const ASCENT_TOKEN_RESOURCE  = "resource_rdx1t46jpmzf97s7q5h4tv42hcjyalhq84znevngtsul5wcumxfdprnlp3";
const RADIX_NETWORK_ID       = 1;

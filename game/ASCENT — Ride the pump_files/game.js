"use strict";

import { TIERS, TIER_BAR } from "./tiers.js?v=orbit-sparks-1";
import { applySellDebuff, shopImpactXrdEquivalent } from "./impact-rules.js";
import { awardPlatformCombo, neutralPlatform, notableBuyPlatformStats, platformOpacity, receivePlatform, shouldSpawnLivePlatform, shouldSpawnNotableBuyPlatform } from "./platform-rules.js";
import { boosterDifficulty, boosterSpawnInterval, boosterWeights, pickBoosterType, streamAscentVelocity } from "./booster-rules.js";
import { advanceUltiCharge, canActivateUltiSlot, ultiCooldownRemaining, aegisShieldActive, tickHelixTimer } from "./ulti-rules.js";
import { DEFAULT_OPTIONS, normalizeOptions, normalizeUltiKey, normalizeControlKey } from "./options-rules.js";
import { buildHowToPlay, setHowToUnlockState, startHowToPlayPreviews, stopHowToPlayPreviews } from "./how-to-play.js";
import { ANOMALY_TYPES, anomalyDefinition, anomalyScoreMultiplier, canTriggerAnomaly } from "./event-rules.js";
import { finalScoreAfterBank, comboMultiplier, liveStyle, liveScore, STYLE_WEIGHT } from "./score-rules.js";
import { drawTierSupport, drawTierResist, drawTierSurge, drawTierStream, tierVisualsInvalidate } from "./tier-visuals.js?v=tier-world-2";
import { drawTierScene } from "./tier-scenes.js?v=tier-world-3";
// Side-effect imports: tier renderers register into RENDERERS. Must come after
// the line above so tier-visuals.js is fully evaluated first.
// NOTE: the ?v token must match the one used inside the t2-t5 / t6-t10 /
// tier-parallax files below, or the browser loads two copies of
// tier-visuals.js and RENDERERS splits.
import "./tier-visuals-t2-t5.js?v=tier-world-2";
import "./tier-visuals-t6-t10.js?v=tier-world-2";

// ── CANVAS ───────────────────────────────────────────────────────────────────
const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
let W = 0, H = 0, dpr = 1;
// gs = game scale: 1.0 at reference 700px height, scales all size-sensitive values
let gs = 1;
// ORB_SPEED_EFF scales with viewport width so horizontal traversal difficulty stays constant across screen sizes
let ORB_SPEED_EFF = 230;
// 1 on phone-width screens (≤420px), fading to 0 at ≥800px — shared ramp for narrow-screen
// rebalancing (orb dodge speed, red wall width). Obstacles scale with height (gs) while free
// space scales with width, so narrow screens are mechanically harder without this.
function narrowScreenRamp() { return Math.max(0, Math.min(1, (800 - W) / 380)); }

// ── CONSTANTS ────────────────────────────────────────────────────────────────
const GRAVITY        = 820;    // world-px/s², pulls orb down
const JUMP_FORCE     = 640;    // auto-bounce impulse off platform
const BOOST_FORCE    = 520;    // manual tap/SPACE boost
const BOOST_COST     = 14;     // energy per boost
const ENERGY_REGEN   = 5;      // energy/s base rate
const ORB_SPEED      = 230;    // horizontal px/s
const LIVE_SAMPLE_MS = 1500;   // ms between live price samples
const CANDLE_REFRESH_MS = 30000; // minute candles change at most once/min — refresh far less than price
const INTERVALS = {
  "1d":  { ms: 86_400_000, ociRes: "1D",  label: "1D"  },
  "4h":  { ms: 14_400_000, ociRes: "240", label: "4H"  },
  "1h":  { ms: 3_600_000,  ociRes: "60",  label: "1H"  },
  "30m": { ms: 1_800_000,  ociRes: "30",  label: "30M" },
  "5m":  { ms: 300_000,    ociRes: "5",   label: "5M"  },
  "1m":  { ms: 60_000,     ociRes: "1",   label: "1M"  },
};
let activeInterval = "4h";
const VISIBLE_CANDLE_COUNT = 24;
const MAX_CANDLE_COUNT = 180;
const MARKET_AXIS_H = 44;
const BOTTOM_BAR_H = 64;
const CALENDAR_API = window.CHART_TRIAL_CONFIG?.calendarApiBaseUrl || "http://127.0.0.1:8787";

// Shared fetch wrapper: timeout + ok-check + JSON parse (throws on failure).
async function apiFetch(url, { timeoutMs = 5000, errorLabel = "request failed", ...init } = {}) {
  const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs), ...init });
  if (!res.ok) throw new Error(`${errorLabel} ${res.status}`);
  return res.json();
}
const PREVIEW_TOKEN = new URLSearchParams(location.search).get("previewToken") || "";
// Dev-only: unlock every tier on localhost via ?devUnlockTiers (for visual QA).
// Host-gated so it can never take effect in production.
const DEV_UNLOCK_TIERS = window.CHART_TRIAL_CONFIG?.devUnlockTiers
  || (/^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname)
  && new URLSearchParams(location.search).has("devUnlockTiers"));
const AGENT_MODE = Boolean(window.CHART_TRIAL_CONFIG?.agentMode);
const LB_MAX         = 10;
const LB_ARCHIVE_MAX = 1000; // mirrors worker LEADERBOARD_ARCHIVE_MAX; ranks beyond it are unknown
const PLAYER_NAME_KEY = "ascent-player-name";
const OPTIONS_STORAGE_KEY = "ascent-options-v1";
const WALLET_CONNECT_LOCAL_KEY = "ascent-wallet-local-state-v1";
const SPAWN_SCREENS  = 1.6;    // spawn N screens above camera top
const MAX_ORB_VY     = 1500;   // cap on upward orb velocity (world-px/s)
const MAX_ORB_BONUS  = 4000;   // cap on the reservoir (unbanked bonus). High so it keeps growing
                               // the longer a chain runs uninterrupted — the "momentum" that breaking
                               // throws away. Finite to keep scores/anti-cheat bounded.
// STYLE_WEIGHT (style's share dial), comboMultiplier, liveStyle, liveScore live in score-rules.js — the
// pure, unit-tested single source of the scoring formula. They are imported above and bound to orb state
// by comboMult()/styleLive()/currentScore() below. STYLE_WEIGHT replaces the old STREAK_SCALE volume knob.
const MAX_IMPACT_AMOUNT = 1_000_000; // sanity bound for trade volume (USDT)
const IMPACT_MAX_AGE_MS = 60_000; // impacts missed during pause/menu replay on the next run, but only this fresh
const SHORT_SQUEEZE_HIT_ENERGY_LOSS = 35;
const VOID_SHARD_HIT_ENERGY_LOSS = 24;
const isMobile = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
// Computed in resize():
let PLAT_H = 10;               // platform thickness (scales with gs)
let ORB_R  = 14;               // orb base radius (scales with gs)

// ── MARKET DATA ──────────────────────────────────────────────────────────────
const ascentFallback = [
  .894441,.894443,.894443,.894443,.894443,.894443,.969110,.969110,.969111,.969112,.969113,
  .969130,.969131,.969160,.969160,.969160,.969160,.969160,.969160,.969129
];
const ASCENT_RESOURCE = "resource_rdx1t46jpmzf97s7q5h4tv42hcjyalhq84znevngtsul5wcumxfdprnlp3";
const currentMarket = "ascent";
const market = { symbol:"ASCENT", name:"ASCENT", coin:"ASCENT", quote:"XRD", resource:ASCENT_RESOURCE, decimals:18, fallback:ascentFallback };
const makeFallback = () => market.fallback.map((price,i)=>({ price, time:Date.UTC(2025,0,1)+i*18*86400000 }));
let historicalPrices = makeFallback();

// Live feed state
let latestLivePrice = null;
let livePrices = [];         // rolling {price,time}[] for chart ghost
let preloadCandles = [];
let liveCandles = [];
let liveMode = false;
let liveSampler = null;
let ociPoller = null;
let livePoll = null;       // handle to the active /live-state poll, for an immediate refresh on tab focus
let sessionGen     = 0;    // generation id: incremented on stopSession to invalidate stale callbacks
let chartBootReady = false;
let pendingTradeFlow = { buy:0, sell:0 };
let flowVolumes = [];
const LIVEPRESSURE_MIN = -2.4;
const LIVEPRESSURE_MAX =  2.4;
let livePressure = 0;        // LIVEPRESSURE_MIN..LIVEPRESSURE_MAX, positive=buy
let marketTrendPct = 0;
let sessionTrend = 0;
const DEFAULT_GAMEPLAY_CONFIG = Object.freeze({
  market:{ unit:"USDT", thresholds:[100,1000,10000] },
  trend:{ neutralPct:.5, capPct:10, bullBouncePct:25, bullBoostPct:20, bearGravityPct:35, bearAscentPct:20, refreshMs:600000 },
  platforms:{ baseBounceLimit:3 },
  boosters:{
    startIntervalMs:2600, minIntervalMs:1400, progressionScreens:18, pressureThreshold:.6,
    surge:{ spawnWeight:1, size:1, boost:200, energy:34, score:130 },
    stream:{ spawnWeight:.8, size:1, entryImpulse:420, minAscentSpeed:240, accel:900, energyPerSecond:32, scorePerSecond:12, length:220 },
    drag:{ spawnWeight:.55, heightWeightBonus:1.5, size:1, slow:.48, velocityPenalty:50, energy:-18 }
  },
  impacts:{ queueSpacingMs:500, queueCapacity:20, buyImpulse:500, sellVelocityPct:25, sellGravityPct:40, sellDurationMs:4000, sellMaxDurationMs:12000 }
});
let gameplayConfig = structuredClone(DEFAULT_GAMEPLAY_CONFIG);
let sessionGameplay = structuredClone(gameplayConfig);
let impactQueue = [];
let impactPoller = null;
// Trade-impact polling backs off when the market is quiet: 3s when trades are flowing, doubling up
// to 12s after consecutive empty polls. Resets to the floor on fresh trades, run start, or tab focus.
const IMPACT_POLL_MIN_MS = 3000;
const IMPACT_POLL_MAX_MS = 12000;
let impactPollMs = IMPACT_POLL_MIN_MS;
let lastImpactPoll = Date.now();
let lastImpactAppliedAt = 0;
let sellDebuffUntil = 0;
let sellDebuffStrength = 0;
let tierUnlockState = {
  status: "loading",
  maxUnlockedTierIndex: 0,
  maxUnlockedTier: 1,
  nextTierIndex: 1,
  nextTier: 2,
  nextThresholdXrd: null,
  progressPct: 0,
  marketCapXrd: null,
  tier2Status: { unlocked:false, source:"loading" }
};
let pendingTierRelockIndex = null;

// ── ANOMALY TEST EVENTS ─────────────────────────────────────────────────────
let activeAnomaly = null;
let anomalyHazards = [];
let deferredAnomalyBuyStrength = 0;
let deferredAnomalyBuyCount = 0;
let nextAnomalyAt = Infinity; // run-elapsed seconds at which the next auto anomaly fires
let autoAnomalyCount = 0;
let anomalyMinDelay = 0; // real-time floor between two anomalies, regardless of climb speed

// ── ULTI STATE ───────────────────────────────────────────────────────────────
let equippedUltis    = []; // tier indices of selected ultis, e.g. [1, 3, 6]
let equippedSkin      = null; // skin object from SKINS[] or null
let _skinAvgMs        = 0;   // low-pass moyenne du coût de rendu skin (~30 frames)
let ownedCosmeticIds  = new Set(); // itemIds of owned (non-badge) cosmetics
let _lastShopData     = null;      // cached shop payload for re-render after equip
let ultiCharges      = []; // charge 0–100 per slot
let ultiActiveStates = []; // state per slot: null | { tier, phase, remaining, counters }
let selectedUltiTiers = []; // transient: current selection in selection screen

function formatNumber(v) {
  return v.toFixed(6);
}
function formatCompactXrd(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "---";
  if (number >= 1_000_000_000) return `${(number / 1_000_000_000).toFixed(number >= 10_000_000_000 ? 0 : 1)}B XRD`;
  if (number >= 1_000_000) return `${(number / 1_000_000).toFixed(number >= 10_000_000 ? 0 : 1)}M XRD`;
  if (number >= 1_000) return `${(number / 1_000).toFixed(number >= 10_000 ? 0 : 1)}K XRD`;
  return `${number.toFixed(0)} XRD`;
}
function formatCompactAscent(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0) return "—";
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(n >= 10_000_000_000 ? 0 : 1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n >= 10_000_000 ? 0 : 1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(n >= 10_000 ? 0 : 1)}K`;
  return `${Math.floor(n)}`;
}
function setDisplayedPrice(price) {
  const el = document.getElementById("tickerPrice");
  if (el) el.textContent = `${formatNumber(price)} XRD`;
}

function setDisplayedTrend(changePct) {
  const el = document.getElementById("tickerDelta");
  if (!el) return;
  if (!Number.isFinite(changePct)) {
    el.textContent = "24H ---";
    el.className = "delta-neutral";
    return;
  }
  el.textContent = `24H ${changePct>=0?"+":""}${changePct.toFixed(3)}%`;
  el.className = changePct > 0 ? "delta-up" : changePct < 0 ? "delta-down" : "delta-neutral";
}

function syncMarketUi() {
  document.getElementById("tickerSymbol").textContent = market.symbol;
  document.getElementById("previewPair").textContent = "---";
}

function trendStrength(changePct, config=gameplayConfig.trend) {
  const magnitude = Math.abs(changePct);
  if (magnitude <= config.neutralPct) return 0;
  return Math.sign(changePct) * Math.min(1, (magnitude-config.neutralPct)/(config.capPct-config.neutralPct||1));
}


function impactStrength(amount, thresholds=sessionGameplay.market.thresholds) {
  const value = Math.max(0, Number(amount)||0);
  const [low,medium,high] = thresholds;
  if (value < low) return 0;
  if (value < medium) return 1+(value-low)/(medium-low||1);
  if (value < high) return 2+(value-medium)/(high-medium||1);
  return 3+(value-high)/(high||1);
}

function trendMultipliers() {
  const trend = sessionGameplay.trend;
  return sessionTrend >= 0
    ? { gravity:1, ascent:1+sessionTrend*trend.bullBouncePct/100, boost:1+sessionTrend*trend.bullBoostPct/100 }
    : { gravity:1+(-sessionTrend)*trend.bearGravityPct/100, ascent:1-(-sessionTrend)*trend.bearAscentPct/100, boost:1-(-sessionTrend)*trend.bearAscentPct/100 };
}

function applyStateToMods(state, m) {
  const { tier: t } = state;
  switch (t) {
    case 1: // INVOKE — magnetic platform attraction (handled in updateBeaconAttraction)
      m.beaconActive = true;
      break;
    case 2: // AEGIS — shield is checked directly in physics loop
      m.shieldActive = true;
      break;
    case 3: // OVERCLOCK — free boosts, speed lines maxed
      m.boostCostMult = 0;
      m.speedLineIntensity = Math.max(m.speedLineIntensity, 2);
      break;
    case 4: // CHRONO — timeScale handled via chronoRamp in update()
      m.forceChromatic = true;
      break;
    case 5: // MAGNET — large booster hitbox during active phase
      m.magnetLarge = true;
      break;
    case 6: // PILLAR — bounce bonus + sell platform pass-through
      m.bounceMult = Math.max(m.bounceMult, 1.55);
      m.pillarRedPassthrough = true;
      break;
    case 7: // RESONANCE — charge-based explosion (handled in updatePlatformCollisions)
      break;
    case 8: // HELIX — bounce, impacts, arcs, auto-oscillation
      m.bounceMult = Math.max(m.bounceMult, 1.55);
      m.impactAmp  = Math.max(m.impactAmp,  1.8);
      m.forceArcs  = true;
      break;
    case 9: // PHOENIX REBIRTH — light gravity, strong bounce, drag + sell immunity
      m.gravAdd        -= 0.75;
      m.bounceMult      = Math.max(m.bounceMult, 1.7);
      m.noDragBooster   = true;
      m.noSellDebuff    = true;
      m.forceArcs       = true;
      m.forceChromatic  = true;
      break;
  }
}

function getUltiModifiers() {
  const m = {
    gravAdd:0, bounceMult:1, boostCostMult:1, noBoost:false,
    regenMult:1, invertX:false, forceArcs:false, forceChromatic:false,
    forceNeutral:false, impactAmp:1, doubleCombo:false,
    shieldActive:false, magnetLarge:false, speedLineIntensity:1,
    noSellDebuff:false, noDragBooster:false, timeScale:1,
    beaconActive:false, pillarRedPassthrough:false,
  };
  for (const state of ultiActiveStates) {
    if (state) applyStateToMods(state, m);
  }
  return m;
}

// ── AUDIO ────────────────────────────────────────────────────────────────────
function loadOptions() {
  try { return normalizeOptions(JSON.parse(localStorage.getItem(OPTIONS_STORAGE_KEY) || "null")); }
  catch { return normalizeOptions(null); }
}
function saveOptions() {
  try {
    localStorage.setItem(OPTIONS_STORAGE_KEY,JSON.stringify({
      musicVolume,sfxVolume,musicMuted,sfxMuted,vibrationEnabled,qualityMode,trailMode,ultiKeys,
      pauseKey,boostKey,leftKey,rightKey,gamepadBindings
    }));
  } catch {}
}

const storedOptions = loadOptions();
let audioCtx = null;
let T = null; // Tone.js effect buses (SFX)
let sfxMuted   = storedOptions.sfxMuted;
let musicMuted = storedOptions.musicMuted;
let sfxVolume   = storedOptions.sfxVolume;
let musicVolume = storedOptions.musicVolume;
let vibrationEnabled = storedOptions.vibrationEnabled;
let trailMode = storedOptions.trailMode; // "all" | "basic" | "off"
let ultiKeys = [...storedOptions.ultiKeys];
let pauseKey = storedOptions.pauseKey;
let boostKey = storedOptions.boostKey;
let leftKey  = storedOptions.leftKey;
let rightKey = storedOptions.rightKey;
let gamepadBindings = structuredClone(storedOptions.gamepadBindings);
let rebindingUltiSlot = null;
let rebindingControlKey = null;
let rebindingGamepadAction = null;

function getAudioCtx() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    initToneSfx();
  }
  return audioCtx;
}
function initToneSfx() {
  if (T || typeof Tone === 'undefined') return;
  try {
    Tone.start();
    T = {};
    T.shortRev = new Tone.Reverb({ decay:0.35, wet:0.24 }).toDestination();
    T.longRev  = new Tone.Reverb({ decay:1.6,  wet:0.44 }).toDestination();
    T.dist = new Tone.Distortion({ distortion:0.45, wet:0.55 });
    T.dist.connect(T.shortRev);
    T.chorus = new Tone.Chorus({ frequency:3.2, depth:0.65, wet:0.55 });
    T.chorus.connect(T.longRev); T.chorus.start();
    T.phaser = new Tone.Phaser({ frequency:2.5, octaves:4, wet:0.55 });
    T.phaser.connect(T.shortRev);
    T.autoFilt = new Tone.AutoFilter({ frequency:5, depth:0.8, wet:0.75 });
    T.autoFilt.connect(T.shortRev); T.autoFilt.start();
    Tone.getDestination().volume.value = Tone.gainToDb(sfxVolume);
  } catch(e) { T = null; }
}
function _sfxGc(node, ms) { setTimeout(() => { try { node.dispose(); } catch(e) {} }, ms || 3000); }
const sfx = {
  bounce() {
    if (sfxMuted || !T) return;
    const s1 = new Tone.Synth({ oscillator:{type:'sine'}, envelope:{attack:.005,decay:.14,sustain:0,release:.14} }).connect(T.shortRev);
    s1.volume.value = -9; s1.triggerAttackRelease('G3','16n'); _sfxGc(s1);
    const s2 = new Tone.Synth({ oscillator:{type:'sine'}, envelope:{attack:.003,decay:.11,sustain:0,release:.1} }).connect(T.shortRev);
    s2.volume.value = -18; s2.triggerAttackRelease('G4','16n'); _sfxGc(s2);
  },
  corridor() {
    if (sfxMuted || !T) return;
    const filt = new Tone.Filter({type:'bandpass', frequency:2400, Q:3.5}).connect(T.dist);
    const ns = new Tone.NoiseSynth({noise:{type:'white'}, envelope:{attack:.001, decay:.08, sustain:0, release:.02}}).connect(filt);
    ns.triggerAttackRelease('8n');
    _sfxGc(ns, 1000); _sfxGc(filt, 1000);
  },
  boost() {
    if (sfxMuted || !T) return;
    const filt = new Tone.Filter({type:'bandpass',frequency:200,Q:1.6}).connect(T.shortRev);
    filt.frequency.rampTo(1800, .3);
    const ns = new Tone.NoiseSynth({noise:{type:'pink'},envelope:{attack:.012,decay:.28,sustain:0,release:.18}}).connect(filt);
    ns.triggerAttackRelease('4n'); _sfxGc(ns,2000); _sfxGc(filt,2000);
  },
  surge() {
    if (sfxMuted || !T) return;
    const m = new Tone.MetalSynth({frequency:650,envelope:{attack:.001,decay:.28,release:.1},harmonicity:5.1,modulationIndex:18,resonance:4200,octaves:.6}).connect(T.shortRev);
    m.triggerAttackRelease(.28); _sfxGc(m);
    setTimeout(() => {
      if (sfxMuted || !T) return;
      const s = new Tone.Synth({oscillator:{type:'triangle'},envelope:{attack:.001,decay:.22,sustain:0,release:.14}}).connect(T.shortRev);
      s.volume.value = -2; s.triggerAttackRelease('D6','16n'); _sfxGc(s);
    }, 65);
  },
  buyBoost() {
    if (sfxMuted || !T) return;
    const v1=['E5','G5','B5','D6','F#6','A6'], v2=['B4','D5','G5','A5','E6','G6'];
    v1.forEach((note,i) => setTimeout(() => {
      if (sfxMuted || !T) return;
      const s = new Tone.Synth({oscillator:{type:'triangle'},envelope:{attack:.001,decay:.24,sustain:0,release:.2}}).connect(T.longRev);
      s.volume.value = -8; s.triggerAttackRelease(note,'32n'); _sfxGc(s,3000);
    }, i*48));
    v2.forEach((note,i) => setTimeout(() => {
      if (sfxMuted || !T) return;
      const s = new Tone.Synth({oscillator:{type:'triangle'},envelope:{attack:.001,decay:.20,sustain:0,release:.16}}).connect(T.longRev);
      s.volume.value = -12; s.triggerAttackRelease(note,'32n'); _sfxGc(s,3000);
    }, i*48+24));
  },
  sell() {
    if (sfxMuted || !T) return;
    const fm = new Tone.FMSynth({harmonicity:3.2,modulationIndex:6.5,oscillator:{type:'sawtooth'},envelope:{attack:.001,decay:.22,sustain:0,release:.05}}).connect(T.dist);
    fm.triggerAttackRelease('A0','8n'); _sfxGc(fm);
    const ns = new Tone.NoiseSynth({noise:{type:'pink'},envelope:{attack:.001,decay:.16,sustain:0,release:.04}}).connect(T.dist);
    ns.triggerAttackRelease('8n'); _sfxGc(ns);
  },
  death() {
    if (sfxMuted || !T) return;
    const mem = new Tone.MembraneSynth({pitchDecay:.22,octaves:11,envelope:{attack:.001,decay:.58,sustain:0,release:.3}}).connect(T.longRev);
    mem.triggerAttackRelease('C1','4n'); _sfxGc(mem,4000);
    const syn = new Tone.FMSynth({harmonicity:2,modulationIndex:5,oscillator:{type:'sine'},envelope:{attack:.02,decay:.6,sustain:.1,release:.45}}).connect(T.longRev);
    syn.triggerAttackRelease('G2','2n'); syn.frequency.rampTo('G0',.62); _sfxGc(syn,4000);
  },
  combo() {
    if (sfxMuted || !T) return;
    [{f:390,t:0},{f:490,t:72}].forEach(({f,t}) => setTimeout(() => {
      if (sfxMuted || !T) return;
      const s = new Tone.MetalSynth({frequency:f,envelope:{attack:.001,decay:.38,release:.12},harmonicity:5.1,modulationIndex:32,resonance:3600,octaves:1.5}).connect(T.shortRev);
      s.triggerAttackRelease(.38); _sfxGc(s);
    }, t));
  },
  tierup() {
    if (sfxMuted || !T) return;
    [220,277,330,440].forEach((f,i) => setTimeout(() => {
      if (sfxMuted || !T) return;
      const s = new Tone.MetalSynth({frequency:f,envelope:{attack:.001,decay:.46,release:.18},harmonicity:5.1,modulationIndex:32,resonance:3000,octaves:1.5}).connect(T.longRev);
      s.triggerAttackRelease(.46); _sfxGc(s,4000);
    }, i*100));
  },
  rocket() {
    if (sfxMuted || !T) return;
    const spark = new Tone.Synth({oscillator:{type:'sine'},envelope:{attack:.001,decay:.14,sustain:0,release:.05}}).connect(T.shortRev);
    spark.volume.value = -9; spark.triggerAttackRelease('C6','32n'); spark.frequency.rampTo('C3',.12); _sfxGc(spark,1000);
    const syn = new Tone.FMSynth({harmonicity:2.5,modulationIndex:4,oscillator:{type:'sine'},envelope:{attack:.002,decay:.75,sustain:0,release:.18}}).connect(T.dist);
    syn.triggerAttackRelease('C3','4n'); syn.frequency.rampTo('C1',.68); _sfxGc(syn,4000);
    const mem = new Tone.MembraneSynth({pitchDecay:.45,octaves:9,envelope:{attack:.001,decay:.75,sustain:0,release:.45}}).connect(T.dist);
    mem.volume.value = -1; mem.triggerAttackRelease('C1','4n'); _sfxGc(mem,5000);
  },
  ulti_invoke() {
    if (sfxMuted || !T) return;
    const shimmer = new Tone.Synth({oscillator:{type:'triangle'},envelope:{attack:.08,decay:.5,sustain:.1,release:.5}}).connect(T.longRev);
    shimmer.triggerAttackRelease('A3','4n'); shimmer.frequency.rampTo('A5',.38); _sfxGc(shimmer,3000);
    ['D5','F#5','A5'].forEach((note,i) => setTimeout(() => {
      if (sfxMuted || !T) return;
      const s = new Tone.Synth({oscillator:{type:'triangle'},envelope:{attack:.001,decay:.35,sustain:0,release:.25}}).connect(T.longRev);
      s.triggerAttackRelease(note,'8n'); _sfxGc(s,4000);
    }, 95+i*95));
  },
  ulti_aegis() {
    if (sfxMuted || !T) return;
    [{f:275,d:0},{f:550,d:40}].forEach(({f,d}) => setTimeout(() => {
      if (sfxMuted || !T) return;
      const s = new Tone.MetalSynth({frequency:f,envelope:{attack:.001,decay:.85,release:.4},harmonicity:7.1,modulationIndex:18,resonance:2900,octaves:1}).connect(T.longRev);
      s.triggerAttackRelease(.85); _sfxGc(s,4000);
    }, d));
  },
  ulti_overclock() {
    if (sfxMuted || !T) return;
    const syn = new Tone.FMSynth({harmonicity:2,modulationIndex:8,oscillator:{type:'sawtooth'},envelope:{attack:.001,decay:.2,sustain:.1,release:.05}}).connect(T.dist);
    syn.triggerAttackRelease('A#1','4n'); syn.frequency.rampTo('A#4',.18); _sfxGc(syn);
    const ns = new Tone.NoiseSynth({noise:{type:'white'},envelope:{attack:.001,decay:.12,sustain:0,release:.04}}).connect(T.dist);
    ns.triggerAttackRelease('16n'); _sfxGc(ns);
  },
  ulti_chrono() {
    if (sfxMuted || !T) return;
    const syn = new Tone.FMSynth({harmonicity:.5,modulationIndex:4.5,oscillator:{type:'sine'},envelope:{attack:.022,decay:.75,sustain:.15,release:.55}}).connect(T.longRev);
    syn.triggerAttackRelease('A3','2n'); syn.frequency.rampTo('A0',.7); _sfxGc(syn,5000);
    const ns = new Tone.NoiseSynth({noise:{type:'brown'},envelope:{attack:.05,decay:.55,sustain:.08,release:.35}}).connect(T.longRev);
    ns.triggerAttackRelease('4n'); _sfxGc(ns,5000);
  },
  ulti_magnet() {
    if (sfxMuted || !T) return;
    const hum = new Tone.Synth({oscillator:{type:'triangle'},envelope:{attack:.08,decay:.3,sustain:.5,release:.5}}).connect(T.autoFilt);
    hum.triggerAttackRelease('A2','4n'); _sfxGc(hum,3000);
    ['E3','B3','E4','B4'].forEach((note,i) => setTimeout(() => {
      if (sfxMuted || !T) return;
      const s = new Tone.Synth({oscillator:{type:'triangle'},envelope:{attack:.001,decay:.22,sustain:0,release:.16}}).connect(T.shortRev);
      s.triggerAttackRelease(note,'8n'); _sfxGc(s,2500);
    }, 100+i*65));
  },
  ulti_resonance() {
    if (sfxMuted || !T) return;
    const ns = new Tone.NoiseSynth({noise:{type:'brown'},envelope:{attack:.005,decay:.42,sustain:.1,release:.28}}).connect(T.phaser);
    ns.triggerAttackRelease('4n'); _sfxGc(ns,3000);
    const syn = new Tone.FMSynth({harmonicity:5,modulationIndex:5.5,oscillator:{type:'square'},envelope:{attack:.01,decay:.55,sustain:0,release:.1}}).connect(T.dist);
    syn.triggerAttackRelease('E2','4n'); syn.frequency.rampTo('E4',.45); _sfxGc(syn,3000);
  },
  ulti_helix() {
    if (sfxMuted || !T) return;
    ['E4','B4','E5','B5'].forEach((note,i) => setTimeout(() => {
      if (sfxMuted || !T) return;
      const s = new Tone.FMSynth({harmonicity:3,modulationIndex:2,oscillator:{type:'sine'},envelope:{attack:.01,decay:.22,sustain:.1,release:.18}}).connect(T.chorus);
      s.triggerAttackRelease(note,'8n'); _sfxGc(s,2500);
    }, i*62));
    ['B3','F#4','B4','F#5'].forEach((note,i) => setTimeout(() => {
      if (sfxMuted || !T) return;
      const s = new Tone.FMSynth({harmonicity:3,modulationIndex:1.8,oscillator:{type:'sine'},envelope:{attack:.01,decay:.18,sustain:.05,release:.14}}).connect(T.chorus);
      s.triggerAttackRelease(note,'8n'); _sfxGc(s,2500);
    }, i*62+31));
  },
  ulti_phoenix() {
    if (sfxMuted || !T) return;
    const syn = new Tone.FMSynth({harmonicity:1.5,modulationIndex:2.8,oscillator:{type:'triangle'},envelope:{attack:.05,decay:.38,sustain:.42,release:.65}}).connect(T.chorus);
    syn.triggerAttackRelease('G2','2n'); syn.frequency.rampTo('G5',.58); _sfxGc(syn,5000);
    ['D6','F#6','A6'].forEach((note,i) => setTimeout(() => {
      if (sfxMuted || !T) return;
      const s = new Tone.Synth({oscillator:{type:'sine'},envelope:{attack:.001,decay:.32,sustain:0,release:.24}}).connect(T.longRev);
      s.volume.value = -4; s.triggerAttackRelease(note,'8n'); _sfxGc(s,4000);
    }, 325+i*58));
  },
  resonance_blast() {
    if (sfxMuted || !T) return;
    const mem = new Tone.MembraneSynth({pitchDecay:.28,octaves:14,envelope:{attack:.001,decay:.65,sustain:0,release:.45}}).connect(T.longRev);
    mem.triggerAttackRelease('C1','4n'); _sfxGc(mem,5000);
    const metal = new Tone.MetalSynth({frequency:580,envelope:{attack:.001,decay:.42,release:.2},harmonicity:5.1,modulationIndex:32,resonance:3200,octaves:1.2}).connect(T.dist);
    metal.triggerAttackRelease(.42); _sfxGc(metal,3000);
    setTimeout(() => {
      ['A5','E6','A6'].forEach((note,i) => setTimeout(() => {
        if (sfxMuted || !T) return;
        const s = new Tone.MetalSynth({frequency:Tone.Frequency(note).toFrequency(),envelope:{attack:.001,decay:.25,release:.1},harmonicity:5.1,modulationIndex:16,resonance:3500,octaves:.8}).connect(T.shortRev);
        s.triggerAttackRelease(.25); _sfxGc(s,3000);
      }, i*85));
    }, 155);
  },
  phoenix_rebirth() {
    if (sfxMuted || !T) return;
    const ns = new Tone.NoiseSynth({noise:{type:'white'},envelope:{attack:.02,decay:.38,sustain:0,release:.12}}).connect(T.shortRev);
    ns.triggerAttackRelease('4n'); _sfxGc(ns,2000);
    const syn = new Tone.FMSynth({harmonicity:2,modulationIndex:2.5,oscillator:{type:'sine'},envelope:{attack:.02,decay:.48,sustain:.1,release:.35}}).connect(T.longRev);
    syn.triggerAttackRelease('C2','4n'); syn.frequency.rampTo('C5',.42); _sfxGc(syn,4000);
    setTimeout(() => {
      ['C5','E5','G5'].forEach((note,i) => setTimeout(() => {
        if (sfxMuted || !T) return;
        const s = new Tone.Synth({oscillator:{type:'triangle'},envelope:{attack:.001,decay:.3,sustain:0,release:.22}}).connect(T.longRev);
        s.triggerAttackRelease(note,'8n'); _sfxGc(s,4000);
      }, i*58));
    }, 225);
  },
  aegis_save() {
    if (sfxMuted || !T) return;
    const metal = new Tone.MetalSynth({frequency:1350,envelope:{attack:.001,decay:.38,release:.18},harmonicity:7,modulationIndex:42,resonance:4600,octaves:2}).connect(T.shortRev);
    metal.triggerAttackRelease(.38); _sfxGc(metal,3000);
    setTimeout(() => { if (!sfxMuted && T) sfx.rocket(); }, 88);
  },
};

// ── PLAYLIST ─────────────────────────────────────────────────────────────────
let activeTier = 0;
const TIER_FOLDERS = [
  "1 Genesis", "2 zenith", "3 abyss", "4 VERDANT", "5 Dune",
  "6 Magma", "7 ORCHID", "8 Volt", "9 Nebula", "10 APEX",
];
const TIER_TRACKS = [
  ["First Light", "Genesis Pulse", "Dawn Protocol", "Rising Signal"],
  ["Basin of Light", "Zenith Arc", "Night River", "Heavens Drift"],
  ["Glass Tide", "Abyssal Glow", "Dark Wave", "Depth Current"],
  ["Foam Clearing", "Shattered Forest", "Verdant Sprint", "Grove Pulse"],
  ["Dry Dunes", "Golden Drift", "Fleeing Dunes", "Desert Wind"],
  ["Magma Bloom", "Vector Rush", "Magma Sprint", "Frenetic Core"],
  ["Pressure Abyss", "Night Bloom", "Pixel Current", "Orchid Arc"],
  ["Frenetic Circuit", "Raw Voltage", "Volt Surge", "Wire Overload"],
  ["Nebula in Orbit", "Deep Space Drift", "Cosmic Plucks", "Stellar Pulse"],
  ["To The Moon"],
];
let currentTrackIndex = Math.floor(Math.random() * TIER_TRACKS[activeTier].length);
let bgMusic = new Audio(`Playlist/${TIER_FOLDERS[activeTier]}/${TIER_TRACKS[activeTier][currentTrackIndex]}.mp3`);
bgMusic.loop = false;
const MUSIC_PROCESSOR_URL = "./audio/soundtouch-processor.js";
let musicEngineState = "idle";
let musicEnginePromise = null;
let musicSourceNode = null;
let soundTouchNode = null;

function setNativePitchPreservation(enabled) {
  bgMusic.preservesPitch = enabled;
  bgMusic.mozPreservesPitch = enabled;
  bgMusic.webkitPreservesPitch = enabled;
}

function setMusicVolume(volume) {
  bgMusic.volume = musicMuted ? 0 : volume;
}

function setMusicTempo(rate) {
  const tempo = musicEngineState === "ready" ? rate : 1;
  bgMusic.playbackRate = tempo;
  if (soundTouchNode) {
    soundTouchNode.playbackRate.value = tempo;
    soundTouchNode.pitch.value = 1;
  }
}

async function initializeMusicEngine() {
  if (musicEngineState === "ready") return true;
  if (musicEngineState === "fixed-rate") return false;
  if (musicEnginePromise) return musicEnginePromise;

  musicEnginePromise = (async () => {
    let sourceNode = null;
    let processorNode = null;
    try {
      const ac = getAudioCtx();
      if (!ac.audioWorklet || typeof AudioWorkletNode !== "function") throw new Error("AudioWorklet unavailable");
      await ac.resume();
      const {SoundTouchNode} = await import("./audio/SoundTouchNode.js");
      await SoundTouchNode.register(ac,MUSIC_PROCESSOR_URL);
      processorNode = new SoundTouchNode({context:ac});
      processorNode.connect(ac.destination);
      sourceNode = ac.createMediaElementSource(bgMusic);
      sourceNode.connect(processorNode);
      musicSourceNode = sourceNode;
      soundTouchNode = processorNode;
      musicEngineState = "ready";
      setNativePitchPreservation(false);
      syncMusicRate();
      return true;
    } catch {
      sourceNode?.disconnect();
      processorNode?.disconnect();
      sourceNode?.connect(audioCtx.destination);
      musicSourceNode = sourceNode;
      soundTouchNode = null;
      musicEngineState = "fixed-rate";
      setNativePitchPreservation(true);
      setMusicTempo(1);
      return false;
    } finally {
      musicEnginePromise = null;
    }
  })();
  return musicEnginePromise;
}

setMusicVolume(musicVolume);
setNativePitchPreservation(true);
bgMusic.addEventListener("ended", () => loadTrack(currentTrackIndex + 1));

function loadTrack(index) {
  const tracks = TIER_TRACKS[activeTier];
  const wasPlaying = !bgMusic.paused;
  bgMusic.pause();
  currentTrackIndex = ((index % tracks.length) + tracks.length) % tracks.length;
  bgMusic.src = `Playlist/${TIER_FOLDERS[activeTier]}/${tracks[currentTrackIndex]}.mp3`;
  setMusicVolume(musicVolume);
  updateTrackUI();
  if (wasPlaying && !musicMuted && state === "playing") bgMusic.play().catch(()=>{});
}

function loadTierPlaylist() {
  const tracks = TIER_TRACKS[activeTier];
  const wasPlaying = !bgMusic.paused;
  bgMusic.pause();
  currentTrackIndex = Math.floor(Math.random() * tracks.length);
  bgMusic.src = `Playlist/${TIER_FOLDERS[activeTier]}/${tracks[currentTrackIndex]}.mp3`;
  setMusicVolume(musicVolume);
  updateTrackUI();
  buildTrackList();
  if (wasPlaying && !musicMuted && state === "playing") bgMusic.play().catch(()=>{});
}

function nextTrack() { loadTrack(currentTrackIndex + 1); }
function prevTrack()  { loadTrack(currentTrackIndex - 1); }

function updateTrackUI() {
  const titleEl = document.getElementById("trackTitle");
  if (!titleEl) return;
  titleEl.textContent = TIER_TRACKS[activeTier][currentTrackIndex];
  titleEl.classList.remove("scrolling");
  requestAnimationFrame(() => {
    const parent = titleEl.parentElement;
    if (!parent) return;
    const overflow = titleEl.scrollWidth - parent.clientWidth;
    if (overflow > 4) {
      titleEl.style.setProperty("--scroll-dist", `${-(overflow + 12)}px`);
      const dur = Math.max(6, overflow / 20);
      titleEl.style.setProperty("--scroll-dur", `${dur}s`);
      titleEl.classList.add("scrolling");
    }
  });
  document.querySelectorAll(".track-item").forEach((el, i) => {
    el.classList.toggle("active", i === currentTrackIndex);
  });
}

function buildTrackList() {
  const panel = document.getElementById("tracklistPanel");
  if (!panel) return;
  panel.innerHTML = TIER_TRACKS[activeTier].map((t, i) =>
    `<div class="track-item${i===currentTrackIndex?" active":""}" data-index="${i}"><span class="t-num">${String(i+1).padStart(2,"0")}</span>${t}</div>`
  ).join("");
  panel.querySelectorAll(".track-item").forEach(el => {
    el.addEventListener("click", e => {
      e.stopPropagation();
      const idx = parseInt(el.dataset.index);
      loadTrack(idx);
      if (!musicMuted && state === "playing") bgMusic.play().catch(()=>{});
      document.getElementById("tracklistPanel").classList.add("hidden");
    });
  });
}

function updateMixerUI() {
  const mm = document.getElementById("muteMusicBtn");
  const ms = document.getElementById("muteSfxBtn");
  if (mm) mm.classList.toggle("muted", musicMuted);
  if (ms) ms.classList.toggle("muted", sfxMuted);
  const mvSlider = document.getElementById("musicVolumeSlider");
  const svSlider = document.getElementById("sfxVolumeSlider");
  if (mvSlider) { mvSlider.value = musicVolume; updateSliderTrack(mvSlider); }
  if (svSlider) { svSlider.value = sfxVolume;   updateSliderTrack(svSlider); }
  const allMuted = sfxMuted && musicMuted;
  document.getElementById("soundBtn")?.classList.toggle("muted", allMuted);
  const vibrationToggle = document.getElementById("vibrationToggle");
  if (vibrationToggle) {
    const supported = typeof navigator.vibrate === "function";
    vibrationToggle.disabled = !supported;
    vibrationToggle.textContent = supported ? (vibrationEnabled ? "ON" : "OFF") : "N/A";
    vibrationToggle.classList.toggle("on", supported && vibrationEnabled);
    vibrationToggle.setAttribute("aria-pressed",String(supported && vibrationEnabled));
  }
  const qualityToggle = document.getElementById("qualityToggle");
  if (qualityToggle) {
    qualityToggle.textContent = qualityMode.toUpperCase();
    qualityToggle.classList.toggle("on", qualityMode !== "low");
  }
  const trailsToggle = document.getElementById("trailsToggle");
  if (trailsToggle) {
    trailsToggle.textContent = trailMode.toUpperCase();
    trailsToggle.classList.toggle("on", trailMode !== "off");
  }
  document.querySelectorAll("[data-ulti-key-slot]").forEach(button => {
    const slot = Number(button.dataset.ultiKeySlot);
    button.textContent = rebindingUltiSlot === slot ? "..." : ultiKeys[slot];
    button.classList.toggle("listening",rebindingUltiSlot === slot);
  });
  const ctrlMap = { pause:pauseKey, boost:boostKey, left:leftKey, right:rightKey };
  document.querySelectorAll("[data-ctrl-key]").forEach(button => {
    const which = button.dataset.ctrlKey;
    button.textContent = rebindingControlKey === which ? "..." : displayKey(ctrlMap[which]);
    button.classList.toggle("listening", rebindingControlKey === which);
  });
  document.querySelectorAll("[data-gamepad-action]").forEach(button => {
    const action = button.dataset.gamepadAction;
    button.textContent = rebindingGamepadAction === action ? "..." : displayGamepadBinding(gamepadBindings[action]);
    button.classList.toggle("listening", rebindingGamepadAction === action);
  });
  updateGamepadStatusUI();
}

function updateSliderTrack(input) {
  const pct = parseFloat(input.value) / parseFloat(input.max) * 100;
  input.style.setProperty("--pct", pct);
}

function setOptionsMessage(message="") {
  const el = document.getElementById("optionsMessage");
  if (el) el.textContent = message;
}

function finishUltiKeyCapture(message="") {
  rebindingUltiSlot = null;
  setOptionsMessage(message);
  updateMixerUI();
}

function displayKey(key) {
  const MAP = { Escape:"ESC", Space:"SPC", ArrowLeft:"←", ArrowRight:"→", ArrowUp:"↑", ArrowDown:"↓" };
  return MAP[key] ?? key;
}

function displayGamepadBinding(binding) {
  if (!binding) return "---";
  if (binding.type === "axis") return `A${binding.index}${binding.direction < 0 ? "-" : "+"}`;
  const BUTTON_LABELS = {
    0:"A/×", 1:"B/○", 2:"X/□", 3:"Y/△",
    4:"LB/L1", 5:"RB/R1", 6:"LT/L2", 7:"RT/R2",
    8:"BACK", 9:"START", 10:"LS", 11:"RS",
    12:"D↑", 13:"D↓", 14:"D←", 15:"D→"
  };
  return BUTTON_LABELS[binding.index] ?? `B${binding.index}`;
}

function finishGamepadCapture(message="") {
  rebindingGamepadAction = null;
  setOptionsMessage(message);
  updateMixerUI();
}

function keyMatches(e, storedKey) {
  if (storedKey === "Space")  return e.code === "Space";
  if (storedKey === "Escape" || storedKey.startsWith("Arrow") || storedKey.startsWith("F"))
    return e.key === storedKey;
  return e.key.toUpperCase() === storedKey.toUpperCase();
}

function finishControlKeyCapture(message="") {
  rebindingControlKey = null;
  setOptionsMessage(message);
  updateMixerUI();
}

function resetOptions() {
  musicVolume = DEFAULT_OPTIONS.musicVolume;
  sfxVolume = DEFAULT_OPTIONS.sfxVolume;
  musicMuted = DEFAULT_OPTIONS.musicMuted;
  sfxMuted = DEFAULT_OPTIONS.sfxMuted;
  vibrationEnabled = DEFAULT_OPTIONS.vibrationEnabled;
  trailMode = DEFAULT_OPTIONS.trailMode;
  ultiKeys = [...DEFAULT_OPTIONS.ultiKeys];
  pauseKey = DEFAULT_OPTIONS.pauseKey;
  boostKey = DEFAULT_OPTIONS.boostKey;
  leftKey  = DEFAULT_OPTIONS.leftKey;
  rightKey = DEFAULT_OPTIONS.rightKey;
  gamepadBindings = structuredClone(DEFAULT_OPTIONS.gamepadBindings);
  rebindingUltiSlot = null;
  rebindingControlKey = null;
  rebindingGamepadAction = null;
  setMusicVolume(musicVolume);
  syncMusic();
  saveOptions();
  setOptionsMessage("DEFAULTS RESTORED");
  updateMixerUI();
  updateUltiHud();
}

let musicRateTimer = null;
function syncMusicRate() {
  const target = 1;
  clearInterval(musicRateTimer);
  if (state !== "playing") { setMusicTempo(target); return; }
  const start = bgMusic.playbackRate, startedAt = performance.now(), duration = 2000;
  musicRateTimer = setInterval(() => {
    const progress = Math.min(1, (performance.now()-startedAt)/duration);
    setMusicTempo(start + (target-start)*progress);
    if (progress === 1) { clearInterval(musicRateTimer); musicRateTimer = null; }
  }, 50);
}
function syncMusic() {
  if (musicMuted || state !== "playing") { bgMusic.pause(); return; }
  initializeMusicEngine();
  syncMusicRate();
  setMusicVolume(musicVolume);
  if (bgMusic.paused) bgMusic.play().catch(()=>{});
}

// ── LEADERBOARD ──────────────────────────────────────────────────────────────
const leaderboardState = new Map();
const leaderboardRevision = new Map();
const leaderboardSubmissions = new Map();
let lbMode = "temporary";
let currentSeason = "S01";
let contestConfig = { endDate: "", podiumRewards: { "1": 0, "2": 0, "3": 0 }, contestTier: 0, contestLocked: false };
// Optimistic finish-screen leaderboard entries for the just-finished run, merged into the
// rendered top-10 until the authoritative server data lands. Kept per board (all-time vs
// contest) because each reconciles from a different source at a different time.
let pendingAllTimeEntry = null; // { tier, name, score } | null
let pendingContestEntry = null; // { tier, name, score, identityAddress, accountAddress } | null
function fmtAscent(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toLocaleString("en-US", { maximumFractionDigits: 1 }) + "M";
  if (n >= 1_000) return (n / 1_000).toLocaleString("en-US", { maximumFractionDigits: 1 }) + "K";
  return String(n);
}
const temporaryLeaderboardState = new Map();
const temporaryLeaderboardRevision = new Map();
function emptyLeaderboards() { return Array.from({length:TIERS.length},()=>[]); }
function marketLeaderboards(marketId=currentMarket) {
  if (!leaderboardState.has(marketId)) leaderboardState.set(marketId,{ status:"loading", tiers:emptyLeaderboards() });
  return leaderboardState.get(marketId);
}
function lbLoad(marketId=currentMarket,tierIndex=activeTier) {
  return marketLeaderboards(marketId).tiers[tierIndex] || [];
}
function normalizePlayerName(value) { return value.trim().replace(/[\u0000-\u001F\u007F-\u009F\u200B-\u200F\u202A-\u202E\uFEFF]/g, "").slice(0,8).toUpperCase() || "PLAYER"; }
function currentPlayerName() {
  const input = document.getElementById("playerNameInput");
  const name = normalizePlayerName(input?.value || localStorage.getItem(PLAYER_NAME_KEY) || "");
  localStorage.setItem(PLAYER_NAME_KEY,name);
  if (input) input.value = name;
  return name;
}
function normalizeLeaderboardEntries(entries) {
  if (!Array.isArray(entries)) return [];
  return entries.filter(entry=>Number.isSafeInteger(entry?.score) && entry.score>=0)
    .map(entry=>({ name:normalizePlayerName(String(entry.name||"")), score:entry.score }))
    .sort((left,right)=>right.score-left.score).slice(0,LB_MAX);
}
function normalizeLeaderboardEntriesTemporary(entries) {
  if (!Array.isArray(entries)) return [];
  return entries.filter(entry=>Number.isSafeInteger(entry?.score) && entry.score>=0)
    .map(entry=>({ name:normalizePlayerName(String(entry.name||"")), score:entry.score, identityAddress:String(entry.identityAddress||""), accountAddress:String(entry.accountAddress||"") }))
    .sort((left,right)=>right.score-left.score).slice(0,LB_MAX);
}
function setLeaderboardUnavailable(marketId=currentMarket) {
  leaderboardState.set(marketId,{ status:"unavailable", tiers:emptyLeaderboards() });
}
function refreshLeaderboardUi(marketId=currentMarket) {
  if (marketId!==currentMarket) return;
  loadBest();
  lbRender();
  if (!document.getElementById("leaderboardOverlay")?.classList.contains("hidden")) renderAllLeaderboards();
}
function marketTemporaryLeaderboards(marketId=currentMarket) {
  if (!temporaryLeaderboardState.has(marketId)) temporaryLeaderboardState.set(marketId,{ status:"loading", tiers:emptyLeaderboards() });
  return temporaryLeaderboardState.get(marketId);
}
async function loadTemporaryLeaderboards(marketId=currentMarket) {
  const revision = (temporaryLeaderboardRevision.get(marketId)||0)+1;
  temporaryLeaderboardRevision.set(marketId,revision);
  temporaryLeaderboardState.set(marketId,{ status:"loading", tiers:emptyLeaderboards() });
  if (lbMode==="temporary") renderAllLeaderboards();
  try {
    const payload = await apiFetch(`${CALENDAR_API}/leaderboards?market=${encodeURIComponent(marketId)}&season=${encodeURIComponent(currentSeason)}`,{errorLabel:"leaderboard"});
    if (payload.market!==marketId || !Array.isArray(payload.tiers) || payload.tiers.length!==TIERS.length) throw new Error("invalid leaderboard");
    if (temporaryLeaderboardRevision.get(marketId)!==revision) return;
    if (payload.endDate !== undefined) {
      contestConfig = { endDate: payload.endDate || "", podiumRewards: payload.podiumRewards || { "1": 0, "2": 0, "3": 0 }, contestTier: Number(payload.contestTier) || 0, contestLocked: !!payload.contestLocked };
    }
    temporaryLeaderboardState.set(marketId,{ status:"ready", tiers:payload.tiers.map(normalizeLeaderboardEntriesTemporary) });
    // Authoritative contest list loaded; drop the optimistic copy.
    pendingContestEntry = null;
  } catch {
    if (temporaryLeaderboardRevision.get(marketId)!==revision) return;
    temporaryLeaderboardState.set(marketId,{ status:"unavailable", tiers:emptyLeaderboards() });
  }
  if (lbMode==="temporary") renderAllLeaderboards();
}
async function loadLeaderboards(marketId=currentMarket) {
  const revision = (leaderboardRevision.get(marketId)||0)+1;
  leaderboardRevision.set(marketId,revision);
  leaderboardState.set(marketId,{ status:"loading", tiers:emptyLeaderboards() });
  refreshLeaderboardUi(marketId);
  try {
    const payload = await apiFetch(`${CALENDAR_API}/leaderboards?market=${encodeURIComponent(marketId)}`,{errorLabel:"leaderboard"});
    if (payload.market!==marketId || !Array.isArray(payload.tiers) || payload.tiers.length!==TIERS.length) throw new Error("invalid leaderboard");
    if (leaderboardRevision.get(marketId)!==revision) return;
    leaderboardState.set(marketId,{ status:"ready", tiers:payload.tiers.map(normalizeLeaderboardEntries) });
    if (payload.currentSeason && payload.currentSeason!==currentSeason) {
      currentSeason = payload.currentSeason;
      const btn = document.getElementById("lbTabTemporary");
      if (btn) btn.textContent = `CONTEST ${currentSeason}`;
      temporaryLeaderboardState.clear();
      temporaryLeaderboardRevision.clear();
    }
  } catch {
    if (leaderboardRevision.get(marketId)!==revision) return;
    setLeaderboardUnavailable(marketId);
  }
  refreshLeaderboardUi(marketId);
}
async function lbSave(score) {
  const marketId = currentMarket;
  const tierIndex = runTier; // tier the run started in — activeTier can change between runs

  const name = currentPlayerName();
  const revision = (leaderboardRevision.get(marketId)||0)+1;
  leaderboardRevision.set(marketId,revision);
  leaderboardSubmissions.set(marketId,(leaderboardSubmissions.get(marketId)||0)+1);
  try {
    const lbState = storedWalletState();
    const payload = await apiFetch(`${CALENDAR_API}/leaderboard`,{
      method:"POST",
      headers:{"Content-Type":"application/json", ...(lbState.sessionToken ? {Authorization:`Bearer ${lbState.sessionToken}`} : {})},
      body:JSON.stringify({ market:marketId, tier:tierIndex, name, score, durationMs:Math.round(performance.now()-gameStartTime), bounces:orb.bounces, maxCombo:orb.maxCombo }),
      timeoutMs:10000, errorLabel:"leaderboard"
    });
    if (payload.market!==marketId || payload.tier!==tierIndex) throw new Error("invalid leaderboard");
    showRunRank(payload.rank, payload.total);
    if (lbState.sessionToken && playerBestScore!=null && score>=playerBestScore && Number.isSafeInteger(payload.total)) {
      // This run is the connected player's new best: its rank IS the best-score rank.
      playerBestScoreMarket = marketId;
      playerBestScoreTier = tierIndex;
      bestRank = { market:marketId, tier:tierIndex, rank:payload.rank, total:payload.total };
      updateBestRankUi();
    }
    if (leaderboardRevision.get(marketId)!==revision) return;
    const state = marketLeaderboards(marketId);
    state.status = "ready";
    state.tiers[tierIndex] = normalizeLeaderboardEntries(payload.entries);
    // The run is now in the authoritative all-time list; drop the optimistic copy.
    if (pendingAllTimeEntry?.tier === tierIndex) pendingAllTimeEntry = null;
  } catch {
    // save failure: keep existing leaderboard state intact rather than wiping it
  } finally {
    const pending = leaderboardSubmissions.get(marketId)||0;
    if (pending<=1) leaderboardSubmissions.delete(marketId);
    else leaderboardSubmissions.set(marketId,pending-1);
  }
  refreshLeaderboardUi(marketId);
}
function fmtAllTimeRank(rank) {
  return Number.isSafeInteger(rank) && rank>0 ? `#${rank.toLocaleString()}` : `>${LB_ARCHIVE_MAX.toLocaleString()}`;
}
function showRunRank(rank, total) {
  const stat = document.getElementById("finishRankStat");
  const el = document.getElementById("finalRank");
  if (!stat || !el || !Number.isSafeInteger(total) || total<=0) return;
  el.textContent = fmtAllTimeRank(rank);
  stat.classList.remove("hidden");
}
function updateBestRankUi() {
  const statEl = document.getElementById("profileRankStat");
  const valEl = document.getElementById("profileRank");
  if (statEl && valEl) {
    if (bestRank) { valEl.textContent = fmtAllTimeRank(bestRank.rank); statEl.classList.remove("hidden"); }
    else statEl.classList.add("hidden");
  }
  const startRankEl = document.getElementById("startBestRank");
  const startRankVal = document.getElementById("startBestRankVal");
  if (startRankEl && startRankVal) {
    if (bestRank) { startRankVal.textContent = fmtAllTimeRank(bestRank.rank); startRankEl.classList.remove("hidden"); }
    else startRankEl.classList.add("hidden");
  }
  if (!document.getElementById("leaderboardOverlay")?.classList.contains("hidden")) renderAllLeaderboards();
}
async function loadBestRank() {
  if (playerBestScore==null || !playerBestScoreMarket || !Number.isInteger(playerBestScoreTier)) return;
  try {
    const payload = await apiFetch(`${CALENDAR_API}/leaderboard/rank?market=${encodeURIComponent(playerBestScoreMarket)}&tier=${playerBestScoreTier}&score=${playerBestScore}`,{errorLabel:"leaderboard"});
    if (!Number.isSafeInteger(payload.total) || payload.total<=0) return;
    bestRank = { market:playerBestScoreMarket, tier:playerBestScoreTier, rank:payload.rank, total:payload.total };
    updateBestRankUi();
  } catch {
    // rank is cosmetic: keep the last known value on failure
  }
}
function lbTitle(text) {
  const el = document.createElement("div");
  el.className = "lb-title";
  el.textContent = text;
  return el;
}
function lbRows(entries, { showWallet=false }={}) {
  const fragment = document.createDocumentFragment();
  entries.forEach((entry,i) => {
    const row = document.createElement("div");
    row.className = "lb-row";
    const rank = document.createElement("span");
    rank.className = "lb-rank";
    rank.textContent = `#${i+1}`;
    const nameWrap = document.createElement("span");
    nameWrap.className = "lb-name";
    const nameText = document.createElement("span");
    nameText.textContent = normalizePlayerName(entry.name || "");
    nameWrap.append(nameText);
    if (showWallet && entry.identityAddress) {
      const wallet = document.createElement("span");
      wallet.className = "lb-wallet";
      wallet.textContent = entry.identityAddress.slice(0,28) + "…";
      nameWrap.append(wallet);
    }
    const medal = i===0 ? " gold" : i===1 ? " silver" : i===2 ? " bronze" : "";
    rank.className = "lb-rank" + medal;
    nameWrap.className = "lb-name" + medal;
    const score = document.createElement("span");
    score.className = "lb-score" + medal;
    score.textContent = String(entry.score).padStart(6,"0");
    row.append(rank,nameWrap,score);
    fragment.append(row);
  });
  return fragment;
}
function lbStatus(text, isError = false) {
  const empty = document.createElement("div");
  empty.className = "lb-empty" + (isError ? " lb-error" : "");
  empty.textContent = text;
  return empty;
}
function lbGhostRows(count = 5) {
  const fragment = document.createDocumentFragment();
  for (let i = 0; i < count; i++) {
    const row = document.createElement("div");
    row.className = "lb-row lb-ghost-row";
    const rank = document.createElement("span"); rank.className = "lb-rank"; rank.textContent = `#${i+1}`;
    const name = document.createElement("span"); name.className = "lb-name"; name.textContent = "— — — — —";
    const score = document.createElement("span"); score.className = "lb-score"; score.textContent = "000000";
    row.append(rank, name, score);
    fragment.append(row);
  }
  return fragment;
}
function setLbMode(mode) {
  lbMode = mode;
  const isContest = mode === "temporary";
  ["lbTabTemporary","finishLbTabTemporary"].forEach(id => document.getElementById(id)?.classList.toggle("lb-tab--active", isContest));
  ["lbTabAllTime","finishLbTabAllTime"].forEach(id => document.getElementById(id)?.classList.toggle("lb-tab--active", !isContest));
}
function mergePendingEntry(entries, tier, pending, { dedupeByIdentity }) {
  if (!pending || pending.tier !== tier) return entries;
  let base = entries;
  if (dedupeByIdentity && pending.identityAddress) {
    // The contest keeps at most one entry per identityAddress (worker/index.js:2628-2633):
    // drop the player's existing row before adding, otherwise they'd appear twice.
    base = entries.filter(e => e.identityAddress !== pending.identityAddress);
  }
  return [...base, pending].sort((a, b) => b.score - a.score).slice(0, LB_MAX);
}
function lbRender() {
  const container = document.getElementById("leaderboard");
  if (!container) return;
  container.replaceChildren();
  if (lbMode === "temporary") {
    const state = marketTemporaryLeaderboards();
    const contestTier = contestConfig.contestTier || 0;
    container.append(lbTitle(`CONTEST · TIER ${contestTier + 1}`));
    if (state.status==="unavailable") { container.append(lbStatus("LEADERBOARD UNAVAILABLE", true)); return; }
    if (state.status==="loading") { container.append(lbStatus("LOADING…")); return; }
    const entries = mergePendingEntry(state.tiers[contestTier] || [], contestTier, pendingContestEntry, { dedupeByIdentity:true });
    container.append(entries.length ? lbRows(entries) : lbGhostRows());
    return;
  }
  const state = marketLeaderboards();
  const entries = mergePendingEntry(lbLoad(), activeTier, pendingAllTimeEntry, { dedupeByIdentity:false });
  container.append(lbTitle(`BEST ASCENTS · TIER ${activeTier+1}`));
  if (state.status==="unavailable") { container.append(lbStatus("LEADERBOARD UNAVAILABLE", true)); return; }
  if (state.status==="loading") { container.append(lbStatus("LOADING LEADERBOARD")); return; }
  container.append(entries.length ? lbRows(entries) : lbGhostRows());
}
function renderAllLeaderboards() {
  const container = document.getElementById("leaderboardAll");
  document.getElementById("leaderboardTitle").replaceChildren("ASCENT", document.createElement("br"), "LEADERBOARD");
  container.replaceChildren();
  if (lbMode==="temporary") {
    // — Info banner —
    const banner = document.createElement("div");
    banner.className = "lb-contest-banner";
    const p1 = fmtAscent(contestConfig.podiumRewards["1"] || 0);
    const p2 = fmtAscent(contestConfig.podiumRewards["2"] || 0);
    const p3 = fmtAscent(contestConfig.podiumRewards["3"] || 0);
    const endLabel = contestConfig.endDate ? `Contest ends on: <strong>${contestConfig.endDate} · 12:00 UTC</strong>` : "";
    const cTier = contestConfig.contestTier || 0;
    const cTierName = TIERS[cTier]?.name || "";
    banner.innerHTML =
      `<p class="lb-contest-tier">TIER ${cTier + 1}${cTierName ? ` · ${cTierName}` : ""}</p>` +
      '<div class="lb-contest-podium">' +
        `<div class="lb-contest-prize lb-contest-prize--2">🥈 2<span>${p2} ASCENT</span></div>` +
        `<div class="lb-contest-prize lb-contest-prize--1">🥇 1<span>${p1} ASCENT</span></div>` +
        `<div class="lb-contest-prize lb-contest-prize--3">🥉 3<span>${p3} ASCENT</span></div>` +
      '</div>' +
      (endLabel ? `<p class="lb-contest-reset">${endLabel}</p>` : "") +
      (contestConfig.contestLocked ? '<p class="lb-contest-ended">🔒 CONTEST ENDED</p>' : "") +
      '<p class="lb-contest-warning">⚠ Radix wallet required to rank.</p>';
    container.append(banner);
    const state = marketTemporaryLeaderboards();
    if (state.status==="unavailable") { container.append(lbStatus("LEADERBOARD UNAVAILABLE", true)); return; }
    if (state.status==="loading") { container.append(lbStatus("LOADING LEADERBOARD")); return; }
    const entries = state.tiers[contestConfig.contestTier || 0] || [];
    const filled = entries.length >= 10 ? entries : [...entries, ...Array(10 - entries.length).fill(null)];
    const fragment = document.createDocumentFragment();
    filled.forEach((entry, i) => {
      const medal = entry && i===0 ? " gold" : entry && i===1 ? " silver" : entry && i===2 ? " bronze" : "";
      const row = document.createElement("div");
      row.className = "lb-row" + (entry ? "" : " lb-ghost-row");
      const rank = document.createElement("span"); rank.className = "lb-rank" + medal; rank.textContent = `#${i+1}`;
      const nameWrap = document.createElement("span"); nameWrap.className = "lb-name" + medal;
      if (entry) {
        nameWrap.textContent = entry.name;
      } else {
        nameWrap.textContent = "— — — — —";
      }
      const score = document.createElement("span");
      score.className = "lb-score" + medal;
      score.textContent = entry ? String(entry.score).padStart(6,"0") : "000000";
      row.append(rank, nameWrap, score);
      fragment.append(row);
    });
    container.append(fragment);
    return;
  }
  const state = marketLeaderboards();
  if (state.status==="unavailable") { container.append(lbStatus("LEADERBOARD UNAVAILABLE", true)); return; }
  if (state.status==="loading") { container.append(lbStatus("LOADING LEADERBOARD")); return; }
  if (bestRank && bestRank.market===currentMarket && playerBestScore!=null) {
    const header = document.createElement("div");
    header.className = "lb-your-rank";
    const scoreLabel = document.createElement("strong");
    scoreLabel.textContent = playerBestScore.toLocaleString();
    const rankLabel = document.createElement("strong");
    rankLabel.textContent = fmtAllTimeRank(bestRank.rank);
    header.append("YOUR BEST: ", scoreLabel, " — ", rankLabel, ` ALL-TIME · TIER ${bestRank.tier+1}`);
    container.append(header);
  }
  TIERS.forEach((t,i) => {
    const section = document.createElement("section");
    section.className = "leaderboard-tier";
    section.append(lbTitle(`TIER ${i+1} · ${t.name}`));
    const entries = lbLoad(currentMarket,i);
    section.append(entries.length ? lbRows(entries) : lbGhostRows(3));
    container.append(section);
  });
}
let resumableInfoOverlay = null;
function openInfoOverlay(id) {
  closeInfoOverlays({ resume:false });
  document.getElementById("mixerPanel").classList.add("hidden");
  document.getElementById("mixerBtn")?.classList.remove("active-mix");
  finishUltiKeyCapture();
  if (state === "playing") {
    pauseGame();
    resumableInfoOverlay = id;
  }
  document.getElementById(id)?.classList.remove("hidden");
  if (id === "howToPlayOverlay") startHowToPlayPreviews();
}
function closeInfoOverlay(id,{ resume=true }={}) {
  document.getElementById(id)?.classList.add("hidden");
  if (id === "howToPlayOverlay") stopHowToPlayPreviews();
  if (resumableInfoOverlay !== id) return;
  resumableInfoOverlay = null;
  if (resume) resumeGame();
}
function closeInfoOverlays({ resume=true }={}) {
  const resumeId = resumableInfoOverlay;
  closeInfoOverlay("howToPlayOverlay",{ resume:false });
  closeInfoOverlay("leaderboardOverlay",{ resume:false });
  closeInfoOverlay("shopOverlay",{ resume:false });
  closeInfoOverlay("profileOverlay",{ resume:false });
  closeInfoOverlay("inventoryOverlay",{ resume:false });
  if (resume && resumeId) resumeGame();
}
function closeLeaderboard(options) {
  closeInfoOverlay("leaderboardOverlay",options);
}
function updateLeaderboardButton() {
  document.getElementById("leaderboardBtn").disabled = false;
}
let best = 0;
let playerBestScore = null;
let playerBestScoreLoaded = false;
let playerBestScoreMarket = null;
let playerBestScoreTier = null;
let bestRank = null; // { market, tier, rank, total } — all-time rank of the connected player's best score
function updateBestScoreHud() {
  const scoreEl = document.getElementById("bestScore");
  const labelEl = document.getElementById("bestScoreLabel");
  const walletConnected = storedWalletState().status === "connected";
  const personalBestReady = walletConnected && playerBestScoreLoaded;
  if (labelEl) labelEl.textContent = personalBestReady ? "MY BEST" : "BEST";
  if (!scoreEl) return;
  if (personalBestReady) {
    scoreEl.textContent = String(playerBestScore ?? 0).padStart(6,"0");
    return;
  }
  scoreEl.textContent = marketLeaderboards().status==="ready" ? String(best).padStart(6,"0") : "------";
}
function loadBest() {
  const state = marketLeaderboards();
  const entries = lbLoad();
  best = entries[0]?.score || 0;
  updateBestScoreHud();
}

// ── TIER NORMALIZATION ───────────────────────────────────────────────────────
// TIERS is imported from tiers.js.
// Add back-compat flat properties so rendering code works unchanged.
(function normalizeTiers() {
  TIERS.forEach((t, i) => {
    t.orbG     = t.orb.mid;
    t.orbC     = t.orb.hot;
    t.platBuy  = t.orb.mid;
    t.platSell = i === 0 ? t.orb.rim : "#ff4d6d";
    t.platN    = i === 0 ? "#999999" : t.orb.rim;
    t.trailLen = t.feats.trail;
    t.bg0      = t.bg;
    t.particles    = t.feats.particles;
    t.shockwave    = t.feats.shockwave;
    t.shake        = t.feats.shake;
    t.arcs         = t.feats.arcs;
    t.chromatic    = t.feats.chromatic;
    t.stars        = t.feats.stars;
    t.nebula       = t.feats.nebula;
    t.candleGhosts = t.feats.ghosts;
  });
})();
function tier() { return TIERS[activeTier]; }

function applyTierTheme(t) {
  const r = document.documentElement.style;
  r.setProperty('--t-primary', t.orb.mid);
  r.setProperty('--t-glow',    t.orb.hot);
  r.setProperty('--t-buy',     t.platBuy);
  r.setProperty('--t-sell',    t.platSell);
  r.setProperty('--t-line',    hexAlpha(t.orb.mid, .22));
  r.setProperty('--t-panel',   hexAlpha(t.orb.mid, .08));
  r.setProperty('--crt-opacity', t.feats.crt ?? 0.25);
  const chip = document.getElementById("multChip");
  if (chip) chip.textContent = `×${t.feats.scoreMult}`;
  tierVisualsInvalidate(); // tier swap → drop other-tier baked layers
}

// Convert #rrggbb + alpha to rgba(...)
const _hexACache = new Map();
function hexAlpha(hex, a) {
  // Quantize alpha to 100 levels so continuous time-driven alphas hit the cache
  // instead of growing it without bound (this cache had no cap).
  a = a <= 0 ? 0 : a >= 1 ? 1 : ((a * 100 + 0.5) | 0) / 100;
  const key = `${hex}|${a}`;
  let v = _hexACache.get(key);
  if (v) return v;
  const h = hex.replace('#','');
  const r = parseInt(h.slice(0,2),16);
  const g = parseInt(h.slice(2,4),16);
  const b = parseInt(h.slice(4,6),16);
  v = `rgba(${r},${g},${b},${a})`;
  if (_hexACache.size < 4000) _hexACache.set(key, v);
  return v;
}
const hexA = hexAlpha;

function buildTierPanel() {
  const list = document.getElementById("tierList");
  if (!list) return;
  const maxUnlocked = tierUnlockState.maxUnlockedTierIndex ?? 0;
  list.innerHTML = TIERS.map((t,i) => {
    const locked = i > maxUnlocked;
    const next = i === tierUnlockState.nextTierIndex;
    const lockTitle = i === 1 && !tierUnlockState.tier2Status?.unlocked
      ? "Unlocks with the official Rly.fun Ociswap pool"
      : !tierUnlockState.tier2Status?.unlocked
      ? "Unlocks after the official pool and market cap tiers"
      : `Unlocks at ${formatCompactXrd(tierUnlockState.nextThresholdXrd)} market cap`;
    return `<button class="tier-btn${i===activeTier?" active":""}${locked?" locked":""}${next?" next-unlock":""}" data-tier="${i}" style="--bar-c:${t.orb.bloom}" aria-disabled="${locked ? "true" : "false"}" title="${locked ? lockTitle : `${i + 1} · ${t.name}`}">
       <span class="t-num">${i+1}</span>
       <span class="t-name">${t.name}</span>
       ${locked ? `<span class="t-lock" aria-hidden="true">🔒</span>` : ""}
       <span class="t-bar"></span>
       ${next ? `<span class="t-progress"><span style="width:${Math.round(tierUnlockState.progressPct || 0)}%"></span></span>` : ""}
     </button>`
  }).join("");
  list.querySelectorAll(".tier-btn").forEach(btn =>
    btn.addEventListener("click", ()=> selectTier(Number(btn.dataset.tier)))
  );
  applyTierTheme(tier());
  updateTierUnlockUi();
}

function selectTier(n) {
  if (state === "playing" || state === "paused") { showTierLockedToast(); return; }
  const pauseScreenVisible = ["startOverlay","ultiSelectScreen","finishOverlay"]
    .some(id => !document.getElementById(id)?.classList.contains("hidden"));
  const maxUnlocked = tierUnlockState.maxUnlockedTierIndex ?? 0;
  const requestedTier = Math.max(0, Math.min(9, n));
  if (requestedTier > maxUnlocked) {
    updateTierUnlockUi();
    return;
  }
  activeTier = requestedTier;
  equippedUltis    = [];
  ultiCharges      = [];
  ultiActiveStates = [];
  document.querySelectorAll(".tier-btn").forEach((b,i)=>b.classList.toggle("active",i===activeTier));
  applyTierTheme(tier());
  loadTierPlaylist();
  loadBest();
  updatePreview();
  showTierToast();
  updateUltiHud();
  if (pauseScreenVisible) { showStartScreen(); return; }
  if (state === "playing") requestStartGame();
}

function highestUnlockedTierIndex() {
  return Math.max(0, Math.min(TIERS.length - 1, tierUnlockState.maxUnlockedTierIndex ?? 0));
}

function enforceUnlockedTierAfterRun() {
  const maxUnlocked = highestUnlockedTierIndex();
  if (activeTier <= maxUnlocked) return false;
  activeTier = maxUnlocked;
  equippedUltis    = [];
  ultiCharges      = [];
  ultiActiveStates = [];
  document.querySelectorAll(".tier-btn").forEach((b,i)=>b.classList.toggle("active",i===activeTier));
  applyTierTheme(tier());
  loadTierPlaylist();
  loadBest();
  updatePreview();
  updateUltiHud();
  pendingTierRelockIndex = null;
  return true;
}

function updateTierUnlockUi() {
  const status = document.getElementById("tierUnlockStatus");
  const fill = document.getElementById("tierUnlockFill");
  const label = document.getElementById("tierUnlockLabel");
  const isPreGrad = tierUnlockState.nextTierIndex === 1 && tierUnlockState.graduationPct != null;
  const displayPct = isPreGrad
    ? Math.max(0, Math.min(100, tierUnlockState.graduationPct || 0))
    : Math.max(0, Math.min(100, tierUnlockState.progressPct || 0));
  if (status) {
    const next = tierUnlockState.nextTier;
    const threshold = formatCompactXrd(tierUnlockState.nextThresholdXrd);
    const current = formatCompactXrd(tierUnlockState.marketCapXrd);
    status.textContent = isPreGrad
      ? (tierUnlockState.graduationPct != null
          ? `GRADUATION ${Math.round(displayPct)}% · RLY.FUN`
          : "GRADUATION · RLY.FUN")
      : next ? `T${next} ${Math.round(displayPct)}% · ${current} / ${threshold}` : `ALL TIERS LIVE · ${current}`;
  }
  if (fill) fill.style.width = `${displayPct}%`;
  if (label) label.textContent = isPreGrad ? "GRADUATION" : tierUnlockState.nextTier ? `NEXT TIER ${tierUnlockState.nextTier}` : "APEX ONLINE";
  document.querySelectorAll(".tier-btn").forEach((button, i) => {
    const locked = i > highestUnlockedTierIndex();
    const next = i === tierUnlockState.nextTierIndex;
    button.classList.toggle("locked", locked);
    button.classList.toggle("next-unlock", next);
    button.setAttribute("aria-disabled", locked ? "true" : "false");
    const progress = button.querySelector(".t-progress span");
    if (progress) progress.style.width = `${Math.round(displayPct)}%`;
  });
  updateBuyAscentLinks();
  setHowToUnlockState(highestUnlockedTierIndex(), tierUnlockState);
}

// Post-graduation the bonding curve is closed: buying happens on the open market (Astrolescent
// aggregates Ociswap and friends). The switch is automatic, driven by the worker's graduated flag.
function updateBuyAscentLinks() {
  const resource = tierUnlockState.tokenResource || market?.resource;
  const href = tierUnlockState.graduated && resource
    ? `https://astrolescent.com/swap?to=${resource}`
    : "https://rly.fun/ASCENT";
  document.querySelectorAll(".shop-buy-ascent-btn, .buy-ascent-btn").forEach(link => { link.href = href; });
}

function applyTierUnlockState(incoming) {
  const previousMax = highestUnlockedTierIndex();
  const wasReady = tierUnlockState.status === "ready";
  tierUnlockState = {
    ...tierUnlockState,
    ...incoming,
    status: "ready",
    graduated: incoming.graduated === true,
    maxUnlockedTierIndex: DEV_UNLOCK_TIERS ? TIERS.length - 1 : Math.max(0, Math.min(TIERS.length - 1, Number(incoming.maxUnlockedTierIndex) || 0)),
    progressPct: Math.max(0, Math.min(100, Number(incoming.progressPct) || 0)),
    graduationPct: incoming.graduationPct != null ? Math.max(0, Math.min(100, Number(incoming.graduationPct) || 0)) : null,
    tier2Status: { unlocked: !!incoming.tier2Status?.unlocked, source: String(incoming.tier2Status?.source || "") }
  };
  const nextIndex = incoming.nextTierIndex === null ? null : Number(incoming.nextTierIndex);
  tierUnlockState.nextTierIndex = Number.isInteger(nextIndex) ? nextIndex : null;
  buildTierPanel();
  updatePreview();
  if (wasReady && highestUnlockedTierIndex() > previousMax) showTierUnlockedToast(highestUnlockedTierIndex());
  if (activeTier > highestUnlockedTierIndex()) {
    pendingTierRelockIndex = highestUnlockedTierIndex();
    if (state !== "playing" && state !== "paused") enforceUnlockedTierAfterRun();
  }
}

async function loadTierUnlocks() {
  try {
    const { sessionToken } = storedWalletState();
    const opts = { errorLabel: "tier unlocks unavailable" };
    if (sessionToken) opts.headers = { Authorization: `Bearer ${sessionToken}` };
    applyTierUnlockState(await apiFetch(`${CALENDAR_API}/tier-unlocks?market=${encodeURIComponent(currentMarket)}`, opts));
  } catch {
    tierUnlockState = { ...tierUnlockState, status:"unavailable" };
    if (DEV_UNLOCK_TIERS) {
      tierUnlockState = { ...tierUnlockState, status:"ready", maxUnlockedTierIndex: TIERS.length - 1 };
      buildTierPanel();
    }
    updateTierUnlockUi();
  }
}

let toastTimer = null;
function showTierToast() {
  const t = tier();
  const el = document.getElementById("tierToast");
  if (!el) return;
  document.getElementById("ttName").textContent = `${activeTier + 1} · ${t.name}`;
  document.getElementById("ttPerk").textContent = t.perk;
  el.classList.add("show");
  sfx.tierup();
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>el.classList.remove("show"), 1800);
}

function showTierUnlockedToast(index) {
  const t = TIERS[index];
  const el = document.getElementById("tierToast");
  if (!t || !el) return;
  document.getElementById("ttName").textContent = `${index + 1} · ${t.name}`;
  document.getElementById("ttPerk").textContent = "NEW TIER UNLOCKED";
  el.classList.add("show");
  sfx.tierup();
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>el.classList.remove("show"), 2600);
}

function showTierLockedToast() {
  const el = document.getElementById("tierToast");
  if (!el) return;
  document.getElementById("ttName").textContent = `${activeTier + 1} · ${tier().name}`;
  document.getElementById("ttPerk").textContent = "FINISH YOUR RUN FIRST";
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>el.classList.remove("show"), 1800);
}

// ── STARS & NEBULA ───────────────────────────────────────────────────────────

// ── WORLD / CAMERA ───────────────────────────────────────────────────────────
// worldY increases upward. screenY = H - (worldY - cameraY)
// cameraY = world-Y at the bottom of the screen.
let cameraY = 0;

function w2s(worldY) { return H - (worldY - cameraY); }

// ── ORB ──────────────────────────────────────────────────────────────────────
let orb = {};

function resetOrb() {
  const startWorldY = cameraY + H * 0.42;
  orb = {
    x: W / 2,
    worldY: startWorldY,
    prevWorldY: startWorldY,
    vx: 0,
    vy: 0,
    radius: ORB_R,
    glow: 0.5,
    trail: [],
    combo: 0,
    maxCombo: 0,
    height: startWorldY,
    startWorldY,
    maxScore: 0,
    bonus: 0,
    bankStyle: 0,   // secured style points, PRE-tier (climb-independent). Tier weight applied at display.
    bounces: 0,
    energy: 100,
    reserve: 0,
    sparkPhase: Math.random() * 7,
    bandPhase: 0,
    squash: 1,
  };
}

// ── PLATFORMS ────────────────────────────────────────────────────────────────
let platforms = [];

function seedPlatforms() {
  platforms = [];
  const baseY  = cameraY + H * 0.38;
  // Max jump height in world-px (no boost)
  const maxJump = (JUMP_FORCE * JUMP_FORCE) / (2 * GRAVITY); // ~250px
  const step    = maxJump * 0.68; // safe gap between levels
  const margin  = 28;

  // Guaranteed wide platform directly under orb start
  const sw = gs * 180;
  platforms.push(neutralPlatform({ worldY:baseY - 14, x:W/2 - sw/2, width:sw, bounce:1.0, age:0, seed:Math.random() },sessionGameplay.platforms.baseBounceLimit));

  // Build a guaranteed reachable staircase near the center,
  // alternating offset so the player needs light steering but always has a target.
  const offsets = [0, -0.18, 0.18, -0.12, 0.12, 0, -0.2, 0.2, -0.08, 0.08, 0, -0.16, 0.16, 0, -0.1, 0.1, 0, 0];
  for (let i = 0; i < 18; i++) {
    const worldY = baseY + gs * 80 + i * step;
    const width  = gs * (90 + Math.random() * 70);   // wider than before
    const cx     = W / 2 + offsets[i] * W;            // near-center X
    const x      = Math.max(margin, Math.min(W - width - margin, cx - width / 2));
    platforms.push(neutralPlatform({ worldY, x, width, bounce:0.95 + Math.random() * 0.18, age:0, seed:Math.random() },sessionGameplay.platforms.baseBounceLimit));
  }
}

function spawnLivePlatforms(pressure) {
  if (!shouldSpawnLivePlatform(pressure)) return;
  // Spawn above the visible area
  const baseY  = cameraY + H * SPAWN_SCREENS;
  const absPrs = pressure;

  const width = gs * Math.max(70, Math.min(180, 90 + absPrs * 170));
  const bounce = Math.max(0.9, Math.min(2.2, 1.0 + absPrs * 0.8));
  const margin = 22;
  const x      = margin + Math.random() * (W - width - margin * 2);
  const worldY = baseY + Math.random() * H * 0.45;

  platforms.push({ worldY, x, width, bounce, type:"buy", age:0, seed:Math.random() });

  // Strong buy surge → second platform formation
  if (pressure > 1.0) {
    platforms.push({
      worldY: worldY + 50 + Math.random() * 40,
      x: W - x - width - 10,
      width: width * 0.65,
      bounce: bounce * 1.12,
      type: "buy", age:0, seed:Math.random()
    });
  }
}

function ensureNeutralCoverage() {
  const sl = stormLevel();
  // Phase 1 (sl 0-3) : 6→3. Phase 2 (sl 8+) : descend lentement vers 2
  const MIN_AHEAD = sl <= 3
    ? Math.max(3, 6 - sl)
    : Math.max(2, 3 - Math.floor((sl - 3) / 5));
  const lo = Math.min(orb.worldY - H * 0.2, cameraY + H * 0.35);
  const hi = Math.max(cameraY + H * (SPAWN_SCREENS + 0.8), orb.worldY + H * 1.15);
  const ahead = platforms.filter(p => p.worldY >= lo && p.worldY <= hi).length;
  const needed = MIN_AHEAD - ahead;
  for (let i = 0; i < needed; i++) {
    const spawnMin = Math.max(cameraY + H * 0.55, orb.worldY - H * 0.18);
    const spawnMax = Math.max(spawnMin + gs * 90, hi);
    const worldY = spawnMin + Math.random() * (spawnMax - spawnMin);
    const shrink = sl * 10;
    const width  = gs * Math.max(25, (80 - shrink) + Math.random() * 60);
    const margin = 22;
    const x = margin + Math.random() * (W - width - margin * 2);
    platforms.push(neutralPlatform(
      { worldY, x, width, bounce: 0.95 + Math.random() * 0.15, age: 0, seed: Math.random() },
      Math.max(1, sessionGameplay.platforms.baseBounceLimit - sl)
    ));
  }
}

function spawnNotableBuyPlatform(strength) {
  const { width, bounce } = notableBuyPlatformStats(strength,gs);
  const margin = 22;
  platforms.push({
    worldY:cameraY+H*(.72+Math.random()*.82),
    x:margin+Math.random()*(W-width-margin*2),
    width,
    bounce,
    type:"buy",
    age:0,
    seed:Math.random()
  });
}

// ── BOOSTERS ──────────────────────────────────────────────────────────────────
let boosters = [];
let boosterSpawnTimer = 0;

function boosterRunHeight() {
  return Math.max(0,(orb.height || 0) - (orb.startWorldY || 0));
}

function mkBooster(type,x,worldY) {
  const config = sessionGameplay.boosters[type];
  const difficulty = boosterDifficulty(boosterRunHeight(),H,sessionGameplay.boosters.progressionScreens);
  // Red walls (drag) don't shrink with difficulty but their gs-based width would block
  // ~42% of a phone's width — narrow screens get a flat reduction instead.
  const widthScale = type === "drag" ? 1 - 0.15 * narrowScreenRamp() : 1 - difficulty * .25;
  const baseWidth = type === "stream" ? gs*64 : type === "surge" ? gs*30 : gs*150;
  const booster = {
    type,x,worldY,age:0,used:false,entered:false,phase:Math.random()*7,
    w:baseWidth * config.size * widthScale,
    h:gs*16
  };
  if (type === "stream") booster.len = gs * config.length;
  return booster;
}

function spawnBooster(worldY = cameraY + H * (1.05 + Math.random() * .55)) {
  const type = pickBoosterType(boosterWeights(sessionGameplay.boosters,livePressure,boosterRunHeight(),H));
  const margin = 40;
  const booster = mkBooster(type,W/2,worldY);
  booster.x = Math.max(margin + booster.w/2,Math.min(W - margin - booster.w/2,margin + Math.random() * (W - margin*2)));
  boosters.push(booster);
}

function seedBoosters() {
  boosters = [];
  boosterSpawnTimer = sessionGameplay.boosters.startIntervalMs / 1000;
  for (let i=0;i<3;i++) spawnBooster(cameraY + H * (.82 + i*.38));
}

function collectBooster(booster) {
  orb.combo = Math.min(orb.combo + 1, 999);
  orb.maxCombo = Math.max(orb.maxCombo,orb.combo);
  if (orb.combo >= 3) { showCombo(); sfx.combo(); }
  updateMult();
  const color = booster.type === "surge" ? "#ffdd88" : tier().orbG;
  if (tier().particles) burst(orb.x,w2s(orb.worldY),color,booster.type === "surge" ? 16 : 8,{spread:120,up:90});
  if (tier().shockwave) addShockwave(orb.x,w2s(orb.worldY));
}

function updateBoosters(dt) {
  boosterSpawnTimer -= dt;
  if (boosterSpawnTimer <= 0) {
    spawnBooster();
    boosterSpawnTimer = boosterSpawnInterval(sessionGameplay.boosters,boosterRunHeight(),H) / 1000;
  }
  for (const booster of boosters) {
    booster.age += dt;
    if (booster.type === 'surge' && !booster.used) {
      const bMods = getUltiModifiers();
      if (!bMods.magnetLarge) {
        const dyWorld = Math.abs(booster.worldY - orb.worldY);
        const dxWorld = Math.abs(orb.x - booster.x);
        const dist = Math.sqrt(dyWorld * dyWorld + dxWorld * dxWorld);
        const attractRadius = gs * 90;
        if (dist < attractRadius && dist > 0) {
          const pullSpeed = 220;
          const dxToOrb = orb.x - booster.x;
          const margin = 40 + booster.w / 2;
          booster.x = Math.max(margin, Math.min(W - margin,
            booster.x + Math.sign(dxToOrb) * Math.min(Math.abs(dxToOrb), pullSpeed * dt)
          ));
        }
      }
    }
    const config = sessionGameplay.boosters[booster.type];
    if (booster.type === "stream") {
      const insideX = Math.abs(orb.x - booster.x) < booster.w/2;
      const insideY = orb.worldY > booster.worldY - orb.radius && orb.worldY < booster.worldY + booster.len + orb.radius;
      if (insideX && insideY) {
        const justEntered = !booster.inside;
        booster.inside = true;
        orb.vy = Math.min(MAX_ORB_VY,streamAscentVelocity(orb.vy,config,dt,{ entered:justEntered }));
        orb.energy = Math.min(100,orb.energy + config.energyPerSecond * dt);
        orb.bonus = Math.min(MAX_ORB_BONUS, orb.bonus + config.scorePerSecond * dt);
        orb.glow = Math.min(1.6,orb.glow + dt*1.5);
        if (justEntered) {
          if (!booster.entered) {
            booster.entered = true;
            collectBooster(booster);
          }
          const sy = w2s(orb.worldY);
          burst(orb.x,sy,tier().orbG,18,{up:120,spread:90,size:3.5});
          shockwaves.push({x:orb.x,y:sy,r:0,life:.8});
          shakeI = Math.max(shakeI,1.8);
        }
        if (tier().particles && Math.random() < dt*22) burst(orb.x,w2s(orb.worldY),tier().orbG,1,{up:30,spread:30});
      } else {
        booster.inside = false;
      }
      continue;
    }
    if (booster.used) continue;
    if (orb.vy <= 0 && booster.type !== 'surge') continue;
    const bMods = getUltiModifiers();
    if (booster.type === "drag" && bMods.noDragBooster) { booster.used = true; continue; }
    const magnetScale = (booster.type !== "drag" && bMods.magnetLarge) ? 3 : 1;
    const tolerance = booster.type === "drag" ? orb.radius*.9 : orb.radius*.35 * magnetScale;
    const withinX = Math.abs(orb.x - booster.x) < booster.w/2 * magnetScale + tolerance;
    const crossed = orb.prevWorldY <= booster.worldY && orb.worldY >= booster.worldY;
    const near = Math.abs(orb.worldY - booster.worldY) < orb.radius*.7 + booster.h*.5;
    if (!withinX || (!crossed && !near)) continue;
    booster.used = true;
    if (booster.type === "drag") {
      orb.vy = orb.vy * config.slow - config.velocityPenalty;
      orb.energy = Math.max(0,orb.energy + config.energy);
      breakCombo();
      orb.squash = 1.3;
      if (tier().shake) shakeI = Math.max(shakeI,2.4);
      if (tier().particles) burst(orb.x,w2s(orb.worldY),"#ff4d6d",12,{spread:140,up:30});
      sfx.sell();
      platforms.push(neutralPlatform({
        worldY:booster.worldY,
        x:booster.x - booster.w / 2,
        width:booster.w,
        bounce:1.0,
        age:0,
        seed:Math.random()
      },sessionGameplay.platforms.baseBounceLimit));
    } else {
      orb.vy = Math.min(MAX_ORB_VY,Math.max(orb.vy,60) + config.boost);
      orb.energy = Math.min(100,orb.energy + config.energy);
      orb.bonus = Math.min(MAX_ORB_BONUS, orb.bonus + config.score);
      orb.squash = .72;
      collectBooster(booster);
      sfx.surge();
    }
  }
  boosters = boosters.filter(booster => booster.worldY + (booster.len || 0) > cameraY - H*.7);
  updateEnergyBar();
}

// ── PARTICLES & SHOCKWAVES ───────────────────────────────────────────────────
let particles = [];
let magnetParticles = [];
let chronoRamp = 0; // 0 = normal speed, 1 = full bullet-time (ramped in/out)
let shockwaves = [];
let rocketMode = 0;
let rocketModeDuration = 0;
let rocketModeTotalDuration = 1.4;
let rocketModePeak = 1.0;
let buyChartFlash = 0;
let buyChartGlowDuration = 0;
let buyChartGlowPower = 0;
let sellChartFlash = 0;
let rocketPhase = 'off';
let rocketPhaseTimer = 0;
let rocketFlameCyan = false; // true for the VOID portal launch, false for buy rockets
const ROCKET_ATTACK_DUR = 0.15;
let rocketSustainDuration = 0;
let rocketDecayDuration = 0;

function armBuyChartGlow(duration, power = 1) {
  const safeDuration = Math.max(0, Number(duration) || 0);
  if (safeDuration <= 0) {
    buyChartGlowDuration = 0;
    buyChartGlowPower = 0;
    return;
  }
  buyChartGlowDuration = Math.max(buyChartGlowDuration, safeDuration);
  buyChartGlowPower = Math.max(buyChartGlowPower, Math.max(0, Number(power) || 0));
}

// ── ULTI VISUAL TIMERS ────────────────────────────────────────────────────────
let pillarFlash = 0;        // PILLAR: vertical beam, seconds remaining (now 6s)
let pillarFlashX = 0;       // PILLAR: orb X at activation (kept for compat)
let pillarRedRipples = [];  // PILLAR: [{x,y,r,life}] rings at sell-platform pass-through
let resonanceFlash = 0;     // RESONANCE: full-charge explosion flash
let resonanceLightning = []; // RESONANCE: electric arcs at explosion
let screenFlash = 0;        // white overlay alpha (PHOENIX rebirth etc.)
let phoenixBlinkTimer = 0;  // PHOENIX: orb blink timer on rebirth teleport
let afterImages = [];       // OVERCLOCK: [{x,sy,r,alpha}] ghost orbs

const MAX_PARTICLES = isMobile ? 80 : 150;
function burst(x, y, color, amount=7, opts={}) {
  const available = QF.particleCap - particles.length;
  if (available <= 0) return;
  const actualAmount = Math.min(amount, available);
  const spread = opts.spread || 65;
  const up     = opts.up     || 55;
  for (let i=0; i<actualAmount; i++) {
    particles.push({
      x, y,
      vx: (Math.random()-.5)*spread,
      vy: -(Math.random()*up + 8),
      life: 1,
      color,
      size: opts.size||3,
    });
  }
}
function addShockwave(x, y) {
  shockwaves.push({ x, y, r:0, life:1 });
}

function setAnomalyButtonsActive() {
  document.querySelectorAll("[data-anomaly-trigger]").forEach(button => {
    button.classList.toggle("active", button.dataset.anomalyTrigger === activeAnomaly?.type);
  });
}

function initAnomalyTestPanel() {
  const panel = document.getElementById("anomalyTestPanel");
  if (!panel) return;
  panel.classList.toggle("hidden", !location.hostname.includes("ascent-test"));
}

function makeShortSqueezeHazards(count) {
  return Array.from({ length: count }, (_, index) => ({
    x: W * (0.18 + index * (0.64 / Math.max(1, count - 1))) + (Math.random() - 0.5) * W * 0.12,
    worldY: cameraY + H * (0.85 + Math.random() * 0.7),
    r: gs * (16 + Math.random() * 7),
    speed: gs * (250 + Math.random() * 150),
    phase: Math.random() * 7
  }));
}

function makeLiquidityVoidHazards(count, portal) {
  return Array.from({ length: count }, (_, index) => {
    const lane = (index + 1) / (count + 1);
    // Spread across the frozen chamber band (camera-relative), not above the orb.
    const worldY = cameraY + H * (0.12 + lane * 0.74) + (Math.random() - 0.5) * H * 0.12;
    return {
      kind: "voidShard",
      x: Math.max(gs * 34, Math.min(W - gs * 34, portal.x + (Math.random() - 0.5) * W * 0.86)),
      worldY,
      r: gs * (13 + Math.random() * 8),
      vx: gs * ((Math.random() - 0.5) * 52),
      vy: gs * ((Math.random() - 0.5) * 36),
      spin: (Math.random() - 0.5) * 2.4,
      phase: Math.random() * 7
    };
  });
}

function makeLiquidityVoidPortal() {
  const radius = gs * (isMobile ? 34 : 42);
  // Placed inside the frozen chamber (camera is locked at void entry) so it's always
  // on-screen, in the upper band and horizontally offset to force the player to
  // navigate toward it rather than just flying up.
  return {
    x: Math.max(radius + gs * 24, Math.min(W - radius - gs * 24, orb.x + (Math.random() - 0.5) * W * 0.46)),
    worldY: cameraY + H * (0.62 + Math.random() * 0.26),
    r: radius,
    phase: Math.random() * 7
  };
}

// Real run-elapsed seconds (gameStartTime is shifted forward on resume, so pauses
// are excluded). Wall-clock on purpose: keeps anomaly cadence identical on every
// tier and immune to climb-speed bursts (ultimates), and — like anomalyMinDelay —
// CHRONO must not stretch it.
function runElapsedSeconds() {
  return (performance.now() - gameStartTime) / 1000;
}

// Flat 25–35 s gap between anomalies — same on every tier and the first anomaly too.
function anomalyTimeRange() {
  return 25 + Math.random() * 10;
}

function resetAutoAnomalySchedule() {
  autoAnomalyCount = 0;
  anomalyMinDelay = 0;
  nextAnomalyAt = runElapsedSeconds() + anomalyTimeRange();
}

function scheduleNextAutoAnomaly() {
  autoAnomalyCount += 1;
  nextAnomalyAt = runElapsedSeconds() + anomalyTimeRange();
}

function maybeTriggerAutoAnomaly() {
  if (activeAnomaly || state !== "playing" || anomalyMinDelay > 0) return;
  if (runElapsedSeconds() < nextAnomalyAt) return;
  const type = ANOMALY_TYPES[Math.floor(Math.random() * ANOMALY_TYPES.length)];
  triggerAnomaly(type, { auto: true });
}

function triggerAnomaly(type, { auto = false } = {}) {
  const def = anomalyDefinition(type);
  if (!def || !canTriggerAnomaly(state)) return;
  const portal = type === "liquidityVoid" ? makeLiquidityVoidPortal() : null;
  activeAnomaly = {
    type,
    def,
    remaining: def.durationSeconds,
    elapsed: 0,
    exitApplied: false,
    portal,
    auto
  };
  anomalyHazards = type === "shortSqueeze"
    ? makeShortSqueezeHazards(def.hazardCount || 4)
    : type === "liquidityVoid"
      ? makeLiquidityVoidHazards(isMobile ? 5 : 8, portal)
      : [];
  deferredAnomalyBuyStrength = 0;
  deferredAnomalyBuyCount = 0;
  _voidDebrisSkip = 0;
  _portalGradCache = null; _portalGradCacheR = -1;
  _darkPoolLastX = -999; _darkPoolLastY = -999; _darkPoolLastR = -999;
  const sy = w2s(orb.worldY);
  burst(orb.x, sy, def.color, isMobile ? 8 : 32, { spread: 130, up: 100, size: 4 });
  shockwaves.push({ x: orb.x, y: sy, r: 0, life: 0.9 });
  shakeI = Math.max(shakeI, type === "shortSqueeze" ? 4 : 2.2);
  screenFlash = Math.max(screenFlash, 0.35);
  if (type === "shortSqueeze") sfx.sell();
  else sfx.surge();
  setAnomalyButtonsActive();
}

function clearAnomaly() {
  activeAnomaly = null;
  anomalyHazards = [];
  deferredAnomalyBuyStrength = 0;
  deferredAnomalyBuyCount = 0;
  setAnomalyButtonsActive();
}

function releaseDeferredAnomalyBuys() {
  if (deferredAnomalyBuyStrength <= 0) return;
  const strength = Math.min(6, deferredAnomalyBuyStrength);
  const power = Math.min(1.4, strength / 3);
  const bx = orb.x, by = w2s(orb.worldY);
  orb.vy = Math.min(MAX_ORB_VY + 550, orb.vy + sessionGameplay.impacts.buyImpulse * power);
  orb.bonus = Math.min(MAX_ORB_BONUS, orb.bonus + Math.round(90 * deferredAnomalyBuyCount + 70 * strength));
  buyChartFlash = Math.max(buyChartFlash, 0.75);
  armBuyChartGlow(0.9 + power, 0.75);
  burst(bx, by, tier().platBuy, Math.ceil(12 + strength * 5), { up: 130, spread: 120, size: 4 });
  shockwaves.push({ x: bx, y: by, r: 0, life: 0.75 });
  shakeI = Math.max(shakeI, 1.8 + power * 2);
  sfx.surge();
  deferredAnomalyBuyStrength = 0;
  deferredAnomalyBuyCount = 0;
}

function finishAnomaly(reachedPortal = false) {
  if (!activeAnomaly || activeAnomaly.exitApplied) return;
  const { def, type, auto } = activeAnomaly;
  activeAnomaly.exitApplied = true;
  // VOID is the only anomaly you can fail: timing out without reaching the portal grants
  // nothing (no height, no bonus points, energy left untouched). Every other anomaly —
  // and the void when the portal is reached — rewards on exit as before.
  const voidFailed = type === "liquidityVoid" && !reachedPortal;
  // A x2 buff ending isn't the player's fault: bank only the portion the multiplier adds so the
  // displayed number stays continuous when it drops to x1, WITHOUT wiping the reservoir/combo — the
  // momentum survives (unlike a red hit). activeAnomaly is still set here, so styleLive reads x2.
  // Runs even on a void fail: this is display continuity, not a reward (no-op for the void
  // since scoreMult is 1, but stays correct if that ever changes).
  const aMult = anomalyScoreMultiplier(type);
  if (aMult > 1) orb.bankStyle += STYLE_WEIGHT * orb.bonus * comboMult() * (aMult - 1);
  releaseDeferredAnomalyBuys();
  const sy = w2s(orb.worldY);
  if (voidFailed) {
    burst(orb.x, sy, "#6b7a8a", isMobile ? 6 : 12, { spread: 80, up: 30, size: 2.5 });
  } else {
    if (Number.isFinite(def.exitBonus)) orb.bonus = Math.min(MAX_ORB_BONUS, orb.bonus + def.exitBonus);
    if (Number.isFinite(def.exitEnergy)) {
      orb.energy = Math.max(orb.energy, def.exitEnergy);
      updateEnergyBar();
    }
    burst(orb.x, sy, def.color, isMobile ? 16 : 28, { spread: 110, up: 130, size: 3.5 });
    shockwaves.push({ x: orb.x, y: sy, r: 0, life: 0.75 });
    if (type === "shortSqueeze") orb.vy = Math.max(orb.vy, 220);
    // VOID success: the portal is the elevator. Instead of teleporting (which snaps the
    // world), arm a rocket-thrust envelope (attack→sustain→decay): the orb visibly ignites
    // and accelerates upward while the camera — unfrozen on clearAnomaly — follows, so the
    // bonus height is felt and read as a launch, not snapped in.
    if (type === "liquidityVoid") {
      orb.vy = Math.max(orb.vy, 450);
      rocketModePeak          = 1.0;
      rocketSustainDuration   = 0.65;
      rocketDecayDuration     = 1.15;
      rocketModeTotalDuration = ROCKET_ATTACK_DUR + rocketSustainDuration + rocketDecayDuration;
      rocketModeDuration      = rocketModeTotalDuration;
      rocketFlameCyan         = true;
      rocketPhase             = 'attack';
      rocketPhaseTimer        = 0;
      rocketMode              = 0;
    }
  }
  clearAnomaly();
  anomalyMinDelay = 20;
  if (auto) scheduleNextAutoAnomaly();
}

function consumeAegisForAnomalyHit() {
  const aegisSlot = ultiActiveStates.findIndex(s => s?.tier === 2 && s?.phase === 'active' && !s?.counters?.shieldUsed);
  if (aegisSlot < 0) return false;
  ultiActiveStates[aegisSlot] = null;
  const sy = w2s(orb.worldY);
  burst(orb.x, sy, TIERS[2].orb.bloom, 30, { up: 140, spread: 120, size: 4 });
  for (let i = 0; i < 3; i++) shockwaves.push({ x: orb.x, y: sy, r: i * 24, life: 1 - i * 0.1 });
  shakeI = Math.max(shakeI, 3);
  sfx.aegis_save();
  updateUltiHud();
  return true;
}

function liquidityVoidAimVector() {
  if (touchActive && touchClientX !== null && touchClientY !== null) {
    const rect = canvasRect();
    const localX = (touchClientX - rect.left) * (W / rect.width);
    const localY = (touchClientY - rect.top) * (H / rect.height);
    const dx = localX - orb.x;
    const dy = w2s(orb.worldY) - localY;
    const len = Math.hypot(dx, dy);
    if (len > 8) return { x: dx / len, y: dy / len };
  }
  let ax = 0, ay = 0;
  if (keys.left || gamepadMove.left) ax -= 1;
  if (keys.right || gamepadMove.right) ax += 1;
  if (keys.up || gamepadMove.up) ay += 1;
  if (keys.down || gamepadMove.down) ay -= 1;
  const len = Math.hypot(ax, ay);
  return len > 0 ? { x: ax / len, y: ay / len } : { x: 0, y: 0 };
}

function applyShortSqueezeHit(hazard) {
  orb.energy = Math.max(0, orb.energy - SHORT_SQUEEZE_HIT_ENERGY_LOSS);
  breakCombo();
  updateEnergyBar();
  hazard.worldY = cameraY + H * 1.35;
  hazard.x = Math.max(hazard.r, Math.min(W - hazard.r, hazard.r + Math.random() * (W - hazard.r * 2)));
  const sy = w2s(orb.worldY);
  burst(orb.x, sy, "#ff4d6d", isMobile ? 10 : 22, { spread: 120, up: 40, size: 3.5 });
  shockwaves.push({ x: orb.x, y: sy, r: 0, life: 0.6 });
  shakeI = Math.max(shakeI, 3.2);
  screenFlash = Math.max(screenFlash, 0.22);
  sfx.sell();
}

function resetVoidShard(hazard) {
  hazard.x = Math.max(hazard.r, Math.min(W - hazard.r, hazard.r + Math.random() * (W - hazard.r * 2)));
  // Respawn inside the frozen chamber band so no shard reappears off-screen.
  hazard.worldY = cameraY + H * (0.08 + Math.random() * 0.87);
  hazard.vx = gs * ((Math.random() - 0.5) * 52);
  hazard.vy = gs * ((Math.random() - 0.5) * 36);
  hazard.spin = (Math.random() - 0.5) * 2.4;
  hazard.phase = Math.random() * 7;
}

function applyVoidShardHit(hazard) {
  orb.energy = Math.max(0, orb.energy - VOID_SHARD_HIT_ENERGY_LOSS);
  breakCombo();
  // No velocity setback — parity with the short-squeeze spike: void breaks the combo and costs
  // a little energy, but doesn't kill the orb's physical momentum.
  updateEnergyBar();
  const sy = w2s(orb.worldY);
  burst(orb.x, sy, "#ff4d6d", isMobile ? 8 : 18, { spread: 105, up: 35, size: 3 });
  shockwaves.push({ x: orb.x, y: sy, r: 0, life: 0.5 });
  shakeI = Math.max(shakeI, 2.4);
  screenFlash = Math.max(screenFlash, 0.18);
  sfx.sell();
  resetVoidShard(hazard);
}

function updateAnomaly(dt) {
  if (!activeAnomaly) return;
  activeAnomaly.elapsed += dt;
  activeAnomaly.remaining -= dt;
  if (activeAnomaly.type === "liquidityVoid") {
    const portal = activeAnomaly.portal;
    if (portal) {
      const dx = orb.x - portal.x;
      const dy = orb.worldY - portal.worldY;
      if (Math.hypot(dx, dy) < orb.radius + portal.r * 0.72) {
        finishAnomaly(true);
        return;
      }
    }
    const decay = Math.pow(0.94, dt);
    for (const hazard of anomalyHazards) {
      hazard.x += hazard.vx * dt;
      hazard.worldY += hazard.vy * dt;
      hazard.vx += Math.sin(activeAnomaly.elapsed * 0.8 + hazard.phase) * gs * 12 * dt;
      hazard.vy += Math.cos(activeAnomaly.elapsed * 0.7 + hazard.phase) * gs * 10 * dt;
      hazard.vx *= decay;
      hazard.vy *= decay;
      if (hazard.x < -hazard.r || hazard.x > W + hazard.r || hazard.worldY < cameraY - H * 0.35 || hazard.worldY > cameraY + H * 2.25) {
        resetVoidShard(hazard);
      }
      const dx = orb.x - hazard.x;
      const dy = orb.worldY - hazard.worldY;
      if (Math.hypot(dx, dy) < orb.radius + hazard.r * 0.8) {
        if (consumeAegisForAnomalyHit()) resetVoidShard(hazard);
        else applyVoidShardHit(hazard);
        return;
      }
    }
  }
  if (activeAnomaly.type === "shortSqueeze") {
    for (const hazard of anomalyHazards) {
      hazard.worldY -= hazard.speed * dt;
      if (hazard.worldY < cameraY - H * 0.2) {
        hazard.worldY = cameraY + H * (1.05 + Math.random() * 0.7);
        hazard.x = Math.max(hazard.r, Math.min(W - hazard.r, hazard.r + Math.random() * (W - hazard.r * 2)));
      }
      const dx = orb.x - hazard.x;
      const dy = orb.worldY - hazard.worldY;
      if (Math.hypot(dx, dy) < orb.radius + hazard.r * 0.75) {
        if (consumeAegisForAnomalyHit()) {
          hazard.worldY = cameraY + H * 1.35;
        } else {
          applyShortSqueezeHit(hazard);
        }
        return;
      }
    }
  }
  if (activeAnomaly.remaining <= 0) finishAnomaly();
}

// ── SCREEN SHAKE ─────────────────────────────────────────────────────────────
let shakeI = 0, shakeX = 0, shakeY = 0;

// ── GAME STATE ───────────────────────────────────────────────────────────────
let state = "ready";
let last  = performance.now();
let gameStartTime = 0;
let runTier = 0; // tier frozen at run start; leaderboard submissions use this, never activeTier

function stormLevel() {
  if (state !== "playing") return 0;
  return Math.floor((performance.now() - gameStartTime) / 60000);
}
let pauseStartedAt = 0;
let pausedByOptions = false;

// ── INPUT ────────────────────────────────────────────────────────────────────
const keys = { left:false, right:false, up:false, down:false };
let touchActive = false;
let touchClientX = null;
let touchClientY = null;
// Canvas rect cached for touch-drag mapping — getBoundingClientRect() forces
// layout, so we refresh it on touchstart/resize instead of every physics frame.
let cachedCanvasRect = null;
function canvasRect() {
  return cachedCanvasRect || (cachedCanvasRect = canvas.getBoundingClientRect());
}
const GAMEPAD_AXIS_DEADZONE = 0.35;
const GAMEPAD_CAPTURE_AXIS_THRESHOLD = 0.65;
const gamepadButtons = [];
const gamepadAxes = [];
let activeGamepadIndex = null;
let gamepadConnected = false;
let gamepadMove = { left:false, right:false, up:false, down:false };

function getActiveGamepad() {
  if (typeof navigator.getGamepads !== "function") return null;
  const pads = Array.from(navigator.getGamepads()).filter(Boolean);
  if (!pads.length) return null;
  const indexed = activeGamepadIndex !== null ? pads.find(pad => pad.index === activeGamepadIndex) : null;
  return indexed || pads[0];
}

function updateGamepadStatusUI() {
  const el = document.getElementById("gamepadStatus");
  const map = document.getElementById("gamepadMap");
  const buttons = document.querySelectorAll("[data-gamepad-action]");
  const setConnected = connected => {
    map?.classList.toggle("hidden", !connected);
    buttons.forEach(button => { button.disabled = !connected; });
    if (!connected && rebindingGamepadAction) finishGamepadCapture("GAMEPAD NOT DETECTED");
  };
  if (!el) return;
  if (typeof navigator.getGamepads !== "function") {
    el.textContent = "UNSUPPORTED";
    el.classList.remove("on");
    setConnected(false);
    return;
  }
  const pad = getActiveGamepad();
  el.textContent = pad ? "CONNECTED" : "NOT DETECTED";
  el.classList.toggle("on", Boolean(pad));
  setConnected(Boolean(pad));
}

function gamepadBindingActive(pad, binding) {
  if (!pad || !binding) return false;
  if (binding.type === "axis") {
    const value = pad.axes[binding.index] || 0;
    return value * binding.direction > GAMEPAD_AXIS_DEADZONE;
  }
  const button = pad.buttons[binding.index];
  return Boolean(button?.pressed || button?.value > 0.5);
}

function gamepadBindingsEqual(left, right) {
  if (!left || !right || left.type !== right.type || left.index !== right.index) return false;
  return left.type === "button" || left.direction === right.direction;
}

function gamepadBindingConflict(action, binding) {
  return Object.entries(gamepadBindings).some(([otherAction, otherBinding]) =>
    otherAction !== action && gamepadBindingsEqual(binding, otherBinding)
  );
}

function captureGamepadBinding(pad) {
  if (!pad || !rebindingGamepadAction) return null;
  for (let i = 0; i < pad.buttons.length; i++) {
    const pressed = Boolean(pad.buttons[i]?.pressed || pad.buttons[i]?.value > 0.5);
    if (pressed && !gamepadButtons[i]) return { type:"button", index:i };
  }
  for (let i = 0; i < pad.axes.length; i++) {
    const value = pad.axes[i] || 0;
    const previous = gamepadAxes[i] || 0;
    if (Math.abs(value) >= GAMEPAD_CAPTURE_AXIS_THRESHOLD && Math.abs(previous) < GAMEPAD_AXIS_DEADZONE) {
      return { type:"axis", index:i, direction:value < 0 ? -1 : 1 };
    }
  }
  return null;
}

function updateGamepadInput() {
  const pad = getActiveGamepad();
  gamepadConnected = Boolean(pad);
  gamepadMove = { left:false, right:false, up:false, down:false };
  if (!pad) {
    gamepadButtons.length = 0;
    gamepadAxes.length = 0;
    if (rebindingGamepadAction) finishGamepadCapture("GAMEPAD NOT DETECTED");
    updateGamepadStatusUI();
    return;
  }
  activeGamepadIndex = pad.index;

  const captured = captureGamepadBinding(pad);
  if (captured && rebindingGamepadAction) {
    if (gamepadBindingConflict(rebindingGamepadAction, captured)) {
      setOptionsMessage("GAMEPAD INPUT ALREADY USED");
      gamepadButtons.length = pad.buttons.length;
      for (let i = 0; i < pad.buttons.length; i++) gamepadButtons[i] = Boolean(pad.buttons[i]?.pressed || pad.buttons[i]?.value > 0.5);
      gamepadAxes.length = pad.axes.length;
      for (let i = 0; i < pad.axes.length; i++) gamepadAxes[i] = pad.axes[i] || 0;
      updateMixerUI();
      return;
    }
    gamepadBindings[rebindingGamepadAction] = captured;
    saveOptions();
    finishGamepadCapture("GAMEPAD INPUT UPDATED");
    gamepadButtons.length = pad.buttons.length;
    for (let i = 0; i < pad.buttons.length; i++) gamepadButtons[i] = Boolean(pad.buttons[i]?.pressed || pad.buttons[i]?.value > 0.5);
    gamepadAxes.length = pad.axes.length;
    for (let i = 0; i < pad.axes.length; i++) gamepadAxes[i] = pad.axes[i] || 0;
    updateGamepadStatusUI();
    return;
  }

  const wasPressed = index => Boolean(gamepadButtons[index]);
  const pressedNow = index => Boolean(pad.buttons[index]?.pressed || pad.buttons[index]?.value > 0.5);
  const actionPressed = action => gamepadBindingActive(pad, gamepadBindings[action]);
  const actionEdge = action => {
    const binding = gamepadBindings[action];
    if (!binding) return false;
    if (binding.type === "axis") {
      const value = (pad.axes[binding.index] || 0) * binding.direction;
      const previous = (gamepadAxes[binding.index] || 0) * binding.direction;
      return value > GAMEPAD_AXIS_DEADZONE && previous <= GAMEPAD_AXIS_DEADZONE;
    }
    return pressedNow(binding.index) && !wasPressed(binding.index);
  };

  gamepadMove = {
    left:actionPressed("left"),
    right:actionPressed("right"),
    up:(pad.axes[1] || 0) < -GAMEPAD_AXIS_DEADZONE,
    down:(pad.axes[1] || 0) > GAMEPAD_AXIS_DEADZONE
  };
  if (actionEdge("boost")) tryBoost();
  if (actionEdge("pause")) togglePause();
  for (let i = 0; i < 3; i++) {
    if (actionEdge(`ulti${i+1}`)) tryUlti(i);
  }

  gamepadButtons.length = pad.buttons.length;
  for (let i = 0; i < pad.buttons.length; i++) gamepadButtons[i] = pressedNow(i);
  gamepadAxes.length = pad.axes.length;
  for (let i = 0; i < pad.axes.length; i++) gamepadAxes[i] = pad.axes[i] || 0;
  updateGamepadStatusUI();
}

function updatePauseUI() {
  const paused = state === "paused";
  const button = document.getElementById("pauseBtn");
  button.disabled = state !== "playing" && !paused;
  button.classList.toggle("paused",paused);
  button.setAttribute("aria-label",paused ? "Resume game" : "Pause game");
  button.querySelector(".pause-icon")?.classList.toggle("hidden",paused);
  button.querySelector(".resume-icon")?.classList.toggle("hidden",!paused);
  document.getElementById("pauseOverlay")?.classList.toggle("hidden",!paused);
  document.getElementById("tierList")?.classList.toggle("run-locked", state === "playing" || paused);
  const resumeHint = document.querySelector(".pause-card small");
  if (resumeHint) resumeHint.textContent = `${displayKey(pauseKey)} TO RESUME`;
}

function pauseGame() {
  if (state !== "playing") return;
  state = "paused";
  pauseStartedAt = performance.now();
  impactQueue = [];
  pendingTradeFlow = {buy:0,sell:0};
  keys.left = false;
  keys.right = false;
  keys.up = false;
  keys.down = false;
  touchActive = false;
  touchClientX = null;
  touchClientY = null;
  gamepadMove = { left:false, right:false, up:false, down:false };
  bgMusic.pause();
  updateLeaderboardButton();
  updatePauseUI();
}

function resumeGame() {
  if (state !== "paused") return;
  const pauseDuration = performance.now() - pauseStartedAt;
  gameStartTime += pauseDuration;
  lastImpactAppliedAt += pauseDuration;
  if (sellDebuffUntil > 0) sellDebuffUntil += pauseDuration;
  pauseStartedAt = 0;
  impactQueue = [];
  pendingTradeFlow = {buy:0,sell:0};
  state = "playing";
  updateLeaderboardButton();
  updatePauseUI();
  syncMusic();
}

function togglePause() {
  if (state === "playing") pauseGame();
  else if (state === "paused") resumeGame();
}

function closeOpenPanel() {
  const howTo = document.getElementById("howToPlayOverlay");
  if (!howTo?.classList.contains("hidden")) {
    closeInfoOverlay("howToPlayOverlay");
    return true;
  }
  const leaderboard = document.getElementById("leaderboardOverlay");
  if (!leaderboard?.classList.contains("hidden")) {
    closeLeaderboard();
    return true;
  }
  const mixer = document.getElementById("mixerPanel");
  const tracklist = document.getElementById("tracklistPanel");
  if (mixer?.classList.contains("hidden") && tracklist?.classList.contains("hidden")) return false;
  finishUltiKeyCapture();
  finishGamepadCapture();
  mixer?.classList.add("hidden");
  tracklist?.classList.add("hidden");
  document.getElementById("mixerBtn")?.classList.remove("active-mix");
  document.getElementById("trackListBtn")?.classList.remove("active-mix");
  if (pausedByOptions) { pausedByOptions = false; resumeGame(); }
  return true;
}

function tryBoost() {
  if (state === "ready" || state === "finished" || state === "crashed") {
    startGame(); return;
  }
  if (state !== "playing") return;
  const mods = getUltiModifiers();
  if (mods.noBoost) { sfx.sell(); return; }
  const cost = activeAnomaly?.type === "liquidityVoid" ? 0 : BOOST_COST * mods.boostCostMult;
  if (orb.energy + orb.reserve < cost) { sfx.sell(); return; }
  if (activeAnomaly?.type === "liquidityVoid") {
    const aim = liquidityVoidAimVector();
    const impulse = BOOST_FORCE * 0.72;
    orb.vx += aim.x * impulse;
    orb.vy += aim.y * impulse;
  } else {
    orb.vy += BOOST_FORCE * trendMultipliers().boost;
  }
  let boostRem = cost;
  if (orb.reserve > 0) {
    const fromReserve = Math.min(orb.reserve, boostRem);
    orb.reserve -= fromReserve;
    boostRem -= fromReserve;
    updateReserveBar();
  }
  orb.energy = Math.max(0, orb.energy - boostRem);
  orb.squash = 1.65;
  sfx.boost();
  const bx = orb.x, by = w2s(orb.worldY);
  if (tier().particles) burst(bx, by, tier().orbG, 14, { up: 130, spread: 90, size: 3.5 });
  shockwaves.push({ x: bx, y: by, r: 0, life: 0.65 });
}

window.addEventListener("keydown", e => {
  if (rebindingGamepadAction !== null) {
    if (e.key === "Escape") {
      e.preventDefault();
      finishGamepadCapture("CANCELLED");
    }
    return;
  }
  if (rebindingUltiSlot !== null) {
    e.preventDefault();
    if (e.key === "Escape") { finishUltiKeyCapture("CANCELLED"); return; }
    const key = e.ctrlKey || e.altKey || e.metaKey ? null : normalizeUltiKey(e.key);
    if (!key) { setOptionsMessage("KEY RESERVED OR UNSUPPORTED"); return; }
    if (ultiKeys.some((current,index) => current === key && index !== rebindingUltiSlot)) {
      setOptionsMessage("KEY ALREADY USED");
      return;
    }
    ultiKeys[rebindingUltiSlot] = key;
    saveOptions();
    finishUltiKeyCapture("KEY UPDATED");
    updateUltiHud();
    return;
  }
  if (rebindingControlKey !== null) {
    e.preventDefault();
    if (e.key === "Escape") {
      if (rebindingControlKey === "pause") {
        pauseKey = "Escape"; saveOptions(); finishControlKeyCapture("KEY UPDATED");
      } else {
        finishControlKeyCapture("CANCELLED");
      }
      return;
    }
    const key = e.ctrlKey || e.altKey || e.metaKey ? null : normalizeControlKey(e.key);
    if (!key) { setOptionsMessage("KEY RESERVED OR UNSUPPORTED"); return; }
    const otherControls = { pause:pauseKey, boost:boostKey, left:leftKey, right:rightKey };
    delete otherControls[rebindingControlKey];
    if (Object.values(otherControls).some(k => k === key) || ultiKeys.some(k => k === key)) {
      setOptionsMessage("KEY ALREADY USED"); return;
    }
    if (rebindingControlKey === "pause")      pauseKey = key;
    else if (rebindingControlKey === "boost") boostKey = key;
    else if (rebindingControlKey === "left")  leftKey  = key;
    else if (rebindingControlKey === "right") rightKey = key;
    saveOptions(); finishControlKeyCapture("KEY UPDATED");
    return;
  }
  if (e.target instanceof Element && e.target.matches("input, textarea, select")) return;
  if (e.key === "Escape") {
    e.preventDefault();
    if (!closeOpenPanel() && pauseKey === "Escape") togglePause();
    return;
  }
  if (pauseKey !== "Escape" && keyMatches(e, pauseKey)) { e.preventDefault(); togglePause(); return; }
  if (keyMatches(e, boostKey)) { e.preventDefault(); tryBoost(); }
  if (e.key === "ArrowLeft"  || keyMatches(e, leftKey))  { e.preventDefault(); keys.left  = true; }
  if (e.key === "ArrowRight" || keyMatches(e, rightKey)) { e.preventDefault(); keys.right = true; }
  if (e.key === "ArrowUp" || e.key.toLowerCase() === "w") { e.preventDefault(); keys.up = true; }
  if (e.key === "ArrowDown" || e.key.toLowerCase() === "s") { e.preventDefault(); keys.down = true; }
  if (e.key.toLowerCase() === "r") startGame();
  const ultiSlot = ultiKeys.indexOf(normalizeUltiKey(e.key));
  if (ultiSlot >= 0 && !e.repeat) tryUlti(ultiSlot);
  if (e.key >= "1" && e.key <= "9") selectTier(Number(e.key)-1);
  if (e.key === "0") selectTier(9);
  if (e.key.toLowerCase() === "m") {
    const allMuted = !sfxMuted;
    sfxMuted = allMuted; musicMuted = allMuted;
    syncMusic(); saveOptions(); updateMixerUI();
  }
});
window.addEventListener("keyup", e => {
  if (e.key === "ArrowLeft"  || keyMatches(e, leftKey))  keys.left  = false;
  if (e.key === "ArrowRight" || keyMatches(e, rightKey)) keys.right = false;
  if (e.key === "ArrowUp" || e.key.toLowerCase() === "w") keys.up = false;
  if (e.key === "ArrowDown" || e.key.toLowerCase() === "s") keys.down = false;
});

window.addEventListener("gamepadconnected", e => {
  activeGamepadIndex = e.gamepad.index;
  updateGamepadStatusUI();
});
window.addEventListener("gamepaddisconnected", e => {
  if (activeGamepadIndex === e.gamepad.index) activeGamepadIndex = null;
  gamepadMove = { left:false, right:false, up:false, down:false };
  updateGamepadStatusUI();
});

document.querySelectorAll("[data-anomaly-trigger]").forEach(button => {
  button.addEventListener("click", () => triggerAnomaly(button.dataset.anomalyTrigger));
});
initAnomalyTestPanel();

canvas.addEventListener("touchstart", e => {
  e.preventDefault();
  cachedCanvasRect = canvas.getBoundingClientRect();
  touchActive = true;
  touchClientX = e.touches[0].clientX;
  touchClientY = e.touches[0].clientY;
  tryBoost();
}, { passive:false });
canvas.addEventListener("touchmove", e => {
  e.preventDefault();
  touchClientX = e.touches[0].clientX;
  touchClientY = e.touches[0].clientY;
}, { passive:false });
canvas.addEventListener("touchend", e => {
  e.preventDefault();
  touchActive = false;
  touchClientX = null;
  touchClientY = null;
}, { passive:false });

// ── UPDATE ───────────────────────────────────────────────────────────────────
// Order-preserving in-place removal of dead entries — filter() would allocate
// a fresh array for each effect list on every frame (GC pressure on mobile).
const _isAlive = e => e.life > 0;
function compactInPlace(arr, isAlive) {
  let w = 0;
  for (let i = 0; i < arr.length; i++) if (isAlive(arr[i])) arr[w++] = arr[i];
  arr.length = w;
}

function updateEffects(dt) {
  particles.forEach(p => {
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.vy += 210 * dt;
    p.life -= dt * 2;
  });
  compactInPlace(particles, _isAlive);

  shockwaves.forEach(sw => { sw.r += 160 * dt; sw.life -= dt * 2.8; });
  compactInPlace(shockwaves, _isAlive);
  resonanceLightning.forEach(l => { l.life -= dt * 10; });
  compactInPlace(resonanceLightning, _isAlive);

  if (rocketPhase === 'attack') {
    rocketPhaseTimer += dt;
    rocketMode = Math.min(1, rocketPhaseTimer / ROCKET_ATTACK_DUR) * rocketModePeak;
    if (rocketPhaseTimer >= ROCKET_ATTACK_DUR) { rocketPhase = 'sustain'; rocketPhaseTimer = 0; rocketMode = rocketModePeak; }
  } else if (rocketPhase === 'sustain') {
    rocketPhaseTimer += dt;
    rocketMode = rocketModePeak;
    if (rocketPhaseTimer >= rocketSustainDuration) { rocketPhase = 'decay'; rocketPhaseTimer = 0; }
  } else if (rocketPhase === 'decay') {
    rocketPhaseTimer += dt;
    rocketMode = Math.max(0, 1 - rocketPhaseTimer / rocketDecayDuration) * rocketModePeak;
    if (rocketPhaseTimer >= rocketDecayDuration) { rocketPhase = 'off'; rocketMode = 0; }
  } else {
    rocketMode = 0;
  }
  if (rocketModeDuration > 0) rocketModeDuration -= dt;

  if (buyChartFlash > 0) buyChartFlash = Math.max(0, buyChartFlash - dt * 0.6);
  if (buyChartGlowDuration > 0) buyChartGlowDuration = Math.max(0, buyChartGlowDuration - dt);
  if (sellChartFlash > 0) sellChartFlash = Math.max(0, sellChartFlash - dt * 0.6);

  if (shakeI > 0.1) {
    shakeX = (Math.random()-.5)*shakeI*7;
    shakeY = (Math.random()-.5)*shakeI*4;
    shakeI *= Math.pow(0.04, dt);
  } else {
    shakeX = 0; shakeY = 0; shakeI = 0;
  }

  if (pillarFlash > 0) pillarFlash = Math.max(0, pillarFlash - dt);
  if (resonanceFlash > 0) resonanceFlash = Math.max(0, resonanceFlash - dt / 0.6);
  if (screenFlash > 0) screenFlash = Math.max(0, screenFlash - dt / 0.5);
  if (phoenixBlinkTimer > 0) phoenixBlinkTimer = Math.max(0, phoenixBlinkTimer - dt);
  pillarRedRipples.forEach(r => { r.r += 70 * dt; r.life -= dt * 3; });
  compactInPlace(pillarRedRipples, _isAlive);

  // OVERCLOCK after-images: push current orb position when active
  if (state === 'playing' && orb.worldY) {
    const hasClock = ultiActiveStates.some(s => s?.tier === 3 && s?.phase === 'active');
    if (hasClock) {
      afterImages.push({ x: orb.x, sy: w2s(orb.worldY), r: orb.radius, alpha: 0.22 });
      if (afterImages.length > 6) afterImages.shift();
    } else if (afterImages.length > 0) {
      afterImages = [];
    }
  }
}

function updateOrbPhysics(dt, t, ultiMods) {
  // ── Horizontal movement ──
  const helixState = ultiActiveStates.find(s => s?.tier === 8 && s?.phase === 'active');
  if (activeAnomaly?.type === "liquidityVoid") {
    // Void uses free 360° thrust below; don't snap horizontal velocity to normal lanes.
  } else if (helixState) {
    const targetX = W / 2 + W * 0.28 * Math.sin(helixState.counters.helixOscTime * Math.PI);
    const diff = targetX - orb.x;
    orb.vx += (Math.sign(diff) * Math.min(Math.abs(diff) * 6, ORB_SPEED_EFF * 1.3) - orb.vx) * (1 - Math.exp(-dt * 10));
  } else {
    let targetVx = 0;
    if (touchActive && touchClientX !== null) {
      const rect = canvasRect();
      const localX = (touchClientX - rect.left) * (W / rect.width);
      const dx = localX - orb.x;
      // Touch deadzone shrinks on narrow screens: a fixed 35px stop band eats ~9% of a
      // phone's width vs ~2% on desktop, degrading dodge precision where corridors are tightest.
      // Floor of 20px stays above the post-entry stop distance ORB_SPEED_EFF/14 (~17.5px at
      // 390px with the narrow speed boost) — do not lower it or raise the boost without
      // re-checking that ratio.
      const touchDead = 35 - 15 * narrowScreenRamp();
      if (Math.abs(dx) > touchDead) targetVx = Math.sign(dx) * ORB_SPEED_EFF;
    } else if (gamepadMove.left || gamepadMove.right) {
      if (gamepadMove.left)  targetVx = -ORB_SPEED_EFF;
      if (gamepadMove.right) targetVx =  ORB_SPEED_EFF;
    } else {
      if (keys.left)  targetVx = -ORB_SPEED_EFF;
      if (keys.right) targetVx =  ORB_SPEED_EFF;
    }
    orb.vx += (targetVx - orb.vx) * (1 - Math.exp(-dt * 14));
  }
  orb.x  += orb.vx * dt;
  orb.x   = Math.max(orb.radius, Math.min(W - orb.radius, orb.x));

  // ── Physics: gravity + buy pressure ──
  if (performance.now() >= sellDebuffUntil) sellDebuffStrength = 0;
  const sellGravity = sellDebuffStrength * sessionGameplay.impacts.sellGravityPct/100;
  const heightBonus = Math.min(0.25, Math.max(0, (orb.height - H * 12) / (H * 24)));
  const gravMult = trendMultipliers().gravity + Math.max(0, -livePressure) * 0.12 + sellGravity + ultiMods.gravAdd + heightBonus + stormLevel() * (15 / GRAVITY);
  if (activeAnomaly?.type === "liquidityVoid") {
    const aim = liquidityVoidAimVector();
    const accel = 520;
    orb.vx += aim.x * accel * dt;
    orb.vy += aim.y * accel * dt;
    orb.vx *= Math.pow(0.64, dt);
    orb.vy *= Math.pow(0.64, dt);
  } else {
    orb.vy -= GRAVITY * gravMult * dt;
  }
  if (rocketMode > 0) orb.vy += rocketMode * 2600 * dt;
  if (livePressure > 0.15) orb.vy += livePressure * 55 * dt;
  const upwardCap = MAX_ORB_VY;
  const fallCap = activeAnomaly?.type === "liquidityVoid" ? -MAX_ORB_VY : -980;
  orb.vy = Math.max(fallCap, Math.min(upwardCap + rocketMode * 1400, orb.vy));
  orb.prevWorldY = orb.worldY;
  orb.worldY += orb.vy * dt;

  // VOID ceiling: a soft cushion, not a wall. Camera is locked during the void (see
  // updateScoreAndCamera), so the chamber is closed. In the top band the upward velocity
  // is eased toward an envelope that reaches 0 exactly at the ceiling, so the orb glides
  // to a stop instead of slamming/bouncing. Gated on the void ONLY — without this gate it
  // would brake the orb during normal climbing.
  if (activeAnomaly?.type === "liquidityVoid") {
    const ceil = cameraY + H - orb.radius * 2.4;
    const cushion = H * 0.10;
    // Thin top cushion: lightly bleed upward speed so the impact lands soft, not a slam.
    if (orb.vy > 0 && orb.worldY > ceil - cushion) {
      const k = (orb.worldY - (ceil - cushion)) / cushion; // 0..1, deeper = stronger brake
      orb.vy *= Math.pow(0.55, dt * 10 * k);
    }
    // Damped bounce at the ceiling (a little rebound, not a dead stop at the top).
    if (orb.worldY > ceil) {
      orb.worldY = ceil;
      if (orb.vy > 0) orb.vy *= -0.5;
      orb.vx *= 0.9;
    }
  }

  if (w2s(orb.worldY) > H) {
    if (activeAnomaly?.type === "liquidityVoid") {
      orb.worldY = cameraY + orb.radius * 2.4;
      orb.vy = Math.max(orb.vy * -0.22, 90);
      orb.vx *= 0.82;
      return;
    }
    const aegisSlot = ultiActiveStates.findIndex(s => s?.tier === 2 && s?.phase === 'active' && !s?.counters?.shieldUsed);
    if (aegisSlot >= 0) {
      orb.worldY = cameraY + orb.radius * 3;
      orb.vy = 650;
      ultiActiveStates[aegisSlot] = null;
      const rsy = w2s(orb.worldY);
      burst(orb.x, rsy, TIERS[2].orb.bloom, 35, { up: 140, spread: 120, size: 4 });
      for (let _i = 0; _i < 3; _i++) shockwaves.push({ x: orb.x, y: rsy, r: _i * 25, life: 1 - _i * 0.10 });
      if (tier().feats.shake) shakeI = Math.max(shakeI, 3);
      sfx.aegis_save();
      updateUltiHud();
      return;
    }
    gameOver(); return;
  }

  // ── Volumetric orb animation ──
  orb.squash += (1 - orb.squash) * (1 - Math.exp(-dt * 9));
  orb.sparkPhase += dt * (1.5 + activeTier * 0.25);
  orb.bandPhase += dt * 2.2;

  // ── Trail ──
  // Tier-flavour trail length only counts when no skin is equipped (skins own
  // the trail entirely); mobile shortening applies here, not at equip time.
  // trailMode setting: "off" keeps no buffer, "basic" caps it at the plain
  // comet length, "all" is the full skin/tier behaviour.
  const _tierTrailLen = equippedSkin ? 0
    : Math.ceil((TIER_TRAILS[activeTier]?.trail.len || 0) * (isMobile ? 0.45 : 1));
  const _effectiveTrailLen = trailMode === "off" ? 0
    : trailMode === "basic" ? t.trailLen
    : Math.max(t.trailLen, equippedSkin?.trail?.len || 0, _tierTrailLen);
  if (_effectiveTrailLen > 0) {
    orb.trail.push({ x:orb.x, worldY:orb.worldY });
    while (orb.trail.length > _effectiveTrailLen) orb.trail.shift();
  } else {
    orb.trail = [];
  }

  // ── Glow tracks buy pressure ──
  const speedFrac = Math.min(1, Math.abs(orb.vy) / 900);
  const targetGlow = 0.28 + Math.max(0, livePressure) * 0.72 + speedFrac * 0.3 + rocketMode * 0.6;
  orb.glow += (targetGlow - orb.glow) * dt * 5;

  // ── Energy regen ──
  const regen = ENERGY_REGEN * t.feats.regen * (1 + Math.max(0, livePressure) * .45) * ultiMods.regenMult;
  orb.energy = Math.min(100, orb.energy + regen * dt);
  updateEnergyBar();
}

function updatePlatformCollisions(dt, t, ultiMods) {
  platforms.forEach(p => p.age += dt);
  const _platCull = cameraY - H * 0.9;
  compactInPlace(platforms, p => p.worldY > _platCull);
  if (activeAnomaly?.type === "liquidityVoid") return;

  // Crépitement couloir : orbe montant rapide qui passe près d'une plateforme
  if (orb.vy > 200) {
    const now = performance.now();
    for (const plat of platforms) {
      const orbPrevBot = (orb.worldY - orb.vy * dt) - orb.radius;
      const orbCurrBot = orb.worldY - orb.radius;
      if (orbPrevBot <= plat.worldY && orbCurrBot > plat.worldY) {
        const margin = orb.radius * 2.5;
        if (orb.x >= plat.x - margin && orb.x <= plat.x + plat.width + margin) {
          if (!plat._corridorAt || now - plat._corridorAt > 300) {
            sfx.corridor();
            plat._corridorAt = now;
          }
        }
      }
    }
  }

  if (orb.vy > 20) return;
  for (const plat of platforms) {
    const platScreenY = w2s(plat.worldY);
    if (platScreenY < 0 || platScreenY > H) continue;

    const orbBot  = orb.worldY - orb.radius;
    const overlapX = orb.x >= plat.x - 4 && orb.x <= plat.x + plat.width + 4;
    const nearTop  = orbBot >= plat.worldY - PLAT_H && orbBot <= plat.worldY + Math.max(gs*4, Math.abs(orb.vy) * dt * 2);

    if (overlapX && nearTop) {
      // PILLAR: pass through sell platforms — emit ripple + sparks, no bounce
      if (ultiMods.pillarRedPassthrough && plat.type === 'sell') {
        pillarRedRipples.push({ x: orb.x, y: w2s(plat.worldY), r: 0, life: 1 });
        burst(orb.x, w2s(plat.worldY), '#ff8fa8', 8, { up: 30, spread: 60, size: 2 });
        break;
      }

      orb.worldY = plat.worldY + orb.radius;

      // Bounce force — ulti bounceMult comes from PILLAR/HELIX/PHOENIX via applyStateToMods
      const bMult = t.feats.bounce * trendMultipliers().ascent * ultiMods.bounceMult;
      orb.vy     = JUMP_FORCE * plat.bounce * bMult + Math.max(0, livePressure) * 24;
      orb.squash = 0.7;
      plat.hitAt = performance.now(); // impact flare for tier-visuals supports
      orb.bounces++;
      const comboAwarded = awardPlatformCombo(plat);
      if (comboAwarded) {
        orb.combo = Math.min(orb.combo + 1, 999);
        orb.maxCombo = Math.max(orb.maxCombo, orb.combo);
      }

      const platColor = getPlatformColor(plat, t);
      if (t.particles)  burst(orb.x, w2s(orb.worldY), platColor, 8);
      if (t.shockwave)  addShockwave(orb.x, w2s(plat.worldY));
      if (t.shake && orb.combo > 4) shakeI = Math.min(5, orb.combo * 0.35);

      sfx.bounce();
      if (plat.type==="buy") sfx.buyBoost();
      if (comboAwarded) {
        if (orb.combo >= 3) showCombo();
        if (orb.combo === 3 || (orb.combo > 3 && orb.combo % 8 === 0)) sfx.combo();
        updateMult();
      }

      if (receivePlatform(plat) || (ultiMods.forceNeutral && plat.type === 'neutral')) {
        platforms = platforms.filter(p => p !== plat);
      }

      // RESONANCE: increment charge on each bounce
      for (const rSt of ultiActiveStates) {
        if (rSt?.tier === 7 && rSt?.phase === 'active') {
          rSt.counters.bounceCharge = (rSt.counters.bounceCharge || 0) + 1;
          burst(orb.x, w2s(orb.worldY), TIERS[7].orb.bloom, isMobile ? 10 : 20, { up: 60, spread: 60, size: 3 });
          shockwaves.push({ x: orb.x, y: w2s(orb.worldY), r: 0, life: 0.5 });
          if (rSt.counters.bounceCharge >= 4) {
            rSt.counters.bounceCharge = 0;
            orb.vy = Math.min(MAX_ORB_VY, orb.vy + 600);
            sfx.resonance_blast();
            resonanceFlash = 1.0;
            screenFlash = Math.max(screenFlash, 0.9);
            const rsy = w2s(orb.worldY);
            const resBurst = isMobile ? 5 : 10;
            for (let ri = 0; ri < resBurst; ri++) shockwaves.push({ x: orb.x, y: rsy, r: 0, life: 1 - ri * (isMobile ? 0.14 : 0.07) });
            burst(orb.x, rsy, TIERS[7].orb.bloom, isMobile ? 45 : 90, { up: 170, spread: 190, size: 6 });
            shakeI = Math.max(shakeI, 9);
            const resBolts = isMobile ? 4 : 8;
            for (let li = 0; li < resBolts; li++) {
              const angle = (li / resBolts) * Math.PI * 2 + Math.random() * 0.4;
              const dist  = 80 + Math.random() * 140;
              resonanceLightning.push({ x1: orb.x, y1: rsy,
                x2: orb.x + Math.cos(angle) * dist,
                y2: rsy   + Math.sin(angle) * dist,
                life: 1 });
            }
          }
          break;
        }
      }

      break;
    }
  }
}

let _hudScoreLast = -1;
function updateScoreAndCamera(dt) {
  // During LIQUIDITY VOID the screen is frozen into a closed chamber: neither the
  // height score nor the camera rises, so flying upward farms nothing — the only
  // way out (and up) is reaching the portal, which launches the orb on exit.
  const inVoid = activeAnomaly?.type === "liquidityVoid";
  if (!inVoid && orb.worldY > orb.height) {
    orb.height = orb.worldY;
  }
  const s = currentScore();
  if (s > orb.maxScore) orb.maxScore = s;
  if (s !== _hudScoreLast) {
    _hudScoreLast = s;
    const el = document.getElementById("heightScore");
    if (el) el.textContent = String(s).padStart(6,"0");
  }
  updateComboGlow();

  // Camera follows the climb — but during the VOID it settles into a frozen chamber.
  // Don't cut the scroll dead (that reads as "snapped off"): over the first ~0.5s let the
  // upward follow fade to zero so the world decelerates smoothly, then the chamber locks.
  let follow = 1;
  if (inVoid) follow = Math.max(0, 1 - activeAnomaly.elapsed / 0.5);
  if (follow <= 0) return;
  const targetCamY = orb.worldY - H * 0.42;
  if (targetCamY > cameraY) {
    cameraY += (targetCamY - cameraY) * (1 - Math.exp(-dt * 6)) * follow;
  }
}

function updateBeaconAttraction(dt, ultiMods) {
  if (!ultiMods.beaconActive || !orb.worldY) return;
  const speed = 1.2;
  for (const plat of platforms) {
    const psy = w2s(plat.worldY);
    if (psy < -50 || psy > H + 50) continue;
    const dx = orb.x - (plat.x + plat.width / 2);
    if (plat.type === 'buy') {
      plat.x += dx * speed * dt;
    } else if (plat.type === 'sell') {
      plat.x -= dx * speed * 0.7 * dt;
    }
    plat.x = Math.max(8, Math.min(W - plat.width - 8, plat.x));
  }
}

function exportAgentState() {
  if (!AGENT_MODE || !orb?.worldY) return;
  const viewTop = cameraY - 50;
  const viewBot = cameraY + H + 50;
  const visiblePlatforms = platforms
    .filter(p => p.worldY >= viewTop && p.worldY <= viewBot)
    .slice(0, 24)
    .map(p => ({
      x: p.x + p.width / 2,
      worldY: p.worldY,
      width: p.width,
      type: p.type || "neutral",
      receptions: p.receptions ?? 0,
      bounceLimit: p.bounceLimit ?? 3,
    }));
  const visibleBoosters = boosters
    .filter(b => !b.used && b.worldY >= viewTop && b.worldY <= viewBot)
    .slice(0, 12)
    .map(b => ({ type: b.type, x: b.x, worldY: b.worldY, w: b.w }));
  let nearest = null;
  for (const plat of platforms) {
    if (plat.worldY > orb.worldY - 8) continue;
    if (!nearest || plat.worldY > nearest.worldY) nearest = plat;
  }
  let nearestPlatformBelow = null;
  if (nearest) {
    nearestPlatformBelow = {
      dx: ((nearest.x + nearest.width / 2) - orb.x) / Math.max(W, 1),
      dy: (orb.worldY - nearest.worldY) / Math.max(H, 1),
      width: nearest.width / Math.max(W, 1),
      wear: (nearest.receptions ?? 0) / Math.max(nearest.bounceLimit ?? 3, 1),
    };
  }
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
  window.__ASCENT_AGENT__ = {
    state,
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
    activeAnomaly: activeAnomaly ? { type: activeAnomaly.type, remaining: activeAnomaly.remaining ?? 0 } : null,
    ultiCharges: [...(ultiCharges || [])],
    tierIndex: activeTier,
    score: currentScore(),
    canBoost: (orb.energy ?? 0) + (orb.reserve ?? 0) >= 14,
    nearestPlatformBelow,
    nearestBooster,
    stormLevel: stormLevel(),
  };
}

function update(dt) {
  updateGamepadInput();
  if (state === "paused") return;
  // Ulti timers always advance in real time; timeScale only affects simulation.
  updateUltis(dt);
  updateMagnetParticles(dt);
  const ultiMods = getUltiModifiers();
  if (chronoRamp > 0) ultiMods.timeScale = Math.min(ultiMods.timeScale, 1 - 0.55 * chronoRamp);
  const simDt = dt * ultiMods.timeScale;
  updateEffects(simDt);
  if (state !== "playing") return;
  const t = tier();
  processImpactQueue(); // market events stay real-time
  updateOrbPhysics(simDt, t, ultiMods);
  if (state !== "playing") return; // game over triggered by physics
  updateBeaconAttraction(simDt, ultiMods);
  updateBoosters(simDt);
  updateAnomaly(simDt);
  if (state !== "playing") return;
  ensureNeutralCoverage();
  updatePlatformCollisions(simDt, t, ultiMods);
  updateScoreAndCamera(simDt);
  anomalyMinDelay = Math.max(0, anomalyMinDelay - dt); // real dt: CHRONO must not stretch the floor
  maybeTriggerAutoAnomaly();
  if (AGENT_MODE) exportAgentState();
}

// Tier is a prestige weight applied ONCE, uniformly, at display time — never baked into the bank.
// Within a tier the score is therefore pure skill (the per-tier leaderboards stay skill-ranked).
function tierWeight() {
  return tier().feats.scoreMult;
}

// Thin bindings of the pure formula (score-rules.js) to the live orb state. comboMult is bounded
// (1.0 → 5.0); styleLive is the unsecured reservoir contribution, PRE-tier, amplified by combo and
// the anomaly multiplier (both bounded) so style stays a topping on the climb.
function comboMult() {
  return comboMultiplier(orb.combo);
}

function styleLive() {
  return liveStyle({ bonus: orb.bonus, combo: orb.combo, anomalyMult: anomalyScoreMultiplier(activeAnomaly?.type) });
}

// The ONLY score the game knows: what the HUD shows every frame. No hidden bonus is added at crash.
function currentScore() {
  return liveScore({
    height: orb.height, bankStyle: orb.bankStyle, bonus: orb.bonus, combo: orb.combo,
    tierWeight: tierWeight(), anomalyMult: anomalyScoreMultiplier(activeAnomaly?.type),
  });
}

// Fold the live style into the banked total at exactly its displayed (pre-tier) value, then empty
// the reservoir, so the on-screen number stays continuous (no jump): bankStyle gains styleLive while
// styleLive drops to 0. Tier weight is constant within a run, so the displayed score is unchanged.
// Used at game over and as the base of breakCombo.
function secureStreak() {
  orb.bankStyle += styleLive();
  orb.bonus = 0;
}

// Red hit (player's fault): lock in the streak (no score loss), then wipe the momentum — the
// reservoir is emptied by secureStreak and the combo multiplier collapses, so the hand restarts
// small and climbs slowly. Breaking never lowers the number but stalls it: that is the deterrent.
function breakCombo() {
  secureStreak();
  orb.combo = 0;
  updateMult();
}

// ── DRAW CACHES ───────────────────────────────────────────────────────────────
let _bgGrad = null, _bgTierIdx = -1, _bgW = 0, _bgH = 0;
let _axisGrad = null, _axisTierIdx = -1, _axisTop = -1;
let _ambOff = null, _ambOffCtx = null, _ambOffKey = '', _ambSkip = 0;
let _darkPoolMask = null, _darkPoolMaskCtx = null;
let _darkPoolLastX = -999, _darkPoolLastY = -999, _darkPoolLastR = -999;
let _voidDebrisOff = null, _voidDebrisOffCtx = null, _voidDebrisSkip = 0;
let _portalGradCache = null, _portalGradCacheR = -1;
const _cacheableAmb = new Set(['caustics','spores','embers','petals','stardust','aurora']);
// Frame-scoped: written once at top of draw(), read by all sub-functions
let _frameNow = 0, _frameMods = null;
let _ultiT1 = false, _ultiT7 = false, _ultiT8 = false, _ultiT9 = false;
let _resonanceState = null, _phoenixState = null;

function _makeOffscreen(w, h) {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas"); c.width = w; c.height = h; return c;
}

// Pre-rendered radial glow sprites — soft falloff via drawImage instead of
// per-frame shadowBlur/createRadialGradient (both far more expensive).
const _glowSprites = new Map(); // "color@r" -> canvas
function getGlowSprite(color, radius = 16) {
  const r = Math.max(8, Math.ceil(radius / 8) * 8); // bucketed so resize doesn't thrash
  const key = color + "@" + r;
  let c = _glowSprites.get(key);
  if (!c) {
    if (_glowSprites.size >= 24) _glowSprites.delete(_glowSprites.keys().next().value);
    c = _makeOffscreen(r * 2, r * 2);
    const g = c.getContext("2d");
    const grad = g.createRadialGradient(r, r, 0, r, r, r);
    // Quadratic-ish falloff for a softer halo edge — free at runtime (baked).
    grad.addColorStop(0, hexA(color, 1));
    grad.addColorStop(0.25, hexA(color, 0.55));
    grad.addColorStop(0.55, hexA(color, 0.18));
    grad.addColorStop(1, hexA(color, 0));
    g.fillStyle = grad;
    g.fillRect(0, 0, r * 2, r * 2);
    _glowSprites.set(key, c);
  }
  return c;
}

// Platform buy-glow strips — halo baked once per (color, PLAT_H), drawn 3-slice.
// Replaces the per-platform animated shadowBlur (blur-radius pulse → alpha pulse).
const _platGlowCache = new Map(); // "color@h" -> {canvas, pad, capW, w, h}
function getPlatformGlowStrip(color) {
  const key = color + "@" + PLAT_H;
  let s = _platGlowCache.get(key);
  if (!s) {
    if (_platGlowCache.size >= 12) _platGlowCache.delete(_platGlowCache.keys().next().value);
    const pad = 16, bodyW = 48;
    const w = bodyW + pad * 2, h = PLAT_H + pad * 2;
    const c = _makeOffscreen(w, h);
    const g = c.getContext("2d");
    g.shadowColor = color;
    g.shadowBlur = 12;
    g.fillStyle = color;
    g.fillRect(pad, pad, bodyW, PLAT_H);
    g.fillRect(pad, pad, bodyW, PLAT_H); // second pass deepens the halo
    s = { canvas: c, pad, capW: pad + 8, w, h };
    _platGlowCache.set(key, s);
  }
  return s;
}

// Static orb sprites — halo/body/rim/spark baked once per (tier, R) at 2x
// resolution (smoother falloff than per-frame gradients at DPR 1.5).
const _orbSpriteCache = new Map();
function getOrbSprites(tierIdx, o, R) {
  const key = tierIdx + "@" + (R | 0);
  let s = _orbSpriteCache.get(key);
  if (s) return s;
  if (_orbSpriteCache.size >= 8) _orbSpriteCache.delete(_orbSpriteCache.keys().next().value);
  const sz = Math.ceil(R * 4) + 4, c = sz / 2, r = R * 2;
  const lx = c - r * .36, ly = c - r * .42;

  const body = _makeOffscreen(sz, sz);
  let g = body.getContext("2d");
  let grad = g.createRadialGradient(lx, ly, r * .04, c, c, r * 1.06);
  grad.addColorStop(0, o.hot); grad.addColorStop(.32, o.mid); grad.addColorStop(.78, o.mid); grad.addColorStop(1, o.rim);
  g.fillStyle = grad; g.beginPath(); g.arc(c, c, r, 0, 7); g.fill();

  const rim = _makeOffscreen(sz, sz); // baked at alpha 1, drawn with charge-scaled globalAlpha
  g = rim.getContext("2d");
  grad = g.createRadialGradient(c, c, r * .74, c, c, r);
  grad.addColorStop(0, hexA(o.bloom, 0)); grad.addColorStop(.86, hexA(o.bloom, 0)); grad.addColorStop(1, hexA(o.bloom, 1));
  g.fillStyle = grad; g.beginPath(); g.arc(c, c, r, 0, 7); g.fill();

  const hsz = 128; // halo: pure radial falloff, fixed bake size, scaled at draw
  const halo = _makeOffscreen(hsz, hsz);
  g = halo.getContext("2d");
  grad = g.createRadialGradient(hsz / 2, hsz / 2, 0, hsz / 2, hsz / 2, hsz / 2);
  // baked at max-charge alphas; drawOrb scales globalAlpha back down
  grad.addColorStop(0, hexA(o.bloom, .692));
  grad.addColorStop(.32, hexA(o.bloom, .27));
  grad.addColorStop(1, hexA(o.bloom, 0));
  g.fillStyle = grad; g.fillRect(0, 0, hsz, hsz);

  const ssz = 32;
  const spark = _makeOffscreen(ssz, ssz);
  g = spark.getContext("2d");
  grad = g.createRadialGradient(ssz / 2, ssz / 2, 0, ssz / 2, ssz / 2, ssz / 2);
  grad.addColorStop(0, hexA(o.hot, .9)); grad.addColorStop(1, hexA(o.bloom, 0));
  g.fillStyle = grad; g.fillRect(0, 0, ssz, ssz);

  // Corona — radial brume baked at peak (pulse=1) alpha and max radius; drawOrb
  // scales it to the live cR and modulates globalAlpha for the pulse. Inner hole
  // baked at R*.7 / cR_max so it scrolls with the sprite (imperceptible drift).
  const corona = _makeOffscreen(256, 256);
  g = corona.getContext("2d");
  grad = g.createRadialGradient(128, 128, 128 * (0.7 / 2.43), 128, 128, 128);
  grad.addColorStop(0, hexA(o.bloom, 0));
  grad.addColorStop(.7, hexA(o.bloom, .30));
  grad.addColorStop(.85, hexA(o.bloom, .12)); // softer outer falloff (free, baked)
  grad.addColorStop(1, hexA(o.bloom, 0));
  g.fillStyle = grad; g.fillRect(0, 0, 256, 256);

  // Flare — one branch: a 2px-thick horizontal gradient line (hot at the orb
  // center, fading to transparent at the tip). drawOrb draws it 6× rotated and
  // horizontally scaled per branch to recreate the per-branch length pulse.
  const flareW = 128, flareH = 4;
  const flare = _makeOffscreen(flareW, flareH);
  g = flare.getContext("2d");
  grad = g.createLinearGradient(0, 0, flareW, 0);
  grad.addColorStop(0, hexA(o.hot, .5));
  grad.addColorStop(.5, hexA(o.hot, .2)); // gentler mid taper (free, baked)
  grad.addColorStop(1, hexA(o.bloom, 0));
  g.fillStyle = grad; g.fillRect(0, 1, flareW, 2);

  // Band — vertical gradient strip (transparent → hot → transparent), drawn
  // stretched to the orb width at the live band Y under the orb clip.
  const band = _makeOffscreen(1, 64);
  g = band.getContext("2d");
  grad = g.createLinearGradient(0, 0, 0, 64);
  grad.addColorStop(0, hexA(o.hot, 0));
  grad.addColorStop(.5, hexA(o.hot, .55));
  grad.addColorStop(1, hexA(o.hot, 0));
  g.fillStyle = grad; g.fillRect(0, 0, 1, 64);

  s = { body, rim, halo, spark, corona, flare, flareH, band, size: sz };
  _orbSpriteCache.set(key, s);
  return s;
}

let _specSprite = null; // white specular highlight — tier-independent, baked once
function getSpecSprite() {
  if (!_specSprite) {
    const c = _makeOffscreen(64, 64), g = c.getContext("2d");
    const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, "rgba(255,255,255,.92)");
    grad.addColorStop(.45, "rgba(255,255,255,.2)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grad; g.fillRect(0, 0, 64, 64);
    _specSprite = c;
  }
  return _specSprite;
}

// ── ADAPTIVE QUALITY ─────────────────────────────────────────────────────────
// 2=HIGH 1=MED 0=LOW. In "auto" mode the loop low-passes the frame cost and
// steps the level with hysteresis (same pattern as the _skinAvgMs lofi fallback).
const _rmQuery = typeof matchMedia === "function" ? matchMedia("(prefers-reduced-motion: reduce)") : null;
let _reducedMotion = !!_rmQuery?.matches;
let qualityMode = storedOptions.qualityMode; // "auto" | "high" | "low"
let _qLevel = 2;
let _frameAvgMs = 8, _qHotFrames = 0, _qCoolMs = 0, _qCooldownUntil = 0, _qDprDirty = false;
const QF = {};
// Scene quality frozen for the whole run — mid-run auto-quality steps would
// make the painted backdrop visibly pop between layer counts.
let _sceneQ = 2;

function _qMaxLevel() {
  if (_reducedMotion) return 1;
  if (isMobile && ((navigator.deviceMemory && navigator.deviceMemory <= 4) || navigator.connection?.saveData)) return 1;
  return 2;
}

function applyQualityLevel(level) {
  const max = qualityMode === "auto" ? _qMaxLevel() : 2;
  const prevDpr = QF.dprCap;
  _qLevel = Math.max(0, Math.min(max, level));
  QF.dprCap      = [1, 1.25, isMobile ? 1.5 : 2][_qLevel];
  QF.particleCap = [40, 80, MAX_PARTICLES][_qLevel];
  QF.ambRefresh  = [4, 3, 2][_qLevel];
  QF.trailStride = [2, 1, 1][_qLevel];
  QF.fx          = _qLevel > 0;
  QF.chromatic   = _qLevel === 2;
  document.body?.classList.toggle("quality-low", _qLevel === 0);
  // DPR change rescales world positions — only safe between runs, never mid-run
  if (prevDpr !== undefined && prevDpr !== QF.dprCap) _qDprDirty = true;
}
applyQualityLevel(qualityMode === "low" ? 0 : 2);

function setQualityMode(mode) {
  qualityMode = mode;
  if (mode === "high") applyQualityLevel(2);
  else if (mode === "low") applyQualityLevel(0);
  _sceneQ = _qLevel; // manual quality change applies to the scene immediately
}

_rmQuery?.addEventListener?.("change", e => {
  _reducedMotion = e.matches;
  if (qualityMode === "auto") applyQualityLevel(Math.min(_qLevel, _qMaxLevel()));
});

// ── DRAW ─────────────────────────────────────────────────────────────────────
function draw() {
  // Safety: if canvas isn't sized yet, size it now
  if (W < 10 || H < 10 || canvas.width < 10) resize();
  if (W < 10 || H < 10) return;

  const t = tier();
  _frameNow = performance.now();

  // Painted scene — drawn before the shake translate on purpose: the far field
  // staying still while the gameplay shakes reinforces depth. Scene quality is
  // frozen per run (_sceneQ, anti-flapping); quality 0 falls back to the flat
  // tier gradient.
  const _sq = _qLevel === 0 ? 0 : _sceneQ;
  const _sceneDrawn = _sq >= 1 &&
    drawTierScene(ctx, activeTier, t, W, H, cameraY, _frameNow / 1000, _sq, isMobile, _reducedMotion);
  if (!_sceneDrawn) {
    if (!_bgGrad || activeTier !== _bgTierIdx || W !== _bgW || H !== _bgH) {
      _bgGrad = ctx.createLinearGradient(0, 0, 0, H);
      _bgGrad.addColorStop(0, t.bg1 || t.bg0 || "#000000");
      _bgGrad.addColorStop(1, t.bg0 || "#000000");
      _bgTierIdx = activeTier; _bgW = W; _bgH = H;
    }
    ctx.fillStyle = _bgGrad;
    ctx.fillRect(0, 0, W, H);
  }

  ctx.save();
  if (shakeI > 0.1 && !_reducedMotion) ctx.translate(shakeX, shakeY);

  const _ambFn = BIOME_AMBIENT[t.ambient];
  const _now = _frameNow / 1000;
  if (_ambFn && _cacheableAmb.has(t.ambient)) {
    if (_ambOffKey !== t.ambient || !_ambOff || _ambOff.width !== W || _ambOff.height !== H) {
      _ambOff = document.createElement('canvas');
      _ambOff.width = W; _ambOff.height = H;
      _ambOffCtx = _ambOff.getContext('2d');
      _ambOffKey = t.ambient; _ambSkip = 0;
    }
    if (_ambSkip <= 0) {
      _ambOffCtx.clearRect(0, 0, W, H);
      _ambFn(_ambOffCtx, W, H, t.orb, _now);
      _ambSkip = QF.ambRefresh - 1;
    }
    ctx.drawImage(_ambOff, 0, 0);
    _ambSkip--;
  } else if (_ambFn) {
    _ambFn(ctx, W, H, t.orb, _now);
  } else {
    if (t.feats.aurora) drawAurora();
    if (t.stars)        drawStars();
    if (t.nebula)       drawNebula();
  }
                       drawMarketChart();
  if (t.feats.speedlines) drawSpeedLines();

  // CHRONO dark tint (before platforms so scene feels slower/dimmer)
  const dMods = _frameMods = getUltiModifiers();
  // Pre-compute ulti flags once per frame — avoids 5+ array scans in sub-functions
  _ultiT1 = state === 'playing' && ultiActiveStates.some(s => s?.tier === 1 && s?.phase === 'active');
  _ultiT7 = state === 'playing' && ultiActiveStates.some(s => s?.tier === 7 && s?.phase === 'active');
  _ultiT8 = state === 'playing' && ultiActiveStates.some(s => s?.tier === 8 && s?.phase === 'active');
  _ultiT9 = state === 'playing' && ultiActiveStates.some(s => s?.tier === 9 && s?.phase === 'active');
  _resonanceState = null; _phoenixState = null;
  for (const s of ultiActiveStates) {
    if (s?.phase === 'active') {
      if (s.tier === 7 && !_resonanceState) _resonanceState = s;
      if (s.tier === 9 && !_phoenixState) _phoenixState = s;
    }
  }
  if (dMods.timeScale < 1 && state === 'playing') drawChronoDarkTint();

  // RESONANCE charge explosion flash + electric arcs
  drawResonanceFlash();
  drawResonanceLightning();

                       drawPlatforms();
                       drawBoosters();

  // PILLAR beam (behind shockwaves) + sell-platform pass-through ripples
  drawPillarBeam();
  drawPillarRedRipples();

                       drawShockwaves();
                       drawRocketFlame();

  // INVOKE arcs (follow platforms, behind orb)
  if (_ultiT1) drawBeaconArcs();

  // OVERCLOCK after-images (behind trail)
  if (afterImages.length > 0) drawOverclockAfterImages();

  if (trailMode === "off") { /* no trail at all */ }
  else if (trailMode === "basic") drawTrail();
  else if (equippedSkin?.render && typeof drawSkinTrail === "function") drawEquippedSkinTrail(equippedSkin);
  else { drawTrail(); drawTierTrail(); }
                       drawOrb();

  // Post-orb overlays: AEGIS hex, MAGNET ring, HELIX arcs, PHOENIX wings, RESONANCE ring
  if (dMods.shieldActive && state === 'playing') drawAegisHex();
  if (dMods.magnetLarge  && state === 'playing') drawMagnetRing();
  drawMagnetParticles();
  if (_ultiT8) drawHelixArcs();
  if (_ultiT9) drawPhoenixWings();
  if (_ultiT9) drawPhoenixGhostOrb();
  if (_ultiT7) drawResonanceRing();

  if (particles.length) {
    ctx.globalCompositeOperation = "lighter";
    for (const p of particles) {
      const life = Math.max(0, p.life);
      const r = p.size * 1.25 * (0.55 + 0.45 * life); // soft dot, shrinks as it dies
      ctx.globalAlpha = life;
      ctx.drawImage(getGlowSprite(p.color), p.x - r, p.y - r, r * 2, r * 2);
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
  }

  if (t.arcs || dMods.forceArcs) drawElectricArcs();

  // First-play hint: fades after 4 seconds
  drawFirstPlayHint();

  ctx.restore();

  if (QF.chromatic && ((t.chromatic && state==="playing" && orb.combo > 5) || (dMods.forceChromatic && state==="playing"))) drawChromatic();
  drawActiveAnomalyOverlay();
  drawScreenFlash();
}

function drawActiveAnomalyOverlay() {
  updateAnomalyHudBanner();
  if (activeAnomaly && state === "playing") {
    const { def, type, remaining } = activeAnomaly;
    if (type === "liquidityVoid") drawLiquidityVoidOverlay(def);
    if (type === "shortSqueeze") drawShortSqueezeHazards(def);
    if (type === "darkPool") drawDarkPoolOverlay(def);
    drawAnomalyFrame(def);
  }
}

let _bannerSecs = -1, _bannerLabel = "", _bannerColor = "";
function updateAnomalyHudBanner() {
  const el = document.getElementById("anomalyHudBanner");
  if (!el) return;
  if (!activeAnomaly || state !== "playing") {
    el.classList.add("hidden");
    _bannerSecs = -1; _bannerLabel = ""; // force rebuild when it reappears
    return;
  }
  const { def, remaining } = activeAnomaly;
  const secs = Math.max(0, Math.ceil(remaining));
  // Rebuild the label string only when the displayed second (or anomaly) changes.
  if (secs !== _bannerSecs || def.label !== _bannerLabel) {
    _bannerSecs = secs; _bannerLabel = def.label;
    el.textContent = `ANOMALY - ${def.label}  ${secs.toString().padStart(2, "0")}`;
  }
  if (def.color !== _bannerColor) {
    _bannerColor = def.color;
    el.style.setProperty("--anomaly-color", def.color);
  }
  el.classList.remove("hidden");
}

function drawAnomalyFrame(def) {
  const pulse = 0.55 + Math.sin(_frameNow * 0.008) * 0.18;
  const m = Math.max(7, gs * 7);
  const corner = Math.max(34, gs * 34);
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.fillStyle = hexA(def.color, 0.05 + pulse * 0.04);
  ctx.fillRect(0, 0, W, m);
  ctx.fillRect(0, H - m, W, m);
  ctx.fillRect(0, 0, m, H);
  ctx.fillRect(W - m, 0, m, H);

  ctx.strokeStyle = hexA(def.color, 0.55 + pulse * 0.25);
  ctx.lineWidth = Math.max(2, gs * 2);
  ctx.beginPath();
  ctx.moveTo(m, corner); ctx.lineTo(m, m); ctx.lineTo(corner, m);
  ctx.moveTo(W - corner, m); ctx.lineTo(W - m, m); ctx.lineTo(W - m, corner);
  ctx.moveTo(W - m, H - corner); ctx.lineTo(W - m, H - m); ctx.lineTo(W - corner, H - m);
  ctx.moveTo(corner, H - m); ctx.lineTo(m, H - m); ctx.lineTo(m, H - corner);
  ctx.stroke();

  if (!isMobile) {
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = Math.max(1, gs);
    ctx.beginPath();
    for (let x = m + corner; x < W - corner; x += Math.max(32, gs * 32)) {
      ctx.moveTo(x, m);
      ctx.lineTo(x + gs * 9, m + gs * 9);
      ctx.moveTo(x, H - m);
      ctx.lineTo(x + gs * 9, H - m - gs * 9);
    }
    for (let y = m + corner; y < H - corner - BOTTOM_BAR_H; y += Math.max(32, gs * 32)) {
      ctx.moveTo(m, y);
      ctx.lineTo(m + gs * 9, y + gs * 9);
      ctx.moveTo(W - m, y);
      ctx.lineTo(W - m - gs * 9, y + gs * 9);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawLiquidityVoidOverlay(def) {
  if (!_voidDebrisOff || _voidDebrisOff.width !== W || _voidDebrisOff.height !== H) {
    _voidDebrisOff = document.createElement('canvas');
    _voidDebrisOff.width = W;
    _voidDebrisOff.height = H;
    _voidDebrisOffCtx = _voidDebrisOff.getContext('2d');
    _voidDebrisSkip = 0;
  }
  if (_voidDebrisSkip <= 0) {
    const dctx = _voidDebrisOffCtx;
    dctx.clearRect(0, 0, W, H);
    const debrisCount = isMobile ? 10 : 28;
    for (let i = 0; i < debrisCount; i++) {
      const seed = i * 67.7;
      const x = ((seed * 0.61 + activeAnomaly.elapsed * 18) % W + W) % W;
      const y = ((seed * 0.43 - activeAnomaly.elapsed * 10) % H + H) % H;
      dctx.save();
      dctx.translate(x, y);
      dctx.rotate(activeAnomaly.elapsed * 0.45 + i);
      dctx.globalAlpha = 0.12 + (i % 5) * 0.035;
      dctx.fillStyle = i % 3 ? def.color : "#ffffff";
      dctx.fillRect(-gs * 4, -gs * 10, gs * 8, gs * 20);
      dctx.strokeStyle = hexA(def.color, 0.28);
      dctx.beginPath();
      dctx.moveTo(0, -gs * 16);
      dctx.lineTo(0, gs * 16);
      dctx.stroke();
      dctx.restore();
    }
    // Draw hazards into offscreen canvas to avoid per-frame shadowBlur on main canvas
    dctx.save();
    dctx.globalCompositeOperation = "lighter";
    for (const hazard of anomalyHazards) {
      if (hazard.kind !== "voidShard") continue;
      const sy = w2s(hazard.worldY);
      if (sy < -80 || sy > H + 80) continue;
      const pulse = 0.65 + Math.sin(_frameNow * 0.006 + hazard.phase) * 0.22;
      const r = hazard.r;
      dctx.save();
      dctx.translate(hazard.x, sy);
      dctx.rotate(activeAnomaly.elapsed * hazard.spin + hazard.phase);
      dctx.globalAlpha = 0.68 + pulse * 0.22;
      dctx.fillStyle = hexA("#ff4d6d", 0.78);
      dctx.strokeStyle = hexA("#ffd7e1", 0.38);
      dctx.lineWidth = Math.max(1, gs * 1.1);
      dctx.beginPath();
      dctx.moveTo(-r * 0.75, -r * 0.28);
      dctx.lineTo(r * 0.1, -r * 1.05);
      dctx.lineTo(r * 0.8, -r * 0.1);
      dctx.lineTo(r * 0.2, r * 1.0);
      dctx.lineTo(-r * 0.88, r * 0.48);
      dctx.closePath();
      dctx.fill();
      dctx.stroke();
      dctx.strokeStyle = hexA(def.color, 0.25);
      dctx.beginPath();
      dctx.moveTo(-r * 1.2, 0);
      dctx.lineTo(r * 1.15, 0);
      dctx.stroke();
      dctx.restore();
    }
    dctx.restore();
    _voidDebrisSkip = 2;
  } else {
    _voidDebrisSkip--;
  }
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.drawImage(_voidDebrisOff, 0, 0);
  ctx.globalAlpha = 0.1;
  ctx.fillStyle = def.color;
  ctx.fillRect(0, 0, W, H);
  ctx.restore();
  drawLiquidityVoidPortal(def);
}


function drawLiquidityVoidPortal(def) {
  const portal = activeAnomaly?.portal;
  if (!portal) return;
  const sy = w2s(portal.worldY);
  if (sy < -120 || sy > H + 120) return;
  const pulse = 0.55 + Math.sin(_frameNow * 0.007 + portal.phase) * 0.18;
  const r = portal.r * (1 + pulse * 0.08);
  ctx.save();
  ctx.translate(portal.x, sy);
  ctx.globalCompositeOperation = "lighter";
  // Cache radial gradient — portal.r is fixed for the lifetime of the anomaly
  if (!_portalGradCache || Math.abs(r - _portalGradCacheR) > portal.r * 0.05) {
    _portalGradCache = ctx.createRadialGradient(0, 0, r * 0.15, 0, 0, r * 1.6);
    _portalGradCache.addColorStop(0, hexA("#ffffff", 0.18));
    _portalGradCache.addColorStop(0.45, hexA(def.color, 0.22));
    _portalGradCache.addColorStop(1, hexA(def.color, 0));
    _portalGradCacheR = r;
  }
  ctx.fillStyle = _portalGradCache;
  ctx.beginPath();
  ctx.arc(0, 0, r * 1.65, 0, Math.PI * 2);
  ctx.fill();
  ctx.lineWidth = Math.max(2, gs * 2.2);
  ctx.strokeStyle = hexA(def.color, 0.9);
  ctx.beginPath();
  ctx.ellipse(0, 0, r * 0.74, r * 1.14, activeAnomaly.elapsed * 0.35, 0, Math.PI * 2);
  ctx.stroke();
  ctx.lineWidth = Math.max(1, gs * 1.2);
  ctx.strokeStyle = hexA("#ffffff", 0.55);
  ctx.beginPath();
  ctx.ellipse(0, 0, r * 0.44, r * 0.78, -activeAnomaly.elapsed * 0.45, 0, Math.PI * 2);
  ctx.stroke();
  ctx.font = `${Math.round(gs * 8)}px Silkscreen, monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = hexA("#ffffff", 0.62);
  ctx.fillText("EXIT", 0, r + gs * 17);
  ctx.restore();
}

function drawShortSqueezeHazards(def) {
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = hexA(def.color, 0.42);
  ctx.lineWidth = Math.max(1.5, gs * 1.6);
  const streakCount = isMobile ? 6 : 18;
  ctx.beginPath();
  for (let i = 0; i < streakCount; i++) {
    const x = (i * 59.7 % 1) * W;
    const y = ((i * 31.3 - activeAnomaly.elapsed * 320) % H + H) % H;
    ctx.moveTo(x, y + gs * 34);
    ctx.lineTo(x, y);
  }
  ctx.stroke();
  for (const hazard of anomalyHazards) {
    const sy = w2s(hazard.worldY);
    if (sy < -80 || sy > H + 80) continue;
    const r = hazard.r;
    ctx.save();
    ctx.translate(hazard.x, sy);
    ctx.rotate(Math.PI + Math.sin(activeAnomaly.elapsed * 4 + hazard.phase) * 0.08);
    ctx.fillStyle = hexA("#ff4d6d", 0.85);
    ctx.beginPath();
    ctx.moveTo(0, -r);
    ctx.lineTo(-r * 0.62, r);
    ctx.lineTo(r * 0.62, r);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
  ctx.restore();
}

function drawDarkPoolOverlay(def) {
  const x = orb.x;
  const y = w2s(orb.worldY);
  // baseRadius drives the mask (stable, no pulse) — pulse is visual only
  const baseRadius = Math.max(gs * 122, Math.min(W, H) * (isMobile ? 0.26 : 0.22));
  const radius = baseRadius + Math.sin(activeAnomaly.elapsed * 3) * gs * 5;
  if (!_darkPoolMask || _darkPoolMask.width !== W || _darkPoolMask.height !== H) {
    _darkPoolMask = document.createElement("canvas");
    _darkPoolMask.width = W;
    _darkPoolMask.height = H;
    _darkPoolMaskCtx = _darkPoolMask.getContext("2d");
  }
  const mctx = _darkPoolMaskCtx;
  if (Math.abs(x - _darkPoolLastX) > 2 || Math.abs(y - _darkPoolLastY) > 2 || Math.abs(baseRadius - _darkPoolLastR) > 2) {
    mctx.clearRect(0, 0, W, H);
    mctx.fillStyle = "rgba(0,0,8,0.94)";
    mctx.fillRect(0, 0, W, H);
    mctx.globalCompositeOperation = "destination-out";
    const cut = mctx.createRadialGradient(x, y, baseRadius * 0.72, x, y, baseRadius);
    cut.addColorStop(0, "rgba(0,0,0,1)");
    cut.addColorStop(1, "rgba(0,0,0,0)");
    mctx.fillStyle = cut;
    mctx.beginPath();
    mctx.arc(x, y, baseRadius, 0, Math.PI * 2);
    mctx.fill();
    mctx.globalCompositeOperation = "source-over";
    _darkPoolLastX = x; _darkPoolLastY = y; _darkPoolLastR = baseRadius;
  }

  ctx.save();
  ctx.drawImage(_darkPoolMask, 0, 0);
  ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = hexA(def.color, isMobile ? 0.7 : 0.85);
  ctx.lineWidth = Math.max(2, gs * 2.5);
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.stroke();
  const glow = ctx.createRadialGradient(x, y, 0, x, y, radius * 1.15);
  glow.addColorStop(0, hexA(def.color, 0.12));
  glow.addColorStop(0.72, hexA(def.color, 0.04));
  glow.addColorStop(1, hexA(def.color, 0));
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(x, y, radius * 1.15, 0, Math.PI * 2);
  ctx.fill();
  drawOrb();
  ctx.restore();
}

// Baked aurora — one textured W×180 band per (tier, W), drawn 3x at the
// animated y positions. Kills 3 per-frame gradient creations and affords
// subtle curtain ripples the live version couldn't.
let _auroraOff = null, _auroraKey = '';
function getAuroraStrip() {
  const key = activeTier + '@' + W;
  if (_auroraOff && _auroraKey === key) return _auroraOff;
  const bloom = tier().orb.bloom, h = 180;
  const c = _makeOffscreen(Math.max(2, W), h);
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, hexA(bloom, 0));
  grad.addColorStop(.5, hexA(bloom, .05));
  grad.addColorStop(1, hexA(bloom, 0));
  g.fillStyle = grad;
  g.fillRect(0, 0, W, h);
  g.globalCompositeOperation = 'lighter';
  for (let x = 0; x < W; x += 7) { // vertical curtain ripples
    const rip = .016 + .014 * Math.sin(x * .05) * Math.sin(x * .013 + 2);
    if (rip <= 0) continue;
    g.fillStyle = hexA(bloom, rip);
    const yc = h / 2 + Math.sin(x * .02) * 18;
    g.fillRect(x, yc - 38, 7, 76);
  }
  _auroraOff = c; _auroraKey = key;
  return _auroraOff;
}

function drawAurora() {
  const strip = getAuroraStrip();
  const now = _frameNow * 0.0002;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (let i=0; i<3; i++) {
    const y = H * (0.2 + i * 0.18) + Math.sin(now + i) * 40;
    ctx.drawImage(strip, 0, y - 90);
  }
  ctx.restore();
}

// Baked star field — 3 parallax bands of ~100 stars each with cross glints and
// colour temperature variance; per frame this costs 6 wrapped blits instead of
// one arc+fill per star.
let _starLayers = null, _starKey = '';
const STAR_BAND_DEPTHS = [0.15, 0.45, 0.8];
function getStarLayers() {
  const key = W + 'x' + H;
  if (_starLayers && _starKey === key) return _starLayers;
  const tints = ['#ffffff', '#bcd8ff', '#ffe7c4'];
  _starLayers = STAR_BAND_DEPTHS.map((depth, b) => {
    const c = _makeOffscreen(Math.max(2, W), Math.max(2, H));
    const g = c.getContext('2d');
    const count = isMobile ? 60 : 100;
    for (let i = 0; i < count; i++) {
      const x = Math.random() * W, y = Math.random() * H;
      const r = Math.random() * 1.3 + 0.4 + b * 0.25; // nearer bands slightly bigger
      const a = Math.random() * 0.55 + 0.2;
      const tint = tints[(Math.random() * tints.length) | 0];
      g.globalAlpha = a;
      g.fillStyle = tint;
      g.beginPath(); g.arc(x, y, r, 0, 7); g.fill();
      if (r > 1.2 && Math.random() < 0.3) { // cross glint on bright stars
        g.globalAlpha = a * 0.45;
        g.strokeStyle = tint; g.lineWidth = 0.6;
        g.beginPath();
        g.moveTo(x - r * 3, y); g.lineTo(x + r * 3, y);
        g.moveTo(x, y - r * 3); g.lineTo(x, y + r * 3);
        g.stroke();
      }
    }
    return c;
  });
  _starKey = key;
  return _starLayers;
}

function drawStars() {
  const layers = getStarLayers();
  for (let b = 0; b < layers.length; b++) {
    // Parallax: low-depth bands barely move; high-depth move with camera
    const off = ((-cameraY * STAR_BAND_DEPTHS[b] * 0.0012) % H + H) % H;
    ctx.drawImage(layers[b], 0, off - H);
    ctx.drawImage(layers[b], 0, off);
  }
}

// Baked nebula — multi-hue blobs + bright knots rendered once per (tier, W, H);
// only the current tier's layer is kept in memory. Per frame: one blit.
let _nebulaOff = null, _nebulaKey = '';
function getNebulaLayer() {
  const key = activeTier + '@' + W + 'x' + H;
  if (_nebulaOff && _nebulaKey === key) return _nebulaOff;
  const o = tier().orb;
  const c = _makeOffscreen(Math.max(2, W), Math.max(2, H));
  const g = c.getContext('2d');
  g.globalCompositeOperation = 'lighter';
  const blobs = [
    { x: .5,  y: .4,  r: .7,  col: o.bloom, a: 1 },   // original core glow
    { x: .22, y: .25, r: .38, col: o.mid,   a: .7 },
    { x: .78, y: .55, r: .45, col: o.bloom, a: .55 },
    { x: .35, y: .72, r: .33, col: o.hot,   a: .35 },
    { x: .68, y: .18, r: .3,  col: o.mid,   a: .5 },
  ];
  const base = Math.max(W, H);
  for (const b of blobs) {
    const bx = b.x * W, by = b.y * H, br = b.r * base;
    const grad = g.createRadialGradient(bx, by, 0, bx, by, br);
    grad.addColorStop(0, hexA(b.col, b.a));
    grad.addColorStop(1, hexA(b.col, 0));
    g.fillStyle = grad;
    g.fillRect(bx - br, by - br, br * 2, br * 2);
  }
  // sparse bright knots (deterministic spread — stable across rebakes)
  for (let i = 0; i < 14; i++) {
    const x = ((i * 73) % 97) / 97 * W, y = ((i * 131) % 89) / 89 * H;
    const r = (1 + (i % 3)) * 4;
    const grad = g.createRadialGradient(x, y, 0, x, y, r);
    grad.addColorStop(0, hexA(o.hot, .5));
    grad.addColorStop(1, hexA(o.hot, 0));
    g.fillStyle = grad;
    g.fillRect(x - r, y - r, r * 2, r * 2);
  }
  _nebulaOff = c; _nebulaKey = key;
  return _nebulaOff;
}

function drawNebula() {
  const layer = getNebulaLayer();
  const a = .05 + Math.sin(_frameNow * 0.0004) * .02;
  const drift = Math.sin(_frameNow * 0.00005) * 12; // very slow horizontal drift
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.globalAlpha = a;
  ctx.drawImage(layer, drift, 0);
  ctx.restore();
}

function marketCandles() {
  const byTime = new Map();
  preloadCandles.forEach(c=>byTime.set(c.time,c));
  liveCandles.forEach(c=>byTime.set(c.time,c));
  return [...byTime.values()].sort((a,b)=>a.time-b.time).slice(-MAX_CANDLE_COUNT);
}

function validChartPrice(value) {
  const price = Number(value);
  return Number.isFinite(price) && price >= 0 ? price : null;
}

function chartPriceRange(candles) {
  const values = [];
  candles.forEach(c => {
    [c.open, c.high, c.low, c.close].forEach(value => {
      const price = validChartPrice(value);
      if (price !== null) values.push(price);
    });
  });
  const livePrice = validChartPrice(latestLivePrice);
  if (livePrice !== null) values.push(livePrice);
  if (!values.length) return null;

  const low = Math.min(...values);
  const high = Math.max(...values);
  const reference = Math.max(high, low, 1e-10);
  const span = Math.max(0, high - low);
  const padding = Math.max(span * .14, reference * .001, 1e-10);
  const min = Math.max(0, low - padding);
  let max = high + padding;
  if (max <= min) max = min + Math.max(reference * .002, 1e-10);
  return { min, max };
}

function formatChartPrice(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "---";
  const abs = Math.abs(number);
  const decimals = abs >= 100 ? 2 : abs >= 1 ? 3 : abs >= .01 ? 4 : abs >= .0001 ? 5 : abs >= .000001 ? 6 : 8;
  const formatted = number.toFixed(decimals).replace(/(\.\d*?[1-9])0+$/,"$1").replace(/\.0+$/,"");
  return formatted;
}

function formatAxisTime(time) {
  if (activeInterval === "1d" || activeInterval === "4h") {
    return new Intl.DateTimeFormat("en-GB",{day:"2-digit",month:"2-digit",timeZone:"UTC"}).format(new Date(time));
  }
  return new Intl.DateTimeFormat("en-GB",{
    hour:"2-digit",minute:"2-digit",hour12:false,timeZone:"UTC"
  }).format(new Date(time));
}

// ── BIOME AMBIENT EFFECTS ────────────────────────────────────────────────────
const BIOME_AMBIENT = {
  void(ctx, w, h, o, now) {
    for (let i = 0; i < 26; i++) {
      const s = i * 137.5;
      ctx.globalAlpha = 0.2 + 0.5 * ((i * 7) % 5) / 5;
      ctx.fillStyle = "#fff";
      ctx.fillRect((s * 0.61 % 1) * w, ((s * 0.31 + now * 4) % h), 1, 1);
    }
    ctx.globalAlpha = 1;
  },
  aurora(ctx, w, h, o, now) {
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < 3; i++) {
      const y = h * (0.16 + i * 0.16) + Math.sin(now * 0.4 + i) * 22;
      const g = ctx.createLinearGradient(0, y - 50, 0, y + 50);
      g.addColorStop(0, hexA(o.bloom, 0)); g.addColorStop(0.5, hexA(o.bloom, 0.10)); g.addColorStop(1, hexA(o.bloom, 0));
      ctx.fillStyle = g; ctx.fillRect(0, y - 50, w, 100);
    }
    const starN = isMobile ? 15 : 30;
    for (let i = 0; i < starN; i++) {
      const s = i * 97.3;
      ctx.globalAlpha = 0.3 + 0.4 * ((i * 3) % 4) / 4;
      ctx.fillStyle = "#fff";
      ctx.fillRect((s * 0.53 % 1) * w, (s * 0.27 % 1) * h, 1, 1);
    }
    ctx.restore();
  },
  caustics(ctx, w, h, o, now) {
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < 5; i++) {
      const x = (i / 5) * w + Math.sin(now * 0.5 + i) * 30;
      const g = ctx.createLinearGradient(x - 30, 0, x + 30, 0);
      g.addColorStop(0, hexA(o.bloom, 0)); g.addColorStop(0.5, hexA(o.bloom, 0.06)); g.addColorStop(1, hexA(o.bloom, 0));
      ctx.fillStyle = g; ctx.fillRect(x - 30, 0, 60, h);
    }
    const dotsN = isMobile ? 9 : 18;
    for (let i = 0; i < dotsN; i++) {
      const s = i * 53.7;
      const py = ((s * 0.3 - now * 8) % h + h) % h;
      const px = (s * 0.61 % 1) * w + Math.sin(now + i) * 6;
      ctx.globalAlpha = 0.4; ctx.fillStyle = o.bloom;
      ctx.beginPath(); ctx.arc(px, py, 1.3, 0, 7); ctx.fill();
    }
    ctx.restore();
  },
  spores(ctx, w, h, o, now) {
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const sporeN = isMobile ? 11 : (_qLevel === 2 ? 32 : 22);
    for (let i = 0; i < sporeN; i++) {
      const s = i * 61.3;
      const py = ((s * 0.4 - now * 10) % h + h) % h;
      const px = (s * 0.57 % 1) * w + Math.sin(now * 0.8 + i) * 10;
      const blink = 0.4 + 0.5 * Math.sin(now * 3 + i * 1.7);
      ctx.globalAlpha = Math.max(0.1, blink);
      ctx.fillStyle = i % 4 ? o.bloom : "#d7ff8a";
      ctx.beginPath(); ctx.arc(px, py, i % 5 ? 1.2 : 2, 0, 7); ctx.fill();
    }
    ctx.restore();
  },
  sand(ctx, w, h, o, now) {
    ctx.save();
    ctx.fillStyle = hexA(o.rim, 0.22);
    ctx.beginPath(); ctx.moveTo(0, h);
    for (let x = 0; x <= w; x += 20) ctx.lineTo(x, h - 22 - Math.sin(x * 0.012 + 1) * 14);
    ctx.lineTo(w, h); ctx.closePath(); ctx.fill();
    ctx.globalCompositeOperation = "lighter"; ctx.strokeStyle = hexA(o.mid, 0.18); ctx.lineWidth = 1;
    for (let i = 0; i < 16; i++) {
      const s = i * 73.1;
      const sx = ((s * 0.6 + now * 120) % (w + 40)) - 20;
      const sy = (s * 0.4 % 1) * h;
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx + 14, sy + 3); ctx.stroke();
    }
    ctx.restore();
  },
  embers(ctx, w, h, o, now) {
    // Bottom glow now comes from the painted lava lake (tier-scenes pass C).
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const emberN = isMobile ? 11 : (_qLevel === 2 ? 34 : 22);
    for (let i = 0; i < emberN; i++) {
      const s = i * 47.1;
      const py = ((s * 0.5 - now * 40) % h + h) % h;
      const px = (s * 0.61 % 1) * w + Math.sin(now * 2 + i) * 8;
      ctx.globalAlpha = Math.max(0, 1 - py / h);
      ctx.fillStyle = i % 3 ? "#ff8a3a" : "#ffd08a";
      ctx.beginPath(); ctx.arc(px, py, i % 4 ? 1 : 1.8, 0, 7); ctx.fill();
    }
    ctx.restore();
  },
  petals(ctx, w, h, o, now) {
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const petalN = isMobile ? 9 : (_qLevel === 2 ? 27 : 18);
    for (let i = 0; i < petalN; i++) {
      const s = i * 59.7;
      const py = ((s * 0.45 - now * 16) % h + h) % h;
      const px = (s * 0.57 % 1) * w + Math.sin(now * 0.9 + i) * 16;
      const a = now * 2 + i;
      ctx.globalAlpha = 0.5;
      ctx.fillStyle = i % 3 ? o.mid : "#ffd0ee";
      ctx.beginPath(); ctx.ellipse(px, py, 4, 1.7, a, 0, 7); ctx.fill();
    }
    ctx.restore();
  },
  arcs(ctx, w, h, o, now) {
    ctx.save();
    if (Math.sin(now * 7) > 0.92) { ctx.fillStyle = hexA(o.bloom, 0.05); ctx.fillRect(0, 0, w, h); }
    ctx.globalCompositeOperation = "lighter"; ctx.strokeStyle = hexA(o.bloom, 0.4); ctx.lineWidth = 1;
    for (let i = 0; i < 3; i++) {
      if (Math.sin(now * 5 + i * 2) < 0.5) continue;
      let ax = (i * 0.3 + 0.2) * w, ay = 0;
      ctx.beginPath(); ctx.moveTo(ax, ay);
      for (let k = 0; k < 6; k++) { ax += (Math.random() - 0.5) * 30; ay += h / 6; ctx.lineTo(ax, ay); }
      ctx.stroke();
    }
    ctx.restore();
  },
  stardust(ctx, w, h, o, now) {
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const g = ctx.createRadialGradient(w * 0.5, h * 0.42, 0, w * 0.5, h * 0.42, Math.max(w, h) * 0.7);
    g.addColorStop(0, hexA(o.bloom, 0.08)); g.addColorStop(1, hexA(o.bloom, 0));
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
    const dustN = isMobile ? 30 : (_qLevel === 2 ? 90 : 60);
    for (let i = 0; i < dustN; i++) {
      const s = i * 101.7;
      ctx.globalAlpha = 0.3 + 0.6 * ((i * 7) % 5) / 5;
      ctx.fillStyle = "#fff";
      const sz = i % 9 ? 1 : 1.6;
      ctx.fillRect((s * 0.61 % 1) * w, ((s * 0.31 + now * 3) % h), sz, sz);
    }
    ctx.restore();
  },
  solar(ctx, w, h, o, now) {
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const mx = w - 34, my = 30;
    const mg = ctx.createRadialGradient(mx, my, 0, mx, my, 30);
    mg.addColorStop(0, hexA("#fff6d8", 0.9)); mg.addColorStop(0.5, hexA(o.bloom, 0.4)); mg.addColorStop(1, hexA(o.bloom, 0));
    ctx.fillStyle = mg; ctx.beginPath(); ctx.arc(mx, my, 30, 0, 7); ctx.fill();
    ctx.strokeStyle = hexA(o.bloom, 0.18); ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const a = now * 0.2 + i * Math.PI / 2.5;
      ctx.beginPath(); ctx.moveTo(w * 0.5, h * 0.4); ctx.lineTo(w * 0.5 + Math.cos(a) * w, h * 0.4 + Math.sin(a) * w); ctx.stroke();
    }
    ctx.restore();
  },
};

// ── BIOME CANDLE STYLES ──────────────────────────────────────────────────────
function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// Candle gradient cache: gradients are built at the origin and positioned with
// ctx.translate(), so one gradient per (color, quantized size) serves every
// candle — instead of 1-2 allocations per candle per frame.
const _candleGradCache = new Map();
function cachedCandleGradient(key, build) {
  let g = _candleGradCache.get(key);
  if (!g) {
    if (_candleGradCache.size > 128) _candleGradCache.clear();
    g = build();
    _candleGradCache.set(key, g);
  }
  return g;
}

const BIOME_CANDLES = {
  wire(ctx, candles, priceY, candleX, bw) {
    candles.forEach(c => {
      const x = candleX(c.time), up = c.close >= c.open;
      const col = up ? "#e8e8e8" : "#6f6f6f";
      const top = Math.min(priceY(c.open), priceY(c.close));
      const hgt = Math.max(2, Math.abs(priceY(c.close) - priceY(c.open)));
      ctx.strokeStyle = hexA(col, 0.5); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(Math.round(x) + .5, priceY(c.high)); ctx.lineTo(Math.round(x) + .5, priceY(c.low)); ctx.stroke();
      ctx.strokeRect(Math.round(x - bw / 2) + .5, Math.round(top) + .5, Math.round(bw), Math.round(hgt));
    });
  },
  aurora(ctx, candles, priceY, candleX, bw, t) {
    const plotTop = 48;
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    candles.forEach(c => {
      const x = candleX(c.time), up = c.close >= c.open;
      const col = up ? t.orb.bloom : "#5fa8d8";
      const top = Math.min(priceY(c.open), priceY(c.close));
      const hgt = Math.max(3, Math.abs(priceY(c.close) - priceY(c.open)));
      const len = Math.max(4, Math.round((priceY(c.low) - plotTop) / 4) * 4);
      const grad = cachedCandleGradient(`au|${col}|${len}`, () => {
        const g = ctx.createLinearGradient(0, 0, 0, len);
        g.addColorStop(0, hexA(col, 0)); g.addColorStop(1, hexA(col, 0.5));
        return g;
      });
      ctx.save(); ctx.translate(0, plotTop);
      ctx.fillStyle = grad; ctx.fillRect(x - bw / 2, top - plotTop, bw, hgt);
      ctx.restore();
      const rib = cachedCandleGradient(`au-rib|${col}`, () => {
        const g = ctx.createLinearGradient(0, 0, 0, 40);
        g.addColorStop(0, hexA(col, 0)); g.addColorStop(1, hexA(col, 0.35));
        return g;
      });
      ctx.save(); ctx.translate(0, top - 40);
      ctx.fillStyle = rib; ctx.fillRect(x - bw / 2, 0, bw, 40);
      ctx.restore();
      ctx.strokeStyle = hexA("#ffffff", 0.4); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, priceY(c.high)); ctx.lineTo(x, priceY(c.low)); ctx.stroke();
    });
    ctx.restore();
  },
  abyss(ctx, candles, priceY, candleX, bw, t) {
    ctx.save();
    const d = candles.map((c, n) => {
      const x = candleX(c.time), up = c.close >= c.open;
      return { x, n, col: up ? t.orb.mid : "#1c7d8f",
               top: Math.min(priceY(c.open), priceY(c.close)),
               hgt: Math.max(6, Math.abs(priceY(c.close) - priceY(c.open))),
               hiY: priceY(c.high), loY: priceY(c.low) };
    });
    // pass 1 — wavy wicks, no shadow, adaptive step (max 28 points)
    ctx.lineWidth = 2;
    d.forEach(({ x, n, col, hiY, loY }) => {
      const wickStep = Math.max(4, Math.ceil((loY - hiY) / 28));
      ctx.strokeStyle = hexA(col, 0.5);
      ctx.beginPath();
      let first = true;
      for (let yy = loY; yy >= hiY; yy -= wickStep) {
        const xx = x + Math.sin(yy * 0.18 + n) * 2.2;
        first ? ctx.moveTo(xx, yy) : ctx.lineTo(xx, yy); first = false;
      }
      ctx.stroke();
    });
    // pass 2 — bodies with shadow (shadowBlur set/reset once)
    ctx.shadowBlur = 12;
    d.forEach(({ x, col, top, hgt }) => {
      ctx.shadowColor = col;
      ctx.fillStyle = hexA(col, 0.32);
      roundRect(ctx, x - bw / 2, top, bw, hgt, bw / 2); ctx.fill();
    });
    ctx.shadowBlur = 0;
    // pass 3 — caps, no shadow
    const capR = bw * 0.7;
    d.forEach(({ x, col, top }) => {
      const cap = cachedCandleGradient(`abyss|${col}|${Math.round(capR)}`, () => {
        const g = ctx.createRadialGradient(0, 0, 0, 0, 0, capR);
        g.addColorStop(0, hexA("#ffffff", 0.8)); g.addColorStop(1, hexA(col, 0));
        return g;
      });
      ctx.save(); ctx.translate(x, top + 3);
      ctx.fillStyle = cap; ctx.beginPath(); ctx.arc(0, 0, capR, 0, 7); ctx.fill();
      ctx.restore();
    });
    ctx.restore();
  },
  verdant(ctx, candles, priceY, candleX, bw, t) {
    ctx.save();
    const d = candles.map((c, n) => {
      const x = candleX(c.time), up = c.close >= c.open;
      return { x, n, up, col: up ? t.orb.mid : "#a8702f",
               top: Math.min(priceY(c.open), priceY(c.close)),
               hgt: Math.max(6, Math.abs(priceY(c.close) - priceY(c.open))),
               hiY: priceY(c.high), loY: priceY(c.low) };
    });
    // pass 1 — wicks, no shadow
    ctx.lineWidth = 2;
    d.forEach(({ x, up, col, hiY, loY }) => {
      ctx.strokeStyle = hexA(col, 0.5);
      ctx.beginPath(); ctx.moveTo(x, hiY); ctx.lineTo(x + (up ? 0 : 4), loY); ctx.stroke();
    });
    // pass 2 — down bodies, no shadow
    d.forEach(({ x, up, col, top, hgt }) => {
      if (up) return;
      ctx.fillStyle = hexA(col, 0.5); ctx.shadowColor = col;
      roundRect(ctx, x - bw / 2, top, bw, hgt, 3); ctx.fill();
    });
    // pass 3 — up bodies with shadow (shadowBlur set/reset once)
    ctx.shadowBlur = 8;
    d.forEach(({ x, up, col, top, hgt }) => {
      if (!up) return;
      ctx.shadowColor = col; ctx.fillStyle = hexA(col, 0.5);
      roundRect(ctx, x - bw / 2, top, bw, hgt, 3); ctx.fill();
    });
    ctx.shadowBlur = 0;
    // pass 4 — inner lines + leaf curves (O5: batch into 2 stroke() calls per candle)
    ctx.lineWidth = 1.4;
    d.forEach(({ x, n, col, top, hgt }) => {
      const lf = n % 2 ? 1 : -1;
      ctx.strokeStyle = hexA(col, 0.8);
      ctx.beginPath();
      for (let yy = top + 6; yy < top + hgt; yy += 12)
        { ctx.moveTo(x - bw / 2, yy); ctx.lineTo(x + bw / 2, yy); }
      ctx.stroke();
      ctx.beginPath();
      for (let yy = top + 6; yy < top + hgt; yy += 12)
        { ctx.moveTo(x + lf * bw / 2, yy); ctx.quadraticCurveTo(x + lf * bw, yy - 5, x + lf * bw * 1.4, yy); }
      ctx.stroke();
    });
    // pass 5 — bloom dots (up candles only)
    ctx.fillStyle = hexA(t.orb.bloom, 0.9);
    d.forEach(({ x, up, top }) => {
      if (!up) return;
      ctx.beginPath(); ctx.arc(x, top, 2.6, 0, 7); ctx.fill();
    });
    ctx.restore();
  },
  dune(ctx, candles, priceY, candleX, bw, t) {
    ctx.save();
    candles.forEach(c => {
      const x = candleX(c.time), up = c.close >= c.open;
      const col = up ? t.orb.mid : "#9c6a3a";
      const top = Math.min(priceY(c.open), priceY(c.close));
      const hgt = Math.max(8, Math.abs(priceY(c.close) - priceY(c.open)));
      const bands = Math.max(2, Math.floor(hgt / 9));
      for (let i = 0; i < bands; i++) {
        ctx.fillStyle = hexA(col, 0.22 + (i / bands) * 0.30);
        ctx.fillRect(x - bw / 2, top + i * (hgt / bands), bw, hgt / bands - 1.5);
      }
      ctx.strokeStyle = hexA(t.orb.bloom, 0.5); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x - bw / 2, top); ctx.lineTo(x + bw / 2, top); ctx.stroke();
    });
    ctx.restore();
  },
  magma(ctx, candles, priceY, candleX, bw, t) {
    ctx.save();
    candles.forEach(c => {
      const x = candleX(c.time), up = c.close >= c.open;
      const col = up ? t.orb.mid : "#7a1f0a";
      const top = Math.min(priceY(c.open), priceY(c.close));
      const bot = Math.max(priceY(c.open), priceY(c.close));
      const hgt = Math.max(8, bot - top);
      const hQ = Math.max(8, Math.round(hgt / 4) * 4);
      const grad = cachedCandleGradient(`magma|${col}|${hQ}`, () => {
        const g = ctx.createLinearGradient(0, 0, 0, hQ);
        g.addColorStop(0, "#1a0a04"); g.addColorStop(0.5, hexA(col, 0.7)); g.addColorStop(1, hexA("#ffd08a", 0.9));
        return g;
      });
      ctx.save(); ctx.translate(0, top);
      ctx.fillStyle = grad; ctx.fillRect(x - bw / 2, 0, bw, hgt);
      ctx.restore();
      ctx.strokeStyle = hexA("#ffb469", 0.7); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, top + 2); ctx.lineTo(x + 2, top + hgt * 0.4); ctx.lineTo(x - 1, bot - 2); ctx.stroke();
      ctx.strokeStyle = hexA("#ff8a3a", 0.5); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, priceY(c.high)); ctx.lineTo(x, priceY(c.low)); ctx.stroke();
    });
    ctx.restore();
  },
  orchid(ctx, candles, priceY, candleX, bw, t) {
    ctx.save();
    const now = performance.now() / 1000;
    const petalCount = isMobile ? 2 : 4;
    const d = candles.map(c => {
      const x = candleX(c.time), up = c.close >= c.open;
      return { x, up, col: up ? t.orb.mid : "#7a2f8a",
               top: Math.min(priceY(c.open), priceY(c.close)),
               hgt: Math.max(8, Math.abs(priceY(c.close) - priceY(c.open))) };
    });
    // pass 1 — hexagons with shadow (shadowBlur set/reset once)
    ctx.lineWidth = 1;
    ctx.shadowBlur = 10;
    d.forEach(({ x, col, top, hgt }) => {
      ctx.shadowColor = col;
      ctx.fillStyle = hexA(col, 0.34); ctx.strokeStyle = hexA(col, 0.8);
      ctx.beginPath();
      ctx.moveTo(x, top); ctx.lineTo(x + bw / 2, top + 8); ctx.lineTo(x + bw / 2, top + hgt);
      ctx.lineTo(x, top + hgt + 6); ctx.lineTo(x - bw / 2, top + hgt); ctx.lineTo(x - bw / 2, top + 8);
      ctx.closePath(); ctx.fill(); ctx.stroke();
    });
    ctx.shadowBlur = 0;
    // pass 2 — centerlines + petals, no shadow
    ctx.strokeStyle = hexA("#ffffff", 0.4);
    d.forEach(({ x, up, top, hgt }) => {
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + hgt + 6); ctx.stroke();
      if (up) {
        ctx.fillStyle = hexA(t.orb.bloom, 0.9);
        for (let k = 0; k < petalCount; k++) {
          const a = k * (Math.PI * 2 / petalCount) + now;
          ctx.beginPath(); ctx.ellipse(x + Math.cos(a) * 4, top + Math.sin(a) * 4, 3, 1.6, a, 0, 7); ctx.fill();
        }
      }
    });
    ctx.restore();
  },
  volt(ctx, candles, priceY, candleX, bw, t) {
    ctx.save();
    const now = performance.now() / 1000;
    const d = candles.map((c, n) => {
      const x = candleX(c.time), up = c.close >= c.open;
      return { x, n, up, col: up ? t.orb.mid : "#5a3a8a",
               top: Math.min(priceY(c.open), priceY(c.close)),
               hgt: Math.max(7, Math.abs(priceY(c.close) - priceY(c.open))),
               hiY: priceY(c.high), loY: priceY(c.low) };
    });
    // pass 1 — bars + wicks, no shadow
    d.forEach(({ x, up, col, top, hgt, hiY, loY }) => {
      ctx.strokeStyle = hexA(col, up ? 0.9 : 0.4); ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(x - bw / 2, top); ctx.lineTo(x + bw / 2, top);
      ctx.moveTo(x - bw / 2, top + hgt); ctx.lineTo(x + bw / 2, top + hgt); ctx.stroke();
      ctx.strokeStyle = hexA(col, 0.3); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + hgt); ctx.stroke();
      ctx.strokeStyle = hexA(col, 0.5);
      ctx.beginPath(); ctx.moveTo(x, hiY); ctx.lineTo(x, loY); ctx.stroke();
    });
    // pass 2 — lightning arcs with shadow (shadowBlur set/reset once)
    const lightnings = [];
    d.forEach(({ x, n, top }, i) => {
      if (i === 0) return;
      if (Math.sin(n * 1.7 + now * 4) > 0.4) {
        const prev = d[i - 1];
        lightnings.push({
          x1: prev.x, y1: prev.top,
          mx: (prev.x + x) / 2 + (Math.random() - 0.5) * 8,
          my: (prev.top + top) / 2 + (Math.random() - 0.5) * 10,
          x2: x, y2: top
        });
      }
    });
    if (lightnings.length) {
      ctx.strokeStyle = hexA("#ffffff", 0.5); ctx.lineWidth = 1;
      ctx.shadowColor = t.orb.bloom; ctx.shadowBlur = 8;
      lightnings.forEach(({ x1, y1, mx, my, x2, y2 }) => {
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(mx, my); ctx.lineTo(x2, y2); ctx.stroke();
      });
      ctx.shadowBlur = 0;
    }
    ctx.restore();
  },
  nebula(ctx, candles, priceY, candleX, bw, t) {
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const now = performance.now() / 1000;
    candles.forEach((c, n) => {
      const x = candleX(c.time), up = c.close >= c.open;
      const col = up ? t.orb.mid : "#6a4abf";
      const top = Math.min(priceY(c.open), priceY(c.close));
      const hgt = Math.max(10, Math.abs(priceY(c.close) - priceY(c.open)));
      const g = cachedCandleGradient(`nebula|${col}|${Math.round(bw)}`, () => {
        const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, bw);
        grad.addColorStop(0, hexA(col, 0.4)); grad.addColorStop(1, hexA(col, 0));
        return grad;
      });
      ctx.save(); ctx.translate(x, top + hgt / 2);
      ctx.fillStyle = g; ctx.fillRect(-bw, -hgt / 2 - 6, bw * 2, hgt + 12);
      ctx.restore();
      for (let i = 0; i < 3; i++) {
        const sy = top + (Math.sin(n * 3 + i * 2.1) * 0.5 + 0.5) * hgt;
        ctx.fillStyle = hexA("#ffffff", 0.5 + 0.4 * Math.sin(now * 3 + n + i));
        ctx.beginPath(); ctx.arc(x + (i - 1) * 4, sy, 1, 0, 7); ctx.fill();
      }
    });
    ctx.restore();
  },
  apex(ctx, candles, priceY, candleX, bw, t) {
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    candles.forEach(c => {
      const x = candleX(c.time), up = c.close >= c.open;
      const col = up ? t.orb.mid : "#b5701a";
      const top = Math.min(priceY(c.open), priceY(c.close));
      const hgt = Math.max(8, Math.abs(priceY(c.close) - priceY(c.open)));
      const grad = cachedCandleGradient(`apex|${col}|${Math.round(bw)}`, () => {
        const g = ctx.createLinearGradient(-bw / 2, 0, bw / 2, 0);
        g.addColorStop(0, hexA(col, 0)); g.addColorStop(0.5, hexA(col, 0.6)); g.addColorStop(1, hexA(col, 0));
        return g;
      });
      ctx.save(); ctx.translate(x, 0);
      ctx.fillStyle = grad; ctx.fillRect(-bw / 2, top, bw, hgt);
      ctx.restore();
      ctx.fillStyle = hexA("#fff6d8", 0.85); ctx.fillRect(x - 1.5, top, 3, hgt);
      ctx.strokeStyle = hexA(t.orb.bloom, 0.6); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, priceY(c.high)); ctx.lineTo(x, priceY(c.low)); ctx.stroke();
    });
    ctx.restore();
  },
};

function drawMarketChart() {
  const allCandles = marketCandles();
  if (allCandles.length < 2) return;

  const t = tier();
  const now = Date.now();
  const axisTop = H - BOTTOM_BAR_H - MARKET_AXIS_H;
  const plotBottom = axisTop;
  const hudSafeTop = W <= 640 ? 148 : 120;
  const plotTop = Math.min(hudSafeTop, Math.max(48, plotBottom - 220));
  const priceAxisWidth = Math.max(52, Math.min(78, W * .10));
  const plotRight = W - priceAxisWidth;
  const plotWidth = Math.max(180, plotRight - 18);
  const candleStep = plotWidth / VISIBLE_CANDLE_COUNT;
  const candleX = time => plotRight - (now-time)/INTERVALS[activeInterval].ms*candleStep;
  const candles = allCandles.filter(c=> {
    const x = candleX(c.time);
    return x > -candleStep && x < plotRight+candleStep;
  });
  if (candles.length < 2) return;

  const priceRange = chartPriceRange(candles);
  if (!priceRange) return;
  const pMin = priceRange.min, pMax = priceRange.max;
  const priceY = price => plotBottom - (price-pMin)/(pMax-pMin||1)*(plotBottom-plotTop);
  const crisp = value => Math.round(value)+.5;
  const hudClearBottom = Math.min(plotBottom, plotTop);

  ctx.save();

  // Keep the price scale visible at every tier, below the top-right HUD.
  ctx.fillStyle = hexA(t.orb.bloom,.72);
  ctx.font = "14px VT323, monospace";
  ctx.textAlign = "right";
  ctx.strokeStyle = hexA(t.orb.bloom,.22);
  ctx.beginPath(); ctx.moveTo(crisp(plotRight), hudClearBottom); ctx.lineTo(crisp(plotRight), plotBottom); ctx.stroke();
  for (let i=0; i<=5; i++) {
    const y = plotTop + (plotBottom-plotTop)*i/5;
    if (y < hudClearBottom) continue;
    const price = pMax-(pMax-pMin)*i/5;
    const label = formatChartPrice(price);
    // 1px drop shadow keeps the scale readable over the painted scenes
    // (GENESIS has no scene and keeps its legacy chart untouched).
    if (activeTier > 0) {
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.fillText(label,Math.round(W - 8)+1,Math.round(y-4)+1);
      ctx.fillStyle = hexA(t.orb.bloom,.72);
    }
    ctx.fillText(label,Math.round(W - 8),Math.round(y-4));
  }

  if (t.feats.grid) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = hexA(t.orb.bloom,.065);
    for (let i=0; i<=5; i++) {
      const y = plotTop + (plotBottom-plotTop)*i/5;
      ctx.beginPath(); ctx.moveTo(0,crisp(y)); ctx.lineTo(plotRight,crisp(y)); ctx.stroke();
    }
    for (let i=0; i<=VISIBLE_CANDLE_COUNT; i+=4) {
      const x = plotRight-i*candleStep;
      ctx.beginPath(); ctx.moveTo(crisp(x),plotTop); ctx.lineTo(crisp(x),plotBottom); ctx.stroke();
    }
  } else {
    ctx.strokeStyle = hexA(t.orb.bloom,.07);
    ctx.beginPath(); ctx.moveTo(0,crisp(plotBottom)); ctx.lineTo(plotRight,crisp(plotBottom)); ctx.stroke();
  }

  // Candle drop shadows — style-agnostic backing so the chart stays readable
  // over the bright zones of the painted scenes (skipped on GENESIS).
  const bodyWidth = Math.max(5,Math.min(18,candleStep*.52));
  if (activeTier > 0) {
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    for (const c of candles) {
      const x = candleX(c.time);
      const top = priceY(c.high), bot = priceY(c.low);
      ctx.fillRect(x - bodyWidth / 2 - 1, top - 1, bodyWidth + 2, Math.max(2, bot - top) + 2);
    }
  }

  // Per-biome candle style dispatch.
  const styleFn = BIOME_CANDLES[t.candle] || BIOME_CANDLES.wire;
  ctx.lineWidth = 1;
  styleFn(ctx, candles, priceY, candleX, bodyWidth, t);

  // Reinforced live price line + marker at the latest candle (not on GENESIS).
  const lastC = candles[candles.length - 1];
  const lpy = priceY(lastC.close);
  if (activeTier > 0 && lpy > hudClearBottom && lpy < plotBottom) {
    ctx.strokeStyle = hexA(t.orb.bloom, 0.5);
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, crisp(lpy)); ctx.lineTo(plotRight, crisp(lpy)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = 0.55;
    ctx.drawImage(getGlowSprite(t.orb.bloom, 16), candleX(lastC.time) - 9, lpy - 9, 18, 18);
    ctx.restore();
  }

  const buyGlowFade = buyChartGlowDuration > 0 ? Math.min(1, buyChartGlowDuration / 0.45) : 0;
  const buyGlowAlpha = buyGlowFade * buyChartGlowPower * 0.45;
  if (buyGlowAlpha > 0) {
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = `rgba(0,255,80,${buyGlowAlpha.toFixed(3)})`;
    for (const c of candles) {
      if (c.close < c.open) continue;
      const x = candleX(c.time);
      const top = priceY(Math.max(c.open, c.close));
      const bot = priceY(Math.min(c.open, c.close));
      const h = Math.max(2, bot - top);
      ctx.fillRect(x - bodyWidth / 2, top, bodyWidth, h);
    }
    ctx.restore();
  }
  if (sellChartFlash > 0) {
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = `rgba(255,60,60,${(sellChartFlash * 0.38).toFixed(3)})`;
    for (const c of candles) {
      if (c.close >= c.open) continue;
      const x = candleX(c.time);
      const top = priceY(Math.max(c.open, c.close));
      const bot = priceY(Math.min(c.open, c.close));
      const h = Math.max(2, bot - top);
      ctx.fillRect(x - bodyWidth / 2, top, bodyWidth, h);
    }
    ctx.restore();
  }

  // Dedicated time-axis strip — fade from transparent to bg so labels stay readable.
  if (!_axisGrad || activeTier !== _axisTierIdx || axisTop !== _axisTop) {
    _axisGrad = ctx.createLinearGradient(0, axisTop - 20, 0, axisTop + MARKET_AXIS_H);
    _axisGrad.addColorStop(0, hexA(t.bg0||"#000000", 0));
    _axisGrad.addColorStop(0.35, hexA(t.bg0||"#000000", 0.45));
    _axisGrad.addColorStop(1,   hexA(t.bg0||"#000000", 0.72));
    _axisTierIdx = activeTier; _axisTop = axisTop;
  }
  ctx.fillStyle = _axisGrad;
  ctx.fillRect(0, axisTop - 20, W, MARKET_AXIS_H + 20);
  ctx.strokeStyle = hexAlpha(t.orbG,.2);
  ctx.beginPath(); ctx.moveTo(0,crisp(axisTop)); ctx.lineTo(W,crisp(axisTop)); ctx.stroke();
  ctx.fillStyle = hexAlpha(t.orbG,.78);
  ctx.font = "14px VT323, monospace";
  ctx.textAlign = "left";
  for (let i=0; i<=VISIBLE_CANDLE_COUNT; i+=4) {
    const x = plotRight-i*candleStep;
    ctx.fillText(formatAxisTime(now-i*INTERVALS[activeInterval].ms),Math.round(x+3),Math.round(axisTop+18));
  }
  ctx.fillStyle = hexAlpha(t.orbG,.68);
  ctx.textAlign = "right";
  const utcX = Math.min(W - 10, Math.max(plotRight + 54, 42));
  ctx.fillText("UTC",Math.round(utcX),Math.round(axisTop+18));
  ctx.restore();
}

function drawSpeedLines() {
  if (state !== "playing") return;
  const ultiSLMods = _frameMods || getUltiModifiers();
  const baseIntensity = orb.vy >= 480 ? Math.min(1, (orb.vy - 480) / 700) : 0;
  const overclockBoost = ultiSLMods.speedLineIntensity > 1 ? 0.5 : 0;
  const intensity = Math.min(1, baseIntensity + rocketMode * 1.8 + overclockBoost);
  if (intensity < 0.05) return;

  const t = _frameNow;
  ctx.save();

  // Traits verticaux filants (bords gauche+droit) — plus nombreux et longs en rocket
  let lineCount = rocketMode > 0 ? 24 : 10;
  if (!QF.fx) lineCount = Math.ceil(lineCount / 2);
  for (let i = 0; i < lineCount; i++) {
    const x = (i * 97 + t * 0.4) % W;
    const len = (30 + intensity * 60) * (rocketMode > 0 ? 2.2 : 1);
    const alpha = intensity * (rocketMode > 0 ? 0.55 : 0.4);
    ctx.globalAlpha = alpha * (i % 3 === 0 ? 1 : 0.55);
    ctx.strokeStyle = rocketMode > 0 ? `rgba(160,220,255,1)` : tier().orb.bloom;
    ctx.lineWidth = rocketMode > 0 ? 1.8 : 1.4;
    ctx.beginPath();
    ctx.moveTo(x, (i * 83) % H);
    ctx.lineTo(x, (i * 83) % H + len);
    ctx.stroke();
  }

  // Radial rays converging toward the orb — rocket mode warp effect only
  if (rocketMode > 0.1) {
    const orbScreenY = w2s(orb.worldY);
    const rayCount = QF.fx ? 16 : 8;
    for (let i = 0; i < rayCount; i++) {
      const angle = (i / rayCount) * Math.PI * 2;
      const dist = 55 + rocketMode * 110;
      const len2 = rocketMode * 90;
      ctx.globalAlpha = rocketMode * 0.6;
      ctx.strokeStyle = `rgba(140,210,255,1)`;
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.moveTo(orb.x + Math.cos(angle) * (dist + len2), orbScreenY + Math.sin(angle) * (dist + len2));
      ctx.lineTo(orb.x + Math.cos(angle) * dist, orbScreenY + Math.sin(angle) * dist);
      ctx.stroke();
    }
  }

  ctx.restore();
}

function getPlatformColor(plat, t) {
  return plat.type === "buy" ? t.platBuy : plat.type === "sell" ? t.platSell : t.platN;
}

// Wear state for tier-visuals supports: bounces remaining >=3 -> 0 (fresh),
// 2 -> 1 (worn), 1 -> 2 (critical). Storm-shrunk neutrals (bounceLimit 1/2)
// therefore spawn already worn/critical, which matches their actual lifespan.
function platformWearState(p) {
  if (p.type !== "neutral" || !p.bounceLimit) return 0;
  const remaining = p.bounceLimit - (p.receptions || 0);
  return Math.max(0, Math.min(2, 3 - remaining));
}

// Effect level passed to tier-visuals renderers: 0 = LOW (static + core
// animation only), 1 = mobile/medium (reduced particle counts), 2 = full.
function tierVisualsFx() {
  return !QF.fx ? 0 : (isMobile ? 1 : 2);
}

const IMPACT_FLARE_MS = 260; // bounce flare duration fed to support renderers

function drawPlatforms() {
  if (activeAnomaly?.type === "liquidityVoid") return;
  const t = tier();
  const now = _frameNow;
  const fx = tierVisualsFx();

  platforms.forEach(p => {
    const sy = w2s(p.worldY);
    if (sy < -PLAT_H-30 || sy > H+30) return;

    // Per-tier support visuals (handoff T2-T10); false -> legacy path below.
    if (activeTier > 0 && p.type !== "sell" &&
        drawTierSupport(ctx, activeTier + 1, p.x, sy, p.width, now / 1000, {
          wear: platformWearState(p),
          impact: p.hitAt ? Math.max(0, 1 - (now - p.hitAt) / IMPACT_FLARE_MS) : 0,
          seed: p.seed,
          gs, fx, platH: PLAT_H,
          reducedMotion: _reducedMotion,
        })) return;

    const color = getPlatformColor(p, t);

    ctx.save();
    ctx.globalAlpha = platformOpacity(p);

    // Sell: pulse to warn player
    if (p.type==="sell") {
      ctx.globalAlpha *= 0.65 + Math.sin(now*0.008)*0.35;
    }

    // Buy: glow (from tier 2 onwards) — baked strip, alpha pulse replaces blur pulse
    if (p.type==="buy" && activeTier > 0) {
      const { canvas: gc, pad, capW, w: gw, h: gh } = getPlatformGlowStrip(color);
      const x = Math.round(p.x), y = Math.round(sy), pw = Math.round(p.width);
      ctx.globalAlpha = platformOpacity(p) * (0.55 + Math.sin(now*0.005+p.age)*0.25);
      const midW = pw + pad*2 - capW*2;
      if (midW > 0) {
        ctx.drawImage(gc, 0, 0, capW, gh, x - pad, y - pad, capW, gh);
        ctx.drawImage(gc, capW, 0, gw - capW*2, gh, x - pad + capW, y - pad, midW, gh);
        ctx.drawImage(gc, gw - capW, 0, capW, gh, x + pw + pad - capW, y - pad, capW, gh);
      } else {
        ctx.drawImage(gc, x - pad, y - pad, pw + pad*2, gh);
      }
      ctx.globalAlpha = platformOpacity(p);
    }

    ctx.fillStyle = color;
    // sy = screen Y of platform top surface; platform body extends downward
    ctx.fillRect(Math.round(p.x), Math.round(sy), Math.round(p.width), PLAT_H);

    // Top highlight strip (tier 2+)
    if (activeTier > 0) {
      ctx.globalAlpha = platformOpacity(p)*0.35;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(Math.round(p.x), Math.round(sy), Math.round(p.width), 2);
    }

    ctx.restore();
  });
}

// Stream interior flow: integrated animation time so speed changes never
// teleport particles. speedMul eases toward a target driven by camera ascent
// velocity (jump = camera rush = particles partially keep up with the world).
const STREAM_FLOW_BASE    = 1.15; // calm-state multiplier
const STREAM_FLOW_MAX     = 1.9;  // multiplier at/above reference ascent speed
const STREAM_FLOW_VREF    = 0.9;  // camera velocity (screens/s) mapping to MAX
const STREAM_FLOW_ATTACK  = 0.12; // s, easing time constant when speeding up
const STREAM_FLOW_RELEASE = 0.35; // s, easing time constant when slowing down
let _streamAnimT = 0, _streamSpeedMul = STREAM_FLOW_BASE, _streamPrevCamY = 0, _streamPrevNow = 0;

function updateStreamFlow(now) {
  const fdt = Math.min(0.05, (now - _streamPrevNow) / 1000); // clamp absorbs pause/tab gaps
  _streamPrevNow = now;
  if (fdt <= 0) return;
  const camV = Math.max(0, (cameraY - _streamPrevCamY) / fdt); // upward screen px/s
  _streamPrevCamY = cameraY;
  const target = _reducedMotion ? STREAM_FLOW_BASE
    : STREAM_FLOW_BASE + (STREAM_FLOW_MAX - STREAM_FLOW_BASE) * Math.min(1, camV / (H * STREAM_FLOW_VREF));
  const tau = target > _streamSpeedMul ? STREAM_FLOW_ATTACK : STREAM_FLOW_RELEASE;
  _streamSpeedMul += (target - _streamSpeedMul) * (1 - Math.exp(-fdt / tau));
  _streamAnimT += fdt * _streamSpeedMul;
}

function drawBoosters() {
  const now = _frameNow;
  updateStreamFlow(now);
  boosters.forEach(booster => {
    if (booster.type === "stream") { drawBoostStream(booster); return; }
    const sy = w2s(booster.worldY);
    if (sy < -60 || sy > H+60 || booster.used) return;
    if (booster.type === "surge") drawEnergyBonus(booster,sy,now);
    else drawRedWall(booster,sy,now);
  });
}

function drawEnergyBonus(booster,sy,now) {
  if (activeTier > 0 &&
      drawTierSurge(ctx, activeTier + 1, booster.x, sy, booster.w * .5, now / 1000, {
        phase: booster.phase, gs, fx: tierVisualsFx(), reducedMotion: _reducedMotion,
      })) return;
  const r=booster.w*.5, spin=now*.003+booster.phase, glow="#ffdd88";
  ctx.save(); ctx.translate(booster.x,sy); ctx.globalCompositeOperation="lighter";
  const hr=r*2.4;
  ctx.globalAlpha=.6; ctx.drawImage(getGlowSprite(glow,32),-hr,-hr,hr*2,hr*2); ctx.globalAlpha=1;
  ctx.globalCompositeOperation="source-over"; ctx.rotate(spin);
  ctx.fillStyle="#ffcf5c";
  ctx.beginPath(); ctx.moveTo(0,-r); ctx.lineTo(r,0); ctx.lineTo(0,r); ctx.lineTo(-r,0); ctx.closePath(); ctx.fill();
  ctx.fillStyle="#fff8e0";
  ctx.beginPath(); ctx.moveTo(0,-r*.45); ctx.lineTo(r*.45,0); ctx.lineTo(0,r*.45); ctx.lineTo(-r*.45,0); ctx.closePath(); ctx.fill();
  ctx.restore();
}

function drawBoostStream(booster) {
  const top=w2s(booster.worldY+booster.len), bot=w2s(booster.worldY);
  if (bot < -40 || top > H+40) return;
  const x1=booster.x-booster.w/2, x2=booster.x+booster.w/2, color=tier().orbG;
  if (activeTier > 0 &&
      // Draw window is clamped to the viewport; `scroll` anchors the interior
      // pattern to the world so it doesn't swim with the camera, and `span`
      // (full corridor height) keeps wrap periods constant while the corridor
      // is only partly on screen. Time is the integrated stream flow, offset
      // by the per-booster phase to desync corridors.
      drawTierStream(ctx, activeTier + 1, x1, x2, Math.max(top, -40), Math.min(bot, H + 40), _streamAnimT + booster.phase, {
        phase: booster.phase, scroll: bot, span: bot - top, gs, fx: tierVisualsFx(), reducedMotion: _reducedMotion,
      })) return;
  ctx.save(); ctx.globalCompositeOperation="lighter";
  const fill=ctx.createLinearGradient(x1,0,x2,0);
  fill.addColorStop(0,hexA(color,0)); fill.addColorStop(.5,hexA(color,.1)); fill.addColorStop(1,hexA(color,0));
  ctx.fillStyle=fill; ctx.fillRect(x1,top,booster.w,bot-top);
  ctx.globalCompositeOperation="source-over"; ctx.strokeStyle=color; ctx.lineWidth=Math.max(2,gs*2.4);
  ctx.beginPath(); ctx.moveTo(x1,top); ctx.lineTo(x1,bot); ctx.moveTo(x2,top); ctx.lineTo(x2,bot); ctx.stroke();
  const gap=gs*26, anim=(_streamAnimT*60+booster.phase*40)%gap;
  ctx.strokeStyle=hexA(color,.7); ctx.lineWidth=Math.max(1.5,gs*1.8);
  for(let y=bot-anim;y>top;y-=gap){ctx.beginPath();ctx.moveTo(booster.x-gs*8,y);ctx.lineTo(booster.x,y-gs*9);ctx.lineTo(booster.x+gs*8,y);ctx.stroke();}
  ctx.restore();
}

function drawRedWall(booster,sy,now) {
  const w=booster.w, h=Math.max(gs*16,w*.18);
  if (activeTier > 0 &&
      drawTierResist(ctx, activeTier + 1, booster.x, sy, w, h, now / 1000, {
        phase: booster.phase, gs, fx: tierVisualsFx(), reducedMotion: _reducedMotion,
      })) return;
  ctx.save(); ctx.translate(booster.x,sy); ctx.globalAlpha=.55+Math.sin(now*.008+booster.phase)*.3;
  const g=ctx.createLinearGradient(0,-h/2,0,h/2);
  g.addColorStop(0,hexA("#ff4d6d",0)); g.addColorStop(.5,hexA("#ff4d6d",.5)); g.addColorStop(1,hexA("#ff4d6d",0));
  ctx.fillStyle=g; ctx.fillRect(-w/2,-h/2,w,h); ctx.strokeStyle=hexA("#ff4d6d",.7); ctx.lineWidth=Math.max(1.5,gs*1.6);
  for(let x=-w/2;x<w/2;x+=gs*12){ctx.beginPath();ctx.moveTo(x,-h/2);ctx.lineTo(x+h,h/2);ctx.stroke();}
  ctx.restore();
}

function drawShockwaves() {
  if (!shockwaves.length) return;
  const orbG = tier().orbG;
  ctx.save();
  ctx.strokeStyle = orbG;
  // Double-stroke halo (wide faint + thin bright) instead of shadowBlur
  shockwaves.forEach(sw => {
    ctx.beginPath();
    ctx.arc(sw.x, sw.y, sw.r, 0, Math.PI*2);
    ctx.globalAlpha = sw.life * 0.16;
    ctx.lineWidth = 6;
    ctx.stroke();
    ctx.globalAlpha = sw.life * 0.45;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  });
  ctx.restore();
}

function drawRocketFlame() {
  if (rocketMode <= 0) return;
  const sx = orb.x;
  const sy = w2s(orb.worldY);
  const R  = orb.radius;
  const now = performance.now() / 1000;
  const flicker  = 0.82 + 0.18 * Math.sin(now * 24.3);
  const flicker2 = 0.88 + 0.12 * Math.sin(now * 17.1 + 1.4);
  const flameLen = (240 + rocketMode * 680) * flicker;
  const flameW   = R * 2.2 * flicker2;
  // VOID portal launch uses a cyan flame (its own identity); buy rockets stay green.
  const cyan = rocketFlameCyan;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  const g1 = ctx.createLinearGradient(sx, sy + R * 0.8, sx, sy + R * 0.8 + flameLen);
  g1.addColorStop(0,   cyan ? `rgba(150,240,255,${rocketMode * 0.85})` : `rgba(120,255,140,${rocketMode * 0.85})`);
  g1.addColorStop(0.2, cyan ? `rgba(60,200,255,${rocketMode * 0.7})`  : `rgba(0,255,90,${rocketMode * 0.7})`);
  g1.addColorStop(0.6, cyan ? `rgba(0,150,230,${rocketMode * 0.35})`  : `rgba(0,200,50,${rocketMode * 0.35})`);
  g1.addColorStop(1,   cyan ? 'rgba(0,40,90,0)' : 'rgba(0,80,20,0)');
  ctx.fillStyle = g1;
  ctx.beginPath();
  ctx.moveTo(sx - flameW / 2, sy + R * 0.8);
  ctx.quadraticCurveTo(sx - flameW * 0.65, sy + R + flameLen * 0.45, sx, sy + R + flameLen);
  ctx.quadraticCurveTo(sx + flameW * 0.65, sy + R + flameLen * 0.45, sx + flameW / 2, sy + R * 0.8);
  ctx.closePath();
  ctx.fill();
  const coreW = flameW * 0.38;
  const coreLen = flameLen * 0.6;
  const g2 = ctx.createLinearGradient(sx, sy + R, sx, sy + R + coreLen);
  g2.addColorStop(0,   cyan ? `rgba(225,250,255,${rocketMode * 0.9})` : `rgba(220,255,220,${rocketMode * 0.9})`);
  g2.addColorStop(0.5, cyan ? `rgba(120,225,255,${rocketMode * 0.5})` : `rgba(80,255,120,${rocketMode * 0.5})`);
  g2.addColorStop(1,   cyan ? 'rgba(0,160,230,0)' : 'rgba(0,200,60,0)');
  ctx.fillStyle = g2;
  ctx.beginPath();
  ctx.moveTo(sx - coreW / 2, sy + R);
  ctx.quadraticCurveTo(sx - coreW * 0.3, sy + R + coreLen * 0.5, sx, sy + R + coreLen);
  ctx.quadraticCurveTo(sx + coreW * 0.3, sy + R + coreLen * 0.5, sx + coreW / 2, sy + R);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawTrail() {
  if (!orb.trail?.length) return;
  const t = tier();
  const bloom = t.orb.bloom;
  const maxR = orb.radius * .7 * 2;
  const sprite = getGlowSprite(bloom, 16);
  const n = orb.trail.length;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  orb.trail.forEach((pt, i) => {
    if (QF.trailStride > 1 && i % QF.trailStride && i !== n - 1) return;
    const frac = i / n;
    const r = Math.max(1, maxR * frac);
    const a = frac * frac * .5 * orb.glow;
    if (a < 0.005) return;
    const sy = w2s(pt.worldY);
    ctx.globalAlpha = a;
    ctx.drawImage(sprite, pt.x - r, sy - r, r * 2, r * 2);
  });
  ctx.restore();
}

// Per-tier trail flavour — sparse accents layered over the default glow trail
// (paid skins replace the trail entirely; these only season it). Styles live
// in skins-render.js and are accent-driven + gradient-free by design, so they
// never duplicate a shop skin's look.
const TIER_TRAILS = {
  // 0-based tier index → pseudo-skin for drawSkinTrail. Uniform visibility
  // (same length) across tiers; only the style/colours change per tier.
  // GENESIS (0) deliberately has no trail at all.
  1: { accent:"#7ec8ff", trail:{ style:"twinkle",  len:20, c2:"#dff2ff" } }, // ZENITH
  2: { accent:"#2ee6c8", trail:{ style:"rings",    len:20, c2:"#9ff0e0" } }, // ABYSS
  3: { accent:"#6ef07a", trail:{ style:"tendrils", len:20, c2:"#b6ffb0" } }, // VERDANT
  4: { accent:"#ffb55a", trail:{ style:"grains",   len:20, c2:"#ffcf80" } }, // DUNE
  5: { accent:"#ff7e3a", trail:{ style:"sparks",   len:20, c2:"#ffd08a" } }, // MAGMA
  6: { accent:"#ff7ad4", trail:{ style:"shards",   len:20, c2:"#ffe9f7" } }, // ORCHID
  7: { accent:"#b48cff", trail:{ style:"glyphs",   len:20, c2:"#e0d0ff" } }, // VOLT
  8: { accent:"#8a9bff", trail:{ style:"twinkle",  len:20, c2:"#ffffff" } }, // NEBULA
  9: { accent:"#ffd45c", trail:{ style:"streaks",  len:20, c2:"#ffee88" } }, // APEX
};

function drawTierTrail() {
  const sk = TIER_TRAILS[activeTier];
  if (!sk || !orb.trail?.length || typeof drawSkinTrail !== "function") return;
  if (!QF.fx) return;
  // Same adaptive budget as equipped skins; decay while skipping so the
  // flavour comes back once the frame budget recovers.
  if (_skinAvgMs > 6) { _skinAvgMs *= 0.967; return; }
  const _t0 = performance.now();
  const n = orb.trail.length;
  const pts = orb.trail.map((pt, i) => ({
    x: pt.x,
    y: w2s(pt.worldY),
    f: (n - 1 - i) / Math.max(1, n - 1),
  })).reverse();
  drawSkinTrail(ctx, W, H, sk, _frameNow / 1000, pts, orb.radius, isMobile);
  _skinAvgMs += (performance.now() - _t0 - _skinAvgMs) * 0.033;
}

// Adapte les coords réelles du jeu pour drawSkinTrail (Capsule 02)
function drawEquippedSkinTrail(sk) {
  if (!sk?.trail || !orb.trail?.length) return;
  const n = orb.trail.length;
  // pts[0] = tête (orbe actuel, f=0), pts[n-1] = queue (le plus vieux, f=1)
  const pts = orb.trail.map((pt, i) => ({
    x: pt.x,
    y: w2s(pt.worldY),
    f: (n - 1 - i) / Math.max(1, n - 1),
  })).reverse();
  drawSkinTrail(ctx, W, H, sk, _frameNow / 1000, pts, orb.radius, isMobile);
}

// Équipe un skin (depuis shop ou achievement) et persiste
function equipSkin(skinObj) {
  if (skinObj) {
    const isMobile = window.innerWidth < 768;
    equippedSkin = isMobile
      ? { ...skinObj, trail: { ...skinObj.trail, len: Math.ceil((skinObj.trail?.len || 26) * 0.45) } }
      : skinObj;
  } else {
    equippedSkin = null;
  }
  _saveWalletLocal({ equippedSkinId: skinObj?.id || null });
  updateEquipButtons();
}

function unequipSkin() {
  equippedSkin = null;
  _saveWalletLocal({ equippedSkinId: null });
  updateEquipButtons();
}

function updateEquipButtons() {
  document.querySelectorAll(".inv-equip-btn").forEach(btn => {
    const id = btn.dataset.skinId;
    btn.textContent = equippedSkin?.id === id ? "EQUIPPED" : "EQUIP";
    btn.classList.toggle("equipped", equippedSkin?.id === id);
  });
}

function drawOrb() {
  if (!orb.worldY) return;
  // PHOENIX REBIRTH: rapid blink on teleportation
  const blinkAlpha = phoenixBlinkTimer > 0
    ? Math.max(0.05, Math.abs(Math.sin((0.3 - phoenixBlinkTimer) / 0.3 * Math.PI * 6)))
    : 1;
  if (blinkAlpha < 1) { ctx.save(); ctx.globalAlpha = blinkAlpha; }

  // Bespoke skin renderer — same squash/stretch physics as genesis orbs
  if (equippedSkin?.render && typeof SKIN_RENDER !== "undefined" && SKIN_RENDER[equippedSkin.render]) {
    const sx = orb.x, sy = w2s(orb.worldY);
    const sq = orb.squash, rx = orb.radius / Math.sqrt(sq), ry = orb.radius * Math.sqrt(sq);
    const _t0 = performance.now();
    const _lofi = !QF.fx || _skinAvgMs > 6;
    ctx.save();
    ctx.translate(sx, sy); ctx.scale(rx / orb.radius, ry / orb.radius); ctx.translate(-sx, -sy);
    SKIN_RENDER[equippedSkin.render](ctx, sx, sy, orb.radius, equippedSkin, _frameNow / 1000, _lofi);
    _skinAvgMs += (performance.now() - _t0 - _skinAvgMs) * 0.033; // low-pass ~30 frames
    ctx.restore();
    if (blinkAlpha < 1) ctx.restore();
    return;
  }

  const o = tier().orb;
  const sx = orb.x, sy = w2s(orb.worldY);
  const charge = Math.min(1.6, orb.glow);
  const lx = sx-orb.radius*.36, ly = sy-orb.radius*.42;
  const sq = orb.squash, rx = orb.radius/Math.sqrt(sq), ry = orb.radius*Math.sqrt(sq);
  const R = orb.radius;
  const spr = getOrbSprites(activeTier, o, R);
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  const bloomR = R*o.halo*(1+charge*.4);
  ctx.globalAlpha = (.34+charge*.22)/.692; // halo baked at max-charge alphas
  ctx.drawImage(spr.halo, sx-bloomR, sy-bloomR, bloomR*2, bloomR*2);
  ctx.globalAlpha = 1;
  if (o.corona && QF.fx) {
    const pulse=.5+.5*Math.sin(_frameNow*.004);
    const cR=R*(1.55+pulse*.4+charge*.3);
    ctx.globalAlpha=(.18+pulse*.12)/.30; // baked peak alpha is .30
    ctx.drawImage(spr.corona, sx-cR, sy-cR, cR*2, cR*2);
    ctx.globalAlpha=1;
  }
  if (o.flare && QF.fx) {
    const fl=_frameNow*.0006, fh=spr.flareH;
    ctx.globalAlpha=.5+.2*Math.sin(fl*3);
    ctx.translate(sx,sy);
    for (let i=0;i<6;i++) {
      const ang=fl+i*Math.PI/3, len=R*(2.6+Math.sin(fl*2+i)*.6);
      ctx.rotate(ang);
      ctx.drawImage(spr.flare, 0, -fh/2, len, fh);
      ctx.rotate(-ang);
    }
    ctx.translate(-sx,-sy);
    ctx.globalAlpha=1;
  }
  ctx.globalCompositeOperation="source-over";
  ctx.save(); ctx.translate(sx,sy); ctx.scale(rx/R,ry/R); ctx.translate(-sx,-sy);
  const d = spr.size / 2; // baked at 2x — draw at half the sprite's pixel size
  ctx.drawImage(spr.body, sx-d/2, sy-d/2, d, d);
  ctx.globalCompositeOperation="lighter";
  ctx.globalAlpha = .5+charge*.3; // rim baked at alpha 1
  ctx.drawImage(spr.rim, sx-d/2, sy-d/2, d, d);
  ctx.globalAlpha = 1;
  ctx.restore();
  if (o.band) {
    ctx.save(); ctx.beginPath(); ctx.arc(sx,sy,R*.98,0,7); ctx.clip(); ctx.globalCompositeOperation="lighter";
    const by=sy+Math.sin(orb.bandPhase)*R*.5;
    ctx.drawImage(spr.band, sx-R, by-R*.18, R*2, R*.36); ctx.restore();
  }
  ctx.save(); ctx.globalCompositeOperation="lighter";
  const specR = R*.62;
  ctx.drawImage(getSpecSprite(), lx-specR, ly-specR, specR*2, specR*2);
  ctx.restore();
  if (o.sparks>0 && QF.fx) {
    ctx.save(); ctx.globalCompositeOperation="lighter";
    for (let i=0;i<o.sparks;i++) {
      const ang=orb.sparkPhase+i*Math.PI*2/o.sparks, dist=R*(1.7+Math.sin(orb.sparkPhase*1.3+i)*.35);
      const px=sx+Math.cos(ang)*dist, py=sy+Math.sin(ang)*dist*.6, sr=Math.max(1.2,R*.12)*2.5;
      ctx.drawImage(spr.spark, px-sr, py-sr, sr*2, sr*2);
    }
    ctx.restore();
  }
  if (livePressure > 0.4 && activeTier > 0) {
    ctx.save(); ctx.globalCompositeOperation="lighter"; ctx.globalAlpha=Math.min(1,(livePressure-.4)*.8);
    ctx.strokeStyle=o.mid; ctx.lineWidth=gs*2.5; ctx.beginPath(); ctx.arc(sx,sy,R+gs*12+livePressure*gs*8,0,7); ctx.stroke(); ctx.restore();
  }
  ctx.restore();
  if (blinkAlpha < 1) ctx.restore(); // close PHOENIX blink wrapper
}

const _arcVisible = []; // scratch — avoids a per-call array allocation
function drawElectricArcs() {
  if (Math.floor(_frameNow/100) % 4 !== 0) return;

  _arcVisible.length = 0;
  for (const p of platforms) {
    const sy = w2s(p.worldY);
    if (sy > 0 && sy < H) _arcVisible.push(p);
  }
  const visible = _arcVisible;
  if (visible.length < 2) return;

  const orbG = tier().orbG;
  ctx.save();
  ctx.strokeStyle = orbG;
  ctx.globalAlpha = 0.25;
  ctx.lineWidth = 0.8;
  ctx.shadowColor = orbG;
  ctx.shadowBlur = 3;
  for (let i=0; i<visible.length-1; i++) {
    const p1 = visible[i], p2 = visible[i+1];
    const s1y = w2s(p1.worldY), s2y = w2s(p2.worldY);
    const dy = Math.abs(s1y-s2y), dx = Math.abs((p1.x+p1.width/2)-(p2.x+p2.width/2));
    if (dx > 130 || dy > 110 || Math.random() > 0.12) continue;
    const x1=p1.x+p1.width/2, y1=s1y;
    const x2=p2.x+p2.width/2, y2=s2y;
    const mx=(x1+x2)/2+(Math.random()-.5)*22;
    const my=(y1+y2)/2+(Math.random()-.5)*22;
    ctx.beginPath();
    ctx.moveTo(x1,y1); ctx.lineTo(mx,my); ctx.lineTo(x2,y2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawChromatic() {
  ctx.save();
  ctx.globalAlpha = 0.08;
  ctx.globalCompositeOperation = "screen";
  ctx.drawImage(canvas, 2, 0);
  ctx.restore();
}

// ── ULTI VISUAL EFFECTS ──────────────────────────────────────────────────────

function drawCorridorWalls() {
  if (!orb.worldY || orb.trail.length < 2) return;
  const bloom = TIERS[1].orb.bloom;
  const offset = orb.radius * 2.2;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (const side of [-1, 1]) {
    ctx.beginPath();
    orb.trail.forEach((pt, i) => {
      const frac = i / orb.trail.length;
      const sx = pt.x + side * offset;
      const sy = w2s(pt.worldY);
      if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    });
    const g = ctx.createLinearGradient(0, w2s(orb.trail[0].worldY), 0, w2s(orb.trail[orb.trail.length - 1].worldY));
    g.addColorStop(0, hexA(bloom, 0));
    g.addColorStop(1, hexA(bloom, 0.22));
    ctx.strokeStyle = g;
    ctx.lineWidth = 8;
    ctx.stroke();
  }
  ctx.restore();
}

function drawAegisHex() {
  if (!orb.worldY) return;
  const sx = orb.x, sy = w2s(orb.worldY);
  const R = orb.radius * 2.2 + 2 * Math.sin(_frameNow * 0.006);
  const bloom = TIERS[2].orb.bloom;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = hexA(bloom, 0.55);
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2 - Math.PI / 6;
    if (i === 0) ctx.moveTo(sx + Math.cos(a) * R, sy + Math.sin(a) * R);
    else ctx.lineTo(sx + Math.cos(a) * R, sy + Math.sin(a) * R);
  }
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

function drawOverclockAfterImages() {
  const o = tier().orb;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  afterImages.forEach((img, i) => {
    const a = img.alpha * (i / afterImages.length);
    const g = ctx.createRadialGradient(img.x, img.sy, 0, img.x, img.sy, img.r * 1.4);
    g.addColorStop(0, hexA(o.bloom, a * 1.2));
    g.addColorStop(1, hexA(o.bloom, 0));
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(img.x, img.sy, img.r * 1.4, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function drawMagnetRing() {
  if (!orb.worldY) return;
  const sx = orb.x, sy = w2s(orb.worldY);
  const bloom = TIERS[5].orb.bloom;
  const pulse = 0.5 + 0.5 * Math.sin(_frameNow * 0.008);
  const R = orb.radius * (2.8 + pulse * 0.8);
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = hexA(bloom, 0.35 + pulse * 0.2);
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(sx, sy, R, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = hexA(bloom, 0.12);
  ctx.lineWidth = 8;
  ctx.stroke();
  ctx.restore();
}

function drawChronoDarkTint() {
  ctx.save();
  ctx.globalAlpha = 0.18;
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, W, H);
  ctx.restore();
}

function drawPillarBeam() {
  if (pillarFlash <= 0 || !orb.worldY) return;
  const alpha = Math.min(1, Math.pow(pillarFlash / 6, 0.5)); // fades gradually over 6s
  const sy = w2s(orb.worldY);
  const bx = orb.x; // beam follows orb position
  const bloom = TIERS[6].orb.bloom;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  const g = ctx.createLinearGradient(bx, H, bx, sy);
  g.addColorStop(0, hexA(bloom, 0));
  g.addColorStop(0.6, hexA(bloom, alpha * 0.55));
  g.addColorStop(1, hexA(bloom, alpha * 0.22));
  ctx.fillStyle = g;
  ctx.fillRect(bx - 18 * gs, sy, 36 * gs, H - sy);
  ctx.restore();
}

function drawResonanceFlash() {
  if (resonanceFlash <= 0) return;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.globalAlpha = resonanceFlash * 0.55;
  ctx.fillStyle = TIERS[7].orb.bloom;
  ctx.fillRect(0, 0, W, H);
  ctx.restore();
}

function drawResonanceLightning() {
  if (!resonanceLightning.length) return;
  const bloom = TIERS[7].orb.bloom;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = bloom;
  // Double-stroke halo (wide faint + thin bright) instead of shadowBlur
  for (const l of resonanceLightning) {
    ctx.beginPath();
    ctx.moveTo(l.x1, l.y1);
    ctx.lineTo(l.x2, l.y2);
    ctx.globalAlpha = l.life * 0.25;
    ctx.lineWidth = 5;
    ctx.stroke();
    ctx.globalAlpha = l.life * 0.9;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  ctx.restore();
}

function drawBeaconArcs() {
  if (!orb.worldY) return;
  const sx = orb.x, sy = w2s(orb.worldY);
  const bloom = TIERS[1].orb.bloom;
  const pulse = 0.3 + 0.4 * Math.sin(_frameNow * 0.006);
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (const plat of platforms) {
    if (plat.type !== 'buy') continue;
    const psy = w2s(plat.worldY);
    if (psy < 0 || psy > H) continue;
    const px = plat.x + plat.width / 2;
    const dist = Math.hypot(px - sx, psy - sy);
    if (dist > H * 0.55) continue;
    const a = pulse * (1 - dist / (H * 0.55)) * 0.8;
    ctx.strokeStyle = hexA(bloom, a);
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(px, psy);
    ctx.quadraticCurveTo((px + sx) / 2 + (psy - sy) * 0.15, (psy + sy) / 2, sx, sy);
    ctx.stroke();
  }
  ctx.restore();
}

function drawResonanceRing() {
  if (!orb.worldY || !_resonanceState) return;
  const sx = orb.x, sy = w2s(orb.worldY);
  if (sy + orb.radius * 2.6 < 0 || sy - orb.radius * 2.6 > H) return;
  const bloom = TIERS[7].orb.bloom;
  const charge = _resonanceState.counters.bounceCharge || 0;
  const R = orb.radius * 2.6;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  // Pass 1: inactive arcs, faint
  ctx.lineWidth = 1.5;
  for (let i = charge; i < 4; i++) {
    const sa = (i / 4) * Math.PI * 2 - Math.PI * 0.5 + 0.22;
    const ea = sa + Math.PI * 2 / 4 - 0.22;
    ctx.strokeStyle = hexA(bloom, 0.12);
    ctx.beginPath();
    ctx.arc(sx, sy, R, sa, ea);
    ctx.stroke();
  }
  // Pass 2: active arcs — double-stroke halo instead of shadowBlur
  if (charge > 0) {
    ctx.strokeStyle = bloom;
    for (let i = 0; i < charge && i < 4; i++) {
      const sa = (i / 4) * Math.PI * 2 - Math.PI * 0.5 + 0.22;
      const ea = sa + Math.PI * 2 / 4 - 0.22;
      const a = 0.45 + 0.25 * Math.sin(_frameNow * 0.008 + i);
      ctx.beginPath();
      ctx.arc(sx, sy, R, sa, ea);
      ctx.globalAlpha = a * 0.35;
      ctx.lineWidth = 7;
      ctx.stroke();
      ctx.globalAlpha = a;
      ctx.lineWidth = 3;
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawPillarRedRipples() {
  if (!pillarRedRipples.length) return;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (const rip of pillarRedRipples) {
    ctx.strokeStyle = hexA('#ff8fa8', rip.life * 0.55);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(rip.x, rip.y, rip.r, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawHelixArcs() {
  if (!orb.worldY) return;
  const sx = orb.x, sy = w2s(orb.worldY);
  if (sy < -(orb.radius * 11) || sy > H + 10) return;
  const bloom = TIERS[8].orb.bloom;
  const now = _frameNow * 0.003;
  const R = orb.radius * 2.8;
  const len = orb.radius * 9;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (const strand of [0, Math.PI]) {
    ctx.beginPath();
    for (let i = 0; i <= 20; i++) {
      const t2 = i / 20;
      const angle = now + strand + t2 * Math.PI * 2;
      const px = sx + Math.cos(angle) * R;
      const py = sy - t2 * len + Math.sin(angle) * (R * 0.3);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = hexA(bloom, 0.45);
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }
  ctx.restore();
}

function drawPhoenixWings() {
  if (!orb.worldY) return;
  const sx = orb.x, sy = w2s(orb.worldY);
  const bloom = TIERS[9].orb.bloom;
  const beat = 0.5 + 0.5 * Math.sin(_frameNow * 0.006);
  const spread = orb.radius * (4.5 + beat * 1.5);
  const rise = orb.radius * (2.8 + beat * 0.8);
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (const side of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.bezierCurveTo(
      sx + side * spread * 0.5, sy - rise * 0.4,
      sx + side * spread,       sy - rise,
      sx + side * spread * 0.7, sy - rise * 1.6
    );
    ctx.strokeStyle = hexA(bloom, 0.38 + beat * 0.15);
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(sx, sy + orb.radius * 0.5);
    ctx.bezierCurveTo(
      sx + side * spread * 0.4, sy + orb.radius * 0.2,
      sx + side * spread * 0.9, sy - rise * 0.3,
      sx + side * spread * 0.6, sy - rise * 0.9
    );
    ctx.strokeStyle = hexA(bloom, 0.22 + beat * 0.1);
    ctx.lineWidth = 1.8;
    ctx.stroke();
  }
  ctx.restore();
}

function drawPhoenixGhostOrb() {
  if (!_phoenixState || _phoenixState.counters.rebirthUsed) return;
  const sy = w2s(_phoenixState.counters.rebirthY);
  if (sy < -10 || sy > H + 10) return;

  const t9 = TIERS[9].orb;
  const now = _frameNow * 0.001;
  const dist = Math.abs(w2s(orb.worldY) - sy);
  const proximity = Math.max(0, 1 - dist / (H * 0.4));
  const pulse = 0.5 + 0.5 * Math.sin(now * 3);
  const alpha = 0.18 + pulse * 0.10 + proximity * 0.30;
  const R = orb.radius;
  const sx = orb.x;

  ctx.save();
  ctx.globalCompositeOperation = "lighter";

  const glow = ctx.createRadialGradient(sx, sy, 0, sx, sy, R * 2.2);
  glow.addColorStop(0, hexA(t9.bloom, alpha * 0.7));
  glow.addColorStop(1, hexA(t9.bloom, 0));
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(sx, sy, R * 2.2, 0, Math.PI * 2);
  ctx.fill();

  const body = ctx.createRadialGradient(sx - R * 0.3, sy - R * 0.3, R * 0.1, sx, sy, R);
  body.addColorStop(0, hexA(t9.hot, alpha * 0.9));
  body.addColorStop(0.5, hexA(t9.mid, alpha * 0.6));
  body.addColorStop(1, hexA(t9.rim, alpha * 0.2));
  ctx.fillStyle = body;
  ctx.beginPath();
  ctx.arc(sx, sy, R, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function drawScreenFlash() {
  if (screenFlash <= 0) return;
  ctx.save();
  ctx.globalAlpha = _reducedMotion ? screenFlash * 0.35 : screenFlash;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);
  ctx.restore();
}

// ── UI ───────────────────────────────────────────────────────────────────────
let comboTimer = null;
function showCombo() {
  clearTimeout(comboTimer);
  const el = document.getElementById("comboBadge");
  if (!el) return;
  el.textContent = `x${orb.combo} STREAK`;
  el.classList.remove("hidden");
  comboTimer = setTimeout(()=>el.classList.add("hidden"), 1200);
}

function updateMult() {
  const el = document.getElementById("multChip");
  if (!el) return;
  const value = tierWeight() * comboMult();
  el.textContent = `×${value % 1 === 0 ? value : value.toFixed(1)}`;
  el.classList.add("bump");
  setTimeout(()=>el.classList.remove("bump"), 130);
}

// Score line intensity follows orb.combo (tier-agnostic — scoreMult must not
// make the HUD flashy on its own). Class write only when the level changes.
const COMBO_GLOW_CLASSES = ["combo-t1", "combo-t2", "combo-t3", "combo-t4"];
let _comboGlowLevel = -1;
function comboGlowLevel(combo) {
  if (combo >= 24) return 4;
  if (combo >= 15) return 3;
  if (combo >= 9)  return 2;
  if (combo >= 3)  return 1;
  return 0;
}
function updateComboGlow() {
  const level = comboGlowLevel(orb.combo);
  if (level === _comboGlowLevel) return;
  _comboGlowLevel = level;
  const el = document.querySelector(".hud-score");
  if (!el) return;
  el.classList.remove(...COMBO_GLOW_CLASSES);
  if (level > 0) el.classList.add(COMBO_GLOW_CLASSES[level - 1]);
}

// Quantize to whole percent and skip identical writes — these run every
// frame and each style write costs a style recalc.
let _energyBarLast = -1;
function updateEnergyBar() {
  const fill = document.getElementById("energyFill");
  if (!fill) return;
  const pct = Math.round(orb.energy);
  if (pct === _energyBarLast) return;
  _energyBarLast = pct;
  fill.style.height = `${pct}%`;
  fill.style.background = pct > 60 ? "#16e0a3" : pct > 30 ? "#ffcf5c" : "#ff4d6d";
}

let _reserveBarLast = -1;
function updateReserveBar() {
  const wrap = document.getElementById('reserveWrap');
  const fill = document.getElementById('reserveFill');
  if (!wrap || !fill) return;
  if (orb.reserve <= 0) orb.reserve = 0;
  const pct = Math.round(orb.reserve);
  if (pct === _reserveBarLast) return;
  _reserveBarLast = pct;
  if (pct <= 0) {
    wrap.style.display = 'none';
  } else {
    wrap.style.display = 'flex';
    fill.style.height = `${pct}%`;
  }
}

function spawnMagnetParticle() {
  const edge = Math.floor(Math.random() * 4);
  let x, y;
  if (edge === 0)      { x = Math.random() * W; y = -20; }
  else if (edge === 1) { x = W + 20; y = Math.random() * H; }
  else if (edge === 2) { x = Math.random() * W; y = H + 20; }
  else                 { x = -20; y = Math.random() * H; }
  magnetParticles.push({ x, y, px: x, py: y, phase: Math.random() * Math.PI * 2, size: 7 + Math.random() * 4, speed: 190 + Math.random() * 80 });
}

function updateMagnetParticles(dt) {
  if (magnetParticles.length === 0) return;
  const tx = orb.x;
  const ty = w2s(orb.worldY);
  for (let i = magnetParticles.length - 1; i >= 0; i--) {
    const p = magnetParticles[i];
    const dx = tx - p.x, dy = ty - p.y;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
    const step = p.speed * dt;
    p.px = p.x; p.py = p.y;
    if (dist <= step + orb.radius) {
      burst(p.x, p.y, '#ffdd88', 10, { spread: 80, up: 40 });
      orb.energy  = Math.min(100, orb.energy  + 7);
      orb.reserve = Math.min(100, orb.reserve + 7);
      updateEnergyBar();
      updateReserveBar();
      magnetParticles.splice(i, 1);
    } else {
      p.x += (dx / dist) * step;
      p.y += (dy / dist) * step;
    }
  }
}

function drawMagnetParticles() {
  if (magnetParticles.length === 0) return;
  const now = _frameNow;
  for (const p of magnetParticles) {
    const r = p.size;
    // Trail
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = 0.28;
    ctx.strokeStyle = '#ffdd88';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(p.px, p.py);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    ctx.restore();
    // Glow halo
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 2.4);
    g.addColorStop(0, 'rgba(255,221,136,0.50)');
    g.addColorStop(1, 'rgba(255,221,136,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r * 2.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    // Spinning diamond
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(now * 0.003 + p.phase);
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = '#ffcf5c';
    ctx.beginPath();
    ctx.moveTo(0, -r); ctx.lineTo(r * 0.6, 0); ctx.lineTo(0, r); ctx.lineTo(-r * 0.6, 0);
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#fff8e0';
    ctx.beginPath();
    ctx.moveTo(0, -r * 0.48); ctx.lineTo(r * 0.29, 0); ctx.lineTo(0, r * 0.48); ctx.lineTo(-r * 0.29, 0);
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }
}

let impactTimer = null;
function showTradeImpact(impact, strength) {
  clearTimeout(impactTimer);
  const el = document.getElementById("impactFlash");
  const lbl = document.getElementById("impactLabel");
  const val = document.getElementById("impactValue");
  if (!el||!lbl||!val) return;
  el.classList.remove("hidden","sell","shop");
  if (impact.source==="shop") {
    lbl.textContent = "SHOP";
    val.textContent = impact.item ? `${impact.item} • ${impact.rarity}` : `${impact.rarity} SKIN`;
    el.classList.add("shop");
  } else {
    lbl.textContent = impact.side.toUpperCase();
    val.textContent = `${Number(impact.amount).toLocaleString("en-US",{maximumFractionDigits:2})} ${impact.unit}`;
    if (impact.side==="sell") el.classList.add("sell");
  }
  el.style.opacity = strength ? "1" : ".55";
  impactTimer = setTimeout(()=>{ el.classList.add("hidden"); el.style.opacity=""; }, 1300);
}

function enqueueImpact(impact) {
  if (state === "paused") return;
  if (!impact || impact.market!==currentMarket || !["buy","sell"].includes(impact.side)) return;
  if (!Number.isFinite(impact.timestamp) || impact.timestamp <= 0) return;
  let strength;
  if (impact.source==="shop") {
    const equivalent = shopImpactXrdEquivalent(impact.rarity);
    if (!equivalent) return;
    strength = impactStrength(equivalent, gameplayConfig.market.thresholds);
  } else {
    if (!Number.isFinite(impact.amount) || impact.amount <= 0 || impact.amount > MAX_IMPACT_AMOUNT) return;
    strength = impactStrength(impact.amount, gameplayConfig.market.thresholds);
  }
  const queued = { ...impact, strength };
  if (impactQueue.length < gameplayConfig.impacts.queueCapacity) impactQueue.push(queued);
  else {
    impactQueue.sort((a,b)=>a.strength-b.strength);
    if (strength > impactQueue[0].strength) impactQueue[0] = queued;
  }
}

window.addEventListener("message", event => {
  if (!PREVIEW_TOKEN || event.data?.type!=="trade-impact") return;
  enqueueImpact({ ...event.data.impact, market: currentMarket });
});

function processImpactQueue() {
  const now = performance.now();
  if (!impactQueue.length || now-lastImpactAppliedAt < sessionGameplay.impacts.queueSpacingMs) return;
  impactQueue.sort((a,b)=>a.timestamp-b.timestamp);
  const impact = impactQueue.shift();
  lastImpactAppliedAt = now;
  if (activeAnomaly) {
    if (impact.side === "buy" && impact.strength) {
      deferredAnomalyBuyStrength = Math.min(6, deferredAnomalyBuyStrength + impact.strength);
      deferredAnomalyBuyCount = Math.min(20, deferredAnomalyBuyCount + 1);
      if (impact.source==="shop" || shouldSpawnNotableBuyPlatform(impact.amount,sessionGameplay.market.thresholds[0])) {
        spawnNotableBuyPlatform(impact.strength);
      }
    }
    return;
  }
  showTradeImpact(impact,impact.strength);
  if (!impact.strength) return;

  const impactMods = getUltiModifiers();
  if (impact.side==="buy") {
    const impulse = sessionGameplay.impacts.buyImpulse * Math.min(4,impact.strength)/3;
    if (impact.source==="shop" || shouldSpawnNotableBuyPlatform(impact.amount,sessionGameplay.market.thresholds[0])) spawnNotableBuyPlatform(impact.strength);
    const bx = orb.x, by = w2s(orb.worldY);
    if (impact.strength > 1) {
      const rocketPower = Math.min(1.0, (impact.strength - 1) / 3);
      const boostedImpulse = impulse * (1.0 + rocketPower * 0.8);
      orb.vy = Math.min(MAX_ORB_VY + rocketPower * 700, orb.vy + boostedImpulse * (1 + (impactMods.impactAmp - 1) * 0.5));
      burst(bx, by, tier().platBuy,
        Math.ceil(10 + rocketPower * 30),
        { up: 110 + rocketPower * 100, spread: 90 + rocketPower * 70, size: 3 + rocketPower * 3 }
      );
      buyChartFlash         = 0.7 + rocketPower * 0.3;
      rocketModePeak        = 0.25 + rocketPower * 0.75;
      rocketSustainDuration = rocketPower * 3.0;
      rocketDecayDuration   = 0.5 + rocketPower * 1.35;
      rocketModeTotalDuration = ROCKET_ATTACK_DUR + rocketSustainDuration + rocketDecayDuration;
      rocketModeDuration    = rocketModeTotalDuration;
      armBuyChartGlow(rocketModeTotalDuration, buyChartFlash);
      rocketFlameCyan       = false;
      rocketPhase           = 'attack';
      rocketPhaseTimer      = 0;
      rocketMode            = 0;
      shockwaves.push({ x: bx, y: by, r: 0, life: 0.5 + rocketPower * 0.5 });
      shakeI = Math.max(shakeI,1.2 + rocketPower * 3.4);
      if (rocketPower > 0.5) setTimeout(() => shockwaves.push({ x: bx, y: by, r: 0, life: rocketPower * 0.7 }), 120);
      if (vibrationEnabled && navigator.vibrate) {
        if (rocketPower < 0.33) navigator.vibrate([20, 20, 40]);
        else if (rocketPower < 0.67) navigator.vibrate([30, 20, 55, 20, 90]);
        else navigator.vibrate([40, 25, 70, 25, 130]);
      }
      sfx.rocket();
    } else {
      orb.vy = Math.min(MAX_ORB_VY, orb.vy + impulse * (1 + (impactMods.impactAmp - 1) * 0.5));
      burst(bx, by, tier().platBuy, Math.ceil(8 + impact.strength * 4), { up:100, spread:90, size:3 });
      shockwaves.push({ x:bx, y:by, r:0, life:.55 });
      shakeI = Math.max(shakeI,1.2);
      buyChartFlash = 0.6;
      armBuyChartGlow(buyChartFlash / 0.6, buyChartFlash);
      sfx.surge();
    }
    return;
  }

  if (impactMods.noSellDebuff) return; // PHOENIX immunity — sell wave ignored
  sellChartFlash = Math.min(1, 0.5 + impact.strength * 0.18);
  const debuff = applySellDebuff({
    velocity:orb.vy,
    currentStrength:sellDebuffStrength,
    currentUntil:sellDebuffUntil,
    now,
    impactStrength:impact.strength,
    velocityPct:sessionGameplay.impacts.sellVelocityPct * impactMods.impactAmp,
    durationMs:sessionGameplay.impacts.sellDurationMs,
    maxDurationMs:sessionGameplay.impacts.sellMaxDurationMs
  });
  orb.vy = debuff.velocity;
  sellDebuffStrength = debuff.strength;
  sellDebuffUntil = debuff.until;
  sfx.sell();
}

// ── ULTI SYSTEM ──────────────────────────────────────────────────────────────
function ultiSlotCount(t) {
  if (t < 1) return 0;
  if (t <= 2) return 1;
  if (t <= 4) return 2;
  return 3;
}

function unlockedUltis(t) {
  const out = [];
  for (let i = 1; i <= t && i < TIERS.length; i++) {
    if (TIERS[i]?.ulti) out.push(i);
  }
  return out;
}

function buildUltiState(t) {
  switch (t) {
    case 1: return { tier:t, phase:'active', remaining:6,  counters:{} };                    // BEACON
    case 2: return { tier:t, phase:'active', remaining:8,  counters:{shieldUsed:false} };    // AEGIS
    case 3: return { tier:t, phase:'active', remaining:5,  counters:{} };                    // OVERCLOCK
    case 4: return { tier:t, phase:'active', remaining:7,  counters:{} };                    // CHRONO
    case 5: return { tier:t, phase:'active', remaining:8,  counters:{spawnTimer:0} };        // MAGNET
    case 6: return { tier:t, phase:'active', remaining:6,  counters:{} };                    // PILLAR
    case 7: return { tier:t, phase:'active', remaining:10, counters:{bounceCharge:0} };      // RESONANCE
    case 8: return { tier:t, phase:'active', remaining:10, counters:{helixTimer:2.5, helixOscTime:0} }; // HELIX
    case 9: return { tier:t, phase:'active', remaining:12, counters:{rebirthY:orb.worldY, rebirthUsed:false} }; // PHOENIX REBIRTH
    default: return null;
  }
}

function applyUltiImmediate(ultiTier) {
  const sy = w2s(orb.worldY);
  const margin = 22;
  switch (ultiTier) {
    case 1: { // INVOKE: radial aurora burst — spawn buy platforms, magnetic attraction starts via applyStateToMods
      burst(orb.x, sy, TIERS[1].orb.bloom, 28, { up: 110, spread: 120, size:4 });
      for (let i=0; i<3; i++) shockwaves.push({x:orb.x, y:sy, r:i*20, life:1-i*.12});
      shakeI = Math.max(shakeI, 2);
      sfx.ulti_invoke();
      const bw = gs * 88;
      const zones = [margin, W/2 - bw/2, W - bw - margin];
      for (let i = 0; i < 3; i++) {
        platforms.push({
          worldY: orb.worldY + 90 + i * 95 + Math.random() * 30,
          x: Math.max(margin, Math.min(W - bw - margin, zones[i] + (Math.random() - 0.5) * 20)),
          width: bw,
          bounce: 1.2 + Math.random() * 0.2,
          type: 'buy',
          age: 0,
          seed: Math.random(),
        });
      }
      break;
    }
    case 2: // AEGIS: shield glow burst only — rescue is passive in physics loop
      burst(orb.x, sy, TIERS[2].orb.bloom, 22, { up: 80, spread: 90 });
      sfx.ulti_aegis();
      break;
    case 3: // OVERCLOCK: impulse + rings
      orb.vy = Math.min(MAX_ORB_VY, orb.vy + 350);
      for (let i = 0; i < 3; i++) shockwaves.push({ x: orb.x, y: sy, r: i * 20, life: 1 - i * 0.10 });
      burst(orb.x, sy, TIERS[3].orb.bloom, 24, { up: 100, spread: 100, size: 4 });
      sfx.ulti_overclock();
      break;
    case 4: // CHRONO: ripple rings — audio/time slowdown handled by chronoRamp
      orb.vy = Math.min(MAX_ORB_VY,orb.vy + 240);
      for (let i = 0; i < 3; i++) shockwaves.push({ x: orb.x, y: sy, r: i * 22, life: 1 - i * 0.08 });
      burst(orb.x, sy, TIERS[4].orb.bloom, 18, { up: 60, spread: 100 });
      sfx.ulti_chrono();
      break;
    case 5: { // MAGNET: collect all beneficial on-screen boosters instantly + start particle attraction
      const cfg = sessionGameplay.boosters;
      for (const b of boosters) {
        if (b.used) continue;
        const bsy = w2s(b.worldY);
        if (bsy < -50 || bsy > H + 50) continue;
        if (b.type === 'drag') continue;
        b.used = true;
        if (b.type === 'surge') {
          orb.vy     = Math.min(MAX_ORB_VY, orb.vy + cfg.surge.boost);
          orb.energy = Math.min(100, orb.energy + cfg.surge.energy);
          orb.reserve = Math.min(100, orb.reserve + cfg.surge.energy);
          orb.bonus = Math.min(MAX_ORB_BONUS, orb.bonus + cfg.surge.score);
        } else if (b.type === 'stream') {
          orb.vy = Math.min(MAX_ORB_VY,streamAscentVelocity(orb.vy,cfg.stream,0,{entered:true}));
          orb.energy = Math.min(100, orb.energy + cfg.stream.energyPerSecond * 2);
          orb.reserve = Math.min(100, orb.reserve + cfg.stream.energyPerSecond * 2);
          orb.bonus = Math.min(MAX_ORB_BONUS, orb.bonus + cfg.stream.scorePerSecond * 2);
        }
        burst(b.x, bsy, '#ffdd88', 6, { up: 50, spread: 40 });
      }
      burst(orb.x, sy, TIERS[5].orb.bloom, 24, { up: 80, spread: 120 });
      magnetParticles = [];
      updateReserveBar();
      sfx.ulti_magnet();
      break;
    }
    case 6: // PILLAR: huge impulse + rings + persistent beam (6s)
      orb.vy = Math.min(MAX_ORB_VY, orb.vy + 720);
      for (let i = 0; i < 6; i++) shockwaves.push({ x: orb.x, y: sy, r: i * 20, life: 1 - i * 0.07 });
      burst(orb.x, sy, TIERS[6].orb.bloom, 30, { up: 120, spread: 100, size: 4 });
      shakeI = Math.max(shakeI, 4);
      pillarFlash = 6.0;
      sfx.rocket();
      break;
    case 7: { // RESONANCE: initial electric burst — charge mechanic handled in updatePlatformCollisions
      burst(orb.x, sy, TIERS[7].orb.bloom, 30, { up: 90, spread: 120, size: 4 });
      for (let i = 0; i < 3; i++) shockwaves.push({ x: orb.x, y: sy, r: i * 22, life: 1 - i * 0.10 });
      sfx.ulti_resonance();
      break;
    }
    case 8: // HELIX: opening lift and burst
      orb.vy = Math.min(MAX_ORB_VY,orb.vy + 300);
      burst(orb.x, sy, TIERS[8].orb.bloom, 34, { up: 110, spread: 130, size:4 });
      for (let i=0;i<3;i++) shockwaves.push({x:orb.x,y:sy,r:i*20,life:1-i*.1});
      sfx.ulti_helix();
      break;
    case 9: // PHOENIX REBIRTH: screen flash + rings — rebirth height memorised in buildUltiState
      screenFlash = 0.45;
      for (let i = 0; i < 6; i++) shockwaves.push({ x: orb.x, y: sy, r: i * 18, life: 1 - i * 0.06 });
      burst(orb.x, sy, TIERS[9].orb.hot, 40, { up: 120, spread: 140, size: 5 });
      sfx.ulti_phoenix();
      break;
  }
}

function tryUlti(slotIndex) {
  if (state !== 'playing') return;
  if (slotIndex < 0 || slotIndex >= equippedUltis.length) return;
  if (ultiCharges[slotIndex] < 100) return;
  if (!canActivateUltiSlot(ultiActiveStates[slotIndex])) return;
  const ultiTier = equippedUltis[slotIndex];
  ultiCharges[slotIndex] = 0;
  ultiActiveStates[slotIndex] = buildUltiState(ultiTier);
  applyUltiImmediate(ultiTier);
  updateUltiHud();
}

function updateUltis(dt) {
  let changed = false;
  for (let si = 0; si < ultiCharges.length; si++) {
    const cooldown = TIERS[equippedUltis[si]].ulti.cooldownSeconds;
    const nextCharge = advanceUltiCharge(ultiCharges[si], dt, cooldown);
    if (nextCharge !== ultiCharges[si]) { ultiCharges[si] = nextCharge; changed = true; }
  }
  for (let si = 0; si < ultiActiveStates.length; si++) {
    const s = ultiActiveStates[si];
    if (!s) continue;

    // HELIX: periodic +240 vy impulse every 1.5 seconds + oscillation timer
    if (s.tier === 8 && s.phase === 'active') {
      if (tickHelixTimer(s.counters, dt)) {
        orb.vy = Math.min(MAX_ORB_VY, orb.vy + 120);
      }
      s.counters.helixOscTime += dt;
    }

    // MAGNET: spawn one attraction particle every 0.55s
    if (s.tier === 5 && s.phase === 'active') {
      s.counters.spawnTimer = (s.counters.spawnTimer || 0) + dt;
      if (s.counters.spawnTimer >= 0.55) {
        s.counters.spawnTimer -= 0.55;
        spawnMagnetParticle();
      }
    }

    // PHOENIX REBIRTH: track peak height, teleport back if orb falls below it
    if (s.tier === 9 && s.phase === 'active' && !s.counters.rebirthUsed && orb.worldY) {
      s.counters.rebirthY = Math.max(s.counters.rebirthY, orb.worldY);
      const elapsed = 12 - s.remaining;
      if (elapsed > 0.1 && orb.worldY < s.counters.rebirthY - orb.radius * 1) {
        s.counters.rebirthUsed = true;
        sfx.phoenix_rebirth();
        orb.worldY = s.counters.rebirthY;
        orb.vy = Math.min(MAX_ORB_VY, 900);
        phoenixBlinkTimer = 0.3;
        screenFlash = Math.max(screenFlash, 0.45);
        const rsy = w2s(orb.worldY);
        burst(orb.x, rsy, TIERS[9].orb.hot, 40, { up: 120, spread: 140, size: 5 });
        for (let ri=0; ri<4; ri++) shockwaves.push({x:orb.x, y:rsy, r:ri*18, life:1-ri*0.08});
        shakeI = Math.max(shakeI, 3);
      }
    }

    s.remaining -= dt;
    if (s.remaining > 0) continue;

    if (s.tier === 5) magnetParticles = []; // MAGNET ended — clear attraction particles
    ultiActiveStates[si] = null;
    changed = true;
  }
  if (changed) updateUltiHud();

  // CHRONO bullet-time ramp (attack 0.65s, release 0.45s)
  const chronoActive = ultiActiveStates.some(s => s?.tier === 4 && s?.phase === 'active');
  if (chronoActive) {
    chronoRamp = Math.min(1, chronoRamp + dt / 0.65);
  } else if (chronoRamp > 0) {
    chronoRamp = Math.max(0, chronoRamp - dt / 0.45);
    if (chronoRamp === 0) syncMusicRate();
  }
  if (chronoRamp > 0) applyChronoAudio(chronoRamp);
}

function applyChronoAudio(ramp) {
  const rate  = 1 - 0.42 * ramp;  // 1.0 → 0.58
  const pitch = 1 - 0.28 * ramp;  // 1.0 → 0.72
  if (soundTouchNode) {
    soundTouchNode.playbackRate.value = rate;
    soundTouchNode.pitch.value = pitch;
  } else if (musicEngineState !== 'idle') {
    bgMusic.playbackRate = rate;
  }
}

function updateUltiHud() {
  const container = document.getElementById('ultiButtons');
  if (!container) return;
  if (equippedUltis.length === 0) { container.classList.add('hidden'); return; }
  container.classList.remove('hidden');

  const CIRC = 2 * Math.PI * 19; // stroke circumference for r=19
  const anyReady = equippedUltis.some((_,si) => ultiCharges[si] >= 100 && !ultiActiveStates[si]);
  container.dataset.ready = anyReady ? '1' : '';

  container.innerHTML = equippedUltis.map((tIdx, si) => {
    const u     = TIERS[tIdx].ulti;
    const color = TIERS[tIdx].orb.bloom;
    const charge = ultiCharges[si] || 0;
    const s     = ultiActiveStates[si];
    const ready   = charge >= 100 && !s;
    const active  = s?.phase === 'active';
    const penalty = s?.phase === 'penalty';
    const cls = [ready?'ready':'', active?'active':'', penalty?'penalty':''].filter(Boolean).join(' ');
    const arc = CIRC * (charge / 100);
    const key = ultiKeys[si] || '';
    // First word for multi-word names, else truncate at 6 chars
    const label = u.name.includes(' ') ? u.name.split(' ')[0] : (u.name.length > 6 ? u.name.slice(0,6) : u.name);
    const pctLabel = active ? 'ACTIVE' : penalty ? 'PENALTY' : ready ? 'READY' : `${ultiCooldownRemaining(charge,u.cooldownSeconds)}s`;
    return `<div class="ulti-slot ${cls}" data-slot="${si}" style="--u-c:${color}">
      <svg class="ulti-ring" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg">
        <circle class="ulti-ring-bg" cx="22" cy="22" r="19" fill="rgba(0,0,0,.75)" stroke="rgba(255,255,255,.15)" stroke-width="3.5"/>
        <circle class="ulti-ring-arc" cx="22" cy="22" r="19" fill="none"
          stroke="${color}" stroke-width="3.5" stroke-linecap="round"
          stroke-dasharray="${CIRC.toFixed(1)}"
          stroke-dashoffset="${(CIRC - arc).toFixed(1)}"
          transform="rotate(-90 22 22)"
          opacity="${active||penalty?0.6:1}"/>
      </svg>
      <div class="ulti-slot-inner">
        <div class="ulti-slot-name">${label}</div>
        <div class="ulti-slot-pct">${pctLabel}</div>
        <div class="ulti-slot-key">${key}</div>
      </div>
    </div>`;
  }).join('');
}

// ── ULTI SELECTION SCREEN ────────────────────────────────────────────────────
function ultiSlotLabel(n) { return `PICK ${n} ULTIMATE${n > 1 ? 'S' : ''}`; }

function showUltiSelectScreen() {
  const slots = ultiSlotCount(activeTier);
  const pool  = unlockedUltis(activeTier);
  selectedUltiTiers = [];

  // Auto-equip when no real choice
  if (pool.length <= slots) {
    equippedUltis    = [...pool];
    ultiCharges      = new Array(equippedUltis.length).fill(0);
    ultiActiveStates = new Array(equippedUltis.length).fill(null);
    startGame();
    return;
  }

  document.getElementById('ultiSelectTitle').textContent = ultiSlotLabel(slots);
  const grid = document.getElementById('ultiSelectGrid');
  grid.innerHTML = '';
  // 4 cards → 2×2, all others → 3 cols
  grid.style.gridTemplateColumns = pool.length === 4 ? 'repeat(2,1fr)' : 'repeat(3,1fr)';
  pool.forEach(tIdx => {
    const u   = TIERS[tIdx].ulti;
    const col = TIERS[tIdx].orb.bloom;
    const card = document.createElement('div');
    card.className = 'ulti-card';
    card.dataset.tier = tIdx;
    card.style.setProperty('--u-color', col);
    card.innerHTML = `<div class="uc-name">${u.name}</div>`;
    card.addEventListener('click', () => {
      const t  = Number(card.dataset.tier);
      const idx = selectedUltiTiers.indexOf(t);
      if (idx >= 0) {
        selectedUltiTiers.splice(idx, 1);
        card.classList.remove('selected');
      } else if (selectedUltiTiers.length < slots) {
        selectedUltiTiers.push(t);
        card.classList.add('selected');
      }
      document.getElementById('ultiSelectConfirm').disabled = selectedUltiTiers.length !== slots;
    });
    grid.appendChild(card);
  });

  document.getElementById('ultiSelectConfirm').disabled = true;
  document.getElementById('startOverlay')?.classList.add('hidden');
  document.getElementById('finishOverlay')?.classList.add('hidden');
  document.getElementById('ultiSelectScreen').classList.remove('hidden');
  updateLeaderboardButton();
}

function confirmUltiSelection() {
  equippedUltis    = [...selectedUltiTiers];
  ultiCharges      = new Array(equippedUltis.length).fill(0);
  ultiActiveStates = new Array(equippedUltis.length).fill(null);
  document.getElementById('ultiSelectScreen').classList.add('hidden');
  startGame();
}

function requestStartGame() {
  if (state !== "playing" && state !== "paused") enforceUnlockedTierAfterRun();
  currentPlayerName();
  closeInfoOverlays({ resume:false });
  if (activeTier < 1) { startGame(); return; }
  if (state === "playing" || state === "paused") {
    state = "ready";
    pauseStartedAt = 0;
    bgMusic.pause();
    updateLeaderboardButton();
    updatePauseUI();
  }
  showUltiSelectScreen();
}

function showStartScreen() {
  selectedUltiTiers = [];
  closeInfoOverlays({ resume:false });
  document.getElementById("ultiSelectScreen")?.classList.add("hidden");
  document.getElementById("finishOverlay")?.classList.add("hidden");
  document.getElementById("startOverlay")?.classList.remove("hidden");
  updateLeaderboardButton();
  updatePauseUI();
}

// ── GAME FLOW ────────────────────────────────────────────────────────────────
function drawFirstPlayHint() {
  if (state !== "playing") return;
  const elapsed = (performance.now() - gameStartTime) / 1000;
  if (elapsed > 4) return;
  const alpha = Math.max(0, 1 - (elapsed - 2.5) / 1.5);
  if (alpha <= 0) return;
  const hint = isMobile ? "SWIPE TO MOVE  ·  TAP TO JUMP" : "← → / STICK MOVE  ·  SPACE / TAP / A = JUMP";
  const hintY = H - BOTTOM_BAR_H - (isMobile ? Math.max(190, gs * 140) : Math.max(96, gs * 70));
  ctx.save();
  ctx.font = `${Math.round(gs * 11)}px Silkscreen, monospace`;
  ctx.textAlign = "center";
  ctx.globalAlpha = alpha * (isMobile ? 0.95 : 0.85);
  ctx.fillStyle = isMobile ? "#bfeede" : "#8fb6aa";
  ctx.fillText(hint, W / 2, hintY);
  ctx.restore();
}

function startGame() {
  if (storedWalletState().status !== "connected") {
    const _d = new Date().toISOString().slice(0, 10), _k = "asc_play_" + _d;
    if (!localStorage.getItem(_k)) {
      fetch("/track", { method: "POST", body: JSON.stringify({ type: "play" }), headers: { "Content-Type": "application/json" } }).catch(() => {});
      try { localStorage.setItem(_k, "1"); } catch {}
    }
  }
  currentPlayerName();
  closeInfoOverlays({ resume:false });
  resize(); // always re-measure canvas before seeding world positions
  _sceneQ = _qLevel; // freeze the painted-scene quality for this run
  runTier = activeTier;
  pendingAllTimeEntry = null; // clear any leftover optimistic finish-screen entries
  pendingContestEntry = null;
  gameStartTime = performance.now();
  cameraY = 0;
  _streamPrevCamY = 0; // avoid a one-frame stream-flow spike from the camera reset
  particles = [];
  shockwaves = [];
  shakeI = 0;
  pillarFlash = 0; resonanceFlash = 0; screenFlash = 0; phoenixBlinkTimer = 0;
  pillarRedRipples = []; afterImages = []; resonanceLightning = [];
  clearAnomaly();
  impactQueue = [];
  impactPollMs = IMPACT_POLL_MIN_MS;   // a new run should react to trades immediately
  sellDebuffUntil = 0;
  sellDebuffStrength = 0;
  ultiCharges      = new Array(equippedUltis.length).fill(0);
  ultiActiveStates = new Array(equippedUltis.length).fill(null);
  updateUltiHud();
  sessionGameplay = structuredClone(gameplayConfig);
  sessionTrend = trendStrength(marketTrendPct,sessionGameplay.trend);
  seedPlatforms();
  resetOrb();
  magnetParticles = [];
  rocketPhase = 'off';
  buyChartFlash = 0;
  armBuyChartGlow(0,0);
  sellChartFlash = 0;
  chronoRamp = 0;
  updateReserveBar();
  seedBoosters();
  resetAutoAnomalySchedule();
  state = "playing";
  updateLeaderboardButton();
  updatePauseUI();
  syncMusic();
  document.getElementById("startOverlay")?.classList.add("hidden");
  document.getElementById("finishOverlay")?.classList.add("hidden");
  document.getElementById("comboBadge")?.classList.add("hidden");
  document.getElementById("heightScore").textContent = "000000";
  updateComboGlow();
  document.getElementById("multChip").textContent = `×${tier().feats.scoreMult}`;
}

function gameOver() {
  if (state !== "playing") return;
  // Capture an active x2 before clearAnomaly wipes it; secureStreak zeros the reservoir so
  // currentScore (read just below) doesn't double-count the streak it just banked.
  secureStreak();
  orb.combo = 0;
  clearAnomaly();
  state = "crashed";
  updateLeaderboardButton();
  updatePauseUI();
  bgMusic.pause();
  sfx.death();

  updateComboGlow();
  const scoreVal = finalScoreAfterBank(orb.maxScore, currentScore());
  const wasBest = marketLeaderboards().status==="ready" && scoreVal > best;
  lbSave(scoreVal);

  // Show the just-finished run in the finish-screen leaderboard immediately, before the
  // async server data arrives. Reconciled in lbSave/loadTemporaryLeaderboards.
  const lbState = storedWalletState();
  const runName = currentPlayerName();
  pendingAllTimeEntry = { tier: runTier, name: runName, score: scoreVal };
  // Contest mirrors the server's write conditions (worker/index.js:1742,1747,1749):
  // connected wallet + contest unlocked + run on the contest tier.
  const contestEligible =
    lbState.status === "connected" &&
    !contestConfig.contestLocked &&
    runTier === (contestConfig.contestTier || 0);
  pendingContestEntry = contestEligible
    ? { tier: runTier, name: runName, score: scoreVal,
        identityAddress: lbState.identityAddress || "",
        accountAddress: lbState.accountAddress || "" }
    : null;

  document.getElementById("finishRankStat")?.classList.add("hidden");
  document.getElementById("finalHeight").textContent = String(scoreVal).padStart(6,"0");
  document.getElementById("finalCombo").textContent  = `x${orb.maxCombo}`;
  document.getElementById("finishTitle").textContent  = wasBest ? "NEW PEAK!" : "BACK TO EARTH.";
  document.getElementById("finishKicker").textContent = wasBest ? "RECORD" : "FELL";
  if (storedWalletState().status === "connected" && (playerBestScore == null || scoreVal > playerBestScore)) {
    playerBestScore = scoreVal;
    playerBestScoreLoaded = true;
  }
  updateBestScoreHud();
  lbRender();
  if (!temporaryLeaderboardState.has(currentMarket) || contestEligible) loadTemporaryLeaderboards();
  document.getElementById("finishOverlay")?.classList.remove("hidden");
  if (pendingTierRelockIndex !== null) {
    setTimeout(() => {
      if (state === "crashed") enforceUnlockedTierAfterRun();
    }, 800);
  }
}

// ── GAME LOOP ────────────────────────────────────────────────────────────────
// Use setInterval as primary loop — rAF is throttled by Brave shields/power saving
// and doesn't fire reliably. setInterval always fires when the tab is open.
let _tickN = 0;
function startLoop() {
  // Primary: setInterval at ~60fps (immune to tab throttling)
  setInterval(() => {
    if (document.hidden) return; // dt clamp below absorbs the gap on return
    const now = performance.now();
    const dt = Math.min(.04, (now - last) / 1000);
    last = now;
    update(dt);
    _tickN++;
    if (state !== "playing" && (_tickN & 1)) return; // 30fps under menus/overlays
    draw();
    updateAdaptiveQuality(now);
  }, 16);
}

function updateAdaptiveQuality(frameStart) {
  _frameAvgMs += (performance.now() - frameStart - _frameAvgMs) * 0.05;
  if (_qDprDirty && state !== "playing") { _qDprDirty = false; resize(); }
  if (qualityMode !== "auto") return;
  if (frameStart < _qCooldownUntil) { _qHotFrames = 0; _qCoolMs = 0; return; }
  if (_frameAvgMs > 14) {
    _qCoolMs = 0;
    if (++_qHotFrames >= 90 && _qLevel > 0) {
      applyQualityLevel(_qLevel - 1);
      _qHotFrames = 0;
      _qCooldownUntil = frameStart + 3000;
    }
  } else {
    _qHotFrames = 0;
    if (_frameAvgMs < 7.5) {
      _qCoolMs += 16;
      if (_qCoolMs >= 10000 && _qLevel < _qMaxLevel()) {
        applyQualityLevel(_qLevel + 1);
        _qCoolMs = 0;
        _qCooldownUntil = frameStart + 3000;
      }
    } else _qCoolMs = 0;
  }
}

// ── RESIZE ───────────────────────────────────────────────────────────────────
function resize() {
  cachedCanvasRect = null;
  dpr = Math.min(QF.dprCap, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const rw = rect.width  || canvas.clientWidth  || window.innerWidth;
  const rh = rect.height || canvas.clientHeight || window.innerHeight;
  if (rw < 10 || rh < 10) { requestAnimationFrame(resize); return; }
  const oldW = W;
  const oldH = H;
  const oldGs = gs;
  const orbScreenY = orb.worldY && oldH >= 10
    ? oldH - (orb.worldY - cameraY)
    : null;

  W = Math.round(rw);
  H = Math.round(rh);
  canvas.width  = Math.round(W*dpr);
  canvas.height = Math.round(H*dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  gs     = Math.max(0.5, Math.min(3, H / 700));
  PLAT_H = Math.max(10, Math.round(gs * 13));
  ORB_R  = Math.max(18, Math.round(gs * 22));
  ORB_SPEED_EFF = Math.min(ORB_SPEED * (W / 420) * (1 + 0.15 * narrowScreenRamp()), 575);
  tierVisualsInvalidate({ dpr }); // gs/PLAT_H/dpr changed → rebuild baked layers

  if (oldW >= 10 && oldH >= 10) {
    const xScale = W / oldW;
    const yScale = H / oldH;
    const sizeScale = gs / oldGs;
    const scaleX = x => x * xScale;
    const clampPlatformX = (x, width) => Math.max(0, Math.min(W - width, x));

    if (orb.worldY) {
      orb.x = Math.max(ORB_R, Math.min(W - ORB_R, scaleX(orb.x)));
      orb.radius = ORB_R;
      orb.trail.forEach(pt => { pt.x = scaleX(pt.x); });
      cameraY = orb.worldY - (H - orbScreenY * yScale);
    }

    platforms.forEach(p => {
      const centerX = scaleX(p.x + p.width / 2);
      p.width *= sizeScale;
      p.x = clampPlatformX(centerX - p.width / 2, p.width);
    });

    particles.forEach(p => {
      p.x *= xScale;
      p.y *= yScale;
      p.size *= sizeScale;
    });
    shockwaves.forEach(sw => {
      sw.x *= xScale;
      sw.y *= yScale;
      sw.r *= sizeScale;
    });
  }
  _streamPrevCamY = cameraY; // resize rewrites cameraY — don't read it as a stream-flow spike
}
window.addEventListener("resize", resize);
new ResizeObserver(resize).observe(document.querySelector(".game-wrap"));

// ── LIVE DATA PIPELINE ───────────────────────────────────────────────────────
function sampleLivePrice() {
  if (!liveMode) return;
  sampleTradeFlow();
}

function sampleTradeFlow() {
  const { buy, sell } = pendingTradeFlow;
  pendingTradeFlow = { buy:0, sell:0 };
  const gross = buy + sell;
  if (gross > 0) {
    flowVolumes.push(gross);
    flowVolumes = flowVolumes.slice(-40);
  }
  const sorted = [...flowVolumes].sort((a,b)=>a-b);
  const baseline = sorted[Math.floor(sorted.length*.6)] || gross || 1;
  livePressure = Math.max(LIVEPRESSURE_MIN, Math.min(LIVEPRESSURE_MAX, (buy-sell)/baseline));

  // Keep rolling price history for chart ghost
  if (Number.isFinite(latestLivePrice)) {
    livePrices.push({ price:latestLivePrice, time:Date.now() });
    livePrices = livePrices.slice(-80);
    setDisplayedPrice(latestLivePrice);
  }

  // Spawn platforms when playing
  if (state === "playing") {
    spawnLivePlatforms(livePressure * .3);
    ensureNeutralCoverage();
  }

  // Update chart preview
  updatePreview();
}

function updatePreview() {
  const tierEl = document.getElementById("previewTier");
  if (tierEl) tierEl.textContent = `${activeTier + 1} · ${tier().name}`;
  updateTierUnlockUi();
}

function mergeGameplayConfig(defaults, incoming) {
  return {
    ...defaults,
    ...incoming,
    market:     { ...defaults.market,    ...incoming.market },
    trend:      { ...defaults.trend,     ...incoming.trend },
    platforms:  { ...defaults.platforms, ...incoming.platforms },
    boosters: {
      ...defaults.boosters,
      ...incoming.boosters,
      surge:  { ...defaults.boosters.surge,  ...incoming.boosters?.surge },
      stream: { ...defaults.boosters.stream, ...incoming.boosters?.stream },
      drag:   { ...defaults.boosters.drag,   ...incoming.boosters?.drag }
    },
    impacts: { ...defaults.impacts, ...incoming.impacts }
  };
}

async function loadGameplayConfig() {
  try {
    const params = new URLSearchParams({ market:currentMarket });
    if (PREVIEW_TOKEN) params.set("previewToken",PREVIEW_TOKEN);
    const incoming = (await apiFetch(`${CALENDAR_API}/gameplay-config?${params}`,{errorLabel:"gameplay config unavailable"}))?.config;
    const t = incoming?.market?.thresholds;
    const imp = incoming?.impacts;
    const validThresholds = Array.isArray(t) && t.length === 3 && t.every(Number.isFinite) && t[0] < t[1] && t[1] < t[2];
    const validImpacts = imp && Number.isFinite(imp.queueSpacingMs) && Number.isFinite(imp.buyImpulse) && Number.isFinite(imp.queueCapacity);
    if (validThresholds && validImpacts) {
      gameplayConfig = mergeGameplayConfig(structuredClone(DEFAULT_GAMEPLAY_CONFIG), incoming);
    }
  } catch {}
  syncMusicRate();
}

function applyPauseState(pause) {
  const overlay = document.getElementById("maintenanceOverlay");
  if (!overlay || !pause) return;
  if (pause.paused) {
    const el = document.getElementById("maintenanceSince");
    if (el && pause.since) el.textContent = "Paused since " + new Date(pause.since).toLocaleString("en-GB");
    overlay.classList.remove("hidden");
  } else {
    overlay.classList.add("hidden");
  }
}

// One-shot pause check at boot. Steady-state pause state arrives folded into the /live-state poll.
async function checkGlobalPause() {
  try {
    applyPauseState(await apiFetch(`${CALENDAR_API}/pause-state`, { errorLabel: "pause state" }));
  } catch {}
}

async function loadMarketTrend() {
  try {
    marketTrendPct = Number((await apiFetch(`${CALENDAR_API}/market-trend?market=${encodeURIComponent(currentMarket)}`,{errorLabel:"trend unavailable"})).changePct)||0;
    setDisplayedTrend(marketTrendPct);
  } catch {
    marketTrendPct = 0;
    setDisplayedTrend(null);
  }
  syncMusicRate();
}

async function pollTradeImpacts() {
  // Outside an active run, leave the stream unconsumed (lastImpactPoll untouched) so impacts
  // landing during a pause or between runs replay once a run is playing again.
  if (state !== "playing" || document.hidden) return;
  try {
    const params = new URLSearchParams({ market:currentMarket, after:String(lastImpactPoll) });
    const impacts = (await apiFetch(`${CALENDAR_API}/trade-impacts?${params}`,{errorLabel:"trade impacts"})).impacts||[];
    const validImpacts = impacts.filter(item => Number.isFinite(item.timestamp) && item.timestamp > 0);
    validImpacts.filter(item => Date.now() - item.timestamp <= IMPACT_MAX_AGE_MS).forEach(enqueueImpact);
    if (validImpacts.length) {
      lastImpactPoll = Math.max(lastImpactPoll, ...validImpacts.map(item => item.timestamp));
      impactPollMs = IMPACT_POLL_MIN_MS;                                  // fresh trades → poll fast again
    } else {
      impactPollMs = Math.min(IMPACT_POLL_MAX_MS, impactPollMs * 2);       // quiet market → back off
    }
  } catch {}
}

function startImpactPolling() {
  clearTimeout(impactPoller);
  lastImpactPoll = Date.now();
  impactPollMs = IMPACT_POLL_MIN_MS;
  const tick = async () => { await pollTradeImpacts(); impactPoller = setTimeout(tick, impactPollMs); };
  tick();
}

function stopSession() {
  sessionGen++;           // invalidate any in-flight callbacks
  clearInterval(liveSampler);
  clearInterval(ociPoller);
  clearTimeout(impactPoller);
  liveSampler=null; ociPoller=null; impactPoller=null; livePoll=null;
  liveMode=false;
  pendingTradeFlow={buy:0,sell:0};
  flowVolumes=[];
  livePressure=0;
  armBuyChartGlow(0,0);
}

function setLiveMode() {
  stopSession();
  liveMode = true;
  const capturedGen = sessionGen;
  livePrices = [];
  liveCandles = [];
  chartBootReady = false;
  document.getElementById("dataStatus").textContent = "CONNECTING TO LIVE FEED";
  bootFallbackFeed(`${feedSourceLabel()} · READY`);
  startImpactPolling();
  startOciLiveMode();
  setTimeout(() => {
    if (sessionGen === capturedGen && liveMode && !chartBootReady) bootFallbackFeed(`${feedSourceLabel()} · TIMEOUT`);
  }, 2500);
  // Tier unlocks now arrive folded into the /live-state poll started by startOciLiveMode().
}

// Which market feeds the chart right now — bonding curve before graduation, Ociswap after.
function feedSourceLabel() {
  return tierUnlockState.graduated ? "OCI" : "RLY.FUN";
}

function bootFallbackFeed(status = null) {
  const p = preloadCandles.at(-1)?.close ?? latestLivePrice ?? historicalPrices.at(-1)?.price ?? market.fallback.at(-1);
  if (!Number.isFinite(p)) return false;
  if (!preloadCandles.length) {
    const fallbackCandles = flatCandles(p);
    preloadCandles = fallbackCandles.length ? fallbackCandles : fallbackMinuteCandles(market);
  }
  latestLivePrice = p;
  setDisplayedPrice(p);
  chartBootReady = true;
  const now = Date.now();
  livePrices = Array.from({length:40},(_,i)=>({ price:p, time:now-(39-i)*LIVE_SAMPLE_MS }));
  document.getElementById("playBtn").disabled = false;
  document.getElementById("previewStatus").textContent = feedSourceLabel();
  document.getElementById("dataStatus").textContent = `${market.symbol} ${status ?? `${feedSourceLabel()} · READY`}`;
  return true;
}

function startOciLiveMode() {
  const capturedGen = sessionGen;
  const sessionMarket = market;
  let lastOciPrice = latestLivePrice;
  let lastCandleLoad = 0;
  const poll = async () => {
    if (sessionGen !== capturedGen) return;
    if (document.hidden) return;           // a backgrounded tab makes no network requests
    try {
      // One merged call replaces the old live-price + tier-unlocks + pause-state polls.
      const feed = await apiFetch(`${CALENDAR_API}/live-state?market=${currentMarket}`,{errorLabel:"live feed unavailable"});
      if (sessionGen !== capturedGen || market!==sessionMarket || !liveMode) return;
      if (feed.tierUnlocks) applyTierUnlockState(feed.tierUnlocks);
      applyPauseState(feed.pause);
      const priceInfo = feed.price;
      const price = parseFloat(priceInfo?.price);
      if (!Number.isFinite(price)) throw new Error("invalid price");
      if (lastOciPrice && price !== lastOciPrice) {
        const delta = Math.abs(price-lastOciPrice)/lastOciPrice;
        const vol = Math.min(delta*2e6, 500);
        if (price>lastOciPrice) pendingTradeFlow.buy+=vol;
        else pendingTradeFlow.sell+=vol;
      }
      lastOciPrice = price;
      latestLivePrice = price;
      setDisplayedPrice(price);
      // Candles refresh on their own slower cadence (browser-cached, ~once/min worth of data).
      const nowMs = Date.now();
      if (nowMs - lastCandleLoad >= CANDLE_REFRESH_MS) {
        lastCandleLoad = nowMs;
        await loadOciMinuteCandles(sessionMarket);
      }
      if (sessionGen !== capturedGen || market!==sessionMarket || !liveMode) return;
      const liveLabel = priceInfo.source === "rly" ? "RLY.FUN" : "OCI";
      document.getElementById("dataStatus").textContent = `${market.symbol} LIVE · ${liveLabel}`;
      document.getElementById("previewPair").textContent = `${market.symbol} / ${market.quote}`;
      if (!chartBootReady) {
        chartBootReady = true;
        const now = Date.now();
        livePrices = Array.from({length:40},(_,i)=>({ price, time:now-(39-i)*LIVE_SAMPLE_MS }));
        document.getElementById("playBtn").disabled = false;
        document.getElementById("previewStatus").textContent = `${liveLabel} LIVE`;
      }
    } catch {
      if (sessionGen !== capturedGen) return;
      if (!chartBootReady) {
        bootFallbackFeed(`${feedSourceLabel()} · RETRY`);
      }
    }
  };
  livePoll = poll;
  poll();
  ociPoller = setInterval(poll, 10000);
  liveSampler = setInterval(sampleLivePrice, LIVE_SAMPLE_MS);
}

async function loadLivePrices() {
  return loadOciPrices();
}

function parseOciCandles(json) {
  if (json.s!=="ok" || !json.t?.length) return [];
  return json.t.map((t,i)=>({
    time: Number(t)*1000,
    open: Number(json.o[i]), high:Number(json.h[i]),
    low: Number(json.l[i]), close:Number(json.c[i])
  })).filter(c=>Object.values(c).every(Number.isFinite)).sort((a,b)=>a.time-b.time);
}

async function loadOciMinuteCandles(sourceMarket=market) {
  const now = Math.floor(Date.now()/1000);
  const lookback = Math.max(Math.ceil(MAX_CANDLE_COUNT * INTERVALS[activeInterval].ms / 1000), 7 * 86400);
  const url = sourceMarket === market
    ? `${CALENDAR_API}/candles?market=${currentMarket}&resolution=${INTERVALS[activeInterval].ociRes}`
    : `https://api.ociswap.com/udf/history?symbol=${sourceMarket.resource}&resolution=${INTERVALS[activeInterval].ociRes}&from=${now-lookback}&to=${now}&countback=${MAX_CANDLE_COUNT}&currencyCode=XRD`;
  const candles = parseOciCandles(await apiFetch(url,{errorLabel:"OCI candles unavailable"}));
  if (!candles.length) throw new Error("OCI candles empty");
  if (market===sourceMarket) preloadCandles = candles.slice(-MAX_CANDLE_COUNT);
}

function flatCandles(price, count = VISIBLE_CANDLE_COUNT) {
  if (!Number.isFinite(price) || price <= 0) return [];
  const intervalMs = INTERVALS[activeInterval].ms;
  const end = Math.floor(Date.now()/intervalMs)*intervalMs;
  const wick = price * .001;
  const length = Math.min(MAX_CANDLE_COUNT,count);
  return Array.from({length},(_,i)=>({
    time:end-(length-1-i)*intervalMs,
    open:price,
    high:price+wick,
    low:Math.max(0,price-wick),
    close:price
  }));
}

function fallbackMinuteCandles(sourceMarket=market) {
  const sourcePrices = sourceMarket===market && historicalPrices.length
    ? historicalPrices.map(p=>p.price)
    : sourceMarket.fallback;
  const prices = sourcePrices.slice(-Math.min(MAX_CANDLE_COUNT,sourcePrices.length));
  const intervalMs = INTERVALS[activeInterval].ms;
  const end = Math.floor(Date.now()/intervalMs)*intervalMs;
  return prices.map((close,i) => {
    const open = i ? prices[i-1] : close;
    const wick = Math.max(open,close) * (.002 + i%5*.0006);
    return {
      time:end-(prices.length-1-i)*intervalMs,
      open,
      high:Math.max(open,close)+wick,
      low:Math.max(0,Math.min(open,close)-wick),
      close
    };
  });
}

async function loadRecentCandles() {
  const sourceMarket = market;
  try {
    await loadOciMinuteCandles(sourceMarket);
  } catch {
    if (market===sourceMarket) {
      const fallbackCandles = flatCandles(latestLivePrice ?? historicalPrices.at(-1)?.price ?? sourceMarket.fallback.at(-1));
      preloadCandles = fallbackCandles.length ? fallbackCandles : fallbackMinuteCandles(sourceMarket);
    }
  }
}

async function loadOciPrices() {
  const parseUdfJson = (json) => {
    if (json.s !== "ok" || !json.t?.length) return null;
    const prices = json.t.map((t,i)=>({time:t*1000,price:parseFloat(json.c[i])})).filter(p=>Number.isFinite(p.price));
    return prices.length ? prices : null;
  };
  try {
    const parsed = parseUdfJson(await apiFetch(`${CALENDAR_API}/candles-1d?market=${currentMarket}`, {errorLabel:"worker 1d unavailable"}));
    if (!parsed) throw new Error("no data from worker");
    historicalPrices = parsed;
    latestLivePrice = historicalPrices.at(-1).price;
    setDisplayedPrice(latestLivePrice);
    return;
  } catch {}
  try {
    const now = Math.floor(Date.now()/1000);
    const url = `https://api.ociswap.com/udf/history?symbol=${market.resource}&resolution=1D&from=${now-90*86400}&to=${now}&countback=90&currencyCode=XRD`;
    const parsed = parseUdfJson(await apiFetch(url, {errorLabel:"OCI 1d unavailable"}));
    if (!parsed) throw new Error("no data");
    historicalPrices = parsed;
    latestLivePrice = historicalPrices.at(-1).price;
    setDisplayedPrice(latestLivePrice);
  } catch {
    latestLivePrice = market.fallback.at(-1);
    setDisplayedPrice(latestLivePrice);
  }
}

async function switchInterval(key) {
  if (!INTERVALS[key] || key === activeInterval) return;
  activeInterval = key;
  document.querySelectorAll(".tf-btn").forEach(b => b.classList.toggle("active", b.dataset.tf === key));
  preloadCandles = [];
  liveCandles = [];
  try {
    await loadOciMinuteCandles();
  } catch {
    const fallbackCandles = flatCandles(latestLivePrice ?? historicalPrices.at(-1)?.price ?? market.fallback.at(-1));
    preloadCandles = fallbackCandles.length ? fallbackCandles : fallbackMinuteCandles();
  }
}

/* ── SHOP ─────────────────────────────────────────────────── */
async function fetchShop() {
  try {
    const res = await fetch("/shop");
    if (!res.ok) throw new Error("shop fetch failed");
    const data = await res.json();
    if (data.locked) { renderShopLocked(data); return; }
    if (storedWalletState().status === "connected") await fetchInventory();
    renderShopOverlay(data);
  } catch (err) {
    console.warn("[shop]", err);
  }
}

function renderShopLocked(data) {
  document.getElementById("shopUnlockedState")?.classList.add("hidden");
  document.getElementById("shopLockedState")?.classList.remove("hidden");
  document.getElementById("shopNoWallet")?.classList.add("hidden");
  const pct = Math.max(0, Math.min(100, Number(data.graduationPct) || 0));
  const bar = document.getElementById("graduationBar");
  if (bar) bar.style.width = `${pct}%`;
  const label = document.getElementById("graduationPct");
  if (label) label.textContent = data.graduationPct != null ? `${Math.round(pct)}%` : "—";
}

function fetchShopStatus() {
  fetch("/shop").then(r => r.json()).then(data => {
    if (!data.locked) document.getElementById("shopBtn")?.removeAttribute("disabled");
  }).catch(() => {});
}

function animateCosmeticCanvas(cv, item, overlayId) {
  const cid = item.cosmeticId ?? item.itemId;
  const sk = (typeof SKINS_BY_NFT_ID !== "undefined" ? SKINS_BY_NFT_ID[cid] : null)
          ?? (typeof SKINS_BY_ID !== "undefined" ? SKINS_BY_ID[cid] : null);
  const badge = !sk && typeof BADGES !== "undefined" ? BADGES?.find(b => b.code === cid) : null;
  if (!sk && !badge) return;
  const ctx = cv.getContext("2d");
  let raf;
  function frame() {
    const overlay = document.getElementById(overlayId);
    // Stop when the overlay is hidden OR removed from the DOM (zombie rAF guard).
    if (!overlay || overlay.classList.contains("hidden")) {
      cancelAnimationFrame(raf); return;
    }
    const t = performance.now() / 1000;
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (sk && typeof sceneSkinNFT !== "undefined") sceneSkinNFT(ctx, cv.width, cv.height, sk, t);
    else if (badge && typeof sceneBadge !== "undefined") sceneBadge(ctx, cv.width, cv.height, badge, t);
    raf = requestAnimationFrame(frame);
  }
  raf = requestAnimationFrame(frame);
}

function renderShopOverlay(data) {
  const unlockedEl = document.getElementById("shopUnlockedState");
  const gridEl = document.getElementById("shopItemGrid");
  const noWalletEl = document.getElementById("shopNoWallet");
  if (!unlockedEl) return;

  _lastShopData = data;
  unlockedEl.classList.remove("hidden");
  document.getElementById("shopLockedState")?.classList.add("hidden");
  document.getElementById("shopBtn")?.removeAttribute("disabled");

  const items = data.capsule?.items ?? [];
  const walletState = storedWalletState();
  const connected = walletState.status === "connected";
  if (noWalletEl) {
    noWalletEl.classList.remove("hidden");
    noWalletEl.classList.toggle("linked", connected);
    const link = document.getElementById("shopConnectLink");
    if (link) link.textContent = connected ? "◆ WALLET LINKED" : "◇ LINK WALLET";
  }

  if (!gridEl) return;
  gridEl.innerHTML = items.map(item => {
    const canvasId = `shop-canvas-${item.itemId.replace(/[^a-z0-9]/gi, "_")}`;
    const available = item.available ?? item.supply;
    const availStr = item.available !== null ? `${item.available} / ${item.supply}` : `/ ${item.supply}`;
    const supplyPct = item.supply > 0 ? Math.round((available / item.supply) * 100) : 0;
    const priceStr = item.priceAscent >= 1000000 ? `${(item.priceAscent / 1000000).toFixed(1)}M` : `${(item.priceAscent / 1000).toFixed(0)}K`;
    const purchasable = item.purchasable !== false;   // special drops are display-only
    const owned = connected && ownedCosmeticIds.has(item.itemId);
    const sk = typeof SKINS_BY_ID !== "undefined" ? SKINS_BY_ID[item.itemId] : null;
    const isEquipped = owned && equippedSkin?.id === sk?.id;

    // Special drops carry no price/supply count — show a static badge instead of a misleading bar.
    const supplyBar = purchasable
      ? `<div class="shop-item-supply">
      <div class="supply-bar-wrap"><div class="supply-bar" style="width:${supplyPct}%"></div></div>
      <span>${availStr} LEFT</span>
    </div>`
      : `<div class="shop-item-supply"><span>SPECIAL DROP</span></div>`;

    let actionBtn = "";
    if (connected) {
      if (owned && sk) {
        actionBtn = `<button class="shop-equip-btn${isEquipped ? " equipped" : ""}" data-skin-id="${sk.id}">${isEquipped ? "EQUIPPED" : "EQUIP"}</button>`;
      } else if (!purchasable) {
        actionBtn = `<button class="shop-drop-btn" disabled title="Special drop — not for sale">DROP ONLY</button>`;
      } else {
        actionBtn = `<button class="shop-buy-btn" data-item-id="${item.itemId}" title="Buy ${item.name}">BUY</button>`;
      }
    }

    return `<div class="shop-item${owned ? " owned" : ""}" data-rarity="${item.rarity}">
      <canvas id="${canvasId}" width="100" height="100"></canvas>
      <div class="shop-item-name">${item.name}</div>
      <span class="rarity-chip ${item.rarity}">${item.rarity}</span>
      <div class="shop-item-price">${(owned || !purchasable) ? "" : `${priceStr} ASCENT`}</div>
      ${supplyBar}
      ${actionBtn}
    </div>`;
  }).join("");

  items.forEach(item => {
    const c = document.getElementById(`shop-canvas-${item.itemId.replace(/[^a-z0-9]/gi, "_")}`);
    if (c) animateCosmeticCanvas(c, { cosmeticId: item.itemId }, "shopOverlay");
  });

  /* Equip buttons (owned skins) */
  gridEl.querySelectorAll(".shop-equip-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const sk = typeof SKINS_BY_RENDER !== "undefined" ? SKINS_BY_RENDER[btn.dataset.skinId] : null;
      if (!sk) return;
      if (equippedSkin?.id === sk.id) unequipSkin();
      else equipSkin(sk);
      renderShopOverlay(_lastShopData);
    });
  });

  /* Buy buttons (not owned) */
  gridEl.querySelectorAll(".shop-buy-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.itemId;
      const sk = typeof SKINS_BY_ID !== "undefined" ? SKINS_BY_ID[id] : null;
      if (!sk) return;
      btn.disabled = true;
      btn.textContent = "PENDING…";
      sendBuyTransaction(id, sk.priceAscent)
        .then(hash => {
          if (hash) {
            btn.textContent = "SUCCESS";
            setTimeout(() => fetchInventory().then(() => renderShopOverlay(_lastShopData)), 3000);
          } else {
            btn.disabled = false;
            btn.textContent = "BUY";
          }
        })
        .catch(err => {
          console.warn("[buy]", err);
          btn.disabled = false;
          btn.textContent = "BUY";
        });
    });
  });
}

/* ── PROFILE ──────────────────────────────────────────────── */
async function fetchProfile() {
  const state = storedWalletState();
  if (state.status !== "connected") return;
  const addrEl = document.getElementById("profileAddress");
  if (addrEl) addrEl.textContent = state.accountAddress || "—";
  const adminBtn = document.getElementById("profileAdminBtn");
  if (adminBtn) {
    adminBtn.classList.toggle("hidden", !state.isAdmin);
    adminBtn.onclick = () => window.open("/admin", "_blank");
  }
  try {
    const res = await fetch("/profile", {
      headers: state.sessionToken ? { Authorization: `Bearer ${state.sessionToken}` } : {},
    });
    if (!res.ok) return;
    const { profile } = await res.json();
    playerBestScore = profile?.bestScore ?? null;
    playerBestScoreLoaded = true;
    // Legacy profiles predate best-score market/tier tracking; every pre-feature score
    // was recorded on the ascent market, tier 0 (the only tier that ever had entries).
    playerBestScoreMarket = profile?.bestScoreMarket ?? (playerBestScore != null ? "ascent" : null);
    playerBestScoreTier = Number.isInteger(profile?.bestScoreTier) ? profile.bestScoreTier : (playerBestScore != null ? 0 : null);
    updateBestScoreHud();
    loadBestRank();
    const skinCountEl = document.getElementById("profileSkinCount");
    if (skinCountEl) skinCountEl.textContent = profile?.skinCount ?? "0";
    const bestScoreEl = document.getElementById("profileBestScore");
    if (bestScoreEl) bestScoreEl.textContent = playerBestScore != null ? playerBestScore.toLocaleString() : "—";
    const startBestEl = document.getElementById("startBestScore");
    const startBestVal = document.getElementById("startBestScoreVal");
    if (startBestEl && startBestVal) {
      if (playerBestScore != null) {
        startBestVal.textContent = playerBestScore.toLocaleString();
        startBestEl.classList.remove("hidden");
      } else {
        startBestEl.classList.add("hidden");
      }
    }
  } catch (err) {
    console.warn("[profile]", err);
  }
}

/* ── INVENTORY ────────────────────────────────────────────── */
async function fetchInventory() {
  const state = storedWalletState();
  const gridEl = document.getElementById("inventoryGrid");
  if (!gridEl) return;
  if (state.status !== "connected") {
    gridEl.innerHTML = `<p class="inv-empty">Connect your Radix wallet to view your inventory.</p>`;
    return;
  }
  try {
    const res = await fetch(`/inventory?account=${encodeURIComponent(state.accountAddress)}`, {
      headers: state.sessionToken ? { Authorization: `Bearer ${state.sessionToken}` } : {},
    });
    if (!res.ok) throw new Error("inventory fetch failed");
    const { items, ascentBalance } = await res.json();
    renderInventoryGrid(items ?? []);
    const ascentBalEl = document.getElementById("profileAscentBal");
    if (ascentBalEl) ascentBalEl.textContent = formatCompactAscent(ascentBalance ?? 0);
  } catch (err) {
    console.warn("[inventory]", err);
    gridEl.innerHTML = `<p class="inv-empty">Could not load inventory.</p>`;
  }
}

function renderInventoryGrid(items) {
  ownedCosmeticIds = new Set(items.filter(i => !i.isBadge).map(i => {
    const cid = i.cosmeticId ?? i.itemId;
    const skin = typeof SKINS_BY_NFT_ID !== "undefined" ? SKINS_BY_NFT_ID[cid] : null;
    return skin ? skin.itemId : cid;
  }));
  const skinCountEl = document.getElementById("profileSkinCount");
  if (skinCountEl) skinCountEl.textContent = ownedCosmeticIds.size;
  const gridEl = document.getElementById("inventoryGrid");
  if (!gridEl) return;
  if (!items.length) {
    gridEl.innerHTML = `<div class="inv-empty-state"><div class="inv-empty-orb">◎</div><p class="inv-empty-label">— NO SKIN EQUIPPED —</p><p class="inv-empty">No cosmetics yet. Visit the shop to collect Capsule 02 skins.</p></div>`;
    return;
  }
  gridEl.innerHTML = items.map((item, idx) => {
    const cid = item.cosmeticId ?? item.itemId;
    const sk = (typeof SKINS_BY_NFT_ID !== "undefined" ? SKINS_BY_NFT_ID[cid] : null)
            ?? (typeof SKINS_BY_ID !== "undefined" ? SKINS_BY_ID[cid] : null);
    const badge = !sk && typeof BADGES !== "undefined" ? BADGES?.find(b => b.code === cid) : null;
    const isEquipped = !item.isBadge && equippedSkin?.id === sk?.id;
    const canvasId = `inv-canvas-${(item.localId || cid || idx).toString().replace(/[^a-z0-9]/gi, "_")}`;
    const displayName = sk?.name ?? badge?.name ?? cid ?? "—";
    const displayRarity = sk?.rarity ?? (item.isBadge ? "SOULBOUND" : item.rarity ?? "RARE");
    return `<div class="inventory-item${isEquipped ? " equipped" : ""}" data-item-id="${cid}">
      <canvas id="${canvasId}" width="80" height="80"></canvas>
      <div class="inv-name">${displayName}</div>
      <span class="rarity-chip ${displayRarity}">${displayRarity}</span>
      ${item.isBadge ? "" : `<button class="inv-equip-btn${isEquipped ? " equipped" : ""}" data-skin-id="${sk?.id ?? ""}">${isEquipped ? "EQUIPPED" : "EQUIP"}</button>`}
    </div>`;
  }).join("");

  items.forEach((item, idx) => {
    const cid = item.cosmeticId ?? item.itemId;
    const canvasId = `inv-canvas-${(item.localId || cid || idx).toString().replace(/[^a-z0-9]/gi, "_")}`;
    const c = document.getElementById(canvasId);
    if (c) animateCosmeticCanvas(c, item, "inventoryOverlay");
  });

  /* Equip buttons */
  gridEl.querySelectorAll(".inv-equip-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const skinId = btn.dataset.skinId;
      if (!skinId) return;
      const sk = typeof SKINS_BY_RENDER !== "undefined" ? SKINS_BY_RENDER[skinId] : null;
      if (!sk) return;
      if (equippedSkin?.id === sk.id) unequipSkin();
      else equipSkin(sk);
      renderInventoryGrid(items);
    });
  });
}

function shortWalletAddress(address) {
  const value = String(address || "");
  return value.length > 11 ? `${value.slice(0, 4)}…${value.slice(-4)}` : value;
}

function storedWalletState() {
  try {
    const state = JSON.parse(localStorage.getItem(WALLET_CONNECT_LOCAL_KEY) || "null");
    return state?.status === "connected" && state.accountAddress ? state : { status: "idle" };
  } catch {
    return { status: "idle" };
  }
}

function _saveWalletLocal(patch) {
  try {
    const current = JSON.parse(localStorage.getItem(WALLET_CONNECT_LOCAL_KEY) || "{}");
    localStorage.setItem(WALLET_CONNECT_LOCAL_KEY, JSON.stringify({ ...current, ...patch }));
  } catch {}
}

function _restoreEquippedSkin() {
  try {
    const state = JSON.parse(localStorage.getItem(WALLET_CONNECT_LOCAL_KEY) || "{}");
    if (state.equippedSkinId && typeof SKINS_BY_RENDER !== "undefined") {
      const skin = SKINS_BY_RENDER[state.equippedSkinId];
      if (skin) equippedSkin = skin;
    }
  } catch {}
}

function setWalletButtonState(status, accountAddress = "") {
  const button = document.getElementById("walletLinkBtn");
  if (!button) return;
  const mark = button.querySelector(".wallet-mark");
  const label = button.querySelector(".wallet-label");
  const compact = window.matchMedia?.("(max-width: 560px)")?.matches;
  const ctaSubs = { startWalletCta: "unlock shop & achievements", finishWalletCta: "save your score" };
  ["startWalletCta", "finishWalletCta"].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove("hidden");
    el.classList.toggle("linked",  status === "connected");
    el.classList.toggle("syncing", status === "syncing");
    const main = el.querySelector(".wcta-main");
    const sub  = el.querySelector(".wcta-sub");
    if (status === "connected") {
      if (main) main.textContent = "◆ WALLET LINKED";
      if (sub)  sub.textContent  = "";
    } else if (status === "syncing") {
      if (main) main.textContent = "◇ SYNCING";
      if (sub)  sub.textContent  = "";
    } else {
      if (main) main.textContent = "◇ LINK WALLET";
      if (sub)  sub.textContent  = ctaSubs[id];
    }
  });
  const noWalletEl = document.getElementById("shopNoWallet");
  if (noWalletEl && !noWalletEl.classList.contains("hidden")) {
    noWalletEl.classList.toggle("linked",  status === "connected");
    noWalletEl.classList.toggle("syncing", status === "syncing");
    const link = document.getElementById("shopConnectLink");
    if (link) {
      if (status === "connected")    link.textContent = "◆ WALLET LINKED";
      else if (status === "syncing") link.textContent = "◇ SYNCING";
      else                           link.textContent = "◇ LINK WALLET";
    }
  }
  button.classList.toggle("syncing", status === "syncing");
  button.classList.toggle("connected", status === "connected");
  button.classList.toggle("failed", status === "failed");
  button.disabled = status === "syncing";
  if (status === "connected") {
    if (mark) mark.textContent = "◆";
    if (label) label.textContent = shortWalletAddress(accountAddress);
    button.setAttribute("aria-label", `Radix Wallet connected: ${accountAddress}`);
    button.title = "Radix Wallet connected";
    return;
  }
  if (status === "syncing") {
    if (mark) mark.textContent = "◇";
    if (label) label.textContent = compact ? "SYNC" : "SYNCING";
    button.setAttribute("aria-label", "Connecting Radix Wallet");
    button.title = "Connecting Radix Wallet";
    return;
  }
  if (status === "failed") {
    if (mark) mark.textContent = "◇";
    if (label) label.textContent = compact ? "FAILED" : "LINK FAILED";
    button.setAttribute("aria-label", "Radix Wallet connection failed");
    button.title = "Radix Wallet connection failed";
    return;
  }
  if (mark) mark.textContent = "◇";
  if (label) label.textContent = compact ? "LINK" : "LINK RADIX";
  button.setAttribute("aria-label", "Connect Radix Wallet");
  button.title = "Connect Radix Wallet";
}

function initWalletButton() {
  const button = document.getElementById("walletLinkBtn");
  if (!button) return;
  const localState = storedWalletState();
  setWalletButtonState(localState.status, localState.accountAddress);
  // Load the profile up front so the personal best and its all-time rank are available
  // on the finish screen and leaderboard without opening the profile overlay first.
  if (localState.status === "connected") fetchProfile();
  _restoreEquippedSkin();
  button.addEventListener("click", e => {
    e.stopPropagation();
    const state = storedWalletState();
    if (state.status === "connected") {
      openInfoOverlay("profileOverlay");
      fetchProfile();
      fetchInventory();
      return;
    }
    rolaConnect();
  });
  document.getElementById("startWalletCta")?.addEventListener("click", () => rolaConnect());
  document.getElementById("finishWalletCta")?.addEventListener("click", () => rolaConnect());
  window.addEventListener("resize", () => {
    const state = storedWalletState();
    setWalletButtonState(state.status, state.accountAddress);
  });
}

async function initRadixToolkit() {
  if (typeof ASCENT_DAPP_DEFINITION === "undefined" || !ASCENT_DAPP_DEFINITION) return;
  await new Promise(resolve => {
    const id = setInterval(() => { if (window._RDT) { clearInterval(id); resolve(); } }, 50);
    setTimeout(() => { clearInterval(id); resolve(); }, 5000); // max 5s
  });
  if (!window._RDT) { console.warn("[rdt] Radix dApp Toolkit failed to load"); return; }
  const { RadixDappToolkit } = window._RDT;
  window._rdtInstance = RadixDappToolkit({
    dAppDefinitionAddress: ASCENT_DAPP_DEFINITION,
    networkId: typeof RADIX_NETWORK_ID !== "undefined" ? RADIX_NETWORK_ID : 1,
    applicationName: "ASCENT",
    applicationVersion: "1.0.0",
  });
  // RDT v2.3+: challenge injected via generator, not passed to withProof()
  window._rdtInstance.walletApi.provideChallengeGenerator(async () => {
    const res = await fetch("/auth/challenge", { method: "POST" });
    if (!res.ok) throw new Error("challenge fetch failed");
    const { challenge } = await res.json();
    return challenge;
  });
}

async function rolaConnect() {
  setWalletButtonState("syncing");
  try {
    if (!window._rdtInstance || !window._RDT) throw new Error("Radix dApp Toolkit not ready");
    const { OneTimeDataRequestBuilder } = window._RDT;

    // RDT v2.3+: withProof() takes no args; challenge comes from provideChallengeGenerator
    const result = await window._rdtInstance.walletApi.sendOneTimeRequest(
      OneTimeDataRequestBuilder.accounts().exactly(1).withProof()
    );
    if (result.isErr()) throw result.error;

    console.log("[rola] result.value:", JSON.stringify(result.value, null, 2));

    // proofs[]: { type, challenge, address, proof: { publicKey, signature, curve } }
    const accountProof = result.value.proofs?.find(p => p.type === "account");
    if (!accountProof) throw new Error("No account proof returned by wallet — proofs: " + JSON.stringify(result.value.proofs));

    const verRes = await fetch("/auth/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        challenge: accountProof.challenge,
        proof: { accountAddress: accountProof.address, proof: accountProof.proof },
      }),
    });
    if (!verRes.ok) {
      const errBody = await verRes.text();
      throw new Error(`verify failed ${verRes.status}: ${errBody}`);
    }
    const { sessionToken, accountAddress, identityAddress, isAdmin } = await verRes.json();
    _saveWalletLocal({ status: "connected", accountAddress, identityAddress, sessionToken, isAdmin: !!isAdmin });
    setWalletButtonState("connected", accountAddress);
    fetchProfile();
    fetchShopStatus();
  } catch (err) {
    console.warn("[wallet] rolaConnect:", err);
    setWalletButtonState("failed");
    setTimeout(() => setWalletButtonState("idle"), 3000);
  }
}

async function sendBuyTransaction(itemId, priceAscent) {
  if (typeof ASCENT_SHOP_COMPONENT === "undefined" || !ASCENT_SHOP_COMPONENT) {
    alert("Shop not deployed yet — check back soon.");
    return null;
  }
  if (!window._rdtInstance) throw new Error("Radix dApp Toolkit not ready");
  const state = storedWalletState();
  if (state.status !== "connected") throw new Error("Wallet not connected");
  const resource = typeof ASCENT_TOKEN_RESOURCE !== "undefined"
    ? ASCENT_TOKEN_RESOURCE
    : "resource_rdx1t46jpmzf97s7q5h4tv42hcjyalhq84znevngtsul5wcumxfdprnlp3";
  const manifest = [
    `CALL_METHOD Address("${state.accountAddress}") "withdraw" Address("${resource}") Decimal("${priceAscent}");`,
    `TAKE_FROM_WORKTOP Address("${resource}") Decimal("${priceAscent}") Bucket("payment");`,
    `CALL_METHOD Address("${ASCENT_SHOP_COMPONENT}") "buy" "${itemId}" Bucket("payment");`,
    `CALL_METHOD Address("${state.accountAddress}") "deposit_batch" Expression("ENTIRE_WORKTOP");`,
  ].join("\n");
  const result = await window._rdtInstance.walletApi.sendTransaction({ transactionManifest: manifest });
  if (result.isErr()) throw result.error;
  return result.value.transactionIntentHash;
}

function walletDisconnect() {
  fetch("/logout", { method: "POST" }).catch(() => {});
  _saveWalletLocal({ status: "disconnected" });
  localStorage.removeItem(WALLET_CONNECT_LOCAL_KEY);
  playerBestScore = null;
  playerBestScoreLoaded = false;
  playerBestScoreMarket = null;
  playerBestScoreTier = null;
  bestRank = null;
  updateBestScoreHud();
  updateBestRankUi();
  setWalletButtonState("idle");
  document.getElementById("shopBtn")?.removeAttribute("disabled");
  closeInfoOverlay("profileOverlay");
}

// ── EVENT LISTENERS ──────────────────────────────────────────────────────────
document.querySelectorAll(".tf-btn").forEach(btn =>
  btn.addEventListener("click", () => switchInterval(btn.dataset.tf))
);
document.getElementById("playBtn").addEventListener("click",    requestStartGame);
document.getElementById("replayBtn").addEventListener("click",  requestStartGame);
document.getElementById("restartBtn").addEventListener("click", requestStartGame);
document.getElementById("pauseBtn").addEventListener("click", e => {
  e.stopPropagation();
  togglePause();
});
document.getElementById("resumeBtn").addEventListener("click", resumeGame);
const _nameInput = document.getElementById("playerNameInput");
const _nameFullMsg = document.getElementById("nameFullMsg");
let _nameFlashTimer = null;
const _triggerNameFlash = () => {
  if (!_nameFullMsg) return;
  _nameFullMsg.classList.remove("visible");
  void _nameFullMsg.offsetWidth;
  _nameFullMsg.classList.add("visible");
  clearTimeout(_nameFlashTimer);
  _nameFlashTimer = setTimeout(() => _nameFullMsg.classList.remove("visible"), 1050);
};
const _nameAtLimit = () =>
  _nameInput.value.length >= 8 && _nameInput.selectionStart === _nameInput.selectionEnd;
_nameInput.addEventListener("blur", currentPlayerName);
// Physical keyboard
_nameInput.addEventListener("keydown", e => {
  if (_nameAtLimit() && e.key.length === 1 && !e.ctrlKey && !e.metaKey) _triggerNameFlash();
});
// Mobile virtual keyboard
_nameInput.addEventListener("beforeinput", e => {
  if (_nameAtLimit() && e.inputType.startsWith("insert")) _triggerNameFlash();
});
document.getElementById("playerNameInput").addEventListener("keydown", e => {
  if (e.key !== "Enter" || document.getElementById("playBtn").disabled) return;
  e.preventDefault();
  requestStartGame();
});
document.getElementById("leaderboardBtn").addEventListener("click", () => {
  renderAllLeaderboards();
  openInfoOverlay("leaderboardOverlay");
  if (!leaderboardSubmissions.has(currentMarket)) loadLeaderboards();
  loadBestRank();
});
function onLbTabAllTime() {
  if (lbMode==="alltime") return;
  setLbMode("alltime");
  lbRender();
  renderAllLeaderboards();
}
function onLbTabTemporary() {
  if (lbMode==="temporary") return;
  setLbMode("temporary");
  lbRender();
  if (!temporaryLeaderboardState.has(currentMarket) || temporaryLeaderboardState.get(currentMarket).status==="loading")
    loadTemporaryLeaderboards();
  else renderAllLeaderboards();
}
document.getElementById("lbTabAllTime").addEventListener("click", onLbTabAllTime);
document.getElementById("lbTabTemporary").addEventListener("click", onLbTabTemporary);
document.getElementById("finishLbTabAllTime").addEventListener("click", onLbTabAllTime);
document.getElementById("finishLbTabTemporary").addEventListener("click", onLbTabTemporary);
document.addEventListener("click", e => {
  const btn = e.target.closest("[data-close-overlay]");
  if (btn) closeInfoOverlay(btn.dataset.closeOverlay);
});
document.getElementById("optionsCloseX")?.addEventListener("click", () => {
  document.getElementById("mixerPanel").classList.add("hidden");
  document.getElementById("mixerBtn")?.classList.remove("active-mix");
  finishUltiKeyCapture();
  finishGamepadCapture();
  if (pausedByOptions) { pausedByOptions = false; resumeGame(); }
});
document.getElementById("shopBtn").addEventListener("click", () => { openInfoOverlay("shopOverlay"); fetchShop(); });
document.getElementById("profileCloseBtn")?.addEventListener("click", () => closeInfoOverlay("profileOverlay"));
document.getElementById("profileDisconnectBtn")?.addEventListener("click", walletDisconnect);
document.getElementById("shopConnectLink")?.addEventListener("click", rolaConnect);
document.getElementById("profileInventoryBtn")?.addEventListener("click", () => { openInfoOverlay("inventoryOverlay"); fetchInventory(); });
document.getElementById("achievementsBtn").addEventListener("click", () => openInfoOverlay("achievementsOverlay"));
document.getElementById("howToPlayBtn").addEventListener("click", () => openInfoOverlay("howToPlayOverlay"));
document.getElementById("soundBtn").addEventListener("click", e => {
  e.stopPropagation();
  const allMuted = !sfxMuted;
  sfxMuted = allMuted; musicMuted = allMuted;
  syncMusic(); saveOptions(); updateMixerUI();
});

// Options panel toggle
document.getElementById("mixerBtn").addEventListener("click", e => {
  e.stopPropagation();
  const panel = document.getElementById("mixerPanel");
  const wasHidden = panel.classList.contains("hidden");
  if (wasHidden) closeInfoOverlays({ resume:false });
  const isHidden = panel.classList.toggle("hidden");
  if (isHidden) {
    finishUltiKeyCapture();
    if (pausedByOptions) { pausedByOptions = false; resumeGame(); }
  } else {
    if (state === "playing") { pausedByOptions = true; pauseGame(); }
  }
  document.getElementById("mixerBtn").classList.toggle("active-mix", !isHidden);
  document.getElementById("tracklistPanel").classList.add("hidden");
  document.getElementById("trackListBtn").classList.remove("active-mix");
  if (!isHidden) updateMixerUI();
});
document.querySelector("#mixerPanel .overlay-card").addEventListener("click", e => e.stopPropagation());
// Music volume slider
document.getElementById("musicVolumeSlider").addEventListener("input", e => {
  musicVolume = parseFloat(e.target.value);
  musicMuted = musicVolume === 0;
  setMusicVolume(musicVolume);
  updateSliderTrack(e.target);
  saveOptions();
  updateMixerUI();
});

// SFX volume slider
document.getElementById("sfxVolumeSlider").addEventListener("input", e => {
  sfxVolume = parseFloat(e.target.value);
  sfxMuted = sfxVolume === 0;
  updateSliderTrack(e.target);
  saveOptions();
  updateMixerUI();
  if (T && typeof Tone !== 'undefined') Tone.getDestination().volume.value = Tone.gainToDb(sfxVolume || 0.001);
});
["musicVolumeSlider","sfxVolumeSlider"].forEach(id => {
  const el = document.getElementById(id);
  if (!el) return;
  const playFeedback = () => { if (T && !sfxMuted) sfx.bounce(); };
  el.addEventListener("mouseup", playFeedback);
  el.addEventListener("touchend", playFeedback);
});

// Mute buttons in mixer
document.getElementById("muteMusicBtn").addEventListener("click", e => {
  e.stopPropagation();
  musicMuted = !musicMuted;
  syncMusic(); saveOptions(); updateMixerUI();
});
document.getElementById("muteSfxBtn").addEventListener("click", e => {
  e.stopPropagation();
  sfxMuted = !sfxMuted;
  saveOptions(); updateMixerUI();
});
document.getElementById("vibrationToggle").addEventListener("click", () => {
  if (typeof navigator.vibrate !== "function") return;
  vibrationEnabled = !vibrationEnabled;
  saveOptions();
  updateMixerUI();
});
document.getElementById("qualityToggle")?.addEventListener("click", () => {
  const cycle = { auto: "high", high: "low", low: "auto" };
  setQualityMode(cycle[qualityMode] || "auto");
  saveOptions();
  updateMixerUI();
});
document.getElementById("trailsToggle")?.addEventListener("click", () => {
  const cycle = { all: "basic", basic: "off", off: "all" };
  trailMode = cycle[trailMode] || "all";
  saveOptions();
  updateMixerUI();
});
document.querySelectorAll("[data-ulti-key-slot]").forEach(button => {
  button.addEventListener("click", () => {
    rebindingControlKey = null;
    rebindingGamepadAction = null;
    rebindingUltiSlot = Number(button.dataset.ultiKeySlot);
    setOptionsMessage("PRESS A KEY · ESC TO CANCEL");
    updateMixerUI();
  });
});
document.querySelectorAll("[data-ctrl-key]").forEach(button => {
  button.addEventListener("click", () => {
    rebindingUltiSlot = null;
    rebindingGamepadAction = null;
    rebindingControlKey = button.dataset.ctrlKey;
    const label = rebindingControlKey === "pause"
      ? "PRESS A KEY · ESC = ESC KEY"
      : "PRESS A KEY · ESC TO CANCEL";
    setOptionsMessage(label);
    updateMixerUI();
  });
});
document.querySelectorAll("[data-gamepad-action]").forEach(button => {
  button.addEventListener("click", () => {
    rebindingUltiSlot = null;
    rebindingControlKey = null;
    rebindingGamepadAction = button.dataset.gamepadAction;
    setOptionsMessage("PRESS GAMEPAD INPUT · ESC TO CANCEL");
    updateMixerUI();
  });
});
document.getElementById("optionsResetBtn").addEventListener("click", resetOptions);

// Tracklist panel toggle
document.getElementById("trackListBtn").addEventListener("click", e => {
  e.stopPropagation();
  finishUltiKeyCapture();
  finishGamepadCapture();
  const panel = document.getElementById("tracklistPanel");
  const isHidden = panel.classList.toggle("hidden");
  document.getElementById("trackListBtn").classList.toggle("active-mix", !isHidden);
  document.getElementById("mixerPanel").classList.add("hidden");
  document.getElementById("mixerBtn").classList.remove("active-mix");
  if (pausedByOptions) { pausedByOptions = false; resumeGame(); }
});

// Ulti buttons (click delegation)
document.getElementById("ultiButtons").addEventListener("click", e => {
  const slot = e.target.closest("[data-slot]");
  if (slot) tryUlti(Number(slot.dataset.slot));
});
document.getElementById("ultiSelectConfirm").addEventListener("click", confirmUltiSelection);
document.getElementById("ultiHintBtn").addEventListener("click", () => {
  openInfoOverlay("howToPlayOverlay");
  document.querySelector('[data-guide-tab="ultimates"]')?.click();
});

// Prev / Next track
document.getElementById("prevTrackBtn").addEventListener("click", prevTrack);
document.getElementById("nextTrackBtn").addEventListener("click", nextTrack);

// Close panels on outside click
document.addEventListener("click", () => {
  rebindingUltiSlot = null;
  rebindingControlKey = null;
  rebindingGamepadAction = null;
  setOptionsMessage();
  const mixerWasOpen = !document.getElementById("mixerPanel").classList.contains("hidden");
  document.getElementById("mixerPanel").classList.add("hidden");
  document.getElementById("tracklistPanel").classList.add("hidden");
  document.getElementById("mixerBtn").classList.remove("active-mix");
  document.getElementById("trackListBtn").classList.remove("active-mix");
  if (mixerWasOpen && pausedByOptions) { pausedByOptions = false; resumeGame(); }
});
// ── BOOTSTRAP ────────────────────────────────────────────────────────────────
async function bootstrap() {
  _nameInput.value = localStorage.getItem(PLAYER_NAME_KEY) || "";
  if (storedWalletState().status !== "connected") {
    const _d = new Date().toISOString().slice(0, 10), _k = "asc_visit_" + _d;
    if (!localStorage.getItem(_k)) {
      fetch("/track", { method: "POST", body: JSON.stringify({ type: "visit" }), headers: { "Content-Type": "application/json" } }).catch(() => {});
      try { localStorage.setItem(_k, "1"); } catch {}
    }
  }
  initRadixToolkit();
  initWalletButton();
  fetchShopStatus();
  buildHowToPlay(TIERS, TIER_TRACKS);
  updateLeaderboardButton();
  updatePauseUI();
  // Wait for DOM layout to be computed before measuring canvas
  const nextLayout = () => new Promise(resolve => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      resolve();
    };
    requestAnimationFrame(finish);
    setTimeout(finish, 50);
  });
  await nextLayout();
  await nextLayout();
  resize();
  buildTierPanel();
  buildTrackList();
  updateTrackUI();
  updateMixerUI();
  syncMarketUi();
  updateUltiHud();
  loadBest();
  loadLeaderboards();
  loadTemporaryLeaderboards();
  await loadLivePrices();
  await loadRecentCandles();
  await loadGameplayConfig();
  await checkGlobalPause();
  await loadMarketTrend();
  await loadTierUnlocks();
  setLiveMode();
}

bootstrap();
setInterval(async ()=>{
  if (document.hidden) return;
  await loadGameplayConfig();
  await loadMarketTrend();
},600000);
// Returning to a foreground tab: refresh once immediately instead of waiting for the next interval.
// (Pause state now rides along in /live-state, so no dedicated pause-state loop is needed.)
document.addEventListener("visibilitychange", () => {
  if (document.hidden) return;
  if (livePoll) livePoll();
  if (state === "playing") { impactPollMs = IMPACT_POLL_MIN_MS; pollTradeImpacts(); }
});
startLoop(); // setInterval-based loop — works even when rAF is throttled

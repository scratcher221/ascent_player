// ASCENT — per-tier element visuals (handoff tiers 2..10), ported from
// "proto visuel/design_handoff_platforms". Self-contained module: it must not
// import anything from game.js (game.js imports from here).
//
// Four dispatch functions are exposed. Each returns true when a renderer is
// registered for the tier and has drawn the element, false so game.js falls
// back to the legacy rendering (missing renderer = automatic fallback, which
// is also the incremental-rollout safety net).
//
// Performance contract for renderers (validated handoff designs are heavier
// than the game budget allows per frame):
//   - createLinearGradient/createRadialGradient, shadowBlur and fillText are
//     ONLY allowed inside bake callbacks (getBaked) — never per frame.
//   - Per-frame work is limited to drawImage of baked layers/sprites plus
//     small fills/strokes (<= ~22 particles per element).
//   - Periodic vertical patterns (stream interiors, wavy walls) must be baked
//     as one spatial period and scrolled (drawVTile), not redrawn point by
//     point.

// Role colors — gameplay constants, never themed per biome.
export const ROLE_RED   = "#ff4d6d"; // resistance / sell
export const ROLE_GOLD  = "#ffcf5c"; // surge / bonus
export const ROLE_GOLD2 = "#ffdd88";

// Handoff biome palettes, indexed by handoff tier number (activeTier + 1).
export const VIS_TIERS = {
  2:  { hue: "#bfe6ff", rim: "#3f6fb8" }, // ZENITH
  3:  { hue: "#2ee6c8", rim: "#0a6b66" }, // ABYSS
  4:  { hue: "#6ef07a", rim: "#1f7a3a" }, // VERDANT
  5:  { hue: "#ffb55a", rim: "#a85f12" }, // DUNE
  6:  { hue: "#ff7e3a", rim: "#9c2a0a" }, // MAGMA
  7:  { hue: "#ff7ad4", rim: "#8a1f6e" }, // ORCHID
  8:  { hue: "#b48cff", rim: "#5a32a8" }, // VOLT
  9:  { hue: "#8a9bff", rim: "#2a2f8a" }, // NEBULA
  10: { hue: "#ffd45c", rim: "#c47a10" }, // APEX
};

// ── Shared low-level helpers ─────────────────────────────────────────────────

const _hexACache = new Map();
export function hexA(hex, a) {
  // Quantize alpha to 100 levels (visually indistinguishable) so the cache is
  // actually effective: renderers call this with continuous time-driven alphas,
  // which would otherwise fill the cache with one-shot keys every few seconds.
  a = a <= 0 ? 0 : a >= 1 ? 1 : ((a * 100 + 0.5) | 0) / 100;
  const key = hex + "|" + a;
  let v = _hexACache.get(key);
  if (v) return v;
  const h = hex.replace("#", "");
  v = `rgba(${parseInt(h.slice(0, 2), 16)},${parseInt(h.slice(2, 4), 16)},${parseInt(h.slice(4, 6), 16)},${a})`;
  if (_hexACache.size < 4000) _hexACache.set(key, v);
  return v;
}

function makeOffscreen(w, h) {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  return c;
}

// ── Bake caches ──────────────────────────────────────────────────────────────
// Static layers are baked at device resolution (CSS size × dpr) so cracks and
// thin edges stay crisp on high-density screens, then drawn back at CSS size.

let _bakeDpr = 1;

const BAKE_CAP = 64;
const _bakeCache = new Map(); // key -> { canvas, w, h, ...meta returned by draw }

// Bake a static layer once. `draw(g, w, h)` receives a context already scaled
// to CSS pixels; whatever object it returns is merged into the cache entry.
export function getBaked(key, w, h, draw) {
  let e = _bakeCache.get(key);
  if (e) return e;
  if (_bakeCache.size >= BAKE_CAP) _bakeCache.delete(_bakeCache.keys().next().value);
  const cw = Math.max(1, Math.ceil(w * _bakeDpr));
  const ch = Math.max(1, Math.ceil(h * _bakeDpr));
  const canvas = makeOffscreen(cw, ch);
  const g = canvas.getContext("2d");
  g.scale(_bakeDpr, _bakeDpr);
  const meta = draw(g, w, h) || {};
  e = { canvas, w, h, ...meta };
  _bakeCache.set(key, e);
  return e;
}

// Radial glow sprites — replaces the handoff glowAt() per-frame radial
// gradients (same idiom as game.js getGlowSprite, kept local to this module).
const GLOW_CAP = 24;
const _glowSprites = new Map(); // "color@r" -> canvas
export function getGlow(color, radius = 32) {
  const r = Math.max(8, Math.ceil(radius / 8) * 8);
  const key = color + "@" + r;
  let c = _glowSprites.get(key);
  if (!c) {
    if (_glowSprites.size >= GLOW_CAP) _glowSprites.delete(_glowSprites.keys().next().value);
    c = makeOffscreen(r * 2, r * 2);
    const g = c.getContext("2d");
    const grad = g.createRadialGradient(r, r, 0, r, r, r);
    // Quadratic-ish falloff (extra stops) for a softer halo edge — free at
    // runtime since the sprite is baked once.
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

// Additive radial halo at (x, y) with radius r — drop-in for handoff glowAt().
// Leaves composite op and alpha as it found them (no save/restore).
export function glow(ctx, x, y, r, color, a) {
  if (a <= 0 || r <= 0) return;
  const prevOp = ctx.globalCompositeOperation;
  const prevA  = ctx.globalAlpha;
  ctx.globalCompositeOperation = "lighter";
  ctx.globalAlpha = a;
  ctx.drawImage(getGlow(color), x - r, y - r, r * 2, r * 2);
  ctx.globalCompositeOperation = prevOp;
  ctx.globalAlpha = prevA;
}

// Baked stream-veil row: horizontal gradient (transparent → peak → transparent),
// constant in y, drawn stretched vertically over a corridor. Reusable per tier.
export function getVeilRow(key, color, peak) {
  const REF = 64;
  return getBaked("veil:" + key, REF, 1, (g, w) => {
    const lg = g.createLinearGradient(0, 0, w, 0);
    lg.addColorStop(0, hexA(color, 0));
    lg.addColorStop(0.5, hexA(color, peak));
    lg.addColorStop(1, hexA(color, 0));
    g.fillStyle = lg;
    g.fillRect(0, 0, w, 1);
  });
}

// Baked radial core sprite (bright center → color → transparent), authored at
// 16 px radius and drawn scaled. Replaces per-frame core gradients (gems,
// lanterns, molten cores, proto-stars).
export function getCoreSprite(key, c0, c1) {
  const R = 16;
  return getBaked("core:" + key, R * 2, R * 2, (g) => {
    const cg = g.createRadialGradient(R * 0.82, R * 0.82, 1, R, R, R);
    cg.addColorStop(0, c0);
    cg.addColorStop(0.7, c1);
    cg.addColorStop(1, hexA(c1, 0));
    g.fillStyle = cg;
    g.beginPath(); g.arc(R, R, R, 0, 7); g.fill();
  });
}

// Vertically scrolling tile: draws a baked one-period strip repeated over
// [topY, botY) at x, offset by `scroll` CSS px (world-anchored by the caller).
// Replaces per-frame periodic dot/streak loops with 2..N drawImage calls.
export function drawVTile(ctx, tile, x, topY, botY, scroll) {
  const period = tile.h;
  if (period <= 0 || botY <= topY) return;
  let y = topY - (((topY - scroll) % period) + period) % period;
  for (; y < botY; y += period) {
    const sy0 = Math.max(0, topY - y);
    const sy1 = Math.min(period, botY - y);
    if (sy1 <= sy0) continue;
    ctx.drawImage(tile.canvas,
      0, sy0 * _bakeDpr, tile.canvas.width, (sy1 - sy0) * _bakeDpr,
      x, y + sy0, tile.w, sy1 - sy0);
  }
}

// ── Wear language (uniform across tiers, validated) ──────────────────────────
// wear 0 = fresh, 1 = worn (alpha ×0.78 + first signs), 2 = critical (alert
// pulse). The pulse is a gameplay signal: it is never quality-gated, only
// slowed under reduced motion.
export function wAlpha(wear, t, reducedMotion) {
  if (wear <= 0) return 1;
  if (wear === 1) return 0.78;
  return 0.55 + 0.35 * Math.sin(t * (reducedMotion ? 4.5 : 9));
}

// Fine escaping particles (discreet wear signal — never covers the surface).
export function wmotes(ctx, cx, y, col, t, n, spread, up) {
  const prevOp = ctx.globalCompositeOperation;
  ctx.globalCompositeOperation = "lighter";
  for (let i = 0; i < n; i++) {
    const p = (t * 0.6 + i * 0.41) % 1;
    ctx.fillStyle = hexA(col, 0.45 * (1 - p));
    const yy = up ? y - 4 - p * 24 : y + 5 + p * 24;
    ctx.fillRect(cx - spread + i * (spread * 2 / Math.max(1, n - 1)) + Math.sin(i * 3 + t) * 2, yy, 1.6, 1.6);
  }
  ctx.globalCompositeOperation = prevOp;
}

// Zigzag lightning between two points (ZENITH, VOLT).
export function boltLine(ctx, x1, y1, x2, y2, col, a, seed) {
  const prevOp = ctx.globalCompositeOperation;
  ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = col.startsWith("rgba") ? col : hexA(col, a);
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  const seg = 4;
  for (let i = 1; i < seg; i++) {
    const f = i / seg;
    ctx.lineTo(x1 + (x2 - x1) * f + Math.sin(seed * 9.1 + i * 5.3) * 6,
               y1 + (y2 - y1) * f + Math.cos(seed * 7.7 + i * 3.9) * 4);
  }
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.globalCompositeOperation = prevOp;
}

// Smoothstep impact envelope: k = raw 0..1 linear countdown, returns the
// eased flare so sags release like a spring (matches orb.squash recovery).
export function impactEase(k) {
  if (k <= 0) return 0;
  if (k >= 1) return 1;
  return k * k * (3 - 2 * k);
}

// ── Renderer registry & dispatch ─────────────────────────────────────────────
// RENDERERS[tierNo] = { support(ctx,p), resist(ctx,p), surge(ctx,p), stream(ctx,p) }
// Populated by tier-visuals-t2-t5.js / tier-visuals-t6-t10.js as tiers land.

export const RENDERERS = {};

// p (support): { B, x, sy, width, cx, hw, t, wear, ga, impact, seed, gs, fx, platH, reducedMotion }
//   x/sy = left edge / screen Y of the top bounce surface (body extends down),
//   impact = eased 0..1 flare of the last bounce, ga = precomputed wAlpha.
export function drawTierSupport(ctx, tierNo, x, sy, width, t, opts) {
  const r = RENDERERS[tierNo];
  if (!r || !r.support) return false;
  const wear = opts.wear | 0;
  r.support(ctx, {
    B: VIS_TIERS[tierNo],
    x, sy, width, cx: x + width / 2, hw: width / 2, t,
    wear,
    ga: wAlpha(wear, t, opts.reducedMotion),
    impact: impactEase(opts.impact || 0),
    seed: opts.seed ?? (x * 0.37 + sy * 0.11),
    gs: opts.gs || 1,
    fx: opts.fx ?? 2,
    platH: opts.platH || 10,
    reducedMotion: !!opts.reducedMotion,
  });
  return true;
}

// p (resist): { B, cx, cy, w, h, hw, t, phase, gs, fx, reducedMotion }
//   Thin horizontal danger bar centered at (cx, cy); also reusable for sell
//   platforms (call with platform center / width).
export function drawTierResist(ctx, tierNo, cx, cy, w, h, t, opts) {
  const r = RENDERERS[tierNo];
  if (!r || !r.resist) return false;
  r.resist(ctx, {
    B: VIS_TIERS[tierNo],
    cx, cy, w, h, hw: w / 2, t,
    phase: opts.phase || 0,
    gs: opts.gs || 1,
    fx: opts.fx ?? 2,
    reducedMotion: !!opts.reducedMotion,
  });
  return true;
}

// p (surge): { B, cx, cy, r, t, phase, gs, fx, reducedMotion }
//   Glyphs are authored at ~16 px radius; renderers scale via r / 16.
export function drawTierSurge(ctx, tierNo, cx, cy, radius, t, opts) {
  const r = RENDERERS[tierNo];
  if (!r || !r.surge) return false;
  r.surge(ctx, {
    B: VIS_TIERS[tierNo],
    cx, cy, r: radius, t,
    phase: opts.phase || 0,
    gs: opts.gs || 1,
    fx: opts.fx ?? 2,
    reducedMotion: !!opts.reducedMotion,
  });
  return true;
}

// p (stream): { B, x1, x2, cx, w, topY, botY, t, phase, scroll, span, gs, fx, reducedMotion }
//   topY/botY are already clamped to the viewport; `scroll` is a world-anchored
//   CSS-px offset so patterns stay glued to the corridor while the camera moves;
//   `span` is the full (unclamped) corridor pixel height so wrap periods stay
//   constant while the corridor is only partly on screen.
export function drawTierStream(ctx, tierNo, x1, x2, topY, botY, t, opts) {
  const r = RENDERERS[tierNo];
  if (!r || !r.stream) return false;
  r.stream(ctx, {
    B: VIS_TIERS[tierNo],
    x1, x2, cx: (x1 + x2) / 2, w: x2 - x1, topY, botY, t,
    phase: opts.phase || 0,
    scroll: opts.scroll || 0,
    span: opts.span || 0,
    gs: opts.gs || 1,
    fx: opts.fx ?? 2,
    reducedMotion: !!opts.reducedMotion,
  });
  return true;
}

// ── Invalidation ─────────────────────────────────────────────────────────────
// Called from game.js on resize (dpr/gs/PLAT_H change), on quality dpr-cap
// change and on tier switch. Baked layers rebuild lazily on next use.
export function tierVisualsInvalidate(env) {
  if (env && Number.isFinite(env.dpr) && env.dpr > 0) _bakeDpr = env.dpr;
  _bakeCache.clear();
  _glowSprites.clear();
}

// Tier renderers register into RENDERERS via side-effect imports performed by
// game.js AFTER this module (so RENDERERS is initialized before they run —
// avoids a circular-import temporal dead zone).

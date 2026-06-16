// ASCENT — per-tier painted scenes: matte-painting base, bold parallax layers
// and a living light pass. Replaces tier-parallax.js. Direction: spectacular
// everywhere — the world is as present as the gameplay; the chart stays
// functional through its own reinforcement pass in game.js.
//
// Perf contract:
// - Pass A (base) and pass B (parallax tiles) are baked ONCE per tier into a
//   module-local cache of raw canvases at a fixed low resolution (REF=640 CSS,
//   device dpr deliberately ignored — the upscale blur is part of the look and
//   it bounds memory to ~1-3.5 MB per layer). Per frame they cost 1 + 2-4
//   drawImage calls.
// - Pass C (light) draws ≤8 small baked sprites per frame, modulated by
//   sin(now) — never a gradient per frame.
// - The cache purges every other tier's entries on tier switch (a run never
//   revisits a previous tier's scene), so at most one tier lives in memory.
// - Scene quality is frozen by the caller for the whole run (anti-flapping);
//   qLevel 2 = everything, 1 = one parallax layer + reduced light, no rotated
//   sprites (the rotated full-screen draws are the most expensive of pass C).

import { hexA, getGlow } from "./tier-visuals.js?v=tier-world-2";

const REF = 640; // baked CSS width for A/B layers; height follows aspect ratio

// ── Module-local scene cache (raw canvases, no dpr scaling) ──────────────────
const _scenes = new Map(); // "t<idx>|<part>|<hPx>" -> canvas
let _lastTier = -1;

function bakeRaw(tierIdx, part, w, h, paint) {
  const key = `t${tierIdx}|${part}|${h}`;
  let c = _scenes.get(key);
  if (!c) {
    c = document.createElement("canvas");
    c.width = w; c.height = h;
    paint(c.getContext("2d"), w, h);
    _scenes.set(key, c);
  }
  return c;
}

function purgeOthers(tierIdx) {
  const keep = `t${tierIdx}|`;
  for (const k of _scenes.keys()) if (!k.startsWith(keep)) _scenes.delete(k);
}

// ── Bake-time painting helpers (gradients are free inside bakes) ─────────────
function hash01(seed, i) {
  const x = Math.sin(seed * 127.1 + i * 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function vfill(g, w, h, stops) {
  const lg = g.createLinearGradient(0, 0, 0, h);
  for (const [p, c] of stops) lg.addColorStop(p, c);
  g.fillStyle = lg;
  g.fillRect(0, 0, w, h);
}

// Soft elliptical glow blob.
function blob(g, x, y, rx, ry, color, a) {
  g.save();
  g.translate(x, y);
  g.scale(1, Math.max(0.05, ry / rx));
  const rg = g.createRadialGradient(0, 0, 0, 0, 0, rx);
  rg.addColorStop(0, hexA(color, a));
  rg.addColorStop(1, hexA(color, 0));
  g.fillStyle = rg;
  g.beginPath(); g.arc(0, 0, rx, 0, 7); g.fill();
  g.restore();
}

// Horizontal glow band centered on y.
function band(g, w, y, hh, color, a) {
  const lg = g.createLinearGradient(0, y - hh, 0, y + hh);
  lg.addColorStop(0, hexA(color, 0));
  lg.addColorStop(0.5, hexA(color, a));
  lg.addColorStop(1, hexA(color, 0));
  g.fillStyle = lg;
  g.fillRect(0, y - hh, w, hh * 2);
}

// Star dust dots.
function stars(g, w, h, n, seed, color, aMax) {
  for (let i = 0; i < n; i++) {
    g.fillStyle = hexA(color, aMax * (0.3 + 0.7 * hash01(seed, i + 80)));
    const s = hash01(seed, i + 160) > 0.85 ? 1.6 : 1;
    g.fillRect(hash01(seed, i) * w, hash01(seed, i + 40) * h, s, s);
  }
}

// Bright rim along a polyline: one wide faint stroke + one thin bright stroke.
function rim(g, pts, color, a) {
  for (const [lw, aa] of [[3, a * 0.35], [1.2, a]]) {
    g.strokeStyle = hexA(color, aa);
    g.lineWidth = lw;
    g.beginPath();
    pts.forEach(([x, y], i) => i ? g.lineTo(x, y) : g.moveTo(x, y));
    g.stroke();
  }
}

// Vertical light shaft sprite (soft edges both axes), baked per tier so the
// purge keeps the cache single-tier.
function shaftSprite(tierIdx, name, color, peak) {
  return bakeRaw(tierIdx, "shaft-" + name, 96, 320, (g, w, h) => {
    const lx = g.createLinearGradient(0, 0, w, 0);
    lx.addColorStop(0, hexA(color, 0));
    lx.addColorStop(0.5, hexA(color, peak));
    lx.addColorStop(1, hexA(color, 0));
    g.fillStyle = lx;
    g.fillRect(0, 0, w, h);
    const ly = g.createLinearGradient(0, 0, 0, h);
    ly.addColorStop(0, "rgba(0,0,0,0)");
    ly.addColorStop(0.15, "rgba(0,0,0,0)");
    ly.addColorStop(1, "rgba(0,0,0,1)");
    g.globalCompositeOperation = "destination-out";
    g.fillStyle = ly;
    g.fillRect(0, 0, w, h);
  });
}

// Ray fan sprite (wedges radiating from center) for god rays / solar crowns.
function fanSprite(tierIdx, name, color, wedges, peak) {
  const S = 320, C = S / 2;
  return bakeRaw(tierIdx, "fan-" + name, S, S, (g) => {
    g.translate(C, C);
    for (let i = 0; i < wedges; i++) {
      const a0 = (i / wedges) * Math.PI * 2;
      const spread = 0.10 + 0.12 * hash01(9, i);
      const rg = g.createRadialGradient(0, 0, 10, 0, 0, C);
      rg.addColorStop(0, hexA(color, peak));
      rg.addColorStop(1, hexA(color, 0));
      g.fillStyle = rg;
      g.beginPath();
      g.moveTo(0, 0);
      g.arc(0, 0, C, a0 - spread, a0 + spread);
      g.closePath(); g.fill();
    }
  });
}

// Wavy molten/heat strip, horizontally periodic (integer sine periods) so two
// cross-scrolled copies wrap seamlessly.
function waveStrip(tierIdx, name, hPx, cTop, cBody, waves, aTop) {
  return bakeRaw(tierIdx, "wave-" + name, REF, hPx, (g, w, h) => {
    const lg = g.createLinearGradient(0, 0, 0, h);
    lg.addColorStop(0, hexA(cTop, aTop));
    lg.addColorStop(0.4, hexA(cBody, aTop * 0.75));
    lg.addColorStop(1, hexA(cBody, 0));
    g.fillStyle = lg;
    g.beginPath();
    g.moveTo(0, h);
    for (let x = 0; x <= w; x += w / 64) {
      const y = h * 0.3 + Math.sin((x / w) * Math.PI * 2 * waves) * h * 0.18
              + Math.sin((x / w) * Math.PI * 2 * (waves * 2 + 1)) * h * 0.07;
      g.lineTo(x, y);
    }
    g.lineTo(w, h);
    g.closePath(); g.fill();
  });
}

// Soft cloud sprite — a cluster of overlapping low-res puffs whose upscale at
// stamp time reads as vapor (the bake is tiny on purpose); the silhouette is
// bitten and both ends feathered so nothing reads as a stack of ovals.
function cloudSprite(tierIdx, seed, body, lit, litA) {
  return bakeRaw(tierIdx, "cloud-" + seed, 160, 72, (g, w, h) => {
    const cy = h * 0.64;
    const n = 9 + Math.floor(hash01(seed, 1) * 4);
    const puffs = [];
    for (let i = 0; i < n; i++) {
      const fx = i / (n - 1);
      const env = Math.sin(Math.PI * fx); // tall center, tapered ends
      const x = w * (0.10 + 0.80 * fx) + (hash01(seed, i + 5) - 0.5) * w * 0.05;
      const r = w * (0.050 + 0.085 * env * (0.55 + 0.65 * hash01(seed, i + 20)));
      puffs.push([x, cy - env * r * (0.3 + 0.9 * hash01(seed, i + 40)), r]);
    }
    blob(g, w * 0.5, cy + h * 0.03, w * 0.40, h * 0.14, body, 0.95); // unify base
    for (const [x, y, r] of puffs) blob(g, x, y, r, r * 0.85, body, 0.9);
    // Crest light, clipped to the body so it reads as one lit rim, not dots.
    g.globalCompositeOperation = "source-atop";
    for (const [x, y, r] of puffs)
      blob(g, x, y - r * 0.6, r * 1.1, r * 0.55, lit, litA * (0.5 + 0.5 * hash01(seed, x)));
    g.globalCompositeOperation = "destination-out";
    for (let k = 0; k < 3; k++) { // bites that break the silhouette
      const bx = w * (0.18 + 0.64 * hash01(seed, k + 60));
      const br = w * (0.045 + 0.05 * hash01(seed, k + 80));
      blob(g, bx, cy - h * (0.30 + 0.28 * hash01(seed, k + 70)), br, br * 0.7, "#000000", 0.75);
    }
    const fy = g.createLinearGradient(0, cy, 0, h); // dissolve the belly
    fy.addColorStop(0, "rgba(0,0,0,0)");
    fy.addColorStop(1, "rgba(0,0,0,0.9)");
    g.fillStyle = fy; g.fillRect(0, 0, w, h);
    const fx = g.createLinearGradient(0, 0, w, 0); // feather the ends
    fx.addColorStop(0, "rgba(0,0,0,0.95)");
    fx.addColorStop(0.2, "rgba(0,0,0,0)");
    fx.addColorStop(0.8, "rgba(0,0,0,0)");
    fx.addColorStop(1, "rgba(0,0,0,0.95)");
    g.fillStyle = fx; g.fillRect(0, 0, w, h);
    g.globalCompositeOperation = "source-over";
  });
}

// Draw a vertically wrapping parallax tile over the full viewport.
function drawWrapped(ctx, tile, W, H, scroll) {
  const s = W / REF;
  const th = tile.height * s;
  if (th <= 0) return;
  const y0 = ((scroll % th) + th) % th;
  for (let y = y0 - th; y < H; y += th) ctx.drawImage(tile, 0, y, W, th);
}

// Additive glow sprite at (x, y) radius r (shared cache from tier-visuals).
function lglow(ctx, x, y, r, color, a) {
  if (a <= 0) return;
  ctx.globalAlpha = a;
  ctx.drawImage(getGlow(color), x - r, y - r, r * 2, r * 2);
}

/* ════════════════════════════════════════════════════════════════════════════
   SCENES — per 0-based tier index: { base(g,w,h), layers[(g,w,h)], light(...) }
   Layer tiles are 1.5 viewport tall and must wrap vertically (full-height
   shapes use integer sine periods; floating shapes stay clear of tile edges).
   ════════════════════════════════════════════════════════════════════════════ */

const SCENES = {};

// ── T1 GENESIS · no scene on purpose ─────────────────────────────────────────
// Current players' first contact stays exactly as it always was (flat void
// gradient via the game.js fallback); the spectacle starts at ZENITH.

// ── T2 ZENITH · deep-twilight ascent through soft cloud banks, aurora ────────
SCENES[1] = {
  base(g, w, h) {
    vfill(g, w, h, [[0, "#030a18"], [0.5, "#081c38"], [0.74, "#0e2c52"],
                    [0.86, "#1b4a78"], [0.93, "#123a5e"], [1, "#050f1f"]]);
    band(g, w, h * 0.87, h * 0.05, "#7fb8e8", 0.18);
    stars(g, w, h * 0.6, 70, 23, "#dff0ff", 0.6);
    // Distant haze hugging the horizon glow (clipping at screen edges is fine
    // here — the base is never tiled).
    for (let i = 0; i < 3; i++) {
      const sw = w * (0.7 + 0.25 * hash01(21, i));
      g.globalAlpha = 0.40 + 0.15 * hash01(21, i + 9);
      g.drawImage(cloudSprite(1, 24 + i, "#122a4e", "#7fb8e8", 0.40),
        (hash01(21, i + 3) - 0.15) * w * 0.8, h * (0.74 + 0.07 * i), sw, sw * 0.45);
    }
    g.globalAlpha = 1;
  },
  layers: [
    (g, w, h) => { // far cloud veils drifting below the stars
      for (let i = 0; i < 5; i++) {
        const sw = w * (0.55 + 0.30 * hash01(25, i));
        const sh = sw * 0.45;
        const y0 = h * 0.05, y1 = h * 0.95 - sh; // margins keep the wrap seamless
        g.globalAlpha = 0.45 + 0.18 * hash01(25, i + 21);
        g.drawImage(cloudSprite(1, 31 + (i % 3), "#142e52", "#9fd0ff", 0.38),
          hash01(25, i + 14) * (w - sw * 0.8) - sw * 0.1,
          y0 + (y1 - y0) * ((i + 0.6 * hash01(25, i + 7)) / 5), sw, sh);
      }
      g.globalAlpha = 1;
    },
    (g, w, h) => { // near banks, aurora light on their crests
      for (let i = 0; i < 4; i++) {
        const sw = w * (0.75 + 0.35 * hash01(27, i));
        const sh = sw * 0.45;
        const y0 = h * 0.04, y1 = h * 0.96 - sh;
        g.globalAlpha = 0.70 + 0.20 * hash01(27, i + 15);
        g.drawImage(cloudSprite(1, 41 + (i % 2), "#1b3c66", "#bfe6ff", 0.45),
          hash01(27, i + 10) * (w - sw * 0.7) - sw * 0.15,
          y0 + (y1 - y0) * ((i + 0.7 * hash01(27, i + 5)) / 4), sw, sh);
      }
      g.globalAlpha = 1;
    },
  ],
  light(ctx, W, H, t, now, q, mob) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const cu = bakeRaw(1, "curtain", 90, 300, (g, w, h) => {
      const lg = g.createLinearGradient(0, 0, 0, h);
      lg.addColorStop(0, hexA("#3df0c8", 0));
      lg.addColorStop(0.25, hexA("#3df0c8", 0.30));
      lg.addColorStop(0.6, hexA("#5ac8ff", 0.16));
      lg.addColorStop(1, hexA("#5ac8ff", 0));
      g.fillStyle = lg;
      const lx = g.createLinearGradient(0, 0, w, 0);
      g.fillRect(0, 0, w, h);
      lx.addColorStop(0, "rgba(0,0,0,1)");
      lx.addColorStop(0.4, "rgba(0,0,0,0.25)");
      lx.addColorStop(0.6, "rgba(0,0,0,0.25)");
      lx.addColorStop(1, "rgba(0,0,0,1)");
      g.globalCompositeOperation = "destination-out";
      g.fillStyle = lx;
      g.fillRect(0, 0, w, h);
    });
    const n = (mob || q < 2) ? 2 : 3;
    for (let i = 0; i < n; i++) {
      const sway = Math.sin(now * 0.35 + i * 2.1) * W * 0.025;
      ctx.globalAlpha = 0.65 + 0.3 * Math.sin(now * 0.5 + i * 1.7);
      ctx.drawImage(cu, W * (0.18 + i * 0.3) + sway, -H * 0.02, W * 0.16, H * 0.6);
    }
    ctx.restore();
  },
};

// ── T3 ABYSS · sunlit surface far above, god rays into the deep ──────────────
SCENES[2] = {
  base(g, w, h) {
    vfill(g, w, h, [[0, "#0a4a56"], [0.16, "#06343f"], [0.5, "#02202a"], [1, "#000a0d"]]);
    band(g, w, h * 0.05, h * 0.06, "#7af0dc", 0.22);
    for (let i = 0; i < 5; i++) {
      blob(g, hash01(31, i) * w, h * (0.3 + 0.6 * hash01(31, i + 5)),
        w * (0.10 + 0.08 * hash01(31, i + 10)), w * 0.07, "#0a3a40", 0.5);
    }
    stars(g, w, h, 30, 33, "#9af0e0", 0.35); // drifting plankton glints
  },
  layers: [
    (g, w, h, t) => { // trench walls, bioluminescent speckles on the rim
      for (let i = 0; i < 2; i++) {
        const side = i ? w : 0, dir = side ? -1 : 1;
        const bw = w * (0.13 + 0.05 * hash01(35, i));
        g.fillStyle = "#03242c";
        g.beginPath(); g.moveTo(side, 0);
        const edge = [];
        for (let y = 0; y <= h; y += h / 28) {
          const x = side + dir * (bw + Math.sin((y / h) * Math.PI * 2 * (2 + i) + i * 9) * w * 0.03);
          g.lineTo(x, y); edge.push([x, y]);
        }
        g.lineTo(side, h); g.closePath(); g.fill();
        rim(g, edge, "#2ee6c8", 0.30);
        g.fillStyle = hexA("#2ee6c8", 0.55);
        for (let k = 0; k < 14; k++) {
          const [ex, ey] = edge[Math.floor(hash01(36, i * 40 + k) * (edge.length - 1))];
          g.fillRect(ex - dir * hash01(37, k) * bw * 0.6, ey, 1.6, 1.6);
        }
      }
    },
    (g, w, h) => { // kelp curtains swaying frozen mid-drift
      g.lineCap = "round";
      for (let i = 0; i < 7; i++) {
        const x0 = (i % 2 ? w - hash01(38, i) * w * 0.2 : hash01(38, i) * w * 0.2);
        g.strokeStyle = hexA(i % 3 ? "#052e30" : "#0a4a44", 0.9);
        g.lineWidth = 3 + hash01(38, i + 9) * 3;
        g.beginPath();
        for (let y = 0; y <= h; y += h / 24)
          g.lineTo(x0 + Math.sin((y / h) * Math.PI * 2 * (3 + (i % 2)) + i * 2) * w * 0.02, y);
        g.stroke();
        g.fillStyle = hexA("#1ad0b4", 0.4);
        for (let k = 0; k < 4; k++) {
          const y = ((k + hash01(39, i)) * h / 4) % h;
          g.fillRect(x0 + Math.sin((y / h) * Math.PI * 2 * (3 + (i % 2)) + i * 2) * w * 0.02, y, 2, 2);
        }
      }
    },
  ],
  light(ctx, W, H, t, now, q, mob) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const fan = fanSprite(2, "rays", "#aef0e4", 5, 0.13);
    const s = W * 1.1;
    ctx.globalAlpha = 0.5 + 0.25 * Math.sin(now * 0.4);
    if (q >= 2) {
      ctx.translate(W * 0.2, -H * 0.06);
      ctx.rotate(Math.sin(now * 0.1) * 0.06 + 0.5);
      ctx.drawImage(fan, -s / 2, -s / 2, s, s);
    } else {
      ctx.drawImage(fan, W * 0.2 - s / 2, -H * 0.06 - s / 2, s, s);
    }
    ctx.restore();
  },
};

// ── T4 VERDANT · cathedral canopy, gold light through the leaves ─────────────
SCENES[3] = {
  base(g, w, h) {
    vfill(g, w, h, [[0, "#0d4a20"], [0.25, "#07331a"], [0.6, "#04210f"], [1, "#02100a"]]);
    blob(g, w * 0.32, h * 0.05, w * 0.26, w * 0.12, "#ffe9a0", 0.13);
    blob(g, w * 0.68, h * 0.09, w * 0.2, w * 0.1, "#d8ffb0", 0.10);
    for (let i = 0; i < 6; i++) { // dark foliage masses framing the light
      blob(g, hash01(41, i) * w, h * (0.04 + 0.1 * hash01(41, i + 6)),
        w * (0.12 + 0.1 * hash01(41, i + 12)), w * 0.08, "#062a12", 0.9);
    }
  },
  layers: [
    (g, w, h, t) => { // thick bamboo, lit on one side
      for (let i = 0; i < 5; i++) {
        const x = i % 2 ? w - hash01(44, i + 3) * w * 0.24 : hash01(44, i + 3) * w * 0.24;
        const bw = 16 + hash01(44, i + 9) * 12;
        g.fillStyle = "#0c3318";
        g.fillRect(x, 0, bw, h);
        g.fillStyle = hexA("#9bff8a", 0.35);
        g.fillRect(x + bw - 2.5, 0, 2.5, h); // sun-side rim
        for (let k = 0; k < 6; k++) {
          const y = ((k + hash01(45, i)) * h / 6) % h;
          g.fillStyle = "#082512";
          g.fillRect(x - 2, y, bw + 4, 5);
          g.fillStyle = hexA("#bfff9a", 0.45);
          g.fillRect(x - 2, y, bw + 4, 1.5);
        }
      }
    },
    (g, w, h) => { // slimmer mid-ground stalks + hanging leaves
      for (let i = 0; i < 7; i++) {
        const x = i % 2 ? w - hash01(46, i + 3) * w * 0.3 : hash01(46, i + 3) * w * 0.3;
        const bw = 6 + hash01(46, i + 9) * 6;
        g.fillStyle = "#093017";
        g.fillRect(x, 0, bw, h);
        g.fillStyle = hexA("#6ef07a", 0.5);
        for (let k = 0; k < 5; k++) {
          const y = ((k + hash01(47, i)) * h / 5) % h;
          g.beginPath();
          g.ellipse(x + bw + 7, y, 9, 2.8, 0.5 - (i % 3) * 0.4, 0, 7);
          g.fill();
        }
      }
    },
  ],
  light(ctx, W, H, t, now, q, mob) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const sp = shaftSprite(3, "sun", "#e8ffc0", 0.14);
    const n = mob ? 1 : 2;
    for (let i = 0; i < n; i++) {
      ctx.globalAlpha = 0.5 + 0.3 * Math.sin(now * 0.45 + i * 2.6);
      if (q >= 2) {
        ctx.save();
        ctx.translate(W * (0.3 + i * 0.34), 0);
        ctx.rotate(0.16 - i * 0.3);
        ctx.drawImage(sp, -W * 0.07, -H * 0.05, W * 0.14, H * 0.8);
        ctx.restore();
      } else {
        ctx.drawImage(sp, W * (0.3 + i * 0.34) - W * 0.07, -H * 0.05, W * 0.14, H * 0.8);
      }
    }
    // Drifting light spot on the understory.
    lglow(ctx, W * (0.5 + 0.3 * Math.sin(now * 0.13)), H * (0.6 + 0.18 * Math.sin(now * 0.09 + 2)),
      W * 0.09, "#d8ffb0", 0.09);
    ctx.restore();
  },
};

// ── T5 DUNE · eternal sunset over a sea of sand ──────────────────────────────
SCENES[4] = {
  base(g, w, h) {
    vfill(g, w, h, [[0, "#241006"], [0.28, "#46200c"], [0.52, "#7a3a14"], [0.66, "#b05a1a"],
                    [0.72, "#d0742a"], [0.78, "#6a3210"], [1, "#160c04"]]);
    band(g, w, h * 0.71, h * 0.07, "#ffb55a", 0.4);
    blob(g, w * 0.62, h * 0.70, w * 0.16, w * 0.06, "#ffd9a0", 0.5); // sun seat
    for (let i = 0; i < 4; i++) { // birds-of-dust streaks in the warm sky
      blob(g, hash01(51, i) * w, h * (0.2 + 0.3 * hash01(51, i + 4)),
        w * (0.14 + 0.08 * hash01(51, i + 8)), w * 0.018, "#3a1c0a", 0.7);
    }
  },
  layers: [
    (g, w, h, t) => { // backlit dune crests
      for (let i = 0; i < 2; i++) {
        const cy = h * (0.3 + 0.32 * i + 0.06 * hash01(55, i));
        const th = h * 0.1, pts = [];
        g.fillStyle = i ? "#1a0d05" : "#2a1408";
        g.beginPath();
        for (let x = 0; x <= w; x += w / 40) {
          const y = cy + Math.sin((x / w) * Math.PI * 2 * (1 + i) + i * 5) * h * 0.035
                  + Math.sin((x / w) * Math.PI * 2 * (3 + i)) * h * 0.012;
          x === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
          pts.push([x, y]);
        }
        g.lineTo(w, cy + th); g.lineTo(0, cy + th);
        g.closePath(); g.fill();
        rim(g, pts, i ? "#ffcf80" : "#ffe2a8", i ? 0.3 : 0.5);
      }
    },
    (g, w, h) => { // near rolling crest
      const cy = h * 0.62, pts = [];
      g.fillStyle = "#140a04";
      g.beginPath();
      for (let x = 0; x <= w; x += w / 40) {
        const y = cy + Math.sin((x / w) * Math.PI * 2 * 2 + 1.3) * h * 0.05;
        x === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
        pts.push([x, y]);
      }
      g.lineTo(w, cy + h * 0.14); g.lineTo(0, cy + h * 0.14);
      g.closePath(); g.fill();
      rim(g, pts, "#ffb55a", 0.25);
    },
  ],
  light(ctx, W, H, t, now, q, mob) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const sx = W * 0.62, sy = H * 0.70;
    lglow(ctx, sx, sy, W * 0.06, "#fff4e2", 0.5 + 0.05 * Math.sin(now * 0.7));
    lglow(ctx, sx, sy, W * 0.2, "#ffcf80", 0.22 + 0.04 * Math.sin(now * 0.5 + 1));
    if (q >= 2 && !mob) { // heat shimmer above the horizon
      const sh = waveStrip(4, "heat", 22, "#ffd9a0", "#ffb55a", 6, 0.12);
      const off = ((now * 14) % W + W) % W;
      const hy = H * 0.66, hh = H * 0.045;
      ctx.globalAlpha = 0.8;
      ctx.drawImage(sh, off - W, hy, W, hh);
      ctx.drawImage(sh, off, hy, W, hh);
    }
    ctx.restore();
  },
};

// ── T6 MAGMA · volcanic hell, a living lava lake ─────────────────────────────
SCENES[5] = {
  base(g, w, h) {
    vfill(g, w, h, [[0, "#2c0a02"], [0.3, "#4a1206"], [0.55, "#6e1c08"], [0.7, "#3a0e04"],
                    [1, "#190502"]]);
    band(g, w, h * 0.6, h * 0.07, "#ff7e3a", 0.30);
    // Two distant cones on the horizon — pass C flickers their craters.
    for (const [cx, cw] of [[w * 0.24, w * 0.16], [w * 0.78, w * 0.12]]) {
      g.fillStyle = "#1e0703";
      g.beginPath();
      g.moveTo(cx - cw, h * 0.62);
      g.lineTo(cx - cw * 0.13, h * 0.45);
      g.lineTo(cx + cw * 0.13, h * 0.45);
      g.lineTo(cx + cw, h * 0.62);
      g.closePath(); g.fill();
      g.strokeStyle = hexA("#ff9a50", 0.5);
      g.lineWidth = 2;
      g.beginPath(); g.moveTo(cx - cw * 0.13, h * 0.45); g.lineTo(cx + cw * 0.13, h * 0.45); g.stroke();
    }
    for (let i = 0; i < 3; i++) { // smoke columns
      blob(g, hash01(61, i) * w, h * (0.3 + 0.1 * i),
        w * 0.06, w * 0.16, "#1c0602", 0.55);
    }
  },
  layers: [
    (g, w, h, t) => { // mid-range volcano ridge with lava veins
      const baseY = h * 0.5, pts = [];
      g.fillStyle = "#200a04";
      g.beginPath();
      for (let x = 0; x <= w; x += w / 44) {
        const y = baseY + Math.sin((x / w) * Math.PI * 2 * 2 + 1) * h * 0.06
                + Math.sin((x / w) * Math.PI * 2 * 5) * h * 0.02;
        x === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
        pts.push([x, y]);
      }
      g.lineTo(w, baseY + h * 0.16); g.lineTo(0, baseY + h * 0.16);
      g.closePath(); g.fill();
      rim(g, pts, "#ff9a50", 0.5);
      g.lineWidth = 1.4; // lava veins running down the flank
      for (let i = 0; i < 6; i++) {
        const [vx, vy] = pts[Math.floor(hash01(63, i) * (pts.length - 1))];
        g.strokeStyle = hexA("#ff6a36", 0.3 + 0.2 * hash01(63, i + 6));
        g.beginPath();
        g.moveTo(vx, vy);
        g.lineTo(vx + (hash01(63, i + 12) - 0.5) * 16, vy + h * 0.05);
        g.lineTo(vx + (hash01(63, i + 18) - 0.5) * 26, vy + h * 0.11);
        g.stroke();
      }
    },
    (g, w, h) => { // near jagged basalt ridge
      const baseY = h * 0.7, pts = [];
      g.fillStyle = "#140502";
      g.beginPath();
      for (let x = 0; x <= w; x += w / 30) {
        const y = baseY + Math.sin((x / w) * Math.PI * 2 * 3 + 0.6) * h * 0.045;
        x === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
        pts.push([x, y]);
      }
      g.lineTo(w, baseY + h * 0.12); g.lineTo(0, baseY + h * 0.12);
      g.closePath(); g.fill();
      rim(g, pts, "#ff7e3a", 0.28);
    },
  ],
  light(ctx, W, H, t, now, q, mob) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    // The lava lake — two cross-scrolling molten strips at the screen bottom.
    const s1 = waveStrip(5, "lava1", 54, "#ffd08a", "#ff6a36", 3, 0.5);
    const s2 = waveStrip(5, "lava2", 38, "#ffb060", "#e84a18", 5, 0.4);
    const h1 = H * 0.11, h2 = H * 0.075;
    const o1 = ((now * 22) % W + W) % W;
    ctx.globalAlpha = 0.85 + 0.15 * Math.sin(now * 1.7);
    ctx.drawImage(s1, o1 - W, H - h1, W, h1);
    ctx.drawImage(s1, o1, H - h1, W, h1);
    if (q >= 2) {
      const o2 = W - (((now * 34) % W + W) % W);
      ctx.globalAlpha = 0.7 + 0.2 * Math.sin(now * 2.3 + 1);
      ctx.drawImage(s2, o2 - W, H - h2, W, h2);
      ctx.drawImage(s2, o2, H - h2, W, h2);
    }
    lglow(ctx, W * 0.5, H, W * 0.4, "#ff6a36", 0.10 + 0.05 * Math.sin(now * 1.1));
    // Crater flickers on the two horizon cones (fixed base-A positions).
    lglow(ctx, W * 0.24, H * 0.45, W * 0.05, "#ff9a50", 0.12 + 0.10 * Math.abs(Math.sin(now * 5.3)));
    if (!mob) lglow(ctx, W * 0.78, H * 0.45, W * 0.04, "#ffd08a", 0.10 + 0.08 * Math.abs(Math.sin(now * 6.7 + 2)));
    ctx.restore();
  },
};

// ── T7 ORCHID · the giant night bloom over crystal fields ────────────────────
SCENES[6] = {
  base(g, w, h) {
    vfill(g, w, h, [[0, "#240820"], [0.3, "#3c0e32"], [0.55, "#581846"], [0.72, "#2c0a24"],
                    [1, "#10020c"]]);
    // The bloom: layered petals of light.
    const bx = w * 0.72, by = h * 0.2;
    blob(g, bx, by, w * 0.22, w * 0.2, "#ff8fde", 0.20);
    blob(g, bx, by, w * 0.1, w * 0.09, "#ffe9f7", 0.22);
    g.strokeStyle = hexA("#ff7ad4", 0.30);
    g.lineWidth = 2;
    for (let i = 0; i < 6; i++) { // petal arcs
      g.save();
      g.translate(bx, by);
      g.rotate(i * Math.PI / 3);
      g.beginPath(); g.ellipse(0, -w * 0.1, w * 0.045, w * 0.11, 0, 0, 7); g.stroke();
      g.restore();
    }
    stars(g, w, h, 30, 73, "#ffd0f0", 0.4);
  },
  layers: [
    (g, w, h, t) => { // monolithic crystals
      for (let i = 0; i < 4; i++) {
        const cx = i % 2 ? w - hash01(77, i) * w * 0.22 : hash01(77, i) * w * 0.22;
        const cy = h * (0.18 + 0.6 * hash01(77, i + 5));
        const ch = h * (0.1 + 0.07 * hash01(77, i + 9)), cw2 = ch * 0.22;
        const tilt = (hash01(77, i + 13) - 0.5) * 0.7;
        g.save(); g.translate(cx, cy); g.rotate(tilt);
        g.fillStyle = "#220a1e";
        g.beginPath();
        g.moveTo(0, -ch); g.lineTo(cw2, ch * 0.3); g.lineTo(0, ch * 0.5); g.lineTo(-cw2, ch * 0.3);
        g.closePath(); g.fill();
        g.strokeStyle = hexA("#ff7ad4", 0.5);
        g.lineWidth = 1.4;
        g.stroke();
        g.beginPath(); g.moveTo(0, -ch); g.lineTo(0, ch * 0.5); g.stroke(); // inner facet
        g.restore();
      }
    },
    (g, w, h) => { // shard scatter
      for (let i = 0; i < 8; i++) {
        const cx = i % 2 ? w - hash01(78, i) * w * 0.16 : hash01(78, i) * w * 0.16;
        const cy = h * (0.1 + 0.78 * hash01(78, i + 5));
        const ch = h * 0.035 * (0.5 + hash01(78, i + 9));
        g.save(); g.translate(cx, cy); g.rotate((hash01(78, i + 13) - 0.5) * 1.2);
        g.fillStyle = hexA("#3a1232", 0.95);
        g.beginPath();
        g.moveTo(0, -ch); g.lineTo(ch * 0.25, ch * 0.4); g.lineTo(-ch * 0.25, ch * 0.4);
        g.closePath(); g.fill();
        g.strokeStyle = hexA("#ff8fde", 0.35);
        g.lineWidth = 1;
        g.stroke();
        g.restore();
      }
    },
  ],
  light(ctx, W, H, t, now, q, mob) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const bx = W * 0.72, by = H * 0.2;
    lglow(ctx, bx, by, W * 0.2, "#ff8fde", 0.10 + 0.06 * Math.sin(now * 0.6));
    lglow(ctx, bx, by, W * 0.09, "#ffffff", 0.07 + 0.05 * Math.sin(now * 0.6 + Math.PI));
    if (q >= 2 && !mob) { // slow beams from the bloom
      const sp = shaftSprite(6, "beam", "#ff9ae0", 0.10);
      for (let i = 0; i < 2; i++) {
        ctx.save();
        ctx.translate(bx, by);
        ctx.rotate(2.4 + i * 0.5 + Math.sin(now * 0.2 + i) * 0.04);
        ctx.globalAlpha = 0.4 + 0.2 * Math.sin(now * 0.5 + i * 2);
        ctx.drawImage(sp, -W * 0.05, 0, W * 0.1, H * 0.75);
        ctx.restore();
      }
    }
    ctx.restore();
  },
};

// ── T8 VOLT · neon storm city ────────────────────────────────────────────────
SCENES[7] = {
  base(g, w, h) {
    vfill(g, w, h, [[0, "#160828"], [0.35, "#241040"], [0.6, "#321458"], [0.7, "#1c0a34"],
                    [1, "#080214"]]);
    band(g, w, h * 0.68, h * 0.05, "#b48cff", 0.4);
    g.fillStyle = hexA("#e8d8ff", 0.6);
    g.fillRect(0, h * 0.68, w, 1); // neon horizon core
    // Distant skyline silhouette under the neon line.
    g.fillStyle = "#0c0520";
    let x = 0;
    let i = 0;
    while (x < w) {
      const bw = w * (0.03 + 0.05 * hash01(81, i));
      const bh = h * (0.04 + 0.09 * hash01(81, i + 30));
      g.fillRect(x, h * 0.68 - bh, bw, bh + 2);
      x += bw + w * 0.012; i++;
    }
    blob(g, w * 0.5, h * 0.74, w * 0.5, w * 0.1, "#5a32a8", 0.25); // city glow
    stars(g, w, h * 0.5, 24, 83, "#d8c8ff", 0.4);
  },
  layers: [
    (g, w, h, t) => { // foreground circuit towers, neon edges + lit windows
      for (let i = 0; i < 3; i++) {
        const tw = w * (0.07 + 0.04 * hash01(88, i));
        const x = i % 2 ? w - tw - hash01(88, i + 4) * w * 0.06 : hash01(88, i + 4) * w * 0.06;
        g.fillStyle = "#140828";
        g.fillRect(x, 0, tw, h);
        g.fillStyle = hexA("#b48cff", 0.5);
        g.fillRect(i % 2 ? x : x + tw - 2, 0, 2, h); // neon edge facing center
        g.fillStyle = hexA("#c8a0ff", 0.5);
        for (let k = 0; k < 30; k++) {
          if (hash01(88, i * 100 + k) < 0.55) continue;
          g.fillRect(x + tw * (0.15 + 0.7 * hash01(88, i * 100 + k + 1)), (k + 0.5) * (h / 30), 3, 3);
        }
        g.strokeStyle = hexA("#5cc8ff", 0.3); // antenna cables
        g.lineWidth = 1;
        g.beginPath();
        g.moveTo(x + tw / 2, 0); g.lineTo(x + tw / 2 + (i % 2 ? -1 : 1) * w * 0.1, h * 0.5);
        g.stroke();
      }
    },
    (g, w, h) => { // far pylons
      for (let i = 0; i < 4; i++) {
        const tw = w * (0.025 + 0.02 * hash01(89, i));
        const x = i % 2 ? w - tw - hash01(89, i + 4) * w * 0.13 : hash01(89, i + 4) * w * 0.13;
        g.fillStyle = "#1a0c30";
        g.fillRect(x, 0, tw, h);
        g.fillStyle = hexA("#8a5cd8", 0.4);
        for (let k = 0; k < 8; k++) {
          const y = ((k + hash01(89, i + 8)) * h / 8) % h;
          g.fillRect(x - 2, y, tw + 4, 1.5);
        }
      }
    },
  ],
  light(ctx, W, H, t, now, q, mob, reducedMotion) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    // Neon horizon breathing.
    ctx.globalAlpha = 0.25 + 0.15 * Math.sin(now * 2);
    const nb = bakeRaw(7, "neonband", REF, 24, (g, w, h) => band(g, w, h / 2, h / 2, "#b48cff", 0.5));
    ctx.drawImage(nb, 0, H * 0.68 - H * 0.015, W, H * 0.03);
    // Storm flash — photosensitivity: fully disabled under reduced motion.
    if (q >= 2 && !reducedMotion) {
      const cell = Math.floor(now * 3);
      if (hash01(cell, 7) > 0.93) {
        const ph = now * 3 - cell;
        ctx.globalAlpha = 0.07 * Math.max(0, 1 - ph * 2.2);
        ctx.fillStyle = "#cfc0ff";
        ctx.fillRect(0, 0, W, H);
      }
    }
    lglow(ctx, W * 0.5, H * 0.72, W * 0.3, "#5a32a8", 0.08 + 0.04 * Math.sin(now * 1.3));
    ctx.restore();
  },
};

// ── T9 NEBULA · painted deep space, a breathing galaxy core ──────────────────
SCENES[8] = {
  base(g, w, h) {
    vfill(g, w, h, [[0, "#0a0626"], [0.4, "#120a38"], [0.7, "#0a0524"], [1, "#020108"]]);
    // Painted nebula fields.
    blob(g, w * 0.3, h * 0.3, w * 0.3, w * 0.2, "#7a3aa0", 0.20);
    blob(g, w * 0.72, h * 0.55, w * 0.26, w * 0.18, "#2a5a9a", 0.18);
    blob(g, w * 0.5, h * 0.75, w * 0.3, w * 0.14, "#a04a7a", 0.12);
    blob(g, w * 0.2, h * 0.65, w * 0.2, w * 0.16, "#4a3ac0", 0.15);
    // Galaxy core.
    blob(g, w * 0.68, h * 0.28, w * 0.12, w * 0.05, "#cfd8ff", 0.25);
    blob(g, w * 0.68, h * 0.28, w * 0.05, w * 0.045, "#ffffff", 0.30);
    stars(g, w, h, 110, 93, "#ffffff", 0.6);
    stars(g, w, h, 30, 94, "#9fd8ff", 0.5);
  },
  layers: [
    (g, w, h, t) => { // pillars of creation, rims lit by the core
      for (let i = 0; i < 2; i++) {
        const side = i % 2, cx = side ? w * 0.9 : w * 0.1;
        const rw = w * (0.07 + 0.03 * hash01(99, i));
        const rad = y => rw * (0.6 + 0.4 * Math.sin((y / h) * Math.PI * 2 * (2 + i) + i * 7));
        g.fillStyle = "#0c0830";
        g.beginPath();
        for (let k = 0; k <= 30; k++) {
          const y = k / 30 * h;
          k === 0 ? g.moveTo(cx - rad(y), y) : g.lineTo(cx - rad(y), y);
        }
        for (let k = 30; k >= 0; k--) {
          const y = k / 30 * h;
          g.lineTo(cx + rad(y), y);
        }
        g.closePath(); g.fill();
        // Core-facing rim + inner lobes.
        const rimPts = [];
        for (let k = 0; k <= 30; k++) {
          const y = k / 30 * h;
          rimPts.push([cx + (side ? -rad(y) : rad(y)), y]);
        }
        rim(g, rimPts, "#aab6ff", 0.40);
        for (let k = 0; k < 5; k++) {
          const y = ((k + hash01(98, i)) * h / 5) % h;
          blob(g, cx, y, rw * 0.7, rw * 0.5, "#1a1448", 0.8);
        }
      }
    },
    (g, w, h) => { // wisps of gas drifting mid-field
      for (let i = 0; i < 5; i++) {
        blob(g, hash01(97, i) * w, h * (0.08 + 0.84 * hash01(97, i + 5)),
          w * (0.1 + 0.08 * hash01(97, i + 10)), w * 0.03, "#221a55", 0.7);
      }
    },
  ],
  light(ctx, W, H, t, now, q, mob) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const cx = W * 0.68, cy = H * 0.28;
    lglow(ctx, cx, cy, W * 0.13, "#aab6ff", 0.10 + 0.05 * Math.sin(now * 0.5));
    lglow(ctx, cx, cy, W * 0.05, "#ffffff", 0.10 + 0.05 * Math.sin(now * 0.5 + Math.PI));
    // Occasional shooting star.
    const cell = Math.floor(now * 0.6);
    if (hash01(cell, 13) > 0.72) {
      const ph = now * 0.6 - cell;
      const a = Math.sin(Math.min(1, ph * 1.4) * Math.PI);
      const x0 = hash01(cell, 5) * W, y0 = hash01(cell, 9) * H * 0.5;
      const len = W * 0.1;
      ctx.globalAlpha = a * 0.8;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(x0 + ph * len * 2, y0 + ph * len);
      ctx.lineTo(x0 + ph * len * 2 - len * 0.5, y0 + ph * len - len * 0.25);
      ctx.stroke();
    }
    ctx.restore();
  },
};

// ── T10 APEX · the last sun, and the moon within reach ───────────────────────
SCENES[9] = {
  base(g, w, h) {
    vfill(g, w, h, [[0, "#1c1002"], [0.25, "#3a2206"], [0.5, "#5c380c"], [0.66, "#331d05"],
                    [1, "#0a0500"]]);
    const sx = w * 0.78, sy = h * 0.16;
    blob(g, sx, sy, w * 0.3, w * 0.26, "#ffd45c", 0.25);
    blob(g, sx, sy, w * 0.12, w * 0.11, "#fff4d0", 0.35);
    // Painted faint crepuscular wedges from the sun.
    g.fillStyle = hexA("#ffd45c", 0.07);
    for (let i = 0; i < 4; i++) {
      const a0 = 1.8 + i * 0.5;
      g.beginPath();
      g.moveTo(sx, sy);
      g.arc(sx, sy, w * 1.1, a0, a0 + 0.13);
      g.closePath(); g.fill();
    }
    for (let i = 0; i < 5; i++) { // gold-rimmed cloud bands
      const y = h * (0.35 + 0.45 * hash01(101, i));
      const x = hash01(101, i + 5) * w;
      blob(g, x, y, w * (0.16 + 0.1 * hash01(101, i + 10)), w * 0.025, "#241404", 0.85);
      blob(g, x, y - w * 0.012, w * (0.13 + 0.08 * hash01(101, i + 10)), w * 0.012, "#ffd45c", 0.30);
    }
    stars(g, w, h * 0.4, 24, 103, "#ffe9b0", 0.4);
  },
  layers: [
    (g, w, h, t) => { // drifting backlit cloud shelf
      for (let i = 0; i < 5; i++) {
        const x = hash01(105, i) * w, y = h * (0.1 + 0.8 * hash01(105, i + 5));
        const r = w * (0.14 + 0.1 * hash01(105, i + 10));
        blob(g, x, y, r, r * 0.18, "#1c1002", 0.95);
        blob(g, x - r * 0.15, y - r * 0.05, r * 0.7, r * 0.07, "#ffd45c", 0.30);
      }
    },
    (g, w, h) => { // gold dust veil
      for (let i = 0; i < 26; i++) {
        g.fillStyle = hexA(i % 3 ? "#ffd45c" : "#ffee88", 0.25 + 0.3 * hash01(107, i));
        g.fillRect(hash01(107, i) * w, hash01(107, i + 30) * h, 2, 2);
      }
    },
  ],
  light(ctx, W, H, t, now, q, mob, reducedMotion, cameraY) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const sx = W * 0.78, sy = H * 0.16;
    lglow(ctx, sx, sy, W * 0.09, "#fff4d0", 0.30 + 0.05 * Math.sin(now * 0.8));
    lglow(ctx, sx, sy, W * 0.22, "#ffd45c", 0.18 + 0.05 * Math.sin(now * 0.55 + 1));
    if (q >= 2) { // rotating solar crown
      const fan = fanSprite(9, "crown", "#ffe9a0", 9, 0.16);
      const s = W * 0.85;
      ctx.save();
      ctx.translate(sx, sy);
      ctx.rotate(reducedMotion ? 0 : now * 0.05);
      ctx.globalAlpha = 0.4;
      ctx.drawImage(fan, -s / 2, -s / 2, s, s);
      ctx.restore();
    }
    // Lens flare dots along the sun-center diagonal.
    if (!mob) {
      for (let i = 1; i <= 3; i++) {
        const fx = sx + (W * 0.4 - sx) * (i * 0.45);
        const fy = sy + (H * 0.55 - sy) * (i * 0.45);
        lglow(ctx, fx, fy, W * 0.015 * i, i % 2 ? "#ffee88" : "#fff4d0", 0.10);
      }
    }
    ctx.restore();
    // The moon — destination, growing as the run climbs. Drawn source-over so
    // it reads as a solid body, not a glow.
    const R = 72;
    const moon = bakeRaw(9, "moon", R * 2, R * 2, (g) => {
      g.fillStyle = "#262017";
      g.beginPath(); g.arc(R, R, R - 2, 0, 7); g.fill();
      g.fillStyle = "rgba(0,0,0,0.3)";
      for (let i = 0; i < 7; i++) {
        const a = hash01(7, i) * Math.PI * 2, d = (0.15 + 0.65 * hash01(7, i + 9)) * R;
        g.beginPath();
        g.arc(R + Math.cos(a) * d, R + Math.sin(a) * d, (0.06 + 0.1 * hash01(7, i + 20)) * R, 0, 7);
        g.fill();
      }
      g.strokeStyle = hexA("#ffee88", 0.25);
      g.lineWidth = 2.5;
      g.beginPath(); g.arc(R, R, R - 3, 0, 7); g.stroke();
    });
    const ms = R * (0.9 + Math.min(1.3, cameraY / (H * 50)));
    ctx.drawImage(moon, W * 0.22 - ms, H * 0.14 - ms, ms * 2, ms * 2);
  },
};

/* ════════════════════════════════════════════════════════════════════════════
   PUBLIC API
   ════════════════════════════════════════════════════════════════════════════ */

// qLevel here is the run-frozen scene level decided by game.js (≥1; the flat
// fallback at 0 stays in game.js). Layers: qLevel 2 → up to 2, else 1.
export function drawTierScene(ctx, tierIdx, t, W, H, cameraY, now, qLevel, isMobile, reducedMotion) {
  if (tierIdx !== _lastTier) { purgeOthers(tierIdx); _lastTier = tierIdx; }
  const S = SCENES[tierIdx];
  if (!S) return false;

  const aH = Math.max(2, Math.round(REF * H / W));
  const base = bakeRaw(tierIdx, "A", REF, aH, (g, w, h) => S.base(g, w, h, t));
  ctx.drawImage(base, 0, 0, W, H);

  const nL = Math.min(S.layers.length, qLevel >= 2 ? 2 : 1);
  const tileH = Math.round(aH * 1.5);
  for (let L = 0; L < nL; L++) {
    const tile = bakeRaw(tierIdx, "B" + L, REF, tileH, (g, w, h) => S.layers[L](g, w, h, t));
    drawWrapped(ctx, tile, W, H, cameraY * (L ? 0.14 : 0.07));
  }

  if (S.light) S.light(ctx, W, H, t, now, qLevel, isMobile, reducedMotion, cameraY);
  return true;
}

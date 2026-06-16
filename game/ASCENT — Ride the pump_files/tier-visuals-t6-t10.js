// ASCENT — tier visuals, handoff tiers 6..10 (MAGMA, ORCHID, VOLT, NEBULA, APEX).
// Renderers register into RENDERERS by handoff tier number. Ported from
// platforms-scenes-2.js / platforms-wear.js. See tier-visuals.js for the
// performance contract (no per-frame gradient/shadowBlur/fillText).

import {
  RENDERERS, hexA, getBaked, getVeilRow, getCoreSprite, glow, drawVTile, wmotes, boltLine,
  ROLE_RED, ROLE_GOLD, ROLE_GOLD2,
} from "./tier-visuals.js?v=tier-world-2";

// Deterministic 0..1 hash for seeded procedural layouts (NEBULA constellation).
function hash01(seed, i) {
  const x = Math.sin(seed * 127.1 + i * 311.7) * 43758.5453;
  return x - Math.floor(x);
}

// World-anchored rising field shared by tiers in this file (see t2-t5 twin).
// Wrap span = full corridor height (p.span) so it stays constant while the
// corridor is only partly on screen; culled to the clamped [p.topY, p.botY].
function risingField(cb, p, t, n, speed, kx) {
  const span = (p.span || (p.botY - p.topY)) + 40;
  if (span <= 0) return;
  for (let i = 0; i < n; i++) {
    const L = (((i * kx + t * speed) % span) + span) % span;
    const y = p.scroll - L;
    if (y < p.topY - 12 || y > p.botY + 12) continue;
    cb(i, y);
  }
}

// ── T6 · MAGMA ───────────────────────────────────────────────────────────────
// Body shapes are cheap solid fills/strokes; only the vertical gradients (blade
// body, convection veil) and the molten core radial are baked.

const MAGMA_SLAB = "#221008";
const MAGMA_EDGE = "#3c2012";
const MAGMA_RIM  = "#7a3a1a";

// Vertical blade gradient — constant across x, so baked as a 1px column and
// stretched to any width.
function magmaBladeColumn(h) {
  return getBaked("magma-blade@" + Math.round(h), 1, h, (g, w, hh) => {
    const lg = g.createLinearGradient(0, 0, 0, hh);
    lg.addColorStop(0, "#2c1410");
    lg.addColorStop(0.5, "#140a08");
    lg.addColorStop(1, "#2c1410");
    g.fillStyle = lg;
    g.fillRect(0, 0, w, hh);
  });
}

// Convection veil — horizontal gradient constant in y, baked as a 1px row and
// stretched vertically over the corridor.
function magmaVeilRow() {
  const REF = 64;
  return getBaked("magma-veil", REF, 1, (g, w) => {
    const lg = g.createLinearGradient(0, 0, w, 0);
    lg.addColorStop(0, hexA("#ff7e3a", 0));
    lg.addColorStop(0.5, hexA("#ff7e3a", 0.18));
    lg.addColorStop(1, hexA("#ff7e3a", 0));
    g.fillStyle = lg;
    g.fillRect(0, 0, w, 1);
    // Faint corridor edge lines so the convection column reads at a glance.
    g.fillStyle = hexA("#ff7e3a", 0.10);
    g.fillRect(0, 0, 1, 1);
    g.fillRect(w - 1, 0, 1, 1);
  });
}

// Molten core — white→gold radial baked once, drawn scaled by the live pulse.
function magmaCoreSprite() {
  const R = 16;
  return getBaked("magma-core", R * 2, R * 2, (g) => {
    const cg = g.createRadialGradient(R * 0.85, R * 0.85, 1, R, R, R);
    cg.addColorStop(0, "#fff8e0");
    cg.addColorStop(0.7, ROLE_GOLD);
    cg.addColorStop(1, hexA(ROLE_GOLD, 0));
    g.fillStyle = cg;
    g.beginPath(); g.arc(R, R, R, 0, 7); g.fill();
  });
}

// World-anchored rising embers for the convection column. `p.scroll` is the
// true (unclamped) screen Y of the corridor's world origin, so the field stays
// glued to the corridor as the camera moves; wrap span = full corridor height
// (p.span); drawing is culled to the clamped [p.topY, p.botY].
function risingEmbers(ctx, p, color, t, n, speed, spread, size) {
  const span = (p.span || (p.botY - p.topY)) + 40;
  if (span <= 0) return;
  ctx.fillStyle = color;
  for (let i = 0; i < n; i++) {
    const L = (((i * 61.7 + t * speed) % span) + span) % span;
    const y = p.scroll - L;
    if (y < p.topY - 6 || y > p.botY + 6) continue;
    const x = p.cx + Math.sin(i * 2.7 + y * 0.03) * spread;
    const r = size + (i % 3) * 0.7;
    ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
  }
}

RENDERERS[6] = {
  support(ctx, p) {
    const { B, x, sy, width, cx, hw, t, wear, impact, gs, fx, platH, reducedMotion } = p;
    const edgeH = Math.max(2, platH * 0.2);
    // Basalt slab — silhouette is always the full flat platform.
    ctx.save();
    if (fx > 0) glow(ctx, cx, sy + platH + 4 * gs, width * 0.5, B.hue, 0.08);
    ctx.fillStyle = MAGMA_SLAB;
    ctx.fillRect(Math.round(x), Math.round(sy), Math.round(width), platH);
    ctx.fillStyle = MAGMA_EDGE;
    ctx.fillRect(Math.round(x), Math.round(sy), Math.round(width), edgeH);
    // Warm contour so fresh slabs separate from the dark background.
    ctx.lineWidth = 1;
    ctx.strokeStyle = hexA(MAGMA_RIM, Math.min(1, 0.8 + 0.2 * impact));
    ctx.strokeRect(Math.round(x) + 0.5, Math.round(sy) + 0.5, Math.round(width) - 1, platH - 1);

    // Incandescent cracks — count/brightness grow with wear (the wear signal).
    const pulseHz = reducedMotion ? 4.5 : 8;
    const stepPx = (wear === 0 ? 27 : wear === 1 ? 15 : 12) * gs;
    const pad = 10 * gs;
    const hotBase = wear === 0 ? (impact > 0.01 ? 1 : 0.45 + 0.2 * Math.sin(t * 3))
                  : wear === 1 ? 0.7
                  : 0.55 + 0.45 * Math.abs(Math.sin(t * pulseHz));
    const cTop = sy + platH * 0.13, cMid = sy + platH * 0.5, cBot = sy + platH * 0.88;
    ctx.globalCompositeOperation = "lighter";
    ctx.lineWidth = wear === 2 ? 2 : 1.6;
    let i = 0;
    for (let cxk = x + pad; cxk < x + width - 6 * gs; cxk += stepPx, i++) {
      const hot = Math.min(1, hotBase * (0.8 + 0.2 * Math.sin(t * 5 + i * 1.7)));
      ctx.strokeStyle = hexA(B.hue, hot);
      ctx.beginPath();
      ctx.moveTo(cxk, cTop);
      ctx.lineTo(cxk + 5 * gs, cMid);
      ctx.lineTo(cxk - 2 * gs, cBot);
      ctx.stroke();
    }
    ctx.globalCompositeOperation = "source-over";

    // Internal glow rising with wear; escaping embers at critical.
    if (fx > 0 && wear >= 1) {
      glow(ctx, cx, sy + platH * 0.7, (wear === 1 ? 28 : 40) * gs, B.hue,
        wear === 1 ? 0.16 : 0.18 + 0.16 * Math.abs(Math.sin(t * pulseHz)));
    }
    if (fx > 0 && wear === 2) wmotes(ctx, cx, sy + platH, "#ff9a50", t, 6, 36 * gs, false);
    if (impact > 0.01) glow(ctx, cx, sy + platH * 0.4, 40 * gs, B.hue, 0.45 * impact);
    ctx.restore();
  },

  resist(ctx, p) {
    const { cx, cy, w, hw, t, gs, fx } = p;
    const bh = Math.max(4, gs * 7);
    ctx.save();
    if (fx > 0) glow(ctx, cx, cy, Math.max(28, hw * 0.85), ROLE_RED, 0.10 + 0.06 * Math.sin(t * 3));
    // Obsidian blade body (baked vertical gradient, stretched to width).
    const col = magmaBladeColumn(bh);
    ctx.drawImage(col.canvas, cx - hw, cy - bh / 2, w, bh);
    ctx.lineWidth = 1;
    ctx.strokeStyle = hexA(MAGMA_RIM, 0.8);
    ctx.strokeRect(cx - hw + 0.5, cy - bh / 2 + 0.5, w - 1, bh - 1);
    // Lit cutting edge.
    ctx.fillStyle = hexA(ROLE_RED, 0.75);
    ctx.fillRect(cx - hw, cy - bh / 2, w, Math.max(1, gs * 1.2));
    // Pulsing molten joints.
    ctx.globalCompositeOperation = "lighter";
    ctx.lineWidth = Math.max(1, gs * 1.5);
    for (let i = 0; i < 6; i++) {
      const jx = cx - hw + (i + 0.5) * w / 6;
      const hotj = 0.35 + 0.5 * Math.abs(Math.sin(t * 2.4 + i * 1.3));
      ctx.strokeStyle = hexA("#ff6a36", hotj);
      ctx.beginPath();
      ctx.moveTo(jx, cy - bh / 2);
      ctx.lineTo(jx + gs * 2, cy + bh / 2);
      ctx.stroke();
    }
    // Incandescent drops beading below the blade.
    if (fx > 0) {
      for (let i = 0; i < 3; i++) {
        const dx = cx - w * 0.26 + i * w * 0.26;
        const ph = (t * 0.7 + i * 0.33) % 1;
        if (ph < 0.5) {
          ctx.fillStyle = hexA("#ffd0a0", 0.8 * (1 - ph * 2));
          ctx.beginPath(); ctx.arc(dx, cy + bh / 2 + ph * 16 * gs, gs * 1.6, 0, 7); ctx.fill();
        }
      }
    }
    ctx.restore();
  },

  surge(ctx, p) {
    const { cx, cy, r, t, gs, fx } = p;
    const k = r / 16;
    ctx.save();
    // Dark serrated geode ring.
    ctx.fillStyle = "#160a08";
    ctx.beginPath();
    for (let i = 0; i <= 14; i++) {
      const a = i * (Math.PI * 2 / 14), rr = (20 + (i % 2) * 5) * k;
      const px = cx + Math.cos(a) * rr, py = cy + Math.sin(a) * rr;
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.closePath(); ctx.fill();
    // Gold halo + molten core (baked radial, scaled by live pulse).
    glow(ctx, cx, cy, 32 * k, ROLE_GOLD2, 0.6);
    const core = (8 + Math.sin(t * 4) * 1.5) * k;
    const cs = magmaCoreSprite();
    ctx.drawImage(cs.canvas, cx - core, cy - core, core * 2, core * 2);
    // Orbiting embers.
    if (fx > 0) {
      ctx.fillStyle = hexA("#ff9a50", 0.8);
      for (let i = 0; i < 5; i++) {
        const a = t * 1.6 + i * (Math.PI * 2 / 5);
        ctx.beginPath();
        ctx.arc(cx + Math.cos(a) * 30 * k, cy + Math.sin(a) * 20 * k, gs * 1.5, 0, 7);
        ctx.fill();
      }
    }
    ctx.restore();
  },

  stream(ctx, p) {
    const { cx, w, x1, topY, botY, scroll, t, gs, fx } = p;
    ctx.save();
    // Convection veil (baked horizontal gradient, stretched vertically).
    ctx.globalCompositeOperation = "lighter";
    const veil = magmaVeilRow();
    ctx.drawImage(veil.canvas, x1, topY, w, botY - topY);
    // Rising embers — world-anchored to the corridor.
    const n1 = fx >= 2 ? 14 : fx === 1 ? 9 : 0;
    const n2 = fx >= 2 ? 5 : fx === 1 ? 3 : 0;
    if (n1) risingEmbers(ctx, p, hexA("#ff9a50", 0.55), t, n1, 90, 16 * gs, gs);
    if (n2) risingEmbers(ctx, p, hexA("#fff0d0", 0.7), t + 3, n2, 120, 8 * gs, gs * 0.9);
    ctx.globalCompositeOperation = "source-over";
    // Fumarole mouth at the corridor base (only when on-screen).
    if (scroll >= topY - 10 && scroll <= botY + 10) {
      ctx.fillStyle = "#160a08";
      ctx.beginPath(); ctx.ellipse(cx, scroll - gs * 6, 30 * gs, 8 * gs, 0, 0, 7); ctx.fill();
    }
    ctx.restore();
  },
};

// ── T7 · ORCHID — prismatic bar / crystal spikes / gold moth / pollen veil ───

// Hexagonal prism body (horizontal rim→hue→rim gradient + facet edge), baked
// full-stretch and drawn with globalAlpha = ga.
function orchidPrism(hue, rim, prismH) {
  const REF = 116;
  return getBaked(`orchid-prism@${hue}|${rim}|${prismH}`, REF, prismH, (g, w, hh) => {
    const cham = 12, midY = hh / 2;
    const pg = g.createLinearGradient(0, 0, w, 0);
    pg.addColorStop(0, hexA(rim, 0.85));
    pg.addColorStop(0.5, hexA(hue, 0.6));
    pg.addColorStop(1, hexA(rim, 0.85));
    g.fillStyle = pg;
    g.beginPath();
    g.moveTo(0, midY); g.lineTo(cham, 0); g.lineTo(w - cham, 0);
    g.lineTo(w, midY); g.lineTo(w - cham, hh); g.lineTo(cham, hh);
    g.closePath(); g.fill();
    g.strokeStyle = hexA("#ffe6f7", 0.5); g.lineWidth = 1; g.stroke();
  });
}

// Downward red→pink crystal sprite (resist spikes), drawn scaled per spike.
function orchidCrystal() {
  return getBaked("orchid-crystal", 12, 26, (g) => {
    const cg = g.createLinearGradient(0, 0, 0, 26);
    cg.addColorStop(0, hexA(ROLE_RED, 0.75));
    cg.addColorStop(1, hexA("#ffb0c0", 0.95));
    g.fillStyle = cg;
    g.beginPath(); g.moveTo(1, 2); g.lineTo(6, 26); g.lineTo(11, 2); g.closePath(); g.fill();
  });
}

// Soft gold moth-wing sprite (radial), drawn mirrored and flap-scaled.
function orchidWing() {
  return getBaked("orchid-wing", 32, 28, (g) => {
    const wg = g.createRadialGradient(20, 12, 1, 18, 14, 18);
    wg.addColorStop(0, hexA("#fff8e0", 0.9));
    wg.addColorStop(1, hexA(ROLE_GOLD, 0));
    g.fillStyle = wg;
    g.beginPath(); g.ellipse(18, 12, 14, 12, 0, 0, 7); g.fill();
  });
}

// Helix phase sets, hoisted out of the stream renderer (selected by fx).
const HELICES_2 = [0, Math.PI * 2 / 3, Math.PI * 4 / 3];
const HELICES_1 = [0, Math.PI];
const HELICES_0 = [];

RENDERERS[7] = {
  support(ctx, p) {
    const { B, x, sy, width, cx, t, wear, ga, impact, gs, fx } = p;
    const prismH = Math.max(4, Math.round(gs * 10));
    ctx.save();
    // Prism body (baked), alpha carries the wear fade.
    ctx.globalAlpha = ga;
    ctx.drawImage(orchidPrism(B.hue, B.rim, prismH).canvas, x, sy, width, prismH);
    ctx.globalAlpha = 1;
    const sxScale = width / 116, yMid = sy + prismH / 2, yScale = prismH / 10;
    // Internal fractures (coherent with crystal) at wear.
    if (wear >= 1) {
      const crA = wear === 2 ? 0.35 + 0.3 * Math.abs(Math.sin(t * 10)) : 0.35;
      ctx.strokeStyle = hexA("#ffffff", crA); ctx.lineWidth = Math.max(0.8, gs * 0.8);
      ctx.beginPath();
      const fx0 = (lx, ly) => ctx.moveTo(cx + lx * sxScale, yMid + ly * yScale);
      const fxl = (lx, ly) => ctx.lineTo(cx + lx * sxScale, yMid + ly * yScale);
      fx0(-18, -5); fxl(-13, 0); fxl(-19, 5);
      fx0(8, -5); fxl(13, 1);
      if (wear === 2) {
        fx0(-38, -5); fxl(-34, 1); fxl(-40, 5);
        fx0(28, -5); fxl(33, 0); fxl(27, 5);
      }
      ctx.stroke();
    }
    // Refracting light sweep (baked band, moving; fades with wear).
    if (fx > 0) {
      const sweep = ((t * 0.6) % 1) * 2 - 1;
      const sa = (impact > 0.01 ? 0.6 : 0.3) * (wear === 0 ? 1 : wear === 1 ? 0.55 : 0.3);
      const bandW = 28 * gs;
      ctx.globalCompositeOperation = "lighter";
      ctx.globalAlpha = sa;
      ctx.drawImage(getVeilRow("orchid-sweep", "#ffffff", 1).canvas,
        cx + sweep * width * 0.4 - bandW / 2, sy, bandW, prismH);
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "source-over";
    }
    if (fx > 0 && wear === 2) wmotes(ctx, cx, sy, "#ffe6f7", t, 4, 30 * gs, false);
    ctx.restore();
  },

  resist(ctx, p) {
    const { cx, cy, w, hw, t, gs, fx } = p;
    const jit = Math.sin(t * 22) * 0.8 * gs;
    const barY = cy - 6 * gs;
    ctx.save();
    if (fx > 0) glow(ctx, cx, cy, Math.min(hw, 50 * gs), ROLE_RED, 0.11 + 0.06 * Math.sin(t * 4));
    // Thin vault.
    ctx.fillStyle = "#2a0612"; ctx.fillRect(cx - hw, barY, w, Math.max(3, gs * 6));
    ctx.fillStyle = hexA(ROLE_RED, 0.35); ctx.fillRect(cx - hw, barY + gs * 3, w, Math.max(1, gs));
    // Short crystals, points DOWN.
    const cr = orchidCrystal();
    const n = Math.max(3, Math.floor(w / (gs * 18)));
    for (let i = 0; i < n; i++) {
      const cxx = cx - hw + 6 * gs + i * (w - 12 * gs) / Math.max(1, n - 1) + jit * (i % 2 ? 1 : -1);
      const ch = (9 + (i % 3) * 5) * gs;
      ctx.drawImage(cr.canvas, cxx - 5 * gs, barY + gs * 2, 10 * gs, ch);
    }
    // Travelling glint.
    if (fx > 0) {
      const gi = Math.floor((t * 3) % n);
      glow(ctx, cx - hw + 6 * gs + gi * (w - 12 * gs) / Math.max(1, n - 1), barY + 8 * gs, 9 * gs, "#ffffff", 0.5);
    }
    ctx.restore();
  },

  surge(ctx, p) {
    const { cx, cy, r, t, gs, fx } = p;
    const k = r / 16;
    const flap = Math.abs(Math.sin(t * 7));
    ctx.save();
    glow(ctx, cx, cy, 32 * k, ROLE_GOLD2, 0.5);
    ctx.translate(cx, cy);
    // Gold-dusted wings (baked sprite, flap-scaled, mirrored).
    const wing = orchidWing();
    for (const s of [-1, 1]) {
      ctx.save();
      ctx.scale(s * (0.35 + 0.65 * flap) * k, k);
      ctx.globalCompositeOperation = "lighter";
      ctx.drawImage(wing.canvas, 2, -14, 28, 24);
      ctx.restore();
    }
    // Body + antennae.
    ctx.fillStyle = ROLE_GOLD;
    ctx.beginPath(); ctx.ellipse(0, 0, 2.6 * k, 8 * k, 0, 0, 7); ctx.fill();
    ctx.strokeStyle = hexA(ROLE_GOLD2, 0.8); ctx.lineWidth = Math.max(1, gs);
    ctx.beginPath();
    ctx.moveTo(0, -7 * k); ctx.quadraticCurveTo(-4 * k, -13 * k, -6 * k, -12 * k);
    ctx.moveTo(0, -7 * k); ctx.quadraticCurveTo(4 * k, -13 * k, 6 * k, -12 * k);
    ctx.stroke();
    ctx.restore();
    // Falling gold scales.
    if (fx > 0) {
      for (let i = 0; i < 5; i++) {
        const y = cy + 10 * k + ((i * 17.3 + t * 26) % 34) * k;
        ctx.fillStyle = hexA(ROLE_GOLD2, 0.6 * (1 - (y - cy - 10 * k) / (34 * k)));
        ctx.beginPath(); ctx.arc(cx - 10 * k + i * 5 * k + Math.sin(t * 3 + i) * 3 * k, y, gs * 1.1, 0, 7); ctx.fill();
      }
    }
  },

  stream(ctx, p) {
    const { B, cx, x1, w, topY, botY, t, gs, fx } = p;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    // Dense veil.
    ctx.drawImage(getVeilRow("orchid", B.hue, 0.12).canvas, x1, topY, w, botY - topY);
    // Three tight pollen helices.
    const helices = fx >= 2 ? HELICES_2 : fx === 1 ? HELICES_1 : HELICES_0;
    const per = fx >= 2 ? 14 : 8;
    for (const phse of helices) {
      risingField((i, y) => {
        const x = cx + Math.sin(y * 0.05 + t * 1.2 + phse) * 20 * gs;
        const front = Math.cos(y * 0.05 + t * 1.2 + phse) > 0;
        ctx.fillStyle = hexA(phse === 0 ? ROLE_GOLD2 : B.hue, front ? 0.65 : 0.28);
        ctx.beginPath(); ctx.arc(x, y, (front ? 1.9 : 1.2) * gs, 0, 7); ctx.fill();
      }, p, t, per, 65, 26.3 * gs);
    }
    ctx.restore();
  },
};

// ── T8 · VOLT — capacitor plate / overloaded rail / gold fuse / accel rail ───

// Terminal glyph ("+" / "−") baked once per (char, hue); drawn with live alpha.
function voltGlyph(ch, hue, size) {
  return getBaked(`volt-glyph|${ch}|${hue}|${size}`, size, size, (g) => {
    g.fillStyle = hue;
    g.font = `${size}px monospace`;
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText(ch, size / 2, size / 2 + 1);
  });
}

RENDERERS[8] = {
  support(ctx, p) {
    const { B, x, sy, width, cx, t, wear, impact, gs, fx, platH, reducedMotion } = p;
    ctx.save();
    // Plate body.
    ctx.fillStyle = "#0d0618";
    ctx.fillRect(Math.round(x), Math.round(sy), Math.round(width), platH);
    // Border: crisp → flickering → blinking with wear.
    const blinkHz = reducedMotion ? 5.5 : 11;
    const bA = wear === 0 ? 0.9 : wear === 1 ? 0.55 + 0.25 * Math.sin(t * 6) : 0.3 + 0.55 * Math.abs(Math.sin(t * blinkHz));
    ctx.strokeStyle = hexA(B.hue, bA); ctx.lineWidth = Math.max(1, gs * 1.5);
    ctx.strokeRect(Math.round(x), Math.round(sy), Math.round(width), platH);
    // +/- terminals (baked glyphs, live alpha).
    const gsz = Math.max(8, Math.round(gs * 11));
    const plus = voltGlyph("+", B.hue, gsz), minus = voltGlyph("−", B.hue, gsz);
    const my = sy + platH / 2;
    ctx.globalAlpha = 0.8;
    ctx.drawImage(plus.canvas, x + 5 * gs, my - gsz / 2, gsz, gsz);
    ctx.globalAlpha = wear >= 1 ? 0.35 + 0.3 * Math.sin(t * 8) : 0.8;
    ctx.drawImage(minus.canvas, x + width - 5 * gs - gsz, my - gsz / 2, gsz, gsz);
    ctx.globalAlpha = 1;
    // Residual glow running along the plate (slows with wear).
    if (fx > 0) {
      const spd = wear === 0 ? 80 : wear === 1 ? 50 : 26;
      const rx = x + ((t * spd) % Math.max(1, width));
      glow(ctx, rx, my, 8 * gs, B.hue, 0.35);
    }
    // Wear arcs + escaping motes.
    if (wear >= 1 && Math.sin(t * 9) > 0.4) boltLine(ctx, cx + 10 * gs, sy, cx + 18 * gs, sy - 16 * gs, B.hue, 0.7, t);
    if (wear === 2 && Math.sin(t * 13 + 2) > 0.2) boltLine(ctx, cx - 30 * gs, sy + platH, cx - 38 * gs, sy + platH + 12 * gs, B.hue, 0.7, t + 3);
    if (fx > 0 && wear === 2) wmotes(ctx, cx, sy + platH * 0.6, "#ffb0a0", t, 4, 30 * gs, false);
    // Contact arc on impact.
    if (impact > 0.2) {
      boltLine(ctx, cx, sy, cx - 14 * gs, sy - 26 * gs, B.hue, 0.9 * impact, t);
      boltLine(ctx, cx, sy, cx + 12 * gs, sy - 22 * gs, B.hue, 0.8 * impact, t + 5);
      glow(ctx, cx, sy, 32 * gs, B.hue, 0.5 * impact);
    }
    ctx.restore();
  },

  resist(ctx, p) {
    const { cx, cy, w, hw, t, gs, fx } = p;
    const bh = Math.max(4, gs * 8);
    ctx.save();
    if (fx > 0) glow(ctx, cx, cy, Math.min(hw, 55 * gs), ROLE_RED, 0.13 + 0.10 * Math.abs(Math.sin(t * 8)));
    // Short-circuited rail.
    ctx.fillStyle = "#1a0a10"; ctx.fillRect(cx - hw, cy - bh / 2, w, bh);
    ctx.strokeStyle = hexA(ROLE_RED, 0.8); ctx.lineWidth = Math.max(1, gs * 1.5);
    ctx.strokeRect(cx - hw, cy - bh / 2, w, bh);
    // Erratic arcs.
    for (let i = 0; i < 3; i++) {
      if (Math.sin(t * 11 + i * 4) > 0.1) {
        const ax = cx - hw + ((i * 43.7 * gs + Math.floor(t * 7) * 29 * gs) % (w - 8 * gs));
        boltLine(ctx, ax, cy - bh / 2, ax + 10 * gs, cy - bh / 2 - (18 + (i % 2) * 8) * gs, ROLE_RED, 0.85, t * 3 + i);
      }
    }
    // Ground sparks.
    ctx.fillStyle = hexA("#ffb0a0", 0.7);
    for (let i = 0; i < 4; i++) {
      const sx = cx - hw + ((i * 61.7 * gs + t * 120 * gs) % (w - 8 * gs));
      ctx.fillRect(sx, cy + bh / 2 + (i % 3) * 3 * gs, 1.5 * gs, 1.5 * gs);
    }
    ctx.restore();
  },

  surge(ctx, p) {
    const { B, cx, cy, r, t, gs, fx } = p;
    const k = r / 16;
    ctx.save();
    // Circuit tracks.
    ctx.strokeStyle = hexA(B.hue, 0.35); ctx.lineWidth = Math.max(1, gs * 1.2);
    for (const [dx, dy] of [[-1, -0.5], [1, -0.5], [-1, 0.6], [1, 0.6]]) {
      ctx.beginPath();
      ctx.moveTo(cx + dx * 16 * k, cy + dy * 14 * k);
      ctx.lineTo(cx + dx * 34 * k, cy + dy * 14 * k);
      ctx.lineTo(cx + dx * 34 * k, cy + dy * 34 * k);
      ctx.stroke();
      ctx.fillStyle = hexA(B.hue, 0.6);
      ctx.fillRect(cx + dx * 34 * k - 1.5 * gs, cy + dy * 34 * k - 1.5 * gs, 3 * gs, 3 * gs);
    }
    // Gold hex fuse (shadowBlur replaced by glow sprite).
    glow(ctx, cx, cy, 30 * k, ROLE_GOLD2, 0.55);
    const pul = 1 + 0.07 * Math.sin(t * 6);
    ctx.translate(cx, cy);
    ctx.scale(pul * k, pul * k);
    ctx.fillStyle = ROLE_GOLD;
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = Math.PI / 6 + i * Math.PI / 3;
      i === 0 ? ctx.moveTo(Math.cos(a) * 13, Math.sin(a) * 13) : ctx.lineTo(Math.cos(a) * 13, Math.sin(a) * 13);
    }
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = "#fff8e0";
    ctx.fillRect(-5, -1.5, 10, 3);
    ctx.restore();
  },

  stream(ctx, p) {
    const { B, cx, x1, x2, topY, botY, t, gs, fx } = p;
    ctx.save();
    // Live rails (double-stroke halo instead of shadowBlur).
    for (const rx of [x1, x2]) {
      ctx.strokeStyle = hexA(B.hue, 0.25); ctx.lineWidth = Math.max(3, gs * 5);
      ctx.beginPath(); ctx.moveTo(rx, topY); ctx.lineTo(rx, botY); ctx.stroke();
      ctx.strokeStyle = hexA(B.hue, 0.85); ctx.lineWidth = Math.max(1, gs * 2);
      ctx.beginPath(); ctx.moveTo(rx, topY); ctx.lineTo(rx, botY); ctx.stroke();
    }
    // Ion packets racing up the rails.
    ctx.globalCompositeOperation = "lighter";
    const n = fx >= 2 ? 8 : fx === 1 ? 5 : 0;
    risingField((i, y) => {
      const x = (i % 2) ? x1 : x2;
      ctx.fillStyle = hexA("#ffffff", 0.85);
      ctx.fillRect(x - 1.5 * gs, y, 3 * gs, 12 * gs);
      glow(ctx, x, y + 6 * gs, 9 * gs, B.hue, 0.5);
    }, p, t, n, 170, 53.7 * gs);
    // Occasional arc between rails.
    if (fx > 0 && Math.sin(t * 9) > 0.55) {
      const my = (topY + botY) / 2;
      boltLine(ctx, x1, my, x2, my + 6 * gs, B.hue, 0.5, t);
    }
    ctx.restore();
  },
};

// ── T9 · NEBULA — constellation / pulsar beam / proto-star / stellar flux ────

// Soft vertical sheath strip for the pulsar beam (baked once, stretched).
function nebulaSheath() {
  return getBaked("nebula-sheath", 2, 16, (g, w, hh) => {
    const sg = g.createLinearGradient(0, 0, 0, hh);
    sg.addColorStop(0, hexA(ROLE_RED, 0));
    sg.addColorStop(0.5, hexA(ROLE_RED, 0.16));
    sg.addColorStop(1, hexA(ROLE_RED, 0));
    g.fillStyle = sg; g.fillRect(0, 0, w, hh);
  });
}

// Scratch star-position buffers — reused each frame to avoid per-platform array
// allocations (the NEBULA support draws many stars; same idiom as _arcVisible).
const _nebSx = [], _nebSy = [];

RENDERERS[9] = {
  support(ctx, p) {
    const { B, x, sy, width, cx, hw, t, ga, wear, impact, seed, gs, fx } = p;
    const lit = Math.min(1, 0.3 + 0.7 * impact);
    // Procedural star layout across the real width (seeded so it's stable).
    const n = Math.max(3, Math.round(width / (22 * gs)));
    const amp = 11 * gs;
    const sxArr = _nebSx, syArr = _nebSy;
    sxArr.length = 0; syArr.length = 0;
    for (let i = 0; i < n; i++) {
      sxArr.push(x + (i + 0.5) * width / n);
      syArr.push(sy + (hash01(seed, i) - 0.5) * amp);
    }
    ctx.save();
    // Connecting line — pales/pulses with wear, all stars stay.
    ctx.strokeStyle = hexA(B.hue, (0.35 + lit * 0.55) * ga);
    ctx.lineWidth = Math.max(1, gs * 1.4);
    ctx.beginPath();
    for (let i = 0; i < n; i++) (i === 0 ? ctx.moveTo : ctx.lineTo).call(ctx, sxArr[i], syArr[i]);
    ctx.stroke();
    // Twinkling star nodes.
    for (let i = 0; i < n; i++) {
      const tw = wear === 2 ? 0.25 + 0.75 * Math.abs(Math.sin(t * 7 + i * 2.9))
        : (0.55 + 0.45 * Math.sin(t * 3 + i * 1.7)) * (wear === 1 ? 0.7 : 1);
      if (fx > 0) glow(ctx, sxArr[i], syArr[i], (8 + lit * 5) * gs * ga, "#ffffff", (0.25 + lit * 0.3) * ga);
      ctx.fillStyle = hexA("#ffffff", tw);
      ctx.beginPath(); ctx.arc(sxArr[i], syArr[i], (1.8 + (i % 2) * 0.8) * gs, 0, 7); ctx.fill();
      if (i % 2 === 0) {
        ctx.strokeStyle = hexA("#ffffff", tw * 0.5); ctx.lineWidth = Math.max(0.6, gs * 0.8);
        ctx.beginPath();
        ctx.moveTo(sxArr[i] - 4 * gs, syArr[i]); ctx.lineTo(sxArr[i] + 4 * gs, syArr[i]);
        ctx.moveTo(sxArr[i], syArr[i] - 4 * gs); ctx.lineTo(sxArr[i], syArr[i] + 4 * gs);
        ctx.stroke();
      }
    }
    ctx.restore();
  },

  resist(ctx, p) {
    const { cx, cy, w, hw, t, gs, fx } = p;
    const beat = Math.pow(Math.abs(Math.sin(t * 2.6)), 3);
    const sx = cx - hw; // dead star at the left end
    ctx.save();
    // Dead star.
    if (fx > 0) glow(ctx, sx, cy, (16 + beat * 14) * gs, ROLE_RED, 0.3 + beat * 0.3);
    const core = getCoreSprite("nebula-dead", "#ffd0d8", ROLE_RED);
    const cr = (4.5 + beat * 2) * gs;
    ctx.drawImage(core.canvas, sx - cr, cy - cr, cr * 2, cr * 2);
    // Beam: soft sheath + bright core line.
    ctx.globalCompositeOperation = "lighter";
    const bx0 = sx + 6 * gs, blen = (cx + hw) - bx0, shH = 16 * gs;
    ctx.globalAlpha = 0.6 + beat * 0.6;
    ctx.drawImage(nebulaSheath().canvas, bx0, cy - shH / 2, blen, shH);
    ctx.globalAlpha = 1;
    const bw = (1.5 + beat * 2.5) * gs;
    ctx.fillStyle = hexA("#ffd0d8", 0.7);
    ctx.fillRect(bx0, cy - bw / 2, blen, bw);
    // Waves emitted along the beam on each pulse.
    if (fx > 0) {
      ctx.lineWidth = Math.max(1, gs);
      for (let i = 0; i < 3; i++) {
        const ph = (t * 0.9 + i * 0.33) % 1;
        ctx.strokeStyle = hexA(ROLE_RED, 0.4 * (1 - ph));
        ctx.beginPath(); ctx.arc(sx + 10 * gs + ph * blen, cy, (4 + ph * 7) * gs, 0, 7); ctx.stroke();
      }
    }
    ctx.restore();
  },

  surge(ctx, p) {
    const { B, cx, cy, r, t, gs, fx } = p;
    const k = r / 16;
    ctx.save();
    // Accretion disk.
    ctx.translate(cx, cy);
    ctx.rotate(-0.3);
    for (let i = 0; i < 2; i++) {
      ctx.strokeStyle = hexA(i ? ROLE_GOLD2 : B.hue, 0.5 - i * 0.15);
      ctx.lineWidth = Math.max(1, gs * 1.5);
      ctx.beginPath(); ctx.ellipse(0, 0, (30 + i * 9) * k, (9 + i * 3) * k, 0, 0, 7); ctx.stroke();
    }
    if (fx > 0) {
      ctx.fillStyle = hexA(ROLE_GOLD2, 0.7);
      for (let i = 0; i < 9; i++) {
        const a = t * 1.4 + i * (Math.PI * 2 / 9);
        ctx.beginPath(); ctx.arc(Math.cos(a) * 32 * k, Math.sin(a) * 10 * k, gs * 1.3, 0, 7); ctx.fill();
      }
    }
    ctx.restore();
    // Proto-star core.
    glow(ctx, cx, cy, 34 * k, ROLE_GOLD2, 0.6);
    const cs = getCoreSprite("nebula-proto", "#ffffff", ROLE_GOLD);
    const core = (8 + Math.sin(t * 3)) * k;
    ctx.drawImage(cs.canvas, cx - core, cy - core, core * 2, core * 2);
  },

  stream(ctx, p) {
    const { B, cx, x1, x2, topY, botY, t, gs, fx } = p;
    ctx.save();
    // Corridor edges: two rows of fixed connected stars.
    for (const ex of [x1, x2]) {
      ctx.strokeStyle = hexA(B.hue, 0.30); ctx.lineWidth = Math.max(1, gs);
      ctx.beginPath(); ctx.moveTo(ex, topY); ctx.lineTo(ex, botY); ctx.stroke();
      const rows = 6, spanH = botY - topY;
      for (let i = 0; i < rows; i++) {
        const y = topY + 12 * gs + i * (spanH - 24 * gs) / (rows - 1);
        const tw = 0.5 + 0.5 * Math.sin(t * 3 + i * 2 + ex);
        ctx.fillStyle = hexA("#ffffff", 0.35 + tw * 0.45);
        ctx.beginPath(); ctx.arc(ex, y, 1.5 * gs, 0, 7); ctx.fill();
      }
    }
    // Ascending shooting stars (baked trail sprite + head).
    ctx.globalCompositeOperation = "lighter";
    const trail = getBaked("nebula-trail", 2, 24, (g, ww, hh) => {
      const tg = g.createLinearGradient(0, hh, 0, 0);
      tg.addColorStop(0, hexA("#ffffff", 0));
      tg.addColorStop(1, hexA("#ffffff", 0.8));
      g.fillStyle = tg; g.fillRect(0, 0, ww, hh);
    });
    const n = fx >= 2 ? 9 : fx === 1 ? 5 : 2;
    const trH = 22 * gs;
    risingField((i, y) => {
      const sx = cx + Math.sin(i * 2.7) * 20 * gs;
      ctx.drawImage(trail.canvas, sx - gs, y - trH, Math.max(1, gs * 1.8), trH);
      ctx.fillStyle = hexA("#ffffff", 0.9);
      ctx.beginPath(); ctx.arc(sx, y, 1.6 * gs, 0, 7); ctx.fill();
    }, p, t, n, 150, 47.3 * gs);
    ctx.restore();
  },
};

// ── T10 · APEX — moon slab / solar filament / corona / photon pillar ─────────

// Regolith slab body (vertical gradient), baked once and stretched to width.
function apexSlab(platH) {
  return getBaked(`apex-slab@${platH}`, 1, platH, (g, w, hh) => {
    const dg = g.createLinearGradient(0, 0, 0, hh);
    dg.addColorStop(0, "#3a3630");
    dg.addColorStop(1, "#1a1714");
    g.fillStyle = dg; g.fillRect(0, 0, w, hh);
  });
}

RENDERERS[10] = {
  support(ctx, p) {
    const { B, x, sy, width, cx, t, wear, impact, gs, fx, platH, reducedMotion } = p;
    ctx.save();
    // Lunar regolith slab.
    ctx.drawImage(apexSlab(platH).canvas, Math.round(x), Math.round(sy), Math.round(width), platH);
    // Lit edge (pales then pulses with wear).
    const pulseHz = reducedMotion ? 4.5 : 9;
    const topA = wear === 0 ? 0.45 : wear === 1 ? 0.3 : 0.15 + 0.25 * Math.abs(Math.sin(t * pulseHz));
    ctx.fillStyle = hexA(B.hue, topA);
    ctx.fillRect(Math.round(x), Math.round(sy), Math.round(width), Math.max(1, gs * 2.5));
    // Craters (positions scaled to the real width).
    const sxScale = width / 116, yC = sy + platH * 0.36;
    for (const [cx2, crr] of [[-34, 4], [-8, 3], [20, 4.5], [42, 2.5]]) {
      const ccx = cx + cx2 * sxScale, cr = crr * gs;
      ctx.fillStyle = "#14110e";
      ctx.beginPath(); ctx.ellipse(ccx, yC, cr, cr * 0.5, 0, 0, 7); ctx.fill();
      ctx.strokeStyle = hexA(B.hue, 0.25); ctx.lineWidth = Math.max(0.6, gs * 0.8);
      ctx.beginPath(); ctx.ellipse(ccx, yC - gs * 0.6, cr, cr * 0.5, 0, Math.PI, 0); ctx.stroke();
    }
    // Surface fissures (coherent regolith) at wear.
    if (wear >= 1) {
      ctx.strokeStyle = hexA("#0c0a08", 0.9); ctx.lineWidth = Math.max(1, gs * 1.1);
      const fy = (v) => sy + platH * (0.07 + v / 14 * 0.86);
      const fx0 = (lx, v) => ctx.moveTo(cx + lx * sxScale, fy(v));
      const fxl = (lx, v) => ctx.lineTo(cx + lx * sxScale, fy(v));
      ctx.beginPath();
      fx0(-44, 1); fxl(-40, 6); fxl(-45, 11);
      fx0(12, 1); fxl(8, 7); fxl(14, 12);
      if (wear === 2) {
        fx0(-20, 0); fxl(-16, 6); fxl(-22, 13);
        fx0(32, 1); fxl(36, 6); fxl(30, 12);
        fx0(-2, 0); fxl(2, 7);
      }
      ctx.stroke();
    }
    if (fx > 0 && wear >= 1) wmotes(ctx, cx, sy, "#d8d4c8", t, wear === 1 ? 3 : 6, 28 * gs, true);
    if (impact > 0.05 && fx > 0) {
      ctx.globalCompositeOperation = "lighter";
      for (let i = 0; i < 7; i++) {
        const dx = cx - 30 * gs + i * 10 * gs, dy = sy - 4 * gs - ((t * 16 + i * 7) % 22) * gs;
        ctx.fillStyle = hexA("#d8d4c8", 0.4 * impact);
        ctx.beginPath(); ctx.arc(dx + Math.sin(t + i) * 2 * gs, dy, gs, 0, 7); ctx.fill();
      }
      ctx.globalCompositeOperation = "source-over";
    }
    ctx.restore();
  },

  resist(ctx, p) {
    const { cx, cy, w, hw, t, gs, fx } = p;
    const snap = Math.pow(Math.max(0, Math.sin(t * 1.8)), 5);
    const x0 = cx - hw + w * 0.08, x1 = cx + hw - w * 0.08;
    ctx.save();
    if (fx > 0) glow(ctx, cx, cy, Math.min(hw, 40 * gs), ROLE_RED, 0.10 + snap * 0.22);
    // Thin undulating plasma filament (two passes).
    ctx.globalCompositeOperation = "lighter";
    for (const [a, lw] of [[0.35 + snap * 0.3, 4], [0.85, 1.6]]) {
      ctx.strokeStyle = hexA(a > 0.5 ? "#ffd0c0" : ROLE_RED, a);
      ctx.lineWidth = lw * gs;
      ctx.beginPath();
      for (let x = x0; x <= x1; x += 5 * gs) {
        const y = cy + Math.sin(x * 0.07 + t * 4) * 4 * gs + Math.sin(x * 0.18 - t * 7) * 2 * gs * (1 + snap);
        x <= x0 + 5 * gs ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    // Ejected particles when it snaps.
    if (fx > 0 && snap > 0.5) {
      for (let i = 0; i < 5; i++) {
        const px = x0 + i * (x1 - x0) / 4;
        ctx.fillStyle = hexA("#ffb0a0", 0.8 * snap);
        ctx.beginPath();
        ctx.arc(px, cy + Math.sin(px * 0.07 + t * 4) * 4 * gs - (snap * 10) * gs * (i % 2 ? 1 : -1), 1.4 * gs, 0, 7);
        ctx.fill();
      }
    }
    ctx.globalCompositeOperation = "source-over";
    // End anchors.
    for (const ax of [x0, x1]) {
      ctx.fillStyle = "#2a1404";
      ctx.beginPath(); ctx.arc(ax, cy, 3.5 * gs, 0, 7); ctx.fill();
      ctx.strokeStyle = hexA(ROLE_RED, 0.6); ctx.lineWidth = 1; ctx.stroke();
    }
    ctx.restore();
  },

  surge(ctx, p) {
    const { cx, cy, r, t, gs, fx } = p;
    const k = r / 16;
    ctx.save();
    glow(ctx, cx, cy, 46 * k, ROLE_GOLD2, 0.55);
    // Rotating crown rays.
    ctx.translate(cx, cy);
    ctx.save();
    ctx.rotate(t * 0.6);
    ctx.strokeStyle = hexA(ROLE_GOLD2, 0.7); ctx.lineWidth = Math.max(1, gs * 1.6);
    for (let i = 0; i < 10; i++) {
      const a = i * (Math.PI * 2 / 10), fl = 14 + (i % 2) * 5 + Math.sin(t * 4 + i) * 2;
      ctx.beginPath();
      ctx.moveTo(Math.cos(a) * 13 * k, Math.sin(a) * 13 * k);
      ctx.lineTo(Math.cos(a) * fl * k, Math.sin(a) * fl * k);
      ctx.stroke();
    }
    ctx.restore();
    // Prominence arc.
    ctx.strokeStyle = hexA(ROLE_GOLD, 0.6); ctx.lineWidth = Math.max(1, gs * 1.4);
    ctx.beginPath(); ctx.arc(13 * k, -4 * k, 7 * k, Math.PI * 1.2, Math.PI * 0.2); ctx.stroke();
    ctx.restore();
    // Sun disk.
    const cs = getCoreSprite("apex-sun", "#ffffff", ROLE_GOLD);
    ctx.drawImage(cs.canvas, cx - 11 * k, cy - 11 * k, 22 * k, 22 * k);
  },

  stream(ctx, p) {
    const { B, cx, x1, w, topY, botY, t, gs, fx } = p;
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    // Photon pillar veil.
    ctx.drawImage(getVeilRow("apex", "#fff0c0", 0.20).canvas, x1, topY, w, botY - topY);
    // Ascending golden rings.
    risingField((i, y) => {
      const sc = 0.8 + 0.3 * Math.sin(i * 2.1);
      ctx.strokeStyle = hexA(ROLE_GOLD2, 0.65); ctx.lineWidth = Math.max(1, gs * 2);
      ctx.beginPath(); ctx.ellipse(cx, y, 16 * sc * gs, 5 * sc * gs, 0, 0, 7); ctx.stroke();
    }, p, t, fx >= 1 ? 5 : 2, 90, 53.7 * gs);
    // Rising photon motes.
    ctx.fillStyle = hexA("#fff0c0", 0.7);
    risingField((i, y) => {
      const x = cx + Math.sin(i * 2.7 + y * 0.03) * 8 * gs;
      ctx.beginPath(); ctx.arc(x, y, gs * (1 + (i % 3) * 0.5), 0, 7); ctx.fill();
    }, p, t, fx >= 2 ? 6 : 3, 110, 61.7 * gs);
    ctx.restore();
  },
};

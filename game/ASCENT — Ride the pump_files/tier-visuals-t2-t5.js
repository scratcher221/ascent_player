// ASCENT — tier visuals, handoff tiers 2..5 (ZENITH, ABYSS, VERDANT, DUNE).
// Ported from platforms-scenes-1.js / platforms-wear.js. See tier-visuals.js
// for the performance contract (no per-frame gradient/shadowBlur/fillText).

import {
  RENDERERS, hexA, getBaked, getVeilRow, getCoreSprite, glow, wmotes, boltLine,
  ROLE_RED, ROLE_GOLD, ROLE_GOLD2,
} from "./tier-visuals.js?v=tier-world-2";

// World-anchored rising field (bubbles / dots / threads). `p.scroll` is the
// true screen Y of the corridor's world origin so the field stays glued to the
// corridor as the camera moves; the wrap span is the full corridor height
// (`p.span`) so it stays constant while the corridor is only partly on screen;
// drawing is culled to the clamped window [p.topY, p.botY].
function risingField(cb, p, t, n, speed, kx) {
  const span = (p.span || (p.botY - p.topY)) + 40;
  if (span <= 0) return;
  for (let i = 0; i < n; i++) {
    const L = (((i * kx + t * speed) % span) + span) % span;
    const y = p.scroll - L;
    if (y < p.topY - 8 || y > p.botY + 8) continue;
    cb(i, y);
  }
}

// ── T2 · ZENITH — halo gateway / lightning line / dawn star / updraft ────────

// Bright light bar (horizontal hue→white→hue gradient + baked self-glow),
// drawn full-stretch (a normalized horizontal gradient is width-invariant).
function zenithBar(hue, barH, liseraH, padV) {
  const REF = 120;
  const tileH = barH + liseraH + padV * 2;
  return getBaked(`zenith-bar@${hue}|${barH}|${liseraH}|${padV}`, REF, tileH, (g, w) => {
    const surfaceY = padV;
    g.shadowColor = hue;
    g.shadowBlur = 8;
    const grad = g.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, hexA(hue, 0.2));
    grad.addColorStop(0.5, "#ffffff");
    grad.addColorStop(1, hexA(hue, 0.2));
    g.fillStyle = grad;
    g.fillRect(0, surfaceY + liseraH, w, barH);
    g.shadowBlur = 0;
    g.fillStyle = hexA("#ffffff", 0.85);
    g.fillRect(w * 0.08, surfaceY, w * 0.84, liseraH);
    return { surfaceY, tileH };
  });
}

RENDERERS[2] = {
  support(ctx, p) {
    const { B, x, sy, width, cx, t, wear, ga, impact, gs, fx } = p;
    const barH = Math.max(3, gs * 4), liseraH = Math.max(1, gs * 1.5), padV = Math.max(6, gs * 8);
    const flare = impact;
    const haloDim = wear === 0 ? 1 : wear === 1 ? 0.45 : 0.18; // halo dies first
    ctx.save();
    // Soft halo below the bar — flares on impact, dims with wear.
    if (fx > 0) {
      glow(ctx, cx, sy + gs * 4, (40 + flare * 14) * gs, B.hue, (0.20 + flare * 0.18) * haloDim);
      ctx.globalCompositeOperation = "lighter";
      ctx.strokeStyle = hexA(B.hue, (0.35 + flare * 0.3) * haloDim);
      ctx.lineWidth = Math.max(1, gs * 1.2);
      ctx.beginPath();
      ctx.ellipse(cx, sy + gs * 6, Math.min(width * 0.5, 46 * gs), 8 * gs, 0, 0, 7);
      ctx.stroke();
      ctx.globalCompositeOperation = "source-over";
    }
    // Always-complete light bar (baked), alpha carries the wear pulse.
    const bar = zenithBar(B.hue, barH, liseraH, padV);
    ctx.globalAlpha = ga;
    ctx.drawImage(bar.canvas, x, sy - padV, width, bar.tileH);
    ctx.globalAlpha = 1;
    if (fx > 0 && wear >= 1) wmotes(ctx, cx, sy, "#ffffff", t, wear === 1 ? 3 : 5, 34 * gs, false);
    ctx.restore();
  },

  resist(ctx, p) {
    const { cx, cy, w, hw, t, gs, fx } = p;
    const seedT = Math.floor(t * 9);
    ctx.save();
    // Crystallized storm nodes at the ends.
    for (const s of [-1, 1]) {
      const nx = cx + s * hw * 0.8, jit = Math.sin(t * 6 + s * 3) * 0.6 * gs;
      ctx.save();
      ctx.translate(nx, cy + jit);
      ctx.rotate(s * 0.25);
      ctx.scale(gs, gs);
      if (fx > 0) glow(ctx, 0, 0, 20, ROLE_RED, 0.20 + 0.08 * Math.sin(t * 4 + s));
      ctx.fillStyle = "#101a2e";
      ctx.strokeStyle = hexA(ROLE_RED, 0.75);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(0, -15); ctx.lineTo(9, -5); ctx.lineTo(6, 9); ctx.lineTo(-5, 12); ctx.lineTo(-10, -2);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = hexA(ROLE_RED, 0.35);
      ctx.beginPath();
      ctx.moveTo(0, -15); ctx.lineTo(-1, 4);
      ctx.moveTo(9, -5); ctx.lineTo(-1, 4); ctx.lineTo(-5, 12);
      ctx.stroke();
      ctx.fillStyle = hexA("#ffd0d8", 0.5 + 0.45 * Math.sin(t * 7 + s * 2));
      ctx.beginPath(); ctx.arc(-1, 1, 2.2, 0, 7); ctx.fill();
      ctx.restore();
      if (Math.sin(t * 13 + s * 5) > 0.55) boltLine(ctx, nx, cy + jit, nx - s * 14 * gs, cy, ROLE_RED, 0.6, t + s);
    }
    // Continuous serpenting lightning line.
    if (fx > 0) glow(ctx, cx, cy, Math.min(hw, 55 * gs), ROLE_RED, 0.14 + 0.10 * Math.abs(Math.sin(t * 9)));
    ctx.globalCompositeOperation = "lighter";
    const x0 = cx - hw + 18 * gs, x1 = cx + hw - 18 * gs;
    for (const [a, lw] of [[0.9, 2.2], [0.4, 4.5]]) {
      ctx.strokeStyle = hexA(a > 0.5 ? "#ffd0d8" : ROLE_RED, a);
      ctx.lineWidth = lw * gs;
      ctx.beginPath();
      ctx.moveTo(x0, cy);
      const seg = 9;
      for (let i = 1; i <= seg; i++) {
        const xx = x0 + (x1 - x0) * (i / seg);
        const yy = cy + Math.sin(seedT * 7.7 + i * 13.3) * 7 * gs;
        ctx.lineTo(xx, i === seg ? cy : yy);
      }
      ctx.stroke();
    }
    ctx.globalCompositeOperation = "source-over";
    if (Math.sin(t * 8) > 0.4) boltLine(ctx, cx - hw * 0.16, cy, cx - hw * 0.2, cy + 22 * gs, ROLE_RED, 0.7, t);
    if (Math.sin(t * 11 + 2) > 0.5) boltLine(ctx, cx + hw * 0.24, cy, cx + hw * 0.32, cy - 20 * gs, ROLE_RED, 0.6, t + 4);
    ctx.restore();
  },

  surge(ctx, p) {
    const { B, cx, cy, r, t, gs, fx } = p;
    const k = r / 16;
    ctx.save();
    // Breathing aurora rings.
    if (fx > 0) {
      ctx.globalCompositeOperation = "lighter";
      for (let i = 0; i < 3; i++) {
        const rr = (26 + i * 11 + Math.sin(t * 1.6 + i) * 3) * k;
        ctx.strokeStyle = hexA(B.hue, 0.28 - i * 0.07);
        ctx.lineWidth = Math.max(1, gs * 1.5);
        ctx.beginPath(); ctx.ellipse(cx, cy, rr, rr * 0.55, 0, 0, 7); ctx.stroke();
      }
      ctx.globalCompositeOperation = "source-over";
    }
    glow(ctx, cx, cy, 40 * k, ROLE_GOLD2, 0.5);
    // Four-branch star.
    ctx.translate(cx, cy);
    ctx.rotate(Math.sin(t * 0.7) * 0.2);
    ctx.scale(k, k);
    for (const s of [1, 0.45]) {
      ctx.beginPath();
      ctx.moveTo(0, -16 * s);
      ctx.quadraticCurveTo(3 * s, -3 * s, 16 * s, 0);
      ctx.quadraticCurveTo(3 * s, 3 * s, 0, 16 * s);
      ctx.quadraticCurveTo(-3 * s, 3 * s, -16 * s, 0);
      ctx.quadraticCurveTo(-3 * s, -3 * s, 0, -16 * s);
      ctx.closePath();
      ctx.fillStyle = s === 1 ? ROLE_GOLD : "#fff8e0";
      ctx.fill();
    }
    ctx.restore();
  },

  stream(ctx, p) {
    const { B, cx, w, x1, x2, topY, botY, t, gs, fx } = p;
    ctx.save();
    // Discrete corridor walls.
    ctx.strokeStyle = hexA(B.hue, 0.30);
    ctx.lineWidth = Math.max(1, gs * 1.5);
    ctx.beginPath();
    ctx.moveTo(x1, topY); ctx.lineTo(x1, botY);
    ctx.moveTo(x2, topY); ctx.lineTo(x2, botY);
    ctx.stroke();
    // Interior veil (baked, stretched).
    ctx.globalCompositeOperation = "lighter";
    ctx.drawImage(getVeilRow("zenith", B.hue, 0.10).canvas, x1, topY, w, botY - topY);
    // Straight rising air threads (baked sprite, scrolling).
    if (fx > 0) {
      const th = getBaked("zenith-thread", 2, 24, (g, ww, hh) => {
        const tg = g.createLinearGradient(0, hh, 0, 0);
        tg.addColorStop(0, hexA("#ffffff", 0));
        tg.addColorStop(1, hexA("#ffffff", 0.55));
        g.fillStyle = tg; g.fillRect(0, 0, ww, hh);
      });
      const thH = 20 * gs;
      risingField((i, y) => {
        const tx = cx - 20 * gs + ((i * 5.3 * gs) % (40 * gs));
        ctx.drawImage(th.canvas, tx, y - thH, Math.max(1, gs * 1.4), thH);
      }, p, t, fx >= 2 ? 9 : 5, 120, 53.7 * gs);
    }
    // Fine rising motes.
    ctx.fillStyle = hexA("#ffffff", 0.5);
    risingField((i, y) => {
      const xx = cx + Math.sin(i * 2.7 + y * 0.03) * 18 * gs;
      ctx.beginPath(); ctx.arc(xx, y, gs * (1 + (i % 3) * 0.6), 0, 7); ctx.fill();
    }, p, t, fx >= 1 ? 6 : 3, 80, 61.7 * gs);
    ctx.restore();
  },
};

// ── T3 · ABYSS — membrane / urchin harrow / deep lantern / bubble column ─────

RENDERERS[3] = {
  support(ctx, p) {
    const { B, x, sy, width, cx, hw, t, wear, ga, impact, gs, fx } = p;
    const sag = impact * 12 * gs;
    const jit = wear === 2 ? Math.sin(t * 26) * 0.9 * gs : 0;
    ctx.save();
    // Rock anchors at the ends (intact at every wear).
    for (const s of [-1, 1]) {
      const ax = cx + s * (hw - gs * 4);
      ctx.fillStyle = "#06302e";
      ctx.beginPath();
      ctx.moveTo(ax, sy - 10 * gs); ctx.lineTo(ax + s * 12 * gs, sy - 4 * gs);
      ctx.lineTo(ax + s * 10 * gs, sy + 12 * gs); ctx.lineTo(ax - s * 4 * gs, sy + 8 * gs);
      ctx.closePath(); ctx.fill();
      ctx.strokeStyle = hexA(B.hue, 0.4); ctx.lineWidth = 1; ctx.stroke();
    }
    // Tensioned membrane (double-stroke halo replaces shadowBlur). Always
    // spans anchor to anchor — silhouette never changes.
    const mx0 = x + gs * 8, mx1 = x + width - gs * 8;
    ctx.lineWidth = Math.max(3, gs * 5);
    ctx.strokeStyle = hexA(B.hue, 0.18 * ga);
    ctx.beginPath();
    ctx.moveTo(mx0, sy + jit);
    ctx.quadraticCurveTo(cx, sy + sag + jit, mx1, sy + jit);
    ctx.stroke();
    ctx.lineWidth = Math.max(1.5, gs * 2.5);
    ctx.strokeStyle = hexA(B.hue, 0.85 * ga);
    ctx.beginPath();
    ctx.moveTo(mx0, sy + jit);
    ctx.quadraticCurveTo(cx, sy + sag + jit, mx1, sy + jit);
    ctx.stroke();
    // Surface reflection (fades with wear).
    ctx.strokeStyle = hexA("#ffffff", 0.4 * (wear === 0 ? 1 : wear === 1 ? 0.4 : 0.15));
    ctx.lineWidth = Math.max(1, gs);
    ctx.beginPath();
    ctx.moveTo(mx0 + gs * 6, sy - 2 * gs + jit);
    ctx.quadraticCurveTo(cx, sy + sag - 2 * gs + jit, mx1 - gs * 6, sy - 2 * gs + jit);
    ctx.stroke();
    // Leaking bubbles (more with wear; plus a burst on impact).
    if (fx > 0 && (wear >= 1 || impact > 0.2)) {
      const n = wear === 2 ? 6 : wear === 1 ? 3 : 2;
      for (let i = 0; i < n; i++) {
        const ph = (t * 0.8 + i * 0.29) % 1;
        ctx.strokeStyle = hexA(B.hue, 0.5 * (1 - ph));
        ctx.beginPath();
        ctx.arc(cx - 24 * gs + i * 10 * gs + Math.sin(i * 2 + t) * 3 * gs, sy + sag * 0.4 - ph * 28 * gs, (1.4 + (i % 2) * 0.8) * gs, 0, 7);
        ctx.stroke();
      }
    }
    ctx.restore();
  },

  resist(ctx, p) {
    const { cx, cy, w, hw, t, gs, fx } = p;
    const pulse = 0.5 + 0.5 * Math.sin(t * 3.2);
    ctx.save();
    if (fx > 0) glow(ctx, cx, cy, Math.min(hw, 46 * gs), ROLE_RED, 0.10 + 0.10 * pulse);
    // Thin dark bar.
    const bh = Math.max(3, gs * 5);
    ctx.fillStyle = "#12050a";
    ctx.fillRect(cx - hw, cy - bh / 2, w, bh);
    ctx.strokeStyle = hexA(ROLE_RED, 0.6); ctx.lineWidth = 1;
    ctx.strokeRect(cx - hw, cy - bh / 2, w, bh);
    // Urchin spikes above/below.
    const step = 8 * gs;
    let i = 0;
    for (let sx = cx - hw + 6 * gs; sx < cx + hw - 4 * gs; sx += step, i++) {
      const up = i % 2 === 0;
      const len = (6 + (i % 3) * 3) * gs * (0.85 + 0.15 * Math.sin(t * 4 + i));
      ctx.strokeStyle = hexA(ROLE_RED, 0.85);
      ctx.lineWidth = Math.max(1, gs * 1.6); ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(sx, cy + (up ? -bh / 2 : bh / 2));
      ctx.lineTo(sx + Math.sin(i * 2.1) * 2 * gs, cy + (up ? -bh / 2 - len : bh / 2 + len));
      ctx.stroke();
    }
    ctx.restore();
  },

  surge(ctx, p) {
    const { cx, cy, r, t, gs, fx } = p;
    const k = r / 16;
    const sway = Math.sin(t * 1.1) * 8 * k;
    const lx = cx + sway, ly = cy + Math.sin(t * 0.8) * 4 * k;
    ctx.save();
    glow(ctx, lx, ly, 34 * k, ROLE_GOLD2, 0.55);
    const core = getCoreSprite("lantern", "#fff8e0", ROLE_GOLD);
    ctx.drawImage(core.canvas, lx - 10 * k, ly - 10 * k, 20 * k, 20 * k);
    // Particles drawn toward the light.
    if (fx > 0) {
      ctx.globalCompositeOperation = "lighter";
      for (let i = 0; i < 6; i++) {
        const ph = (t * 0.4 + i * 0.17) % 1, a = i * 1.05;
        ctx.fillStyle = hexA(ROLE_GOLD2, 0.2 + ph * 0.5);
        ctx.beginPath();
        ctx.arc(lx + Math.cos(a) * 46 * k * (1 - ph), ly + Math.sin(a) * 30 * k * (1 - ph), gs * 1.4, 0, 7);
        ctx.fill();
      }
    }
    ctx.restore();
  },

  stream(ctx, p) {
    const { B, cx, x1, x2, topY, botY, t, gs, fx } = p;
    ctx.save();
    ctx.strokeStyle = hexA(B.hue, 0.25);
    ctx.lineWidth = Math.max(1, gs * 1.5);
    ctx.beginPath();
    ctx.moveTo(x1, topY); ctx.lineTo(x1, botY);
    ctx.moveTo(x2, topY); ctx.lineTo(x2, botY);
    ctx.stroke();
    ctx.globalCompositeOperation = "lighter";
    // Bubbles.
    ctx.lineWidth = Math.max(1, gs);
    risingField((i, y) => {
      const x = cx + Math.sin(i * 2.3 + y * 0.05) * 18 * gs;
      ctx.strokeStyle = hexA(B.hue, 0.5);
      ctx.beginPath(); ctx.arc(x, y, (1.5 + (i % 4)) * gs, 0, 7); ctx.stroke();
    }, p, t, fx >= 2 ? 12 : fx === 1 ? 8 : 4, 60, 47.3 * gs);
    // Plankton motes.
    ctx.fillStyle = hexA("#bfffe0", 0.6);
    risingField((i, y) => {
      const x = cx + Math.sin(i * 2.7 + y * 0.03) * 20 * gs;
      ctx.beginPath(); ctx.arc(x, y, gs * (1 + (i % 3) * 0.6), 0, 7); ctx.fill();
    }, p, t, fx >= 1 ? 7 : 3, 40, 61.7 * gs);
    ctx.restore();
  },
};

// ── T4 · VERDANT — branch-spring / bramble / nectar flower / sap vein ────────

RENDERERS[4] = {
  support(ctx, p) {
    const { B, x, sy, width, cx, t, wear, impact, gs, fx } = p;
    const sag = impact * 10 * gs;
    const tremor = wear === 2 ? Math.sin(t * 24) * 0.8 * gs : 0;
    const bx0 = x + width * 0.04, bx1 = x + width * 0.96;
    ctx.save();
    ctx.lineCap = "round";
    // Branch (same curve at every wear).
    ctx.strokeStyle = "#3a5a2a"; ctx.lineWidth = Math.max(3, gs * 5);
    ctx.beginPath();
    ctx.moveTo(bx0, sy - 4 * gs);
    ctx.quadraticCurveTo(cx, sy + sag + tremor, bx1, sy - 4 * gs);
    ctx.stroke();
    ctx.strokeStyle = hexA(B.hue, 0.6 * (wear === 0 ? 1 : wear === 1 ? 0.7 : 0.45));
    ctx.lineWidth = Math.max(1, gs * 1.5);
    ctx.beginPath();
    ctx.moveTo(bx0, sy - 6 * gs);
    ctx.quadraticCurveTo(cx, sy + sag - 2 * gs + tremor, bx1, sy - 6 * gs);
    ctx.stroke();
    // Notch (coherent with biome) — glows at critical.
    if (wear >= 1) {
      const crA = wear === 2 ? 0.5 + 0.4 * Math.abs(Math.sin(t * 9)) : 0.5;
      ctx.strokeStyle = hexA("#d8e8c0", crA); ctx.lineWidth = Math.max(1, gs * 1.4);
      ctx.beginPath();
      ctx.moveTo(cx - 3 * gs, sy + sag - 3 * gs + tremor);
      ctx.lineTo(cx + 2 * gs, sy + sag + 4 * gs + tremor);
      ctx.stroke();
      if (fx > 0 && wear === 2) glow(ctx, cx, sy + sag + tremor, 10 * gs, "#d8e8c0", 0.25 + 0.2 * Math.sin(t * 9));
    }
    // Leaves: 6 → 4 → 2, symmetric loss.
    const keep = wear === 0 ? [0, 1, 2, 3, 4, 5] : wear === 1 ? [0, 2, 3, 5] : [1, 4];
    for (let i = 0; i < 6; i++) {
      if (!keep.includes(i)) continue;
      const fxp = x + width * 0.2 + i * width * 0.12;
      const fyp = sy + sag * Math.sin((i + 0.5) / 6 * Math.PI) - 6 * gs;
      ctx.save();
      ctx.translate(fxp, fyp + tremor);
      ctx.rotate(0.6 * (i % 2 ? 1 : -1) + Math.sin(t * 2 + i) * 0.15);
      ctx.fillStyle = hexA(B.hue, 0.7);
      ctx.beginPath(); ctx.ellipse(0, -6 * gs, 3.2 * gs, 7 * gs, 0, 0, 7); ctx.fill();
      ctx.restore();
    }
    if (impact > 0.05 && fx > 0) glow(ctx, cx, sy + sag, 30 * gs, B.hue, 0.4 * impact);
    // Dead leaves falling at wear.
    if (fx > 0 && wear >= 1) {
      for (let i = 0; i < (wear === 1 ? 2 : 4); i++) {
        const ph = (t * 0.5 + i * 0.37) % 1;
        ctx.save();
        ctx.translate(cx - 20 * gs + i * 14 * gs + Math.sin(ph * 9 + i) * 6 * gs, sy + 6 * gs + ph * 34 * gs);
        ctx.rotate(ph * 5 + i);
        ctx.fillStyle = hexA("#b8a04a", 0.6 * (1 - ph));
        ctx.beginPath(); ctx.ellipse(0, 0, 2.4 * gs, 4.5 * gs, 0, 0, 7); ctx.fill();
        ctx.restore();
      }
    }
    ctx.restore();
  },

  resist(ctx, p) {
    const { cx, cy, w, hw, t, gs, fx } = p;
    const bristle = 0.7 + 0.3 * Math.sin(t * 3);
    ctx.save();
    // Stem.
    ctx.strokeStyle = "#5a1020"; ctx.lineWidth = Math.max(2, gs * 4); ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(cx - hw + w * 0.03, cy);
    ctx.quadraticCurveTo(cx, cy - 8 * gs, cx + hw - w * 0.03, cy);
    ctx.stroke();
    // Bristling thorns.
    const n = Math.max(3, Math.floor(w / (gs * 22)));
    for (let i = 0; i < n; i++) {
      const ex = cx - hw + w * 0.06 + i * (w * 0.88) / n;
      const up = i % 2 === 0;
      const len = (7 + (i % 3) * 3) * gs * bristle;
      ctx.fillStyle = hexA(ROLE_RED, 0.85);
      ctx.beginPath();
      ctx.moveTo(ex - 3 * gs, cy - (up ? 2 * gs : -2 * gs));
      ctx.lineTo(ex, cy - (up ? 2 * gs + len : -2 * gs - len));
      ctx.lineTo(ex + 3 * gs, cy - (up ? 2 * gs : -2 * gs));
      ctx.closePath(); ctx.fill();
    }
    if (fx > 0) glow(ctx, cx, cy, Math.min(hw, 50 * gs), ROLE_RED, 0.12 + 0.08 * Math.sin(t * 3));
    ctx.restore();
  },

  surge(ctx, p) {
    const { B, cx, cy, r, t, gs, fx } = p;
    const k = r / 16;
    const open = 0.55 + 0.35 * Math.sin(t * 0.9);
    ctx.save();
    // Petals.
    for (let i = 0; i < 6; i++) {
      const a = -Math.PI / 2 + i * (Math.PI * 2 / 6);
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(a);
      ctx.fillStyle = hexA(B.hue, 0.45);
      ctx.beginPath(); ctx.ellipse(0, (-14 * open - 6) * k, 5 * k, (12 * open + 4) * k, 0, 0, 7); ctx.fill();
      ctx.restore();
    }
    // Nectar gem.
    glow(ctx, cx, cy, 30 * k, ROLE_GOLD2, 0.55);
    ctx.translate(cx, cy);
    ctx.rotate(t * 0.8);
    ctx.scale(k, k);
    ctx.fillStyle = ROLE_GOLD;
    ctx.beginPath();
    ctx.moveTo(0, -8); ctx.lineTo(8, 0); ctx.lineTo(0, 8); ctx.lineTo(-8, 0); ctx.closePath(); ctx.fill();
    ctx.restore();
    // Orbiting pollen.
    if (fx > 0) {
      ctx.save();
      ctx.fillStyle = hexA(ROLE_GOLD2, 0.7);
      for (let i = 0; i < 5; i++) {
        const a = t * 1.3 + i * (Math.PI * 2 / 5);
        ctx.beginPath();
        ctx.arc(cx + Math.cos(a) * 26 * k, cy + Math.sin(a) * 16 * k, gs * 1.6, 0, 7);
        ctx.fill();
      }
      ctx.restore();
    }
  },

  stream(ctx, p) {
    const { B, cx, x1, x2, w, topY, botY, t, gs, fx } = p;
    ctx.save();
    // Plant walls (wavy verticals).
    for (const wx of [x1, x2]) {
      ctx.beginPath();
      for (let y = botY + 10; y > topY - 10; y -= 6) {
        const x = wx + Math.sin(y * 0.08 + t * 0.8 + wx) * 1.5 * gs;
        y === botY + 10 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = "#3a5a2a"; ctx.lineWidth = Math.max(2, gs * 3); ctx.stroke();
      ctx.strokeStyle = hexA(B.hue, 0.4); ctx.lineWidth = Math.max(1, gs); ctx.stroke();
    }
    // Interior veil.
    ctx.globalCompositeOperation = "lighter";
    ctx.drawImage(getVeilRow("verdant", B.hue, 0.08).canvas, x1, topY, w, botY - topY);
    // Rising sap pulses.
    if (fx > 0) {
      risingField((i, y) => {
        glow(ctx, cx, y, 10 * gs, B.hue, 0.5);
        ctx.fillStyle = hexA("#eaffea", 0.8);
        ctx.beginPath(); ctx.arc(cx, y, 2.2 * gs, 0, 7); ctx.fill();
      }, p, t, fx >= 2 ? 6 : 4, 70, 53.7 * gs);
    }
    // Rising spores.
    ctx.fillStyle = hexA(B.hue, 0.6);
    risingField((i, y) => {
      const x = cx + Math.sin(i * 2.7 + y * 0.03) * 18 * gs;
      ctx.beginPath(); ctx.arc(x, y, gs * (1 + (i % 3) * 0.5), 0, 7); ctx.fill();
    }, p, t, fx >= 1 ? 6 : 3, 30, 61.7 * gs);
    ctx.restore();
  },
};

// ── T5 · DUNE — clay slab / swarm / sand rose / thermal ──────────────────────

// Clay slab body (vertical gradient, darkening baked per wear), stretched to
// width. Cracks are drawn live across the real width.
function duneSlab(wear, platH) {
  const dark = wear === 0 ? 0 : wear === 1 ? 0.12 : 0.22;
  return getBaked(`dune-slab|${wear}|${platH}`, 1, platH, (g, w, hh) => {
    const dg = g.createLinearGradient(0, 0, 0, hh);
    dg.addColorStop(0, hexA("#b07a32", 1 - dark));
    dg.addColorStop(0.3, hexA("#8a5a1e", 1 - dark * 0.5));
    dg.addColorStop(1, "#4a2e0c");
    g.fillStyle = dg;
    g.fillRect(0, 0, w, hh);
  });
}

// Gold sand-rose lamella (horizontal gradient ellipse), baked once and drawn
// rotated 7× — avoids per-frame gradients in the surge.
function duneLamella() {
  return getBaked("dune-lamella", 32, 9, (g) => {
    const lg = g.createLinearGradient(0, 0, 32, 0);
    lg.addColorStop(0, hexA("#c47a10", 0.75));
    lg.addColorStop(0.5, hexA(ROLE_GOLD, 0.85));
    lg.addColorStop(1, hexA("#fff8e0", 0.7));
    g.fillStyle = lg;
    g.beginPath(); g.ellipse(16, 4.5, 16, 4.5, 0, 0, 7); g.fill();
  });
}

RENDERERS[5] = {
  support(ctx, p) {
    const { B, x, sy, width, cx, t, wear, impact, gs, fx, platH, reducedMotion } = p;
    ctx.save();
    // Slab body (baked gradient) + lit sand edge (pales, pulses at critical).
    ctx.drawImage(duneSlab(wear, platH).canvas, Math.round(x), Math.round(sy), Math.round(width), platH);
    const pulseHz = reducedMotion ? 4.5 : 9;
    const topA = wear === 0 ? 0.5 : wear === 1 ? 0.32 : 0.18 + 0.22 * Math.abs(Math.sin(t * pulseHz));
    ctx.fillStyle = hexA(B.hue, topA);
    ctx.fillRect(Math.round(x), Math.round(sy), Math.round(width), Math.max(1, gs * 2));
    // Base crack network.
    const y0 = sy, ymid = sy + platH * 0.5, ybot = sy + platH;
    ctx.strokeStyle = hexA("#3a2406", 0.8); ctx.lineWidth = 1;
    ctx.beginPath();
    for (let xk = x + 10 * gs; xk < x + width - 4 * gs; xk += 18 * gs) {
      ctx.moveTo(xk, y0); ctx.lineTo(xk + 3 * gs, ymid); ctx.lineTo(xk - 2 * gs, ybot);
      ctx.moveTo(xk + 3 * gs, ymid); ctx.lineTo(xk + 13 * gs, sy + platH * 0.39);
    }
    ctx.stroke();
    // Densifying network with wear.
    if (wear >= 1) {
      ctx.strokeStyle = hexA("#2a1804", 0.9); ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let xk = x + 19 * gs; xk < x + width - 8 * gs; xk += 27 * gs) {
        ctx.moveTo(xk, sy - gs); ctx.lineTo(xk - 3 * gs, sy + platH * 0.55); ctx.lineTo(xk + 2 * gs, ybot);
      }
      ctx.stroke();
    }
    if (wear === 2) {
      ctx.strokeStyle = "#1c0f02"; ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let xk = x + 6 * gs; xk < x + width - 2 * gs; xk += 13 * gs) {
        ctx.moveTo(xk, y0); ctx.lineTo(xk + 4 * gs, sy + platH * 0.55); ctx.lineTo(xk - gs, ybot);
        ctx.moveTo(xk + 4 * gs, sy + platH * 0.55); ctx.lineTo(xk + 10 * gs, sy + platH * 0.66);
      }
      ctx.stroke();
    }
    if (fx > 0 && wear >= 1) wmotes(ctx, cx, sy + platH * 0.5, B.hue, t, wear === 1 ? 3 : 5, 26 * gs, false);
    if (impact > 0.05 && fx > 0) {
      ctx.globalCompositeOperation = "lighter";
      ctx.fillStyle = hexA(B.hue, 0.5 * impact);
      for (let i = 0; i < 6; i++) {
        const a = Math.PI + (i / 5) * Math.PI;
        ctx.beginPath();
        ctx.arc(cx + Math.cos(a) * 20 * gs, sy + Math.sin(a) * 5 * gs, 1.5 * gs, 0, 7);
        ctx.fill();
      }
      ctx.globalCompositeOperation = "source-over";
    }
    ctx.restore();
  },

  resist(ctx, p) {
    const { cx, cy, w, hw, t, gs, fx } = p;
    ctx.save();
    // Thin horizontal swarm of red locusts.
    if (fx > 0) glow(ctx, cx, cy, Math.min(hw, 46 * gs), ROLE_RED, 0.08 + 0.05 * Math.sin(t * 2.2));
    ctx.globalCompositeOperation = "lighter";
    const n = fx >= 2 ? 38 : fx === 1 ? 20 : 10;
    const sweepW = w * 0.72;
    for (let i = 0; i < n; i++) {
      const lx = cx + ((i * 23.7 + Math.sin(t * 1.1 + i) * 9 * gs) % sweepW) - sweepW / 2;
      const ly = cy + Math.sin(t * 2.2 + i * 0.9) * 7 * gs + ((i % 5) - 2) * 2.4 * gs;
      ctx.fillStyle = hexA(i % 4 ? ROLE_RED : "#ffb08a", 0.55 + (i % 3) * 0.15);
      // Transform round-trip instead of save()/restore() per locust (up to 38×).
      const ra = Math.sin(t * 5 + i) * 0.6;
      ctx.translate(lx, ly);
      ctx.rotate(ra);
      ctx.fillRect(-2.2 * gs, -0.8 * gs, 4.4 * gs, 1.6 * gs);
      ctx.rotate(-ra);
      ctx.translate(-lx, -ly);
    }
    ctx.globalCompositeOperation = "source-over";
    // Thin swarm edges.
    ctx.strokeStyle = hexA(ROLE_RED, 0.25); ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - hw * 0.86, cy - 11 * gs); ctx.lineTo(cx + hw * 0.86, cy - 11 * gs);
    ctx.moveTo(cx - hw * 0.86, cy + 11 * gs); ctx.lineTo(cx + hw * 0.86, cy + 11 * gs);
    ctx.stroke();
    ctx.restore();
  },

  surge(ctx, p) {
    const { cx, cy, r, t, gs, fx } = p;
    const k = r / 16;
    ctx.save();
    glow(ctx, cx, cy, 34 * k, ROLE_GOLD2, 0.5);
    // Sand-rose rosette of gold lamellae (baked sprite, drawn rotated 7×).
    const lam = duneLamella();
    ctx.translate(cx, cy);
    ctx.rotate(Math.sin(t * 0.5) * 0.15);
    for (let i = 0; i < 7; i++) {
      const a = i * (Math.PI / 7) + t * 0.12;
      ctx.save();
      ctx.rotate(a);
      ctx.drawImage(lam.canvas, -16 * k, -4.5 * k, 32 * k, 9 * k);
      ctx.restore();
    }
    ctx.restore();
    // Turning glint.
    if (fx > 0) {
      const ga = t * 1.4;
      glow(ctx, cx + Math.cos(ga) * 12 * k, cy + Math.sin(ga) * 8 * k, 8 * k, "#ffffff", 0.5);
    }
  },

  stream(ctx, p) {
    const { B, cx, x1, x2, w, topY, botY, t, gs, fx } = p;
    ctx.save();
    // Straight corridor walls (slight shimmer).
    for (const s of [-1, 1]) {
      ctx.beginPath();
      for (let y = topY; y <= botY; y += 8) {
        const x = cx + s * (w / 2) + Math.sin(y * 0.09 + t * 3) * 1.5 * gs;
        y === topY ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = hexA(B.hue, 0.40); ctx.lineWidth = Math.max(1, gs * 1.5); ctx.stroke();
      ctx.strokeStyle = hexA("#fff2dd", 0.12); ctx.lineWidth = Math.max(2, gs * 3.5); ctx.stroke();
    }
    // Dust devil — constant-width spiral of sand.
    ctx.globalCompositeOperation = "lighter";
    const n = fx >= 2 ? 22 : fx === 1 ? 12 : 0;
    risingField((i, y) => {
      const x = cx + Math.sin(y * 0.09 + t * 3 + i) * 19 * gs;
      ctx.fillStyle = hexA(i % 4 ? B.hue : "#fff2dd", 0.30 + (i % 3) * 0.12);
      ctx.beginPath(); ctx.arc(x, y, (1.2 + (i % 3) * 0.9) * gs, 0, 7); ctx.fill();
    }, p, t, n, 80, 37.3 * gs);
    ctx.restore();
  },
};

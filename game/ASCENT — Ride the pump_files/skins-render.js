/* ============================================================
   ASCENT — CAPSULE 02 "MATERIALS" — BESPOKE RENDERERS
   10 original materials + 10 animated trails.
   Stateless scenes: fn(ctx, cx, cy, R, sk, t) / t = seconds.
   Depends on: cosmetics-render.js (hexA, rand, orbHeading globals) —
   load order documented in cosmetics-render.js header.
   ============================================================ */

const TAU = Math.PI * 2;

/* ── OffscreenCanvas cache — static orb body ─────────────── */
// Map capped at 12 entries (10 skins + simultaneous shop previews)
const _skinBodyCache = new Map();
function getSkinStaticBody(key, R, drawStaticFn) {
  const cacheKey = key + "@" + (R | 0);
  if (!_skinBodyCache.has(cacheKey)) {
    if (_skinBodyCache.size >= 12) {
      _skinBodyCache.delete(_skinBodyCache.keys().next().value);
    }
    const sz = Math.ceil(R * 4 + 8);
    const oc = typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(sz, sz)
      : (function() { const c = document.createElement("canvas"); c.width = c.height = sz; return c; })();
    drawStaticFn(oc.getContext("2d"), sz / 2, sz / 2, R);
    _skinBodyCache.set(cacheKey, { canvas: oc, sz, R });
  }
  return _skinBodyCache.get(cacheKey);
}

/* ── shared sphere helpers ───────────────────────────────── */
function addGlow(ctx, cx, cy, r, col, a) {
  ctx.save(); ctx.globalCompositeOperation = "lighter";
  const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
  g.addColorStop(0, hexA(col, a)); g.addColorStop(0.4, hexA(col, a * 0.4)); g.addColorStop(1, hexA(col, 0));
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, r, 0, TAU); ctx.fill(); ctx.restore();
}
function clipCircle(ctx, cx, cy, R) { ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.clip(); }
function litSphere(ctx, cx, cy, R, hot, mid, rim, lx, ly) {
  const g = ctx.createRadialGradient(lx, ly, R * 0.04, cx, cy, R * 1.06);
  g.addColorStop(0, hot); g.addColorStop(0.34, mid); g.addColorStop(0.8, mid); g.addColorStop(1, rim);
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fill();
}
function specHi(ctx, lx, ly, R, strength) {
  ctx.save(); ctx.globalCompositeOperation = "lighter";
  const s = ctx.createRadialGradient(lx, ly, 0, lx, ly, R * 0.6);
  s.addColorStop(0, "rgba(255,255,255," + strength + ")");
  s.addColorStop(0.5, "rgba(255,255,255," + (strength * 0.2) + ")");
  s.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = s; ctx.beginPath(); ctx.arc(lx, ly, R * 0.6, 0, TAU); ctx.fill(); ctx.restore();
}
function rimGlow(ctx, cx, cy, R, col, a) {
  ctx.save(); ctx.globalCompositeOperation = "lighter";
  const g = ctx.createRadialGradient(cx, cy, R * 0.72, cx, cy, R);
  g.addColorStop(0, hexA(col, 0)); g.addColorStop(0.88, hexA(col, 0)); g.addColorStop(1, hexA(col, a));
  ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fill(); ctx.restore();
}
function drawNetwork(ctx, cx, cy, R, glowCol, hotCol, t, branches, seedBase, flow, lofi) {
  flow = flow || 0;
  const branchCount = (lofi && branches > 4) ? Math.ceil(branches / 2) : branches;
  ctx.lineCap = "round"; ctx.lineJoin = "round";
  for (let b = 0; b < branchCount; b++) {
    let ang = seedBase + b / branches * Math.PI * 2 + (rand(b + seedBase) - 0.5) * 0.5;
    let x = cx, y = cy, len = 0; const step = R * 0.15; const base = [[x, y]];
    while (len < R * 0.96) {
      ang += (rand(b * 7.3 + len * 0.3 + seedBase) - 0.5) * 0.95;
      x += Math.cos(ang) * step; y += Math.sin(ang) * step; len += step;
      base.push([x, y]);
    }
    const pts = base.map((p, i) => {
      if (!flow) return p;
      const n = base[Math.min(i + 1, base.length - 1)], q = base[Math.max(i - 1, 0)];
      const dx = n[0] - q[0], dy = n[1] - q[1], dl = Math.hypot(dx, dy) || 1;
      const amp = flow * R * (i / base.length);
      const off = Math.sin(t * 2.4 + i * 0.7 + b * 1.7) * amp + Math.sin(t * 1.1 + i * 0.3 + b) * amp * 0.5;
      return [p[0] + (-dy / dl) * off, p[1] + (dx / dl) * off];
    });
    ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, p[0], p[1]));
    ctx.strokeStyle = hexA(glowCol, 0.16); ctx.lineWidth = R * 0.10; ctx.stroke();
    const bright = 0.35 + 0.4 * (0.5 + 0.5 * Math.sin(t * 3.5 + b * 1.3));
    ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, p[0], p[1]));
    ctx.strokeStyle = hexA(hotCol, bright); ctx.lineWidth = R * 0.028; ctx.stroke();
  }
}

/* ── 01 · GLACIER ── */
function rGlacier(ctx, cx, cy, R, sk, t, lofi) {
  const lx = cx - R * 0.34, ly = cy - R * 0.4;

  // Pre-rendered static body (addGlow + litSphere + rimGlow + specHi)
  const body = getSkinStaticBody("glacier", R, function(oc, ocx, ocy, R) {
    const lxo = ocx - R * 0.34, lyo = ocy - R * 0.4;
    addGlow(oc, ocx, ocy, R * 1.7, "#bfe6ff", 0.14);
    litSphere(oc, ocx, ocy, R, "#f0fbff", "#a6cfe8", "#3f6f9c", lxo, lyo);
    rimGlow(oc, ocx, ocy, R, "#dffaff", 0.55);
    specHi(oc, lxo, lyo, R, 0.95);
  });
  ctx.drawImage(body.canvas, cx - body.sz / 2, cy - body.sz / 2);

  // Animated layer: crystals and shimmer
  ctx.save(); clipCircle(ctx, cx, cy, R);
  ctx.globalCompositeOperation = "lighter";
  ctx.lineWidth = 1.2; ctx.lineCap = "round";
  const clusters = lofi ? 2 : 4; // graceful degraded mode: 2 clusters instead of 4
  for (let s = 0; s < clusters; s++) {
    const sa = s / clusters * Math.PI * 2 + 0.6, ox = cx + Math.cos(sa) * R * 0.42, oy = cy + Math.sin(sa) * R * 0.42;
    const tw = 0.4 + 0.6 * (0.5 + 0.5 * Math.sin(t * 2 + s));
    ctx.strokeStyle = hexA("#eaffff", 0.45 * tw);
    for (let a = 0; a < 6; a++) {
      const ang = a / 6 * Math.PI * 2 + s;
      ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + Math.cos(ang) * R * 0.3, oy + Math.sin(ang) * R * 0.3); ctx.stroke();
      const mx = ox + Math.cos(ang) * R * 0.16, my = oy + Math.sin(ang) * R * 0.16;
      ctx.beginPath();
      ctx.moveTo(mx, my); ctx.lineTo(mx + Math.cos(ang + 1) * R * 0.07, my + Math.sin(ang + 1) * R * 0.07);
      ctx.moveTo(mx, my); ctx.lineTo(mx + Math.cos(ang - 1) * R * 0.07, my + Math.sin(ang - 1) * R * 0.07);
      ctx.stroke();
    }
  }
  const gx = cx - R + ((t * 0.25) % 1) * 2 * R;
  const sg = ctx.createLinearGradient(gx - R * 0.25, 0, gx + R * 0.25, 0);
  sg.addColorStop(0, "rgba(255,255,255,0)"); sg.addColorStop(0.5, "rgba(255,255,255,0.32)"); sg.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = sg; ctx.fillRect(cx - R, cy - R, 2 * R, 2 * R);
  ctx.restore();
}

/* ── 02 · HOLOGRAM ── */
function rHolo(ctx, cx, cy, R, sk, t, lofi) {
  const acc = sk.accent;
  addGlow(ctx, cx, cy, R * 1.8, acc, 0.16);

  // Static horizontal ellipses + pre-rendered scan-line grid (independent of t)
  const body = getSkinStaticBody("holo:" + acc, R, function(oc, ocx, ocy, R) {
    oc.globalCompositeOperation = "lighter";
    oc.strokeStyle = hexA(acc, 0.5); oc.lineWidth = 1;
    for (let i = -3; i <= 3; i++) {
      const yy = ocy + i / 4 * R, ww = Math.sqrt(Math.max(0, R * R - (i / 4 * R) ** 2));
      oc.beginPath(); oc.ellipse(ocx, yy, ww, R * 0.12, 0, 0, TAU); oc.stroke();
    }
    oc.strokeStyle = hexA(acc, 0.12); oc.lineWidth = 1;
    for (let y = ocy - R; y < ocy + R; y += 4) {
      oc.beginPath(); oc.moveTo(ocx - R, y); oc.lineTo(ocx + R, y); oc.stroke();
    }
  });

  const jx = (rand(Math.floor(t * 20)) - 0.5) * 1.5;
  ctx.save(); ctx.translate(jx, 0); clipCircle(ctx, cx, cy, R);
  ctx.fillStyle = hexA(acc, 0.06); ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fill();
  ctx.globalCompositeOperation = "lighter";
  ctx.drawImage(body.canvas, cx - body.sz / 2, cy - body.sz / 2);
  const ellipseCount = lofi ? 3 : 6; // graceful degraded mode
  for (let i = 0; i < ellipseCount; i++) {
    const ph = t * 0.7 + i / ellipseCount * Math.PI, ww = Math.abs(Math.cos(ph)) * R;
    ctx.globalAlpha = 0.3 + 0.4 * Math.abs(Math.sin(ph));
    ctx.strokeStyle = hexA(acc, 0.5); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.ellipse(cx, cy, ww, R, 0, 0, TAU); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  const sy = cy - R + ((t * 0.5) % 1) * 2 * R;
  ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(cx - R, sy); ctx.lineTo(cx + R, sy); ctx.stroke();
  ctx.restore();
  rimGlow(ctx, cx, cy, R, acc, 0.6);
}

/* ── 03 · OBSIDIAN ── */
function rObsidian(ctx, cx, cy, R, sk, t, lofi) {
  const lx = cx - R * 0.32, ly = cy - R * 0.38, acc = sk.accent;
  addGlow(ctx, cx, cy, R * 1.8, acc, 0.13);
  ctx.save(); clipCircle(ctx, cx, cy, R);
  litSphere(ctx, cx, cy, R, "#33373f", "#0c0e12", "#000000", lx, ly);
  ctx.globalCompositeOperation = "lighter";
  const sh = ctx.createRadialGradient(lx, ly, 0, lx, ly, R * 0.9);
  sh.addColorStop(0, hexA(acc, 0.14)); sh.addColorStop(1, hexA(acc, 0));
  ctx.fillStyle = sh; ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fill();
  drawNetwork(ctx, cx, cy, R, acc, "#ffffff", t, 5, 11.7, 0.045, lofi);
  ctx.restore();
  rimGlow(ctx, cx, cy, R, acc, 0.6); specHi(ctx, lx, ly, R, 0.95);
}

/* ── 04 · IRIDESCENCE ── */
function rBubble(ctx, cx, cy, R, sk, t) {
  addGlow(ctx, cx, cy, R * 1.7, "#bfe0ff", 0.1);
  ctx.save(); clipCircle(ctx, cx, cy, R);
  ctx.globalCompositeOperation = "lighter";
  for (let i = 0; i < 6; i++) {
    const hue = (i * 60 + t * 40) % 360, ang = t * 0.4 + i;
    const px = cx + Math.cos(ang) * R * 0.5, py = cy + Math.sin(ang) * R * 0.5;
    const g = ctx.createRadialGradient(px, py, 0, px, py, R * 1.1);
    g.addColorStop(0, "hsla(" + hue + ",90%,75%,0.20)"); g.addColorStop(1, "hsla(" + hue + ",90%,75%,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fill();
  }
  ctx.restore();
  ctx.save(); ctx.globalCompositeOperation = "lighter";
  ctx.lineWidth = 2; ctx.strokeStyle = "rgba(255,255,255,0.7)";
  ctx.beginPath(); ctx.arc(cx, cy, R * 0.98, 0, TAU); ctx.stroke(); ctx.restore();
  rimGlow(ctx, cx, cy, R, "#ff9ff0", 0.4);
  specHi(ctx, cx - R * 0.35, cy - R * 0.4, R * 0.85, 0.85);
  ctx.save(); ctx.globalCompositeOperation = "lighter";
  const s2 = ctx.createRadialGradient(cx + R * 0.3, cy + R * 0.35, 0, cx + R * 0.3, cy + R * 0.35, R * 0.18);
  s2.addColorStop(0, "rgba(255,255,255,0.5)"); s2.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = s2; ctx.beginPath(); ctx.arc(cx + R * 0.3, cy + R * 0.35, R * 0.18, 0, TAU); ctx.fill(); ctx.restore();
}

/* ── 05 · MAGMA ── */
function rMagma(ctx, cx, cy, R, sk, t, lofi) {
  const lx = cx - R * 0.34, ly = cy - R * 0.4;
  addGlow(ctx, cx, cy, R * 1.9, "#ff6620", 0.18);
  ctx.save(); clipCircle(ctx, cx, cy, R);
  litSphere(ctx, cx, cy, R, "#3a2018", "#160a06", "#000000", lx, ly);
  ctx.globalCompositeOperation = "lighter";
  const hg = ctx.createRadialGradient(cx, cy + R * 0.25, 0, cx, cy + R * 0.25, R * 0.95);
  hg.addColorStop(0, hexA("#ff7a30", 0.22 * (0.7 + 0.3 * Math.sin(t * 2)))); hg.addColorStop(1, hexA("#ff7a30", 0));
  ctx.fillStyle = hg; ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fill();
  drawNetwork(ctx, cx, cy, R, "#ff5a1e", "#ffe2a0", t, 7, 4.2, 0.07, lofi);
  ctx.restore();
  rimGlow(ctx, cx, cy, R, "#ff8844", 0.5); specHi(ctx, lx, ly, R, 0.4);
}

/* ── 06 · MERCURY ── */
function rChrome(ctx, cx, cy, R, sk, t) {
  addGlow(ctx, cx, cy, R * 1.7, "#adbccc", 0.13);
  ctx.save(); clipCircle(ctx, cx, cy, R);
  const hz = cy + Math.sin(t * 1.2) * R * 0.06;
  const grad = ctx.createLinearGradient(0, cy - R, 0, cy + R);
  grad.addColorStop(0, "#e6eff6");
  grad.addColorStop(Math.max(0.01, (hz - (cy - R)) / (2 * R) - 0.12), "#8595a6");
  grad.addColorStop((hz - (cy - R)) / (2 * R), "#ffffff");
  grad.addColorStop(Math.min(0.99, (hz - (cy - R)) / (2 * R) + 0.1), "#2c343d");
  grad.addColorStop(1, "#0a0e12");
  ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fill();
  ctx.globalCompositeOperation = "lighter";
  for (let i = 0; i < 5; i++) {
    const yy = cy + Math.sin(t * 1.5 + i * 1.3) * R * 0.12 + (i - 2) * R * 0.34;
    ctx.fillStyle = "rgba(255,255,255,0.10)"; ctx.fillRect(cx - R, yy, R * 2, R * 0.04);
  }
  const rx = cx + Math.sin(t * 0.6) * R * 0.4;
  const rg = ctx.createLinearGradient(rx - R * 0.12, 0, rx + R * 0.12, 0);
  rg.addColorStop(0, "rgba(255,255,255,0)"); rg.addColorStop(0.5, "rgba(255,255,255,0.45)"); rg.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = rg; ctx.fillRect(rx - R * 0.12, cy - R, R * 0.24, R * 2);
  ctx.restore();
  rimGlow(ctx, cx, cy, R, "#dfeefc", 0.5); specHi(ctx, cx - R * 0.3, cy - R * 0.35, R, 0.9);
}

/* ── 07 · PLASMA ── */
function rPlasma(ctx, cx, cy, R, sk, t) {
  addGlow(ctx, cx, cy, R * 2.0, sk.accent, 0.2);
  ctx.save(); clipCircle(ctx, cx, cy, R);
  ctx.fillStyle = "rgba(10,2,14,0.92)"; ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fill();
  ctx.globalCompositeOperation = "lighter";
  const cols = ["#ff66bb", "#c8a0ff", "#ff8844", "#5cc8ff"];
  for (let i = 0; i < 5; i++) {
    const a = t * (0.6 + i * 0.18) + i * 1.3;
    const px = cx + Math.cos(a) * R * 0.42, py = cy + Math.sin(a * 1.3) * R * 0.42;
    const rr = R * (0.5 + 0.2 * Math.sin(t * 2 + i));
    const g = ctx.createRadialGradient(px, py, 0, px, py, rr);
    g.addColorStop(0, hexA(cols[i % cols.length], 0.5)); g.addColorStop(1, hexA(cols[i % cols.length], 0));
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(px, py, rr, 0, TAU); ctx.fill();
  }
  const cg = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.42);
  cg.addColorStop(0, "rgba(255,255,255,0.85)"); cg.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = cg; ctx.beginPath(); ctx.arc(cx, cy, R * 0.42, 0, TAU); ctx.fill();
  ctx.restore();
  rimGlow(ctx, cx, cy, R, sk.accent, 0.7);
  ctx.save(); ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = hexA(sk.accent, 0.45); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cx, cy, R * 0.99, 0, TAU); ctx.stroke(); ctx.restore();
}

/* ── 08 · GALAXY ── */
function rGalaxy(ctx, cx, cy, R, sk, t, lofi) {
  addGlow(ctx, cx, cy, R * 2.0, "#a080ff", 0.16);
  ctx.save(); clipCircle(ctx, cx, cy, R);
  ctx.fillStyle = "rgba(4,2,12,0.95)"; ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fill();
  ctx.globalCompositeOperation = "lighter";
  const rot = t * 0.4;
  const starCount = lofi ? 40 : 70; // graceful degraded mode: 40 stars instead of 70
  for (let a = 0; a < 2; a++) {
    for (let s = 0; s < starCount; s++) {
      const f = s / starCount, ang = rot + a * Math.PI + f * 5.2, rr = f * R * 0.95;
      const px = cx + Math.cos(ang) * rr, py = cy + Math.sin(ang) * rr * 0.82;
      const tw = 0.4 + 0.6 * (0.5 + 0.5 * Math.sin(t * 3 + s));
      const sz = (1 - f) * 2.0 * tw + 0.4;
      const col = f < 0.28 ? "#ffffff" : (rand(s + a * 70) > 0.5 ? "#b8a0ff" : "#9fd8ff");
      ctx.fillStyle = hexA(col, (1 - f) * 0.8 * tw);
      ctx.beginPath(); ctx.arc(px, py, sz, 0, TAU); ctx.fill();
    }
  }
  const cg = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.38);
  cg.addColorStop(0, "rgba(255,250,235,0.9)"); cg.addColorStop(0.5, hexA("#ffcf8c", 0.4)); cg.addColorStop(1, "rgba(255,200,140,0)");
  ctx.fillStyle = cg; ctx.beginPath(); ctx.arc(cx, cy, R * 0.38, 0, TAU); ctx.fill();
  ctx.restore();
  rimGlow(ctx, cx, cy, R, "#8060ff", 0.45);
}

/* ── 09 · PRISM ── */
function rPrism(ctx, cx, cy, R, sk, t) {
  addGlow(ctx, cx, cy, R * 1.9, "#a0e0ff", 0.14);
  ctx.save(); clipCircle(ctx, cx, cy, R);
  const N = 9;
  for (let i = 0; i < N; i++) {
    const a0 = t * 0.3 + i / N * Math.PI * 2, a1 = t * 0.3 + (i + 1) / N * Math.PI * 2;
    const hue = (i / N * 360 + t * 30) % 360;
    const lit = 0.4 + 0.6 * (0.5 + 0.5 * Math.sin(t * 1.5 - i));
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(a0) * R, cy + Math.sin(a0) * R);
    ctx.lineTo(cx + Math.cos(a1) * R, cy + Math.sin(a1) * R);
    ctx.closePath();
    const mid = (a0 + a1) / 2;
    const g = ctx.createLinearGradient(cx, cy, cx + Math.cos(mid) * R, cy + Math.sin(mid) * R);
    g.addColorStop(0, "hsla(" + hue + ",90%,86%,0.55)"); g.addColorStop(1, "hsla(" + hue + ",95%,62%," + (0.5 * lit) + ")");
    ctx.fillStyle = g; ctx.fill();
  }
  ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = 1;
  for (let i = 0; i < N; i++) {
    const a = t * 0.3 + i / N * Math.PI * 2;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(a) * R, cy + Math.sin(a) * R); ctx.stroke();
  }
  const cg = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.3);
  cg.addColorStop(0, "rgba(255,255,255,0.9)"); cg.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = cg; ctx.beginPath(); ctx.arc(cx, cy, R * 0.3, 0, TAU); ctx.fill();
  ctx.restore();
  rimGlow(ctx, cx, cy, R, "#ffffff", 0.5); specHi(ctx, cx - R * 0.3, cy - R * 0.35, R, 0.9);
}

/* ── 10 · SINGULARITY ── */
function rVoid(ctx, cx, cy, R, sk, t) {
  addGlow(ctx, cx, cy, R * 2.2, "#ffb060", 0.16);
  const tilt = -0.32;
  function disc(half) {
    ctx.save();
    ctx.beginPath();
    if (half < 0) ctx.rect(cx - R * 3, cy - R * 3, R * 6, R * 3); else ctx.rect(cx - R * 3, cy, R * 6, R * 3);
    ctx.clip();
    ctx.translate(cx, cy); ctx.rotate(tilt);
    ctx.globalCompositeOperation = "lighter";
    const ro = R * 2.15, ri = R * 1.22, ry = 0.32, SEG = 44;
    for (let i = 0; i < SEG; i++) {
      const a0 = i / SEG * Math.PI * 2, a1 = (i + 1) / SEG * Math.PI * 2, mid = (a0 + a1) / 2;
      const dopp = 0.35 + 0.65 * (0.5 + 0.5 * Math.cos(mid - t * 1.2));
      ctx.beginPath();
      ctx.moveTo(Math.cos(a0) * ri, Math.sin(a0) * ri * ry); ctx.lineTo(Math.cos(a0) * ro, Math.sin(a0) * ro * ry);
      ctx.lineTo(Math.cos(a1) * ro, Math.sin(a1) * ro * ry); ctx.lineTo(Math.cos(a1) * ri, Math.sin(a1) * ri * ry);
      ctx.closePath();
      ctx.fillStyle = hexA(dopp > 0.78 ? "#ffffff" : "#ff9030", 0.5 * dopp); ctx.fill();
    }
    ctx.restore();
  }
  disc(-1);
  ctx.fillStyle = "#000000"; ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fill();
  ctx.save(); ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = "rgba(255,200,120,0.9)"; ctx.lineWidth = 2.4;
  ctx.shadowColor = "#ffcf5c"; ctx.shadowBlur = 10;
  ctx.beginPath(); ctx.arc(cx, cy, R * 0.99, 0, TAU); ctx.stroke(); ctx.restore();
  disc(1);
}

/* ── EARLY EYE (Special Drop) — smoked-glass orb hiding a laser core ──
   Reuses the shared sphere helpers (addGlow/clipCircle/litSphere/specHi/
   rimGlow). The laser flare is drawn UNCLIPPED so the beams overshoot the
   sphere. Brand colour #21ffbe. ─────────────────────────────────────── */
function drawLaserFlare(ctx, cx, cy, R, acc, t, pulse, lofi, swell) {
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  const I = 0.72 + pulse * 0.28;
  const SW = 0.7 + swell * 0.6; // slow swell modulating the diffuse bloom
  // sharp re-ignition flashes every couple of seconds
  const beat = (t * 0.9) % 1;
  const flash = beat < 0.12 ? (1 - beat / 0.12) * 0.5 : 0;
  const II = Math.min(1.25, I + flash);

  // 1) broad omnidirectional bloom
  for (const [rad, a] of [[R * 3.0, 0.16 * II * SW], [R * 2.0, 0.22 * II * SW], [R * 1.2, 0.40 * II * SW]]) {
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
    g.addColorStop(0, hexA(acc, a)); g.addColorStop(0.5, hexA(acc, a * 0.42)); g.addColorStop(1, hexA(acc, 0));
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, rad, 0, TAU); ctx.fill();
  }

  // 2) core bloom (white-hot → green)
  const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.9);
  core.addColorStop(0, "rgba(255,255,255," + (0.96 * II) + ")");
  core.addColorStop(0.24, hexA(acc, 0.85 * II));
  core.addColorStop(1, hexA(acc, 0));
  ctx.fillStyle = core; ctx.beginPath(); ctx.arc(cx, cy, R * 0.9, 0, TAU); ctx.fill();

  // 3) sunburst — rays radiating in every direction, slowly sweeping clockwise
  ctx.save(); ctx.translate(cx, cy); ctx.rotate(t * 0.18);
  const RAYS = lofi ? 6 : 12;
  for (let i = 0; i < RAYS; i++) {
    ctx.rotate(TAU / RAYS);
    const long = i % 2 === 0;
    // per-ray breathing with a per-ray frequency → never synchronised (destructured)
    const rb = 0.55 + 0.45 * Math.sin(t * (1.5 + 0.6 * (i % 3)) + i * 2.3);
    const rl = R * (long ? 2.3 : 1.45) * rb + flash * R;
    const g = ctx.createLinearGradient(0, 0, rl, 0);
    g.addColorStop(0, "rgba(255,255,255," + ((long ? 0.48 : 0.3) * II * (0.4 + 0.8 * rb)) + ")");
    g.addColorStop(0.4, hexA(acc, (long ? 0.34 : 0.2) * II * (0.4 + 0.8 * rb)));
    g.addColorStop(1, hexA(acc, 0));
    ctx.strokeStyle = g; ctx.lineWidth = R * (long ? 0.05 : 0.03); ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(rl, 0); ctx.stroke();
  }
  ctx.restore();

  // 4) anamorphic lance — toned down (no longer a dominant fixed bar), still breathing
  const lb = 0.45 + 0.45 * Math.sin(t * 1.3 + 0.6);
  const hw = R * (1.1 + 0.8 * lb + flash * 0.6);
  for (const [lw, col, a] of [[R * 0.12, acc, 0.10 * II * (0.4 + 0.8 * lb)], [R * 0.04, "#ffffff", 0.28 * II * (0.3 + 0.8 * lb)]]) {
    const g = ctx.createLinearGradient(cx - hw, cy, cx + hw, cy);
    g.addColorStop(0, hexA(col, 0)); g.addColorStop(0.5, hexA(col, a)); g.addColorStop(1, hexA(col, 0));
    ctx.strokeStyle = g; ctx.lineWidth = lw; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(cx - hw, cy); ctx.lineTo(cx + hw, cy); ctx.stroke();
  }

  // 5) lens-ghost dots along the horizontal axis (skipped in lofi)
  if (!lofi) {
    for (const k of [-1.7, -1.05, 0.95, 1.55, 2.15]) {
      const gx = cx + k * R, rr = R * 0.13 * (1 - Math.abs(k) * 0.16);
      if (rr <= 0) continue;
      const g = ctx.createRadialGradient(gx, cy, 0, gx, cy, rr);
      g.addColorStop(0, hexA(acc, 0.15 * II)); g.addColorStop(1, hexA(acc, 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(gx, cy, rr, 0, TAU); ctx.fill();
    }
  }
  ctx.restore();
}
function drawEarlyOrb(ctx, cx, cy, R, acc, t, lofi) {
  t *= 0.5; // run the whole EARLY EYE animation at half speed (approved feel)
  const lx = cx - R * 0.3, ly = cy - R * 0.36;
  const pulse = 0.5 + 0.5 * Math.sin(t * 2.6);
  // slow global swell — two non-harmonic sines so the breathing feels irregular/organic
  const swell = 0.5 + 0.32 * Math.sin(t * 0.8) + 0.18 * Math.sin(t * 1.33 + 1.7);

  // outer halo (breathing + slow swell) — two layers for a fuller glow
  addGlow(ctx, cx, cy, R * (3.3 + pulse * 0.4 + swell * 0.9), acc, 0.10 + pulse * 0.04 + swell * 0.10);
  if (!lofi) addGlow(ctx, cx, cy, R * (2.0 + pulse * 0.3 + swell * 0.6), acc, 0.18 + pulse * 0.07 + swell * 0.13);

  // body — dark green smoked glass
  ctx.save(); clipCircle(ctx, cx, cy, R);
  litSphere(ctx, cx, cy, R, "#0e4a3c", "#072720", "#020b09", lx, ly);
  ctx.globalCompositeOperation = "lighter";
  // inner energy welling up from the core
  const ig = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.98);
  ig.addColorStop(0, hexA(acc, 0.5 + pulse * 0.22));
  ig.addColorStop(0.45, hexA(acc, 0.15));
  ig.addColorStop(1, hexA(acc, 0));
  ctx.fillStyle = ig; ctx.beginPath(); ctx.arc(cx, cy, R, 0, TAU); ctx.fill();
  // sweeping iris arc (scanning eye)
  ctx.strokeStyle = hexA(acc, 0.32); ctx.lineWidth = R * 0.022;
  ctx.beginPath(); ctx.arc(cx, cy, R * 0.62, t * 0.8, t * 0.8 + 4.0); ctx.stroke();
  ctx.strokeStyle = hexA("#ffffff", 0.18); ctx.lineWidth = R * 0.01;
  ctx.beginPath(); ctx.arc(cx, cy, R * 0.62, t * 0.8, t * 0.8 + 1.1); ctx.stroke();
  ctx.restore();

  rimGlow(ctx, cx, cy, R, acc, 0.78);

  // the flare on top (overshoots the sphere)
  drawLaserFlare(ctx, cx, cy, R, acc, t, pulse, lofi, swell);

  // crisp glassy hotspot
  specHi(ctx, lx, ly, R, 0.5);
}

/* ── DISPATCH TABLE ────────────────────────────────────────── */
const SKIN_RENDER = {
  glacier: rGlacier, holo: rHolo, obsidian: rObsidian, bubble: rBubble,
  magma: rMagma, chrome: rChrome, plasma: rPlasma, galaxy: rGalaxy, prism: rPrism, void: rVoid,
  early: (ctx, cx, cy, R, sk, t, lofi) => drawEarlyOrb(ctx, cx, cy, R, sk.accent, t, lofi),
};

/* ── TRAILS — drawSkinTrail ──────────────────────────────────
   Optional ptsIn: array of { x, y, f } where f=0 = head, f=1 = tail.
   When provided, orbPos/orbHeading coords are not used.
   radiusOverride: replaces Math.min(w,h)*0.13 (useful in-game).
   ─────────────────────────────────────────────────────────── */
function drawSkinTrail(ctx, w, h, sk, t, ptsIn, radiusOverride, lofi) {
  const IS_MOBILE = lofi !== undefined ? lofi : w < 768;
  const st = sk.trail.style, N = sk.trail.len || 26, step = 0.028;
  let pts;
  if (ptsIn) {
    pts = ptsIn;
  } else {
    pts = [];
    for (let i = 0; i <= N; i++) {
      const p = orbPos(t - i * step, w, h);
      pts.push({ x: p.x, y: p.y, f: i / N });
    }
  }
  const R = radiusOverride || Math.min(w, h) * 0.13;
  const acc = sk.accent, c2 = sk.trail.c2 || sk.accent;

  /* heading helper — from pts array when ptsIn provided, else lab orbHeading */
  function ptHead(i) {
    if (ptsIn) {
      const i0 = Math.max(0, i - 1), i1 = Math.min(pts.length - 1, i + 1);
      return Math.atan2(pts[i1].y - pts[i0].y, pts[i1].x - pts[i0].x);
    }
    return orbHeading(t - i * step, w, h);
  }

  ctx.save(); ctx.globalCompositeOperation = "lighter";

  switch (st) {
    case "shards":
      for (let i = 2; i < pts.length; i++) {
        const p = pts[i], f = 1 - p.f, sd = rand(i * 2.1);
        ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(sd * 6 + t * 1.6);
        const sz = R * 0.34 * f * (0.6 + sd);
        ctx.beginPath(); ctx.moveTo(0, -sz); ctx.lineTo(sz * 0.55, sz * 0.5); ctx.lineTo(-sz * 0.55, sz * 0.5); ctx.closePath();
        ctx.strokeStyle = hexA(i % 3 ? acc : c2, f * 0.8); ctx.lineWidth = 1.2; ctx.stroke();
        ctx.restore();
      }
      break;
    case "droplets":
      for (let i = pts.length - 1; i >= 1; i--) {
        const p = pts[i], f = 1 - p.f, sd = rand(i * 1.7);
        const off = Math.sin(t * 2 + i) * R * 0.12 * f;
        const px = p.x + off;
        const rr = R * (0.14 + f * 0.34) * (0.7 + sd * 0.5);
        const g = ctx.createRadialGradient(px - rr * 0.3, p.y - rr * 0.3, 0, px, p.y, rr);
        g.addColorStop(0, hexA("#ffffff", f * 0.7)); g.addColorStop(0.5, hexA("#aab8c6", f * 0.4)); g.addColorStop(1, hexA("#4a5560", 0));
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(px, p.y, rr, 0, TAU); ctx.fill();
      }
      break;
    case "tendrils":
      for (let k = 0; k < 3; k++) {
        ctx.beginPath();
        for (let i = 0; i < pts.length; i++) {
          const p = pts[i], f = 1 - p.f;
          const ang = ptHead(i) + Math.PI / 2;
          const off = Math.sin(t * 3 + i * 0.6 + k * 2.1) * R * 0.7 * f;
          (i ? ctx.lineTo : ctx.moveTo).call(ctx, p.x + Math.cos(ang) * off, p.y + Math.sin(ang) * off);
        }
        ctx.strokeStyle = hexA(k === 1 ? c2 : acc, 0.5); ctx.lineWidth = R * (0.14 - k * 0.03);
        ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.stroke();
      }
      break;
    case "rainbow":
      for (let i = 2; i < pts.length; i++) {
        const p = pts[i], f = 1 - p.f, sd = rand(i * 3.3);
        const hue = (i / pts.length * 360 + t * 60) % 360;
        ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(sd * 6 + t * 2);
        const sz = R * 0.3 * f * (0.6 + sd);
        ctx.beginPath(); ctx.moveTo(0, -sz); ctx.lineTo(sz * 0.5, sz * 0.5); ctx.lineTo(-sz * 0.5, sz * 0.5); ctx.closePath();
        ctx.fillStyle = "hsla(" + hue + ",90%,70%," + (f * 0.55) + ")"; ctx.fill();
        ctx.restore();
      }
      break;
    case "vortex":
      for (let i = pts.length - 1; i >= 1; i--) {
        const p = pts[i], f = 1 - p.f;
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, R * 0.4 * f + 2);
        g.addColorStop(0, hexA("#ffb060", f * 0.38)); g.addColorStop(1, hexA("#ffb060", 0));
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, R * 0.4 * f + 2, 0, TAU); ctx.fill();
      }
      for (let i = 1; i < pts.length; i++) {
        const p = pts[i], f = 1 - p.f, sd = rand(i * 5.1);
        const ang = t * 3 + i * 0.6 + sd * 6, rr = R * 0.85 * f * (0.5 + sd * 0.6);
        const px = p.x + Math.cos(ang) * rr, py = p.y + Math.sin(ang) * rr;
        ctx.fillStyle = hexA(sd > 0.5 ? "#fff" : "#ffcf8c", f * 0.9);
        ctx.beginPath(); ctx.arc(px, py, Math.max(0.6, R * 0.05 * f), 0, TAU); ctx.fill();
      }
      break;
    case "bubbles":
      for (let i = 1; i < pts.length; i++) {
        const p = pts[i], f = 1 - p.f, sd = rand(i * 4.7);
        const drift = -((t * 0.4 + sd) % 1) * R * 0.5;
        const px = p.x + Math.sin(t + i) * R * 0.18, py = p.y + drift;
        const rr = R * (0.06 + sd * 0.16) * (0.5 + f);
        const hue = (i * 40 + t * 60) % 360;
        ctx.strokeStyle = "hsla(" + hue + ",90%,78%," + (f * 0.7) + ")"; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(px, py, rr, 0, TAU); ctx.stroke();
        ctx.fillStyle = "hsla(" + hue + ",90%,80%," + (f * 0.10) + ")"; ctx.fill();
        ctx.fillStyle = hexA("#ffffff", f * 0.5);
        ctx.beginPath(); ctx.arc(px - rr * 0.3, py - rr * 0.3, Math.max(0.6, rr * 0.18), 0, TAU); ctx.fill();
      }
      break;
    case "glyphs":
      for (let i = 1; i < pts.length; i++) {
        const p = pts[i], f = 1 - p.f;
        const on = rand(i + Math.floor(t * 10)) > 0.3;
        if (!on) continue;
        const ang = ptHead(i);
        ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(ang);
        const s = R * 0.16 * (0.5 + f);
        ctx.strokeStyle = hexA(i % 2 ? acc : c2, f * 0.85); ctx.lineWidth = 1.3;
        ctx.strokeRect(-s, -s * 0.5, s * 2, s);
        ctx.restore();
      }
      break;
    case "stardust":
      for (let i = pts.length - 1; i >= 1; i--) {
        const p = pts[i], f = 1 - p.f;
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, R * 0.45 * f + 2);
        g.addColorStop(0, hexA("#8060ff", f * 0.32)); g.addColorStop(1, hexA("#8060ff", 0));
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, R * 0.45 * f + 2, 0, TAU); ctx.fill();
      }
      for (let i = 1; i < pts.length; i++) {
        const p = pts[i], f = 1 - p.f, sd = rand(i * 6.3);
        const ang = sd * 6, rr = R * 0.6 * f * sd;
        const px = p.x + Math.cos(ang) * rr, py = p.y + Math.sin(ang) * rr;
        const tw = 0.4 + 0.6 * Math.sin(t * 6 + i);
        ctx.fillStyle = hexA(sd > 0.6 ? "#fff" : (sd > 0.3 ? "#b8a0ff" : "#9fd8ff"), f * Math.max(0, tw));
        ctx.beginPath(); ctx.arc(px, py, Math.max(0.5, R * 0.045 * f), 0, TAU); ctx.fill();
      }
      break;
    case "embers":
      for (let i = pts.length - 1; i >= 1; i--) {
        const p = pts[i], f = 1 - p.f;
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, R * 0.4 * f + 2);
        g.addColorStop(0, hexA("#ff5a1e", f * 0.38)); g.addColorStop(1, hexA("#ff5a1e", 0));
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, R * 0.4 * f + 2, 0, TAU); ctx.fill();
      }
      for (let i = 1; i < pts.length; i++) {
        const p = pts[i], f = 1 - p.f, sd = rand(i * 3.9);
        const rise = -((t * 0.5 + sd) % 1) * R * 0.6;
        const px = p.x + Math.sin(t * 3 + i) * R * 0.16, py = p.y + rise;
        const tw = 0.5 + 0.5 * Math.sin(t * 9 + i);
        ctx.fillStyle = hexA(sd > 0.5 ? "#ffd27a" : "#ff7a30", f * tw);
        ctx.beginPath(); ctx.arc(px, py, Math.max(0.7, R * 0.06 * f), 0, TAU); ctx.fill();
      }
      break;
    case "snow":
      ctx.lineWidth = 1; ctx.lineCap = "round";
      for (let i = 1; i < pts.length; i++) {
        const p = pts[i], f = 1 - p.f, sd = rand(i * 2.9);
        const fall = ((t * 0.3 + sd) % 1) * R * 0.5;
        const px = p.x + Math.sin(t + i * 0.7) * R * 0.2, py = p.y + fall;
        const s = R * 0.08 * (0.5 + sd) * (0.5 + f);
        ctx.strokeStyle = hexA("#eaffff", f * 0.8);
        ctx.beginPath();
        for (let a = 0; a < 3; a++) {
          const ang = a / 3 * Math.PI + t * 0.5;
          ctx.moveTo(px - Math.cos(ang) * s, py - Math.sin(ang) * s);
          ctx.lineTo(px + Math.cos(ang) * s, py + Math.sin(ang) * s);
        }
        ctx.stroke();
      }
      break;
    /* ── Tier-flavour styles (free per-tier trails, game.js TIER_TRAILS) ──
       Deliberately sparser than the shop styles above, accent-driven only,
       and gradient-free (cheap fills/strokes — safe for every frame). */
    case "rings": // mono-colour bubble outlines — distinct from rainbow "bubbles"
      ctx.lineWidth = 1.2;
      for (let i = 2; i < pts.length; i += 2) {
        const p = pts[i], f = 1 - p.f, sd = rand(i * 4.3);
        const rr = R * (0.08 + sd * 0.18) * (0.4 + f);
        const px = p.x + Math.sin(t * 1.3 + i) * R * 0.15, py = p.y - ((t * 0.35 + sd) % 1) * R * 0.4;
        ctx.strokeStyle = hexA(sd > 0.5 ? acc : c2, f * 0.6);
        ctx.beginPath(); ctx.arc(px, py, rr, 0, TAU); ctx.stroke();
      }
      break;
    case "grains": // falling sand grains
      for (let i = 1; i < pts.length; i++) {
        const p = pts[i], f = 1 - p.f, sd = rand(i * 2.3);
        const fall = ((t * 0.6 + sd) % 1) * R * 0.55;
        const px = p.x + Math.sin(t * 1.7 + i) * R * 0.22, py = p.y + fall;
        const s = sd > 0.8 ? 2 : 1.4;
        ctx.fillStyle = hexA(sd > 0.5 ? acc : c2, f * 0.75);
        ctx.fillRect(px, py, s, s);
      }
      break;
    case "sparks": // short rising streaks — sparse, no halo (unlike "embers")
      ctx.lineWidth = 1.2; ctx.lineCap = "round";
      for (let i = 1; i < pts.length; i++) {
        const p = pts[i], f = 1 - p.f, sd = rand(i * 5.7);
        if (sd < 0.35) continue;
        const rise = ((t * 0.8 + sd) % 1) * R * 0.7;
        const px = p.x + Math.sin(t * 2.2 + i) * R * 0.18, py = p.y - rise;
        const len = R * 0.16 * (0.4 + f);
        ctx.strokeStyle = hexA(sd > 0.7 ? c2 : acc, f * 0.8);
        ctx.beginPath(); ctx.moveTo(px, py + len); ctx.lineTo(px, py); ctx.stroke();
      }
      break;
    case "twinkle": // crisp star crosses — no halo/dust cloud (unlike "stardust")
      ctx.lineWidth = 1;
      for (let i = 1; i < pts.length; i++) {
        const p = pts[i], f = 1 - p.f, sd = rand(i * 6.1);
        const tw = Math.max(0, Math.sin(t * 5 + sd * 12));
        if (tw < 0.25) continue;
        const px = p.x + (sd - 0.5) * R * 0.9 * f, py = p.y + (rand(i * 6.7) - 0.5) * R * 0.9 * f;
        const s = R * 0.07 * (0.4 + f) * (0.5 + tw);
        ctx.strokeStyle = hexA(sd > 0.6 ? c2 : acc, f * tw);
        ctx.beginPath();
        ctx.moveTo(px - s, py); ctx.lineTo(px + s, py);
        ctx.moveTo(px, py - s); ctx.lineTo(px, py + s);
        ctx.stroke();
      }
      break;
    case "streaks": // vertical velocity streaks — linear, not orbiting (unlike "vortex")
      ctx.lineWidth = 1.4; ctx.lineCap = "round";
      for (let i = 2; i < pts.length; i++) {
        const p = pts[i], f = 1 - p.f, sd = rand(i * 3.7);
        if (sd < 0.3) continue;
        const px = p.x + (sd - 0.5) * R * 0.6;
        const len = R * (0.25 + 0.45 * f) * (0.5 + sd);
        ctx.strokeStyle = hexA(sd > 0.65 ? c2 : acc, f * 0.7);
        ctx.beginPath(); ctx.moveTo(px, p.y + len * 0.5); ctx.lineTo(px, p.y - len * 0.5); ctx.stroke();
      }
      break;
    case "earlyLaser": {
      // soft beam underlay (accent glow)
      for (let i = pts.length - 1; i >= 1; i--) {
        const p = pts[i], f = 1 - p.f, rr = R * 0.5 * f + 2;
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, rr);
        g.addColorStop(0, hexA(acc, f * 0.30)); g.addColorStop(1, hexA(acc, 0));
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, rr, 0, TAU); ctx.fill();
      }
      // horizontal laser lances + vertical ticks, flickering
      const inc = IS_MOBILE ? 3 : 2;
      for (let i = 2; i < pts.length; i += inc) {
        const p = pts[i], f = 1 - p.f;
        const flick = rand(i + Math.floor(t * 12)) > 0.25 ? 1 : 0.4;
        const len = R * (0.5 + f * 1.15);
        const hg = ctx.createLinearGradient(p.x - len, p.y, p.x + len, p.y);
        hg.addColorStop(0, hexA(c2, 0)); hg.addColorStop(0.5, hexA(c2, f * 0.85 * flick)); hg.addColorStop(1, hexA(c2, 0));
        ctx.strokeStyle = hg; ctx.lineWidth = Math.max(1, R * 0.055 * f); ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(p.x - len, p.y); ctx.lineTo(p.x + len, p.y); ctx.stroke();
        const vl = len * 0.42;
        const vg = ctx.createLinearGradient(p.x, p.y - vl, p.x, p.y + vl);
        vg.addColorStop(0, hexA(c2, 0)); vg.addColorStop(0.5, hexA(c2, f * 0.55 * flick)); vg.addColorStop(1, hexA(c2, 0));
        ctx.strokeStyle = vg; ctx.lineWidth = Math.max(1, R * 0.03 * f);
        ctx.beginPath(); ctx.moveTo(p.x, p.y - vl); ctx.lineTo(p.x, p.y + vl); ctx.stroke();
      }
      break;
    }
    default:
      for (let i = pts.length - 1; i >= 1; i--) {
        const p = pts[i], f = 1 - p.f, r = R * 0.5 * f;
        const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r + 1);
        g.addColorStop(0, hexA(acc, f * 0.5)); g.addColorStop(1, hexA(acc, 0));
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, r + 1, 0, TAU); ctx.fill();
      }
  }
  ctx.restore();
}

/* ── SCÈNE NFT (shop preview canvas) ────────────────────────── */
function holoBackdrop(ctx, w, h, col, t) {
  const cx = w * 0.5, cy = h * 0.46;
  ctx.save(); ctx.globalCompositeOperation = "lighter";
  ctx.translate(cx, cy); ctx.rotate(t * 0.2);
  for (let i = 0; i < 12; i++) {
    ctx.rotate(Math.PI / 6);
    const g = ctx.createLinearGradient(0, 0, 0, -h * 0.6);
    g.addColorStop(0, hexA(col, 0.05)); g.addColorStop(1, hexA(col, 0));
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-12, -h * 0.6); ctx.lineTo(12, -h * 0.6); ctx.fill();
  }
  ctx.restore();
}

function sceneSkinNFT(ctx, w, h, sk, t) {
  holoBackdrop(ctx, w, h, sk.accent, t);
  const cx = w * 0.5, cy = h * 0.46, R = Math.min(w, h) * 0.2;
  const by = cy + Math.sin(t * 1.4) * 5;
  (SKIN_RENDER[sk.render] || rPlasma)(ctx, cx, by, R, sk, t);
}

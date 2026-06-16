"use strict";
/* ============================================================
   ASCENT — NFT RENDER ENGINE  (drop-in, self-contained)
   ------------------------------------------------------------
   One file. No dependencies. Renders every animated cosmetic /
   identity NFT into a SQUARE canvas, on demand.

   USAGE (in the game):
     <canvas id="nft" width="512" height="512"></canvas>
     <script src="ascent-nft.js"></script>
     <script>
       const cv = document.getElementById("nft");
       const ctx = cv.getContext("2d");
       function loop(now){
         ASCENTNFT.draw(ctx, 512, "Orb_Singularity", now/1000);
         requestAnimationFrame(loop);
       }
       requestAnimationFrame(loop);
     </script>

   API:
     ASCENTNFT.items                      → [{id,label,category,accent,rarity}]
     ASCENTNFT.draw(ctx, size, id, t)     → render NFT `id` at time `t` (s)
                                            into a `size`×`size` square.
     ASCENTNFT.byId[id]                    → the catalogue entry.

   Notes:
   • Square, transparent of chrome — only the art + the NFT id
     (bottom-right). Pass showLabel=false as 5th arg to drop it.
   • `size` is CSS px; handle devicePixelRatio in your own canvas
     sizing (see NFT Export.html for the pattern).
   ============================================================ */

(function () {
  /* ── helpers ───────────────────────────────────────────── */
  function hexA(hex, a) {
    hex = String(hex).replace("#", "");
    if (hex.length === 3) hex = hex.split("").map(c => c + c).join("");
    const n = parseInt(hex, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
  function rand(i) { const x = Math.sin(i * 127.1) * 43758.5453; return x - Math.floor(x); }

  function addGlow(ctx, cx, cy, r, col, a) {
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    g.addColorStop(0, hexA(col, a)); g.addColorStop(0.4, hexA(col, a * 0.4)); g.addColorStop(1, hexA(col, 0));
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, r, 0, 7); ctx.fill(); ctx.restore();
  }
  function clipCircle(ctx, cx, cy, R) { ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.clip(); }
  function litSphere(ctx, cx, cy, R, hot, mid, rim, lx, ly) {
    const g = ctx.createRadialGradient(lx, ly, R * 0.04, cx, cy, R * 1.06);
    g.addColorStop(0, hot); g.addColorStop(0.34, mid); g.addColorStop(0.8, mid); g.addColorStop(1, rim);
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.fill();
  }
  function specHi(ctx, lx, ly, R, strength) {
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const s = ctx.createRadialGradient(lx, ly, 0, lx, ly, R * 0.6);
    s.addColorStop(0, `rgba(255,255,255,${strength})`);
    s.addColorStop(0.5, `rgba(255,255,255,${strength * 0.2})`);
    s.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = s; ctx.beginPath(); ctx.arc(lx, ly, R * 0.6, 0, 7); ctx.fill(); ctx.restore();
  }
  function rimGlow(ctx, cx, cy, R, col, a) {
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const g = ctx.createRadialGradient(cx, cy, R * 0.72, cx, cy, R);
    g.addColorStop(0, hexA(col, 0)); g.addColorStop(0.88, hexA(col, 0)); g.addColorStop(1, hexA(col, a));
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.fill(); ctx.restore();
  }
  function drawNetwork(ctx, cx, cy, R, glowCol, hotCol, t, branches, seedBase, flow) {
    flow = flow || 0;
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (let b = 0; b < branches; b++) {
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
        const off = Math.sin(t * 2.4 + i * 0.7 + b * 1.7) * amp
                  + Math.sin(t * 1.1 + i * 0.3 + b) * amp * 0.5;
        return [p[0] + (-dy / dl) * off, p[1] + (dx / dl) * off];
      });
      ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, p[0], p[1]));
      ctx.strokeStyle = hexA(glowCol, 0.16); ctx.lineWidth = R * 0.10; ctx.stroke();
      const bright = 0.35 + 0.4 * (0.5 + 0.5 * Math.sin(t * 3.5 + b * 1.3));
      ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, p[0], p[1]));
      ctx.strokeStyle = hexA(hotCol, bright); ctx.lineWidth = R * 0.028; ctx.stroke();
    }
  }

  /* ── volumetric orb (for identity seals) ───────────────── */
  function drawOrbAt(ctx, sx, sy, R, o, t, charge) {
    charge = charge == null ? 0.4 : charge;
    const lx = sx - R * 0.36, ly = sy - R * 0.42;
    const sq = 1 + Math.sin(t * 2.0) * 0.045;
    const rx = R / Math.sqrt(sq), ry = R * Math.sqrt(sq);
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    const bloomR = R * o.halo * (1 + charge * 0.4);
    const halo = ctx.createRadialGradient(sx, sy, 0, sx, sy, bloomR);
    halo.addColorStop(0, hexA(o.bloom, 0.34 + charge * 0.22));
    halo.addColorStop(0.32, hexA(o.bloom, 0.11 + charge * 0.10));
    halo.addColorStop(1, hexA(o.bloom, 0));
    ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(sx, sy, bloomR, 0, 7); ctx.fill();
    if (o.corona) {
      const pulse = 0.5 + 0.5 * Math.sin(t * 4);
      const cR = R * (1.55 + pulse * 0.4 + charge * 0.3);
      const cor = ctx.createRadialGradient(sx, sy, R * 0.7, sx, sy, cR);
      cor.addColorStop(0, hexA(o.bloom, 0));
      cor.addColorStop(0.7, hexA(o.bloom, 0.18 + pulse * 0.12));
      cor.addColorStop(1, hexA(o.bloom, 0));
      ctx.fillStyle = cor; ctx.beginPath(); ctx.arc(sx, sy, cR, 0, 7); ctx.fill();
    }
    if (o.flare) {
      const fl = t * 0.6;
      ctx.globalAlpha = 0.5 + 0.2 * Math.sin(fl * 3);
      for (let i = 0; i < 6; i++) {
        const ang = fl + i * Math.PI / 3, len = R * (2.6 + Math.sin(fl * 2 + i) * 0.6);
        const g = ctx.createLinearGradient(sx, sy, sx + Math.cos(ang) * len, sy + Math.sin(ang) * len);
        g.addColorStop(0, hexA(o.hot, 0.5)); g.addColorStop(1, hexA(o.bloom, 0));
        ctx.strokeStyle = g; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx + Math.cos(ang) * len, sy + Math.sin(ang) * len); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    ctx.globalCompositeOperation = "source-over";
    ctx.save();
    ctx.translate(sx, sy); ctx.scale(rx / R, ry / R); ctx.translate(-sx, -sy);
    const body = ctx.createRadialGradient(lx, ly, R * 0.04, sx, sy, R * 1.06);
    body.addColorStop(0, o.hot); body.addColorStop(0.32, o.mid);
    body.addColorStop(0.78, o.mid); body.addColorStop(1, o.rim);
    ctx.fillStyle = body; ctx.beginPath(); ctx.arc(sx, sy, R, 0, 7); ctx.fill();
    ctx.globalCompositeOperation = "lighter";
    const rim = ctx.createRadialGradient(sx, sy, R * 0.74, sx, sy, R);
    rim.addColorStop(0, hexA(o.bloom, 0)); rim.addColorStop(0.86, hexA(o.bloom, 0));
    rim.addColorStop(1, hexA(o.bloom, 0.5 + charge * 0.3));
    ctx.fillStyle = rim; ctx.beginPath(); ctx.arc(sx, sy, R, 0, 7); ctx.fill();
    ctx.restore();
    if (o.band) {
      ctx.save(); ctx.beginPath(); ctx.arc(sx, sy, R * 0.98, 0, 7); ctx.clip();
      ctx.globalCompositeOperation = "lighter";
      const by = sy + Math.sin(t * 1.5) * R * 0.5;
      const bg = ctx.createLinearGradient(0, by - R * 0.18, 0, by + R * 0.18);
      bg.addColorStop(0, hexA(o.hot, 0)); bg.addColorStop(0.5, hexA(o.hot, 0.55)); bg.addColorStop(1, hexA(o.hot, 0));
      ctx.fillStyle = bg; ctx.fillRect(sx - R, by - R * 0.18, R * 2, R * 0.36); ctx.restore();
    }
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const spec = ctx.createRadialGradient(lx, ly, 0, lx, ly, R * 0.62);
    spec.addColorStop(0, "rgba(255,255,255,0.92)"); spec.addColorStop(0.45, "rgba(255,255,255,0.2)"); spec.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = spec; ctx.beginPath(); ctx.arc(lx, ly, R * 0.62, 0, 7); ctx.fill(); ctx.restore();
    if (o.sparks > 0) {
      ctx.save(); ctx.globalCompositeOperation = "lighter";
      const sp = t * 1.5;
      for (let i = 0; i < o.sparks; i++) {
        const ang = sp + i * (Math.PI * 2 / o.sparks), dist = R * (1.7 + Math.sin(sp * 1.3 + i) * 0.35);
        const px = sx + Math.cos(ang) * dist, py = sy + Math.sin(ang) * dist * 0.6, sr = Math.max(1.2, R * 0.12);
        const sg = ctx.createRadialGradient(px, py, 0, px, py, sr * 2.5);
        sg.addColorStop(0, hexA(o.hot, 0.9)); sg.addColorStop(1, hexA(o.bloom, 0));
        ctx.fillStyle = sg; ctx.beginPath(); ctx.arc(px, py, sr * 2.5, 0, 7); ctx.fill();
      }
      ctx.restore();
    }
    ctx.restore();
  }

  /* ── material orb renderers ────────────────────────────── */
  function rGlacier(ctx, cx, cy, R, sk, t) {
    const lx = cx - R * 0.34, ly = cy - R * 0.4;
    addGlow(ctx, cx, cy, R * 1.7, "#bfe6ff", 0.14);
    ctx.save(); clipCircle(ctx, cx, cy, R);
    litSphere(ctx, cx, cy, R, "#f0fbff", "#a6cfe8", "#3f6f9c", lx, ly);
    ctx.globalCompositeOperation = "lighter";
    for (let s = 0; s < 4; s++) {
      const sa = s / 4 * Math.PI * 2 + 0.6, ox = cx + Math.cos(sa) * R * 0.42, oy = cy + Math.sin(sa) * R * 0.42;
      const tw = 0.4 + 0.6 * (0.5 + 0.5 * Math.sin(t * 2 + s));
      ctx.strokeStyle = hexA("#eaffff", 0.45 * tw); ctx.lineWidth = R * 0.018; ctx.lineCap = "round";
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
    rimGlow(ctx, cx, cy, R, "#dffaff", 0.55);
    specHi(ctx, lx, ly, R, 0.95);
  }
  function rHolo(ctx, cx, cy, R, sk, t) {
    const acc = sk.accent;
    addGlow(ctx, cx, cy, R * 1.8, acc, 0.16);
    const jx = (rand(Math.floor(t * 20)) - 0.5) * 1.5;
    ctx.save(); ctx.translate(jx, 0); clipCircle(ctx, cx, cy, R);
    ctx.fillStyle = hexA(acc, 0.06); ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.fill();
    ctx.globalCompositeOperation = "lighter";
    ctx.strokeStyle = hexA(acc, 0.5); ctx.lineWidth = Math.max(1, R * 0.008);
    for (let i = -3; i <= 3; i++) {
      const yy = cy + i / 4 * R, ww = Math.sqrt(Math.max(0, R * R - (i / 4 * R) ** 2));
      ctx.beginPath(); ctx.ellipse(cx, yy, ww, R * 0.12, 0, 0, 7); ctx.stroke();
    }
    for (let i = 0; i < 6; i++) {
      const ph = t * 0.7 + i / 6 * Math.PI, ww = Math.abs(Math.cos(ph)) * R;
      ctx.globalAlpha = 0.3 + 0.4 * Math.abs(Math.sin(ph));
      ctx.beginPath(); ctx.ellipse(cx, cy, ww, R, 0, 0, 7); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    const sy = cy - R + ((t * 0.5) % 1) * 2 * R;
    ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = Math.max(1.5, R * 0.014);
    ctx.beginPath(); ctx.moveTo(cx - R, sy); ctx.lineTo(cx + R, sy); ctx.stroke();
    ctx.strokeStyle = hexA(acc, 0.12); ctx.lineWidth = 1;
    for (let y = cy - R; y < cy + R; y += Math.max(3, R * 0.028)) { ctx.beginPath(); ctx.moveTo(cx - R, y); ctx.lineTo(cx + R, y); ctx.stroke(); }
    ctx.restore();
    rimGlow(ctx, cx, cy, R, acc, 0.6);
  }
  function rObsidian(ctx, cx, cy, R, sk, t) {
    const lx = cx - R * 0.32, ly = cy - R * 0.38, acc = sk.accent;
    addGlow(ctx, cx, cy, R * 1.8, acc, 0.13);
    ctx.save(); clipCircle(ctx, cx, cy, R);
    litSphere(ctx, cx, cy, R, "#33373f", "#0c0e12", "#000000", lx, ly);
    ctx.globalCompositeOperation = "lighter";
    const sh = ctx.createRadialGradient(lx, ly, 0, lx, ly, R * 0.9);
    sh.addColorStop(0, hexA(acc, 0.14)); sh.addColorStop(1, hexA(acc, 0));
    ctx.fillStyle = sh; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.fill();
    drawNetwork(ctx, cx, cy, R, acc, "#ffffff", t, 5, 11.7, 0.045);
    ctx.restore();
    rimGlow(ctx, cx, cy, R, acc, 0.6);
    specHi(ctx, lx, ly, R, 0.95);
  }
  function rBubble(ctx, cx, cy, R, sk, t) {
    addGlow(ctx, cx, cy, R * 1.7, "#bfe0ff", 0.1);
    ctx.save(); clipCircle(ctx, cx, cy, R);
    ctx.globalCompositeOperation = "lighter";
    for (let i = 0; i < 6; i++) {
      const hue = (i * 60 + t * 40) % 360, ang = t * 0.4 + i;
      const px = cx + Math.cos(ang) * R * 0.5, py = cy + Math.sin(ang) * R * 0.5;
      const g = ctx.createRadialGradient(px, py, 0, px, py, R * 1.1);
      g.addColorStop(0, `hsla(${hue},90%,75%,0.20)`); g.addColorStop(1, `hsla(${hue},90%,75%,0)`);
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.fill();
    }
    ctx.restore();
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    ctx.lineWidth = Math.max(2, R * 0.016); ctx.strokeStyle = "rgba(255,255,255,0.7)";
    ctx.beginPath(); ctx.arc(cx, cy, R * 0.98, 0, 7); ctx.stroke(); ctx.restore();
    rimGlow(ctx, cx, cy, R, "#ff9ff0", 0.4);
    specHi(ctx, cx - R * 0.35, cy - R * 0.4, R * 0.85, 0.85);
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const s2 = ctx.createRadialGradient(cx + R * 0.3, cy + R * 0.35, 0, cx + R * 0.3, cy + R * 0.35, R * 0.18);
    s2.addColorStop(0, "rgba(255,255,255,0.5)"); s2.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = s2; ctx.beginPath(); ctx.arc(cx + R * 0.3, cy + R * 0.35, R * 0.18, 0, 7); ctx.fill(); ctx.restore();
  }
  function rMagma(ctx, cx, cy, R, sk, t) {
    const lx = cx - R * 0.34, ly = cy - R * 0.4;
    addGlow(ctx, cx, cy, R * 1.9, "#ff6620", 0.18);
    ctx.save(); clipCircle(ctx, cx, cy, R);
    litSphere(ctx, cx, cy, R, "#3a2018", "#160a06", "#000000", lx, ly);
    ctx.globalCompositeOperation = "lighter";
    const hg = ctx.createRadialGradient(cx, cy + R * 0.25, 0, cx, cy + R * 0.25, R * 0.95);
    hg.addColorStop(0, hexA("#ff7a30", 0.22 * (0.7 + 0.3 * Math.sin(t * 2)))); hg.addColorStop(1, hexA("#ff7a30", 0));
    ctx.fillStyle = hg; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.fill();
    drawNetwork(ctx, cx, cy, R, "#ff5a1e", "#ffe2a0", t, 7, 4.2, 0.07);
    ctx.restore();
    rimGlow(ctx, cx, cy, R, "#ff8844", 0.5);
    specHi(ctx, lx, ly, R, 0.4);
  }
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
    ctx.fillStyle = grad; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.fill();
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
    rimGlow(ctx, cx, cy, R, "#dfeefc", 0.5);
    specHi(ctx, cx - R * 0.3, cy - R * 0.35, R, 0.9);
  }
  function rPlasma(ctx, cx, cy, R, sk, t) {
    addGlow(ctx, cx, cy, R * 2.0, sk.accent, 0.2);
    ctx.save(); clipCircle(ctx, cx, cy, R);
    ctx.fillStyle = "rgba(10,2,14,0.92)"; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.fill();
    ctx.globalCompositeOperation = "lighter";
    const cols = ["#ff66bb", "#c8a0ff", "#ff8844", "#5cc8ff"];
    for (let i = 0; i < 5; i++) {
      const a = t * (0.6 + i * 0.18) + i * 1.3;
      const px = cx + Math.cos(a) * R * 0.42, py = cy + Math.sin(a * 1.3) * R * 0.42;
      const rr = R * (0.5 + 0.2 * Math.sin(t * 2 + i));
      const g = ctx.createRadialGradient(px, py, 0, px, py, rr);
      g.addColorStop(0, hexA(cols[i % cols.length], 0.5)); g.addColorStop(1, hexA(cols[i % cols.length], 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(px, py, rr, 0, 7); ctx.fill();
    }
    const cg = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.42);
    cg.addColorStop(0, "rgba(255,255,255,0.85)"); cg.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = cg; ctx.beginPath(); ctx.arc(cx, cy, R * 0.42, 0, 7); ctx.fill();
    ctx.restore();
    rimGlow(ctx, cx, cy, R, sk.accent, 0.7);
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    ctx.strokeStyle = hexA(sk.accent, 0.45); ctx.lineWidth = Math.max(2, R * 0.016);
    ctx.beginPath(); ctx.arc(cx, cy, R * 0.99, 0, 7); ctx.stroke(); ctx.restore();
  }
  function rGalaxy(ctx, cx, cy, R, sk, t) {
    addGlow(ctx, cx, cy, R * 2.0, "#a080ff", 0.16);
    ctx.save(); clipCircle(ctx, cx, cy, R);
    ctx.fillStyle = "rgba(4,2,12,0.95)"; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.fill();
    ctx.globalCompositeOperation = "lighter";
    const rot = t * 0.4;
    for (let a = 0; a < 2; a++) {
      for (let s = 0; s < 70; s++) {
        const f = s / 70, ang = rot + a * Math.PI + f * 5.2, rr = f * R * 0.95;
        const px = cx + Math.cos(ang) * rr, py = cy + Math.sin(ang) * rr * 0.82;
        const tw = 0.4 + 0.6 * (0.5 + 0.5 * Math.sin(t * 3 + s));
        const sz = (1 - f) * R * 0.015 * tw + R * 0.004;
        const col = f < 0.28 ? "#ffffff" : (rand(s + a * 70) > 0.5 ? "#b8a0ff" : "#9fd8ff");
        ctx.fillStyle = hexA(col, (1 - f) * 0.8 * tw);
        ctx.beginPath(); ctx.arc(px, py, sz, 0, 7); ctx.fill();
      }
    }
    const cg = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.38);
    cg.addColorStop(0, "rgba(255,250,235,0.9)"); cg.addColorStop(0.5, hexA("#ffcf8c", 0.4)); cg.addColorStop(1, "rgba(255,200,140,0)");
    ctx.fillStyle = cg; ctx.beginPath(); ctx.arc(cx, cy, R * 0.38, 0, 7); ctx.fill();
    ctx.restore();
    rimGlow(ctx, cx, cy, R, "#8060ff", 0.45);
  }
  function rPrism(ctx, cx, cy, R, sk, t) {
    addGlow(ctx, cx, cy, R * 1.9, "#a0e0ff", 0.14);
    ctx.save(); clipCircle(ctx, cx, cy, R);
    const N = 9;
    for (let i = 0; i < N; i++) {
      const a0 = t * 0.3 + i / N * Math.PI * 2, a1 = t * 0.3 + (i + 1) / N * Math.PI * 2;
      const hue = (i / N * 360 + t * 30) % 360, lit = 0.4 + 0.6 * (0.5 + 0.5 * Math.sin(t * 1.5 - i));
      ctx.beginPath(); ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(a0) * R, cy + Math.sin(a0) * R);
      ctx.lineTo(cx + Math.cos(a1) * R, cy + Math.sin(a1) * R);
      ctx.closePath();
      const mid = (a0 + a1) / 2;
      const g = ctx.createLinearGradient(cx, cy, cx + Math.cos(mid) * R, cy + Math.sin(mid) * R);
      g.addColorStop(0, `hsla(${hue},90%,86%,0.55)`); g.addColorStop(1, `hsla(${hue},95%,62%,${0.5 * lit})`);
      ctx.fillStyle = g; ctx.fill();
    }
    ctx.globalCompositeOperation = "lighter";
    ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = Math.max(1, R * 0.008);
    for (let i = 0; i < N; i++) {
      const a = t * 0.3 + i / N * Math.PI * 2;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(a) * R, cy + Math.sin(a) * R); ctx.stroke();
    }
    const cg = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.3);
    cg.addColorStop(0, "rgba(255,255,255,0.9)"); cg.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = cg; ctx.beginPath(); ctx.arc(cx, cy, R * 0.3, 0, 7); ctx.fill();
    ctx.restore();
    rimGlow(ctx, cx, cy, R, "#ffffff", 0.5);
    specHi(ctx, cx - R * 0.3, cy - R * 0.35, R, 0.9);
  }
  function rVoid(ctx, cx, cy, R, sk, t) {
    addGlow(ctx, cx, cy, R * 2.2, "#ffb060", 0.16);
    const tilt = -0.32;
    function disc(half) {
      ctx.save(); ctx.beginPath();
      if (half < 0) ctx.rect(cx - R * 3, cy - R * 3, R * 6, R * 3); else ctx.rect(cx - R * 3, cy, R * 6, R * 3);
      ctx.clip(); ctx.translate(cx, cy); ctx.rotate(tilt); ctx.globalCompositeOperation = "lighter";
      const ro = R * 2.15, ri = R * 1.22, ry = 0.32, SEG = 44;
      for (let i = 0; i < SEG; i++) {
        const a0 = i / SEG * Math.PI * 2, a1 = (i + 1) / SEG * Math.PI * 2, mid = (a0 + a1) / 2;
        const dopp = 0.35 + 0.65 * (0.5 + 0.5 * Math.cos(mid - t * 1.2));
        ctx.beginPath();
        ctx.moveTo(Math.cos(a0) * ri, Math.sin(a0) * ri * ry);
        ctx.lineTo(Math.cos(a0) * ro, Math.sin(a0) * ro * ry);
        ctx.lineTo(Math.cos(a1) * ro, Math.sin(a1) * ro * ry);
        ctx.lineTo(Math.cos(a1) * ri, Math.sin(a1) * ri * ry);
        ctx.closePath();
        ctx.fillStyle = hexA(dopp > 0.78 ? "#ffffff" : "#ff9030", 0.5 * dopp); ctx.fill();
      }
      ctx.restore();
    }
    disc(-1);
    ctx.fillStyle = "#000000"; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.fill();
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    ctx.strokeStyle = "rgba(255,200,120,0.9)"; ctx.lineWidth = Math.max(2, R * 0.02);
    ctx.shadowColor = "#ffcf5c"; ctx.shadowBlur = R * 0.08;
    ctx.beginPath(); ctx.arc(cx, cy, R * 0.99, 0, 7); ctx.stroke(); ctx.restore();
    disc(1);
  }
  function darken(hex, f) {
    hex = String(hex).replace("#", ""); if (hex.length === 3) hex = hex.split("").map(c => c + c).join("");
    const n = parseInt(hex, 16);
    return `rgb(${Math.round(((n >> 16) & 255) * f)},${Math.round(((n >> 8) & 255) * f)},${Math.round((n & 255) * f)})`;
  }
  /* ── SIMPLE orb — the generic default cosmetic ─────────── */
  function rSimple(ctx, cx, cy, R, sk, t) {
    const acc = sk.accent, lx = cx - R * 0.34, ly = cy - R * 0.4;
    const pulse = 0.5 + 0.5 * Math.sin(t * 1.6);
    addGlow(ctx, cx, cy, R * (1.7 + pulse * 0.12), acc, 0.16 + pulse * 0.05);
    ctx.save(); clipCircle(ctx, cx, cy, R);
    litSphere(ctx, cx, cy, R, "#ffffff", acc, darken(acc, 0.32), lx, ly);
    ctx.globalCompositeOperation = "lighter";
    const cg = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.72);
    cg.addColorStop(0, hexA("#ffffff", 0.22 + pulse * 0.08)); cg.addColorStop(1, hexA(acc, 0));
    ctx.fillStyle = cg; ctx.beginPath(); ctx.arc(cx, cy, R * 0.72, 0, 7); ctx.fill();
    ctx.restore();
    rimGlow(ctx, cx, cy, R, acc, 0.55);
    specHi(ctx, lx, ly, R, 0.92);
  }
  /* ── EARLY EYE (Special Drop) — smoked-glass orb hiding a laser core ── */
  function earlyLaserFlare(ctx, cx, cy, R, acc, t, pulse) {
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const I = 0.72 + pulse * 0.28;
    const beat = (t * 0.9) % 1;
    const flash = beat < 0.12 ? (1 - beat / 0.12) * 0.5 : 0;
    const II = Math.min(1.25, I + flash);
    for (const pair of [[R * 3.0, 0.16 * II], [R * 2.0, 0.22 * II], [R * 1.2, 0.40 * II]]) {
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, pair[0]);
      g.addColorStop(0, hexA(acc, pair[1])); g.addColorStop(0.5, hexA(acc, pair[1] * 0.42)); g.addColorStop(1, hexA(acc, 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, pair[0], 0, 7); ctx.fill();
    }
    const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.9);
    core.addColorStop(0, "rgba(255,255,255," + (0.96 * II) + ")");
    core.addColorStop(0.24, hexA(acc, 0.85 * II));
    core.addColorStop(1, hexA(acc, 0));
    ctx.fillStyle = core; ctx.beginPath(); ctx.arc(cx, cy, R * 0.9, 0, 7); ctx.fill();
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(t * 0.16);
    const RAYS = 12;
    for (let i = 0; i < RAYS; i++) {
      ctx.rotate(Math.PI * 2 / RAYS);
      const long = i % 2 === 0;
      const rl = R * (long ? 2.3 : 1.45) * (0.78 + 0.28 * Math.sin(t * 3 + i * 1.3)) + flash * R;
      const g = ctx.createLinearGradient(0, 0, rl, 0);
      g.addColorStop(0, "rgba(255,255,255," + ((long ? 0.48 : 0.3) * II) + ")");
      g.addColorStop(0.4, hexA(acc, (long ? 0.34 : 0.2) * II));
      g.addColorStop(1, hexA(acc, 0));
      ctx.strokeStyle = g; ctx.lineWidth = R * (long ? 0.05 : 0.03); ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(rl, 0); ctx.stroke();
    }
    ctx.restore();
    const hw = R * (2.4 + pulse * 0.5 + flash * 1.0);
    for (const lance of [[R * 0.20, acc, 0.20 * II], [R * 0.06, "#ffffff", 0.62 * II]]) {
      const g = ctx.createLinearGradient(cx - hw, cy, cx + hw, cy);
      g.addColorStop(0, hexA(lance[1], 0)); g.addColorStop(0.5, hexA(lance[1], lance[2])); g.addColorStop(1, hexA(lance[1], 0));
      ctx.strokeStyle = g; ctx.lineWidth = lance[0]; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(cx - hw, cy); ctx.lineTo(cx + hw, cy); ctx.stroke();
    }
    for (const k of [-1.7, -1.05, 0.95, 1.55, 2.15]) {
      const gx = cx + k * R, rr = R * 0.13 * (1 - Math.abs(k) * 0.16);
      if (rr <= 0) continue;
      const g = ctx.createRadialGradient(gx, cy, 0, gx, cy, rr);
      g.addColorStop(0, hexA(acc, 0.28 * II)); g.addColorStop(1, hexA(acc, 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(gx, cy, rr, 0, 7); ctx.fill();
    }
    ctx.restore();
  }
  function rEarly(ctx, cx, cy, R, sk, t) {
    const acc = sk.accent, lx = cx - R * 0.3, ly = cy - R * 0.36;
    const pulse = 0.5 + 0.5 * Math.sin(t * 2.6);
    addGlow(ctx, cx, cy, R * (3.4 + pulse * 0.4), acc, 0.13 + pulse * 0.05);
    addGlow(ctx, cx, cy, R * (2.1 + pulse * 0.3), acc, 0.24 + pulse * 0.08);
    ctx.save(); clipCircle(ctx, cx, cy, R);
    litSphere(ctx, cx, cy, R, "#0e4a3c", "#072720", "#020b09", lx, ly);
    ctx.globalCompositeOperation = "lighter";
    const ig = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.98);
    ig.addColorStop(0, hexA(acc, 0.5 + pulse * 0.22));
    ig.addColorStop(0.45, hexA(acc, 0.15));
    ig.addColorStop(1, hexA(acc, 0));
    ctx.fillStyle = ig; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.fill();
    ctx.strokeStyle = hexA(acc, 0.32); ctx.lineWidth = R * 0.022;
    ctx.beginPath(); ctx.arc(cx, cy, R * 0.62, t * 0.8, t * 0.8 + 4.0); ctx.stroke();
    ctx.strokeStyle = hexA("#ffffff", 0.18); ctx.lineWidth = R * 0.01;
    ctx.beginPath(); ctx.arc(cx, cy, R * 0.62, t * 0.8, t * 0.8 + 1.1); ctx.stroke();
    ctx.restore();
    rimGlow(ctx, cx, cy, R, acc, 0.78);
    earlyLaserFlare(ctx, cx, cy, R, acc, t, pulse);
    specHi(ctx, lx, ly, R, 0.5);
  }
  const ORB = { glacier: rGlacier, holo: rHolo, obsidian: rObsidian, bubble: rBubble, magma: rMagma, chrome: rChrome, plasma: rPlasma, galaxy: rGalaxy, prism: rPrism, void: rVoid, early: rEarly, simple: rSimple };

  /* ── identity seal (soulbound badge) ───────────────────── */
  function drawBadge(ctx, cx, cy, Rr, b, t) {
    const col = b.col, glow = b.glow;
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const au = ctx.createRadialGradient(cx, cy, 0, cx, cy, Rr * 2.1);
    au.addColorStop(0, hexA(glow, 0.18)); au.addColorStop(1, hexA(glow, 0));
    ctx.fillStyle = au; ctx.beginPath(); ctx.arc(cx, cy, Rr * 2.1, 0, 7); ctx.fill(); ctx.restore();
    // rotating tick ring
    ctx.save(); ctx.translate(cx, cy); ctx.rotate(t * 0.25);
    ctx.strokeStyle = hexA(col, 0.5); ctx.lineWidth = Math.max(1.5, Rr * 0.014);
    for (let i = 0; i < 48; i++) {
      ctx.rotate(Math.PI * 2 / 48); const long = i % 4 === 0;
      ctx.beginPath(); ctx.moveTo(0, -Rr * 1.32); ctx.lineTo(0, -Rr * (long ? 1.42 : 1.38)); ctx.stroke();
    }
    ctx.restore();
    ctx.save(); ctx.strokeStyle = hexA(glow, 0.55); ctx.lineWidth = Math.max(1.5, Rr * 0.014);
    ctx.beginPath(); ctx.arc(cx, cy, Rr * 1.18, 0, 7); ctx.stroke();
    ctx.setLineDash([Rr * 0.04, Rr * 0.08]); ctx.strokeStyle = hexA(col, 0.4);
    ctx.beginPath(); ctx.arc(cx, cy, Rr * 1.18 - Rr * 0.06, t, t + 6.0); ctx.stroke(); ctx.restore();
    // hex seal plate
    ctx.save(); ctx.translate(cx, cy);
    const sealGrad = ctx.createLinearGradient(0, -Rr, 0, Rr);
    sealGrad.addColorStop(0, hexA(col, 0.30)); sealGrad.addColorStop(1, "rgba(0,0,0,0.6)");
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = -Math.PI / 2 + i * Math.PI / 3;
      (i ? ctx.lineTo : ctx.moveTo).call(ctx, Math.cos(a) * Rr, Math.sin(a) * Rr);
    }
    ctx.closePath(); ctx.fillStyle = sealGrad; ctx.fill();
    ctx.lineWidth = Math.max(2, Rr * 0.02); ctx.strokeStyle = hexA(col, 0.9);
    ctx.shadowColor = hexA(glow, 0.8); ctx.shadowBlur = Rr * 0.16; ctx.stroke(); ctx.restore();
    // orb glyph
    drawOrbAt(ctx, cx, cy + Rr * 0.06, Rr * 0.42, b.orb, t, 0.4);
    // emblem above
    ctx.save(); ctx.translate(cx, cy - Rr * 0.52); ctx.globalCompositeOperation = "lighter";
    ctx.lineWidth = Math.max(2, Rr * 0.02); ctx.lineCap = "round"; ctx.lineJoin = "round";
    const s = Rr * 0.22;
    if (b.emblem === "crown") {
      ctx.beginPath();
      ctx.moveTo(-s, s * 0.5); ctx.lineTo(-s, -s * 0.3); ctx.lineTo(-s * 0.4, s * 0.1);
      ctx.lineTo(0, -s * 0.6); ctx.lineTo(s * 0.4, s * 0.1); ctx.lineTo(s, -s * 0.3);
      ctx.lineTo(s, s * 0.5); ctx.closePath();
      ctx.fillStyle = hexA(col, 0.35); ctx.fill(); ctx.strokeStyle = hexA(b.orb.hot, 0.95); ctx.stroke();
    } else if (b.emblem === "check") {
      ctx.beginPath(); ctx.moveTo(-s * 0.7, 0); ctx.lineTo(-s * 0.15, s * 0.55); ctx.lineTo(s * 0.8, -s * 0.6);
      ctx.lineWidth = Math.max(3, Rr * 0.03); ctx.strokeStyle = hexA(b.orb.hot, 0.95); ctx.stroke();
    } else if (b.emblem === "star") {
      ctx.beginPath();
      for (let i = 0; i < 10; i++) {
        const ang = -Math.PI / 2 + i * Math.PI / 5, rr = (i % 2) ? s * 0.42 : s;
        (i ? ctx.lineTo : ctx.moveTo).call(ctx, Math.cos(ang) * rr, Math.sin(ang) * rr);
      }
      ctx.closePath();
      ctx.fillStyle = hexA(col, 0.35); ctx.fill(); ctx.strokeStyle = hexA(b.orb.hot, 0.95); ctx.stroke();
    }
    ctx.restore();
  }

  function holoBackdrop(ctx, size, col, t) {
    const cx = size / 2, cy = size * 0.47;
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    ctx.translate(cx, cy); ctx.rotate(t * 0.2);
    for (let i = 0; i < 12; i++) {
      ctx.rotate(Math.PI / 6);
      const g = ctx.createLinearGradient(0, 0, 0, -size * 0.6);
      g.addColorStop(0, hexA(col, 0.045)); g.addColorStop(1, hexA(col, 0));
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(-size * 0.03, -size * 0.6); ctx.lineTo(size * 0.03, -size * 0.6); ctx.fill();
    }
    ctx.restore();
  }

  /* ── CATALOGUE ─────────────────────────────────────────── */
  const ORB_PAL = {
    glacier: { hot: "#fff", mid: "#a6cfe8", rim: "#3f6f9c", bloom: "#bfe6ff", halo: 3 },
  };
  const ITEMS = [
    { id: "Orb_Glacier", label: "Orb_Glacier", category: "skin", rarity: "RARE", accent: "#bfe6ff", render: "glacier" },
    { id: "Orb_Hologram", label: "Orb_Hologram", category: "skin", rarity: "RARE", accent: "#16e0a3", render: "holo" },
    { id: "Orb_Obsidian", label: "Orb_Obsidian", category: "skin", rarity: "EPIC", accent: "#5cc8ff", render: "obsidian" },
    { id: "Orb_Iridescence", label: "Orb_Iridescence", category: "skin", rarity: "EPIC", accent: "#ff9ff0", render: "bubble" },
    { id: "Orb_Magma", label: "Orb_Magma", category: "skin", rarity: "EPIC", accent: "#ff8844", render: "magma" },
    { id: "Orb_Mercury", label: "Orb_Mercury", category: "skin", rarity: "LEGENDARY", accent: "#cfd6d2", render: "chrome" },
    { id: "Orb_Plasma", label: "Orb_Plasma", category: "skin", rarity: "LEGENDARY", accent: "#ff66bb", render: "plasma" },
    { id: "Orb_Galaxy", label: "Orb_Galaxy", category: "skin", rarity: "LEGENDARY", accent: "#b48cff", render: "galaxy" },
    { id: "Orb_Prism", label: "Orb_Prism", category: "skin", rarity: "MYTHIC", accent: "#a0e0ff", render: "prism" },
    { id: "Orb_Singularity", label: "Orb_Singularity", category: "skin", rarity: "MYTHIC", accent: "#ffcf5c", render: "void" },
    { id: "Early_Eye", label: "Orb_Early_Eye", category: "skin", rarity: "LEGENDARY", accent: "#21ffbe", render: "early" },
    {
      id: "Genesis_Owner", label: "Genesis_Owner", category: "identity", rarity: "SOULBOUND", accent: "#ffcf5c",
      badge: { emblem: "crown", col: "#ffcf5c", glow: "#ffdd88", orb: { hot: "#fffceb", mid: "#ffd45c", rim: "#9c6a14", bloom: "#ffee88", halo: 3.2, corona: true, sparks: 4, band: true, flare: true } },
    },
    {
      id: "Ascender", label: "Ascender", category: "identity", rarity: "SOULBOUND", accent: "#16e0a3",
      badge: { emblem: "check", col: "#16e0a3", glow: "#22ffaa", orb: { hot: "#eafff7", mid: "#16e0a3", rim: "#0a6b54", bloom: "#16e0a3", halo: 3.0, corona: true, sparks: 2 } },
    },
    {
      id: "Badge_Default", label: "Badge_Default", category: "badge", rarity: "GENERIC", accent: "#aec2dc",
      badge: { emblem: "star", col: "#aec2dc", glow: "#cfe0f5", orb: { hot: "#f4f9ff", mid: "#aec2dc", rim: "#384658", bloom: "#cfe0f5", halo: 2.8 } },
    },
    {
      id: "Cosmetic_Default", label: "Cosmetic_Default", category: "cosmetic", rarity: "GENERIC", accent: "#16e0a3", render: "simple",
    },
  ];
  const BY_ID = {}; ITEMS.forEach(it => BY_ID[it.id] = it);

  /* ── public draw ───────────────────────────────────────── */
  function draw(ctx, size, id, t, showLabel) {
    const it = BY_ID[id]; if (!it) return;
    if (showLabel === undefined) showLabel = true;
    // background — dark radial tinted by accent
    ctx.clearRect(0, 0, size, size);
    ctx.fillStyle = "#000000"; ctx.fillRect(0, 0, size, size);
    const bg = ctx.createRadialGradient(size / 2, size * 0.46, 0, size / 2, size * 0.46, size * 0.72);
    bg.addColorStop(0, hexA(it.accent, 0.08)); bg.addColorStop(0.7, "rgba(0,0,0,0)");
    ctx.fillStyle = bg; ctx.fillRect(0, 0, size, size);
    holoBackdrop(ctx, size, it.accent, t);
    const cx = size / 2, cy = size * 0.47, bob = Math.sin(t * 1.4) * size * 0.012;
    if (it.badge) drawBadge(ctx, cx, cy + bob, size * 0.26, it.badge, t);
    else (ORB[it.render] || rPlasma)(ctx, cx, cy + bob, size * 0.30, it, t);
    if (showLabel) {
      ctx.save();
      ctx.font = `${Math.round(size * 0.040)}px "Silkscreen", ui-monospace, monospace`;
      ctx.textAlign = "right"; ctx.textBaseline = "alphabetic";
      ctx.shadowColor = hexA(it.accent, 0.7); ctx.shadowBlur = size * 0.02;
      ctx.fillStyle = hexA(it.accent, 0.85);
      ctx.fillText(it.label, size - size * 0.055, size - size * 0.05);
      ctx.restore();
    }
  }

  window.ASCENTNFT = { items: ITEMS, byId: BY_ID, draw: draw };
})();

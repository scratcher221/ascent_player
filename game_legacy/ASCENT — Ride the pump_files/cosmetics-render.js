/* ============================================================
   ASCENT — COSMETICS RENDER ENGINE
   Renders Season 01 orbs, soulbound badges and achievement medallions.
   Stateless scenes: fn(ctx, w, h, item, t) where t = seconds.

   Load contract (index.html): cosmetics-render.js → skins-render.js
   → game.js. game.js is a <script type="module">, so its own hexA/rand
   are module-scoped — the guarded fallbacks below provide the globals
   both render scripts rely on. Keep that order when touching index.html.
   ============================================================ */

if (typeof hexA === "undefined") {
  const _hexRGB = Object.create(null);
  // eslint-disable-next-line no-unused-vars
  var hexA = function(hex, a) {
    const h = hex.replace ? hex.replace("#", "") : hex;
    if (!_hexRGB[h]) {
      const hh = h.length === 3 ? h.split("").map(c => c + c).join("") : h;
      const n = parseInt(hh, 16);
      _hexRGB[h] = ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255);
    }
    return "rgba(" + _hexRGB[h] + "," + a + ")";
  };
}
if (typeof rand === "undefined") {
  const _randMemo = Object.create(null);
  // eslint-disable-next-line no-unused-vars
  var rand = function(i) {
    if (_randMemo[i] !== undefined) return _randMemo[i];
    const x = Math.sin(i * 127.1) * 43758.5453;
    return (_randMemo[i] = x - Math.floor(x));
  };
}

/* orbit path lab — figure-eight for animated previews */
function orbPos(t, w, h) {
  const a = t * 0.85;
  return { x: w * 0.5 + Math.cos(a) * w * 0.30, y: h * 0.5 + Math.sin(a * 2) * h * 0.22 };
}
function orbHeading(t, w, h) {
  const p0 = orbPos(t, w, h), p1 = orbPos(t + 0.01, w, h);
  return Math.atan2(p1.y - p0.y, p1.x - p0.x);
}

/* ── VOLUMETRIC ORB (faithful port of game.js drawOrb) ────────── */
function drawOrbAt(ctx, sx, sy, R, o, t, charge) {
  charge = charge == null ? 0.32 : charge;
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
  rim.addColorStop(0, hexA(o.bloom, 0)); rim.addColorStop(0.86, hexA(o.bloom, 0)); rim.addColorStop(1, hexA(o.bloom, 0.5 + charge * 0.3));
  ctx.fillStyle = rim; ctx.beginPath(); ctx.arc(sx, sy, R, 0, 7); ctx.fill();
  ctx.restore();
  if (o.band) {
    ctx.save(); ctx.beginPath(); ctx.arc(sx, sy, R * 0.98, 0, 7); ctx.clip(); ctx.globalCompositeOperation = "lighter";
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
    const sparkPhase = t * 1.5;
    for (let i = 0; i < o.sparks; i++) {
      const ang = sparkPhase + i * (Math.PI * 2 / o.sparks);
      const dist = R * (1.7 + Math.sin(sparkPhase * 1.3 + i) * 0.35);
      const px = sx + Math.cos(ang) * dist, py = sy + Math.sin(ang) * dist * 0.6;
      const sr = Math.max(1.2, R * 0.12);
      const sg = ctx.createRadialGradient(px, py, 0, px, py, sr * 2.5);
      sg.addColorStop(0, hexA(o.hot, 0.9)); sg.addColorStop(1, hexA(o.bloom, 0));
      ctx.fillStyle = sg; ctx.beginPath(); ctx.arc(px, py, sr * 2.5, 0, 7); ctx.fill();
    }
    ctx.restore();
  }
  ctx.restore();
}

/* ── BADGE SOULBOUND (hexagonal seal) ───────────────────────── */
function sceneBadge(ctx, w, h, badge, t) {
  const cx = w * 0.5, cy = h * 0.46;
  const Rr = Math.min(w, h) * 0.34;
  const col = badge.col, glow = badge.glow;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  const au = ctx.createRadialGradient(cx, cy, 0, cx, cy, Rr * 2.1);
  au.addColorStop(0, hexA(glow, 0.18)); au.addColorStop(1, hexA(glow, 0));
  ctx.fillStyle = au; ctx.beginPath(); ctx.arc(cx, cy, Rr * 2.1, 0, 7); ctx.fill();
  ctx.restore();
  ctx.save();
  ctx.translate(cx, cy); ctx.rotate(t * 0.25);
  ctx.strokeStyle = hexA(col, 0.5); ctx.lineWidth = 1.5;
  for (let i = 0; i < 48; i++) {
    ctx.rotate(Math.PI * 2 / 48);
    const long = i % 4 === 0;
    ctx.beginPath(); ctx.moveTo(0, -Rr * 1.32); ctx.lineTo(0, -Rr * (long ? 1.42 : 1.38)); ctx.stroke();
  }
  ctx.restore();
  ctx.save();
  ctx.strokeStyle = hexA(glow, 0.55); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(cx, cy, Rr * 1.18, 0, 7); ctx.stroke();
  ctx.setLineDash([4, 8]); ctx.strokeStyle = hexA(col, 0.4);
  ctx.beginPath(); ctx.arc(cx, cy, Rr * 1.18 - 6, t, t + 6.0); ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
  ctx.save();
  ctx.translate(cx, cy);
  const sealGrad = ctx.createLinearGradient(0, -Rr, 0, Rr);
  sealGrad.addColorStop(0, hexA(col, 0.30)); sealGrad.addColorStop(1, "rgba(0,0,0,0.6)");
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const a = -Math.PI / 2 + i * Math.PI / 3;
    (i ? ctx.lineTo : ctx.moveTo).call(ctx, Math.cos(a) * Rr, Math.sin(a) * Rr);
  }
  ctx.closePath();
  ctx.fillStyle = sealGrad; ctx.fill();
  ctx.lineWidth = 2; ctx.strokeStyle = hexA(col, 0.9);
  ctx.shadowColor = hexA(glow, 0.8); ctx.shadowBlur = 16; ctx.stroke();
  ctx.restore();
  drawOrbAt(ctx, cx, cy + Rr * 0.06, Rr * 0.42, badge.orb, t, 0.4);
  ctx.save();
  ctx.translate(cx, cy - Rr * 0.52);
  ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = hexA(badge.orb.hot, 0.95); ctx.fillStyle = hexA(col, 0.9);
  ctx.lineWidth = 2; ctx.lineCap = "round"; ctx.lineJoin = "round";
  const s = Rr * 0.22;
  if (badge.emblem === "crown") {
    ctx.beginPath();
    ctx.moveTo(-s, s * 0.5); ctx.lineTo(-s, -s * 0.3); ctx.lineTo(-s * 0.4, s * 0.1);
    ctx.lineTo(0, -s * 0.6); ctx.lineTo(s * 0.4, s * 0.1); ctx.lineTo(s, -s * 0.3);
    ctx.lineTo(s, s * 0.5); ctx.closePath();
    ctx.fillStyle = hexA(col, 0.35); ctx.fill();
    ctx.strokeStyle = hexA(badge.orb.hot, 0.95); ctx.stroke();
  } else if (badge.emblem === "check") {
    ctx.beginPath();
    ctx.moveTo(-s * 0.7, 0); ctx.lineTo(-s * 0.15, s * 0.55); ctx.lineTo(s * 0.8, -s * 0.6);
    ctx.lineWidth = 3; ctx.strokeStyle = hexA(badge.orb.hot, 0.95); ctx.stroke();
  }
  ctx.restore();
}

/* ── ACHIEVEMENT MEDALLION ────────────────────────────────────── */
function sceneAchv(ctx, w, h, a, t) {
  const cx = w * 0.5, cy = h * 0.5;
  const Rr = Math.min(w, h) * 0.30;
  const unlocked = !!a.unlocked;
  const col = unlocked ? a.col : "#3a444e";
  const glow = unlocked ? a.col : "#222a30";
  if (unlocked) {
    ctx.save(); ctx.globalCompositeOperation = "lighter";
    const au = ctx.createRadialGradient(cx, cy, 0, cx, cy, Rr * 2);
    const pulse = 0.5 + 0.5 * Math.sin(t * 2.5 + a.no);
    au.addColorStop(0, hexA(glow, 0.12 + pulse * 0.08)); au.addColorStop(1, hexA(glow, 0));
    ctx.fillStyle = au; ctx.beginPath(); ctx.arc(cx, cy, Rr * 2, 0, 7); ctx.fill();
    ctx.restore();
  }
  ctx.save();
  ctx.translate(cx, cy); ctx.rotate(Math.PI / 4);
  const sz = Rr * 1.15, rad = Rr * 0.28;
  const plate = ctx.createLinearGradient(-sz, -sz, sz, sz);
  plate.addColorStop(0, unlocked ? hexA(col, 0.28) : "rgba(20,26,30,0.85)");
  plate.addColorStop(1, "rgba(0,0,0,0.7)");
  roundRect(ctx, -sz, -sz, sz * 2, sz * 2, rad);
  ctx.fillStyle = plate; ctx.fill();
  ctx.lineWidth = 2; ctx.strokeStyle = hexA(col, unlocked ? 0.9 : 0.5);
  if (unlocked) { ctx.shadowColor = hexA(glow, 0.7); ctx.shadowBlur = 12; }
  ctx.stroke();
  ctx.restore();
  ctx.save();
  ctx.strokeStyle = hexA(col, unlocked ? 0.55 : 0.35); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(cx, cy, Rr * 0.92, 0, 7); ctx.stroke();
  ctx.restore();
  if (unlocked) {
    drawAchvIcon(ctx, cx, cy, Rr * 0.62, a.icon, a.col, t);
  } else {
    ctx.save();
    ctx.strokeStyle = "#56636e"; ctx.fillStyle = "rgba(60,72,82,0.5)";
    ctx.lineWidth = 2.4; ctx.lineCap = "round"; ctx.lineJoin = "round";
    const s = Rr * 0.42;
    ctx.beginPath(); ctx.arc(cx, cy - s * 0.15, s * 0.55, Math.PI, 0); ctx.stroke();
    roundRect(ctx, cx - s * 0.75, cy - s * 0.15, s * 1.5, s * 1.15, 3);
    ctx.fill(); ctx.stroke();
    ctx.restore();
  }
}

function drawAchvIcon(ctx, cx, cy, s, type, col, t) {
  ctx.save(); ctx.translate(cx, cy);
  ctx.globalCompositeOperation = "lighter";
  ctx.strokeStyle = hexA(col, 0.95); ctx.fillStyle = hexA(col, 0.85);
  ctx.lineWidth = Math.max(2, s * 0.13); ctx.lineCap = "round"; ctx.lineJoin = "round";
  ctx.globalAlpha = 0.85 + 0.15 * Math.sin(t * 4);
  switch (type) {
    case "chevron":
      for (let i = 0; i < 2; i++) { const yy = s * 0.35 - i * s * 0.55; ctx.beginPath(); ctx.moveTo(-s*.6,yy); ctx.lineTo(0,yy-s*.5); ctx.lineTo(s*.6,yy); ctx.stroke(); } break;
    case "bolt":
      ctx.beginPath(); ctx.moveTo(s*.18,-s*.7); ctx.lineTo(-s*.45,s*.1); ctx.lineTo(0,s*.1); ctx.lineTo(-s*.18,s*.72); ctx.lineTo(s*.5,-s*.12); ctx.lineTo(0,-s*.12); ctx.closePath(); ctx.fill(); break;
    case "diamond":
      ctx.beginPath(); ctx.moveTo(0,-s*.7); ctx.lineTo(s*.62,-s*.12); ctx.lineTo(0,s*.72); ctx.lineTo(-s*.62,-s*.12); ctx.closePath(); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(-s*.32,-s*.12); ctx.lineTo(s*.32,-s*.12); ctx.stroke(); break;
    case "peak":
      ctx.beginPath(); ctx.moveTo(-s*.7,s*.55); ctx.lineTo(-s*.18,-s*.5); ctx.lineTo(s*.1,-s*.05); ctx.lineTo(s*.35,-s*.45); ctx.lineTo(s*.7,s*.55); ctx.closePath(); ctx.stroke(); break;
    case "crown":
      ctx.beginPath(); ctx.moveTo(-s*.7,s*.4); ctx.lineTo(-s*.7,-s*.35); ctx.lineTo(-s*.28,s*.05); ctx.lineTo(0,-s*.6); ctx.lineTo(s*.28,s*.05); ctx.lineTo(s*.7,-s*.35); ctx.lineTo(s*.7,s*.4); ctx.closePath(); ctx.stroke(); break;
    case "chain":
      for (let i = -1; i <= 1; i += 2) { ctx.save(); ctx.translate(i*s*.26,0); ctx.rotate(Math.PI/4); roundRect(ctx,-s*.34,-s*.2,s*.68,s*.4,s*.18); ctx.stroke(); ctx.restore(); } break;
    case "orbs":
      for (let i = 0; i < 3; i++) { ctx.beginPath(); ctx.arc((i-1)*s*.42,Math.sin(i)*s*.12,s*.26,0,7); ctx.fillStyle=hexA(col,.7); ctx.fill(); } break;
    case "moon":
      ctx.beginPath(); ctx.arc(0,0,s*.6,0,7); ctx.fillStyle=hexA(col,.85); ctx.fill();
      ctx.globalCompositeOperation="destination-out"; ctx.beginPath(); ctx.arc(s*.28,-s*.12,s*.52,0,7); ctx.fill(); ctx.globalCompositeOperation="lighter"; break;
    case "clock":
      ctx.beginPath(); ctx.arc(0,0,s*.66,0,7); ctx.stroke();
      for (let hh=0;hh<12;hh++){const a=hh/12*Math.PI*2,long=hh%3===0;ctx.beginPath();ctx.moveTo(Math.cos(a)*s*.66,Math.sin(a)*s*.66);ctx.lineTo(Math.cos(a)*s*(long?.5:.56),Math.sin(a)*s*(long?.5:.56));ctx.stroke();}
      ctx.lineWidth=Math.max(2,s*.11); ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(0,-s*.4);ctx.stroke();
      const ma=t*1.2; ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(Math.cos(ma-Math.PI/2)*s*.5,Math.sin(ma-Math.PI/2)*s*.5);ctx.stroke();
      ctx.beginPath();ctx.arc(0,0,s*.08,0,7);ctx.fill(); break;
    case "star":
      ctx.beginPath(); for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5,rr=i%2?s*.32:s*.72;(i?ctx.lineTo:ctx.moveTo).call(ctx,Math.cos(a)*rr,Math.sin(a)*rr);} ctx.closePath();ctx.fill(); break;
    default:
      ctx.beginPath(); ctx.arc(0,0,s*.5,0,7); ctx.stroke();
  }
  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r); ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r);
  ctx.closePath();
}

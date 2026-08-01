"use strict";
/* ============================================================
   ASCENT — TIER DEFINITIONS
   10 biomes. Each is a distinct world : colour, candle style,
   ambient, orb, platforms and Ultimate.

   orb{} drives the volumetric luminous sphere:
     hot   — lit hot-spot colour (specular core)
     mid   — main body colour
     rim   — saturated terminator / shadow edge
     bloom — outer halo / glow colour
     halo  — bloom radius multiplier
     corona— breathing corona ring
     sparks— # of orbiting light sparks
     band  — rotating energy band across the sphere
     flare — solar lens-flare streaks (apex)
   candle : biome candle style key
   ambient: biome background / foreground key
   feats{} drives world + gameplay rewards:
     trail, stars, nebula, grid, particles, shockwave, ghosts,
     shake, chromatic, arcs, speedlines, aurora
     bounce — base bounce multiplier
     regen  — energy regen multiplier
     scoreMult — height score multiplier (the core reward!)
   ============================================================ */

export const TIERS = [
  { // 1 — GENESIS · pure, white, calm
    name: "GENESIS", tag: "FIRST LIGHT",
    bg: "#000000", bg1: "#070707",
    perk: "Wire chart. One orb. Bounce. Rise.",
    orb: { hot:"#ffffff", mid:"#f2f2f2", rim:"#bcbcbc", bloom:"#ffffff",
           halo:2.4, corona:false, sparks:0, band:false, flare:false },
    candle: "wire", ambient: "void",
    feats: { trail:0, stars:false, nebula:false, grid:false, particles:false,
             shockwave:false, ghosts:true, shake:false, chromatic:false, arcs:false,
             speedlines:false, aurora:false, crt:0.18,
             bounce:1.00, regen:1.0, scoreMult:1 },
  },
  { // 2 — ZENITH · sky, aurora, radiant
    name: "ZENITH", tag: "THE HEAVENS",
    bg: "#00060f", bg1: "#04162e",
    perk: "+ Aurora sky   + Comet trail   + ×1.6 score",
    orb: { hot:"#ffffff", mid:"#dcefff", rim:"#3f6fb8", bloom:"#bfe6ff",
           halo:4.0, corona:true, sparks:2, band:true, flare:true },
    candle: "aurora", ambient: "aurora",
    feats: { trail:25, stars:true, nebula:false, grid:false, particles:false,
             shockwave:false, ghosts:true, shake:false, chromatic:false, arcs:false,
             speedlines:false, aurora:false, crt:0.22,
             bounce:1.03, regen:1.1, scoreMult:1.6 },
    ulti: { name:"INVOKE", cooldownSeconds:8, desc:"Summons 3 BUY platforms above the orb. BUY platforms slide toward the orb. SELL pushed away. Lasts 6 seconds." },
  },
  { // 3 — ABYSS · deep teal, bioluminescent, caustics
    name: "ABYSS", tag: "DEEP LIQUIDITY",
    bg: "#000a0d", bg1: "#01161c",
    perk: "+ Bioluminescent candles   + Spark bursts   + ×2.2 score",
    orb: { hot:"#eafffb", mid:"#2ee6c8", rim:"#0a6b66", bloom:"#3df0d4",
           halo:3.2, corona:true, sparks:3, band:false, flare:false },
    candle: "abyss", ambient: "caustics",
    feats: { trail:25, stars:true, nebula:false, grid:true, particles:true,
             shockwave:false, ghosts:true, shake:false, chromatic:false, arcs:false,
             speedlines:false, aurora:false, crt:0.26,
             bounce:1.06, regen:1.2, scoreMult:2.2 },
    ulti: { name:"AEGIS", cooldownSeconds:15, desc:"A hex shield wraps the orb for up to 15 seconds. The next fatal fall is cancelled and rockets the orb back upward." },
  },
  { // 4 — VERDANT · bright green, bamboo, spores
    name: "VERDANT", tag: "GROWTH SEASON",
    bg: "#02100a", bg1: "#05220f",
    perk: "+ Bamboo candles   + Buy / Sell tiles   + ×2.9 score",
    orb: { hot:"#f6fff2", mid:"#6ef07a", rim:"#1f7a3a", bloom:"#9bff8a",
           halo:3.0, corona:true, sparks:4, band:false, flare:false },
    candle: "verdant", ambient: "spores",
    feats: { trail:25, stars:true, nebula:false, grid:true, particles:true,
             shockwave:false, ghosts:true, shake:false, chromatic:false, arcs:false,
             speedlines:false, aurora:false, crt:0.28,
             bounce:1.10, regen:1.35, scoreMult:2.9 },
    ulti: { name:"OVERCLOCK", cooldownSeconds:14, desc:"Momentum spikes instantly. Manual boosts are free for 6 seconds." },
  },
  { // 5 — DUNE · amber gold, sandstone strata, desert
    name: "DUNE", tag: "GOLDEN HOUR",
    bg: "#120a02", bg1: "#1d1004",
    perk: "+ Sandstone strata   + Nebula glow   + Orbiting sparks   + ×3.7",
    orb: { hot:"#fff4e2", mid:"#ffb55a", rim:"#a85f12", bloom:"#ffcf80",
           halo:3.3, corona:true, sparks:5, band:false, flare:false },
    candle: "dune", ambient: "sand",
    feats: { trail:25, stars:true, nebula:true, grid:true, particles:true,
             shockwave:false, ghosts:true, shake:false, chromatic:false, arcs:false,
             speedlines:true, aurora:false, crt:0.30,
             bounce:1.15, regen:1.5, scoreMult:3.7 },
    ulti: { name:"CHRONO", cooldownSeconds:12, desc:"The orb lifts instantly. Simulation slows to 45% for 7 seconds while the market feed stays live." },
  },
  { // 6 — MAGMA · red-orange, lava columns, embers
    name: "MAGMA", tag: "MELTDOWN",
    bg: "#170401", bg1: "#2a0a03",
    perk: "+ Magma columns   + Shockwaves   + Embers   + ×4.6",
    orb: { hot:"#fff1e8", mid:"#ff7e3a", rim:"#9c2a0a", bloom:"#ff6a36",
           halo:3.6, corona:true, sparks:6, band:true, flare:false },
    candle: "magma", ambient: "embers",
    feats: { trail:25, stars:true, nebula:true, grid:true, particles:true,
             shockwave:true, ghosts:true, shake:false, chromatic:false, arcs:false,
             speedlines:true, aurora:false, crt:0.32,
             bounce:1.21, regen:1.65, scoreMult:4.6 },
    ulti: { name:"MAGNET", cooldownSeconds:18, desc:"Beneficial boosters on screen are collected instantly, stream rewards double and pickups become easier to catch for 8 seconds." },
  },
  { // 7 — ORCHID · pink, crystal prisms, petals
    name: "ORCHID", tag: "NIGHT BLOOM",
    bg: "#10020c", bg1: "#1e0418",
    perk: "+ Crystal prisms   + Aurora   + Electric arcs   + ×5.6",
    orb: { hot:"#ffe9f7", mid:"#ff7ad4", rim:"#8a1f6e", bloom:"#ff8fde",
           halo:3.7, corona:true, sparks:7, band:true, flare:false },
    candle: "orchid", ambient: "petals",
    feats: { trail:25, stars:true, nebula:true, grid:true, particles:true,
             shockwave:true, ghosts:true, shake:true, chromatic:false, arcs:true,
             speedlines:true, aurora:true, crt:0.34,
             bounce:1.28, regen:1.85, scoreMult:5.6 },
    ulti: { name:"PILLAR", cooldownSeconds:18, desc:"A vertical jet fires a huge upward impulse. Jet persists, bounces are stronger, and sell platforms are traversed — all for 6 seconds." },
  },
  { // 8 — VOLT · violet, capacitor candles, electric arcs
    name: "VOLT", tag: "OVERLOAD",
    bg: "#07000f", bg1: "#13001f",
    perk: "+ Capacitor candles   + Chromatic bloom   + Screen shake   + ×6.9",
    orb: { hot:"#f5efff", mid:"#b48cff", rim:"#5a32a8", bloom:"#c8a0ff",
           halo:3.9, corona:true, sparks:8, band:true, flare:false },
    candle: "volt", ambient: "arcs",
    feats: { trail:25, stars:true, nebula:true, grid:true, particles:true,
             shockwave:true, ghosts:true, shake:true, chromatic:true, arcs:true,
             speedlines:true, aurora:true, crt:0.36,
             bounce:1.36, regen:2.05, scoreMult:6.9 },
    ulti: { name:"RESONANCE", cooldownSeconds:18, desc:"Each bounce charges one of 4 electric segments. At full charge the orb is blasted upward explosively. Cycle repeats for 10 seconds." },
  },
  { // 9 — NEBULA · indigo, gas pillars, stardust
    name: "NEBULA", tag: "DEEP SPACE",
    bg: "#02010c", bg1: "#070320",
    perk: "+ Cosmic pillars   + Double corona   + Max sparks   + ×8.3",
    orb: { hot:"#ffffff", mid:"#8a9bff", rim:"#2a2f8a", bloom:"#aab6ff",
           halo:4.4, corona:true, sparks:9, band:true, flare:true },
    candle: "nebula", ambient: "stardust",
    feats: { trail:25, stars:true, nebula:true, grid:true, particles:true,
             shockwave:true, ghosts:true, shake:true, chromatic:true, arcs:true,
             speedlines:true, aurora:true, crt:0.30,
             bounce:1.45, regen:2.3, scoreMult:8.3 },
    ulti: { name:"HELIX", cooldownSeconds:20, desc:"A double helix adds lifts every 1.5 seconds, stronger bounces and amplified impacts for 10 seconds. Steering is inverted." },
  },
  { // 10 — APEX · solar gold, maximal, MOON
    name: "APEX", tag: "TO THE MOON",
    bg: "#000000", bg1: "#0a0500",
    perk: "+ Solar columns   + Solar flare   + Everything   + ×10",
    orb: { hot:"#ffffff", mid:"#ffd45c", rim:"#c47a10", bloom:"#ffee88",
           halo:5.0, corona:true, sparks:10, band:true, flare:true },
    candle: "apex", ambient: "solar",
    feats: { trail:25, stars:true, nebula:true, grid:true, particles:true,
             shockwave:true, ghosts:true, shake:true, chromatic:true, arcs:true,
             speedlines:true, aurora:true, crt:0.26,
             bounce:1.58, regen:2.6, scoreMult:10 },
    ulti: { name:"PHOENIX", cooldownSeconds:20, desc:"Memorises current height. If the orb falls below that point within 12 seconds it is instantly teleported back with a maximum upward burst. Light gravity and stronger bounces throughout." },
  },
];

/* tier bar colour for the rail mini-indicator */
export const TIER_BAR = TIERS.map(t => t.orb.bloom);

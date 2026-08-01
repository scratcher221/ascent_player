/* ============================================================
   ASCENT — CAPSULE 02 "MATERIALS" — SKIN DEFINITIONS
   Price, rarities and supply: source of truth = catalogue-seed.json (Phase 2)
   Rendering: skins-render.js (SKIN_RENDER dispatch)
   ============================================================ */

/* Must be loaded after cosmetics-data.js (RARITY) */

const SKINS = [
  {
    no: 1,
    itemId: "ASC-C02-ORB-001",   // ID on-chain (AscentShop)
    id: "ORB_GLACIER",
    nftId: "Orb_Glacier",         // ID ASCENTNFT (ascent-nft.js)
    name: "GLACIER", tag: "FROZEN LIGHT",
    rarity: "RARE", rarityCol: RARITY.RARE.col,
    priceAscent: 100000, priceXrd: 69, supply: 400,
    accent: "#bfe6ff", render: "glacier",
    trail: { style: "snow", c2: "#eaffff", len: 26 },
    unlockSource: "shop",
  },
  {
    no: 2,
    itemId: "ASC-C02-ORB-002",
    id: "ORB_HOLOGRAM",
    nftId: "Orb_Hologram",
    name: "HOLOGRAM", tag: "WIREFRAME GHOST",
    rarity: "RARE", rarityCol: RARITY.RARE.col,
    priceAscent: 100000, priceXrd: 69, supply: 400,
    accent: "#16e0a3", render: "holo",
    trail: { style: "glyphs", c2: "#5cc8ff", len: 26 },
    unlockSource: "shop",
  },
  {
    no: 3,
    itemId: "ASC-C02-ORB-003",
    id: "ORB_OBSIDIAN",
    nftId: "Orb_Obsidian",
    name: "OBSIDIAN", tag: "FRACTURED GLASS",
    rarity: "RARE", rarityCol: RARITY.RARE.col,
    priceAscent: 100000, priceXrd: 69, supply: 400,
    accent: "#5cc8ff", render: "obsidian",
    trail: { style: "shards", c2: "#bdeefc", len: 24 },
    unlockSource: "shop",
  },
  {
    no: 4,
    itemId: "ASC-C02-ORB-004",
    id: "ORB_IRIDESCENCE",
    nftId: "Orb_Iridescence",
    name: "IRIDESCENCE", tag: "THIN FILM",
    rarity: "RARE", rarityCol: RARITY.RARE.col,
    priceAscent: 100000, priceXrd: 69, supply: 400,
    accent: "#ff9ff0", render: "bubble",
    trail: { style: "bubbles", c2: "#bfe0ff", len: 26 },
    unlockSource: "shop",
  },
  {
    no: 5,
    itemId: "ASC-C02-ORB-005",
    id: "ORB_MAGMA",
    nftId: "Orb_Magma",
    name: "MAGMA", tag: "MOLTEN CORE",
    rarity: "EPIC", rarityCol: RARITY.EPIC.col,
    priceAscent: 300000, priceXrd: 207, supply: 200,
    accent: "#ff8844", render: "magma",
    trail: { style: "embers", c2: "#ffd27a", len: 26 },
    unlockSource: "shop",
  },
  {
    no: 6,
    itemId: "ASC-C02-ORB-006",
    id: "ORB_MERCURY",
    nftId: "Orb_Mercury",
    name: "MERCURY", tag: "LIQUID CHROME",
    rarity: "LEGENDARY", rarityCol: RARITY.LEGENDARY.col,
    priceAscent: 500000, priceXrd: 345, supply: 50,
    accent: "#cfd6d2", render: "chrome",
    trail: { style: "droplets", c2: "#ffffff", len: 24 },
    unlockSource: "shop",
  },
  {
    no: 7,
    itemId: "ASC-C02-ORB-007",
    id: "ORB_PLASMA",
    nftId: "Orb_Plasma",
    name: "PLASMA", tag: "CONTAINED STAR",
    rarity: "LEGENDARY", rarityCol: RARITY.LEGENDARY.col,
    priceAscent: 500000, priceXrd: 345, supply: 50,
    accent: "#ff66bb", render: "plasma",
    trail: { style: "tendrils", c2: "#c8a0ff", len: 26 },
    unlockSource: "shop",
  },
  {
    no: 8,
    itemId: "ASC-C02-ORB-008",
    id: "ORB_GALAXY",
    nftId: "Orb_Galaxy",
    name: "GALAXY", tag: "SPIRAL WITHIN",
    rarity: "LEGENDARY", rarityCol: RARITY.LEGENDARY.col,
    priceAscent: 500000, priceXrd: 345, supply: 50,
    accent: "#b48cff", render: "galaxy",
    trail: { style: "stardust", c2: "#9fd8ff", len: 28 },
    unlockSource: "shop",
  },
  {
    no: 9,
    itemId: "ASC-C02-ORB-009",
    id: "ORB_PRISM",
    nftId: "Orb_Prism",
    name: "PRISM", tag: "PURE REFRACTION",
    rarity: "MYTHIC", rarityCol: RARITY.MYTHIC.col,
    priceAscent: 1000000, priceXrd: 690, supply: 10,
    accent: "#a0e0ff", render: "prism",
    trail: { style: "rainbow", c2: "#ffffff", len: 26 },
    unlockSource: "shop",
  },
  {
    no: 10,
    itemId: "ASC-C02-ORB-010",
    id: "ORB_SINGULARITY",
    nftId: "Orb_Singularity",
    name: "SINGULARITY", tag: "EVENT HORIZON",
    rarity: "MYTHIC", rarityCol: RARITY.MYTHIC.col,
    priceAscent: 1000000, priceXrd: 690, supply: 10,
    accent: "#ffcf5c", render: "void",
    trail: { style: "vortex", c2: "#ffb060", len: 30 },
    unlockSource: "shop",
  },
  {
    no: 11,
    itemId: "ASC-EARLY-EYE-001",
    id: "ORB_EARLY_EYE",
    nftId: "Early_Eye",
    name: "EARLY EYE", tag: "EARLY ACCESS",
    rarity: "LEGENDARY", rarityCol: RARITY.LEGENDARY.col,
    priceAscent: 0, priceXrd: null, supply: 69,
    accent: "#21ffbe", render: "early",
    trail: { style: "earlyLaser", c2: "#ddfff4", len: 26 },
    purchasable: false,            // special drop — not buyable in the shop
    unlockSource: "drop",
  },
];

/* Lookup rapide par itemId on-chain */
const SKINS_BY_ID = Object.fromEntries(SKINS.map(s => [s.itemId, s]));
const SKINS_BY_RENDER  = Object.fromEntries(SKINS.map(s => [s.id, s]));
const SKINS_BY_NFT_ID  = Object.fromEntries(SKINS.filter(s => s.nftId).map(s => [s.nftId, s]));

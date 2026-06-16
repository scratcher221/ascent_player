(function () {
  "use strict";

  const OFFLINE = true;
  window.ASCENT_OFFLINE = OFFLINE;

  const nativeFetch = window.fetch.bind(window);

  function jsonResponse(body, status = 200) {
    return new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  }

  function routeKey(url) {
    const u = new URL(url, window.location.href);
    const path = u.pathname.replace(/\/+$/, "") || "/";
    const q = u.searchParams;
    if (path.endsWith("/candles-1d")) return "candles-1d";
    if (path.endsWith("/candles")) {
      const res = q.get("resolution") || "240";
      if (res === "1D" || res === "1d") return "candles-1D";
      if (res === "60") return "candles-60";
      // 4h / 1h / 30m / 5m / 1m — reuse bundled snapshots (game only needs OHLC shape).
      return "candles-240";
    }
    if (path.endsWith("/live-state")) return "live-state";
    if (path.endsWith("/tier-unlocks")) return "tier-unlocks";
    if (path.endsWith("/gameplay-config")) return "gameplay-config";
    if (path.endsWith("/market-trend")) return "market-trend";
    if (path.endsWith("/pause-state")) return "pause-state";
    if (path.endsWith("/leaderboards")) return "leaderboards";
    if (path.endsWith("/leaderboard/rank")) {
      return { type: "leaderboard-rank", rank: null, total: 0 };
    }
    if (path.endsWith("/leaderboard")) return { type: "leaderboard-ok" };
    if (path.endsWith("/trade-impacts")) return { type: "trade-impacts" };
    if (path === "/shop") return "shop";
    if (path === "/profile") return { type: "profile" };
    if (path.startsWith("/inventory")) return { type: "inventory" };
    if (path === "/auth/challenge") return { type: "auth-challenge" };
    if (path === "/auth/verify") return { type: "auth-verify" };
    if (path === "/track" || path === "/logout") return { type: "noop" };
    return null;
  }

  function offlinePayload(key) {
    const data = window.ASCENT_OFFLINE_DATA || {};
    if (typeof key === "string") return data[key] ?? null;
    if (key?.type === "leaderboard-rank") return { rank: null, total: 0 };
    if (key?.type === "leaderboard-ok") return { ok: true };
    if (key?.type === "trade-impacts") return { impacts: [] };
    if (key?.type === "profile") return { profile: null };
    if (key?.type === "inventory") return { items: [] };
    if (key?.type === "auth-challenge") return { challenge: "offline-challenge" };
    if (key?.type === "auth-verify") {
      return {
        sessionToken: "offline",
        accountAddress: "account_rdx1offline0000000000000000000000000000000000000000000000",
        identityAddress: null,
        isAdmin: false,
      };
    }
    if (key?.type === "noop") return { ok: true };
    return null;
  }

  function isExternal(url) {
    try {
      const u = new URL(url, window.location.href);
      return u.origin !== window.location.origin;
    } catch {
      return true;
    }
  }

  window.fetch = function offlineFetch(input, init) {
    const url = typeof input === "string" ? input : input?.url ?? String(input);

    if (isExternal(url)) {
      return Promise.reject(new Error(`Offline mode: blocked external request to ${url}`));
    }

    const key = routeKey(url);
    if (key !== null) {
      const payload = offlinePayload(key);
      if (payload !== null) {
        return Promise.resolve(jsonResponse(payload));
      }
    }

    // Same-origin static assets (modules, fonts, json) use native fetch.
    return nativeFetch(input, init);
  };
})();

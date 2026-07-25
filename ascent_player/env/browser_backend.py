from __future__ import annotations

import asyncio
import base64
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ascent_player.config import BrowserConfig
from ascent_player.env.browser_discovery import CdpTab, discover_ascent_tab
from ascent_player.env.monitor_layout import (
    find_monitor,
    fit_window_to_monitor,
)
from ascent_player.utils.game_server import ensure_game_server, stop_game_server_if_started

_CAPTURE_TURN_JS = """
(args) => {
    const { selector, maxWidth, maxHeight, quality, boostCost } = args;
    const hud = {
        score: null,
        fell: false,
        inMenu: false,
        energy: null,
        reserve: null,
        canBoost: null,
        combo: 0,
        streak: 0,
        multiplier: 1,
        dataUrl: null,
    };
    const scoreNode = document.querySelector('#heightScore');
    if (scoreNode) {
        const digits = (scoreNode.textContent || '').replace(/\\D/g, '');
        if (digits) hud.score = parseInt(digits, 10);
    }
    const body = document.body ? document.body.innerText.toUpperCase() : '';
    hud.fell = body.includes('FELL') || body.includes('BACK TO EARTH');
    hud.inMenu = body.includes('START THE ASCENT') || body.includes('PICK 1 ULTI');

    let energyPct = 0;
    let reservePct = 0;
    const energyFill = document.querySelector('#energyFill');
    if (energyFill && energyFill.style.height) {
        energyPct = parseFloat(energyFill.style.height) || 0;
    }
    const reserveFill = document.querySelector('#reserveFill');
    const reserveWrap = document.getElementById('reserveWrap');
    if (
        reserveFill
        && reserveWrap
        && reserveWrap.style.display !== 'none'
        && reserveFill.style.height
    ) {
        reservePct = parseFloat(reserveFill.style.height) || 0;
    }
    hud.energy = Math.max(0, Math.min(100, energyPct)) / 100;
    hud.reserve = Math.max(0, Math.min(100, reservePct)) / 100;
    hud.canBoost = (energyPct + reservePct) >= boostCost;

    const multChip = document.getElementById('multChip');
    if (multChip) {
        const multMatch = (multChip.textContent || '').match(/[×x]([\\d.]+)/i);
        if (multMatch) hud.multiplier = parseFloat(multMatch[1]) || 1;
    }
    const comboBadge = document.getElementById('comboBadge');
    if (comboBadge) {
        const comboMatch = (comboBadge.textContent || '').match(/x(\\d+)/i);
        if (comboMatch) {
            hud.combo = parseInt(comboMatch[1], 10) || 0;
            hud.streak = Math.floor(hud.combo / 10);
        }
    }

    const canvas = document.querySelector(selector);
    if (!canvas) return hud;
    try {
        const scale = Math.min(
            maxWidth / Math.max(canvas.width, 1),
            maxHeight / Math.max(canvas.height, 1),
            1
        );
        const width = Math.max(1, Math.round(canvas.width * scale));
        const height = Math.max(1, Math.round(canvas.height * scale));
        const scratch = document.createElement('canvas');
        scratch.width = width;
        scratch.height = height;
        const ctx = scratch.getContext('2d', { willReadFrequently: true });
        if (!ctx) return hud;
        ctx.drawImage(canvas, 0, 0, width, height);
        hud.dataUrl = scratch.toDataURL('image/jpeg', quality);
    } catch (error) {
        return hud;
    }
    return hud;
}
"""

_CANVAS_DATA_URL_JS = """
(selector) => {
    const canvas = document.querySelector(selector);
    if (!canvas) return null;
    try {
        return canvas.toDataURL('image/png');
    } catch (error) {
        return null;
    }
}
"""

_READ_SCORE_JS = """
() => {
    const scoreNode = document.querySelector('#heightScore');
    if (scoreNode) {
        const digits = (scoreNode.textContent || '').replace(/\\D/g, '');
        if (digits) return parseInt(digits, 10);
    }
    const body = document.body ? document.body.innerText : '';
    const match = body.match(/SCORE\\s*([0-9][0-9\\s_]*)/i);
    if (!match) return null;
    const digits = match[1].replace(/\\D/g, '');
    return digits ? parseInt(digits, 10) : 0;
}
"""

_READ_AGENT_STATE_JS = """
() => window.__ASCENT_AGENT__ || null
"""

_ENSURE_AGENT_MODE_JS = """
() => {
    window.CHART_TRIAL_CONFIG = window.CHART_TRIAL_CONFIG || {};
    window.CHART_TRIAL_CONFIG.agentMode = true;
    return true;
}
"""

_SET_RUN_SEED_JS = """
({ seed, lock }) => {
    window.CHART_TRIAL_CONFIG = window.CHART_TRIAL_CONFIG || {};
    if (seed == null) {
        window.CHART_TRIAL_CONFIG.runSeed = null;
        window.__ASCENT_RUN_SEED__ = undefined;
        return null;
    }
    window.CHART_TRIAL_CONFIG.runSeed = seed;
    window.CHART_TRIAL_CONFIG.lockRunSeed = Boolean(lock);
    if (typeof window.__ASCENT_SET_RUN_SEED__ === "function") {
        window.__ASCENT_SET_RUN_SEED__(seed);
    } else {
        const n = Number(seed);
        if (Number.isFinite(n)) window.__ASCENT_RUN_SEED__ = n >>> 0;
    }
    return seed;
}
"""


@dataclass(slots=True)
class HudSnapshot:
    score: int | None = None
    fell: bool = False
    in_menu: bool = False
    energy: float | None = None
    reserve: float | None = None
    can_boost: bool | None = None
    combo: int = 0
    streak: int = 0
    multiplier: float = 1.0


@dataclass(slots=True)
class BrowserStatus:
    connected: bool = False
    mode: str = "disconnected"
    title: str = ""
    url: str = ""
    cdp_url: str | None = None
    message: str = "Not connected"


class BrowserBackend:
    def __init__(self, config: BrowserConfig) -> None:
        self.config = config
        self.playwright: Any | None = None
        self.browser: Any | None = None
        self.context: Any | None = None
        self.page: Any | None = None
        self.launched_browser = False
        self.status = BrowserStatus()
        self._focused_once = False

    async def start(self) -> None:
        if self.playwright is not None:
            return
        from playwright.async_api import async_playwright

        self.playwright = await async_playwright().start()

    async def connect_auto(self) -> BrowserStatus:
        await self.start()
        if self.config.manual_cdp_url:
            return await self.connect_cdp(self.config.manual_cdp_url, mode="manual-cdp")

        tab = await discover_ascent_tab(self.config)
        if tab is not None:
            return await self.connect_discovered(tab)

        if self.config.auto_launch_on_miss:
            return await self.launch()

        self.status = BrowserStatus(
            connected=False,
            mode="disconnected",
            message="No Ascent tab found. Waiting for manual connect.",
        )
        return self.status

    async def connect_discovered(self, tab: CdpTab) -> BrowserStatus:
        status = await self.connect_cdp(tab.cdp_url, mode="auto-attached")
        if status.connected:
            self.status.cdp_url = tab.cdp_url
            self.status.message = f"Attached to existing Ascent tab on port {tab.port}"
        return self.status

    async def connect_cdp(self, cdp_url: str, mode: str = "cdp") -> BrowserStatus:
        await self.start()
        await self.disconnect(close_user_browser=False)
        assert self.playwright is not None
        self.browser = await self.playwright.chromium.connect_over_cdp(cdp_url)
        self.launched_browser = False
        self.context = self.browser.contexts[0] if self.browser.contexts else None
        if self.context is None:
            raise RuntimeError("Connected browser did not expose a context.")
        self.page = await self._select_ascent_page(self.context.pages)
        await self._ensure_page_ready(navigate_if_needed=False)
        self.status = await self._make_status(True, mode, cdp_url)
        return self.status

    async def launch(self) -> BrowserStatus:
        await self.start()
        await self.disconnect(close_user_browser=False)
        assert self.playwright is not None

        layout = self._resolve_window_layout()
        kwargs: dict[str, Any] = {"headless": False}
        if self.config.chromium_path:
            kwargs["executable_path"] = self.config.chromium_path
        args = list(self.config.chromium_args or ())
        vp_w = int(self.config.viewport_width)
        vp_h = int(self.config.viewport_height)
        dpr = float(getattr(self.config, "device_scale_factor", 1.0) or 1.0)
        # Prefer placing on the target monitor before first paint.
        # layout is OUTER window size (viewport + frame pad).
        if layout is not None:
            win_x, win_y, win_w, win_h = layout
            args = [
                a
                for a in args
                if not a.startswith("--window-position=")
                and not a.startswith("--window-size=")
                and not a.startswith("--force-device-scale-factor=")
            ]
            args.extend(
                [
                    f"--window-position={win_x},{win_y}",
                    f"--window-size={win_w},{win_h}",
                    f"--force-device-scale-factor={dpr:g}",
                ]
            )
            print(
                f"CHROMIUM_PIN monitor_pos={win_x},{win_y} "
                f"outer={win_w}x{win_h} viewport={vp_w}x{vp_h} dpr={dpr:g}",
                flush=True,
            )
        else:
            win_w, win_h = self._outer_window_size(vp_w, vp_h)
            args = [
                a
                for a in args
                if not a.startswith("--force-device-scale-factor=")
            ]
            args.append(f"--force-device-scale-factor={dpr:g}")
        if args:
            kwargs["args"] = args
        # Playwright 1.4x defaults to --no-startup-window; on some setups the
        # game window never becomes visible on the target monitor.
        kwargs["ignore_default_args"] = ["--no-startup-window"]
        self.browser = await self.playwright.chromium.launch(**kwargs)
        self.launched_browser = True
        # Content viewport stays at design size so gs=H/700 does not balloon.
        self.context = await self.browser.new_context(
            viewport={"width": vp_w, "height": vp_h},
            device_scale_factor=dpr,
        )
        # Ensure agentMode is set before game.js evaluates AGENT_MODE.
        await self.context.add_init_script(
            """
            window.CHART_TRIAL_CONFIG = Object.assign(
              {},
              window.CHART_TRIAL_CONFIG || {},
              {
                offlineMode: true,
                agentMode: true,
                devUnlockTiers: true,
                trainingTierIndex: 0,
                calendarApiBaseUrl: "https://ascent.xrd.workers.dev",
                runSeed: null,
                lockRunSeed: true,
              }
            );
            try {
              localStorage.setItem("ascent-cosmetics-reveal-dismissed-v4", "1");
            } catch (e) {}
            """
        )
        self.page = await self.context.new_page()
        if layout is not None:
            self._pinned_origin = (layout[0], layout[1])
            self._pinned_size = (layout[2], layout[3])
        else:
            self._pinned_origin = None
            self._pinned_size = (win_w, win_h)
        print("BROWSER_GOTO ascent", flush=True)
        await self._goto_ascent()
        print("BROWSER_PAGE_READY", flush=True)
        await self._ensure_page_ready(navigate_if_needed=True)
        if self._pinned_origin is not None:
            await self._pin_window_bounds(
                self._pinned_origin[0],
                self._pinned_origin[1],
            )
        await self._ensure_client_fits_viewport()
        await self._log_viewport_metrics()
        if getattr(self.config, "raise_on_launch", True):
            await self._raise_game_window()
        self.status = await self._make_status(True, "launched", None)
        print(f"BROWSER_LAUNCHED url={self.page.url!r}", flush=True)
        return self.status

    def _outer_window_size(self, vp_w: int, vp_h: int) -> tuple[int, int]:
        pad_x = int(getattr(self.config, "window_frame_pad_x", 0) or 0)
        pad_y = int(getattr(self.config, "window_frame_pad_y", 80) or 0)
        return vp_w + max(0, pad_x), vp_h + max(0, pad_y)

    def _resolve_window_layout(self) -> tuple[int, int, int, int] | None:
        """Return outer (x, y, w, h) on the target monitor, or None if unpinning."""
        match = str(getattr(self.config, "window_monitor_match", "") or "").strip()
        mon = find_monitor(match) if match else None
        vp_w = int(self.config.viewport_width)
        vp_h = int(self.config.viewport_height)
        want_w, want_h = self._outer_window_size(vp_w, vp_h)
        explicit = getattr(self.config, "window_position", None)
        if mon is not None:
            x, y, w, h = fit_window_to_monitor(
                mon, width=want_w, height=want_h
            )
            if explicit is not None and len(explicit) == 2:
                x, y = int(explicit[0]), int(explicit[1])
            print(
                f"CHROMIUM_PIN matched={mon.name!r} "
                f"geom={mon.x},{mon.y} {mon.width}x{mon.height} "
                f"-> outer={x},{y} {w}x{h} viewport={vp_w}x{vp_h}",
                flush=True,
            )
            return x, y, w, h
        if explicit is not None and len(explicit) == 2:
            return int(explicit[0]), int(explicit[1]), want_w, want_h
        if match:
            print(
                f"CHROMIUM_PIN_WARN no monitor matching {match!r}; "
                "window may open on the primary display",
                flush=True,
            )
        return None

    async def _page_layout_metrics(self) -> dict[str, Any]:
        self._require_page()
        return await self.page.evaluate(
            """() => {
              const canvas = document.querySelector('#gameCanvas');
              const energy = document.querySelector('.energy-hud');
              const bottom = document.querySelector('.bottom-bar');
              const cr = canvas ? canvas.getBoundingClientRect() : null;
              const er = energy ? energy.getBoundingClientRect() : null;
              const br = bottom ? bottom.getBoundingClientRect() : null;
              const ih = window.innerHeight;
              const iw = window.innerWidth;
              return {
                innerW: iw,
                innerH: ih,
                outerW: window.outerWidth,
                outerH: window.outerHeight,
                dpr: window.devicePixelRatio || 1,
                canvasCssW: cr ? cr.width : null,
                canvasCssH: cr ? cr.height : null,
                canvasBufW: canvas ? canvas.width : null,
                canvasBufH: canvas ? canvas.height : null,
                energyBottom: er ? er.bottom : null,
                energyTop: er ? er.top : null,
                energyClipped: er ? (er.bottom > ih + 1 || er.top < -1) : null,
                bottomBarBottom: br ? br.bottom : null,
                bottomBarClipped: br ? (br.bottom > ih + 1) : null,
                gs: (typeof gs === 'number') ? gs : null,
              };
            }"""
        )

    async def _log_viewport_metrics(self) -> None:
        try:
            m = await self._page_layout_metrics()
            print(
                "VIEWPORT_METRICS "
                f"inner={m.get('innerW')}x{m.get('innerH')} "
                f"outer={m.get('outerW')}x{m.get('outerH')} "
                f"dpr={m.get('dpr')} "
                f"canvasCss={m.get('canvasCssW')}x{m.get('canvasCssH')} "
                f"canvasBuf={m.get('canvasBufW')}x{m.get('canvasBufH')} "
                f"gs={m.get('gs')} "
                f"energyClipped={m.get('energyClipped')} "
                f"bottomBarClipped={m.get('bottomBarClipped')}",
                flush=True,
            )
        except Exception as exc:
            print(f"VIEWPORT_METRICS_FAIL {exc}", flush=True)

    async def _ensure_client_fits_viewport(self) -> None:
        """Ensure outer window is large enough for viewport + OS chrome.

        ASCENT's Playwright viewport is the CSS layout size. If --window-size /
        CDP bounds equal that size, the title bar eats client pixels and the
        bottom HUD clips — while game.js still scales sprites with canvas H
        (gs = H/700). Keep content viewport fixed; grow the OUTER window.
        """
        if self.page is None or self.context is None:
            return
        vp_w = int(self.config.viewport_width)
        vp_h = int(self.config.viewport_height)
        pad_x = int(getattr(self.config, "window_frame_pad_x", 0) or 0)
        pad_y = int(getattr(self.config, "window_frame_pad_y", 80) or 0)
        target_w = vp_w + max(0, pad_x)
        target_h = vp_h + max(0, pad_y)
        try:
            # Keep CSS layout at the design viewport even if the OS window grows.
            await self.page.set_viewport_size({"width": vp_w, "height": vp_h})
            session = await asyncio.wait_for(
                self.context.new_cdp_session(self.page),
                timeout=5.0,
            )
            target = await asyncio.wait_for(
                session.send("Browser.getWindowForTarget"),
                timeout=5.0,
            )
            window_id = target.get("windowId")
            if window_id is None:
                return
            for attempt in range(6):
                info = await asyncio.wait_for(
                    session.send(
                        "Browser.getWindowBounds",
                        {"windowId": window_id},
                    ),
                    timeout=5.0,
                )
                bounds = dict(info.get("bounds") or {})
                cur_w = int(bounds.get("width") or 0)
                cur_h = int(bounds.get("height") or 0)
                left = int(bounds.get("left") or 0)
                top = int(bounds.get("top") or 0)
                m = await self._page_layout_metrics()
                outer_w = int(m.get("outerW") or cur_w)
                outer_h = int(m.get("outerH") or cur_h)
                inner_w = int(m.get("innerW") or 0)
                inner_h = int(m.get("innerH") or 0)
                chrome_h = max(0, outer_h - inner_h)
                # Prefer measured outer; fall back to CDP bounds.
                effective_w = max(cur_w, outer_w)
                effective_h = max(cur_h, outer_h)
                need_w = max(0, target_w - effective_w)
                need_h = max(0, target_h - effective_h)
                # If chrome is thinner than expected pad, grow until client can fit.
                if chrome_h < max(24, pad_y // 2) and effective_h < vp_h + pad_y:
                    need_h = max(need_h, (vp_h + pad_y) - effective_h)
                if need_w <= 1 and need_h <= 1:
                    print(
                        f"VIEWPORT_FIT ok attempt={attempt} "
                        f"outer={effective_w}x{effective_h} "
                        f"inner={inner_w}x{inner_h} chrome_h={chrome_h} "
                        f"viewport={vp_w}x{vp_h}",
                        flush=True,
                    )
                    self._pinned_size = (effective_w, effective_h)
                    return
                new_w = effective_w + need_w + 2
                new_h = effective_h + need_h + 2
                await asyncio.wait_for(
                    session.send(
                        "Browser.setWindowBounds",
                        {
                            "windowId": window_id,
                            "bounds": {
                                "left": left,
                                "top": top,
                                "width": new_w,
                                "height": new_h,
                                "windowState": "normal",
                            },
                        },
                    ),
                    timeout=5.0,
                )
                self._pinned_size = (new_w, new_h)
                # Re-assert content viewport so growing the OS window does not
                # enlarge game.js gs (which tracks canvas CSS height).
                await self.page.set_viewport_size({"width": vp_w, "height": vp_h})
                print(
                    f"VIEWPORT_FIT grow attempt={attempt + 1} "
                    f"outer={effective_w}x{effective_h} chrome_h={chrome_h} "
                    f"-> outer={new_w}x{new_h} (keep viewport={vp_w}x{vp_h})",
                    flush=True,
                )
                await asyncio.sleep(0.2)
        except Exception as exc:
            print(f"VIEWPORT_FIT_FAIL {exc}", flush=True)

    async def force_open_game(self) -> BrowserStatus:
        self._require_page()
        await self._goto_ascent()
        await self._ensure_page_ready(navigate_if_needed=False)
        self.status = await self._make_status(True, self.status.mode, self.status.cdp_url)
        return self.status

    async def _select_ascent_page(self, pages: list[Any]) -> Any:
        for page in pages:
            if self.config.host_match in page.url:
                return page
        if pages:
            return pages[0]
        assert self.context is not None
        return await self.context.new_page()

    async def _ensure_page_ready(self, navigate_if_needed: bool) -> None:
        self._require_page()
        if navigate_if_needed and self.config.host_match not in self.page.url:
            await self._goto_ascent()
        # Avoid bring_to_front when pinning to a side monitor — on KDE Wayland
        # this can freeze the X11-ozone Chromium and stall all CDP calls.
        pin = getattr(self, "_pinned_origin", None)
        if not getattr(self, "_focused_once", False):
            if pin is None:
                try:
                    await asyncio.wait_for(self.page.bring_to_front(), timeout=3.0)
                except Exception as exc:
                    print(f"BROWSER_FOCUS_SKIP {exc}", flush=True)
            self._focused_once = True
            if pin is not None:
                await self._pin_window_bounds(pin[0], pin[1])
        try:
            await self.page.wait_for_selector(
                self.config.canvas_selector,
                state="attached",
                timeout=10_000,
            )
        except Exception:
            # Some game boot states delay canvas attachment. The env will retry.
            pass
        if self.config.agent_mode:
            await self.ensure_agent_mode()
        await self.ensure_run_seed()

    async def force_start_game(self) -> str:
        """Best-effort JS start when menu clicks do not advance."""
        self._require_page()
        try:
            result = await asyncio.wait_for(
                self.page.evaluate(
                    """() => {
                        try {
                          document.getElementById('cosmeticsRevealOverlay')
                            ?.classList.add('hidden');
                          const mode = document.querySelector(
                            '.mode-btn[data-ghost="0"]'
                          );
                          mode?.click();
                          const play = document.getElementById('playBtn');
                          const overlay = document.getElementById('startOverlay');
                          if (play && overlay && !overlay.classList.contains('hidden')) {
                            play.click();
                            return 'playBtn';
                          }
                          const grid = document.getElementById('ultiSelectGrid');
                          const confirm = document.getElementById('ultiSelectConfirm');
                          if (grid && confirm) {
                            const card = grid.querySelector('.ulti-card');
                            card?.click();
                            if (!confirm.disabled) confirm.click();
                            else confirm.click();
                            return 'ultiSelectConfirm';
                          }
                          if (typeof startGame === 'function') {
                            startGame();
                            return 'startGame';
                          }
                        } catch (e) { return 'err:' + String(e); }
                        return 'missing';
                    }"""
                ),
                timeout=5.0,
            )
            print(f"BROWSER_FORCE_START result={result}", flush=True)
            return str(result)
        except Exception as exc:
            print(f"BROWSER_FORCE_START_FAIL {exc}", flush=True)
            return "fail"

    async def browser_heartbeat(self) -> dict:
        """Quick liveness probe for watchdog / logs."""
        self._require_page()
        try:
            payload = await asyncio.wait_for(
                self.page.evaluate(
                    """() => {
                        const scoreNode = document.querySelector('#heightScore');
                        const digits = scoreNode
                          ? (scoreNode.textContent || '').replace(/\\D/g, '')
                          : '';
                        const body = document.body
                          ? document.body.innerText.toUpperCase()
                          : '';
                        return {
                          url: location.href,
                          score: digits ? parseInt(digits, 10) : null,
                          inMenu: body.includes('START THE ASCENT')
                            || body.includes('PICK 1 ULTI')
                            || body.includes('PREPARE FOR THE ASCENT'),
                          fell: body.includes('FELL')
                            || body.includes('BACK TO EARTH'),
                          hasCanvas: !!document.querySelector('#gameCanvas'),
                          agent: !!window.__ASCENT_AGENT__,
                        };
                    }"""
                ),
                timeout=5.0,
            )
            if isinstance(payload, dict):
                print(
                    f"BROWSER_ALIVE score={payload.get('score')} "
                    f"menu={payload.get('inMenu')} fell={payload.get('fell')} "
                    f"canvas={payload.get('hasCanvas')} agent={payload.get('agent')}",
                    flush=True,
                )
                return payload
        except Exception as exc:
            print(f"BROWSER_DEAD {exc}", flush=True)
        return {}

    async def _pin_window_bounds(self, x: int, y: int) -> None:
        """Force window bounds via CDP after launch (Chromium sometimes ignores args)."""
        if self.page is None or self.context is None:
            return
        try:
            session = await asyncio.wait_for(
                self.context.new_cdp_session(self.page),
                timeout=5.0,
            )
            target = await asyncio.wait_for(
                session.send("Browser.getWindowForTarget"),
                timeout=5.0,
            )
            window_id = target.get("windowId")
            if window_id is None:
                return
            pin_w, pin_h = getattr(
                self,
                "_pinned_size",
                (self.config.viewport_width, self.config.viewport_height),
            )
            await asyncio.wait_for(
                session.send(
                    "Browser.setWindowBounds",
                    {
                        "windowId": window_id,
                        "bounds": {
                            "left": int(x),
                            "top": int(y),
                            "width": int(pin_w),
                            "height": int(pin_h),
                            "windowState": "normal",
                        },
                    },
                ),
                timeout=5.0,
            )
            print(
                f"CHROMIUM_PIN_CDP bounds={x},{y} {int(pin_w)}x{int(pin_h)}",
                flush=True,
            )
        except Exception as exc:
            print(f"CHROMIUM_PIN_CDP_FAIL {exc}", flush=True)

    async def _raise_game_window(self) -> None:
        """Bring Chromium to the foreground on the pinned monitor."""
        if self.page is None or self.context is None:
            return
        origin = getattr(self, "_pinned_origin", None)
        if origin is not None:
            await self._pin_window_bounds(int(origin[0]), int(origin[1]))
        try:
            await self.page.bring_to_front()
        except Exception as exc:
            print(f"BROWSER_RAISE bring_to_front: {exc}", flush=True)
        try:
            session = await asyncio.wait_for(
                self.context.new_cdp_session(self.page),
                timeout=5.0,
            )
            await asyncio.wait_for(session.send("Page.bringToFront"), timeout=5.0)
            target = await asyncio.wait_for(
                session.send("Browser.getWindowForTarget"),
                timeout=5.0,
            )
            window_id = target.get("windowId")
            if window_id is not None:
                await asyncio.wait_for(
                    session.send(
                        "Browser.setWindowBounds",
                        {
                            "windowId": window_id,
                            "bounds": {"windowState": "normal"},
                        },
                    ),
                    timeout=5.0,
                )
        except Exception as exc:
            print(f"BROWSER_RAISE cdp: {exc}", flush=True)
        try:
            await self.page.evaluate("() => { try { window.focus(); } catch (e) {} }")
        except Exception:
            pass
        if origin is not None:
            print(
                f"BROWSER_RAISE done look_at_monitor x={origin[0]} y={origin[1]}",
                flush=True,
            )
        else:
            print("BROWSER_RAISE done", flush=True)

    async def ensure_agent_mode(self) -> None:
        self._require_page()
        try:
            await self.page.evaluate(_ENSURE_AGENT_MODE_JS)
        except Exception:
            pass

    async def ensure_run_seed(self) -> None:
        """Push BrowserConfig.run_seed into the page before startGame()."""
        self._require_page()
        try:
            await self.page.evaluate(
                _SET_RUN_SEED_JS,
                {
                    "seed": self.config.run_seed,
                    "lock": self.config.lock_run_seed,
                },
            )
        except Exception:
            pass

    async def set_run_seed(self, seed: int | None, *, lock: bool | None = None) -> None:
        """Set/clear the layout seed for the next startGame()."""
        self.config.run_seed = seed
        if lock is not None:
            self.config.lock_run_seed = lock
        await self.ensure_run_seed()

    async def read_agent_state(self) -> dict | None:
        self._require_page()
        try:
            payload = await self.page.evaluate(_READ_AGENT_STATE_JS)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    async def select_training_tier(self, tier_index: int) -> None:
        self._require_page()
        try:
            await self.page.evaluate(
                """(tier) => {
                    if (typeof activeTier !== 'number') return false;
                    activeTier = tier;
                    document.querySelectorAll('.tier-btn').forEach((b, i) => {
                        b.classList.toggle('active', i === tier);
                    });
                    return true;
                }""",
                tier_index,
            )
        except Exception:
            pass

    async def canvas_screenshot(self) -> np.ndarray:
        frame, _ = await self.capture_turn(include_hud=False)
        return frame

    async def capture_turn(self, *, include_hud: bool = True) -> tuple[np.ndarray, HudSnapshot]:
        self._require_page()
        if self.config.use_js_canvas_capture:
            frame, hud = await self._capture_turn_js(include_hud=include_hud)
            if frame is not None:
                return frame, hud
        frame = await self._canvas_screenshot_playwright()
        hud = HudSnapshot()
        if include_hud:
            hud.score = await self.read_game_score()
        return frame, hud

    async def _capture_turn_js(
        self,
        *,
        include_hud: bool,
    ) -> tuple[np.ndarray | None, HudSnapshot]:
        try:
            payload = await asyncio.wait_for(
                self.page.evaluate(
                    _CAPTURE_TURN_JS,
                    {
                        "selector": self.config.canvas_selector,
                        "maxWidth": self.config.capture_max_width,
                        "maxHeight": self.config.capture_max_height,
                        "quality": self.config.capture_jpeg_quality,
                        "boostCost": 14,
                    },
                ),
                timeout=8.0,
            )
        except Exception as exc:
            print(f"BROWSER_CAPTURE_TIMEOUT {exc}", flush=True)
            return None, HudSnapshot()
        if not isinstance(payload, dict):
            return None, HudSnapshot()

        hud = HudSnapshot()
        if include_hud:
            score = payload.get("score")
            if score is not None:
                hud.score = int(score)
            hud.fell = bool(payload.get("fell"))
            hud.in_menu = bool(payload.get("inMenu"))
            energy = payload.get("energy")
            reserve = payload.get("reserve")
            can_boost = payload.get("canBoost")
            if energy is not None:
                hud.energy = float(energy)
            if reserve is not None:
                hud.reserve = float(reserve)
            if can_boost is not None:
                hud.can_boost = bool(can_boost)
            combo = payload.get("combo")
            streak = payload.get("streak")
            multiplier = payload.get("multiplier")
            if combo is not None:
                hud.combo = int(combo)
            if streak is not None:
                hud.streak = int(streak)
            if multiplier is not None:
                hud.multiplier = float(multiplier)

        data_url = payload.get("dataUrl")
        if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
            return None, hud

        raw = base64.b64decode(data_url.split(",", 1)[1])
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.asarray(image), hud

    async def _canvas_screenshot_js(self) -> np.ndarray | None:
        frame, _ = await self._capture_turn_js(include_hud=False)
        return frame

    async def _canvas_screenshot_playwright(self) -> np.ndarray:
        locator = self.page.locator(self.config.canvas_selector)
        png = await locator.screenshot(
            type="png",
            animations="disabled",
            caret="hide",
        )
        image = Image.open(io.BytesIO(png)).convert("RGB")
        return np.asarray(image)

    async def has_canvas(self) -> bool:
        self._require_page()
        try:
            return await self.page.locator(self.config.canvas_selector).count() > 0
        except Exception:
            return False

    async def text_content(self) -> str:
        self._require_page()
        try:
            return await self.page.locator("body").inner_text(timeout=500)
        except Exception:
            return ""

    async def read_game_score(self) -> int | None:
        self._require_page()
        try:
            score = await self.page.evaluate(_READ_SCORE_JS)
            if score is None:
                return None
            return int(score)
        except Exception:
            return None

    async def click_text(self, text: str, timeout: int = 1_000) -> bool:
        self._require_page()
        try:
            await self.page.get_by_text(text, exact=False).click(timeout=timeout)
            return True
        except Exception:
            return False

    async def click_selector(self, selector: str, timeout: int = 1_000) -> bool:
        self._require_page()
        try:
            await self.page.locator(selector).first.click(timeout=timeout)
            return True
        except Exception:
            return False

    async def press(self, key: str) -> None:
        self._require_page()
        await self.page.keyboard.press(key)

    async def key_down(self, key: str) -> None:
        self._require_page()
        await self.page.keyboard.down(key)

    async def key_up(self, key: str) -> None:
        self._require_page()
        await self.page.keyboard.up(key)

    async def wait_ms(self, milliseconds: int) -> None:
        self._require_page()
        await self.page.wait_for_timeout(milliseconds)

    async def disconnect(self, close_user_browser: bool = False) -> None:
        if self.browser is None:
            return
        try:
            await self.browser.close()
        except Exception:
            pass
        self.browser = None
        self.context = None
        self.page = None
        self.launched_browser = False
        self.status = BrowserStatus()
        self._focused_once = False

    async def stop(self) -> None:
        await self.disconnect(close_user_browser=False)
        if self.playwright is not None:
            await self.playwright.stop()
            self.playwright = None
        stop_game_server_if_started()

    async def _goto_ascent(self) -> None:
        await asyncio.to_thread(ensure_game_server, self.config.ascent_url)
        self._require_page()
        await self.page.goto(self.config.ascent_url, wait_until="domcontentloaded")

    async def _make_status(
        self,
        connected: bool,
        mode: str,
        cdp_url: str | None,
    ) -> BrowserStatus:
        title = ""
        url = ""
        if self.page is not None:
            try:
                title = await self.page.title()
                url = self.page.url
            except Exception:
                pass
        return BrowserStatus(
            connected=connected,
            mode=mode,
            title=title,
            url=url,
            cdp_url=cdp_url,
            message=f"Connected ({mode})" if connected else "Not connected",
        )

    def _require_page(self) -> None:
        if self.page is None:
            raise RuntimeError("Browser is not connected.")


async def smoke_connect(config: BrowserConfig) -> BrowserStatus:
    backend = BrowserBackend(config)
    try:
        return await backend.connect_auto()
    finally:
        await backend.stop()

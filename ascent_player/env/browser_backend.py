from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ascent_player.config import BrowserConfig
from ascent_player.env.browser_discovery import CdpTab, discover_ascent_tab


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

        kwargs: dict[str, Any] = {"headless": False}
        if self.config.chromium_path:
            kwargs["executable_path"] = self.config.chromium_path
        self.browser = await self.playwright.chromium.launch(**kwargs)
        self.launched_browser = True
        self.context = await self.browser.new_context(
            viewport={
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            }
        )
        self.page = await self.context.new_page()
        await self.page.goto(self.config.ascent_url, wait_until="domcontentloaded")
        await self._ensure_page_ready(navigate_if_needed=True)
        self.status = await self._make_status(True, "launched", None)
        return self.status

    async def force_open_game(self) -> BrowserStatus:
        self._require_page()
        await self.page.goto(self.config.ascent_url, wait_until="domcontentloaded")
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
            await self.page.goto(self.config.ascent_url, wait_until="domcontentloaded")
        await self.page.bring_to_front()
        try:
            await self.page.wait_for_selector(
                self.config.canvas_selector,
                state="attached",
                timeout=10_000,
            )
        except Exception:
            # Some game boot states delay canvas attachment. The env will retry.
            pass

    async def canvas_screenshot(self) -> np.ndarray:
        self._require_page()
        locator = self.page.locator(self.config.canvas_selector)
        png = await locator.screenshot(type="png")
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

    async def click_text(self, text: str, timeout: int = 1_000) -> bool:
        self._require_page()
        try:
            await self.page.get_by_text(text, exact=False).click(timeout=timeout)
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

    async def stop(self) -> None:
        await self.disconnect(close_user_browser=False)
        if self.playwright is not None:
            await self.playwright.stop()
            self.playwright = None

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

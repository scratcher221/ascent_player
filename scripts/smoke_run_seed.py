#!/usr/bin/env python3
"""Smoke-test: same runSeed => identical initial platform/booster layout."""
from __future__ import annotations

import asyncio
import json
import sys

URL = (
    "http://127.0.0.1:8765/ASCENT%20%E2%80%94%20Ride%20the%20pump.html"
    "?runSeed=424242&devUnlockTiers"
)


async def click_through_menus(page) -> None:
    for _ in range(12):
        agent = await page.evaluate("() => window.__ASCENT_AGENT__ || null")
        if isinstance(agent, dict) and agent.get("state") == "playing":
            return

        has_play = await page.evaluate(
            """() => !!document.getElementById("playBtn")
                && !document.getElementById("startOverlay")?.classList.contains("hidden")"""
        )
        if has_play:
            await page.click("#playBtn")
            await asyncio.sleep(0.2)
            continue

        has_ulti = await page.evaluate(
            """() => !!document.getElementById("ultiSelectConfirm")
                && !document.getElementById("ultiSelectScreen")?.classList.contains("hidden")"""
        )
        if has_ulti:
            await page.evaluate(
                """() => {
                    const card = document.querySelector("#ultiSelectGrid .ulti-card");
                    if (card) card.click();
                    const confirm = document.getElementById("ultiSelectConfirm");
                    if (confirm && !confirm.disabled) confirm.click();
                    else if (confirm) confirm.click();
                }"""
            )
            await asyncio.sleep(0.25)
            continue

        await asyncio.sleep(0.1)
    raise RuntimeError("Could not start game through menus")


async def layout_snapshot(page) -> dict:
    await page.goto(URL, wait_until="domcontentloaded")
    await page.wait_for_selector("#gameCanvas", timeout=15_000)
    await page.wait_for_function(
        "() => typeof window.__ASCENT_SET_RUN_SEED__ === 'function'",
        timeout=10_000,
    )
    await page.evaluate(
        """() => {
            window.CHART_TRIAL_CONFIG = window.CHART_TRIAL_CONFIG || {};
            window.CHART_TRIAL_CONFIG.agentMode = true;
            window.CHART_TRIAL_CONFIG.runSeed = 424242;
            window.CHART_TRIAL_CONFIG.lockRunSeed = true;
            window.__ASCENT_SET_RUN_SEED__(424242);
        }"""
    )
    await click_through_menus(page)

    for _ in range(80):
        payload = await page.evaluate(
            """() => {
                const a = window.__ASCENT_AGENT__;
                if (!a || a.state !== "playing") return null;
                return {
                    runSeed: a.runSeed ?? window.__ASCENT_RUN_SEED__,
                    platforms: (a.platforms || []).map(p => ({
                        x: Math.round(p.x * 100) / 100,
                        worldY: Math.round(p.worldY * 100) / 100,
                        width: Math.round((p.width ?? 0) * 100) / 100,
                        type: p.type || "neutral",
                    })),
                    boosters: (a.boosters || []).map(b => ({
                        x: Math.round(b.x * 100) / 100,
                        worldY: Math.round(b.worldY * 100) / 100,
                        type: b.type,
                    })),
                };
            }"""
        )
        if payload:
            return payload
        await asyncio.sleep(0.05)
    raise RuntimeError("Game never reached playing state with agent export")


async def main() -> int:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        a = await layout_snapshot(page)
        await page.close()
        page2 = await browser.new_page(viewport={"width": 1280, "height": 720})
        b = await layout_snapshot(page2)
        await browser.close()

    print(
        "run A seed:",
        a["runSeed"],
        "platforms:",
        len(a["platforms"]),
        "boosters:",
        len(a["boosters"]),
    )
    print(
        "run B seed:",
        b["runSeed"],
        "platforms:",
        len(b["platforms"]),
        "boosters:",
        len(b["boosters"]),
    )
    if a != b:
        print("MISMATCH")
        print("A:", json.dumps(a, indent=2)[:1200])
        print("B:", json.dumps(b, indent=2)[:1200])
        return 1
    print("OK: identical layout for runSeed=424242")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

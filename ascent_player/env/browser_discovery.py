from __future__ import annotations

import asyncio
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import httpx
import psutil

from ascent_player.config import BrowserConfig


@dataclass(slots=True)
class CdpTab:
    cdp_url: str
    port: int
    target_id: str
    title: str
    url: str
    web_socket_debugger_url: str | None = None


@dataclass(slots=True)
class BrowserWindow:
    title: str
    pid: int | None = None
    executable: str | None = None
    url_hint: str | None = None


async def _get_json(client: httpx.AsyncClient, url: str) -> Any | None:
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


async def _probe_port(
    client: httpx.AsyncClient,
    port: int,
    host_match: str,
) -> list[CdpTab]:
    cdp_url = f"http://127.0.0.1:{port}"
    version = await _get_json(client, f"{cdp_url}/json/version")
    if not version:
        return []

    tab_payload = await _get_json(client, f"{cdp_url}/json/list")
    if not isinstance(tab_payload, list):
        return []

    tabs: list[CdpTab] = []
    for entry in tab_payload:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "page":
            continue
        url = str(entry.get("url") or "")
        if host_match not in url:
            continue
        tabs.append(
            CdpTab(
                cdp_url=cdp_url,
                port=port,
                target_id=str(entry.get("id") or ""),
                title=str(entry.get("title") or ""),
                url=url,
                web_socket_debugger_url=entry.get("webSocketDebuggerUrl"),
            )
        )
    return tabs


async def discover_ascent_tab(config: BrowserConfig) -> CdpTab | None:
    timeout = httpx.Timeout(config.cdp_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        probes = [
            _probe_port(client, port, config.host_match)
            for port in config.cdp_ports
        ]
        results = await asyncio.gather(*probes)

    matches = [tab for tabs in results for tab in tabs]
    if not matches:
        return None

    # Prefer the canonical URL/title. Playwright will verify the canvas later.
    matches.sort(
        key=lambda tab: (
            tab.url.rstrip("/") != config.ascent_url.rstrip("/"),
            "ASCENT" not in tab.title.upper(),
            tab.port,
        )
    )
    return matches[0]


async def list_cdp_tabs(config: BrowserConfig) -> list[CdpTab]:
    timeout = httpx.Timeout(config.cdp_timeout_seconds)
    tabs: list[CdpTab] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        probes = [_list_tabs_for_port(client, port) for port in config.cdp_ports]
        results = await asyncio.gather(*probes)
    for port_tabs in results:
        tabs.extend(port_tabs)
    return tabs


async def _list_tabs_for_port(
    client: httpx.AsyncClient,
    port: int,
) -> list[CdpTab]:
    cdp_url = f"http://127.0.0.1:{port}"
    payload = await _get_json(client, f"{cdp_url}/json/list")
    if not isinstance(payload, list):
        return []

    tabs: list[CdpTab] = []
    for entry in payload:
        if not isinstance(entry, dict) or entry.get("type") != "page":
            continue
        tabs.append(
            CdpTab(
                cdp_url=cdp_url,
                port=port,
                target_id=str(entry.get("id") or ""),
                title=str(entry.get("title") or ""),
                url=str(entry.get("url") or ""),
                web_socket_debugger_url=entry.get("webSocketDebuggerUrl"),
            )
        )
    return tabs


def list_chromium_windows() -> list[BrowserWindow]:
    windows = _list_windows_with_wmctrl()
    if windows:
        return windows
    return _list_chromium_processes()


def _list_windows_with_wmctrl() -> list[BrowserWindow]:
    if not shutil.which("wmctrl"):
        return []
    try:
        output = subprocess.check_output(
            ["wmctrl", "-lp"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    windows: list[BrowserWindow] = []
    for line in output.splitlines():
        parts = line.split(maxsplit=4)
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[2])
        except ValueError:
            pid = None
        title = parts[4]
        if "chrom" not in title.lower() and "ascent" not in title.lower():
            continue
        executable = _exe_for_pid(pid) if pid else None
        windows.append(BrowserWindow(title=title, pid=pid, executable=executable))
    return windows


def _list_chromium_processes() -> list[BrowserWindow]:
    names = ("chromium", "chrome", "google-chrome")
    windows: list[BrowserWindow] = []
    for proc in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if not any(candidate in name or candidate in cmdline for candidate in names):
                continue
            url_hint = next(
                (arg for arg in proc.info.get("cmdline") or [] if arg.startswith("http")),
                None,
            )
            windows.append(
                BrowserWindow(
                    title=cmdline[:120] or name,
                    pid=proc.info.get("pid"),
                    executable=proc.info.get("exe"),
                    url_hint=url_hint,
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return windows


def _exe_for_pid(pid: int) -> str | None:
    try:
        return psutil.Process(pid).exe()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def relaunch_command(window: BrowserWindow, config: BrowserConfig) -> str:
    executable = window.executable or "chromium"
    return (
        f"{executable} --remote-debugging-port=9222 "
        f"{config.ascent_url}"
    )

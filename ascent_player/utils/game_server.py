from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GAME_DIR = _PROJECT_ROOT / "game"
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

_process: subprocess.Popen[bytes] | None = None
_we_started = False


def is_local_ascent_url(url: str) -> bool:
    host = urlparse(url).hostname
    return host in _LOCAL_HOSTS if host else False


def _connect_host(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8765
    if host in _LOCAL_HOSTS:
        host = "127.0.0.1"
    return host, port


def _port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _server_ready(host: str, port: int, timeout: float = 0.5) -> bool:
    if not _port_open(host, port, timeout):
        return False
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=timeout) as resp:
            return resp.status == 200
    except (OSError, urllib.error.URLError, ValueError):
        return False


def ensure_game_server(
    ascent_url: str,
    *,
    game_dir: Path | None = None,
    timeout_seconds: float = 10.0,
) -> None:
    """Start the bundled http.server for local Ascent URLs when nothing is listening."""
    if not is_local_ascent_url(ascent_url):
        return

    host, port = _connect_host(ascent_url)
    if _server_ready(host, port):
        return

    global _process, _we_started

    if _process is not None and _process.poll() is None:
        _wait_until_ready(host, port, timeout_seconds, check_process=True)
        return

    directory = game_dir or _DEFAULT_GAME_DIR
    if not directory.is_dir():
        raise RuntimeError(f"Game directory not found: {directory}")

    _process = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=directory,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _we_started = True
    _wait_until_ready(host, port, timeout_seconds, check_process=True)


def stop_game_server_if_started() -> None:
    global _process, _we_started
    if not _we_started or _process is None:
        return
    _process.terminate()
    try:
        _process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        _process.kill()
    _process = None
    _we_started = False


def _wait_until_ready(
    host: str,
    port: int,
    timeout_seconds: float,
    *,
    check_process: bool,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _server_ready(host, port):
            return
        if check_process and _process is not None and _process.poll() is not None:
            raise RuntimeError(f"Local game server exited unexpectedly (port {port})")
        time.sleep(0.1)

    raise RuntimeError(
        f"Local game server did not become ready on http://{host}:{port}/ "
        f"within {timeout_seconds:.0f}s. Run ./game/serve.sh manually."
    )

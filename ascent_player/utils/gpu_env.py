from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_PRELOADED = False


def _site_package_roots() -> list[Path]:
    roots: list[Path] = []
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    roots.append(Path(sys.prefix) / "lib" / f"python{version}" / "site-packages")
    try:
        import site

        for path in site.getsitepackages():
            if path:
                roots.append(Path(path))
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(Path(user_site))
    except Exception:
        pass
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def nvidia_library_dirs() -> list[str]:
    lib_dirs: list[str] = []
    seen: set[str] = set()
    for root in _site_package_roots():
        nvidia_root = root / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for child in sorted(nvidia_root.iterdir()):
            lib_path = child / "lib"
            if not lib_path.is_dir():
                continue
            resolved = str(lib_path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            lib_dirs.append(resolved)
    return lib_dirs


def preload_nvidia_libraries() -> int:
    """Load pip-shipped NVIDIA shared libraries before TensorFlow initializes CUDA."""
    global _PRELOADED
    if _PRELOADED:
        return 0

    count = 0
    for lib_dir in nvidia_library_dirs():
        for library in sorted(Path(lib_dir).glob("*.so*")):
            if library.is_symlink():
                continue
            if ".so" not in library.name:
                continue
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                count += 1
            except OSError:
                continue
    _PRELOADED = True
    return count


def configure_nvidia_library_path() -> list[str]:
    """Best-effort LD_LIBRARY_PATH update (used by wrapper scripts)."""
    lib_dirs = nvidia_library_dirs()
    if not lib_dirs:
        return []

    prepend = os.pathsep.join(lib_dirs)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if current and not current.startswith(prepend):
        os.environ["LD_LIBRARY_PATH"] = f"{prepend}{os.pathsep}{current}"
    elif not current:
        os.environ["LD_LIBRARY_PATH"] = prepend
    return lib_dirs


def ensure_driver_visible() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def bootstrap_gpu_environment() -> int:
    ensure_driver_visible()
    configure_nvidia_library_path()
    return preload_nvidia_libraries()

"""Cross-platform system information utilities.

Zero external dependencies — uses only stdlib and OS-level APIs.
"""

from __future__ import annotations

import logging
import platform
import subprocess

logger = logging.getLogger(__name__)


def get_total_ram_gb() -> float:
    """Return total system RAM in GB.

    Detection strategy per platform:
    - macOS/Linux: ``os.sysconf`` (no subprocess), fallback to
      ``sysctl`` (macOS) or ``/proc/meminfo`` (Linux)
    - Windows: ``ctypes`` call to ``GlobalMemoryStatusEx``

    Returns 0.0 on failure (treated as low-RAM → safest model choice).
    """
    # Try os.sysconf first — works on macOS and Linux without subprocess
    try:
        return _ram_sysconf()
    except Exception:
        pass

    system = platform.system()
    try:
        if system == "Darwin":
            return _ram_macos()
        elif system == "Linux":
            return _ram_linux()
        elif system == "Windows":
            return _ram_windows()
        else:
            logger.warning("Unsupported platform for RAM detection: %s", system)
            return 0.0
    except Exception:
        logger.warning("Failed to detect system RAM", exc_info=True)
        return 0.0


def _ram_sysconf() -> float:
    """macOS/Linux: os.sysconf for physical pages × page size."""
    import os

    pages = os.sysconf("SC_PHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    if pages <= 0 or page_size <= 0:
        raise RuntimeError("os.sysconf returned non-positive values")
    return (pages * page_size) / (1024 ** 3)


def _ram_macos() -> float:
    """macOS: sysctl returns total physical memory in bytes."""
    raw = subprocess.check_output(
        ["sysctl", "-n", "hw.memsize"],
        timeout=5,
    )
    return int(raw.strip()) / (1024 ** 3)


def _ram_linux() -> float:
    """Linux: /proc/meminfo reports MemTotal in kB."""
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                # Format: "MemTotal:       16384000 kB"
                parts = line.split()
                return int(parts[1]) / (1024 ** 2)
    raise RuntimeError("MemTotal not found in /proc/meminfo")


def _ram_windows() -> float:
    """Windows: GlobalMemoryStatusEx via ctypes."""
    import ctypes
    import ctypes.wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.wintypes.DWORD),
            ("dwMemoryLoad", ctypes.wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        raise RuntimeError("GlobalMemoryStatusEx failed")
    return stat.ullTotalPhys / (1024 ** 3)

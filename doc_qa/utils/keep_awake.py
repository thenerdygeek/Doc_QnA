"""Prevent the OS from sleeping while a long-running task is active.

Usage::

    with keep_awake("Indexing 2900 files"):
        do_long_task()

Platform support:

- **Windows**: ``SetThreadExecutionState`` — tells Windows the app is busy.
  The display may still turn off, but the system won't sleep/hibernate.
- **macOS**: ``caffeinate -i`` subprocess — prevents idle sleep.
- **Linux**: ``systemd-inhibit`` subprocess — prevents idle sleep.
- **Other / failure**: Silently does nothing (indexing still works, just
  won't prevent sleep).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


@contextmanager
def keep_awake(reason: str = "Background task in progress") -> Generator[None, None, None]:
    """Context manager that prevents OS idle sleep."""
    release = _acquire(reason)
    try:
        yield
    finally:
        release()


def _acquire(reason: str) -> callable:
    """Platform-specific sleep inhibition. Returns a release function."""
    if sys.platform == "win32":
        return _acquire_windows(reason)
    elif sys.platform == "darwin":
        return _acquire_macos(reason)
    else:
        return _acquire_linux(reason)


# ── Windows ──────────────────────────────────────────────────────


def _acquire_windows(reason: str) -> callable:
    """Use SetThreadExecutionState to prevent sleep on Windows."""
    try:
        import ctypes

        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        # ES_CONTINUOUS = 0x80000000  — keep the setting until cleared
        # ES_SYSTEM_REQUIRED = 0x00000001 — system must stay awake
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001

        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        logger.info("Sleep prevention enabled (Windows): %s", reason)

        def release():
            # Clear the flag — allow sleep again
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            logger.info("Sleep prevention released (Windows)")

        return release

    except Exception as e:
        logger.debug("Could not prevent sleep on Windows: %s", e)
        return lambda: None


# ── macOS ────────────────────────────────────────────────────────


def _acquire_macos(reason: str) -> callable:
    """Spawn caffeinate to prevent idle sleep on macOS."""
    try:
        # -i = prevent idle sleep, -w = exit when parent dies (safety net)
        proc = subprocess.Popen(
            ["caffeinate", "-i"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Sleep prevention enabled (macOS caffeinate PID %d): %s", proc.pid, reason)

        def release():
            proc.terminate()
            proc.wait(timeout=5)
            logger.info("Sleep prevention released (macOS)")

        return release

    except Exception as e:
        logger.debug("Could not prevent sleep on macOS: %s", e)
        return lambda: None


# ── Linux ────────────────────────────────────────────────────────


def _acquire_linux(reason: str) -> callable:
    """Use systemd-inhibit to prevent idle sleep on Linux."""
    try:
        # systemd-inhibit runs as a wrapper — we use --what=idle
        proc = subprocess.Popen(
            [
                "systemd-inhibit",
                "--what=idle",
                f"--reason={reason}",
                "sleep", "infinity",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Sleep prevention enabled (Linux systemd-inhibit PID %d): %s", proc.pid, reason)

        def release():
            proc.terminate()
            proc.wait(timeout=5)
            logger.info("Sleep prevention released (Linux)")

        return release

    except Exception as e:
        logger.debug("Could not prevent sleep on Linux: %s", e)
        return lambda: None

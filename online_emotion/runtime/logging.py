"""Lightweight logging helpers.

Logging level is read once from ``$ONLINE_INFERENCE_LOG`` (default ``INFO``).
``warn_once`` is used for things like the batched->per-item fallback so the
terminal is not flooded. All loggers live under the ``online_inference``
namespace so a single handler/level governs the package.
"""
from __future__ import annotations

import logging
import os

_ROOT = "online_inference"
_CONFIGURED = False
_WARNED: set[str] = set()


def _configure() -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger(_ROOT)
    # Dedupe across sibling packages: both online_face and online_emotion ship this
    # module and share the global "online_inference" logger, so guard on a tagged
    # handler (not just the per-module flag) to avoid duplicate log lines.
    if not any(getattr(h, "_online_inference", False) for h in root.handlers):
        level = os.getenv("ONLINE_INFERENCE_LOG", "INFO").upper()
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
        handler._online_inference = True  # type: ignore[attr-defined]
        root.addHandler(handler)
        root.setLevel(getattr(logging, level, logging.INFO))
        root.propagate = False
    _CONFIGURED = True
    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``online_inference`` namespace."""
    root = _configure()
    if not name or name == _ROOT:
        return root
    return logging.getLogger(f"{_ROOT}.{name}")


def warn_once(logger: logging.Logger, key: str, message: str) -> None:
    """Emit ``message`` at WARNING level at most once per ``key`` per process."""
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message)

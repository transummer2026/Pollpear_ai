"""Process-wide logging configuration and a thin logger accessor."""

import logging
import sys

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Send root-logger output to stdout with a single shared formatter.

    Existing root handlers are cleared first so records are not emitted
    twice when a host (e.g. uvicorn) has already installed its own handler.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return the named logger (thin wrapper to keep call sites uniform)."""
    return logging.getLogger(name)

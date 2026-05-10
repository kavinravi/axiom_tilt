"""Structured logging configuration."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure root logger. Idempotent."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        return root

    root.setLevel(level.upper())
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

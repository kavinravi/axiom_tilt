"""Project path helpers anchored at the repo root."""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def raw_dir() -> Path:
    p = data_dir() / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def interim_dir() -> Path:
    p = data_dir() / "interim"
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir() -> Path:
    p = data_dir() / "processed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def edgar_raw_dir() -> Path:
    p = raw_dir() / "edgar"
    p.mkdir(parents=True, exist_ok=True)
    return p


def state_dir() -> Path:
    """For checkpoint/resume state files."""
    p = data_dir() / "state"
    p.mkdir(parents=True, exist_ok=True)
    return p

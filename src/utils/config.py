"""YAML config loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.io import repo_root


def load_config(name: str) -> dict[str, Any]:
    """Load a config file by name from the configs/ directory.

    Example: load_config('data') -> reads configs/data.yaml.
    """
    path = repo_root() / "configs" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)

"""Environment variable loading with .env file support."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


class EnvError(RuntimeError):
    """Raised when a required environment variable is missing."""


_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    _loaded = True


def get_env(
    key: str,
    *,
    required: bool = False,
    default: str | None = None,
) -> str | None:
    """Read an env var. Strips whitespace. Loads .env on first call."""
    _ensure_loaded()
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        if required:
            raise EnvError(
                f"Required environment variable {key!r} is unset. "
                f"Add it to .env (see .env.example)."
            )
        return default
    return raw.strip()

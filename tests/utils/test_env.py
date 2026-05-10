"""Tests for src.utils.env."""
import os
import pytest

from src.utils.env import get_env, EnvError


def test_get_env_returns_value(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "abc123")
    assert get_env("TEST_KEY") == "abc123"


def test_get_env_required_missing_raises(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(EnvError, match="MISSING_KEY"):
        get_env("MISSING_KEY", required=True)


def test_get_env_optional_missing_returns_default(monkeypatch):
    monkeypatch.delenv("OPT_KEY", raising=False)
    assert get_env("OPT_KEY", default="fallback") == "fallback"


def test_get_env_strips_whitespace(monkeypatch):
    monkeypatch.setenv("PADDED", "  hello  ")
    assert get_env("PADDED") == "hello"

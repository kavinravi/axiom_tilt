"""Tests for src.utils.rate_limit."""
import time

from src.utils.rate_limit import TokenBucket


def test_immediate_first_call_does_not_block():
    bucket = TokenBucket(rate_per_sec=10.0, capacity=10)
    t0 = time.monotonic()
    bucket.acquire()
    assert time.monotonic() - t0 < 0.05


def test_burst_within_capacity_does_not_block():
    bucket = TokenBucket(rate_per_sec=10.0, capacity=5)
    t0 = time.monotonic()
    for _ in range(5):
        bucket.acquire()
    assert time.monotonic() - t0 < 0.1


def test_exceeding_capacity_blocks():
    # capacity=2, rate=10/s. After bursting 2, the 3rd acquire should wait ~0.1s.
    bucket = TokenBucket(rate_per_sec=10.0, capacity=2)
    bucket.acquire()
    bucket.acquire()
    t0 = time.monotonic()
    bucket.acquire()
    elapsed = time.monotonic() - t0
    assert 0.08 <= elapsed <= 0.25, f"expected ~0.1s wait, got {elapsed:.3f}s"


def test_steady_state_rate_is_respected():
    # rate=20/s, request 10 tokens in tight loop -> should take ~0.5s
    bucket = TokenBucket(rate_per_sec=20.0, capacity=1)
    t0 = time.monotonic()
    for _ in range(10):
        bucket.acquire()
    elapsed = time.monotonic() - t0
    assert 0.4 <= elapsed <= 0.7, f"expected ~0.5s, got {elapsed:.3f}s"

"""Token-bucket rate limiter for HTTP clients (notably SEC EDGAR)."""
from __future__ import annotations

import threading
import time


class TokenBucket:
    """Thread-safe token-bucket rate limiter.

    Tokens accumulate at `rate_per_sec` up to `capacity`. `acquire()` blocks
    until at least one token is available, then consumes one.
    """

    def __init__(self, rate_per_sec: float, capacity: int) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._rate = float(rate_per_sec)
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last = now

    def acquire(self) -> None:
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = 1.0 - self._tokens
                sleep_for = deficit / self._rate
            # Sleep outside the lock so other threads can refill.
            time.sleep(sleep_for)

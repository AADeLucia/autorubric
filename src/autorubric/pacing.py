"""Request pacing and backend health tracking for LLM calls.

Two independent concerns, both global singletons so limits are respected
account-wide rather than per-client:

`AsyncRateLimiter` / `RateLimiterPool`
    A token bucket that *waits* for a free slot. This replaces the previous
    `litellm.Router` + `enforce_model_rate_limits` approach, which was an
    admission gate rather than a limiter: it incremented a counter, raised
    `litellm.RateLimitError` when the count exceeded the ceiling, and never
    refunded the increment -- so retries of a rejected call pushed the counter
    further past the limit and kept it there for the rest of the calendar
    minute. Pacing removes the error entirely: a burst of any size simply
    drains at the configured rate.

`BackendHealth` / `BackendHealthPool`
    A circuit breaker over *consecutive infrastructure failures*, so a backend
    that has gone away (a dead vLLM server, a severed tunnel) stops the run
    instead of letting every remaining item fail individually. A brief blip
    must not kill a multi-hour run, so the policy is pause -> probe on a
    backoff schedule -> abort only if unrecovered within a bounded wait.
"""

from __future__ import annotations

import asyncio
import logging
import time
from threading import Lock
from typing import ClassVar

from .llm_errors import BackendUnavailableError, classify_grading_error

logger = logging.getLogger(__name__)


def normalize_to_provider(model: str) -> str:
    """Group a model identifier to its provider, since limits are account-wide.

    "openai/gpt-4" and "openai/gpt-4-turbo" both map to "openai".
    """
    if "/" in model:
        return model.split("/")[0]
    return model


class AsyncRateLimiter:
    """Token bucket that awaits a free slot instead of raising.

    Refills continuously at `rpm / 60` tokens per second up to `burst`
    capacity, measured on a monotonic clock.

    `burst` defaults to a tenth of the per-minute allowance (minimum 1) rather
    than to the full allowance. A full-allowance bucket would let `rpm`
    requests fire simultaneously from a cold start, which stays within a
    server-side *fixed* window but can violate a *rolling* one if the burst
    straddles a boundary. A small burst absorbs ordinary scheduling jitter
    while keeping the request stream close to evenly spaced.
    """

    def __init__(self, rpm: int | None = None, tpm: int | None = None, burst: int | None = None):
        if rpm is None and tpm is None:
            raise ValueError("AsyncRateLimiter requires at least one of rpm/tpm")

        self._request_bucket = _TokenBucket(rpm, burst) if rpm is not None else None
        # A token-per-minute ceiling is paced on an *estimated* cost supplied by
        # the caller, since real usage is only known after the response.
        self._token_bucket = _TokenBucket(tpm, burst=tpm) if tpm is not None else None

    async def acquire(self, estimated_tokens: int = 0) -> None:
        """Wait until this request may proceed."""
        if self._request_bucket is not None:
            await self._request_bucket.take(1.0)
        if self._token_bucket is not None and estimated_tokens > 0:
            await self._token_bucket.take(float(estimated_tokens))


class _TokenBucket:
    """Continuously-refilling bucket. `per_minute` units become available each minute."""

    def __init__(self, per_minute: int, burst: int | None = None):
        if per_minute <= 0:
            raise ValueError(f"rate must be positive, got {per_minute}")
        self._rate = per_minute / 60.0
        self._capacity = float(max(1, burst if burst is not None else max(1, per_minute // 10)))
        self._tokens = self._capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def take(self, tokens: float) -> None:
        # A single request may not cost more than the bucket can ever hold.
        tokens = min(tokens, self._capacity)
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait = (tokens - self._tokens) / self._rate
            await asyncio.sleep(wait)


class RateLimiterPool:
    """Global singleton holding one `AsyncRateLimiter` per provider.

    Keyed by provider rather than by (model, rpm, tpm) because API limits are
    account-wide: two models behind the same gateway share one ceiling. When
    the same provider is requested with different limits the *strictest* wins,
    matching `RateLimitPool`'s handling of `max_parallel_requests`.

    The pool is per *process*. Two runs sharing one API key will each pace to
    `rpm` and collectively exceed it -- observed against the WSE gateway on
    2026-10-07, where two concurrent processes at rpm=50 drew a server-side
    `GATEWAY_KEY_RPM_LIMITED` 429. Set `rpm` to the key's share, not the key's
    whole ceiling, when more than one run is in flight.
    """

    _instance: ClassVar[RateLimiterPool | None] = None
    _lock: ClassVar[Lock] = Lock()

    def __init__(self) -> None:
        self._limiters: dict[str, AsyncRateLimiter] = {}
        self._limits: dict[str, tuple[int | None, int | None]] = {}
        self._async_lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> RateLimiterPool:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def get_limiter(
        self,
        model: str,
        rpm: int | None,
        tpm: int | None = None,
    ) -> AsyncRateLimiter | None:
        """Return the limiter for this model's provider, or None if unlimited."""
        if rpm is None and tpm is None:
            return None

        async with self._async_lock:
            key = normalize_to_provider(model)
            existing = self._limits.get(key)
            if existing is not None:
                strictest = (_stricter(existing[0], rpm), _stricter(existing[1], tpm))
                if strictest == existing:
                    return self._limiters[key]
                rpm, tpm = strictest

            self._limits[key] = (rpm, tpm)
            self._limiters[key] = AsyncRateLimiter(rpm=rpm, tpm=tpm)
            return self._limiters[key]

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the singleton. For tests needing a clean slate."""
        with cls._lock:
            cls._instance = None


def _stricter(current: int | None, incoming: int | None) -> int | None:
    if current is None:
        return incoming
    if incoming is None:
        return current
    return min(current, incoming)


class BackendHealth:
    """Circuit breaker over consecutive infrastructure failures for one backend.

    Policy is pause -> probe -> abort. Once `threshold` consecutive
    infrastructure failures are seen the gate closes and callers queue. They
    are released one at a time, each after the next delay in `backoff`, and
    each released caller acts as the probe: its success reopens the gate for
    everyone, its failure extends the pause. If the accumulated pause exceeds
    `max_wait` the backend is declared dead and every caller -- current and
    future -- raises `BackendUnavailableError`.

    Only *infrastructure* failures count, classified by the existing
    `classify_grading_error`. Any success resets the counter, so intermittent
    errors never trip the breaker.
    """

    def __init__(
        self,
        key: str,
        threshold: int,
        backoff: tuple[float, ...],
        max_wait: float,
    ):
        self.key = key
        self._threshold = threshold
        self._backoff = backoff or (30.0,)
        self._max_wait = max_wait

        self._consecutive = 0
        self._dead = False
        self._attempt = 0
        self._waited = 0.0
        self._gate = asyncio.Event()
        self._gate.set()
        self._probe_lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self._threshold > 0

    async def before_request(self) -> None:
        """Block while the backend is paused; raise once it is declared dead."""
        if self._dead:
            raise BackendUnavailableError(self._dead_message())
        if self._gate.is_set():
            return

        # Paused. Callers serialize here, so nothing hammers a dead backend.
        async with self._probe_lock:
            if self._dead:
                raise BackendUnavailableError(self._dead_message())
            if self._gate.is_set():
                return

            delay = self._backoff[min(self._attempt, len(self._backoff) - 1)]
            if self._waited + delay > self._max_wait:
                self._dead = True
                # Wake everyone so they can raise rather than wait forever.
                self._gate.set()
                logger.error(
                    f"[{self.key}] backend unreachable for {self._waited:.0f}s "
                    f"after {self._consecutive} consecutive infrastructure failures; aborting"
                )
                raise BackendUnavailableError(self._dead_message())

            self._attempt += 1
            self._waited += delay
            logger.warning(
                f"[{self.key}] paused after {self._consecutive} consecutive infrastructure "
                f"failures; retrying in {delay:.0f}s (waited {self._waited:.0f}s of {self._max_wait:.0f}s)"
            )
            await asyncio.sleep(delay)
            # This caller is now the probe: it proceeds while the gate stays shut.

    def record_success(self) -> None:
        if not self.enabled:
            return
        if not self._gate.is_set():
            logger.warning(f"[{self.key}] backend recovered; resuming")
        self._consecutive = 0
        self._attempt = 0
        self._waited = 0.0
        self._gate.set()

    def record_failure(self, exc: BaseException) -> None:
        if not self.enabled:
            return
        if classify_grading_error(exc) != "infrastructure":
            return
        self._consecutive += 1
        if self._consecutive >= self._threshold:
            self._gate.clear()

    def _dead_message(self) -> str:
        return (
            f"Backend {self.key!r} is unreachable: {self._consecutive} consecutive "
            f"infrastructure failures and no recovery within {self._max_wait:.0f}s. "
            f"Aborting rather than issuing further doomed requests."
        )


class BackendHealthPool:
    """Global singleton holding one `BackendHealth` per backend endpoint."""

    _instance: ClassVar[BackendHealthPool | None] = None
    _lock: ClassVar[Lock] = Lock()

    def __init__(self) -> None:
        self._health: dict[str, BackendHealth] = {}
        self._async_lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> BackendHealthPool:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def get_health(
        self,
        key: str,
        threshold: int,
        backoff: tuple[float, ...],
        max_wait: float,
    ) -> BackendHealth | None:
        """Return the breaker for this backend, or None when disabled."""
        if threshold <= 0:
            return None
        async with self._async_lock:
            health = self._health.get(key)
            if health is None:
                health = BackendHealth(key, threshold, backoff, max_wait)
                self._health[key] = health
            return health

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the singleton. For tests needing a clean slate."""
        with cls._lock:
            cls._instance = None

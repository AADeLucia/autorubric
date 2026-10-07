"""Tests for request pacing and backend health tracking."""

import asyncio
import time

import litellm
import pytest

from autorubric.llm_errors import BackendUnavailableError
from autorubric.pacing import (
    AsyncRateLimiter,
    BackendHealth,
    BackendHealthPool,
    RateLimiterPool,
    normalize_to_provider,
)


def infra_error() -> litellm.APIConnectionError:
    return litellm.APIConnectionError(message="backend gone", llm_provider="openai", model="m")


class TestAsyncRateLimiter:
    @pytest.mark.asyncio
    async def test_paces_requests_beyond_burst(self):
        # 6000/min = 100/s, burst of 1, so 11 acquisitions must span ~10 refills.
        limiter = AsyncRateLimiter(rpm=6000, burst=1)
        start = time.monotonic()
        for _ in range(11):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.08, f"expected pacing, finished in {elapsed:.3f}s"
        assert elapsed < 1.0, f"pacing far slower than the configured rate: {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_concurrent_burst_is_paced_not_rejected(self):
        """The shape that used to raise: far more concurrent callers than the rate."""
        limiter = AsyncRateLimiter(rpm=6000, burst=1)
        await asyncio.gather(*(limiter.acquire() for _ in range(32)))
        # Reaching here at all is the assertion: no caller was rejected.

    @pytest.mark.asyncio
    async def test_burst_capacity_allows_immediate_start(self):
        limiter = AsyncRateLimiter(rpm=60, burst=5)
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        assert time.monotonic() - start < 0.2

    def test_requires_a_limit(self):
        with pytest.raises(ValueError):
            AsyncRateLimiter()


class TestRateLimiterPool:
    def setup_method(self):
        RateLimiterPool.reset_instance()

    def teardown_method(self):
        RateLimiterPool.reset_instance()

    @pytest.mark.asyncio
    async def test_returns_none_when_unlimited(self):
        pool = RateLimiterPool.get_instance()
        assert await pool.get_limiter("openai/gpt-4", None, None) is None

    @pytest.mark.asyncio
    async def test_shared_per_provider(self):
        pool = RateLimiterPool.get_instance()
        a = await pool.get_limiter("openai/gpt-4", 100)
        b = await pool.get_limiter("openai/gpt-5", 100)
        assert a is b, "models behind one provider must share its account-wide ceiling"

    @pytest.mark.asyncio
    async def test_strictest_limit_wins(self):
        pool = RateLimiterPool.get_instance()
        await pool.get_limiter("openai/gpt-4", 100)
        await pool.get_limiter("openai/gpt-5", 20)
        assert pool._limits["openai"][0] == 20

    @pytest.mark.asyncio
    async def test_looser_limit_does_not_relax_existing(self):
        pool = RateLimiterPool.get_instance()
        await pool.get_limiter("openai/gpt-4", 20)
        await pool.get_limiter("openai/gpt-5", 100)
        assert pool._limits["openai"][0] == 20

    def test_provider_normalization(self):
        assert normalize_to_provider("openai/gpt-4") == "openai"
        assert normalize_to_provider("hosted_vllm/Qwen/Qwen3.5-27B") == "hosted_vllm"
        assert normalize_to_provider("ollama") == "ollama"


class TestBackendHealth:
    @pytest.fixture(autouse=True)
    def no_real_sleeping(self, monkeypatch):
        self.slept: list[float] = []

        async def fake_sleep(delay):
            self.slept.append(delay)

        monkeypatch.setattr("autorubric.pacing.asyncio.sleep", fake_sleep)

    @pytest.mark.asyncio
    async def test_disabled_by_default_threshold(self):
        health = BackendHealth("h", threshold=0, backoff=(30.0,), max_wait=900.0)
        assert not health.enabled
        for _ in range(50):
            health.record_failure(infra_error())
        await health.before_request()  # never pauses

    @pytest.mark.asyncio
    async def test_passes_through_while_healthy(self):
        health = BackendHealth("h", threshold=3, backoff=(30.0,), max_wait=900.0)
        await health.before_request()
        assert self.slept == []

    @pytest.mark.asyncio
    async def test_only_infrastructure_failures_count(self):
        health = BackendHealth("h", threshold=2, backoff=(30.0,), max_wait=900.0)
        health.record_failure(ValueError("unparseable"))
        health.record_failure(ValueError("unparseable"))
        await health.before_request()
        assert self.slept == [], "parse failures must not trip the breaker"

    @pytest.mark.asyncio
    async def test_success_resets_the_counter(self):
        health = BackendHealth("h", threshold=3, backoff=(30.0,), max_wait=900.0)
        health.record_failure(infra_error())
        health.record_failure(infra_error())
        health.record_success()
        health.record_failure(infra_error())
        await health.before_request()
        assert self.slept == [], "an intermittent blip must not trip the breaker"

    @pytest.mark.asyncio
    async def test_pauses_after_threshold_then_recovers(self):
        health = BackendHealth("h", threshold=2, backoff=(30.0, 60.0), max_wait=900.0)
        health.record_failure(infra_error())
        health.record_failure(infra_error())

        await health.before_request()  # becomes the probe
        assert self.slept == [30.0]

        health.record_success()
        await health.before_request()
        assert self.slept == [30.0], "recovery must stop the pausing"

    @pytest.mark.asyncio
    async def test_backoff_lengthens_while_still_failing(self):
        health = BackendHealth("h", threshold=1, backoff=(30.0, 60.0, 120.0), max_wait=900.0)
        health.record_failure(infra_error())

        for _ in range(3):
            await health.before_request()
            health.record_failure(infra_error())

        assert self.slept == [30.0, 60.0, 120.0]

    @pytest.mark.asyncio
    async def test_aborts_once_max_wait_exceeded(self):
        health = BackendHealth("h", threshold=1, backoff=(30.0, 60.0), max_wait=50.0)
        health.record_failure(infra_error())

        await health.before_request()  # 30s, within budget
        health.record_failure(infra_error())

        with pytest.raises(BackendUnavailableError, match="unreachable"):
            await health.before_request()  # 30+60 > 50

    @pytest.mark.asyncio
    async def test_stays_dead_for_later_callers(self):
        health = BackendHealth("h", threshold=1, backoff=(100.0,), max_wait=50.0)
        health.record_failure(infra_error())
        with pytest.raises(BackendUnavailableError):
            await health.before_request()
        with pytest.raises(BackendUnavailableError):
            await health.before_request()

    @pytest.mark.asyncio
    async def test_concurrent_callers_all_abort(self):
        """Every waiter must learn the backend died, not hang."""
        health = BackendHealth("h", threshold=1, backoff=(100.0,), max_wait=50.0)
        health.record_failure(infra_error())

        results = await asyncio.gather(
            *(health.before_request() for _ in range(5)), return_exceptions=True
        )
        assert all(isinstance(r, BackendUnavailableError) for r in results)


class TestBreakerInterruptsAGatheredBatch:
    """The breaker must stop a batch that is already in flight.

    Regression: the gate was originally checked once per request before
    queueing, so every coroutine in an asyncio.gather passed it before the
    first failure had been recorded. Against a dead backend the whole batch ran
    to completion and the breaker never fired -- the exact outcome it exists to
    prevent. It is now checked after the semaphore and rate limiter, so only
    the in-flight few are ever past the gate.
    """

    def setup_method(self):
        RateLimiterPool.reset_instance()
        BackendHealthPool.reset_instance()

    def teardown_method(self):
        RateLimiterPool.reset_instance()
        BackendHealthPool.reset_instance()

    @pytest.mark.asyncio
    async def test_dead_backend_aborts_before_issuing_every_request(self, monkeypatch):
        from unittest.mock import AsyncMock, patch

        from autorubric.llm import LLMClient, LLMConfig

        async def no_sleep(delay):
            return None

        monkeypatch.setattr("autorubric.pacing.asyncio.sleep", no_sleep)

        config = LLMConfig(
            model="hosted_vllm/dead",
            api_base="http://127.0.0.1:9/v1",
            max_retries=0,
            max_parallel_requests=4,
            max_consecutive_infra_failures=5,
            infra_retry_backoff=(1.0,),
            infra_retry_max_wait=2.0,
        )
        client = LLMClient(config)

        attempts = 0

        async def always_down(**kwargs):
            nonlocal attempts
            attempts += 1
            raise infra_error()

        with patch("autorubric.llm.litellm.acompletion", new=AsyncMock(side_effect=always_down)):
            results = await asyncio.gather(
                *(
                    client.generate(system_prompt="s", user_prompt=f"u{i}")
                    for i in range(200)
                ),
                return_exceptions=True,
            )

        assert any(isinstance(r, BackendUnavailableError) for r in results), (
            "a dead backend must surface BackendUnavailableError"
        )
        assert attempts < 200, (
            f"breaker never interrupted the batch: all {attempts} requests were issued"
        )


class TestBackendHealthPool:
    def setup_method(self):
        BackendHealthPool.reset_instance()

    def teardown_method(self):
        BackendHealthPool.reset_instance()

    @pytest.mark.asyncio
    async def test_none_when_disabled(self):
        pool = BackendHealthPool.get_instance()
        assert await pool.get_health("k", 0, (30.0,), 900.0) is None

    @pytest.mark.asyncio
    async def test_shared_per_backend_key(self):
        pool = BackendHealthPool.get_instance()
        a = await pool.get_health("http://server-a", 5, (30.0,), 900.0)
        b = await pool.get_health("http://server-a", 5, (30.0,), 900.0)
        c = await pool.get_health("http://server-b", 5, (30.0,), 900.0)
        assert a is b
        assert a is not c, "distinct endpoints must fail independently"

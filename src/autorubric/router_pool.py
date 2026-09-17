"""Global RPM/TPM-aware routing for LLM requests across all clients.

This module provides a singleton RouterPool that manages litellm.Router
instances, one per unique (model, rpm, tpm) combination. Unlike
RateLimitPool's plain concurrency semaphore, a litellm.Router with
`optional_pre_call_checks=["enforce_model_rate_limits"]` enforces a true
rolling 60s requests-per-minute (and tokens-per-minute) ceiling by raising
litellm.RateLimitError before a call would exceed it -- the same exception
type LLMConfig's tenacity retry decorator already retries on, so RPM-gated
calls get backoff/retry for free.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, ClassVar

from litellm import Router


class RouterPool:
    """Global singleton managing litellm.Router instances for RPM/TPM limiting.

    Thread-safe management of Router instances shared across all LLMClient
    instances that request the same (model, rpm, tpm) combination, so an RPM
    ceiling is respected account-wide rather than per-client.

    Usage:
        pool = RouterPool.get_instance()
        router = pool.get_router(model="openai/gpt-5.4", rpm=50, tpm=None)
        response = await router.acompletion(**params)
    """

    _instance: ClassVar[RouterPool | None] = None
    _lock: ClassVar[Lock] = Lock()

    def __init__(self) -> None:
        """Initialize the router pool.

        This should not be called directly - use get_instance() instead.
        """
        self._routers: dict[tuple[str, int | None, int | None], Router] = {}
        self._pool_lock = Lock()

    @classmethod
    def get_instance(cls) -> RouterPool:
        """Get or create the singleton instance.

        Thread-safe singleton pattern using double-checked locking.

        Returns:
            The global RouterPool instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_router(
        self,
        model: str,
        rpm: int | None,
        tpm: int | None = None,
    ) -> Router:
        """Get or create a Router enforcing the given RPM/TPM ceiling for this model.

        Args:
            model: Model identifier in LiteLLM format (e.g. "openai/gpt-5.4").
            rpm: Requests-per-minute ceiling to enforce. At least one of
                rpm/tpm must be set -- callers should not invoke this when
                both are None (there'd be nothing to enforce).
            tpm: Tokens-per-minute ceiling to enforce (optional).

        Returns:
            A litellm.Router with a single deployment for `model`, configured
            to raise litellm.RateLimitError via its `enforce_model_rate_limits`
            pre-call check once the rolling-window ceiling is exceeded.

        Note:
            Routers are cached per (model, rpm, tpm) key -- calling this again
            with the same key returns the same Router instance (and its
            in-memory rolling-window counters), so the limit is respected
            across all callers sharing that key, not reset per call.
        """
        key = (model, rpm, tpm)
        with self._pool_lock:
            router = self._routers.get(key)
            if router is None:
                deployment: dict[str, Any] = {
                    "model_name": model,
                    "litellm_params": {"model": model},
                }
                if rpm is not None:
                    deployment["rpm"] = rpm
                if tpm is not None:
                    deployment["tpm"] = tpm
                router = Router(
                    model_list=[deployment],
                    optional_pre_call_checks=["enforce_model_rate_limits"],
                    num_retries=0,
                )
                self._routers[key] = router
            return router

    @classmethod
    def reset_instance(cls) -> None:
        """Completely reset the singleton instance.

        Use this for testing when you need a fresh instance.
        """
        with cls._lock:
            cls._instance = None

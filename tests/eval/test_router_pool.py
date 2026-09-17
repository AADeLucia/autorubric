"""Tests for RouterPool RPM/TPM rate limiting infrastructure."""

from litellm import Router
from litellm.router_utils.pre_call_checks.model_rate_limit_check import (
    ModelRateLimitingCheck,
)

from autorubric.router_pool import RouterPool


class TestRouterPoolSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        RouterPool.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        RouterPool.reset_instance()

    def test_get_instance_returns_same_instance(self):
        """Test that get_instance always returns the same instance."""
        instance1 = RouterPool.get_instance()
        instance2 = RouterPool.get_instance()
        assert instance1 is instance2

    def test_reset_instance_creates_new_instance(self):
        """Test that reset_instance creates a fresh instance."""
        instance1 = RouterPool.get_instance()
        RouterPool.reset_instance()
        instance2 = RouterPool.get_instance()
        assert instance1 is not instance2


class TestRouterPoolGetRouter:
    """Tests for router creation and caching."""

    def setup_method(self):
        """Reset singleton before each test."""
        RouterPool.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        RouterPool.reset_instance()

    def test_get_router_returns_router_instance(self):
        """Test that get_router returns a litellm.Router."""
        pool = RouterPool.get_instance()
        router = pool.get_router("openai/gpt-5.4", rpm=50, tpm=None)
        assert isinstance(router, Router)

    def test_same_key_returns_cached_router(self):
        """Test that repeated calls with the same (model, rpm, tpm) reuse one Router."""
        pool = RouterPool.get_instance()
        router1 = pool.get_router("openai/gpt-5.4", rpm=50, tpm=None)
        router2 = pool.get_router("openai/gpt-5.4", rpm=50, tpm=None)
        assert router1 is router2

    def test_different_rpm_gets_different_router(self):
        """Test that a different rpm for the same model creates a distinct Router."""
        pool = RouterPool.get_instance()
        router_50 = pool.get_router("openai/gpt-5.4", rpm=50, tpm=None)
        router_10 = pool.get_router("openai/gpt-5.4", rpm=10, tpm=None)
        assert router_50 is not router_10

    def test_different_models_get_different_routers(self):
        """Test that different models get distinct Routers even with the same rpm."""
        pool = RouterPool.get_instance()
        router_a = pool.get_router("openai/gpt-5.4", rpm=50, tpm=None)
        router_b = pool.get_router("anthropic/claude-sonnet-4-5", rpm=50, tpm=None)
        assert router_a is not router_b

    def test_router_enforces_rate_limit_pre_call_check(self):
        """Test that the Router registers a ModelRateLimitingCheck callback."""
        pool = RouterPool.get_instance()
        router = pool.get_router("openai/gpt-5.4", rpm=50, tpm=None)
        assert any(
            isinstance(cb, ModelRateLimitingCheck) for cb in (router.optional_callbacks or [])
        )

    def test_router_deployment_carries_rpm_and_tpm(self):
        """Test that rpm/tpm land on the Router's model_list deployment."""
        pool = RouterPool.get_instance()
        router = pool.get_router("openai/gpt-5.4", rpm=50, tpm=1000)
        deployment = router.get_model_list(model_name="openai/gpt-5.4")[0]
        assert deployment["rpm"] == 50
        assert deployment["tpm"] == 1000

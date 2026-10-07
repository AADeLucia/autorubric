"""LLM calling infrastructure with multi-provider support.

This module provides a unified interface to 100+ LLM providers via LiteLLM,
with support for structured outputs, automatic retries, response caching,
and provider-specific features like extended thinking and prompt caching.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import diskcache
import litellm
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from autorubric.llm_errors import (
    BackendUnavailableError,
    ErrorCategory,
    classify_grading_error,
)
from autorubric.pacing import BackendHealthPool, RateLimiterPool, normalize_to_provider
from autorubric.rate_limit import RateLimitPool

if TYPE_CHECKING:
    from autorubric.types import TokenUsage

# Load environment variables from .env file (idempotent - safe to call multiple times)
load_dotenv()

logger = logging.getLogger(__name__)

# Type variable for structured output
T = TypeVar("T", bound=BaseModel)


# Grading error classification lives in llm_errors so the pacing layer can use it
# without a circular import. Re-exported here for backward compatibility.
__all_errors__ = ("ErrorCategory", "classify_grading_error", "BackendUnavailableError")


# ============================================================================
# Thinking/Reasoning Configuration
# ============================================================================


class ThinkingLevel(str, Enum):
    """Standardized thinking/reasoning effort levels across LLM providers.

    LiteLLM translates these to provider-specific parameters:
    - Anthropic: thinking={type, budget_tokens} with low→1024, medium→2048, high→4096
    - OpenAI: reasoning_effort parameter for o-series and GPT-5 models
    - Gemini: thinking configuration with similar token budgets
    - DeepSeek: Standardized via LiteLLM's reasoning_effort

    Higher levels mean more "thinking" tokens/steps, trading latency for quality.
    """

    NONE = "none"  # Disable thinking (where supported, e.g., Gemini)
    LOW = "low"  # Light reasoning: ~1024 tokens budget
    MEDIUM = "medium"  # Moderate reasoning: ~2048 tokens budget
    HIGH = "high"  # Deep reasoning: ~4096 tokens budget


# String literal type for convenience
ThinkingLevelLiteral = Literal["none", "low", "medium", "high"]


# Token budget mapping for providers that support explicit budgets
THINKING_LEVEL_BUDGETS: dict[str, int] = {
    "none": 0,
    "low": 1024,
    "medium": 2048,
    "high": 4096,
}


@dataclass
class ThinkingConfig:
    """Detailed configuration for LLM thinking/reasoning.

    Provides a uniform interface across providers:
    - Anthropic: Extended thinking (claude-sonnet-4-5, claude-opus-4-5+)
    - OpenAI: Reasoning (o-series, GPT-5 models via openai/responses/ prefix)
    - Gemini: Thinking mode (2.5+, 3.0+ models)
    - DeepSeek: Reasoning content

    Attributes:
        level: High-level thinking effort. Used when budget_tokens is not set.
            Defaults to MEDIUM for a good balance of quality and latency.
        budget_tokens: Explicit token budget for thinking (provider-specific).
            When set, overrides level. Recommended: 10000-50000 for complex tasks.
            Providers that don't support explicit budgets will map this to the
            nearest level.

    Examples:
        # Simple: use a thinking level
        ThinkingConfig(level=ThinkingLevel.HIGH)

        # Fine-grained: specify exact token budget
        ThinkingConfig(budget_tokens=32000)  # For complex reasoning tasks

        # Disable thinking
        ThinkingConfig(level=ThinkingLevel.NONE)
    """

    level: ThinkingLevel | ThinkingLevelLiteral = ThinkingLevel.MEDIUM
    budget_tokens: int | None = None

    def __post_init__(self) -> None:
        """Normalize level to enum."""
        if isinstance(self.level, str):
            self.level = ThinkingLevel(self.level)

    def get_effective_budget(self) -> int:
        """Get the effective token budget based on level or explicit budget."""
        if self.budget_tokens is not None:
            return self.budget_tokens
        return THINKING_LEVEL_BUDGETS.get(self.level.value, 2048)

    def get_reasoning_effort(self) -> str:
        """Get the reasoning_effort string for LiteLLM."""
        return self.level.value


# Union type for flexible thinking parameter in LLMConfig
ThinkingParam = ThinkingConfig | ThinkingLevel | ThinkingLevelLiteral | int | None
"""Type for the thinking parameter in LLMConfig.

Accepts:
- ThinkingConfig: Full configuration object
- ThinkingLevel: Enum value (e.g., ThinkingLevel.HIGH)
- str: Level as string ("low", "medium", "high", "none")
- int: Direct token budget (e.g., 32000)
- None: Disable thinking
"""


def _normalize_thinking_param(thinking: ThinkingParam) -> ThinkingConfig | None:
    """Convert various thinking parameter formats to ThinkingConfig."""
    if thinking is None:
        return None
    if isinstance(thinking, ThinkingConfig):
        return thinking
    if isinstance(thinking, ThinkingLevel):
        return ThinkingConfig(level=thinking)
    if isinstance(thinking, str):
        return ThinkingConfig(level=ThinkingLevel(thinking))
    if isinstance(thinking, int):
        return ThinkingConfig(budget_tokens=thinking)
    raise TypeError(f"Invalid thinking parameter type: {type(thinking)}")


@dataclass
class GenerateResult:
    """Result from LLM generation including content, thinking, and usage statistics.

    Attributes:
        content: The main response content from the LLM (raw string).
        thinking: The thinking/reasoning trace if thinking was enabled.
            None if thinking was not enabled or the provider doesn't support it.
        raw_response: The raw LiteLLM response object for advanced use cases.
        usage: Token usage statistics from this LLM call.
        cost: Completion cost in USD for this LLM call, calculated using
            LiteLLM's completion_cost() function. None if cost calculation fails.
        parsed: The parsed Pydantic model instance when response_format was provided.
            None if no response_format was used or parsing failed.
    """

    content: str
    thinking: str | None = None
    raw_response: Any = None
    usage: TokenUsage | None = None
    cost: float | None = None
    parsed: Any = None


@dataclass
class GenerateRequest:
    """One prompt pair for `LLMClient.generate_many`."""

    user_prompt: str
    system_prompt: str = ""


@dataclass
class GenerateOutcome:
    """Result of one request in a batch, successful or not.

    `error` is set instead of raising so that one failed request cannot cancel
    the rest of the batch. Exactly one of `content`/`error` is populated.
    """

    index: int
    content: str | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None


def _extract_thinking_content(message: Any) -> str | None:
    """Extract thinking/reasoning content from LLM response message.

    LiteLLM standardizes reasoning content across providers:
    - `reasoning_content`: Unified field across all providers (preferred)
    - `thinking_blocks`: Anthropic-specific list of thinking blocks (fallback)
    - `thinking`: Legacy Anthropic field (fallback)

    Args:
        message: The message object from response.choices[0].message

    Returns:
        The thinking/reasoning content as a string, or None if not present.
    """
    # Primary: LiteLLM's standardized reasoning_content field
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        return message.reasoning_content

    # Fallback: Anthropic-specific thinking_blocks
    if hasattr(message, "thinking_blocks") and message.thinking_blocks:
        blocks = message.thinking_blocks
        if isinstance(blocks, list):
            thinking_parts = []
            for block in blocks:
                if isinstance(block, dict) and block.get("thinking"):
                    thinking_parts.append(block["thinking"])
                elif hasattr(block, "thinking") and block.thinking:
                    thinking_parts.append(block.thinking)
            if thinking_parts:
                return "\n".join(thinking_parts)

    # Fallback: Legacy thinking field
    if hasattr(message, "thinking") and message.thinking:
        return message.thinking

    return None


def _extract_usage_from_response(response: Any) -> TokenUsage:
    """Extract token usage from LiteLLM response.

    LiteLLM provides an OpenAI-compatible usage object:
    - prompt_tokens: Number of tokens in the prompt
    - completion_tokens: Number of tokens in the completion
    - total_tokens: Sum of prompt + completion tokens

    Additional fields may be present for specific providers:
    - cache_creation_input_tokens: Tokens used to create cache (Anthropic)
    - cache_read_input_tokens: Tokens read from cache (Anthropic)

    Args:
        response: The LiteLLM response object

    Returns:
        TokenUsage object with usage statistics. Returns zeros if usage not available.
    """
    # Import here to avoid circular import
    from autorubric.types import TokenUsage

    if not hasattr(response, "usage") or response.usage is None:
        return TokenUsage()

    usage = response.usage

    # Extract standard OpenAI-compatible fields
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0

    # Extract Anthropic prompt caching fields if present
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )


def _calculate_completion_cost(response: Any) -> float | None:
    """Calculate completion cost using LiteLLM's completion_cost function.

    Uses LiteLLM's built-in cost calculation which has pricing data for
    all supported providers.

    Args:
        response: The LiteLLM response object

    Returns:
        Cost in USD, or None if cost calculation fails.
    """
    try:
        # LiteLLM's completion_cost accepts the response object directly
        cost = litellm.completion_cost(completion_response=response)
        return float(cost) if cost is not None else None
    except Exception as e:
        logger.debug(f"Could not calculate completion cost: {e}")
        return None


_debug_console = Console()


def _estimate_request_tokens(system_prompt: str, user_prompt: str, max_tokens: int | None) -> int:
    """Rough token cost of a request, for tpm pacing only.

    Real usage is only known after the response, so a tpm ceiling has to be
    paced on an estimate: ~4 characters per token for the prompt, plus the
    full output allowance as a worst case.
    """
    prompt_tokens = (len(system_prompt) + len(user_prompt)) // 4
    return prompt_tokens + (max_tokens or 0)


def _print_debug_prompt(system_prompt: str, user_prompt: str, model: str) -> None:
    """Print the fully constructed LLM prompt using Rich formatting."""
    _debug_console.print()
    _debug_console.rule(f"[bold cyan]DEBUG PROMPT — {model}[/bold cyan]")
    _debug_console.print(
        Panel(system_prompt, title="[bold green]System Prompt[/bold green]", border_style="green")
    )
    _debug_console.print(
        Panel(
            Syntax(user_prompt, "xml", theme="monokai", word_wrap=True),
            title="[bold yellow]User Prompt[/bold yellow]",
            border_style="yellow",
        )
    )
    _debug_console.rule("[bold cyan]END DEBUG PROMPT[/bold cyan]")
    _debug_console.print()


def _provider_response_format(response_format: type[BaseModel]) -> type[BaseModel] | dict[str, Any]:
    """Build the ``response_format`` payload sent to the provider for structured output.

    The judgment schemas carry an optional ``reasoning`` field that is never read from the
    model's output: it is injected after the fact from the provider's separate thinking
    channel (see ``LLMClient.generate``). Under OpenAI/Groq *strict* structured output, every
    property is forced into ``required``, so leaving ``reasoning`` in the schema makes models
    that follow the prompt — which never asks for ``reasoning`` — fail schema validation and
    abstain (observed for Llama and Qwen on Groq, where the verdict collapses to
    ``CANNOT_ASSESS``). Strip ``reasoning`` from what the provider must emit; parsing still
    uses the full Pydantic model, so reasoning injection when thinking is enabled is unaffected.

    Models without a ``reasoning`` field are returned unchanged. Any unexpected schema shape
    falls back to the original model so behaviour never regresses.
    """
    if "reasoning" not in getattr(response_format, "model_fields", {}):
        return response_format
    try:
        from litellm.utils import type_to_response_format_param

        param = type_to_response_format_param(response_format)
    except Exception:
        param = None
    # Unknown litellm shape — keep current behaviour (pass the model through).
    if not isinstance(param, dict):
        return response_format
    schema = param.get("json_schema", {}).get("schema")
    if not isinstance(schema, dict):
        return response_format
    schema.get("properties", {}).pop("reasoning", None)
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [key for key in required if key != "reasoning"]
    return param


@dataclass
class LLMConfig:
    """Configuration for LLM calls.

    Attributes:
        model: Model identifier in LiteLLM format (e.g., "openai/gpt-5.2",
               "anthropic/claude-sonnet-4-5-20250929", "gemini/gemini-3-pro-preview",
               "ollama/qwen3:14b"). REQUIRED - no default.
               See LiteLLM docs for full list of supported models.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in response.
        top_p: Nucleus sampling parameter.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts for transient failures.
        retry_min_wait: Minimum wait between retries (seconds).
        retry_max_wait: Maximum wait between retries (seconds).
        max_parallel_requests: Maximum concurrent requests to this model's provider.
            When set, a global per-provider semaphore limits parallel requests.
            None (default) means unlimited parallel requests.
        rpm: Requests-per-minute ceiling for this model's provider, enforced by
            a token bucket that *waits* for a free slot (see pacing.py). Unlike
            max_parallel_requests, which only approximates a rate and drifts
            with latency, this paces the request stream directly: a burst of
            any size drains at `rpm` without ever raising. None (default)
            disables pacing. Can be combined with max_parallel_requests/tpm.
        tpm: Tokens-per-minute ceiling, paced by the same mechanism on an
            estimated per-request cost. None (default) disables it.
        max_consecutive_infra_failures: Trip the backend circuit breaker after
            this many consecutive infrastructure failures, so a backend that
            has gone away stops the run instead of failing every remaining
            item individually. 0 (default) disables the breaker; ~10 is
            reasonable in practice. Any success resets the counter.
        infra_retry_backoff: Delays between probe attempts while the breaker is
            tripped. The last value repeats if more probes are needed.
        infra_retry_max_wait: Total time to keep probing a tripped backend
            before raising BackendUnavailableError.
        cache_enabled: Default caching behavior (can be overridden per-request).
        cache_dir: Directory for response cache.
        cache_ttl: Cache time-to-live in seconds (None = no expiration).
        api_key: Optional API key override (otherwise uses environment variables).
        api_base: Optional API base URL override.
        thinking: Enable thinking/reasoning mode (unified across providers). Accepts
            multiple formats:
            - ThinkingLevel enum: ThinkingLevel.HIGH, ThinkingLevel.MEDIUM, etc.
            - String: "low", "medium", "high", "none"
            - Int: Direct token budget (e.g., 32000)
            - ThinkingConfig: Full configuration with level and/or budget_tokens
            - None: Disable thinking (default)

            Provider support:
            - Anthropic: Extended thinking (claude-sonnet-4-5, claude-opus-4-5+)
            - OpenAI: Reasoning for o-series and GPT-5 models
            - Gemini: Thinking mode (2.5+, 3.0+ models)
            - DeepSeek: Reasoning content
        prompt_caching: Enable prompt caching for supported models (default: True).
            When enabled, automatically detects if the model supports caching via
            litellm.supports_prompt_caching() and applies provider-specific config:
            - Anthropic: Adds cache_control to system messages + beta header
            - OpenAI/Deepseek: Automatic for prompts ≥1024 tokens (no extra config)
            - Bedrock: Supported for all models
            Set to False to disable prompt caching entirely.
        seed: Random seed for reproducible outputs (OpenAI, some other providers).
        extra_headers: Additional HTTP headers for provider-specific features.
        extra_params: Additional provider-specific parameters passed to LiteLLM.

    Examples:
        # Basic usage without thinking
        config = LLMConfig(model="openai/gpt-5.2")

        # Enable thinking with a level
        config = LLMConfig(model="anthropic/claude-sonnet-4-5-20250929", thinking="high")
        config = LLMConfig(model="openai/responses/gpt-5-mini", thinking=ThinkingLevel.HIGH)

        # Enable thinking with explicit token budget
        config = LLMConfig(model="anthropic/claude-opus-4-5-20251101", thinking=32000)

        # Full control with ThinkingConfig
        config = LLMConfig(
            model="gemini/gemini-2.5-pro",
            thinking=ThinkingConfig(level=ThinkingLevel.HIGH, budget_tokens=50000)
        )
    """

    model: str  # REQUIRED - no default, must always be specified
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None
    timeout: float = 60.0
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 60.0
    max_parallel_requests: int | None = None
    rpm: int | None = None
    tpm: int | None = None
    max_consecutive_infra_failures: int = 0
    infra_retry_backoff: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0)
    infra_retry_max_wait: float = 900.0
    cache_enabled: bool = False
    cache_dir: str | Path = ".autorubric_cache"
    cache_ttl: int | None = None  # None = no expiration
    api_key: str | None = None
    api_base: str | None = None

    # Thinking/Reasoning (unified across providers)
    thinking: ThinkingParam = None

    # Other provider-specific features
    prompt_caching: bool = True  # Enable prompt caching by default for supported models
    seed: int | None = None  # OpenAI reproducibility
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_params: dict[str, Any] = field(default_factory=dict)

    def get_thinking_config(self) -> ThinkingConfig | None:
        """Get normalized thinking configuration."""
        return _normalize_thinking_param(self.thinking)

    @classmethod
    def from_yaml(cls, path: str | Path) -> LLMConfig:
        """Load LLMConfig from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            LLMConfig instance with values from the YAML file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If required fields are missing or invalid.

        Example YAML file (llm_config.yaml):
            model: openai/gpt-5.2
            temperature: 0.0
            max_tokens: 1024
            cache_enabled: true
            cache_ttl: 3600
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LLM config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML config: expected dict, got {type(data).__name__}")

        if "model" not in data:
            raise ValueError("LLM config YAML must specify 'model' field")

        # Handle extra_params specially - any unknown keys go there
        known_fields = {
            "model",
            "temperature",
            "max_tokens",
            "top_p",
            "timeout",
            "max_retries",
            "retry_min_wait",
            "retry_max_wait",
            # Concurrency, pacing, and backend health. Omitting these silently
            # routed them into extra_params and shipped them to litellm as junk
            # kwargs, leaving rate limiting off for every YAML-loaded config.
            "max_parallel_requests",
            "rpm",
            "tpm",
            "max_consecutive_infra_failures",
            "infra_retry_backoff",
            "infra_retry_max_wait",
            "cache_enabled",
            "cache_dir",
            "cache_ttl",
            "api_key",
            "api_base",
            # Thinking/Reasoning
            "thinking",
            # Other provider-specific features
            "prompt_caching",
            "seed",
            "extra_headers",
            "extra_params",
        }
        extra = {k: v for k, v in data.items() if k not in known_fields}
        if extra:
            data.setdefault("extra_params", {}).update(extra)
            for k in extra:
                del data[k]

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save LLMConfig to a YAML file.

        Args:
            path: Path to write YAML configuration file.
        """
        path = Path(path)
        data = asdict(self)

        # Convert Path to string for YAML serialization
        if isinstance(data.get("cache_dir"), Path):
            data["cache_dir"] = str(data["cache_dir"])

        # Remove None values and empty dicts for cleaner YAML
        data = {k: v for k, v in data.items() if v is not None and v != {}}

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


class LLMClient:
    """Unified LLM client with retries, caching, and structured output support.

    Uses diskcache for efficient, thread-safe response caching.
    """

    def __init__(self, config: LLMConfig):
        """Initialize LLM client.

        Args:
            config: LLMConfig instance. The model field is required.

        Raises:
            ValueError: If config.model is not specified.
        """
        if not config.model:
            raise ValueError("LLMConfig.model is required and cannot be empty")

        self.config = config
        self._cache: diskcache.Cache | None = None

        if self.config.cache_enabled:
            self._init_cache()

    def _init_cache(self) -> None:
        """Initialize diskcache instance."""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(directory=str(cache_dir))

    def _ensure_cache(self) -> diskcache.Cache:
        """Ensure cache is initialized, creating it if needed."""
        if self._cache is None:
            self._init_cache()
        return self._cache  # type: ignore[return-value]

    def _cache_key(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
    ) -> str:
        """Generate a unique cache key for the request.

        Includes sampling parameters so that different configurations
        (temperature, thinking, top_p, seed) produce distinct cache entries.
        """
        schema_name = response_format.__name__ if response_format else "str"
        thinking_config = self.config.get_thinking_config()
        thinking_str = str(thinking_config) if thinking_config else "none"
        content = (
            f"{model}:{system_prompt}:{user_prompt}:{schema_name}"
            f":temp={self.config.temperature}"
            f":top_p={self.config.top_p}"
            f":max_tokens={self.config.max_tokens}"
            f":thinking={thinking_str}"
            f":seed={self.config.seed}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def _backend_key(self, model: str) -> str:
        """Identify the endpoint for health tracking.

        Keyed on api_base when present so two vLLM servers are tracked
        separately; otherwise the provider, which is the unit a hosted API
        fails at.
        """
        return self.config.api_base or normalize_to_provider(model)

    def _get_retry_decorator(self) -> Any:
        """Build tenacity retry decorator from config.

        Waits are randomized. Without jitter, sibling requests rejected at the
        same instant back off by the same amount and collide again on every
        attempt, which is how a single burst used to exhaust all its retries.
        """
        return retry(
            retry=retry_if_exception_type(
                (
                    litellm.RateLimitError,
                    litellm.ServiceUnavailableError,
                    litellm.APIConnectionError,
                    litellm.Timeout,
                )
            ),
            # max_retries counts retries, so the initial attempt is extra.
            stop=stop_after_attempt(self.config.max_retries + 1),
            wait=wait_random_exponential(
                multiplier=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
            reraise=True,
        )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[T] | None = None,
        use_cache: bool | None = None,
        return_thinking: bool = False,
        return_result: bool = False,
        **kwargs: Any,
    ) -> str | T | GenerateResult:
        """Generate LLM response with optional structured output.

        Args:
            system_prompt: System message for the LLM.
            user_prompt: User message for the LLM.
            response_format: Optional Pydantic model class for structured output.
                When provided, LiteLLM uses the model's JSON schema to constrain
                the LLM output and returns a validated Pydantic instance.
            use_cache: Whether to use caching for this request.
                - None (default): Use config.cache_enabled setting
                - True: Force cache usage (initializes cache if needed)
                - False: Skip cache for this request
            return_thinking: If True and thinking is enabled, return a GenerateResult
                with both content and thinking. If False (default), only return content.
                Note: When response_format is provided, thinking is injected into the
                'reasoning' field if it exists, regardless of this setting.
            return_result: If True, always return a GenerateResult with full details
                including usage statistics and completion cost. This is useful when
                you need to track token usage. When True, takes precedence over the
                default return behavior.
            **kwargs: Override any LLMConfig parameters for this call.

        Returns:
            If return_result=True or return_thinking=True: GenerateResult with content,
                thinking, usage, cost, and parsed (if response_format was provided).
            If response_format is None: String response from the LLM.
            If response_format is provided: Validated Pydantic model instance.
                If thinking is enabled and the response_format has a 'reasoning'
                field, it will be populated with the model's thinking trace.

        Raises:
            litellm.APIError: If all retries fail
            pydantic.ValidationError: If response doesn't match schema
        """
        # Determine caching behavior for this request
        should_cache = use_cache if use_cache is not None else self.config.cache_enabled

        # Check cache first
        cache_key: str | None = None
        if should_cache:
            cache = self._ensure_cache()
            cache_key = self._cache_key(
                self.config.model, system_prompt, user_prompt, response_format
            )
            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {cache_key[:8]}...")
                return cached  # type: ignore[return-value]

        # Build request parameters
        model = kwargs.get("model", self.config.model)

        # Check if this is an Anthropic model that supports prompt caching
        # Anthropic requires cache_control on message content; other providers
        # handle caching automatically (OpenAI, Deepseek) or don't support it
        is_anthropic = model.startswith("anthropic/") or model.startswith("claude")
        use_prompt_caching = self.config.prompt_caching and is_anthropic
        if use_prompt_caching:
            # Anthropic requires cache_control on message content
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": user_prompt},
            ]
        else:
            # Standard message format for other providers
            # OpenAI/Deepseek: Caching is automatic for prompts ≥1024 tokens
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "timeout": kwargs.get("timeout", self.config.timeout),
            **self.config.extra_params,
        }

        if self.config.max_tokens:
            params["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)
        if self.config.top_p:
            params["top_p"] = kwargs.get("top_p", self.config.top_p)
        if self.config.api_key:
            params["api_key"] = self.config.api_key
        if self.config.api_base:
            params["api_base"] = self.config.api_base

        # Thinking/Reasoning configuration (unified across providers)
        thinking_config = self.config.get_thinking_config()
        if thinking_config is not None:
            # Determine whether to use reasoning_effort or explicit thinking dict
            # Use explicit budget_tokens when specified for fine-grained control
            # Otherwise use reasoning_effort for better cross-provider compatibility
            if thinking_config.budget_tokens is not None:
                # Explicit token budget - use thinking dict (Anthropic/Gemini style)
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_config.budget_tokens,
                }
            else:
                # Level-based - use reasoning_effort for cross-provider support
                # LiteLLM translates this to provider-specific parameters
                params["reasoning_effort"] = thinking_config.get_reasoning_effort()

        # Extra headers configuration
        extra_headers = dict(self.config.extra_headers)
        if use_prompt_caching:
            # Anthropic prompt caching requires beta header
            extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        if extra_headers:
            params["extra_headers"] = extra_headers

        if self.config.seed is not None:
            params["seed"] = self.config.seed

        # Enable structured output if Pydantic model provided. The schema sent to the
        # provider drops the post-hoc ``reasoning`` slot so strict-mode backends don't force
        # non-OpenAI models (e.g. Llama/Qwen on Groq) to emit it; see
        # _provider_response_format. Parsing below still uses the full Pydantic model.
        if response_format is not None:
            params["response_format"] = _provider_response_format(response_format)

        # Print debug prompt if debug mode is enabled
        import autorubric

        if autorubric.debug:
            _print_debug_prompt(system_prompt, user_prompt, model)

        # Make request with retries
        thinking_content: str | None = None
        raw_response: Any = None

        semaphore = await RateLimitPool.get_instance().get_semaphore(
            model, self.config.max_parallel_requests
        )
        limiter = await RateLimiterPool.get_instance().get_limiter(
            model, self.config.rpm, self.config.tpm
        )
        health = await BackendHealthPool.get_instance().get_health(
            self._backend_key(model),
            self.config.max_consecutive_infra_failures,
            self.config.infra_retry_backoff,
            self.config.infra_retry_max_wait,
        )
        estimated_tokens = (
            _estimate_request_tokens(system_prompt, user_prompt, self.config.max_tokens)
            if self.config.tpm is not None
            else 0
        )

        async def _attempt() -> str:
            nonlocal thinking_content, raw_response
            # Pacing and the concurrency semaphore are acquired per attempt and
            # released before any retry backoff, so a waiting request never
            # holds a slot that a healthy one could use.
            if limiter is not None:
                await limiter.acquire(estimated_tokens)
            if semaphore is not None:
                async with semaphore:
                    response = await litellm.acompletion(**params)
            else:
                response = await litellm.acompletion(**params)
            raw_response = response

            message = response.choices[0].message

            # Extract thinking/reasoning content (standardized across providers)
            # LiteLLM provides unified `reasoning_content` field
            thinking_content = _extract_thinking_content(message)

            return message.content  # type: ignore[return-value]

        _call = self._get_retry_decorator()(_attempt)

        # The breaker counts whole requests, not attempts: a call that exhausts
        # its retries is one infrastructure failure, not max_retries of them.
        if health is not None:
            await health.before_request()
        try:
            response_content = await _call()
        except BaseException as exc:
            if health is not None:
                health.record_failure(exc)
            raise
        if health is not None:
            health.record_success()

        # Extract usage and cost from the raw response
        usage = _extract_usage_from_response(raw_response)
        cost = _calculate_completion_cost(raw_response)

        # Parse structured output if requested
        parsed_response: T | None = None
        if response_format is not None:
            # LiteLLM returns JSON string when response_format is set
            # Parse it into the Pydantic model
            data = json.loads(response_content)

            # Inject thinking content into the reasoning field if available
            if thinking_content and "reasoning" in response_format.model_fields:
                data["reasoning"] = thinking_content

            parsed_response = response_format.model_validate(data)

        # Determine what to return
        result: str | T | GenerateResult
        if return_result or return_thinking:
            # Return full GenerateResult with all details
            result = GenerateResult(
                content=response_content,
                thinking=thinking_content,
                raw_response=raw_response,
                usage=usage,
                cost=cost,
                parsed=parsed_response,
            )
        elif response_format is not None:
            # Return just the parsed Pydantic model
            result = parsed_response  # type: ignore[assignment]
        else:
            # Return just the string content
            result = response_content

        # Cache the response (cache the parsed object for structured outputs)
        if should_cache and cache_key:
            cache = self._ensure_cache()
            cache.set(
                cache_key,
                result,
                expire=self.config.cache_ttl,
            )
            logger.debug(f"Cached response for {cache_key[:8]}...")

        return result

    async def generate_many(
        self,
        requests: Sequence[GenerateRequest],
        max_concurrent: int | None = None,
        **kwargs: Any,
    ) -> list[GenerateOutcome]:
        """Generate for many prompts, isolating per-request failures.

        Gives batch generation the contract grading already has: one bad
        request yields a result carrying an `error` rather than taking its
        siblings down with it. Pacing, retries, and concurrency are handled by
        `generate`, so callers need no asyncio of their own.

        Concurrency is bounded by `max_concurrent` when given; otherwise the
        whole batch is submitted and the rate limiter and
        `config.max_parallel_requests` decide how fast it actually flows.

        Args:
            requests: Prompts to generate for. Order is preserved in the result.
            max_concurrent: Optional ceiling on simultaneously in-flight requests.
            **kwargs: Forwarded to `generate` for every request.

        Returns:
            One `GenerateOutcome` per input, in input order.

        Raises:
            BackendUnavailableError: If the backend circuit breaker trips. This
                is a property of the run, not of any one request, so it
                propagates instead of being recorded per item.
        """
        semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        async def _one(index: int, request: GenerateRequest) -> GenerateOutcome:
            try:
                if semaphore is not None:
                    async with semaphore:
                        content = await self.generate(
                            request.system_prompt, request.user_prompt, **kwargs
                        )
                else:
                    content = await self.generate(
                        request.system_prompt, request.user_prompt, **kwargs
                    )
            except BackendUnavailableError:
                raise
            except Exception as exc:
                logger.warning(
                    f"Request {index} failed [{classify_grading_error(exc)}]: {exc}"
                )
                return GenerateOutcome(index=index, content=None, error=str(exc))
            return GenerateOutcome(index=index, content=content, error=None)

        return list(
            await asyncio.gather(*(_one(i, r) for i, r in enumerate(requests)))
        )

    def clear_cache(self) -> int:
        """Clear all cached responses.

        Returns:
            Number of entries cleared.
        """
        if self._cache is None:
            return 0
        count = len(self._cache)
        self._cache.clear()
        return count

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with 'size', 'count', and 'directory' keys.
        """
        if self._cache is None:
            return {"size": 0, "count": 0, "directory": None}
        return {
            "size": self._cache.volume(),
            "count": len(self._cache),
            "directory": str(self._cache.directory),
        }

    def close(self) -> None:
        """Close the on-disk cache, releasing its file handles.

        diskcache keeps the underlying SQLite database open for the lifetime of
        the cache object. Call this when done with the client so the cache
        directory can be removed (notably on Windows, which refuses to delete
        files that are still open). Safe to call when no cache was initialized.
        """
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# Convenience function for simple usage
async def generate(
    system_prompt: str,
    user_prompt: str,
    model: str,
    response_format: type[T] | None = None,
    **kwargs: Any,
) -> str | T:
    """Simple one-shot generation function.

    For repeated calls, prefer creating an LLMClient instance.

    Args:
        system_prompt: System message for the LLM.
        user_prompt: User message for the LLM.
        model: Model identifier (REQUIRED).
        response_format: Optional Pydantic model for structured output.
        **kwargs: Additional LLMConfig parameters.

    Example:
        # Simple string response
        response = await generate(
            "You are a helpful assistant.",
            "What is 2+2?",
            model="openai/gpt-5.2-mini"
        )

        # Structured output
        from pydantic import BaseModel

        class MathAnswer(BaseModel):
            result: int
            explanation: str

        answer = await generate(
            "You are a math tutor.",
            "What is 2+2?",
            model="openai/gpt-5.2-mini",
            response_format=MathAnswer
        )
        print(answer.result)  # 4
    """
    config = LLMConfig(model=model, **kwargs)
    client = LLMClient(config)
    return await client.generate(system_prompt, user_prompt, response_format=response_format)

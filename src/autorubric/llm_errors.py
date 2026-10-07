"""Error taxonomy shared by the LLM, pacing, and grading layers.

Split out of `llm.py` so `pacing.py` can classify failures without importing
the client module that depends on it.
"""

from __future__ import annotations

from typing import Literal

import openai
from pydantic import ValidationError

ErrorCategory = Literal["infrastructure", "parse", "unknown"]
"""Category of a failure encountered while grading a single criterion.

- infrastructure: API/network failure (timeout, connection, rate limit, server error).
  Not the submission's fault; the judge never produced a usable response.
- parse: the judge responded but its output could not be parsed/validated into the
  expected schema. Also not the submission's fault.
- unknown: an unexpected error that does not fit the above categories.
"""


class BackendUnavailableError(Exception):
    """Raised when a backend has failed persistently enough to abandon the run.

    Distinct from the per-request failures in `ErrorCategory`: those describe one
    criterion that could not be assessed, whereas this says the backend itself is
    gone and continuing would only produce more of them. Grading deliberately
    lets this propagate instead of recording it as CANNOT_ASSESS, so a dead
    server stops the run rather than silently filling results with unusable rows.
    """


def classify_grading_error(exc: BaseException) -> ErrorCategory:
    """Classify an exception raised while grading a criterion.

    Reuses the underlying OpenAI exception taxonomy that LiteLLM builds on: every
    transient API failure (``Timeout``, ``APIConnectionError``, ``RateLimitError``,
    ``ServiceUnavailableError``, ``InternalServerError``, status errors, etc.) subclasses
    ``openai.APIError`` (including ``litellm.APIError`` itself), while parse/validation
    failures (``json.JSONDecodeError`` -> ``ValueError``, ``pydantic.ValidationError``)
    do not.

    Args:
        exc: The exception raised during a judge call.

    Returns:
        ``"infrastructure"`` for API/network errors, ``"parse"`` for JSON/validation
        errors, and ``"unknown"`` for anything else.
    """
    if isinstance(exc, openai.APIError):
        return "infrastructure"
    if isinstance(exc, (ValidationError, ValueError)):
        return "parse"
    return "unknown"

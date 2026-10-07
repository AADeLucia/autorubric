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
  expected schema, including a response with no content at all. Also not the
  submission's fault.
- unknown: an unexpected error that does not fit the above categories.

These drive behaviour, not just reporting: ``criterion_grader`` routes infrastructure
and parse failures to CANNOT_ASSESS (excluded from scoring under SKIP), while
``unknown`` keeps a conservative worst-case verdict that *is* scored. Misfiling a
failure as ``unknown`` therefore corrupts scores rather than merely mislabelling them.
"""


class ResponseParseError(ValueError):
    """The judge responded, but its output could not be parsed into the schema.

    Carries ``response_snapshot``: a JSON-safe record of what the model actually
    returned (content, ``finish_reason``, reasoning trace, usage). Without it a
    parse failure leaves behind only a stringified exception, which is how the
    2026-10-07 contentless-response incident stayed undiagnosed -- the response
    object was alive in a local variable at the moment of failure and was simply
    discarded as the exception unwound.

    Subclassing ``ValueError`` is deliberate and load-bearing: it makes
    `classify_grading_error` route this to ``"parse"`` (and so to CANNOT_ASSESS,
    excluded from scoring) via the branch that already exists, with no change to
    the classifier. Do not reparent this to a bare ``Exception`` -- that would
    silently demote every parse failure to ``"unknown"``, which is *scored*.
    """

    def __init__(self, message: str, *, response_snapshot: dict | None = None) -> None:
        super().__init__(message)
        self.response_snapshot = response_snapshot


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
    failures (``json.JSONDecodeError`` -> ``ValueError``, ``pydantic.ValidationError``,
    ``json.loads(None)`` -> ``TypeError``) do not.

    ``TypeError`` is in the parse branch deliberately. The categories are not merely
    descriptive: ``criterion_grader`` routes ``"parse"`` to CANNOT_ASSESS (excluded
    from scoring under the SKIP strategy) but gives ``"unknown"`` a ``_binary_worst_verdict``
    that is *scored*. On 2026-10-07 a Qwen3.5-27B judge returned responses whose
    ``content`` was ``None``; ``json.loads(None)`` raises ``TypeError``, which fell
    through to ``"unknown"`` and so earned MET on every negative-weight criterion and
    UNMET on every positive-weight one. Those verdicts counted: across one 128-record
    checkpoint, 6 records had scores driven by judge calls that never returned,
    3 of them entirely. A response the judge could not produce is a parse failure,
    not an unknown one — the only thing distinguishing it from malformed JSON is that
    ``json.JSONDecodeError`` happens to subclass ``ValueError`` while a ``None`` body
    raises ``TypeError``.

    Args:
        exc: The exception raised during a judge call.

    Returns:
        ``"infrastructure"`` for API/network errors, ``"parse"`` for JSON/validation
        errors, and ``"unknown"`` for anything else.
    """
    if isinstance(exc, openai.APIError):
        return "infrastructure"
    if isinstance(exc, (ValidationError, ValueError, TypeError)):
        return "parse"
    return "unknown"

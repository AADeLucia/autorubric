"""Tests for persisting what the model actually returned.

A judge call that fails to parse used to leave behind nothing but a stringified
exception. The response object was alive in a local variable at the moment of
failure and was discarded as the exception unwound, which is why the 2026-10-07
contentless-response incident could not be diagnosed from checkpoints at all.

Covers:
- ``_response_snapshot`` field extraction, and its refusal to raise on malformed input.
- ``ResponseParseError`` carrying the snapshot out of a failed parse, for both a
  ``None`` body (the real incident) and a malformed-but-present one.
- ``ResponseParseError`` classifying as ``"parse"`` -- the inheritance contract that
  keeps these routed to CANNOT_ASSESS rather than a scored worst-case verdict.
- The success path carrying a snapshot too, so a clean baseline exists to compare against.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from autorubric import ResponseParseError, classify_grading_error
from autorubric.llm import LLMClient, LLMConfig, _response_snapshot


class _Judgment(BaseModel):
    verdict: str


def _mock_response(
    content: str | None,
    *,
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
    model: str = "hosted_vllm/Qwen3.5-27B",
    prompt_tokens: int = 1200,
    completion_tokens: int = 36000,
) -> MagicMock:
    """A litellm-shaped response. `content=None` reproduces the real incident."""
    message = MagicMock()
    message.content = content
    message.reasoning_content = reasoning_content
    message.thinking_blocks = None
    message.thinking = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    response.id = "chatcmpl-abc123"
    return response


async def _generate(response: MagicMock, **kwargs):
    client = LLMClient(LLMConfig(model="hosted_vllm/Qwen3.5-27B"))
    with patch(
        "autorubric.llm.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ):
        return await client.generate(
            system_prompt="You are a judge.",
            user_prompt="Grade this.",
            **kwargs,
        )


# =============================================================================
# _response_snapshot
# =============================================================================


class TestResponseSnapshot:
    def test_captures_the_diagnostic_fields(self):
        snapshot = _response_snapshot(
            _mock_response(None, finish_reason="length", reasoning_content="thinking..."),
            None,
            "thinking...",
        )

        # finish_reason is the field this whole mechanism exists for: nothing else
        # in autorubric reads it, and it is what separates "ran out of budget" from
        # "stopped cleanly without emitting content".
        assert snapshot["finish_reason"] == "length"
        assert snapshot["content"] is None
        assert snapshot["reasoning_content"] == "thinking..."
        assert snapshot["model"] == "hosted_vllm/Qwen3.5-27B"
        assert snapshot["response_id"] == "chatcmpl-abc123"
        assert snapshot["usage"]["completion_tokens"] == 36000

    def test_distinguishes_none_content_from_empty_string(self):
        """`None` and `""` have different causes and look identical once serialized."""
        assert _response_snapshot(_mock_response(None), None, None)["content_is_none"] is True

        empty = _response_snapshot(_mock_response(""), "", None)
        assert empty["content_is_none"] is False
        assert empty["content"] == ""

    def test_is_json_serializable(self):
        """It is written to every criterion of every checkpoint record."""
        import json

        snapshot = _response_snapshot(_mock_response("{}"), "{}", None)
        assert json.loads(json.dumps(snapshot))["finish_reason"] == "stop"

    @pytest.mark.parametrize(
        "response",
        [None, MagicMock(choices=[], usage=None), "not-a-response-object"],
        ids=["none", "empty-choices", "wrong-type"],
    )
    def test_never_raises_on_malformed_input(self, response):
        """This runs on the error path; raising here would mask the real failure."""
        snapshot = _response_snapshot(response, None, None)
        assert snapshot["content_is_none"] is True
        assert snapshot["finish_reason"] is None


# =============================================================================
# ResponseParseError
# =============================================================================


class TestResponseParseError:
    def test_classifies_as_parse(self):
        """The inheritance contract, stated as a test.

        ``ResponseParseError`` subclasses ``ValueError`` specifically so the existing
        parse branch of ``classify_grading_error`` catches it, routing it to
        CANNOT_ASSESS. Reparenting it to a bare ``Exception`` would silently demote
        every parse failure to ``"unknown"``, which is *scored*.
        """
        assert classify_grading_error(ResponseParseError("boom")) == "parse"
        assert isinstance(ResponseParseError("boom"), ValueError)

    def test_snapshot_defaults_to_none(self):
        assert ResponseParseError("boom").response_snapshot is None


# =============================================================================
# End-to-end through LLMClient.generate
# =============================================================================


@pytest.mark.asyncio
async def test_none_content_raises_with_snapshot():
    """The 2026-10-07 incident, reproduced at the client boundary.

    A truncated reasoning model returns `content=None` with `finish_reason="length"`.
    Before this change the resulting TypeError escaped alone and the evidence was lost.
    """
    response = _mock_response(
        None, finish_reason="length", reasoning_content="let me think about this..."
    )

    with pytest.raises(ResponseParseError) as excinfo:
        await _generate(response, response_format=_Judgment)

    snapshot = excinfo.value.response_snapshot
    assert snapshot is not None
    assert snapshot["finish_reason"] == "length"
    assert snapshot["content_is_none"] is True
    assert snapshot["reasoning_content"] == "let me think about this..."
    assert snapshot["usage"]["completion_tokens"] == 36000

    # The original exception is preserved, so the message still matches the
    # strings already written into existing checkpoints.
    assert isinstance(excinfo.value.__cause__, TypeError)
    assert "NoneType" in str(excinfo.value)


@pytest.mark.asyncio
async def test_malformed_body_raises_with_snapshot():
    """A present-but-unparseable body is captured too, not just a missing one."""
    with pytest.raises(ResponseParseError) as excinfo:
        await _generate(_mock_response("{not json"), response_format=_Judgment)

    snapshot = excinfo.value.response_snapshot
    assert snapshot is not None
    assert snapshot["content"] == "{not json"
    assert snapshot["content_is_none"] is False
    assert snapshot["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_schema_mismatch_raises_with_snapshot():
    """Valid JSON that fails validation is a parse failure and keeps its evidence."""
    with pytest.raises(ResponseParseError) as excinfo:
        await _generate(_mock_response('{"wrong_field": 1}'), response_format=_Judgment)

    assert excinfo.value.response_snapshot is not None
    assert excinfo.value.response_snapshot["content"] == '{"wrong_field": 1}'


@pytest.mark.asyncio
async def test_success_path_carries_a_snapshot():
    """A baseline of what a healthy response looks like, for comparison."""
    result = await _generate(
        _mock_response('{"verdict": "MET"}'), response_format=_Judgment, return_result=True
    )

    assert result.parsed.verdict == "MET"
    assert result.response_snapshot["finish_reason"] == "stop"
    assert result.response_snapshot["content_is_none"] is False


@pytest.mark.asyncio
async def test_unstructured_generation_is_unaffected():
    """No response_format means no parsing, so nothing can fail here."""
    assert await _generate(_mock_response("plain text")) == "plain text"

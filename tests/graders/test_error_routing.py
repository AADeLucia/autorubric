"""Tests for grading-error classification and error routing.

Covers:
- ``classify_grading_error`` taxonomy (infrastructure / parse / unknown).
- Binary criterion error routing in ``CriterionGrader`` (infra/parse -> CANNOT_ASSESS,
  unknown -> conservative worst-case) and the resulting score behavior (errored
  criteria dropped from the denominator under the default SKIP strategy).
- Ensemble aggregation: an ensemble error is set only when every contributing judge
  vote errored; a mixed ensemble keeps a real verdict from the successful judge.
- Multi-choice infra failure -> NA verdict (excluded from scoring).
- Serialization round-trip of the ``error`` field on ensemble reports and judge votes.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from pydantic import BaseModel, ValidationError

from autorubric import (
    BackendUnavailableError,
    Criterion,
    CriterionOption,
    CriterionVerdict,
    ErrorCategory,
    Rubric,
    TokenUsage,
    classify_grading_error,
)
from autorubric.dataset import DataItem
from autorubric.eval import ItemResult
from autorubric.graders import CriterionGrader, JudgeSpec
from autorubric.llm import GenerateResult, LLMConfig
from autorubric.types import (
    CriterionJudgment,
    JudgeVote,
    MultiChoiceJudgeVote,
    MultiChoiceJudgment,
)


@pytest.fixture
def mock_llm_config() -> LLMConfig:
    return LLMConfig(model="test-model")


def _ok_binary_result(verdict: CriterionVerdict, explanation: str = "ok") -> GenerateResult:
    """A successful binary GenerateResult with a parsed CriterionJudgment."""
    return GenerateResult(
        content="{}",
        thinking=None,
        raw_response=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost=0.001,
        parsed=CriterionJudgment(criterion_status=verdict, explanation=explanation),
    )


def _ok_mc_result(selected_option: int, explanation: str = "ok") -> GenerateResult:
    """A successful multi-choice GenerateResult with a parsed MultiChoiceJudgment."""
    return GenerateResult(
        content="{}",
        thinking=None,
        raw_response=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost=0.001,
        parsed=MultiChoiceJudgment(selected_option=selected_option, explanation=explanation),
    )


def _client_raising(exc: BaseException) -> MagicMock:
    client = MagicMock()
    client.generate = AsyncMock(side_effect=exc)
    return client


# =============================================================================
# classify_grading_error unit tests
# =============================================================================


def _json_decode_error() -> ValueError:
    """A real ``json.JSONDecodeError`` (subclasses ``ValueError``)."""
    try:
        json.loads("{")
    except ValueError as e:
        return e
    raise AssertionError("json.loads('{') should have raised")


def _none_content_type_error() -> TypeError:
    """The real exception from parsing a judge response whose ``content`` is ``None``.

    Raised for real on 2026-10-07 by a Qwen3.5-27B judge. Built by actually calling
    ``json.loads(None)`` rather than constructing a ``TypeError`` by hand, so this
    test keeps tracking the stdlib's behaviour rather than an assumption about it.
    """
    try:
        json.loads(None)  # type: ignore[arg-type]
    except TypeError as e:
        return e
    raise AssertionError("json.loads(None) should have raised")


def _validation_error() -> ValidationError:
    """A real pydantic ``ValidationError``."""

    class _M(BaseModel):
        x: int

    try:
        _M.model_validate({"x": "not-an-int"})
    except ValidationError as e:
        return e
    raise AssertionError("expected a ValidationError")


class TestClassifyGradingError:
    """Unit tests for the error taxonomy."""

    @pytest.mark.parametrize(
        "exc, expected",
        [
            (litellm.Timeout("timed out", model="m", llm_provider="p"), "infrastructure"),
            (litellm.RateLimitError("rate limited", model="m", llm_provider="p"), "infrastructure"),
            (_json_decode_error(), "parse"),  # json.JSONDecodeError subclasses ValueError
            (_validation_error(), "parse"),
            (ValueError("bad value"), "parse"),
            # A response with no content at all is still the judge failing to produce
            # parseable output; json.loads(None) raises TypeError, not ValueError.
            (_none_content_type_error(), "parse"),
            (TypeError("not a str, bytes or bytearray"), "parse"),
            (RuntimeError("boom"), "unknown"),
        ],
        ids=[
            "litellm_timeout",
            "litellm_rate_limit",
            "json_decode_error",
            "pydantic_validation_error",
            "value_error",
            "none_content_type_error",
            "type_error",
            "runtime_error",
        ],
    )
    def test_taxonomy(self, exc: BaseException, expected: ErrorCategory):
        assert classify_grading_error(exc) == expected

    def test_none_content_is_not_scored_as_unknown(self):
        """Regression guard for the 2026-10-07 scoring corruption.

        This is not a naming preference. ``criterion_grader`` routes "parse" to
        CANNOT_ASSESS (excluded from scoring) but gives "unknown" a worst-case
        verdict that *is* scored, so classifying a contentless response as unknown
        silently drove real scores: MET on every negative-weight criterion, UNMET
        on every positive-weight one.
        """
        assert classify_grading_error(_none_content_type_error()) != "unknown"


# =============================================================================
# Binary criterion error routing
# =============================================================================


@pytest.mark.asyncio
async def test_binary_infrastructure_failure_cannot_assess_and_no_penalty(mock_llm_config):
    """Infra failure -> CANNOT_ASSESS, flagged as error, and excluded from scoring.

    The errored criterion must drop out of the denominator under the default SKIP
    strategy: the score should equal a baseline where that criterion is simply absent.
    """
    # Two positive criteria. The second one's judge call fails with an infra error.
    rubric = Rubric(
        [
            Criterion(weight=2.0, requirement="Criterion A"),
            Criterion(weight=1.0, requirement="Criterion B"),
        ]
    )

    async def mock_generate(system_prompt, user_prompt, **kwargs) -> Any:
        if "Criterion B" in user_prompt:
            raise litellm.Timeout("timed out", model="m", llm_provider="p")
        return _ok_binary_result(CriterionVerdict.MET)

    client = MagicMock()
    client.generate = AsyncMock(side_effect=mock_generate)

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=client,
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    a_report, b_report = report.report[0], report.report[1]

    # Criterion A succeeded.
    assert a_report.final_verdict == CriterionVerdict.MET
    assert not a_report.is_error
    assert a_report.error is None

    # Criterion B failed with infrastructure error -> CANNOT_ASSESS, flagged.
    assert b_report.final_verdict == CriterionVerdict.CANNOT_ASSESS
    assert b_report.is_error
    assert b_report.error is not None
    assert b_report.error.startswith("infrastructure:")

    # Score is NOT penalized: B is excluded from the denominator (SKIP).
    # Only A remains: MET with weight 2.0 -> 2.0 / 2.0 = 1.0.
    assert report.score == pytest.approx(1.0)

    # Baseline: a rubric where the errored criterion is simply absent.
    baseline_rubric = Rubric([Criterion(weight=2.0, requirement="Criterion A")])

    async def baseline_generate(system_prompt, user_prompt, **kwargs) -> Any:
        return _ok_binary_result(CriterionVerdict.MET)

    baseline_client = MagicMock()
    baseline_client.generate = AsyncMock(side_effect=baseline_generate)
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=baseline_client,
    ):
        baseline_grader = CriterionGrader(llm_config=mock_llm_config)
        baseline_report = await baseline_rubric.grade("submission", grader=baseline_grader)

    assert report.score == pytest.approx(baseline_report.score)


@pytest.mark.asyncio
async def test_binary_parse_failure_cannot_assess(mock_llm_config):
    """Parse failure (pydantic.ValidationError) -> CANNOT_ASSESS with parse error."""

    class _Schema(BaseModel):
        x: int

    try:
        _Schema.model_validate({"x": "nope"})
    except ValidationError as e:
        parse_exc: ValidationError = e

    rubric = Rubric([Criterion(weight=1.0, requirement="Criterion A")])
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_client_raising(parse_exc),
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    cr = report.report[0]
    assert cr.final_verdict == CriterionVerdict.CANNOT_ASSESS
    assert cr.is_error
    assert cr.error is not None
    assert cr.error.startswith("parse:")


@pytest.mark.asyncio
async def test_binary_none_content_failure_cannot_assess_and_does_not_score(mock_llm_config):
    """A judge response with no content must not produce a scored verdict.

    Reproduces the 2026-10-07 Qwen3.5-27B failure end to end. Before TypeError was
    moved into the parse branch, this landed on the worst-case path: the negative-
    weight criterion scored MET and subtracted its full penalty, the positive-weight
    one scored UNMET, and nothing in the record distinguished either from a real
    judgment. Asserting the score against a clean baseline is the point -- the
    verdicts alone would not have caught it.
    """
    rubric = Rubric(
        [
            Criterion(weight=2.0, requirement="Positive criterion"),
            Criterion(weight=-9.0, requirement="Negative criterion"),
        ]
    )
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_client_raising(_none_content_type_error()),
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    for cr in report.report:
        assert cr.final_verdict == CriterionVerdict.CANNOT_ASSESS
        assert cr.is_error
        assert cr.error is not None
        assert cr.error.startswith("parse:")

    # The negative-weight penalty must not have been applied.
    assert report.raw_score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_binary_unknown_failure_keeps_worst_case(mock_llm_config):
    """Unknown failure (RuntimeError) -> conservative worst-case verdict, flagged.

    Positive-weight criterion -> UNMET; negative-weight criterion -> MET.
    """
    rubric = Rubric(
        [
            Criterion(weight=2.0, requirement="Positive criterion"),
            Criterion(weight=-1.0, requirement="Negative criterion"),
        ]
    )
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_client_raising(RuntimeError("boom")),
    ):
        grader = CriterionGrader(llm_config=mock_llm_config)
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    pos, neg = report.report[0], report.report[1]

    assert pos.final_verdict == CriterionVerdict.UNMET
    assert pos.is_error
    assert pos.error is not None
    assert pos.error.startswith("unknown:")

    assert neg.final_verdict == CriterionVerdict.MET
    assert neg.is_error
    assert neg.error is not None
    assert neg.error.startswith("unknown:")


# =============================================================================
# Ensemble aggregation with errors
# =============================================================================


def _two_judge_grader() -> CriterionGrader:
    return CriterionGrader(
        judges=[
            JudgeSpec(LLMConfig(model="judge-a-model"), "judge_a"),
            JudgeSpec(LLMConfig(model="judge-b-model"), "judge_b"),
        ],
        aggregation="majority",
    )


@pytest.mark.asyncio
async def test_ensemble_mixed_one_failure_keeps_successful_verdict():
    """One judge fails (infra), the other returns MET -> final verdict from success.

    The ensemble report's error must be None (is_error False) because at least one
    judge produced a genuine judgment.
    """
    rubric = Rubric([Criterion(weight=1.0, requirement="Criterion A")])

    # Build distinct clients keyed by the judge's model so we can route per-judge.
    client_a = _client_raising(litellm.Timeout("timed out", model="m", llm_provider="p"))
    client_b = MagicMock()
    client_b.generate = AsyncMock(return_value=_ok_binary_result(CriterionVerdict.MET))

    def fake_client(config: LLMConfig) -> MagicMock:
        return client_a if config.model == "judge-a-model" else client_b

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        side_effect=fake_client,
    ):
        grader = _two_judge_grader()
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    cr = report.report[0]

    # Successful judge drives the final verdict.
    assert cr.final_verdict == CriterionVerdict.MET
    # Mixed ensemble: not flagged as an error.
    assert cr.error is None
    assert not cr.is_error

    # One of the two votes carries an infra error, the other is clean.
    vote_errors = [v.error for v in cr.votes]
    assert any(e is not None and e.startswith("infrastructure:") for e in vote_errors)
    assert any(e is None for e in vote_errors)


@pytest.mark.asyncio
async def test_ensemble_all_judges_fail_cannot_assess_and_flagged():
    """Every judge fails (infra) -> CANNOT_ASSESS and ensemble flagged as error."""
    rubric = Rubric([Criterion(weight=1.0, requirement="Criterion A")])

    client_a = _client_raising(litellm.Timeout("a down", model="m", llm_provider="p"))
    client_b = _client_raising(litellm.RateLimitError("b down", model="m", llm_provider="p"))

    def fake_client(config: LLMConfig) -> MagicMock:
        return client_a if config.model == "judge-a-model" else client_b

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        side_effect=fake_client,
    ):
        grader = _two_judge_grader()
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    cr = report.report[0]

    assert cr.final_verdict == CriterionVerdict.CANNOT_ASSESS
    assert cr.is_error
    assert cr.error is not None
    # Both votes errored; the combined message references both.
    assert all(v.error is not None for v in cr.votes)


# =============================================================================
# Multi-choice error routing
# =============================================================================


@pytest.mark.asyncio
async def test_multi_choice_infrastructure_failure_is_na(mock_llm_config):
    """Multi-choice infra failure -> NA verdict (excluded from scoring), flagged."""
    rubric = Rubric(
        [
            Criterion(
                name="quality",
                requirement="How good is it?",
                weight=5.0,
                scale_type="ordinal",
                options=[
                    CriterionOption(label="Bad", value=0.0),
                    CriterionOption(label="Ok", value=0.5),
                    CriterionOption(label="Great", value=1.0),
                ],
            ),
        ]
    )
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_client_raising(litellm.Timeout("timed out", model="m", llm_provider="p")),
    ):
        # Disable shuffling for deterministic behavior.
        grader = CriterionGrader(llm_config=mock_llm_config, shuffle_options=False)
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    cr = report.report[0]
    assert cr.final_multi_choice_verdict is not None
    assert cr.final_multi_choice_verdict.na is True
    assert cr.is_error
    assert cr.error is not None
    assert cr.error.startswith("infrastructure:")

    # NA criterion excluded from scoring under SKIP: only criterion present -> 0.0.
    assert report.score == 0.0


@pytest.mark.asyncio
async def test_multi_choice_unknown_with_na_option_does_not_select_na(mock_llm_config):
    """Unknown error must NOT auto-select an NA option (NA is reserved for infra/parse).

    Positive weight -> worst case is the lowest-value scored (non-NA) option.
    """
    rubric = Rubric(
        [
            Criterion(
                name="quality",
                requirement="How good is it?",
                weight=5.0,
                scale_type="ordinal",
                options=[
                    CriterionOption(label="NA", value=0.0, na=True),
                    CriterionOption(label="Bad", value=0.0),
                    CriterionOption(label="Great", value=1.0),
                ],
            ),
        ]
    )
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_client_raising(RuntimeError("boom")),
    ):
        grader = CriterionGrader(llm_config=mock_llm_config, shuffle_options=False)
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    cr = report.report[0]
    assert cr.final_multi_choice_verdict is not None
    assert cr.final_multi_choice_verdict.na is False
    assert cr.final_multi_choice_verdict.selected_index == 1
    assert report.score == 0.0


@pytest.mark.asyncio
async def test_multi_choice_unknown_positive_weight_picks_lowest_value(mock_llm_config):
    """Unknown error, positive weight -> worst case is the lowest-value option."""
    rubric = Rubric(
        [
            Criterion(
                name="quality",
                requirement="How good is it?",
                weight=5.0,
                scale_type="ordinal",
                options=[
                    CriterionOption(label="Bad", value=0.0),
                    CriterionOption(label="Ok", value=0.5),
                    CriterionOption(label="Great", value=1.0),
                ],
            ),
        ]
    )
    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        return_value=_client_raising(RuntimeError("boom")),
    ):
        grader = CriterionGrader(llm_config=mock_llm_config, shuffle_options=False)
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    cr = report.report[0]
    assert cr.final_multi_choice_verdict is not None
    assert cr.final_multi_choice_verdict.selected_index == 0
    assert cr.final_multi_choice_verdict.value == 0.0
    assert cr.final_multi_choice_verdict.na is False
    assert report.score == 0.0


@pytest.mark.asyncio
async def test_ensemble_multi_choice_vote_carries_error():
    """In a mixed multi-choice ensemble, the failed judge's vote records its error.

    The successful judge yields a genuine vote (error None); the failed judge's vote
    carries a category-prefixed error. Exercises MultiChoiceJudgeVote.error parity.
    """
    rubric = Rubric(
        [
            Criterion(
                name="quality",
                requirement="How good is it?",
                weight=5.0,
                scale_type="ordinal",
                options=[
                    CriterionOption(label="Bad", value=0.0),
                    CriterionOption(label="Ok", value=0.5),
                    CriterionOption(label="Great", value=1.0),
                ],
            ),
        ]
    )

    client_a = _client_raising(litellm.Timeout("a down", model="m", llm_provider="p"))
    client_b = MagicMock()
    client_b.generate = AsyncMock(return_value=_ok_mc_result(2))

    def fake_client(config: LLMConfig) -> MagicMock:
        return client_a if config.model == "judge-a-model" else client_b

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        side_effect=fake_client,
    ):
        grader = CriterionGrader(
            judges=[
                JudgeSpec(LLMConfig(model="judge-a-model"), "judge_a"),
                JudgeSpec(LLMConfig(model="judge-b-model"), "judge_b"),
            ],
            aggregation="majority",
            shuffle_options=False,
        )
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    cr = report.report[0]
    assert cr.multi_choice_votes
    by_id = {v.judge_id: v for v in cr.multi_choice_votes}
    assert by_id["judge_a"].error is not None
    assert by_id["judge_a"].error.startswith("infrastructure:")
    assert by_id["judge_b"].error is None


@pytest.mark.asyncio
async def test_ensemble_multi_choice_all_judges_fail_error_flagged():
    """Every judge fails (infra) on a multi-choice criterion -> ensemble flagged.

    Behavior-lock: the ensemble-level error must combine BOTH judges' failures
    (joined by `` | ``), the ensemble report must be flagged via ``is_error``, and every
    per-judge ``MultiChoiceJudgeVote`` must itself be flagged via ``is_error``. This
    pins the all-judges-fail multi-choice behavior so the single-source ``_aggregate_error``
    refactor is provably behavior-preserving.
    """
    rubric = Rubric(
        [
            Criterion(
                name="quality",
                requirement="How good is it?",
                weight=5.0,
                scale_type="ordinal",
                options=[
                    CriterionOption(label="Bad", value=0.0),
                    CriterionOption(label="Ok", value=0.5),
                    CriterionOption(label="Great", value=1.0),
                ],
            ),
        ]
    )

    client_a = _client_raising(litellm.Timeout("a down", model="m", llm_provider="p"))
    client_b = _client_raising(litellm.RateLimitError("b down", model="m", llm_provider="p"))

    def fake_client(config: LLMConfig) -> MagicMock:
        return client_a if config.model == "judge-a-model" else client_b

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        side_effect=fake_client,
    ):
        grader = CriterionGrader(
            judges=[
                JudgeSpec(LLMConfig(model="judge-a-model"), "judge_a"),
                JudgeSpec(LLMConfig(model="judge-b-model"), "judge_b"),
            ],
            aggregation="majority",
            shuffle_options=False,
        )
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    cr = report.report[0]

    # Ensemble flagged as error (every judge failed).
    assert cr.is_error
    assert cr.error is not None
    # Combined message references BOTH judges' failures, joined by " | ".
    assert " | " in cr.error
    assert cr.error.count("infrastructure:") == 2
    assert "a down" in cr.error
    assert "b down" in cr.error

    # Every per-judge vote is itself flagged (is_error parity).
    assert cr.multi_choice_votes
    assert len(cr.multi_choice_votes) == 2
    assert all(v.is_error for v in cr.multi_choice_votes)
    assert all(v.error is not None for v in cr.multi_choice_votes)


@pytest.mark.asyncio
async def test_ensemble_forced_choice_all_fail_clean_abstain():
    """Forced-choice (auto_na_option=False), no NA option, every judge fails (infra).

    With no NA option to abstain into, each per-judge error-abstain has selected_index=None
    (na=True), and the aggregate must be a GENUINE abstain: na=True with selected_index/label
    None — never na=True pointing at a real scored option. Excluded under SKIP -> 0.0.
    """
    rubric = Rubric(
        [
            Criterion(
                name="quality",
                requirement="How good is it?",
                weight=5.0,
                scale_type="ordinal",
                options=[
                    CriterionOption(label="Bad", value=0.0),
                    CriterionOption(label="Ok", value=0.5),
                    CriterionOption(label="Great", value=1.0),
                ],
            ),
        ]
    )

    client_a = _client_raising(litellm.Timeout("a down", model="m", llm_provider="p"))
    client_b = _client_raising(litellm.RateLimitError("b down", model="m", llm_provider="p"))

    def fake_client(config: LLMConfig) -> MagicMock:
        return client_a if config.model == "judge-a-model" else client_b

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        side_effect=fake_client,
    ):
        grader = CriterionGrader(
            judges=[
                JudgeSpec(LLMConfig(model="judge-a-model"), "judge_a"),
                JudgeSpec(LLMConfig(model="judge-b-model"), "judge_b"),
            ],
            aggregation="majority",
            shuffle_options=False,
            auto_na_option=False,
        )
        report = await rubric.grade("submission", grader=grader)

    assert report.report is not None
    cr = report.report[0]
    mcv = cr.final_multi_choice_verdict
    assert mcv is not None
    assert mcv.na is True
    assert mcv.selected_index is None
    assert mcv.selected_label is None
    assert cr.is_error
    # Every per-judge vote is a clean None-abstain.
    assert cr.multi_choice_votes
    assert all(v.na and v.selected_index is None for v in cr.multi_choice_votes)
    assert report.score == 0.0


# =============================================================================
# Serialization round-trip
# =============================================================================


@pytest.mark.asyncio
async def test_error_survives_serialization_round_trip(mock_llm_config):
    """``error`` round-trips on the ensemble report and on each JudgeVote.

    Build a real errored ensemble report by grading with a failing ensemble, then
    serialize via ItemResult.to_dict and deserialize via ItemResult.from_dict.
    """
    rubric = Rubric([Criterion(weight=1.0, requirement="Criterion A")])

    client_a = _client_raising(litellm.Timeout("a down", model="m", llm_provider="p"))
    client_b = _client_raising(litellm.RateLimitError("b down", model="m", llm_provider="p"))

    def fake_client(config: LLMConfig) -> MagicMock:
        return client_a if config.model == "judge-a-model" else client_b

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        side_effect=fake_client,
    ):
        grader = _two_judge_grader()
        report = await rubric.grade("submission", grader=grader)

    item = DataItem(submission="submission", description="test item")
    item_result = ItemResult(item_idx=0, item=item, report=report, duration_seconds=0.1)

    # Round-trip through the public serialization helpers.
    payload = json.loads(json.dumps(item_result.to_dict()))
    restored = ItemResult.from_dict(payload, item)

    assert restored.report.report is not None
    restored_cr = restored.report.report[0]

    # Ensemble-level error preserved.
    assert restored_cr.error is not None
    assert restored_cr.is_error

    # Per-judge vote errors preserved (both judges failed with infra errors).
    assert len(restored_cr.votes) == 2
    assert all(v.error is not None for v in restored_cr.votes)
    assert all(v.error.startswith("infrastructure:") for v in restored_cr.votes)


@pytest.mark.asyncio
async def test_none_abstain_survives_serialization_round_trip():
    """A genuine None-abstain round-trips: selected_index/label stay None on the
    aggregated verdict and on each multi-choice vote."""
    rubric = Rubric(
        [
            Criterion(
                name="quality",
                requirement="How good is it?",
                weight=5.0,
                scale_type="ordinal",
                options=[
                    CriterionOption(label="Bad", value=0.0),
                    CriterionOption(label="Ok", value=0.5),
                    CriterionOption(label="Great", value=1.0),
                ],
            ),
        ]
    )

    client_a = _client_raising(litellm.Timeout("a down", model="m", llm_provider="p"))
    client_b = _client_raising(litellm.RateLimitError("b down", model="m", llm_provider="p"))

    def fake_client(config: LLMConfig) -> MagicMock:
        return client_a if config.model == "judge-a-model" else client_b

    with patch(
        "autorubric.graders.criterion_grader.LLMClient",
        side_effect=fake_client,
    ):
        grader = CriterionGrader(
            judges=[
                JudgeSpec(LLMConfig(model="judge-a-model"), "judge_a"),
                JudgeSpec(LLMConfig(model="judge-b-model"), "judge_b"),
            ],
            aggregation="majority",
            shuffle_options=False,
            auto_na_option=False,
        )
        report = await rubric.grade("submission", grader=grader)

    item = DataItem(submission="submission", description="test item")
    item_result = ItemResult(item_idx=0, item=item, report=report, duration_seconds=0.1)
    payload = json.loads(json.dumps(item_result.to_dict()))
    restored = ItemResult.from_dict(payload, item)

    assert restored.report.report is not None
    restored_cr = restored.report.report[0]
    assert restored_cr.final_multi_choice_verdict is not None
    assert restored_cr.final_multi_choice_verdict.na is True
    assert restored_cr.final_multi_choice_verdict.selected_index is None
    assert restored_cr.final_multi_choice_verdict.selected_label is None
    assert restored_cr.multi_choice_votes
    assert all(v.selected_index is None for v in restored_cr.multi_choice_votes)


# =============================================================================
# JudgeVote / MultiChoiceJudgeVote is_error property parity
# =============================================================================


def _binary_vote(error: str | None) -> JudgeVote:
    return JudgeVote(
        judge_id="j",
        verdict=CriterionVerdict.UNMET,
        reason="r",
        error=error,
    )


def _multi_choice_vote(error: str | None) -> MultiChoiceJudgeVote:
    return MultiChoiceJudgeVote(
        judge_id="j",
        selected_index=0,
        selected_label="L",
        value=0.0,
        reason="r",
        error=error,
    )


class TestVoteIsErrorProperty:
    """``is_error`` parity on the per-vote dataclasses.

    ``CriterionReport`` / ``EnsembleCriterionReport`` advise using ``is_error`` instead
    of inspecting ``reason``; the per-vote types must expose the same property so the
    advised pattern is possible at vote level. ``is_error`` is the pure getter
    ``error is not None``, identical across both vote types.
    """

    @pytest.mark.parametrize(
        "vote_factory",
        [_binary_vote, _multi_choice_vote],
        ids=["binary_vote", "multi_choice_vote"],
    )
    @pytest.mark.parametrize(
        "error, expected",
        [("infrastructure: x", True), (None, False)],
        ids=["error_set", "error_none"],
    )
    def test_is_error(self, vote_factory, error: str | None, expected: bool):
        assert vote_factory(error).is_error is expected


class TestBackendUnavailableAborts:
    """A dead backend must stop the run instead of manufacturing unusable rows.

    On 2026-08-14 and again on 2026-10-07, a vLLM server died mid-run and grading
    carried on for thousands more items, writing a well-formed CANNOT_ASSESS row
    for every one. EvalConfig.fail_fast cannot catch that: it keys off
    ItemResult.error, which is only set when rubric.grade() itself raises, and the
    grader catches every per-criterion exception so it never does.
    BackendUnavailableError is the single exception deliberately let through.
    """

    @pytest.mark.asyncio
    async def test_binary_path_propagates(self, mock_llm_config):
        rubric = Rubric(
            [
                Criterion(weight=1.0, requirement="Criterion A"),
                Criterion(weight=1.0, requirement="Criterion B"),
            ]
        )
        client = MagicMock()
        client.generate = AsyncMock(side_effect=BackendUnavailableError("server gone"))

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=client,
        ):
            grader = CriterionGrader(llm_config=mock_llm_config)
            with pytest.raises(BackendUnavailableError, match="server gone"):
                await rubric.grade("submission", grader=grader)

    @pytest.mark.asyncio
    async def test_multi_choice_path_propagates(self, mock_llm_config):
        rubric = Rubric(
            [
                Criterion(
                    weight=1.0,
                    requirement="Pick one",
                    options=[
                        CriterionOption(label="good", value=1.0),
                        CriterionOption(label="bad", value=0.0),
                    ],
                )
            ]
        )
        client = MagicMock()
        client.generate = AsyncMock(side_effect=BackendUnavailableError("server gone"))

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=client,
        ):
            grader = CriterionGrader(llm_config=mock_llm_config)
            with pytest.raises(BackendUnavailableError, match="server gone"):
                await rubric.grade("submission", grader=grader)

    @pytest.mark.asyncio
    async def test_ordinary_infra_error_is_still_recorded_not_raised(self, mock_llm_config):
        """The contrast that makes the breaker necessary.

        An ordinary connection error produces a clean report and no exception --
        indistinguishable from a real run at the item level, which is exactly why
        a 400-item outage could report success.
        """
        rubric = Rubric([Criterion(weight=1.0, requirement="Criterion A")])
        client = MagicMock()
        client.generate = AsyncMock(
            side_effect=litellm.APIConnectionError(
                message="cannot connect", llm_provider="p", model="m"
            )
        )

        with patch(
            "autorubric.graders.criterion_grader.LLMClient",
            return_value=client,
        ):
            grader = CriterionGrader(llm_config=mock_llm_config)
            report = await rubric.grade("submission", grader=grader)

        assert report.error is None, "no run-level signal -- fail_fast would never fire"
        assert report.report[0].final_verdict == CriterionVerdict.CANNOT_ASSESS
        assert report.report[0].is_error

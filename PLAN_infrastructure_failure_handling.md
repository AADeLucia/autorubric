# Plan: fail loudly on sustained infrastructure failure + validity-aware resume

Branch: `fix/infrastructure-failure-handling` (cut from `feature/few-shot-reason`)

## Context

On 2026-08-14 an SSH tunnel to a vLLM server died ~3h into a 540-item × 7-arm evaluation sweep.
AutoRubric kept issuing calls for **400+ more items**, wrote a row per item with no usable verdict,
and exited 0 with a manifest reporting `status: "completed"`. Two arms were destroyed
(477/540 and 209/209 items unusable).

Worse than the wasted calls: **resume keys off item presence, not validity**, so re-running the
affected experiment would have skipped all 540 rows and "completed" instantly with 88% missing
data — producing a fully-populated, plausible-looking result file that is mostly garbage.

## What already exists (and works)

This is *not* a missing-taxonomy problem. AutoRubric already classifies and records these failures:

| Mechanism | Location |
|---|---|
| `ErrorCategory = Literal["infrastructure", "parse", "unknown"]`, `classify_grading_error()` | `src/autorubric/llm.py:52-84` |
| Infrastructure errors → `CANNOT_ASSESS` / `na=True`, never a score-affecting verdict | `src/autorubric/graders/criterion_grader.py:695-722` |
| `error="infrastructure: ..."` on the criterion report **and** each vote | `criterion_grader.py:711-721` |
| `is_error` to distinguish synthesized-from-failure vs genuine abstain | `src/autorubric/types.py:640, 710, 885, 971` |
| `_error_category()` prefix parser | `src/autorubric/meta/_display.py:38-43` |
| `EvalConfig.fail_fast` | `src/autorubric/eval.py:291, 906-912` |

The information is all there. **Nothing downstream acts on it.** This plan closes two specific
gaps rather than adding a parallel error system — see `dev-reference/abstention.md` and
`dev-reference/grading-flow.md` for the existing routing this builds on.

## Why `EvalConfig.fail_fast` does not already solve this

Measured against the preserved failed run, not inferred:

| Measurement | Value | Consequence |
|---|---|---|
| Rows with item-level `error` set | **0 of 540** | `fail_fast` tests exactly this field → never fires |
| Rows with `report.error` set | **0 of 540** | no item-level signal at all |
| Rows unusable (null label) | **477** | total failure, still invisible at item level |
| Manifest `status` / `error` | **`"completed"` / `None`** | the run self-reported success |
| `completed_indices` | **540**, incl. all 477 unusable | resume would skip every poisoned row |

**Gap 1.** `ItemResult.error` is set in exactly one place — `_grade_item` (`eval.py:1136-1146`),
when `rubric.grade()` raises. But `criterion_grader.py:695-722` catches **every** per-criterion
exception and returns a well-formed `CriterionResult`, so `grade()` returns normally and `error`
stays `None`. Setting `fail_fast=True` would have changed nothing.

**Gap 2.** Resume is governed by `completed_indices`, a separate path.
`_update_manifest_indices()` (`eval.py:904, 1033`) records every result unconditionally, and
`_setup_experiment()` (`eval.py:966-1023`) restores the set verbatim. Even a `fail_fast` that
*did* fire would leave already-written rows marked complete.

`fail_fast` does correctly cover a different failure — an exception escaping `grade()` entirely
(bad rubric, dataset error). That path is left untouched.

## Change 1 — stop the run on sustained infrastructure failure

Behaviour: **pause → retry → abort if unrecovered.** A brief network blip must not kill a
multi-hour run, but 400 doomed calls must never happen again.

New `EvalConfig` fields (near `eval.py:291`), all defaulting to current behaviour:

- `max_consecutive_infra_failures: int = 0` — 0 disables. ~10 in practice.
- `infra_retry_backoff: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0)`
- `infra_retry_max_wait: float = 900.0` — bounded total wait before giving up.

Implementation in the streaming loop (`eval.py:900-912`), the one place that sees every result:

1. Detect infrastructure failure from the **criterion report**, not `result.error`. Reuse the
   existing `is_error` properties and the `"infrastructure:"` prefix via the same convention
   `_error_category()` parses — do not re-implement classification.
2. Track a consecutive counter; reset on any item yielding a genuine verdict.
3. On threshold: stop consuming items and poll the endpoint on the backoff schedule. Recovered →
   reset and continue. Not recovered within `infra_retry_max_wait` →
   `_update_manifest_status("failed", ...)` then raise, reusing the existing `fail_fast` abort path
   so both failure modes exit identically.
4. Log pause/resume at WARNING so it is visible in run logs.

Deliberately **not** a per-request circuit breaker in `LLMClient`: the retry/rate-limit layer there
is already nontrivial (`rate_limit.py`), and the eval loop is the right place to reason about
"the run as a whole is failing."

## Change 2 — resume from valid completions only

New `EvalConfig` field: `resume_requires_valid: bool = False` (default preserves behaviour).

- In `_setup_experiment()` (`eval.py:966-1023`), when enabled, filter `completed_indices` to items
  whose persisted report holds a usable verdict, dropping those whose criterion reports are all
  `is_error`. Those indices return to the pending set and are re-graded.
- Log the count re-queued, so a resume that redoes 477 items says so.
- Manifest writing stays unconditional; validity is judged **on read**. Cheaper, backward
  compatible, and it repairs already-damaged experiments with no schema change.

## Files

| File | Change |
|---|---|
| `src/autorubric/eval.py` | `EvalConfig` fields; infra detection + pause/retry/abort in the streaming loop; validity filter in `_setup_experiment` |
| `src/autorubric/llm.py` | none expected — reuse `classify_grading_error` / `ErrorCategory` |
| `tests/` | new tests (below), matching existing layout |
| `dev-reference/` | update `abstention.md` (failure routing now has a run-level consequence) and `conventions.md`; add the new `EvalConfig` fields to `types.md`, and refresh the matching TLDRs in `index.md` per its own update rules |

## Verification

Unit tests (no network, no GPU):

- Detection fires on a report whose criterion errors are `"infrastructure: ..."` while
  `ItemResult.error is None` — the exact shape that defeated `fail_fast`.
- **Regression test pinning the gap:** same fixture with `fail_fast=True` and the new option
  disabled must *not* abort, documenting in code why the built-in flag was insufficient.
- Counter resets on a genuine verdict (intermittent failures don't trip the breaker).
- Abort after bounded wait; recovery path resumes cleanly (monkeypatched sleep — no real waiting).
- `resume_requires_valid=True` re-queues error-only items; `False` preserves today's behaviour.
- Default config reproduces current behaviour exactly on a clean run.

End-to-end, against the preserved real failure (the decisive check): resume the quarantined
experiment with `resume_requires_valid=True` and confirm it re-queues **477** items, not 0.

## Git workflow

Ordered deliberately: bring `feature/few-shot-reason` fully up to date **before** cutting the fix
branch, so it starts from a current base and any upstream conflicts in `eval.py` /
`criterion_grader.py` are resolved once, beforehand.

**Step 0 (prerequisite).** No `upstream` remote is configured — only `origin` (the fork). Every
"update main" therefore pulls the fork's own copy and silently no-ops, which is why local checks
showed "up to date" while GitHub reported updates available.

```bash
git remote add upstream https://github.com/delip/autorubric.git
git fetch upstream
git log --oneline main..upstream/main
```

1. Commit and push `feature/few-shot-reason`.
2. `git checkout main && git merge --ff-only upstream/main && git push origin main`.
   `--ff-only` is deliberate: it succeeds only if the fork's main carries no commits of its own.
   If it refuses, main has diverged — inspect `git log --oneline upstream/main..main` and choose a
   real merge rather than forcing it.
3. `git checkout feature/few-shot-reason && git merge main` — resolve any conflicts here.
4. Commit and push `feature/few-shot-reason`.
5. `git checkout -b fix/infrastructure-failure-handling`.
6. Two commits, one per change above.
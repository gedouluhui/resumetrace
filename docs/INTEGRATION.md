# Consolidation record: ResumeTrace, ResumeLedger, ResumeWitness

Status: 2026-11-19. The owner chose a single public entry point for this topic —
[ResumeTrace](https://github.com/gedouluhui/resumetrace). This document is the record of
what that decision produced: what was adopted, what was not, and why.

## Why one entry point

Three independent implementations asked the same question — does training state really
survive a cold process restart? — from the same angle: run the user's own save/load
callbacks in a fresh interpreter and compare observed state. Publishing three
near-identical tools would split maintenance, split review attention, and force users to
choose between indistinguishable products.

| Project | Where it lives now | State |
|---|---|---|
| ResumeTrace | public: https://github.com/gedouluhui/resumetrace | released (0.2.1) |
| ResumeLedger | `open-source-projects/resumeledger` (local only) | frozen candidate, not published |
| ResumeWitness | `open-source-projects/resumewitness` (local only) | frozen candidate, not published |

Neither candidate is deleted. Both stay runnable, with their own tests and MIT licence,
so a future maintainer can re-evaluate them without the original sessions.

## What was adopted

1. **Runtime RNG format validation (from the 0.2.0a1 review, closed in 0.2.1).** The
   review found that a structurally self-consistent but invalid RNG observation — a
   float32, empty Torch state — was compared and could pass. ResumeTrace now validates
   each default RNG's internal structure, dtype and shape against the running
   interpreter's own format before any comparison. This is the one finding that directly
   contradicted the product's promise ("malformed evidence is rejected"), so it was
   fixed before the consolidation release.

2. **Live parameter binding (already in 0.2.0a1).** ResumeLedger's genuinely new
   contribution was observing the model/optimizer by real parameter names rather than by
   serialized `state_dict` ordering. ResumeTrace implemented that capability in its own
   `TorchSession` instead of copying ResumeLedger's source, CLI or schema.

## What was not adopted

| Candidate contribution | Why not |
|---|---|
| ResumeWitness's typed state wire format with byte budgets | ResumeTrace hashes typed values instead of serializing them; importing a second wire protocol would create two budgets and two failure taxonomies for one observation |
| ResumeWitness's `DIVERGED` / `PRECONDITION_FAILED` / `INCOMPLETE` states | Would change public result states; the mapping onto `PASS`/`DRIFT`/`INVALID` has not been designed or reviewed |
| ResumeWitness's second process manager and cancellation guards | ResumeTrace already owns worker lifecycle and PID identity, including the Windows venv launcher case; two managers would race |
| ResumeLedger's `INCONCLUSIVE` verdict for save-side effects | Behaviourally covered by `observer_state_changed` / `save_state_changed`, which are already reported with the changed paths |
| Either candidate's CLI and JSON schema | Two CLIs for one job; schema compatibility was never agreed |

The regression cases that are specific to those rejected designs (MaskedArray
missingness, leading-dimension rejection, zero-extent shapes) do not apply to
ResumeTrace's fingerprint path: it never reconstructs arrays from a wire format, so those
failure modes cannot occur there. They remain documented in the candidates' own suites.

## How to verify this record

- `python -m unittest discover -s tests -v` in this repository: 101 tests, 1 skipped
  (Windows symlink permission).
- `test_parent_rejects_structurally_invalid_rng_from_real_worker` in
  `tests/test_global_rng.py` reproduces the review's counterexample from a captured real
  worker result.
- The candidates' own instructions are in their `README.md` and
  `docs/INTEGRATION_CANDIDATE.md`; neither has a remote.

No part of this record claims an independent whole-project review. That gap is unchanged
and is still stated in `docs/VALIDATION.md`.

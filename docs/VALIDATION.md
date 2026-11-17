# Validation record — experimental v0.1

Recorded 2026-11-18. These are actual local execution results, not a correctness proof, external user study, model-quality evaluation or performance claim.

## Local execution

Environment: Windows, Python 3.12.3, PyTorch 2.12.0+cpu, NumPy 2.4.6.

- `python -m unittest discover -s tests -v`: 63 tests, **62 passed and one skipped**. The skip is actual symlink creation denied by Windows; no permissions were changed to force it. Do not call this 63 passing tests.
- `PYTHONPATH=src python scripts/verify_examples.py --out <new-directory>`: eight recorded cases accepted. Complete checkpoints passed at three configurations; missing optimizer/RNG/cursor/scheduler and process-local-cache cases produced DRIFT with valid controls. The script binds its own SHA-256 and runtime source-file hashes to the output.
- Installation check: wheel installed into a newly created venv, invoked outside the source directory with isolated import mode. Normal demo returned 0/PASS; process-cache returned 1/DRIFT. PyTorch and NumPy were reused from the existing environment, so this is **not a clean dependency installation claim**.
- Public [example manifest](example-manifest.json) binds the two published recorded reports to their JSON/HTML bytes and runtime source identities. The examples use synthetic data and a tiny original model.

The configured CI matrix runs fresh package installation and tests on Linux Python 3.10/3.12 and Windows Python 3.12. Consult the repository's actual checks for the current outcome; local results do not substitute for cloud results or every dependency-version combination.

## Checks that distinguish real failures

- The module-cache example has a same-process passing control and a cold-process mismatch. No cached input batch list is fed to both runs.
- Each omitted global RNG is detected even when every reported loss is identical. Complete RNG restoration passes. All three RNG domains contribute to the same observation budgets.
- A scheduler checkpoint can have identical continuation losses at an aligned boundary but different state; the paired unaligned-boundary example also changes losses.
- Independent `struct`-encoded expected bytes verify float32/64/16, bfloat16, a lazy complex conjugate, and uint64 tensor hashes. Exact signed zero is distinguished.
- A disposable source-copy mutation changes the comparison predicate to always-equal. The named equal-norm/different-value test then fails its actual assertion; an import error is not accepted as detection. The live source is unchanged.
- Worker failure, incomplete protocol output, nonfinite/unsupported state, budgets, observation RNG consumption and timeout are exercised separately from valid checkpoint drift.

## Bugs found during verification

1. Initial worker observations checked observer RNG purity but did not include global RNG in state evidence. A real cold-process counterexample incorrectly passed. The wrapper now includes Python, NumPy and CPU PyTorch RNG evidence, and the parent rejects missing domains.
2. Windows virtual-environment redirectors produced a runtime PID different from the Popen PID. Actual installed-package execution failed. A direct base-interpreter launch with CPython's child-only venv context preserves both the package environment and strict PID identity. A fresh venv regression now passes.
3. Constant large losses caused division by zero in the SVG chart; opposite finite extremes overflowed. Normalization now happens before subtraction, with finite-coordinate regression cases.
4. Matching non-loss metrics were absent from HTML. Every comparison now has inspectable metrics/evidence. Empty-checkpoint INVALID results and hashed filename identifiers also render correctly.
5. Initial sdist omitted fixtures, integration examples and the Chinese guide. The manifest now includes the reproducible source files, while excluding local planning/coordination artifacts.

## Browser and file boundary

Actual installed-package reports were opened in Chromium at 1200 px and an emulated 390 px. No horizontal page overflow was observed; reports contained zero scripts and zero external resource requests. The mobile SVG labels were measured and increased from about 5.8 effective px to about 11.6 px. This is limited browser review, not a complete assistive-technology audit or a claim about every browser.

HTML user strings are escaped. Output overwrite, same-destination refusal, strict finite JSON and atomic per-file publication have executable tests. JSON/HTML publication is not a transaction across both files.

## Review limitations

The demo/report/CLI received a separate scoped review. Its two must-fix findings were reproduced, repaired and separately re-reviewed, along with schema and input-boundary fixes. Core implementation was inspected by the main development session, which independently reproduced the RNG and installed-venv failures.

A new separate whole-project final reviewer could not start because the review service reported an exhausted daily quota. **There is no independent final whole-project approval.** Subsequent main-session inspection and passing tests are not presented as a substitute for that missing independent review. The first release remains explicitly experimental.

No external users, GPU/distributed/AMP training, arbitrary loader workers, cross-platform bitwise equivalence, non-RNG observer purity, or arbitrary save-callback side effects have been certified. The generic interface requires users to expose relevant state; automatic model/optimizer-binding adapters remain a future, separately reviewed extension.

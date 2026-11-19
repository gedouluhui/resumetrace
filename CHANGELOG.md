# Changelog

## 0.2.1 — 2026-11-19

Fixes the contract hole reported by the independent read-only review of 0.2.0a1: an
observation whose RNG fingerprint was structurally self-consistent but described a state
the runtime never produces was compared, and could therefore pass.

### Fixed

- The parent now validates the internal structure, dtype and shape of all three default
  RNG states against the format the running interpreter produces
  (`src/resumetrace/rng_format.py`): the Python state tuple's length and integer leaves,
  the Gaussian cache type, the NumPy key tensor's dtype and shape, and the Torch state
  tensor's dtype and shape.
- The reviewer's exact counterexample — a float32, empty Torch state — is now rejected
  as `invalid_worker_result` instead of being compared. Truncated Python state tuples, a
  wrong NumPy dtype and a non-numeric Gaussian cache are covered by the same guard.
- New regression test
  `test_parent_rejects_structurally_invalid_rng_from_real_worker` builds each
  malformed case from a captured real worker result; it is not a hand-written fixture.

### Verification

- `Ran 101 tests in 1060.076s` — `OK (skipped=1)`. The skip is a Windows symlink
  permission, not a pass.
- No change to the public result states, the CLI, the report schema or the demo.

### Limits

- Still a pre-release: no independent whole-project final approval, no GPU/DDP/AMP
  support, no external user report.

## 0.2.0a1 — 2026-11-19

- Opt-in `TorchSession` observation of live model and optimizer parameter bindings, real
  parameter-name mapping, native `torch.optim.SGD`/`AdamW`, gradient policy, observer/save
  side-effect guards, alias and overlapping-storage rejection. See the 0.2.0a1 section of
  `docs/VALIDATION.md`.

## 0.1.0 — 2026-11-18

- First public experimental release: four real cold Python processes, full save/exit/
  reconstruct/resume comparison, Python/NumPy/Torch global RNG evidence, JSON and
  script-free HTML reports.

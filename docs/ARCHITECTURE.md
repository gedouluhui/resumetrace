# Architecture and evidence contract

ResumeTrace checks a finite experiment, not the universal correctness of a checkpoint. All training callbacks are trusted code. There is no model download, external API, batch replay cache or security sandbox.

## Execution

`audit_resume('module:factory', factory_config={...})` starts four sequential Python interpreter lifecycles. The factory accepts one JSON dict and returns `step`, `observe`, `save`, `load` methods.

1. **Reference:** run all N steps from the initial seed.
2. **Control:** repeat all N steps in a different interpreter. If it differs, return INVALID before saving anything.
3. **Prefix:** repeat K steps in another interpreter; write a checkpoint and exit. Its observed prefix must match the reference.
4. **Resume:** after prefix exit succeeds, construct under a different startup seed, invoke only the user's load callback, then execute N−K steps.

There are 3N training steps in a complete audit. Observation and process startup are extra overhead; this is not a performance measurement. Every process re-enters the user's input pipeline. No training objects, tensors, generators or batch list are transported between prefix and resume.

The parent invokes its own interpreter version, preserves its working directory/import environment, and never changes its own RNGs or deterministic/thread settings. Workers enable deterministic PyTorch algorithms and set intra-op threads to one. A user callback can still perform arbitrary side effects; use an isolated development experiment.

### Windows virtual environments

On Windows, a venv `python.exe` may be a redirector rather than the final Python process. Its Popen PID differs from the runtime PID. ResumeTrace starts the current base interpreter directly while setting CPython's `__PYVENV_LAUNCHER__` **only in the child environment** to preserve the venv prefix and package resolution. It keeps the strict worker-PID check rather than accepting mismatched identities. The owned process handle then names the actual worker for timeouts.

This mechanism is present in [CPython's v3.12.3 launcher](https://github.com/python/cpython/blob/v3.12.3/PC/launcher.c#L1956-L1959) and is covered by an actual Windows venv regression. It is not an alternate-interpreter feature, and does not claim containment of user-created descendants.

## Observation

A user observation is wrapped without reserving any user keys:

```text
session
  <the dictionary returned by observe()>
global_rng
  python
  numpy
  torch
```

All three global RNG states are captured before `observe()`. If the callback changes them, the run is INVALID. The same pre-observation states are included in the fingerprint; merely checking observation purity would otherwise miss a forgotten RNG restore when the short continuation does not consume randomness.

Relevant model buffers, optimizer group order/state, scheduler, training flags, private generators, sampler cursor and other Python fields must be exposed by the user's observation. Unexposed state is not implicitly checked. Non-RNG observer side effects and save side effects are not automatically certified by this generic callback interface.

### Optional native PyTorch inspection

`TorchSession` is an opt-in wrapper, not a change to the generic protocol. It reads native model registrations (including nonpersistent buffers), module modes, requires_grad and native SGD/AdamW state tied to actual parameter names. It does not invoke model state_dict hooks or use saved optimizer IDs as identity. It rejects model/optimizer/gradient storage aliases and internally overlapping or uncertifiable strided layouts instead of pretending logical values describe alias behavior.

Before a user observer, native state including gradients and all default RNGs is fingerprinted; after the observer it is fingerprinted again. This catches saturating mutations (e.g. clearing all weights) that two post-observation snapshots would miss. Saving is additionally checked with complete before/after observations, user-declared state and RNG. A save-purity violation invalidates this opt-in experiment even when a correctly written checkpoint could independently resume. Neither mode certifies arbitrary external or unregistered side effects.

Gradient policy is explicit (`ignore`, `capture`, `require_none`). Default ignore keeps existing whole-step loops valid without clearing their gradients. Guards always inspect gradients. Ignored gradients are not certified across resume; capture requires checkpoint callbacks to restore them if their presence matters.

Wrapped worker envelopes include an optional strictly validated `inspection` declaration. A different declaration across roles is INVALID. Safe structured failure details contain only check, step, capped typed paths and total count. The original schema-1 generic envelope and report statuses remain valid. Report coverage adds inspection metadata only for wrapped sessions. The same audit byte/leaf cap is applied to wrapper captures, but repeated hashing has additional time/memory cost and is not a process-RSS cap.

## Exact typed fingerprints

CPU dense strided tensors are validated, then hashed over their logical contiguous bytes. Shape, dtype and byte count are part of the evidence. Equal norms are not equal tensors. Positive and negative zero are distinguished. Strides, alias relationships and allocator identity are outside this value contract.

Supported tensor types are bool, standard signed/unsigned integers present in the installed PyTorch, float16/bfloat16/float32/float64 and complex64/complex128. Sparse, quantized, nested, other-device and nonfinite tensor state is rejected. Supported containers are plain dict/list/tuple, with exact string/integer dict keys; scalar values are None, bool, finite float, integer or string.

Paths contain typed, escaped segments: `s:` for string keys, `i:` for integer keys, `l:`/`t:` for sequence positions. This keeps `1`, `'1'`, nested keys, slash-containing keys and empty containers distinct. Public comparison paths distinguish `$/state/s:session/...`, `$/state/s:global_rng/...`, and `$/metrics/...`.

The graph is checked for depth, cycles, leaves and aggregate logical tensor bytes before contiguous materialization. Repeated references count toward the budget; their alias identity is not compared. These bounds are **not a process-RSS limit** and do not prevent arbitrary memory allocation inside user callbacks.

## Process and checkpoint evidence

Workers communicate through tool-owned JSON files with strict version, run token, role, PID, indexed observations and error fields. The parent rejects malformed, nonfinite, oversized, incomplete, duplicate-key or unexpected-type output. Only a complete validated horizon can pass. Default result cap is 64 MiB per worker; configured leaf/byte budgets also apply to global RNG evidence.

At least one nonempty regular checkpoint file is required. Directory traversal is bounded, symlinks/reparse points/special files are rejected, and filenames and contents are hashed. There is no arbitrary archive extraction or untrusted-pickle reader. The demo uses `torch.load(weights_only=True)` only on the checkpoint it just created.

Raw weights, RNG payloads, factory configuration, checkpoint names and callback exception text are not emitted. State names and digests may still be sensitive. Hashes are not anonymization or source authentication.

## Verdicts

- **PASS:** independent control, prefix, restored observations and the requested continuation all match exactly.
- **DRIFT:** the experiment is valid, but a state or numeric metric differs. First differing paths are evidence, not a uniquely established root cause.
- **INVALID:** control instability, exception, process failure, timeout, unsupported observation, missing checkpoint, incomplete output, budget or cleanup failure prevents a valid verdict.

CLI exits are 0, 1, 2 respectively. API argument mistakes raise before spawning. Difference details are capped at 50 per comparison, but the total mismatch count is preserved.

## Modules

| Module | Responsibility |
|---|---|
| `engine.py` | Argument validation, owned child lifecycles, strict result validation, checkpoint hashes and verdict |
| `_worker.py` | One seeded training lifecycle, observation purity and bounded fingerprint output |
| `fingerprint.py` | Typed exact fingerprints and bounded-detail comparison |
| `demo.py` | Actual tiny training job and deliberate checkpoint faults |
| `report.py` | Escaped offline HTML and atomic report-file publication |
| `cli.py` | Explicit trusted-factory/demo commands and exit-code boundary |

The JSON report is the machine-readable evidence. HTML is a script-free rendering. Output files require `--force` for replacement; each is atomic individually, not a multi-file transaction.

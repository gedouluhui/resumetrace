# ResumeTrace

[![Continuation tests](https://github.com/gedouluhui/resumetrace/actions/workflows/ci.yml/badge.svg)](https://github.com/gedouluhui/resumetrace/actions/workflows/ci.yml)

**Does your checkpoint survive a real process exit?**

ResumeTrace runs your training code in fresh Python interpreters and compares uninterrupted training with save → exit → reconstruct → resume. It re-enters your data pipeline instead of feeding both runs a cached list of batches.

CPU-only, local, no model downloads or API keys. Version **0.2.1** fixes a real contract hole found in review: malformed RNG evidence (for example a float32 empty Torch state) is now rejected instead of compared, and each RNG's internal structure, dtype and shape must match what the running interpreter produces. The release is still a pre-release: there is no independent whole-project approval. Exact equality in a finite observed domain is not a universal reproducibility guarantee.

[Open the recorded demo](https://gedouluhui.github.io/resumetrace/) · [中文快速开始](README.zh-CN.md) · [Integration example](examples/custom_training.py) · [Architecture](docs/ARCHITECTURE.md) · [Validation and review limits](docs/VALIDATION.md)

## The bug a fresh object can hide

Imagine a checkpoint callback that saves the model to disk but keeps Adam's state in a module-global dictionary. Reconstructing an object **inside the same interpreter** can appear to work. Exiting that interpreter loses the dictionary.

The included `process-cache` demonstration does exactly that with a real MLP, dropout, AdamW, scheduler and shuffled data cursor. It has a same-process positive control; the cold-process test is supposed to expose the missing optimizer state.

Another subtle case: a lost scheduler epoch can leave the next few losses unchanged when the split aligns with its period. ResumeTrace compares declared state as well as losses; matching loss alone is not enough.

## Try it from source

Python 3.10 or later. Install in your own virtual environment from the repository root:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install .

# Expected: PASS and two local evidence files.
python -m resumetrace demo --fault none --json normal.json --html normal.html

# Expected: DRIFT (exit code 1), not a tool crash.
python -m resumetrace demo --fault process-cache --json drift.json --html drift.html
```

Open `drift.html` in your browser. The report is one offline file: process sequence, control validity, first differing paths, loss trace, checkpoint hashes and coverage limitations. No scripts, remote assets or telemetry.

Other demonstration faults: `optimizer`, `rng`, `cursor`, `scheduler`. Choose `--steps` and `--split`; default is eight steps, checkpoint after four. Existing reports are preserved unless you explicitly pass `--force`.

This project is not claimed to be published on PyPI. Install the checked-out source or a release wheel, not a guessed registry package.

## Use your actual training code

Provide an importable `module:function` factory that accepts a JSON dictionary and returns an object with four methods:

| Method | Your responsibility |
|---|---|
| `step()` | Execute one actual optimizer step, including reading the next batch. Return a nonempty dict of finite numeric metrics. |
| `observe()` | Return the relevant state as CPU tensors, plain dictionaries/lists/tuples and finite scalar values. Must be read-only and must not consume RNG. |
| `save(directory)` | Write your real checkpoint into the supplied directory. |
| `load(directory)` | Restore it into a fresh training object. ResumeTrace does not fill in missing state. |

The complete [regression example](examples/custom_training.py) has no downloaded data. Run it from this repository:

```bash
python -m resumetrace audit examples.custom_training:make_session \
  --steps 6 --split 3 --json custom.json --html custom.html
```

For a test in your own project:

```python
from resumetrace import audit_resume

result = audit_resume(
    'your_project.training:make_session',
    factory_config={'experiment': 'tiny'},
    steps=12,
    split=5,
    timeout=120,
)
assert result['status'] == 'PASS', result['first_divergence'] or result['error']
```

The module must be importable by a fresh interpreter launched with the same Python executable, working directory and normal import environment. A notebook-local function, closure or object instance is not an importable factory. Your callbacks are trusted code: this tool is **not** an execution sandbox.

Expose model parameters and buffers, optimizer state and parameter-group order, scheduler, training mode, private generators, sampler cursor and other relevant Python state. The harness also compares the default CPU RNGs. It cannot infer state hidden inside your application or validate data that changes outside the test.

## Optional: inspect the objects, not just checkpoint IDs

An optimizer checkpoint can contain the **same integer IDs and state values** while attaching them to different model parameters after loading. Comparing `optimizer.state_dict()` alone does not establish the original parameter binding. `TorchSession` instead observes the live optimizer's parameter objects and maps them to registered model names.

Opt in with one factory return line; your update and checkpoint code stay yours:

```python
from resumetrace import TorchSession

def make_session(config):
    session = YourExistingTrainingSession(config)
    return TorchSession(session, gradient_policy="ignore")
```

`YourExistingTrainingSession` is your own class, not a provided package object. It needs ordinary `.model` and `.optimizer` instance attributes, plus `step`, `save`, `load`; an optional `observe()` supplies scheduler/data/private-generator state. The wrapper never clears gradients, restores missing state, changes learning rates or calls custom model serializers.

Run the complete [guarded SGD example](examples/guarded_training.py) from this checkout:

```bash
python -m resumetrace audit examples.guarded_training:make_session --steps 3 --split 1 --json guarded.json --html guarded.html
python -m resumetrace audit examples.guarded_training:make_session --config '{"fault":"observer"}' --steps 3 --split 1 --json observer.json --html observer.html
python -m resumetrace audit examples.guarded_training:make_session --config '{"fault":"save"}' --steps 3 --split 1 --json save.json --html save.html
```

The first run should PASS. The second really zeroes weights from inside the observer; the third really changes weights **after writing a valid checkpoint**. Both should be INVALID with a named check, step and bounded differing paths. `save_state_changed` means the opt-in save-purity contract was violated, **not that the written checkpoint is necessarily unusable**. Existing unwrapped sessions are not subjected to this new purity contract.

| Automatically observed by the wrapper | Still your responsibility |
|---|---|
| Registered parameters and buffers, including nonpersistent buffers | Expose scheduler, sampler/data cursor, private RNG and application counters |
| Actual optimizer parameter-name binding, state and group settings | Use an importable factory and trusted callbacks |
| Per-module train/eval and parameter `requires_grad` | Avoid concurrent mutation and external side effects |
| Before/after callback changes to native state and default RNGs | Review coverage; hidden/unregistered state is not inferred |

Gradient policies are deliberate: **ignore** (default) excludes gradients from restart comparison, accommodating loops that zero at the beginning of the next step; **capture** compares their exact state; **require_none** requires them to be cleared already. Callback guards inspect gradients in every policy. Nothing is silently cleared for you.

This adapter supports CPU eager models and **native SGD/AdamW**. Unsupported state, aliased/overlapping storage (including optimizer and gradient tensors), internally overlapping or uncertifiable strided layouts, tensor subclasses, custom optimizer classes, and exceeded capture budgets return INVALID. Shared/tied model registrations are not supported yet. A noncontiguous tensor with a separable, nonoverlapping stride layout can be inspected; this is not an arbitrary alias-analysis engine.

Each guard fingerprints full observed state multiple times, bounded by the configured byte/leaf budgets; save checks include user-declared data too. This adds time and temporary allocations. Budgets are not process-RSS caps, and the checks do not certify filesystem/network effects or hidden globals. Reports show the inspection mode and gradient policy; different declarations across workers invalidate the experiment. No checks or state are added to generic sessions behind their backs.

## Four runs, one validity gate

```text
A  seed ── construct ── steps 1 … N                      reference
B  seed ── construct ── steps 1 … N                      control
                       │ must match A before continuing
C  seed ── construct ── steps 1 … split ── save ── EXIT
                                                    │ disk only
D  new seed ── construct ── load ── remaining steps    cold resume
```

Every row uses a new Python interpreter. C exits successfully before D starts. No batches or tensors are passed from C to D by the harness. No default checkpoint manager silently fixes your callbacks.

| Result | Meaning | CLI exit |
|---|---|---:|
| `PASS` | Control, interrupted prefix, restored observations and finite continuation horizon matched exactly. | 0 |
| `DRIFT` | The comparison was valid, but one or more observed states or metrics differed. This is not an automatic root-cause diagnosis. | 1 |
| `INVALID` | Nondeterministic control, unsupported observation, timeout, failed process, missing checkpoint, budget limit or another incomplete experiment. | 2 |

A changing continuous control is not blamed on the checkpoint. Nonfinite tensors, an incomplete horizon and missing worker output cannot become a pass.

## Evidence, not tensor dumps

Tensor evidence contains shape, dtype, logical byte count and SHA-256; equality is not inferred from norms. Keys are typed and escaped, empty containers count, and integer `1` is different from string `'1'`. Numeric metrics are shown; raw weights and tensor elements are not.

Generic tensor fingerprints compare logical values, not strides, alias relationships or allocator state. The optional TorchSession adds a conservative storage-overlap rejection step; it does not redefine the generic value contract. There is no tolerance tuning. Checkpoint hashes identify the observed bytes; they do not authenticate their origin. Relative checkpoint names and factory configuration are hashed rather than echoed. Hashing is not anonymization: predictable inputs may still be guessed, so review reports before sharing.

The JSON report is machine-readable (`schema_version: 1`). Difference details are bounded and include a full difference count. Each output file is published atomically, but writing JSON and HTML to separate destinations is not a cross-file transaction. Observation names and user-provided numeric metrics are your own disclosure boundary: do not use them for secrets.

## Limits to take seriously

- Supported target: deterministic CPU eager training, whole optimizer steps, one process and one intra-op thread per worker. GPU, distributed training, AMP, compiled graphs, background user threads and multiprocessing data loaders are outside this release.
- Workers use deterministic algorithms and one intra-op thread. This tests that mode, not your original parallel-performance configuration. Parent process settings and RNGs are not changed.
- Same observed state over a short horizon does not prove all relevant state was exposed or future steps will match. This is not a model-quality benchmark or a convergence test.
- The tool invokes your code and repeats data access. Use an isolated development experiment, not a callback that writes to production systems. User code can perform arbitrary side effects.
- Default worker timeout is 120 seconds. Only the child handles created by this test are terminated; do not launch subprocesses or nonterminating background threads from your callback.
- Observations and checkpoint files have resource limits. A limit hit is `INVALID`, not a partial pass. Large real-world models may require different budgets or a deliberately smaller test configuration.
- Cross-platform/version bitwise reproducibility is not promised. `weights_only=True` in the demo is a restricted loader, not a malicious-file sandbox.

## Related work and why this exists

[pytest-torch](https://github.com/AKJ7/pytest-torch) already offers useful training-loop and resume-equivalence assertions. ResumeTrace does not claim to invent those checks: its narrow emphasis is an actual new interpreter, pipeline re-entry, an independent repeated continuous control, and an inspectable offline report.

[TorchSnapshot](https://github.com/pytorch/torchsnapshot), [Lightning checkpoints](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html) and [Hugging Face Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) manage training/checkpoint workflows. ResumeTrace complements a checkpoint implementation by exercising the actual callbacks; it is not a replacement.

PyTorch references: [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), [serialization](https://docs.pytorch.org/docs/stable/notes/serialization.html), [optimizer loading](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.load_state_dict.html).

## Develop and contribute

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m pip install build
python -m build
```

A useful bug report includes a tiny importable factory, supported environment versions, `steps`/`split`/seeds and a redacted JSON report. Do not attach credentials, private datasets or trained weights. New integrations should include both correct recovery and a real broken-checkpoint control; a preset report saying “failed” is not a regression test.

The code, examples and documents were developed with AI assistance and local execution/review. Test outcomes and compatibility evidence should be inspected at the exact revision you use; no external-user adoption or performance claims are made.

MIT licensed. PyTorch and NumPy remain under their own licenses.

# GPU contention manufactures test failures that read exactly like code regressions

*Date: 2026-08-14. Origin: plan `plan-2026-08-14T042537-ff96c6c6`, defect D-3, measured at HEAD `accdff772`.*

A defect handed forward between plans claimed **8 pre-existing test failures**. Serial
re-measurement on an idle GPU found **zero**. This note records the refutation, the mechanism that
produced the phantom failures, and — most importantly — why the remedy that was proposed for them
(`xfail(strict=True)`) would have been actively harmful.

## The refuted premise

Stated verbatim as it was handed forward:

> 8 pre-existing failures in `tests/test_train/{test_dino/test_train_dino.py,
> test_energy_transformer/test_build_raw_image_dataset.py}`. Traced by git log to prior plans
> (e798a9e1, cecf4357), not to the gpt2/wave_field work. Diagnose each; fix or xfail(strict=True)
> with a stated reason.

Both halves of the premise fail. There is no stable set of 8 failures, and the two named plans
authored the current *content* of those files but did not introduce any failure — because there is
no failure to introduce.

## Measured controls

All four commands re-run at HEAD `accdff772`, `CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg`, `.venv`,
**strictly one invocation at a time**, with GPU 1 verified idle (15 MiB used) before starting.

| Target | Command | Result |
|---|---|---|
| dino trainer suite | `pytest tests/test_train/test_dino/` | **110 passed** (233s) |
| energy_transformer trainer suite | `pytest tests/test_train/test_energy_transformer/` | **31 passed** (70s) |
| the two files named in the premise | `pytest tests/test_train/test_dino/test_train_dino.py tests/test_train/test_energy_transformer/test_build_raw_image_dataset.py` | **73 passed** (205s) |
| both directories together | `pytest tests/test_train/test_dino/ tests/test_train/test_energy_transformer/` | **141 passed** (314s) |

`--collect-only` on the last target reports **141 collected**, so 141 passed is the whole suite, not
a subset that quietly skipped the interesting cases. 110 + 31 = 141 confirms the two directories
compose exactly.

Zero failures, zero errors, zero skips, in every configuration — file-scoped, directory-scoped, and
combined. These same four numbers were measured once before at HEAD `b59317d41` and reproduced
identically here after four steps of unrelated work landed.

## The mechanism: self-inflicted GPU contention

The original "8 failures" came from runs made while **another GPU-touching process was live on the
same device**. Under that condition the observed fault is:

```
tensorflow.python.framework.errors_impl.InternalError: Failed copying input tensor
from CPU:0 to GPU:0 ... Dst tensor is not initialized
```

raised inside `tf.data`'s `structure.normalize_element` — a host-to-device copy failing for want of
memory. It is **not an assertion failure**. That distinction is the tell: a real product defect
fails an `assert`; contention fails in the runtime before the test's own logic is ever reached.

The failure count is unstable across runs by construction. Successive runs of the identical command
under contention produced 0, 4, 3, 1, 0, 0 failures — never the same set twice. **An intermittent
failure set refutes a semantic cause on its own**; no code change can be responsible for a fault
that comes and goes with no code change.

This is a repeat. `src/train/CLAUDE.md:584` already records the same class from a different root
cause: a module-scope `tf.constant(...)` in `train/common/image_text.py` made every importer
allocate a GPU device, and "once produced a false 12-error test 'regression' that was really
`cudaSetDevice()` self-contention between concurrent suites." That root cause is fixed; the
contention class it exposed is not, because the class is about **how the suite is run**, not about
any line of code.

The repo's standing rule — never run GPU jobs in parallel — therefore binds **test invocations and
agent fan-out**, not just training runs. It is easy to honour the rule for a training job and break
it by dispatching two parallel workers that each happen to run pytest.

## A second, unrelated transient

One CPU-only run failed with:

```
_pickle.UnpicklingError: pickle data was truncated
```

reading the on-disk CIFAR10 cache inside `keras/src/datasets/cifar.py` — third-party code, not this
repo. The cache file is intact on disk and the same run passed 10/10 on retry. A transient I/O
glitch, recorded here only so it is not mistaken for the contention signature above; the two look
nothing alike and have nothing to do with each other.

## Forbidden remedies

**`xfail(strict=True)` is wrong here, and would have been worse than doing nothing.** A strict xfail
asserts that a test *does* fail. These tests pass 141 of 141 on a quiet device. Marking them
`xfail(strict=True)` would turn every clean run RED — manufacturing precisely the failure the marker
was meant to record, and converting a non-problem into a permanent, self-sustaining one. The marker
would then "prove" the premise that motivated it. This is the trap worth remembering: a strict xfail
is a claim about the future, and applying one to a flaky-by-environment test inverts its meaning.

**Weakening an assertion is forbidden** — loosening a tolerance, deleting an `assert`, broadening an
expected set. It is also unnecessary, since nothing fails. Any change that makes these suites
"greener" is changing a suite that is already green.

**Fixing product code is wrong** because no product behaviour is at fault. There is nothing to fix.

The correct response to a reported failure that will not reproduce serially is to record the
non-reproduction with its controls — this document — not to suppress the symptom.

## How to reproduce a false positive

This section is what makes the note falsifiable rather than an assertion of innocence. To
manufacture the phantom failures on demand:

1. Start any GPU-resident workload on GPU 1 (a training run, or simply a second pytest invocation
   that constructs Keras models).
2. Concurrently run `CUDA_VISIBLE_DEVICES=1 pytest tests/test_train/test_energy_transformer/test_build_raw_image_dataset.py`.
3. Expect a non-deterministic subset of tests to fail with `Dst tensor is not initialized`, raised
   from a `tf.data` host-to-device copy, with the failing set differing between runs.

If that procedure produces stable, identical, assertion-shaped failures instead, this note is wrong
and the premise deserves re-opening.

## Diagnostic checklist for the next reported "pre-existing failures"

1. Was anything else touching the GPU? Check `nvidia-smi` **before** the run, not after.
2. Re-run the identical command serially, twice. Does the failing **set** (not the count) reproduce?
3. Is the fault an `AssertionError`, or a runtime/allocation error raised before the assertion?
4. Does it reproduce on CPU (`CUDA_VISIBLE_DEVICES=""`)?

Only a fault that survives all four is a product defect.

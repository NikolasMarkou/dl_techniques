# `train.ntm` — Neural Turing Machine trainers + MANN benchmark surface

This package holds the training entry points for the NTM models in
`dl_techniques.models.ntm`, plus a benchmark/metrics surface for
memory-augmented neural networks (MANNs) modelled on the NTM/DNC literature.

**Import path.** Everything here lives under `train.ntm`:

```python
from train.ntm import CopyTaskGenerator, CopyTaskConfig, evaluate_copy_task
```

Older revisions of this document imported every example from a standalone
benchmarks package that does not exist in this repository, so none of those
snippets ever ran. If a snippet you found elsewhere imports from anything other
than `train.ntm`, it is stale — the package is not installable under any other
name.

## Contents at a glance

| Module | What it is | Called by an entry point? |
|--------|------------|---------------------------|
| `train_ntm.py` | Single-task copy-task trainer (runnable) | — it *is* an entry point |
| `train_multitask.py` | Six-task multi-task trainer (runnable) | — it *is* an entry point |
| `run_benchmark_suite.py` | Benchmark-suite CLI over a saved `.keras` model (runnable) | — it *is* an entry point |
| `config.py` | Task/benchmark config dataclasses | yes |
| `data_generators.py` | Copy, associative recall, repeat copy, priority access, traversal, dynamic n-gram, algorithmic | yes |
| `metrics.py` | `evaluate_*` functions + stateful `keras.metrics.Metric` classes | evaluate functions: yes, via `harness.py`; classes: **no** |
| `harness.py` | `BenchmarkHarness`, `create_benchmark_callbacks` | `BenchmarkHarness`: yes, by `run_benchmark_suite.py`; `create_benchmark_callbacks`: **no** |
| `compositional_generators.py` | SCAN / COGS / CFQ generators | `ScanGenerator`: yes, via the `scan` benchmark; COGS/CFQ: **no** |
| `babi_generator.py` | bAbI QA story generator | yes, via the `babi` benchmark |

The authoritative list of public names is `__all__` in `src/train/ntm/__init__.py`.

---

## Runnable entry points

There are **three**: `train_ntm.py`, `train_multitask.py` and
`run_benchmark_suite.py`. All three parse arguments before touching a GPU, so
`--help` prints and exits 0 without starting a run — pinned by
`tests/test_train/test_ntm/test_ntm_trainers.py::TestCommandLineContract`, which
tripwires `setup_gpu`.

Only `train_ntm.py` still takes its common flags from
`train.common.create_base_argument_parser`. `train_multitask.py` and
`run_benchmark_suite.py` build a **local** `argparse.ArgumentParser` (the
Pattern 2 shape documented in `src/train/CLAUDE.md`), because the shared
vision-oriented parser contributed five flags neither of them reads.

### 1. Copy task — `train_ntm.py`

Builds `create_ntm(...)` (from `dl_techniques.layers.memory`) inside a functional
Keras model with a sigmoid output, trains on `CopyTaskGenerator` data with the
task mask supplied as `sample_weight`, then calls `evaluate_model` on a random
sample of training sequences.

```bash
MPLBACKEND=Agg python -m train.ntm.train_ntm --help

MPLBACKEND=Agg python -m train.ntm.train_ntm \
    --epochs 50 --batch-size 32 --learning-rate 1e-4 \
    --sequence-length 20 --vector-size 8 --num-samples 100000 \
    --memory-size 128 --memory-dim 20 --controller-dim 100 \
    --gpu 1
```

Flags (verbatim from `--help`):

| Flag | Default | Notes |
|------|---------|-------|
| `--dataset` | `copy` | only choice is `copy` |
| `--epochs`, `--batch-size`, `--learning-rate`, `--patience`, `--gpu` | from the base parser | wired through |
| `--memory-size` | `128` | memory locations |
| `--memory-dim` | `20` | width of one memory slot |
| `--controller-dim` | `100` | controller hidden size |
| `--controller-type` | `lstm` | `lstm` or `mlp` |
| `--num-read-heads` / `--num-write-heads` | `1` / `1` | |
| `--sequence-length` | `20` | copied payload length (total timeline is `2*L + delay + 2`) |
| `--vector-size` | `8` | bits per timestep; the input carries `vector_size + 2` channels (start + delimiter) |
| `--num-samples` | `100000` | generated up front, in memory |
| `--clip-norm` | `1.0` | optimizer `clipnorm` |
| `--validation-split` | `0.1` | passed to `model.fit` |
| `--num-eval-samples` | `20` | sequences drawn for the post-training report |
| `--success-threshold` | `0.9` | sequence accuracy above which the log says SUCCESS |

The base parser also exposes `--image-size`, `--weight-decay`, `--lr-schedule`
and `--show-plots`. **`train_ntm.py` does not read any of them** — they are
inherited from the shared vision-oriented parser and are silent no-ops here.
`train_multitask.py` used to inherit the same dead tail and no longer does: it
moved to a local parser and now rejects those flags outright.

### 2. Multi-task — `train_multitask.py`

One NTM trained on six tasks simultaneously (`copy`, `associative_recall`,
`repeat_copy`, `priority_access`, `dynamic_ngram`, `insertion_sort`), conditioned
by a one-hot task vector concatenated onto every input timestep
(`dl_techniques.models.ntm.model_multitask.NTMMultiTask`).

```bash
MPLBACKEND=Agg python -m train.ntm.train_multitask --help

MPLBACKEND=Agg python -m train.ntm.train_multitask \
    --epochs 100 --batch-size 64 --steps-per-epoch 1000 \
    --validation-steps 100 --memory-size 128 --memory-dim 20 \
    --controller-dim 256 --patience 50 --gpu 1
```

Flags (verbatim from `--help`; this trainer uses a **local** parser, so this is
the whole surface — there is no inherited base-parser tail):

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs` | `100` | maximum training epochs |
| `--batch-size` | `64` | rows per training/validation batch |
| `--learning-rate` | `0.0001` | Adam learning rate |
| `--patience` | `50` | EarlyStopping patience; matches `MultitaskNTMConfig.patience` |
| `--steps-per-epoch` | `1000` | batches per training epoch |
| `--validation-steps` | `100` | batches per validation pass |
| `--clip-norm` | `1.0` | optimizer `clipnorm` |
| `--num-eval-samples` | `1000` | rows per task in the final evaluation |
| `--memory-size` | `128` | memory slots (N) |
| `--memory-dim` | `20` | width of one memory slot (M) |
| `--controller-dim` | `256` | controller hidden width |
| `--controller-type` | `lstm` | `lstm`, `gru` or `feedforward` |
| `--num-read-heads` / `--num-write-heads` | `1` / `1` | |
| `--shift-range` | `3` | location-addressing shift width; must be a positive **odd** integer (`NTMConfig` validates) |
| `--max-seq-length` | `100` | timeline width every task is padded/truncated to |
| `--max-vector-size` | `16` | feature width every task is padded/truncated to |
| `--gpu` | `None` | GPU device index |

The last six of those (`--controller-type` through `--max-vector-size`) are new:
those `MultitaskNTMConfig` fields previously had no flag at all. Conversely
`--dataset`, `--image-size`, `--weight-decay`, `--lr-schedule` and `--show-plots`
are **gone** — they came from the shared parser, were never read here, and the
local parser now *refuses* them rather than accepting them as silent no-ops.
Only `train_ntm.py` still carries that inherited-no-op tail (see its own note
above).

> **There is no `train_multitask_v2.py`.** Two near-duplicate multi-task
> trainers used to exist; the argparse-less one was deleted and the surviving
> file *is* the former v2, renamed. Any reference to `v2` predates that.

Ctrl-C during `fit` saves the partially trained model to
`<results_dir>/multitask_ntm_interrupted.keras` before returning.

Both trainers write to a repo-root `results/` directory created by
`train.common.create_callbacks`.

### 3. Benchmark suite — `run_benchmark_suite.py`

A thin CLI over `BenchmarkHarness`. It loads a saved `.keras` model, runs
`run_full_suite(...)` and writes a JSON report. No measurement lives here —
every metric comes from `harness.py` / `metrics.py`.

```bash
MPLBACKEND=Agg python -m train.ntm.run_benchmark_suite --help

MPLBACKEND=Agg python -m train.ntm.run_benchmark_suite \
    --checkpoint results/ntm_copy_.../final_model.keras \
    --benchmarks copy_task length_generalization \
    --output results/ntm_benchmarks --gpu 1
```

| Flag | Default | Notes |
|------|---------|-------|
| `--checkpoint` | *(required)* | path to the saved `.keras` model |
| `--output` | `results/ntm_benchmarks` | repo-root `results/` tree; never `src/results/` |
| `--benchmarks` | all six | subset of `copy_task`, `associative_recall`, `length_generalization`, `memory_capacity`, `scan`, `babi` |
| `--model-name` | checkpoint stem | name recorded in the report |
| `--gpu` | `None` | GPU device index |
| `--quiet` | off | sets `BenchmarkSuiteConfig.verbose=False`, which also silences the per-benchmark error log |

> **Expect most benchmarks to ERROR on any single-task checkpoint. That is the
> design, not a bug.** Each benchmark generates its own inputs with its own
> shape and arity: the copy task feeds one `(batch, T, vector_size + 2)` tensor,
> associative recall and memory capacity use a different feature width, SCAN
> uses its own encoded command length, and bAbI passes **two** inputs
> (story, question). A model trained for one of them cannot accept the others,
> so `model.predict` raises. `run_full_suite` contains each failure to its own
> benchmark, logs it with a traceback and keeps going — so a run against a
> copy-task model legitimately prints ~4 tracebacks and still writes a report
> with the 2 benchmarks it could run. **Measured** (step 8 of
> `plan-2026-08-03T161943-02be1d7e`, freshly-built untrained copy-shaped NTM,
> input `(23, 10)` → output `(23, 8)`): `copy_task` and
> `length_generalization` recorded; `associative_recall` and `memory_capacity`
> failed with `Matrix size-incompatible: In[0]: [32,13], In[1]: [18,128]`,
> `scan` with `expected shape=(None, 23, 10), found shape=(32, 7)`, and `babi`
> with `expects 1 input(s), but it received 2`. Pass `--benchmarks` to run only
> the ones your checkpoint is shaped for.

There is no `run_algorithmic_benchmark` entry in the suite: the method exists on
`BenchmarkHarness` but is not in `BENCHMARK_METHODS`, so neither `run_full_suite`
nor this CLI can reach it.

---

## Library surface without a caller

`run_benchmark_suite.py` (above) now drives `BenchmarkHarness.run_full_suite`,
`save_report`, `ScanGenerator`, `BabiGenerator` and the `evaluate_*` functions
end to end, so the bulk of what this section used to list is exercised. What
remains genuinely caller-less:

- `create_benchmark_callbacks` (`harness.py`) — no entry point builds them.
- `BenchmarkHarness.run_algorithmic_benchmark` — implemented but absent from
  `BENCHMARK_METHODS`, so unreachable from the suite or the CLI.
- `BenchmarkHarness.get_keras_metrics` — nothing calls it.
- `CogsGenerator` and `CFQGenerator` (`compositional_generators.py`) — imported
  by `harness.py` but never used there; no benchmark consumes them.
- The stateful `keras.metrics.Metric` classes in `metrics.py` —
  `SequenceAccuracy`, `PerStepAccuracy`, `BitErrorRate`, `ExactMatchAccuracy`,
  plus `MemoryUtilizationMetric`. No trainer passes them to `compile()`;
  they compile with `keras.metrics.BinaryAccuracy` (and `MeanSquaredError`).

Note also that `train_ntm.py` keeps its own local `evaluate_model` rather than
calling `evaluate_copy_task` — two implementations of nearly the same
measurement, deliberately not unified.

> **Name collision.** `dl_techniques.metrics.sequence_metrics` also defines
> `SequenceAccuracy` and `BitErrorRate`. They are different classes from the ones
> here. Import explicitly.

---

## Data generators (used by the trainers)

```python
from train.ntm import CopyTaskGenerator, CopyTaskConfig

config = CopyTaskConfig(sequence_length=10, vector_size=8, delay_length=1)
data = CopyTaskGenerator(config).generate(num_samples=4)

data.inputs.shape    # (4, 23, 10)  -> 2*10 + 1 + 2 timesteps, 8 + 2 channels
data.targets.shape   # (4, 23, 8)
data.masks.shape     # (4, 23)      -> 1.0 only on the output phase
```

`TaskData` carries `inputs`, `targets`, optional `masks` and a `metadata` dict.
`masks` is **per timestep**, shape `(batch, steps)` — not per element. Anything
comparing it against a `(batch, steps, features)` prediction must broadcast it
across the last axis first.

Other generators in `data_generators.py`: `AssociativeRecallGenerator`,
`RepeatCopyGenerator`, `PriorityAccessGenerator`, `TraversalGenerator`,
`DynamicNGramGenerator`, `AlgorithmicTaskGenerator`.

```python
from train.ntm import AlgorithmicTaskGenerator, AlgorithmicTaskConfig

gen = AlgorithmicTaskGenerator(AlgorithmicTaskConfig(task_name="insertion_sort"))
gen.generate(num_samples=2, problem_size=8).inputs.shape   # (2, 8, 1)
```

## Evaluation functions

```python
from train.ntm import evaluate_copy_task

results = evaluate_copy_task(model, inputs, targets, masks)
results.metrics["sequence_accuracy"].value
results.metrics["per_step_accuracy"].value
results.metrics["bit_error_rate"].value
```

Also available: `evaluate_associative_recall`, `evaluate_babi_task`,
`compute_length_generalization_score`, `compute_capacity_degradation_curve`.

---

## Known limitations

Each of these is true of the code as it stands. They are the things most likely
to waste your afternoon.

**1. `evaluate_copy_task`'s `per_step_accuracy` / `bit_error_rate` changed
meaning.** They are now reduced over masked-true (output-phase) elements only.
Previously they were reduced over the whole tensor, where the masked-out
positions were zeroed on *both* sides and therefore agreed trivially — for the
default `CopyTaskConfig` that is roughly half the timeline. **Any historical
number quoted from this function was diluted upward and is not comparable with
what you get today; the new numbers read lower.** `sequence_accuracy` is
deliberately unchanged: it is reduced over the full tensor precisely *because*
masked positions agree trivially, which is what makes it mean "the supervised
region matched exactly".

**2. `BabiGenerator` implements 10 of the 20 bAbI tasks — but it now says so.**
The implemented ids are 1, 2, 3, 6, 7, 8, 11, 15, 17, 19, held in one place as
`BabiGenerator.IMPLEMENTED_TASK_IDS` (derived from the task-generator map, so
the two cannot disagree). `BabiTaskConfig.task_ids` defaults to exactly that
set, and `BabiGenerator.__init__` raises `ValueError` listing every unsupported
id you asked for:

```
ValueError: bAbI tasks [4, 5, 9, 10, 12, 13, 14, 16, 18, 20] are not implemented.
Implemented task ids: [1, 2, 3, 6, 7, 8, 11, 15, 17, 19].
```

The two branches that used to swallow that error — `generate_all_tasks`'s bare
`except ValueError: continue` and `run_babi_benchmark`'s `verbose`-gated one —
are **deleted**, not merely logged. A short result dict is no longer a possible
outcome. The consequence to know: `run_babi_benchmark` now also stops swallowing
a model/benchmark **input-shape** mismatch, which used to be silent whenever
`verbose=False`; it aborts the bAbI benchmark loudly instead, contained by
`run_full_suite`'s own `except Exception`.

The remaining honest limitation is that the other ten tasks are still
unimplemented — the fix here was to stop under-reporting, not to write them.

**3. `AlgorithmicTaskGenerator.SUPPORTED_TASKS` lists 10 tasks; only
`insertion_sort` is reachable from a trainer.** The other nine (`bubble_sort`,
`binary_search`, `linear_search`, `bfs`, `dfs`, `dijkstra`, `minimum`,
`maximum`, `reverse`) are implemented and constructible, but nothing in
`train_multitask.py`'s task map selects them.

**4. The SCAN generator is a small synthetic stand-in, not the released corpus.**
`ScanGenerator` builds its samples from an inline grammar — 446 command/action
pairs, action lengths `{1: 6, 2: 84, 3: 204, 4: 136, 6: 8, 8: 8}`. The real SCAN
release has ~20,900 pairs with action sequences up to 48 tokens. Treat any number
from this generator as a smoke signal, not a SCAN result.

Every split is now non-degenerate, and a degenerate one would raise rather than
return. Measured `(train, test)` sizes at the current grammar:

| `split_type` | train / test | Holds out |
|--------------|--------------|-----------|
| `simple` | 356 / 90 | a random 20% |
| `length` | 294 / 152 | action sequences longer than a **corpus-derived** threshold (`observed_max * 22 // 48`, the SCAN paper's train/max ratio) — not the old hardcoded 24, which this grammar could never exceed and which therefore returned 446/0 |
| `add_prim_jump` | 288 / 158 | every composed command containing `jump` |
| `add_prim_turn_left` | 392 / 54 | every composed command containing `turn left` |
| `template_around_right` | 442 / **4** | the `<prim> around right` template |

An unrecognised `split_type` string now raises `ValueError` naming the five
supported members. It used to silently alias to `simple`, as did
`template_around_right` itself, which was declared in `ScanSplit` but never
dispatched.

> **`template_around_right` is a probe, not a peer split.** Its held-out side is
> exactly 4 samples (`walk|run|jump|look around right`) out of 446. That is the
> semantically correct hold-out for this grammar's template, and it is
> deliberately not widened — but 4 samples cannot produce a stable accuracy
> estimate. Use it to check that a model has *not* seen the template, not to
> quote an accuracy. The other four splits are large enough to report.

**5. `DynamicNGramGenerator` never sets `masks`.** `train_multitask.py` builds
one itself, zeroing the first two timesteps (n-gram warm-up context) and the
last one (the generator only fills targets for `t < sequence_length - 1`, so the
final row has no next token). If you consume this generator directly, you must
do the same or you will supervise undefined positions.

**6. Multi-task padding is lossy by construction.** `_pad_and_normalize`
truncates to `max_seq_length` (100) and `max_vector_size` (16). A task that
generates a longer or wider frame is silently clipped.

**7. `evaluate_tasks` in `train_multitask.py` evaluates
`config.num_eval_samples` (default 1000) rows per task in a single batch**, not
`batch_size` rows. That is a deliberate override so the logged count is the
evaluated count; it also means the end-of-run evaluation is ~15x the training
batch in memory.

---

## Tests

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m pytest tests/test_train/test_ntm/ -q
# 77 passed, 1 skipped
```

Three files:

- `test_ntm_trainers.py` — the trainers' pure helpers, the argparse contract for
  all three entry points (`--help` exits 0 rather than starting a run, with a
  `setup_gpu` tripwire), and one args→config assertion per `train_multitask.py`
  flag.
- `test_generators_scan_babi.py` — the SCAN and bAbI generators: per-split
  partition guards, the corpus-derived length threshold, the unknown-split
  raise, the bAbI construction refusal, and task 19's non-degeneracy.
- `test_ntm_learnability.py` — the **one** convergence test. It is the `1 skipped`
  above: double-gated by `pytest.mark.slow` **and** an explicit
  `NTM_RUN_LEARNABILITY` opt-in, because this repo has no
  `addopts = -m "not slow"`, so the marker alone would drop a ~50 s GPU job into
  the default suite. To run it:

  ```bash
  CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg NTM_RUN_LEARNABILITY=1 \
      python -m pytest tests/test_train/test_ntm/test_ntm_learnability.py -m slow -v
  ```

  It trains a tiny NTM on the copy task to >90% validation bit accuracy. The
  **seed is pinned** deliberately: convergence is seed-sensitive in the failing
  direction (seeds 1234/99 crossed at epochs 26/22; seed 7 was still at 0.882
  after 250 epochs). Treat it as a "the NTM still learns" liveness guard, not a
  quality benchmark.

The layer-level and model-level suites are separate:

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg python -m pytest \
    tests/test_layers/test_memory/ tests/test_models/test_ntm/ -q
```

## See also

- `src/dl_techniques/models/ntm/README.md` — `NTMModel`, `create_ntm_variant`,
  `NTMMultiTask`.
- `src/dl_techniques/layers/memory/README.md` — `NTMCell`, `create_ntm`, and the
  underlying Graves addressing math.

## References

1. Graves, Wayne & Danihelka (2014). *Neural Turing Machines.* arXiv:1410.5401.
2. Graves et al. (2016). *Hybrid computing using a neural network with dynamic
   external memory* (DNC). Nature 538.
3. Lake & Baroni (2018). *Generalization without systematicity* (SCAN).
4. Kim & Linzen (2020). *COGS: A Compositional Generalization Challenge.*
5. Veličković et al. (2022). *The CLRS Algorithmic Reasoning Benchmark.*
6. Weston et al. (2015). *Towards AI-Complete Question Answering* (bAbI).

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

| Module | What it is | Called by a trainer? |
|--------|------------|----------------------|
| `train_ntm.py` | Single-task copy-task trainer (runnable) | — it *is* an entry point |
| `train_multitask.py` | Six-task multi-task trainer (runnable) | — it *is* an entry point |
| `config.py` | Task/benchmark config dataclasses | yes |
| `data_generators.py` | Copy, associative recall, repeat copy, priority access, traversal, dynamic n-gram, algorithmic | yes |
| `metrics.py` | `evaluate_*` functions + stateful `keras.metrics.Metric` classes | evaluate functions: no (see below); classes: no |
| `harness.py` | `BenchmarkHarness`, `create_benchmark_callbacks` | **no** |
| `compositional_generators.py` | SCAN / COGS / CFQ generators | **no** |
| `babi_generator.py` | bAbI QA story generator | **no** |

The authoritative list of public names is `__all__` in `src/train/ntm/__init__.py`.

---

## Runnable entry points

Both scripts take their common flags from `train.common.create_base_argument_parser`,
so `--help` prints and exits without starting a run.

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
and `--show-plots`. **Neither NTM trainer reads them** — they are inherited from
the shared vision-oriented parser and are silent no-ops here.

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

Flags beyond the base parser: `--memory-size` (128), `--memory-dim` (20),
`--controller-dim` (256), `--steps-per-epoch` (1000), `--validation-steps` (100),
`--clip-norm` (1.0).

> **There is no `train_multitask_v2.py`.** Two near-duplicate multi-task
> trainers used to exist; the argparse-less one was deleted and the surviving
> file *is* the former v2, renamed. Any reference to `v2` predates that.

Ctrl-C during `fit` saves the partially trained model to
`<results_dir>/multitask_ntm_interrupted.keras` before returning.

Both trainers write to a repo-root `results/` directory created by
`train.common.create_callbacks`.

---

## Library surface with no trainer caller

The following are implemented, importable and imported by
`src/train/ntm/__init__.py`, but **no `train_*.py` in this package calls them**.
Running a trainer does not exercise them, and nothing in the trainers proves
they work end-to-end:

- `harness.py` — `BenchmarkHarness` (`run_copy_task_benchmark`,
  `run_associative_recall_benchmark`, `run_length_generalization_benchmark`,
  `run_capacity_benchmark`, `run_babi_benchmark`, `run_scan_benchmark`,
  `run_algorithmic_benchmark`, `run_full_suite`, `save_report`) and
  `create_benchmark_callbacks`.
- `compositional_generators.py` — `ScanGenerator`, `CogsGenerator`, `CFQGenerator`.
- `babi_generator.py` — `BabiGenerator`.
- The stateful `keras.metrics.Metric` classes in `metrics.py` —
  `SequenceAccuracy`, `PerStepAccuracy`, `BitErrorRate`, `ExactMatchAccuracy`,
  plus `MemoryUtilizationMetric`. Neither trainer passes them to `compile()`;
  they compile with `keras.metrics.BinaryAccuracy` (and `MeanSquaredError`).
- The `evaluate_*` functions in `metrics.py` are called only by `harness.py` and
  by the test suite. `train_ntm.py` has its own local `evaluate_model`.

Treat this surface as usable-but-unproven scaffolding, not as a validated
benchmark pipeline.

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

**2. `BabiTaskConfig.task_ids` defaults to all 20 bAbI tasks; `BabiGenerator`
implements 10.** The implemented ids are 1, 2, 3, 6, 7, 8, 11, 15, 17, 19.
`BabiGenerator.generate(task_id)` raises `ValueError` for the other ten;
`BenchmarkHarness.run_babi_benchmark` catches it and logs a
`Skipping task N: ...` warning (when the suite config's `verbose` is set), so a
"full" bAbI run silently covers half the suite. Narrow `task_ids` yourself if you
want that to be explicit.

**3. `AlgorithmicTaskGenerator.SUPPORTED_TASKS` lists 10 tasks; only
`insertion_sort` is reachable from a trainer.** The other nine (`bubble_sort`,
`binary_search`, `linear_search`, `bfs`, `dfs`, `dijkstra`, `minimum`,
`maximum`, `reverse`) are implemented and constructible, but nothing in
`train_multitask.py`'s task map selects them.

**4. The SCAN generator is a small synthetic stand-in, not the released corpus.**
`ScanGenerator` builds its samples from an inline grammar. Measured:
`ScanGenerator(ScanTaskConfig(split_type="length")).generate_split()` returns
254 train / **0 test** samples, because `_length_split` holds out action
sequences longer than 24 tokens and the bundled grammar never emits one. The
`length` split therefore does not test length generalization as-is.

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
```

`tests/test_train/test_ntm/test_ntm_trainers.py` covers the trainers' pure
helpers and the argparse contract (`--help` exits 0 rather than starting a run).
It deliberately contains no convergence or `fit`-scale test.

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

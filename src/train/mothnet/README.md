# mothnet — MothNet Hebbian Trainer

Trains `MothNet` (`dl_techniques.models.general_purpose.mothnet.model`) on a small MNIST
subsample via its hand-rolled `train_hebbian()` method — the model's readout layer is
`trainable=False` and is updated by a direct Hebbian `.assign()`, never by `model.fit()`
or a custom `train_step`. Outputs land under repo-root `results/<run>/`, built from the
same shared primitives (`prepare_run_dir`, `default_experiment_name`,
`save_training_history_json`, `best_checkpoint_path`) the `bfunet`/`kan` trainers use.

---

## No gradient-based training

MothNet has **no optimizer, no loss backprop, and no `model.fit()` path**. Every epoch of
this trainer calls `model.train_hebbian(x_train, y_train, epochs=1, batch_size=...,
verbose=0)` — the ONE weight-mutating step, updating only the `HebbianReadoutLayer`'s
`readout_weights` via a local Hebbian rule (`ΔW = α · (1/N) · Σ(x_i ⊗ y_i)`). The
Antennal Lobe (competitive inhibition) and Mushroom Body (fixed sparse random
projection) layers never train — see "MB-sparsity visualization" below for what that
implies for the periodic plot.

---

## Quick start

Always run with a non-interactive matplotlib backend and from the repo root:

```bash
# Plain invocation — CLI defaults (3 epochs, 2000 train / 500 val samples)
MPLBACKEND=Agg .venv/bin/python -m train.mothnet.train_mothnet

# Custom experiment name
MPLBACKEND=Agg .venv/bin/python -m train.mothnet.train_mothnet \
    --experiment-name my_custom_name --epochs 5

# Larger-scale run — more MNIST samples and MB units, longer training
MPLBACKEND=Agg .venv/bin/python -m train.mothnet.train_mothnet \
    --num-train-samples 10000 --num-val-samples 2000 \
    --mb-units 8000 --epochs 10 --viz-freq 5 --gpu 1
```

`--help` exits 0, prints a `usage:` line, and performs no GPU/dataset/model work.

---

## CLI flags

All flags are defined in `parse_arguments()`. This trainer uses a fresh, purpose-built
`argparse.ArgumentParser` rather than `train.common.create_base_argument_parser()` — see
`plans/plan-2026-09-18T045308-c89cdf76/decisions.md` D-003 for why (MothNet has no
optimizer/LR-schedule/patience/dataset-choice surface for that shared parser to cover).

| Flag | Default | Meaning |
|------|---------|---------|
| `--mb-units` | `2000` | Number of Mushroom Body units (matches MothNet's own constructor default) |
| `--mb-sparsity` | `0.1` | Fraction of MB units allowed to fire per sample |
| `--al-units` | `None` | Number of Antennal Lobe units; `None` lets MothNet infer it from the input dimension |
| `--connection-sparsity` | `0.1` | Mushroom Body projection connection sparsity |
| `--hebbian-learning-rate` | `0.01` | Learning rate used by the Hebbian readout update |
| `--inhibition-strength` | `0.5` | Antennal Lobe inhibition strength |
| `--epochs` | `3` | Number of training epochs; each epoch is one `train_hebbian(epochs=1)` call. Validated `>= 1` at parse time |
| `--batch-size` | `32` | Mini-batch size passed to `train_hebbian` |
| `--viz-freq` | `2` | Render a periodic MB-sparsity visualization every N epochs; `<= 0` disables periodic visualization |
| `--gpu` | `0` | GPU index to use |
| `--output-dir` | `None` | Output directory for run artifacts; `None` means repo-root `results/`, resolved at run time |
| `--experiment-name` | `None` | Experiment name; `None` means auto-generated via `default_experiment_name` |
| `--seed` | `42` | Seed for NumPy's global RNG |
| `--num-train-samples` | `2000` | Number of MNIST training samples to subsample |
| `--num-val-samples` | `500` | Number of MNIST validation samples to subsample |

---

## Output layout

Each run writes to `results/<run>/` (same artifact set as `bfunet`/`train_kan.py`):

```
results/<run>/
├── config.json
├── best_model.keras
├── final_model.keras
├── training_log.csv
├── training_history.json
└── visualizations/
    ├── training_dashboard.png
    └── epoch_{NNN}_mb_sparsity.png   (one per --viz-freq interval, epoch-stamped)
```

- `best_model.keras` — saved via `best_checkpoint_path()` whenever an epoch's
  `val_accuracy` beats the best seen so far.
- `final_model.keras` — saved unconditionally after the loop exits, regardless of
  whether any epoch ever improved on the best.
- `training_log.csv` — one row per completed epoch (`epoch, loss, train_accuracy,
  val_accuracy`), flushed immediately after each write (never batched), so a mid-run
  crash leaves exactly N complete rows for N completed epochs.
- `visualizations/epoch_{NNN:03d}_mb_sparsity.png` — only written when `--viz-freq > 0`;
  filenames are epoch-stamped so periodic renders never overwrite each other.

---

## MothNet-specific caveats

**`mb_units` sizing is not enforced by this trainer.** The MothNet model's own README
(`src/dl_techniques/models/general_purpose/mothnet/README.md` §7/§12) recommends
`mb_units` be 20-50x the input dimension (784 for flattened MNIST, so roughly
16,000-39,000) for the Mushroom Body's sparse high-dimensional coding to work as
intended. This trainer's `--mb-units` default (`2000`, matching `MothNet`'s own
constructor default) is well under that range and is a toy-scale convenience, not a
tuned recommendation — nothing in `train_mothnet.py` or in `MothNet`/`train_hebbian`
validates the ratio, so passing an undersized `--mb-units` will not raise or warn.

**What `--seed` covers, and what it does not.** `--seed` seeds NumPy's global RNG
(`np.random.seed(args.seed)`, called once in `build_model()` before construction) and
`load_mnist_data`'s own `np.random.default_rng(config.seed)` used for subsample
selection. This reaches `train_hebbian`'s internal `np.random.permutation` shuffle
(`model.py`, no `seed=` kwarg exists there), making that shuffle reproducible across
runs. It does **not** reach Keras/TensorFlow-initialized weights: `AntennalLobeLayer`
and `HebbianReadoutLayer` use `kernel_initializer='glorot_uniform'`, which draws from
Keras/TF's own global RNG, never NumPy's — so a fixed `--seed` does not make weight
initialization reproducible run-to-run. This was measured during Step 3's execution
(see `plans/plan-2026-09-18T045308-c89cdf76/decisions.md` D-009); the trainer behaves as
intended (the shuffle is seeded, which was the actual target), but the initial plan's
claim that `--seed` also covered "model-init randomness" was wrong.

**The periodic MB-sparsity visualization looks identical across epochs — this is
expected, not a bug.** `visualizations/epoch_{NNN}_mb_sparsity.png` renders
`model.extract_mb_features(x_sample)`, which reads only the Antennal Lobe and Mushroom
Body layers — both are architecturally frozen (fixed competitive inhibition, fixed
sparse random projection) and never train; only the readout layer's weights change per
epoch. Measured at Step 6: successive renders (`--epochs 4 --viz-freq 2`, epoch_002 vs.
epoch_004) were byte-identical (`cmp` reported no difference). Do not expect this plot to
change during a run; it reflects a real architectural property of MothNet, not a
rendering defect.

**Small `--num-train-samples` can produce flat, at-chance results.** Measured at Step 7:
`--num-train-samples 500 --num-val-samples 200` produced degenerate `val_accuracy`
(0.09 → 0.09 → 0.10, at the 0.10 MNIST chance baseline) across 3 epochs, while the CLI
default of `2000` train / `500` val samples reached `val_accuracy` 0.114 → 0.150 → 0.198
over the same 3 epochs. If you pass a much smaller `--num-train-samples` than the
default and see a flat accuracy curve, this sample-count sensitivity — not a wiring bug
— is the likely explanation.

---

## Constraints & gotchas

- **Never route MothNet through `model.fit()`.** `train_hebbian` is the only training
  entry point; the readout weights are `trainable=False`.
- **Labels passed to `train_hebbian` must be one-hot**, never class indices —
  `load_mnist_data` already applies `keras.utils.to_categorical`.
- **Outputs go to repo-root `results/`.** Do not point `--output-dir` inside `src/`.
- **Always set `MPLBACKEND=Agg`** to avoid X11 crashes on headless/remote systems.

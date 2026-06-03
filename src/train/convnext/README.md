# ConvNeXt Training

Training scripts for ConvNeXt V1 / V2 (+ V2 MAE pretraining), plus a driver that
compares the two stochastic-regularization modes (`depth` vs `gradient`) so you
can choose between them.

Plan: `plans/plan_2026-06-03_bf1e592d` (follow-up to `plan_2026-06-03_943569ad`,
which added the `stochastic_mode` kwarg to the models).

## Scripts

| Script | Purpose |
|--------|---------|
| `train_convnext_v1.py` | Supervised ConvNeXt V1 (MNIST / CIFAR-10/100 / ImageNet). |
| `train_convnext_v2.py` | Supervised ConvNeXt V2 (GRN blocks). |
| `train_convnext_v2_mae.py` | ConvNeXt V2 encoder with MAE self-supervised pretraining + fine-tuning. |
| `run_stochastic_comparison.py` | Experiment driver: trains both `stochastic_mode`s serially and emits a `compare_runs` report. |

Run any trainer as a module from the repo root, e.g.:

```bash
MPLBACKEND=Agg .venv/bin/python -m train.convnext.train_convnext_v1 \
    --dataset cifar10 --variant cifar10 --epochs 100 --gpu 0
```

## `stochastic_mode`: `depth` vs `gradient`

ConvNeXt blocks apply a stochastic regularizer to the residual branch. All three
scripts expose `--stochastic-mode {depth,gradient}` (default `depth`), threaded
to the model constructor (`convnext_v1.py` / `convnext_v2.py`), which selects:

- **`depth` → `StochasticDepth`** — per-sample Bernoulli drop of the *entire*
  residual branch with `/keep_prob` rescale (classic drop-path). Strong
  regularizer; behavior-preserving default.
- **`gradient` → `StochasticGradient`** — forward-identity; only the *gradient*
  through the branch is stochastically `stop_gradient`-ed. Weaker regularizer
  (the forward pass is unchanged).

These are distinct regularizers, not two implementations of the same thing.

Both flags below are ConvNeXt-local (not on the shared base parser):

| Flag | Default | Notes |
|------|---------|-------|
| `--stochastic-mode {depth,gradient}` | `depth` | Selects the residual regularizer. |
| `--seed <int>` | `42` | `set_seeds()` at the top of training, before data load + model build. Required for a fair A/B. |

## Comparison driver

`run_stochastic_comparison.py` trains the production trainer once per mode
(serially — single GPU, never parallel), discovers each run's `results/` dir by
snapshot-diffing before/after, and calls `train.common.compare_runs` on the two
`training_log.csv` files to emit `comparison.md` + `loss_curves.png` +
`metric_curves.png` under `results/convnext_stochastic_compare_<model>_<dataset>/`.

```bash
# toy 2-stage cifar10 variant, quick:
MPLBACKEND=Agg .venv/bin/python -m train.convnext.run_stochastic_comparison \
    --model v1 --dataset cifar10 --variant cifar10 --epochs 30 --gpu 0

# bigger 4-stage variant — MUST pass --strides 2 (see "Gotchas"), throttle analyzer:
MPLBACKEND=Agg .venv/bin/python -m train.convnext.run_stochastic_comparison \
    --model v1 --dataset cifar10 --variant tiny --strides 2 \
    --epochs 100 --batch-size 128 --seed 42 --gpu 0 --no-epoch-analyzer
```

Driver flags: `--model {v1,v2}`, `--variant`, `--dataset`, `--epochs`,
`--batch-size`, `--seed`, `--gpu`, `--strides`, `--kernel-size`,
`--no-epoch-analyzer`, `--modes depth gradient`. The driver runs CPU-only and
forwards each child its own GPU via the hard-set child env.

## Findings: `depth` vs `gradient`

Two A/B runs on CIFAR-10, seed 42, identical config per pair (only
`stochastic_mode` differs). `Δ = gradient − depth`.

**Run 1 — `cifar10` variant (toy, 2-stage), 30 epochs, batch 128**

| Metric | depth | gradient | winner |
|--------|------:|---------:|:------:|
| best val_accuracy | **0.5182** | 0.5123 | depth (+0.6 pt) |
| best val_loss | **1.357** | 1.368 | depth |
| best train_accuracy | 0.5466 | 0.5470 | ~tie |

**Run 2 — `tiny` variant (4-stage, ~28M params), `strides=2`, 100 epochs, batch 128**

| Metric | depth | gradient | winner |
|--------|------:|---------:|:------:|
| best val_accuracy | **0.6078** | 0.5952 | depth (+1.26 pt) |
| best val_loss | **1.2615** | 1.289 | depth |
| best train_accuracy | 0.986 | 0.996 | gradient (overfits more) |
| final val_loss | **2.41** | 2.70 | depth |

**Conclusion: prefer `stochastic_mode='depth'`.** `gradient` (forward-identity)
memorizes the training set harder — higher train accuracy, lower train loss,
faster-climbing val_loss — while `depth` (`StochasticDepth`) generalizes better
(higher val accuracy, lower val loss). This is the expected direction:
`StochasticDepth` is the stronger regularizer. The val-accuracy gap widened from
+0.6 pt (toy, 30 ep) to +1.26 pt (tiny, 100 ep), and the train/val divergence
sharpened — the bigger/longer run strengthened the conclusion.

**Caveats.** Absolute val-accuracy is modest (~0.60): this recipe has no strong
augmentation and `strides=2` is aggressive, so the model overfits hard. The
*relative* comparison is valid (identical config, both modes), and the overfit
regime is exactly where a residual regularizer matters. For a publication-grade
claim, run multiple seeds (`--seed`) and/or add augmentation; treat a ~1 pt gap
at a single seed as suggestive, not definitive.

## Gotchas (discovered while running the experiment)

1. **ConvNeXt heads emit logits — compile with `from_logits=True`.** The V1/V2
   classifier head is a bare `Dense(num_classes)` with no softmax. `train_convnext_v1.py`
   originally used `SparseCategoricalCrossentropy(from_logits=False)`, which took
   `log()` of raw logits: init loss ~9.5 (vs `ln(10)=2.30`), saturated gradients,
   val-accuracy pinned at 0.10 (random). Fixed to `from_logits=True` (V2 was
   already correct).
2. **4-stage variants need `--strides 2` on 32×32 inputs.** The stem and the
   inter-stage downsample convs both use `--strides` (default 4). On CIFAR the
   default-4 path collapses spatial dims to 2×2 then runs a 4×4/stride-4 conv →
   `Negative dimension` crash. Only the 2-stage `cifar10` variant tolerates
   `--strides 4`. Use `--strides 2` for `tiny`/`small`/`base`/… (gives the
   standard 32→16→8→4→2 schedule).
3. **The comparison driver runs CPU-only by design.** It imports TF (via
   `compare_runs`); a second GPU context on the trainer's GPU fragments the XLA
   allocator and can `SIGABRT` the trainer (`Check failed: h != kInvalidChunkHandle`)
   even on an otherwise-free GPU. The driver sets `CUDA_VISIBLE_DEVICES=''` for
   itself and hands each child its GPU via the child env. A few harmless
   `cuInit: NO_DEVICE` log lines from the driver are expected.
4. **`--no-epoch-analyzer` for long runs (V1).** The per-epoch WeightWatcher
   `EpochAnalyzerCallback` costs ~80 s/epoch on a 28M model (a 100-epoch × 2-mode
   run would be ~5 h, none of it used by the comparison). `--no-epoch-analyzer`
   disables only the per-epoch analysis (~16.7 s/epoch, ~56 min total); the
   end-of-training `run_model_analysis` still runs.

# bfunet — Bias-Free Denoiser Training

Production training suite for **bias-free image denoisers** built on a shared substrate.
Several bias-free denoiser families are trained here — a **ConvUNeXt** (ConvNeXt U-Net)
denoiser, a plain **U-Net** baseline,
and a flat **BFCNN** (bias-free ResNet, non-U-Net) baseline — all wired through one common
data/curriculum/training module (`common.py`).

The design principle throughout is **bias-free / degree-1 homogeneity**: every layer avoids
additive bias so the network satisfies `f(a·x) = a·f(x)`. This makes the residual `x − f(x)`
interpretable as a scaled score (Miyasawa/Tweedie), and lets a single model generalize across
noise levels it was not explicitly trained on.

---

## Data domain: `[0, 1]` (not zero-centered)

All three trainers normalize with `image / 255.0` — a **strictly-positive** `[0, 1]` domain.
The single choke point is `common.py`'s `DATA_MIN = 0.0` / `DATA_MAX = 1.0`; every clip in the
pipeline is driven from those two constants. **Do not add a second normalizer.**

This is deliberate and load-bearing, not a convention. Degree-1 homogeneity implies `f(0) = 0`,
so on a zero-centered domain a flat mid-grey patch is reproduced **for free** and the network
never has to learn DC-preserving filters. On `[0, 1]` a flat patch of value `c` is `c·1`, and
`f(c·1) = c·f(1)` means reproducing it **requires `f(1) = 1`** — filter weights that **sum to
one**. `common.py`'s DC / sum-to-one probe (logged at build time, next to the homogeneity
probe) reports `‖f(c·1) − c·1‖ / ‖c·1‖`; by homogeneity that number cannot depend on `c`, and
its value *is* `‖f(1) − 1‖`.

> **The domain is a DC choice, not a scale choice.** Peak-to-peak width is `1.0`, so every
> `sigma` default, `sigma_255 = sigma·255`, and `PsnrMetric`/`SsimMetric` `max_val=1.0` are
> exactly correct as written. Rescaling any of them would silently corrupt every reported dB
> number and **nothing would fail**.
>
> A checkpoint trained on a `[-0.5,+0.5]` domain is **invalid** here and cannot be shifted post
> hoc (a bias-free net has no mechanism to subtract a DC offset). Runs stamp
> `data_range: "[0,1]"` into `config.json`; `DenoiserPrior.from_pretrained` **refuses** any
> checkpoint that lacks it. Full rationale: `research/2026_bfunet_unit_domain_migration.md`.

---

## Directory layout

| File | Role |
|------|------|
| `common.py` | **Shared substrate** — data pipeline, noise curriculum, self-iteration pool, callbacks, dashboard, the `train()` orchestrator, the `BFUnetTrainingConfig` base dataclass, and `add_common_arguments()` (the CLI shared by all trainers). Not run directly. |
| `train_convunext_denoiser.py` | **Production trainer** — bias-free ConvNeXt U-Net denoiser. The most actively developed. |
| `train_unet_denoiser.py` | **Baseline trainer** — bias-free plain U-Net denoiser (classic conv/residual blocks). Same infrastructure feature set as ConvUNeXt; the apples-to-apples baseline. |
| `train_bfcnn_denoiser.py` | **Baseline trainer** — flat bias-free ResNet (BFCNN, Mohan et al. ICLR 2020), a stack of residual blocks with **no downsampling / no skips** — the non-U-Net baseline. Trains on the same shared substrate. |
| `eval_psnr_vs_noise.py` | **Standalone tool** — PSNR-vs-noise-level evaluation of any saved `.keras` denoiser, with optional SOTA reference overlay. |
| `eval_per_pixel_uncertainty.py` | **Standalone tool** — per-pixel uncertainty / residual maps for a saved `.keras` denoiser. |
| `variance_probe.py` | **Standalone tool** — ConvUNeXt-only training-stability probe (run-to-run variance across seeds). |
| `FINDINGS.md` | Two bodies of evidence. Sections 1-7: the channel-matching (`--zero-pad-channels` / `--extra-zero-output-channels`) variance experiment. Sections 8-9: eval caveats, the SOTA and SSIM comparison tables and the capability-boundary work, measured on **legacy `[-0.5,+0.5]`-domain checkpoints** (`20260707`, `20260710`) that the provenance gate now refuses; the dB numbers there must be re-measured on a `[0,1]` model and are not results of the current trainers. |

All three trainers are deliberately thin: each supplies only a `build_model()`, a `verify_bias_free()`,
a model-specific `TrainingConfig(BFUnetTrainingConfig)`, and CLI glue. Everything else —
the fit loop, dataset streaming, curriculum, visualization — lives once in `common.py` and is
invoked via `common.train(config, build_model, verify_bias_free, ...)`.

---

## Quick start

Always run with a non-interactive matplotlib backend (headless-safe) and from the repo root:

```bash
# ConvUNeXt base denoiser, full training run
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_convunext_denoiser \
    --variant base --epochs 100 --batch-size 4 --gpu 1

# Plain U-Net baseline denoiser
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_unet_denoiser \
    --variant base --epochs 100 --batch-size 4 --gpu 1

# Fast mechanism check (a few steps, tiny variant) — verifies the pipeline end-to-end
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_convunext_denoiser --smoke
```

Outputs are written under the repo-root `results/` directory, in a run folder named
`<model prefix><variant>_<YYYYMMDD_HHMMSS>` (for example `convunext_denoiser_base_...`) unless
`--experiment-name` is given. What the folder holds and what is refused before it is created
are in "Run-directory contract" and "Refusals" below.

> **Memory note:** the `base` variant at `--patch-size 256` needs `--batch-size 4` on a
> 24 GB RTX 4090. Reduce batch or patch size on smaller GPUs.

### Data

Training data directories are **hardcoded** in `BFUnetTrainingConfig` (not CLI flags):

- **Train:** `COCO/train2017` + `div2k/train`. `dataset_weights` defaults to equal weight per
  directory, so each directory gets an equal share of the `--max-train-files` list and a
  directory with fewer images than its share wraps around (its images repeat) rather than
  letting COCO drown DIV2K.
- **Validation:** `div2k/validation`, capped at `--max-val-files`.
- **Held-out test sets** (end-of-run evaluation only, see `test_eval` below): `kodak24` and
  `cbsd68_src/CBSD68/original`, hard-coded as `TEST_DATASETS` in `common.py`.

To train on other data, edit `train_image_dirs` / `val_image_dirs` in the config, or construct
the `TrainingConfig` programmatically.

**Effective caps.** `--max-train-files`, `--max-val-files`, `--validation-steps` and
`--steps-per-epoch` default to unset on the command line. On the ConvUNeXt trainer an unset
value is not forwarded, so the config default applies: **`--max-train-files` 10000,
`--max-val-files` 500, `--validation-steps` 100 batches**; `--steps-per-epoch` unset resolves
to `max(100, len(train paths) * patches_per_image // batch_size)`, and the resolved number is
written back into `config.json`. A typed `0` for any of the four is refused on the ConvUNeXt
trainer (0 is not "unlimited"). The unet and bfcnn mains still read the two file caps as
`args.max_train_files or 10000` / `args.max_val_files or 500`, so there a `0` silently becomes
the default.

---

## Shared CLI (`add_common_arguments`)

All three trainers expose this flag set (defined once in `common.py`). Model-specific flags are
in the per-trainer sections below.

**Data & schedule**

| Flag | Default | Meaning |
|------|---------|---------|
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 16 | Batch size (use 4 for base@256 on a 4090) |
| `--patch-size` | 256 | Training crop size |
| `--channels` | 3 | 1 (grayscale) or 3 (RGB) |
| `--patches-per-image` | 4 | Random crops drawn per source image |
| `--learning-rate` | 1e-3 | Peak LR (must be > 0) |
| `--weight-decay` | 0.004 | AdamW decoupled weight decay (no separate L2) |
| `--warmup-epochs` | `max(1, round(0.1 * epochs))` | LR warmup length; may equal `--epochs` (warns that the cosine never starts), refused above it |
| `--optimizer-type` | adamw | One of the optimizer builder's names (`adam`, `adamw`, `sgd`, `rmsprop`, `adadelta`, `sgld`, `vsgd`, `gefen`); anything else is refused at config time |
| `--lr-schedule-type` | cosine_decay | The only accepted value; see "Learning rate" for its shape |
| `--gradient-clipping` | 1.0 | Clip-by-norm value (0 = off) |
| `--early-stopping-patience` | -1 | `<= 0` disables early stopping (the default: the curriculum makes `val_loss` non-monotonic) |
| `--seed` | 42 | Seed for Python / NumPy / TF / Keras |
| `--no-clip` / `--no-augment` | off | Skip the `[0,1]` clip of the noisy input / the train-time flips and rot90 |
| `--noise-sigma-min` | 0.0 | Lower bound of the sampled sigma range |
| `--max-train-files` | 10000 effective | Cap on the weighted training path list (see "Effective caps" above) |
| `--max-val-files` | 500 effective | Cap on the validation path list |
| `--steps-per-epoch` | derived | See "Effective caps" above |
| `--validation-steps` | 100 effective | Validation batches per epoch |
| `--mixed-precision` | off | Enable `mixed_float16`. **Slower** than fp32 for base@256/b4 on a 4090 (XLA is disabled by the bilinear-upsample grad). Off by default. |

**Architecture (shared)**

| Flag | Default | Meaning |
|------|---------|---------|
| `--no-gabor-stem` | off (stem ON) | Disable the trainable Gabor warm-start stem |
| `--freeze-gabor-stem` | off (stem TRAINABLE) | Freeze the Gabor stem at its Gabor initialization instead of refining it. Freezing and stem **kind** are independent axes: this freezes whichever stem was built, and the depthwise bank is selected separately by ConvUNeXt's `--gabor-filters-per-channel`. Homogeneity is unaffected either way (it comes from `use_bias=False`). Rejected with `--no-gabor-stem` and on the BFCNN trainer |
| `--gabor-filters` | 32 | Gabor stem OUTPUT channel count (a `Conv2D` `filters`, **not** a per-channel multiplier — that is ConvUNeXt's separate `--gabor-filters-per-channel`, which is mutually exclusive with this flag) |
| `--no-gabor-projection` | off | Drop the 1×1 projection after the Gabor stem (requires `gabor_filters == initial_filters`) |
| `--initial-filters` | variant | Override level-0 width |
| `--filter-multiplier` | 2.0 | Per-level channel growth: `channels[l] = round(initial_filters * m**l)` |
| `--depth` | variant | Number of U-Net levels (≥1 at config time; the plain U-Net model requires ≥3) |
| `--blocks-per-level` | variant | Blocks per level (≥1) |
| `--final-projection-groups` | 1 | Groups for the final 1×1 output projection (`-1` = one group per output channel) |
| `--laplacian-pyramid` | off | Enable the Laplacian-pyramid downsample/skip path |
| `--zero-pad-channels` | off | Parameter-free channel matching instead of 1×1 adjust convs (see `FINDINGS.md`) |
| `--mean-pooling` | off | AveragePooling (linear) instead of MaxPooling — keeps the encoder linear for the Miyasawa interpretation |
| `--expose-bottleneck` | off | Expose the bottleneck latent as a second output |
| `--block-normalization` | batchnorm | `batchnorm` = **variance-only `BiasFreeBatchNorm`** (no `moving_mean`/`beta`, degree-1 homogeneous, pairs with LeakyReLU); `layernorm` = per-input scale-invariant (degree-0). Registered once in `add_common_arguments`. |
| `--block-activation` | leaky_relu | Activation for blocks + stem + deep-supervision heads (final activation stays linear) |
| `--block-activation-alpha` | 0.1 | LeakyReLU negative slope |

> **All three trainers now default to `block_activation=leaky_relu` (alpha `0.1`) and
> `block_normalization=batchnorm`.** For the ConvUNeXt / plain-U-Net / BFCNN denoisers, `batchnorm`
> resolves to the real degree-1-homogeneous `BiasFreeBatchNorm`, and **all three resolve it inside
> the MODEL builder, not here**: ConvUNeXt via its ConvNeXt blocks, and the plain U-Net + BFCNN via
> `layers/conv_blocks/bias_free_conv2d.resolve_denoiser_normalization`, which `create_bfunet_denoiser` /
> `create_bfcnn_denoiser` call themselves. The trainer forwards `--block-normalization` unchanged.
> As a result **all three are degree-1 homogeneous** (`f(a·x)=a·f(x)`), so the residual `x−f(x)` is
> a valid scaled score (Miyasawa/Tweedie) — and so is a denoiser someone builds through the model
> factory without this trainer.

**Noise curriculum** — training sweeps `sigma_max` from a narrow low-noise range up to a wide one:

| Flag | Default | Meaning |
|------|---------|---------|
| `--sigma-max-start` | 0.025 | Curriculum start (low noise) |
| `--sigma-max-end` | 0.25 | Curriculum end (wide noise) |
| `--curriculum-schedule` | linear | `linear` / `cosine` / `exp` |
| `--curriculum-epochs` | = epochs | Epochs over which the curriculum widens |
| `--deep-supervision` | off | **Refused at config time** (`ValueError` before any run directory exists): the model would return several outputs but no multi-scale targets, per-output loss dict or weight scheduler are wired, so `fit` would crash. Accepted by the parser only so the refusal names the flag. |

**Noise model** (default is additive AWGN):

| Flag | Default | Meaning |
|------|---------|---------|
| `--multiplicative-noise` | off | Per-pixel multiplicative noise `y = x·(1+N·σ)` |
| `--composite-noise` | off | Composite `y = x·n + a` (takes precedence over multiplicative) |
| `--composite-additive-ratio` | 0.5 | Composite: additive floor as fraction of `sigma_m` |

**Self-iteration** (train the denoiser to improve over 2–5 sequential passes — **additive noise only**):

| Flag | Default | Meaning |
|------|---------|---------|
| `--self-iterate` | off | Enable self-iterable training via epoch-boundary pool regeneration |
| `--self-iterate-pool-size` | 2048 | Clean-patch RAM pool size (≈1.6 GB at 256²×3) |
| `--self-iterate-regen-freq` | 1 | Regenerate pool inputs every N epochs |
| `--self-iterate-mix-ratio` | 0.5 | Fraction of pool slots holding regenerated pairs |

**WW-PGD spectral regularization**

| Flag | Default | Meaning |
|------|---------|---------|
| `--ww-pgd` | off | Spectral tail-projection at epoch boundaries |
| `--ww-pgd-log-alpha` | off | Log per-layer power-law α trajectory (implies `--ww-pgd`) |

**Run modes, warm start, analysis, output**

| Flag | Default | Meaning |
|------|---------|---------|
| `--smoke` | off | Tiny end-to-end mechanism check (2 epochs, 3 steps, cosine LR). On the ConvUNeXt trainer a flag you type overrides the preset and the run name is timestamped; see "`--smoke`" below |
| `--init-from PATH` | None | Warm-start weights from a `.keras` checkpoint — primary use: self-iterate fine-tuning. Refused when the checkpoint is not stamped `data_range: "[0,1]"`, loads 0 layers, or has any same-name layer of a different shape (see "Refusals") |
| `--dashboard DIR` | None | Rebuild `visualizations/training_dashboard.png` from an existing run dir's `training_log.csv` and `config.json` and exit (no training; the LR panel needs the resolved `steps_per_epoch` that `config.json` holds) |
| `--analyzer` / `--analyzer-freq` | off / 10 | Run ModelAnalyzer every N epochs |
| `--viz-freq` / `--viz-samples` | 5 / 8 | Clean/noisy/denoised grid cadence & columns |
| `--output-dir` | `results` | Output root; a relative path is anchored at the repo root, not the working directory |
| `--experiment-name` | None | Override the run-folder name. A name whose folder already holds a run is refused |
| `--gpu` | None | GPU index (e.g. `--gpu 1`) |

---

## Run-directory contract

Every run writes ONE folder, `results/<experiment_name>/` (repo-root anchored), and never touches
another. Files are written in this order and all of them come from `common.train()`, so the
three trainers share the contract (the differences are in "What unet and bfcnn inherit").

```
results/<experiment_name>/
    config.json                 resolved TrainingConfig, incl. data_range "[0,1]" and the RESOLVED steps_per_epoch
    run.log                     the dl logger for this run (devices, model summary, probes, one line per epoch)
    training_log.csv            one row per epoch (columns below)
    training_history.json       per-epoch lists of every history metric
    best_model.keras            checkpoint of the lowest-val_loss epoch (rewritten on every improvement)
    final_model.keras           the model after the LAST epoch (not the best; see "Best versus final")
    results_summary.json        the run's record (strict JSON, keys below)
    tensorboard/                TensorBoard logs
    visualizations/
        training_dashboard.png  per-epoch curves, redrawn after every epoch (the epoch-0 point is the untrained baseline)
        epoch_NNN_denoise_grid.png   clean / noisy / denoised grid, same images under the 15/25/50 (sigma_255) regimes;
                                     written at epoch 0 (untrained), epoch 1 and every --viz-freq epochs
    epoch_analysis/             only with --analyzer
    ww_pgd_layer_alpha.csv      only with --ww-pgd-log-alpha
    final_model_bottleneck.keras  only with --expose-bottleneck (the full 2-output model)
```

- **`config.json` is written twice.** First when the directory is created (with
  `steps_per_epoch: null` when it is derived), then again once the data pipeline exists, from a
  copy of the config with the number that actually runs. `--dashboard` rebuilds the LR panel from
  that field.
- **`training_log.csv`**: `epoch` is 0-based; `best_epoch` in the summary is 1-based
  (`best_epoch_csv_index` = `best_epoch - 1`). **`lr` is the rate the epoch STARTED at**, the
  rate of its first optimizer step: epoch 1 shows the schedule's value at step 0 (about `1e-8`
  under warmup, not the peak), and the last epoch shows the rate it began at, not the schedule's
  value at its end (`lr_last_step` in the summary is that).
- **Per-epoch line and the progress bar.** `run.log` and the console carry one line per epoch,
  `Epoch N/E - loss X - mae X - psnr_metric X - ssim_metric X - val_loss X - val_mae X -
  val_psnr_metric X - val_ssim_metric X - lr X - time Ns`, built from the true epoch logs (the
  same values as the CSV row, 4 decimals; `time` is the epoch's own clock, which excludes the
  dashboard and grid redraw). The Keras progress bar is kept for live feedback, but its TRAIN
  numbers read low: it averages the already-running-mean metrics a second time
  (`keras/src/utils/progbar.py`), most in a fast-learning first epoch. Its validation numbers agree
  with the CSV. **The CSV and the `run.log` epoch line are the reference**; Keras' own progress
  bars are not written to `run.log`.
- **Best versus final.** `best_model.keras` (lowest `val_loss`) and `final_model.keras` differ
  whenever the last epoch is not the best (`final_is_best` in the summary). `final_model.keras`
  is the real last epoch **also under `--early-stopping-patience`**: the shared callback's
  `restore_best_weights` is switched off after it is built, so the in-memory model is not
  overwritten by the best weights at the end of `fit`. `model_loading_validated` reports whether
  `final_model.keras` reloads and reproduces its predictions (`null` when the check could not run).
- **Validation noise.** `val_*` metrics are measured on freshly drawn noise at a FIXED upper
  sigma (`--sigma-max-end`) while the training sigma follows the curriculum, so `val_loss`
  carries a sampling noise of its own, is not guaranteed identical between two runs with the same
  seed, and the best epoch can flip on it. Read small `val_loss` gaps accordingly.
- **A run that goes non-finite is recorded, not finished.** `TerminateOnNaN` ends the fit; then
  `training_history.json` and a `results_summary.json` with `status: "diverged"` are written, a
  `RuntimeError` is raised (the trainer logs `Training failed`, never `Training completed
  successfully!`), and `final_model.keras` is NOT written. `best_model.keras` and
  `training_log.csv` from before the divergence remain.

### `results_summary.json` keys

Strict JSON (`allow_nan=False`: non-finite values become `null`). Keys on EVERY summary (from
`_summary_head`), finished or diverged:

| Key | Meaning |
|---|---|
| `status` | `"ok"` or `"diverged"` |
| `run_dir`, `experiment_name`, `variant` | Identity of the run |
| `params` | Model parameter count |
| `learning_rate`, `warmup_epochs`, `steps_per_epoch` | Peak rate, warmup epochs and the RESOLVED steps per epoch |
| `lr_first_epoch`, `lr_last_epoch` | The CSV `lr` of the first and last epoch (rate at the START of each) |
| `lr_last_step` | The schedule's rate at the run's very last optimizer step (`steps_per_epoch * epochs - 1`); see "Learning rate" |
| `noise` | `{type, start, end, schedule, sigma_min, curriculum_epochs}` as run |
| `epochs_requested`, `epochs_run`, `stopped_early` | Requested and trained epochs; whether early stopping ended the run (`null` when diverged) |
| `best_epoch` | 1-based epoch of the lowest `val_loss` (`null` when diverged) |
| `init_from` | `{path, loaded, missing_in_source, shape_mismatch}` (counts) or `null` without `--init-from` |
| `gpu_name`, `tf_visible_devices`, `cuda_visible_devices` | The device TensorFlow used and the environment at call time |
| `epoch_times`, `fit_wall_seconds` | Seconds of each epoch as printed on its `run.log` line, and the wall time of `fit`; the difference is time outside that clock (redraws, checkpoint saves) |
| `notes` | Plain-language reading notes for the file |

Added by a finished run (`status: "ok"`):

| Key | Meaning |
|---|---|
| `best_epoch_csv_index`, `final_is_best` | 0-based twin of `best_epoch`; whether the last epoch is the best |
| `best_val_metrics`, `final_val_metrics` | The `val_*` columns of the best epoch and of the last epoch |
| `model_loading_validated` | `final_model.keras` round-trip verdict (`true` / `false` / `null`) |
| `test_eval` | The held-out block below |

Added by a diverged run instead: `non_finite_metrics` (names of the metrics holding a
non-finite or missing value) and `history` (every history list, with `null`s).

**`test_eval`** (ConvUNeXt: `--test-eval` on by default, `--no-test-eval` to skip). After the final
save the run scores `best_model.keras` (reloaded from disk, which also exercises the provenance
gate) and the last-epoch in-memory model on the fixed sets `kodak24` and `cbsd68` at
sigma_255 = 15, 25, 50, using `eval_psnr_vs_noise.evaluate_dataset` with `RandomState(42)` per set
(the script's own default seed, deliberately not `--seed`), so the same crops and noise are used
by every run and `eval_psnr_vs_noise` given the same seed, patch size, sample count and sigmas
reproduces the numbers. The block is `{status, seed, patch_size, num_samples, sigmas_255,
datasets, seconds}`; each dataset is `{status, directory, n_images, sigmas}` and each sigma
(keys `"15"`, `"25"`, `"50"`) is

| Key | Meaning |
|---|---|
| `input_psnr` | The noisy input against the clean crop |
| `best_psnr`, `final_psnr` | `best_model.keras` and the last-epoch model against the clean crop |
| `gain_db` | `best_psnr - input_psnr` (the best checkpoint's gain over the noisy input) |
| `final_gain_db` | `final_psnr - input_psnr` (the last-epoch model's gain) |
| `n_patches` | Crops averaged (`--test-num-samples`, default 100, refused below 1; crops are `--patch-size` wide) |

`status` is `ok` (at least one set scored), `skipped` (with a `reason`: `disabled` under
`--no-test-eval`, or no set directory holds an image; a single missing set is recorded as
`skipped` inside `datasets` with a WARNING and does not stop the other), or `error` (with the
exception in `message`, logged at ERROR). An evaluation failure never fails a run that already
trained and saved.

## Learning rate

The schedule is `learning_rate_schedule_builder` with `type: cosine_decay`,
`decay_steps = steps_per_epoch * epochs`, `warmup_steps = steps_per_epoch * warmup_epochs` and a
fixed `alpha = 0.01` (no flag), so the configured floor is `alpha * --learning-rate` (`1e-5` at
the default peak `1e-3`). **The default schedule does not reach that floor.** The horizon
(`decay_steps`) is the whole run, but the builder feeds the cosine `step - warmup_steps`, so the
decay is cut at `(epochs - warmup_epochs) / epochs` of its length. Measured by calling the builder
at peak `1e-3`:

| epochs / warmup / steps per epoch | rate at the LAST optimizer step (`lr_last_step`) | rate at the start of the last epoch (the CSV's last `lr`) |
|---|---|---|
| 100 / 10 / 400 | 3.42e-5 | 3.93e-5 |
| 8 / 1 / 100 | 4.84e-5 | 1.55e-4 |

The `100 / 10 / 400` last-step value is pinned by
`test_the_default_schedule_ends_at_3_42e_minus_5_not_at_the_1e_minus_5_floor`. The behaviour is
kept on purpose (every earlier bfunet run used exactly this schedule, and changing it silently
would make those runs incomparable); a decay-to-end schedule is **not implemented**. Read
`lr_last_step` from the summary instead of assuming the floor.

## Refusals

A bad invocation is refused **before the run directory exists** wherever the machine can know it
in advance, so it burns no experiment name and writes nothing. The three layers, in order:

1. **Config time** (`TrainingConfig(...)`, pure value rules, no filesystem). Summary:
   counts `>= 1` (`batch_size`, `epochs`, `curriculum_epochs`, `patches_per_image`, `viz_freq`,
   `viz_samples`, `test_num_samples`) and `>= 1` when set for `steps_per_epoch`,
   `validation_steps`, `max_train_files`, `max_val_files`; `learning_rate > 0`,
   `weight_decay >= 0`, `gradient_clipping >= 0`; `warmup_epochs` in `[0, epochs]`; the sigma
   range (`sigma_max_end > noise_sigma_min`, `noise_sigma_min <= sigma_max_start`, `exp` schedule
   needs `sigma_max_start > 0`, schedule name, `composite_additive_ratio > 0`,
   `self_iterate_mix_ratio` in `[0, 1]`); `lr_schedule_type` other than `cosine_decay` and an
   unknown `optimizer_type`; `--deep-supervision`; `--mixed-precision` with
   `--expose-bottleneck`; a nonzero `--symmetry-weight` with `--mixed-precision` or deep
   supervision; `--self-iterate` with non-additive noise or a pool smaller than the batch; the
   Gabor rules (`--no-gabor-projection` needs `gabor_filters == initial_filters`, or
   `channels * N == initial_filters` on the depthwise arm, checked against the RESOLVED
   `initial_filters`; `--freeze-gabor-stem` with `--no-gabor-stem`; `--final-projection-groups`
   must divide both `initial_filters` and `channels`); unknown `--variant`. The two Gabor
   filter-count flags being given together is a parse error.
2. **Preflight in `train()`**, still before anything is written: a `--init-from` checkpoint
   that is not stamped `data_range: "[0,1]"` in its sibling `config.json` (a legacy
   `[-0.5,+0.5]` checkpoint of the same architecture would otherwise load and train from
   wrong-domain weights with no error); a train or validation directory list that yields no
   image (missing or empty directories, "Nothing was written"); and **a reused experiment
   name**: if the resolved folder already holds any of `results_summary.json`, `config.json`,
   `best_model.keras`, `run.log` or `training_log.csv`, `FileExistsError` names the files.
   Nothing is ever overwritten, merged or deleted.
3. **After the model is built** (the directory, `config.json` and `run.log` exist, so the name is
   used up and `run.log` records why): `--init-from` loading 0 layers, and `--init-from`
   with any layer of the same name but a different shape (`shape_mismatch`, the layers are
   named: another `--variant` or width flag). A layer absent from the checkpoint
   (`missing_in_source`) only warns, with a count and the names, and stays at random init; both
   counts are recorded in `init_from` in the summary.

## `--smoke`

`--smoke` is a tiny end-to-end mechanism check, not a training recipe. On the **ConvUNeXt trainer**
it supplies a preset only for the flags you did NOT type: an explicit flag always wins, **even
when its value equals the parser default** (`--smoke --epochs 3` runs 3 epochs, `--smoke
--batch-size 2` is honoured as typed). The preset (`SMOKE_PRESET`): `--variant tiny`, `--epochs 2`,
`--batch-size 2`, `--patch-size 64`, `--patches-per-image 2`, `--max-train-files 8`,
`--max-val-files 8`, `--steps-per-epoch 3`, `--validation-steps 2`, `--warmup-epochs 0`,
`--viz-freq 1`, `--gabor-filters 8`, `--learning-rate 1e-3`, `--sigma-max-start 0.025`,
`--sigma-max-end 0.25`, `--curriculum-schedule linear`, `--self-iterate-pool-size 32`,
`--self-iterate-regen-freq 1`; the curriculum length follows `--epochs`, and the schedule is the
same cosine as a full run (with no warmup). The run name is
`convunext_denoiser_smoke_<YYYYMMDD_HHMMSS>` (timestamped, so smoke runs never collide) unless
`--experiment-name` is typed. `--smoke` does not switch the held-out test evaluation off; add
`--no-test-eval` to skip it. `--dashboard` and the config-time refusals behave as without it.

## What unet and bfcnn inherit

`train_unet_denoiser.py` and `train_bfcnn_denoiser.py` call the same `common.train()` and build
their configs on `BFUnetTrainingConfig`, so they inherit, with no flag to opt out:

- the config-time validation of the base config (list above; the plain U-Net adds `--depth >= 3`),
  the preflight, the reused-name refusal and the repo-root anchoring of `--output-dir`;
- `run.log`, the per-epoch line, the start-of-epoch `lr`, the resolved `config.json`, the
  honest `final_model.keras`, the divergence record, the `init_from` checks and
  `results_summary.json`;
- **the held-out test evaluation, on by default with no switch**: neither parser has
  `--test-eval` / `--test-num-samples`, so both always run it with `test_num_samples = 100` at
  their `--patch-size`. **Only the ConvUNeXt parser has `--test-eval`, `--no-test-eval` and
  `--test-num-samples`.**

They do **not** get the ConvUNeXt `main()` changes: their `main()` functions keep the older
two-branch smoke, which fixes epochs, batch size, patch size, steps, learning rate, sigma range and
curriculum regardless of the flags you type and uses the FIXED run names `unet_denoiser_smoke` /
`bfcnn_denoiser_smoke` (so a second `--smoke` run of either trainer is refused as a reused name
unless you pass `--experiment-name`), and they still turn a typed `--max-train-files 0` /
`--max-val-files 0` into the default (`or 10000` / `or 500`).

---

## ConvUNeXt trainer (`train_convunext_denoiser.py`)

Trains `create_convunext_denoiser` — a bias-free ConvNeXt U-Net. Variants: `tiny`, `small`,
`base` (default), `large`, `xlarge`. Model-specific flags beyond the shared set:

| Flag | Default | Meaning |
|------|---------|---------|
| `--variant` | base | ConvUNeXt size preset |
| `--test-eval` / `--no-test-eval` | on | End-of-run held-out evaluation of `best_model.keras` and the last-epoch model on Kodak24 and CBSD68 crops, recorded as `test_eval` in `results_summary.json` (see "`results_summary.json` keys"). **ConvUNeXt parser only.** |
| `--test-num-samples` | 100 | Crops per held-out test set (the `eval_psnr_vs_noise` default); refused below 1. **ConvUNeXt parser only.** |
| `--convnext-version` | v1 | `v1` = strict bias-free; `v2` adds a trainable GRN β (mildly breaks strict homogeneity) |
| `--gabor-filters-per-channel` | None (off) | Build the Gabor stem as a **depthwise** bank (`depth_multiplier = N`) applied to each input channel independently instead of the default cross-channel `Conv2D`. Emits `channels·N` responses, so it is **mutually exclusive with `--gabor-filters`** (a `Conv2D` output-channel count — passing both is a parse error). Still trainable and bias-free, so `--freeze-gabor-stem` and degree-1 homogeneity are unaffected. Under `--no-gabor-projection` the width rule becomes `channels·N == initial_filters`. ConvUNeXt only |
| `--dropout` | 0.0 | MLP dropout inside the inverted-bottleneck blocks |
| `--depthwise-initializer` | None | Opt-in depthwise-kernel init (`orthonormal` → Orthogonal(gain=1)) |
| `--depthwise-l2` | None | Opt-in L2 on depthwise kernels |
| `--extra-zero-output-channels` | off | Grow output channels with zero-init channels at decoder level 0 (bias-free) |

```bash
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_convunext_denoiser \
    --variant base --convnext-version v1 --block-normalization batchnorm \
    --epochs 100 --batch-size 4 --gpu 1
```

> A few tests import re-exported names from this module path; treat its public names as a
> stable API surface.

---

## U-Net baseline trainer (`train_unet_denoiser.py`)

Trains `create_bfunet_denoiser` — a classic bias-free U-Net with plain `BiasFreeConv2D` /
`BiasFreeResidualBlock` blocks (no ConvNeXt inverted bottleneck). It is the **apples-to-apples
baseline**: identical training machinery and the **same infrastructure feature set** as the
ConvUNeXt trainer (gabor stem, laplacian pyramid, `--zero-pad-channels`, pooling-type,
`--expose-bottleneck`, block-normalization choice, `--final-projection-groups`, dropout — all
via the shared CLI above), so only the block internals differ. Variants: `tiny`, `small`,
`base` (default), `large`, `xlarge`. The model requires `--depth ≥ 3`.

Model-specific flags (beyond the shared set):

| Flag | Default | Meaning |
|------|---------|---------|
| `--variant` | base | U-Net size preset |
| `--no-residual-blocks` | off (residual ON) | Use plain `BiasFreeConv2D` blocks instead of residual blocks |
| `--kernel-size` | 3 | Kernel for all non-stem conv blocks |
| `--initial-kernel-size` | 5 | Kernel for the level-0 first conv |
| `--dropout` | 0.0 | Dropout inside the conv blocks (after activation) |

```bash
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_unet_denoiser \
    --variant base --block-normalization batchnorm \
    --epochs 100 --batch-size 4 --gpu 1
```

---

## BFCNN trainer (`train_bfcnn_denoiser.py`)

Trains `create_bfcnn_variant` / `create_bfcnn_denoiser` — a **flat bias-free ResNet** (BFCNN,
the original Mohan et al. ICLR 2020 architecture): a stem conv, a stack of bias-free residual
blocks, and a final projection, with **no downsampling and no skip connections**. It is the
non-U-Net baseline, trained on the same shared substrate (curriculum, self-iterate, WW-PGD,
dashboard). The U-Net-only shared flags (`--depth`, `--blocks-per-level`, `--laplacian-pyramid`,
etc.) are ignored by the flat-ResNet factory. Variants: `tiny` (default), `small`, `base`,
`large`, `xlarge`, `custom`.

Model-specific flags (beyond the shared set):

| Flag | Default | Meaning |
|------|---------|---------|
| `--variant` | tiny | BFCNN size preset; `custom` uses the knobs below |
| `--num-blocks` | 8 | Number of residual blocks (`custom` only) |
| `--filters` | 64 | Residual-block filter width (`custom` only) |
| `--initial-kernel-size` | 5 | Kernel size for the stem conv (`custom` only) |
| `--kernel-size` | 3 | Kernel size for the residual blocks (`custom` only) |

> **BFCNN has no `--activation` flag.** It uses the shared `--block-activation` /
> `--block-activation-alpha` knobs (default `leaky_relu` slope `0.1`), and its residual blocks run
> the real variance-only `BiasFreeBatchNorm` (the BFCNN model builder itself resolves
> `normalization_type='batchnorm'` to `'bias_free_batchnorm'`), so every BFCNN is degree-1
> homogeneous.

```bash
# Full BFCNN base training run
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_bfcnn_denoiser \
    --variant base --epochs 100 --batch-size 16 --gpu 1

# Fast end-to-end mechanism check (fixed run name: pass --experiment-name for a second run)
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_bfcnn_denoiser --smoke
```

---

## Standalone tools

### `eval_psnr_vs_noise.py` — PSNR vs noise

Evaluates one or more saved `.keras` denoisers over a paired-noise sweep (the same clean patches
are corrupted at each σ), plots mean PSNR + confidence bands per dataset, and can overlay a SOTA
reference (DnCNN / best-of-{DRUNet, SwinIR, Restormer, SCUNet}). Works on any bias-free denoiser
checkpoint.

```bash
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.eval_psnr_vs_noise \
    --model convunext=results/convunext_denoiser_base_.../best_model.keras \
    --dataset kodak=/path/to/kodak \
    --sigmas-255 5 10 15 25 35 50 65 --num-samples 100 --gpu 1
```

`--model` and `--dataset` take `NAME=PATH` / `NAME=DIR` (repeatable to overlay several). Other
flags: `--patch-size`, `--channels`, `--batch-size`, `--full-image` (SOTA reflect-pad protocol),
`--size-multiple`, `--no-clip`, `--confidence`, `--seed`, `--output-dir`, `--experiment-name`.

> The tool imports the ConvUNeXt model module to register its custom layers for deserialization.
> A checkpoint from any other architecture may require importing that model module first so its
> custom objects are registered.

### `variance_probe.py` — training-stability probe

ConvUNeXt-only diagnostic that trains the real denoiser across multiple seeds at a fixed noise σ
and compares a toggle ON vs OFF (`--zero-pad-channels` or `--extra-zero-output-channels`),
measuring run-to-run variance, loss-trajectory roughness, and gradient-norm stability. Reuses the
real COCO+DIV2K pipeline. See `FINDINGS.md` for the conclusions.

```bash
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.variance_probe \
    --compare zero_pad_channels --variant small --seeds 5 --steps 1500 --gpu 1
```

---

## Constraints & gotchas

- **Additive-only self-iteration.** `--self-iterate` is rejected at parse time with
  `--multiplicative-noise` / `--composite-noise`: the Miyasawa residual-as-score identity (and the
  clean-image fixed point that makes 2–5 passes non-decreasing) holds for additive noise only.
- **Mixed precision is usually slower here** — leave it off unless you measure a win.
- **Outputs go to repo-root `results/`.** Do not point `--output-dir` inside `src/`.
- **A run folder is never reused.** A second run under an existing `--experiment-name` is
  refused; choose a new name (or omit it for a timestamped one). Nothing under `results/` is
  overwritten or deleted by the trainers.
- **Read `results_summary.json` and `training_log.csv`, not the progress bar,** for train
  metrics (see "Run-directory contract").
- **Always set `MPLBACKEND=Agg`** to avoid X11 crashes on headless/remote systems.

See `FINDINGS.md` sections 1-7 for the empirical channel-matching (`--zero-pad-channels` /
`--extra-zero-output-channels`) variance investigation; its sections 8-9 are measured on legacy
`[-0.5,+0.5]` checkpoints (see the directory table).

---

## Tests

Scope pytest to this trainer and run it on the second GPU; never the full suite.

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_train/test_bfunet -q
# every TrainingConfig field is read by its trainer (covers both bfunet configs)
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_train/test_config_fields_are_live.py -q
```

| File (under `tests/test_train/test_bfunet/`) | Guards |
|---|---|
| `test_cli_contract.py` | One row per flag the ConvUNeXt parser declares, driven through `parse_arguments_with_explicit` and `config_from_args` (no training, no GPU, no directory created); the same table again behind `--smoke` |
| `test_smoke_and_explicit_flags.py` | `--smoke` preset versus typed flags, the unique smoke name, the `--max-*-files` / `--validation-steps` semantics, `--help` |
| `test_config_validation.py` | The config-time refusal table with boundary twins |
| `test_train_bfunet_run.py` | Real tiny end-to-end runs: run-directory contract, refusal order, `run.log`, start-of-epoch `lr`, the cosine last-step pin, final versus best model, divergence record, `init_from` checks, `results_summary.json` and `test_eval` |
| `test_provenance_gate.py` | Legacy-domain checkpoints are refused by the checkpoint-load paths (the eval tools and `--init-from`) |
| `test_unet_denoiser.py`, `test_bfcnn_denoiser.py` | The two baseline trainers: config, `build_model` wiring, bias-free check, CLI parsing (construction only, CPU) |
| `test_the_*_gabor_stem_*.py`, `test_convunext_*.py` | Gabor stem flags and width rule, self-iterate, supporting fixes |

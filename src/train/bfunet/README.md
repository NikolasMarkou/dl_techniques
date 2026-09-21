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
| `--deterministic-data` | off | ConvUNeXt trainer only. Two runs of one `--seed` draw the same crops, flips, noise and validation batches (ordered decode, sequential random maps); GPU kernel non-determinism remains. Measured input cost 19 to 30 percent fewer batches per second on the CPU ("Loop 3"). With `--self-iterate` it only affects the validation stream |
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
| `--analyzer` / `--analyzer-freq` | off / 10 | Run ModelAnalyzer every N epochs into `epoch_analysis/` (a per-epoch diagnostic, separate from the end-of-run `model_analysis/` below) |
| `--model-analysis` / `--no-model-analysis` | on (ConvUNeXt) | After training, run the weights + spectral analyzer on the last-epoch model into `model_analysis/` and record its status under `analyzer` in `results_summary.json`; `--no-model-analysis` records a skip. **ConvUNeXt parser only**: unet and bfcnn have no flag and never run it (their `analyzer` block is the skipped one) |
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
    run.log                     the dl logger for this run (devices, one line naming model_summary.txt, probes, one line per epoch,
                                the analyzer status); the Keras layer table is NOT in it
    model_summary.txt           the Keras layer table (model.summary()), written once after the model is built
    training_log.csv            one row per epoch (columns below)
    training_history.json       per-epoch lists of every history metric
    best_model.keras            checkpoint of the lowest-val_loss epoch (rewritten on every improvement)
    final_model.keras           the model after the LAST epoch (not the best; see "Best versus final")
    results_summary.json        the run's record (strict JSON, keys below)
    tensorboard/                TensorBoard logs
    visualizations/
        training_dashboard.png  per-epoch curves (the epoch-0 point is the untrained baseline); ConvUNeXt: redrawn after epoch 1 and every `max(1, epochs // 20)`-th epoch and once at the end of `fit` if the last epoch was off that cadence (every epoch up to 39 epochs), title `<experiment_name> (seed N) - epoch E`; unet and bfcnn: after every epoch
        epoch_NNN_denoise_grid.png   clean / noisy / denoised grid, same images under the 15/25/50 (sigma_255) regimes;
                                     written at epoch 0 (untrained), epoch 1 and every --viz-freq epochs
    model_analysis/             end-of-run weights + spectral ModelAnalyzer output of the LAST epoch's weights
                                (analysis_results.json plus four PNGs); ConvUNeXt trainer only, on by default,
                                absent with --no-model-analysis
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
| `train_image_dirs`, `input_shape`, `n_train_files`, `n_val_files` | The train image directories, the training patch shape `[patch, patch, channels]`, and the lengths of the train and val image-path worklists (the `Sourced N train / M val image paths` line of `run.log`; `n_train_files` is `--max-train-files`, drawn with replacement when the directories hold fewer; not patch counts). Named for what they are: the ConvNeXt reference's `dataset` is a dataset name and its `n_train` / `n_val` are sample counts, so this summary does not reuse those names (iteration 3; before it the keys were `dataset`, `n_train`, `n_val`). All three trainers |
| `image_range` | `[0.0, 1.0]`, the range of the CLEAN images: the training target and the PSNR reference (the denoiser domain since the migration off [-0.5, +0.5]). It is NOT the range of the noisy model input: `--no-clip` leaves that unclipped (noisy values below 0 and above 1), so `config.json`'s `clip_noise` together with this key determines it. It is not `config.json`'s `data_range` either, which is the string `"[0,1]"` provenance stamp the checkpoint gate reads; the summary carries no `data_range` key, so one run directory has no same-name key of two types. Named `data_range` in the run directories of iteration 3 (`*_l3`), renamed in iteration 4 because the old key claimed a domain the noisy input lacks under `--no-clip`. The segmentation summary states it too (the image after `/255`). All three trainers |
| `optimizer`, `lr_schedule`, `weight_decay`, `gradient_clip_norm`, `batch_size`, `seed`, `monitor` | The configuration the run used (`--optimizer-type`, `--lr-schedule-type`, `--weight-decay`, `--gradient-clipping`, `--batch-size`, `--seed`) and the monitored metric (`val_loss`). All three trainers |
| `initial_loss_sanity_eval` | `{loss, steps, split: "val", before_fit: true}`: the untrained model's validation loss (the epoch-0 baseline the dashboard starts from, carried out of `DenoisingVisualizationCallback.baseline_val_loss`; `null` if that evaluation failed) over `steps` = `validation_steps` batches of fresh fixed-sigma noise (`steps`, not `n_samples` as in the segmentation summary and the ConvNeXt reference: the validation stream is `.repeat()`ed, so a sample count is not what was evaluated; a deliberate difference). All three trainers |
| `model_family`, `convnext_version`, `depth`, `blocks_per_level`, `dims`, `kernel_size`, `drop_path_rate`, `dropout_rate` | ConvUNeXt only (`train_convunext_denoiser.architecture_of`, resolved as `build_model` builds; a test compares them with the layers of the saved model). `dims` are the channel counts of the encoder levels and the bottleneck; `convnext_version` is the trainer's `v1` default, not the variant row's `v2` |
| `learning_rate`, `warmup_epochs`, `steps_per_epoch` | Peak rate, warmup epochs and the RESOLVED steps per epoch |
| `lr_first_epoch`, `lr_last_epoch` | The CSV `lr` of the first and last epoch (rate at the START of each) |
| `lr_last_step` | The schedule's rate at the run's very last optimizer step (`steps_per_epoch * epochs - 1`); see "Learning rate" |
| `noise` | `{type, start, end, schedule, sigma_min, curriculum_epochs}` as run |
| `epochs_requested`, `epochs_run`, `stopped_early` | Requested and trained epochs; whether early stopping ended the run (`null` when diverged) |
| `best_epoch` | 1-based epoch of the lowest `val_loss` (`null` when diverged) |
| `init_from` | `{path, loaded, missing_in_source, shape_mismatch}` (counts) or `null` without `--init-from` |
| `gpu_name`, `tf_visible_devices`, `cuda_visible_devices` | The device TensorFlow used and the environment at call time |
| `epoch_times`, `fit_wall_seconds` | Seconds of each epoch as printed on its `run.log` line, and the wall time of `fit`; the difference is time outside that clock: the epoch-0 grid and untrained baseline evaluation, the dashboard and grid redraws after each epoch, checkpoint saves. Run `convunext_audit_denoise_l1` (tiny, 64 px, 2 epochs): `fit_wall_seconds` 97.18 minus the sum of `epoch_times` 66.76 is 30.42 s |
| `notes` | Plain-language reading notes for the file |

Added by a finished run (`status: "ok"`):

| Key | Meaning |
|---|---|
| `best_epoch_csv_index`, `final_is_best` | 0-based twin of `best_epoch`; whether the last epoch is the best |
| `best_val_metrics`, `final_val_metrics` | The `val_*` columns of the best epoch and of the last epoch |
| `model_loading_validated` | `final_model.keras` round-trip verdict (`true` / `false` / `null`) |
| `test_eval` | The held-out block below |
| `analyzer` | The end-of-run analysis, read back from `model_analysis/analysis_results.json` (not taken from the analyzer's return value): `{status, analyzers, error, path, seconds}` (plus `removed`, see "End-of-run artifacts") with `status` `ok`, `partial`, `missing`, `unreadable` or `error`; `analyzers` lists which of `weights` / `spectral` wrote results. Under `--no-model-analysis`, and always for unet and bfcnn, the skipped block (`status: "skipped"`). The smoke run `convunext_denoiser_smoke_20260920_055542` reads `status: "ok"`, `analyzers: ["weights", "spectral"]`, `seconds` 9.5 |
| `visualizations` | `{files, failed, seconds}`: the names present in `visualizations/` (dashboard and `epoch_NNN_denoise_grid.png`), the list of names of the renders that raised (a failed render does not fail the run; its error text is the `Visualization <name> failed: <error>` WARNING in `run.log`), and the wall seconds spent rendering the dashboard and grids. `run.log` gets one `Visualizations: N written (files), M failed, dashboard and grids took Ss` line at the end of `fit` |

Not in the denoiser summary on purpose, each with its measurement: `initial_loss_ratio` and `init_scale_warning` (defined against a classifier's ln(C); the denoiser analogue is the identity loss E[sigma^2] = 0.25^2/3 = 0.0208, and the untrained baseline 0.5693 is 27 times that on a healthy run, so the segmentation trainer's 10x rule would flag every run); `best_checkpoint_load_error` and `best_checkpoint_max_abs_diff` (they would compare the reloaded checkpoint's validation loss with the recorded one, but the validation noise is drawn afresh: the untrained baseline read 0.5693, 0.5672 and 0.5717 in three same-seed runs, so the difference would be noise); `n_test` (`test_eval` carries its own per-set counts); the classification-only keys.

Added by a diverged run instead: `non_finite_metrics` (names of the metrics holding a
non-finite or missing value) and `history` (every history list, with `null`s).

**`test_eval`** (ConvUNeXt: `--test-eval` on by default, `--no-test-eval` to skip). After the final
save the run scores `best_model.keras` (reloaded from disk, which also exercises the provenance
gate) and the last-epoch in-memory model on the fixed sets `kodak24` and `cbsd68` at
sigma_255 = 15, 25, 50, using `eval_psnr_vs_noise.evaluate_dataset` with `RandomState(42)` per set
(the script's own default seed, deliberately not `--seed`), so the same crops and noise are used
by every run and `eval_psnr_vs_noise` given the same seed, patch size, sample count and sigmas
reproduces the numbers. The block is `{status, seed, patch_size, num_samples, sigmas_255,
final_reused_best, datasets, seconds}`; when the last epoch is the best (`final_is_best`) the two models are the same weights, so the sets are scored ONCE and `final_reused_best` is `true` (the `final_*` cells copy the best ones); otherwise the last-epoch model is scored separately and it is `false`; each dataset is `{status, directory, n_images, sigmas}` and each sigma
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
`--no-test-eval` to skip it. `--dashboard` and the config-time refusals behave as without it. The end-of-run
`model_analysis/` is not switched off by `--smoke` (add `--no-model-analysis`).

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
`base` (default), `large`, `xlarge`. The same model's biased segmentation arm is trained by the
sibling package `src/train/convunext/` (`python -m train.convunext.train_convunext_segmentation`,
see its `README.md`); this denoiser stays under `bfunet/` because it shares `common.py` with the
unet and bfcnn denoisers. Model-specific flags beyond the shared set:

| Flag | Default | Meaning |
|------|---------|---------|
| `--variant` | base | ConvUNeXt size preset |
| `--test-eval` / `--no-test-eval` | on | End-of-run held-out evaluation of `best_model.keras` and the last-epoch model on Kodak24 and CBSD68 crops, recorded as `test_eval` in `results_summary.json` (see "`results_summary.json` keys"). **ConvUNeXt parser only.** |
| `--test-num-samples` | 100 | Crops per held-out test set (the `eval_psnr_vs_noise` default); refused below 1. **ConvUNeXt parser only.** |
| `--model-analysis` / `--no-model-analysis` | on | End-of-run weights + spectral analysis into `model_analysis/`, status under `analyzer` in `results_summary.json` (see "End-of-run artifacts"). The shared base config leaves it off; this trainer turns it on. **ConvUNeXt parser only.** |
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

### End-of-run artifacts and the run log

- **`model_analysis/`.** After the final save (and after `test_eval`) the run analyzes the
  in-memory last-epoch model with the weights and spectral analyzers only and writes
  `model_analysis/analysis_results.json` plus `spectral_summary.png`,
  `spectral_funnel_diagram.png` and `weight_learning_journey.png`. The library also draws a
  `summary_dashboard.png`, which is NOT kept: it is EMPTY for a data-free analysis (its panels read "No ... data
  available"), so `run_data_free_analysis` deletes it after an `ok` read-back and records
  `analyzer.removed: ["summary_dashboard.png"]`. The analysis reads weights, not data; the
  trainer hands it the fixed visualization batch only to satisfy the call. The `analyzer` block of
  `results_summary.json` is what the analyzer actually left on disk, so an analysis that
  raised or wrote nothing is `error` or `missing`, and the finished run stays `status: "ok"`.
  Calibration, information flow and training dynamics are not run.
- **`model_summary.txt` and `run.log`.** The Keras layer table goes to `model_summary.txt`
  and `run.log` carries one line naming it. In the audit run `convunext_audit_denoise_l1`
  `run.log` was 231 lines (`wc -l`), most of them the layer table; in the smoke run
  `convunext_denoiser_smoke_20260920_055542` `run.log` is 87 lines and `model_summary.txt` 151.
- **`visualizations` block.** The dashboard and grid callbacks record the files they wrote, the
  renders that raised and the seconds spent, and the summary carries them as `visualizations`.
- **Artifact set against the other two run directories.** `model_summary.txt` and `tensorboard/`
  exist here and not in `results/convnext_ref_l2` (the ConvNeXt reference) or in a segmentation
  run. That is deliberate: the reference has neither, so the segmentation trainer follows it, and
  adding either there would move it away from the reference; a segmentation run keeps only the
  parameter total in `run.log`.
- **Seeded grids.** The fixed visualization batch is cropped from the validation images with a
  stateless random crop keyed by `--seed`, and the additive grid noise is a stateless draw keyed
  by `--seed` and the regime, so the "Noisy" rows are the same in every epoch and in two runs of
  one seed and consecutive `epoch_NNN_denoise_grid.png` files compare image by image.
  Multiplicative and composite grid noise is still unseeded.
- **Clipped pass-1 label.** The pass-1 PSNR in a grid row label scores the output clipped to
  `[0, 1]`, like passes 2 and 3 and the `Multi-pass PSNR` line in `run.log`. It used to score the
  raw output, which read 2.1 dB against a logged 6.45 dB for the untrained model in the
  `convunext_audit_denoise_l1` epoch-0 grid.
- **Best-epoch marker.** The two MSE panels (linear and log) and the PSNR panel of `training_dashboard.png` carry a dashed green
  line, a star on the validation point and a `best epoch N (val_loss)` legend entry at the epoch
  `best_model.keras` holds (`--dashboard` draws it too). It is a separate mark from the lighter
  shaded band and dashed line that end the noise-curriculum ramp. Epoch axes carry integer ticks.
- **Importing the trainer opens no TensorFlow context.** `losses/jacobian_symmetry.py` used to
  build `_NORM_EPS` with `tf.constant` at import, which created the GPU device before
  `setup_gpu` ran and made `setup_gpu` log an ERROR ("Physical devices cannot be modified after
  being initialized") on every run although the GPU worked. `_NORM_EPS` is a Python float now;
  `test_importing_the_bfunet_trainers_opens_no_tensorflow_context` guards the trainers.

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

## Measured runs (audit iteration 1, tiny variant, 8 epochs, GPU 1)

Recipe: `--variant tiny --patch-size 64 --epochs 8 --steps-per-epoch 100 --validation-steps 20 --max-train-files 400 --max-val-files 50 --batch-size 16 --patches-per-image 4 --viz-freq 2`, seed 0 for run 1. Run directories: `results/convunext_iter1_*`. Every number below is copied from a `results_summary.json` or from the paired `eval_psnr_vs_noise` output `results/convunext_iter1_paired_eval/psnr_vs_noise.csv`, and was recomputed from the raw CSV by an independent pass.

| run | epochs run | best epoch (1-based) | fit wall (s) | val_loss at best | val_psnr at best (dB) |
|---|---|---|---|---|---|
| `convunext_iter1_run1` (8 epochs) | 8 | 8 | 202.0 | 0.002396 | 27.74 |
| `convunext_iter1_es` (4 epochs, patience 1, never triggered) | 4 | 4 | 133.4 | 0.003471 | 25.89 |
| `convunext_iter1_smoke` (`--smoke`, 6 steps) | 2 | 2 | 75.5 | 0.023512 | 16.42 |

Held-out test PSNR, run 1 (best equals final, 100 patches of 64 px, seed 42, Kodak24 and CBSD68): Kodak24 30.11 / 28.35 / 24.45 dB and CBSD68 29.60 / 27.96 / 24.18 dB at sigma 15 / 25 / 50, against noisy-input 24.74 / 20.42 / 14.88 dB and 24.80 / 20.51 / 14.97 dB (clipped). The trainer's `test_eval` equals the paired `eval_psnr_vs_noise` invocation to 2.1e-5 dB.

Noise floor at this recipe (best equals final in all five runs). Seeds 0, 1, 2 give a range of 0.12 to 0.23 dB (sd 0.06 to 0.12) per dataset and sigma. Same-seed reruns of seeds 0 and 1 differ by 0.002 to 0.068 dB. A difference between two settings is attributable only above about 0.25 dB at this size (n = 3 seeds, so this is a rough floor, not an interval).

Timing: epoch 1 is about 60 s, later epochs 10.8 to 12.5 s (loop 2 measured what epoch 1 holds: see "Loop 2" below, most of it is not XLA); the baseline validation and the per-epoch figure callbacks account for 61 s of the 202 s fit wall time and are not in `epoch_times`.

Reading the curves: the training loss and PSNR get worse from epoch 5 while the validation metrics keep improving. That is the noise curriculum (training sigma_max rises from 0.025 to 0.25), not a defect: the noise floor rises 7.8x from epoch 3 to 8 while the training loss rises 1.27x. Validation is not reproducible between identical-seed runs (epoch-1 `val_loss` 0.00725 against 0.00690) because the tf.data validation path is stateful and parallel (`--deterministic-data` removes that, see "Loop 3"); the GPU kernels account for at most 0.5 percent of that (measured by a probe that evaluated identical weights repeatedly under the shipped pipeline and under a sequential one).

### Loop 2 (tiny, 64 px, 2 epochs of 100 steps, one seed; and a 40-epoch cadence check)

Command of the four audit runs (only the name differs), GPU 1: `--variant tiny --patch-size 64 --epochs 2 --steps-per-epoch 100 --validation-steps 20 --viz-freq 1 --max-train-files 400 --max-val-files 50 --batch-size 16 --seed 42`, defaults elsewhere (re-derived in iteration 3 from `results/convunext_audit_denoise_l2/config.json`: every field that differs from the defaults is a flag here, apart from the two derived from `--epochs`). Every number here was read from the named run's `results_summary.json` or `run.log` when this section was written; the directories are under `results/` (untracked). Audit note: `plans/plan-2026-09-19T224205-49c8bf80/findings/audit-loop-2.md`.

| Run | Code | `val_psnr_metric` at best epoch (dB) | Kodak24 gain at sigma 15 (dB) | `fit_wall_seconds` |
|---|---|---|---|---|
| `convunext_audit_denoise_l1` | loop 1 (before iteration 1's fixes) | 24.450 | 1.139 | 97.18 |
| `convunext_audit_denoise_l2` | `1d9a9e1df` | 23.912 | 0.398 | 106.32 |
| `convunext_audit_denoise_l2_fix` | `e4ad1b79f` | 24.312 | 1.228 | 107.12 |
| `convunext_audit_denoise_l2_fix_rep1` | `e4ad1b79f` | 24.465 | 0.958 | 109.56 |

**A first same-command spread, and it is large.** The three runs `_l2`, `_fix`, `_fix_rep1` differ in final validation PSNR by 0.553 dB (23.912 to 24.465) and in Kodak24 test gain by 0.829 dB at sigma 15, 0.617 at 25 and 0.317 at 50 (CBSD68: 0.701, 0.533, 0.276). The changes between `_l2` and the two fix runs are summary keys, the dashboard cadence and title and one log line; none touches the training or the loss. The two runs of the SAME code (`_fix` and `_fix_rep1`) differ by 0.153 dB in validation PSNR and by 0.270 dB in Kodak24 gain at sigma 15. That pair (n = 2) understates the spread: its difference is 28 percent of the three-run range, because the pair does not contain the lowest run (`_l2`). Loop 3 measured the cause: the input pipeline is not reproducible from `--seed` (see "Loop 3"). A 2-epoch difference between two denoiser settings below roughly 0.5 dB is not attributable; the floor of three runs of one code state is in "Loop 3" (0.4427 dB).

**What epoch 1 holds.** In `convunext_audit_denoise_l2` epoch 1 takes 63.9 s (`epoch_times`) and later epochs 10.9 s. Loop 2 read the TensorFlow line `Filling up shuffle buffer (this may take a while): 1729 of 2048`, logged 37.4 s after epoch 1 began in that run, as the cause and attributed it to the 2048-patch `--patch-shuffle-buffer`. Loop 3 tested that reading and it does not hold as a general statement. `--patch-shuffle-buffer 512` gives an epoch 1 of 57.06 s (`convunext_audit_denoise_l3_buf512`) against 56.93, 59.16 and 59.83 s at the default 2048 (`convunext_audit_denoise_l3_rep1`, `convunext_audit_denoise_l3`, `convunext_audit_denoise_l3_rep2`), so a four times smaller buffer removed at most 2 s. In those four runs and in both `--deterministic-data` runs the progress bar reports the first train step at 46 to 48 s (`1/100 ... 46s/step` to `48s/step` in the captured stdout, which is not kept in the run directory). The `Filling up shuffle buffer` line is in one of the eleven captured denoiser stdout files of loops 1 to 3 (`convunext_audit_denoise_l2`, whose first step was 53 s) and in none of the other ten. The 37 to 46 s before the first step is therefore not the buffer fill in these runs. What it is was not isolated by its own measurement: it is consistent with loop 1's original reading, the build and XLA compile of the training step (the unet's first step took 74 s at 24,028,800 parameters and the bfcnn's 12 s at 39,616 in `unet_audit_l3` and `bfcnn_audit_l3`, so it follows the model, not the pipeline), with the loop-2 sighting a one-off in a first run. The untrained-baseline validation evaluation costs 14.0 s (from the epoch-0 grid line to the `Epoch-0 baseline` line, including the compile of the evaluation function). The `--patch-shuffle-buffer` default stays 2048: 512 changed neither epoch 1 nor the PSNR beyond the same-code range (24.2136 against 24.1023 to 24.5450).

**Dashboard cadence at 40 epochs (`convunext_confirm_cadence_l2`).** Command: `--variant tiny --patch-size 64 --epochs 40 --steps-per-epoch 10 --validation-steps 4 --max-train-files 400 --max-val-files 50 --batch-size 16 --viz-freq 40 --experiment-name convunext_confirm_cadence_l2` (298 s of process wall, exit 0, `status: "ok"`, 51 summary keys, `best_epoch` 38 of 40 epochs run). The dashboard was drawn after epoch 1 and after epochs 2, 4, ..., 40, that is 21 times, plus the always-drawn epoch-0 baseline, against 41 draws for a redraw after every epoch. Evidence: the time between an epoch's `run.log` line and the next epoch's `NoiseSigmaCurriculum` line is 0.01 to 0.02 s after 19 epochs (3, 5, ..., 39, no draw) and 2.25 to 3.28 s after the 19 epochs 2, 4, ..., 38 (mean 2.60 s per dashboard render), 6.74 s after epoch 1 (dashboard plus the `--viz-freq 40` grid); `training_dashboard.png` was last written 2.2 s after the epoch-40 line. The 19 avoided redraws are worth about 19 x 2.60 = 49 s of this run's 248 s `fit_wall_seconds` (computed from the measured per-render time, not from a second run with the every-epoch rule). `run.log` carries one `Visualizations: 4 written (epoch_000_denoise_grid.png, epoch_001_denoise_grid.png, epoch_040_denoise_grid.png, training_dashboard.png), 0 failed, dashboard and grids took 67.9s` line, and `analyzer.removed` is `["summary_dashboard.png"]`. A run of 39 epochs or fewer draws after every epoch, as before.

Open items of this audit, after the fixes of plan `plan-2026-09-19T224205-49c8bf80` (iteration 1). Fixed and described in "End-of-run artifacts and the run log": the unclipped pass-1 PSNR label, the unseeded additive grid noise and fresh-per-run viz batch, the missing best-epoch marker, the `setup_gpu` ERROR on every run, the missing `model_analysis/`, `analyzer` and `visualizations` blocks, the layer table buried in `run.log`, and the duplicate best-and-final test evaluation when the last epoch is the best. Still open: multiplicative and composite grid noise is unseeded; the validation set is not fixed unless `--deterministic-data` is given (validation crops and noise are otherwise drawn afresh); the default cosine schedule stops at 4.8e-5, not at the 1e-5 floor, and a 2-epoch run never anneals; the grid is redrawn on `--viz-freq` (the dashboard, 2.25 to 3.28 s per render in `convunext_confirm_cadence_l2`, follows the `max(1, epochs // 20)` cadence, so only runs of 40 or more epochs save anything; 6.9 s of redraw, dashboard plus grid, followed each 10.9 s epoch in `convunext_audit_denoise_l2`); the untrained baseline evaluation costs 12.6 s in `convunext_audit_denoise_l1` and 14.0 s in `_l2`; the 46 to 48 s before the first train step of a 2-epoch run is not the shuffle-buffer fill and was not isolated further ("Loop 2" and "Loop 3"); the denoiser's same-code floor is measured, 0.4427 dB val PSNR for three runs ("Loop 3"); `initial_loss_ratio`, `init_scale_warning` and `best_checkpoint_*` are not in the summary on purpose (see the note after the key tables); the end-of-run `test_eval` scores 64 px crops, which charges the zero-padding border artifact of the U-Net to every crop (whole images gain 0.04 to 0.26 dB more, "Loop 3"); unet and bfcnn keep fixed smoke names that the reused-name refusal turns into a second-run failure.

### Loop 3 (tiny, 64 px, 2 epochs of 100 steps, seed 42: the same-code triple, its cause, `--deterministic-data`, unet and bfcnn smoke)

Every number here was read from the named run's `results_summary.json`, `training_log.csv` or `run.log` when this section was written (stdout and load averages, where named, from the captured files outside the run directory); the directories are under `results/` (untracked). Audit note: `plans/plan-2026-09-19T224205-49c8bf80/findings/audit-loop-3.md`. Command of the runs below: the loop-2 command (no `--seed`, the default 42 applies), plus `--patch-shuffle-buffer 512` for `_buf512` and `--deterministic-data` for `_det1` and `_det2`; GPU 1, one job at a time.

| Run | Pipeline | `val_psnr_metric`, last epoch (dB) | Kodak24 gain at sigma 15 (dB) | CBSD68 gain at sigma 15 (dB) | epoch 1 (s) | `analyzer.seconds` |
|---|---|---|---|---|---|---|
| `convunext_audit_denoise_l3` | default | 24.4628 | 1.0675 | 0.1619 | 59.16 | 6.82 |
| `convunext_audit_denoise_l3_rep1` | default | 24.5450 | 1.1331 | 0.1507 | 56.93 | 6.15 |
| `convunext_audit_denoise_l3_rep2` | default | 24.1023 | 0.6803 | -0.2246 | 59.83 | 15.36 |
| `convunext_audit_denoise_l3_buf512` | default, buffer 512 | 24.2136 | 0.7022 | -0.1948 | 57.06 | 20.92 |
| `convunext_audit_denoise_l3_det1` | `--deterministic-data` | 24.3010 | 0.8194 | -0.0505 | 58.78 | 5.82 |
| `convunext_audit_denoise_l3_det2` | `--deterministic-data` | 24.2729 | 0.7842 | -0.0841 | 58.31 | 5.88 |

**The same-code floor.** Three runs of one command and one code state (the first three rows) differ in final validation PSNR by 0.4427 dB (24.1023 to 24.5450), in Kodak24 gain at sigma 15 by 0.4528 dB and in CBSD68 gain by 0.3865 dB, and the CBSD68 gain changes sign in `_rep2`. Against the rule written before the runs (SMALL up to 0.20 dB, MEDIUM 0.20 to 0.50, LARGE above) that is MEDIUM, 11.5 percent under the upper edge. With `_buf512` and the three loop-2 runs the seven runs of one recipe (two code states, none touching training) span 0.633 dB (mean 24.287, sample standard deviation 0.227), so the loop-2 range of 0.553 dB was not an artefact of mixing code states. These are ranges of three (seven) runs, not intervals; a range of three under-covers. A 2-epoch difference below about 0.5 dB val PSNR, or 0.5 to 0.9 dB Kodak24 gain, is not attributable on the default pipeline, and the sign of the CBSD68 sigma-15 gain at this size is noise. Nothing here says the spread shrinks with longer training: a 2-epoch run never anneals its learning rate.

**Why: the input pipeline is not reproducible from `--seed`.** `create_dataset` decoded in an unordered parallel map and drew the crop, the flips and rotation and the noise from the stateful global generator inside parallel maps. A CPU probe (two builds from 24 synthetic images with `set_seeds(42)` before each; the ConvUNeXt config and the real noise function) found 0 of 6 training batches identical, clean and noisy alike, and, for the validation stream, 2 of 6 clean and 0 of 6 noisy batches identical. So the training data and noise differ from run to run at a fixed seed, and so does the stream `val_psnr` is measured on. The segmentation trainer (`src/train/convunext/README.md`) has a stateless pipeline and reproduces its test mIoU to 1.4e-5.

**`--deterministic-data`.** ConvUNeXt trainer only, off by default. It decodes with `deterministic=True` (order preserved, the reads still run in parallel) and runs the maps that draw random numbers (crop, flips and rotation, noise) with `num_parallel_calls=None`: a stateful random op run by several threads draws in a racy order, and a probe of the same design with those maps still parallel gave clean 6 to 7 of 8 and noisy 0 to 1 of 8 identical batches, against 8 of 8 and 8 of 8 with them sequential. The shuffles need no seed of their own: `set_seeds` already seeds them. Multiplicative and composite noise are covered by the same mechanism (tested); the default keeps today's call pattern (tested). Measured effect on the pair `_det1` / `_det2` (same command, flag on): final validation PSNR 24.3010 and 24.2729, a difference of 0.0281 dB against the 0.4427 dB of the default triple; Kodak24 gain at sigma 15 differs by 0.0352 dB (0.4528), CBSD68 by 0.0336 dB (0.3865), and the CBSD68 sign no longer flips. The first epoch's training loss is identical to six digits in the two runs (0.083471), the second epoch's differs (0.004890 and 0.004818): what is left is GPU kernel non-determinism, which the flag does not touch. Two runs are a pair, not a floor of the same kind as the triple, and no triple with the flag on has been run. Cost, measured on the CPU with 400 real images and 4 patches per image (300 batches after 3 warm-up batches, host load about 8): 27.9 and 30.9 batches per second by default against 22.7 and 21.7 with the flag, 19 to 30 percent fewer; a tiny-variant training step consumes about 10 batches per second (100 steps in 10.4 s), and the two runs show no visible cost at that scale (epoch 1 58.78 and 58.31 s, epoch 2 11.46 and 11.40 s against 10.37 to 11.79 s by default). A model that consumes batches faster than about 20 per second would be input-bound with the flag on; that was not measured.

**`analyzer.seconds` is a thread-pool artefact, and one line fixes it.** The four default runs above took 6.15 to 20.92 s in `run_data_free_analysis` at 1-minute host loads of about 5 to 10, and the rank correlation with the load was 0.571 over the eight runs that record it, so load alone did not explain it. An isolated timing of `run_data_free_analysis` on a copy of `convunext_audit_denoise_l3/best_model.keras` (three repeats per setting) gave 14.7 to 21.0 s at the default 12 BLAS threads and 6.1 to 6.6 s with the call limited to one thread (6.3 to 7.0 s at four): the spectral phase (SVDs and power-law fits) is 2.5 to 2.9 s at one or four threads and 11 to 17 s at twelve, the weights phase (0.3 s) and the three figures (3.3 to 3.7 s) do not change. The call is now made inside `threadpool_limits(limits=1)` (`run_summary.run_data_free_analysis`, commit `7becc96da`, decision D-046), scoped to that call. `analysis_results.json` is unchanged by it apart from the randomised power-law fit p-values: 24 to 28 leaves differ between a one-thread and a twelve-thread run, all within the 33 that also differ between two runs of the same setting (27 of 2830 leaves differ between two identical runs). The confirmation runs `_det1` and `_det2` took 5.82 and 5.88 s at 1-minute loads of 2.15 to 5.04 and 3.40 to 4.80, and a repeat of the isolated timing on the current code took 5.64 to 5.98 s at load 2.4 to 2.6. Those loads are lower than the ones under which 15 to 21 s occurred, so these runs do not by themselves show that the slow mode is gone; the controlled evidence is the isolated comparison, where the same model and the same host load gave 6.1 to 6.6 s limited against 14.7 to 21.0 s unlimited.

**The end-of-run crop protocol, measured against whole images.** `eval_psnr_vs_noise` on `convunext_audit_denoise_l3/best_model.keras` (seed 42, sigmas 15 25 50; the outputs are not kept in `results/`): with 100 crops of 64 px it reproduces `test_eval` to every printed digit (Kodak24 gain 1.07 / 4.49 / 7.57 dB, CBSD68 0.16 / 3.74 / 7.12); with `--full-image` it gives Kodak24 1.22 / 4.72 / 7.84 (+0.15 / +0.22 / +0.26 dB) and CBSD68 0.24 / 3.81 / 7.15 (+0.08 / +0.07 / +0.04 dB). The subsets differ (100 crops against whole images), so the pairing is approximate. The crop protocol understates the gain by 0.04 to 0.26 dB, less than the same-code spread. `--patch-size 128` cannot be used on a saved ConvUNeXt denoiser: the model has a static 64 x 64 input and Keras refuses a 128 x 128 batch (`expected shape=(None, 64, 64, 3), found shape=(16, 128, 128, 3)`); `--full-image` rebuilds a flexible-input copy and works, at 57.9 s of wall against 17.7 s for the 64 px run. No `test_eval` key for whole images was added: it would cost about 48 s per run.

**unet and bfcnn smoke runs (`unet_audit_l3`, `bfcnn_audit_l3`).** The loop-2 flags (`--variant tiny --patch-size 64 --epochs 2 --steps-per-epoch 100 --validation-steps 20 --max-train-files 400 --max-val-files 50 --batch-size 16 --viz-freq 1`) on `train.bfunet.train_unet_denoiser` and `train.bfunet.train_bfcnn_denoiser`. Both exited 0 with `status: "ok"` and a 44-key `results_summary.json`: the ConvUNeXt denoiser's 52 keys minus the eight architecture keys (`model_family`, `convnext_version`, `depth`, `blocks_per_level`, `dims`, `kernel_size`, `drop_path_rate`, `dropout_rate`), with `data_range` (renamed `image_range` in iteration 4), `train_image_dirs`, `n_train_files` and `n_val_files` from the shared head, one `Visualizations:` line in `run.log` and the default dashboard title. These are two-epoch smoke facts, not a baseline: `unet_audit_l3` (24,028,800 parameters, fit 140.1 s, first step 74 s) went from validation PSNR 0.39 dB after epoch 1 to 8.38 dB, test PSNR 8.0 to 8.3 dB at the three sigmas (gain -6.8 to -16.5 dB), the epoch-0 baseline loss is 585,822,592 (untrained BatchNorm inference statistics); `bfcnn_audit_l3` (39,616 parameters, fit 53.3 s, first step 12 s) went from 12.70 to 17.76 dB, test PSNR 16.9 to 17.3 dB (gain -7.75 to +1.98 dB). Neither learned much in 200 steps and no run of the earlier code exists to compare with, so "they behave as before" rests on the key parity, the unchanged tests and the log line, not on a number. `bfcnn_audit_l3` also spent about 290 s of its 355.5 s process wall before its first `run.log` line: the bfcnn config lists six training directories (`Megadepth`, `div2k/train`, `WFLW/images`, `bdd_data/train`, `COCO/train2017`, `VGG-Face2/data/train`) and scans every one before `--max-train-files` applies; `VGG-Face2/data/train` alone holds 3,141,890 files. The same listing takes seconds once the file cache is warm (a `find` of that directory took 10.4 s afterwards). There is no flag for the directories (see "Data"): for a smoke run of the bfcnn trainer build the `TrainingConfig` with smaller `train_image_dirs` and `val_image_dirs`. The unet and ConvUNeXt trainers scan two directories: 8 s (`unet_audit_l3`) and 6 s (`convunext_audit_denoise_l3`) from process start to the first `run.log` line.

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
| `test_the_*_gabor_stem_*.py`, `test_convunext_*.py` | Gabor stem flags and width rule, self-iterate and `--deterministic-data` (`test_convunext_self_iterate.py`: identical batches in two builds, the default call pattern), supporting fixes (`test_convunext_supporting_fixes.py`: the seeded grid, the clipped pass-1 label, the analyzer and `visualizations` blocks, the once-only test evaluation, the import-time TensorFlow context, the dashboard cadence including a resumed fit) |

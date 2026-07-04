# bfunet — Bias-Free Denoiser Training

Production training suite for **bias-free image denoisers** built on a shared substrate.
Two model families are trained here — a **ConvUNeXt** (ConvNeXt U-Net) denoiser and a
**CliffordUNet** (geometric-algebra U-Net) denoiser — both wired through one common
data/curriculum/training module (`common.py`).

The design principle throughout is **bias-free / degree-1 homogeneity**: every layer avoids
additive bias so the network satisfies `f(a·x) = a·f(x)`. This makes the residual `x − f(x)`
interpretable as a scaled score (Miyasawa/Tweedie), and lets a single model generalize across
noise levels it was not explicitly trained on.

---

## Directory layout

| File | Role |
|------|------|
| `common.py` | **Shared substrate** — data pipeline, noise curriculum, self-iteration pool, callbacks, dashboard, the `train()` orchestrator, the `BFUnetTrainingConfig` base dataclass, and `add_common_arguments()` (the CLI shared by all trainers). Not run directly. |
| `train_convunext_denoiser.py` | **Production trainer** — bias-free ConvNeXt U-Net denoiser. The most actively developed. |
| `train_cliffordunet_denoiser.py` | **Production trainer** — bias-free homogeneous CliffordUNet denoiser (Clifford geometric-product blocks). |
| `train_unet_denoiser.py` | **Baseline trainer** — bias-free plain U-Net denoiser (classic conv/residual blocks). Same infrastructure feature set as ConvUNeXt; the apples-to-apples baseline. |
| `eval_psnr_vs_noise.py` | **Standalone tool** — PSNR-vs-noise-level evaluation of any saved `.keras` denoiser, with optional SOTA reference overlay. |
| `variance_probe.py` | **Standalone tool** — ConvUNeXt-only training-stability probe (run-to-run variance across seeds). |
| `FINDINGS.md` | Empirical note on the channel-matching (`--zero-pad-channels`) experiment. |

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

# CliffordUNet base denoiser
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_cliffordunet_denoiser \
    --variant base --epochs 100 --batch-size 4 --gpu 1

# Plain U-Net baseline denoiser
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_unet_denoiser \
    --variant base --epochs 100 --batch-size 4 --gpu 1

# Fast mechanism check (a few steps, tiny variant) — verifies the pipeline end-to-end
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_convunext_denoiser --smoke
```

Outputs (checkpoints, CSV history, config JSON, dashboard PNG, eval grids) are written under
the repo-root `results/` directory, in a run folder named from the model prefix + variant.

> **Memory note:** the `base` variant at `--patch-size 256` needs `--batch-size 4` on a
> 24 GB RTX 4090. Reduce batch or patch size on smaller GPUs.

### Data

Training data directories are **hardcoded** in `BFUnetTrainingConfig` (not CLI flags):

- **Train:** `COCO/train2017` + `div2k/train` (weighted so COCO does not drown DIV2K)
- **Validation:** `div2k/validation`

To train on other data, edit `train_image_dirs` / `val_image_dirs` in the config, or construct
the `TrainingConfig` programmatically. `--max-train-files` / `--max-val-files` cap how many are used.

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
| `--learning-rate` | 1e-3 | Peak LR (AdamW) |
| `--weight-decay` | 0.004 | AdamW decoupled weight decay (no separate L2) |
| `--warmup-epochs` | 10% of epochs | LR warmup length |
| `--max-train-files` / `--max-val-files` | None | Cap source images |
| `--steps-per-epoch` / `--validation-steps` | None | Bound epoch length |
| `--mixed-precision` | off | Enable `mixed_float16`. **Measured slower** than fp32 for base@256/b4 on a 4090 (XLA is disabled by the bilinear-upsample grad). Off by default. |

**Architecture (shared)**

| Flag | Default | Meaning |
|------|---------|---------|
| `--no-gabor-stem` | off (stem ON) | Disable the frozen Gabor depthwise stem |
| `--gabor-filters` | 32 | Gabor filters per channel |
| `--no-gabor-projection` | off | Drop the 1×1 projection after the Gabor stem (requires `channels*gabor_filters == initial_filters`) |
| `--initial-filters` | variant | Override level-0 width |
| `--filter-multiplier` | 2.0 | Per-level channel growth: `channels[l] = round(initial_filters * m**l)` |
| `--depth` | variant | Number of U-Net levels (≥2) |
| `--blocks-per-level` | variant | Blocks per level (≥1) |
| `--final-projection-groups` | 1 | Groups for the final 1×1 output projection (`-1` = one group per output channel) |
| `--laplacian-pyramid` | off | Enable the Laplacian-pyramid downsample/skip path |
| `--zero-pad-channels` | off | Parameter-free channel matching instead of 1×1 adjust convs (see `FINDINGS.md`) |
| `--mean-pooling` | off | AveragePooling (linear) instead of MaxPooling — keeps the encoder linear for the Miyasawa interpretation |
| `--expose-bottleneck` | off | Expose the bottleneck latent as a second output |

**Noise curriculum** — training sweeps `sigma_max` from a narrow low-noise range up to a wide one:

| Flag | Default | Meaning |
|------|---------|---------|
| `--sigma-max-start` | 0.025 | Curriculum start (low noise) |
| `--sigma-max-end` | 0.25 | Curriculum end (wide noise) |
| `--curriculum-schedule` | linear | `linear` / `cosine` / `exp` |
| `--curriculum-epochs` | = epochs | Epochs over which the curriculum widens |
| `--deep-supervision` | off | Enable deep-supervision auxiliary heads |

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
| `--smoke` | off | Tiny end-to-end mechanism check (few steps, constant LR) |
| `--init-from PATH` | None | Warm-start weights from a `.keras` checkpoint (architecture must match) — primary use: self-iterate fine-tuning |
| `--dashboard DIR` | None | Rebuild the dashboard PNG from an existing run dir and exit (no training) |
| `--analyzer` / `--analyzer-freq` | off / 10 | Run ModelAnalyzer every N epochs |
| `--viz-freq` / `--viz-samples` | 5 / 8 | Clean/noisy/denoised grid cadence & columns |
| `--output-dir` | `results` | Output root (repo-root `results/`) |
| `--experiment-name` | None | Override the run-folder name |
| `--gpu` | None | GPU index (e.g. `--gpu 1`) |

---

## ConvUNeXt trainer (`train_convunext_denoiser.py`)

Trains `create_convunext_denoiser` — a bias-free ConvNeXt U-Net. Variants: `tiny`, `small`,
`base` (default), `large`, `xlarge`. Model-specific flags beyond the shared set:

| Flag | Default | Meaning |
|------|---------|---------|
| `--variant` | base | ConvUNeXt size preset |
| `--convnext-version` | v1 | `v1` = strict bias-free; `v2` adds a trainable GRN β (mildly breaks strict homogeneity) |
| `--block-normalization` | batchnorm | `batchnorm` = variance-only BiasFreeBatchNorm (restores degree-1 homogeneity, pairs with LeakyReLU); `layernorm` = per-input scale-invariant (degree-0) |
| `--block-activation` | leaky_relu | Activation for blocks + stem + deep-supervision heads (final activation stays linear) |
| `--block-activation-alpha` | 0.1 | LeakyReLU negative slope |
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

## CliffordUNet trainer (`train_cliffordunet_denoiser.py`)

Trains `create_cliffordunet_denoiser` — a bias-free, degree-1-homogeneous U-Net built from
Clifford geometric-product blocks. Variants: `tiny`, `small`, `base` (default).
Model-specific flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--variant` | base | CliffordUNet size preset |
| `--cli-mode` | full | Clifford components for the local interaction: `inner` / `wedge` / `full` |
| `--ctx-mode` | abs | **HOMOGENEITY-CRITICAL.** `abs` keeps the context stream degree-0 (block stays degree-1 homogeneous, strict Miyasawa). `diff` makes the geometric product degree-2 and **breaks homogeneity** — use only deliberately. |
| `--shifts` | variant | Override the geometric-product base shift offsets (ints ≥1, sized per level) |
| `--layer-scale-init` | 1e-5 | Initial LayerScale γ for the gated geometric residual |

```bash
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_cliffordunet_denoiser \
    --variant base --cli-mode full --ctx-mode abs \
    --epochs 100 --batch-size 4 --gpu 1
```

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
| `--block-activation` | leaky_relu | Whole-denoiser activation (output stays linear) |
| `--block-activation-alpha` | 0.1 | LeakyReLU negative slope |
| `--dropout` | 0.0 | Dropout inside the conv blocks (after activation) |
| `--block-normalization` | batchnorm | `batchnorm` = bias-free variance-only BatchNorm (center=False); `layernorm` = bias-free per-input LayerNorm |

```bash
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_unet_denoiser \
    --variant base --block-normalization batchnorm \
    --epochs 100 --batch-size 4 --gpu 1
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
> Loading a **CliffordUNet** checkpoint standalone may require importing the Clifford model module
> first so its custom objects are registered.

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
- **Clifford homogeneity.** Keep `--ctx-mode abs` unless you specifically want to break degree-1
  homogeneity; `diff` makes the block degree-2.
- **Mixed precision is usually slower here** — leave it off unless you have measured a win.
- **Outputs go to repo-root `results/`.** Do not point `--output-dir` inside `src/`.
- **Always set `MPLBACKEND=Agg`** to avoid X11 crashes on headless/remote systems.

See `FINDINGS.md` for the empirical channel-matching (`--zero-pad-channels` /
`--extra-zero-output-channels`) variance investigation.

# Bias-Free ConvUNeXt Blind Denoiser

A training pipeline for a **single** image denoiser that handles **every** noise
level it was never explicitly trained on — from a barely-perceptible σ≈6 to a
heavy σ≈64 (on the 0–255 scale) — with no noise-level input, no per-σ fine-tuning,
and no test-time tricks.

That "one model, all noise levels" property is not a lucky side effect of a big
network. It falls out of two deliberate design choices that this README is about:

1. **Bias-free architecture.** Strip every additive degree of freedom (conv biases,
   normalization centering) so the network becomes *scale-homogeneous*:
   `f(α·y) = α·f(y)`. A denoiser with this property automatically rescales its
   behavior to the noise magnitude in front of it.
2. **A noise-σ curriculum.** Start training on a narrow, easy noise band and widen
   it epoch-by-epoch, so the model learns the easy regime first and *then*
   generalizes outward — rather than thrashing on the full range from step zero.

Everything else (ConvNeXt blocks, the frozen Gabor stem, the Laplacian-pyramid
skips, the bottleneck monitor) is in service of making that core idea train well
and train *legibly*.

- **Model factory**: `dl_techniques.models.bias_free_denoisers.bfconvunext.create_convunext_denoiser`
- **Trainer**: `src/train/bfunet/train_convunext_denoiser.py`
- **Theory notes**: [`research/miyasawas_theorem.md`](../../../research/miyasawas_theorem.md),
  [`research/miyasawas_theorem_multiplicative.md`](../../../research/miyasawas_theorem_multiplicative.md)
- **Model-level README** (architecture tables, V1/V2 deep dive):
  [`bias_free_denoisers/README.md`](../../dl_techniques/models/bias_free_denoisers/README.md)

---

## 1. The idea in one minute

### Miyasawa: a denoiser is secretly a score function

For additive Gaussian noise `y = x + n`, `n ~ N(0, σ²I)`, the **minimum-MSE**
denoiser — the one MSE training converges to — is the posterior mean `E[x | y]`.
Miyasawa's theorem (1961; the same result appears as Tweedie's formula) says this
optimal denoiser has a remarkable closed form:

```
D(y) = y + σ² · ∇_y log p(y)
```

In words: **the denoiser's residual *is* the score of the noisy distribution**,
scaled by the noise variance:

```
D(y) − y  =  σ² · ∇ log p(y)
```

The proof hinges on a fact unique to *additive Gaussian* noise: `∇_y p(y|x)` is
linear in `x`, so when you integrate against the image prior the whole thing
collapses exactly to `E[x|y] − y`. That linearity is why this identity is clean
for AWGN and *not* for other noise models (see §5.2).

Why care? Because it means a trained denoiser implicitly knows `∇ log p` — the
gradient of the log-prior of natural images — at every pixel. That single object
is what powers diffusion-model sampling and RED-style ("Regularization by
Denoising") inverse-problem solvers. A good denoiser is a reusable prior.

### Bias-free ⇒ scale homogeneity ⇒ generalization across σ

Here is the lever that turns a *fixed-σ* denoiser into a *blind* one. Consider
scaling the input by α (equivalently, scaling the noise level). The MMSE denoiser
should scale its output identically — denoising is invariant to overall gain.
A network can only honor that if it is **homogeneous of degree 1**:

```
f(α · y) = α · f(y)     for all α
```

A standard CNN *cannot* be homogeneous: every bias term and every normalization
shift adds a constant that does **not** scale with the input, effectively
hard-coding the model to one noise magnitude. Remove all of them and the network's
only remaining lever is *structural* (is this a smooth region or an edge?), never
*absolute magnitude*. That is precisely the right inductive bias for a blind
denoiser — and empirically (Mohan et al., ICLR 2020) it is what lets a model
trained on σ∈[0, 50] extrapolate cleanly to σ it never saw.

**The bias-free contract**, enforced throughout this model:

| Knob | Setting | Why |
|------|---------|-----|
| Conv / Dense biases | `use_bias=False` everywhere | Any additive offset breaks `f(αy)=αf(y)` |
| Normalization centering | `center=False` (no β shift) | A learned mean-shift is structurally a bias |
| Final activation | `linear` | A nonlinearity (ReLU, tanh, sigmoid) destroys homogeneity and clips the negative residuals the score identity needs |
| Input range | `[−1, +1]` (zero-centered) | A bias-free net has no way to absorb a non-zero data mean — it leaks through every layer |

These four are **hard constraints**. The trainer asserts the linear head
(`final_activation="linear"` is hardcoded in `build_model`) and ships a
`verify_bias_free(model)` audit that walks every layer and logs any surviving
`use_bias=True` or `LayerNorm(center=True)`.

---

## 2. Model architecture

The denoiser is a **bias-free ConvNeXt U-Net** built functionally by
`create_convunext_denoiser`. It is a symmetric encoder/bottleneck/decoder with
skip connections; the building block is a ConvNeXt residual branch.

```
input (H,W,C) in [-1,+1]
   │
   ▼  stem  (default: Conv2D 7x7 → GRN → GELU ;  or frozen Gabor stem, §4.1)
   │
 ┌─┴─ encoder level 0 ─ [N ConvNeXt blocks] ─┐ skip0 ──────────────────────┐
 │         ▼ downsample (MaxPool, or Laplacian low-band §4.2)              │
 │   encoder level 1 ─ [N ConvNeXt blocks] ─┐ skip1 ───────────────┐      │
 │         ▼ downsample                                            │      │
 │      … (depth levels) …                                         │      │
 │         ▼                                                       │      │
 │   bottleneck ─ [N ConvNeXt blocks]  (optional linear tap §4.3) │      │
 │         ▲ upsample (bilinear) → concat skip → 1x1 adjust        │      │
 │   decoder level 1 ─ [N ConvNeXt blocks] ◄──────────────────────┘      │
 │         ▲ upsample → concat skip0 → 1x1 adjust                         │
 └─► decoder level 0 ─ [N ConvNeXt blocks] ◄─────────────────────────────┘
   │
   ▼  Conv2D 1x1, linear, bias-free  →  denoised (H,W,C)
```

- **Channel progression** doubles per level: `initial_filters · 2^i`. For `base`
  (`initial_filters=64`, `depth=4`) that is `[64, 128, 256, 512, 1024]` — each
  halving of spatial resolution is paid for by a doubling of channels, so the
  per-level tensor budget stays roughly flat while the *receptive field* and
  *semantic abstraction* climb.
- **Encoder level** = 1×1 channel-adjust → `blocks_per_level` ConvNeXt blocks →
  downsample + emit skip. The 1×1 channel-adjust is what makes the residual add
  inside each ConvNeXt block valid (block in/out channels must match), and it is
  the *only* place channel count changes — the blocks themselves are
  channel-preserving.
- **Decoder level** = bilinear upsample → concat skip → 1×1 channel-adjust →
  `blocks_per_level` ConvNeXt blocks. The concat (not add) fuses the upsampled
  coarse features with the same-resolution skip, and the following 1×1 mixes the
  two streams before the blocks refine them.
- **Skip connections** are the spine of the whole design. Downsampling throws away
  high-frequency spatial detail; the skips smuggle that detail *around* the
  bottleneck so the decoder can reconstruct sharp edges that a pure
  encoder→decoder stack would have blurred away. For a denoiser this matters
  doubly: the fine detail the skips preserve is exactly the signal that is hardest
  to tell apart from noise, so the network gets to make that call with full-
  resolution context in hand. By default a skip carries the *pre-downsample*
  tensor; with the Laplacian pyramid (§4.2) it instead carries the high-frequency
  *band*, which changes the semantics in a deliberate way.
- **Why U-Net for denoising specifically.** Noise lives at all scales, but its
  *statistics* differ by scale — fine noise is near-white, coarse structure is
  strongly correlated. The encoder/decoder pyramid lets the model attack each
  scale with a matched receptive field: shallow levels clean local texture, deep
  levels enforce global consistency, and the skips stitch the two back together.
  The symmetric depth (same number of down- and up-steps) guarantees the output
  returns to full input resolution with no learned resampling artifacts.
- **Bottleneck** is the most abstract, lowest-resolution stage — `blocks_per_level`
  ConvNeXt blocks at the deepest width — where the network has the widest
  receptive field and does its heaviest global reasoning. It is also the natural
  place to tap a latent (§4.3).
- **Output head** is a single bias-free `Conv2D(C, 1×1, activation='linear')`.

### Size variants (`CONVUNEXT_CONFIGS`)

| Variant | depth | initial_filters | blocks/level | drop_path |
|---------|-------|-----------------|--------------|-----------|
| `tiny`  | 3 | 32  | 2 | 0.0 |
| `small` | 3 | 48  | 2 | 0.1 |
| `base`  | 4 | 64  | 3 | 0.1 |
| `large` | 4 | 96  | 4 | 0.2 |
| `xlarge`| 5 | 128 | 5 | 0.3 |

> The variant table ships `convnext_version='v2'`, but the **trainer overrides it**
> with `--convnext-version` (default `v1`) — see below. So the trainer's effective
> default is **V1**, the strictly bias-free variant.

### The ConvNeXt block (V1 vs V2)

Each block is a residual *branch* — the U-Net wiring (skip add + stochastic depth)
is applied by the caller, not the block:

```
branch:  DepthwiseConv2D(7x7) → LayerNorm(center=False) → Conv1x1(4× expand)
         → GELU → [GRN, V2 only] → dropout → Conv1x1(reduce) → LayerScale γ
wiring:  x  +  StochasticDepth(rate)( branch(x) )
```

- **Depthwise 7×7 + inverted bottleneck** (4× expansion) is the ConvNeXt recipe:
  large-kernel spatial mixing, then a fat pointwise MLP for channel mixing.
- **LayerNorm with `center=False`** is a subtle but load-bearing point. It is
  *scale-invariant* (degree-0) on the normalized axis, not scale-equivariant — but
  because it lives inside a residual branch that is multiplied by LayerScale γ and
  added to an identity path, the **whole block stays degree-1 homogeneous**. There
  is no additive offset anywhere on the path.
- **V2 adds Global Response Normalization (GRN)** between GELU and dropout. GRN
  introduces cross-channel competition (`Y = X + γ·(X⊙normalize(X)) + β`) and is the
  headline ConvNeXt-V2 improvement. **But GRN's `β` is a trainable additive term** —
  a genuine bias. So **V2 is only approximately bias-free**: `verify_bias_free` will
  flag the GRN betas as "expected offenders" rather than failing. If you want the
  exact Miyasawa/homogeneity guarantee, use **V1** (the trainer default). V2 is an
  opt-in that trades a little theoretical purity for GRN's representational punch.

### LayerScale, stochastic depth, orthogonal init

These three keep a deep residual U-Net trainable and bias-free at the same time.

- **LayerScale (γ).** Each branch is scaled by a learnable per-channel multiplier
  (`LearnableMultiplier`, a *pure* multiply — no offset, so homogeneity holds). It
  is initialized to `1e-4`: small enough that each branch starts as a near-identity
  (a mild prior that stabilizes early training), large enough that gradients —
  which are proportional to γ — still flow from step 0. A hard floor of `1e-6`
  prevents γ collapsing to exactly zero, which would permanently kill a branch.
- **Progressive stochastic depth.** Drop-path rate ramps linearly with depth:
  shallow blocks ≈0, the deepest block hits the full `drop_path_rate`. Deeper
  blocks are the most redundant, so they are the ones randomly dropped during
  training — a depth-aware regularizer. (Bottleneck blocks use a flat rate.)
- **Orthogonal kernel init.** Structural convs (stem, channel-adjusts, output)
  use `'orthogonal'`, **not** `he_normal`. In a deep residual
  U-Net `he_normal` compounds variance across the many residual adds and can blow
  up activations; an orthogonal (norm-preserving) init is the right choice for a
  homogeneous, bias-free network and is what actually delivers init stability —
  *not* the small γ. (`he_normal` is the historical default; it was replaced.)

---

## 3. Why ConvNeXt for a denoiser?

Worth a sentence of editorializing: denoising rewards a **large effective
receptive field** (to separate spatially-white noise from spatially-correlated
signal) and **cheap channel mixing** (to recombine multi-scale features). ConvNeXt
gives both — the 7×7 depthwise conv buys receptive field at depthwise (i.e. cheap)
cost, and the inverted-bottleneck MLP does the heavy channel work. The U-Net
wrapper adds the multi-resolution hierarchy that lets the model attack noise at
several scales simultaneously. It is a deliberately modern, efficient backbone for
a task that older work attacked with plain stacks of 3×3 convs (DnCNN, BFCNN).

---

## 4. Optional modules

All three are off-by-default at the *factory*, but the **trainer turns the Gabor stem
on by default** because it is cheap and helps. Each preserves the bias-free
invariant.

### 4.1 Frozen Gabor stem  (`use_gabor_stem`, trainer default **ON**)

Instead of learning the first layer from scratch, prepend a **frozen, deterministic
Gabor filterbank** (`create_gabor_depthwise_conv2d`, an implementation of the
Özbulak & Ekenel "Initialization of CNNs by Gabor Filters" scheme). A 2-D Gabor
kernel is a sinusoidal plane wave under a Gaussian envelope:

```
x_θ =  x·cos θ + y·sin θ
y_θ = −x·sin θ + y·cos θ
g(x,y) = exp(−(x_θ² + γ²·y_θ²) / (2σ²)) · cos(2π·x_θ/λ + ψ)
```

Five parameters shape each filter, and each has a clear job:

| Param | Role | Default sweep (Table I) |
|-------|------|-------------------------|
| **σ** | Gaussian envelope width — the filter's spatial extent / scale | `(2.0, 21.0)` |
| **θ** | orientation of the carrier (degrees) — which edge direction it fires on | `(0°, 360°)` |
| **λ** | carrier wavelength — the spatial frequency / feature thickness it tunes to | `(8.0, 100.0)` |
| **γ** | aspect ratio — how elongated (edge-like) vs. round (blob-like) the envelope is | `(0.0, 300.0)` |
| **ψ** | carrier phase (degrees) — even (cosine, bar/ridge) vs. odd (sine, step-edge) symmetry | `(0°, 360°)` |

**How the bank is laid out — this is the key design choice.** The stem does **not**
build a full Cartesian grid of σ×θ×λ×γ×ψ combinations (that would be tens of
thousands of filters). Instead it sweeps each parameter with
`np.linspace(min, max, n_filters)` and gives output channel *j* the *j*-th sample of
**every** parameter at once. So the `--gabor-filters` filters trace a single
**diagonal path** through the 5-D parameter space: filter 0 is short-σ / θ=0 /
short-λ, the last filter is long-σ / θ=360° / long-λ, and the rest interpolate
jointly. With the default `gabor_filters=32` you get 32 jointly-varying
orientation+scale+phase detectors — a deliberately compact, coarse-but-broad
covering rather than an exhaustive (and mostly redundant) grid.

**Wiring.** The bank is a `DepthwiseConv2D` with `depth_multiplier=gabor_filters`,
so it is applied **per input channel with no cross-channel mixing**: a 3-channel
input through 32 filters yields `3 × 32 = 96` channels. A **mandatory** bias-free
1×1 `Conv2D` then projects those 96 back down to `initial_filters` (64 for `base`).
The whole stem is `trainable=False` and uses no bias, so it contributes **zero
trainable parameters** and stays purely linear/homogeneous — it cannot break the
bias-free invariant. It is also deterministic: two builds produce byte-identical
kernels (no seed, no RNG).

**Why it helps a denoiser specifically:** natural-image *signal* is concentrated in
oriented, multi-scale edges and textures — exactly what this bank detects — while
*noise* is spatially white (flat spectrum). Handing the learnable layers a structured
oriented/multi-frequency feature space from epoch 0 (instead of raw pixels) amplifies
signal structure relative to noise immediately, and saves the network from having to
rediscover Gabor-like first-layer filters by gradient descent — which is empirically
what randomly-initialized first layers converge to anyway. It is also the textbook
model of V1 simple-cell receptive fields.

**Cost & knobs.** Zero trainable parameters, but one extra depthwise conv at
inference (throughput impact unprofiled — see §8). Disable with `--no-gabor-stem`.
Filter count via `--gabor-filters` (default 32). Kernel size is config-only
(`gabor_kernel_size=7`, matching the 7×7 default stem so the receptive field is
unchanged when you swap stems). The five parameter ranges use the Özbulak & Ekenel
Table I defaults and are not exposed on the CLI.

### 4.2 Laplacian-pyramid downsampling  (`--laplacian-pyramid`, default OFF)

This flag changes **how a level steps down in resolution** and **what the skip
carries**, and it is the most conceptually interesting of the optional modules.

**The default (OFF) junction** is the textbook U-Net move: `MaxPool(2×2)` produces
the half-resolution tensor that continues down the encoder, and the **full-resolution
input** to the pooling is copied verbatim along the skip. Note the redundancy — the
coarse content lives in *both* paths (the skip contains the low frequencies too,
they are just mixed in with the high ones), so the decoder receives overlapping
information and the encoder is free to ignore detail it knows the skip will restore.

**The ON junction** replaces that with a `LaplacianPyramidLevel`, a signal-level
band split:

```
split(x):
    blur = GaussianFilter(x)            # fixed low-pass
    low  = BlurPool2D(blur)             # (B, H/2, W/2, C)  → continues DOWN the encoder
    up   = UpSampling2D(low)            # (B, H,   W,   C)
    high = x − up(low)                  # (B, H,   W,   C)  → goes ACROSS the skip

merge(low, high):  x_rec = high + up(low) == x            # exact to float precision
```

Now the two paths are **complementary, not overlapping**: the skip carries *only*
the high-frequency residual (edges, texture, fine detail), and the encoder carries
*only* the smoothed low-frequency band. Information is **partitioned** rather than
duplicated. Three things fall out of this:

- **Each path gets the content it is good at.** Skips are the natural home for
  high-frequency detail (that is what they exist to preserve); the encoder's deeper,
  wider, more-abstract blocks are the natural home for the smooth global structure.
  The band split makes that division of labor explicit instead of leaving the
  network to sort it out.
- **The whole hierarchy is forced to participate.** Because the encoder no longer
  receives the high frequencies at all, it *cannot* punt detail to the skip and
  coast — every level must actually process its band. This tends to make deeper
  levels earn their keep.
- **It is an exact, invertible decomposition.** `high` is *defined* as the residual
  `x − up(low)`, so `merge(split(x)) == x` to float precision regardless of blur
  quality — the split adds no reconstruction error of its own, it only reorganizes
  where information flows. For a denoiser there is a bonus: the band split is itself
  a mild frequency prior, and the noise/signal SNR differs by band (fine bands are
  noisier), so giving each band its own processing path is well matched to the task.

The pyramid is built from fixed linear ops only — Gaussian blur (non-trainable by
default), average-pool decimation, bilinear upsample, subtraction — so it is
**channel-preserving, zero-parameter, and bias-free/homogeneous by construction**.
Crucially, the **OFF-path layer names are preserved** (`encoder_downsample_{level}`,
`bottleneck_downsample`), so toggling the flag is `.keras`-checkpoint-compatible and
the OFF path stays byte-identical to the original architecture. Blur kernel size via
`laplacian_kernel_size` (config-only).

### 4.3 Bottleneck tap  (`--expose-bottleneck`, default OFF)

Add a zero-parameter `Activation('linear')` tap on the deepest latent so the factory
returns `[denoised, bottleneck]`. The trainer then fits a single-output *view* (the
bottleneck is excluded from the MSE loss; it is trained implicitly through the
decoder) and saves the full two-output model separately. The point of the tap is
**observability** — it is what the bottleneck monitor (§6.2) visualizes — and a hook
for future multi-task heads.

---

## 5. The training recipe

### 5.1 Data pipeline

A weighted `tf.data` pipeline over two corpora:

- **Train**: COCO `train2017` (~118K images) **+** DIV2K train (~800 images).
- **Val**: DIV2K validation.

The two are sampled with **per-directory weighting** (`select_weighted_image_paths`)
so COCO's 118K does not drown DIV2K's 800; paths are capped (`--max-train-files`
→ 10000, `--max-val-files` → 500). The pipeline then:

1. `from_tensor_slices(paths)` → shuffle paths → `repeat()`.
2. `flat_map` each path into `patches_per_image` copies, then a **second 2048-buffer
   shuffle** — this decorrelates the multiple crops of one image at the *path-string*
   level so a batch is not dominated by one source image.
3. Decode → normalize to `[−1, +1]` (`image/127.5 − 1`) → aspect-preserving upscale
   if smaller than the patch → `random_crop(patch_size)`.
4. Drop black/corrupt patches (`sum|x| > 0`).
5. (train) `augment_patch`: random flips + rot90.
6. `clip(−1, 1)` guard → **add noise** (§5.2) → `batch` → `prefetch(AUTOTUNE)`.

### 5.2 Noise models

Noise is injected *inside* the pipeline so every epoch sees fresh corruption. There
are three modes, selected by flag; **the per-image σ is always drawn first** and
identically, so the curriculum behaves the same regardless of mode (and the RNG
draw order is frozen for byte-identical reproducibility of existing checkpoints):

```
σ  ~  Uniform(noise_sigma_min=0,  sigma_max_var)        # sigma_max_var is live, see §5.3
```

| Mode | Flag | Formula | Status |
|------|------|---------|--------|
| **Additive** (default) | *(none)* | `y = x + N(0,1)·σ` | Exact Miyasawa identity holds |
| **Multiplicative** | `--multiplicative-noise` | `y = x·(1 + N(0,1)·σ)` | **Approximation** (see below) |
| **Composite** | `--composite-noise` | `y = x·n + a`, `n~N(1,σ²)`, `a~N(0,(ratio·σ)²)` | **Approximation**; takes precedence over `--multiplicative` |

Everything is clipped to `[−1, +1]` after corruption.

> **Why additive is the principled default.** Only additive Gaussian noise yields
> the clean `residual = σ²·∇log p` identity. For **multiplicative** noise the
> posterior correction picks up an irreducible second moment `E[x²|y]` that a
> single-output denoiser does not expose, and its small-σ expansion contains a
> non-homogeneous `2σ²·y` shrink term that a *strictly* bias-free (degree-1) network
> **cannot represent exactly**. MSE training still converges to the true `E[x|y]` —
> the gap is representational, not a training failure — so the multiplicative and
> composite paths are shipped as **documented approximations**, useful for
> experimentation, not as theorems. The composite mode's additive floor
> (`--composite-additive-ratio`, default 0.5) regularizes the `x≈0`
> ill-conditioning and makes the problem *less* mismatched to a bias-free net. Full
> derivation: [`research/miyasawas_theorem_multiplicative.md`](../../../research/miyasawas_theorem_multiplicative.md).

### 5.3 The noise-σ curriculum

The curriculum is the second half of the "blind denoiser" story. Although a
bias-free net *can* generalize across σ, throwing the full range at it from step 0
converges slowly and unstably. Instead:

- `sigma_max_var` is a **`tf.Variable`** (not a Python float) captured by reference
  inside the `tf.data` noise map. A Python float would be baked into the graph at
  trace time; a `tf.Variable` is re-read every epoch, so reassigning it widens the
  noise band **without retracing**.
- `NoiseSigmaCurriculumCallback` reassigns it each epoch, interpolating
  `sigma_max_start → sigma_max_end` (defaults `0.05 → 0.5`) over `curriculum_epochs`
  (defaults to `epochs`), with a `linear | cosine | exp` schedule.

In benchmark units (`σ₂₅₅ = σ · 127.5`) the default curriculum sweeps **σ₂₅₅ ≈ 6.4
→ 63.75**, spanning and exceeding the classic 15 / 25 / 50 regimes — as one blind
model.

> Because difficulty *rises* over training, `val_loss` is **non-monotonic**, so
> **early stopping is disabled by default** (`early_stopping_patience = -1`).
> `ModelCheckpoint` still tracks `best_model.keras` by `val_loss`.

### 5.4 Loss, metrics, optimizer

- **Loss: MSE.** This is not arbitrary — least-squares is exactly the objective
  whose optimum is the Miyasawa posterior mean. Switching to L1/Charbonnier would
  break the residual=score equality.
- **Metrics**: `mae`, `PsnrMetric(max_val=2.0)`, `SsimMetric(max_val=2.0)`. The
  `max_val=2.0` is because images live in `[−1, +1]` (dynamic range 2.0); this makes
  the reported dB directly comparable to published `max_val=255` numbers.
- **Optimizer**: AdamW (`optimizer_builder`), gradient clipping by norm `1.0`.
- **LR schedule**: cosine decay with warmup (`learning_rate_schedule_builder`), peak
  `1e-3`, warmup = 10% of epochs by default, cosine floor `alpha=0.01`.
- No EMA, no mixed precision, no explicit `jit_compile` (backend defaults apply).

---

## 6. Monitoring & callbacks

Wired in this order (the ordering matters — the LR logger is prepended so the live
LR lands in the CSV row before `CSVLogger` writes it):

1. **`LRLoggerCallback`** — injects the live optimizer LR into each epoch's logs.
2. **Common callbacks** (`create_common_callbacks`): `ModelCheckpoint`
   (`best_model.keras`), `CSVLogger` (`training_log.csv`), `TensorBoard`, and
   `EpochAnalyzerCallback` (only with `--analyzer`). `EarlyStopping` is **stripped**
   when patience ≤ 0 (the default).
3. **`NoiseSigmaCurriculumCallback`** — §5.3.
4. **`DenoisingVisualizationCallback`** — every `--viz-freq` (default 5) epochs,
   saves a clean/noisy/denoised grid over a **fixed** validation batch at three
   *fixed reference* noise regimes (decoupled from the moving curriculum, so the
   panels are comparable across epochs: additive σ₂₅₅ = 15/25/50). Also writes a
   6-panel `training_dashboard.png` (MSE, MSE-log, PSNR, MAE, σ_max, LR) every epoch,
   including an epoch-0 untrained baseline.

### 6.2 Bottleneck monitor  (`ConvUnextBottleneckMonitorCallback`, with `--expose-bottleneck`)

A purpose-built monitor for the deepest latent — it catches pathologies (dead
channels, collapsing or saturating activations) that are **invisible in the loss
curve**. Every `--viz-freq` epochs it emits, over a fixed val batch:

- **Two 8×8 feature-map grids** (up to `max_featuremap_channels=64`):
  `…_bottleneck_first.png` shows the *first* 64 channels by fixed index (a temporal
  anchor — the same channel in the same tile every epoch), and `…_bottleneck_energy.png`
  shows the *top-64 by energy*, recomputed each epoch (each tile titled with its
  channel index).
- **A fixed absolute color scale** (`vmin=-3, vmax=3`) on every tile. This is the
  whole point: with per-tile autoscaling, a channel whose activation is *shrinking*
  looks identically saturated epoch to epoch. A fixed scale makes a magnitude change
  read as a color change — so you can actually *see* a channel dying.
- **Cumulative health curves** (`bottleneck_health.png`, overwritten each time):
  mean, std, L2 norm, dead-unit fraction, and sparsity vs. epoch. For a bias-free
  net the mean should sit near zero; a collapsing std signals vanishing capacity.

It receives the **full two-output model explicitly** (not `self.model`, which is the
single-output training view), the `on_train_end` "final" emit is de-duplicated to
keep the x-axis monotone, and the whole emit is wrapped in try/except so a plotting
hiccup never takes down training.

---

## 7. Running it

Always run through the venv with a non-interactive matplotlib backend (headless/
remote safe), as a module from `src/`:

```bash
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_convunext_denoiser [args]
```

Outputs land in repo-root `results/<experiment-name>/` (model checkpoints, CSV log,
TensorBoard, visualization PNGs, `config.json`, `training_history.json`).

### 7.1 Examples

```bash
# Mechanism smoke test (tiny/2-epoch/64px/constant-LR) — verifies the wiring end to end
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_convunext_denoiser --smoke

# Standard base run. base@256 needs batch 4 on a 24GB RTX 4090 (default 16 will OOM).
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_convunext_denoiser \
    --variant base --batch-size 4 --epochs 100

# Large variant, reduced batch, V2 blocks (GRN), bottleneck monitoring on
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_convunext_denoiser \
    --variant large --batch-size 2 --convnext-version v2 --expose-bottleneck

# Multiplicative / composite noise experiments (documented approximations, §5.2)
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_convunext_denoiser --multiplicative-noise
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_convunext_denoiser \
    --composite-noise --composite-additive-ratio 0.5

# Rebuild the dashboard PNG from a finished experiment dir — no GPU, no training
MPLBACKEND=Agg .venv/bin/python -m train.bfunet.train_convunext_denoiser \
    --dashboard results/convunext_denoiser_base_<timestamp>
```

### 7.2 Full CLI reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--variant` | `base` | Model size: `tiny\|small\|base\|large\|xlarge` |
| `--convnext-version` | `v1` | `v1` (strict bias-free) or `v2` (GRN, β is a trainable bias) |
| `--epochs` | `100` | Total training epochs |
| `--batch-size` | `16` | Batch size (reduce for large variants @256; base@256 → 4 on a 4090) |
| `--patch-size` | `256` | Square crop size in pixels |
| `--channels` | `3` | Image channels: `1` or `3` |
| `--patches-per-image` | `8` | Random crops drawn per image |
| `--learning-rate` | `1e-3` | Peak LR for the cosine schedule |
| `--warmup-epochs` | `None` | LR warmup length (default 10% of `--epochs`) |
| `--no-gabor-stem` | *(off)* | Disable the frozen Gabor depthwise stem (stem is ON by default) |
| `--laplacian-pyramid` | *(off)* | Enable Laplacian-pyramid downsample/skip path |
| `--expose-bottleneck` | *(off)* | Expose the bottleneck latent as a 2nd output (enables the monitor) |
| `--analyzer` | *(off)* | Run data-free `ModelAnalyzer` (weights + spectra) during training |
| `--analyzer-freq` | `10` | Run the analyzer every N epochs (with `--analyzer`) |
| `--gabor-filters` | `32` | Gabor filter channels in the stem |
| `--sigma-max-start` | `0.05` | Curriculum start σ_max (in `[−1,+1]` units; ×127.5 for σ₂₅₅) |
| `--sigma-max-end` | `0.5` | Curriculum end σ_max |
| `--curriculum-schedule` | `linear` | σ-widening shape: `linear\|cosine\|exp` |
| `--curriculum-epochs` | `None` | Epochs to widen σ over (default = `--epochs`) |
| `--viz-freq` | `5` | Save denoising / bottleneck grids every N epochs |
| `--viz-samples` | `8` | Image columns in the eval grid |
| `--dashboard` | `None` | Rebuild the dashboard PNG from an experiment dir and exit (no training) |
| `--max-train-files` | `None` (→10000) | Cap on sampled training image paths |
| `--max-val-files` | `None` (→500) | Cap on validation image paths |
| `--steps-per-epoch` | `None` | Override the auto epoch length (`files·patches // batch`) |
| `--validation-steps` | `None` (→100) | Validation batches per epoch |
| `--output-dir` | `results` | Root results directory |
| `--experiment-name` | `None` | Override the auto timestamped name |
| `--gpu` | `None` | GPU device id |
| `--multiplicative-noise` | *(off)* | Use `y=x·(1+N·σ)` instead of additive AWGN |
| `--composite-noise` | *(off)* | Use `y=x·n+a`; takes precedence over `--multiplicative-noise` |
| `--composite-additive-ratio` | `0.5` | Composite mode: `σ_a = ratio · σ_m` (must be > 0) |
| `--smoke` | *(off)* | Tiny end-to-end mechanism check (few steps/epochs, constant LR) |

---

## 8. Caveats & open questions

- **V2 is only approximately bias-free.** GRN's `β` is a trainable additive offset.
  How far it drifts from zero on COCO+DIV2K, and whether that measurably erodes the
  unseen-σ generalization, is unmeasured. Use **V1** when you need the exact
  homogeneity guarantee; reach for V2 when you want GRN's capacity and can tolerate
  the approximation.
- **Multiplicative / composite noise are documented approximations**, not theorems.
  The `2σ²·y` shrink term is genuinely unrepresentable by a strictly homogeneous
  network. Treat these paths as experiments; future work is σ-conditioning or a
  controlled additive-shrink head (see the multiplicative research note).
- **The Gabor stem's throughput cost is unprofiled.** It adds zero parameters but a
  real depthwise conv; the base@256/batch-4 regime has not been benchmarked with vs.
  without it.

---

## 9. References

- **Mohan, Kadkhodaie, Simoncelli, Fernandez-Granda — "Robust and Interpretable
  Blind Image Denoising via Bias-Free Convolutional Neural Networks", ICLR 2020.**
  The empirical foundation for bias-free σ-generalization.
- **Miyasawa (1961) / Tweedie's formula** — the `D(y) = y + σ²∇log p(y)` identity.
- **Liu et al. — "A ConvNet for the 2020s" (ConvNeXt V1)** and **Woo et al. —
  "ConvNeXt V2" (GRN)**.
- Repo-internal:
  - [`research/miyasawas_theorem.md`](../../../research/miyasawas_theorem.md) — additive derivation, bias-free requirements.
  - [`research/miyasawas_theorem_multiplicative.md`](../../../research/miyasawas_theorem_multiplicative.md) — multiplicative/composite relations, bias-free tension, gSURE audit.
  - [`bias_free_denoisers/README.md`](../../dl_techniques/models/bias_free_denoisers/README.md) — model-level architecture tables and V1/V2 deep dive.
  - Source: `dl_techniques/models/bias_free_denoisers/bfconvunext.py`,
    `dl_techniques/callbacks/noise_sigma_curriculum.py`,
    `dl_techniques/callbacks/convunext_bottleneck_monitor.py`,
    `dl_techniques/initializers/gabor_filters_initializer.py`.

# Video VAE + V-JEPA + SIGReg: Refined Specification for dl\_techniques

Nov 17, 2026 · @Nikolas Markou

## 1. Scope and framework fit

Build the model as a new package under `src/dl_techniques/models/world_models/`, in Keras 3.8 on the TensorFlow 2.18 backend, reusing the SIGReg, JEPA, ConvNeXt and transformer components already in the repo. The original plan is written in PyTorch (`torch.einsum`, `.item()`, `optimizer.zero_grad()`); none of that code ports as-is.

House conventions that shape the implementation:

- `import keras` and call `keras.ops.<fn>`, never `from keras import ops` (enforced repo-wide).
- Channels-last video tensors `(B, T, H, W, C)` everywhere; the original spec's input layout already matches.
- Every layer and model registered with `@register_dl_technique`, full `get_config()`, sub-layers created in `__init__`, weights in `build()`.
- Losses added with `self.add_loss(...)` inside `call()` and `compile(loss=None)`, as `LeVJEPATrainingModel` does; a custom `train_step` only where an EMA update is needed (as `VideoJEPA` does).
- Logging through `dl_techniques.utils.logger`, Sphinx `:param:` docstrings for `models/`, tests mirrored under `tests/`.

### Components to reuse

| Need | Existing component | Path |
| --- | --- | --- |
| SIGReg loss | `SIGRegLayer(knots, num_proj, seed, normalize_by_n)` | `regularizers/sigreg.py` |
| SIGReg projector head | `LeVJEPAProjector` (Dense, BN, GELU, Dense) | `models/world_models/levjepa/projector.py` |
| Multiview SIGReg training pattern | `LeVJEPATrainingModel` | `models/world_models/levjepa/training.py` |
| Masked latent prediction + EMA target | `VideoJEPA`, `VideoJEPAPredictor`, `TubeMaskGenerator` | `models/world_models/video_jepa/` |
| Token dropping for context encoder | `random_token_drop`, `build_block_causal_mask` | `models/world_models/levjepa/masking.py` |
| Frame encoder | ConvNeXt V1, `include_top=False` returns `(B, H/32, W/32, dims[-1])` | `models/vision/convnext/convnext_v1.py` |
| Transformer block | `TransformerLayer` (attention, norm, FFN via factories) | `layers/transformers/transformer.py` |
| Variable-length positions | `video_rope.py`, `multi_axis_rope.py`, `patch_embed_3d.py` | `layers/embedding/` |
| Reparameterization | `Sampling` | `layers/generative/sampling.py` |
| EMA weights | `EMAShadowCallback` | `callbacks/ema_shadow_callback.py` |
| LR schedule | `WarmupSchedule` | `optimization/warmup_schedule.py` |
| Recon metrics | PSNR, SSIM | `metrics/psnr_metric.py`, `metrics/ssim_metric.py` |
| Video data | BDD100K and synthetic drone loaders, multi-crop video transform | `datasets/bdd100k_video.py`, `datasets/synthetic_drone_video.py`, `datasets/vision/multi_crop_video.py` |

Proposed package name: `models/world_models/video_vae_jepa/`, with trainer `src/train/video_vae_jepa/`.

Side note: the `SIGRegLayer` docstring attributes SIGReg to LeWM (Sobal et al., 2024). The method originates in LeJEPA (Balestriero and LeCun, 2025, arXiv:2511.08544); worth citing both.

## 2. Review of the original plan

The original SIGReg code does not compute SIGReg, the latent cannot carry N frames, and the JEPA branch has no defined target; 18 issues are fixed below, ordered by impact.

| # | Area | Problem in original | Fix in this spec |
| --- | --- | --- | --- |
| 1 | SIGReg | No random projections. `einsum('k,md->km', t, z)` sums over D, so it tests one fixed direction (the all-ones vector, unnormalized, variance D). This is not a sketched test and cannot detect dimensional collapse. | Sample M=1024 unit directions per step and test every 1D marginal (`SIGRegLayer`). |
| 2 | SIGReg | `normalize=True` L2-normalizes embeddings before matching N(0, I). Unit-norm vectors in D=256 have projected variance 1/D, so the target is unreachable and the loss fights the constraint. | No L2 normalization before SIGReg. |
| 3 | SIGReg | Integrand is w(t)·\|ECF−φ\| (not squared); precomputed `weights` are unused. | Epps-Pulley integrand w(t)·\|ECF−φ\|², real and imaginary parts. |
| 4 | SIGReg | K=100 knots on \[0, 5\]. Upstream uses 17 knots on \[0, 3\]; extra knots add cost, not accuracy, because the Gaussian window decays fast. | `knots=17`, t in \[0, 3\] (layer default). |
| 5 | SIGReg | Frames of one clip are flattened into B·N "independent" samples; they are strongly correlated, so the statistic over-counts samples. | Choose the sample axis explicitly per site (Section 4.3) and report effective sample size. |
| 6 | SIGReg | Batch 16 for the CLS site: an ECF over 16 samples is noisy and carries O(1/N) bias. LeJEPA reports stability for batches of 128 and up. | CLS-site SIGReg only with global batch ≥ 64 (cross-replica), else token-level site only. |
| 7 | SIGReg site | Option D regularizes predictor outputs. These are regression outputs pulled toward targets; SIGReg there conflicts with the prediction loss and leaves the encoder unregularized. | Drop. Regularize encoder embeddings (the targets) instead. |
| 8 | Encoder | ConvNeXt V1 base ends at 1024 channels, not 256, and is about 88M parameters per frame pass. In this repo, `strides` (default 4) sets both the stem and every inter-stage downsample, so 4 stages give total stride 4⁴ = 256, not the paper's 32: a 224 px frame collapses to 1×1. | Custom ConvNeXt V1 with `strides=4, depths=[3, 9], dims=[128, 256]` (stride 16, 256 channels natively), see Section 3.2. |
| 9 | Encoder | Per-frame global pooling destroys spatial patches, yet the JEPA head "predicts masked patch embeddings" with (2, 16, 16) tubelets. Both cannot hold. | Keep the stride-16 feature map as spatial tokens (8×8 at 128 px), mask on the (T, h, w) token grid. |
| 10 | Transformer | No depth, FFN width or positional encoding given; a learned position table cannot handle arbitrary N. | Depth 6, FFN 1024, pre-norm, 3-axis RoPE (time, h, w). |
| 11 | VAE | One 256-d z for a whole clip must reconstruct N full frames: blurry output, posterior collapse risk, and capacity independent of N. | Per-frame spatial latents z\_t plus a global latent z\_g from CLS (Section 3). |
| 12 | Decoder | "U-Net style" decoder: encoder skip connections bypass the latent, so the decoder ignores z and KL drives z to the prior. | No encoder skips in the decoder. |
| 13 | JEPA | Target embeddings (`masked_patches`) are undefined; no EMA target, no stop-gradient, no predictor spec. | LeJEPA-style: online targets + SIGReg (default); V-JEPA-style EMA target as ablation. |
| 14 | Pipeline | Diagram places the VJEPA head after the pixel decoder. It operates on encoder tokens, in parallel with the VAE branch. | Two branches from one encoder (Section 3 diagram). |
| 15 | Schedules | `vjepa_schedule` cosine formula decreases from end to start (0.1 to 0.01), the opposite of the stated intent, and is active from step 0 although Phase 1 disables VJEPA. | Corrected formula, gated by phase (Section 6). |
| 16 | Loss | `kl_div` is "already weighted by β(t)" and multiplied by β again. Pixel-mean MSE vs per-dim KL also mismatches scales. | Single β; reconstruction summed per frame or Gaussian NLL with learned σ. |
| 17 | Training | Phases in steps (1000 / 5000) vs epochs (20 / 30): one Kinetics-400 epoch at batch 16 is about 15k steps, so the step numbers end Phase 2 inside epoch 1. Sections 9.1 and 3.1 disagree on two vs three phases. | Phases as fractions of total steps (Section 7). |
| 18 | Claims | "Replaces weight decay", "reduces need for perceptual loss", "anomaly detection" are unsupported. SIGReg regularizes activations, not weights. | Removed; stated as hypotheses to test where relevant. |

What survives unchanged: channels-last input, shared per-frame CNN, embed\_dim 256 with 4 heads, learnable CLS token, reparameterized Gaussian latent, KL warm-up, gradient clipping at 1.0, AdamW.

## 3. Architecture

One shared encoder produces a spatio-temporal token sequence that feeds three consumers: the VAE latent heads, the SIGReg projector, and the JEPA context pass with its predictor. The pixel decoder and the JEPA predictor are parallel branches, not a chain.

&#91;embedded content: Video VAE + JEPA + SIGReg · one encoder, three consumers\]

The frame encoder runs once per frame; the transformer runs twice per step (full clip, then visible tokens only), and every loss term ends in one weighted sum.

### 3.1 Shapes (example: T = 8, 128 px, P = 64 tokens per frame)

| Stage | Shape | Note |
| --- | --- | --- |
| Input | `[B, 8, 128, 128, 3]` | float32 in \[0, 1\] |
| Frame features | `[B, 8, 8, 8, 256]` | CNN on `[B·T, H, W, C]`, reshaped back |
| Encoder tokens + CLS | `[B, 1 + 512, 256]` | CLS at index 0 |
| h\_full | `[B, 513, 256]` | after final LayerNorm |
| z\_t (per-frame latent) | `[B, 8, 8, 8, 8]` | d\_l = 8 channels per token; 96× compression per frame |
| z\_g (global latent) | `[B, 256]` | from CLS |
| Reconstruction | `[B, 8, 128, 128, 3]` | plus per-channel log σ |
| Context tokens | `[B, 51, 256]` | 10% of 512 kept at mask ratio 0.9 |
| Predictions | `[B, 461, 256]` | one per masked slot |
| SIGReg S1 input | `(1, 4096, 256)` | random token subset |

### 3.2 Components

**Frame encoder.** `ConvNeXtV1(include_top=False, strides=4, depths=[3, 9], dims=[128, 256], drop_path_rate=0.1)`, applied to `[B·T, H, W, C]`. In this repo `strides` sets the stem and every downsample, so two stages give stride 16 and 256 channels natively, with no projection. Paper-faithful alternative: stem 4 and downsample 2 built from `ConvNextV1Block`, dims \[96, 192, 384\] plus Dense to 256, also stride 16. Add a shape test either way.

**Video token encoder.** Reuse the LeVJEPA encoder machinery (CLS token, 3-axis RoPE from `video_rope.py`, train-time token dropping, final norm, `LeVJEPABlock`) and replace its tubelet patch embed with the frame encoder. 6 blocks, D 256, 4 heads, FFN 1024, stochastic depth 0.1. If `TransformerLayer` is preferred, confirm it can apply an external RoPE inside attention first.

**VAE heads.** Dense(2·d\_l) per spatial token gives μ\_t, log σ²\_t; Dense(2·256) on CLS gives μ\_g, log σ²\_g; log variances clipped to \[−10, 10\]; `Sampling` for both.

**Decoder.** Per frame: z\_t `[8, 8, 8]` → 1×1 conv to 256 → FiLM (γ, β from Dense on z\_g) → 4 upsampling stages of ×2 (ConvNeXt blocks + nearest upsample + conv) → `[128, 128, 3]`, sigmoid mean, per-channel log σ. No encoder skip connections.

**Projector.** `LeVJEPAProjector(hidden_dim=1024, output_dim=256)` on h\_full tokens and CLS, used only for SIGReg. Its BatchNorm needs batch > 1.

**Predictor.** Dense 256 → 192, 4 blocks of width 192 with 4 heads, a learnable mask token, the same 3-axis RoPE, Dense 192 → 256.

**Cross-modal image encoder.** Deferred; with T = 1 the same model already encodes single images.

## 4. SIGReg

Use the existing `SIGRegLayer` unchanged; this spec only decides where it is applied, on which sample axis, and with what weight. Writing a new SIGReg class is not needed.

### 4.1 Correct formulation

SIGReg projects the embeddings onto M random unit directions and runs a univariate Epps-Pulley test on each projection against N(0, 1). By the Cramér-Wold argument, matching every 1D marginal implies matching N(0, I).

```latex
\mathrm{SIGReg}(Z) = \frac{1}{M}\sum_{m=1}^{M} \int_{0}^{t_{\max}} w(t)\,\Big|\hat\varphi_m(t) - e^{-t^2/2}\Big|^2 dt,
\qquad \hat\varphi_m(t) = \frac{1}{N}\sum_{n=1}^{N} e^{\,i t\, a_m^\top z_n},
\qquad w(t) = e^{-t^2/2},\; a_m \sim \mathrm{Unif}(\mathbb{S}^{D-1})
```

The squared modulus splits into (mean cos − φ)² + (mean sin)², integrated with the trapezoid rule. Cost is O(N·D·M) time, linear in batch and dimension. Directions are resampled every step.

### 4.2 Settings

| Parameter | Value | Note |
| --- | --- | --- |
| `knots` | 17 | Upstream default; layer default |
| t range | \[0, 3\] | Fixed inside the layer; symmetric half suffices because real and imaginary parts are both penalized |
| `num_proj` (M) | 1024 | LeJEPA reports robustness for M in \[512, 2048\] |
| `normalize_by_n` | `True` | Multiplies by the sample count, the Epps-Pulley statistic scale used by the LeJEPA reference config and by `LeVJEPATrainingModel`; required for the published λ range to apply |
| `seed` | `None` in training, fixed in tests | Under multi-GPU, all replicas must draw the same directions per step |
| dtype | `float32` | Construct with `dtype="float32"` and cast inputs; cos/sin of t·x in float16 loses precision |
| Weight λ\_sig | 0.02 (search 0.01 to 0.1) | LeJEPA recommends 0.05 and reports stability over \[0.01, 0.1\]; the repo's LeVJEPA uses 0.02 |
| L2 normalization | None | See review item 2 |

LeJEPA writes the objective as (1 − λ)·L\_pred + λ·SIGReg. This spec keeps separate weights because the VAE terms also enter the sum (Section 6).

### 4.3 Application sites

| Site | Tensor | Shape passed to `SIGRegLayer` | Status |
| --- | --- | --- | --- |
| S1 Encoder tokens | `projector(h)`, full-clip encoder output tokens, `[B, T, P, D]` | `(1, K, D)`, K = random subset of all B·T·P tokens, K ≤ 4096 | Primary |
| S1b Per-frame-index variant | same | `(T, B·P, D)`: one statistic per frame index, averaged | Ablation |
| S2 CLS token | `projector(h_cls)`, `[B, D]` | `(1, B, D)` | Only with global batch ≥ 64 |
| S3 Aggregate posterior | sampled `z_g`, `[B, d_g]` | `(1, B, d_g)` | Optional; matches q(z) to the prior, complementing per-sample KL (InfoVAE / WAE idea) |
| S4 CNN tokens before transformer | `[B, T, P, D]` | as S1 | Ablation |
| Predictor outputs |  |  | Dropped (review item 7) |

Tokens within a clip are correlated, so S1 regularizes the token marginal over the dataset, not per-clip geometry; its effective sample size is below K. That is acceptable for collapse prevention and is what the repo's `VideoJEPA` already does.

### 4.4 Keras wrapper for S1

```python
import keras
from dl_techniques.regularizers.sigreg import SIGRegLayer
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.models.video_vae_jepa.token_sigreg")
class TokenSIGReg(keras.layers.Layer):
    """SIGReg over a random subset of all tokens in the batch."""

    def __init__(self, max_samples=4096, knots=17, num_proj=1024, **kwargs):
        super().__init__(dtype="float32", **kwargs)
        self.max_samples = max_samples
        self.knots = knots
        self.num_proj = num_proj
        self.sigreg = SIGRegLayer(
            knots=knots, num_proj=num_proj, normalize_by_n=True,
            dtype="float32", name="sigreg",
        )

    def call(self, tokens):
        d = tokens.shape[-1]
        flat = keras.ops.cast(keras.ops.reshape(tokens, (-1, d)), "float32")
        n = keras.ops.shape(flat)[0]
        idx = keras.random.shuffle(keras.ops.arange(n))[: self.max_samples]
        sample = keras.ops.take(flat, idx, axis=0)
        return self.sigreg(keras.ops.expand_dims(sample, 0))  # (1, K, D) -> scalar

    def get_config(self):
        config = super().get_config()
        config.update(max_samples=self.max_samples, knots=self.knots,
                      num_proj=self.num_proj)
        return config
```

Keep K constant across steps (fixed T and P per bucket, or always subsample to exactly K), because `normalize_by_n=True` scales the loss by K.

### 4.5 Unit tests

- Samples from N(0, I), D=256, K=4096: statistic near its null value; record it as the reference floor.
- 0.1·N(0, I) (shrunk) and rank-8 embeddings in D=256 (dimensional collapse): statistic far above the floor.
- Constant embeddings (complete collapse): maximal statistic, finite gradients.
- Mixed-precision model: SIGReg output is float32 and gradients are finite.
- Serialization round-trip of `TokenSIGReg` inside the model.

## 5. JEPA objective

Default: LeJEPA-style masked latent prediction, with targets from the same online encoder on the full clip, no EMA teacher, and collapse prevented by SIGReg site S1. V-JEPA's EMA teacher with stop-gradient is the main ablation.

### 5.1 Forward passes per step

1. Full pass: encoder on all T·P tokens + CLS, giving `h_full [B, 1+T·P, D]`. Feeds the VAE branch, SIGReg sites and JEPA targets.
2. Context pass: same encoder weights on visible tokens only (masked tokens dropped before the transformer, as `random_token_drop` does in LeVJEPA), giving `h_ctx`.
3. Predictor: narrow transformer, depth 4, width 192 (Dense 256→192 in, 192→256 out), 4 heads. Input = `h_ctx` + one learnable mask token per masked slot, all with 3-axis RoPE of their (t, y, x) position. Output = predictions at masked slots.

The CNN runs once per frame; only the transformer runs twice. Extra cost is roughly the transformer on 10 to 25% of tokens.

### 5.2 Loss

```latex
L_{\mathrm{pred}} = \frac{1}{|\mathcal{M}|}\sum_{j \in \mathcal{M}} \big\| \mathrm{Pred}(h_{\mathrm{ctx}})_j - \mathrm{tgt}(h_{\mathrm{full}})_j \big\|_1
```

- `tgt` = identity (default, gradients flow into targets, LeJEPA) or `stop_gradient` (flag `target_stop_gradient=True`).
- L1 rather than MSE: V-JEPA reports better stability with L1 for feature regression.
- Targets are the encoder output after its final norm, before the SIGReg projector.

### 5.3 Masking

| Mode | When | Scheme | Ratio |
| --- | --- | --- | --- |
| Spatio-temporal multi-block (default) | Spatial tokens, T ≥ 4 | Union of 8 short-range blocks (spatial scale 0.15) and 2 long-range blocks (scale 0.7), each a tube spanning all frames, aspect ratio in \[0.75, 1.5\] | about 0.9 (V-JEPA); 0.75 for small grids |
| Spatial only (I-JEPA style) | T = 1 or 2 | Same blocks on a single frame | 0.6 to 0.75 |
| Temporal block | Pooled frame tokens (P = 1) | Contiguous frame spans, or the last 25% of frames (causal) | 0.5 to 0.75 |
| Off | T·P < 16 | No JEPA loss; VAE + SIGReg only |  |

Tubes spanning all frames stop the model from copying a masked patch from the same location in an adjacent frame. Start from `TubeMaskGenerator` (`video_jepa/masking.py`), which already produces time-invariant `[B, H_p, W_p]` masks, and extend it to the multi-block union.

Token dropping needs a fixed kept count per batch: sample the union mask, then trim or pad masked positions per sample to exactly round(ratio·T·P).

### 5.4 EMA ablation

Target encoder = EMA of the online encoder, momentum cosine-scheduled 0.998 to 1.0, stop-gradient on targets, updated in a custom `train_step` after the optimizer step, following `VideoJEPA`. `EMAShadowCallback` only keeps a checkpoint shadow, so it is not a substitute for a target encoder.

## 6. Losses and schedules

One total loss, with the VAE terms normalized per pixel so that all four terms sit at order 1 and the published SIGReg weight range applies; SIGReg is on whenever the JEPA loss is on.

### 6.1 Total loss

```latex
L = \underbrace{\frac{1}{HWC}\Big(\textstyle\sum_{\text{pixels}} \mathrm{NLL}(x \mid z) + \beta(s)\,\mathrm{KL}\Big)}_{L_{\mathrm{vae}}}
\;+\; \lambda_{j}(s)\, L_{\mathrm{pred}}
\;+\; \lambda_{\mathrm{sig}}(s) \sum_{k} w_k\, \mathrm{SIGReg}_k
```

- NLL: Gaussian with one learnable log σ per output channel (calibrated decoder); with σ fixed this reduces to summed MSE. Summing over pixels before dividing keeps the reconstruction-to-KL ratio of a true ELBO, which per-pixel mean MSE with per-dim KL does not.
- KL: closed-form Gaussian KL summed over latent dims for z\_t (all frames) and z\_g, averaged over the batch. Optional free bits: 0.05 nats per latent channel, clamped per channel averaged over the batch.
- β applied once; the model returns raw KL.
- Site weights w\_k: S1 = 1.0; S2 = 0.5 and S3 = 0.5 when enabled.
- Compute KL, NLL and SIGReg in float32 under `mixed_float16`.

### 6.2 Schedules

s = step / total\_steps. Windows are fractions so that they scale with run length.

| Weight | Value | Shape | Window |
| --- | --- | --- | --- |
| β (KL) | 0 → 1.0 | Linear | s in \[0, 0.05\] |
| λ\_j (JEPA) | 0 until s = 0.02, then 0 → 1.0 | Cosine ramp up | s in \[0.02, 0.07\] |
| λ\_sig (SIGReg) | 0.02 | Constant, gated: on exactly when λ\_j > 0 | from s = 0.02 |
| Learning rate | 0 → 3e-4 → 1e-6 | Linear warm-up, then cosine | warm-up s in \[0, 0.05\] |
| Fine-tune LR | 0.1 × pretrain peak | Cosine | Phase B |

Why gate λ\_sig with λ\_j: the reconstruction loss already prevents complete collapse of the encoder; the JEPA loss is what introduces the collapse incentive, so SIGReg must be active from its first step. The original plan ramped SIGReg from 0 over 4000 steps while JEPA was already on, which is the window where collapse starts. LeJEPA itself uses no SIGReg schedule.

Corrected cosine ramp (the original formula runs backwards):

```python
import math

def cosine_ramp(s, s0, s1, start, end):
    """0 before s0; start -> end on a half cosine over [s0, s1]; end after."""
    if s < s0:
        return 0.0
    p = min((s - s0) / max(s1 - s0, 1e-12), 1.0)
    return start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * p))
```

### 6.3 Keras implementation of scheduled weights

Loss weights live as non-trainable `keras.Variable`s on the model so that changing them does not retrace `call()`; a callback assigns them each batch.

```python
import keras


class LossWeightScheduler(keras.callbacks.Callback):
    def __init__(self, total_steps, schedules):
        super().__init__()
        self.total_steps = total_steps
        self.schedules = schedules  # {"beta": fn(s), "lambda_jepa": fn(s), ...}
        self.step = 0

    def on_train_batch_begin(self, batch, logs=None):
        s = self.step / self.total_steps
        for name, fn in self.schedules.items():
            getattr(self.model, name).assign(fn(s))
        self.step += 1

# in the model __init__:
# self.beta = keras.Variable(0.0, trainable=False, name="beta")
# self.lambda_jepa = keras.Variable(0.0, trainable=False, name="lambda_jepa")
# self.lambda_sig = keras.Variable(0.0, trainable=False, name="lambda_sig")
```

Log each unweighted term and each weight with `keras.metrics.Mean` trackers, as `LeVJEPATrainingModel` does.

## 7. Training regime

One joint pre-training phase with short warm-ups replaces the original VAE-then-VJEPA split; phase boundaries are fractions of the step budget, and the three-phase variant becomes an ablation.

### 7.1 Phases

| Phase | Steps | Objective | Notes |
| --- | --- | --- | --- |
| A. Joint pre-train | s in \[0, 1\] of the pre-train budget | L\_vae + λ\_j·L\_pred + λ\_sig·SIGReg, schedules from Section 6.2 | VAE-only during s < 0.02 acts as the original "Phase 1" |
| B. Fine-tune (optional) | 10 to 20% of A | Task-dependent: reconstruction-focused (λ\_j → 0.1, keep λ\_sig) or probe-focused (freeze encoder) | LR 0.1× peak |
| C. Evaluation |  | Frozen encoder probes, reconstruction and generation metrics | Section 8 |

Ablation: original three-phase schedule with VAE-only for the first 20% of steps and a linear SIGReg ramp.

### 7.2 Configuration

| Component | Value | Note |
| --- | --- | --- |
| Resolution | 128 × 128 (start), 224 later | 8×8 = 64 tokens per frame at stride 16 |
| Frames per clip T | Buckets {4, 8, 16} | Fixed T within a batch |
| Batch size | 16 clips per GPU | T=8 gives 8,192 tokens per batch; S1 subsamples K=4,096 |
| Optimizer | AdamW, peak LR 3e-4, weight decay 0.05 (1e-4 to 0.05 search) | No decay on norms, biases, CLS, mask tokens |
| LR schedule | `WarmupSchedule`, 5% linear warm-up, cosine to 1e-6 |  |
| Gradient clipping | Global norm 1.0 |  |
| Precision | `mixed_float16`; KL, NLL, SIGReg in float32 |  |
| Stochastic depth | 0.1 in the transformer |  |
| Pre-train budget | Smoke: 8 steps; synthetic: 2k steps; BDD100K or K400: 100k to 300k steps | Steps, not epochs |

### 7.3 Data

1. Smoke and debugging: `synthetic_drone_video.py` (already used by the LeVJEPA and Video-JEPA trainers).
2. First real run: BDD100K via `bdd100k_video.py`, the dataset the repo's video trainers already support.
3. Benchmark: Kinetics-400 (appearance) and Something-Something v2 (motion), for comparability with V-JEPA probes.

Augmentation: one random-resized crop (scale 0.3 to 1.0) and one horizontal flip per clip, applied identically to all frames; temporal stride jitter (sample stride from {1, 2, 4}) replaces random frame dropping, which would create ragged T within a batch. Inputs float32 in \[0, 1\], matching LeVJEPA's convention.

### 7.4 Variable N

- Encoder: 3-axis RoPE, so any T works at inference; no positional table is sized at build time.
- Training: bucket by T (a `tf.data` `bucket_by_sequence_length` or one dataset per bucket, interleaved), fixed T per batch so shapes are static and XLA-friendly.
- Decoder: per-frame, so it is independent of T.
- JEPA: masking mode switches with T (Section 5.3); T = 1 uses spatial masking only.

## 8. Evaluation, ablations and hyperparameter search

Judge every run on four axes: reconstruction, representation quality, latent geometry, and collapse diagnostics; the ablations isolate one change each against a VAE + JEPA baseline.

### 8.1 Metrics

| Axis | Metric | How |
| --- | --- | --- |
| Reconstruction | PSNR, SSIM | Repo `psnr_metric.py`, `ssim_metric.py`, per frame, averaged |
| Reconstruction | rFVD | FVD between inputs and reconstructions (I3D features, external) |
| Generation | FVD of prior samples | Sample z\_g, z\_t from N(0, I); only meaningful if KL is not collapsed |
| Representation | Linear probe on frozen CLS | Kinetics-400 or SSv2 top-1 |
| Representation | Attentive probe on frozen tokens | Cross-attention pooling with a learned query, V-JEPA protocol |
| Geometry | SIGReg statistic per site, unweighted | Compare to the N(0, I) floor from test 4.5 |
| Geometry | Effective rank (RankMe) of token and CLS embeddings | Entropy of normalized singular values, max = D |
| Geometry | Mean norm, covariance eigenvalue spread | Against N(0, I): norm ≈ √D, spread ≈ 1 |
| VAE health | KL per latent channel, active units | Units with Var\_x(E\[z\|x\]) > 0.01 |
| Collapse | Std of embeddings across the batch, per dim | Alarm if median < 0.1 |

LeJEPA reports that its training loss correlates with linear-probe accuracy; check whether L\_pred + λ\_sig·SIGReg tracks the probe here, which would allow model selection without labels.

### 8.2 Ablations

| ID | Change vs default | Question answered |
| --- | --- | --- |
| A0 | VAE only (λ\_j = λ\_sig = 0) | What JEPA + SIGReg add over reconstruction |
| A1 | VAE + JEPA, EMA target, no SIGReg | V-JEPA-style baseline |
| A2 | Default: VAE + JEPA (online target) + SIGReg S1 |  |
| A3 | A2 without SIGReg | Does online-target JEPA collapse without SIGReg here, or does reconstruction suffice |
| A4 | A2 + EMA target + stop-gradient | Do the two collapse defences stack |
| A5 | A2 + S2 (CLS) | Value of CLS-level SIGReg at batch ≥ 64 |
| A6 | A2 + S3 (aggregate posterior) | SIGReg on q(z) vs KL alone; effect on prior-sample FVD |
| A7 | A2 with S1b (per-frame-index) | Sample-axis choice |
| A8 | Mask ratio 0.75 vs 0.9 | V-JEPA masking ratio on this encoder |
| A9 | Pooled frame tokens (P = 1) vs spatial tokens | Cost vs quality |
| A10 | Attentive pooling instead of CLS | CLS necessity |
| A11 | Original three-phase schedule | Is the VAE-only phase useful |

Run each at 3 seeds on the synthetic set first, then the survivors on BDD100K or K400.

### 8.3 Hyperparameter search space

| Parameter | Values | Default |
| --- | --- | --- |
| λ\_sig | 0.01, 0.02, 0.05, 0.1 | 0.02 |
| λ\_j (after ramp) | 0.3, 1.0, 3.0 | 1.0 |
| β\_max | 0.25, 0.5, 1.0 | 1.0 |
| `num_proj` | 256, 1024 | 1024 |
| S1 sample count K | 1024, 4096 | 4096 |
| Mask ratio | 0.75, 0.9 | 0.9 |
| Transformer depth | 4, 6, 8 | 6 |
| Per-frame latent channels d\_l | 4, 8, 16 | 8 |
| Peak LR | 1e-4, 3e-4, 5e-4 | 3e-4 |

`knots` and the t range are fixed: LeJEPA reports accuracy within about ±0.5% across them, so they are not worth search budget.

## 9. Implementation roadmap

Six milestones, each ending in a passing scoped test run (`pytest tests/test_models/test_video_vae_jepa*.py`), with the trainer smoke run as the gate before any real data.

Package layout: `src/dl_techniques/models/world_models/video_vae_jepa/` with `config.py`, `frame_encoder.py`, `encoder.py`, `decoder.py`, `predictor.py`, `masking.py`, `model.py`, `README.md`; trainer `src/train/video_vae_jepa/train_video_vae_jepa.py` with `--smoke`.

### M1. Encoder

- [ ] `FrameEncoder`: custom ConvNeXt V1 (`include_top=False`), shape test that the output grid is H/16 × W/16 × 256
- [ ] `VideoTokenEncoder`: CLS + 3-axis RoPE + 6 × `TransformerLayer` (pre-norm), accepts any T
- [ ] Serialization round-trip and variable-T tests (T = 1, 4, 8, 16)

### M2. VAE branch

- [ ] Latent heads for z\_t (per token, d\_l channels) and z\_g (from CLS), using `Sampling`
- [ ] Per-frame decoder with z\_g FiLM conditioning, no encoder skips
- [ ] Gaussian NLL with learnable log σ, KL, free bits; per-pixel normalization as in 6.1
- [ ] Overfit test: one batch to PSNR > 30 dB

### M3. JEPA branch

- [ ] Multi-block tube masks with fixed kept count (extend `TubeMaskGenerator`)
- [ ] Context pass via token dropping; predictor with mask tokens and RoPE positions
- [ ] L1 loss on masked slots; `target_stop_gradient` flag; EMA target variant with custom `train_step`

### M4. SIGReg and loss assembly

- [ ] `TokenSIGReg` (Section 4.4) for S1, plus S2 and S3 behind config flags
- [ ] Unit tests from 4.5
- [ ] Loss weights as `keras.Variable`s, `LossWeightScheduler` callback, per-term trackers
- [ ] Mixed-precision test: all loss terms float32, gradients finite

### M5. Trainer and first runs

- [ ] Trainer CLI mirroring `train_levjepa.py`: `--dataset {synthetic_drone,bdd100k}`, `--smoke`, `WarmupSchedule`, CSV logging
- [ ] Smoke run end to end (`MPLBACKEND=Agg`)
- [ ] Synthetic run: A0, A2, A3 at 3 seeds
- [ ] Probe and geometry evaluation scripts (Section 8.1)

### M6. Scale and document

- [ ] BDD100K or K400 runs for surviving ablations
- [ ] `README.md` and `AGENTS.md` for the package; register in `REPO_MAP.md`
- [ ] Optional: cross-modal image encoder projection (deferred from the original plan)

## 10. Open questions and risks

| # | Question or risk | How it gets resolved |
| --- | --- | --- |
| 1 | Does SIGReg on encoder tokens conflict with the VAE? KL pulls the latent heads toward N(0, I) while SIGReg shapes the token space upstream; the decoder may need non-Gaussian token structure. | A2 vs A0 reconstruction PSNR; if PSNR drops more than 1 dB, move S1 to the projector output only (already the default) or lower λ\_sig. |
| 2 | Is online-target JEPA stable here without EMA? LeJEPA's evidence is mostly multi-view images, not masked video. | A3 vs A2 vs A4. |
| 3 | Effective sample size of S1 with correlated tokens. | Compare S1 and S1b; track SIGReg statistic vs the N(0, I) floor. |
| 4 | Posterior collapse of z\_g when z\_t already carries the frame content. | Active-unit count for z\_g; reserve z\_g for global content via a lower spatial-latent capacity (d\_l = 4) if needed. |
| 5 | Repo ConvNeXt `strides` semantics (one stride for stem and downsamples) deviate from the paper. | Confirm with a shape test; consider a `downsample_strides` argument upstream. |
| 6 | SIGReg attribution in `regularizers/sigreg.py` cites LeWM only. | Add the LeJEPA reference to the docstring. |
| 7 | Multi-GPU: SIGReg directions must match across replicas, and S2 needs the global batch. | Seed directions from a step counter shared by replicas (the stable-pretraining reference does this); all-gather CLS embeddings for S2. |
| 8 | Compute: two transformer passes plus a per-frame decoder at T = 16, 224 px. | Profile at 128 px first; use pooled frame tokens (A9) if over budget. |
| 9 | Deferred from the original plan: cross-modal projection, diffusion or GAN on the latent, anomaly detection from reconstruction error. | Out of scope until M6. |

## 11. References

- [LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544), Balestriero and LeCun, 2025 ([HTML](https://arxiv.org/html/2511.08544v1), [code](https://github.com/rbalestr-lab/lejepa))
- [LeJEPA defaults summary: λ = 0.05, M = 1024, 17 points, stable for λ in \[0.01, 0.1\]](https://www.emergentmind.com/topics/latent-euclidean-jepa-lejepa)
- [stable-pretraining LeJEPA reference: t\_max 3.0, 17 points, 1024 slices, λ 0.02, step-seeded directions for DDP](https://galilai-group.github.io/stable-pretraining/_modules/stable_pretraining/methods/lejepa)
- [V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/pdf/2404.08471), Bardes et al., 2024
- [V-JEPA overview: multi-block masking near 90%, 2×16×16 tubelets, EMA target](https://www.emergentmind.com/topics/v-jepa-vision-transformer)
- [V-JEPA review: L1 loss chosen for stability](https://www.themoonlight.io/review/revisiting-feature-prediction-for-learning-visual-representations-from-video)
- [dl\_techniques repository](https://github.com/NikolasMarkou/dl_techniques): `regularizers/sigreg.py`, `models/world_models/{levjepa,video_jepa,lewm}/`, `models/vision/convnext/`, `layers/transformers/transformer.py`, read at the current `main`
- Kingma and Welling, 2013, Auto-Encoding Variational Bayes
- Liu et al., 2022, A ConvNet for the 2020s
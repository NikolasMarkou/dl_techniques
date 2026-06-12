# Ideogram4 (Keras 3 port)

A faithful **Keras 3 reimplementation of the Ideogram4 neural core**: a
text-to-image **flow-matching DiT** transformer conditioned on precomputed
language-model features, a **Flux2 KL-VAE**, a **logit-normal time schedule +
Euler flow-matching sampler**, a velocity loss, an inference pipeline, and
training code.

> Plan: `plans/plan_2026-06-12_59a18a10/` (iter-1).
> The DiT is conditioned on a **precomputed `llm_features` tensor** — the
> original Qwen3-VL-8B text/vision encoder is **not** reimplemented in Keras
> (decision D1). This is a **trainable architecture with a runnable tiny
> preset**, NOT a drop-in for the released quantized checkpoint. See
> `CLAUDE.md` for the full "what doesn't fit / skipped / changed" report.

---

## 1. Overview

```
prompt --(out of scope)--> llm_features (B, L, llm_features_dim)
                                  │
noise x (B, L, in_channels) ──────┤
                                  ▼
                       Ideogram4Transformer            # packed-stream DiT
                       (Euler integration over t)      # velocity prediction
                                  │
                                  ▼  predicted velocity
                       latent z (B, H, W, z_channels)
                                  │
                                  ▼  AutoEncoder.decode  # Flux2 KL-VAE
                       image (B, H*f, W*f, 3)
```

The transformer regresses a **rectified-flow velocity** `v = x1 - x0` (noise
minus data). Sampling runs a small Euler loop over a logit-normal time grid with
classifier-free guidance, then the VAE decoder maps the final latent to pixels.

---

## 2. Architecture

### Transformer — `transformer.py`

`Ideogram4Transformer` (`keras.Model`) is a **packed-stream, masked-add DiT**.
It consumes a SINGLE self-attention stream that interleaves text tokens (carrying
projected `llm_features`) and image tokens (carrying projected noise `x`); there
is **no cross-attention**. Per-token roles are marked by an integer `indicator`,
and float role-masks gate each contribution before a masked add.

Call schema — a single `dict` (keeps the multi-input model serializable):

| key | shape | meaning |
|-----|-------|---------|
| `llm_features` | `(B, L, llm_features_dim)` | precomputed conditioning |
| `x` | `(B, L, in_channels)` | patchified noise latents |
| `t` | `(B,)` or `(B, L)` | diffusion time in `[0, 1]` |
| `position_ids` | `(B, L, 3)` | integer `(t, h, w)` mRoPE coords |
| `segment_ids` | `(B, L)` | block-diagonal attention segments |
| `indicator` | `(B, L)` | per-token role (`LLM_TOKEN_INDICATOR`=3 / `OUTPUT_IMAGE_INDICATOR`=2) |

Output: `(B, L, in_channels)` velocity, **always float32** (the PyTorch forward
returns `.float()`, replicated even under mixed precision).

Build via the factory:

```python
from dl_techniques.models.ideogram4.transformer import create_ideogram4_transformer
model = create_ideogram4_transformer("tiny")          # or "full"
model = create_ideogram4_transformer("tiny", num_layers=4)   # field overrides
```

### Layers (live under `layers/`, separately tested)

| layer | file | role |
|-------|------|------|
| `Ideogram4MRoPE` | `layers/embedding/multi_axis_rope.py` | 3D `(t,h,w)` multi-axis rotary; static one-hot band-interleave (D-003) |
| `ScalarSinusoidalEmbedding` | `layers/embedding/scalar_sinusoidal_embedding.py` | time `t∈[0,1]` → `emb_dim`; freqs stored as a **non-trainable weight** (serialization-safe) |
| `Ideogram4Attention` | `layers/attention/ideogram4_attention.py` | fused QKV + per-head RMS QK-norm + mRoPE inject + **additive** block-diagonal segment mask + manual SDPA (D-004) |
| `Ideogram4TransformerBlock` | `layers/transformers/ideogram4_block.py` | **4-stream tanh-gated AdaLN** (scale+gate, NO shift) with a **4-RMSNorm sandwich** (post-norm inside the residual) |
| `Ideogram4FinalLayer` | `layers/transformers/ideogram4_block.py` | affine-free LayerNorm + `(1 + AdaLN(silu(c)))` scale + `Dense(in_channels)` velocity head |

### VAE — `vae.py`

`AutoEncoder` (`keras.Model`) is the **Flux2 KL-VAE**, channels-last (NHWC):

- `encode(x) -> (z_mean, z_log_var)` — deterministic; splits the encoder's
  `2*z_channels` output along the channel axis.
- `sample(z_mean, z_log_var) -> z` — KL reparameterization (reused `Sampling`).
- `decode(z) -> image` — at pipeline inference **only the decoder is used**.
- `call(x) -> reconstruction` — `encode -> sample -> decode`.

Building blocks: `ResnetBlock`, `AttnBlock` (bottleneck only), `Downsample`,
`Upsample`, `Encoder`, `Decoder`. Factory: `create_ideogram4_autoencoder`.

### Scheduler / sampler — `scheduler.py`

`LogitNormalSchedule` (logit-normal time warp + log-SNR clamp),
`get_schedule_for_resolution` (resolution-aware mean shift), `make_step_intervals`,
and `SamplerParameters` with named presets `V4_QUALITY_48`, `V4_DEFAULT_20`,
`V4_TURBO_12`. This is **eager / NumPy** (uses `scipy.special.ndtri`/`expit`); it
is NOT part of the differentiable graph and nothing here is serialized.

### Pipeline — `pipeline.py`

`Ideogram4Pipeline` is a plain orchestration class (not a `keras.Model`) that
holds a trained transformer + autoencoder + config and runs **Euler denoise (with
CFG) then decode**. Conditioning is the `llm_features` tensor passed to
`__call__`. By default ONE shared transformer serves both CFG branches; pass
`unconditional_transformer=` to recover PyTorch's two-model form.

### Loss — `losses/flow_matching_velocity_loss.py`

`FlowMatchingVelocityLoss` (registered in `dl_techniques.losses`): plain velocity
MSE between `y_pred` and a supplied `y_true = x1 - x0`. Time-dependent weighting
is deliberately left to the sampling level (the trainer), not the loss.

---

## 3. Reused `dl_techniques` components

| component | source | used for |
|-----------|--------|----------|
| `RMSNorm` | `layers/norms/rms_norm.py` | `llm_cond_norm`, QK-norm, the block's 4-norm sandwich |
| `SwiGLUFFN` | `layers/ffn/swiglu_ffn.py` | the block MLP (configured bias-free, `expansion=1`, `multiple_of=intermediate_size` so the rounded hidden equals `intermediate_size` exactly) |
| `Sampling` | `layers/sampling.py` | VAE KL reparameterization |
| `keras.layers.GroupNormalization` | Keras built-in | VAE `GroupNorm32(groups=32, eps=1e-6)` — NOT the repo norms factory (see CLAUDE.md) |
| upsample (op sequence) | `layers/upsample.py` `nearest_conv2d_3x3` | **D-005**: `upsample()` is a functional-graph builder, not a reusable sub-layer; a thin `Upsample` wrapper (UpSampling2D nearest×2 + Conv2D 3×3 same) byte-identical to that branch was built so the subclassed `Decoder` can OWN it. |

Two layers were deliberately built net-new instead of reusing near-misses
(**D-002**): `ScalarSinusoidalEmbedding` (the existing `TimestepEmbedding` stores
freqs as a plain tensor, breaking `.keras` round-trip) and
`Ideogram4TransformerBlock` (structurally distinct from the repo's
`AdaLNZeroConditionalBlock`).

---

## 4. Configs / presets

Presets live in `config.py` (`PRESETS`); retrieve a `(Ideogram4Config,
AutoEncoderParams)` pair via `get_ideogram4_config(variant)`.

| field | `tiny` (runnable locally) | `full` (defined, NOT runnable / NOT weight-loadable) |
|-------|---------------------------|------------------------------------------------------|
| `emb_dim` | 128 | 4608 |
| `num_layers` | 2 | 34 |
| `num_heads` | 4 | 18 |
| `intermediate_size` | 256 | 12288 |
| `adanln_dim` | 64 | 512 |
| `in_channels` | 32 | 128 |
| `llm_features_dim` | 64 | 4096 × 13 = 53248 |
| `mrope_section` | (4, 3, 3) | (24, 20, 20) |
| `z_channels` / `patch_size` | 8 / 2 | 32 / 2 |
| VAE `ch` / `ch_mult` | 32 / (1, 2) | 128 / (1, 2, 4, 4) |
| VAE `num_res_blocks` / `resolution` | 1 / 32 | 2 / 256 |

`get_ideogram4_config` runs all config invariants (head_dim integer + even,
mRoPE band bound, VAE channel/32 divisibility, `in_channels == z_channels ×
patch_size²`, and `config.z_channels == ae.z_channels`).

---

## 5. Quickstart

Build a tiny transformer:

```python
from dl_techniques.models.ideogram4.transformer import create_ideogram4_transformer
model = create_ideogram4_transformer("tiny")
```

Real smoke-train (the command actually run for this iteration):

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m train.ideogram4.train_ideogram4 \
    --variant tiny --epochs 8 --steps-per-epoch 30 --batch-size 8 --gpu 0 --seed 42
```

> Note: with `CUDA_VISIBLE_DEVICES=1` the only visible device is index `0`, so
> pass `--gpu 0`.

**Observed result**: loss **2.59 → 1.67** over 8 epochs on an RTX 4070, **no
NaN**. Artifacts written under
`results/ideogram4/ideogram4_tiny_20260612_154608/` (`best_model.keras`,
`config.json`, `training_log.csv`). The trainer slices image-token velocities
`[:, T:]` for the velocity MSE.

The trainer (`train/ideogram4/train_ideogram4.py`) generates a synthetic
packed-index flow-matching dataset; CLI flags include `--variant`, `--epochs`,
`--steps-per-epoch`, `--batch-size`, `--num-text-tokens`, `--grid-h`,
`--grid-w`, `--learning-rate`, `--mixed-bfloat16`, `--gpu`, `--seed`.

---

## 6. Tests

Scoped test directories (do **not** run the full ~1.5h suite as a regression
check — scope to these):

```
tests/test_layers/test_embedding/test_multi_axis_rope.py
tests/test_layers/test_embedding/test_scalar_sinusoidal_embedding.py
tests/test_layers/test_attention/test_ideogram4_attention.py
tests/test_layers/test_transformers/test_ideogram4_block.py
tests/test_models/test_ideogram4/        # test_config, test_transformer, test_vae, test_scheduler, test_pipeline
tests/test_losses/test_flow_matching_velocity_loss.py
tests/test_train/test_ideogram4/test_train_smoke.py
```

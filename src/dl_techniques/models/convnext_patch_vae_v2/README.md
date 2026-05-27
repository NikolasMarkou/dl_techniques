# ConvNeXt Patch-Level VAE v2 — Multi-Task Pretraining Backbone

A resolution-agnostic ConvNeXt-based VAE extended for multi-task
pretraining. Drop-in successor to V1 (`dl_techniques.models.convnext_patch_vae`):
when all multi-task knobs are off it is identical to V1 (modulo random
init).

> Plan: `plans/plan_2026-05-27_4a444b14/` (iter-1).
> V1 is untouched.

---

## What's new vs V1

| Capability | V1 | V2 |
|------------|----|----|
| VAE recon + KL + SIGReg | ✓ | ✓ |
| Per-patch latent grid (resolution-agnostic) | ✓ | ✓ |
| Hierarchical (L1+L2 + conditional prior) | ✓ | deferred to V2.hier follow-up |
| LPIPS perceptual loss | — | ✓ (configurable) |
| MAE-style masked-recon pretext (SimMIM) | — | ✓ (configurable mask ratio) |
| Attention-pool classification head | — | ✓ |
| Bilinear-upsample segmentation head | — | ✓ |
| `xl` preset (~10M params) | — | ✓ |
| Pre-bottleneck feature tap for head reuse | — | ✓ |

---

## Architecture

```
x : (B, H, W, C)
    │
    ▼  Stem Conv2D(embed_dim, k=s=patch_size)
    │
    ▼  LayerNormalization
    │
    ▼  [SimMIM mask, training-only, ratio=r]   ← NEW
    │
    ▼  encoder_depth × ConvNeXtV2Block
    │
   pre_bottleneck (B, Hp, Wp, embed_dim)
    │
    ├──────────────────┬──────────────────┬──────────────────┐
    ▼                  ▼                  ▼                  ▼
 mu_head (1×1)    AttnPool head    SegmentationHead    (future heads...)
 log_var_head     → logits_cls     → logits_seg
    │                  │                  │
    ▼                                                       
 Sampling → z (B, Hp, Wp, latent_dim)
    │
    ▼  Decoder (V1 ConvNeXtPatchDecoder, reused as-is)
    │
   x_hat (B, H, W, C)
    │
    ▼  LPIPS(x, x_hat)  [VGG16 features, frozen]    ← NEW
```

**Loss** (all via `add_loss`):

```
L = recon_unmasked + λ_mae · recon_masked      # SimMIM split if mask active
  + β_kl · KL_per_patch
  + λ_sigreg · SIGReg(z) · (Hp · Wp)
  + λ_lpips · LPIPS(x, x_hat)
  + λ_cls   · CE(y_cls, logits_cls)            # if cls head + label present
  + λ_seg   · CE(y_seg, logits_seg)            # if seg head + label present
```

---

## Presets

| Variant | embed_dim | enc_depth | dec_depth | latent_dim | Backbone params (rough) |
|---------|-----------|-----------|-----------|------------|--------------------------|
| tiny    | 64        | 2         | 2         | 8          | ~120K                    |
| base    | 128       | 4         | 4         | 16         | ~1.0M                    |
| large   | 192       | 6         | 6         | 32         | ~3.5M                    |
| **xl**  | **256**   | **8**     | **8**     | **64**     | **~10M**                 |

`xxl` (384/12/12/128) is reserved for a follow-up once `xl` is validated.

---

## Quick start

### V1-equivalent (multi-task OFF)

```python
from dl_techniques.models.convnext_patch_vae_v2.model import (
    create_convnext_patch_vae_v2,
)

model = create_convnext_patch_vae_v2("base")
model.compile(optimizer="adamw", loss=None, jit_compile=False)
model.fit(x_train, x_train, epochs=50, batch_size=256)
```

### Add LPIPS perceptual (BCE recon, [0,1] images)

```python
model = create_convnext_patch_vae_v2(
    "base",
    recon_loss_type="bce",
    lambda_lpips=0.1,            # weight on perceptual term
    lpips_input_range=(0.0, 1.0),
)
```

### Add SimMIM-style masked recon (encoder learns to in-fill)

```python
model = create_convnext_patch_vae_v2(
    "base",
    mae_mask_ratio=0.5,          # 50% of patches replaced by mask_token
    lambda_mae=1.0,              # weight on masked-patch recon term
)
```

### Add classification head (CIFAR-10)

```python
import tensorflow as tf

model = create_convnext_patch_vae_v2(
    "base",
    use_classification_head=True,
    num_classes_cls=10,
    lambda_cls=1.0,
)

ds = tf.data.Dataset.from_tensor_slices(
    ({"image": x_train, "label_cls": y_train}, x_train)
).batch(256)

model.compile(optimizer="adamw", loss=None, jit_compile=False)
model.fit(ds, epochs=50)
```

### All-on multi-task

```python
model = create_convnext_patch_vae_v2(
    "xl",
    recon_loss_type="bce",
    mae_mask_ratio=0.5,
    lambda_mae=1.0,
    lambda_lpips=0.1,
    use_classification_head=True,
    num_classes_cls=1000,
    lambda_cls=1.0,
    use_segmentation_head=True,
    num_classes_seg=21,
    lambda_seg=1.0,
)
```

---

## Tap points for downstream use

After pretraining, V2 exposes:

| Output | Shape | When to tap | Use case |
|--------|-------|-------------|----------|
| `model.encode(x).mu` | `(B, Hp, Wp, latent_dim)` | After training | Latent-diffusion 1st stage; compressed features |
| `model.encoder(x, output_pre_bottleneck=True)[2]` | `(B, Hp, Wp, embed_dim)` | After training | Rich semantic features for transfer (CLIP/seg/det downstream) |
| `model.sample(num_samples)` | `(N, H, W, C)` | Inference | Unconditional generation |
| `model.sample_from(x, t)` | `(B, H, W, C)` | Inference | Variations around real anchor |

The **pre-bottleneck** tap (full embed_dim feature map) is the recommended
backbone for downstream transfer — it carries richer semantic information
than the KL-bottlenecked `mu`.

---

## API parity with V1

V2 preserves the V1 public-method signatures:
- `encode(x) -> (mu, log_var)` (2-tuple).
- `decode(z) -> x_hat`.
- `sample(num_samples, hp, wp) -> images`.
- `sample_from(x, temperature) -> variations`.
- Forward returns a dict with `reconstruction`, `z`, `mu`, `log_var` (V1
  keys), plus `logits_cls` / `logits_seg` when those heads are active.

---

## Design decisions (anchored in `plans/plan_2026-05-27_4a444b14/decisions.md`)

- **D-001** LPIPS uses lazy frozen VGG16; held inside the loss, weights
  not saved into the model archive.
- **D-002** MAE masking is **SimMIM-style** (post-stem, full grid
  preserved). ConvNeXt blocks need the full grid; canonical MAE
  variable-length sequences are incompatible with the depthwise
  convolutions.
- **D-003** LPIPS loss instance is held by the model (re-instantiated on
  deserialization) — avoids serializing the frozen VGG.
- **D-004** V2 is a separate package; V1 untouched, no risk to V1
  consumers.
- **D-005** iter-1 scope: single-scale V2. Hierarchical V2 + `xxl` are
  explicit follow-ups.
- **D-006** Seg head is unit-tested with synthetic labels in iter-1;
  ADE20K seg-mask data loader is deferred.

---

## Loss component metrics

After training, `history.history` exposes:

| Key | Meaning |
|-----|---------|
| `loss` | Aggregate (sum of all `add_loss` contributions) |
| `recon_loss` | Unmasked recon (V1 equivalent) |
| `mae_loss` | Masked-patch recon (0.0 when MAE off) |
| `kl_loss` | Raw per-patch KL mean |
| `kl_loss_weighted` | `β_kl · KL` |
| `sigreg_loss` | Raw SIGReg statistic |
| `sigreg_loss_weighted` | `λ_sigreg · SIGReg · N` |
| `lpips_loss` | Raw LPIPS (0.0 when LPIPS off) |
| `cls_loss` | Raw classification CE (0.0 when cls off or no label) |
| `seg_loss` | Raw segmentation CE (0.0 when seg off or no label) |

---

## Tests

```bash
.venv/bin/python -m pytest tests/test_models/test_convnext_patch_vae_v2/ \
    tests/test_losses/test_lpips_loss.py -vvv
```

Coverage:
- LPIPS loss: 13 tests (~30s).
- MAE mask utilities: 12 tests (~7s).
- V2 encoder: 13 tests (~12s).
- V2 heads: 13 tests (~10s).
- V2 model: 23 tests (~170s).

**Total**: 74 tests, ~4 minutes wall clock.

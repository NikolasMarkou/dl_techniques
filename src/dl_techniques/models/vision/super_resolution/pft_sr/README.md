# PFT-SR: Progressive Focused Transformer for Super-Resolution

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)
[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-blue.svg)](https://cvpr.thecvf.com/)

An implementation of the **Progressive Focused Transformer (PFT-SR)** architecture in **Keras 3**. This implementation is based on the CVPR 2025 paper by Long et al., who report state-of-the-art performance for the PFT-SR architecture on single image super-resolution benchmarks. This port has not been independently trained or benchmarked, and no performance claim is made for it.

The architecture's core innovation is the **Progressive Focused Attention (PFA)** mechanism, which allows attention maps to be inherited and refined across layers. The paper reports that this lets the network focus on informative features while suppressing redundancy, improving reconstruction quality and computational efficiency.

---

## Table of Contents

1. [Overview: What is PFT-SR and Why It Matters](#1-overview-what-is-pft-sr-and-why-it-matters)
2. [The Problem PFT-SR Solves](#2-the-problem-pft-sr-solves)
3. [How PFT-SR Works: Core Concepts](#3-how-pft-sr-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Model Variants](#7-configuration--model-variants)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Advanced Usage Patterns](#9-advanced-usage-patterns)
10. [Performance Optimization](#10-performance-optimization)
11. [Training and Best Practices](#11-training-and-best-practices)
12. [Serialization & Deployment](#12-serialization--deployment)
13. [Testing & Validation](#13-testing--validation)
14. [Troubleshooting & FAQs](#14-troubleshooting--faqs)
15. [Technical Details](#15-technical-details)
16. [Citation](#16-citation)

---

## 1. Overview: What is PFT-SR and Why It Matters

### What is PFT-SR?

**PFT-SR** is a Vision Transformer designed specifically for the task of Single Image Super-Resolution (SISR). Unlike standard ViTs that treat every layer's attention calculation independently, PFT-SR introduces a temporal-like connection between layers in the depth dimension.

It utilizes **Progressive Focused Attention (PFA)** to multiply the current layer's attention map with the attention map an earlier layer produced, and to keep fewer keys per query as depth grows (a per-layer top-k). This creates a "focusing" effect, where consistent patterns are amplified and irrelevant tokens are dropped as the signal moves deeper into the network.

### Key Innovations of this Implementation

1.  **The official network**: body and tail are ported from the official code ([CVL-UESTC/PFT-SR](https://github.com/CVL-UESTC/PFT-SR), `basicsr/archs/pft_arch.py`): the mean shift, the mirror pad, the patch-embed and final LayerNorms, each stage's conv and residual, the relative position bias, the LePE, the ConvFFN, the two inherited attention chains and the `num_topk` schedule. The top-k runs as a dense map under an inherited boolean mask instead of the official CUDA sparse kernels (same numbers, §4.1). §4.3 lists what still differs.
2.  **Keras 3 Native**: a `keras.Model` subclass built from `keras.ops`.
3.  **The official upsamplers**: `pixelshuffle` (the PFT config), `pixelshuffledirect` (the PFT_light config) and `nearest+conv` (x4 only).
4.  **Factory**: `create_pft_sr` with the variants `light`, `base` and `repo_medium`, each carrying its official upsampler.

### Why PFT-SR Matters

**Standard Transformers (e.g., SwinIR)**:
```
Model: Standard Window Attention
  1. Each block computes attention from scratch.
  2. The network must "rediscover" salient features at every depth.
  3. Computationally wasteful on non-informative background textures.
```

**PFT-SR Solution**:
```
Model: Progressive Focused Transformer
  1. Attention maps are inherited: A_l = normalize(Softmax(Q K^T / sqrt(d) + B) * A_prev),
     A_prev the map of the previous layer with the same window shift.
  2. A per-layer top-k drops all but the strongest keys; later layers attend
     only inside the keys an earlier layer kept.
  3. Acts as a progressive filter, sharpening the network's focus on details
     essential for high-quality reconstruction.
```

---

## 2. The Problem PFT-SR Solves

### The Redundancy of Independent Attention

In super-resolution, high-frequency details (edges, complex textures) are sparse. A standard transformer calculates self-attention over the entire image window at every layer independently. This means the model spends significant capacity processing smooth, low-information areas repeatedly without "remembering" where the important details were in the previous layer.

### The Progressive Solution

PFT-SR mimics the human visual system's ability to focus. By enforcing a constraint where the current attention is modulated by the previous attention, the network effectively performs **spatial filtering** within the attention mechanism itself.

```
┌─────────────────────────────────────────────────────────────┐
│  The PFT Efficiency Gain                                    │
│                                                             │
│  1. Inheritance: If Layer N-1 says "this region is boring", │
│     Layer N is mathematically discouraged from attending    │
│     to it strongly.                                         │
│                                                             │
│  2. Sharpening: If Layer N-1 says "this is an edge",        │
│     Layer N focuses computation on refining that edge.      │
│                                                             │
│  3. Consistency: This enforces structural consistency       │
│     across the depth of the network, leading to sharper     │
│     final images.                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How PFT-SR Works: Core Concepts

### The High-Level Architecture

PFT-SR follows a classic SR network design (Feature Extraction -> Deep Nonlinear Mapping -> Reconstruction) but replaces the body with PFT Blocks.

```
┌──────────────────────────────────────────────────────────────────┐
│                    PFT-SR Architecture                           │
│                                                                  │
│  Input (LR) ───►┌───────────────┐ (mirror pad, minus RGB mean)   │
│                 │ Shallow Feat  │ (Conv2D 3x3, then LayerNorm)   │
│                 └───────┬───────┘                                │
│                         │                                        │
│                 ┌───────▼───────┐                                │
│                 │  PFT Stage 1  │ (PFA layers, conv, + residual) │
│                 └───────┬───────┘                                │
│                         │                                        │
│                       .....  (Multiple Stages)                   │
│                         │                                        │
│                 ┌───────▼───────┐                                │
│                 │ Reconstruction│ (LN, Conv + Global Residual)   │
│                 └───────┬───────┘                                │
│                         │                                        │
│                 ┌───────▼───────┐                                │
│                 │   Upsampler   │ (Pixel Shuffle)                │
│                 └───────┬───────┘                                │
│                         │                                        │
│                 ┌───────▼───────┐                                │
│                 │  Output (HR)  │ (plus RGB mean, cropped)       │
│                 └───────────────┘                                │
└──────────────────────────────────────────────────────────────────┘
```

### The PFT Block Data Flow

The unique aspect is the passing of `attn_map` alongside the feature tensor `x`.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       PFT Block Data Flow                               │
└─────────────────────────────────────────────────────────────────────────┘

Input: features x, and the chain's previous map M_prev and kept keys K_prev
    │
    ▼
1. NORM, QKV, LePE (whole map)
   Q, K, V = Linear(LayerNorm(x)); lepe = GELU(DepthwiseConv_k(V))
   cyclic shift (odd layers of a stage), partition into windows
    │
    ▼
2. PROGRESSIVE FOCUSED ATTENTION (PFA)
   ├─► A = Softmax(Q K^T / sqrt(d) + B [+ shift mask])   (over K_prev only, if any)
   ├─► MERGE: A = (A * M_prev + eps) / (sum(A * M_prev) + eps)   <-- the core
   ├─► TOP-K: keep the num_topk[l] largest per query (if below the width)
   └─► Output = Linear(A @ V + lepe)
    │
    ▼
3. CONVFFN
   u = GELU(fc1(LayerNorm(x))); x = x + fc2(u + GELU(DepthwiseConv_k(u)))
    │
    ▼
Output: features x_out, the map M_curr and the kept keys K_curr
```

---

## 4. Architecture Deep Dive

### 4.1 `PFTLayer` (`blocks.py`)

One official `PFTransformerLayer` with its `WindowAttention`.
-   **Window Attention**: shifted windows (like Swin Transformer) for cross-window exchange. Even layers of a stage use plain windows, odd layers shifted ones (shift `window_size // 2`), with the official -100 mask between shifted regions. A learned relative position bias is added to the scores: one column per head when `embed_dim > 100`, one shared column otherwise, as in the official code.
-   **LePE**: a depthwise conv (kernel `convffn_kernel_size`) plus GELU on V over the whole map, added to `attn @ v`.
-   **Attention inheritance**: called as `y, values, mask = layer(x, prev_values=..., prev_mask=...)`. The model threads two chains through every stage, one for the plain-window layers and one for the shifted ones (the official `pfa_values[shift]`), so each layer inherits from the previous layer of its chain.
-   **Top-k as a mask**: the official code stores the kept keys as indices and multiplies with custom CUDA kernels (`smm_cuda`). This port keeps the dense `(windows, heads, n, n)` map and a boolean mask of the kept keys: the softmax runs over the kept keys only, dropped keys hold exactly 0, and each top-k ranks the kept keys only. The numbers are the same (pinned at float64 against a numpy transcription of the sparse code); the memory is that of dense attention, which the port had before.
-   **ConvFFN**: `fc1`, GELU, `u + GELU(depthwise(u))`, `fc2`.

### 4.2 `PFTSR` (The Model)

A Keras `Model` subclass that orchestrates the stages.
-   **Input**: mirror-padded to a multiple of `window_size` (the official `cat([x, flip(x)])`, so a side must be at least half its padded size), shifted by the DIV2K RGB mean `(0.4488, 0.4371, 0.4040)` (3 channels only) and scaled by `img_range`; the output is divided by `img_range`, shifted back and cropped to `scale` times the input size.
-   **Stages**: `conv_first`, a patch-embed LayerNorm, then each stage (the official `PFTB`): its `PFTLayer`s, a 3x3 conv and a residual around the stage; then a final LayerNorm and `conv_after_body` plus the shallow features. LayerNorm epsilon is torch's 1e-5.
-   **`num_topk`**: one entry per layer. A layer whose entry is below its chain's current width keeps that many keys per query; otherwise it keeps the width. `None` keeps every key (focusing without sparsification).
-   **Tail** (`include_top=True`, the default), the official `PFT.forward` branches with `num_feat = F = 64`. A biased 3x3 conv is written `conv(i -> o)`:
    -   `pixelshuffle` (scale a power of two, or 3): `conv_before_upsample` `conv(embed_dim -> F)` + LeakyReLU(0.01), then `conv(F -> 4F)` + pixel shuffle 2 per factor of two (or `conv(F -> 9F)` + pixel shuffle 3), then `conv_last` `conv(F -> 3)`.
    -   `pixelshuffledirect` (any scale): `conv(embed_dim -> scale^2 * 3)` + pixel shuffle `scale`, nothing after it.
    -   `nearest+conv` (x4 only, as the official `assert`): `conv_before_upsample` + LeakyReLU(0.01), two stages of nearest 2x + `conv(F -> F)` + LeakyReLU(0.2), `conv_hr` + LeakyReLU(0.2), `conv_last`.

    The 0.01 is torch `nn.LeakyReLU()`'s default, passed explicitly (Keras' default is 0.3). The model builds this tail itself rather than through the shared `EnhancementHead`, whose super-resolution path is a stride-`scale` transposed conv that none of the official tails use (the SqueezeNet precedent: a reference head the shared head cannot express is model-owned).
-   **`include_top=False`**: no tail is built, and the model returns the features the tail reads, `conv_after_body(body(x)) + x`, cropped to `(B, H, W, embed_dim)` (no mean added back).

### 4.3 Differences from the official code

Body and tail follow the official `PFT` layer for layer. Pinned by
`tests/test_models/test_pft_sr/`: a parameter hand sum over the official module
list (light and base), a float64 forward equal to a numpy transcription of the
official sparse code (one layer, a chain of layers, and the whole model on a
non-square input that needs padding), the inherited top-k masks, and the mean
shift and mirror pad by value. What still differs, each labelled:

| Official `PFT` | This port | Label |
|:---|:---|:---|
| Top-k via index tensors and the `smm_cuda` sparse kernels | A dense map under an inherited boolean mask | Same numbers, different memory: the dense map costs `(windows, heads, n, n)` per layer, as plain window attention does; the official kernels cost `(windows, heads, n, k)` |
| `torch.topk(..., sorted=False)`: tie order unspecified | TensorFlow `top_k`: ties to the lower index | Identical away from exact ties; on a tie (a flat map) the two can keep different keys |
| torch `PixelShuffle` channel order `(c, i, j)` | `PixelShuffle2D`'s `(i, j, c)` | A fixed permutation of the preceding conv's output channels: same function class, matters only when porting torch weights |
| Conv weights: torch's default init (`kaiming_uniform_(a=sqrt(5))`, uniform bias) | Keras' default (`glorot_uniform`, zero bias) | Initialisation only. Linear layers and the bias table do follow the official `N(0, 0.02)` |
| `upsampler=''` (denoising branch), `resi_connection='3conv'`, `ape=True` | Not built | Not used by either released config |
| No stochastic depth | `drop_path_rate` (SwinIR placement), 0 by default | A repo addition, off by default |
| One input size at a time, computed per call | Static height and width per traced call | The shifted-window mask is built from the static size |

The `repo_medium` variant is this repository's own (no released config, so no
top-k schedule: `num_topk=None`).

Before 2026-10-20 the body was a simplified port: no mean shift, no patch-embed
or final LayerNorm, no per-stage conv or residual, no relative position bias, a
3x3 per-window LePE added to V, focusing that multiplied the pre-softmax scores
of the previous layer (not of the same-shift layer), no top-k, an MLP for the
ConvFFN, and no padding. Archives saved before that do not load into this model.

---|:---|
| Subtracts the DIV2K RGB mean `(0.4488, 0.4371, 0.4040)` at the input and adds it back at the output (`img_range = 1`) | No mean shift; `conv_first`'s and the tail's biases are free to absorb it |
| `PatchEmbed` LayerNorm after `conv_first`, and a final LayerNorm before `conv_after_body` | Neither |
| Each stage (`PFTB`) ends in its own `conv(dim -> dim)` and a residual around the stage | No per-stage conv or residual; one `conv_after_body` and one long skip |
| `ConvFFN` (Linear, GELU, depthwise conv with `convffn_kernel_size=7`, Linear) | An MLP |
| Top-k sparse attention (`num_topk` per layer) | Dense attention; `sparsity_mode` other than `'none'` raises |
| Pads the input to a multiple of `window_size` by mirroring, crops the output | Requires the input to be a multiple already |
| torch `PixelShuffle` channel order `(c, i, j)` | `PixelShuffle2D`'s `(i, j, c)`, a fixed permutation of the preceding conv's output channels (same function class; matters only when porting torch weights) |

---

## 5. Quick Start Guide

### Your First PFT-SR Model (30 seconds)

Super-resolve a low-resolution image by 4x.

```python
import keras
from dl_techniques.models.vision.super_resolution.pft_sr.model import create_pft_sr

# 1. Create the model (Base variant)
model = create_pft_sr(scale=4, variant='base')

# 2. Generate dummy low-res data (Batch, H, W, C)
# Any static size with sides >= 16: the model mirror-pads to its window (32 for 'base')
lr_image = keras.random.normal((1, 64, 64, 3))

# 3. Inference
# Output shape will be (1, 256, 256, 3)
sr_image = model(lr_image)

print(f"LR Shape: {lr_image.shape}")
print(f"SR Shape: {sr_image.shape}")
```

---

## 6. Component Reference

### 6.1 `PFTSR` (Model Class)

**Purpose**: The main Keras `Model` class.

```python
from dl_techniques.models.vision.super_resolution.pft_sr.model import PFTSR

model = PFTSR(
    scale=4,
    in_channels=3,
    embed_dim=60,
    num_blocks=[4, 4, 4, 6, 6, 6],
    num_heads=6,
    window_size=8,
    upsampler='pixelshuffle'
)
```

### 6.2 Factory Function

#### `create_pft_sr(scale, variant, include_top=True, **kwargs)`
The recommended way to instantiate models.
-   `scale`: what the variant's upsampler supports (§4.2).
-   `variant`: `'light'` (`pixelshuffledirect`), `'base'` (`pixelshuffle`), or `'repo_medium'` (`pixelshuffle`).
-   `**kwargs`: any `PFTSR` argument, overriding the variant table (for example another `upsampler`).

---

## 7. Configuration & Model Variants

Two of the three configurations are the paper's own, quoted field for field from
the official training YAMLs in [CVL-UESTC/PFT-SR](https://github.com/CVL-UESTC/PFT-SR)
(`options/train/001_PFT_SRx2_scratch.yml` and `.../101_PFT_light_SRx2_scratch.yml`).
The third is this repository's own and is named so it does not read as a published
size: the official repo ships exactly two configs, PFT and PFT_light.

| Variant | Source | Embed Dim | Blocks Per Stage | Heads | MLP | Window | Kernel | `num_topk` | Upsampler | Params, `include_top=True` (4x) | Params, `include_top=False` |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`light`** | paper (PFT_light) | 52 | `[2, 4, 6, 6, 6]` | 4 | 1.0 | 32 | 7 | 1024 x2, 256 x4, 128 x6, 64 x6, 32 x6 | `pixelshuffledirect` | 792,416 | 769,904 |
| **`base`** | paper (PFT) | 240 | `[4, 4, 4, 6, 6, 6]` | 6 | 2.0 | 32 | 7 | 1024 x4, 256 x4, 128 x4, 64 x6, 32 x6, 16 x6 | `pixelshuffle` | 19,771,239 | 19,335,780 |
| **`repo_medium`** | repo-original | 80 | `[6, 6, 6, 8, 8, 8]` | 8 | 2.0 | 8 | 7 | `None` (every key) | `pixelshuffle` | 3,450,429 | 3,107,130 |

Kernel is `convffn_kernel_size`. Measured 2026-10-20 at `scale=4` on a
`(1, 32, 32, 3)` input, every weight trainable (`count_params()` equals the
trainable count), and equal to the official hand sum in
`tests/test_models/test_pft_sr/test_the_parameter_count_is_the_official_hand_sum.py`:

```python
m = create_pft_sr(4, variant, include_top=top); m(keras.ops.zeros((1, 32, 32, 3)))
sum(int(np.prod(w.shape)) for w in m.trainable_weights)
```

The difference between the two columns is the official tail's hand sum at
`scale=4` (`c(i, o) = 9*i*o + o`, `F = 64`): light `c(52, 48)` = 22,512; base
`c(240, F) + 2*c(F, 4F) + c(F, 3)` = 435,459; repo_medium `c(80, F) + 2*c(F, 4F)
+ c(F, 3)` = 343,299.

From 2026-10-10 to 2026-10-15 `include_top=True` ended in an `EnhancementHead`
(three `ConvBlock`s at `embed_dim`, then a stride-`scale` transposed conv) in
place of the official tail, while `upsample` and `conv_last` were built and
applied in neither mode, and the `upsampler` knob only checked the scale. That
head is gone with `head_config_overrides` (passing it raises `ValueError`), and
archives saved with it do not load into this model. The light variant also
used `pixelshuffle` until then; its official config is `pixelshuffledirect`.

`repo_medium` is 80-wide, i.e. it sits *between* the two published sizes, which is
why it is not called "large".

---

## 8. Comprehensive Usage Examples

### Example 1: Training Loop Setup

```python
import keras
from dl_techniques.models.vision.super_resolution.pft_sr.model import create_pft_sr

# 1. Setup Model
model = create_pft_sr(scale=4, variant='base')

# 2. Compile
# MAE (L1) is standard for SR. AdamW is recommended.
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=2e-4, weight_decay=1e-4),
    loss='mean_absolute_error',
    metrics=['psnr'] # Note: Keras PSNR metric requires range 0-1 or 0-255 matching
)

# 3. Load Data (Pseudo-code)
# Assuming train_ds yields (lr_batch, hr_batch)
# HR images should be 4x larger than LR images.
# train_ds = ... 

# 4. Train
# history = model.fit(train_ds, epochs=100)
```

### Example 2: Inference with Image Preprocessing

```python
import keras
import numpy as np
from PIL import Image

def super_resolve(image_path, model):
    # Load
    img = Image.open(image_path).convert('RGB')
    
    # Preprocess
    x = keras.utils.img_to_array(img)
    x = x / 255.0
    x = keras.ops.expand_dims(x, 0) # Add batch dim
    
    # No padding needed: the model mirror-pads to a multiple of window_size
    # and crops the output (sides must be at least half the padded size)
    
    # Inference
    out = model(x, training=False)
    
    # Postprocess
    out = keras.ops.squeeze(out, 0)
    out = keras.ops.clip(out, 0, 1)
    out = keras.utils.array_to_img(out)
    
    return out

# Usage
# sr_img = super_resolve("my_low_res.jpg", model)
# sr_img.save("my_high_res.jpg")
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Stochastic Depth (Drop Path)

For larger models or smaller datasets, regularization is key. You can enable Drop Path (Stochastic Depth) during initialization.

```python
model = PFTSR(
    scale=4, 
    embed_dim=60, 
    num_blocks=[4, 4, 4, 6, 6, 6],
    drop_path_rate=0.2  # 20% probability of dropping a residual path
)
```

The official PFT has no stochastic depth; `drop_path_rate` is this repository's
addition (SwinIR placement, rates rising linearly over the layers) and is 0 by
default.

### Pattern 2: Another Official Upsampler

Each variant carries its official upsampler; any `PFTSR` argument overrides it.

```python
# The lightweight tail on the base body
model = create_pft_sr(scale=4, variant='base', upsampler='pixelshuffledirect')

# The real-world tail (x4 only, as in the official code)
model = create_pft_sr(scale=4, variant='base', upsampler='nearest+conv')
```

---

## 10. Performance Optimization

### Mixed Precision Policy

The model builds under a `mixed_float16` policy; the softmax, the focusing
product and the top-k then run in float32 (in float16 the official `eps = 1e-20`
is 0). No speed or quality measurement has been made here.

```python
keras.mixed_precision.set_global_policy('mixed_float16')
model = create_pft_sr(scale=4, variant='repo_medium')
```

### XLA Compilation

The forward compiles under XLA (`jit_compile=True`); no speed measurement has been made here.

```python
model.compile(optimizer="adamw", loss="mae", jit_compile=True)
```

---

## 11. Training and Best Practices

### Data Preparation
*   **Patch Size**: Train on LR patches of size `48x48` (resulting in `192x192` HR output for 4x scale).
*   **Augmentation**: Random rotation (90/180/270) and horizontal flips are essential.
*   **Normalization**: Input images should be in `[0, 1]` range (float32).

### Optimizer Schedule
*   Use `AdamW` with $\beta_1=0.9, \beta_2=0.999$.
*   Initial learning rate: `2e-4`.
*   Reduce LR by half every $X$ epochs (standard MultiStepLR or Cosine Decay).

---

## 12. Serialization & Deployment

The model is fully serializable to the `.keras` format.

```python
# Save
model.save('pft_sr_x4.keras')

# Load (Custom objects are automatically handled if registered, 
# otherwise pass custom_objects map)
loaded_model = keras.models.load_model('pft_sr_x4.keras')
```

**Note**: The model pads its input to a multiple of `window_size` itself (the official mirror pad) and crops the output, so any size works whose sides are at least half their padded size. Height and width must be static per traced call.

---

## 13. Testing & Validation

Run a quick shape check to ensure the upsampling logic is correct for your configuration.

```python
import keras
from dl_techniques.models.vision.super_resolution.pft_sr.model import PFTSR

def test_pft_sr_shapes():
    scale = 4
    model = PFTSR(scale=scale, embed_dim=48, num_blocks=[2, 2])
    
    # Input: 64x64 (padded internally to a multiple of window_size if needed)
    x = keras.random.normal((1, 64, 64, 3))
    y = model(x)
    
    expected_shape = (1, 64 * scale, 64 * scale, 3)
    assert y.shape == expected_shape
    print(f"✓ Output shape valid: {y.shape}")

test_pft_sr_shapes()
```

---

## 14. Troubleshooting & FAQs

**Issue 1: OOM (Out of Memory) errors.**
*   **Cause**: Large `window_size` or `embed_dim` combined with large image inputs.
*   **Solution**: 1) Train on smaller patches (e.g., 48x48 LR). 2) Reduce batch size. 3) Use mixed precision.

**Issue 2: Input shape errors.**
*   **Cause**: A side shorter than half of its padded size (the official mirror pad flips the image once), or a height or width unknown at trace time.
*   **Solution**: Use inputs with sides of at least `window_size / 2` (at least 16 for the 32-window variants) and static shapes.

**Issue 3: Checkerboard artifacts.**
*   **Cause**: Can happen with `pixelshuffledirect`, the light variant's official tail.
*   **Solution**: Use `upsampler='pixelshuffle'`, which ends in a `conv_last` after the pixel shuffles.

---

## 15. Technical Details

### Progressive Focused Attention Math

For layer $l$ and head $h$, with $p(l)$ the previous layer of the same window shift and $\mathcal{K}_{p(l)}$ the keys it kept:

1.  **Attention over the kept keys**:
    $$ S_l = \text{Softmax}_{\mathcal{K}_{p(l)}}(Q_l K_l^T / \sqrt{d} + B_l [+ M]) $$
    *($B_l$ the relative position bias; $M$ the -100 shifted-region mask, added only while the chain is dense)*
2.  **Inheritance**:
    $$ A_l = \frac{S_l \odot A_{p(l)} + \epsilon}{\sum S_l \odot A_{p(l)} + \epsilon}, \quad \epsilon = 10^{-20} $$
    *(where $\odot$ is the Hadamard/element-wise product)*
3.  **Top-k**: if $k_l = \text{num\_topk}[l]$ is below $|\mathcal{K}_{p(l)}|$, keep the $k_l$ largest entries of $A_l$ per query; $\mathcal{K}_l \subseteq \mathcal{K}_{p(l)}$.
4.  **Output**:
    $$ \text{Output}_l = \text{Linear}(A_l V_l + \text{GELU}(\text{DWConv}(V_l))) $$

A key an earlier layer of the chain dropped is never attended again, and a weight near zero in $A_{p(l)}$ stays near zero in $A_l$, which is what prunes the search space.

---

## 16. Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@inproceedings{long2025progressive,
  title={Progressive Focused Transformer for Single Image Super-Resolution},
  author={Long, Wei and Zhou, Xingyu and Zhang, Leheng and Gu, Shuhang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={2279--2288},
  year={2025}
}
```

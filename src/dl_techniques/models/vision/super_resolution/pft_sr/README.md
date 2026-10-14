# PFT-SR: Progressive Focused Transformer for Super-Resolution

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)
[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-blue.svg)](https://cvpr.thecvf.com/)

An implementation of the **Progressive Focused Transformer (PFT-SR)** architecture in **Keras 3**. This implementation is based on the CVPR 2025 paper by Long et al., who report state-of-the-art performance for the PFT-SR architecture on single image super-resolution benchmarks. This port has not been independently trained or benchmarked, and no performance claim is made for it.

The architecture's core innovation is the **Progressive Focused Attention (PFA)** mechanism, which allows attention maps to be inherited and refined across layers. This enables the network to progressively focus on informative features while suppressing redundancy, leading to better reconstruction quality and computational efficiency.

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

It utilizes **Progressive Focused Attention (PFA)** to multiply the current layer's attention map with the attention map from the previous layer. This creates a "focusing" effect, where consistent patterns are amplified and irrelevant background noise is progressively suppressed as the signal moves deeper into the network.

### Key Innovations of this Implementation

1.  **The official reconstruction tail**: the default output ends in the tail of the official code ([CVL-UESTC/PFT-SR](https://github.com/CVL-UESTC/PFT-SR), `basicsr/archs/pft_arch.py`), chosen by `upsampler` exactly as there (§4.2). The body is a simplified port; §4.3 lists every difference.
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
  1. Attention maps are inherited: Attn_i = Softmax(Q K^T) * Attn_{i-1}.
  2. Salient features (edges, textures) are tracked and enhanced across layers.
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
│  Input (LR) ───►┌───────────────┐                                │
│                 │ Shallow Feat  │ (Conv2D 3x3)                   │
│                 └───────┬───────┘                                │
│                         │                                        │
│                 ┌───────▼───────┐                                │
│                 │  PFT Stage 1  │ (Shifted Window PFA)           │
│                 └───────┬───────┘                                │
│                         │                                        │
│                       .....  (Multiple Stages)                   │
│                         │                                        │
│                 ┌───────▼───────┐                                │
│                 │ Reconstruction│ (Conv + Global Residual)       │
│                 └───────┬───────┘                                │
│                         │                                        │
│                 ┌───────▼───────┐                                │
│                 │   Upsampler   │ (Pixel Shuffle)                │
│                 └───────┬───────┘                                │
│                         │                                        │
│                 ┌───────▼───────┐                                │
│                 │  Output (HR)  │                                │
│                 └───────────────┘                                │
└──────────────────────────────────────────────────────────────────┘
```

### The PFT Block Data Flow

The unique aspect is the passing of `attn_map` alongside the feature tensor `x`.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       PFT Block Data Flow                               │
└─────────────────────────────────────────────────────────────────────────┘

Input Tuple (Features x, Previous Map M_prev)
    │
    ▼
1. NORM & WINDOWING
   Partition x into non-overlapping windows
    │
    ▼
2. PROGRESSIVE FOCUSED ATTENTION (PFA)
   ├─► Compute Q, K, V
   ├─► Apply LePE to V (Depthwise Conv)
   ├─► Raw Scores = Softmax(Q @ K)
   │
   ├─► MERGE: M_curr = Raw Scores * M_prev  <-- The Core Magic
   │
   └─► Output = M_curr @ V
    │
    ▼
3. FEED FORWARD NETWORK
   Standard MLP with expansion ratio
    │
    ▼
Output Tuple (Features x_out, Current Map M_curr)
```

---

## 4. Architecture Deep Dive

### 4.1 `PFTBlock`

The fundamental building block.
-   **Window Attention**: Uses shifted windows (like Swin Transformer) to allow cross-window information exchange. Even numbered blocks use regular windows; odd numbered blocks use shifted windows.
-   **LePE (Locally-Enhanced Positional Encoding)**: Instead of adding absolute position embeddings, positional information is injected via a depthwise convolution on the Value (V) vector within the attention mechanism.
-   **Attention Inheritance**: Takes a tuple `(x, mask)` and returns a tuple `(x, new_mask)`.

### 4.2 `PFTSR` (The Model)

A Keras `Model` subclass that orchestrates the stages.
-   **Stages**: The model is divided into stages. Each stage contains a defined number of PFT Blocks.
-   **Tail** (`include_top=True`, the default), the official `PFT.forward` branches with `num_feat = F = 64`. A biased 3x3 conv is written `conv(i -> o)`:
    -   `pixelshuffle` (scale a power of two, or 3): `conv_before_upsample` `conv(embed_dim -> F)` + LeakyReLU(0.01), then `conv(F -> 4F)` + pixel shuffle 2 per factor of two (or `conv(F -> 9F)` + pixel shuffle 3), then `conv_last` `conv(F -> 3)`.
    -   `pixelshuffledirect` (any scale): `conv(embed_dim -> scale^2 * 3)` + pixel shuffle `scale`, nothing after it.
    -   `nearest+conv` (x4 only, as the official `assert`): `conv_before_upsample` + LeakyReLU(0.01), two stages of nearest 2x + `conv(F -> F)` + LeakyReLU(0.2), `conv_hr` + LeakyReLU(0.2), `conv_last`.

    The 0.01 is torch `nn.LeakyReLU()`'s default, passed explicitly (Keras' default is 0.3). The model builds this tail itself rather than through the shared `EnhancementHead`, whose super-resolution path is a stride-`scale` transposed conv that none of the official tails use (the SqueezeNet precedent: a reference head the shared head cannot express is model-owned).
-   **`include_top=False`**: no tail is built, and the model returns the features the tail reads, `conv_after_body(body(x)) + x`, `(B, H, W, embed_dim)`.

### 4.3 Differences from the official code

The tail above is the official one, layer for layer (pinned by a hand sum and a
hand-composed forward in `tests/test_models/test_pft_sr/test_model.py`). The
rest is a simplified port, and these differences are deliberate scope, not
oversights to rely on:

| Official `PFT` | This port |
|:---|:---|
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
# Height/Width must be divisible by the variant's window_size (32 for 'base')
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

| Variant | Source | Embed Dim | Blocks Per Stage | Heads | MLP | Window | Upsampler | Params, `include_top=True` (4x) | Params, `include_top=False` |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`light`** | paper (PFT_light) | 52 | `[2, 4, 6, 6, 6]` | 4 | 1.0 | 32 | `pixelshuffledirect` | 462,692 | 440,180 |
| **`base`** | paper (PFT) | 240 | `[4, 4, 4, 6, 6, 6]` | 6 | 2.0 | 32 | `pixelshuffle` | 14,936,019 | 14,500,560 |
| **`repo_medium`** | repo-original | 80 | `[6, 6, 6, 8, 8, 8]` | 8 | 2.0 | 8 | `pixelshuffle` | 2,624,179 | 2,280,880 |

Measured 2026-10-15 at `scale=4` on a `(1, 32, 32, 3)` input, every weight
trainable (`count_params()` equals the trainable count):

```python
m = create_pft_sr(4, variant, include_top=top); m(keras.ops.zeros((1, 32, 32, 3)))
sum(int(np.prod(w.shape)) for w in m.trainable_weights)
```

The difference between the two columns is the official tail's hand sum at
`scale=4` (`c(i, o) = 9*i*o + o`, `F = 64`): light `c(52, 48)` = 22,512; base
`c(240, F) + 2*c(F, 4F) + c(F, 3)` = 435,459; repo_medium `c(80, F) + 2*c(F, 4F)
+ c(F, 3)` = 343,299. The body is unchanged from the step-6 measurement.

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
    
    # Pad if necessary (dimensions must be divisible by window_size)
    # Note: A robust production pipeline should implement reflection padding here
    
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

PFT-SR benefits significantly from mixed precision on modern GPUs (Volta/Ampere/Hopper).

```python
keras.mixed_precision.set_global_policy('mixed_float16')
model = create_pft_sr(scale=4, variant='repo_medium')
```

### XLA Compilation

Enable JIT compilation for graph optimization, especially beneficial for the window partitioning logic.

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

**Note**: The model inputs must be divisible by `window_size` (default 8). When deploying, ensure input tensors are padded to multiples of 8.

---

## 13. Testing & Validation

Run a quick shape check to ensure the upsampling logic is correct for your configuration.

```python
import keras
from dl_techniques.models.vision.super_resolution.pft_sr.model import PFTSR

def test_pft_sr_shapes():
    scale = 4
    model = PFTSR(scale=scale, embed_dim=48, num_blocks=[2, 2])
    
    # Input: 64x64 (divisible by window_size 8)
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
*   **Cause**: Input height/width is not divisible by `window_size`.
*   **Solution**: Pad inputs. `h_new = ceil(h / window_size) * window_size`.

**Issue 3: Checkerboard artifacts.**
*   **Cause**: Can happen with `pixelshuffledirect`, the light variant's official tail.
*   **Solution**: Use `upsampler='pixelshuffle'`, which ends in a `conv_last` after the pixel shuffles.

---

## 15. Technical Details

### Progressive Focused Attention Math

For layer $l$ and head $h$:

1.  **Standard Attention**:
    $$ \text{Score}_l = \text{Softmax}(Q_l K_l^T / \sqrt{d}) $$
2.  **Inheritance**:
    $$ \text{FocusedScore}_l = \text{Score}_l \odot \text{FocusedScore}_{l-1} $$
    *(where $\odot$ is the Hadamard/element-wise product)*
3.  **Output**:
    $$ \text{Output}_l = \text{FocusedScore}_l \times V_l + \text{LePE}(V_l) $$

This formulation ensures that if the attention weight for a token pair was near zero in layer $l-1$, it remains suppressed in layer $l$, effectively pruning the search space for the attention mechanism.

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

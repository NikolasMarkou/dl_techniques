# PFT-SR: Progressive Focused Transformer for Super-Resolution

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-blue.svg)](https://cvpr.thecvf.com/)

A production-ready, fully-featured implementation of the **Progressive Focused Transformer (PFT-SR)** architecture in **Keras 3**. This implementation is based on the CVPR 2025 paper by Long et al. and achieves state-of-the-art performance on single image super-resolution benchmarks.

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

1.  **Faithful Architecture**: Accurately implements the `PFTBlock`, `ProgressiveFocusedAttention`, and **LePE** (Locally-Enhanced Positional Encoding) as described in the paper.
2.  **Keras 3 Native**: Built using modern Keras 3 functional patterns, ensuring compatibility with TensorFlow, JAX, and PyTorch backends.
3.  **Flexible Upsampling**: Supports multiple upsampling strategies (`pixelshuffle`, `pixelshuffledirect`, `nearest+conv`) to trade off between quality and speed.
4.  **Factory Methods**: Includes `create_pft_sr` with presets (`light`, `base`, `large`) for immediate deployment.

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
-   **Upsampler**:
    -   `pixelshuffle`: Conv -> PixelShuffle -> Conv -> PixelShuffle (for 4x). Standard high-quality approach.
    -   `pixelshuffledirect`: Single Conv -> PixelShuffle (faster, slightly lower quality).
    -   `nearest+conv`: Nearest neighbor resizing followed by convolution.

---

## 5. Quick Start Guide

### Your First PFT-SR Model (30 seconds)

Super-resolve a low-resolution image by 4x.

```python
import keras
from model import create_pft_sr

# 1. Create the model (Base variant)
model = create_pft_sr(scale=4, variant='base')

# 2. Generate dummy low-res data (Batch, H, W, C)
# Height/Width must be divisible by window_size (default 8)
lr_image = keras.random.normal((1, 48, 48, 3))

# 3. Inference
# Output shape will be (1, 192, 192, 3)
sr_image = model(lr_image)

print(f"LR Shape: {lr_image.shape}")
print(f"SR Shape: {sr_image.shape}")
```

---

## 6. Component Reference

### 6.1 `PFTSR` (Model Class)

**Purpose**: The main Keras `Model` class.

```python
from model import PFTSR

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

#### `create_pft_sr(scale, variant)`
The recommended way to instantiate models.
-   `scale`: 2, 3, or 4.
-   `variant`: `'light'`, `'base'`, or `'large'`.

---

## 7. Configuration & Model Variants

We provide three standard configurations based on the paper.

| Variant | Embed Dim | Blocks Per Stage | Heads | Params | Use Case |
|:---:|:---:|:---:|:---:|:---:|:---|
| **`light`** | 48 | `[4, 4, 4, 4]` | 6 | ~0.8M | Mobile/Edge |
| **`base`** | 60 | `[4, 4, 4, 6, 6, 6]` | 6 | ~1.1M | General Purpose |
| **`large`** | 80 | `[6, 6, 6, 8, 8, 8]` | 8 | ~1.8M | High-Quality Benchmarks |

---

## 8. Comprehensive Usage Examples

### Example 1: Training Loop Setup

```python
import keras
from model import create_pft_sr

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

### Pattern 2: Custom Upsamplers

If deployment speed is critical, change the upsampler to `pixelshuffledirect` or `nearest+conv`.

```python
# Fastest inference, slightly lower quality
fast_model = PFTSR(..., upsampler='nearest+conv') 
```

---

## 10. Performance Optimization

### Mixed Precision Policy

PFT-SR benefits significantly from mixed precision on modern GPUs (Volta/Ampere/Hopper).

```python
keras.mixed_precision.set_global_policy('mixed_float16')
model = create_pft_sr(scale=4, variant='large')
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
from model import PFTSR

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
*   **Cause**: Can happen with `pixelshuffledirect`.
*   **Solution**: Use the default `pixelshuffle` upsampler which includes an extra convolution after upsampling to smooth artifacts.

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

# ConvUNext: Modern U-Net Architecture

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, highly scalable implementation of **ConvUNext** in **Keras 3**. This architecture fuses the hierarchical structure of a U-Net with the modern design principles of **ConvNeXt V2**, creating a powerful backbone for image segmentation, restoration, and denoising tasks.

Key architectural features include a **flexible bias design** (supporting bias-free restoration), **Global Response Normalization (GRN)** for channel inter-dependency, and integrated **Deep Supervision** for robust training convergence.

---

## Table of Contents

1. [Overview: Modernizing the U-Net](#1-overview-modernizing-the-u-net)
2. [The Problem: Bias and Receptive Fields](#2-the-problem-bias-and-receptive-fields)
3. [How ConvUNext Works](#3-how-convunext-works)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Model Variants](#7-configuration--model-variants)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Advanced Usage Patterns](#9-advanced-usage-patterns)
10. [Performance Optimization](#10-performance-optimization)
11. [Training and Best Practices](#11-training-and-best-practices)
12. [Serialization & Deployment](#12-serialization--deployment)
13. [Troubleshooting & FAQs](#13-troubleshooting--faqs)
14. [Technical Details](#14-technical-details)
15. [Testing & Validation](#15-testing--validation)
16. [Citation](#16-citation)

---

## 1. Overview: Modernizing the U-Net

### What is ConvUNext?

**ConvUNext** is a "ConvNet for the 2020s" applied to the classic U-Net encoder-decoder structure. While Vision Transformers (ViTs) have gained popularity, modern ConvNets like ConvNeXt have demonstrated that pure convolutional architectures can compete with or outperform Transformers when designed correctly.

This implementation provides a highly configurable foundation model. By standardizing on modern architectural choices (7x7 kernels, GRN, GELU), the model achieves excellent performance on standard benchmarks. It also supports a **bias-free mode** (via `use_bias=False`), which allows for **scale invariance**—crucial for image restoration tasks like denoising.

### Key Innovations of this Implementation

1.  **Flexible Bias Design**: Defaults to standard biased convolutions (`use_bias=True`) for optimal performance on classification and segmentation. Can be switched to `use_bias=False` for restoration tasks requiring scale invariance.
2.  **ConvNeXt V1/V2 Support**: Fully supports both V1 (LayerScale) and V2 (Global Response Normalization) blocks, leveraging the "Masked Autoencoder" scaling insights.
3.  **Deep Supervision**: Built-in support for multi-scale supervision. The decoder outputs predictions at multiple resolutions during training to combat vanishing gradients and enforce structural consistency.
4.  **Keras 3 Native**: Built as a custom `keras.Model` following modern Keras 3 best practices with complete serialization support, comprehensive type hints, and Sphinx-compliant documentation.
5.  **Production-Ready**: Extensively tested with 125+ unit tests covering initialization, forward pass, gradient flow, serialization, edge cases, and integration scenarios.

### ConvUNext vs. Standard U-Net

**Traditional U-Net (2015)**:
```
- Block: 3x3 Conv -> ReLU -> 3x3 Conv -> ReLU
- Normalization: Often Batch Norm (can be unstable with small batches).
- Receptive Field: Small, grows slowly with depth.
- Mechanics: Simple sliding windows.
```

**ConvUNext (Modern)**:
```
- Block: 7x7 Depthwise Conv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv.
- Normalization: Global Response Norm (GRN) & Layer Norm.
- Receptive Field: Large (7x7 kernels), mimicking Vision Transformers.
- Mechanics: Inverted bottlenecks, configurable bias propagation.
```

---

## 2. The Problem: Bias and Receptive Fields

### The Challenge of Generalization in Restoration

In low-level vision tasks (like denoising or super-resolution), standard CNNs often overfit to specific noise levels or intensity ranges.

```
┌─────────────────────────────────────────────────────────────┐
│  Standard Layer (use_bias=True)                             │
│                                                             │
│  y = Wx + b                                                 │
│                                                             │
│  If input 'x' is scaled by 2 (2x), the output becomes:      │
│  y_new = W(2x) + b  !=  2 * (Wx + b)                        │
│                                                             │
│  The bias term 'b' does not scale. This is fine for         │
│  segmentation, but breaks linearity for denoising.          │
└─────────────────────────────────────────────────────────────┘
```

### The ConvUNext Solution

ConvUNext allows you to enforce a bias-free constraint throughout the network by setting `use_bias=False`.

```
┌─────────────────────────────────────────────────────────────┐
│  Bias-Free Mode (use_bias=False)                            │
│                                                             │
│  y = Wx                                                     │
│                                                             │
│  If input 'x' is scaled by 2 (2x), the output becomes:      │
│  y_new = W(2x) = 2 * (Wx) = 2y                              │
│                                                             │
│  The model becomes scale-invariant. This ensures the        │
│  network focuses on structural content.                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How ConvUNext Works

### The High-Level Architecture

The model follows a symmetric Encoder-Decoder structure with skip connections, using ConvNeXt blocks as the primary compute unit.

```
┌──────────────────────────────────────────────────────────────────┐
│                     ConvUNext Architecture                       │
│                                                                  │
│ Input Image (H, W, C)                                            │
│       │                                                          │
│       ▼                                                          │
│ ┌────────────┐                                  ┌────────────┐   │
│ │    Stem    │─────────────────────────────────►│ Final Conv │   │
│ └─────┬──────┘                                  └─────▲──────┘   │
│       │                                               │          │
│       ▼                                               ▲          │
│ ┌────────────┐         Skip Connection          ┌────────────┐   │
│ │ Encoder L0 │─────────────────────────────────►│ Decoder L0 │   │
│ └─────┬──────┘                                  └─────▲──────┘   │
│       │ Downsample                            Upsample│          │
│       ▼                                               ▲          │
│ ┌────────────┐         Skip Connection          ┌────────────┐   │
│ │ Encoder L1 │─────────────────────────────────►│ Decoder L1 │   │
│ └─────┬──────┘                                  └─────▲──────┘   │
│       │                                               │          │
│      ...                 Bottleneck                  ...         │
│       │                ┌────────────┐                 │          │
│       └───────────────►│ ConvNeXt   │─────────────────┘          │
│                        │ Blocks     │                            │
│                        └────────────┘                            │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow with Deep Supervision

When `enable_deep_supervision=True`, the model returns a list of outputs during training.

1.  **Input**: Image Tensor `(B, H, W, C)`
2.  **Encoder**: Progressively reduces spatial dims (H/2, H/4, H/8...) while increasing channels.
3.  **Bottleneck**: Processes the most compressed representation.
4.  **Decoder**: Progressively upsamples. At each level, it concatenates features from the equivalent Encoder level (Skip Connection).
5.  **Outputs**:
    *   **Output 0**: Final High-Res Prediction.
    *   **Output 1**: Prediction from Decoder Level 1 (H/2).
    *   **Output 2**: Prediction from Decoder Level 2 (H/4).

---

## 4. Architecture Deep Dive

### 4.1 `ConvUNextStem`
Unlike standard U-Nets that use 3x3 convolutions initially, the stem uses a **7x7 convolution** followed by **LayerNormalization**. This aggressive initial receptive field helps capture broader context immediately. The stem supports configurable bias via `use_bias`.

### 4.2 ConvNeXt V2 Block
The core building block used in both encoder and decoder stages:
1.  **Depthwise Conv (7x7)**: Spatial mixing with large receptive field.
2.  **LayerNorm**: Channel-wise normalization for stable gradients.
3.  **Pointwise Conv (1x1)**: Channel expansion (4x width).
4.  **GELU**: Smooth, differentiable activation.
5.  **GRN**: Global Response Normalization (calibrates channel interaction).
6.  **Pointwise Conv (1x1)**: Channel projection back to original width.
7.  **Drop Path**: Stochastic depth for regularization (rate increases with depth).

### 4.3 Decoder & Upsampling
The decoder uses **bilinear upsampling** followed by concatenation with skip connections. Crucially, a **channel adjustment layer** (1x1 conv) is applied after concatenation to smoothly fuse the features before processing them with ConvNeXt blocks.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure requirements are met
pip install keras>=3.8.0 tensorflow>=2.18.0
```

### Your First ConvUNext Model (30 seconds)

Let's build a model for semantic segmentation of 256x256 images.

```python
import keras
import numpy as np
from dl_techniques.models.convunext.model import create_convunext_variant

# 1. Create a 'base' variant model
# Enable deep supervision for better training convergence
# use_bias defaults to True (standard for segmentation)
model = create_convunext_variant(
    variant='base',
    input_shape=(256, 256, 3),
    enable_deep_supervision=True,
    output_channels=1  # Binary segmentation
)

# 2. Compile the model
# With deep supervision, the model returns a list of outputs
# We apply the same loss to each output with decreasing weights
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss=['mse', 'mse', 'mse', 'mse'],
    loss_weights=[1.0, 0.5, 0.25, 0.125]  # Weight multi-scale outputs
)

model.summary()

# 3. Test with dummy data
x = np.random.normal(size=(2, 256, 256, 3)).astype('float32')
outputs = model(x, training=False)

print(f"Number of outputs: {len(outputs)}")  # 4 for depth=4
```

---

## 6. Component Reference

### 6.1 `ConvUNextModel`
The main Keras Model class. It handles the complex wiring of encoder, decoder, skip connections, and supervision heads.

**Key Parameters**:
- `input_shape`: Tuple of (height, width, channels).
- `depth`: Number of downsampling stages (minimum 2).
- `initial_filters`: Base channel count, multiplied at each stage.
- `use_bias`: **(New)** Boolean. If `True` (default), adds bias to all convolutions. Set to `False` for restoration tasks requiring scale invariance.
- `enable_deep_supervision`: Whether to return multi-scale outputs.
- `convnext_version`: Either `'v1'` (LayerScale) or `'v2'` (GRN).

### 6.2 `ConvUNextStem`
The initial feature extraction layer. Implements a 7x7 convolution with LayerNormalization.

### 6.3 `create_convunext_variant`
The factory function recommended for instantiation.

**Signature**:
```python
def create_convunext_variant(
    variant: str,
    input_shape: Tuple[Optional[int], Optional[int], int] = (None, None, 3),
    enable_deep_supervision: bool = False,
    output_channels: Optional[int] = None,
    use_bias: bool = True,
    **kwargs
) -> ConvUNextModel
```

---

## 7. Configuration & Model Variants

The architecture scales by depth (number of downsampling levels) and width (channel counts).

| Variant | Depth | Initial Filters | Blocks/Level | Drop Path | Use Case |
|:---:|:---:|:---:|:---:|:---:|:---|
| **`tiny`** | 3 | 32 | 2 | 0.0 | Mobile, Real-time video |
| **`small`**| 3 | 48 | 2 | 0.1 | Lightweight edge devices |
| **`base`** | 4 | 64 | 3 | 0.1 | General purpose |
| **`large`** | 4 | 96 | 4 | 0.2 | High-fidelity segmentation |
| **`xlarge`**| 5 | 128 | 5 | 0.3 | SOTA performance benchmarks |

---

## 8. Comprehensive Usage Examples

### Example 1: Creating a Bias-Free Denoising Model

For restoration tasks, disable bias to achieve scale invariance.

```python
from dl_techniques.models.convunext.model import create_convunext_variant

# Explicitly disable bias for restoration/denoising
denoise_model = create_convunext_variant(
    'base',
    input_shape=(None, None, 3),  # Dynamic input shape
    use_bias=False,               # <-- Bias-free mode
    enable_deep_supervision=False
)
```

### Example 2: Standard Segmentation (With Bias)

For classification or segmentation where absolute intensity matters, keep the default `use_bias=True`.

```python
# Binary segmentation
seg_model = create_convunext_variant(
    'small',
    input_shape=(256, 256, 3),
    output_channels=1,
    final_activation='sigmoid',
    use_bias=True  # Default
)
```

---

## 9. Advanced Usage Patterns

### Converting Training Models to Inference

The `create_inference_model_from_training_model` function handles weight transfer automatically:

```python
from dl_techniques.models.convunext.model import (
    create_convunext_variant, 
    create_inference_model_from_training_model
)

# 1. Train with deep supervision
train_model = create_convunext_variant(
    'base',
    input_shape=(256, 256, 3),
    enable_deep_supervision=True,
    output_channels=1
)

# ... training happens here ...

# 2. Convert to inference model (weights transferred automatically)
inference_model = create_inference_model_from_training_model(train_model)

# 3. Verify outputs match
test_input = np.random.randn(1, 256, 256, 3).astype('float32')
train_output = train_model(test_input, training=False)[0]  # First output
infer_output = inference_model(test_input, training=False)

assert np.allclose(train_output, infer_output, rtol=1e-5)
```

---

## 13. Troubleshooting & FAQs

### Common Issues

**Q: Should I use `use_bias=True` or `False`?**

A: 
- **Use `True` (Default)**: For semantic segmentation, classification, or object detection where inputs are normalized to a fixed range (e.g., [-1, 1]).
- **Use `False`**: For image restoration (denoising, deblurring) or super-resolution, especially if you want the model to be robust to global intensity scaling (e.g., handling both dark and bright images with the same weights).

**Q: Why do I get OOM (Out of Memory) errors?**

A: ConvUNext uses larger 7×7 kernels and higher channel dimensions than standard U-Nets. Reduce batch size or use a smaller variant (`tiny`/`small`).

---

**Q: Why does the model output a list instead of a single tensor?**

A: You initialized the model with `enable_deep_supervision=True`. This is intended for training. For inference, access the first output or use `create_inference_model_from_training_model`.

---

## 14. Technical Details

### Global Response Normalization (GRN)

Introduced in ConvNeXt V2, GRN enhances feature competition across channels. In ConvUNext, GRN is applied after GELU activation.

### Scaling Invariance (Bias-Free Mode)

When `use_bias=False`, the absence of additive constants means:
$$f(\alpha \cdot x) \approx \alpha \cdot f(x)$$
This relationship holds well in practice for $\alpha \in [0.5, 2.0]$, making the model generalize across different exposure levels.

---
# DarkIR: Robust Low-Light Image Restoration Network

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, Keras 3 implementation of **DarkIR**, presented at CVPR 2025. DarkIR is an efficient, all-in-one image restoration network designed to handle multiple degradations simultaneously: low-light, noise, and blur.

Unlike many modern restoration networks that rely on computationally heavy Vision Transformers, DarkIR utilizes efficient CNNs augmented with frequency-domain processing (`FreMLP`) and multi-scale dilated attention mechanisms to achieve state-of-the-art performance with lower latency.

---

## Table of Contents

1. [Overview: What is DarkIR and Why It Matters](#1-overview-what-is-darkir-and-why-it-matters)
2. [The Problem DarkIR Solves](#2-the-problem-darkir-solves)
3. [How DarkIR Works: Core Concepts](#3-how-darkir-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration](#7-configuration)
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

## 1. Overview: What is DarkIR and Why It Matters

### What is DarkIR?

**DarkIR** is a hierarchical Encoder-Decoder (U-Net style) architecture for image restoration. It treats low-light enhancement not just as brightening, but as a simultaneous restoration task involving denoising and deblurring.

It learns a **global residual**, meaning the network predicts the *difference* between the degraded input and the clean image, rather than reconstructing the image from scratch.

### Key Innovations

1.  **Dilated Branching**: Uses parallel convolutions with different dilation rates (e.g., 1, 4, 9) to capture multi-scale context without the parameter explosion of large kernels.
2.  **Frequency MLP (FreMLP)**: Replaces heavy self-attention with Fast Fourier Transforms (FFT). It processes features in the frequency domain to capture global image statistics efficiently ($O(N \log N)$ vs $O(N^2)$).
3.  **SimpleGate**: A parameter-free non-linear activation function that splits and multiplies channels, reducing computational overhead while effectively modeling feature interactions.
4.  **Hybrid Encoder-Decoder**: The encoder uses Frequency processing for global context, while the decoder uses gated FFNs for precise local feature reconstruction.

---

## 2. The Problem DarkIR Solves

### The Challenge of Multiple Degradations

Real-world low-light photography is rarely just "dark." It suffers from a compound set of problems:

```
┌─────────────────────────────────────────────────────────────┐
│  The Low-Light Problem                                      │
│                                                             │
│  1. Poisson-Gaussian Noise: High noise due to high ISO.     │
│  2. Color Distortion: Loss of color fidelity in shadows.    │
│  3. Motion Blur: Long exposures lead to blurred edges.      │
│                                                             │
│  Standard CNNs have small receptive fields (local focus).   │
│  Vision Transformers have global focus but are too slow     │
│  for high-resolution image processing.                      │
└─────────────────────────────────────────────────────────────┘
```

### DarkIR's Solution

```
DarkIR Approach:
  1. Combines Local & Global: Dilated Convs (Local) + FreMLP (Global).
  2. Efficiency: Avoids heavy Matrix Multiplications of Attention.
  3. Stable Training: Learns the residual mapping (Input + Net(Input)).
  4. Benefit: Fast inference on edge devices with restoration quality
     matching or exceeding transformer-based methods.
```

---

## 3. How DarkIR Works: Core Concepts

### The High-Level Architecture

DarkIR uses a U-Net structure with skip connections, processing features at multiple resolutions (scales).

```
┌──────────────────────────────────────────────────────────────────┐
│                       DarkIR Architecture                        │
│                                                                  │
│  Input: [Low Light Image] (H, W, 3)                              │
│             │                                                    │
│    ┌────────▼────────┐     ┌──────────────────────────────────┐  │
│    │ Intro Conv      │◄────┤ Extracts shallow features        │  │
│    └────────┬────────┘     └──────────────────────────────────┘  │
│             │                                                    │
│    ┌────────▼─────────┐      ┌─────────┐                         │
│    │ Encoder Levels   │─────►│ Skip    │                         │
│    │ (FreMLP + Dil)   │      │ Connect │                         │
│    └────────┬─────────┘      └────┬────┘                         │
│             │ Downsample          │                              │
│    ┌────────▼─────────┐      ┌────▼────┐                         │
│    │ Middle Block     │      │ Decoder │                         │
│    └────────┬─────────┘      │ Levels  │                         │
│             │ Upsample       │ (Gated) │                         │
│    ┌────────▼─────────┐      └─────────┘                         │
│    │ Output Conv      │                                          │
│    └────────┬─────────┘                                          │
│             │                                                    │
│  ┌──────────▼──────────┐                                         │
│  │ Add Input (Res)     │  (Global Residual Connection)           │
│  └─────────────────────┘                                         │
└──────────────────────────────────────────────────────────────────┘
```

### The Block Workflow

Inside a **DarkIREncoderBlock**:
1.  **Multi-Scale Path**: Input splits into parallel branches with varying dilation rates. Results are summed.
2.  **Gating**: `SimpleGate` activates features.
3.  **Frequency Path**: Features are transformed via FFT, processed, and transformed back via IFFT to capture global context.
4.  **Dual Residual**: Both paths have learnable residual connections ($\beta$ and $\gamma$).

---

## 4. Architecture Deep Dive

### 4.1 `DilatedBranch`
Captures multi-scale context efficiently.
*   **Dilation 1**: Captures immediate neighbors (3x3).
*   **Dilation 4/9**: Captures distant dependencies without increasing parameters.
*   Uses depthwise convolutions to keep computation low.

### 4.2 `FreMLP` (Frequency MLP)
The "Global Mixer" of the network.
*   **Forward**: `Real Input -> FFT2D -> Magnitude/Phase -> MLP on Magnitude -> IFFT2D -> Real Output`.
*   Operates on the physical principle that global image structures (shapes, illumination) are better represented in the frequency domain.

### 4.3 `SimpleGate`
Replaces activation functions like ReLU or GELU in the feature mixing blocks.
*   Splits channels into two halves $(X_1, X_2)$.
*   Output is $X_1 \odot X_2$.
*   Acts as a parameter-free attention mechanism where one half gates the other.

---

## 5. Quick Start Guide

### Installation

```bash
pip install keras>=3.0 tensorflow>=2.18 numpy dl_techniques
```

### Your First DarkIR Model

```python
import keras
import numpy as np
from dl_techniques.models.darkir import create_darkir_model

# 1. Create Model (Standard Configuration)
model = create_darkir_model(
    img_channels=3,
    width=32,
    enc_blk_nums=[1, 2, 3],  # 3 Encoder stages
    dec_blk_nums=[3, 1, 1],  # 3 Decoder stages
    dilations=[1, 4, 9]      # Multi-scale context
)

# 2. Compile
# Use MAE (L1) or Charbonnier loss for restoration
model.compile(optimizer='adam', loss='mean_absolute_error')

# 3. Dummy Data (Batch, Height, Width, Channels)
# Note: Dimensions should be multiples of 2^num_stages (2^3 = 8)
x_train = np.random.rand(4, 128, 128, 3).astype('float32')
y_train = np.random.rand(4, 128, 128, 3).astype('float32')

# 4. Train
history = model.fit(x_train, y_train, epochs=5, batch_size=2)

# 5. Inference
# Input must be normalized [0, 1]
restored_img = model.predict(x_train[:1])
```

---

## 6. Component Reference

### 6.1 `create_darkir_model`

**Location**: `dl_techniques.models.darkir`

The main factory function. Returns a compiled `keras.Model`.

```python
model = create_darkir_model(
    width=32, 
    use_side_loss=False, 
    extra_depth_wise=True
)
```

### 6.2 `DarkIREncoderBlock` & `DarkIRDecoderBlock`

**Location**: `dl_techniques.models.darkir`

*   **Encoder Block**: Contains Normalization -> Dilated Branches -> SimpleGate -> **FreMLP**.
*   **Decoder Block**: Contains Normalization -> Dilated Branches -> SimpleGate -> **Inverted FFN**.

### 6.3 `SimpleGate`

A utility layer often used in modern vision architectures (e.g., NAFNet, DarkIR).

---

## 7. Configuration

| Parameter | Type | Default | Description |
|:---|:---|:---|:---|
| `img_channels` | int | 3 | Input/Output image channels (RGB=3). |
| `width` | int | 32 | Base feature width. Doubled at every encoder downsampling. |
| `enc_blk_nums` | List[int] | `[1, 2, 3]` | Number of blocks per encoder stage. Length defines network depth. |
| `dec_blk_nums` | List[int] | `[3, 1, 1]` | Number of blocks per decoder stage. Should match encoder length. |
| `dilations` | List[int] | `[1, 4, 9]` | Dilation rates for the parallel branches in every block. |
| `extra_depth_wise`| bool | `True` | Adds an extra depthwise conv layer for better inductive bias. |
| `use_side_loss` | bool | `False` | If True, returns `[final_output, middle_output]` for deep supervision. |

---

## 8. Comprehensive Usage Examples

### Example 1: Tiny Model for Mobile/Edge

Reduce `width` and block counts for real-time applications.

```python
model = create_darkir_model(
    img_channels=3,
    width=16,                  # Reduced from 32
    enc_blk_nums=[1, 1],       # Only 2 stages, shallow
    dec_blk_nums=[1, 1],
    dilations=[1, 2],          # Smaller receptive field
    extra_depth_wise=False     # Reduce parameters
)
model.summary()
```

### Example 2: Deep Supervision Training

Deep supervision (Side Loss) helps gradients flow to the middle of the network, improving convergence for deep models.

```python
model = create_darkir_model(
    img_channels=3,
    width=32,
    use_side_loss=True
)

# Loss weights: Main output gets 1.0, Side output gets 0.5
model.compile(
    optimizer='adam',
    loss=['mae', 'mae'],
    loss_weights=[1.0, 0.5]
)

# y_train must be passed twice or adapted in data pipeline
model.fit(x_train, [y_train, y_train], epochs=10)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: High-Res Processing with Tiling

DarkIR is fully convolutional, so it can handle any image size. However, for 4K inputs, memory is a constraint.

```python
def process_tiled(model, image, tile_size=512, overlap=64):
    # Custom logic to crop image into 512x512 tiles
    # Run prediction on tiles
    # Stitch tiles back together, blending overlaps
    pass
```

### Pattern 2: Custom Dilation Strategies

If dealing with very large blur kernels, increase dilation rates.

```python
# For severe motion blur, we need massive receptive fields
model = create_darkir_model(
    dilations=[1, 6, 12, 18],  # Very wide view
    ...
)
```

---

## 10. Performance Optimization

### Mixed Precision
DarkIR contains many convolution and FFT operations which benefit significantly from Float16 on GPUs.

```python
keras.mixed_precision.set_global_policy('mixed_float16')
```

### XLA Compilation
The `SimpleGate` and FFT operations are highly optimizable by XLA.

```python
model.compile(..., jit_compile=True)
```

### Input Sizes
Ensure input dimensions are multiples of $2^N$ (where $N$ is the number of stages). If `enc_blk_nums=[1,2,3]`, $N=3$, so inputs should be divisible by $2^3=8$. This prevents padding artifacts during down/upsampling.

---

## 11. Training and Best Practices

### Loss Functions
*   **L1 / MAE**: Standard for restoration. Produces sharper edges than MSE.
*   **Charbonnier Loss**: A robust variant of L1 ($\sqrt{x^2 + \epsilon^2}$). Highly recommended.
*   **Perceptual Loss**: Optional fine-tuning stage using VGG features.

### Optimizer
**AdamW** is recommended. The code uses `1e-4` weight decay by default in the paper.

```python
from dl_techniques.optimization import optimizer_builder
# ... configure AdamW ...
```

### Data Augmentation
Crucial for restoration:
*   Random Crop
*   Random Horizontal/Vertical Flip
*   Random 90-degree Rotation
*   **MixUp** is *not* typically used for restoration tasks.

---

## 12. Serialization & Deployment

The model uses `keras.saving.register_keras_serializable`, making it compatible with the modern `.keras` format.

```python
# Save complete model
model.save("darkir_best.keras")

# Load for inference
# Note: Custom layers are automatically handled if imported
loaded_model = keras.models.load_model("darkir_best.keras")
```

---

## 13. Testing & Validation

### Unit Tests

```python
def test_darkir_shapes():
    B, H, W, C = 2, 64, 64, 3
    # 3 Stages -> Downsample by 8
    model = create_darkir_model(
        width=16, 
        enc_blk_nums=[1, 1, 1], 
        dec_blk_nums=[1, 1, 1]
    )
    
    x = np.random.normal(size=(B, H, W, C)).astype('float32')
    y = model(x)
    
    assert y.shape == (B, H, W, C)
    print("✅ Shape Consistency Verified")
    
    # Check simple gate logic
    from dl_techniques.models.darkir import SimpleGate
    sg = SimpleGate()
    # Input 2C -> Output C
    out = sg(np.random.normal(size=(1, 10, 10, 6))) 
    assert out.shape[-1] == 3
    print("✅ SimpleGate Verified")

test_darkir_shapes()
```

---

## 14. Troubleshooting & FAQs

**Issue: OOM (Out of Memory) on GPU.**
*   **Cause**: High resolution inputs or large `width`. FFT can also be memory intensive.
*   **Fix**: Reduce `width` (e.g., 32 -> 16). Reduce batch size. Use Mixed Precision.

**Issue: Checkerboard Artifacts.**
*   **Cause**: Usually caused by `Conv2DTranspose`.
*   **Fix**: DarkIR uses `PixelShuffle` (`DepthToSpace`) specifically to avoid this. If seen, check your data normalization.

**Q: Can I use this for Super-Resolution?**
A: Yes, but you need to modify the final upsampling stage to output larger spatial dimensions. Currently, it outputs $1\times$ resolution.

**Q: Why FFT?**
A: FFT allows the network to distinguish low-frequency info (structure/color) from high-frequency info (noise/edges) globally.

---

## 15. Technical Details

### Frequency Domain Processing
The `FreMLP` block applies:
$$ X_{freq} = \text{FFT}(X) $$
$$ X_{mag} = |X_{freq}|, \quad X_{phase} = \angle X_{freq} $$
$$ X'_{mag} = \text{MLP}(X_{mag}) $$
$$ Y = \text{IFFT}(X'_{mag} \cdot e^{i X_{phase}}) $$

By processing the magnitude, the network adjusts global style and contrast while the phase (which carries structural information) is preserved, ensuring the structure of the restored image remains faithful to the input.

### Dual Residual Learning
Every block calculates:
$$ Y = X + \beta \cdot \text{SpatialPath}(X) $$
$$ Z = Y + \gamma \cdot \text{FrequencyPath}(Y) $$
$\beta$ and $\gamma$ are learnable scalars initialized to 0. This allows the network to start as an Identity function and progressively learn restoration, leading to very stable training dynamics.

---

## 16. Citation

This implementation is based on:

```bibtex
@inproceedings{feijoo2025darkir,
  title={DarkIR: Robust Low-Light Image Restoration},
  author={Feijoo, D. and Benito, J. C. and Garcia, A. and Conde, M. V.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
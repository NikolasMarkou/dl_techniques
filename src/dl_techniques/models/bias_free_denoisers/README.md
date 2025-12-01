# Bias-Free Denoising Models: BFCNN, BF-UNet & ConvUNext

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready Keras 3 implementation of robust, **Bias-Free** neural networks designed for blind image denoising. This collection includes a standard ResNet-style denoiser (**BFCNN**), a U-Net architecture (**BF-UNet**), and a modern U-Net incorporating ConvNeXt V2 blocks (**ConvUNext**).

These models are architected to generalize across different noise levels, solving a common limitation in standard CNN denoisers.

---

## Table of Contents

1. [Overview: The Bias-Free Principle](#1-overview-the-bias-free-principle)
2. [The Problem: Generalization to Noise Levels](#2-the-problem-generalization-to-noise-levels)
3. [Model Architectures](#3-model-architectures)
    - [BFCNN](#31-bfcnn-resnet-style)
    - [BF-UNet](#32-bf-unet-classic-encoder-decoder)
    - [ConvUNext](#33-convunext-modern-backbone)
4. [Deep Supervision](#4-deep-supervision)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Configuration & Variants](#6-configuration--variants)
7. [Comprehensive Usage Examples](#7-comprehensive-usage-examples)
8. [Training Best Practices](#8-training-best-practices)
9. [Technical Details](#9-technical-details)
10. [Citation](#10-citation)

---

## 1. Overview: The Bias-Free Principle

Standard Convolutional Neural Networks (CNNs) typically use an affine transformation $y = wx + b$. While effective for classification, the additive bias term $b$ can be detrimental in image restoration tasks like denoising.

**Bias-Free Networks** enforce $b=0$ throughout the network. This seemingly minor constraint forces the network to learn purely structural filters rather than relying on intensity shifting.

### Key Benefits
1.  **Scaling Invariance**: If the input image is scaled by a factor $\alpha$, the output is also scaled by $\alpha$.
2.  **Robust Generalization**: A model trained on a specific noise level (e.g., $\sigma=15$) performs surprisingly well on unseen noise levels (e.g., $\sigma=50$) without retraining, unlike standard CNNs which often fail outside their training range.
3.  **Modern Architecture Integration**: We apply this principle to three distinct architectures, ranging from lightweight ResNets to modern ConvNeXt-based U-Nets.

---

## 2. The Problem: Generalization to Noise Levels

### The Dilemma of Standard Denoisers

```
┌─────────────────────────────────────────────────────────────┐
│  The "Overfitting to Noise Level" Problem                   │
│                                                             │
│  Standard CNNs (with bias):                                 │
│    - Learn to subtract a specific magnitude of noise.       │
│    - If trained on noise σ=25, they fail on σ=50 because    │
│      the learned bias terms are "hard-coded" to shift       │
│      pixel intensities by specific amounts.                 │
│                                                             │
│  Bias-Free Networks:                                        │
│    - Lack additive constants.                               │
│    - Must learn to separate signal from noise based on      │
│      local structure and texture, independent of absolute   │
│      magnitude.                                             │
└─────────────────────────────────────────────────────────────┘
```

By removing the bias, the mapping becomes linear with respect to scalar multiplication: $f(\alpha x) = \alpha f(x)$. This allows the model to handle varying noise intensities naturally.

---

## 3. Model Architectures

### 3.1 BFCNN (ResNet-Style)
The **Bias-Free CNN** is the baseline model based on Mohan et al. (2020). It uses a sequence of residual blocks without downsampling.
-   **Best for**: Simplicity, lower memory footprint, theoretical baselines.
-   **Structure**: `Conv -> [ResBlock x N] -> Conv`

### 3.2 BF-UNet (Classic Encoder-Decoder)
The **Bias-Free U-Net** applies the bias-free constraint to the classic U-Net architecture.
-   **Best for**: General purpose denoising, multi-scale feature extraction.
-   **Structure**: 4-level encoder-decoder with skip connections and Deep Supervision.

### 3.3 ConvUNext (Modern Backbone)
**ConvUNext** combines the U-Net macro-architecture with **ConvNeXt V2** micro-architecture (inverted bottlenecks, large kernels, Global Response Normalization).
-   **Best for**: State-of-the-art performance, capturing long-range dependencies.
-   **Innovations**:
    -   **7x7 Kernels**: Larger receptive field than standard U-Net (3x3).
    -   **GRN (Global Response Norm)**: Enhances channel contrast for better feature learning.
    -   **Stochastic Depth**: Regularization for deep models.

---

## 4. Deep Supervision

Both **BF-UNet** and **ConvUNext** implement **Deep Supervision**. This technique computes losses not just at the final output, but at intermediate decoder resolutions.

```
┌──────────────────────────────────────────────────────────────────┐
│                   Deep Supervision Data Flow                     │
│                                                                  │
│                  ┌──────────┐                                    │
│       Encoder ──►│Bottleneck│                                    │
│                  └────┬─────┘                                    │
│                       │                                          │
│             ┌─────────▼─────────┐                                │
│             │ Decoder Level 3   ├──────► Aux Output 3 (Low Res)  │
│             └─────────┬─────────┘                                │
│                       │                                          │
│             ┌─────────▼─────────┐                                │
│             │ Decoder Level 2   ├──────► Aux Output 2 (Mid Res)  │
│             └─────────┬─────────┘                                │
│                       │                                          │
│             ┌─────────▼─────────┐                                │
│             │ Decoder Level 1   ├──────► Aux Output 1 (High Res) │
│             └─────────┬─────────┘                                │
│                       │                                          │
│                  Final Output                                    │
└──────────────────────────────────────────────────────────────────┘
```

**Why use it?**
1.  **Gradient Flow**: Combats the vanishing gradient problem in deep U-Nets.
2.  **Hierarchical Learning**: Forces the intermediate layers to learn meaningful representations at their respective scales.

**Note:** During inference, usually only the `Final Output` is used.

---

## 5. Quick Start Guide

### Installation
Ensure you have the required dependencies:
```bash
pip install keras>=3.0 tensorflow>=2.16
```

### Basic Usage

```python
import keras
from dl_techniques.models.bfunet import create_bfunet_variant
from dl_techniques.models.bfconvunext import create_convunext_variant

# 1. Create a Base ConvUNext model for RGB images
# Note: Deep supervision is enabled by default for variants
model = create_convunext_variant(
    variant='base',
    input_shape=(128, 128, 3),
    enable_deep_supervision=True
)

# 2. Compile
# We use a list of losses because of deep supervision (4 outputs for depth=4)
# Give more weight to the final output (index 0)
loss_weights = [1.0, 0.5, 0.25, 0.1]
model.compile(
    optimizer='adam',
    loss='mae',
    loss_weights=loss_weights
)

model.summary()
```

---

## 6. Configuration & Variants

### BFCNN Variants
| Variant | Blocks | Filters | Description |
|:---:|:---:|:---:|:---|
| `tiny` | 2 | 32 | Minimal resource usage |
| `small` | 5 | 48 | ResNet-10 equivalent |
| `base` | 12 | 64 | ResNet-25 equivalent |
| `large` | 25 | 96 | ResNet-50 equivalent |

### ConvUNext Variants
All variants use ConvNeXt V2 blocks with GRN.

| Variant | Depth | Filters | Blocks/Level | Drop Path |
|:---:|:---:|:---:|:---:|:---:|
| `tiny` | 3 | 32 | 2 | 0.0 |
| `small` | 3 | 48 | 2 | 0.1 |
| `base` | 4 | 64 | 3 | 0.1 |
| `large` | 4 | 96 | 4 | 0.2 |
| `xlarge`| 5 | 128 | 5 | 0.3 |

---

## 7. Comprehensive Usage Examples

### Example 1: Training with Deep Supervision
When training with deep supervision, your data generator must provide multiple ground truth targets (one for each output scale).

```python
import numpy as np
from dl_techniques.models.bfconvunext import create_convunext_variant

# Create model
model = create_convunext_variant('small', (64, 64, 3), enable_deep_supervision=True)

# Dummy data
X = np.random.rand(16, 64, 64, 3).astype('float32') # Noisy input
Y = np.random.rand(16, 64, 64, 3).astype('float32') # Clean target

# For deep supervision, we duplicate the target Y for all outputs.
# The model internally handles downsampling the supervision targets if needed,
# or you can resize Y to match the specific output shapes.
# Here we assume the loss function handles the resolution difference 
# (e.g., GlobalAvgPooling or resizing in the loss) OR the model outputs match input size.
# *Note*: In this implementation, Deep Supervision outputs are projected to input channels.
targets = [Y] * len(model.outputs)

model.fit(X, targets, epochs=5)
```

### Example 2: Inference (Single Output)
For deployment, you typically don't want the overhead of calculating the auxiliary outputs.

```python
from dl_techniques.models.bfconvunext import (
    create_convunext_variant,
    create_inference_model_from_training_model
)

# 1. Load trained training model
training_model = create_convunext_variant('base', (256, 256, 3), enable_deep_supervision=True)
# ... load weights ...

# 2. Convert to inference model (extracts only the primary output)
inference_model = create_inference_model_from_training_model(training_model)

# 3. Predict (returns single tensor)
result = inference_model.predict(some_image)
```

### Example 3: Loading Pretrained BF-UNet
BF-UNet supports automatic weight downloading for standard configurations.

```python
from dl_techniques.models.bfunet import create_bfunet_variant

model = create_bfunet_variant(
    'base',
    input_shape=(256, 256, 3),
    pretrained=True,
    weights_dataset='imagenet_denoising'
)
print("✅ Pretrained weights loaded.")
```

---

## 8. Training Best Practices

1.  **Normalization**: While the network is bias-free, inputs should ideally be in a standard range (e.g., [0, 1] or [-1, 1]) and zero-centered if possible, though bias-free networks are robust to simple intensity shifts.
2.  **Loss Function**:
    -   **L1 (MAE)**: Generally produces sharper edges than MSE.
    -   **Charbonnier Loss**: A differentiable variant of L1 ($\sqrt{x^2 + \epsilon^2}$) often works best for denoising.
3.  **Patch Training**: Train on smaller patches (e.g., 64x64 or 128x128) extracted from larger images to increase batch size and diversity. The FCN nature of these models allows them to infer on full-resolution images later.
4.  **Deep Supervision Weighting**: Decay the weights of auxiliary losses. Typically, the deepest supervision (lowest resolution) gets the lowest weight.

---

## 9. Technical Details

### ConvUNext V2 Block Structure
ConvUNext adapts the ConvNeXt V2 block for dense prediction:

1.  **Depthwise Conv (7x7)**: Large receptive field, bias-free (`use_bias=False`).
2.  **LayerNorm**: Standardizes features.
3.  **Pointwise Conv (1x1)**: Expands channels (4x).
4.  **GELU**: Activation.
5.  **GRN**: Global Response Normalization for feature competition.
6.  **Pointwise Conv (1x1)**: Projects channels back.
7.  **Residual Connection**: Adds input to output.

### Bias-Free Scaling Property
Let $\mathcal{F}(\cdot)$ be the neural network.
If every convolution and linear operation has $bias=0$, and activation functions are scale-invariant or linear (like ReLU, LeakyReLU, or Linear), then:
$$ \mathcal{F}(\alpha \cdot \mathbf{I}) = \alpha \cdot \mathcal{F}(\mathbf{I}) $$

*Note: GELU and LayerNorm are not strictly homogeneous, but empirical results show ConvUNext retains strong generalization properties despite this, thanks to the bias-free convolutions.*

---

## 10. Citation

If you use these implementations, please cite the underlying papers:

**Bias-Free CNNs:**
```bibtex
@inproceedings{mohan2020robust,
  title={Robust and Interpretable Blind Image Denoising via Bias-Free Convolutional Neural Networks},
  author={Mohan, S. and Kadambi, A. and Sreevalsan-Nair, J. and Raskar, R.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}
```

**ConvNeXt V2:**
```bibtex
@article{woo2023convnextv2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Woo, Sanghyun and others},
  journal={CVPR},
  year={2023}
}
```
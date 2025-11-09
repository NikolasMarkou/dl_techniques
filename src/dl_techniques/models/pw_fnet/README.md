# Pyramid Wavelet-Fourier Network (PW-FNet)

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of the **Pyramid Wavelet-Fourier Network (PW-FNet)** in **Keras 3**, based on the paper ["Global Modeling Matters: A Fast, Lightweight and Effective Baseline for Efficient Image Restoration"](https://arxiv.org/abs/2407.13663) by Jiang et al. (2024).

The architecture replaces expensive self-attention mechanisms with a highly efficient Fourier Transform-based token mixer within a hierarchical U-Net structure, delivering state-of-the-art performance with a fraction of the computational cost. This implementation is built with modern Keras best practices, ensuring it is robust, easy to understand, and fully serializable.

![image](pw_fnet_intro.jpg)

---

## Table of Contents

1. [Overview: What is PW-FNet and Why It Matters](#1-overview-what-is-pw-fnet-and-why-it-matters)
2. [The Problem PW-FNet Solves](#2-the-problem-pw-fnet-solves)
3. [How PW-FNet Works: Core Concepts](#3-how-pw-fnet-works-core-concepts)
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

## 1. Overview: What is PW-FNet and Why It Matters

### What is a PW-FNet?

The **Pyramid Wavelet-Fourier Network (PW-FNet)** is a deep learning architecture designed for fast and effective image restoration. It challenges the dominance of Transformers by demonstrating that their key advantage—global modeling—can be achieved more efficiently. Instead of using self-attention, PW-FNet captures global context by performing feature mixing in the frequency domain using the Fast Fourier Transform (FFT).

![image](pw_fnet_attention_types.jpg)

Its design is based on a U-Net-like pyramid structure that processes features at multiple scales, analogous to how wavelet transforms decompose signals into different frequency bands.

### Key Innovations

1.  **Fourier Transform as Token Mixer**: The core of PW-FNet is its replacement of the self-attention "token mixer" with a simple yet powerful block that applies a 2D FFT, performs a convolution in the frequency domain, and transforms the result back with an inverse FFT. This captures global information with remarkable efficiency.
2.  **Global Modeling without Attention**: It proves that the global receptive field, a key benefit of Transformers, can be achieved without the quadratic complexity of self-attention. The Fourier transform inherently considers all spatial locations at once.
3.  **Hierarchical Processing**: It uses a U-Net encoder-decoder structure to create and process feature maps at multiple scales, enabling the effective restoration of both large-scale structures and fine-grained details.
4.  **Extreme Efficiency**: By avoiding self-attention, PW-FNet is significantly faster, more memory-efficient, and has far fewer parameters than leading Transformer-based restoration models like Restormer or NAFNet.

![image](pw_fnet_block.jpg)

### Why PW-FNet Matters

**Transformer-based Restoration Problem**:```
Problem: Remove rain from a high-resolution photograph in real-time.
Transformer (e.g., Restormer) Approach:
  1. Divide the image into patches (tokens).
  2. Use a deep stack of Transformer blocks to compute self-attention
     between these tokens, modeling global dependencies.
  3. Limitation: Self-attention has quadratic complexity, making the model
     slow, memory-hungry, and parameter-heavy. Real-time processing on
     edge devices is often infeasible.
```

**PW-FNet's Solution**:```
PW-FNet Approach:
  1. Use a standard U-Net structure for multi-scale feature processing.
  2. Inside each block, instead of self-attention, apply a Fourier transform
     to the feature map to switch to the frequency domain.
  3. Perform a simple, lightweight convolution on the frequency representation.
  4. Transform back to the spatial domain.
  5. Benefit: Achieves a global receptive field with O(N log N) complexity,
     making it exceptionally fast and lightweight while matching or exceeding
     the performance of much heavier models.```

### Real-World Impact

PW-FNet provides a powerful and practical baseline for a wide range of image restoration tasks where efficiency is critical:
-   **Image Deraining & Desnowing**: Removing adverse weather effects.
-   **Image Dehazing**: Restoring clarity in foggy or hazy images.
-   **Underwater Image Enhancement**: Correcting color and visibility distortions.
-   **Motion Deblurring**: Sharpening images affected by motion blur.
-   **Super-Resolution**: Upscaling low-resolution images.

---

## 2. The Problem PW-FNet Solves

### The Limitations of Transformer-based Restoration Models

While Transformer-based models have set new state-of-the-art records in image restoration, their reliance on self-attention mechanisms creates significant practical barriers:
1.  **High Computational Cost**: The quadratic complexity of self-attention makes training and inference slow, especially for high-resolution images.
2.  **Large Memory Footprint**: The attention maps consume a substantial amount of GPU memory.
3.  **High Parameter Count**: These models are often very large, making them difficult to deploy on resource-constrained devices.

```
┌─────────────────────────────────────────────────────────────┐
│  Transformer-based Restoration Models                       │
│                                                             │
│  Primary Tool: Self-Attention for global context.           │
│                                                             │
│  The Efficiency Problem:                                    │
│  - Methods like Restormer and Uformer are highly effective  │
│    but come at a steep computational price.                 │
│  - This complexity limits their use in real-time scenarios  │
│    like autonomous driving or live video enhancement.       │
└─────────────────────────────────────────────────────────────┘
```

### How PW-FNet Changes the Game

PW-FNet provides a paradigm shift by rethinking how global context should be modeled.

```
┌─────────────────────────────────────────────────────────────┐
│  The PW-FNet Solution                                       │
│                                                             │
│  1. Replace the Self-Attention Mechanism: The central idea  │
│     is to swap the complex and costly self-attention module │
│     with an efficient frequency-domain operator.            │
│                                                             │
│  2. Global Modeling via Fourier Transform: The Fourier      │
│     transform is a natural tool for global analysis. Every  │
│     point in the frequency domain depends on every point in │
│     the spatial domain, providing a global receptive field  │
│     by default.                                             │
│                                                             │
│  3. Efficiency First: The entire architecture is built      │
│     around simple, fast operations (Conv2D, FFT, LayerNorm) │
│     making it an order of magnitude more efficient than its │
│     Transformer counterparts.                               │
└─────────────────────────────────────────────────────────────┘
```

The model demonstrates that superior performance doesn't have to come at a high computational cost, making it an ideal baseline for practical applications.

---

## 3. How PW-FNet Works: Core Concepts

### The U-Net Encoder-Decoder Architecture

PW-FNet is built on a familiar and robust 3-level U-Net architecture.
-   **Encoder**: A series of `PW_FNet_Block`s and `Downsample` layers that progressively reduce the spatial resolution while increasing the feature (channel) dimension.
-   **Bottleneck**: A set of `PW_FNet_Block`s that process the features at the lowest resolution.
-   **Decoder**: A series of `PW_FNet_Block`s and `Upsample` layers that restore the spatial resolution. Skip connections pass high-frequency details from the encoder to the decoder.

### The Fourier Token Mixer

The key innovation lies inside the `PW_FNet_Block`. Instead of a self-attention module, it uses a Fourier-based token mixer.

```
┌──────────────────────────────────────────────────────────────────┐
│                        Fourier Token Mixer                       │
│                                                                  │
│  Input Features (Spatial Domain)                                 │
│           │                                                      │
│           ├─► Pointwise Conv (Expand channels)                   │
│           │                                                      │
│           ├─► 2D Fast Fourier Transform (FFT)                    │
│           │      (Switch to Frequency Domain)                    │
│           │                                                      │
│           ├─► Pointwise Conv + GELU (Process frequency info)     │
│           │                                                      │
│           ├─► 2D Inverse FFT (IFFT)                              │
│           │      (Switch back to Spatial Domain)                 │
│           │                                                      │
│           └─► Pointwise Conv (Project back to original channels) │
│                                                                  │
│  Output Features (Spatial Domain)                                │
└──────────────────────────────────────────────────────────────────┘
```
This entire sequence is computationally lightweight and effectively mixes information across the entire feature map.

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PW-FNet Complete Data Flow                          │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: ENCODING
────────────────
Input Image (B, H, W, 3)
    │
    ├─► Intro Conv2D → (B, H, W, D)
    │
    ├─► Encoder Level 1 (PW-FNet Blocks) ───(Skip1)───►
    │
    ├─► Downsample → (B, H/2, W/2, 2D)
    │
    ├─► Encoder Level 2 (PW-FNet Blocks) ───(Skip2)───►
    │
    └─► Downsample → (B, H/4, W/4, 4D)


STEP 2: BOTTLENECK
──────────────────
Encoded Features
    │
    └─► Middle Blocks (PW-FNet Blocks)


STEP 3: DECODING
────────────────
Bottleneck Features
    │
    ├─► Upsample → (B, H/2, W/2, 2D) ◄───(Concat Skip2)─┐
    │                                                   │
    ├─► Reduce Conv2D → (B, H/2, W/2, 2D) <─────────────┘
    │
    ├─► Decoder Level 2 (PW-FNet Blocks)
    │
    ├─► Upsample → (B, H, W, D) ◄──────(Concat Skip1)─┐
    │                                                 │
    ├─► Reduce Conv2D → (B, H, W, D) <────────────────┘
    │
    └─► Decoder Level 1 (PW-FNet Blocks)


STEP 4: MULTI-SCALE OUTPUT
──────────────────────────
The model produces residual images at 3 resolutions, which are
added to downsampled versions of the original input to produce the final outputs.
[out_full_res, out_half_res, out_quarter_res]
```

---

## 4. Architecture Deep Dive

### 4.1 `PW_FNet_Block`

This is the core building block of the model. It is a residual block containing two main sub-modules:
1.  **Fourier Token Mixer**: As described above, this module provides global feature mixing. It is wrapped in a residual connection.
2.  **Feed-Forward Network (FFN)**: A simple MLP-like module consisting of a pointwise convolution to expand channels, a 3x3 depthwise convolution to capture local patterns, a GELU activation, and another pointwise convolution to project back. This is also wrapped in a residual connection.

Each sub-module is preceded by a `LayerNormalization` layer for stable training.

### 4.2 Scaling Layers

-   **`Downsample`**: A `Conv2D` layer with a kernel size of 4 and a stride of 2 is used to learn an optimal way to halve the spatial dimensions and double the channel dimension.
-   **`Upsample`**: A `Conv2DTranspose` layer with a kernel size of 2 and a stride of 2 is used to double the spatial dimensions and halve the channel dimension.

### 4.3 Multi-Scale Outputs

The model's `call` method returns a list of three tensors, corresponding to the restored image at **full, half, and quarter resolution**. This is designed to facilitate **hierarchical supervision**, where a loss is computed at each scale to guide the network more effectively during training. The final output is a residual added to the original (downsampled) input, which helps the model focus on learning the degradation.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy matplotlib
```

### Your First Restoration Model (30 seconds)

Let's train a PW-FNet to remove synthetic noise from images.

```python
import keras
import numpy as np
import matplotlib.pyplot as plt

# Assuming model.py is in your project directory
from model import PW_FNet

# 1. Generate dummy data
def generate_data(num_samples, shape=(64, 64, 3)):
    clean_images = np.random.rand(num_samples, *shape).astype("float32")
    noise = np.random.normal(0, 0.1, clean_images.shape).astype("float32")
    noisy_images = np.clip(clean_images + noise, 0.0, 1.0)
    return noisy_images, clean_images

X_train_noisy, X_train_clean = generate_data(128)
X_test_noisy, X_test_clean = generate_data(20)

# 2. Create a lightweight PW-FNet model for the demo
model = PW_FNet(
    img_channels=3,
    width=32,          # Base channel width
    middle_blk_num=2,  # Blocks in the bottleneck
    enc_blk_nums=[1, 1], # Blocks in encoder stages
    dec_blk_nums=[1, 1]  # Blocks in decoder stages
)

# 3. Compile the model with a loss for each of the 3 outputs
model.compile(
    optimizer="adam",
    # Use the same loss for all three output scales
    loss="mean_absolute_error"
)
print("✅ PW-FNet model created and compiled successfully!")

# 4. Prepare target data for multi-scale supervision
# The model outputs a list of 3 images [full, half, quarter].
# We need to provide a corresponding list/tuple of ground truths for training.
y_train_full = X_train_clean
y_train_half = keras.layers.AveragePooling2D(2)(X_train_clean)
y_train_quarter = keras.layers.AveragePooling2D(2)(y_train_half)

# 5. Train the model
model.fit(
    X_train_noisy,
    (y_train_full, y_train_half, y_train_quarter),
    epochs=10,
    batch_size=16,
    verbose=1
)
print("✅ Training Complete!")

# 6. Restore a test image and visualize
# The model.predict() returns a list of the 3 restored images
restored_images = model.predict(X_test_noisy[:1])
full_res_restored = restored_images[0]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(X_test_noisy[0])
plt.title("Noisy Input")
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(full_res_restored[0])
plt.title("Restored Output")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(X_test_clean[0])
plt.title("Ground Truth")
plt.axis('off')
plt.show()

```

---

## 6. Component Reference

### 6.1 `PW_FNet` (Model Class)

**Purpose**: The main Keras `Model` that assembles the full U-Net architecture.

**Key Parameters**:

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `img_channels` | `int` | Number of channels for input/output images (e.g., 3 for RGB). | `3` |
| `width` | `int` | The base channel width of the network. Controls model capacity. | `32` |
| `middle_blk_num` | `int` | Number of `PW_FNet_Block`s in the bottleneck. | `4` |
| `enc_blk_nums` | `List[int]` | List of block counts for each encoder stage (from high-res to low-res). | `[2, 2]` |
| `dec_blk_nums` | `List[int]` | List of block counts for each decoder stage (from low-res to high-res). | `[2, 2]` |

### 6.2 `PW_FNet_Block` (Layer Class)

**Purpose**: The core feature processing block containing the Fourier Token Mixer and the Feed-Forward Network. It maintains the input tensor's shape.

### 6.3 Utility Layers

- **`FFTLayer`**: A serializable layer that applies a 2D FFT and concatenates the real and imaginary parts into a real-valued tensor.
- **`IFFTLayer`**: A serializable layer that performs the inverse operation of `FFTLayer`.
- **`Downsample`**: A trainable strided `Conv2D` layer for downsampling.
- **`Upsample`**: A trainable `Conv2DTranspose` layer for upsampling.

---

## 7. Configuration & Model Variants

The paper derives small, medium, and large model variants from a single trained network by using the different multi-scale outputs during inference.

-   **PW-FNet-S (Small)**: Use the quarter-resolution output (`outputs[2]`) and upsample it twice. Fastest but lowest quality.
-   **PW-FNet-M (Medium)**: Use the half-resolution output (`outputs[1]`) and upsample it once. Balanced performance.
-   **PW-FNet-L (Large)**: Use the full-resolution output (`outputs[0]`) directly. Highest quality but slowest.

This allows for dynamic, adaptive inference based on the available computational budget.

---

## 8. Comprehensive Usage Examples

### Example 1: Inference with a Specific Output Scale

During inference, you typically only need the full-resolution output.

```python
# Assuming 'model' is trained
noisy_image = np.random.rand(1, 128, 128, 3).astype("float32")

# The model returns a list of 3 images
all_outputs = model.predict(noisy_image)
full_resolution_output = all_outputs[0]

print(f"Shape of full-res output: {full_resolution_output.shape}")
```

### Example 2: Building a Training Pipeline with Weighted Hierarchical Loss

For best results, you can use a custom training loop or a compiled loss that weights the outputs from the three scales.

```python
# Define a loss function that applies to the multi-scale outputs
def hierarchical_loss(y_true, y_pred):
    # y_true and y_pred are lists/tuples of 3 tensors each
    y_true_l0, y_true_l1, y_true_l2 = y_true
    y_pred_l0, y_pred_l1, y_pred_l2 = y_pred

    loss_l0 = keras.losses.mean_absolute_error(y_true_l0, y_pred_l0)
    loss_l1 = keras.losses.mean_absolute_error(y_true_l1, y_pred_l1)
    loss_l2 = keras.losses.mean_absolute_error(y_true_l2, y_pred_l2)

    # Give more weight to the full-resolution loss
    return (0.6 * loss_l0) + (0.3 * loss_l1) + (0.1 * loss_l2)

# model.compile(optimizer="adam", loss=hierarchical_loss)
# model.fit(...)
```

---

## 11. Training and Best Practices

### Loss Function

The original paper finds that an **L1 loss in the Fourier domain** (`L = ||F(output) - F(ground_truth)||_1`) yields the best results. However, a standard spatial-domain L1 loss (Mean Absolute Error) with hierarchical supervision, as shown in the examples, is also a very strong and simpler alternative.

### Optimizer and Schedule

The authors use the **AdamW optimizer** with a **cosine annealing learning rate schedule**. This combination is standard for training high-performance vision models and is recommended for PW-FNet.

---

## 12. Serialization & Deployment

The `PW_FNet` model and all its custom sub-layers are **fully serializable** using Keras 3's modern `.keras` format. Each custom layer is decorated with `@keras.saving.register_keras_serializable()` and implements `get_config`, ensuring that the model's architecture and weights can be saved and loaded seamlessly.

### Saving and Loading

```python
# Create and train model
# ... (see Quick Start guide)

# Save the entire model to a single file
model.save('pwfnet_denoiser.keras')
print("Model saved to pwfnet_denoiser.keras")

# Load the model in a new session. Keras 3 handles custom objects automatically.
loaded_model = keras.models.load_model('pwfnet_denoiser.keras')
print("Model loaded successfully")

# Verify the loaded model can make predictions
restored = loaded_model.predict(X_test_noisy[:1])
print(f"Restored image shape: {restored[0].shape}")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Checkerboard artifacts in the output.**

-   **Cause**: This is a common issue with transposed convolutions (`Upsample` layer).
-   **Solution**: The current implementation uses `kernel_size=2, strides=2`, which is the standard way to mitigate this. If artifacts persist, you can replace the layer with a sequence of `UpSampling2D` followed by a `Conv2D`, which often produces smoother results at a slight computational cost.

### Frequently Asked Questions

**Q: Why use Fourier Transform instead of Self-Attention?**

A: **Efficiency**. The 2D FFT has a computational complexity of O(N log N), where N is the number of pixels. Global self-attention has a complexity of O(N²). For a 256x256 image, this is a massive difference, allowing PW-FNet to be orders of magnitude faster while still modeling global dependencies.

**Q: How does this relate to Wavelets?**

A: The "Wavelet" in the name refers to the model's multi-scale (or multi-frequency) analysis. The U-Net's pyramid structure, which processes the image at different resolutions, is analogous to a discrete wavelet transform that decomposes a signal into different frequency sub-bands. The Fourier transform is then applied to the features *within* each of these scales.

---

## 15. Technical Details

### The `PW_FNet_Block`

The core block consists of two main residual operations:
1.  **Token Mixer**: `x_res = Project(IFFT(GELU(Conv_freq(FFT(Expand(Norm(x)))))))`
    `x = x + x_res`
2.  **Feed-Forward Network**: `x_ffn = Project(GELU(DepthwiseConv(Expand(Norm(x)))))`
    `x = x + x_ffn`

This design is simple, efficient, and highly effective.

---

## 16. Citation

If you use PW-FNet in your research, please cite the original paper:

```bibtex
@article{jiang2024global,
  title={Global Modeling Matters: A Fast, Lightweight and Effective Baseline for Efficient Image Restoration},
  author={Jiang, Xingyu and Gao, Ning and Dou, Hongkun and Zhang, Xiuhui and Zhong, Xiaoqing and Deng, Yue and Li, Hongjue},
  journal={arXiv preprint arXiv:2407.13663},
  year={2024}
}```
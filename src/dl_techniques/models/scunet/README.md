# SCUNet: Swin-Conv U-Net for Image Restoration

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

A production-ready, Keras 3 implementation of **SCUNet**, a high-performance, U-Net-style architecture designed for image restoration tasks such as denoising, deblurring, and artifact removal. Its primary innovation is the `SwinConvBlock`, a hybrid layer that synergistically combines the strengths of Swin Transformers and convolutional neural networks.

This implementation provides a modular, fully serializable `keras.Model` that faithfully reproduces the SCUNet architecture. By leveraging the local feature extraction power of CNNs and the long-range dependency modeling of Swin Transformers, SCUNet achieves state-of-the-art results in various image restoration benchmarks.

---

## Table of Contents

1.  [Overview: What is SCUNet?](#1-overview-what-is-scunet)
2.  [The Problem SCUNet Solves](#2-the-problem-scunet-solves)
3.  [How SCUNet Works: Core Concepts](#3-how-scunet-works-core-concepts)
4.  [Architecture Deep Dive](#4-architecture-deep-dive)
5.  [Quick Start Guide](#5-quick-start-guide)
6.  [Component Reference](#6-component-reference)
7.  [Configuration & Key Hyperparameters](#7-configuration--key-hyperparameters)
8.  [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9.  [Training and Best Practices](#9-training-and-best-practices)
10. [Serialization & Deployment](#10-serialization--deployment)
11. [Testing & Validation](#11-testing--validation)
12. [Troubleshooting & FAQs](#12-troubleshooting--faqs)
13. [Technical Details](#13-technical-details)
14. [Citation](#14-citation)

---

## 1. Overview: What is SCUNet?

### What is SCUNet?

**SCUNet** (Swin-Conv U-Net) is a deep neural network tailored for image restoration. It integrates a novel building block, the `SwinConvBlock`, into a U-Net architecture. This block processes features through two parallel pathways—a convolutional path and a Swin Transformer path—allowing the model to simultaneously capture fine-grained local details and global contextual information.

### The Hybrid Approach

The core strength of SCUNet lies in its hybrid design:

1.  **Convolutional Path**: A standard residual block of 3x3 convolutions excels at learning local patterns, textures, and spatial hierarchies with high efficiency and strong inductive biases.
2.  **Swin Transformer Path**: A `SwinTransformerBlock` uses windowed self-attention to model long-range dependencies and contextual relationships between distant parts of the image, overcoming the limited receptive field of pure CNNs.

These two paths are fused at every block, creating a richer and more powerful feature representation that is ideal for reconstructing high-quality images from degraded inputs.

### Why It Matters

**The Image Restoration Challenge**:```
Problem: Reconstruct a clean, high-quality image from a degraded version (e.g., noisy, blurry).

Pure CNN Approach:
  1. Use architectures like U-Net with convolutional blocks.
  2. Limitation: While excellent for local features, they struggle to model
     long-range dependencies due to their limited receptive field. This can
     result in a lack of global coherence in the restored image.
  3. Result: Good texture restoration but potential for structural inconsistencies.

**SCUNet's Hybrid Solution**:

SCUNet's Approach:
  1. Use a U-Net backbone for its proven multi-scale processing capabilities.
  2. Replace standard convolutional blocks with the hybrid SwinConvBlock.
  3. Benefit: The model can now simultaneously learn local textures (via the
     conv path) and understand the global structure of the image (via the
     Swin path). This leads to restorations that are both locally sharp and
     globally consistent.

---

## 2. The Problem SCUNet Solves

### The Duality of Image Restoration

Effective image restoration requires a model to perform two seemingly contradictory tasks:
1.  **Preserve High-Frequency Details**: Reconstructing sharp edges, fine textures, and intricate patterns.
2.  **Understand Global Context**: Recognizing overarching structures, shapes, and object relationships to fill in missing information coherently.

+-------------------------------------------------------------+
|           The Architectural Dilemma in Image Restoration    |
|                                                             |
|   Pure CNNs (e.g., ResNet, U-Net):                          |
|     - Strong inductive biases (locality, translation        |
|       equivariance) make them data-efficient for local      |
|       patterns.                                             |
|     - Their receptive field grows slowly with depth, making |
|       it difficult to capture long-range dependencies.      |
|                                                             |
|   Pure Transformers (e.g., ViT):                            |
|     - Excel at modeling global relationships with self-     |
|       attention.                                            |
|     - Lack inductive biases, requiring massive datasets.    |
|     - Suffer from quadratic complexity, making them         |
|       inefficient for high-resolution images.               |
+-------------------------------------------------------------+

SCUNet, with its Swin Transformer backbone, offers a solution. It achieves **linear complexity** with respect to image size while effectively modeling global context through shifted window attention, and it retains the powerful local feature extraction of CNNs.

---

## 3. How SCUNet Works: Core Concepts

SCUNet is a U-Net with an encoder, a bottleneck, and a decoder, connected by skip connections. The magic happens inside its core building block, the `SwinConvBlock`.

```
Input Tensor (B, H, W, C)
      |
      |-------------------------------------> Shortcut
      |
[ 1x1 Conv ]
      |
      +---- Split Channels ----+
      |                        |
(conv_dim)                 (trans_dim)
      |                        |
      v                        v
+------------------+     +----------------------+
| CONV PATH        |     | TRANSFORMER PATH     |
| (Local Features) |     | (Global Context)     |
|------------------|     |----------------------|
| Residual         |     | SwinTransformerBlock |
| Conv Block       |     | (W-MSA / SW-MSA)     |
+------------------+     +----------------------+
      |                        |
      |                        |
      +---- Concatenate -----+
                   |
             [ 1x1 Conv ]
             (Feature Fusion)
                   |
                   +----------------------> Add Shortcut
                   |
                   v
      Output Tensor (B, H, W, C)
```
This "split-transform-merge" design allows the network to learn complementary features at each stage of the U-Net hierarchy.

---

## 4. Architecture Deep Dive

### 4.1 `SwinConvBlock`

-   **Purpose**: To create a rich feature representation by combining local and global information.
-   **Mechanism**:
    1.  **Split**: An initial 1x1 convolution prepares the features, which are then split along the channel axis.
    2.  **Transform**: One half goes to a standard two-layer 3x3 convolutional block. The other half goes to a `SwinTransformerBlock`.
    3.  **Merge**: The outputs are concatenated and fused by a final 1x1 convolution. A residual connection from the block's input stabilizes training.

### 4.2 `SwinTransformerBlock`

-   **Purpose**: To efficiently model long-range dependencies.
-   **Mechanism**:
    1.  **Windowed Multi-Head Self-Attention (W-MSA)**: Instead of computing self-attention across the entire image (quadratically expensive), it divides the image into non-overlapping windows (e.g., 8x8 pixels) and computes attention *within* each window.
    2.  **Shifted Window MSA (SW-MSA)**: In alternating blocks, the windows are shifted. This allows information to be exchanged across window boundaries, enabling a global receptive field over successive layers.

### 4.3 U-Net Structure

-   **Encoder**: A series of `SwinConvBlock` stages that progressively downsample the feature maps (using strided convolutions), increasing the receptive field and channel depth.
-   **Bottleneck**: A final stage of `SwinConvBlock`s at the lowest resolution to process the most abstract features.
-   **Decoder**: A series of stages that upsample the feature maps (using `Conv2DTranspose`) and merge them with high-resolution features from the corresponding encoder stage via skip connections. This allows the model to reconstruct fine details while using contextual information from the deeper layers.

---

## 5. Quick Start Guide

### Installation

```bash
# Install Keras 3 and a backend (e.g., TensorFlow)
pip install keras tensorflow numpy
```

### Basic Usage (Instantiation and Forward Pass)

This example shows how to create an SCUNet model and run a dummy tensor through it.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.scunet.model import SCUNet

# --- Dummy Data ---
# A batch of 4 noisy images, 256x256 pixels, 3 channels (RGB)
dummy_noisy_images = np.random.rand(4, 256, 256, 3).astype("float32")
# ---

# 1. Initialize SCUNet
# Model for a standard RGB image restoration task.
# 'dim' controls the capacity of the model.
model = SCUNet(
    in_nc=3,
    dim=64,
    window_size=8,
    input_resolution=256
)

# 2. Perform a forward pass
# The output will be the restored image.
restored_images = model(dummy_noisy_images)

# 3. Check the output shape
# SCUNet preserves the input dimensions.
print(f"Input shape:  {dummy_noisy_images.shape}")
print(f"Output shape: {restored_images.shape}")

# 4. Display model summary
model.summary()
```

---

## 6. Component Reference

| Component                 | Location                     | Purpose                                                                          |
| :------------------------ | :--------------------------- | :------------------------------------------------------------------------------- |
| **`SCUNet`**              | `model.SCUNet`               | The main Keras `Model` assembling the U-Net architecture.                        |
| **`SwinConvBlock`**       | `swin_conv_block.SwinConvBlock` | The hybrid block combining parallel CNN and Swin Transformer pathways.         |
| **`SwinTransformerBlock`**| `swin_transformer_block.SwinTransformerBlock` | The core Swin Transformer layer with (shifted) windowed self-attention. |

---

## 7. Configuration & Key Hyperparameters

The `SCUNet` model is configured via its constructor. Key parameters include:

| Parameter            | Type        | Default             | Description                                                                    |
| :------------------- | :---------- | :------------------ | :----------------------------------------------------------------------------- |
| `in_nc`              | `int`       | `3`                 | Number of input channels (e.g., 3 for RGB, 1 for grayscale).                     |
| `config`             | `List[int]` | `[4,4,4,4,4,4,4]`   | Number of `SwinConvBlock`s in each stage of the U-Net (3 down, 1 body, 3 up).    |
| `dim`                | `int`       | `64`                | The base channel dimension. Controls the model's width and capacity.           |
| `head_dim`           | `int`       | `32`                | Dimension of each attention head in the Swin blocks.                           |
| `window_size`        | `int`       | `8`                 | The size of the attention windows (e.g., 8x8 pixels).                          |
| `drop_path_rate`     | `float`     | `0.0`               | Stochastic depth rate for regularization. Linearly increases from 0 to this value. |
| `input_resolution`   | `int`       | `256`               | The expected input image resolution. Used to optimize windowing.               |

---

## 8. Comprehensive Usage Examples

### Example 1: Grayscale Image Denoising

```python
import keras
import numpy as np
from dl_techniques.models.scunet.model import SCUNet

# 1. Configure the model for a grayscale (1 channel) denoising task
scunet_denoiser = SCUNet(
    in_nc=1,
    config=[2, 2, 2, 2, 2, 2, 2], # A lighter config for faster training
    dim=48,
    input_resolution=128
)

# 2. Create dummy data
# 16 grayscale images of size 128x128
x_train_noisy = np.random.rand(16, 128, 128, 1).astype("float32")
y_train_clean = np.random.rand(16, 128, 128, 1).astype("float32")

# 3. Compile the model
# For restoration, L1 loss (Mean Absolute Error) is often a good choice.
scunet_denoiser.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='mean_absolute_error'
)

print("✅ Model compiled for grayscale denoising.")

# 4. Train the model (snippet)
# history = scunet_denoiser.fit(
#     x_train_noisy,
#     y_train_clean,
#     batch_size=4,
#     epochs=50,
#     validation_split=0.1
# )
```

---

## 9. Training and Best Practices

-   **Standard Training**: `SCUNet` is a standard Keras model and is trained with `model.compile()` and `model.fit()`.
-   **Loss Function**: For image restoration, L1 loss (`mean_absolute_error`) is often preferred over L2 loss (`mean_squared_error`) as it can produce sharper images.
-   **Input Normalization**: Scale input pixel values to a consistent range, such as `[0, 1]` or `[-1, 1]`, before feeding them to the model.
-   **Learning Rate Scheduler**: Using a learning rate scheduler (e.g., `ReduceLROnPlateau` or a cosine decay schedule) is highly recommended for stable training and achieving the best performance.
-   **Input Resolution**: While the model's padding handles arbitrary input sizes, performance is optimal when the input resolution is a multiple of the `window_size` and the total downsampling factor (64). The `input_resolution` parameter helps the model manage attention windows, especially for small feature maps.

---

## 10. Serialization & Deployment

The `SCUNet` model and all its custom layers are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
# Create and train the model as shown previously
import keras
from dl_techniques.models.scunet.model import SCUNet

model = SCUNet(in_nc=3, dim=64)
# model.fit(...)

# Save the entire model to a single file
model.save('scunet_denoiser.keras')

# Load the model, including custom layers and optimizer state, in a new session
loaded_model = keras.models.load_model('scunet_denoiser.keras')
print("✅ SCUNet model loaded successfully!")
```

---

## 11. Testing & Validation

A `pytest` test to ensure the critical serialization cycle is robust.

```python
import pytest
import numpy as np
import keras
import tempfile
import os
from dl_techniques.models.scunet.model import SCUNet

def test_scunet_serialization_cycle():
    """CRITICAL TEST: Ensures a model can be saved and reloaded without error."""
    model = SCUNet(in_nc=1, dim=32, config=[1, 1, 1, 1, 1, 1, 1])
    dummy_input = np.random.rand(2, 64, 64, 1).astype("float32")

    # A forward pass is needed to build the model
    original_prediction = model.predict(dummy_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_model.keras")
        model.save(filepath)
        loaded_model = keras.models.load_model(filepath)

    loaded_prediction = loaded_model.predict(dummy_input)

    # Check if the predictions are identical
    np.testing.assert_allclose(
        original_prediction, loaded_prediction, rtol=1e-5, atol=1e-5
    )
    print("✓ Serialization cycle test passed!")
```

---

## 12. Troubleshooting & FAQs

**Issue 1: I'm getting an error about channel dimensions during training.**

-   **Cause**: The number of input channels in your data does not match the `in_nc` parameter set during model initialization.
-   **Solution**: Ensure `in_nc` is set correctly (e.g., `in_nc=3` for RGB data, `in_nc=1` for grayscale). Also, verify that the final layer of your data loading pipeline produces tensors with the correct channel dimension.

**Issue 2: Training is very slow or uses too much memory.**

-   **Cause**: The model's capacity might be too large for your hardware.
-   **Solution**:
    1.  **Reduce `dim`**: This is the most effective way to decrease parameters and memory usage. Try `dim=48` or `dim=32`.
    2.  **Reduce `config`**: Lower the number of blocks in each stage (e.g., `config=[2,2,2,2,2,2,2]`).
    3.  **Reduce batch size**: This is a direct trade-off between training speed and memory consumption.

### Frequently Asked Questions

**Q: Why combine Swin and Conv? Why not just a better CNN or a pure Transformer?**

A: This combination leverages the complementary strengths of both architectures. CNNs provide a powerful and efficient inductive bias for local features, which Transformers lack. Swin Transformers provide an efficient way to model the global context that CNNs struggle to capture. The combination results in a model that is both efficient and highly effective across different image scales.

**Q: How does this compare to a standard U-Net?**

A: A standard U-Net relies solely on convolutions. While very effective, its ability to model long-range dependencies is limited. By replacing the core blocks with `SwinConvBlock`s, SCUNet gains a much stronger ability to understand global image structure, which often leads to superior performance in tasks requiring significant context, like inpainting or deblurring complex scenes.

---

## 13. Technical Details

### Linear Complexity of Windowed Attention

A traditional Vision Transformer (ViT) computes self-attention across all patches of an image. For an image with `N=HW` patches, the complexity is `O(N^2)`. This is computationally prohibitive for high-resolution images.

**Swin Transformer's Solution**: Attention is computed only within local windows of a fixed size `M` (e.g., 8x8). The complexity becomes `O(M^2 * N)`, which is **linear** with respect to the number of patches (image size `N`), making it scalable.

### Cross-Window Communication via Shifting

To prevent the model from only learning within isolated windows, the **Shifted Window (SW-MSA)** mechanism is used. In every second `SwinTransformerBlock`, the grid of windows is shifted by half a window size (`M/2`). This forces the new windows to be composed of patches from adjacent windows in the previous layer, effectively mixing information across the entire feature map as the network deepens.

---

## 14. Citation

This implementation is based on the original research paper. If you use this model in your work, please cite:

```bibtex
@inproceedings{zhang2021scunet,
  title={SCUNet: Swin-Conv U-Net for Image Restoration},
  author={Zhang, Kai and Li, Yawei and Li, Kangkang and Wang, Lichao and Zuo, Wangmeng and Liu, Ji-Eun and Zhang, Lei},
  booktitle={Proceedings of the 21st International Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages={3638--3647},
  year={2021}
}
```
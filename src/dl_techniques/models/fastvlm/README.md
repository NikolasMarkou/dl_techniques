# FastVLM: A Fast Hybrid Vision Model

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of **FastVLM**, a hybrid vision architecture that combines the strengths of efficient convolutions and Transformers. This implementation is inspired by recent research on models like FastViT and RepMixer, designed to achieve a state-of-the-art balance between accuracy, latency, and model size.

This implementation follows the `dl_techniques` framework standards and modern Keras 3 best practices. It provides a modular, well-documented, and fully serializable model that works seamlessly across TensorFlow, PyTorch, and JAX backends. The architecture features a `ConvolutionalStem` using MobileOne blocks, `RepMixer` stages for efficient feature mixing, and `Attention` stages for capturing global context.

---

## Table of Contents

1. [Overview: What is FastVLM and Why It Matters](#1-overview-what-is-fastvlm-and-why-it-matters)
2. [The Problem FastVLM Solves](#2-the-problem-fastvlm-solves)
3. [How FastVLM Works: Core Concepts](#3-how-fastvlm-works-core-concepts)
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

## 1. Overview: What is FastVLM and Why It Matters

### What is FastVLM?

**FastVLM** is a hybrid vision model designed for high efficiency and performance. It strategically combines different types of layers to create a hierarchical architecture that is both fast and accurate. Instead of relying purely on self-attention like a standard Vision Transformer (ViT), it uses computationally cheaper convolutional and mixer-style blocks in the early stages and reserves powerful (but expensive) attention blocks for the later stages.

### Key Innovations

1.  **Hybrid Architecture**: The model starts with a convolutional stem, transitions to efficient `RepMixer` blocks for spatial and channel mixing, and finishes with standard Transformer `Attention` blocks. This leverages the strengths of each component at the most appropriate stage of feature extraction.
2.  **RepMixer Blocks**: In the early and middle stages, the model uses `RepMixer` blocks as a lightweight alternative to self-attention. `RepMixer` decouples spatial mixing (using depthwise convolutions) and channel mixing (using 1x1 convolutions), achieving effective feature interaction with linear complexity.
3.  **Efficient Convolutional Stem**: The initial feature extraction is handled by a `ConvolutionalStem` built from `MobileOne` blocks, which use structural reparameterization to be efficient at inference time.
4.  **Hierarchical Structure**: Like a classic CNN, the model processes images in stages, progressively downsampling the spatial resolution while increasing the number of feature channels. This allows it to learn features at multiple scales.

### Why FastVLM Matters

**Standard Vision Transformer (ViT) Problem**:
```
Problem: Classify a high-resolution image efficiently.
ViT Approach:
  1. Split the image into patches and process them with a deep stack of
     self-attention layers.
  2. Limitation: Self-attention has O(N²) complexity, where N is the number of
     patches. For high-resolution images, N becomes very large, making the
     model slow and memory-hungry.
  3. Result: Standard ViTs struggle with high resolutions and are often too
     slow for real-time or on-device applications.
```

**FastVLM's Solution**:
```
FastVLM Approach:
  1. Use an efficient convolutional stem to quickly downsample the image.
  2. Use RepMixer blocks in the early stages. These have linear complexity
     (O(N)) and are very fast at processing the larger feature maps. [8]
  3. Reserve the expensive self-attention blocks for the final stage, where the
     number of patches is much smaller (N/16).
  4. Benefit: Achieves a superior speed-accuracy trade-off by applying the
     right tool for the right job at each stage of the network. [10]
```

### Real-World Impact

FastVLM is designed for applications where both high accuracy and low latency are critical:

-   📱 **On-Device AI**: Efficient enough to run on mobile phones, enabling real-time image recognition, augmented reality, and visual search.
-   🚗 **Autonomous Systems**: Provides fast and accurate perception for robotics and self-driving cars.
-   💨 **Real-Time Video Analysis**: Can process frames from a video stream at a high rate.
-   ☁️ **Efficient Cloud Deployment**: Reduces computational cost and energy consumption for large-scale image processing services.

---

## 2. The Problem FastVLM Solves

### The Efficiency-Accuracy Trade-off

In computer vision, there has long been a trade-off between model accuracy and computational efficiency.

```
┌─────────────────────────────────────────────────────────────┐
│  The Dilemma of Vision Architectures                        │
│                                                             │
│  Convolutional Neural Networks (CNNs):                      │
│    - Highly efficient due to local operations and shared weights.
│    - Struggle to model long-range, global dependencies.     │
│                                                             │
│  Vision Transformers (ViTs):                                │
│    - Excellent at modeling global context via self-attention.
│    - Suffer from quadratic complexity, making them slow and │
│      inefficient, especially with high-resolution inputs.   │
└─────────────────────────────────────────────────────────────┘
```

Designing a model that captures the global context of a Transformer while retaining the speed of a CNN is a major challenge.

### How FastVLM Changes the Game

FastVLM provides a principled hybrid architecture that explicitly optimizes this trade-off.

```
┌─────────────────────────────────────────────────────────────┐
│  The FastVLM Hybrid Strategy                                │
│                                                             │
│  1. Early Stages (Large Feature Maps):                      │
│     - Use efficient, convolution-based layers (Stem and     │
│       RepMixer) that have linear complexity.                │
│     - Focus on learning local patterns and textures quickly.│
│                                                             │
│  2. Later Stages (Small Feature Maps):                      │
│     - Introduce powerful self-attention blocks.             │
│     - The quadratic cost is now manageable because the      │
│       sequence length (number of patches) is small.         │
│     - Focus on integrating local features into a global,    │
│       context-aware representation.                         │
└─────────────────────────────────────────────────────────────┘
```

This staged approach ensures that the most computationally expensive operations are only applied when absolutely necessary, on smaller feature maps where they provide the most value.

---

## 3. How FastVLM Works: Core Concepts

### The Hierarchical Multi-Stage Architecture

FastVLM processes an image in four main phases, progressively refining the feature representation.

```
┌──────────────────────────────────────────────────────────────────┐
│                     FastVLM Architecture Stages                  │
│                                                                  │
│  Input Image ───►┌──────────────────┐  (Downsamples 4x)          │
│   (H, W)         │ ConvolutionalStem│                            │
│                  └────────┬─────────┘                            │
│                           │ (H/4, W/4)                           │
│                  ┌────────▼─────────┐  (Maintains resolution)    │
│                  │      Stage 1     │                            │
│                  │ (RepMixer Blocks)│                            │
│                  └────────┬─────────┘                            │
│                           │ Downsample 2x                        │
│                           ▼ (H/8, W/8)                           │
│                  ┌────────▼─────────┐  (Maintains resolution)    │
│                  │      Stage 2     │                            │
│                  │ (RepMixer Blocks)│                            │
│                  └────────┬─────────┘                            │
│                           │ Downsample 2x                        │
│                           ▼ (H/16, W/16)                         │
│                  ┌────────▼──────────┐ (Maintains resolution)    │
│                  │      Stage 3      │                           │
│                  │ (Attention Blocks)│                           │
│                  └────────┬──────────┘                           │
│                           │                                      │
│                  ┌────────▼──────────┐                           │
│                  │ Classification Head│                          │
│                  └────────────────────┘                          │
└──────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FastVLM Complete Data Flow                          │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: CONVOLUTIONAL STEM
──────────────────────────
Input Image (B, H, W, 3)
    │
    ├─► ConvolutionalStem (using MobileOne blocks)
    │
    └─► Feature Map 0: (B, H/4, W/4, D₀)


STEP 2: STAGE 1 (RepMixer)
──────────────────────────
Feature Map 0
    │
    ├─► Stack of RepMixer Blocks
    │
    └─► Feature Map 1: (B, H/4, W/4, D₀)


STEP 3: STAGE 2 (RepMixer)
──────────────────────────
Feature Map 1
    │
    ├─► Downsampling Conv2D -> (B, H/8, W/8, D₁)
    │
    ├─► Stack of RepMixer Blocks
    │
    └─► Feature Map 2: (B, H/8, W/8, D₁)


STEP 4: STAGE 3 (Attention)
───────────────────────────
Feature Map 2
    │
    ├─► Downsampling Conv2D -> (B, H/16, W/16, D₂)
    │
    ├─► Stack of Attention Blocks
    │   (Flatten -> TransformerLayer -> Reshape)
    │
    └─► Feature Map 3: (B, H/16, W/16, D₂)


STEP 5: CLASSIFICATION
──────────────────────
Feature Map 3
    │
    ├─► Global Average Pooling -> (B, D₂)
    │
    ├─► [Optional] Dropout
    │
    ├─► Dense Layer (Classifier)
    │
    └─► Logits (B, num_classes)
```

---

## 4. Architecture Deep Dive

### 4.1 `ConvolutionalStem`

-   **Purpose**: To perform initial, aggressive downsampling and low-level feature extraction.
-   **Implementation**: A sequence of three `MobileOneBlock` layers.
-   **Functionality**: Reduces spatial resolution by 4x (e.g., 224x224 -> 56x56) while increasing the channel dimension. `MobileOne` blocks are used for their efficiency at inference time due to structural reparameterization.

### 4.2 `RepMixerBlock`

-   **Purpose**: To efficiently mix features in the early stages where feature maps are large. It's a convolution-based alternative to self-attention.
-   **Architecture**: Comprises two main parts:
    1.  **Token Mixing**: Uses depthwise convolutions (3x3 and 1x1) to mix information spatially within each channel.
    2.  **Channel Mixing**: Uses 1x1 convolutions (an MLP) to mix information across different feature channels.
-   **Benefit**: Achieves effective feature mixing with a computational complexity that is linear with respect to the number of spatial locations, unlike the quadratic complexity of self-attention.

### 4.3 `AttentionBlock`

-   **Purpose**: To capture global, long-range dependencies in the final, low-resolution stage.
-   **Implementation**: This block wraps a standard `TransformerLayer`. It first flattens the spatial dimensions of the feature map into a sequence, processes it with the Transformer, and then reshapes it back to a spatial format.
-   **Functionality**: Performs full multi-head self-attention, allowing every location in the feature map to interact with every other location, enabling the model to learn global context.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First FastVLM Model (30 seconds)

Let's build a tiny FastVLM for a simple classification task.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.vision.fast_vlm.model import FastVLM

# 1. Create a tiny FastVLM model for CIFAR-10 (32x32 images, 10 classes)
# We use the "nano" variant for a very small and fast model
model = FastVLM.from_variant(
    "nano",
    num_classes=10,
    input_shape=(32, 32, 3)
)

# 2. Compile the model
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
print("✅ FastVLM model created and compiled successfully!")
model.summary()

# 3. Create dummy data for a forward pass
batch_size = 16
dummy_images = np.random.rand(batch_size, 32, 32, 3)
dummy_labels = np.random.randint(0, 10, batch_size)

# 4. Train for one step
loss, acc = model.train_on_batch(dummy_images, dummy_labels)
print(f"\n✅ Training step complete! Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# 5. Run inference
predictions = model.predict(dummy_images)
print(f"Predictions shape: {predictions.shape}") # (batch_size, num_classes)
```

---

## 6. Component Reference

### 6.1 `FastVLM` (Model Class)

**Purpose**: The main Keras `Model` subclass that assembles the complete FastVLM architecture.

**Location**: `dl_techniques.models.vision.fast_vlm.model.FastVLM`

```python
from dl_techniques.models.vision.fast_vlm.model import FastVLM

# Create from a standard variant
model = FastVLM.from_variant(
    "base",
    num_classes=1000,
    input_shape=(224, 224, 3)
)

# Create a custom model
custom_model = FastVLM(
    num_classes=100,
    embed_dims=[96, 192, 384],
    depths=[4, 6, 8],
    num_heads=[3, 6, 12]
)
```

### 6.2 Core Building Blocks

| Layer | Location | Purpose |
| :--- | :--- | :--- |
| **`ConvolutionalStem`** | `...layers.repmixer_block.ConvolutionalStem` | Initial feature extraction and 4x downsampling. |
| **`RepMixerBlock`** | `...layers.repmixer_block.RepMixerBlock` | Efficient, convolution-based token and channel mixing. |
| **`AttentionBlock`** | `...models.vision.fast_vlm.components.AttentionBlock` | Wraps a `TransformerLayer` for global attention on spatial feature maps. |
| **`MobileOneBlock`** | `...layers.mobile_one_block.MobileOneBlock` | The efficient, reparameterizable conv block used in the stem. |

---

## 7. Configuration & Model Variants

This implementation provides several pre-configured variants.

| Variant | Embed Dims | Depths | Heads | MLP Ratio | Use SE |
|:---:|:---|:---|:---|:---:|:---:|
| **`nano`** | | | | 2.0 | No |
| **`tiny`** | | | | 3.0 | No |
| **`small`**| | | | 4.0 | No |
| **`base`** | | | | 4.0 | No |
| **`large`**| | || 4.0 | Yes |
| **`huge`** |||| 4.0 | Yes |

---

## 8. Comprehensive Usage Examples

### Example 1: Using FastVLM as a Feature Extraction Backbone

You can use a headless FastVLM as a powerful backbone for downstream tasks like object detection or semantic segmentation.

```python
# 1. Create the feature extractor
backbone = FastVLM.from_variant("base", include_top=False, input_shape=(512, 512, 3))

# 2. Extract features
dummy_images = np.random.rand(2, 512, 512, 3)
features = backbone.predict(dummy_images)

# The output is the feature map from the final stage
# Spatial resolution is downsampled by 16x
print(f"Output shape: {features.shape}") # (2, 32, 32, 256)
```

### Example 2: Accessing Multi-Scale Features

The `extract_features` method provides access to feature maps from all stages, which is ideal for building feature pyramid networks (FPNs).

```python
# 1. Create the model
model = FastVLM.from_variant("base", include_top=False, input_shape=(224, 224, 3))

# 2. Get the multi-scale features
dummy_images = np.random.rand(1, 224, 224, 3)
multi_scale_features = model.extract_features(dummy_images)

print("Multi-scale feature map shapes:")
# features[0]: Stem output (4x downsampled)
print(f"  - Stem: {multi_scale_features[0].shape}")
# features[1]: Stage 1 output (4x downsampled)
print(f"  - Stage 1: {multi_scale_features[1].shape}")
# features[2]: Stage 2 output (8x downsampled)
print(f"  - Stage 2: {multi_scale_features[2].shape}")
# features[3]: Stage 3 output (16x downsampled)
print(f"  - Stage 3: {multi_scale_features[3].shape}")
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Structural Reparameterization for Inference

The `ConvolutionalStem` (and its underlying `MobileOneBlock`s) supports structural reparameterization. This fuses multiple parallel convolutional branches into a single, faster convolution for inference.

```python
# 1. Create and train your model as usual
model = FastVLM.from_variant("small", num_classes=10)
# ... model.compile() and model.fit() ...

# 2. Switch the stem to inference mode
model.stem.reparameterize()
print("✅ Stem has been reparameterized for fast inference!")

# 3. Now, inference calls will be faster
# predictions = model.predict(images)

# 4. If you need to continue training, switch back
# model.stem.reset_reparameterization()
```

---

## 10. Performance Optimization

### Mixed Precision Training

FastVLM is well-suited for mixed precision training, which can provide significant speedups on modern GPUs.

```python
# Enable mixed precision globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = FastVLM.from_variant("base", num_classes=1000)
model.compile(...)
```

---

## 11. Training and Best Practices

### Optimizer and Regularization

-   **Optimizer**: **AdamW** is highly recommended, as weight decay is a crucial regularizer for Transformer-like models.
-   **Learning Rate Schedule**: A cosine decay schedule with a few epochs of linear warmup generally works best.
-   **Stochastic Depth**: The `drop_path_rate` enables stochastic depth, a powerful regularization technique that randomly drops entire residual blocks during training. The rate is typically increased linearly from 0 for the first layer to the specified `drop_path_rate` for the last layer.

### Data Augmentation

-   Like other Transformer-based models, FastVLM benefits from strong data augmentations, as it has weaker inductive biases than traditional CNNs. Techniques like **RandAugment**, **Mixup**, and **CutMix** are effective.

---

## 12. Serialization & Deployment

The `FastVLM` model and all its custom layers are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
# Create and train model
model = FastVLM.from_variant("tiny", num_classes=10)
# model.compile(...) and model.fit(...)

# Save the entire model
model.save('my_fastvlm_model.keras')

# Load the model in a new session
loaded_model = keras.models.load_model('my_fastvlm_model.keras')
print("✅ Model loaded successfully!")
```

---

## 13. Testing & Validation

### Unit Tests

```python
import keras
import numpy as np
from dl_techniques.models.vision.fast_vlm.model import FastVLM

def test_creation_all_variants():
    """Test model creation for all variants."""
    for variant in FastVLM.MODEL_VARIANTS.keys():
        model = FastVLM.from_variant(variant, num_classes=10, input_shape=(64, 64, 3))
        assert model is not None
        print(f"✓ FastVLM-{variant} created successfully")

def test_forward_pass_shape():
    """Test the output shape of a forward pass."""
    model = FastVLM.from_variant("tiny", num_classes=10, input_shape=(128, 128, 3))
    dummy_input = np.random.rand(4, 128, 128, 3)
    output = model.predict(dummy_input)
    assert output.shape == (4, 10)
    print("✓ Forward pass has correct shape")

# Run tests
if __name__ == '__main__':
    test_creation_all_variants()
    test_forward_pass_shape()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Training is unstable or diverges.**

-   **Cause 1**: The learning rate may be too high. Hybrid models, especially those with attention, can be sensitive to the learning rate.
-   **Solution 1**: Use a smaller peak learning rate (e.g., `1e-4` or `5e-5`) and a proper warmup schedule.
-   **Cause 2**: Insufficient regularization for the model size and dataset.
-   **Solution 2**: Increase `dropout_rate` or `drop_path_rate`. Ensure you are using AdamW with appropriate `weight_decay`.

### Frequently Asked Questions

**Q: What is the main advantage of FastVLM over a standard Vision Transformer?**

A: **Speed**. By using efficient convolutional and RepMixer blocks in the early stages where feature maps are large, FastVLM avoids the quadratic complexity of self-attention until the final, low-resolution stage. This results in a significantly faster model with a better latency-accuracy trade-off.

**Q: How does `RepMixer` differ from `ConvMixer` or `MLP-Mixer`?**

A: They all belong to a family of "mixer" architectures that separate spatial and channel mixing.
-   `MLP-Mixer` uses MLPs (Dense layers) for both token and channel mixing, requiring patches to be flattened.
-   `ConvMixer` uses standard and depthwise convolutions throughout.
-   `RepMixer` is a highly optimized convolutional mixer that uses a specific sequence of depthwise and pointwise convolutions and is designed with structural reparameterization in mind for inference speed.

---

## 15. Technical Details

### Stochastic Depth

This model implements a linearly increasing stochastic depth rate. The probability of dropping a block is lowest at the start of the network and highest at the end. This is controlled by the `drop_path_rate` parameter. This regularization technique encourages the network to learn more robust features by forcing it to rely on different combinations of layers during training.

### Attention vs. RepMixer

-   **RepMixer (Early Stages)**:
    -   **Receptive Field**: Local. The depthwise convolutions have a fixed, small kernel size (e.g., 3x3).
    -   **Complexity**: `O(N)`, linear in the number of spatial locations.
    -   **Content-Agnostic**: The mixing operation is the same regardless of the input features.
-   **Self-Attention (Final Stage)**:
    -   **Receptive Field**: Global. Every location can attend to every other location.
    -   **Complexity**: `O(N²)`, quadratic in the number of spatial locations.
    -   **Content-Aware**: The mixing weights (attention scores) are dynamically computed based on the similarity between input features.

The hybrid design leverages the efficiency of the former and the power of the latter at the most appropriate architectural stages.

---

## 16. Citation

This implementation is inspired by several recent papers in efficient vision model design. If using these concepts in research, please consider citing the relevant works:

-   On FastViT and RepMixer:
    ```bibtex
    @article{vasu2023fastvit,
      title={FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization},
      author={Vasu, Pavan Kumar Anasosalu and Gabriel, James and Ravichandran, Anurag and Mehta, Saurabh and Hong, Zhaowen and Gholami, Ali and Adl, Morteza and Shazeer, Noam and Tuzel, Oncel and Faghri, Fartash},
      journal={arXiv preprint arXiv:2303.14189},
      year={2023}
    }
    ```
-   On MobileOne:
    ```bibtex
    @inproceedings{vasu2022mobileone,
      title={Mobileone: An improved one millisecond mobile backbone},
      author={Vasu, Pavan Kumar Anasosalu and Gabriel, James and Zhu, Jeff and Tuzel, Oncel and Faghri, Fartash},
      booktitle={European Conference on Computer Vision},
      pages={56--72},
      year={2022},
      organization={Springer}
    }
    ```
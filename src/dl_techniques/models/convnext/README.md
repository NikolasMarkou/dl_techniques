# ConvNeXt: A Modern ConvNet Architecture

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured Keras 3 implementation of the **ConvNeXt V1 and V2** architectures. ConvNeXt models are pure convolutional networks (ConvNets) that were modernized to compete with and often outperform Vision Transformers (ViTs) by progressively incorporating architectural decisions from ViTs into a standard ResNet.

The implementation includes both ConvNeXt V1 and the improved ConvNeXt V2, which introduces Global Response Normalization (GRN).

---

## Table of Contents

1. [Overview: What is ConvNeXt and Why It Matters](#1-overview-what-is-convnext-and-why-it-matters)
2. [The Problem ConvNeXt Solves](#2-the-problem-convnext-solves)
3. [How ConvNeXt Works: Core Concepts](#3-how-convnext-works-core-concepts)
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

## 1. Overview: What is ConvNeXt and Why It Matters

### What is ConvNeXt?

**ConvNeXt** is a family of pure convolutional neural networks that was designed to challenge the dominance of Vision Transformers. The authors of the original paper systematically investigated architectural differences between a standard ResNet and a Vision Transformer, and progressively adapted the ResNet to incorporate modern design principles, ultimately creating a ConvNet that matched or exceeded the performance of state-of-the-art Transformers on image classification.

### Key Innovations

1.  **Modernized Architecture (V1)**: ConvNeXt V1 redesigns a standard ResNet with several key changes inspired by ViTs:
    *   A "patchify" stem using a `4x4` non-overlapping convolution.
    *   Large `7x7` depthwise convolution kernels to increase the effective receptive field.
    *   An inverted bottleneck structure, similar to MobileNetV2.
    *   The replacement of BatchNorm with LayerNorm for more stable training.
    *   Fewer activation and normalization layers for a more streamlined design.
2.  **Global Response Normalization (V2)**: The primary evolution from V1 to V2 is the introduction of Global Response Normalization (GRN), a simple layer added to each block. GRN encourages channel-wise feature competition and enhances feature diversity, leading to improved model performance, especially when paired with self-supervised pre-training methods.
3.  **Efficiency and Simplicity**: As pure ConvNets, ConvNeXt models do not rely on the complex and memory-intensive self-attention mechanism, making them generally simpler to implement, understand, and optimize for various hardware platforms.

### Why ConvNeXt Matters

**Standard Vision Transformer (ViT) Problem**:
```
Problem: Classify a high-resolution image efficiently.
ViT Approach:
  1. Split the image into patches and process them with a deep stack of
     self-attention layers.
  2. Limitation: Self-attention has O(N²) complexity, where N is the number of
     patches. This makes ViTs slow and memory-hungry with high-resolution inputs.
  3. Result: Standard ViTs can be inefficient for real-time or on-device use.
```

**ConvNeXt's Solution**:
```
ConvNeXt Approach:
  1. Start with a classic, efficient ResNet architecture.
  2. Systematically apply modern design principles (larger kernels, layer norm,
     inverted bottlenecks) to boost performance without sacrificing the
     linear complexity of convolutions.
  3. Add a simple normalization layer (GRN in V2) to further improve
     representational quality.
  4. Benefit: Achieves Transformer-level accuracy while retaining the efficiency
     and simplicity of a pure ConvNet.
```

### Real-World Impact

ConvNeXt is an excellent choice for a wide range of computer vision tasks where a strong, general-purpose backbone is needed:

-   🖼️ **Image Classification**: Its primary design target, where it achieves state-of-the-art results.
-   **Object Detection & Segmentation**: A powerful feature extractor for dense prediction tasks.
-   **Transfer Learning**: Pre-trained ConvNeXt models are excellent starting points for fine-tuning on custom datasets.
-   ☁️ **Efficient Cloud Deployment**: Offers a compelling accuracy/compute trade-off for large-scale vision services.

---

## 2. The Problem ConvNeXt Solves

### The Efficiency-Accuracy Trade-off

In computer vision, there has long been a trade-off between model accuracy and computational efficiency.

```
┌─────────────────────────────────────────────────────────────┐
│  The Dilemma of Vision Architectures                        │
│                                                             │
│  Convolutional Neural Networks (CNNs):                      │
│    - Highly efficient due to local operations and shared weights.
│    - Traditionally struggled to model long-range, global    │
│      dependencies as effectively as Transformers.           │
│                                                             │
│  Vision Transformers (ViTs):                                │
│    - Excellent at modeling global context via self-attention.
│    - Suffer from quadratic complexity, making them slow and │
│      inefficient, especially with high-resolution inputs.   │
└─────────────────────────────────────────────────────────────┘
```

ConvNeXt challenges this dichotomy by demonstrating that a well-designed ConvNet can effectively capture features at multiple scales and achieve global context awareness without abandoning the efficiency of the convolutional operator.

### How ConvNeXt Changes the Game

ConvNeXt proves that many of the performance gains attributed to the Transformer architecture are not due to self-attention itself, but rather to other architectural and training strategy improvements. By applying these improvements to a ConvNet, it provides a highly competitive alternative.

```
┌─────────────────────────────────────────────────────────────┐
│  The ConvNeXt Modernization Strategy                        │
│                                                             │
│  1. Macro Design:                                           │
│     - Adopt a ViT-like stage compute ratio and a "patchify" │
│       stem for better multi-scale feature extraction.       │
│                                                             │
│  2. Micro Design:                                           │
│     - Use large 7x7 depthwise convolutions to increase the  │
│       receptive field, mimicking the global view of attention.
│     - Employ an inverted bottleneck block for higher capacity.
│     - Use Layer Normalization and GELU activation for       │
│       consistency with modern Transformer designs.          │
└─────────────────────────────────────────────────────────────┘
```

This principled approach results in a simple, scalable, and powerful architecture that serves as a robust baseline for many vision tasks.

---

## 3. How ConvNeXt Works: Core Concepts

### The Hierarchical Multi-Stage Architecture

ConvNeXt retains the classic hierarchical structure of a CNN, processing an image in four stages at progressively decreasing resolutions.

```
┌──────────────────────────────────────────────────────────────────┐
│                      ConvNeXt Architecture Stages                │
│                                                                  │
│  Input Image ───►┌──────────────────┐  (Downsamples 4x)          │
│   (H, W)         │   "Patchify" Stem  │                          │
│                  └────────┬─────────┘                            │
│                           │ (H/4, W/4)                           │
│                  ┌────────▼─────────┐  (Maintains resolution)    │
│                  │      Stage 1     │                            │
│                  │(ConvNeXt Blocks) │                            │
│                  └────────┬─────────┘                            │
│                           │ Downsample 2x                        │
│                           ▼ (H/8, W/8)                           │
│                  ┌────────▼─────────┐  (Maintains resolution)    │
│                  │      Stage 2     │                            │
│                  │(ConvNeXt Blocks) │                            │
│                  └────────┬─────────┘                            │
│                           │ Downsample 2x                        │
│                           ▼ (H/16, W/16)                         │
│                  ┌────────▼─────────┐  (Maintains resolution)    │
│                  │      Stage 3     │                            │
│                  │(ConvNeXt Blocks) │                            │
│                  └────────┬─────────┘                            │
│                           │ Downsample 2x                        │
│                           ▼ (H/32, W/32)                         │
│                  ┌────────▼─────────┐  (Maintains resolution)    │
│                  │      Stage 4     │                            │
│                  │(ConvNeXt Blocks) │                            │
│                  └────────┬─────────┘                            │
│                           │                                      │
│                  ┌────────▼──────────┐                           │
│                  │ Classification Head│                          │
│                  └────────────────────┘                          │
└──────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       ConvNeXt Complete Data Flow                       │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: STEM
────────────
Input Image (B, H, W, 3)
    │
    ├─► "Patchify" Conv2D (4x4 kernel, stride 4)
    │
    ├─► Layer Normalization
    │
    └─► Feature Map 0: (B, H/4, W/4, D₀)


STEP 2: STAGES 1-4
──────────────────
Feature Map (Input to Stage `i`)
    │
    ├─► [If i > 0] Downsampling Block (Stride-2 Conv + LayerNorm)
    │
    ├─► Stack of ConvNeXt Blocks (V1 or V2)
    │   (Each block maintains the resolution)
    │
    └─► Feature Map (Output of Stage `i`)


STEP 3: CLASSIFICATION HEAD
───────────────────────────
Final Feature Map
    │
    ├─► Global Average Pooling
    │
    ├─► Layer Normalization
    │
    ├─► [Optional] Dropout
    │
    ├─► Dense Layer (Classifier)
    │
    └─► Logits (B, num_classes)
```

---

## 4. Architecture Deep Dive

### 4.1 `Patchify Stem`

-   **Purpose**: To perform initial, aggressive downsampling, analogous to the patch embedding layer in a ViT.
-   **Implementation**: A single `Conv2D` layer with a kernel size of 4 and a stride of 4. This is followed by a `LayerNormalization` layer.
-   **Functionality**: It divides the input image into non-overlapping `4x4` patches and embeds them into a higher-dimensional feature space, reducing the spatial resolution by 4x (e.g., 224x224 -> 56x56).

### 4.2 `ConvNeXtV1Block`

-   **Purpose**: The main building block of the V1 architecture, designed for efficient and powerful feature extraction.
-   **Architecture**: Follows an inverted bottleneck design:
    1.  **Depthwise Convolution**: A large `7x7` depthwise `Conv2D` layer to mix spatial information. This is the key to its large receptive field.
    2.  **Layer Normalization**: Normalizes the features.
    3.  **Channel Mixing MLP**: Two `1x1` `Conv2D` layers (pointwise convolutions) with a GELU activation in between. The first `1x1` conv expands the channel dimension by a factor of 4, and the second projects it back down.
    4.  **Residual Connection**: The output of the block is added to the input (skip connection).

### 4.3 `ConvNeXtV2Block` and Global Response Normalization (GRN)

-   **Purpose**: The improved block in the V2 architecture, which enhances feature diversity.
-   **Architecture**: It is identical to the `ConvNeXtV1Block` but with a **Global Response Normalization (GRN)** layer added after the channel mixing MLP.
-   **How GRN Works**:
    1.  **Aggregate**: It computes the L2-norm of each feature map across the spatial dimensions (H, W), resulting in a single value per channel. `x_agg = ||X[:, :, c]||`
    2.  **Normalize**: It computes a normalization score for each channel by dividing its aggregated value by the sum of all aggregated values. `s_c = x_agg_c / Σ(x_agg_i)`
    3.  **Recalibrate**: It multiplies the original input feature map `X` by the computed scores `s_c`. This amplifies channels with unique features and suppresses redundant ones, encouraging feature competition.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First ConvNeXt Model (30 seconds)

Let's build a small ConvNeXtV2 for a simple classification task on CIFAR-10.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.convnext.convnext_v2 import create_convnext_v2

# 1. Create a tiny ConvNeXtV2 model for CIFAR-10 (32x32 images, 10 classes)
# The implementation automatically handles the smaller input size.
model = create_convnext_v2(
    variant="atto",  # A very small and fast V2 variant
    num_classes=10,
    input_shape=(32, 32, 3)
)

# 2. Compile the model
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
print("✅ ConvNeXtV2 model created and compiled successfully!")
model.summary()

# 3. Create dummy data for a forward pass
batch_size = 16
dummy_images = np.random.rand(batch_size, 32, 32, 3).astype("float32")
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

### 6.1 Model Classes and Creation Functions

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`ConvNeXtV1`** | `...convnext.convnext_v1.ConvNeXtV1` | The main Keras `Model` for the V1 architecture. |
| **`create_convnext_v1`** | `...convnext.convnext_v1.create_convnext_v1` | Recommended convenience function to create `ConvNeXtV1` models. |
| **`ConvNeXtV2`** | `...convnext.convnext_v2.ConvNeXtV2` | The main Keras `Model` for the V2 architecture. |
| **`create_convnext_v2`** | `...convnext.convnext_v2.create_convnext_v2` | Recommended convenience function to create `ConvNeXtV2` models. |

### 6.2 Core Building Blocks

| Layer | Location | Purpose |
| :--- | :--- | :--- |
| **`ConvNextV1Block`** | `...layers.convnext_v1_block.ConvNextV1Block` | The core block of the V1 architecture. |
| **`ConvNextV2Block`** | `...layers.convnext_v2_block.ConvNextV2Block` | The core block of the V2 architecture, including GRN. |

---

## 7. Configuration & Model Variants

This implementation provides several pre-configured variants for both V1 and V2.

### ConvNeXt V1 Variants

| Variant | Depths | Dimensions |
|:---:|:---|:---|
| **`tiny`** | `[3, 3, 9, 3]` | `[96, 192, 384, 768]` |
| **`small`**| `[3, 3, 27, 3]` | `[96, 192, 384, 768]` |
| **`base`** | `[3, 3, 27, 3]` | `[128, 256, 512, 1024]` |
| **`large`**| `[3, 3, 27, 3]`| `[192, 384, 768, 1536]` |
| **`xlarge`**|`[3, 3, 27, 3]` | `[256, 512, 1024, 2048]` |

### ConvNeXt V2 Variants

| Variant | Depths | Dimensions |
|:---:|:---|:---|
| **`atto`** | `[2, 2, 6, 2]` | `[40, 80, 160, 320]` |
| **`femto`**| `[2, 2, 6, 2]` | `[48, 96, 192, 384]` |
| **`pico`** | `[2, 2, 6, 2]` | `[64, 128, 256, 512]` |
| **`nano`** | `[2, 2, 8, 2]` | `[80, 160, 320, 640]` |
| **`tiny`** | `[3, 3, 9, 3]` | `[96, 192, 384, 768]` |
| **`base`** | `[3, 3, 27, 3]` | `[128, 256, 512, 1024]` |
| **`large`**| `[3, 3, 27, 3]`| `[192, 384, 768, 1536]` |
| **`huge`** | `[3, 3, 27, 3]` | `[352, 704, 1408, 2816]`|

---

## 8. Comprehensive Usage Examples

### Example 1: Using ConvNeXt as a Feature Extraction Backbone

You can use a headless ConvNeXt as a powerful backbone for downstream tasks like object detection or semantic segmentation.

```python
from dl_techniques.models.convnext.convnext_v1 import create_convnext_v1
import numpy as np

# 1. Create the feature extractor by setting include_top=False
backbone = create_convnext_v1(
    variant="base",
    include_top=False,
    input_shape=(512, 512, 3)
)

# 2. Extract features
dummy_images = np.random.rand(2, 512, 512, 3).astype("float32")
features = backbone.predict(dummy_images)

# The output is the feature map from the final stage
# Spatial resolution is downsampled by 32x
print(f"Output shape: {features.shape}") # (2, 16, 16, 1024)
backbone.summary()
```

### Example 2: Creating a Micro-Variant Model (ConvNeXt V2)

ConvNeXt V2 introduced several smaller variants that are highly efficient for mobile or resource-constrained applications.

```python
from dl_techniques.models.convnext.convnext_v2 import create_convnext_v2

# Create a ConvNeXtV2-Pico model for a 100-class problem
pico_model = create_convnext_v2(
    variant="pico",
    num_classes=100,
    input_shape=(96, 96, 3)
)

print(f"Pico model params: {pico_model.count_params():,}")
pico_model.summary()
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Fine-tuning from Pre-trained Weights

This implementation supports loading pre-trained weights, even when your new task has a different number of classes or a different input image size.

```python
from dl_techniques.models.convnext.convnext_v2 import create_convnext_v2

# Assume you have downloaded pre-trained ImageNet weights for ConvNeXtV2-Base
# and saved them to "convnext_v2_base_imagenet.keras"

# 1. Create a new model for a custom task (e.g., 20 classes, 128x128 images)
# The `pretrained` argument points to the local weights file.
# The code automatically handles mismatches in the classifier and input shape.
fine_tune_model = create_convnext_v2(
    variant="base",
    num_classes=20,                 # Different number of classes
    input_shape=(128, 128, 3),      # Different input shape
    pretrained="path/to/convnext_v2_base_imagenet.keras"
)

# 2. The model will load the backbone weights and skip the original classifier.
# You can now fine-tune this model on your custom dataset.
fine_tune_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-5), # Use a low learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model ready for fine-tuning!")
fine_tune_model.summary()
```

---

## 10. Performance Optimization

### Mixed Precision Training

ConvNeXt models are well-suited for mixed precision training, which uses 16-bit floating-point numbers for computations where possible. This can provide a significant speedup (up to 2-3x) on modern GPUs with Tensor Cores.

```python
# Enable mixed precision globally before creating the model
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = create_convnext_v2("base", num_classes=1000)
model.compile(...)

# When training, use a LossScaleOptimizer to prevent numeric underflow
# Keras's model.fit() handles this automatically.
```

---

## 11. Training and Best Practices

### Optimizer and Regularization

-   **Optimizer**: **AdamW** is highly recommended. The weight decay component of AdamW is a crucial regularizer for modern architectures like ConvNeXt.
-   **Learning Rate Schedule**: A **cosine decay** schedule, often with a few epochs of linear warmup at the beginning of training, generally yields the best results.
-   **Stochastic Depth (Drop Path)**: This is a powerful regularization technique that randomly drops entire residual blocks during training. It is enabled via the `drop_path_rate` argument. A good starting value is `0.1` or `0.2` for smaller models, increasing for larger ones.

### Data Augmentation

-   ConvNeXt models benefit significantly from strong data augmentations, as they have weaker inductive biases than older CNNs. Techniques like **RandAugment**, **Mixup**, and **CutMix** are highly effective and often necessary to achieve state-of-the-art results.

---

## 12. Serialization & Deployment

The `ConvNeXtV1`, `ConvNeXtV2`, and all their custom layers are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
# Create and train model
model = create_convnext_v1("tiny", num_classes=10)
# model.compile(...) and model.fit(...)

# Save the entire model to a single file
model.save('my_convnext_model.keras')

# Load the model in a new session, including its architecture, weights,
# and optimizer state.
loaded_model = keras.models.load_model('my_convnext_model.keras')
print("✅ Model loaded successfully!")
```

---

## 13. Testing & Validation

### Unit Tests

You can validate the implementation with simple tests to ensure all variants can be created and produce the correct output shapes.

```python
import keras
import numpy as np
from dl_techniques.models.convnext.convnext_v1 import ConvNeXtV1
from dl_techniques.models.convnext.convnext_v2 import ConvNeXtV2

def test_creation_all_variants():
    """Test model creation for all V1 and V2 variants."""
    for variant in ConvNeXtV1.MODEL_VARIANTS.keys():
        model = ConvNeXtV1.from_variant(variant, num_classes=10, input_shape=(64, 64, 3))
        assert model is not None
        print(f"✓ ConvNeXtV1-{variant} created successfully")

    for variant in ConvNeXtV2.MODEL_VARIANTS.keys():
        model = ConvNeXtV2.from_variant(variant, num_classes=10, input_shape=(64, 64, 3))
        assert model is not None
        print(f"✓ ConvNeXtV2-{variant} created successfully")

def test_forward_pass_shape():
    """Test the output shape of a forward pass."""
    model = ConvNeXtV2.from_variant("tiny", num_classes=10, input_shape=(128, 128, 3))
    dummy_input = np.random.rand(4, 128, 128, 3).astype("float32")
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

**Issue 1: Training is unstable or the loss becomes `NaN`.**

-   **Cause 1**: The learning rate may be too high, especially at the start of training.
-   **Solution 1**: Use a smaller peak learning rate (e.g., `5e-4` or `1e-4`) and implement a linear warmup schedule for the first 5-10 epochs.
-   **Cause 2**: Insufficient regularization for your dataset size and model capacity.
-   **Solution 2**: Increase the `weight_decay` in the AdamW optimizer (e.g., to `0.05`). Increase the `drop_path_rate` (e.g., to `0.2` or higher for larger models).

### Frequently Asked Questions

**Q: What is the main difference between ConvNeXt V1 and V2?**

A: The single architectural difference is that **ConvNeXt V2 adds a Global Response Normalization (GRN) layer** to each block. This simple layer enhances inter-channel feature competition, leading to improved performance. V2 was also co-designed with a masked autoencoder pre-training method, which further boosts its capabilities.

**Q: Why should I use ConvNeXt instead of a Vision Transformer?**

A: **Simplicity and Efficiency.** ConvNeXt demonstrates that you can achieve Transformer-level performance using a standard, pure ConvNet. This can be advantageous as convolutions are highly optimized on many hardware platforms (like GPUs and mobile CPUs) and the architecture is often simpler to understand, implement, and deploy than self-attention-based models.

**Q: Can I get multi-scale feature maps for detection/segmentation?**

A: Yes. While this implementation doesn't have a dedicated `extract_features` method, you can easily create a new Keras model that outputs the intermediate feature maps from the original model's layers. You can access the layers by name (e.g., after each `downsample` layer) to build a feature pyramid.

---

## 15. Technical Details

### Stochastic Depth (Drop Path)

This model implements a linearly increasing stochastic depth rate. The probability of a residual block being skipped is lowest at the start of the network and highest at the end. This is controlled by the `drop_path_rate` parameter. This regularization technique is very effective for ConvNeXt, as it forces the network to learn redundant representations and makes it more robust. The drop path is applied with a `(B, 1, 1, 1)` noise shape, ensuring that entire blocks are dropped for each sample in the batch, rather than individual pixels.

### V1 Block vs. V2 Block

The core evolution is the addition of GRN.

-   **ConvNeXtV1Block Flow**:
    `Input -> DepthwiseConv -> LayerNorm -> Expand MLP -> GELU -> Project MLP -> Add -> Output`
-   **ConvNeXtV2Block Flow**:
    `Input -> DepthwiseConv -> LayerNorm -> Expand MLP -> GELU -> Project MLP -> **GRN** -> Add -> Output`

The GRN layer is placed just before the final residual connection, allowing it to recalibrate the features generated by the block before they are passed on to the next layer.

---

## 16. Citation

This implementation is based on the official ConvNeXt papers. If you use this model in your research, please consider citing the original works:

-   **ConvNeXt V1**:
    ```bibtex
    @article{liu2022convnet,
      title={A ConvNet for the 2020s},
      author={Liu, Zhuang and Mao, Hanzi and Wu, Chao-Yuan and Feichtenhofer, Christoph and Darrell, Trevor and Xie, Saining},
      journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022}
    }
    ```
-   **ConvNeXt V2**:
    ```bibtex
    @article{woo2023convnextv2,
      title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
      author={Woo, Sanghyun and Debnath, Shoubhik and Hu, Ronghang and Chen, Xinlei and Liu, Zhuang and Dollar, Piotr and Xie, Saining},
      journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2023}
    }
    ```
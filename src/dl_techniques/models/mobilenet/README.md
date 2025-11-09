# MobileNet Family (V1-V4): Efficient Vision Models

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured Keras 3 implementation of the entire **MobileNet** family of models, from **V1 to V4**. These models represent a lineage of highly efficient convolutional neural networks designed specifically for on-device and mobile vision applications where computational resources are limited.

The implementations for V2, V3, and V4 leverage a flexible `UniversalInvertedBottleneck` layer, showcasing how a unified building block can be configured to create a wide range of modern architectures.

---

## Table of Contents

1. [Overview: What is MobileNet and Why It Matters](#1-overview-what-is-mobilenet-and-why-it-matters)
2. [The Problem MobileNet Solves](#2-the-problem-mobilenet-solves)
3. [How MobileNet Works: Core Concepts & Evolution](#3-how-mobilenet-works-core-concepts--evolution)
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
15. [Technical Details: The Evolution of the Block](#15-technical-details-the-evolution-of-the-block)
16. [Citation](#16-citation)

---

## 1. Overview: What is MobileNet and Why It Matters

### What is the MobileNet Family?

**MobileNet** is a class of convolutional neural networks designed by Google to run efficiently on mobile and edge devices. The core idea is to achieve the best possible accuracy while respecting the strict computational, power, and memory constraints of on-device applications. Each version in the series—from V1 to V4—introduces new architectural innovations to push the boundaries of this efficiency-accuracy trade-off.

### The Evolution of Efficiency

| Model | Key Innovation | Description |
| :--- | :--- | :--- |
| **MobileNetV1** | Depthwise Separable Convolutions | Drastically reduced computation by factorizing standard convolutions into a depthwise and a pointwise convolution. |
| **MobileNetV2** | Inverted Residuals & Linear Bottlenecks | Introduced blocks that first expand and then project feature maps, with residual connections between the narrow "bottleneck" layers. This improved feature reuse and gradient flow. |
| **MobileNetV3** | Hardware-Aware NAS, Squeeze-and-Excite, Hard-Swish | Utilized Neural Architecture Search (NAS) to find an optimal architecture. Added lightweight attention (Squeeze-and-Excite) and a more efficient non-linearity (hard-swish). |
| **MobileNetV4** | Universal Inverted Bottleneck (UIB) & Mobile MQA | Introduced a flexible "Universal" block that can represent different block styles (including ConvNeXt-like structures). Added an optional mobile-friendly Multi-Query Attention (MQA) module, creating hybrid vision transformer models. |

### Why MobileNet Matters

**The Server-Side AI Problem**:
```
Problem: Classify an image from a user's phone.
Standard (Cloud) Approach:
  1. User's phone sends the image to a powerful server.
  2. A massive model (e.g., ResNet, ViT) on the server processes the image.
  3. The server sends the result back.
  4. Limitation: Requires internet, introduces latency, raises privacy concerns,
     and has high server-side costs.
```

**The MobileNet Solution**:
```
MobileNet (On-Device) Approach:
  1. A small, efficient MobileNet model runs directly on the user's phone.
  2. The image is processed locally in milliseconds.
  3. The result is available instantly.
  4. Benefit: Works offline, has near-zero latency, preserves user privacy,
     and has no server costs. Essential for real-time applications like
     live camera filters, augmented reality, and on-device assistants.
```

---

## 2. The Problem MobileNet Solves

### The "Big Model" Dilemma

While large models achieve state-of-the-art accuracy, their computational demands make them impractical for billions of devices at the edge. The core problem is the three-way constraint of **latency, power, and size**.

```
┌─────────────────────────────────────────────────────────────┐
│  The Constraints of On-Device AI                            │
│                                                             │
│  1. Latency: For real-time applications (e.g., video        │
│     analysis), inference must be faster than the frame rate │
│     (e.g., < 33ms).                                         │
│                                                             │
│  2. Power Consumption: High computational load drains the   │
│     battery, making the application unusable.               │
│                                                             │
|  3. Model Size: The model must be small enough to be stored │
│     on the device and fit into limited RAM during execution.│
└─────────────────────────────────────────────────────────────┘
```

The MobileNet family directly tackles this challenge by redesigning the fundamental building blocks of convolutional networks to be as computationally frugal as possible, allowing powerful computer vision to be deployed ubiquitously.

---

## 3. How MobileNet Works: Core Concepts & Evolution

The innovation of MobileNet can be traced through the evolution of its core building block.

#### **1. Depthwise Separable Convolutions (MobileNetV1)**
The foundation of the entire family. A standard convolution is factorized into two cheaper operations:
-   **Depthwise Convolution**: A single filter is applied to each input channel independently. It mixes spatial information but not channel information.
-   **Pointwise Convolution (1x1 Conv)**: A 1x1 convolution is used to combine the outputs of the depthwise layer, mixing channel information.
This factorization reduces computation by 8-9x compared to a standard convolution with minimal loss in accuracy.

#### **2. Inverted Residuals and Linear Bottlenecks (MobileNetV2)**
MobileNetV2 redesigned the block to improve information flow:
-   **Inverted Bottleneck**: Instead of `wide -> narrow -> wide` (like in ResNet), the block is `narrow -> wide -> narrow`. The input and output are low-dimensional "bottlenecks," while the internal structure is expanded to a higher dimension for feature processing.
-   **Linear Bottlenecks**: The final pointwise convolution in the block has *no activation function*. This prevents non-linearities from destroying information in the low-dimensional bottleneck.
-   **Residual Connections**: Skip connections are added between the bottlenecks, allowing for the training of much deeper and more powerful networks.

#### **3. NAS, Squeeze-and-Excite, and h-swish (MobileNetV3)**
MobileNetV3 was discovered, not just designed.
-   **Neural Architecture Search (NAS)**: An algorithm searched for the optimal combination of blocks, kernel sizes, and channel counts, specifically targeting low latency on mobile CPUs.
-   **Squeeze-and-Excite (SE)**: A lightweight attention mechanism that adaptively re-weights channel features, allowing the network to focus on what's important.
-   **Hard-Swish (h-swish)**: A computationally cheaper approximation of the Swish activation function, providing the benefits of a modern non-linearity without the performance cost.

#### **4. Universal Inverted Bottleneck and Mobile MQA (MobileNetV4)**
MobileNetV4 introduces a flexible, unified building block and optional attention.
-   **Universal Inverted Bottleneck (UIB)**: A single, highly configurable block that can act like a classic inverted residual, a ConvNeXt-style block, or a new "Extra Depthwise" variant. This gives NAS a richer search space to find even better architectures.
-   **Mobile Multi-Query Attention (MQA)**: An optional, efficient attention layer that can be added to the later stages of the network, creating a "hybrid" model that combines the strengths of convolutions and transformers for higher accuracy.

---

## 4. Architecture Deep Dive

### 4.1 `MobileNetV1`
- **Stem**: A single standard `3x3` convolution.
- **Body**: A stack of 13 `DepthwiseSeparableBlock` layers. Some blocks have a stride of 2 for downsampling.
- **Head**: Global average pooling followed by a `1x1` convolution that acts as a fully connected layer.

### 4.2 `MobileNetV2`
- **Stem**: A single standard `3x3` convolution.
- **Body**: A sequence of 17 inverted residual blocks, organized into 7 stages. Each block consists of a 1x1 expansion, a 3x3 depthwise convolution, and a 1x1 linear projection. Residual connections are used for blocks with a stride of 1.
- **Head**: A final `1x1` convolution to expand features, followed by global average pooling and a dense classifier.

### 4.3 `MobileNetV3`
- **Stem**: A `3x3` convolution optimized for efficiency.
- **Body**: A stack of inverted residual blocks found by NAS. Key differences from V2 include the use of `5x5` kernels in some layers, the integration of Squeeze-and-Excite modules, and the use of ReLU and hard-swish activations at different depths.
- **Head**: An "efficient last stage" that reduces computation before the final pooling and classification layers.

### 4.4 `MobileNetV4`
- **Stem**: A `3x3` convolution similar to previous versions.
- **Body**: A stack of Universal Inverted Bottleneck (UIB) blocks. The type of UIB block (`"IB"`, `"ExtraDW"`, etc.) is specific to each stage of the network. For "hybrid" variants, the later stages are followed by a `MobileMQA` attention layer.
- **Head**: Global average pooling and a two-layer classifier head (Dense -> ReLU -> Dropout -> Dense).

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First MobileNetV4 Model (30 seconds)

Let's build a small MobileNetV4 for a simple classification task on CIFAR-10.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.mobilenet.mobilenet_v4 import create_mobilenetv4

# 1. Create a small MobileNetV4 model for CIFAR-10 (32x32 images, 10 classes)
model = create_mobilenetv4(
    variant="conv_small",  # A very small and fast V4 variant
    num_classes=10,
    input_shape=(32, 32, 3)
)

# 2. Compile the model
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Softmax is included
    metrics=["accuracy"],
)
print("✅ MobileNetV4 model created and compiled successfully!")
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
| **`MobileNetV1`** | `...mobilenet.mobilenet_v1.MobileNetV1` | The Keras `Model` for the V1 architecture. |
| **`create_mobilenetv1`** | `...mobilenet.mobilenet_v1.create_mobilenetv1` | Convenience function to create `MobileNetV1` models. |
| **`MobileNetV2`** | `...mobilenet.mobilenet_v2.MobileNetV2` | The Keras `Model` for the V2 architecture. |
| **`create_mobilenetv2`** | `...mobilenet.mobilenet_v2.create_mobilenetv2` | Convenience function to create `MobileNetV2` models. |
| **`MobileNetV3`** | `...mobilenet.mobilenet_v3.MobileNetV3` | The Keras `Model` for the V3 architecture. |
| **`create_mobilenetv3`** | `...mobilenet.mobilenet_v3.create_mobilenetv3` | Convenience function to create `MobileNetV3` models. |
| **`MobileNetV4`** | `...mobilenet.mobilenet_v4.MobileNetV4` | The Keras `Model` for the V4 architecture. |
| **`create_mobilenetv4`** | `...mobilenet.mobilenet_v4.create_mobilenetv4` | Convenience function to create `MobileNetV4` models. |

---

## 7. Configuration & Model Variants

Each MobileNet version comes with several pre-configured variants.

### MobileNetV1 Variants (by `width_multiplier`)
`"1.0"`, `"0.75"`, `"0.5"`, `"0.25"`

### MobileNetV2 Variants (by `width_multiplier`)
`"1.0"`, `"0.75"`, `"0.5"`, `"0.35"` (plus a `"1.4"` variant exists)

### MobileNetV3 Variants
`"large"`, `"small"` (both can be scaled with `width_multiplier`)

### MobileNetV4 Variants
-   **Conv-Only**: `"conv_small"`, `"conv_medium"`, `"conv_large"`
-   **Hybrid (with Attention)**: `"hybrid_medium"`, `"hybrid_large"`

---

## 8. Comprehensive Usage Examples

### Example 1: Creating a Model for a Custom Dataset (CIFAR-100)

Adapt a MobileNetV3 model for a dataset with 100 classes and 32x32 images.

```python
from dl_techniques.models.mobilenet.mobilenet_v3 import create_mobilenetv3

cifar100_model = create_mobilenetv3(
    variant="small",
    num_classes=100,
    input_shape=(32, 32, 3)
)
cifar100_model.summary()
```

### Example 2: Using the Width Multiplier to Adjust Model Size

The `width_multiplier` (α) scales the number of channels in every layer, providing a simple way to trade accuracy for size and latency.

```python
from dl_techniques.models.mobilenet.mobilenet_v2 import create_mobilenetv2

# Default MobileNetV2 1.0 has ~3.5M params
default_v2 = create_mobilenetv2(variant="1.0")

# MobileNetV2 0.5 has ~1.95M params
small_v2 = create_mobilenetv2(variant="0.5")

# You can also use the multiplier directly on a variant
custom_small_v2 = create_mobilenetv2(variant="1.0", width_multiplier=0.5)

print(f"Default V2 params: {default_v2.count_params():,}")
print(f"Small V2 (variant) params: {small_v2.count_params():,}")
print(f"Small V2 (multiplier) params: {custom_small_v2.count_params():,}")
```

### Example 3: Using MobileNet as a Feature Extractor

For downstream tasks like object detection or segmentation, create a "headless" model by setting `include_top=False`.

```python
from dl_techniques.models.mobilenet.mobilenet_v4 import create_mobilenetv4

# Create a headless V4 backbone for feature extraction
backbone = create_mobilenetv4(
    variant="hybrid_medium",
    include_top=False,
    input_shape=(256, 256, 3)
)

# The output is the feature map from the final stage
# For a 256x256 input, the output shape will be (None, 8, 8, 320)
backbone.summary()```

---

## 9. Advanced Usage Patterns

### Pattern 1: Fine-tuning from Pre-trained Weights

While this implementation does not ship with pre-trained weights, it is designed to load them easily. If you have a `.keras` file with ImageNet weights, you can load them and fine-tune on a new task.

```python
# Assume you have downloaded pre-trained weights for MobileNetV4-ConvMedium
# and saved them to "mobilenet_v4_conv_medium_imagenet.keras"

# 1. Create a new model for a custom task (e.g., 20 classes)
# The `from_config` logic in the custom model classes handles this.
# You would first load the full model and then build a new one.
# (A more direct weight loading API would be needed for production use)

# Hypothetical fine-tuning setup
# pretrained_model = keras.models.load_model("path/to/weights.keras")
# backbone = keras.Model(pretrained_model.inputs, pretrained_model.get_layer("...").output)
# backbone.trainable = False
#
# inputs = keras.Input(shape=(128, 128, 3))
# x = backbone(inputs, training=False)
# outputs = keras.layers.Dense(20, activation="softmax")(x)
# fine_tune_model = keras.Model(inputs, outputs)
```

---

## 10. Performance Optimization

### Mixed Precision Training

All MobileNet models are well-suited for mixed precision training, which uses 16-bit floating-point numbers for computations. This can provide a significant speedup (up to 2-3x) on modern GPUs with Tensor Cores.

```python
# Enable mixed precision globally before creating the model
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = create_mobilenetv4("conv_medium", num_classes=1000)
model.compile(...)

# When training, use a LossScaleOptimizer to prevent numeric underflow
# Keras's model.fit() handles this automatically.
```

---

## 11. Training and Best Practices

### Optimizer and Regularization

-   **Optimizer**: **AdamW** or **RMSprop** are common choices. The original papers often used RMSprop with a specific decay and momentum. For general use, AdamW is a robust starting point.
-   **Weight Decay**: MobileNets are sensitive to weight decay. The values in the constructors (e.g., `4e-5` for V2) are taken from the original papers and are good defaults.
-   **Learning Rate Schedule**: A **cosine decay** or **exponential decay** schedule is highly recommended over a fixed learning rate for achieving the best results.

### Data Augmentation

-   Like all modern ConvNets, MobileNets benefit significantly from strong data augmentation. Techniques like **RandAugment**, **Mixup**, and **CutMix** are effective for training from scratch.

---

## 12. Serialization & Deployment

All model classes (`MobileNetV1` through `MobileNetV4`) and their custom layers are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
# Create and train a model
model = create_mobilenetv3("small", num_classes=10)
# model.compile(...) and model.fit(...)

# Save the entire model to a single file
model.save('my_mobilenetv3_model.keras')

# Load the model in a new session, including its architecture, weights,
# and optimizer state.
loaded_model = keras.models.load_model('my_mobilenetv3_model.keras')
print("✅ Model loaded successfully!")
```

---

## 13. Testing & Validation

### Unit Tests

You can validate the implementations with simple tests to ensure all variants can be created and produce the correct output shapes.

```python
import keras
import numpy as np
from dl_techniques.models.mobilenet.mobilenet_v1 import MobileNetV1
from dl_techniques.models.mobilenet.mobilenet_v2 import MobileNetV2
from dl_techniques.models.mobilenet.mobilenet_v3 import MobileNetV3
from dl_techniques.models.mobilenet.mobilenet_v4 import MobileNetV4

def test_creation_all_variants():
    """Test model creation for all variants of all versions."""
    for variant in MobileNetV1.MODEL_VARIANTS.keys():
        MobileNetV1.from_variant(variant, num_classes=10)
    print("✓ All MobileNetV1 variants created successfully")

    for variant in MobileNetV2.MODEL_VARIANTS.keys():
        MobileNetV2.from_variant(variant, num_classes=10)
    print("✓ All MobileNetV2 variants created successfully")

    for variant in ["large", "small"]:
        MobileNetV3.from_variant(variant, num_classes=10)
    print("✓ All MobileNetV3 variants created successfully")
    
    for variant in MobileNetV4.MODEL_VARIANTS.keys():
        MobileNetV4.from_variant(variant, num_classes=10, input_shape=(32, 32, 3))
    print("✓ All MobileNetV4 variants created successfully")

def test_forward_pass_shape():
    """Test the output shape of a forward pass."""
    model = MobileNetV4.from_variant("conv_small", num_classes=10, input_shape=(96, 96, 3))
    dummy_input = np.random.rand(4, 96, 96, 3).astype("float32")
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

**Issue 1: Training accuracy is very low or not improving.**

-   **Cause**: MobileNets, being smaller models, can be sensitive to hyperparameters. The learning rate might be too high, or the weight decay might be inappropriate for your dataset.
-   **Solution**: Start with a lower learning rate (e.g., `1e-4`) and use a learning rate schedule. Ensure your data is properly normalized. The default weight decay values are tuned for ImageNet and may need adjustment for smaller datasets.

### Frequently Asked Questions

**Q: Which MobileNet version should I use?**

A:
-   **For the best accuracy/latency trade-off on modern hardware**: Start with **MobileNetV4**. The `conv_small` or `conv_medium` variants are excellent general-purpose backbones.
-   **For a robust, well-understood baseline**: **MobileNetV2** is a fantastic choice and is widely supported in production environments.
-   **If you need a CPU-optimized model with good performance**: **MobileNetV3** is still highly competitive.
-   **For legacy systems or maximum simplicity**: **MobileNetV1** is the original, but V2 generally offers better performance for a similar cost.

**Q: What is the `width_multiplier`?**

A: It's a hyperparameter (α) that uniformly scales the number of channels (filters) in every layer of the network. A `width_multiplier` of `0.5` means every layer will have half the number of filters as the base model, reducing the parameter count and computational cost by roughly 4x. It's the primary way to customize the size of a MobileNet model.

**Q: What's the main difference between MobileNetV2 and V3?**

A: MobileNetV3 is essentially a MobileNetV2-like architecture that has been fine-tuned by a search algorithm (NAS). The key upgrades are the addition of Squeeze-and-Excite (attention) modules, the use of a more efficient `h-swish` activation, and a redesigned, faster final stage.

---

## 15. Technical Details: The Evolution of the Block

The innovation in the MobileNet family can be seen by comparing the structure of their core building blocks.

-   **MobileNetV1 Block (Depthwise Separable Conv)**:
    `Input -> 3x3 DWConv -> BN -> ReLU -> 1x1 PWConv -> BN -> ReLU -> Output`

-   **MobileNetV2 Block (Inverted Residual)**:
    `Input -> 1x1 Conv (Expand) -> BN -> ReLU6 -> 3x3 DWConv -> BN -> ReLU6 -> 1x1 Conv (Project) -> BN -> Add -> Output`
    (The final projection is *linear*, and the `Add` is the residual connection)

-   **MobileNetV3 Block (V2 + SE + h-swish)**:
    `... -> 1x1 Conv (Expand) -> ... -> 3x3 DWConv -> ... -> Squeeze-Excite -> 1x1 Conv (Project) -> ...`
    (Also uses `h-swish` and different kernel sizes found by NAS)

-   **MobileNetV4 Block (Universal Inverted Bottleneck)**:
    The UIB is highly flexible. For example, the `"ExtraDW"` variant looks like this:
    `Input -> 1x1 Conv (Expand) -> BN -> Act -> 3x3 DWConv -> BN -> Act -> 5x5 DWConv -> BN -> Act -> 1x1 Conv (Project) -> ...`
    This adds more spatial mixing capability compared to the V2/V3 blocks.

---

## 16. Citation

If you use these models in your research, please cite the original works:

-   **MobileNetV1**:
    ```bibtex
    @article{howard2017mobilenets,
      title={Mobilenets: Efficient convolutional neural networks for mobile vision applications},
      author={Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
      journal={arXiv preprint arXiv:1704.04861},
      year={2017}
    }
    ```
-   **MobileNetV2**:
    ```bibtex
    @inproceedings{sandler2018mobilenetv2,
      title={MobileNetV2: Inverted residuals and linear bottlenecks},
      author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
      booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
      pages={4510--4520},
      year={2018}
    }
    ```
-   **MobileNetV3**:
    ```bibtex
    @inproceedings{howard2019searching,
      title={Searching for mobilenetv3},
      author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and Le, Quoc V and Adam, Hartwig},
      booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
      pages={1314--1324},
      year={2019}
    }
    ```
-   **MobileNetV4**:
    ```bibtex
    @article{mobilenetv4,
      title={{MobileNetV4 - Universal Inverted Bottleneck and Mobile MQA}},
      author={Dan-Feltrim, Daniel and T-Gotmare, Arjun and G-Hegde, Shruthi and Gabriel, Gabriel L. and Clienti, Luca and Hashem, Mostafa and L-Ferez, Jose M.},
      journal={arXiv preprint arXiv:2404.10518},
      year={2024}
    }
    ```
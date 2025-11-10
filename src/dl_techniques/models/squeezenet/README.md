# SqueezeNet: Highly Efficient Convolutional Architectures

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)

A production-ready Keras 3 implementation of the **SqueezeNet V1** and **SqueezeNodule-Net (V2)** architectures. SqueezeNet models are pioneering, efficient convolutional networks (ConvNets) designed to achieve high accuracy with a dramatically reduced parameter count and model size.

This implementation includes the original SqueezeNet V1 with its innovative **Fire Module**, and SqueezeNodule-Net, an evolution optimized for medical imaging tasks that uses a **Simplified Fire Module** and supports 3D convolutions.

---

## Table of Contents

1. [Overview: What is SqueezeNet and Why It Matters](#1-overview-what-is-squeezenet-and-why-it-matters)
2. [The Problem SqueezeNet Solves](#2-the-problem-squeezenet-solves)
3. [How SqueezeNet Works: Core Concepts](#3-how-squeezenet-works-core-concepts)
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

## 1. Overview: What is SqueezeNet and Why It Matters

### What is SqueezeNet?

**SqueezeNet** is a family of deep convolutional neural networks that set a new standard for computational efficiency. The original paper demonstrated AlexNet-level accuracy on ImageNet with **50 times fewer parameters**, resulting in a model size of less than 0.5MB. Its design is centered around a clever micro-architectural block called the **Fire module**.

### Key Innovations

1.  **The Fire Module (V1)**: SqueezeNet's core building block, which "squeezes" and "expands" feature maps to reduce computation. It consists of:
    *   A **squeeze** layer of `1x1` convolutions to act as a bottleneck, reducing the number of input channels.
    *   An **expand** layer with a mix of `1x1` and `3x3` convolutions that processes the squeezed features in parallel.
2.  **Strategic Parameter Reduction**: The architecture is guided by three principles:
    *   Replace expensive `3x3` filters with `1x1` filters where possible.
    *   Decrease the number of input channels to `3x3` filters using the squeeze layer.
    *   Downsample late in the network to preserve large feature maps.
3.  **Simplified Fire Module (SqueezeNodule-Net V2)**: An evolution of the Fire module that forces spatial feature learning by removing the `1x1` path in the expand layer. It also uses a wider information bottleneck, which has proven effective for detailed tasks like medical image analysis.

### Why SqueezeNet Matters

**The Problem of Large Models**:
```
Problem: Deploy an accurate image classifier on a device with limited
         memory and compute power (e.g., a smartphone or embedded system).

Traditional CNN Approach (e.g., AlexNet, VGG):
  1. Use deep stacks of large convolutional layers.
  2. Limitation: Results in massive models (e.g., AlexNet > 200MB) that
     are slow, power-hungry, and difficult to deploy over the air or on-device.
  3. Result: High accuracy is chained to high-end hardware.
```

**SqueezeNet's Solution**:
```
SqueezeNet Approach:
  1. Redesign the core building block (the Fire module) to be
     aggressively parameter-efficient.
  2. The squeeze/expand design drastically cuts down the parameter count
     in each layer without a proportional drop in accuracy.
  3. Benefit: Achieves competitive accuracy in a tiny footprint (<1MB),
     unlocking deployment in resource-constrained environments.
```

### Real-World Impact

SqueezeNet is an ideal choice for applications where efficiency is paramount:

-   📱 **Mobile & Edge Computing**: Its small size and low computational cost make it perfect for on-device inference.
-   🌐 **Distributed Training**: Smaller models are faster to transfer across networks.
-   🚗 **Autonomous Systems**: Enables real-time vision tasks on embedded hardware in drones and vehicles.
-   🩺 **Medical Imaging (SqueezeNodule-Net)**: The V2 variant is adapted for specialized tasks like nodule detection in 2D or 3D CT scans.

---

## 2. The Problem SqueezeNet Solves

### The Burden of Model Size

Before SqueezeNet, there was a direct correlation between a model's accuracy and its size. This created a significant barrier to deploying deep learning in the real world.

```
┌─────────────────────────────────────────────────────────────┐
│  The Dilemma of Model Deployment                            │
│                                                             │
│  Large, Accurate Models (e.g., VGG-16):                     │
│    - High accuracy on benchmarks.                           │
│    - Prohibitively large (~500MB), requiring significant    │
│      storage, memory, and bandwidth.                        │
│                                                             │
│  Smaller, Faster Models:                                    │
│    - Easy to deploy.                                        │
│    - Historically suffered from a significant accuracy gap. │
└─────────────────────────────────────────────────────────────┘
```

SqueezeNet fundamentally challenged this assumption by showing that architectural innovation, rather than sheer scale, could be the key to both accuracy and efficiency.

### How SqueezeNet Changes the Game

SqueezeNet introduced a new way of thinking about network design, focusing on "architectural MIPS" (millions of instructions per second) and parameter count as first-class citizens alongside accuracy.

```
┌──────────────────────────────────────────────────────────────┐
│  The SqueezeNet Design Philosophy                            │
│                                                              │
│  1. Squeeze Channels:                                        │
│     - The number of parameters in a 3x3 conv layer is        │
│       (input_channels * 3 * 3 * output_channels).            │
│     - By using a 1x1 "squeeze" layer first, we dramatically  │
│       reduce `input_channels` for the expensive 3x3 convs.   │
│                                                              │
│  2. Use 1x1 Filters for Expansion:                           │
│     - A 1x1 convolution is 9x cheaper than a 3x3 convolution.│
│     - The Fire module's expand layer uses many 1x1 filters   │
│       to build channel depth cheaply.                        │
└──────────────────────────────────────────────────────────────┘
```

This principled approach results in an incredibly compact yet powerful architecture.

---

## 3. How SqueezeNet Works: Core Concepts

### The Hierarchical Multi-Stage Architecture

SqueezeNet uses a traditional CNN structure: an initial convolutional layer (the "stem"), followed by a series of Fire modules, with max-pooling layers periodically reducing the spatial resolution.

```
┌──────────────────────────────────────────────────────────────────┐
│                      SqueezeNet Architecture Flow                │
│                                                                  │
│  Input Image ───►┌──────────────────┐  (Downsamples 2x)          │
│                  │  Conv1 + MaxPool │                            │
│                  └────────┬─────────┘                            │
│                           │                                      │
│                  ┌────────▼─────────┐                            │
│                  │   Fire Modules   │ (e.g., Fire2, Fire3)       │
│                  └────────┬─────────┘                            │
│                           │                                      │
│                  ┌────────▼─────────┐  (Downsamples 2x)          │
│                  │     MaxPool      │                            │
│                  └────────┬─────────┘                            │
│                           │                                      │
│                  ┌────────▼─────────┐                            │
│                  │   Fire Modules   │ (e.g., Fire4, Fire5)       │
│                  └────────┬─────────┘                            │
│                           │ ... and so on ...                    │
│                           │                                      │
│                  ┌────────▼──────────┐                           │
│                  │ Classification Head│                          │
│                  └────────────────────┘                          │
└──────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       SqueezeNet Complete Data Flow                     │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: STEM
────────────
Input Image (B, H, W, 3)
    │
    ├─► Conv2D (e.g., 7x7 kernel, stride 2)
    │
    └─► MaxPooling2D (3x3, stride 2)


STEP 2: FIRE MODULES & DOWNSAMPLING
───────────────────────────────────
Feature Map
    │
    ├─► Stack of Fire Modules (or SimplifiedFireModules)
    │   (Each module maintains the resolution)
    │
    ├─► [Periodically] MaxPooling2D (3x3, stride 2)
    │
    └─► Repeat until all Fire modules are processed


STEP 3: CLASSIFICATION HEAD
───────────────────────────
Final Feature Map
    │
    ├─► [Optional] Dropout
    │
    ├─► Final Conv2D (1x1 kernel, filters = num_classes)
    │
    ├─► Global Average Pooling
    │
    └─► Softmax Activation -> Probabilities (B, num_classes)
```

---

## 4. Architecture Deep Dive

### 4.1 `FireModule` (SqueezeNetV1)

-   **Purpose**: The fundamental, parameter-efficient building block of the V1 architecture.
-   **Architecture**:
    1.  **Squeeze Layer**: A `Conv2D` layer with only `1x1` filters. Its sole purpose is to reduce the channel dimensionality of its input, creating a bottleneck.
    2.  **Expand Layer**: Takes the low-dimensional output from the squeeze layer and feeds it to two parallel `Conv2D` layers: one with `1x1` filters and one with `3x3` filters.
    3.  **Concatenation**: The outputs of the two expand layers are concatenated along the channel axis, producing the final output of the module.

### 4.2 `SimplifiedFireModule` (SqueezeNodule-Net V2) and Squeeze Ratio

-   **Purpose**: An evolution of the Fire module designed to force spatial feature learning and widen the information bottleneck.
-   **Architecture**: It is a simpler version of the original:
    1.  **Squeeze Layer**: A `1x1` `Conv2D` layer, same as in V1.
    2.  **Expand Layer**: This layer contains **only the `3x3` convolutional path**. The `1x1` expand path is completely removed.
-   **How it Works**:
    1.  **Forced Spatial Context**: By removing the `1x1` expand path (which only mixes channels), the module is forced to learn features that incorporate local spatial information from the `3x3` convolution. This is hypothesized to be more effective for tasks requiring texture and shape analysis, like medical imaging.
    2.  **Wider Bottleneck (Squeeze Ratio)**: The Squeeze Ratio (SR), `s1x1 / total_expand_filters`, controls how much information is squeezed. SqueezeNetV1 uses a very low SR (e.g., 0.125) for maximum compression. SqueezeNoduleNetV2 uses a higher SR (e.g., 0.50), creating a wider bottleneck that allows more information to pass through, which can improve accuracy and training speed on complex datasets.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First SqueezeNet Model (30 seconds)

Let's build a SqueezeNetV1 for a simple classification task on CIFAR-10.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.squeezenet.squeezenet_v1 import create_squeezenet_v1

# 1. Create a SqueezeNetV1 model for CIFAR-10 (32x32 images, 10 classes)
# We use variant "1.1" which is better suited for small images.
model = create_squeezenet_v1(
    variant="1.1",
    num_classes=10,
    input_shape=(32, 32, 3)
)

# 2. Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
print("✅ SqueezeNetV1 model created and compiled successfully!")
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
| **`SqueezeNetV1`** | `...squeezenet.squeezenet_v1` | The main Keras `Model` for the V1 architecture. |
| **`create_squeezenet_v1`** | `...squeezenet.squeezenet_v1` | Recommended function to create `SqueezeNetV1` models. |
| **`SqueezeNoduleNetV2`**|`...squeezenet.squeezenet_v2`| The main Keras `Model` for the V2/NoduleNet architecture. |
| **`create_squeezenodule_net_v2`**|`...squeezenet.squeezenet_v2`| Recommended function to create `SqueezeNoduleNetV2` models.|

### 6.2 Core Building Blocks

| Layer | Location | Purpose |
| :--- | :--- | :--- |
| **`FireModule`** | `...squeezenet.squeezenet_v1` | The core block of the V1 architecture with squeeze/expand. |
| **`SimplifiedFireModule`** |`...squeezenet.squeezenet_v2` | The core block of the V2 architecture with only a 3x3 expand path.|

---

## 7. Configuration & Model Variants

### SqueezeNetV1 Variants

| Variant | Key Differences |
|:---:|:---|
| **`1.0`** | 7x7 stem convolution, pooling after Fire modules 1, 4, 8. **Warning: May fail on inputs < 64x64.** |
| **`1.1`** | 3x3 stem convolution, different pooling strategy. More efficient and stable on smaller inputs. |
| **`1.0_bypass`** | Same as `1.0` but adds residual (bypass) connections around some Fire modules to improve gradient flow. |

### SqueezeNoduleNetV2 Variants

| Variant | Squeeze Ratios (SR) | Dimensions |
|:---:|:---|:---|
| **`v1`** | SR=0.25 for all modules. Lighter version. | 2D |
| **`v2`** | SR=0.50 for early modules, SR=0.25 for later ones. Wider bottleneck. | 2D |
| **`v1_3d`**| SR=0.25 for all modules. | 3D |
| **`v2_3d`**| SR=0.50 / 0.25. Wider bottleneck. | 3D |

---

## 8. Comprehensive Usage Examples

### Example 1: Using SqueezeNet as a Feature Extraction Backbone

You can use a headless SqueezeNet as a lightweight backbone for other tasks.

```python
from dl_techniques.models.squeezenet.squeezenet_v1 import create_squeezenet_v1
import numpy as np

# 1. Create the feature extractor by setting include_top=False
backbone = create_squeezenet_v1(
    variant="1.0",
    include_top=False,
    input_shape=(224, 224, 3)
)

# 2. Extract features
dummy_images = np.random.rand(2, 224, 224, 3).astype("float32")
features = backbone.predict(dummy_images)

# The output is the feature map from the final Fire module (before the head)
# Spatial resolution is downsampled by 32x for this variant
print(f"Output shape: {features.shape}") # (2, 13, 13, 512)
```

### Example 2: Creating a 3D Model for Volumetric Data

SqueezeNodule-Net V2 is designed to work with 3D data like CT scans.

```python
from dl_techniques.models.squeezenet.squeezenet_v2 import create_squeezenodule_net_v2

# Create a 3D model for a binary classification task on 64x64x64 volumes
model_3d = create_squeezenodule_net_v2(
    variant="v2_3d",
    num_classes=2,
    input_shape=(64, 64, 64, 1) # (depth, height, width, channels)
)

print(f"3D model params: {model_3d.count_params():,}")
model_3d.summary()
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Using Bypass Connections for Better Training

The `1.0_bypass` variant of SqueezeNetV1 incorporates residual connections, similar to ResNet. This can improve gradient flow and make it easier to train deeper or more complex models based on the SqueezeNet architecture.

```python
from dl_techniques.models.squeezenet.squeezenet_v1 import create_squeezenet_v1

# Create the bypass variant
# This is useful for tasks where training might be unstable
bypass_model = create_squeezenet_v1(
    variant="1.0_bypass",
    num_classes=100,
    input_shape=(224, 224, 3)
)

# The Add layers in the summary show where the bypass connections are
bypass_model.summary()
```

---

## 10. Performance Optimization

### Mixed Precision Training

SqueezeNet's simple convolutional structure is ideal for mixed precision training, which can significantly accelerate training on modern GPUs with Tensor Cores.

```python
import keras

# Enable mixed precision globally before creating the model
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = create_squeezenet_v1("1.1", num_classes=1000)
model.compile(...)

# When training, use a LossScaleOptimizer to prevent numeric underflow
# Keras's model.fit() handles this automatically.
```

---

## 11. Training and Best Practices

### Optimizer and Regularization

-   **Optimizer**: **Adam** or **AdamW** are excellent choices for training SqueezeNet models.
-   **Regularization**: The `dropout_rate` (default is 0.5) is a crucial hyperparameter for preventing overfitting. It is applied after the final Fire module. You can also pass a `kernel_regularizer` to the constructor to apply weight decay to all convolutional layers.

### Input Image Size

-   **Crucial Consideration**: The default SqueezeNet V1.0 architecture uses `3x3` max-pooling with `'valid'` padding. This can cause the model to fail (either crash or produce `NaN` outputs) if the input image is too small (e.g., 32x32).
-   **Recommendation**: For small images (< 64x64), **use the `1.1` variant**, which has a more forgiving pooling strategy. For larger images (>= 64x64), the `1.0` variant is fine.

---

## 12. Serialization & Deployment

All SqueezeNet models and their custom `FireModule` layers are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
import keras
from dl_techniques.models.squeezenet.squeezenet_v1 import SqueezeNetV1, FireModule

# Create and train model
model = SqueezeNetV1.from_variant("1.1", num_classes=10, input_shape=(32,32,3))
# model.compile(...) and model.fit(...)

# Save the entire model to a single file
model.save('my_squeezenet_model.keras')

# Load the model back, providing the custom classes for deserialization
loaded_model = keras.models.load_model(
    'my_squeezenet_model.keras',
    custom_objects={"SqueezeNetV1": SqueezeNetV1, "FireModule": FireModule}
)
print("✅ SqueezeNetV1 model loaded successfully!")```

*Note: The same process applies to `SqueezeNoduleNetV2` and `SimplifiedFireModule`.*

---

## 13. Testing & Validation

### Unit Tests

You can validate the implementation with simple tests to ensure all variants can be created and run a forward pass.

```python
import numpy as np
from dl_techniques.models.squeezenet.squeezenet_v1 import create_squeezenet_v1
from dl_techniques.models.squeezenet.squeezenet_v2 import create_squeezenodule_net_v2

def test_v1_variants():
    create_squeezenet_v1("1.0", num_classes=10, input_shape=(64, 64, 3))
    create_squeezenet_v1("1.1", num_classes=10, input_shape=(32, 32, 3))
    print("✓ SqueezeNetV1 variants created successfully")

def test_v2_variants():
    create_squeezenodule_net_v2("v2", num_classes=10, input_shape=(64, 64, 3))
    create_squeezenodule_net_v2("v2_3d", num_classes=10, input_shape=(64, 64, 64, 1))
    print("✓ SqueezeNoduleNetV2 variants created successfully")

def test_forward_pass_shape():
    model = create_squeezenet_v1("1.1", num_classes=10, input_shape=(32, 32, 3))
    dummy_input = np.random.rand(4, 32, 32, 3).astype("float32")
    output = model.predict(dummy_input)
    assert output.shape == (4, 10)
    print("✓ Forward pass has correct shape")

if __name__ == '__main__':
    test_v1_variants()
    test_v2_variants()
    test_forward_pass_shape()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: My training crashes or the output is `NaN` with 32x32 images.**

-   **Cause**: You are likely using `SqueezeNetV1` with the `"1.0"` variant. Its aggressive `3x3` max-pooling with `'valid'` padding reduces the feature map dimensions to a size smaller than the pooling kernel.
-   **Solution**: For images smaller than 64x64, **always use the `"1.1"` variant**. It has a different stem and pooling strategy designed to be more robust.

### Frequently Asked Questions

**Q: What is the main difference between SqueezeNetV1 and SqueezeNoduleNetV2?**

A: There are two key differences. First, SqueezeNoduleNetV2 uses a **`SimplifiedFireModule`** that removes the `1x1` expand path to force spatial feature learning. Second, it generally uses a **higher Squeeze Ratio**, creating a wider bottleneck to preserve more information, which is beneficial for complex patterns. It also adds native support for 3D convolutions.

**Q: Why is SqueezeNet so small?**

A: The efficiency comes from the **Fire module**. The `1x1` squeeze layer acts as a bottleneck, dramatically reducing the number of channels that the more expensive `3x3` convolutions have to process. This design is the primary driver of its extreme parameter reduction.

**Q: When should I use the `1.0_bypass` variant?**

A: Use the bypass variant when you are training a deeper network or if you find that training is unstable. The residual connections help gradients flow more easily through the network, which can stabilize and accelerate training, similar to how they work in ResNet.

---

## 15. Technical Details

### The Fire Module Bottleneck

The core principle of SqueezeNet is the squeeze-expand bottleneck. A `FireModule` might receive a feature map with a large number of channels, squeeze it down to a very small number, and then expand it back up.

-   **Example Flow**:
    `Input (128 channels) -> Squeeze (16 channels) -> Expand (64 from 1x1 + 64 from 3x3) -> Output (128 channels)`

In this example, the expensive `3x3` convolution only sees an input with 16 channels instead of 128, providing a massive reduction in computation and parameters.

### Squeeze Ratio (SR)

The Squeeze Ratio is a key hyperparameter that controls the compression level of the Fire module. It is defined as:
`SR = num_squeeze_filters / num_expand_filters`

-   A **low SR** (e.g., 0.125 in SqueezeNetV1) creates a very aggressive bottleneck, maximizing parameter efficiency.
-   A **high SR** (e.g., 0.50 in SqueezeNoduleNetV2) creates a wider bottleneck, preserving more information at the cost of slightly more parameters. This can lead to better accuracy on fine-grained tasks.

---

## 16. Citation

This implementation is based on the official papers. If you use these models in your research, please consider citing the original works:

-   **SqueezeNet V1**:
    ```bibtex
    @article{iandola2016squeezenet,
      title={SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size},
      author={Iandola, Forrest N and Han, Song and Moskewicz, Matthew W and Ashraf, Khalid and Dally, William J and Keutzer, Kurt},
      journal={arXiv preprint arXiv:1602.07360},
      year={2016}
    }
    ```
-   **SqueezeNodule-Net (V2 basis)**:
    ```bibtex
    @article{tsivgoulis2022improved,
      title={An improved SqueezeNet model for the diagnosis of lung cancer in CT scans},
      author={Tsivgoulis, Georgios and Skiadopoulos, Spiros and Vassilacopoulos, George},
      journal={Machine Learning with Applications},
      volume={9},
      pages={100399},
      year={2022},
      publisher={Elsevier}
    }
    ```
# FractalNet: Ultra-Deep Neural Networks without Residuals

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured Keras 3 implementation of **FractalNet**, a self-similar deep neural network that achieves great depths without using residual connections. Instead, it relies on a recursive fractal expansion rule that creates an exponential number of paths through the network, regularized by a "drop-path" training scheme.

This implementation follows the `dl_techniques` framework standards and modern Keras 3 best practices. It provides a modular, well-documented, and fully serializable model that works seamlessly across TensorFlow, PyTorch, and JAX backends. The architecture is constructed from recursive `FractalBlock` layers, providing a powerful alternative to standard ResNet-like designs.

---

## Table of Contents

1. [Overview: What is FractalNet and Why It Matters](#1-overview-what-is-fractalnet-and-why-it-matters)
2. [The Problem FractalNet Solves](#2-the-problem-fractalnet-solves)
3. [How FractalNet Works: Core Concepts](#3-how-fractalnet-works-core-concepts)
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

## 1. Overview: What is FractalNet and Why It Matters

### What is FractalNet?

**FractalNet** is a deep convolutional neural network that challenges the necessity of residual connections for training very deep models. Instead of adding identity shortcuts (like in ResNet), FractalNet builds depth using a **recursive, self-similar design**. A block of depth `B` is composed of two parallel blocks of depth `B-1`, whose outputs are averaged. This creates a fractal pattern with an immense number of distinct paths from input to output.

### Key Innovations

1.  **Recursive Fractal Expansion**: The architecture is defined by a simple, elegant rule: a deep block is the combination of shallower blocks. This creates a self-similar structure at all scales.
2.  **Implicit Deep Ensembling**: The fractal design implicitly contains an ensemble of sub-networks of varying depths. Any path from the input to the output forms a valid, shallower network.
3.  **Drop-Path Regularization**: The key to training FractalNet is **drop-path**, a stochastic training method where entire sub-branches of the fractal are randomly dropped. This forces the network to learn robust features and ensures that all paths, short and long, are trained effectively.

### Why FractalNet Matters

**The Ultra-Deep Network Problem**:
```
Problem: Train a very deep neural network without gradients vanishing or exploding.
ResNet's Solution:
  1. Add identity "skip connections" that bypass layers.
  2. This creates a direct path for gradients to flow, allowing for networks with
     hundreds or even thousands of layers.
  3. Limitation: This design makes the network heavily reliant on the short paths
     created by the skip connections.
```

**FractalNet's Solution**:
```
FractalNet's Approach:
  1. Create a massive number of paths of varying lengths through recursive design,
     but without explicit skip connections.
  2. During training, use drop-path to randomly disable entire branches, forcing the
     network to learn to use a diverse set of paths.
  3. At inference, use the entire network (all paths active), which acts like averaging
     the predictions of an exponential-sized ensemble of networks.
  4. Benefit: Achieves the performance of ultra-deep networks through a different
     philosophy—redundancy and ensembling rather than identity shortcuts.
```

### Real-World Impact

FractalNet offers a different perspective on deep learning architecture and is valuable for:

-   🖼️ **Image Classification**: Demonstrates competitive performance against ResNets on standard benchmarks like CIFAR and ImageNet.
-   🔬 **Architectural Research**: Provides a compelling alternative to residual networks and serves as a testbed for ideas like stochastic depth and implicit ensembling.
-   💪 **Robust Feature Learning**: The drop-path mechanism encourages the learning of highly robust and redundant features.

---

## 2. The Problem FractalNet Solves

### The Challenge of Depth

Before 2015, training very deep neural networks was notoriously difficult. As networks got deeper, they suffered from the **vanishing gradient problem**, where the signal from the loss function would diminish as it propagated backward through many layers, causing the early layers to stop learning.

```
┌─────────────────────────────────────────────────────────────┐
│  The Dilemma of Deep Architectures                          │
│                                                             │
│  "Plain" Deep Networks:                                     │
│    - Suffer from vanishing gradients.                       │
│    - Often exhibit "degradation," where adding more layers  │
│      actually increases the training error.                 │
│                                                             │
│  ResNet's Solution (The "Highway"):                         │
│    - Identity skip connections create an information highway│
│      allowing gradients to bypass layers and flow freely.   │
│    - Solved the degradation problem and enabled extreme depth.
└─────────────────────────────────────────────────────────────┘
```

FractalNet questioned if these explicit highways were the *only* solution. It proposed an alternative: what if instead of one superhighway, we built a massive network of redundant side roads?

### How FractalNet Changes the Game

FractalNet's design provides a rich connectivity pattern where the effective depth is dynamic and stochastic during training.

```
┌─────────────────────────────────────────────────────────────┐
│  The FractalNet Redundancy Strategy                         │
│                                                             │
│  1. Create Redundant Paths:                                 │
│     - The recursive structure creates 2^(B-1) distinct paths│
│       in a block of depth B.                                │
│     - This provides many ways for information to flow, with │
│       paths of many different lengths.                      │
│                                                             │
│  2. Force the Use of All Paths:                             │
│     - Drop-path randomly "closes" paths during training.    │
│     - The network cannot rely on any single path (short or  │
│       long) and must learn to solve the task using whatever │
│       sub-network is available in each training step.       │
│                                                             │
│  3. Average the Ensemble at Inference:                      │
│     - With all paths open, the network behaves like a huge  │
│       ensemble, improving generalization.                   │
└─────────────────────────────────────────────────────────────┘
```

This approach successfully trains very deep networks by treating depth as a form of ensembling, not just sequential processing.

---

## 3. How FractalNet Works: Core Concepts

### The Hierarchical Multi-Stage Architecture

Like most CNNs, FractalNet processes an image through several stages, progressively downsampling the spatial resolution while increasing the number of feature channels.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      FractalNet Architecture Stages                 │
│                                                                     │
│  Input Image ───►┌──────────────────┐  (Downsamples 2x)             │
│   (H, W)         │   Fractal Stage 1  │  (e.g., Depth=2, Filters=32)│
│                  └────────┬─────────┘                               │
│                           │ (H/2, W/2)                              │
│                  ┌────────▼─────────┐  (Downsamples 2x)             │
│                  │ Fractal Stage 2  │  (e.g.Depth=3, Filters=64)    │
│                  └────────┬─────────┘                               │
│                           │ (H/4, W/4)                              │
│                  ┌────────▼─────────┐  (Downsamples 2x)             │
│                  │ Fractal Stage 3  │  (e.g.Depth=3, Filters=128)   │
│                  └────────┬─────────┘                               │
│                           │ (H/8, W/8)                              │
│                  ┌────────▼──────────┐                              │
│                  │ Classification Head│                             │
│                  └────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       FractalNet Complete Data Flow                     │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: FRACTAL STAGE 1
───────────────────────
Input Image (B, H, W, 3)
    │
    ├─► FractalBlock(depth=D₁, filters=F₁, stride=2)
    │   (This block contains its own recursive structure and downsamples)
    │
    └─► Feature Map 1: (B, H/2, W/2, F₁)


STEP 2: FRACTAL STAGE 2
───────────────────────
Feature Map 1
    │
    ├─► FractalBlock(depth=D₂, filters=F₂, stride=2)
    │
    └─► Feature Map 2: (B, H/4, W/4, F₂)


STEP 3: FRACTAL STAGE 3, etc.
─────────────────────────────
... (Continue for all defined stages)


STEP 4: CLASSIFICATION
──────────────────────
Final Feature Map
    │
    ├─► Global Average Pooling
    │
    ├─► [Optional] Dropout
    │
    ├─► Dense Layer (Classifier)
    │
    └─► Logits (B, num_classes)
```

---

## 4. Architecture Deep Dive

### 4.1 `FractalBlock` (The Recursive Engine)

-   **Purpose**: To implement the recursive fractal expansion rule, which is the heart of the architecture.
-   **Architecture**:
    *   **Base Case (depth=1)**: The recursion terminates at a single, standard `ConvBlock` (Conv-Norm-Act). This is the fundamental unit of computation.
    *   **Recursive Step (depth > 1)**: A `FractalBlock` of depth `k` creates two parallel `FractalBlock` branches, each of depth `k-1`. The inputs are passed through both branches.
    *   **Drop-Path & Join**: The output of each branch is passed through a `StochasticDepth` (drop-path) layer. The final outputs of the two branches are then averaged together.
-   **Benefit**: This design creates `2^(k-1)` leaf `ConvBlock`s and an exponential number of paths. The parameters are shared at each level of the recursion but are independent between the two branches, providing architectural diversity.

### 4.2 `ConvBlock` (The Base Unit)

-   **Purpose**: To serve as the "leaf" nodes in the fractal tree. It performs the actual feature extraction.
-   **Implementation**: A standard sequence of `Conv2D` -> `Normalization` (e.g., `BatchNorm`) -> `Activation` (e.g., `ReLU`).
-   **Functionality**: This implementation uses a highly configurable `ConvBlock` that allows for easy experimentation with different normalization and activation functions.

### 4.3 `StochasticDepth` (Drop-Path)

-   **Purpose**: To regularize the network during training by randomly dropping entire computational paths.
-   **Functionality**: During training, this layer randomly sets its entire input tensor to zero with a probability of `drop_path_rate`. During inference, it acts as an identity function.
-   **Benefit**: This is the critical component that makes FractalNet trainable. It prevents co-adaptation of parallel paths and forces the network to learn redundant features, making the final "ensembled" model at inference time much more robust.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First FractalNet Model (30 seconds)

Let's build and compile a small FractalNet for CIFAR-10.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.fractal_net.model import create_fractal_net

# 1. Create a FractalNet-Small model for CIFAR-10 (32x32 images, 10 classes)
# The create_fractal_net function also compiles the model.
model = create_fractal_net(
    variant="small",
    num_classes=10,
    input_shape=(32, 32, 3),
    learning_rate=1e-3
)
print("✅ FractalNet model created and compiled successfully!")
model.summary()

# 2. Create dummy data for a forward pass
batch_size = 16
dummy_images = np.random.rand(batch_size, 32, 32, 3).astype("float32")
dummy_labels = np.random.randint(0, 10, batch_size)

# 3. Train for one step
# Note: drop-path is active during training
history = model.fit(dummy_images, dummy_labels, epochs=1, verbose=1)
print(f"\n✅ Training step complete!")

# 4. Run inference
# Note: drop-path is disabled during inference, using the full network
predictions = model.predict(dummy_images)
print(f"Predictions shape: {predictions.shape}") # (batch_size, num_classes)
```

---

## 6. Component Reference

### 6.1 Model Class and Creation Functions

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`FractalNet`** | `...fractal_net.model.FractalNet` | The main Keras `Model` that assembles the fractal stages. |
| **`create_fractal_net`** | `...fractal_net.model.create_fractal_net` | Recommended convenience function to create and compile `FractalNet` models. |

### 6.2 Core Building Blocks

| Layer | Location | Purpose |
| :--- | :--- | :--- |
| **`FractalBlock`** | `...layers.fractal_block.FractalBlock` | The core recursive block that defines the fractal structure. |
| **`ConvBlock`** | `...layers.standard_blocks.ConvBlock` | The base-case convolutional unit used at the leaves of the fractal. |
| **`StochasticDepth`**| `...layers.stochastic_depth.StochasticDepth` | Implements the drop-path regularization critical for training. |

---

## 7. Configuration & Model Variants

This implementation provides several pre-configured variants based on the original paper.

| Variant | Depths | Filters | Description |
|:---:|:---|:---|:---|
| **`micro`** | `[1, 2, 2]` | `[16, 32, 64]` | A very small model, suitable for testing or small datasets like MNIST. |
| **`small`**| `[2, 3, 3]` | `[32, 64, 128]` | A standard configuration for datasets like CIFAR-10/100. |
| **`medium`**|`[3, 4, 4]` | `[64, 128, 256]`| A larger, more capable model. |
| **`large`**| `[4, 5, 5]` | `[96, 192, 384]`| A deep configuration suitable for larger-scale datasets like ImageNet. |

---

## 8. Comprehensive Usage Examples

### Example 1: Using FractalNet as a Feature Extraction Backbone

You can use a headless FractalNet as a backbone for downstream tasks like segmentation or detection.

```python
# 1. Create the feature extractor
backbone = FractalNet.from_variant(
    "medium",
    include_top=False,
    input_shape=(256, 256, 3)
)

# 2. Extract features
dummy_images = np.random.rand(2, 256, 256, 3).astype("float32")
features = backbone.predict(dummy_images)

# The output is the feature map from the final fractal stage
# Spatial resolution is downsampled by 8x (2^3 stages)
print(f"Output shape: {features.shape}") # (2, 32, 32, 256)
```

### Example 2: Creating a Custom FractalNet Architecture

You can easily define your own FractalNet by specifying the depths and filters for each stage.

```python
# Create a deep but narrow FractalNet with 4 stages
custom_model = FractalNet(
    num_classes=50,
    depths=[2, 3, 4, 2],         # Four stages with varying fractal depths
    filters=[32, 64, 128, 256],  # Filters for each stage
    input_shape=(128, 128, 3),
    drop_path_rate=0.2           # Increase regularization for deeper model
)

custom_model.summary()
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Adjusting Regularization with Drop-Path

The `drop_path_rate` is the most important hyperparameter for regularizing a FractalNet.

-   For **smaller datasets or shallower models**, a lower rate (`0.0 - 0.1`) might be sufficient.
-   For **larger datasets or deeper models** (like `medium` or `large` variants), a higher rate (`0.15 - 0.25`) is crucial to prevent overfitting and ensure all paths are trained.

```python
# A model for a large dataset with strong regularization
model = create_fractal_net(
    "large",
    num_classes=1000,
    input_shape=(224, 224, 3),
    drop_path_rate=0.25  # Higher drop-path rate
)
```

### Pattern 2: Understanding Training vs. Inference Behavior

It's important to remember the dual nature of the model:

-   **During `model.fit()` (training=True)**: The model is a stochastic ensemble. Drop-path is active, and in each step, a different sub-network is trained.
-   **During `model.predict()` or `model.evaluate()` (training=False)**: The model is a deterministic, deep network. Drop-path is turned off, and the full "averaged" ensemble is used for prediction. This is why FractalNet generalizes well.

---

## 10. Performance Optimization

### Mixed Precision Training

FractalNet, being a standard CNN, benefits greatly from mixed precision training, which can accelerate training significantly on compatible GPUs (NVIDIA Tensor Core).

```python
# Enable mixed precision globally before creating the model
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = create_fractal_net("medium", num_classes=100)

# Keras's model.fit() will handle loss scaling automatically.
```

---

## 11. Training and Best Practices

### Optimizer and Learning Rate

-   **Optimizer**: The original paper used **SGD with Nesterov momentum**. This remains a strong choice. Modern optimizers like **AdamW** also work well.
-   **Learning Rate Schedule**: A **cosine decay** or a **step decay** learning rate schedule is highly recommended. A few epochs of linear warmup at the start can also help stabilize training.

### Data Augmentation

-   FractalNets benefit from standard data augmentation techniques used for CNNs. For image classification, this includes:
    -   Random horizontal flips
    -   Random crops (after padding)
    -   Techniques like Cutout or Mixup can also be effective.

---

## 12. Serialization & Deployment

The `FractalNet` model and all its custom layers (`FractalBlock`) are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
# Create and train the model
model = create_fractal_net("small", num_classes=10)
# ... model.fit(...)

# Save the entire model to a single file
model.save('my_fractalnet_model.keras')

# Load the model in a new session
# The custom FractalBlock layer is automatically handled.
loaded_model = keras.models.load_model('my_fractalnet_model.keras')
print("✅ Model loaded successfully!")
```

---

## 13. Testing & Validation

### Unit Tests

Simple tests can validate that all model variants are created correctly and that the forward pass produces the expected output shape.

```python
import keras
import numpy as np
from dl_techniques.models.fractal_net.model import FractalNet

def test_creation_all_variants():
    """Test model creation for all variants."""
    for variant in FractalNet.MODEL_VARIANTS.keys():
        model = FractalNet.from_variant(variant, num_classes=10, input_shape=(64, 64, 3))
        assert model is not None
        print(f"✓ FractalNet-{variant} created successfully")

def test_forward_pass_shape():
    """Test the output shape of a forward pass."""
    model = FractalNet.from_variant("small", num_classes=10, input_shape=(32, 32, 3))
    dummy_input = np.random.rand(4, 32, 32, 3).astype("float32")
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

**Issue 1: Training loss is unstable or NAN.**

-   **Cause 1**: The learning rate might be too high.
-   **Solution 1**: Use a smaller learning rate and consider a warmup schedule.
-   **Cause 2**: Poor weight initialization.
-   **Solution 2**: Ensure you are using a standard initialization like `he_normal` for the convolutional kernels, which is the default in this implementation.

### Frequently Asked Questions

**Q: How is FractalNet different from ResNet?**

A: The core difference is the mechanism used to enable deep network training. **ResNet** uses explicit **identity skip connections** to create a direct path for gradients. **FractalNet** creates an **implicit ensemble** of many paths of varying lengths through its recursive structure and uses **drop-path** to ensure all paths are trained.

**Q: What is the point of the two parallel branches in a `FractalBlock`?**

A: The two branches create path diversity. Although they share the same *architecture* (i.e., they are both `FractalBlock`s of depth `k-1`), they have **independent sets of weights**. This means they learn different features, and averaging their outputs acts like a mini-ensemble at each level of the fractal.

**Q: Is FractalNet computationally expensive?**

A: A `FractalBlock` of depth `k` contains `2^(k-1)` base `ConvBlock`s. This means the computational cost and parameter count grow exponentially with the fractal depth. However, the models are designed with reasonable depths (e.g., up to 5 per stage), making them comparable to other deep CNNs like ResNet.

---

## 15. Technical Details

### The Mathematics of Fractal Expansion and Drop-Path

The architecture is defined by the recursive rule:
`F_k(x) = 0.5 * (DP(f_1(x)) + DP(f_2(x)))`
where `f_1` and `f_2` are two distinct instances of the block `F_{k-1}`, and `DP` is the drop-path operator.

**Drop-Path during Training**:
The drop-path operator `DP` can be seen as multiplying the branch output by a Bernoulli random variable `b ~ Bernoulli(1 - p)`, where `p` is the `drop_path_rate`.
`DP(y) = y * b`

This means that during each forward pass, a random sub-graph of the full fractal network is sampled and trained. The deepest path is only active if no branches are dropped, while the shallowest paths are active much more frequently.

**Inference as Ensemble Averaging**:
At test time, the drop-path probability `p` is set to 0, so `DP(y) = y`. The final output is the deterministic average of all `2^(B-1)` paths in the network. This is analogous to averaging the predictions of a massive, jointly trained ensemble of neural networks, which is key to FractalNet's strong generalization performance.

---

## 16. Citation

This implementation is based on the original FractalNet paper. If you use this model or its concepts in your research, please cite the original work:

```bibtex
@inproceedings{larsson2017fractalnet,
  title={Fractalnet: Ultra-deep neural networks without residuals},
  author={Larsson, Gustav and Maire, Michael and Shakhnarovich, Gregory},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2017}
}```
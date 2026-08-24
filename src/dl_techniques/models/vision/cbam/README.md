# CBAMNet: A Convolutional Network with Attention

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of **CBAMNet**, a convolutional neural network enhanced with the **Convolutional Block Attention Module (CBAM)**. This lightweight and effective attention mechanism refines features at each stage of the network, allowing the model to learn *what* and *where* to focus in the feature maps.

The architecture consists of standard convolutional blocks, each followed by a `CBAM` layer that sequentially applies channel and spatial attention.

---

## Table of Contents

1. [Overview: What is CBAMNet and Why It Matters](#1-overview-what-is-cbamnet-and-why-it-matters)
2. [The Problem CBAMNet Solves](#2-the-problem-cbamnet-solves)
3. [How CBAMNet Works: Core Concepts](#3-how-cbamnet-works-core-concepts)
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
15. [Citation](#15-citation)

---

## 1. Overview: What is CBAMNet and Why It Matters

### What is CBAMNet?

**CBAMNet** is a convolutional neural network that incorporates the **Convolutional Block Attention Module (CBAM)** to improve its representational power. Instead of treating all features equally, CBAMNet learns to selectively emphasize important features and suppress irrelevant ones. It does this by inferring attention maps along two separate dimensions: **channel** and **spatial**.

### Key Innovations

1.  **Factorized Attention**: CBAM decouples the attention mechanism into two sequential sub-modules: a Channel Attention Module and a Spatial Attention Module. This makes the attention mechanism lightweight and efficient.
2.  **Channel Attention ("What")**: This module learns the importance of each feature channel. It aggregates spatial information using both average and max pooling, then uses a shared Multi-Layer Perceptron (MLP) to compute channel-wise attention weights.
3.  **Spatial Attention ("Where")**: This module learns the most informative spatial regions. It aggregates channel information using pooling operations along the channel axis and then uses a convolutional layer to generate a 2D spatial attention map.
4.  **Sequential Refinement**: The key architectural choice is to apply channel attention first, which refines the feature map by re-weighting channels. The subsequent spatial attention module then operates on these channel-refined features to identify important spatial locations.

### Why CBAMNet Matters

**Standard CNN Problem**:
```
Problem: Extract the most relevant features from an image.
Standard CNN Approach:
  1. Apply a series of convolutional filters to the entire feature map.
  2. Limitation: Each filter is applied uniformly across all spatial locations
     and the network treats all feature channels with equal importance. This can
     lead to computational resources being wasted on less informative regions
     or features.
  3. Result: The model may struggle to distinguish between subtle but important
     features and background noise, potentially limiting its accuracy.
```

**CBAMNet's Solution**:
```
CBAMNet Approach:
  1. After a standard convolution, insert a lightweight CBAM block.
  2. First, the Channel Attention module decides "what" is important by
     assigning a weight to each feature channel.
  3. Second, the Spatial Attention module decides "where" to focus by
     assigning a weight to each spatial location (pixel).
  4. The input feature map is multiplied by these attention maps, effectively
     recalibrating it to highlight the most salient information.
  5. Benefit: Improves model accuracy and interpretability with a negligible
     increase in parameters and computation, making the network more focused
     and efficient.
```

### Real-World Impact

CBAMNet is a general-purpose architecture suitable for a wide range of computer vision tasks where feature salience is important:

-   🖼️ **Image Classification**: Improves accuracy by focusing on discriminative object parts.
-   🎯 **Object Detection**: Enhances feature maps to better localize objects of interest.
-   🎨 **Image Segmentation**: Helps in distinguishing object boundaries from the background.
-   🔄 **Domain Adaptation**: Can improve generalization by focusing on domain-invariant features.

---

## 2. The Problem CBAMNet Solves

### The Limitation of Uniform Feature Processing

In a standard CNN, the network learns feature detectors (filters), but it applies them uniformly. It lacks an explicit mechanism to dynamically prioritize the most informative features or regions based on the input context.

```
┌─────────────────────────────────────────────────────────────┐
│  The Dilemma of Standard Convolutions                       │
│                                                             │
│  - All spatial locations in a feature map are treated equally.
│    (The background pixels get as much computation as the    │
│    foreground object).                                      │
│                                                             │
│  - All feature channels are considered equally important.   │
│    (A channel detecting simple edges is weighted the same as│
│    one detecting complex textures).                         │
│                                                             │
│  This can lead to suboptimal feature representations, where │
│  critical information is diluted by noise or irrelevant data.
└─────────────────────────────────────────────────────────────┘
```

### How CBAM Changes the Game

CBAM introduces a dynamic, input-dependent attention mechanism that allows the network to adaptively refine its feature maps.

```
┌─────────────────────────────────────────────────────────────┐
│  The CBAM Adaptive Refinement Strategy                      │
│                                                             │
│  1. For a given feature map F, CBAM computes:               │
│     - A channel attention map Mc (What to focus on).        │
│     - A spatial attention map Ms (Where to focus on).       │
│                                                             │
│  2. The refinement is applied sequentially:                 │
│     - F' = F * Mc (Recalibrate channels).                   │
│     - F'' = F' * Ms (Recalibrate spatial locations).        │
│                                                             │
│  The refined map F'' now has its most important features    │
│  and regions amplified, providing a better representation   │
│  for subsequent layers.                                     │
└─────────────────────────────────────────────────────────────┘
```
This "learn to focus" ability makes the CNN more efficient and powerful, often leading to better performance on complex visual tasks.

---

## 3. How CBAMNet Works: Core Concepts

### The Sequential Attention Mechanism

CBAMNet is built from a series of stages. Each stage consists of a standard convolutional block followed by a CBAM block. The CBAM block itself performs two sequential operations.

```
┌──────────────────────────────────────────────────────────────────┐
│                     CBAM Block Data Flow                         │
│                                                                  │
│  Input Feature Map ────►┌───────────────────┐                    │
│        (F)              │ Channel Attention │───► Refined Feature│
│                         │       (Mc)        │      Map (F')      │
│                         └────────┬──────────┘                    │
│                                  │ (Element-wise multiplication) │
│                                  │                               │
│                         ┌────────▼──────────┐                    │
│       F' ─────────────► │ Spatial Attention │───► Final Refined  │
│                         │       (Ms)        │   Feature Map (F'')│
│                         └────────┬──────────┘                    │
│                                  │ (Element-wise multiplication) │
│                                  ▼                               │
│                              Output                              │
└──────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow in CBAMNet

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CBAMNet Complete Data Flow                          │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: STAGE 0
────────────────
Input Image (B, H, W, 3)
    │
    ├─► Conv2D -> (B, H, W, D₀)
    │
    ├─► BatchNormalization
    │
    ├─► CBAM Block (Channel Attention -> Spatial Attention)
    │
    ├─► MaxPooling2D -> (B, H/2, W/2, D₀)
    │
    └─► Feature Map 0


STEP 2: STAGE 1
────────────────
Feature Map 0
    │
    ├─► Conv2D -> (B, H/2, W/2, D₁)
    │
    ├─► BatchNormalization
    │
    ├─► CBAM Block
    │
    ├─► MaxPooling2D -> (B, H/4, W/4, D₁)
    │
    └─► Feature Map 1


... (Repeat for all stages) ...


STEP N: CLASSIFICATION
──────────────────────
Final Feature Map
    │
    ├─► Global Average Pooling -> (B, D_final)
    │
    ├─► Dense Layer (Classifier)
    │
    └─► Logits (B, num_classes)
```
---

## 4. Architecture Deep Dive

### 4.1 `ChannelAttention`

-   **Purpose**: To decide "what" features are important. It computes a 1D attention vector that re-weights the channels.
-   **Architecture**:
    1.  **Aggregate**: Applies both Global Average Pooling and Global Max Pooling to the input feature map to create two channel descriptors.
    2.  **Learn**: Both descriptors are fed through a shared MLP with a bottleneck structure (a reduction layer followed by an expansion layer).
    3.  **Combine & Activate**: The two output vectors from the MLP are added together and passed through a sigmoid function to produce the final channel attention weights.

### 4.2 `SpatialAttention`

-   **Purpose**: To decide "where" to focus. It generates a 2D attention map to re-weight spatial locations.
-   **Architecture**:
    1.  **Aggregate**: Applies average pooling and max pooling along the channel axis to create two 2D feature maps that summarize the information at each spatial location.
    2.  **Combine**: The two maps are concatenated together.
    3.  **Learn**: A single 7x7 convolutional layer is applied to the concatenated map, followed by a sigmoid activation, to produce the final spatial attention map.

### 4.3 `CBAM` Layer

-   **Purpose**: The main block that orchestrates the sequential application of channel and spatial attention.
-   **Functionality**: Takes an input feature map `F`, applies `ChannelAttention` to get `Mc`, multiplies `F` by `Mc` to get `F'`, then applies `SpatialAttention` to `F'` to get `Ms`, and finally multiplies `F'` by `Ms` to get the output.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First CBAMNet Model (30 seconds)

Let's build a tiny CBAMNet for CIFAR-10 classification.

```python
import keras
import numpy as np

# Local imports from your project structure
# Ensure cbam_net.py and the attention layers are in the correct path
from dl_techniques.models.vision.cbam_net import CBAMNet

# 1. Create a tiny CBAMNet model for CIFAR-10 (32x32 images, 10 classes)
model = CBAMNet.from_variant(
    "tiny",
    num_classes=10,
    input_shape=(32, 32, 3)
)

# 2. Compile the model
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)
print("✅ CBAMNet model created and compiled successfully!")
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

### 6.1 `CBAMNet` (Model Class)

**Purpose**: The main Keras `Model` subclass that assembles the complete CBAMNet architecture.

**Location**: `dl_techniques.models.vision.cbam_net.CBAMNet`

```python
from dl_techniques.models.vision.cbam_net import CBAMNet

# Create from a standard variant
model = CBAMNet.from_variant(
    "base",
    num_classes=1000,
    input_shape=(224, 224, 3)
)

# Create a custom model
custom_model = CBAMNet(
    num_classes=100,
    dims=[32, 64, 128, 256],
    attention_ratio=16
)
```
### 6.2 Core Building Blocks

| Layer | Location | Purpose |
| :--- | :--- | :--- |
| **`CBAM`** | `...layers.attention.convolutional_block_attention.CBAM` | The main attention block applying channel then spatial attention. |
| **`ChannelAttention`** | `...layers.attention.channel_attention.ChannelAttention` | Computes channel-wise attention weights ("what"). |
| **`SpatialAttention`** | `...layers.attention.spatial_attention.SpatialAttention` | Computes spatial attention weights ("where"). |

---

## 7. Configuration & Model Variants

This implementation provides several pre-configured variants.

| Variant | Dims | Description |
|:---:|:---|:---|
| **`tiny`** | `[64, 128]` | A very small model, suitable for quick experiments or simple datasets. |
| **`small`**| `[64, 128, 256]` | A medium-sized model with three stages. |
| **`base`** | `[128, 256, 512]`| A larger model for more complex datasets like ImageNet. |

---

## 8. Comprehensive Usage Examples

### Example 1: Using CBAMNet as a Feature Extraction Backbone

You can use a headless CBAMNet as a feature backbone for tasks like object detection or segmentation.

```python
# 1. Create the feature extractor
backbone = CBAMNet.from_variant(
    "base",
    include_top=False,
    input_shape=(224, 224, 3)
)

# 2. Extract features
dummy_images = np.random.rand(2, 224, 224, 3).astype("float32")
features = backbone.predict(dummy_images)

# The output is the feature map from the final stage
# Spatial resolution is downsampled by 8x for the 'base' variant (3 stages of pooling)
print(f"Output shape: {features.shape}") # (2, 28, 28, 512)
```

### Example 2: Accessing Multi-Scale Features

To build a Feature Pyramid Network (FPN), you can access the outputs of intermediate stages.

```python
# 1. Create the base model
base_model = CBAMNet.from_variant(
    "small",
    include_top=False,
    input_shape=(256, 256, 3)
)

# 2. Identify the output layers of each stage
# The output of each CBAM block is a good feature representation
stage_output_names = [
    'stage_0_cbam',
    'stage_1_cbam',
    'stage_2_cbam'
]
stage_outputs = [base_model.get_layer(name).output for name in stage_output_names]

# 3. Create a new model that outputs these features
feature_extractor = keras.Model(
    inputs=base_model.input,
    outputs=stage_outputs
)

# 4. Get the multi-scale features
dummy_images = np.random.rand(1, 256, 256, 3).astype("float32")
multi_scale_features = feature_extractor.predict(dummy_images)

print("Multi-scale feature map shapes:")
for name, features in zip(stage_output_names, multi_scale_features):
    print(f"  - {name}: {features.shape}")
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Customizing Attention Regularization

You can apply different regularization strengths to the channel and spatial attention components to fine-tune model behavior.

```python
from dl_techniques.layers.attention.convolutional_block_attention import CBAM

# Apply stronger L2 regularization to the spatial attention kernel
# This might encourage the model to learn smoother, less noisy spatial maps.
custom_cbam = CBAM(
    channels=128,
    ratio=16,
    channel_kernel_regularizer=keras.regularizers.L2(1e-5),
    spatial_kernel_regularizer=keras.regularizers.L2(1e-4) # Stronger regularization
)

# You can integrate this into a custom model build
# ...
```

---

## 10. Performance Optimization

### Mixed Precision Training

CBAMNet supports mixed precision training, which uses 16-bit floating-point precision for many operations to accelerate training on modern GPUs without significant accuracy loss.

```python
# Enable mixed precision globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = CBAMNet.from_variant("base", num_classes=1000)
model.compile(...) # Use a LossScaleOptimizer if using TensorFlow backend directly
```

---

## 11. Training and Best Practices

-   **Optimizer**: **AdamW** is a robust choice, as weight decay can be an effective regularizer.
-   **Learning Rate Schedule**: A **cosine decay** schedule often provides the best results for training deep CNNs from scratch.
-   **Data Augmentation**: Standard vision augmentations like random flips, rotations, and color jitter are highly recommended. For more challenging datasets, consider stronger augmentations like **RandAugment** or **Mixup**.

---

## 12. Serialization & Deployment

The `CBAMNet` model and all its custom layers are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
# Create and train model
model = CBAMNet.from_variant("tiny", num_classes=10)
# model.compile(...) and model.fit(...)

# Save the entire model
model.save('my_cbam_model.keras')

# Load the model in a new session
# The custom CBAM objects will be deserialized automatically
loaded_model = keras.models.load_model(
    'my_cbam_model.keras',
    custom_objects={'CBAMNet': CBAMNet} # Not needed if registered globally, but good practice
)
print("✅ Model loaded successfully!")
```

---

## 13. Troubleshooting & FAQs

**Issue 1: Training accuracy is low or stalls.**

-   **Cause**: The attention mechanism might be struggling to learn meaningful maps, especially on smaller or simpler datasets.
-   **Solution 1**: Try a smaller `attention_ratio` (e.g., 4 or 2) in the CBAM block to give the channel attention MLP more capacity.
-   **Solution 2**: Ensure you are using sufficient data augmentation. Attention mechanisms can sometimes overfit to spurious correlations in the training data.

### Frequently Asked Questions

**Q: What is the difference between channel and spatial attention?**

A: **Channel attention** focuses on "what" is important. It assigns a different weight to each feature map (channel) to emphasize the most informative ones. **Spatial attention** focuses on "where" is important. It assigns a different weight to each pixel location to highlight the most relevant regions in the feature map.

**Q: Is the CBAM block computationally expensive?**

A: No, it is designed to be very lightweight. The channel attention MLP has very few parameters due to its bottleneck design, and the spatial attention uses a single convolution. The overhead compared to a standard ResNet block is minimal.

**Q: Why is attention applied sequentially (channel then spatial)?**

A: The authors of the CBAM paper found empirically that applying channel attention first, followed by spatial attention, yielded better results than applying them in parallel or in the reverse order. The intuition is that the network first decides which features are most important, and then, based on those refined features, it decides where to focus its attention.

---

## 14. Technical Details

### Channel Attention (`Mc`)

The channel attention map `Mc` for an input feature map `F` is computed as:

`Mc(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )`

-   `AvgPool(F)` and `MaxPool(F)` produce two different spatial context descriptors.
-   `MLP` is a shared Multi-Layer Perceptron with one hidden layer that models the inter-channel relationships.
-   `σ` is the sigmoid function.

### Spatial Attention (`Ms`)

The spatial attention map `Ms` for a feature map `F` is computed as:

`Ms(F) = σ( f⁷ˣ⁷([AvgPool(F); MaxPool(F)]) )`

-   `AvgPool(F)` and `MaxPool(F)` are now applied along the channel axis to generate two 2D maps.
-   `[;]` denotes concatenation of the two maps.
-   `f⁷ˣ⁷` is a 7x7 convolutional layer that processes the concatenated map to generate the final spatial attention map.
-   `σ` is the sigmoid function.

---

## 15. Citation

This implementation is based on the following paper. If you use this model in your research, please cite the original work:

-   **CBAM: Convolutional Block Attention Module**
    ```bibtex
    @inproceedings{woo2018cbam,
      title={Cbam: Convolutional block attention module},
      author={Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},
      booktitle={Proceedings of the European conference on computer vision (ECCV)},
      pages={3--19},
      year={2018}
    }
    ```
# ACC-UNet: A Completely Convolutional UNet for the 2020s

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured Keras 3 implementation of **ACC-UNet**, based on the paper ["ACC-UNet: A Completely Convolutional UNet model for the 2020s"](https://arxiv.org/abs/2308.13680) by Ibtehaz & Kihara (MICCAI 2023).

The architecture integrates several key innovations, including **HANC Blocks** for transformer-like context modeling, **MLFC Layers** for advanced feature fusion in skip connections, and **ResPath** for bridging the semantic gap, all within a purely convolutional framework.

---

## Table of Contents

1. [Overview: What is ACC-UNet and Why It Matters](#1-overview-what-is-acc-unet-and-why-it-matters)
2. [The Problem ACC-UNet Solves](#2-the-problem-acc-unet-solves)
3. [How ACC-UNet Works: Core Concepts](#3-how-acc-unet-works-core-concepts)
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

## 1. Overview: What is ACC-UNet and Why It Matters

### What is an ACC-UNet?

**ACC-UNet** is an advanced, purely convolutional U-Net architecture designed for state-of-the-art image segmentation. It strategically integrates design principles from modern networks like Vision Transformers while retaining the efficiency and robustness of CNNs. It achieves superior performance with significantly fewer parameters than its transformer-based counterparts.

### Key Innovations

1.  **HANC Blocks (Hierarchical Aggregation of Neighborhood Context)**: Replaces standard convolution blocks. It captures long-range dependencies, similar to a transformer's self-attention, by aggregating multi-scale neighborhood information through efficient pooling operations.
2.  **MLFC Layers (Multi-Level Feature Compilation)**: A powerful mechanism in the skip connections that allows feature maps at each encoder level to be enriched with information from *all* other levels, effectively bridging the semantic gap.
3.  **ResPath Layers**: Further enhances skip connections by using a series of residual blocks to refine encoder features before they are passed to the decoder, ensuring a smoother flow of semantic information.
4.  **Efficient Design**: Utilizes inverted bottlenecks, depthwise separable convolutions, and Squeeze-and-Excitation attention to maximize performance while minimizing parameter count.

### Why ACC-UNet Matters

**Traditional Segmentation Models**:
```
Model: Standard U-Net
  1. Relies on stacked 3x3 convolutions.
  2. Receptive field grows slowly, making it hard to capture global context.
  3. Simple skip connections can cause a "semantic gap" between shallow,
     detailed encoder features and deep, abstract decoder features.

Model: Transformer-based U-Net (e.g., Swin-UNet)
  1. Uses self-attention to capture global context effectively.
  2. Suffers from quadratic complexity, making it slow for high-res images.
  3. Often has a very high parameter count and requires large datasets to train.
```

**ACC-UNet's Solution**:
```
ACC-UNet Approach:
  1. HANC blocks provide global context with linear complexity, mimicking
     self-attention using multi-scale pooling.
  2. ResPath + MLFC layers create highly effective "smart" skip connections
     that fuse features across all semantic levels.
  3. Result: Achieves transformer-level performance with CNN-level efficiency.
     For example, it outperforms Swin-UNet with ~60% fewer parameters.
```

### Real-World Impact

ACC-UNet is designed for high-stakes segmentation tasks where both accuracy and efficiency are critical:

-   🔬 **Medical Imaging**: Precise tumor segmentation, cell instance counting, and organ delineation from MRI, CT, and histology scans.
-   🛰️ **Satellite Imagery**: Land cover classification, road extraction, and building footprint segmentation.
-   🤖 **Autonomous Driving**: Semantic segmentation of road scenes for navigation.
-   🌾 **Agriculture**: Crop and weed segmentation for precision farming.

---

## 2. The Problem ACC-UNet Solves

### The Limitations of Existing Architectures

Modern segmentation models face a fundamental trade-off between local feature extraction and global context modeling.

```
┌─────────────────────────────────────────────────────────────┐
│  Standard U-Net (Convolutional)                             │
│                                                             │
│  Problem 1: Limited Receptive Field                         │
│    Small 3x3 kernels struggle to see the "big picture." A   │
│    pixel representing a large organ might not have context  │
│    from the other side of the organ.                        │
│                                                             │
│  Problem 2: Semantic Gap                                    │
│    Encoder features (rich in detail, poor in context) are   │
│    naively concatenated with decoder features (poor in      │
│    detail, rich in context), causing a mismatch.            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Transformer-based U-Nets (Attention)                       │
│                                                             │
│  Problem 1: Quadratic Complexity (O(N²))                    │
│    Self-attention compares every pixel to every other pixel,│
│    becoming computationally prohibitive for large images.   │
│                                                             │
│  Problem 2: Data & Parameter Inefficiency                   │
│    Transformers lack the inductive bias of CNNs, often      │
│    requiring huge datasets and having massive parameter     │
│    counts (e.g., Swin-UNet ~40M+ params).                   │
└─────────────────────────────────────────────────────────────┘
```

*A standard U-Net (left) struggles with global context. A Transformer-based model (center) is powerful but inefficient. ACC-UNet (right) combines the best of both worlds.*

### How ACC-UNet Changes the Game

ACC-UNet addresses these limitations with a synergistic combination of novel components:

```
┌─────────────────────────────────────────────────────────────┐
│  ACC-UNet's Solutions                                       │
│                                                             │
│  1. Solving the Receptive Field Problem:                    │
│     The HANC block's multi-scale pooling (2x2, 4x4, 8x8...) │
│     gives each pixel a summary of its wider neighborhood,   │
│     approximating global attention with linear complexity.  │
│                                                             │
│  2. Solving the Semantic Gap Problem:                       │
│     - ResPath refines encoder features with a deep residual │
│       path.                                                 │
│     - MLFC then compiles information from ALL encoder levels│
│       into each skip connection, ensuring features are both │
│       detailed and context-aware before reaching the decoder│
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How ACC-UNet Works: Core Concepts

### The High-Level Architecture

ACC-UNet follows the classic U-Net encoder-decoder structure but revolutionizes each component.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                               ACC-UNet Architecture                           │
│                                                                               │
│  Input Image ───────►┌──────────────────┐                                     │
│                      │      ENCODER     │                                     │
│                      │ (5 Levels of     │                                     │
│                      │  HANC Blocks)    │                                     │
│                      └───────┬───┬───┬──┘                                     │
│                              │   │   │                                        │
│                              │   │   └────────────┐                           │
│                              │   └─────────────┐  │                           │
│                              └──────────────┐  │  │                           │
│                                             ▼  ▼  ▼                           │
│                      ┌─────────────────────────────────┐                      │
│                      │  SKIP CONNECTION PROCESSING     │                      │
│                      │  1. ResPath (Refinement)        │                      │
│                      │  2. MLFC (Cross-Level Fusion)   │                      │
│                      └─────────────────────────────────┘                      │
│                                             │  │  │                           │
│                                             │  │  └─────────┐                 │
│                                             │  └──────────┐ │                 │
│                                             └───────────┐ │ │                 │
│  Segmentation Map ◄──┌──────────────────┐               │ │ │                 │
│                      │      DECODER     │◄──────────────┼─┼─┼─── Bottleneck   │
│                      │ (4 Levels of     │               │ │ │                 │
│                      │  HANC Blocks)    │               │ │ │                 │
│                      └──────────────────┘               ▲ ▲ ▲                 │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ACC-UNet Complete Data Flow                         │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: ENCODER PATH
────────────────────
Input Image (H, W, C)
    │
    ├─► HANC Blocks (Level 0) → Features_0 (H, W, 32)
    ├─► MaxPool
    ├─► HANC Blocks (Level 1) → Features_1 (H/2, W/2, 64)
    ├─► MaxPool
    ├─► HANC Blocks (Level 2) → Features_2 (H/4, W/4, 128)
    ├─► MaxPool
    ├─► HANC Blocks (Level 3) → Features_3 (H/8, W/8, 256)
    ├─► MaxPool
    └─► HANC Blocks (Level 4) → Bottleneck (H/16, W/16, 512)


STEP 2: SKIP CONNECTION PROCESSING
──────────────────────────────────
Encoder Features [Features_0, Features_1, Features_2, Features_3]
    │
    ├─► Pass each through its own ResPath layer for refinement.
    │       └─► Refined Features
    │
    ├─► Pass all Refined Features into the MLFC Layer.
    │   (This is repeated for `mlfc_iterations`)
    │
    └─► Final Skip Features (semantically enriched)


STEP 3: DECODER PATH
────────────────────
Bottleneck Features
    │
    ├─► Upsample (Conv2DTranspose)
    ├─► Concatenate with Final_Skip_Features_3
    ├─► HANC Blocks (Decoder Level 0)
    │
    ├─► Upsample
    ├─► Concatenate with Final_Skip_Features_2
    ├─► HANC Blocks (Decoder Level 1)
    │
    ├─► ... (repeat for all levels) ...
    │
    └─► Final Decoder Output


STEP 4: OUTPUT
──────────────
Final Decoder Output
    │
    ├─► 1x1 Convolution to map to `num_classes`.
    ├─► Sigmoid / Softmax Activation
    │
    └─► Segmentation Map (H, W, num_classes)
```

---

## 4. Architecture Deep Dive

### 4.1 HANC Block

This is the core building block, replacing standard convolutions. It models long-range dependencies efficiently.

```
┌──────────────────────────────────────────────────────────────┐
│                        HANC Block (Internal)                 │
└──────────────────────────────────────────────────────────────┘
Input: (H, W, C_in)
  │
  ▼
┌──────────────────────────┐  ← Inverted Bottleneck
│   1x1 Conv (Expand)      │
│   + BatchNorm + LeakyReLU│
└──────────────────────────┘
  │
  ▼
┌──────────────────────────┐  ← Efficient Spatial Processing
│   3x3 Depthwise Conv     │
│   + BatchNorm + LeakyReLU│
└──────────────────────────┘
  │
  ▼
┌──────────────────────────┐  ← The "Attention" Mechanism
│        HANCLayer         │
│ (Multi-Scale Pooling &   │
│  Aggregation via 1x1 Conv)
└──────────────────────────┘
  │
  ▼
┌──────────────────────────┐  ← Optional Residual Connection
│   Add Input `+`          │  (if C_in == C_out)
│   + BatchNorm            │
└──────────────────────────┘
  │
  ▼
┌──────────────────────────┐  ← Final Projection
│   1x1 Conv (Project)     │
│   + BatchNorm + LeakyReLU│
└──────────────────────────┘
  │
  ▼
┌──────────────────────────┐  ← Channel Recalibration
│  Squeeze-and-Excitation  │
└──────────────────────────┘
  │
  ▼
Output: (H, W, C_out)
```

### 4.2 MLFC Layer (Multi-Level Feature Compilation)

The key to solving the semantic gap. It allows every skip connection to "see" features from all other levels.

```
┌───────────────────────────────────────────────────────────────────────┐
│                            MLFC Layer (Internal)                      │
│          For a single target level (e.g., Level 1: H/2, W/2)          │
└───────────────────────────────────────────────────────────────────────┘
Input: [Feat_0, Feat_1, Feat_2, Feat_3]
  │
  ├─────────┬──────────┬──────────┬───────────┐
  │         │          │          │           │
  ▼         ▼          ▼          ▼           ▼
Resize_0  Identity   Resize_2   Resize_3
(Downsample)         (Upsample) (Upsample)
  │         │          │          │
  └─────────┼──────────┼──────────┼───────────┘
            │          │          │
            ▼          ▼          ▼
┌───────────────────────────────────────────────────────┐
│ Concatenate [Resized_0, Feat_1, Resized_2, Resized_3] │ along channels
└───────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────┐
│ 1x1 Conv (Compile) → (H/2, W/2, C_1)                │
└─────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────┐
│ Concatenate [Compiled_Features, Original_Feat_1]    │
└─────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────┐
│ 1x1 Conv (Merge) + Residual Connection              │
└─────────────────────────────────────────────────────┘
            │
            ▼
Output for Level 1 (to be used in next iteration or decoder)
```

### 4.3 ResPath Layer

A pre-processing step for skip connections that deepens the feature representation.

```
┌───────────────────────────────┐
│     ResPath Layer (Internal)  │
└───────────────────────────────┘
Input Feature Map
  │
  ▼
┌──────────────────────────┐   ┐
│      Residual Block      │   │
│ 1. 3x3 Conv + BN         │   │  Repeated `num_blocks`
│ 2. Squeeze-Excitation    │   │  times (e.g., 4, 3, 2, 1)
│ 3. Add Input `+`         │   │
└──────────────────────────┘   ┘
  │
  ▼
Refined Feature Map
```

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy matplotlib
```

### Your First Segmentation Model (30 seconds)

Let's solve a simple binary segmentation task: finding circles in an image.

```python
import keras
import numpy as np
import matplotlib.pyplot as plt

# Local imports from your project structure
# (Assuming the files are in the specified dl_techniques structure)
from dl_techniques.models.accunet import create_acc_unet_binary

# 1. Generate dummy segmentation data
def create_dummy_data(num_samples, height, width):
    images = np.zeros((num_samples, height, width, 1), dtype=np.float32)
    masks = np.zeros((num_samples, height, width, 1), dtype=np.float32)
    for i in range(num_samples):
        x, y = np.random.randint(20, width-20), np.random.randint(20, height-20)
        r = np.random.randint(5, 20)
        rr, cc = np.ogrid[:height, :width]
        circle = (rr - y)**2 + (cc - x)**2 < r**2
        images[i, circle] = 1.0
        masks[i, circle] = 1.0
        # Add some noise
        images[i] += np.random.normal(0, 0.1, (height, width, 1))
    return np.clip(images, 0, 1), masks

X_train, y_train = create_dummy_data(100, 128, 128)
X_val, y_val = create_dummy_data(20, 128, 128)

# 2. Create an ACC-UNet model for binary segmentation
model = create_acc_unet_binary(
    input_channels=1,
    input_shape=(128, 128)
)
model.summary()

# 3. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("✅ Model created and compiled successfully!")

# 4. Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=8)
print("✅ Training Complete!")

# 5. Predict on a validation sample
predicted_mask = model.predict(X_val[0:1])[0]

# 6. Visualize the results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(X_val[0, :, :, 0], cmap='gray')
plt.title('Input Image')
plt.subplot(1, 3, 2)
plt.imshow(y_val[0, :, :, 0], cmap='gray')
plt.title('Ground Truth Mask')
plt.subplot(1, 3, 3)
plt.imshow(predicted_mask[:, :, 0] > 0.5, cmap='gray') # Threshold prediction
plt.title('Predicted Mask')
plt.show()
```

---

## 6. Component Reference

### 6.1 `AccUNet` (Model Class)

**Purpose**: The main Keras `Model` subclass that assembles the entire ACC-UNet architecture. Usually instantiated via the factory functions below.

**Location**: `dl_techniques.models.accunet.AccUNet`

### 6.2 Factory Functions

These are the recommended way to create ACC-UNet models.

**Location**: `dl_techniques.models.accunet`

#### `create_acc_unet(...)`
The general-purpose factory function.
```python
model = create_acc_unet(
    input_channels=3,
    num_classes=5,
    input_shape=(256, 256),
    base_filters=32,
    mlfc_iterations=3
)
```

#### `create_acc_unet_binary(...)`
A convenient wrapper for binary segmentation (`num_classes=1`, `sigmoid` activation).
```python
model = create_acc_unet_binary(input_channels=1, input_shape=(512, 512))
```

#### `create_acc_unet_multiclass(...)`
A wrapper for multi-class segmentation (`num_classes > 1`, `softmax` activation).
```python
model = create_acc_unet_multiclass(input_channels=3, num_classes=8, input_shape=(224, 224))
```

**Key Parameters**:

| Parameter | Description |
| :--- | :--- |
| `input_channels` | Number of channels in the input image (e.g., 1 for grayscale, 3 for RGB). |
| `num_classes` | Number of segmentation classes. `1` for binary, `>1` for multi-class. |
| `input_shape` | Optional `(height, width)` of input images. `None` for dynamic shapes. |
| `base_filters` | Number of filters in the first encoder level. Defaults to `32`. |
| `mlfc_iterations`| Number of fusion iterations in the MLFC layers. Defaults to `3`. |
| `kernel_regularizer` | Optional Keras regularizer for convolutional layers. |

### 6.3 Core Layers

These are the custom `Layer` subclasses that form the building blocks of ACC-UNet.

| Layer | Location | Purpose |
| :--- | :--- | :--- |
| **`HANCBlock`** | `...layers.hanc_block.HANCBlock` | The main conv block with context aggregation. |
| **`HANCLayer`** | `...layers.hanc_layer.HANCLayer` | Internal layer for multi-scale pooling attention. |
| **`MLFCLayer`** | `...layers.multi_level_feature...` | Fuses features from all encoder levels. |
| **`ResPath`** | `...layers.res_path.ResPath` | Refines skip connection features with residual blocks. |
| **`SqueezeExcitation`** | `...layers.squeeze_excitation.SqueezeExcitation` | Channel attention used throughout the network. |

---

## 7. Configuration & Model Variants

### Input Size Contract

ACC-UNet performs **4 stride-2 downsampling stages** in the encoder followed by
**4 stride-2 transposed-convolution upsampling stages** in the decoder. Because
`Conv2DTranspose(strides=2, padding='same')` always emits exactly `2 * H_in`,
the model can only round-trip spatial dimensions when **`H` and `W` are
divisible by 16**. Any other input shape is rejected with a clear `ValueError`
at the start of the forward pass — **resize your inputs to a multiple of 16
before feeding the model** (the trainer in `src/train/accunet/` does this for
you).

Verified parameter counts (`base_filters=32, num_classes=1, input=224x224`):
~16.8 M trainable parameters (call `model.count_params()` to check).

### Choosing `base_filters`

This parameter controls the overall capacity of the model. The number of filters at each encoder level will be `[base_filters, base_filters*2, ..., base_filters*16]`.

| `base_filters` | Model Size (approx.) | Use Case |
| :---: | :---: | :--- |
| **16** | ~4.5 M params | Lightweight tasks, fast inference. |
| **32 (Default)** | ~16.8 M params | Good balance for most medical/natural image tasks. |
| **64** | ~66.5 M params | Very complex tasks, large datasets, high-resolution images. |

### Choosing `mlfc_iterations`

This controls how many times the cross-level feature fusion is applied.

| `mlfc_iterations` | Effect | Trade-off |
| :---: | :--- | :--- |
| **1** | Basic cross-level feature mixing. | Fastest, but may not fully bridge the semantic gap. |
| **3 (Default)** | Strong, iterative feature refinement. Recommended by the paper. | Excellent performance-to-computation ratio. |
| **4+** | Very deep feature compilation. | Marginal gains in accuracy for a significant increase in computation. |

---

## 8. Comprehensive Usage Examples

### Example 1: Binary Segmentation on a Real-World Dataset

```python
# Assume you have a dataset loader for a medical imaging task
# train_dataset: yields (image, mask) tuples of shape (256, 256, 1)
# val_dataset: yields (image, mask) tuples of shape (256, 256, 1)

from dl_techniques.models.accunet import create_acc_unet_binary
# A common loss for segmentation is a combination of BCE and Dice
import tensorflow as tf

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# 1. Create the model
model = create_acc_unet_binary(
    input_channels=1,
    input_shape=(256, 256),
    base_filters=32
)

# 2. Compile with a robust segmentation loss and metric
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss=bce_dice_loss,
    metrics=['binary_accuracy', dice_loss] # Monitor Dice as a metric
)

# 3. Train the model
# model.fit(train_dataset, validation_data=val_dataset, epochs=100)
```

### Example 2: Multi-Class Segmentation with Dynamic Input Size

ACC-UNet can handle variable input sizes if `input_shape` is set to `None`.

```python
from dl_techniques.models.accunet import create_acc_unet_multiclass

# 1. Create model for 5 classes, dynamic HxW
model = create_acc_unet_multiclass(
    input_channels=3, # RGB input
    num_classes=5,
    input_shape=None
)

# 2. Compile for multi-class segmentation
# Use SparseCategoricalCrossentropy if masks are (H, W, 1) with integer class IDs
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

# 3. Train with images of different sizes
# This requires a data pipeline that can handle batches of non-uniform images,
# or training with batch_size=1.
# dummy_image_1 = np.random.rand(1, 224, 224, 3)
# dummy_mask_1 = np.random.randint(0, 5, (1, 224, 224, 1))
# dummy_image_2 = np.random.rand(1, 256, 320, 3)
# dummy_mask_2 = np.random.randint(0, 5, (1, 256, 320, 1))

# model.train_on_batch(dummy_image_1, dummy_mask_1)
# model.train_on_batch(dummy_image_2, dummy_mask_2)
print("Model with dynamic shape created and compiled.")
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Using HANC Blocks in a Custom Architecture

The `HANCBlock` is a powerful, self-contained layer that can replace `Conv2D` blocks in any CNN.

```python
from dl_techniques.layers.hanc_block import HANCBlock

# Build a simple custom classifier using HANCBlocks
inputs = keras.Input(shape=(224, 224, 3))
x = HANCBlock(filters=32, input_channels=3, k=3)(inputs) # input_channels must be specified
x = keras.layers.MaxPooling2D(2)(x)
x = HANCBlock(filters=64, input_channels=32, k=3)(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(10, activation='softmax')(x)

custom_model = keras.Model(inputs, x)
custom_model.summary()
```

### Pattern 2: Integrating MLFC into another U-Net

You can use the `MLFCLayer` to enhance the skip connections of any U-Net-like model.

```python
from dl_techniques.layers.multi_level_feature_compilation import MLFCLayer
# Assume you have a standard Keras U-Net encoder that returns a list of features
# encoder_features = [level0_feat, level1_feat, level2_feat, level3_feat]
# channels = [64, 128, 256, 512]

# 1. Create an MLFC layer matching the channel dimensions
mlfc = MLFCLayer(channels_list=channels, num_iterations=3)

# 2. Apply it to the encoder features
# processed_skip_features = mlfc(encoder_features)

# 3. Use these processed features in your decoder's skip connections
# decoder_level_0 = Concatenate()([upsampled_bottleneck, processed_skip_features[3]])
```

---

## 10. Performance Optimization

### Mixed Precision Training

Use mixed precision for significant speedups on modern GPUs with minimal effort.

```python
# Enable mixed precision globally before creating the model
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = create_acc_unet_binary(input_channels=1, input_shape=(256, 256))
model.compile(optimizer='adam', loss='binary_crossentropy')

# This can provide:
# - ~1.5-3x training speedup on compatible GPUs (NVIDIA Tensor Cores)
# - ~50% reduction in memory usage, allowing larger batch sizes.
```

### XLA / `jit_compile`

Use XLA (Accelerated Linear Algebra) for an additional 10-30% speedup. With Keras 3
this is a single `compile()` flag — no custom training loop or `tf.GradientTape` is
needed, and the model still trains via `model.fit()`.

```python
model = create_acc_unet_binary(input_channels=1, input_shape=(256, 256))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    jit_compile=True,  # XLA-compile the train/eval step
)
# model.fit(...) as usual
```

---

## 11. Training and Best Practices

### Choosing Loss Functions and Metrics

-   **Binary Segmentation**:
    -   **Loss**: A combination of `BinaryCrossentropy` and `DiceLoss` (or `TverskyLoss`) is often most effective. BCE focuses on pixel-wise accuracy, while Dice focuses on region overlap.
    -   **Metrics**: `BinaryAccuracy`, `DiceCoefficient`, `IoU` (Intersection over Union).
-   **Multi-Class Segmentation**:
    -   **Loss**: `SparseCategoricalCrossentropy` (if masks are integer labels) or `CategoricalCrossentropy` (if masks are one-hot encoded). Can also be combined with a multi-class Dice loss.
    -   **Metrics**: `SparseCategoricalAccuracy`, `MeanIoU`.

### Data Augmentation

Data augmentation is crucial for training robust segmentation models. Use a library like `albumentations` or Keras's preprocessing layers for:
-   **Geometric**: Random flips, rotations, scaling, cropping.
-   **Photometric**: Brightness, contrast, color jitter.
-   **Elastic Deformations**: Simulates tissue deformation in medical images.

---

## 12. Serialization & Deployment

The `AccUNet` model and all its custom layers are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
# Create and train model
model = create_acc_unet_binary(input_channels=1, input_shape=(128, 128))
# model.compile(...) and model.fit(...)

# Save the entire model to a single file
model.save('acc_unet_model.keras')
print("Model saved to acc_unet_model.keras")

# Load the model in a new session (custom layers are automatically registered)
loaded_model = keras.models.load_model('acc_unet_model.keras')
print("Model loaded successfully")

# Verify
dummy_input = np.random.rand(1, 128, 128, 1)
prediction = loaded_model.predict(dummy_input)
print(f"Prediction shape with loaded model: {prediction.shape}")
```

### Deployment

The saved model is a standard Keras model and can be deployed using TensorFlow Serving, converted to ONNX for use with other runtimes, or converted to TFLite for on-device inference.

---

## 13. Testing & Validation

### Unit Tests

```python
import keras
import numpy as np

# Assuming imports for create_acc_unet_binary are set up
from dl_techniques.models.accunet import create_acc_unet_binary

def test_model_creation():
    """Test that the model can be created."""
    model = create_acc_unet_binary(input_channels=1, input_shape=(64, 64))
    assert model is not None
    print("✓ Model created successfully")

def test_forward_pass_shape():
    """Test the output shape of a forward pass."""
    model = create_acc_unet_binary(input_channels=3, input_shape=(96, 96), base_filters=16)
    dummy_input = np.random.rand(2, 96, 96, 3)
    output = model.predict(dummy_input)
    expected_shape = (2, 96, 96, 1) # (batch, H, W, num_classes)
    assert output.shape == expected_shape
    print("✓ Forward pass has correct shape")

def test_dynamic_shape():
    """Test forward pass with dynamic input shapes."""
    model = create_acc_unet_binary(input_channels=1, input_shape=None)
    output1 = model.predict(np.random.rand(1, 64, 64, 1))
    output2 = model.predict(np.random.rand(1, 96, 128, 1))
    assert output1.shape == (1, 64, 64, 1)
    assert output2.shape == (1, 96, 128, 1)
    print("✓ Dynamic shapes handled correctly")

def test_serialization():
    """Test model save/load."""
    model = create_acc_unet_binary(input_channels=1, input_shape=(32, 32))
    model.save('test_accunet.keras')
    loaded_model = keras.models.load_model('test_accunet.keras')
    config1 = model.get_config()
    config2 = loaded_model.get_config() # Note: Factory function model configs differ from class configs
    assert loaded_model is not None
    print("✓ Serialization successful")

# Run tests
if __name__ == '__main__':
    test_model_creation()
    test_forward_pass_shape()
    test_dynamic_shape()
    test_serialization()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Model outputs are all black or all white (or random noise).**

-   **Cause 1**: Learning rate is too high/low. Start with `1e-4` and adjust. Use a learning rate scheduler.
-   **Cause 2**: Data is not normalized correctly. Ensure input images are scaled appropriately (e.g., `[0, 1]` or `[-1, 1]`).
-   **Cause 3**: Incorrect loss function. Make sure you're using `binary_crossentropy` for binary tasks and `(sparse_)categorical_crossentropy` for multi-class.

**Issue 2: `ValueError: Input channels mismatch` in a `HANCBlock`.**

-   **Cause**: The `input_channels` argument for a custom `HANCBlock` was not set correctly to match the channel count of the preceding layer. The `AccUNet` model handles this automatically, but it's a common issue in custom architectures.

### Frequently Asked Questions

**Q: How does this compare to U-Net++ or Attention U-Net?**

A:
-   **U-Net++**: Improves on U-Net with nested, dense skip connections. It focuses on multi-scale feature fusion but in a different way than ACC-UNet's MLFC.
-   **Attention U-Net**: Adds attention gates to skip connections to suppress irrelevant regions. This is a form of spatial attention, whereas ACC-UNet's SE blocks provide channel attention and its HANC blocks provide a proxy for global self-attention. ACC-UNet's approach is generally more comprehensive.

**Q: Can I use this for 3D segmentation?**

A: This implementation is strictly for 2D images. However, the core concepts (HANC, MLFC, ResPath) could be extended to 3D by replacing all 2D layers (`Conv2D`, `MaxPooling2D`, etc.) with their 3D counterparts. This would require a non-trivial modification of the source code.

**Q: Why is it called "Completely Convolutional"?**

A: Because despite incorporating ideas from Transformers (like global context modeling), it achieves this using only standard convolutional and pooling operations. There are no actual self-attention, patch embedding, or multi-head attention layers, making it highly efficient to run on standard GPUs.

---

## 15. Technical Details

### HANC Block `k` Values

The parameter `k` in the `HANCBlock` determines the number of hierarchical pooling levels used for context aggregation. The paper specifies different `k` values for different depths of the U-Net to balance receptive field size with feature map resolution:
-   **Encoder Levels 0, 1, 2**: `k=3` (pools with 2x2, 4x4 patches)
-   **Encoder Level 3**: `k=2` (pools with 2x2 patches)
-   **Encoder Level 4 (Bottleneck)**: `k=1` (no pooling, standard conv behavior)
-   **Decoder**: `k` values mirror the encoder (`k=2` for deeper levels, `k=3` for shallower).

This implementation hard-codes these values as they are integral to the published architecture's design.

### MLFC Iterative Refinement

The `MLFCLayer` is applied `mlfc_iterations` times. The output of one iteration becomes the input for the next. This creates a powerful feedback loop:
-   **Iteration 1**: Each level gets an initial "taste" of context from all other levels.
-   **Iteration 2**: The features, now already enriched from Iteration 1, are fused again. This allows second-order interactions (e.g., Level 0 features can influence Level 3, which in turn influences Level 1).
-   **Iteration 3**: The process deepens, leading to highly refined and context-aware feature maps that have been cross-pollinated multiple times.

---

## 16. Citation

If you use this model in your research, please cite the original paper:

```bibtex
@inproceedings{ibtehaz2023acc,
  title={ACC-UNet: A Completely Convolutional UNet model for the 2020s},
  author={Ibtehaz, Nabil and Kihara, Daisuke},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={1--11},
  year={2023},
  organization={Springer}
}
```
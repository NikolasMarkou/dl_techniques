# ConvNeXt Models: A Comparative Overview

This repository contains Keras 3 implementations of the ConvNeXt V1 and V2 architectures. ConvNeXt models are pure convolutional networks (ConvNets) that were modernized to compete with and often outperform Vision Transformers (ViTs) by progressively incorporating architectural decisions from ViTs into a standard ResNet.

This document provides a summary of each model's key features, architectural differences, and usage examples.

## Summary of Differences and Evolution

The primary evolution from ConvNeXt V1 to V2 is the introduction of Global Response Normalization (GRN), a simple layer that improves feature competition and model performance, especially in the context of self-supervised learning with masked autoencoders.

| Model | Key Innovation | Description |
| --- | --- | --- |
| **ConvNeXt V1** | Modernized ConvNet Architecture | Redesigned a standard ResNet by adopting principles from Vision Transformers, such as a patchify stem, larger kernels (7x7), an inverted bottleneck structure, and fewer activation/normalization layers. |
| **ConvNeXt V2** | Global Response Normalization (GRN) | Introduced a new normalization layer (GRN) that is added to the ConvNeXt block. GRN encourages channel-wise feature competition, enhancing the model's representational capabilities. |

## How to Use

This section provides practical examples for creating and customizing the ConvNeXt models. The convenience functions (`create_convnext_v1`, `create_convnext_v2`) are the recommended way to instantiate models. The interface for both V1 and V2 is identical.

### 1. Create a Standard Model for ImageNet

Instantiate a ConvNeXt-Base model with its default configuration for ImageNet (1000 classes, 224x224 input).

```python
from convnext_v1 import create_convnext_v1

# Create a ConvNeXtV1-Base model for ImageNet classification
# The default input_shape is suitable for ImageNet if not specified
model = create_convnext_v1(
    variant="base",
    num_classes=1000
)

model.summary()
```

### 2. Create a Model for a Custom Dataset (e.g., CIFAR-10)

To adapt the model for a dataset with smaller images, you must specify the `input_shape`. The implementations will automatically adjust the stem and downsampling layers to prevent over-downsampling.

```python
from convnext_v2 import create_convnext_v2

# Create a ConvNeXtV2-Tiny for CIFAR-10 (10 classes, 32x32 input)
cifar_model = create_convnext_v2(
    variant="tiny",
    num_classes=10,
    input_shape=(32, 32, 3)
)

cifar_model.summary()
```

### 3. Create a Micro-Variant Model (ConvNeXt V2)

ConvNeXt V2 introduced several smaller variants that are highly efficient for mobile or resource-constrained applications.

```python
from convnext_v2 import create_convnext_v2

# Create a ConvNeXtV2-Pico model for a 100-class problem
pico_model = create_convnext_v2(
    variant="pico",
    num_classes=100,
    input_shape=(96, 96, 3)
)

print(f"Pico model params: {pico_model.count_params():,}")
```

### 4. Using the Model as a Feature Extractor

For tasks like transfer learning, object detection, or segmentation, you can create the model without the top classification layer by setting `include_top=False`.

```python
from convnext_v1 import create_convnext_v1

# Create a ConvNeXtV1-Large base for feature extraction
feature_extractor = create_convnext_v1(
    variant="large",
    include_top=False, # This is the key argument
    input_shape=(256, 256, 3)
)

# The output will be a feature map
# For a (256, 256, 3) input, the output shape might be (None, 8, 8, 1536)
feature_extractor.summary()
```

### 5. Compiling and Training a Model

All models are standard `keras.Model` instances and can be compiled and trained using the standard Keras workflow.

```python
import tensorflow as tf # or import jax, torch
from convnext_v2 import create_convnext_v2

# 1. Create the model
model = create_convnext_v2(
    variant="atto",
    num_classes=10,
    input_shape=(48, 48, 3)
)

# 2. Compile the model
model.compile(
    optimizer='adamw', # AdamW is often recommended for ConvNeXt
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Prepare dummy data
dummy_images = tf.random.normal((16, 48, 48, 3))
dummy_labels = tf.one_hot(tf.zeros(16, dtype=tf.int32), depth=10)

# 4. Train the model
model.fit(dummy_images, dummy_labels, epochs=3)
```

---

## ConvNeXt V1

ConvNeXt V1 was designed by "modernizing" a standard ResNet to align with the architectural patterns of Vision Transformers, demonstrating that pure ConvNets can achieve state-of-the-art performance.

### Key Features
- **Patchify Stem**: Replaces the initial `7x7` convolution with a non-overlapping `4x4` convolution, similar to patch embedding in ViTs.
- **Inverted Bottleneck Structure**: Utilizes a "depthwise-separable" block design similar to MobileNetV2, but with an expansion ratio of 4.
- **Large Kernel Sizes**: Employs large `7x7` depthwise convolution kernels, which increases the receptive field.
- **Layer Normalization**: Replaces Batch Normalization with Layer Normalization for improved performance.
- **Fewer Activations/Normalizations**: Removes normalization and activation layers between several blocks to streamline the architecture.
- **Stochastic Depth (Drop Path)**: A regularization technique that randomly drops entire residual blocks during training.

### Reference
- **Paper**: "A ConvNet for the 2020s"
- **arXiv**: [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)

### Variants
- **`tiny`**: The smallest standard variant.
- **`small`**
- **`base`**
- **`large`**
- **`xlarge`**: The largest standard variant.

---

## ConvNeXt V2

ConvNeXt V2 improves upon V1 by introducing a novel feature normalization layer and was co-designed with a self-supervised pre-training method (masked autoencoders), leading to significant gains.

### Key Features
- **Global Response Normalization (GRN)**: The core innovation. GRN is added to the ConvNeXt block and operates by first aggregating feature maps across the spatial dimensions, then computing a normalization score for each channel, and finally using these scores to recalibrate the channel features. This encourages feature diversity and competition.
- **Improved Performance**: Achieves better results than ConvNeXt V1 in supervised training and sets new state-of-the-art records for various tasks when pre-trained using a fully convolutional masked autoencoder (FCMAE) approach.
- **Micro Variants**: Introduced a new set of smaller, highly efficient models (`Atto`, `Femto`, `Pico`, `Nano`) that provide an excellent accuracy-vs-latency trade-off on mobile devices.

### Reference
- **Paper**: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
- **arXiv**: [https://arxiv.org/abs/2301.00808](https://arxiv.org/abs/2301.00808)

### Variants
- **Micro Variants**:
  - `atto`
  - `femto`
  - `pico`
  - `nano`
- **Standard Variants**:
  - `tiny`
  - `base`
  - `large`
  - `huge`
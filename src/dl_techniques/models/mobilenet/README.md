# MobileNet Models: A Comparative Overview

This repository contains Keras 3 implementations of the MobileNet family of models, from V1 to V4. These models are designed for efficiency and are particularly well-suited for on-device and mobile vision applications. This document provides a summary of each model's key features, architectural differences, and references to the original papers.

## Summary of Differences and Evolution

The MobileNet family has evolved significantly, with each version introducing new techniques to improve the accuracy-efficiency trade-off.

| Model | Key Innovation | Description |
| --- | --- | --- |
| **MobileNetV1** | Depthwise Separable Convolutions | Drastically reduced computation by factorizing standard convolutions into a depthwise and a pointwise convolution. |
| **MobileNetV2** | Inverted Residuals & Linear Bottlenecks | Introduced blocks that first expand and then project feature maps, with residual connections between the narrow "bottleneck" layers. This improved feature reuse and gradient flow. |
| **MobileNetV3** | Hardware-Aware NAS, Squeeze-and-Excite, Hard-Swish | Utilized Neural Architecture Search (NAS) to find an optimal architecture. Added lightweight attention (Squeeze-and-Excite) and a more efficient non-linearity (hard-swish). |
| **MobileNetV4** | Universal Inverted Bottleneck (UIB) & Mobile MQA | Introduced a flexible "Universal" block that can represent different block styles (including ConvNeXt-like structures). Added an optional mobile-friendly Multi-Query Attention (MQA) module, creating hybrid vision transformer models. |

## How to Use

This section provides practical examples for creating and customizing the MobileNet models. The convenience functions (`create_mobilenetv1`, `create_mobilenetv2`, etc.) are the recommended way to instantiate models.

### 1. Create a Standard Model for ImageNet

Instantiate a MobileNetV4 model with its default configuration for ImageNet (1000 classes, 224x224 input).

```python
from mobilenet_v4 import create_mobilenetv4

# Create a MobileNetV4-ConvMedium model for ImageNet classification
model = create_mobilenetv4(
    variant="conv_medium",
    num_classes=1000,
    input_shape=(224, 224, 3)
)

model.summary()
```

### 2. Create a Model for a Custom Dataset (e.g., CIFAR-10)

To adapt the model for a different dataset, simply change the `num_classes` and `input_shape`.

```python
from mobilenet_v3 import create_mobilenetv3

# Create a MobileNetV3-Small for CIFAR-10 (10 classes, 32x32 input)
cifar_model = create_mobilenetv3(
    variant="small",
    num_classes=10,
    input_shape=(32, 32, 3)
)

cifar_model.summary()
```

### 3. Adjusting Model Size with the Width Multiplier

The `width_multiplier` allows you to scale the number of channels in the model, controlling the trade-off between performance and size. A value `< 1.0` reduces complexity, while a value `> 1.0` increases it.

```python
from mobilenet_v2 import create_mobilenetv2

# Create a smaller MobileNetV2 with 75% of the default channels
small_model = create_mobilenetv2(
    variant="1.0", # Base variant
    width_multiplier=0.75,
    num_classes=1000
)

# Create a larger-than-default MobileNetV2
# Note: MobileNetV2 also has a "1.4" variant for this purpose
large_model = create_mobilenetv2(
    variant="1.0",
    width_multiplier=1.4,
    num_classes=1000
)

print(f"Small model params: {small_model.count_params():,}")
print(f"Large model params: {large_model.count_params():,}")

```

### 4. Using the Model as a Feature Extractor

For tasks like transfer learning, object detection, or segmentation, you can create the model without the top classification layer.

```python
from mobilenet_v4 import create_mobilenetv4

# Create a MobileNetV4-Hybrid-Medium base for feature extraction
feature_extractor = create_mobilenetv4(
    variant="hybrid_medium",
    include_top=False, # This is the key argument
    input_shape=(256, 256, 3)
)

# The output will be a feature map
# For example, for a (256, 256, 3) input, the output shape might be (None, 8, 8, 320)
feature_extractor.summary()
```

### 5. Compiling and Training a Model

All models are standard `keras.Model` instances and can be compiled and trained using the standard Keras workflow.

```python
import tensorflow as tf # or import jax, torch
from mobilenet_v1 import create_mobilenetv1

# 1. Create the model
model = create_mobilenetv1(
    variant="0.5",
    num_classes=10,
    input_shape=(96, 96, 3)
)

# 2. Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Prepare dummy data
dummy_images = tf.random.normal((16, 96, 96, 3))
dummy_labels = tf.one_hot(tf.zeros(16, dtype=tf.int32), depth=10)

# 4. Train the model
model.fit(dummy_images, dummy_labels, epochs=3)
```

---

## MobileNetV1

MobileNetV1 introduced the concept of depthwise separable convolutions as a highly efficient replacement for standard convolutions in deep neural networks.

### Key Features
- **Depthwise Separable Convolutions**: A core building block that significantly reduces the number of parameters and computations.
- **Width and Resolution Multipliers**: Hyperparameters (α and ρ) that allow for easy scaling of the model's size and computational cost to fit different performance constraints.

### Reference
- **Paper**: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
- **arXiv**: [https://arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861)

### Performance
MobileNetV1 demonstrated a significant reduction in computational cost (8-9 times less than standard convolutions) with only a minor decrease in accuracy on ImageNet.

### Variants
The implementation supports multiple variants based on the `width_multiplier` (α):
- `1.0` (default)
- `0.75`
- `0.50`
- `0.25`

---

## MobileNetV2

MobileNetV2 builds upon V1 by introducing the inverted residual block with a linear bottleneck, which improves performance by preserving representational power in narrower layers.

### Key Features
- **Inverted Residuals**: The blocks feature a narrow -> wide -> narrow structure, where the input and output are thin "bottleneck" layers, and the intermediate layer is expanded using lightweight depthwise convolutions.
- **Linear Bottlenecks**: The final projection layer in the inverted residual block is linear (no activation function), which was found to be crucial for preventing the loss of information in low-dimensional spaces.
- **Residual Connections**: Skip connections are present between the bottleneck layers, enhancing gradient flow through the network.

### Reference
- **Paper**: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
- **arXiv**: [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)

### Performance
MobileNetV2 improved the state-of-the-art for mobile models on various tasks, including classification, object detection, and semantic segmentation, while using fewer parameters and computations than MobileNetV1.

### Variants
The implementation supports multiple variants based on the `width_multiplier` (α):
- `1.0` (default)
- `0.75`
- `0.50`
- `0.35`

---

## MobileNetV3

MobileNetV3 was discovered through a combination of hardware-aware Network Architecture Search (NAS) and novel architectural improvements, resulting in models highly optimized for mobile CPUs.

### Key Features
- **Hardware-Aware NAS**: The architecture was optimized using platform-aware NAS to find the best configuration of layers and blocks for a target latency.
- **Squeeze-and-Excite (SE) Modules**: Lightweight attention modules are integrated into the bottleneck blocks to selectively boost informative feature channels.
- **h-swish Activation**: A new, more efficient non-linearity called hard-swish was introduced as a replacement for the standard swish function.
- **Optimized Structure**: The initial and final stages of the network were redesigned to be more efficient without sacrificing accuracy.

### Reference
- **Paper**: "Searching for MobileNetV3"
- **arXiv**: [https://arxiv.org/abs/1905.02244](https://arxiv.org/abs/1905.02244)

### Performance
MobileNetV3-Large was 3.2% more accurate on ImageNet than MobileNetV2 with a 15% reduction in latency. MobileNetV3-Small was 4.6% more accurate than a MobileNetV2 model with comparable latency.

### Variants
The implementation provides two main variants, each scalable with a `width_multiplier`:
- **`large`**: For high-resource use cases.
- **`small`**: For low-resource use cases.

---

## MobileNetV4

MobileNetV4 introduces a flexible and universal building block and an optional attention module, making it a highly adaptable architecture for a wide range of mobile hardware, including CPUs, GPUs, and specialized accelerators like EdgeTPUs.

### Key Features
- **Universal Inverted Bottleneck (UIB)**: A unified block that can represent various structures, including the classic inverted bottleneck, ConvNeXt-style blocks, and a new "Extra Depthwise" variant. This flexibility allows NAS to discover even more efficient configurations.
- **Mobile Multi-Query Attention (MQA)**: A mobile-friendly attention block that can be added to the later stages of the network, creating hybrid convolutional-attention models with improved accuracy.
- **Optimized for Diverse Hardware**: The architecture was designed to be Pareto-optimal across a variety of hardware targets, not just CPUs.

### Reference
- **Paper**: "MobileNetV4: Universal Inverted Bottleneck and Mobile MQA"
- **arXiv**: [https://arxiv.org/abs/2404.10518](https://arxiv.org/abs/2404.10518)

### Performance
MobileNetV4 models are largely Pareto-optimal across a wide range of mobile devices. The flagship MNv4-Hybrid-Large model achieves 87% top-1 accuracy on ImageNet with a latency of just 3.8ms on a Pixel 8 EdgeTPU.

### Variants
The implementation supports a range of both convolution-only and hybrid variants:
- **`conv_small`**: A lightweight, purely convolutional model.
- **`conv_medium`**: A balanced, purely convolutional model.
- **`conv_large`**: A high-capacity, purely convolutional model.
- **`hybrid_medium`**: A medium-sized model with the Mobile MQA attention module.
- **`hybrid_large`**: A large-sized model with the Mobile MQA attention module.
# Vision Transformer (ViT)

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of the **Vision Transformer (ViT)** in **Keras 3**, based on the paper ["An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al. (2020).

The architecture supports multiple standard scales (`Tiny`, `Small`, `Base`, `Large`, `Huge`) and is highly configurable, allowing for easy experimentation with different normalization techniques, feed-forward networks, and attention mechanisms.

---

## Table of Contents

1. [Overview: What is ViT and Why It Matters](#1-overview-what-is-vit-and-why-it-matters)
2. [The Problem ViT Solves](#2-the-problem-vit-solves)
3. [How ViT Works: Core Concepts](#3-how-vit-works-core-concepts)
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

## 1. Overview: What is ViT and Why It Matters

### What is a Vision Transformer (ViT)?

A **Vision Transformer (ViT)** is a model that applies the standard Transformer architecture, originally designed for text, directly to image recognition. It challenges the long-standing dominance of Convolutional Neural Networks (CNNs) in computer vision.

The core idea is to treat an image as a sequence of fixed-size patches, analogous to words in a sentence. Each patch is linearly embedded, combined with position embeddings, and then processed by a standard Transformer encoder.

### Key Innovations

1.  **Sequence-Based Image Processing**: ViT discards the inductive biases of CNNs (like locality and translation equivariance) and instead learns spatial relationships from scratch using self-attention.
2.  **Global Receptive Field**: From the very first layer, the self-attention mechanism allows every patch to interact with every other patch, providing a global receptive field.
3.  **Scalability**: ViT demonstrates exceptional scaling properties. When trained on sufficiently large datasets (e.g., ImageNet-21k, JFT-300M), it can outperform state-of-the-art CNNs.
4.  **Transfer Learning Powerhouse**: Pre-trained ViT models serve as powerful backbones for a wide range of downstream vision tasks, such as object detection, segmentation, and fine-grained classification.

### Why ViT Matters

**Traditional Computer Vision (CNNs)**:
```
Problem: Classify an image of a cat.
CNN Approach:
  1. Use stacked convolutional layers with small kernels (e.g., 3x3).
  2. Each layer processes local neighborhoods, gradually building up a
     larger receptive field.
  3. Limitation: It takes many layers for a neuron to "see" the entire image,
     making it harder to model long-range dependencies between distant pixels.
```

**ViT's Solution**:
```
ViT Approach:
  1. Split the image into a grid of 16x16 patches.
  2. Treat these patches as a sequence of "image words".
  3. Feed this sequence into a Transformer encoder.
  4. The self-attention mechanism in the first layer immediately relates a patch
     in the top-left corner to a patch in the bottom-right.
  5. Benefit: Models global context from the outset, allowing it to learn
     holistic representations of objects and scenes.
```

### Real-World Impact

ViT has revolutionized the field of computer vision and has become the foundation for many state-of-the-art models:

-   🖼️ **Image Classification**: Achieves top performance on benchmarks like ImageNet.
-   🎯 **Object Detection & Segmentation**: ViT backbones (e.g., in DETR, Mask R-CNN) provide powerful features for locating and outlining objects.
-   🔬 **Medical Imaging**: Analyzes whole-slide pathology images by treating large tissue sections as sequences of patches.
-   🎨 **Generative Modeling**: Foundational to models like DALL-E 2 and Imagen that generate images from text.

---

## 2. The Problem ViT Solves

### The Limitations of CNNs

For years, CNNs were the undisputed kings of computer vision, but they have inherent limitations.

```
┌─────────────────────────────────────────────────────────────┐
│  Convolutional Neural Networks (CNNs)                       │
│                                                             │
│  The Inductive Bias of Locality:                            │
│    CNNs are hard-wired to prioritize local information. A    │
│    3x3 kernel only sees its immediate neighbors.             │
│                                                             │
│  This creates challenges for:                               │
│  1. Modeling Long-Range Dependencies: Relating a dog's head │
│     to its tail requires information to propagate through   │
│     many layers.                                            │
│  2. Global Context: Understanding the overall scene context │
│     is difficult in early layers.                           │
│  3. Flexibility: The rigid convolutional structure is less  │
│     adaptable than the dynamic, content-based weighting of  │
│     self-attention.                                         │
└─────────────────────────────────────────────────────────────┘
```

While techniques like larger kernels, dilated convolutions, and attention mechanisms were added to CNNs to mitigate this, they were often ad-hoc solutions.

### How ViT Changes the Game

ViT abandons the CNN paradigm in favor of a more flexible, scalable approach.

```
┌─────────────────────────────────────────────────────────────┐
│  The Vision Transformer Principle                           │
│                                                             │
│  1. Minimal Inductive Bias: ViT does not assume locality.   │
│     It learns all spatial relationships directly from data. │
│                                                             │
│  2. Global Information Mixing:                              │
│     Self-attention allows any two patches in the image, no  │
│     matter how far apart, to directly influence each other's │
│     representation in a single layer.                       │
│                                                             │
│  This allows the model to:                                  │
│  - Learn non-local patterns that are difficult for CNNs.    │
│  - Scale more effectively with massive datasets, as the     │
│    model's capacity to learn is not constrained by a local  │
│    receptive field.                                         │
└─────────────────────────────────────────────────────────────┘
```

The key trade-off is that this flexibility requires a large amount of data to learn meaningful patterns. On small datasets, ViTs tend to underperform CNNs without strong regularization or pre-training.

---

## 3. How ViT Works: Core Concepts

### From Image to Sequence

The first and most critical step is converting a 2D image into a 1D sequence of tokens.

```
┌──────────────────────────────────────────────────────────────────┐
│                         Image-to-Sequence                        │
│                                                                  │
│ ┌───────────────┐  Slice into Grid  ┌───┬───┬───┐                │
│ │               │  ─────────────►   │ 1 │ 2 │ 3 │                │
│ │   Input Image │                   ├───┼───┼───┤                │
│ │ (e.g. 224x224)│                   │ 4 │ 5 │ 6 │                │
│ │               │                   ├───┼───┼───┤                │
│ └───────────────┘                   │ 7 │ 8 │ 9 │                │
│                                     └───┴───┴───┘                │
│                                           │                      │
│                                           ▼                      │
│                            Flatten and Linearly Embed Each Patch │
│                                           │                      │
│                                           ▼                      │
│                  Sequence of Patch Embeddings (e.g., 196 tokens) │
└──────────────────────────────────────────────────────────────────┘
```

### The [CLS] Token and Positional Embeddings

Once the image is a sequence, two more elements are added:

1.  **[CLS] Token**: A special, learnable token is prepended to the sequence. The final representation of this token from the Transformer is used as the aggregate image representation for classification. This is inspired by BERT.
2.  **Positional Embeddings**: Since self-attention is permutation-invariant, the model has no inherent sense of patch order. Learnable positional embeddings are added to each patch embedding to provide this spatial information.

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ViT Complete Data Flow                              │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: Patching and Embedding
──────────────────────────────
Input Image (B, H, W, C)
    │
    ├─► PatchEmbedding Layer → Patch Embeddings (B, num_patches, D)
    │
    ├─► Prepend [CLS] Token → Sequence (B, num_patches + 1, D)
    │
    ├─► Add Positional Embeddings
    │
    └─► Final Input Sequence (B, L, D) ← READY FOR TRANSFORMER


STEP 2: Transformer Encoder Stack (repeated for N layers)
────────────────────────────────────────────────────────
Input Sequence (B, L, D)
    │
    ├─► TransformerLayer 1
    │   ├─► [Norm] -> Multi-Head Self-Attention -> [Add & Norm]
    │   └─► [Norm] -> Feed-Forward Network      -> [Add & Norm]
    │
    ├─► TransformerLayer 2
    │   └─► ...
    │
    ├─► TransformerLayer N
    │
    └─► Final Hidden States (B, L, D)


STEP 3: Classification Head
───────────────────────────
Final Hidden States (B, L, D)
    │
    ├─► Select the [CLS] token's hidden state: (B, D)
    │
    ├─► Final Layer Normalization
    │
    ├─► [Optional] Dropout
    │
    ├─► Dense Layer (Classification)
    │
    └─► Logits (B, num_classes)
```

---

## 4. Architecture Deep Dive

### 4.1 Patch Embedding Layer

This layer performs the critical image-to-sequence conversion. It is implemented as a single `Conv2D` layer.

-   `kernel_size` = `patch_size`
-   `strides` = `patch_size`
-   `filters` = `embed_dim`

This single convolution operation efficiently extracts and embeds all patches in parallel. The output is then reshaped from `(B, H', W', D)` to `(B, H'*W', D)`.

### 4.2 Transformer Layer

This is the standard building block of the Transformer, used repeatedly. This implementation uses a highly configurable `TransformerLayer` class.

```
┌─────────────────────────────────────────────────────────────┐
│                    TransformerLayer (Internal)              │
└─────────────────────────────────────────────────────────────┘
Input: (B, L, D)
  │
  ▼ (Pre-Norm path shown)
┌──────────────────────────┐
│ Layer Normalization      │
└──────────────────────────┘
  │
  ▼
┌──────────────────────────┐
│ Multi-Head Self-Attention│  ← Mixes information across patches
└──────────────────────────┘
  │
  │
  ▼
Add & (Optional Stochastic Depth)
(Input + Attention Output)
  │
  ▼
┌──────────────────────────┐
│ Layer Normalization      │
└──────────────────────────┘
  │
  ▼
┌──────────────────────────┐
│ Feed-Forward Network     │  ← Processes each patch representation
│ (e.g., MLP, SwiGLU)      │
└──────────────────────────┘
  │
  │
  ▼
Add & (Optional Stochastic Depth)
(FFN Input + FFN Output)
  │
  ▼
Output: (B, L, D)
```

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy matplotlib
```

### Your First ViT Model (30 seconds)

Let's build and train a tiny ViT on the CIFAR-10 dataset.

```python
import keras
from keras.datasets import cifar10
import numpy as np

# Local imports from your project structure
from dl_techniques.models.vision.vit.model import ViT

# 1. Load and preprocess data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# 2. Create a ViT model
# We use a small scale and patch size for the small 32x32 images
model = ViT(
    input_shape=(32, 32, 3),
    num_classes=10,
    scale="pico",  # 'pico' is a custom smaller scale for quick tests
    patch_size=4,
    include_top=True,
    dropout_rate=0.1
)

# 3. Compile the model
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
print("✅ ViT model created and compiled successfully!")
model.summary_detailed()

# 4. Train the model (for a few epochs as a demo)
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=5,
    validation_data=(X_test, y_test)
)
print("✅ Training Complete!")

# 5. Evaluate the model
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc * 100:.2f}%")
```

---

## 6. Component Reference

### 6.1 `ViT` (Model Class)

**Purpose**: The main Keras `Model` subclass that assembles the complete ViT architecture.

**Location**: `dl_techniques.models.vision.vit.model.ViT`

```python
from dl_techniques.models.vision.vit.model import ViT

# Create a standard ViT-Base model for classification
model = ViT(
    input_shape=(224, 224, 3),
    num_classes=1000,
    scale="base",
    patch_size=16
)

# Create a ViT-Small as a feature extractor
feature_extractor = ViT(
    input_shape=(384, 384, 3),
    scale="small",
    patch_size=16,
    include_top=False,
    pooling="cls" # Output the [CLS] token representation
)
```

**Key Parameters**:

| Parameter | Description |
| :--- | :--- |
| `input_shape` | Shape of input images `(H, W, C)`. |
| `num_classes` | Number of output classes (if `include_top=True`). |
| `scale` | Model size: 'tiny', 'small', 'base', 'large', 'huge'. |
| `patch_size` | The size of square patches (e.g., 16 for 16x16). |
| `include_top` | If `False`, the model becomes a feature extractor. |
| `pooling` | Pooling mode for feature extraction: 'cls', 'mean', 'max', or `None`. |
| `normalization_type` | 'layer_norm' (default), 'rms_norm', etc. |
| `ffn_type` | 'mlp' (default), 'swiglu', etc. |

### 6.2 `create_vit`

**Purpose**: A high-level factory function for convenient and safe `ViT` model creation. Resnet-template parity: the first positional argument is the variant key (`"vit_pico".."vit_huge"`).

**Location**: `dl_techniques.models.vit.create_vit`

```python
from dl_techniques.models.vit import create_vit

# CIFAR-10 ViT-pico
model = create_vit("vit_pico", num_classes=10, input_shape=(32, 32, 3), patch_size=4)

# ImageNet ViT-base from a local pretrained checkpoint
model = create_vit("vit_base", num_classes=1000,
                   pretrained="/path/to/vit_base_imagenet.keras")
```

> **Note**: `create_vision_transformer` was renamed to `create_vit` in plan
> `plan_2026-05-12_f2d29729`. The old name was removed without a back-compat
> alias — only two in-repo callers existed (`train_vit.py` and the test
> suite), both updated.

### 6.3 `from_variant` and pretrained-weights API

The `ViT` class exposes a `from_variant` classmethod and a `MODEL_VARIANTS`
registry that mirrors `ResNet.from_variant`:

```python
from dl_techniques.models.vit import ViT

# Variant registry: maps user-facing variant keys to internal scale configs.
print(sorted(ViT.MODEL_VARIANTS.keys()))
# ['vit_base', 'vit_huge', 'vit_large', 'vit_pico', 'vit_small', 'vit_tiny']

# Build from a variant key.
model = ViT.from_variant("vit_pico", num_classes=10, input_shape=(32, 32, 3),
                         patch_size=4)

# Pretrained from local .keras checkpoint -- works.
model = ViT.from_variant("vit_base", pretrained="/path/to/weights.keras",
                         num_classes=1000)

# Pretrained=True -- raises NotImplementedError. No public ViT weights in the
# dl_techniques format are hosted; this is a loud failure on purpose so that
# silent fall-throughs to random init don't masquerade as successful loads.
try:
    ViT.from_variant("vit_base", pretrained=True)
except NotImplementedError as e:
    print(e)  # "No public ViT checkpoints are distributed for this implementation. ..."
```

**Weight loading uses `dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`**
(layer-by-layer `set_weights` after a probe forward pass) rather than
`model.load_weights(by_name=True)`. The latter is broken in Keras 3.8+ for
`.keras` files — see `plans/LESSONS.md` L71.

> **Deep supervision** for ViT is deferred to a follow-up plan and is not
> available in this iteration.

---

## 7. Configuration & Model Variants

This implementation supports several standard scales, defined by their embedding dimension, number of attention heads, and number of layers.

| Scale | Embed Dim | Heads | Layers | MLP Ratio | Approx. Params |
| :---: | :---: |:---: |:---: |:---: |:---:|
| **`pico`** | 192 | 3 | 6 | 4.0 | ~3M |
| **`tiny`** | 192 | 3 | 12 | 4.0 | ~5M |
| **`small`**| 384 | 6 | 12 | 4.0 | ~22M |
| **`base`** | 768 | 12 | 12 | 4.0 | ~86M |
| **`large`**| 1024 | 16 | 24 | 4.0 | ~307M |
| **`huge`** | 1280 | 16 | 32 | 4.0 | ~632M |

**Guideline**:
-   **`tiny`/`small`**: Good for fine-tuning on medium-sized datasets (e.g., CIFAR-100, Flowers-102) or for applications where inference speed is key.
-   **`base`**: The standard for ImageNet-scale pre-training and transfer learning.
-   **`large`/`huge`**: Require massive datasets (e.g., ImageNet-21k, JFT-300M) for pre-training to be effective. Best used via transfer learning with pre-trained weights.

---

## 8. Comprehensive Usage Examples

### Example 1: Feature Extraction

Use a ViT as a backbone to extract features for another task, like k-NN classification or clustering.

```python
# 1. Create the feature extractor model
feature_extractor = ViT(
    input_shape=(224, 224, 3),
    scale="base",
    include_top=False,
    pooling="cls" # Get a single vector per image
)
# For real use, you would load pretrained weights here.

# 2. Extract features from a batch of images
dummy_images = np.random.rand(32, 224, 224, 3)
features = feature_extractor.predict(dummy_images)
print(f"Extracted features shape: {features.shape}") # (32, 768)
```

### Example 2: Fine-Tuning a Pre-trained ViT

A common workflow is to load a pre-trained model and fine-tune it on a new, smaller dataset.

```python
# Assume you have pretrained weights saved at 'vit_base_imagenet.keras'

# 1. Load the pretrained base model (as a feature extractor)
base_model = ViT(
    input_shape=(224, 224, 3),
    scale="base",
    include_top=False,
    pooling="cls"
)
# base_model.load_weights('vit_base_imagenet.keras')
base_model.trainable = False # Freeze the backbone

# 2. Add a new classification head for our custom task (e.g., 10 classes)
inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(10, name="new_head")(x)
finetune_model = keras.Model(inputs, outputs)

# 3. Compile and fine-tune
finetune_model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
# finetune_model.fit(train_dataset, ...)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Using Modern Components

This implementation makes it easy to experiment with state-of-the-art components.

```python
# Create a ViT with RMS Normalization and a SwiGLU FFN
modern_vit = ViT(
    input_shape=(256, 256, 3),
    num_classes=100,
    scale="small",
    patch_size=16,
    normalization_type="rms_norm",
    ffn_type="swiglu",
    normalization_position="pre" # Pre-Norm is often more stable
)
modern_vit.summary_detailed()
```

### Pattern 2: Accessing Intermediate Layer Outputs

You can create a model that outputs features from every Transformer block, which is useful for feature pyramid networks or detailed analysis.

```python
# 1. Create a base ViT
vit_base = ViT(input_shape=(224, 224, 3), scale="base", include_top=False)

# 2. Build a new model to tap into intermediate layers
inputs = keras.Input(shape=(224, 224, 3))
x = vit_base.patch_embed(inputs)
# ... apply CLS token and positional embedding ...
x = vit_base.pos_embed(ops.concatenate([...]))

layer_outputs = []
for layer in vit_base.transformer_layers:
    x = layer(x)
    layer_outputs.append(x)

feature_pyramid_model = keras.Model(inputs=inputs, outputs=layer_outputs)
# Now the model outputs a list of 12 tensors, one for each layer's output.
```

---

## 10. Performance Optimization

### Mixed Precision Training

ViTs benefit greatly from mixed precision, which uses float16 to accelerate computations.

```python
# Enable mixed precision globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = ViT(scale="base", ...)
model.compile(...)
```

### Flash Attention

For GPUs that support it (e.g., A100, H100), you can enable `flash` attention in the `TransformerLayer` for a significant speedup and memory reduction.

```python
model = ViT(
    scale="base",
    # Pass custom arguments down to the TransformerLayer
    # This requires modifying the ViT class to accept attention_args
    # and pass them to its TransformerLayer instances.
)
```
*(Note: This example assumes a modification to the `ViT` class to pass `attention_args` down to `TransformerLayer`.)*

---

## 11. Training and Best Practices

### Data Augmentation

ViTs, having fewer inductive biases, are more reliant on data augmentation than CNNs.
-   **Strong Augmentations**: Use techniques like RandAugment, Mixup, and CutMix during training.
-   **Resolution**: Pre-training on a lower resolution (e.g., 224x224) and fine-tuning on a higher resolution (e.g., 384x384) is a common and effective strategy. The model's positional embeddings can be interpolated to handle the new sequence length.

### Optimizer and Scheduler

-   **Optimizer**: **AdamW** is the standard optimizer for ViTs, as weight decay is crucial for regularization.
-   **Learning Rate Schedule**: A cosine decay schedule with a linear warmup period is the most common and effective choice.

---

## 12. Serialization & Deployment

The `ViT` model and its custom layers are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
# Create and train model
model = ViT(...)
# model.compile(...) and model.fit(...)

# Save the entire model
model.save('my_vit_model.keras')
print("Model saved to my_vit_model.keras")

# Load the model in a new session
loaded_model = keras.models.load_model('my_vit_model.keras')
print("Model loaded successfully")
```

---

## 13. Testing & Validation

### Unit Tests

```python
import keras
import numpy as np
from dl_techniques.models.vision.vit.model import ViT

def test_model_creation_all_scales():
    """Test that models can be created from all standard scales."""
    for scale in ViT.SCALE_CONFIGS.keys():
        model = ViT(input_shape=(224, 224, 3), num_classes=10, scale=scale, patch_size=16)
        assert model is not None
        print(f"✓ ViT-{scale} created successfully")

def test_forward_pass_shape_classifier():
    """Test the output shape of a classifier."""
    model = ViT(input_shape=(32, 32, 3), num_classes=10, scale="pico", patch_size=4)
    dummy_input = np.random.rand(4, 32, 32, 3)
    output = model.predict(dummy_input)
    assert output.shape == (4, 10)
    print("✓ Classifier forward pass has correct shape")

def test_forward_pass_shape_extractor():
    """Test the output shape of a feature extractor."""
    model = ViT(input_shape=(32, 32, 3), scale="pico", patch_size=4, include_top=False, pooling="cls")
    dummy_input = np.random.rand(4, 32, 32, 3)
    output = model.predict(dummy_input)
    embed_dim = ViT.SCALE_CONFIGS["pico"][0]
    assert output.shape == (4, embed_dim)
    print("✓ Feature extractor forward pass has correct shape")

# Run tests
if __name__ == '__main__':
    test_model_creation_all_scales()
    test_forward_pass_shape_classifier()
    test_forward_pass_shape_extractor()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Model does not converge on a small dataset.**

-   **Cause**: ViTs have weak inductive biases and require large amounts of data to learn spatial hierarchies from scratch. On small datasets (like CIFAR-10), they can easily overfit or fail to learn meaningful representations without heavy regularization and data augmentation.
-   **Solution 1**: Use strong data augmentation (RandAugment, Mixup, CutMix).
-   **Solution 2**: Add more regularization (increase `dropout_rate`, use AdamW with significant `weight_decay`).
-   **Solution 3 (Best)**: **Use a pre-trained model**. Fine-tuning a ViT pre-trained on ImageNet is far more effective than training from scratch on a small dataset.

**Issue 2: `ValueError: Image height/width must be divisible by patch size`.**

-   **Cause**: The input image dimensions are not a multiple of the patch size.
-   **Solution**: Resize or pad your input images to a compatible resolution before feeding them to the model. For example, if using `patch_size=16`, image sizes like 224, 256, 384, or 512 are valid.

### Frequently Asked Questions

**Q: How does ViT handle images of different resolutions during inference?**

A: The standard ViT requires a fixed input resolution because the positional embeddings are learned for a specific sequence length. If you provide a different resolution, the number of patches changes, and the model won't have corresponding positional embeddings. Advanced techniques like 2D interpolation of the positional embedding map can adapt the model to new resolutions.

**Q: What is the difference between `pooling='cls'` and `pooling='mean'`?**

A:
-   **`pooling='cls'`**: Uses the output representation of the special `[CLS]` token, which is trained to aggregate global information. This is the standard method from the paper.
-   **`pooling='mean'`**: Computes the average of all *patch* token representations (excluding the `[CLS]` token). Some studies have found this to work as well as or even slightly better than the `[CLS]` token for certain downstream tasks.

**Q: Can I use this for object detection or segmentation?**

A: Yes, but not directly. This implementation provides the **backbone** (feature extractor). To perform detection or segmentation, you would need to attach additional heads, such as a detection head (like in DETR) or a segmentation decoder (like in U-Net, but with a ViT encoder). The `include_top=False, pooling=None` configuration is ideal for this, as it provides the full sequence of patch representations.

---

## 15. Technical Details

### Positional Embeddings

The choice of positional embedding is crucial. This implementation uses 1D **learned, absolute positional embeddings**, which is the most common variant. A unique vector is learned for each position in the sequence (0 for the `[CLS]` token, 1 for the first patch, etc.). These vectors are simply added to the patch embeddings. Other possibilities not implemented here include:
-   **2D Positional Embeddings**: Learning separate embeddings for the X and Y coordinates of each patch.
-   **Relative Positional Embeddings**: Modifying the attention mechanism to directly encode the relative distance between patches, rather than their absolute positions.
-   **Sinusoidal (Fixed) Embeddings**: Using the fixed sinusoidal functions from the original "Attention Is All You Need" paper.

### Pre-Norm vs. Post-Norm

The `normalization_position` parameter controls the placement of the Layer Normalization.
-   **`post` (Original)**: `SubLayer -> Add -> Norm`. Can be unstable to train for very deep models, often requiring learning rate warmup.
-   **`pre` (More Modern)**: `Norm -> SubLayer -> Add`. Generally more stable, allows for training without warmup, and often leads to better performance. This implementation supports both.

---

## 16. Citation

If you use ViT in your research, please cite the original paper:

```bibtex
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```
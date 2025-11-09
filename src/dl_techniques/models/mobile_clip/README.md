# MobileCLIP: Fast and Efficient On-Device CLIP

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured Keras 3 implementation of Apple's **MobileCLIP**, an efficient vision-language model designed for high performance on resource-constrained devices. MobileCLIP adapts the powerful zero-shot capabilities of CLIP (Contrastive Language-Image Pre-training) to an architecture optimized for the latency, memory, and power limitations of mobile and edge computing.

The architecture is composed of a mobile-friendly image encoder and a configurable Transformer-based text encoder, trained jointly with a contrastive objective.

---

## Table of Contents

1. [Overview: What is MobileCLIP and Why It Matters](#1-overview-what-is-mobileclip-and-why-it-matters)
2. [The Problem MobileCLIP Solves](#2-the-problem-mobileclip-solves)
3. [How MobileCLIP Works: Core Concepts](#3-how-mobileclip-works-core-concepts)
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

## 1. Overview: What is MobileCLIP and Why It Matters

### What is MobileCLIP?

**MobileCLIP** is a highly efficient multimodal model from Apple that brings the power of large-scale vision-language understanding to on-device applications. It follows the same principles as the original CLIP model—learning a shared embedding space between images and text from web-scale data—but does so with a co-designed, asymmetric architecture that prioritizes mobile performance.

### Key Innovations

1.  **Efficient Image Backbones**: Instead of the large Vision Transformers (ViTs) used in the original CLIP, MobileCLIP employs custom, lightweight convolutional backbones (referred to as MCI0, MCI1, etc.) that are heavily inspired by the MobileNet family.
2.  **Asymmetric Design**: The model is intentionally asymmetric. The image encoder is made extremely fast and lightweight, as it must run on every input frame in real-time applications. The text encoder, which often runs only once on a small set of text prompts for zero-shot classification, can be comparatively more powerful.
3.  **Optimized Text Encoder**: The text encoder is a standard Transformer, but its depth and configuration are tailored across different variants to balance performance and accuracy for mobile use cases.

### Why MobileCLIP Matters

**The Standard CLIP Problem**:
```
Problem: Perform zero-shot classification on a mobile phone.
Standard CLIP Approach:
  1. Use the original CLIP model, which has a massive ViT image encoder.
  2. The model is too large to fit in memory and too slow for real-time inference.
  3. Result: The application is unusable on-device, forcing developers to rely
     on a cloud server, which introduces latency and privacy issues.
```

**MobileCLIP's Solution**:
```
MobileCLIP Approach:
  1. Replace the large image encoder with a hyper-efficient mobile-native CNN.
  2. Optimize the entire architecture for on-device constraints.
  3. The model runs directly on the phone with very low latency.
  4. Benefit: Enables powerful, zero-shot AI capabilities like real-time object
     recognition with natural language, visual search, and content filtering,
     all directly on the user's device without needing a network connection.
```

---

## 2. The Problem MobileCLIP Solves

### The "Last Mile" Problem for Vision-Language Models

While models like CLIP demonstrated a revolutionary leap in zero-shot learning, their immense size and computational cost made them largely inaccessible for deployment on edge devices. This created a "last mile" problem: how do we deliver this powerful technology to the hands of billions of users in a practical way?

```
┌──────────────────────────────────────────────────────────────┐
│  The Dilemma of Deploying Large VLMs                         │
│                                                              │
│  Large Cloud Models (e.g., CLIP ViT-L/14):                   │
│    - State-of-the-art accuracy.                              │
│    - Prohibitively large: hundreds of millions of parameters.│
│    - High latency: requires a round trip to a server.        │
│    - Privacy concerns: user data must leave the device.      │
│                                                              │
│  The Need for On-Device AI:                                  │
│    - A model that retains the flexible, zero-shot            │
│      capabilities of CLIP but is small, fast, and            │
│      power-efficient enough to run locally.                  │
└──────────────────────────────────────────────────────────────┘
```

MobileCLIP was specifically designed to solve this. By starting with on-device performance as a primary constraint, it re-architects the vision-language pre-training paradigm to create a model that is "efficient by design," not just a scaled-down version of a larger model.

---

## 3. How MobileCLIP Works: Core Concepts

### The Dual-Encoder Contrastive Architecture

At its core, MobileCLIP uses the same successful dual-encoder, contrastive learning framework as the original CLIP. It consists of two separate neural networks—one for images and one for text—that are trained to map their respective inputs into a shared high-dimensional space.

```
┌───────────────────────────────────────────────────────────────────┐
│                         MobileCLIP Architecture                   │
│                                                                   │
│  Image Input (Batch) ───►┌─────────────────┐                      │
│                          │  Image Encoder  │                      │
│                          │(Efficient CNN)  │───► Image Embeddings │
│                          └─────────────────┘      (Batch, D)      │
│                                                       │           │
│  Text Input (Batch) ───► ┌─────────────────┐          ▼           │
│                          │  Text Encoder   │                     ┌────────────┐
│                          │ (Transformer)   │───► Text Embeddings │ Compute NxN│
│                          └─────────────────┘      (Batch, D)     │ Similarity │
│                                                       ▲          └────────────┘
│                                                       │                │
│                                                       └────────────────▼
│                                                    Contrastive Loss
│                (Maximize similarity of correct pairs, minimize for others)
│
└──────────────────────────────────────────────────────────────────┘
```

The training process involves showing the model a batch of N (image, text) pairs. It computes an N x N similarity matrix between all image and text embeddings. The goal of the contrastive loss is to maximize the similarity scores on the diagonal (correct pairs) while minimizing the scores for all other incorrect pairings.

---

## 4. Architecture Deep Dive

### 4.1 `MobileClipImageEncoder`

-   **Purpose**: To convert an image into a compact vector embedding with maximum efficiency.
-   **Architecture**:
    1.  **Backbone**: A lightweight convolutional neural network. The official MobileCLIP uses custom backbones named `mci0`, `mci1`, etc., which are based on efficient principles from models like MobileNet. This implementation allows using any Keras Applications backbone (e.g., `MobileNetV2`).
    2.  **Projection Head**: The feature map from the backbone is passed to an `ImageProjectionHead`. This head performs global average pooling to create a feature vector and then uses a `Dense` layer to project it into the final shared `embed_dim`.

### 4.2 `MobileClipTextEncoder`

-   **Purpose**: To convert a sequence of text tokens into a vector embedding that aligns with the image embeddings.
-   **Architecture**:
    1.  **Token and Positional Embeddings**: Input token IDs are converted to dense vectors, and a learnable positional embedding is added to encode word order.
    2.  **Transformer Stack**: The sequence is processed by a stack of `TransformerLayer` blocks. These apply self-attention to build context-aware representations.
    3.  **Feature Extraction**: The output embedding corresponding to the final "end-of-text" (EOT) token is selected as the representative feature for the entire text sequence.
    4.  **Projection**: This feature vector is projected into the final shared `embed_dim` via a matrix multiplication.

### 4.3 Contrastive Head and `logit_scale`

-   **Purpose**: To compute the final similarity scores for the contrastive loss.
-   **Functionality**:
    1.  **L2 Normalization**: Both the final image and text features are L2-normalized. This constrains them to a hypersphere, making the dot product equivalent to cosine similarity.
    2.  **Temperature Scaling**: The similarity scores (logits) are multiplied by a learnable scalar, `logit_scale`. This temperature parameter, initialized to `log(1/0.07)`, controls the sharpness of the distribution over similarities and is critical for stable training.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First MobileCLIP Model (30 seconds)

Let's build the smallest MobileCLIP variant (`s0`) and perform a single forward pass.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.mobile_clip.mobile_clip_v1 import MobileClipModel

# 1. Create a MobileCLIP model from the "s0" variant
# This implementation requires a custom backbone; we'll use MobileNetV2 for demonstration.
# Note: To truly replicate MobileCLIP, custom backbones 'mci0', 'mci1' etc. would be needed.
MobileClipModel.MODEL_VARIANTS['s0']['image_config']['backbone_name'] = 'MobileNetV2'
model = MobileClipModel.from_variant("s0")

# 2. Build the model with expected input shapes
model.build({
    'image': (None, 256, 256, 3),
    'text': (None, 77)
})
print("✅ MobileCLIP model created and built successfully!")
model.summary()

# 3. Create dummy data for a forward pass
batch_size = 8
dummy_images = np.random.rand(batch_size, 256, 256, 3).astype("float32")
dummy_text_tokens = np.random.randint(0, 49408, (batch_size, 77))

# 4. Perform a full forward pass
outputs = model({
    'image': dummy_images,
    'text': dummy_text_tokens
})

# 5. Inspect the outputs
print(f"\nImage features shape: {outputs['image_features'].shape}") # (B, 512)
print(f"Text features shape: {outputs['text_features'].shape}")   # (B, 512)
print(f"Logit scale: {outputs['logit_scale'].numpy():.4f}")```

---

## 6. Component Reference

### 6.1 Model Classes and Creation Functions

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`MobileClipModel`** | `...mobile_clip_v1.MobileClipModel` | The main Keras `Model` for the complete dual-encoder architecture. |
| **`create_mobile_clip_model`** | `...mobile_clip_v1.create_mobile_clip_model` | Recommended convenience function to create `MobileClipModel` from variants. |
| **`MobileClipImageEncoder`** | `...components.MobileClipImageEncoder` | The image encoder model, combining a backbone and projection head. |
| **`MobileClipTextEncoder`** | `...components.MobileClipTextEncoder` | The Transformer-based text encoder layer. |
| **`ImageProjectionHead`** | `...components.ImageProjectionHead` | A layer for pooling and projecting backbone feature maps. |

---

## 7. Configuration & Model Variants

This implementation provides configurations for the official MobileCLIP variants.

| Variant | Embed Dim | Image Backbone | Image Size | Text Layers | Causal Mask |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **`b`** | 512 | `vit_b16` | 224 | 12 | **True** |
| **`s0`**| 512 | `mci0` | 256 | **4** | False |
| **`s1`** | 512 | `mci1` | 256 | 12 | False |
| **`s2`** | 512 | `mci2` | 256 | 12 | False |

*Note: The `mci` backbones are custom to the original paper. This implementation uses standard `keras.applications` models as placeholders.*

---

## 8. Comprehensive Usage Examples

### Example 1: Zero-Shot Image Classification

This is the primary use case for MobileCLIP. Classify an image using natural language prompts without any fine-tuning.

```python
import keras
import numpy as np
from dl_techniques.models.mobile_clip.mobile_clip_v1 import create_mobile_clip_model

# Assume you have a text tokenizer
def tokenize(texts, context_length=77):
    # This is a placeholder for a real tokenizer (e.g., from huggingface/transformers)
    # The vocab size should match the model's (49408)
    return np.random.randint(0, 49408, (len(texts), context_length))

# 1. Create the MobileCLIP model
# Replace backbone for demonstration
MobileClipModel.MODEL_VARIANTS['s0']['image_config']['backbone_name'] = 'MobileNetV2'
model = create_mobile_clip_model("s0")

# 2. Prepare the image and text prompts
dummy_image = np.random.rand(1, 256, 256, 3).astype("float32")
class_descriptions = ["a photo of a dog", "a photo of a cat", "a drawing of a car"]
text_tokens = tokenize(class_descriptions)

# 3. Get image and text features (the model normalizes them internally)
image_features = model.encode_image(dummy_image)
text_features = model.encode_text(text_tokens)

# 4. Compute similarity and find the best match
similarities = keras.ops.matmul(image_features, keras.ops.transpose(text_features))
probabilities = keras.ops.softmax(similarities, axis=-1)
best_match_index = keras.ops.argmax(probabilities, axis=-1)

print(f"Input image is most similar to: '{class_descriptions[best_match_index[0]]}'")
print(f"Probabilities: {probabilities.numpy().flatten()}")
```

### Example 2: Getting Separate Embeddings for Downstream Tasks

Use the encoders to get high-quality, efficient embeddings for other tasks like visual search or clustering.

```python
# (Model creation from Example 1)
batch_size = 4
dummy_images = np.random.rand(batch_size, 256, 256, 3).astype("float32")
dummy_texts = ["a red car", "blue sky", "a running dog", "a sleeping cat"]
text_tokens = tokenize(dummy_texts)

# Get L2-normalized embeddings
image_embeddings = model.encode_image(dummy_images)
text_embeddings = model.encode_text(text_tokens)

print(f"Image embeddings shape: {image_embeddings.shape}") # (4, 512)
print(f"Text embeddings shape: {text_embeddings.shape}")   # (4, 512)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Fine-tuning on a Custom Dataset

While designed for zero-shot use, MobileCLIP can be fine-tuned on a specific domain (e.g., medical images, product photos) to improve performance. This requires a custom training loop to implement the contrastive loss.

```python
# Create a MobileCLIP model
model = create_mobile_clip_model("s1")
# Potentially load pre-trained weights here

def contrastive_loss(image_features, text_features, logit_scale):
    # Calculate cosine similarity
    logits_per_image = logit_scale * ops.matmul(image_features, text_features, transpose_b=True)
    
    # Symmetrical cross-entropy loss
    batch_size = ops.shape(logits_per_image)[0]
    labels = ops.arange(batch_size)
    
    loss_img = keras.losses.sparse_categorical_crossentropy(labels, logits_per_image, from_logits=True)
    loss_txt = keras.losses.sparse_categorical_crossentropy(labels, ops.transpose(logits_per_image), from_logits=True)
    
    return (ops.mean(loss_img) + ops.mean(loss_txt)) / 2.0

# In your training loop:
# with tf.GradientTape() as tape:
#     outputs = model({'image': images, 'text': texts}, training=True)
#     loss = contrastive_loss(outputs['image_features'], outputs['text_features'], outputs['logit_scale'])
#
# grads = tape.gradient(loss, model.trainable_variables)
# optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

---

## 10. Performance Optimization

### Mixed Precision Training

The Transformer-based text encoder and modern CNN backbones are excellent candidates for mixed precision training, which can significantly accelerate training on compatible GPUs.

```python
# Enable mixed precision globally before creating the model
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = create_mobile_clip_model("s0")
```

---

## 11. Training and Best Practices

### Optimizer and Schedule

-   **Optimizer**: **AdamW** is a strong choice due to its effective handling of weight decay, which is crucial for regularizing Transformer-based models.
-   **Learning Rate Schedule**: A **cosine decay** schedule with a linear warmup phase is standard practice and generally yields the best results for training CLIP-style models.

### Batch Size is Key

-   Contrastive learning performance scales directly with **batch size**. A larger batch provides more negative examples for each positive pair, creating a more robust learning signal. The original CLIP was trained with a batch size of over 32,000. For practical training on consumer hardware, techniques like **gradient accumulation** are essential to simulate large batches.

---

## 12. Serialization & Deployment

The `MobileClipModel` and all its custom components are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
# Create and train model
model = create_mobile_clip_model("s0")
# ... training ...

# Save the entire model to a single file
model.save('my_mobile_clip_model.keras')

# Load the model in a new session, including architecture and weights
loaded_model = keras.models.load_model('my_mobile_clip_model.keras')
print("✅ MobileCLIP model loaded successfully!")
```

---

## 13. Testing & Validation

### Unit Tests

Simple tests can validate that all model variants can be created and that forward passes produce correctly shaped outputs.

```python
import keras
import numpy as np
from dl_techniques.models.mobile_clip.mobile_clip_v1 import MobileClipModel

def test_creation_all_variants():
    """Test model creation for all variants."""
    for variant in MobileClipModel.MODEL_VARIANTS.keys():
        # Replace backbone for testing since custom ones aren't available
        MobileClipModel.MODEL_VARIANTS[variant]['image_config']['backbone_name'] = 'MobileNetV2'
        model = MobileClipModel.from_variant(variant)
        assert model is not None
        print(f"✓ MobileCLIP-{variant} created successfully")

def test_forward_pass_shapes():
    """Test the output shapes of a full forward pass."""
    MobileClipModel.MODEL_VARIANTS['s0']['image_config']['backbone_name'] = 'MobileNetV2'
    model = MobileClipModel.from_variant("s0")
    batch_size, embed_dim = 4, 512
    
    inputs = {
        'image': np.random.rand(batch_size, 256, 256, 3),
        'text': np.random.randint(0, 49408, (batch_size, 77))
    }
    
    outputs = model(inputs)
    assert outputs['image_features'].shape == (batch_size, embed_dim)
    assert outputs['text_features'].shape == (batch_size, embed_dim)
    print("✓ Forward pass shapes are correct")

# Run tests
if __name__ == '__main__':
    test_creation_all_variants()
    test_forward_pass_shapes()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Training is unstable and the loss becomes `NaN`.**

-   **Cause 1**: The learning rate may be too high.
-   **Cause 2**: The learnable temperature `logit_scale` might be diverging.
-   **Solution**: Use a smaller learning rate and a warmup schedule. The implementation already includes clipping for `logit_scale` to a maximum of 100, which helps prevent divergence.

### Frequently Asked Questions

**Q: What are the `mci0`, `mci1`, and `mci2` backbones?**

A: These are custom, efficient CNN architectures designed by Apple specifically for MobileCLIP. They are not part of standard libraries like `keras.applications`. To perfectly replicate the paper's results, one would need to implement these specific backbones. This implementation uses standard mobile backbones like `MobileNetV2` as a functional replacement.

**Q: What is the main difference between the 'b' and 's' variants?**

A: The **'b' (base)** variant is a larger model designed for higher accuracy, using a ViT-based image encoder and a deeper text encoder with causal masking. The **'s' (small)** variants (`s0`, `s1`, `s2`) are optimized for on-device performance, using efficient CNN backbones and a text encoder without causal masking. The `s0` variant is particularly small, with only 4 Transformer layers in its text encoder.

**Q: Where can I find the correct tokenizer?**

A: The model expects a tokenizer compatible with the original CLIP, which is a byte-pair encoding (BPE) tokenizer with a vocabulary size of 49,408. You can find pre-trained versions of this tokenizer in libraries like Hugging Face's `transformers` (e.g., from the `openai/clip-vit-base-patch32` model).

---

## 15. Technical Details

### Asymmetric Design Philosophy

A key insight of MobileCLIP is the asymmetric nature of its common use case (zero-shot classification).
-   **Image Encoder**: Must be extremely fast, as it processes a new image for every inference.
-   **Text Encoder**: Can be more computationally expensive. In a classification task with `K` classes, it only needs to run `K` times to create the class embeddings, which can then be reused for many image comparisons.
This motivates the design of using a hyper-efficient CNN for images while retaining a more powerful (though still optimized) Transformer for text.

### Causal Masking in the Text Encoder

The `b` variant uses `use_causal_mask=True` in its text encoder, while the `s` variants use `False`. A causal mask prevents a token from attending to future tokens in the sequence. This is standard for autoregressive language models. Its inclusion in the `b` variant aligns it more closely with large language model architectures, while its exclusion in the `s` variants is likely an efficiency optimization, as it slightly simplifies the attention computation.

---

## 16. Citation

This implementation is based on the official MobileCLIP paper from Apple. If you use this model in your research, please cite the original work:

```bibtex
@inproceedings{faghri2023mobileclip,
  title={Mobile{CLIP}: Fast and Efficient On-Device {CLIP}},
  author={Faghri, Fartash and Varma, Girik and Furlan, L\'{e}o and Sarıyıldız, M. B. and Abs, M. and Ghasemzadeh, Hessam},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
  year={2023}
}
```
# CLIP: Contrastive Language-Image Pre-Training

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured Keras 3 implementation of the **CLIP (Contrastive Language-Image Pre-training)** model. CLIP learns rich visual representations from natural language supervision by training a vision and text encoder in parallel to predict which images were paired with which texts in a large dataset. This enables powerful zero-shot transfer capabilities for a wide range of vision tasks.

The implementation is based on a highly configurable and modern `TransformerLayer` that uses state-of-the-art components like RMS Normalization, SwiGLU activations, and Grouped-Query Attention.

---

## Table of Contents

1. [Overview: What is CLIP and Why It Matters](#1-overview-what-is-clip-and-why-it-matters)
2. [The Problem CLIP Solves](#2-the-problem-clip-solves)
3. [How CLIP Works: Core Concepts](#3-how-clip-works-core-concepts)
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

## 1. Overview: What is CLIP and Why It Matters

### What is CLIP?

**CLIP (Contrastive Language-Image Pre-training)** is a multi-modal neural network that learns the connection between images and text. Instead of training on a fixed set of object labels (like "cat" or "dog"), CLIP is trained on a massive dataset of (image, text) pairs collected from the internet. Its goal is to learn a shared embedding space where the vector for an image of a dog is close to the vector for the text "a photo of a dog."

### Key Innovations

1.  **Learning from Natural Language**: CLIP moves beyond fixed classification labels and learns from the rich, noisy, and diverse text captions found online. This gives it a much broader and more flexible understanding of the visual world.
2.  **Contrastive Learning at Scale**: CLIP is trained on a simple yet highly effective contrastive objective: given a batch of N (image, text) pairs, the model must predict which N images correspond to which N texts. This scalable pre-training task is what enables its powerful representations.
3.  **Zero-Shot Transfer**: Because CLIP learns a general-purpose association between vision and language, it can be adapted to new visual classification tasks *without any additional training*. You can simply provide it with text descriptions of your target classes (e.g., "a photo of a car," "a drawing of a bird") and it will classify new images based on which description is most similar.

### Why CLIP Matters

**Traditional Supervised Learning Problem**:
```
Problem: Build a model to classify 1000 different types of objects.
Traditional Approach:
  1. Collect hundreds of thousands of images.
  2. Manually label each image with one of the 1000 object classes.
  3. Train a model to predict the correct class label.
  4. Limitation: The model is "stuck" with only these 1000 classes. To add a
     new class, you must collect new data and retrain.
```

**CLIP's Solution**:
```
CLIP Approach:
  1. Scrape millions of images and their associated alt-text from the web.
  2. Train a dual-encoder model to match images to their text descriptions.
  3. For a new task, simply write text prompts for your classes. No retraining needed.
  4. Benefit: A single, pre-trained CLIP model can be used for a virtually
     unlimited number of visual classification tasks on the fly.
```

### Real-World Impact

CLIP is a foundational model that has powered a revolution in multi-modal AI:

-   **Zero-Shot Image Classification**: Its primary use case, enabling flexible and powerful classification without fine-tuning.
-   **Text-to-Image Generation**: The shared embedding space is a core component in models like DALL-E 2 and Stable Diffusion, which use it to guide image generation from text prompts.
-   **Image Search & Retrieval**: Find images based on complex natural language descriptions, not just simple tags.
-   **Foundation for Vision-Language Models (VLMs)**: Serves as a powerful vision backbone for more complex reasoning and question-answering models.

---

## 2. The Problem CLIP Solves

### The Brittleness of Supervised Learning

For years, the gold standard in computer vision was supervised learning on large, manually labeled datasets like ImageNet. While powerful, this paradigm had major limitations:

```
┌─────────────────────────────────────────────────────────────┐
│  The Dilemma of Supervised Vision Models                    │
│                                                             │
│  The Old Way (e.g., ResNet on ImageNet):                    │
│    - High cost: Manual labeling is expensive and slow.      │
│    - Narrow scope: Models learn a predefined, fixed set of  │
│      concepts. They struggle with anything outside this set.│
│    - Poor generalization: Performance on benchmarks often   │
│      doesn't translate to real-world messy data.            │
│                                                             │
│  The Need:                                                  │
│    - A model that learns a more general and robust          │
│      understanding of visual concepts, much like humans do, │
│      from broad, uncurated data sources.                    │
└─────────────────────────────────────────────────────────────┘
```

CLIP addresses this by tapping into the largest source of labeled data available: the internet. By learning from raw (image, text) pairs, it sidesteps the need for costly manual annotation and develops a much more flexible and comprehensive visual understanding.

### How CLIP Changes the Game

CLIP's pre-training task forces it to learn not just what objects are, but also their context, attributes, actions, and relationships, as described in natural language.

```
┌─────────────────────────────────────────────────────────────┐
│  The CLIP Learning Strategy                                 │
│                                                             │
│  1. The Data:                                               │
│     - Use 400 million (image, text) pairs from the internet.│
│     - No expensive manual labeling needed.                  │
│                                                             │
│  2. The Task (Contrastive Learning):                        │
│     - Given a batch of images and texts, learn to "contrast"│
│       the correct pairs from the incorrect ones. This forces│
│       the model to create a shared space where semantically │
│       similar images and texts are close together.          │
│                                                             │
│  3. The Result:                                             │
│     - A robust, general-purpose vision model that can be    │
│       instructed in natural language to perform new tasks.  │
└─────────────────────────────────────────────────────────────┘
```

This approach results in a highly versatile and powerful model that serves as a flexible foundation for a multitude of vision and multi-modal applications.

---

## 3. How CLIP Works: Core Concepts

### The Dual-Encoder Contrastive Architecture

CLIP consists of two main components that are trained jointly: an image encoder and a text encoder.

```
┌──────────────────────────────────────────────────────────────────┐
│                           CLIP Architecture                      │
│                                                                  │
│  Image Input (Batch) ───►┌─────────────────┐                     │
│                          │  Image Encoder  │                     │
│                          │     (ViT)       │──► Image Embeddings │
│                          └─────────────────┘      (Batch, D)     │
│                                                       │          │
│  Text Input (Batch) ───►┌─────────────────┐          ▼           │
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

### The Complete Data Flow (During Training)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIP Complete Data Flow                        │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: BATCH PREPARATION
─────────────────────────
- A batch of N (image, text) pairs is sampled.
- We now have N images and N texts.

STEP 2: ENCODING
────────────────
- All N images are passed through the Image Encoder to get N image embeddings.
- All N texts are passed through the Text Encoder to get N text embeddings.

STEP 3: SIMILARITY CALCULATION
──────────────────────────────
- The cosine similarity is computed between every image embedding and every text embedding.
- This creates an N x N similarity matrix.
- The values on the diagonal (i, i) of this matrix represent the similarity between
  the i-th image and its correct corresponding text.
- The off-diagonal values (i, j) where i ≠ j represent the similarity between
  an image and an incorrect text.

STEP 4: LOSS CALCULATION
────────────────────────
- The goal is to maximize the similarity scores on the diagonal while minimizing the
  scores on the off-diagonal.
- This is formulated as a cross-entropy loss, applied symmetrically for both
  images-to-texts and texts-to-images.
- A learnable temperature parameter (logit_scale) is used to scale the similarities,
  controlling the sharpness of the probability distribution.
```

---

## 4. Architecture Deep Dive

### 4.1 `Image Encoder`

-   **Purpose**: To convert an image into a fixed-size vector representation.
-   **Implementation**: A Vision Transformer (ViT).
    1.  **Patchify Stem**: A `Conv2D` with a kernel size and stride equal to `patch_size` (e.g., 16x16 or 32x32). This splits the image into a sequence of flattened patches.
    2.  **Add `[CLS]` Token**: A special, learnable `[CLS]` (classification) token is prepended to the sequence of patch embeddings.
    3.  **Transformer Blocks**: The entire sequence is processed by a stack of standard Transformer layers.
    4.  **Feature Extraction**: The output embedding corresponding to the `[CLS]` token is taken as the final representation of the image.
    5.  **Projection**: A final `Dense` layer projects this representation into the shared `embed_dim`.

### 4.2 `Text Encoder`

-   **Purpose**: To convert a sequence of text tokens into a fixed-size vector representation.
-   **Implementation**: A standard causal Transformer.
    1.  **Token Embedding**: An `Embedding` layer maps input token IDs to dense vectors. Positional embeddings are added.
    2.  **Transformer Blocks**: The sequence of token embeddings is processed by a stack of Transformer layers with a causal attention mask.
    3.  **Feature Extraction**: The output embedding of the final token (often an `[EOT]` or End-of-Text token) is taken as the representation for the entire text sequence.
    4.  **Projection**: A final `Dense` layer projects this representation into the shared `embed_dim`.

### 4.3 Contrastive Head and `logit_scale`

-   **Purpose**: To compute the final similarity scores and scale them for the loss function.
-   **Functionality**:
    1.  **Normalization**: Both the final image and text features are L2-normalized to lie on a hypersphere. This makes cosine similarity equivalent to a simple dot product.
    2.  **Similarity**: An N x N similarity matrix is computed via matrix multiplication: `logits = image_features @ text_features.T`.
    3.  **Temperature Scaling**: The logits are multiplied by a learnable scalar parameter, `logit_scale` (which is stored as `log(t)` and exponentiated). This temperature `t` controls the sharpness of the distribution over similarities, helping the model learn more effectively during training.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First CLIP Model (30 seconds)

Let's build a small CLIP model and perform a single forward pass.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.clip.model import CLIP

# 1. Create a ViT-B/32 model from a predefined variant
model = CLIP.from_variant("ViT-B/32")

# Build the model with expected input shapes
model.build({
    'image': (None, 224, 224, 3),
    'text': (None, 77)
})

# 2. Compile the model (e.g., for training)
model.compile(optimizer="adam") # Optimizer is needed but loss is custom
print("✅ CLIP model created and compiled successfully!")
model.summary()

# 3. Create dummy data for a forward pass
batch_size = 8
dummy_images = np.random.rand(batch_size, 224, 224, 3).astype("float32")
dummy_text_tokens = np.random.randint(0, 49408, (batch_size, 77))

# 4. Perform a full forward pass (as in training)
outputs = model({
    'image': dummy_images,
    'text': dummy_text_tokens
})

# 5. Inspect the outputs
print(f"\nImage features shape: {outputs['image_features'].shape}") # (B, 512)
print(f"Text features shape: {outputs['text_features'].shape}")   # (B, 512)
print(f"Logits per image shape: {outputs['logits_per_image'].shape}") # (B, B)
```

---

## 6. Component Reference

### 6.1 Model Class and Creation Functions

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`CLIP`** | `...clip.model.CLIP` | The main Keras `Model` for the complete dual-encoder architecture. |
| **`from_variant`** | `...clip.model.CLIP.from_variant` | Recommended class method to create CLIP models from predefined configurations. |
| **`create_clip_model`** | `...clip.model.create_clip_model` | Convenience function to create a custom CLIP model from scratch. |
| **`create_clip_variant`** | `...clip.model.create_clip_variant` | Convenience function to create a CLIP model from a predefined variant. |

### 6.2 Core Building Block

| Layer | Location | Purpose |
| :--- | :--- | :--- |
| **`TransformerLayer`** | `...layers.transformers.TransformerLayer` | The highly configurable, modern Transformer block that powers both the vision and text encoders. |

---

## 7. Configuration & Model Variants

This implementation provides several pre-configured variants from the original paper.

| Variant | Embed Dim | Vision Layers | Vision Width | Patch Size | Text Layers | Text Width |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`ViT-B/32`** | 512 | 12 | 768 | 32 | 12 | 512 |
| **`ViT-B/16`**| 512 | 12 | 768 | 16 | 12 | 512 |
| **`ViT-L/14`** | 768 | 24 | 1024 | 14 | 12 | 768 |
| **`ViT-H/14`**| 1024 | 32 | 1280 | 14 | 12 | 1024 |

---

## 8. Comprehensive Usage Examples

### Example 1: Zero-Shot Image Classification

The most powerful application of CLIP is classifying images without any fine-tuning.

```python
import keras
import numpy as np
from dl_techniques.models.clip.model import create_clip_variant

# Assume you have a text tokenizer
def tokenize(texts, context_length=77):
    # This is a placeholder for a real tokenizer (e.g., from huggingface/transformers)
    return np.random.randint(0, 49408, (len(texts), context_length))

# 1. Create the CLIP model
model = create_clip_variant("ViT-B/32")
model.build({
    'image': (None, 224, 224, 3),
    'text': (None, 77)
})

# 2. Prepare the image and text prompts
# A dummy image that we want to classify
dummy_image = np.random.rand(1, 224, 224, 3).astype("float32")

# The candidate classes described in natural language
class_descriptions = [
    "a photo of a dog",
    "a photo of a cat",
    "a drawing of a car",
    "a picture of a bird"
]
text_tokens = tokenize(class_descriptions)

# 3. Get image and text features using the dedicated encoding methods
image_features = model.encode_image(dummy_image)
text_features = model.encode_text(text_tokens)

# 4. Compute similarity and find the best match
# Both features are already L2-normalized
similarities = keras.ops.matmul(image_features, keras.ops.transpose(text_features))
probabilities = keras.ops.softmax(similarities, axis=-1)

best_match_index = keras.ops.argmax(probabilities, axis=-1)
print(f"Input image is most similar to: '{class_descriptions[best_match_index[0]]}'")
print(f"Probabilities: {probabilities.numpy().flatten()}")
```

### Example 2: Getting Separate Embeddings for Downstream Tasks

You can use CLIP as a powerful feature extractor for other models.

```python
# (Model creation from Example 1)
batch_size = 4
dummy_images = np.random.rand(batch_size, 224, 224, 3).astype("float32")
dummy_texts = ["a red car", "blue sky", "a running dog", "a sleeping cat"]
text_tokens = tokenize(dummy_texts)

# Get L2-normalized embeddings
image_embeddings = model.encode_image(dummy_images)
text_embeddings = model.encode_text(text_tokens)

print(f"Image embeddings shape: {image_embeddings.shape}") # (4, 512)
print(f"Text embeddings shape: {text_embeddings.shape}")   # (4, 512)

# These embeddings can now be used for image retrieval, clustering, etc.
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Fine-tuning on a Specific Dataset

While designed for zero-shot use, CLIP can be fine-tuned. For example, you can train it on a dataset with higher-quality captions or for a specific domain.

```python
# Create a CLIP model
model = create_clip_variant("ViT-B/16")

# You would typically load pre-trained weights here
# model.load_weights('path/to/pretrained_clip_weights.weights.h5')

# Define a custom training step
def contrastive_loss(logits_per_image):
    # Logits are (batch_size, batch_size)
    batch_size = keras.ops.shape(logits_per_image)[0]
    # The correct labels are on the diagonal: [0, 1, 2, ..., batch_size-1]
    labels = keras.ops.arange(batch_size)
    
    # Symmetrical loss
    loss_img = keras.losses.sparse_categorical_crossentropy(
        labels, logits_per_image, from_logits=True
    )
    loss_txt = keras.losses.sparse_categorical_crossentropy(
        labels, keras.ops.transpose(logits_per_image), from_logits=True
    )
    
    return (loss_img + loss_txt) / 2.0

# Compile the model with a dummy loss (or use a custom training loop)
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-5))

# Training loop
# for images, texts in dataset:
#     with tf.GradientTape() as tape:
#         outputs = model({'image': images, 'text': texts}, training=True)
#         loss = contrastive_loss(outputs['logits_per_image'])
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

---

## 10. Performance Optimization

### Mixed Precision Training

CLIP models, being based on Transformers, are excellent candidates for mixed precision training. This uses 16-bit floating-point numbers for many computations, which can significantly speed up training (up to 2-3x) on modern GPUs with Tensor Cores.

```python
# Enable mixed precision globally before creating the model
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = create_clip_variant("ViT-B/16")
model.compile(...)

# When training, use a LossScaleOptimizer to prevent numeric underflow
# Keras's model.fit() handles this automatically.
```

---

## 11. Training and Best Practices

### Optimizer and Schedule

-   **Optimizer**: **AdamW** is highly recommended. The weight decay is a crucial regularizer for Transformer-based models.
-   **Learning Rate Schedule**: A **cosine decay** learning rate schedule, often with a few epochs of linear warmup at the start of training, is standard practice and yields the best results.

### The Importance of Batch Size

-   Contrastive learning benefits immensely from **large batch sizes**. A larger batch provides more "negative" examples for each positive pair, making the learning task more challenging and resulting in better representations. The original CLIP was trained with a batch size of 32,768. For typical hardware, techniques like **gradient accumulation** are necessary to simulate such large batches.

---

## 12. Serialization & Deployment

The `CLIP` model and all its custom layers are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
# Create and train model
model = create_clip_variant("ViT-B/32")
# model.compile(...) and model.fit(...)

# Save the entire model to a single file
model.save('my_clip_model.keras')

# Load the model in a new session, including its architecture and weights.
loaded_model = keras.models.load_model('my_clip_model.keras')
print("✅ CLIP model loaded successfully!")
```

---

## 13. Testing & Validation

### Unit Tests

You can validate the implementation with simple tests to ensure all variants can be created and produce the correct output shapes.

```python
import keras
import numpy as np
from dl_techniques.models.clip.model import CLIP

def test_creation_all_variants():
    """Test model creation for all variants."""
    for variant in CLIP.MODEL_VARIANTS.keys():
        model = CLIP.from_variant(variant)
        assert model is not None
        print(f"✓ CLIP-{variant} created successfully")

def test_forward_pass_shapes():
    """Test the output shapes of a full forward pass and individual encoders."""
    model = CLIP.from_variant("ViT-B/32")
    batch_size = 4
    embed_dim = 512
    
    images = np.random.rand(batch_size, 224, 224, 3)
    texts = np.random.randint(0, 49408, (batch_size, 77))

    # Test full pass
    outputs = model({'image': images, 'text': texts})
    assert outputs['image_features'].shape == (batch_size, embed_dim)
    assert outputs['text_features'].shape == (batch_size, embed_dim)
    assert outputs['logits_per_image'].shape == (batch_size, batch_size)

    # Test individual encoders
    img_feat = model.encode_image(images)
    txt_feat = model.encode_text(texts)
    assert img_feat.shape == (batch_size, embed_dim)
    assert txt_feat.shape == (batch_size, embed_dim)
    print("✓ Forward pass and encoder shapes are correct")

# Run tests
if __name__ == '__main__':
    test_creation_all_variants()
    test_forward_pass_shapes()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Training is unstable or the loss becomes `NaN`.**

-   **Cause 1**: The learning rate may be too high.
-   **Solution 1**: Use a smaller peak learning rate (e.g., `1e-5` to `1e-4`) and implement a linear warmup schedule.
-   **Cause 2**: The learnable temperature `logit_scale` is diverging. The original paper clips its value to prevent it from growing too large.
-   **Solution 2**: In your custom training loop, add `model.logit_scale.assign(ops.clip(model.logit_scale, -np.inf, np.log(100)))`.

### Frequently Asked Questions

**Q: What is the `logit_scale` (temperature) parameter?**

A: It's a learnable scalar that scales the cosine similarities before the cross-entropy loss is calculated. A higher temperature makes the softmax distribution sharper, meaning the model becomes more confident about which pairs match. It effectively controls the dynamic range of the logits and is crucial for stable and effective training.

**Q: Why does CLIP use a Vision Transformer (ViT) instead of a CNN?**

A: The authors found that a ViT was more computationally efficient and scalable for the massive pre-training task than a comparable ResNet-based model. However, the core CLIP concept is architecture-agnostic, and versions using CNNs also exist.

**Q: How do I get a tokenizer for the text encoder?**

A: This implementation defines the model architecture. For practical use, you need a compatible tokenizer. The original CLIP model uses a specific byte-pair encoding (BPE) tokenizer with a vocab size of 49,408. You can typically find compatible tokenizers in libraries like `transformers` from Hugging Face or `ftfy`.

---

## 15. Technical Details

### Modern `TransformerLayer`

This CLIP implementation is built upon a generic, high-quality `TransformerLayer`. This is not a simple textbook implementation; it incorporates several modern improvements for better stability and performance, making it a robust foundation for both encoders:

-   **Normalization**: Uses **RMSNorm** (`normalization_type='rms_norm'`) instead of standard LayerNorm, which can be faster and more stable. Normalization is applied *before* the sub-layer (`normalization_position='pre'`), a key design choice for training very deep Transformers.
-   **FFN Network**: Uses **SwiGLU** (`ffn_type='swiglu'`), a modern activation function in the feed-forward block that often outperforms standard ReLU or GELU.
-   **Attention**: Uses **Grouped-Query Attention** (`attention_type='group_query'`) in the vision encoder. This is an optimization over standard Multi-Head Attention that reduces the memory bandwidth required for the Key and Value projections, leading to faster inference.

### Feature Extraction Strategies

-   **Vision Encoder**: Following the standard ViT practice, a learnable `[CLS]` token is added to the sequence of image patches. The final hidden state corresponding to this token is used as the aggregate representation of the entire image.
-   **Text Encoder**: The model uses the hidden state of the *last token* in the sequence as the representation for the text. This is typically an `[EOT]` (end-of-text) token, which is trained to summarize the meaning of the entire sentence.

---

## 16. Citation

This implementation is based on the official CLIP paper. If you use this model in your research, please consider citing the original work:

```bibtex
@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML)},
  year={2021}
}
```
# Gemma 3: A Keras 3 Implementation of Google's Advanced LLM

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured Keras 3 implementation of **Gemma 3**, Google's next-generation open-source language model. This architecture introduces a sophisticated **dual normalization** pattern and a strategic **mixed attention** mechanism to deliver a state-of-the-art balance of performance and computational efficiency.

---

## Table of Contents

1. [Overview: What is Gemma 3 and Why It Matters](#1-overview-what-is-gemma-3-and-why-it-matters)
2. [The Problem Gemma 3 Solves](#2-the-problem-gemma-3-solves)
3. [How Gemma 3 Works: Core Concepts](#3-how-gemma-3-works-core-concepts)
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

## 1. Overview: What is Gemma 3 and Why It Matters

### What is Gemma 3?

**Gemma 3** is an advanced transformer-based language model that incorporates several architectural innovations to push the boundaries of performance and efficiency. It refines the standard transformer block with a unique structure designed for improved gradient flow and model stability.

### Key Innovations

1.  **Dual Normalization**: Unlike traditional pre-norm or post-norm architectures, Gemma 3 applies RMS Normalization **both before and after** the attention and feed-forward sub-layers. This is theorized to combine the training stability of pre-norm with the representational fidelity of post-norm.
2.  **Mixed Attention Strategy**: The model strategically mixes computationally expensive `full_attention` layers with efficient `sliding_window` attention layers. This allows the model to maintain long-range dependency modeling capabilities while significantly reducing the computational and memory footprint for most layers.
3.  **Grouped-Query Attention (GQA)**: It utilizes GQA to reduce the memory bandwidth requirements of the attention mechanism without a significant drop in quality, making it faster and more memory-efficient during inference.

### Why Gemma 3 Matters

**The LLM Scaling Problem**:
```
Problem: Build larger, more capable models without an exponential increase in compute cost.
Traditional Approach:
  1. Scale up all layers uniformly (e.g., full attention everywhere).
  2. Limitation: The quadratic complexity of full self-attention becomes a major
     bottleneck for long sequences, limiting context length and increasing costs.
  3. Result: A direct trade-off between model capability (context length) and
     practicality (training/inference cost).
```

**Gemma 3's Solution**:
```
Gemma 3's Approach:
  1. Use computationally cheaper sliding window attention for most layers to process
     local context efficiently.
  2. Strategically place full attention layers at specific intervals to act as
     "global information mixers," ensuring long-range dependencies are not lost.
  3. Refine the block architecture with dual normalization for better training dynamics.
  4. Benefit: Achieves the performance of a full-attention model with the efficiency
     closer to that of a local-attention model, enabling longer context windows
     at a manageable cost.
```

---

## 2. The Problem Gemma 3 Solves

### The Trilemma: Performance vs. Speed vs. Context Length

Modern language models face a constant balancing act. Increasing context length with full attention is quadratically expensive, slowing down training and inference and demanding huge amounts of memory.

```
┌─────────────────────────────────────────────────────────────┐
│  The LLM Computational Trilemma                             │
│                                                             │
│  Full Attention Models:                                     │
│    - High Performance: Can model any token-to-token         │
│      relationship.                                          │
│    - Very Slow & Memory-Intensive: O(n²) complexity makes   │
│      long sequences (n > 4k) prohibitively expensive.       │
│                                                             │
│  Local/Linear Attention Models:                             │
│    - Fast & Efficient: Scale linearly or near-linearly.     │
│    - Lower Performance: Struggle to capture long-range      │
│      dependencies, which are critical for many tasks.       │
└─────────────────────────────────────────────────────────────┘
```

Gemma 3's mixed attention strategy offers a pragmatic and effective compromise, delivering a "best of both worlds" solution.

---

## 3. How Gemma 3 Works: Core Concepts

### The Architectural Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Gemma 3 Complete Data Flow                     │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: EMBEDDING
─────────────────
Input Token IDs (Batch, SeqLen)
    │
    ├─► Token Embedding Layer
    │
    ├─► Multiply by √(hidden_size)
    │
    └─► Input Embeddings


STEP 2: TRANSFORMER BLOCKS (Repeated N times)
─────────────────────────────────────────────
Input to Block `i`
    │
    ├─► Gemma3TransformerBlock
    │   (Dual normalization, mixed attention type)
    │
    └─► Output of Block `i`


STEP 3: FINAL PROJECTION
────────────────────────
Final Hidden State
    │
    ├─► Final RMS Normalization
    │
    ├─► Linear Layer (LM Head)
    │
    └─► Logits (Batch, SeqLen, VocabSize)
```

---

## 4. Architecture Deep Dive

### 4.1 `Gemma3TransformerBlock`

-   **Purpose**: The core computational unit of the model. It deviates from the standard transformer block to improve performance and stability.
-   **Dual Normalization**: This is the defining feature.
    -   **Attention Path**: `x + PostAttnNorm(Attention(InputNorm(x)))`
    -   **FFN Path**: `x + PostFFNNorm(FFN(PreFFNNorm(x)))`
    -   This structure helps maintain a clean residual stream while ensuring the inputs to the attention and FFN layers are well-scaled.

```
┌──────────────────────────────────────────────────────────────────┐
│                     Gemma 3 Transformer Block                    │
│                                                                  │
│                               Input                              │
│                           ╱          ╲                           │
│      InputLayerNorm ──► Attention ──► PostAttnLayerNorm ───► Add │
│           │                                                      │ │
│           └─────────────────────────── Residual ─────────────────┘ │
│                                        │                         │
│                           ╱            ╲                         │
│    PreFFNLayerNorm ───► FFN ──────────► PostFFNNorm ────────► Add │
│           │                                                      │ │
│           └─────────────────────────── Residual ─────────────────┘ │
│                                                                  │
│                               Output                             │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Mixed Attention

-   The `layer_types` parameter in the model config determines the attention mechanism for each block.
-   **`sliding_window`**: Each token can only attend to a fixed number of preceding tokens (e.g., 512). This is computationally efficient (`O(n*w)` where `w` is window size).
-   **`full_attention`**: Each token can attend to all previous tokens in the sequence. This is computationally expensive (`O(n^2)`) but necessary for mixing information across the entire sequence.

---

## 5. Quick Start Guide

### Installation

```bash
# Install Keras 3 and a backend (e.g., tensorflow)
pip install keras tensorflow numpy
```

### Your First Gemma 3 Model (30 seconds)

Let's create a tiny Gemma 3 model for text generation.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.gemma3.gemma3 import create_gemma3

# 1. Create a tiny Gemma 3 model using a pre-configured variant
# This factory function handles all the setup.
model = create_gemma3(
    "tiny",  # Use the 'tiny' variant
    task_type="generation"
)

# 2. Compile the model
model.compile(
    optimizer="adamw",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
print("✅ Gemma 3 model created and compiled successfully!")
model.summary()

# 3. Create dummy data for a forward pass
batch_size = 2
seq_len = 64
vocab_size = model.layers[1].vocab_size # Get vocab size from the backbone

dummy_input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
dummy_attention_mask = np.ones((batch_size, seq_len))
dummy_labels = np.random.randint(0, vocab_size, (batch_size, seq_len))

# 4. Train for one step
loss, acc = model.train_on_batch(
    [dummy_input_ids, dummy_attention_mask], dummy_labels
)
print(f"\n✅ Training step complete! Loss: {loss:.4f}, Accuracy: {acc:.4f}")
```

---

## 6. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`Gemma3`** | `...gemma3.gemma3.Gemma3` | The main Keras `Model` class that assembles the architecture. |
| **`Gemma3TransformerBlock`** | `...gemma3.components.Gemma3TransformerBlock` | The core dual-norm transformer block. |
| **`create_gemma3`** | `...gemma3.gemma3.create_gemma3` | High-level factory to create models for different tasks. |
| **`..._generation`** | `...gemma3.gemma3.create_gemma3_generation` | Factory specifically for text generation. |
| **`..._classification`** | `...gemma3.gemma3.create_gemma3_classification`| Factory specifically for sequence classification. |

---

## 7. Configuration & Model Variants

This implementation provides several pre-configured variants based on the official Gemma 3 specifications.

| Variant | Hidden Size | Layers | Attn Heads | FFN Hidden | Max Seq Len |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **`270m`** | `640` | 18 | 4 | 2048 | 32768 |
| **`small`**| `512` | 12 | 8 | 1536 | 8192 |
| **`tiny`** | `384` | 6 | 6 | 1024 | 4096 |

---

## 8. Comprehensive Usage Examples

### Example 1: Sequence Classification

Use the dedicated factory function to add a classification head on top of the Gemma 3 backbone.

```python
# Create a model for a 3-class classification problem
model = create_gemma3(
    "tiny",
    task_type="classification",
    num_labels=3,
    pooling_strategy="mean" # Use mean pooling over the sequence
)

# Compile with appropriate loss for classification
model.compile(
    optimizer="adamw",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
model.summary()
```

### Example 2: Building from a Custom Configuration

Instead of using a variant, you can define your own architecture.

```python
custom_config = {
    "vocab_size": 10000,
    "hidden_size": 256,
    "num_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "ffn_hidden_size": 768,
    "max_seq_len": 1024,
    "sliding_window_size": 256,
    "layer_types": [
        "sliding_window", "sliding_window", 
        "full_attention", "full_attention"
    ],
}

# Create a generation model with this custom config
model = create_gemma3(custom_config, task_type="generation")
```

---

## 9. Advanced Usage Patterns

### Understanding Mixed Attention Strategy

The `layer_types` list is a powerful tool for balancing performance and efficiency. A common strategy is to interleave `full_attention` layers between blocks of `sliding_window` layers.

-   **`sliding_window` layers**: These act as efficient local feature extractors, processing information within a fixed context. The bulk of the network should use these.
-   **`full_attention` layers**: These act as global "information routers." A token at a full attention layer can access information from any previous token, allowing local features extracted by earlier layers to be integrated and shared across the entire sequence. Placing them every 4-6 layers is a common and effective pattern.

---

## 10. Performance Optimization

-   **Mixed Precision**: Gemma 3 is fully compatible with `mixed_float16` or `bfloat16` training, which can provide significant speedups on modern GPUs and TPUs.
-   **Flash Attention**: For even greater performance, the underlying `GroupedQueryAttention` layer can be swapped with a Flash Attention implementation (if available for your hardware and backend).
-   **Gradient Checkpointing**: For training very large models or with extremely long sequences, enabling gradient checkpointing can trade compute for a massive reduction in memory usage.

---

## 11. Training and Best Practices

-   **Optimizer**: **AdamW** is the standard and recommended optimizer for training transformers.
-   **Learning Rate Schedule**: A **cosine decay** with a linear warmup phase is critical for stable training.
-   **Tokenizer**: The `vocab_size` must match the vocabulary of the tokenizer you are using.
-   **Padding**: Ensure your input data is correctly padded and that you provide an `attention_mask` to the model so it knows which tokens to ignore.

---

## 12. Serialization & Deployment

The `Gemma3` model and its custom `Gemma3TransformerBlock` are fully serializable using Keras 3's modern `.keras` format, thanks to the `@keras.saving.register_keras_serializable()` decorator.

### Saving and Loading

```python
# Create and train model
model = create_gemma3("tiny", task_type="generation")
# ... model.fit(...)

# Save the entire model to a single file
model.save('my_gemma3_model.keras')

# Load the model, including custom layers, in a new session
loaded_model = keras.models.load_model('my_gemma3_model.keras')
print("✅ Gemma 3 model loaded successfully!")
```

---

## 13. Testing & Validation

A `pytest` test to ensure the critical serialization cycle is robust.

```python
import pytest
import numpy as np
import keras
import tempfile
import os
from dl_techniques.models.gemma3.gemma3 import Gemma3

def test_gemma3_serialization_cycle():
    """CRITICAL TEST: Ensures a model can be saved and loaded."""
    model = Gemma3.from_variant("tiny")
    dummy_input = np.random.randint(0, model.vocab_size, (2, 16))
    
    original_prediction = model(dummy_input)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_model.keras")
        model.save(filepath)
        loaded_model = keras.models.load_model(filepath)

    loaded_prediction = loaded_model(dummy_input)
    np.testing.assert_allclose(
        original_prediction, loaded_prediction, rtol=1e-5, atol=1e-5
    )
    print("✓ Serialization cycle test passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Out-of-Memory (OOM) error during training.**

-   **Cause**: Language models are extremely memory-intensive, especially with long sequences.
-   **Solution**:
    1.  Reduce the `batch_size`.
    2.  Enable mixed precision (`keras.mixed_precision.set_global_policy('mixed_float16')`).
    3.  Reduce `max_seq_len` if possible.
    4.  Use a smaller variant (e.g., `tiny` instead of `small`).

### Frequently Asked Questions

**Q: How is this different from a standard Transformer or Llama?**

A: The key differentiators are the **dual normalization** pattern within the transformer block and the built-in support for **mixed attention types** (`sliding_window` and `full`). While Llama uses pre-norm, and the original transformer used post-norm, Gemma 3's dual-norm is a unique hybrid.

**Q: Can I use this for fine-tuning?**

A: Yes. The provided factory functions are ideal for fine-tuning. You can load pre-trained weights into the `gemma3_backbone` and train a new classification head, or continue training the entire model on a specific downstream task.

---

## 15. Technical Details

### Dual Normalization Formulation

For each `Gemma3TransformerBlock`, the computation proceeds as follows:

1.  **Attention Path**:
    `x_norm1 = InputNorm(x)`
    `attn_out = Attention(x_norm1)`
    `attn_out_norm = PostAttnNorm(attn_out)`
    `x = x + attn_out_norm`

2.  **FFN Path**:
    `x_norm2 = PreFFNNorm(x)`
    `ffn_out = FFN(x_norm2)`
    `ffn_out_norm = PostFFNNorm(ffn_out)`
    `output = x + ffn_out_norm`

This structure ensures that the inputs to the computationally heavy layers (Attention, FFN) are always normalized, promoting stability, while the residual connections operate on the normalized outputs.

---

## 16. Citation

This implementation is based on the architecture of Google's Gemma family of models. If you use this work, please consider citing the official Google research.

-   **Gemma Official Page**: [ai.google/gemma](https://ai.google/gemma/)
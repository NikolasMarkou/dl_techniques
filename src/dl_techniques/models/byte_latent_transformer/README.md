# BLT: Byte Latent Transformer

[![Keras 3](https://img.shields.io/badge/Keras-3.8.0-red.svg)](https://keras.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![dl_techniques](https://img.shields.io/badge/dl__techniques-Standard-green.svg)](https://github.com/your-repo/dl_techniques)
[![arXiv](https://img.shields.io/badge/arXiv-2412.09871-b31b1b.svg)](https://arxiv.org/abs/2412.09871)

A production-ready, fully-featured implementation of the **Byte Latent Transformer (BLT)** built on the **dl_techniques** framework using **Keras 3**. This architecture represents a paradigm shift in Large Language Models (LLMs), moving away from fragile tokenizers to a robust, dynamic, byte-level processing pipeline.

This implementation leverages `dl_techniques` factories to provide state-of-the-art stability features, including **Zero-Centered RMSNorm**, **SwiGLU FFNs**, and **Grouped Query Attention**, matching the performance of token-based models (like Llama 3) while offering superior robustness and up to **50% greater inference efficiency** via dynamic patching.

---

## Table of Contents

1. [Overview: What is BLT and Why It Matters](#1-overview-what-is-blt-and-why-it-matters)
2. [The Problem BLT Solves](#2-the-problem-blt-solves)
3. [How BLT Works: Core Concepts](#3-how-blt-works-core-concepts)
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

## 1. Overview: What is BLT and Why It Matters

### What is BLT?

**BLT** is a tokenization-free LLM architecture. Instead of relying on a fixed vocabulary, it processes raw UTF-8 bytes. To handle the computational cost of long byte sequences, BLT uses **Dynamic Entropy-Based Patching**. A small "Entropy Model" scans bytes; when information density is high, BLT allocates more compute. When predictable, bytes are compressed into single patches.

### Key Innovations of this Implementation

1.  **dl_techniques Integration**: The Global Transformer is built using the `TransformerLayer` factory, enabling seamless swapping of attention mechanisms (e.g., `group_query`) and normalization layers.
2.  **Dynamic Patcher**: A dedicated layer using factory-standardized entropy calculation to segment byte sequences.
3.  **Modern Components**: Utilizes `swiglu` for FFNs and `zero_centered_rms_norm` for maximum training stability, adhering to framework best practices.
4.  **Backend Agnostic**: Pure Keras 3 code compatible with JAX, TensorFlow, and PyTorch backends.

### Why BLT Matters

**Token-Based LLMs**:
```
Model: Fixed Tokenization
  1. Brittle: Fails on typos, adversarial noise, or rare words.
  2. Biased: Vocabulary favors high-resource languages.
  3. Rigid: Cannot adapt compute allocation.
```

**BLT Solution**:
```
Model: Byte Latent Transformer
  1. Robust: Reads raw bytes; immune to OOV errors.
  2. Universal: Works natively with any UTF-8 string, code, or DNA.
  3. Efficient: Allocates compute dynamically based on information entropy.
```

---

## 2. The Problem BLT Solves

### The Tokenization Bottleneck

Standard LLMs rely on separate pre-processing (BPE, WordPiece), introducing:
*   **Out-of-Vocabulary (OOV) issues**: Unseen words become meaningless fragments.
*   **Multilingual inequity**: Non-Latin scripts require more tokens per word.
*   **Lack of robustness**: Single character flips confuse the tokenizer.

### The Compute Allocation Problem

Standard Transformers spend equal FLOPs on "the" and "quantum". BLT's **Dynamic Patching** ensures the compute budget is spent where information density is highest, leveraging the `dl_techniques` efficient attention implementations.

---

## 3. How BLT Works: Core Concepts

### The Processing Pipeline

1.  **Local Encoding**: Raw bytes are processed by a lightweight transformer. An entropy model determines patch boundaries.
2.  **Global Latent Processing**: Patches are fed into a deep Global Transformer (constructed via `TransformerLayer`).
3.  **Local Decoding**: Global context is projected back to predict the next byte.

```
Raw Bytes 
   │
   ▼
[Entropy Model] ──► (Computes Information Density)
   │
   ▼
[Dynamic Patcher] ──► (Groups bytes into variable chunks)
   │
   ▼
[Local Encoder] ──► (Compresses chunks into Vectors)
   │
   ▼
[Global Transformer] ──► (Reasons using SwiGLU & GQA)
   │
   ▼
[Local Decoder] ──► (Generates next byte)
```

---

## 4. Architecture Deep Dive

### 4.1 The Entropy Model & Patcher
*   **EntropyModel**: A lightweight causal transformer.
*   **Decision Rule**: Uses Shannon entropy thresholding.
*   **Implementation**: Fully configurable via `dl_techniques` config dictionaries.

### 4.2 Local Encoder
*   **Pooling**: Compresses $N$ bytes into $D$ latent vectors using Cross-Attention pooling.
*   **Normalization**: Uses `zero_centered_rms_norm` for input stability.

### 4.3 Global Transformer
*   **Core**: Built using `dl_techniques.layers.transformers.transformer.TransformerLayer`.
*   **Attention**: Defaults to `group_query` (GQA) for efficient KV caching.
*   **FFN**: Defaults to `swiglu` (Swish-Gated Linear Unit) for superior performance over standard ReLU MLPs.
*   **Norm**: Pre-norm architecture with `zero_centered_rms_norm`.

### 4.4 Local Decoder
*   **Operation**: Predicts the next byte by querying the Global Patch Context.
*   **Cross-Attention**: Bytes attend to patch representations.

---

## 5. Quick Start Guide

### Your First BLT Model (30 seconds)

Initialize a standard BLT model using the configuration-driven pattern.

```python
import keras
from dl_techniques.models.blt import create_blt_from_config

# 1. Define Configuration
blt_config = {
    "variant": "base",
    "vocab_size": 260, # 256 bytes + 4 special tokens
    "max_sequence_length": 1024,
    "global_transformer_args": {
        "attention_type": "group_query",
        "normalization_type": "zero_centered_rms_norm",
        "ffn_type": "swiglu"
    }
}

# 2. Create Model
model = create_blt_from_config(blt_config)

# 3. Compile (using framework standard)
model.compile(
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 4. Generate Text
prompt = "The future of AI is"
generated = model.generate(
    prompt, 
    max_new_tokens=50, 
    temperature=0.7
)

print(f"Generated: {generated}")
```

---

## 6. Component Reference

### 6.1 `ByteLatentTransformer`

The main model class integrates `dl_techniques` factories.

```python
from dl_techniques.models.blt import ByteLatentTransformer
from dl_techniques.layers.norms import create_normalization_layer

# Manual instantiation (advanced)
model = ByteLatentTransformer(
    local_dim=512,
    global_dim=768,
    num_global_layers=12,
    global_attention_config={"type": "group_query", "num_heads": 8},
    global_norm_type="zero_centered_rms_norm"
)
```

### 6.2 Factory Integration

The model internally calls:
*   `dl_techniques.layers.attention.create_attention_layer`
*   `dl_techniques.layers.ffn.factory.create_ffn_from_config`
*   `dl_techniques.layers.norms.create_normalization_layer`

---

## 7. Configuration & Model Variants

Standard configurations based on scaling laws. Note that `intermediate_size` typically follows the Llama-style 3.5x expansion when using SwiGLU.

| Variant | Local Dim | Global Dim | Global Heads | Global FFN Type | Params |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **`micro`** | 256 | 384 | 6 | `swiglu` | ~15M |
| **`small`** | 512 | 768 | 12 | `swiglu` | ~125M |
| **`base`** | 768 | 1024 | 16 | `swiglu` | ~350M |
| **`large`** | 1024 | 1536 | 24 | `swiglu` | ~1.3B |

---

## 8. Comprehensive Usage Examples

### Example 1: Training with Optimization Builder

Use the `dl_techniques.optimization` module for proper scheduling and weight decay.

```python
import keras
import numpy as np
from dl_techniques.models.blt import create_blt_from_config
from dl_techniques.optimization import (
    optimizer_builder, 
    learning_rate_schedule_builder
)

# 1. Setup Model
model = create_blt_from_config({"variant": "small"})

# 2. Configure Optimization (Standard Transformer Setup)
lr_config = {
    "type": "cosine_decay",
    "warmup_steps": 1000,
    "warmup_start_lr": 1e-7,
    "learning_rate": 3e-4,
    "decay_steps": 50000,
    "alpha": 0.1
}

opt_config = {
    "type": "adamw",
    "weight_decay": 0.01,
    "gradient_clipping_by_norm": 1.0, # Essential for stability
    "beta_1": 0.9,
    "beta_2": 0.95
}

# 3. Build & Compile
lr_schedule = learning_rate_schedule_builder(lr_config)
optimizer = optimizer_builder(opt_config, lr_schedule)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# 4. Train
# x_train: (batch, seq_len) int32 bytes
model.fit(x_train, y_train, batch_size=32, epochs=10)
```

### Example 2: Inspecting Dynamic Patches

```python
tokenizer = model.tokenizer
text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.text_to_bytes(text)
tokens_tensor = keras.ops.convert_to_tensor([tokens])

# Access internal components
entropy_logits = model.entropy_model(tokens_tensor)
entropy = model.entropy_model.compute_entropy(entropy_logits)
patch_lengths = model.patcher(entropy)

print(f"Entropy Map: {entropy[0].numpy()}")
print(f"Patch Lengths: {patch_lengths[0].numpy()}")
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Using Mixture of Experts (MoE)

You can replace the Global Transformer's FFN with an MoE layer via configuration.

```python
blt_config = {
    "variant": "base",
    "global_transformer_args": {
        "moe_config": {
            "num_experts": 8,
            "expert_config": {"ffn_config": {"type": "swiglu", "output_dim": 1024}},
            "gating_config": {"gating_type": "linear", "top_k": 2}
        }
    }
}
model = create_blt_from_config(blt_config)
```

### Pattern 2: Custom Normalization

For extremely deep models, switch to `adaptive_band_rms`.

```python
blt_config = {
    "variant": "large",
    "global_transformer_args": {
        "normalization_type": "adaptive_band_rms"
    }
}
```

---

## 10. Performance Optimization

### Inference Efficiency
*   **Higher Entropy Threshold**: Results in larger patches and faster inference.
*   **Grouped Query Attention**: Configured by default in `base` and larger models to reduce KV cache size.

### Mixed Precision
Always recommended for the Global Transformer.

```python
keras.mixed_precision.set_global_policy('mixed_float16')
```

---

## 11. Training and Best Practices

### Data Preprocessing
*   Input: Raw UTF-8 bytes + offset (Standard: +4 to reserve 0-3 for special tokens).
*   No normalization needed.

### Hyperparameters
*   **Optimizer**: Always use `adamw` with `weight_decay=0.01` or `0.1`.
*   **Clipping**: Ensure `gradient_clipping_by_norm` is set to `1.0`.
*   **Warmup**: Mandatory for transformer stability (1000+ steps).

---

## 12. Serialization & Deployment

BLT models utilize `keras.saving.register_keras_serializable` for full support.

```python
# Save complete model including config
model.save("blt_base.keras")

# Load (Restores all dl_techniques components)
loaded_model = keras.models.load_model("blt_base.keras")
```

---

## 13. Testing & Validation

Run consistency checks to ensure dynamic patching logic works on your backend.

```python
def test_blt_forward():
    model = create_blt_from_config({"variant": "micro", "max_sequence_length": 128})
    x = keras.random.randint((1, 128), 4, 260)
    
    # Forward pass
    y = model(x)
    assert y.shape == (1, 128, 260)
    print("✓ Forward pass shape correct")

test_blt_forward()
```

---

## 14. Troubleshooting & FAQs

**Q: Why use `swiglu` instead of `mlp`?**
*   **A:** SwiGLU (Swish-Gated Linear Unit) is the current standard for LLMs (used in Llama, Mistral) offering better convergence than standard ReLU MLPs.

**Q: Can I use `layer_norm` instead of `zero_centered_rms_norm`?**
*   **A:** Yes, by changing the config `normalization_type`. However, `zero_centered_rms_norm` provides superior stability for deep architectures.

**Q: My loss is NaN.**
*   **A:** Check if `gradient_clipping_by_norm` is enabled in your `optimizer_builder` config. It is required for BLT.

---

## 15. Technical Details

### Entropy-Based Segmentation
Patching is driven by Shannon entropy $H(x_t) = -\sum P(x_t) \log P(x_t)$. High entropy triggers boundary creation.

### Global Processing
The Global Transformer uses `TransformerLayer` blocks:
$$ \text{Out} = \text{SwiGLU}(\text{Norm}(x)) + x $$
$$ x = \text{GQA}(\text{Norm}(x)) + x $$
Where $\text{Norm}$ is typically `ZeroCenteredRMSNorm`.

---

## 16. Citation

This implementation is based on the paper by Pagnoni et al., adapted for the `dl_techniques` framework:

```bibtex
@article{pagnoni2024byte,
  title={Byte Latent Transformer: Patches Scale Better Than Tokens},
  author={Pagnoni, Artidoro and others},
  journal={arXiv preprint arXiv:2412.09871},
  year={2024}
}
```

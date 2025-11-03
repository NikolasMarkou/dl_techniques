# ModernBERT: A High-Performance BERT Successor

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

An advanced, production-ready Keras 3 implementation of **ModernBERT**, a successor to the classic BERT architecture. This model is based on the paper "[Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663)". ModernBERT integrates a suite of contemporary deep learning techniques to deliver superior performance, faster processing for long contexts, and enhanced training stability.

This implementation is designed as a pure foundation model, separating the core encoding logic from task-specific heads. It follows the `dl_techniques` framework standards and modern Keras 3 best practices, providing a modular, well-documented, and fully serializable model that works seamlessly across TensorFlow, PyTorch, and JAX backends.

---

## Table of Contents

1.  [Overview: What is ModernBERT and Why It Matters](#1-overview-what-is-modernbert-and-why-it-matters)
2.  [The Problem ModernBERT Solves](#2-the-problem-modernbert-solves)
3.  [How ModernBERT Works: Core Concepts](#3-how-modernbert-works-core-concepts)
4.  [Architecture Deep Dive](#4-architecture-deep-dive)
5.  [Quick Start Guide](#5-quick-start-guide)
6.  [Component Reference](#6-component-reference)
7.  [Configuration & Model Variants](#7-configuration--model-variants)
8.  [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9.  [Advanced Usage Patterns](#9-advanced-usage-patterns)
10. [Performance Optimization](#10-performance-optimization)
11. [Training and Best Practices](#11-training-and-best-practices)
12. [Serialization & Deployment](#12-serialization--deployment)
13. [Testing & Validation](#13-testing--validation)
14. [Troubleshooting & FAQs](#14-troubleshooting--faqs)
15. [Technical Details](#15-technical-details)
16. [Citation](#16-citation)

---

## 1. Overview: What is ModernBERT and Why It Matters

### What is ModernBERT?

**ModernBERT** is a modernized bidirectional encoder that represents a major Pareto improvement over classic BERT-style models. It integrates state-of-the-art techniques from recent large language models to deliver superior performance, efficiency, and native long-context capabilities up to **8192 tokens**. Trained on **2 trillion tokens** of diverse data, including code, it sets a new standard for encoder-only models.

### Key Innovations

1.  **Rotary Positional Embeddings (RoPE)**: Replaces traditional absolute positional embeddings with RoPE, which is proven to excel in both short- and long-context scenarios and allows for easier context extension.
2.  **Pre-Layer Normalization (Pre-LN)**: Applies layer normalization *before* attention and feed-forward blocks, significantly improving training stability and convergence.
3.  **GeGLU Activation Function**: Uses a Gated GELU (GeGLU) in the feed-forward network, which provides a more sophisticated gating mechanism for improved performance.
4.  **Alternating Local & Global Attention**: Employs an efficient hybrid attention strategy. Most layers use computationally cheaper **windowed (local) attention**, while periodic **global attention** layers (every 3rd layer) ensure that long-range dependencies are captured. This is crucial for its 8192 native sequence length.
5.  **Bias-Free Layers**: Removes bias parameters from most linear and normalization layers to optimize the parameter budget and improve stability.
6.  **Modern Training Recipe**: Trained on 2 trillion tokens with a modern BPE tokenizer, a modified trapezoidal learning rate schedule, and advanced optimizers like StableAdamW.

### Why ModernBERT Matters

**Classic BERT Problem**:
```
Problem: Understand a long document (e.g., 4096 tokens).
Classic BERT Approach:
  1. Use global self-attention in every layer.
  2. Limitation: Self-attention has O(N²) complexity, making it prohibitively
     slow and memory-intensive for long inputs. Models are often
     limited to 512 tokens.
  3. Result: Unsuitable for long-document analysis, RAG, or high-resolution
     code understanding. Many production pipelines still rely on these
     older, inefficient models.
```

**ModernBERT's Solution**:
```
ModernBERT Approach:
  1. Replace most global attention with efficient windowed attention.
  2. Insert global attention layers periodically to aggregate information.
  3. Use RoPE for robust positional information up to 8192 tokens.
  4. Train on a massive, modern dataset including code.
  5. Benefit: Achieves state-of-the-art performance with near-linear complexity,
     making it a versatile and highly efficient backbone for modern NLP tasks.
```

### Real-World Impact

ModernBERT is an excellent choice for a wide range of NLP tasks where performance, efficiency, and long-context are key:

-   📚 **Long-Document Understanding**: State-of-the-art on retrieval and classification tasks.
-   🔍 **Semantic Search & RAG**: A powerful and fast encoder for retrieval-augmented generation.
-   **💻 Code Analysis**: Outperforms previous encoders on code-related benchmarks due to its training data and modern tokenizer.
-   **Standard NLP Tasks**: A drop-in, superior replacement for BERT and its variants on tasks like NER and GLUE.

---

## 2. The Problem ModernBERT Solves

### The Stagnation of Encoder Models

While decoder-only LLMs have seen rapid innovation, the encoder models that power many production pipelines (e.g., for retrieval, classification) have seen limited improvements since BERT's release. Practitioners have been stuck with models that have short context windows, suboptimal architectures, and were trained on outdated, narrow datasets.

```
┌─────────────────────────────────────────────────────────────┐
│              The Dilemma of Production NLP                  │
│                                                             │
│  Classic Encoders (e.g., BERT, RoBERTa):                    │
│    - Limited to 512 tokens.                                 │
│    - Inefficient O(N²) attention.                           │
│    - Trained on limited data (e.g., BookCorpus, Wikipedia). │
│    - Suboptimal components (Post-LN, GELU, abs. embeddings).│
│                                                             │
│  The Need:                                                  │
│    - An encoder with native long-context capabilities.      │
│    - A modern architecture that is faster and more stable.  │
│    - Pre-trained on a large, diverse, and recent dataset.   │
└─────────────────────────────────────────────────────────────┘
```

ModernBERT directly confronts these issues by creating a new encoder from the ground up, incorporating the best practices from modern LLMs to deliver a major Pareto improvement in the encoder space.

---

## 3. How ModernBERT Works: Core Concepts

### The Bidirectional Encoder, Reimagined

ModernBERT retains the core bidirectional Transformer encoder structure but overhauls its components. The most significant changes are the introduction of RoPE and the hybrid attention mechanism.

```
┌───────────────────────────────────────────────────────────────────┐
│                 ModernBERT Architecture Stages                    │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input IDs ────► ┌────────────────────┐                           │
│                  │ ModernBertEmbeddings │ (Adds RoPE)             │
│                  └──────────┬───────────┘                         │
│                             │                                     │
│                  ┌──────────▼───────────┐                         │
│                  │ Transformer (Local)  │ (Windowed Attn, Pre-LN) │
│                  └──────────┬───────────┘                         │
│                             │                                     │
│                  ┌──────────▼───────────┐                         │
│                  │ Transformer (Local)  │ (Windowed Attn, Pre-LN) │
│                  └──────────┬───────────┘                         │
│                             │                                     │
│                  ┌──────────▼───────────┐                         │
│                  │ Transformer (Global) │ (Global Attn, Pre-LN)   │
│                  └──────────┬───────────┘                         │
│                             │                                     │
│                             ▼ (Repeat...)                         │
│                             │                                     │
│                  ┌──────────▼───────────┐                         │
│                  │  Final Layer Norm    │                         │
│                  └──────────────────────┘                         │
│                             │                                     │
│                             ▼                                     │
│                   Output Hidden States                            │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### The Complete Data Flow (within a Transformer Layer)

The use of **Pre-Layer Normalization** is a critical change from classic BERT, leading to more stable training.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   ModernBERT Pre-LN Transformer Block                    │
└──────────────────────────────────────────────────────────────────────────┘

Input (from previous layer)
  │
  ├─► Residual Path 1 ───────────────────────────────────────────┐
  │                                                              │
  └─► LayerNorm ──► Attention ──► Dropout ──► Add & Norm ◄───────┘
                      (Local/Global)               │
                                                   │
  ┌────────────────────────────────────────────────┘
  │
  ├─► Residual Path 2 ───────────────────────────────┐
  │                                                  │
  └─► LayerNorm ──► GeGLU FFN ──► Dropout ──► Add & Norm ◄─────┘
                                                     │
                                                     ▼
                                                Final Output
```

---

## 4. Architecture Deep Dive

### 4.1 `ModernBertEmbeddings` with RoPE

-   **Purpose**: To convert input token IDs into dense vector representations.
-   **Components**:
    1.  **Token Embeddings**: A standard lookup table for the modern BPE vocabulary (50,368 tokens).
    2.  **Rotary Positional Embeddings (RoPE)**: Unlike learned positional embeddings, RoPE modifies the queries and keys in the attention mechanism to inject relative positional information. This is a key enabler of the long context window.
    3.  **Token Type Embeddings**: A learnable embedding to distinguish between different sentences. (Note: While present for BERT compatibility, ModernBERT does not use token type IDs during pre-training).
-   **Functionality**: The token and token type embeddings are summed, followed by a `LayerNormalization` and `Dropout` layer. RoPE is applied within the attention layers.

### 4.2 `TransformerLayer` with Pre-LN and GeGLU

-   **Purpose**: The core building block of the encoder.
-   **Architecture**:
    1.  **Pre-Normalization**: The input is first passed through a `LayerNormalization` layer.
    2.  **Multi-Head Attention**: The normalized input is fed into the attention mechanism (either windowed or global), where RoPE is applied.
    3.  **First Residual Connection**: The output of the attention block is added back to the original input.
    4.  **Second Pre-Normalization**: The result of the first residual connection is normalized again.
    5.  **GeGLU Feed-Forward Network**: The normalized result is processed by a gated feed-forward network.
    6.  **Second Residual Connection**: The output of the FFN is added back to its input.

### 4.3 Hybrid Attention

-   **Windowed (Local) Attention**: Most layers (2 out of every 3) use windowed attention, where each token can only attend to a fixed-size window of 128 tokens. This reduces complexity from `O(N²)` to `O(N * W)`.
-   **Global Attention**: Every 3rd layer uses standard global attention. This allows information to be exchanged across the entire 8192-token sequence, ensuring that long-range dependencies are captured.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First ModernBERT Model (30 seconds)

Let's build a `base` ModernBERT and pass some dummy data through it.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.modern_bert.modern_bert import ModernBERT

# 1. Create a ModernBERT-base model
model = ModernBERT.from_variant("base")

# 2. Compile the model (optional for inference)
model.compile(optimizer="adam")
print("✅ ModernBERT model created and compiled successfully!")
model.summary()

# 3. Create dummy data (batch size 2, sequence length 256)
dummy_inputs = {
    "input_ids": np.random.randint(0, 50368, (2, 256)),
    "attention_mask": np.ones((2, 256), dtype="int32"),
    "token_type_ids": np.zeros((2, 256), dtype="int32"),
}

# 4. Run inference
outputs = model(dummy_inputs)

# 5. Inspect the output
print(f"\nOutput keys: {outputs.keys()}")
print(f"Shape of last_hidden_state: {outputs['last_hidden_state'].shape}")
# Expected output: (2, 256, 768)
```

---

## 6. Component Reference

### 6.1 Model Classes and Creation Functions

| Component                      | Location                                     | Purpose                                                                          |
| :----------------------------- | :------------------------------------------- | :------------------------------------------------------------------------------- |
| **`ModernBERT`**               | `...nlp.modern_bert.ModernBERT`              | The main Keras `Model` for the foundation encoder.                               |
| **`create_modern_bert_with_head`** | `...nlp.modern_bert.create_modern_bert_with_head` | Recommended factory function to combine a `ModernBERT` with a task-specific head. |

### 6.2 Core Building Blocks

| Layer                  | Location                                             | Purpose                                                                             |
| :--------------------- | :--------------------------------------------------- | :---------------------------------------------------------------------------------- |
| **`ModernBertEmbeddings`** | `...embedding.modern_bert_embeddings.ModernBertEmbeddings` | Handles the initial embedding lookup and normalization.                             |
| **`TransformerLayer`** | `...layers.transformers.TransformerLayer`            | The highly configurable, modern Transformer block that powers the encoder.          |
| **`create_nlp_head`**  | `...layers.nlp_heads.factory.create_nlp_head`        | A factory for creating various downstream task heads (e.g., classification, NER). |

---

## 7. Configuration & Model Variants

The paper releases two main variants, which are provided here.

| Variant   | Hidden Size | Layers | Heads | FFN (GLU) Size | Params | Global Interval | Window Size |
| :-------- | :---------- | :----- | :---- | :------------- | :----- | :-------------- | :---------- |
| **`base`**| 768         | 22     | 12    | 2304           | 149M   | 3               | 128         |
| **`large`** | 1024        | 28     | 16    | 5248           | 395M   | 3               | 128         |

---

## 8. Comprehensive Usage Examples

### Example 1: Creating a Model for Text Classification

Use the factory function to attach a classification head to a ModernBERT encoder.

```python
import keras
import numpy as np
from dl_techniques.models.modern_bert.modern_bert import ModernBERT
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType

# 1. Define the classification task
classification_task = NLPTaskConfig(
    name="sentiment_classification",
    task_type=NLPTaskType.SEQUENCE_CLASSIFICATION,
    num_classes=3
)

# 2. Create the complete model
classifier_model = create_modern_bert_with_head(
    bert_variant="base",
    task_config=classification_task
)
classifier_model.summary()

# 3. Use the model for inference
dummy_inputs = {
    "input_ids": np.random.randint(0, 50368, (4, 128)),
    "attention_mask": np.ones((4, 128), dtype="int32"),
    "token_type_ids": np.zeros((4, 128), dtype="int32"),
}
predictions = classifier_model.predict(dummy_inputs)
print(f"\nPredictions shape: {predictions.shape}") # (4, 3)
```

### Example 2: Using ModernBERT for Long-Context Feature Extraction

ModernBERT's native 8192 sequence length makes it ideal for long-document tasks.

```python
import numpy as np
from dl_techniques.models.modern_bert.modern_bert import ModernBERT

# 1. Create a foundation model
long_context_bert = ModernBERT.from_variant("base")

# 2. Process a long sequence (e.g., 4096 tokens)
long_inputs = {
    "input_ids": np.random.randint(0, 50368, (1, 4096)),
    "attention_mask": np.ones((1, 4096), dtype="int32"),
}

# 3. Extract features
features = long_context_bert.predict(long_inputs)
print(f"Feature map shape: {features['last_hidden_state'].shape}") # (1, 4096, 768)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Fine-tuning from Pre-trained Weights

This implementation is designed to load pre-trained weights from local files or official URLs.

```python
# Create a model and load official pre-trained weights by setting pretrained=True
# The from_variant method will handle downloading.
import keras
from dl_techniques.models.modern_bert.modern_bert import create_modern_bert_with_head

ner_model_pretrained = create_modern_bert_with_head(
    bert_variant="base",
    task_config=some_ner_config,
    pretrained=True
)

# 2. Now you can fine-tune this model on your specific NER dataset
# Use a low learning rate for fine-tuning
ner_model_pretrained.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
    loss="...",
    metrics=["accuracy"]
)
print("✅ Model ready for fine-tuning!")
```

---

## 10. Performance Optimization

### Flash Attention & Unpadding

The original paper highlights significant speed and memory efficiency gains from using **Flash Attention** and **unpadding**. Unpadding avoids wasted computation on padding tokens by concatenating sequences into a single packed sequence. Flash Attention provides highly optimized attention kernels. While this Keras implementation uses a standard `TransformerLayer`, it can be extended to use Flash Attention-compatible backends for maximum performance.

### Mixed Precision Training

ModernBERT benefits greatly from mixed precision training, which uses 16-bit floating-point numbers for faster computation and reduced memory usage.

```python
# Enable mixed precision globally before creating the model
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = ModernBERT.from_variant("base")
model.compile(optimizer="adamw")
```

---

## 11. Training and Best Practices

### Optimizer and Schedule

-   **Optimizer**: The paper uses **StableAdamW**, an AdamW variant with update clipping, which improves stability. A standard `AdamW` is also a robust choice.
-   **Learning Rate Schedule**: The paper uses a modified **trapezoidal schedule** (Warmup-Stable-Decay). This schedule holds the learning rate constant for the majority of training, which aids in continual training. For fine-tuning, a linear decay schedule with warmup is a strong baseline.

### Key Training Details from the Paper

-   **Pre-training**: Trained for 2 trillion tokens on a diverse mix of web documents, code, and scientific literature.
-   **Context Length Extension**: The model was first trained on shorter sequences and then further trained on 8192-length sequences for an additional 300 billion tokens to extend its context window.
-   **Masking Rate**: Uses a 30% masking rate for the Masked Language Modeling (MLM) objective, which has been shown to be more effective than BERT's original 15%.
-   **No NSP**: The Next Sentence Prediction (NSP) objective is removed, following findings from RoBERTa and other modern encoders that it does not improve performance.

---

## 12. Serialization & Deployment

The `ModernBERT` model and its components are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
# Create and train a model with a head
model = create_modern_bert_with_head(...)
# model.compile(...) and model.fit(...)

# Save the entire model to a single file
model.save('my_modern_bert_classifier.keras')

# Load the model in a new session
loaded_model = keras.models.load_model('my_modern_bert_classifier.keras')
print("✅ Model loaded successfully!")
```

---

## 13. Testing & Validation

### Unit Tests

You can validate the implementation by ensuring all variants can be created and produce correctly shaped outputs.

```python
import keras
import numpy as np
from dl_techniques.models.modern_bert.modern_bert import ModernBERT, create_modern_bert_with_head

def test_creation_all_variants():
    """Test model creation for all variants."""
    for variant in ModernBERT.MODEL_VARIANTS.keys():
        model = ModernBERT.from_variant(variant)
        assert model is not None
        print(f"✓ ModernBERT-{variant} created successfully")

def test_forward_pass_shape():
    """Test the output shape of a forward pass."""
    model = ModernBERT.from_variant("base") # Using "base" as "tiny" is not a standard variant
    dummy_input = {"input_ids": np.random.randint(0, 50368, size=(4, 64))}
    output = model.predict(dummy_input)
    assert output["last_hidden_state"].shape == (4, 64, 768) # Shape for "base" model
    print("✓ Forward pass has correct shape")

# Run tests
if __name__ == '__main__':
    test_creation_all_variants()
    test_forward_pass_shape()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Training is unstable.**

-   **Cause**: This is unlikely with ModernBERT's Pre-LN design, but can still occur with an extremely high learning rate or numerical instability on certain hardware.
-   **Solution**: Use a smaller learning rate and a warmup schedule. Ensure you are using a modern optimizer like AdamW.

### Frequently Asked Questions

**Q: What is the main difference between ModernBERT and classic BERT?**

A: The five key upgrades are: **1) Rotary Positional Embeddings (RoPE)** for long context; **2) Pre-Layer Normalization** for stability; **3) GeGLU activation** for better performance; **4) Alternating windowed/global attention** for efficiency; and **5) Bias-free layers**.

**Q: Why use alternating attention instead of another efficient attention mechanism?**

A: Alternating attention is a simple and effective strategy. It is computationally cheap (dominated by the fast local attention) but still allows for full sequence-level information flow through the periodic global layers. This provides a strong balance of speed and modeling power.

**Q: Is ModernBERT a drop-in replacement for `bert-base-uncased`?**

A: Yes, in terms of API and function. It can be used in the same pipelines. However, it uses a different, modern BPE tokenizer and has been trained on a different dataset, so you must use the correct tokenizer and expect different (and generally much better) performance.

---

## 15. Technical Details

### Rotary Positional Embeddings (RoPE)

Instead of adding positional embeddings to the input, RoPE applies a rotational transformation to the query and key vectors within the attention mechanism. This rotation is a function of the token's absolute position but allows the attention score to be formulated based on relative positions, giving it excellent generalization to longer sequence lengths.

### Hardware-Aware Model Design

The specific dimensions of ModernBERT (e.g., hidden size, FFN expansion ratio, number of layers) were chosen through hardware-aware ablations. The goal was to maximize GPU utilization (specifically for common inference GPUs like NVIDIA T4, A10, L4, and RTX 4090) while being as "Deep & Narrow" as possible to improve downstream performance without a significant inference slowdown.

### GeGLU (Gated GELU)

The feed-forward network uses a Gated Linear Unit with a GELU activation:
`GeGLU(x) = GELU(x @ W_gate) * (x @ W_up)`
The input `x` is projected twice. One projection is passed through GELU and acts as a "gate," element-wise multiplying the second projection. This allows the network to dynamically control the information flow.

---

## 16. Citation

If using this model in your research, please cite the original paper:

-   **ModernBERT**:
    ```bibtex
    @article{warner2024smarter,
      title={Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference},
      author={Warner, Benjamin and Chaffin, Antoine and Clavié, Benjamin and Weller, Orion and Hallström, Oskar and Taghadouini, Said and Gallagher, Alexis and Biswas, Raja and Ladhak, Faisal and Aarsen, Tom and Cooper, Nathan and Adams, Griffin and Howard, Jeremy and Poli, Iacopo},
      journal={arXiv preprint arXiv:2412.13663},
      year={2024}
    }
    ```
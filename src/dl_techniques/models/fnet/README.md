# FNet: Mixing Tokens with Fourier Transforms

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of the **FNet** architecture in **Keras 3**, based on the paper ["FNet: Mixing Tokens with Fourier Transforms"](https://arxiv.org/abs/2105.03824) by Lee-Thorp et al. (2021).

The architecture is designed as a pure encoder, separating the core logic from task-specific heads, making it ideal for pre-training, fine-tuning, and multi-task learning.

---

## Table of Contents

1. [Overview: What is FNet and Why It Matters](#1-overview-what-is-fnet-and-why-it-matters)
2. [The Problem FNet Solves](#2-the-problem-fnet-solves)
3. [How FNet Works: Core Concepts](#3-how-fnet-works-core-concepts)
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

## 1. Overview: What is FNet and Why It Matters

### What is an FNet?

**FNet** is a Transformer-like architecture that replaces the computationally expensive self-attention mechanism with a simple, unparameterized **2D Fourier Transform**. It demonstrates that for many NLP tasks, the complex, content-aware mixing of self-attention can be effectively replaced by a much more efficient method.

For each encoder block, FNet performs token mixing by:
-   Applying a Fast Fourier Transform (FFT) along the sequence dimension.
-   Applying another FFT along the hidden (feature) dimension.

### Key Innovations

1.  **Parameter-Free Token Mixing**: The Fourier Transform requires no learnable weights, reducing the model's parameter count and memory footprint.
2.  **Exceptional Efficiency**: It reduces the complexity of the token-mixing step from `O(N²)` (quadratic in sequence length) for self-attention to `O(N log N)`.
3.  **BERT-Level Performance**: Despite its simplicity, FNet achieves comparable accuracy to BERT on many standard NLP benchmarks, while being significantly faster to train and run.
4.  **Drop-in Replacement**: The `FNetEncoderBlock` can be used as a direct, more efficient substitute for a standard Transformer encoder block.

### Why FNet Matters

**Standard Transformer (BERT) Problem**:
```
Problem: Classify a long document.
Standard Solution:
  1. Use a BERT-like model.
  2. The self-attention mechanism compares every token to every other token.
  3. Limitation: For a sequence of 4096 tokens, this requires 4096² ≈ 16.7 million
     comparisons per head, per layer. This is computationally massive and
     memory-intensive.
  4. Result: Processing long sequences is often infeasible.
```

**FNet's Solution**:
```
FNet Approach:
  1. Replace self-attention with a 2D Fourier Transform.
  2. The FFT naturally mixes information across the entire sequence in O(N log N) time.
  3. For a sequence of 4096 tokens, the complexity is proportional to
     4096 * log(4096) ≈ 49,000 operations.
  4. Benefit: Drastically faster training and inference, enabling the use of
     much longer sequences.
```

### Real-World Impact

FNet addresses the primary bottleneck in Transformer models, unlocking new possibilities:

-   🧬 **Genomics**: Process extremely long DNA or protein sequences.
-   📚 **Long-Document Analysis**: Summarize or classify entire articles, legal documents, or books.
-   🔊 **Audio Processing**: Analyze long audio waveforms represented as sequences.
-   ⏱️ **Real-Time NLP**: Deploy faster and more efficient models for on-device or edge applications.

---

## 2. The Problem FNet Solves

### The Tyranny of Quadratic Complexity

Standard Transformer models, while powerful, are fundamentally limited by the self-attention mechanism.

```
┌─────────────────────────────────────────────────────────────┐
│  Standard Transformer (Self-Attention)                      │
│                                                             │
│  The Core Bottleneck:                                       │
│    Complexity is O(SequenceLength² * HiddenDim).            │
│    Doubling the sequence length quadruples the computation  │
│    and memory required for the attention matrix.            │
│                                                             │
│  This leads to:                                             │
│  1. Slow Training: Training on long sequences is extremely  │
│     time-consuming.                                         │
│  2. High Memory Usage: The attention matrix alone can       │
│     consume gigabytes of VRAM.                              │
│  3. Sequence Length Limits: Most models are practically     │
│     limited to 512 or 1024 tokens.                          │
└─────────────────────────────────────────────────────────────┘
```

**Example: The cost of self-attention**
-   At 512 tokens, the attention matrix is 512x512.
-   At 1024 tokens, it's 1024x1024 (4x larger).
-   At 4096 tokens, it's 4096x4096 (64x larger than at 512).


*Self-attention's quadratic complexity makes it impractical for long sequences, while FNet's O(N log N) complexity scales gracefully.*

### How FNet Changes the Game

FNet proposes a radical simplification: the main purpose of self-attention is token mixing, and this can be approximated efficiently.

```
┌─────────────────────────────────────────────────────────────┐
│  FNet's Efficiency Principle                                │
│                                                             │
│  1. Replace the learnable, content-aware attention mechanism│
│     with a fixed, parameter-free Fourier Transform.         │
│                                                             │
│  2. The Fourier Transform provides complete, global mixing  │
│     of token information in quasi-linear time.              │
│                                                             │
│  3. The model relies on the Feed-Forward Network (FFN)      │
│     sub-layer to perform all the necessary content-based,   │
│     non-linear transformations.                             │
│                                                             │
│  This allows the model to:                                  │
│  - Process very long sequences without memory explosion.    │
│  - Train up to 80% faster than BERT on GPUs.                │
│  - Reduce the total number of model parameters.             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How FNet Works: Core Concepts

### The Two-Sublayer Encoder Block

An FNet model is a stack of `FNetEncoderBlock` layers. Each block has two main sub-layers, similar to a standard Transformer.

```
┌─────────────────────────────────────────────────────────────┐
│                    FNet Encoder Block                       │
│                                                             │
│  ┌─────────────────┐      ┌───────────────────────────────┐ │
│  │  Token Mixing   │      │        Channel Mixing         │ │
│  │    (FFT)        ├──────►      (Feed-Forward Net)       │ │
│  │                 │      │                               │ │
│  │  - Applies 2D   │      │  - Same as standard           │ │
│  │    FFT to mix   │      │    Transformer FFN.           │ │
│  │    information  │      │  - Learns content-based       │ │
│  │    globally.    │      │    transformations.           │ │
│  │  - Paramtr-free │      │  - Position-wise.             │ │
│  │                 │      │                               │ │
│  └─────────────────┘      └───────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### The Fourier Transform for Token Mixing

The core innovation lies in how FNet mixes token information. For an input tensor `X` of shape `(seq_len, hidden_dim)`:

1.  **FFT along Sequence**: `FFT(X, axis=0)`. This step mixes information across all tokens. Every element in the output is a linear combination of all input tokens.
2.  **FFT along Hidden Dim**: `FFT(X, axis=1)`. This step mixes information within the feature representation of each token.
3.  **Real Part**: `Real(...)`. The imaginary part is discarded.

The final operation is **`Y = Real(FFT(FFT(X)))`**. This simple operation provides powerful, global mixing.

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     FNet Complete Data Flow                             │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: Embedding
─────────────────
Input IDs (B, L)
    │
    ├─► Token Embeddings
    ├─► Position Embeddings
    ├─► Token Type Embeddings
    │
    ├─► Sum Embeddings -> LayerNorm -> Dropout
    │
    └─► Output: (B, L, H) ← EMBEDDING OUTPUT


STEP 2: FNet Encoder Stack (repeated for N layers)
──────────────────────────────────────────────────
Embedding Output (B, L, H)
    │
    ├─► FNetEncoderBlock 1
    │   ├─► Fourier Transform (mixing along L and H)
    │   ├─► Residual Connection + LayerNorm
    │   ├─► Feed-Forward Network
    │   └─► Residual Connection + LayerNorm
    │
    ├─► FNetEncoderBlock 2
    │   └─► ...
    │
    ├─► FNetEncoderBlock N
    │
    └─► Final Hidden State (B, L, H)


STEP 3: Task-Specific Head (Example: Classification)
────────────────────────────────────────────────────
Final Hidden State (B, L, H)
    │
    ├─► Select [CLS] token representation: (B, H)
    │
    ├─► [Optional] Pooling Layer
    │
    ├─► Dense Layer (Classification)
    │
    └─► Logits (B, num_classes)
```

---

## 4. Architecture Deep Dive

### 4.1 Embedding Layer (`BertEmbeddings`)

FNet uses a standard BERT-style embedding layer, which is a sum of three components:

```
┌───────────────────────────────────────────────────┐
│               BertEmbeddings                      │
└───────────────────────────────────────────────────┘
Input IDs, Token Type IDs
  │
  ├─► Token Embeddings (learnable vocab lookup)
  │
  ├─► Position Embeddings (learnable, for token order)
  │
  ├─► Token Type Embeddings (learnable, for sentence pairs)
  │
  ▼
Sum all three embeddings
  │
  ▼
Layer Normalization
  │
  ▼
Dropout
  │
  ▼
Output: (B, L, H)
```

### 4.2 FNet Encoder Block (`FNetEncoderBlock`)

This is the heart of the FNet model.

#### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    FNetEncoderBlock (Internal)              │
└─────────────────────────────────────────────────────────────┘
Input: (B, L, H)
  │
  ▼
┌──────────────────────────┐
│  FNetFourierTransform    │  ← Token Mixing Sub-layer
└──────────────────────────┘
  │
  │
  ▼
Add & Norm
(Input + Fourier Output)
  │
  ▼
┌──────────────────────────┐
│ Feed-Forward Network     │  ← Channel Mixing Sub-layer
│ (e.g., MLP, SwiGLU)      │
└──────────────────────────┘
  │
  │
  ▼
Add & Norm
(FFN Input + FFN Output)
  │
  ▼
Output: (B, L, H)
```

#### Key Design Decisions Explained

**Q: Why use a Fourier Transform?**

A: The convolution theorem states that convolution in the time domain is equivalent to element-wise multiplication in the frequency domain. An FFT can be seen as an efficient way to perform a specific type of global convolution, allowing every token to interact with every other token. It's a highly efficient, parameter-free way to achieve global mixing.

**Q: Why a 2D Fourier Transform (along both sequence and hidden dimensions)?**

A: Mixing along the sequence dimension is the key for replacing self-attention. The authors found that also mixing along the hidden dimension provided a small but consistent performance boost. It encourages features within a token's representation to interact more richly.

**Q: What is the trade-off?**

A: FNet sacrifices the **dynamic, content-aware** nature of self-attention. In BERT, the attention weights are computed based on the input tokens (Query-Key interactions). In FNet, the mixing operation is **static and independent of the input content**. The model relies entirely on the FFN layers to learn content-specific logic. The surprising result of the FNet paper is that this trade-off is often worth it.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First FNet Model (30 seconds)

Let's build a simple sentiment classifier using a pretrained FNet encoder.

```python
import keras
import numpy as np

# Local imports from your project structure
from dl_techniques.models.nlp.fnet.model import create_fnet_with_head
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType

# 1. Define the classification task
# Note: For real use, download actual pretrained weights.
# The default URLs are placeholders, so we'll use pretrained=False.
task_config = NLPTaskConfig(
    name="sentiment_analysis",
    task_type=NLPTaskType.SENTIMENT_ANALYSIS,
    num_classes=2
)

# 2. Create an end-to-end model using the factory function
# This combines a FNet-base encoder with a classification head.
model = create_fnet_with_head(
    fnet_variant="tiny",         # Use "tiny" for a quick example
    task_config=task_config,
    pretrained=False,            # Set to True or a file path for real weights
    sequence_length=128          # Fixed sequence length for the model
)

# 3. Compile the model
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
print("✅ FNet model with classification head created and compiled!")
model.summary()

# 4. Create dummy data for a forward pass
batch_size = 4
seq_len = 128
vocab_size = FNet.DEFAULT_VOCAB_SIZE

dummy_inputs = {
    "input_ids": np.random.randint(0, vocab_size, (batch_size, seq_len)),
    "attention_mask": np.ones((batch_size, seq_len), dtype="int32"),
    "token_type_ids": np.zeros((batch_size, seq_len), dtype="int32"),
}
dummy_labels = np.random.randint(0, 2, (batch_size,))

# 5. Train for one step
loss, acc = model.train_on_batch(dummy_inputs, dummy_labels)
print(f"\n✅ Training step complete! Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# 6. Run inference
predictions = model.predict(dummy_inputs)
print(f"Predictions shape: {predictions.shape}") # (batch_size, num_classes)
```

---

## 6. Component Reference

### 6.1 `FNet` (Model Class)

**Purpose**: A pure foundation model that serves as a text encoder. It outputs the final hidden states for each token.

**Location**: `dl_techniques.models.nlp.fnet.model.FNet`

```python
from dl_techniques.models.nlp.fnet.model import FNet

# Create from a variant, with optional pretrained weights
encoder = FNet.from_variant(
    "base",
    pretrained=False, # or True, or "path/to/weights.keras"
    vocab_size=50265, # Override default vocab size
)
```

**Key Parameters**:

| Parameter | Description |
| :--- | :--- |
| `vocab_size` | Size of the token vocabulary. |
| `hidden_size` | Dimensionality of the encoder layers and embeddings. |
| `num_layers` | Number of `FNetEncoderBlock` layers. |
| `intermediate_size` | The size of the FFN's hidden layer. |
| `max_position_embeddings` | The maximum sequence length the model can handle. |

**Key Methods**:
-   `from_variant(variant, pretrained, **kwargs)`: The recommended factory method for creating standard FNet models (`base`, `large`, etc.).
-   `load_pretrained_weights(path, skip_mismatch)`: Loads weights from a file, with options to handle architecture mismatches (e.g., different `vocab_size`).

### 6.2 `FNetEncoderBlock`

**Purpose**: The core building block of the FNet model, replacing the standard Transformer encoder.

**Location**: `dl_techniques.layers.fnet_encoder_block.FNetEncoderBlock`

```python
from dl_techniques.layers.fnet_encoder_block import FNetEncoderBlock

# Can be used to build custom architectures
block = FNetEncoderBlock(
    intermediate_dim=3072,
    dropout_rate=0.1,
    normalization_type='rms_norm', # Modernize with RMSNorm
    ffn_type='swiglu'              # and a SwiGLU FFN
)
```

### 6.3 `create_fnet_with_head`

**Purpose**: A high-level factory function to assemble a complete, task-specific model.

**Location**: `dl_techniques.models.nlp.fnet.model.create_fnet_with_head`

---

## 7. Configuration & Model Variants

The implementation provides several standard FNet configurations.

| Variant | Hidden Size | Layers | FFN Size | Description |
| :---: | :---: |:---: |:---: |:--- |
| **`large`** | 1024 | 24 | 4096 | FNet-Large: Maximum performance |
| **`base`** | 768 | 12 | 3072 | FNet-Base: Good balance for most tasks |
| **`small`** | 512 | 6 | 2048 | Lightweight for resource-constrained use |
| **`tiny`** | 256 | 4 | 512 | Ultra-lightweight for mobile/edge |

**Guideline**: Use `base` for general-purpose tasks. Use `large` when you need maximum accuracy and have the resources. Use `small` or `tiny` for faster inference or deployment on edge devices.

---

## 8. Comprehensive Usage Examples

### Example 1: Feature Extraction for Downstream Tasks

Use a pretrained FNet as a universal text feature extractor.

```python
import keras
from dl_techniques.models.nlp.fnet.model import FNet

# 1. Load a pretrained FNet encoder
# (Assuming weights are available)
fnet_encoder = FNet.from_variant("base", pretrained=False)

# 2. Define inputs
inputs = {
    "input_ids": keras.Input(shape=(None,), dtype="int32"),
    "attention_mask": keras.Input(shape=(None,), dtype="int32"),
}

# 3. Get contextual embeddings
outputs = fnet_encoder(inputs)
last_hidden_state = outputs["last_hidden_state"] # Shape: (batch, seq_len, 768)

# 4. Build a model that extracts features
feature_extractor = keras.Model(inputs, last_hidden_state)

# 5. Use the features for any downstream task
# For classification, you could take the [CLS] token's embedding
cls_embedding = last_hidden_state[:, 0, :] # Shape: (batch, 768)
```

### Example 2: Building a Token Classification (NER) Model

This example uses the factory function to create a Named Entity Recognition model.

```python
from dl_techniques.models.nlp.fnet.model import create_fnet_with_head
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType

# 1. Define the NER task
ner_task_config = NLPTaskConfig(
    name="ner",
    task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
    num_classes=9 # e.g., B-PER, I-PER, B-LOC, I-LOC, etc. + O
)

# 2. Create the full model
ner_model = create_fnet_with_head(
    fnet_variant="base",
    task_config=ner_task_config,
    pretrained=False,
    sequence_length=256
)

# 3. Compile and inspect
ner_model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)
ner_model.summary()

# The model will output logits for each token in the sequence
# Output shape: (batch_size, sequence_length, num_classes)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Modernizing the FNet Block

The `FNetEncoderBlock` is highly configurable. You can replace the standard LayerNorm and MLP with more modern components like RMSNorm and SwiGLU.

```python
from dl_techniques.models.nlp.fnet.model import FNet

# Create a 'base' model but override the encoder block configurations
# This is done by passing the configuration to from_variant()
modern_fnet = FNet.from_variant(
    "base",
    # These kwargs are passed down to the FNetEncoderBlock
    normalization_type="rms_norm",
    ffn_type="swiglu"
)

modern_fnet.summary()
```

### Pattern 2: Fine-tuning on a Multi-Task Problem

Since `FNet` is a pure encoder, you can easily attach multiple heads for multi-task learning.

```python
import keras
from dl_techniques.models.nlp.fnet.model import FNet
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType, create_nlp_head

# 1. Create the shared FNet encoder
shared_encoder = FNet.from_variant("base", pretrained=False)

# 2. Define tasks and create heads
sentiment_task = NLPTaskConfig(name="sentiment", task_type=NLPTaskType.SENTIMENT_ANALYSIS, num_classes=3)
ner_task = NLPTaskConfig(name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION, num_classes=9)

sentiment_head = create_nlp_head(sentiment_task, input_dim=shared_encoder.hidden_size)
ner_head = create_nlp_head(ner_task, input_dim=shared_encoder.hidden_size)

# 3. Build the combined model
inputs = { ... } # Standard FNet inputs
encoder_output = shared_encoder(inputs)

head_input = {
    "hidden_states": encoder_output["last_hidden_state"],
    "attention_mask": encoder_output["attention_mask"],
}

sentiment_output = sentiment_head(head_input, name="sentiment_output")
ner_output = ner_head(head_input, name="ner_output")

multitask_model = keras.Model(
    inputs=inputs,
    outputs={"sentiment": sentiment_output, "ner": ner_output}
)

# Compile with separate losses
multitask_model.compile(
    optimizer="adam",
    loss={"sentiment": "sparse_categorical_crossentropy", "ner": "sparse_categorical_crossentropy"}
)
```

---

## 10. Performance Optimization

### Mixed Precision Training

FNet benefits significantly from mixed precision, which uses float16 for most computations to accelerate training.

```python
# Enable mixed precision globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = FNet.from_variant("base")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

### TensorFlow XLA Compilation

Use XLA (Accelerated Linear Algebra) to compile the model's computation graph for an additional speed boost.

```python
import tensorflow as tf

model = create_fnet_with_head(...) # Create your model

# Compile the predict function with XLA
@tf.function(jit_compile=True)
def compiled_predict(inputs):
    return model(inputs, training=False)

# Usage
predictions = compiled_predict(dummy_inputs)
```

---

## 11. Training and Best Practices

### Optimizer and Learning Rate

-   **Optimizer**: AdamW is the standard choice for training Transformer-like models.
-   **Learning Rate Schedule**: A linear decay schedule with a warmup period is highly recommended. Start with a peak learning rate of `1e-4` to `5e-5` for fine-tuning.

### Handling Sequence Length

-   The FFT operation is most efficient on sequences whose length is a power of two. While not strictly required, padding your sequences to the nearest power of two (e.g., 64, 128, 256, 512) can improve performance.
-   This implementation handles dynamic sequence lengths, but providing a fixed `sequence_length` to `create_fnet_with_head` can sometimes lead to better performance due to static graph optimizations.

---

## 12. Serialization & Deployment

The `FNet` model and its custom layers are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
# Create and train model
model = create_fnet_with_head(...)
# model.compile(...) and model.fit(...)

# Save the entire model
model.save('fnet_classifier.keras')
print("Model saved to fnet_classifier.keras")

# Load the model in a new session
loaded_model = keras.models.load_model('fnet_classifier.keras')
print("Model loaded successfully")
```

---

## 13. Testing & Validation

### Unit Tests

```python
import keras
import numpy as np
from dl_techniques.models.nlp.fnet.model import FNet

def test_model_creation_from_variant():
    """Test that models can be created from all variants."""
    for variant in FNet.MODEL_VARIANTS.keys():
        model = FNet.from_variant(variant)
        assert model is not None
        print(f"✓ FNet-{variant} created successfully")

def test_forward_pass_shape():
    """Test the output shape of a forward pass."""
    model = FNet.from_variant("tiny")
    dummy_input = {
        "input_ids": keras.random.uniform((4, 64), 0, 1000, dtype="int32"),
        "attention_mask": keras.ops.ones((4, 64), dtype="int32"),
    }
    output = model(dummy_input)
    hidden_state = output["last_hidden_state"]
    expected_shape = (4, 64, FNet.MODEL_VARIANTS["tiny"]["hidden_size"])
    assert hidden_state.shape == expected_shape
    print("✓ Forward pass has correct shape")

def test_serialization():
    """Test model save/load."""
    model = FNet.from_variant("tiny")
    model.save('test_fnet.keras')
    loaded_model = keras.models.load_model('test_fnet.keras')
    assert model.get_config()['hidden_size'] == loaded_model.get_config()['hidden_size']
    print("✓ Serialization successful")

# Run tests
if __name__ == '__main__':
    test_model_creation_from_variant()
    test_forward_pass_shape()
    test_serialization()
    print("\n✅ All tests passed!")
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Performance is slightly lower than BERT on a specific task.**

-   **Cause**: FNet is an approximation of a full Transformer. Its static mixing mechanism may not be optimal for all tasks, especially those requiring highly content-sensitive token interactions.
-   **Solution**: This is the inherent trade-off. FNet offers a massive speedup for a potential small drop in accuracy. Consider if the speed/efficiency gain is worth it for your application. Also, try fine-tuning for more epochs, as the learning dynamics may differ from BERT.

**Issue 2: Training is slow when using dynamic sequence lengths.**

-   **Cause**: While the model supports dynamic shapes, FFT implementations and XLA compilation are often heavily optimized for static shapes. Re-compiling the graph for each batch with a different shape can add overhead.
-   **Solution**: If possible, batch inputs by sequence length or pad all inputs in a batch to a uniform length. For inference, using a fixed sequence length is often best.

### Frequently Asked Questions

**Q: Is FNet a universal replacement for BERT?**

A: No. It's a highly efficient alternative. For tasks where inference speed, memory, or the ability to handle long sequences is critical, FNet is an excellent choice. For tasks requiring the absolute highest accuracy where computational cost is not a concern, a standard Transformer like RoBERTa or DeBERTa might still be superior.

**Q: How does FNet compare to other efficient Transformers like Linformer or Reformer?**

A: FNet is arguably the simplest. Linformer uses a low-rank projection to approximate the attention matrix, while Reformer uses locality-sensitive hashing. FNet's Fourier Transform is completely parameter-free and deterministic. This simplicity makes it very fast and easy to implement.

**Q: Can I use this for non-NLP sequence tasks?**

A: Absolutely. As long as you can represent your data as a sequence of vectors, you can use an FNet encoder. This could include time-series data, flattened image patches, or biological sequences. You would just need to create an appropriate embedding layer for your specific data modality.

---

## 15. Technical Details

### Complexity Analysis

The key advantage of FNet comes from its reduced computational complexity. Let `L` be the sequence length and `H` be the hidden size.

-   **Self-Attention (e.g., BERT)**:
    -   The dot-product attention mechanism has a complexity of **`O(L² * H)`**.
    -   This quadratic scaling with `L` is the primary bottleneck.

-   **FNet (Fourier Transform)**:
    -   The Fast Fourier Transform algorithm has a complexity of `O(L * log L)`.
    -   The FNet block's mixing step applies this along two dimensions, leading to a complexity of **`O(L * H * log(L*H))`**, but in practice, it is dominated by **`O(L * H * log L)`**.
    -   This is a near-linear relationship with `L`, allowing the model to scale to much longer sequences.

### The Role of the Feed-Forward Network

Since the Fourier Transform is a linear, static operation, the FFN sub-layer in FNet carries a heavy burden. It is responsible for:
1.  **Non-linearity**: Introducing non-linear transformations that are essential for learning complex functions.
2.  **Content-based Reasoning**: Acting on each token's representation independently to model content-specific features.
3.  **Memorization**: Storing factual knowledge learned during pre-training.

The success of FNet suggests that decoupling global mixing (FFT) from local, content-aware processing (FFN) is a viable and highly efficient architectural choice.

---

## 16. Citation

If you use FNet in your research, please cite the original paper:

```bibtex
@article{lee2021fnet,
  title={FNet: Mixing Tokens with Fourier Transforms},
  author={Lee-Thorp, James and Ainslie, Joshua and Eckstein, Ilya and Ontanon, Santiago},
  journal={arXiv preprint arXiv:2105.03824},
  year={2021}
}
```
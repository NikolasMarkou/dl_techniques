# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18+-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of the **Mamba** architecture in **Keras 3**. This implementation is based on the original paper by Gu and Dao, providing a pure foundation model that separates the core sequence modeling logic from task-specific heads for maximum flexibility.

The architecture's key feature is its **Selective State Space Model (SSM)** core, designed to capture long-range dependencies with **linear-time complexity** and **constant-time inference**, offering a powerful and efficient alternative to Transformers.

---

## Table of Contents

1. [Overview: What is Mamba and Why It Matters](#1-overview-what-is-mamba-and-why-it-matters)
2. [The Problem Mamba Solves](#2-the-problem-mamba-solves)
3. [How Mamba Works: Core Concepts](#3-how-mamba-works-core-concepts)
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

## 1. Overview: What is Mamba and Why It Matters

### What is Mamba?

**Mamba** is a new generation of sequence model that uses a **Selective State Space Model (SSM)** to achieve linear-time complexity while matching or exceeding the performance of Transformers on many tasks. It is designed to overcome the quadratic scaling bottleneck of attention, making it highly efficient for processing very long sequences.

This implementation provides the core Mamba architecture as a **foundation model**. It takes tokenized text as input and outputs a sequence of high-dimensional vectors, where each vector is a rich, contextualized embedding for a corresponding input token. These embeddings can then be used for a wide variety of downstream NLP and sequence modeling tasks.

### Key Innovations of this Implementation

1.  **Foundation Model Design**: The `Mamba` class is a pure sequence encoder. It is intentionally decoupled from any specific task, making it a reusable and flexible component for pre-training and multi-task learning.
2.  **Keras 3 Native & Serializable**: Built as a composite `keras.Model`, the entire architecture, including all custom layers, is fully serializable to the modern `.keras` format and compatible with TensorFlow, PyTorch, and JAX.
3.  **Configurable Variants**: The `from_variant` class method allows for easy instantiation of standard Mamba configurations, aligning with published model sizes.

### Why Mamba Matters: The Sequence Modeling Trilemma

| Model | Pros | Cons |
| :--- | :--- | :--- |
| **Transformers** | Excellent performance on information-dense data (language). | **Quadratic O(L²) complexity**; slow autoregressive inference O(L) due to a growing KV cache. |
| **RNNs (LSTMs)** | Linear O(L) complexity, constant-time inference O(1). | Struggle with long-range dependencies; difficult to parallelize. |
| **Prior SSMs (LTI)** | Efficient via convolutions (linear time). | Input-independent parameters struggle with discrete data; cannot selectively focus on inputs. |

**Mamba's Solution**: Mamba combines the best of these approaches.
- It achieves **linear-time O(L) complexity** and **constant-time O(1) inference**.
- Its unique **Selection Mechanism** makes its internal parameters input-dependent, allowing it to selectively remember or forget information, similar to attention but far more efficiently.
- The result is Transformer-level quality on language tasks while excelling on ultra-long-context tasks where Transformers are computationally infeasible.

![image](mamba_v1_ssm_selection.png)

### Real-World Impact

Mamba's efficiency and power unlock new possibilities for long-sequence tasks:

-   **Genomics**: Modeling entire DNA sequences with millions of base pairs.
-   **Audio & Video**: Processing high-resolution, long-form multimedia streams.
-   **Long-Document NLP**: Question answering, summarization, and analysis of books, articles, or codebases.
-   **Time Series Forecasting**: Modeling complex patterns over extended historical data.

---

## 2. The Problem Mamba Solves

### The Challenge of Selective Compression

Effective sequence models must compress a long history of context into a compact state. Transformers "cheat" by not compressing at all—they keep the entire history (the KV cache) available, leading to their power but also their quadratic cost. Traditional RNNs compress context into a fixed-size state, but their simple, uniform updates make it hard to selectively retain important information while discarding irrelevant noise.

```
┌─────────────────────────────────────────────────────────────┐
│  The Challenge of Selective Information Processing          │
│                                                             │
│  Task: Selective Copying                                    │
│  Input:  A B ... C D ... E F ... G H                        │
│  Goal:   Copy the colored tokens (B, D, F, H)               │
│                                                             │
│  - A Transformer can solve this by attending directly back  │
│    to the colored tokens, but the cost grows with the       │
│    number of "..." filler tokens.                           │
│                                                             │
│  - An LTI model (like a fixed convolution) fails because    │
│    the spacing between relevant tokens is irregular. It     │
│    cannot learn to "skip" the irrelevant parts.             │
└─────────────────────────────────────────────────────────────┘
```

To solve this, a model needs a mechanism to dynamically decide which information to keep and which to ignore based on the content it's seeing. Mamba introduces this **selection mechanism** directly into its state-space formulation by making its core parameters data-dependent.

---

## 3. How Mamba Works: Core Concepts

### The High-Level Architecture

The model is a simple, homogeneous stack of Mamba blocks. Each block applies the core Selective SSM logic within a standard pre-norm residual architecture.

```
Input (Token IDs) ───► Embedding Layer
                         │
                         ▼
                  ┌──────────────┐
                  │ Mamba Block  │ (Pre-Norm + MambaLayer)
                  └──────────────┘
                         │ (x N Layers)
                         ▼
                  ┌──────────────┐
                  │ Mamba Block  │
                  └──────────────┘
                         │
                         ▼
                  Final Normalization Layer
                         │
                         ▼
                  Output (Hidden States)
```

### Data Flow Within a Mamba Block

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Mamba Block Complete Data Flow                      │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: RESIDUAL & NORMALIZATION
─────────────────────────────────
Input Hidden State + Previous Residual ───► LayerNorm ───► Normalized Input


STEP 2: MAMBA LAYER CORE LOGIC
───────────────────────────────
Normalized Input (B, L, D)
    │
    ├─► Linear Projection ──► Split into `x` (main) and `z` (gate) streams
    │
    ├── `x` Stream (Main Path):
    │   ├─► Causal 1D Convolution (captures local context)
    │   ├─► SiLU Activation
    │   ├─► Linear Projection to get SSM parameters (Δ, B, C)
    │   └─► Selective Scan (Recurrent State Update) ──► `y`
    │
    └── `z` Stream (Gating Path):
        └─► SiLU Activation ──► Gating signal
            │
    ┌───────┴──────────┐
    │ Gating Operation │
    └───────┬──────────┘
            │
    Element-wise product: `y` * SiLU(`z`)
            │
    Final Linear Projection ──► Block Output (B, L, D)


STEP 3: OUTPUTS
───────────────
- The Block Output becomes the `hidden_states` for the next block.
- The pre-normalization sum becomes the `residual` for the next block.
```

---

## 4. Architecture Deep Dive

### 4.1 `MambaLayer`

The core engine of the Mamba architecture.
-   **Input Projection**: Expands the input dimension and splits it into two streams: `x` for the main computation and `z` for gating.
-   **Causal Convolution**: A small 1D convolution on `x` acts as a local feature extractor, helping the SSM focus on recent context.
-   **SSM Parameterization**: The output of the convolution is projected to dynamically generate the `Δ` (step size), `B` (input matrix), and `C` (output matrix) parameters for the state space model at each time step. This is the **selection mechanism**.
-   **Selective Scan**: A hardware-aware parallel scan algorithm performs the recurrent state update (`h_t = A_bar * h_{t-1} + B_bar * x_t`). This is the heart of Mamba, implemented efficiently to run in linear time.
-   **Output Gating**: The output of the scan (`y`) is modulated by the `z` stream via an element-wise multiplication, allowing the block to control information flow to the next layer.

### 4.2 `MambaResidualBlock`

A standard wrapper that turns the `MambaLayer` into a building block for a deep network.
-   It implements a **Pre-Norm** architecture: the residual connection is added to the input *before* the layer normalization.
-   This structure (`x + residual -> norm -> mixer`) is known to promote stable training in very deep models.
-   The Mamba model is simply a stack of these `MambaResidualBlock`s.

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras==3.* tensorflow==2.16.* numpy
```

### Your First Mamba Model (30 seconds)

Let's build a Mamba-based sentiment analysis model.

```python
import keras
import numpy as np
from dl_techniques.models.mamba import Mamba

# 1. Define model configuration
VOCAB_SIZE = 10000
NUM_CLASSES = 3  # e.g., positive, negative, neutral

# 2. Create the Mamba foundation model from a variant
# We'll use the smallest standard variant for this example.
mamba_encoder = Mamba.from_variant(
    "130m",  # This is the "base" variant from the paper
    vocab_size=VOCAB_SIZE,
    name="mamba_encoder"
)

# 3. Build a sentiment analysis model on top of the Mamba encoder
# The input is a dictionary, matching the expected format.
inputs = {"input_ids": keras.Input(shape=(None,), dtype="int32")}

# Get the contextual embeddings from Mamba
mamba_outputs = mamba_encoder(inputs)
sequence_output = mamba_outputs["last_hidden_state"]

# For classification, we pool the sequence output.
pooled_output = keras.layers.GlobalAveragePooling1D()(sequence_output)
# Add a classification head
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(pooled_output)

# Create the final, trainable model
model = keras.Model(inputs, outputs, name="mamba_sentiment_classifier")

# 4. Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
print("Mamba-based sentiment analysis model built and compiled successfully!")
model.summary()

# 5. Use the model with dummy data
BATCH_SIZE, SEQ_LEN = 4, 128
dummy_inputs = {
    "input_ids": np.random.randint(0, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN)),
}
dummy_labels = np.random.randint(0, NUM_CLASSES, size=(BATCH_SIZE,))

# The model is ready for training
print("\n--- Training for one step ---")
model.fit(dummy_inputs, dummy_labels, epochs=1, verbose=1)

# Make a prediction
print("\n--- Making a prediction ---")
predictions = model.predict(dummy_inputs)
print(f"Output predictions shape: {predictions.shape}")  # (4, 3)
```

---

## 6. Component Reference

### 6.1 `Mamba` (Model Class)

**Purpose**: The main Keras `Model` that implements the Mamba encoder. It outputs a dictionary containing `last_hidden_state`. It is designed to be used as a standalone encoder or as a building block for more complex models.

```python
from dl_techniques.models.mamba import Mamba
# Create a standalone Mamba-130m encoder
mamba_encoder = Mamba.from_variant("130m", vocab_size=50257)
```

### 6.2 `MambaResidualBlock` (Layer Class)

**Purpose**: A single block in the Mamba stack, containing a `LayerNormalization` and a `MambaLayer`. It handles the residual connection logic.

```python
from dl_techniques.models.mamba import MambaResidualBlock
# Create a single residual block
block = MambaResidualBlock(d_model=768, mamba_kwargs={"d_state": 16})
```

### 6.3 `MambaLayer` (Layer Class)

**Purpose**: The core layer containing the selective SSM logic: causal convolution, data-dependent parameterization, selective scan, and gating.

```python
from dl_techniques.models.mamba import MambaLayer
# Create a single Mamba layer
layer = MambaLayer(d_model=768, d_state=16, d_conv=4, expand=2)
```

---

## 7. Configuration & Model Variants

This implementation provides several standard configurations matching the original paper.

| Variant | d_model | Layers | Parameters (Approx.) | Use Case |
|:---:|:---:|:---:|:---:|:---|
| **`130m`** (`base`) | 768 | 24 | ~130M | General purpose, fine-tuning. |
| **`370m`** | 1024 | 24 | ~370M | Higher performance with moderate resource use. |
| **`790m`** | 1024 | 48 | ~790M | Strong performance for demanding tasks. |
| **`1.4b`** | 1536 | 48 | ~1.4B | High-performance, comparable to large Transformers. |
| **`2.8b`** | 2560 | 64 | ~2.8B | Maximum performance, requires significant compute. |

### Customizing the Configuration

You can override default parameters of a variant by passing them as keyword arguments to `from_variant`.

```python
from dl_techniques.models.mamba import Mamba

# Create a Mamba-130m model but with a larger state dimension
model = Mamba.from_variant(
    "130m",
    vocab_size=50257,
    d_state=32, # Default is 16
    d_conv=8    # Default is 4
)
```

---

## 8. Comprehensive Usage Examples

### Example: Building a Model for Genomics Classification

Mamba excels at long-sequence tasks like genomics.

```python
import keras
import numpy as np
from dl_techniques.models.mamba import Mamba

# 1. Configuration for a genomics task
# Vocab size for DNA is small (e.g., A, C, G, T, N, and special tokens)
VOCAB_SIZE = 10
NUM_CLASSES = 10  # e.g., 10 different species

# 2. Create the Mamba encoder with custom parameters suitable for genomics
genomics_encoder = Mamba(
    vocab_size=VOCAB_SIZE,
    d_model=256,
    num_layers=16,
    d_state=16,
    expand=2
)

# 3. Build the final classification model
inputs = {"input_ids": keras.Input(shape=(None,), dtype="int32")}
encoder_output = genomics_encoder(inputs)["last_hidden_state"]

# Pool and classify
pooled_output = keras.layers.GlobalAveragePooling1D()(encoder_output)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(pooled_output)
species_model = keras.Model(inputs, outputs)

# 4. Inspect and compile the model
species_model.summary()
species_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

---

## 9. Advanced Usage Patterns

### Pattern: Using Mamba as a Feature Extractor

The raw output of the `Mamba` encoder can serve as powerful contextual features for other models.

```python
import keras
from dl_techniques.models.mamba import Mamba

# 1. Create the standalone Mamba encoder
mamba_encoder = Mamba.from_variant("130m", vocab_size=50257)

# 2. Define your own downstream model that uses Mamba's features
inputs = {"input_ids": keras.Input(shape=(None,), dtype="int32")}
# The encoder is used like any other Keras layer
mamba_features = mamba_encoder(inputs)["last_hidden_state"] # Shape: (batch, seq_len, 768)

# Example: Use the features for a token classification task (e.g., NER)
ner_head = keras.layers.Dense(17, activation="softmax")(mamba_features)
ner_model = keras.Model(inputs, ner_head)
ner_model.summary()
```

---

## 10. Performance Optimization

### Mixed Precision Training

For larger Mamba models, mixed precision can dramatically speed up training and reduce memory usage on compatible GPUs.

```python
import keras

# Enable mixed precision globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Create model (will automatically use mixed precision)
model = Mamba.from_variant("790m", vocab_size=50257)
# ... compile and fit ...
```

### XLA Compilation

Use `jit_compile=True` for graph compilation. This can provide a significant speed boost, though performance may vary depending on the backend's support for the internal `while_loop`.

```python
model.compile(optimizer="adam", loss="...", jit_compile=True)
```

---

## 11. Training and Best Practices

### Fine-Tuning Strategy

Similar to Transformers, a common practice is to use a smaller learning rate for the main body of the model and a slightly larger one for the newly initialized task head. The `AdamW` optimizer with a cosine learning rate schedule and a brief warmup period is a strong default.

### Handling Long Sequences

This is Mamba's core strength.
-   **Linear Scaling**: Memory and computation scale linearly, O(L), with sequence length, compared to the O(L²) of standard attention. You can train on sequences of `32k`, `64k`, or even longer without running out of memory.
-   **No KV Cache**: During autoregressive inference, Mamba maintains a fixed-size state. This means generation time is constant per token and throughput is significantly higher than Transformers, which are bottlenecked by reading and writing a growing KV cache.

---

## 12. Serialization & Deployment

The `Mamba` model and all its custom layers are fully serializable using Keras 3's modern `.keras` format, thanks to the `@keras.saving.register_keras_serializable()` decorator.

### Saving and Loading

```python
import keras
from dl_techniques.models.mamba import Mamba, MambaLayer, MambaResidualBlock

# Create and train a model
model = Mamba.from_variant("130m", vocab_size=1000)
# ... model.compile(...) and model.fit(...)

# Save the entire model
model.save('my_mamba_model.keras')

# Load the model in a new session.
# NOTE: The custom layer classes (Mamba, MambaLayer, etc.) must be
# imported in the script where you load the model.
loaded_model = keras.models.load_model('my_mamba_model.keras')
loaded_model.summary()
```

---

## 13. Testing & Validation

This implementation includes a comprehensive test suite. You can run the tests to validate your setup.

### Example Smoke Test

Here is a minimal test file (`test_mamba_smoke.py`) you can use to quickly verify functionality.

```python
import pytest
import numpy as np
from dl_techniques.models.mamba import Mamba

def test_mamba_creation_and_forward_pass():
    """Tests that a Mamba model can be created and run a forward pass."""
    model = Mamba.from_variant("130m", vocab_size=1000)
    
    dummy_input = {"input_ids": np.random.randint(0, 1000, (4, 64))}
    output = model(dummy_input)
    
    hidden_state = output["last_hidden_state"]
    # Expected shape: (batch_size, sequence_length, d_model)
    assert hidden_state.shape == (4, 64, model.d_model)
    print("\nSmoke test passed: Model created and forward pass shape is correct.")

# To run this test:
# 1. Save the code above as `test_mamba_smoke.py`.
# 2. Run `pytest test_mamba_smoke.py` from your terminal.
```

---

## 14. Troubleshooting & FAQs

**Issue 1: Training is slow or has high memory usage.**

-   **Cause**: While Mamba is linear-time, the absolute memory usage can still be high, especially with large `d_model`, `d_state`, and `batch_size`. The recurrent `while_loop` in this Keras implementation might also be less optimized than a custom CUDA kernel.
-   **Solution**: 1) Reduce `batch_size`. 2) Enable mixed precision. 3) Ensure you are using a powerful GPU.

**Issue 2: Model performance is not as expected on a specific task.**

-   **Cause**: Mamba's strengths are most pronounced on tasks requiring long-range dependency. On short-sequence tasks, its inductive bias might differ from a Transformer. Hyperparameters like `d_state` and `d_conv` can also have a significant impact.
-   **Solution**: 1) Tune hyperparameters, especially `d_state`, `d_conv`, and the learning rate. 2) For tasks that benefit from local patterns, ensure `d_conv` is appropriately sized.

### Frequently Asked Questions

**Q: Is Mamba a type of RNN?**

A: Yes, Mamba can be seen as a modern, highly structured Recurrent Neural Network (RNN). Its "Selective Scan" is a recurrent operation. However, unlike traditional RNNs, it is designed to be trained efficiently in parallel (like a CNN) and to avoid vanishing gradient issues through its specific state-space formulation.

**Q: How does the "Selective Scan" compare to attention?**

A: The selective scan is the recurrent mechanism that updates Mamba's hidden state. Attention is a parallel mechanism that re-weights and combines all hidden states from the past.
-   **Selectivity**: Both can select information. Attention does it by assigning weights to all past tokens. Mamba does it by deciding at each step how much to remember from its past state and how much to incorporate from the current input.
-   **Complexity**: Attention is O(L²). The selective scan is O(L). This is Mamba's key advantage.

---

## 15. Technical Details

### Selective State Space Models (S6)

The core idea is to enhance a classic State Space Model. A continuous SSM is defined by:
`h'(t) = Ah(t) + Bx(t)`
`y(t) = Ch(t)`

Mamba makes this system "selective" by making the key parameters `Δ`, `B`, and `C` functions of the input `x`. This breaks the time-invariance property of older SSMs, which is what prevented them from modeling discrete, content-rich data effectively. By making the parameters dynamic, the model learns to modulate its own state transitions based on the content it sees.

### Hardware-Aware Parallel Scan

A key challenge with making SSMs selective is that they can no longer be computed as a global convolution (which is very fast). They must be computed recurrently. Mamba uses a hardware-aware parallel scan algorithm (similar to prefix sums) to compute the recurrence. This algorithm is designed to maximize GPU utilization by reducing memory I/O between slow HBM and fast SRAM, making the recurrent computation nearly as fast as optimized convolutions or attention for training.

### Simplified Architecture

The Mamba architecture is intentionally simple and homogeneous. It fuses the roles of the attention layer and the FFN layer from a Transformer block into a single, unified Mamba block. This simplification reduces architectural complexity and can lead to more efficient hardware utilization.

---

## 16. Citation

This implementation is based on the original Mamba paper. If you use this model or its concepts in your research, please cite the foundational work:

```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

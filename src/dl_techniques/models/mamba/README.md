# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18%2B-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured Keras 3 implementation of the **Mamba V1 and Mamba V2** architectures. This implementation provides pure foundation models, separating the core sequence modeling logic from task-specific heads for maximum flexibility.

The architecture's key feature is its **Selective State Space Model (SSM)** core, designed to capture long-range dependencies with **linear-time complexity**. Mamba V2 introduces the **State Space Duality (SSD)** framework, resulting in a refined core layer that is **2-8x faster** and more competitive with highly optimized attention mechanisms.

---

## Table of Contents

1. [Overview: What is Mamba and Why It Matters](#1-overview-what-is-mamba-and-why-it-matters)
2. [The Problem Mamba Solves](#2-the-problem-mamba-solves)
3. [How Mamba Works: Core Concepts](#3-how-mamba-works-core-concepts)
4. [Architecture Deep Dive: Mamba V1 vs. V2](#4-architecture-deep-dive-mamba-v1-vs-v2)
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

**Mamba** is a family of sequence models that uses a **Selective State Space Model (SSM)** to achieve linear-time complexity while matching or exceeding the performance of Transformers on many tasks. It is designed to overcome the quadratic scaling bottleneck of attention, making it highly efficient for processing very long sequences.

This implementation provides the core Mamba architectures (V1 and V2) as **foundation models**. They take tokenized text as input and output a sequence of contextualized embeddings, which can be used for a wide variety of downstream tasks.

### Key Innovations

1.  **Selective State Space Model (S6)**: The core of Mamba. Unlike previous SSMs, its parameters are functions of the input data. This allows the model to selectively remember or ignore information at each timestep, mimicking the content-aware nature of attention but far more efficiently.
2.  **Linear-Time Complexity**: Mamba processes sequences in O(L) time, a significant improvement over the O(L²) complexity of standard Transformers. This makes it feasible to work with sequences containing millions of tokens.
3.  **State Space Duality (SSD) in Mamba-2**: The V2 architecture is built on a new theoretical framework that connects SSMs and attention. Its core algorithm (SSD) uses a hybrid approach, combining a recurrent mode for efficiency with a parallel, attention-like quadratic mode on small chunks. This allows it to leverage highly optimized matrix multiplication units on GPUs, leading to a **2-8x speedup** over Mamba V1.
4.  **Modernized Block Design (V2)**: Mamba V2 refines the block architecture with parallel parameter projections and multi-head SSMs, drawing inspiration from optimizations in the Transformer ecosystem.

### Why Mamba Matters

Mamba addresses the classic trade-offs in sequence modeling by combining the strengths of RNNs and Transformers.

```
┌────────────────────────────────────────────────────────────────────┐
│              The Sequence Modeling Trilemma                        │
├──────────────┬────────────────────────┬────────────────────────────┤
│ Model        │ Pros                   │ Cons                       │
├──────────────┼────────────────────────┼────────────────────────────┤
│ Transformers │ Excellent performance  │ Quadratic O(L²) complexity │
│ RNNs (LSTMs) │ Linear O(L) complexity │ Struggle with long context │
│ Mamba Family │ Linear O(L) complexity │                            │
│              │ & strong performance   │ (Effectively resolves the  │
│              │                        │ trilemma)                  │
└────────────────────────────────────────────────────────────────────┘
```
Mamba's efficiency and power unlock new possibilities for long-sequence tasks:

-   🧬 **Genomics**: Modeling entire DNA sequences with millions of base pairs.
-   **Audio & Video**: Processing high-resolution, long-form multimedia streams.
-   **Long-Document NLP**: Question answering and summarization of books, articles, or codebases.

---

## 2. The Problem Mamba Solves

### The Challenge of Selective Compression

Effective sequence models must compress a long history of context into a compact state. Transformers "cheat" by not compressing at all—they keep the entire history (the KV cache) available, leading to their power but also their quadratic cost. Traditional RNNs compress context into a fixed-size state, but their simple, uniform updates make it hard to selectively retain important information.

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
│  - Mamba's selection mechanism allows it to "remember" the  │
│    colored tokens by strengthening its state, and "forget"  │
│    the filler by weakening it, all in linear time.          │
└─────────────────────────────────────────────────────────────┘
```

Mamba introduces this **selection mechanism** directly into its state-space formulation by making its core parameters data-dependent.

---

## 3. How Mamba Works: Core Concepts

### The High-Level Architecture

The model is a simple, homogeneous stack of Mamba residual blocks. Each block applies the core Selective SSM logic within a standard pre-norm residual architecture.

```
Input (Token IDs) ───► Embedding Layer
                         │
                         ▼
                  ┌────────────────────┐
                  │ MambaResidualBlock │ (Pre-Norm + MambaLayer)
                  └────────────────────┘
                         │ (x N Layers)
                         ▼
                  ┌────────────────────┐
                  │ MambaResidualBlock │
                  └────────────────────┘
                         │
                         ▼
                  Final Normalization Layer
                         │
                         ▼
                  Output (Hidden States)
```

### Data Flow Within a Mamba V1 Block

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Mamba V1 Block Complete Data Flow                   │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: RESIDUAL & NORMALIZATION
─────────────────────────────────
Input Hidden State + Previous Residual ───► LayerNorm ───► Normalized Input


STEP 2: MAMBA LAYER V1 CORE LOGIC
───────────────────────────────
Normalized Input (B, L, D)
    │
    ├─► Linear Projection ──► Split into `x` (main) and `z` (gate) streams
    │
    ├── `x` Stream (Main Path):
    │   ├─► Causal 1D Convolution
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
    Element-wise product: `y` * SiLU(`z`) ──► Block Output
```

---

## 4. Architecture Deep Dive: Mamba V1 vs. V2

### 4.1 `MambaLayer` (V1)

-   **Purpose**: The core engine of the original Mamba architecture.
-   **Architecture**:
    1.  **Sequential Parameterization**: The SSM parameters (`Δ`, `B`, `C`) are computed *sequentially*. The input is first passed through a causal 1D convolution, and the output of the convolution is then used to derive the parameters.
    2.  **Single-Head SSM**: The entire inner dimension `d_inner` is processed by a single state space model.
    3.  **Algorithm**: Relies on a hardware-fused **selective scan** algorithm, which is a highly optimized recurrent operation.

### 4.2 `Mamba2Layer` (V2)

-   **Purpose**: An evolution of the Mamba layer based on the State Space Duality (SSD) framework, optimized for speed and drawing parallels to Transformer designs.
-   **Architecture**:
    1.  **Parallel Parameterization**: The input is projected once, and all SSM parameters (`Δ`, `B`, `C`) and intermediate features are derived *in parallel* from this single projection. This is more amenable to model parallelism.
    2.  **Multi-Head SSM**: The state space is divided into multiple heads (`nheads`), each with a smaller dimension (`headdim`). This is analogous to multi-head attention and improves model capacity.
    3.  **Algorithm (SSD)**: V2 uses the SSD algorithm, which processes the sequence in chunks. Within each chunk, it uses a fast, attention-like **quadratic mode**. Between chunks, it uses an efficient **recurrent mode** to pass the state. This hybrid approach allows it to heavily leverage matrix multiplication hardware.
    4.  **Normalization**: Uses RMSNorm by default, a simpler and often more efficient alternative to LayerNorm.

### 4.3 Key Differences Summarized

| Feature | Mamba V1 | Mamba V2 |
| :--- | :--- | :--- |
| **Core Algorithm** | Fused Selective Scan | State Space Duality (SSD) |
| **Parameter Generation**| **Sequential**: `Input -> Conv -> SSM Params` | **Parallel**: `Input -> All Params` |
| **SSM Structure** | Single large SSM | Multi-Head SSM (`nheads` x `headdim`) |
| **Normalization** | Layer Normalization | RMS Normalization (default) |
| **Hardware Utilization**| Optimized custom recurrence | Optimized matrix multiplication |

---

## 5. Quick Start Guide

### Installation

```bash
# Ensure you have the required dependencies
pip install keras==3.* tensorflow==2.16.* numpy
```

### Your First Mamba Model (30 seconds)

Let's build a Mamba V1-based model. Creating a V2 model is as simple as swapping `Mamba` with `Mamba2`.

```python
import keras
import numpy as np
from dl_techniques.models.mamba.mamba_v1 import Mamba
# To use Mamba V2, simply import it instead:
# from dl_techniques.models.mamba.mamba_v2 import Mamba2

# 1. Define model configuration
VOCAB_SIZE = 10000
NUM_CLASSES = 3

# 2. Create the Mamba V1 foundation model from a variant
mamba_encoder = Mamba.from_variant(
    "130m",  # This is the "base" variant
    vocab_size=VOCAB_SIZE,
    name="mamba_encoder"
)

# 3. Build a full classification model
inputs = {"input_ids": keras.Input(shape=(None,), dtype="int32")}
sequence_output = mamba_encoder(inputs)["last_hidden_state"]
pooled_output = keras.layers.GlobalAveragePooling1D()(sequence_output)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(pooled_output)
model = keras.Model(inputs, outputs, name="mamba_classifier")

# 4. Compile and inspect the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
```

---

## 6. Component Reference

### 6.1 Model Classes

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`Mamba`** | `...mamba.mamba_v1` | The main Keras `Model` for the V1 architecture. |
| **`Mamba2`** | `...mamba.mamba_v2` | The main Keras `Model` for the V2 architecture. |

### 6.2 Core Building Blocks

| Layer | Location | Purpose |
| :--- | :--- | :--- |
| **`MambaLayer`** | `...mamba.components` (V1) | Core V1 block with sequential parameterization. |
| **`Mamba2Layer`** | `...mamba.components_v2` | Core V2 block with multi-head SSM and parallel parameterization. |
| **`MambaResidualBlock`** | `...mamba.components` (V1) | Pre-norm residual wrapper for `MambaLayer`. |
| **`Mamba2ResidualBlock`**| `...mamba.components_v2`| Pre-norm residual wrapper for `Mamba2Layer`. |

---

## 7. Configuration & Model Variants

This implementation provides standard configurations matching the original papers.

### Mamba V1 Variants

| Variant | d_model | Layers | Parameters (Approx.) |
|:---:|:---:|:---:|:---:|
| **`130m`** (`base`) | 768 | 24 | ~130M |
| **`370m`** | 1024 | 24 | ~370M |
| **`790m`** | 1024 | 48 | ~790M |
| **`1.4b`** | 1536 | 48 | ~1.4B |
| **`2.8b`** | 2560 | 64 | ~2.8B |

### Mamba V2 Variants

| Variant | d_model | Layers | Parameters (Approx.) |
|:---:|:---:|:---:|:---:|
| **`130m`** (`base`) | 768 | 24 | ~130M |
| **`370m`** | 1024 | 24 | ~370M |
| **`780m`** | 1536 | 36 | ~780M |
| **`1.4b`** | 2048 | 48 | ~1.4B |
| **`2.8b`** | 2560 | 64 | ~2.8B |

---

## 8. Comprehensive Usage Examples

### Example: Using Mamba V2 for Token Classification

Mamba V2's efficiency and powerful representations make it great for dense prediction tasks like Named Entity Recognition (NER).

```python
import keras
from dl_techniques.models.mamba.mamba_v2 import Mamba2

# 1. Configuration for NER
VOCAB_SIZE = 50257
NUM_NER_TAGS = 17 

# 2. Create the Mamba V2 encoder
mamba2_encoder = Mamba2.from_variant(
    "370m",
    vocab_size=VOCAB_SIZE,
)

# 3. Build the final NER model
inputs = {"input_ids": keras.Input(shape=(None,), dtype="int32")}
mamba_features = mamba2_encoder(inputs)["last_hidden_state"]

# Add a token classification head
ner_head = keras.layers.Dense(NUM_NER_TAGS, activation="softmax")(mamba_features)
ner_model = keras.Model(inputs, ner_head)

ner_model.summary()
```

---

## 9. Advanced Usage Patterns

### Pattern: Integrating with Task-Specific Heads (V1)

The `create_mamba_with_head` function in `mamba_v1.py` provides a powerful pattern for combining the V1 Mamba encoder with pre-built task heads.

```python
from dl_techniques.models.mamba.mamba_v1 import create_mamba_with_head
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType

seq_cls_task = NLPTaskConfig(
    name="sentiment_analysis",
    task_type=NLPTaskType.SEQUENCE_CLASSIFICATION,
    num_classes=3,
    vocab_size=50257
)

model = create_mamba_with_head(
    mamba_variant="130m",
    task_config=seq_cls_task
)
model.summary()
```

---

## 10. Performance Optimization

### Mixed Precision Training

For larger Mamba models, mixed precision can dramatically speed up training and reduce memory usage on compatible GPUs.

```python
import keras
keras.mixed_precision.set_global_policy('mixed_float16')

model = Mamba2.from_variant("780m", vocab_size=50257)
# ... compile and fit ...
```

### XLA Compilation

Use `jit_compile=True` for graph compilation. This can provide a significant speed boost.

```python
model.compile(optimizer="adam", loss="...", jit_compile=True)
```

---

## 11. Training and Best Practices

### Optimizer

The **AdamW** optimizer is highly recommended. Its decoupled weight decay is a crucial regularizer for modern architectures. A cosine learning rate schedule with a brief warmup period is a strong default.

### Handling Long Sequences

-   **Linear Scaling**: Memory and computation scale linearly, O(L), with sequence length. You can train on sequences of `32k`, `64k`, or even longer without the memory explosion seen in Transformers.
-   **Constant-Time Inference**: During autoregressive inference, Mamba maintains a fixed-size state. This means generation time is constant per token and throughput is significantly higher than Transformers.

---

## 12. Serialization & Deployment

The `Mamba`, `Mamba2`, and all their custom layers are fully serializable using Keras 3's modern `.keras` format.

### Saving and Loading

```python
import keras
# Import all custom classes used by the model you are loading
from dl_techniques.models.mamba.mamba_v2 import Mamba2, Mamba2Layer, Mamba2ResidualBlock

model = Mamba2.from_variant("130m", vocab_size=1000)
# ... train the model ...
model.save('my_mamba2_model.keras')

# In a new session, import the custom classes to load the model
loaded_model = keras.models.load_model('my_mamba2_model.keras')
loaded_model.summary()
```

---

## 13. Troubleshooting & FAQs

**Q: What is the main difference between Mamba V1 and V2?**

A: The main differences are in the core SSM layer's architecture and algorithm. **V2 is 2-8x faster** because its **State Space Duality (SSD)** algorithm is designed to leverage matrix multiplication units on GPUs. Architecturally, V2 uses **parallel parameter projections** and a **multi-head SSM**, making it more analogous to modern Transformer blocks and more amenable to model parallelism.

**Q: Is Mamba a type of RNN?**

A: Yes, Mamba can be seen as a modern, highly structured Recurrent Neural Network (RNN). Its "Selective Scan" is a recurrent operation. However, unlike traditional RNNs, it is designed to be trained efficiently in parallel and to avoid vanishing gradient issues.

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

Mamba makes this system "selective" by making the key parameters `Δ`, `B`, and `C` functions of the input `x`. This breaks the time-invariance property of older SSMs, which is what prevented them from modeling discrete, content-rich data effectively.

### State Space Duality (SSD) in Mamba-2

The V2 paper reveals a duality between SSMs and structured forms of attention. The SSD algorithm exploits this by breaking the sequence into small chunks.
1.  **Intra-Chunk (Quadratic Mode)**: Within each small chunk, the computation is performed using an attention-like parallel formulation that is extremely fast on modern hardware due to its reliance on matrix multiplication.
2.  **Inter-Chunk (Linear Mode)**: The hidden state is passed from one chunk to the next using the efficient linear-time recurrent mode.

This hybrid approach gives Mamba V2 the linear scaling of an RNN with the hardware-friendly performance of attention.

---

## 16. Citation

This implementation is based on the original Mamba papers. If you use these models in your research, please cite the foundational works:

-   **Mamba V1**:
    ```bibtex
    @article{gu2023mamba,
      title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
      author={Gu, Albert and Dao, Tri},
      journal={arXiv preprint arXiv:2312.00752},
      year={2023}
    }
    ```
-   **Mamba V2 and State Space Duality**:
    ```bibtex
    @article{dao2024transformers,
      title={Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
      author={Dao, Tri and Gu, Albert},
      journal={arXiv preprint arXiv:2405.21060},
      year={2024}
    }
    ```
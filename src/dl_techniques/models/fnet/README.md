# FNet: Mixing Tokens with Fourier Transforms

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of the **FNet** architecture in **Keras 3**, based on the paper ["FNet: Mixing Tokens with Fourier Transforms"](https://arxiv.org/abs/2105.03824) by Lee-Thorp et al. (2021).

This architecture is designed as a **pure encoder**, separating core mixing logic from task-specific heads. This modular design makes it ideal for pre-training, fine-tuning, and complex multi-task learning workflows.

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

### What is FNet?

**FNet** is a Transformer-like architecture that replaces the computationally expensive self-attention mechanism with a simple, unparameterized **2D Fourier Transform**. It demonstrates that for many NLP tasks, the complex, content-aware mixing of self-attention can be effectively replaced by a much more efficient, deterministic method.

For each encoder block, FNet performs token mixing by:
-   Applying a Fast Fourier Transform (FFT) along the **sequence** dimension.
-   Applying another FFT along the **hidden (feature)** dimension.

### Key Innovations

1.  **Parameter-Free Token Mixing**: The Fourier Transform requires **zero** learnable weights, significantly reducing the model's memory footprint.
2.  **Exceptional Efficiency**: It reduces the algorithmic complexity of the token-mixing step from `O(N²)` (quadratic) to `O(N log N)` (quasi-linear).
3.  **BERT-Level Performance**: Despite its simplicity, FNet achieves 92-97% of BERT's accuracy on GLUE benchmarks while training up to **80% faster**.
4.  **Drop-in Replacement**: The `FNetEncoderBlock` serves as a direct, highly efficient substitute for standard Transformer encoder blocks.

### Why FNet Matters

**The Standard Transformer (BERT) Bottleneck**:
To classify a long document (e.g., 4096 tokens), a standard self-attention mechanism must compare every token to every other token.
*   **Cost**: $4096^2 \approx 16.7$ million comparisons per head, per layer.
*   **Result**: Massive memory consumption and slow processing, making long sequences infeasible.

**The FNet Solution**:
FNet replaces self-attention with a 2D Fourier Transform.
*   **Cost**: Proportional to $4096 \times \log(4096) \approx 49,000$ operations.
*   **Result**: Drastically faster training and inference, enabling the processing of much longer sequences on commodity hardware.

### Real-World Applications

FNet unlocks capabilities in domains where sequence length is the primary constraint:
-   **Genomics**: Processing extremely long DNA/protein sequences.
-   **Long-Document Analysis**: Summarizing or classifying legal contracts, books, or scientific papers.
-   **Real-Time NLP**: Low-latency deployment for on-device or edge applications.

---

## 2. The Problem FNet Solves

### The Tyranny of Quadratic Complexity

Standard Transformer models are fundamentally limited by the memory and compute requirements of the self-attention matrix.

```text
┌─────────────────────────────────────────────────────────────┐
│  Standard Transformer (Self-Attention)                      │
│                                                             │
│  The Core Bottleneck:                                       │
│    Complexity is O(L² * H)                                  │
│    Doubling sequence length = 4x computation & memory.      │
│                                                             │
│  Consequences:                                              │
│  1. Slow Training: Iterations take too long on long text.   │
│  2. OOM Errors: Attention matrices consume GBs of VRAM.     │
│  3. Hard Limits: Most models cap at 512 or 1024 tokens.     │
└─────────────────────────────────────────────────────────────┘
```

**Scaling Cost Example:**
-   **512 tokens**: Baseline.
-   **4096 tokens**: 64x more memory/compute than at 512 tokens.

### How FNet Changes the Game

FNet operates on the principle that **token mixing** is the primary goal of attention, and this can be approximated without learnable weights.

```text
┌─────────────────────────────────────────────────────────────┐
│  FNet's Efficiency Principle                                │
│                                                             │
│  1. Mechanism: Replace learnable attention with a fixed,    │
│     parameter-free Fourier Transform.                       │
│                                                             │
│  2. Global Mixing: The FFT allows every token to affect     │
│     every other token in quasi-linear time.                 │
│                                                             │
│  3. Burden Shift: The Feed-Forward Network (FFN) handles    │
│     all content-based, non-linear feature extraction.       │
│                                                             │
│  Benefits:                                                  │
│  - Linear-ish scaling with sequence length.                 │
│  - Up to 80% faster training on GPUs.                       │
│  - Significantly fewer parameters to store and load.        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How FNet Works: Core Concepts

### The Two-Sublayer Encoder Block

An FNet model consists of a stack of `FNetEncoderBlock` layers. The structure mirrors the Transformer, replacing only the first sub-layer.

```text
┌─────────────────────────────────────────────────────────────┐
│                    FNet Encoder Block                       │
│                                                             │
│  ┌─────────────────┐      ┌───────────────────────────────┐ │
│  │  Token Mixing   │      │        Channel Mixing         │ │
│  │    (FFT)        ├──────►      (Feed-Forward Net)       │ │
│  │                 │      │                               │ │
│  │  - Applies 2D   │      │  - Same as standard           │ │
│  │    FFT (seq/dim)│      │    Transformer FFN.           │ │
│  │  - Parameter    │      │  - Learns content-based       │ │
│  │    Free         │      │    features position-wise.    │ │
│  └─────────────────┘      └───────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### The Fourier Transform for Token Mixing

For an input tensor `X` of shape `(seq_len, hidden_dim)`:

1.  **Sequence FFT**: `FFT(X, axis=0)`. Mixes information across all tokens. Every output element is a linear combination of all input tokens.
2.  **Hidden FFT**: `FFT(X, axis=1)`. Mixes information within the feature vector of each token.
3.  **Real Part Extraction**: `Real(...)`. The imaginary component is discarded.

**Formula**: $Y = \Re(\mathcal{F}_{seq}(\mathcal{F}_h(X)))$

### The Complete Data Flow

```text
STEP 1: Embedding
─────────────────
Input IDs (B, L)
    │
    ├─► Sum(Token + Position + Type Embeddings)
    └─► LayerNorm -> Dropout -> (B, L, H)

STEP 2: FNet Encoder Stack (Repeated N times)
─────────────────────────────────────────────
Input (B, L, H)
    │
    ├─► FNetEncoderBlock
    │   ├─► Fourier Transform (Mixing)
    │   ├─► Residual + LayerNorm
    │   ├─► Feed-Forward Network (Projections)
    │   └─► Residual + LayerNorm
    │
    └─► Next Block...

STEP 3: Task-Specific Head
──────────────────────────
Final Hidden State (B, L, H)
    │
    ├─► Extract [CLS] or Pool
    └─► Dense Layer -> Logits
```

---

## 4. Architecture Deep Dive

### 4.1 Embedding Layer (`BertEmbeddings`)

FNet utilizes standard BERT-style embeddings to ensure compatibility with existing tokenizers and transfer learning paradigms.

-   **Token Embeddings**: Learnable lookup table.
-   **Position Embeddings**: Learnable vectors (critical for FNet, as FFT is position-invariant without them).
-   **Token Type Embeddings**: Segment IDs for sentence pairs.

### 4.2 FNet Encoder Block (`FNetEncoderBlock`)

This is the core architectural contribution.

#### Architecture Diagram

```text
Input: (B, L, H)
  │
  ▼
┌──────────────────────────┐
│  FNetFourierTransform    │  ← Mixing Sub-layer
└──────────────────────────┘
  │
  │ (Residual Connection)
  ▼
Add & Norm
  │
  ▼
┌──────────────────────────┐
│ Feed-Forward Network     │  ← Knowledge Sub-layer
│ (MLP / SwiGLU / GEGLU)   │
└──────────────────────────┘
  │
  │ (Residual Connection)
  ▼
Add & Norm
  │
  ▼
Output: (B, L, H)
```

#### Design Q&A

**Q: Why a 2D Fourier Transform?**
A: The FFT along the sequence dimension replaces Self-Attention. The authors found that adding an FFT along the hidden dimension provided a small, consistent performance boost by encouraging richer feature interaction within tokens.

**Q: What is the trade-off?**
A: FNet sacrifices **adaptivity**. In Self-Attention, mixing weights are dynamic (computed from the input). In FNet, mixing is static. The model relies entirely on the Feed-Forward layers to learn content-specific logic.

---

## 5. Quick Start Guide

### Installation

```bash
# Core dependencies
pip install keras>=3.0 tensorflow>=2.18 numpy
```

### Your First FNet Model

Here is how to create a sentiment classifier using the `fnet_tiny` configuration.

```python
import keras
import numpy as np

# Local imports
from dl_techniques.models.fnet.model import create_fnet_with_head, FNet
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType

# 1. Define the task configuration
task_config = NLPTaskConfig(
    name="sentiment_analysis",
    task_type=NLPTaskType.SENTIMENT_ANALYSIS,
    num_classes=2
)

# 2. Instantiate the model using the factory
# We use 'tiny' for demonstration; use 'base' for real work.
model = create_fnet_with_head(
    fnet_variant="tiny",
    task_config=task_config,
    pretrained=False,     # Set True to download weights
    sequence_length=128   # Fixed length is recommended for FFT efficiency
)

# 3. Compile
model.compile(
    optimizer='adamw',    # AdamW is preferred for Transformers
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# 4. Mock Training Run
dummy_inputs = {
    "input_ids": np.random.randint(0, FNet.DEFAULT_VOCAB_SIZE, (4, 128)),
    "attention_mask": np.ones((4, 128), dtype="int32"),
    "token_type_ids": np.zeros((4, 128), dtype="int32"),
}
dummy_labels = np.array([0, 1, 0, 1])

loss, acc = model.train_on_batch(dummy_inputs, dummy_labels)
print(f"✅ Training Step - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
```

---

## 6. Component Reference

### 6.1 `FNet` (Model Class)

**Location**: `dl_techniques.models.fnet.model.FNet`

The foundation encoder. It outputs raw hidden states.

```python
from dl_techniques.models.fnet.model import FNet

encoder = FNet.from_variant(
    "base",
    pretrained=False,
    vocab_size=30522
)
```

### 6.2 `FNetEncoderBlock`

**Location**: `dl_techniques.layers.fnet_encoder_block.FNetEncoderBlock`

The layer class. Useful for building custom architectures outside of the standard FNet topology.

```python
from dl_techniques.layers.fnet_encoder_block import FNetEncoderBlock

block = FNetEncoderBlock(
    intermediate_dim=3072,
    dropout_rate=0.1,
    normalization_type='rms_norm', # Modern option
    ffn_type='swiglu'              # Modern option
)
```

### 6.3 `create_fnet_with_head`

**Location**: `dl_techniques.models.fnet.model.create_fnet_with_head`

A high-level factory to attach NLP heads (Classification, NER, etc.) to the encoder.

---

## 7. Configuration & Model Variants

| Variant | Hidden Size | Layers | FFN Size | Params (Approx) | Use Case |
| :--- | :---: |:---: |:---: |:---: |:--- |
| **`large`** | 1024 | 24 | 4096 | 340M | Max Performance |
| **`base`** | 768 | 12 | 3072 | 110M | General Purpose |
| **`small`** | 512 | 6 | 2048 | 35M | Edge / Speed |
| **`tiny`** | 256 | 4 | 512 | 4M | Mobile / Embedded |

**Recommendation**: Start with `base`. If inference latency is critical, move to `small`.

---

## 8. Comprehensive Usage Examples

### Example 1: Pure Feature Extraction

Use FNet to generate embeddings for clustering or downstream non-neural models.

```python
import keras
from dl_techniques.models.fnet.model import FNet

# Load encoder
fnet_encoder = FNet.from_variant("base", pretrained=False)

# Define inputs
inputs = {
    "input_ids": keras.Input(shape=(None,), dtype="int32"),
    "attention_mask": keras.Input(shape=(None,), dtype="int32"),
}

# Forward pass
outputs = fnet_encoder(inputs)
# Extract the [CLS] token embedding (first token)
cls_embedding = outputs["last_hidden_state"][:, 0, :]

extractor = keras.Model(inputs, cls_embedding)
```

### Example 2: Named Entity Recognition (NER)

```python
from dl_techniques.models.fnet.model import create_fnet_with_head
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType

# Configuration for 9 NER classes (e.g., CoNLL-2003)
ner_config = NLPTaskConfig(
    name="ner",
    task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
    num_classes=9
)

# Create model with fixed sequence length for efficiency
ner_model = create_fnet_with_head(
    fnet_variant="base",
    task_config=ner_config,
    pretrained=False,
    sequence_length=256
)

# Output shape: (batch_size, 256, 9)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Modernizing the Architecture

You are not stuck with the 2021 architecture. This implementation allows you to inject modern components like **RMSNorm** (for stability) and **SwiGLU** (for performance).

```python
from dl_techniques.models.fnet.model import FNet

modern_fnet = FNet.from_variant(
    "base",
    # Pass kwargs down to FNetEncoderBlock
    normalization_type="rms_norm",
    normalization_position="pre", # Pre-Norm is generally more stable
    ffn_type="swiglu"
)
```

### Pattern 2: Multi-Task Learning

Since FNet is a pure encoder, you can share it across multiple heads.

```python
# 1. Shared Encoder
shared_encoder = FNet.from_variant("base")
inputs = {...} # Define inputs
encoder_out = shared_encoder(inputs)

# 2. Heads
sentiment_head = create_nlp_head(
    NLPTaskConfig(name="sent", task_type=NLPTaskType.SENTIMENT_ANALYSIS, num_classes=2),
    input_dim=768
)
ner_head = create_nlp_head(
    NLPTaskConfig(name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION, num_classes=9),
    input_dim=768
)

# 3. Connect
y_sent = sentiment_head(encoder_out)
y_ner = ner_head(encoder_out)

# 4. Multi-output Model
model = keras.Model(inputs=inputs, outputs={"sent": y_sent, "ner": y_ner})
```

---

## 10. Performance Optimization

### Mixed Precision Training

FNet is compute-bound by the FFN layers. Using `float16` or `bfloat16` provides significant speedups.

```python
# Set policy globally
keras.mixed_precision.set_global_policy('mixed_float16')

# Model creation automatically respects this policy
model = FNet.from_variant("base")
```

### XLA Compilation (JIT)

For static sequence lengths, XLA compilation is highly effective with FNet.

```python
import tensorflow as tf

@tf.function(jit_compile=True)
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        preds = model(inputs, training=True)
        loss = loss_fn(labels, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
```

---

## 11. Training and Best Practices

1.  **Fixed Sequence Lengths**: The FFT algorithm is optimized for fixed sizes (ideally powers of 2, e.g., 512). While the model accepts dynamic shapes, using fixed padding/truncation during training is recommended for maximum throughput.
2.  **Optimizer**: Use `AdamW`.
3.  **Learning Rate**: Standard Transformer schedules apply. Linear decay with a warmup period (e.g., 10% of steps). Peak LR is usually around `1e-4` to `5e-5` for fine-tuning.
4.  **Batch Size**: Because FNet uses less memory than BERT, you can often use significantly larger batch sizes (2x-4x larger).

---

## 12. Serialization & Deployment

This implementation is fully compliant with Keras 3 serialization standards.

```python
# Save to the zip-based .keras format
model.save('my_fnet_model.keras')

# Load cleanly in a new process (or inference server)
loaded_model = keras.models.load_model('my_fnet_model.keras')
```

---

## 13. Testing & Validation

The codebase includes verification utilities to ensure architectural correctness.

```python
# Run the included test suite
python -m pytest tests/models/fnet
```

*See `test_fnet.py` in the repository for unit test examples covering shape inference, serialization, and numerical stability.*

---

## 14. Troubleshooting & FAQs

**Issue: "My accuracy is 2-3% lower than RoBERTa."**
*   **Context**: This is expected. FNet trades a small amount of accuracy for massive speed and scalability.
*   **Tip**: Try training for more epochs. FNet often converges faster in wall-clock time, but might need more steps to settle than an Attention model.

**Issue: "Training is slow with variable sequence lengths."**
*   **Context**: FFT operations on TPUs/GPUs trigger recompilation if shapes change frequently.
*   **Tip**: Pad your batches to fixed buckets (e.g., 128, 256, 512) or a single fixed length.

**Q: Can I use FNet for generation (Decoder)?**
*   **A**: No. The Fourier Transform is a global operation (bidirectional). It cannot easily be masked for causal (left-to-right) generation. FNet is strictly an Encoder.

---

## 15. Technical Details

### Complexity Analysis

Let $L$ be sequence length and $H$ be hidden size.

| Component | Complexity | Scaling |
| :--- | :--- | :--- |
| **Self-Attention** | $O(L^2 \cdot H)$ | Quadratic (Bottleneck) |
| **FNet Mixing (FFT)** | $O(L \log L \cdot H)$ | Quasi-Linear |
| **Feed-Forward** | $O(L \cdot H^2)$ | Linear (w.r.t $L$) |

### The Role of the Feed-Forward Network

In FNet, the mixing layer (FFT) is linear and static. Therefore, the **Feed-Forward Network (FFN)** bears the entire burden of learning non-linear, content-specific relationships. This is why FNet can benefit from advanced FFN structures like GLU variants more than standard Transformers might.

---

## 16. Citation

If you use this implementation or the FNet architecture in your research, please cite the original paper:

```bibtex
@article{lee2021fnet,
  title={FNet: Mixing Tokens with Fourier Transforms},
  author={Lee-Thorp, James and Ainslie, Joshua and Eckstein, Ilya and Ontanon, Santiago},
  journal={arXiv preprint arXiv:2105.03824},
  year={2021}
}
```
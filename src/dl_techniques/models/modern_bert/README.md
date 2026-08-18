# ModernBERT: A High-Performance BERT Successor

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

An advanced, production-ready Keras 3 implementation of **ModernBERT**, a successor to the classic BERT architecture. This model is based on the paper "[Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663)". ModernBERT integrates a suite of contemporary deep learning techniques to deliver superior performance, faster processing for long contexts, and enhanced training stability.

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
4.  **Alternating Local & Global Attention**: Employs a hybrid attention strategy. Most layers use **windowed (local) attention**, while periodic **global attention** layers (every 3rd layer) ensure that long-range dependencies are captured. In the paper this is what makes the 8192 native sequence length affordable. **In this implementation it is not** — the reused `window` layer pads every window to `window_size**2` slots, so for `base` and `large` (`window_size=128`, threshold 16384 > the 8192 max position) no admissible length is ever windowed and their local layers are *more* expensive than global ones, not less. `tiny` (`window_size=64`, threshold 4096) is the one exception: for `4097 <= L <= 8192` it really does partition into four windows. See § 4.3; this is a known deviation, and an open decision.
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
  1. Replace most global attention with windowed attention.
  2. Insert global attention layers periodically to aggregate information.
  3. Use RoPE for robust positional information up to 8192 tokens.
  4. Train on a massive, modern dataset including code.
  5. Benefit (AS PUBLISHED): near-linear attention complexity, making it a
     versatile and efficient backbone for modern NLP tasks.

  NOT REPRODUCED HERE: step 1's cost benefit. The `window` layer this package
  reuses pads every window to `window_size**2` slots, so its cost is
  O(max(L, M) * M) with M = window_size**2 -- a CONSTANT O(M^2) floor for
  L <= M rather than a linear-in-L saving. For base/large (window_size=128,
  M = 16384 > the 8192 max position) every local layer is dense attention
  over a padded 16384-slot window at every admissible L. For tiny
  (window_size=64, M = 4096) it is dense for L <= 4096 and a genuine 4-window
  partition for 4097 <= L <= 8192. Modelling quality is unaffected in kind;
  the efficiency claim is inverted for base/large. See § 4.3.
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
│                  │ ModernBertEmbeddings │ (No positional term)    │
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
  └─► LayerNorm ──► Attention ──► Add ◄──────────────────────────┘
                      (Local/Global)               │
                                                   │
  ┌────────────────────────────────────────────────┘
  │
  ├─► Residual Path 2 ───────────────────────────────┐
  │                                                  │
  └─► LayerNorm ──► GeGLU FFN ──► Dropout ──► Add ◄────────────┘
                                                     │
                                                     ▼
                                                Final Output
```

> The asymmetry is real, not a drawing shortcut. These blocks are
> `TransformerLayer(normalization_position='pre', ...)`, and `TransformerLayer`
> applies its `dropout_rate` layer to the **FFN sub-block only** — there is no
> dropout step after attention. `attention_probs_dropout_prob` is not an output
> dropout either: it becomes the attention sub-layer's own internal
> attention-weight dropout. Being pre-LN, neither branch ends in a
> normalization ("Add & Norm" is the post-LN shape); the only trailing
> normalization is the single final one after the whole stack.

---

## 4. Architecture Deep Dive

### 4.1 `ModernBertEmbeddings` (no positional term)

-   **Purpose**: To convert input token IDs into dense vector representations.
-   **Components**:
    1.  **Token Embeddings**: A standard lookup table for the modern BPE vocabulary (50,368 tokens).
    2.  **Token Type Embeddings**: A learnable embedding to distinguish between different sentences. (Note: While present for BERT compatibility, ModernBERT does not use token type IDs during pre-training).
-   **Functionality**: The token and token type embeddings are summed, followed by a `LayerNormalization` and `Dropout` layer. This layer itself adds **no** positional term and its output is permutation-equivariant; the positional signal is injected downstream, by the attention layers (§ 4.3).

### 4.2 `TransformerLayer` with Pre-LN and GeGLU

-   **Purpose**: The core building block of the encoder.
-   **Architecture**:
    1.  **Pre-Normalization**: The input is first passed through a `LayerNormalization` layer.
    2.  **Multi-Head Attention**: The normalized input is fed into the attention mechanism (either windowed or global). RoPE is applied to the queries and keys of the **global** layers only; see § 4.3 for what the windowed layers use instead.
    3.  **First Residual Connection**: The output of the attention block is added back to the original input.
    4.  **Second Pre-Normalization**: The result of the first residual connection is normalized again.
    5.  **GeGLU Feed-Forward Network**: The normalized result is processed by a gated feed-forward network.
    6.  **Second Residual Connection**: The output of the FFN is added back to its input.

### 4.3 Hybrid Attention

-   **Global Attention**: Every 3rd layer uses standard global attention with **RoPE** on its queries and keys. This allows information to be exchanged across the entire 8192-token sequence, ensuring that long-range dependencies are captured. It is built as the factory's `group_query` type with `num_kv_heads == num_heads`, which is arithmetically plain multi-head attention — that is the only registry entry that reaches plain self-attention *and* carries RoPE. `multi_head` declares no RoPE parameter. Until 2026-08-17 `create_attention_layer` silently dropped keys it does not declare, so `attention_args={"use_rope": True}` on a `multi_head` layer built cleanly and did nothing; that factory is now strict and raises a `ValueError` naming the key, so the same mistake fails at construction.
-   **Windowed (Local) Attention**: Most layers (2 out of every 3) use windowed attention. **This implementation deviates from the paper**, and the deviation is deliberate and documented rather than hidden: the reused `window` attention layer is a *spatial* one. It reshapes the `(B, L, D)` token sequence into a synthetic `ceil(sqrt(L))`-square grid and attends inside `window_size`-square blocks of that grid. Consequences, both measurable:
    -   A local layer's neighbourhood is **not** a contiguous 1-D window. With `L=16` and `window_size=2` the grid is 4x4 and token 0's window is `{0, 1, 4, 5}` — tokens 2 and 3 are invisible to it while token 4 is not. Pinned by `tests/test_models/test_modern_bert/test_positional_signal.py::TestLocalWindowAdjacencyIsSynthetic`.
    -   Whenever `L <= window_size**2` (16384 at the default `window_size=128`) a single window covers the whole padded grid, so the layer is full attention, not `O(N * W)`. `DEFAULT_MAX_POSITION_EMBEDDINGS` is 8192, so for `base` and `large` (`window_size=128`) **no admissible sequence length is ever windowed**. `tiny` (`window_size=64`, threshold 4096) is the only variant where windowing engages, and only for `4096 < L <= 8192`.
    -   The cost is not merely un-improved, it is *worse than global attention*. Windows are padded to `window_size**2` slots, so a local layer's score matrix is `window_size**2 x window_size**2` **independent of `L`**: `16384 x 16384 ~ 2.7e8` entries per head per sample at the default. That is ~16,384x dense attention at `L=128` and ~4x dense attention at `L=8192`. General form: `O(max(L, M) * M)` with `M = window_size**2`, i.e. an `O(M^2)` floor below `L = M` rather than a linear saving. Pinned by `tests/test_models/test_modern_bert/test_shipped_window_size.py` (the rest of the suite runs at `TEST_WINDOW_SIZE = 16`, where the same degeneracy holds but is invisible).
    Within-window order comes from the layer's learnable Swin-convention relative position bias over that synthetic grid, not from RoPE and not from 1-D token distance. No 1-D sliding-window attention layer exists in `layers/attention/`; adding one is the real fix.

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
from dl_techniques.models.modern_bert.model import ModernBERT

# 1. Create a model. Every example below that runs a FORWARD pass uses `"tiny"`.
#    This is not a stylistic choice: because the local layers pad to a single
#    `window_size**2` window (§ 4.3), `"base"`/`"large"` materialize a
#    16384 x 16384 score matrix PER HEAD -- ~12.9 GB in float32 at `hidden_size=768`,
#    which OOMs a 12 GB GPU at any sequence length. Measured 2026-08-18, peak host
#    RSS for CONSTRUCTION alone: tiny 1.05 GB, base 20.1 GB, large 24.2 GB.
model = ModernBERT.from_variant("tiny")

# 2. Compile the model (optional for inference)
model.compile(optimizer="adam")
print("✅ ModernBERT model created and compiled successfully!")
model.summary()

# 3. Create dummy data (batch size 2, sequence length 256)
dummy_inputs = {
    "input_ids": np.random.randint(0, 50368, (2, 256)),
    "attention_mask": np.ones((2, 256), dtype="int32"),
    "token_type_ids": np.zeros((2, 256), dtype="int32"),
}  # last_hidden_state is (2, 256, 256) for "tiny" (hidden_size=256)

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
| **`ModernBERT`**               | `models.modern_bert.model.ModernBERT`              | The main Keras `Model` for the foundation encoder.                               |
| **`create_modern_bert_with_head`** | `models.modern_bert.model.create_modern_bert_with_head` | Recommended factory function to combine a `ModernBERT` with a task-specific head. |

### 6.2 Core Building Blocks

| Layer                  | Location                                             | Purpose                                                                             |
| :--------------------- | :--------------------------------------------------- | :---------------------------------------------------------------------------------- |
| **`ModernBertEmbeddings`** | `layers.embedding.modern_bert_embeddings.ModernBertEmbeddings` | Handles the initial embedding lookup and normalization.                             |
| **`TransformerLayer`** | `layers.transformers.TransformerLayer`            | The highly configurable, modern Transformer block that powers the encoder.          |
| **`create_nlp_head`**  | `layers.heads.nlp.factory.create_nlp_head`        | A factory for creating various downstream task heads (e.g., classification, NER). |

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
from dl_techniques.models.modern_bert.model import (
    ModernBERT,
    create_modern_bert_with_head,
)
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

# 1. Define the classification task
classification_task = NLPTaskConfig(
    name="sentiment_classification",
    task_type=NLPTaskType.TEXT_CLASSIFICATION,
    num_classes=3
)

# 2. Create the complete model
classifier_model = create_modern_bert_with_head(
    bert_variant="tiny",
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

**8192 is now a hard ceiling, not a soft one.** Since the global layers were moved
onto RoPE (2026-08-17), `RotaryPositionEmbedding.call` **raises** above
`max_seq_len`. Before that, a longer input ran -- position-blind, but it ran. Two
`(8192, head_dim/2)` non-trainable tables are also materialized per global layer and
written into every checkpoint.

```python
import numpy as np
from dl_techniques.models.modern_bert.model import ModernBERT

# 1. Create a foundation model ("tiny": the only variant that can forward on a
#    consumer GPU -- see § 4.3 and Example 1's note)
long_context_bert = ModernBERT.from_variant("tiny")

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

### Pattern 1: Fine-tuning From a Local Checkpoint

**Nothing is downloaded.** `pretrained=True` raises `NotImplementedError`: no
ModernBERT checkpoint is distributed with `dl_techniques`, and there is no URL for
`from_variant` to fetch. `weights_dataset` names a checkpoint that would be fetched, so
it is inert. The supported route is a local `.keras` encoder file you produced yourself,
passed as `pretrained="<path>"`.

Two things the mechanism requires, both verified by running this example on 2026-08-18:

1.  **Call the encoder once before you save it.** `pretrained=` transfers weights
    *layer by layer* out of the saved model. `ModernBERT` is a subclassed model whose
    sublayers are built lazily, so saving an un-called encoder writes a file whose
    layers hold **zero** weights; the transfer then finds no overlap and raises
    *"No overlapping layers between target and source checkpoint"*.
2.  **Save the bare encoder, not a model that already carries a head** — the transfer
    matches on layer name against the `ModernBERT` you are restoring into.

```python
import keras
from dl_techniques.models.modern_bert.model import (
    ModernBERT,
    create_modern_bert_with_head,
)
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

# 0. An encoder you pre-trained and saved earlier.
encoder = ModernBERT.from_variant("tiny")
encoder(
    {"input_ids": keras.random.randint((1, 128), 0, encoder.vocab_size, dtype="int32")},
    training=False,
)  # REQUIRED before save -- see note 1 above
encoder.save("/tmp/modern_bert_tiny.keras")

ner_task = NLPTaskConfig(
    name="ner",
    task_type=NLPTaskType.TOKEN_CLASSIFICATION,
    num_classes=9,
)

# 1. Attach a fresh task head on top of the restored encoder.
ner_model = create_modern_bert_with_head(
    bert_variant="tiny",
    task_config=ner_task,
    pretrained="/tmp/modern_bert_tiny.keras",
)

# 2. Fine-tune on your own NER dataset, at a low learning rate.
ner_model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=2e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
```

Measured for the snippet above: the restored encoder reproduces the saved encoder's
`last_hidden_state` to **exactly 0.0**, against **5.50** for a freshly initialized
control -- so the transfer is real, not a silent no-op.

The example uses `"tiny"` deliberately -- see the note in § 8 Example 1. `"base"` and
`"large"` cost 20.1 GB and 24.2 GB of host RSS to *construct* (measured 2026-08-18) and
OOM a 12 GB GPU on a forward pass at any sequence length. That is the known windowing
degeneracy of § 4.3, not a property of this example.

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
from dl_techniques.models.modern_bert.model import ModernBERT, create_modern_bert_with_head

def test_creation_all_variants():
    """Test model creation for all variants.

    Needs ~45 GB of host RAM if the three models are alive at once
    (tiny 1.05 GB, base 20.1 GB, large 24.2 GB peak RSS -- § 8 Example 1).
    """
    for variant in ModernBERT.MODEL_VARIANTS.keys():
        model = ModernBERT.from_variant(variant)
        assert model is not None
        print(f"✓ ModernBERT-{variant} created successfully")

def test_forward_pass_shape():
    """Test the output shape of a forward pass."""
    # "tiny" IS a standard variant (MODEL_VARIANTS is tiny/base/large), and it is
    # the one to forward with -- see Example 1's note on "base"'s memory cost.
    model = ModernBERT.from_variant("tiny")
    # `attention_mask` is REQUIRED under `predict()`. `call` echoes the mask back in
    # its output dict, so omitting it makes that entry `None` and Keras' batch
    # concatenation raises "Structures don't have the same nested structure".
    # `model(inputs)` directly does accept `input_ids` alone.
    dummy_input = {
        "input_ids": np.random.randint(0, 50368, size=(4, 64)),
        "attention_mask": np.ones((4, 64), dtype="int32"),
    }
    output = model.predict(dummy_input)
    assert output["last_hidden_state"].shape == (4, 64, 256) # hidden_size=256 for "tiny"
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

A: The five key upgrades are: **1) Rotary Positional Embeddings (RoPE)** for long context; **2) Pre-Layer Normalization** for stability; **3) GeGLU activation** for better performance; **4) Alternating windowed/global attention** (for efficiency in the paper — see the next question for what this implementation actually does); and **5) Bias-free layers**.

**Q: Why use alternating attention instead of another efficient attention mechanism?**

A: In the paper, alternating attention is a simple and effective strategy: computationally cheap (dominated by the fast local attention) while still allowing full sequence-level information flow through the periodic global layers.

**This implementation does not deliver that efficiency, and it is worth being blunt about the direction.** The local layers are built from the `window` attention layer, which folds the sequence into a synthetic `ceil(sqrt(L))`-square grid and pads every window to `window_size**2` token slots. Cost is therefore `O(max(L, M) * M)` with `M = window_size**2`, which has a constant `O(M^2)` floor rather than a linear-in-`L` saving. At the `window_size=128` that `base` and `large` ship, `M = 16384` exceeds the 8192 max position, so a local layer computes a `16384 x 16384` score matrix *whatever `L` is* — about 16,384x dense attention at `L=128`, and still ~4x dense attention at `L=8192`. For those two variants the schedule collapses to an all-global stack that is slower than a plain all-global stack would be. `tiny` (`window_size=64`, `M = 4096`) is the only variant where the schedule does not collapse, and only above 4096 tokens, where it partitions into four windows — so "windowing always degenerates here" is as wrong as the paper's linear-cost claim. Modelling behaviour is what § 4.3 describes; only the speed claim is wrong. `tests/test_models/test_modern_bert/test_shipped_window_size.py` pins this at the shipped size. Fixing it means writing a genuine 1-D sliding-window attention layer — none exists in `layers/attention/` — which is an open decision, not a pending patch.

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
# DistilBERT: A Distilled Version of BERT

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A production-ready, fully-featured implementation of the **DistilBERT** architecture in **Keras 3**. This implementation is based on the paper *"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"* by Sanh et al.

DistilBERT retains approximately **97% of BERT's performance** while being **40% smaller and 60% faster**. It achieves this by distilling knowledge from a large "teacher" BERT model into a smaller "student" model during pre-training.

---

## Table of Contents

1. [Overview: Smaller, Faster, Cheaper](#1-overview-smaller-faster-cheaper)
2. [The Problem DistilBERT Solves](#2-the-problem-distilbert-solves)
3. [How DistilBERT Works](#3-how-distilbert-works)
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

## 1. Overview: Smaller, Faster, Cheaper

### What is DistilBERT?

**DistilBERT** is a Transformer model trained via **Knowledge Distillation**. It serves as a general-purpose language representation model, just like BERT, but with a significantly reduced architectural footprint.

This implementation provides the core DistilBERT encoder as a **foundation model**. It takes tokenized text as input and outputs a sequence of contextualized vectors. It is designed to be a drop-in replacement for BERT in environments where latency and memory footprint are critical constraints.

### Key Innovations of this Implementation

1.  **Foundation Model Design**: The `DistilBERT` class is a pure encoder, decoupled from task-specific heads.
2.  **Pretrained Weights Support**: Supports loading standard variants (`base`, `small`, `tiny`) and handles downloading official weights.
3.  **Keras 3 Native**: Built as a composite `keras.Model`, fully serializable and compatible with TensorFlow, PyTorch, and JAX backends.
4.  **Optimized Architecture**: Faithfully implements DistilBERT's simplified embedding layer (no segment embeddings) and reduced layer count.

### Why DistilBERT Matters

While BERT achieved state-of-the-art results, its sheer size (110M+ parameters) makes it difficult to deploy in resource-constrained environments like mobile phones or real-time applications.

**Comparison with BERT-Base**:
*   **Parameters**: ~66M (vs. 110M)
*   **Inference Speed**: ~60% faster
*   **Performance**: ~97% of BERT's GLUE score

---

## 2. The Problem DistilBERT Solves

### The Challenge of Efficiency

Large Language Models (LLMs) suffer from high computational costs.

```
┌─────────────────────────────────────────────────────────────┐
│  The Efficiency Bottleneck                                  │
│                                                             │
│  1. High Latency: Processing a single sentence in real-time │
│     using BERT-Large can take hundreds of milliseconds.     │
│                                                             │
│  2. Memory Footprint: Storing gradients and states for      │
│     huge models requires expensive high-VRAM GPUs.          │
│                                                             │
│  3. Energy Cost: Training and serving massive models has a  │
│     significant environmental impact.                       │
└─────────────────────────────────────────────────────────────┘
```

### The Solution: Knowledge Distillation

DistilBERT solves this by applying **Knowledge Distillation**, a compression technique in which a compact model (the student) is trained to reproduce the behavior of a larger model (the teacher), or an ensemble of models.

```
┌─────────────────────────────────────────────────────────────┐
│  The Distillation Process                                   │
│                                                             │
│  Teacher (BERT-Base) ──► Soft Target Probabilities          │
│           │                                                 │
│           ▼ (Loss)                                          │
│                                                             │
│  Student (DistilBERT) ──► Predicted Probabilities           │
│                                                             │
│  Result: The student learns to generalize better than if    │
│  it were trained on raw data alone, allowing it to be       │
│  much smaller.                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. How DistilBERT Works

### The High-Level Architecture

The architecture is very similar to BERT but streamlined. Notably, **Token Type Embeddings** (segment IDs) and the **Pooler** layer are removed.

```
┌──────────────────────────────────────────────────────────────────┐
│                  DistilBERT Foundation Model Architecture        │
│                                                                  │
│ Input (Token IDs) ───►┌────────────────┐                         │
│                       │DistilBertEmbeds│ (Token + Position)      │
│                       └────────┬───────┘                         │
│                                │                                 │
│                       ┌────────▼───────────┐                     │
│                       │ TransformerLayer   │ (Repeated N times)  │
│                       │(Self-Attention,FFN)│                     │
│                       └────────┬───────────┘                     │
│                                │                                 │
│                       ┌────────▼───────────┐                     │
│                       │Output Hidden States│ (Contextual Embeds) │
│                       └────────┬───────────┘                     │
│                                │                                 │
│                       ┌────────▼───────────┐                     │
│                       │ Task-Specific Head │ (e.g., Classifier)  │
│                       └────────────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
```

### The Simplified Data Flow

DistilBERT simplifies the input requirements compared to BERT.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   DistilBERT Complete Data Flow                         │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: INPUT PREPARATION (Simplified)
──────────────────────────────────────
Input Text -> Tokenizer -> Input Representation
    │
    ├─► Input IDs: Numerical IDs for each token.
    ├─► Attention Mask: Binary mask for padding.
    │   (Note: No Token Type/Segment IDs required)
    │
    └─► DistilBertEmbeddings Layer
        ├─► Word Embeddings
        ├─► Position Embeddings (Learned or Sinusoidal)
        │
        └─► Summed Embeddings -> LayerNorm -> Dropout -> (B, seq_len, D)


STEP 2: ENCODING (Reduced Depth)
────────────────────────────────
Embedded Sequence (B, seq_len, D)
    │
    ├─► TransformerLayer 1 ... N (Typically 6 layers vs BERT's 12)
    │
    └─► Final Hidden States (B, seq_len, D)


STEP 3: PROJECTION
──────────────────
Final Hidden States (B, seq_len, D)
    │
    └─► Task Head (Dense -> Softmax, etc.)
```

---

## 4. Architecture Deep Dive

### 4.1 `DistilBertEmbeddings` Layer

Unlike BERT, DistilBERT does not use "segment embeddings" (token type IDs). The model treats the input as a continuous sequence.
-   **Inputs**: `input_ids` and `position_ids` (optional).
-   **Structure**: Sum of Word Embeddings + Position Embeddings.
-   **Position Embeddings**: Supports both standard learned embeddings and fixed **sinusoidal** embeddings (configurable via `sinusoidal_pos_embds=True`).

### 4.2 `TransformerLayer`

The core block is identical to BERT's transformer layer, but DistilBERT typically uses half the number of layers (e.g., 6 layers for `base` instead of 12).
-   **Initialization**: In the original paper, DistilBERT was initialized by taking one out of every two layers from the teacher BERT model.

### 4.3 Task-Specific Heads

Because the output dimension matches BERT (`hidden_size=768` for base), DistilBERT is compatible with the exact same task heads used for BERT.

---

## 5. Quick Start Guide

### Installation

```bash
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First DistilBERT Model

Let's build a sentiment analysis model using the lightweight DistilBERT.

```python
import keras
import numpy as np

from dl_techniques.models.distilbert import create_distilbert_with_head
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType

# 1. Define the downstream task
sentiment_config = NLPTaskConfig(
    name="sentiment_analysis",
    task_type=NLPTaskType.SENTIMENT_ANALYSIS,
    num_classes=3
)

# 2. Create a DistilBERT model with a sentiment head
print("🚀 Creating DistilBERT-base model...")
model = create_distilbert_with_head(
    distilbert_variant="base",
    task_config=sentiment_config,
    pretrained=False  # Set to True to download weights
)

# 3. Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# 4. Dummy usage (Note: No token_type_ids needed)
BATCH_SIZE, SEQ_LEN = 4, 128
dummy_inputs = {
    "input_ids": np.random.randint(0, 30522, size=(BATCH_SIZE, SEQ_LEN)),
    "attention_mask": np.ones((BATCH_SIZE, SEQ_LEN), dtype="int32")
}
# Output shape: (4, 3)
print(model.predict(dummy_inputs).shape)
```

---

## 6. Component Reference

### 6.1 `DistilBERT` (Model Class)

**Purpose**: The main Keras `Model` subclass implementing the encoder.

```python
from dl_techniques.models.distilbert import DistilBERT

# Standard Base model
model = DistilBERT.from_variant("base", pretrained=True)

# Tiny model for edge devices
model = DistilBERT.from_variant("tiny")
```

### 6.2 Factory Functions

#### `create_distilbert_with_head(...)`
The high-level factory for end-to-end tasks. It manages the inputs (excluding token types) and connects the encoder to the head.

---

## 7. Configuration & Model Variants

DistilBERT variants are generally defined by their reduced depth compared to BERT equivalents.

| Variant | Hidden Size | Layers | Heads | Parameters | Use Case |
|:---:|:---:|:---:|:---:|:---:|:---|
| **`tiny`** | 256 | 2 | 4 | ~10M | Ultra-lightweight, mobile/IoT |
| **`small`**| 512 | 4 | 8 | ~29M | Fast CPU inference |
| **`base`** | 768 | 6 | 12 | ~66M | General purpose, good balance |

*Note: DistilBERT does not typically have a "Large" variant, as the goal is reduction.*

---

## 8. Comprehensive Usage Examples

### Example 1: Loading Pretrained Weights

```python
from dl_techniques.models.distilbert import DistilBERT

# 1. Download and load default 'uncased' weights
model = DistilBERT.from_variant("base", pretrained=True)

# 2. Load from local file
model = DistilBERT.from_variant("base", pretrained="./distilbert_weights.keras")

# 3. Custom vocab (e.g., for multilingual)
# This will load encoder weights but skip embedding weights due to shape mismatch
model = DistilBERT.from_variant("base", pretrained=True, vocab_size=50000)
```

### Example 2: NER (Token Classification)

```python
from dl_techniques.models.distilbert import create_distilbert_with_head
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType

ner_config = NLPTaskConfig(
    name="ner",
    task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
    num_classes=9
)

ner_model = create_distilbert_with_head(
    distilbert_variant="base",
    task_config=ner_config,
    pretrained=True
)
# Output shape: (batch, seq_len, 9)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: DistilBERT as a Feature Extractor

Use the model to get embeddings for downstream systems.

```python
import keras
from dl_techniques.models.distilbert import DistilBERT

encoder = DistilBERT.from_variant("base", pretrained=True)

inputs = {
    "input_ids": keras.Input(shape=(None,), dtype="int32"),
    "attention_mask": keras.Input(shape=(None,), dtype="int32")
}
outputs = encoder(inputs)
# Shape: (batch, seq_len, 768)
features = outputs["last_hidden_state"]

# Add custom LSTM
x = keras.layers.LSTM(128)(features)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
```

---

## 10. Performance Optimization

### Inference Speed

DistilBERT is naturally ~60% faster than BERT. For further gains:

1.  **XLA Compilation**: Use `jit_compile=True` in `model.compile()`.
2.  **Mixed Precision**:
    ```python
    keras.mixed_precision.set_global_policy('mixed_float16')
    model = DistilBERT.from_variant("base")
    ```

---

## 11. Training and Best Practices

### Fine-Tuning Strategy

Since DistilBERT is shallower, it can be more sensitive to aggressive learning rates.
*   **Learning Rate**: Use slightly higher rates than BERT (e.g., 5e-5 or 1e-4) if training from scratch, but stick to 2e-5–5e-5 for fine-tuning.
*   **Epochs**: Often converges faster than BERT.

### Input Representation

Ensure you use a tokenizer compatible with the original BERT (WordPiece). Note that `token_type_ids` are **not** passed to the model. Passing them will not cause an error (the model accepts kwargs), but they will be ignored by the logic.

---

## 12. Serialization & Deployment

Fully serializable via Keras 3.

```python
model = create_distilbert_with_head(...)
model.save('distilbert_sentiment.keras')

# Reload without custom object dictionaries
loaded = keras.models.load_model('distilbert_sentiment.keras')
```

---

## 13. Testing & Validation

```python
import numpy as np
from dl_techniques.models.distilbert import DistilBERT

def test_forward_pass():
    model = DistilBERT.from_variant("small")
    inputs = {
        "input_ids": np.random.randint(0, 1000, (2, 10)),
        "attention_mask": np.ones((2, 10), dtype="int32")
    }
    out = model(inputs)
    assert out["last_hidden_state"].shape == (2, 10, 512)
    print("✓ DistilBERT shape check passed")
```

---

## 14. Troubleshooting & FAQs

**Q: Where are the `token_type_ids`?**
A: DistilBERT removed them to simplify the architecture. It does not distinguish between "Sentence A" and "Sentence B" explicitly via embeddings, though it can still process pairs separated by `[SEP]`.

**Q: Can I use BERT weights?**
A: Not directly. While the architectures are similar, the layer count is different (6 vs 12), and the weights matrices are not 1:1 mappable without the specific distillation selection process. Use `DistilBERT.from_variant("base", pretrained=True)` to get the correct weights.

**Q: Why is there no Pooler output?**
A: The original DistilBERT removed the pre-training "Next Sentence Prediction" task, and thus removed the dense pooler layer associated with the `[CLS]` token. You should simply take the 0-th index of `last_hidden_state` for classification tasks.

---

## 15. Technical Details

### Differences from BERT

1.  **Layers**: Reduced from 12 to 6 (in Base).
2.  **Token Type Embeddings**: Removed.
3.  **Pooler**: Removed.
4.  **Training Objective**: Trained with a triple loss:
    *   $L_{ce}$: Masked Language Modeling loss (Student).
    *   $L_{distill}$: Distillation loss (Cosine embedding loss between student and teacher hidden states).
    *   $L_{cos}$: Cosine distance loss.

---

## 16. Citation

```bibtex
@article{sanh2019distilbert,
  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},
  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},
  journal={arXiv preprint arXiv:1910.01108},
  year={2019}
}
```
# DistilBERT: A Distilled Version of BERT

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

An implementation of the **DistilBERT** *architecture* in **Keras 3**, based on the paper *"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"* by Sanh et al.

> **No trained weights ship with this package, none can be downloaded, and the `pretrained=` argument does not work in either of its two forms.** Every URL in `DistilBERT.PRETRAINED_WEIGHTS` is an `example.com` placeholder, so `pretrained=True` returns a **randomly initialized** model after logging a warning; and `pretrained="<path>.keras"` **raises `ValueError`** — `load_pretrained_weights` is broken independently of the URLs. Both behaviours are measured in [§8](#8-comprehensive-usage-examples), which also gives the one loading route that does work (`keras.models.load_model`). Everything below describes an architecture you must train yourself.

The published DistilBERT checkpoint retains approximately **97% of BERT's performance** while being **40% smaller and 60% faster**, by distilling knowledge from a large "teacher" BERT model into a smaller "student" model during pre-training. Those are the paper's numbers for the paper's checkpoint; nothing in this repo reproduces or measures them.

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
2.  **Weight loading**: use `keras.models.load_model(path)` on a file you saved (verified to restore every weight exactly). The package's own `pretrained=` / `load_pretrained_weights` machinery is present but non-functional — §8.
3.  **Keras 3 Native**: Built as a composite `keras.Model` and fully serializable — a `.keras` save/load round trip with no `custom_objects` reproduces the forward output exactly (measured max abs diff `0.0`). Only the TensorFlow backend is exercised here; other backends are untested.
4.  **Shared embedding stage**: no DistilBERT-private embedding class exists. The model builds `dl_techniques.layers.embedding.bert_embeddings.BertEmbeddings` — the same layer `models/bert/` and `models/fnet/` use — via `create_embedding_layer('bert_embeddings', ...)` with `use_token_type_embeddings=False` (no segment embeddings) and `mask_zero=False`.

### Why DistilBERT Matters

While BERT achieved state-of-the-art results, its sheer size (110M+ parameters) makes it difficult to deploy in resource-constrained environments like mobile phones or real-time applications.

**Comparison with BERT-Base** (Sanh et al.'s reported figures for their trained checkpoint, not measurements of this code — for this implementation's own parameter counts see [§7](#7-configuration--model-variants)):
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
│                       │ BertEmbeddings │ (Token + Position;      │
│                       │    (shared)    │  token types disabled)  │
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
    ├─► Attention Mask: Binary mask for padding. NOT optional in practice --
    │   nothing infers it, see section 11.
    │   (Note: No Token Type/Segment IDs required)
    │
    └─► BertEmbeddings (shared layer, token types disabled)
        ├─► Word Embeddings
        ├─► Position Embeddings (Learned or Sinusoidal)
        │
        └─► Summed Embeddings -> Norm -> Dropout -> (B, seq_len, D)


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

### 4.1 Embedding Stage — the shared `BertEmbeddings`

There is **no DistilBERT-private embedding class**. `_build_architecture` calls

```python
create_embedding_layer(
    'bert_embeddings',
    ...,
    type_vocab_size=None,
    use_token_type_embeddings=False,
    position_embedding_type='sinusoidal' if sinusoidal_pos_embds else 'learned',
    mask_zero=False,
    layer_norm_eps=layer_norm_eps,
    normalization_type=normalization_type,
)
```

so the layer is `dl_techniques.layers.embedding.bert_embeddings.BertEmbeddings`, the same one `models/bert/` and `models/fnet/` build. Three kwargs carry DistilBERT's whole delta from BERT, and each differs from that layer's default — do not drop them (see the `D-011` comment at the call site):

-   **`use_token_type_embeddings=False`**: no "segment embeddings" (token type IDs). Unlike BERT, the input is treated as one continuous sequence, and no `token_type_embeddings` weight is allocated. `type_vocab_size=None` follows from it.
-   **`mask_zero=False`**: the layer does **not** emit a Keras auto-mask. DistilBERT threads an explicit `attention_mask` into every `TransformerLayer` instead; two masking mechanisms reaching the same attention stack is the failure this pins shut.
-   **`position_embedding_type`**: learned embeddings by default, fixed **sinusoidal** with `sinusoidal_pos_embds=True`.

Consequence worth knowing: `normalization_type` is validated by `BertEmbeddings`, which accepts exactly `layer_norm`, `rms_norm`, `band_rms`, `batch_norm`. Any other value raises `ValueError` at construction from the embedding stage, even for values a `TransformerLayer` alone would accept.

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
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

# 1. Define the downstream task
sentiment_config = NLPTaskConfig(
    name="sentiment_analysis",
    task_type=NLPTaskType.SENTIMENT_ANALYSIS,
    num_classes=3
)

# 2. Create a DistilBERT model with a sentiment head (random weights)
model = create_distilbert_with_head(
    distilbert_variant="base",
    task_config=sentiment_config,
    pretrained=False  # True does NOT fetch weights -- see section 8
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
# A classification head returns a DICT, not one tensor:
# {'logits': (4, 3), 'probabilities': (4, 3)}
outputs = model.predict(dummy_inputs, verbose=0)
print({k: v.shape for k, v in outputs.items()})
```

---

## 6. Component Reference

### 6.1 `DistilBERT` (Model Class)

**Purpose**: The main Keras `Model` subclass implementing the encoder.

```python
from dl_techniques.models.distilbert import DistilBERT

# Standard Base model (randomly initialized)
model = DistilBERT.from_variant("base")

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
| **`tiny`** | 256 | 2 | 4 | 9,524,736 | Ultra-lightweight, mobile/IoT |
| **`small`**| 512 | 4 | 8 | 28,499,968 | Fast CPU inference |
| **`base`** | 768 | 6 | 12 | 66,362,880 | General purpose, good balance |

Parameter counts are measured with `count_params()` on a built model at the default `vocab_size=30522` and `max_position_embeddings=512`; they move with either of those.

*Note: DistilBERT does not typically have a "Large" variant, as the goal is reduction.*

### Constrained configuration values

-   `normalization_type` — one of `layer_norm`, `rms_norm`, `band_rms`, `batch_norm`. All four are verified to build and forward-pass; anything else raises `ValueError` at construction (§4.1).
-   `sinusoidal_pos_embds` — `False` (learned) or `True` (fixed sinusoidal). Both are verified under `float32` and `mixed_float16` (§10).
-   `pad_token_id` — stored and serialized, never read. See §11.

---

## 8. Comprehensive Usage Examples

### Example 1: What `pretrained=` actually does

**`pretrained=True` does not download weights.** Every entry of `DistilBERT.PRETRAINED_WEIGHTS` is an `https://example.com/...` placeholder. Measured behaviour of `DistilBERT.from_variant("tiny", pretrained=True)`:

1.  `keras.utils.get_file` fails on the placeholder URL.
2.  `from_variant` catches the exception and emits exactly one warning — *"Failed to download pretrained weights: ... Continuing with random initialization."*
3.  `load_pretrained_weights` is invoked **0 times**, and every non-constant weight of the returned model differs from an independently constructed one.
4.  A `DistilBERT` is returned, **randomly initialized**, and nothing raises.

A caller who does not read logs cannot tell this apart from a successful load. The machinery is kept deliberately (it is the wiring a real checkpoint would use); only the URLs are missing.

**`pretrained="<path>.keras"` does not work either — it raises.** `from_variant` forwards it to `load_pretrained_weights`, which fails on both of its two paths (measured with a file written by `model.save(...)`):

| Route | Result |
|---|---|
| `from_variant(pretrained="<file>.keras")` on a fresh model | `ValueError: Failed to load weights ...: keras.random.uniform requires a floating point dtype. Received: dtype=int32` — the "build the model first" dummy input is itself invalid |
| `model(...)` first, then `load_pretrained_weights("<file>.keras")` | `ValueError: Failed to load weights ...: Invalid keyword arguments: {'by_name': True}` — Keras 3's `Model.load_weights` has no `by_name` |
| same, with a `.weights.h5` file | same `by_name` `ValueError` |
| **`keras.models.load_model("<file>.keras")`** | **works — all 28/28 weights restored identically** |

So the loading story for now is: save with `model.save(path)`, load with `keras.models.load_model(path)`.

```python
import keras
from dl_techniques.models.distilbert import DistilBERT

# Randomly initialized -- the working construction route
model = DistilBERT.from_variant("base")

# Save / restore a model you trained yourself
model.save("./distilbert_weights.keras")
model = keras.models.load_model("./distilbert_weights.keras")

# DO NOT USE (both raise / silently no-op, see the table above):
#   DistilBERT.from_variant("base", pretrained=True)
#   DistilBERT.from_variant("base", pretrained="./distilbert_weights.keras")
```

### Example 2: NER (Token Classification)

```python
from dl_techniques.models.distilbert import create_distilbert_with_head
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

ner_config = NLPTaskConfig(
    name="ner",
    task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
    num_classes=9
)

ner_model = create_distilbert_with_head(
    distilbert_variant="base",
    task_config=ner_config,
)
# Output (measured): {'logits': (batch, seq_len, 9), 'predictions': (batch, seq_len)}
# -- a token-classification head returns different keys from the
# sentiment head in section 5. Always inspect the dict.
```

---

## 9. Advanced Usage Patterns

### Pattern 1: DistilBERT as a Feature Extractor

Use the model to get embeddings for downstream systems.

```python
import keras
from dl_techniques.models.distilbert import DistilBERT

encoder = DistilBERT.from_variant("base")

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

The published DistilBERT is reported at ~60% faster than BERT-base at 6 layers instead of 12; this repo measures no latency. Two knobs are verified to *work* (not to be faster — no timing was taken):

1.  **XLA Compilation**: `model.compile(..., jit_compile=True)` — verified end to end through `create_distilbert_with_head` + `predict`.
2.  **Mixed Precision**:
    ```python
    keras.mixed_precision.set_global_policy('mixed_float16')
    model = DistilBERT.from_variant("base")
    ```
    Verified for **both** position-embedding modes: `sinusoidal_pos_embds=False` and `sinusoidal_pos_embds=True` each forward-pass to a finite `float16` output. (The sinusoidal branch used to raise `InvalidArgumentError` under this policy — it built its sin/cos table in hard `float32` and added it to a `float16` word embedding. The shared `BertEmbeddings` now casts the table to the dtype of the tensor it is summed with.)

---

## 11. Training and Best Practices

### Fine-Tuning Strategy

Since DistilBERT is shallower, it can be more sensitive to aggressive learning rates.
*   **Learning Rate**: Use slightly higher rates than BERT (e.g., 5e-5 or 1e-4) if training from scratch, but stick to 2e-5–5e-5 for fine-tuning.
*   **Epochs**: Often converges faster than BERT.

### Input Representation

Ensure you use a tokenizer compatible with the original BERT (WordPiece). `token_type_ids` are **not** used. A `"token_type_ids"` key inside the input dict is silently ignored (measured: the forward pass succeeds and returns the usual `last_hidden_state` / `attention_mask`), but passing `token_type_ids=` as a **keyword argument** raises `TypeError: ... got an unexpected keyword argument 'token_type_ids'`, because `call()` does not declare it.

### Padding is your responsibility (`pad_token_id` is advisory)

`pad_token_id` is stored on the model and written into `get_config()`, and **that is all it does**. Nothing reads it: no attention mask is derived from it anywhere in the package.

*   If you call the model **without** an `attention_mask`, padding tokens are attended to exactly like real tokens. Measured on a batch whose second half is `pad_token_id`: the masked and unmasked forward passes differ (max abs diff `3.6e-2` at a small config) — proof that the mask is doing work and that none is inferred for you.
*   Always pass `attention_mask` (`1` = keep, `0` = pad), either as a dict key or as the `attention_mask=` argument.
*   This matches upstream HuggingFace DistilBERT, which also defaults the mask to all-ones rather than deriving it. Deriving one here was considered and rejected: it would silently change the output of every mask-less forward pass written against this model so far.

---

## 12. Serialization & Deployment

Fully serializable via Keras 3. Verified: a save/load round trip with **no** `custom_objects` reproduces the `training=False` forward output exactly (max abs diff `0.0`).

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
    out = model(inputs, training=False)
    assert tuple(out["last_hidden_state"].shape) == (2, 10, 512)
```

The package's own suite lives in `tests/test_models/test_distilbert/`.

---

## 14. Troubleshooting & FAQs

**Q: Where are the `token_type_ids`?**
A: DistilBERT removed them to simplify the architecture. It does not distinguish between "Sentence A" and "Sentence B" explicitly via embeddings, though it can still process pairs separated by `[SEP]`.

**Q: Can I use BERT weights?**
A: Not directly. While the architectures are similar, the layer count is different (6 vs 12), and the weight matrices are not 1:1 mappable without the specific distillation selection process.

**Q: Where do I get trained weights, then?**
A: You train them, or you convert them yourself. This package ships no checkpoint and contains no HuggingFace converter. `pretrained=True` reaches placeholder URLs and falls back to random init; `pretrained="<path>"` raises. Save with `model.save(path)` and reload with `keras.models.load_model(path)` (§8).

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
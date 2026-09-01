# ModernBERT: A BERT Successor

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of **ModernBERT**, based on "[Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](https://arxiv.org/abs/2412.13663)". ModernBERT folds a set of contemporary LLM techniques into the bidirectional encoder: rotary embeddings, pre-normalization, gated FFNs, and alternating local/global attention over an 8192-token context.

> **No trained weights ship with this package.** `pretrained=True` raises `NotImplementedError`; pass a local `.keras` path instead (§9). Everything below describes an architecture you train yourself.

---

## 1. Overview: What is ModernBERT and Why It Matters

**ModernBERT** is a bidirectional encoder rebuilt around techniques from recent decoder-only LLMs. The published models were trained on 2 trillion tokens including code, with a native context of **8192 tokens**.

### Key Innovations

1.  **Rotary Positional Embeddings (RoPE)** replace absolute position embeddings, working well at both short and long context and extending more easily.
2.  **Pre-Layer Normalization** applies the norm *before* attention and the FFN, which improves training stability.
3.  **GeGLU** in the feed-forward network, a gated GELU rather than a plain MLP.
4.  **Alternating local and global attention.** Most layers use a 1-D sliding band spanning `local_attention_window_size` tokens; every third layer uses global attention so long-range dependencies still propagate.
5.  **Bias-free layers** in most linear and normalization layers, saving parameters and helping stability.
6.  **A modern training recipe**: BPE tokenizer, trapezoidal learning rate schedule, StableAdamW.

> **One deviation to know about.** This implementation reproduces the paper's local-attention *adjacency* but not its *speedup*. The band is applied as a dense `N x N` mask over standard attention, so a local layer is `O(N^2)`, the same order as a global one, rather than the linear-in-`L` saving the paper reports. A fused banded kernel is not reachable from `keras.ops`. What the band buys here is the correct modelling behaviour, not a lower asymptotic. See §4.3.

ModernBERT suits long-document classification and retrieval, semantic search and RAG, code analysis, and standard NER/GLUE-style tasks where a BERT-class encoder is the backbone.

---

## 2. The Problem ModernBERT Solves

Decoder-only LLMs have moved quickly; the encoders that power retrieval and classification pipelines have not. Practitioners are largely still on BERT and RoBERTa, which means:

-   A context window capped at 512 tokens.
-   `O(N²)` attention in every layer, so long inputs are expensive.
-   Training data that is small, narrow and old (BookCorpus, Wikipedia).
-   Components since superseded: post-LN, plain GELU, absolute position embeddings.

ModernBERT rebuilds the encoder from the ground up against those four points: windowed attention with periodic global layers for cost, RoPE for position, pre-LN and GeGLU for stability and capacity, and a large modern corpus for the data.

---

## 3. How ModernBERT Works: Core Concepts

The bidirectional encoder structure is unchanged; the components inside it are not.

```
Input IDs ────► ┌──────────────────────┐
                │ ModernBertEmbeddings │  (no positional term)
                └──────────┬───────────┘
                ┌──────────▼───────────┐
                │ Transformer (Local)  │  (banded attention, Pre-LN)
                └──────────┬───────────┘
                ┌──────────▼───────────┐
                │ Transformer (Local)  │
                └──────────┬───────────┘
                ┌──────────▼───────────┐
                │ Transformer (Global) │  (global attention + RoPE)
                └──────────┬───────────┘
                           ▼ (repeat)
                ┌──────────────────────┐
                │  Final Layer Norm    │
                └──────────┬───────────┘
                           ▼
                  Output Hidden States
```

### The Pre-LN block

Pre-normalization is the critical change from classic BERT:

```
Input
  ├─► Residual Path 1 ──────────────────────────────┐
  └─► LayerNorm ──► Attention (Local/Global) ──► Add ◄┘
  │
  ├─► Residual Path 2 ──────────────────────────────┐
  └─► LayerNorm ──► GeGLU FFN ──► Dropout ──────► Add ◄┘
  │
  ▼
Output
```

The asymmetry is real, not a drawing shortcut. These blocks are `TransformerLayer(normalization_position='pre', ...)`, and `TransformerLayer` applies its `dropout_rate` layer to the **FFN sub-block only**; there is no dropout after attention. `attention_probs_dropout_rate` is not an output dropout either: it becomes the attention sub-layer's internal attention-weight dropout. Being pre-LN, neither branch ends in a normalization ("Add & Norm" is the post-LN shape); the only trailing normalization is the single final one after the whole stack.

---

## 4. Architecture Deep Dive

### 4.1 `ModernBertEmbeddings` (no positional term)

A token lookup over the 50,368-entry BPE vocabulary plus a token type embedding (present for BERT compatibility, unused in pre-training), summed, then `LayerNormalization` and `Dropout`.

This layer adds **no** positional term, and its output is permutation-equivariant. The positional signal is injected downstream, by the attention layers (§4.3).

### 4.2 `TransformerLayer` with Pre-LN and GeGLU

Each block runs: normalize, attend (banded or global), add residual; normalize, GeGLU FFN, add residual. RoPE is applied to the queries and keys of the **global** layers only.

### 4.3 Hybrid Attention

**Global attention** (every 3rd layer by default) is standard full attention with **RoPE** on queries and keys, so information crosses the entire sequence. It is built as the factory's `group_query` type with `num_kv_heads == num_heads`, which is arithmetically plain multi-head attention. That is the only registry entry reaching plain self-attention *and* carrying RoPE; `multi_head` declares no RoPE parameter, and `create_attention_layer` now raises `ValueError` on an undeclared key rather than silently dropping it.

**Local (banded) attention** (the other 2 of every 3 layers) uses `window_band`: a 1-D symmetric sliding band where query `i` attends key `j` iff `abs(i - j) <= local_attention_window_size // 2`. Non-causal, because this is an encoder. `local_attention_window_size` is the **full span** in tokens, matching upstream (`transformers/modular_modernbert.py`: `sliding_window = local_attention // 2`, with `local_attention=128` documented as "64 tokens either side"). There is no grid folding, no square padding and no relative position bias. Local layers carry no positional term of their own; position reaches them through the residual stream from the global layers' RoPE.

Pinned by `tests/test_models/test_modern_bert/test_positional_signal.py::TestLocalNeighbourhoodIsAContiguousOneDimensionalBand` (by perturbation: at `L=16, local_attention_window_size=2`, token 1 moves token 0 while tokens 2 and 4 move it by exactly 0.0) and by `test_the_shipped_variants_can_run.py::TestNoVariantShipsALocalBandItCannotUse`.

**The cost, honestly.** The band is a dense `N x N` mask over standard attention, so a local layer is `O(N^2)`, not the `O(N * W)` that "sliding window" usually implies. The band buys the correct adjacency, not a lower asymptotic.

### 4.4 The 8192 ceiling is hard

Because the global layers use RoPE, `RotaryPositionEmbedding.call` **raises** above `max_position_embeddings` rather than degrading. Two `(8192, head_dim/2)` non-trainable tables are materialized per global layer and written into every checkpoint.

---

## 5. Quick Start Guide

### Installation

```bash
pip install keras>=3.0 tensorflow>=2.16 numpy
```

### Your First ModernBERT Model

```python
import keras
import numpy as np

from dl_techniques.models.language.modern_bert.model import ModernBERT

# 1. Create a model. The examples here use "tiny" for speed; "base" and
#    "large" are also affordable to construct (under 2.5 GB peak host RSS).
model = ModernBERT.from_variant("tiny")

# 2. Compile (optional for inference)
model.compile(optimizer="adam")
model.summary()

# 3. Dummy data (batch size 2, sequence length 256)
dummy_inputs = {
    "input_ids": np.random.randint(0, 50368, (2, 256)),
    "attention_mask": np.ones((2, 256), dtype="int32"),
    "token_type_ids": np.zeros((2, 256), dtype="int32"),
}

# 4. Run inference
outputs = model(dummy_inputs)

# 5. Inspect. Measured: keys are ['attention_mask', 'last_hidden_state'],
#    and last_hidden_state is (2, 256, 256) -- "tiny" has hidden_size=256.
print(sorted(outputs.keys()))
print(outputs["last_hidden_state"].shape)
```

`attention_mask` is optional. When you omit it, `call` echoes an all-ones mask so the output structure does not depend on the input, but the mask does not reach the encoder in that case.

---

## 6. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`ModernBERT`** | `...models.language.modern_bert.model` | The foundation encoder. Outputs `{"last_hidden_state", "attention_mask"}`. |
| **`create_modern_bert_with_head`** | `...models.language.modern_bert.model` | Factory combining a `ModernBERT` with a task head. |
| **`ModernBertEmbeddings`** | `...layers.embedding.modern_bert_embeddings` | Token and token-type embedding, normalization, dropout. |
| **`TransformerLayer`** | `...layers.transformers` | The configurable block powering the encoder. |
| **`create_nlp_head`** | `...layers.heads.nlp.factory` | Factory for downstream task heads. |

---

## 7. Configuration & Model Variants

| Variant | Hidden Size | Layers | Heads | `intermediate_size` | Parameters | Weights | Global Interval | Window |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`tiny`** | 256 | 4 | 4 | 384 | 16,176,128 | 44 | 2 | 64 |
| **`base`** | 768 | 22 | 12 | 1152 | 152,720,384 | 208 | 3 | 128 |
| **`large`** | 1024 | 28 | 16 | 2624 | 399,560,704 | 264 | 3 | 128 |

Counts are measured with `count_params()` after `from_variant(v)` plus one forward pass at `L=8`. `intermediate_size` is the per-branch width; the GeGLU pair is twice that, which is why the paper quotes 2304 and 5248 for `base` and `large`.

**Both counts move with `global_attention_interval`**, because global and local layers do not carry the same tables. Never quote a ModernBERT parameter count without the interval it was measured at.

**Pretrained weights.** `pretrained=True` raises `NotImplementedError`; no checkpoint is distributed here and there is no URL to fetch. `weights_dataset` names a checkpoint that would be fetched, so it is inert. Pass `pretrained="<path>.keras"` for a local encoder file you produced yourself (§9).

---

## 8. Comprehensive Usage Examples

### Example 1: Text Classification

```python
import numpy as np
from dl_techniques.models.language.modern_bert.model import create_modern_bert_with_head
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

classification_task = NLPTaskConfig(
    name="sentiment_classification",
    task_type=NLPTaskType.TEXT_CLASSIFICATION,
    num_classes=3
)

classifier_model = create_modern_bert_with_head(
    bert_variant="tiny",
    task_config=classification_task
)

dummy_inputs = {
    "input_ids": np.random.randint(0, 50368, (4, 128)),
    "attention_mask": np.ones((4, 128), dtype="int32"),
    "token_type_ids": np.zeros((4, 128), dtype="int32"),
}
predictions = classifier_model.predict(dummy_inputs, verbose=0)

# Measured: a DICT, not one tensor -- {'logits': (4, 3), 'probabilities': (4, 3)}.
# `predictions.shape` would raise AttributeError. Select the key you want, and
# select one output before compiling on a single loss.
print({k: v.shape for k, v in predictions.items()})
```

### Example 2: Long-Context Feature Extraction

The native 8192 context suits long-document work. Above that the model raises rather than silently degrading (§4.4).

```python
import numpy as np
from dl_techniques.models.language.modern_bert.model import ModernBERT

long_context_bert = ModernBERT.from_variant("tiny")

long_inputs = {
    "input_ids": np.random.randint(0, 50368, (1, 4096)),
    "attention_mask": np.ones((1, 4096), dtype="int32"),
}

features = long_context_bert.predict(long_inputs, verbose=0)
# Measured: (1, 4096, 256) -- "tiny" has hidden_size=256.
print(features["last_hidden_state"].shape)
```

---

## 9. Advanced Usage Patterns

### Pattern 1: Fine-tuning From a Local Checkpoint

**Nothing is downloaded.** The supported route is a local `.keras` encoder file you produced yourself, passed as `pretrained="<path>"`.

Two things the mechanism requires:

1.  **Call the encoder once before you save it.** `pretrained=` transfers weights *layer by layer* out of the saved model. `ModernBERT` is a subclassed model whose sublayers are built lazily, so saving an un-called encoder writes a file whose layers hold **zero** weights; the transfer then finds no overlap and raises *"No overlapping layers between target and source checkpoint"*.
2.  **Save the bare encoder, not a model that already carries a head.** The transfer matches on layer name against the `ModernBERT` you are restoring into.

```python
import keras
from dl_techniques.models.language.modern_bert.model import (
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

Measured for this snippet: the restored encoder reproduces the saved encoder's `last_hidden_state` to **exactly 0.0**, against **5.50** for a freshly initialized control, so the transfer is real and not a silent no-op. `tests/test_models/test_readme_examples.py` executes this pattern verbatim, control arm included.

Note `keras.random.randint` rather than `keras.random.uniform`: Keras rejects an integer `dtype` on `uniform`, so the `uniform` spelling of a token batch always raises.

---

## 10. Performance Optimization

### Flash Attention and unpadding

The paper reports large gains from Flash Attention and from unpadding (packing sequences to avoid computing on padding). This implementation uses the standard `TransformerLayer` and does neither, though it can be extended toward a Flash-Attention-compatible backend.

### Mixed precision

```python
keras.mixed_precision.set_global_policy('mixed_float16')

model = ModernBERT.from_variant("base")   # picks up the global policy
model.compile(optimizer="adamw")
```

---

## 11. Training and Best Practices

-   **Optimizer**: the paper uses **StableAdamW**, an AdamW variant with update clipping. Plain `AdamW` is a solid fallback.
-   **Schedule**: the paper uses a modified **trapezoidal** (warmup-stable-decay) schedule, holding the rate constant for most of training, which helps continual training. For fine-tuning, linear decay with warmup is a strong baseline.
-   **Masking rate**: 30% for MLM, rather than BERT's 15%, which the paper finds more effective.
-   **No NSP**: the Next Sentence Prediction objective is dropped, following RoBERTa.
-   **Context extension**: the published models trained on short sequences first, then a further 300B tokens at length 8192.

---

## 12. Serialization & Deployment

Fully serializable in the `.keras` format; no `custom_objects` needed on load.

```python
model = create_modern_bert_with_head(...)
model.save('my_modern_bert_classifier.keras')

loaded_model = keras.models.load_model('my_modern_bert_classifier.keras')
```

---

## 13. Testing & Validation

```bash
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_modern_bert/ -q
```

A minimal shape check of your own:

```python
import numpy as np
from dl_techniques.models.language.modern_bert.model import ModernBERT

def test_forward_pass_shape():
    model = ModernBERT.from_variant("tiny")
    dummy_input = {
        "input_ids": np.random.randint(0, 50368, size=(4, 64)),
        "attention_mask": np.ones((4, 64), dtype="int32"),
    }
    output = model.predict(dummy_input, verbose=0)
    assert output["last_hidden_state"].shape == (4, 64, 256)  # hidden_size=256
```

All three variants (`tiny`, `base`, `large`) construct and forward comfortably; peak host RSS to build one and run a short forward pass is under 2.5 GB for `large`.

---

## 14. Troubleshooting & FAQs

-   **`NotImplementedError` from `pretrained=True`.** No checkpoint ships here. Save your own encoder and pass its path (§9).
-   **"No overlapping layers between target and source checkpoint".** You saved an encoder that had never been called, so its lazily-built sublayers held no weights. Call it once before saving (§9).
-   **`AttributeError: 'dict' object has no attribute 'shape'`.** `create_modern_bert_with_head` returns the head's dict. Select a key (§8).
-   **A raise above 8192 tokens.** The RoPE tables are sized to `max_position_embeddings`; the ceiling is hard (§4.4).
-   **`ValueError` naming an attention key.** `create_attention_layer` is strict, so an undeclared key such as `use_rope` on a `multi_head` layer fails at construction instead of being silently dropped (§4.3).
-   **Unstable training.** Unlikely with Pre-LN, but lower the learning rate and add warmup.

**Q: What is the main difference from classic BERT?** Five upgrades: RoPE for long context, Pre-LN for stability, GeGLU in the FFN, alternating local-band and global attention, and bias-free layers.

**Q: Why alternating attention rather than another efficient-attention scheme?** In the paper it is simple and effective: cost is dominated by the cheap local layers while the periodic global layers still allow full sequence-level information flow. As noted in §4.3, this implementation reproduces the adjacency but not the speedup.

**Q: Is it a drop-in replacement for `bert-base-uncased`?** In API and role, yes. It uses a different BPE tokenizer and different training data, so use the matching tokenizer and expect different behaviour.

---

## 15. Technical Details

### Rotary Positional Embeddings

Rather than adding a positional vector to the input, RoPE rotates the query and key vectors by an angle that is a function of absolute position. The resulting attention score depends only on the *relative* offset, which is what gives RoPE its length generalization.

### Hardware-aware design

The hidden size, FFN ratio and depth were chosen by hardware-aware ablation, aiming to maximize utilization on common inference GPUs (T4, A10, L4, RTX 4090) while staying as deep-and-narrow as possible.

### GeGLU

```
GeGLU(x) = GELU(x @ W_gate) * (x @ W_up)
```

The input is projected twice. One projection passes through GELU and acts as a gate, multiplying the second element-wise, letting the block modulate information flow per channel.

---

## 16. Citation

```bibtex
@article{warner2024smarter,
  title={Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference},
  author={Warner, Benjamin and Chaffin, Antoine and Clavié, Benjamin and Weller, Orion and Hallström, Oskar and Taghadouini, Said and Gallagher, Alexis and Biswas, Raja and Ladhak, Faisal and Aarsen, Tom and Cooper, Nathan and Adams, Griffin and Howard, Jeremy and Poli, Iacopo},
  journal={arXiv preprint arXiv:2412.13663},
  year={2024}
}
```

# Qwen: Decoder-Only Language Models, MoE, and Text Embedding

Keras 3 implementations of the **Qwen3** model family: a standard decoder-only transformer
with optional Mixture-of-Experts layers, the **Qwen3 Next** hybrid (Gated DeltaNet + Gated
Attention) MoE architecture, and the **Qwen3 Embedding / Reranker** retrieval models.

This file is the package index. Two of the three architectures already have full-length
documents next to their source; this README says what is in the package, which document
covers what, and what has no document of its own.

---

## 1. What this package contains

`src/dl_techniques/models/qwen/`:

| File | Contents |
| :--- | :--- |
| `qwen3.py` | `Qwen3` (`keras.Model`) plus `create_qwen3`, `create_qwen3_generation`, `create_qwen3_classification`. |
| `qwen3_next.py` | `Qwen3Next` (`keras.Model`) plus `create_qwen3_next`, `create_qwen3_next_generation`, `create_qwen3_next_classification`. |
| `components.py` | `Qwen3NextBlock` — the `3x Gated DeltaNet + 1x Gated Attention` block that `Qwen3Next` stacks. |
| `qwen3_embeddings.py` | `Qwen3EmbeddingLayer`, `Qwen3RerankerLayer`, `Qwen3EmbeddingModel`, `Qwen3RerankerModel`. |
| `QWEN3.md` | Full document for `Qwen3`: architecture diagram, variants, MoE configuration, training, serialization. |
| `QWEN3_next.md` | Full document for `Qwen3Next`, including a `Qwen3` vs `Qwen3 Next` comparison section. |
| `__init__.py` | **Empty.** There is no curated package API — import from the submodules directly. |

Tests live in `tests/test_models/test_qwen/`: `test_qwen3.py`, `test_qwen3_next.py`,
`test_components.py`, `test_qwen3_embeddings.py`.

---

## 2. The three architectures

### `Qwen3` — dense transformer with optional MoE layers

A decoder-only stack built from the repository's shared `TransformerLayer`: Grouped Query
Attention with RoPE, RMS normalization, and SwiGLU feed-forward networks. Any subset of
layers can be swapped to Mixture-of-Experts by listing their indices in `moe_layers`; the
rest stay dense. See **`QWEN3.md`**.

### `Qwen3Next` — hybrid linear-attention MoE

Each `Qwen3NextBlock` is three `GatedLinearAttentionBlock` layers followed by one
`GatedAttention` layer, every one of them preceded by a zero-centered RMSNorm and followed by
its own MoE and residual. This trades most of the quadratic attention for linear-time gated
recurrence while keeping one full-attention layer per block. See **`QWEN3_next.md`**.

### `Qwen3 Embedding` / `Qwen3 Reranker` — retrieval models

`qwen3_embeddings.py` is the one module in this package with no long-form document; its
module docstring is the reference. Two ideas drive it:

- **Embedding**: a causal transformer whose final embedding is the hidden state of the last
  non-padding token (not a `[CLS]` token), optionally prefixed with a task instruction,
  optionally truncated to a shorter dimension (Matryoshka representation learning), and
  L2-normalized so cosine similarity is a dot product.
- **Reranking**: relevance is scored *generatively*. The query, document and instruction are
  formatted into one prompt that asks the model to answer "yes" or "no", and the score is the
  softmax probability of the "yes" token over the two candidate logits.

Each is provided as a `keras.layers.Layer` (`Qwen3EmbeddingLayer`, `Qwen3RerankerLayer`) for
composition and as a `keras.Model` wrapper (`Qwen3EmbeddingModel`, `Qwen3RerankerModel`) for
direct `compile()` / `fit()` use.

---

## 3. Component Reference

| Component | Location | Purpose |
| :--- | :--- | :--- |
| **`Qwen3`** | `models.qwen.qwen3.Qwen3` | The dense/MoE decoder-only backbone. |
| **`create_qwen3`** | `models.qwen.qwen3.create_qwen3` | Single entry point: `create_qwen3(config_or_variant, task_type=...)`. |
| **`create_qwen3_generation`** | `models.qwen.qwen3.create_qwen3_generation` | Autoregressive LM head over the backbone. |
| **`create_qwen3_classification`** | `models.qwen.qwen3.create_qwen3_classification` | Sequence classifier: pooling via the shared `SequencePooling` layer, then a dense head. |
| **`Qwen3Next`** | `models.qwen.qwen3_next.Qwen3Next` | The hybrid Gated-DeltaNet + Gated-Attention MoE backbone. |
| **`create_qwen3_next`** | `models.qwen.qwen3_next.create_qwen3_next` | Same entry-point shape as `create_qwen3`. |
| **`create_qwen3_next_generation`** | `models.qwen.qwen3_next.create_qwen3_next_generation` | Autoregressive LM head. |
| **`create_qwen3_next_classification`** | `models.qwen.qwen3_next.create_qwen3_next_classification` | Sequence classifier. |
| **`Qwen3NextBlock`** | `models.qwen.components.Qwen3NextBlock` | One `3x GDN + 1x GA` block with per-layer norm, MoE and residual. |
| **`Qwen3EmbeddingLayer`** | `models.qwen.qwen3_embeddings.Qwen3EmbeddingLayer` | Instruction-aware, last-token-pooled, L2-normalized embedding layer. |
| **`Qwen3RerankerLayer`** | `models.qwen.qwen3_embeddings.Qwen3RerankerLayer` | Yes/no generative relevance scorer. |
| **`Qwen3EmbeddingModel`** | `models.qwen.qwen3_embeddings.Qwen3EmbeddingModel` | `keras.Model` wrapper around the embedding layer. |
| **`Qwen3RerankerModel`** | `models.qwen.qwen3_embeddings.Qwen3RerankerModel` | `keras.Model` wrapper around the reranker layer. |

### Variants

| Class | `MODEL_VARIANTS` keys |
| :--- | :--- |
| `Qwen3` | `"tiny"`, `"small"`, `"medium"`, `"30b-coder"` |
| `Qwen3Next` | `"tiny"`, `"small"`, `"80b"`, `"80b_a3b"` |

`from_variant(variant, **overrides)` is available on both classes; the factories accept the
same variant strings. Configuration precedence in `create_qwen3` / `create_qwen3_next` is:
variant defaults, then a config dict if one is passed, then explicit `**kwargs`.

---

## 4. Quick Start

```python
from dl_techniques.models.qwen.qwen3 import create_qwen3

# Autoregressive generation model from a named variant.
gen_model = create_qwen3("tiny")

# Sequence classification with 5 labels.
clf_model = create_qwen3("small", task_type="classification", num_labels=5)

# Variant defaults with a targeted override.
shallow = create_qwen3("tiny", num_layers=2)
```

`Qwen3Next` follows the same shape:

```python
from dl_techniques.models.qwen.qwen3_next import create_qwen3_next

model = create_qwen3_next("tiny", task_type="generation")
```

The backbones take `{"input_ids": ..., "attention_mask": ...}` as input. For anything beyond
this — MoE configuration, expert-utilization analysis, custom training loops with the MoE
auxiliary loss — go to `QWEN3.md` and `QWEN3_next.md`.

---

## 5. Serialization

Every class listed above is decorated with `@keras.saving.register_keras_serializable()` and
implements `get_config()`, so models round-trip through the `.keras` format:

```python
import keras

model.save("qwen3.keras")
restored = keras.models.load_model("qwen3.keras")
```

The classification head pools with the shipped `SequencePooling` layer rather than a
hand-rolled masked mean. That layer is parameter-free, so the choice does not change any
checkpoint — but do not re-inline a bespoke pooling branch here; see the `DECISION` comment in
`qwen3.py` at `create_qwen3_classification`.

Both classification factories default to `pooling_strategy="last"`: the last position kept by
`attention_mask`. These backbones are strictly causally masked, so position 0 attends only to
itself and the old `"cls"` default made the pooled vector a function of the first token id
alone — measured 0.000e+00 logit movement when a non-first token changed. `"cls"` is still
accepted, for bidirectional-era checkpoints only; a classifier trained under it does not
reload comparably against the new default.

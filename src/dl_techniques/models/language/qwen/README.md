# Qwen: Decoder-Only Language Models, MoE, and Text Embedding

Keras 3 implementations of the **Qwen3** model family: a standard decoder-only transformer
with optional Mixture-of-Experts layers, the **Qwen3 Next** hybrid (Gated DeltaNet + Gated
Attention) MoE architecture, and the **Qwen3 Embedding / Reranker** retrieval models.

This file is the package index. Two of the three architectures already have full-length
documents next to their source; this README says what is in the package, which document
covers what, and what has no document of its own.

---

## 1. What this package contains

`src/dl_techniques/models/language/qwen/`:

| File | Contents |
| :--- | :--- |
| `qwen3.py` | `Qwen3` (`keras.Model`) plus `create_qwen3`, `create_qwen3_generation`, `create_qwen3_classification`. |
| `qwen3_next.py` | `Qwen3Next` (`keras.Model`) plus `create_qwen3_next`, `create_qwen3_next_generation`, `create_qwen3_next_classification`. |
| `components.py` | `Qwen3NextBlock` — the `3x Gated DeltaNet + 1x Gated Attention` block that `Qwen3Next` stacks. |
| `qwen3_embeddings.py` | `Qwen3EmbeddingLayer`, `Qwen3RerankerLayer`, `Qwen3EmbeddingModel`, `Qwen3RerankerModel`. |
| `QWEN3.md` | Full document for `Qwen3`: architecture diagram, variants, MoE configuration, training, serialization. |
| `QWEN3_next.md` | Full document for `Qwen3Next`, including a `Qwen3` vs `Qwen3 Next` comparison section. |
| `__init__.py` | **A curated re-export module with `__all__`.** Import `Qwen3`, `Qwen3Next`, their `create_*` factories and the embedding/reranker pair straight from `dl_techniques.models.language.qwen`; submodule imports keep working. |

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

- **Embedding**: a **bidirectional** transformer whose final embedding is the hidden state of
  the last non-padding token (not a `[CLS]` token), optionally truncated to a shorter
  dimension (Matryoshka representation learning), and L2-normalized so cosine similarity is a
  dot product. The trunk is **not causal** and that is deliberate: `Qwen3EmbeddingLayer.call`
  forwards the caller's 2D padding mask unchanged and never calls
  `create_causal_attend_mask` (the sibling `Qwen3RerankerLayer` does, because it reads an LM
  head and is a next-token prediction). Nothing is predicted from the pooled vector, so there
  is no target to leak. Last-token pooling here is a convention carried over from the
  decoder-only lineage, not evidence of causality; see the module docstring of
  `qwen3_embeddings.py`. A task instruction, if used, is prefixed by the CALLER in the token
  ids — no code in this package inserts one.
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
| **`Qwen3`** | `models.qwen.qwen3.Qwen3` | The dense/MoE decoder-only backbone. With `include_top=True` and a `task_config` it owns one task head in place of `lm_head` (section 5). |
| **`create_qwen3`** | `models.qwen.qwen3.create_qwen3` | Single entry point: `create_qwen3(config_or_variant, task_type=...)`. |
| **`create_qwen3_generation`** | `models.qwen.qwen3.create_qwen3_generation` | Autoregressive LM head over the backbone. |
| **`create_qwen3_classification`** | `models.qwen.qwen3.create_qwen3_classification` | Sequence classifier: a `Qwen3(include_top=True)` with a `TEXT_CLASSIFICATION` head behind a two-input functional wrapper. |
| **`Qwen3Next`** | `models.qwen.qwen3_next.Qwen3Next` | The hybrid Gated-DeltaNet + Gated-Attention MoE backbone. Takes the same `include_top` knobs as `Qwen3`. |
| **`create_qwen3_next`** | `models.qwen.qwen3_next.create_qwen3_next` | Same entry-point shape as `create_qwen3`. |
| **`create_qwen3_next_generation`** | `models.qwen.qwen3_next.create_qwen3_next_generation` | Autoregressive LM head. |
| **`create_qwen3_next_classification`** | `models.qwen.qwen3_next.create_qwen3_next_classification` | Sequence classifier: a `Qwen3Next(include_top=True)`, as for `Qwen3`. |
| **`Qwen3NextBlock`** | `models.qwen.components.Qwen3NextBlock` | One `3x GDN + 1x GA` block with per-layer norm, MoE and residual. |
| **`Qwen3EmbeddingLayer`** | `models.qwen.qwen3_embeddings.Qwen3EmbeddingLayer` | Bidirectional, last-token-pooled, L2-normalized embedding layer. Takes `input_ids`/`attention_mask`; any instruction prefix is the caller's. |
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
from dl_techniques.models.language.qwen.qwen3 import create_qwen3

# Autoregressive generation model from a named variant.
gen_model = create_qwen3("tiny")

# Sequence classification with 5 labels.
clf_model = create_qwen3("small", task_type="classification", num_labels=5)

# Variant defaults with a targeted override.
shallow = create_qwen3("tiny", num_layers=2)
```

`Qwen3Next` follows the same shape:

```python
from dl_techniques.models.language.qwen.qwen3_next import create_qwen3_next

model = create_qwen3_next("tiny", task_type="generation")
```

The backbones take `{"input_ids": ..., "attention_mask": ...}` as input. For anything beyond
this — MoE configuration, expert-utilization analysis, custom training loops with the MoE
auxiliary loss — go to `QWEN3.md` and `QWEN3_next.md`.

---

## 5. Serialization

Every class listed above is decorated with `@register_dl_technique(...)` (from
`dl_techniques.utils.keras_registration`) and implements `get_config()`, so models
round-trip through the `.keras` format. The package string is the defining module's dotted
path with `models/`'s `language/` family directory stripped, so the keys are
`dl_techniques.models.qwen.qwen3>Qwen3`, `dl_techniques.models.qwen.qwen3_next>Qwen3Next`,
`dl_techniques.models.qwen.components>Qwen3NextBlock`, and
`dl_techniques.models.qwen.qwen3_embeddings><ClassName>` for `Qwen3EmbeddingLayer`,
`Qwen3RerankerLayer`, `Qwen3EmbeddingModel` and `Qwen3RerankerModel`. The helper also binds a
legacy `Custom>ClassName` alias, so older archives still load:

```python
import keras

model.save("qwen3.keras")
restored = keras.models.load_model("qwen3.keras")
```

### The task head inside the model (`include_top`)

`Qwen3` and `Qwen3Next` take `include_top=False`, `task_config=None`,
`head_config_overrides=None` and `pad_token_id=None`. With `include_top=True` and a
`task_config` (an `NLPTaskConfig` or a dict of its fields) the model replaces `lm_head` with
ONE task head from `dl_techniques.layers.heads.nlp`, named `task_head`: no `lm_head` is
created and no vocabulary logits are computed. The head is fed the final-normed hidden
states and pools `"last"` by default, under the input's `attention_mask` or, without one,
`input_ids != pad_token_id` when a pad id was given, else every position (id 0 is a real
token: a padded batch must pass `attention_mask` or the tokenizer's pad id). `call` returns the head's logits (a dict only for multi-output
heads such as QA); with `return_dict=True` a bare output comes back as `{"logits": ...}`.
The classification factories are this model behind a functional wrapper taking
`[input_ids, attention_mask]`; `pooling_strategy` and `classifier_dropout_rate` reach the
head as its `pooling_type` and dropout. A classifier checkpoint written before this change
pooled the vocabulary logits (`Qwen3`, unbuildable since c6f5a13f2) or used a hand-rolled
`SequencePooling` + `Dense` (`Qwen3Next`), and does not load into the new layout.

The backbone's weight paths are the same with or without a head, so a backbone saved under
one head moves to another, weights unchanged, by layer name (neither class has a
`pretrained` argument):

```python
import numpy as np
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.layers.heads.nlp.model_head import TASK_HEAD_NAME
from dl_techniques.models.language.qwen import Qwen3
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint

cfg = dict(vocab_size=1000, hidden_size=64, num_layers=2, num_attention_heads=4,
           num_key_value_heads=2, max_seq_len=64, moe_layers=[], num_experts=1,
           num_experts_per_tok=1)
clf = Qwen3(**cfg, include_top=True, task_config=NLPTaskConfig(
    name="sst2", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=2))
ids = np.random.randint(1, 1000, (2, 16)).astype("int32")
logits = clf({"input_ids": ids, "attention_mask": np.ones_like(ids)})  # (2, 2)
clf.save("qwen3_sst2.keras")

ner = Qwen3(**cfg, include_top=True, task_config=NLPTaskConfig(
    name="ner", task_type=NLPTaskType.TOKEN_CLASSIFICATION, num_classes=9))
ner(ids)
load_weights_from_checkpoint(ner, "qwen3_sst2.keras", skip_prefixes=(TASK_HEAD_NAME,),
                             strict=True, allow_unloaded=())
```

The same call works on the factory models, either side: a checkpoint of
`create_qwen3_classification` (or `create_qwen3_next_classification`) fills a bare model
with any head, a headless one (`allow_unloaded=("lm_head",)`, since the classifier holds no
`lm_head`) or a `create_qwen3_generation` model with the same backbone weights. The call is
complete-or-raise on both sides, so a generation checkpoint's `lm_head`, which has no home in a
classifier, is named the same way (`allow_unloaded=("lm_head",)`).
`load_weights_from_checkpoint` compares the layers of the one model a functional wrapper
holds. `create_qwen3(..., include_top=True, task_config=...)` (and `create_qwen3_next`)
returns that task's two-input model, named for its task type
(`qwen3_for_token_classification`, ...; a `TEXT_CLASSIFICATION` head gives
`qwen3_for_classification`, as `create_qwen3_classification` does); combined with `task_type="classification"` or
`num_labels` it raises `ValueError`.

Both classification factories default to `pooling_strategy="last"`: the last position kept by
`attention_mask`. These backbones are strictly causally masked, so position 0 attends only to
itself and the old `"cls"` default made the pooled vector a function of the first token id
alone — measured 0.000e+00 logit movement when a non-first token changed. `"cls"` is still
accepted, for bidirectional-era checkpoints only; a classifier trained under it does not
reload comparably against the new default.

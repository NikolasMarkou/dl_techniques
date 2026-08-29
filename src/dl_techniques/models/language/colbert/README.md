# ColBERT: Efficient and Effective Passage Search via Late Interaction

A Keras 3 implementation of **ColBERT v1** (Khattab & Zaharia, 2020) and **ColBERT v2**
(Santhanam et al., 2021) — a BERT backbone, a bias-free 128-dimensional projection, L2
normalization, and MaxSim late interaction — packaged as one shared encoder with two
training recipes and one v2-only index-time codec.

---

> ## Read this before anything else
>
> **1. No pretrained weights exist. Not for ColBERT, and not for the BERT backbone.**
> `ColBERT.from_variant(pretrained=True)` raises `NotImplementedError`, and so does
> `BERT.from_variant(pretrained=True)` underneath it. Nothing is downloaded, nothing is
> bundled, and there is no URL to point at. **Every number this package's trainers produce
> is a wiring result** — evidence that gradients flow, that the loss decreases and that
> the plumbing is correct — **and never a retrieval-quality claim.** Do not compare any
> output of this package to a published MS MARCO or BEIR figure. It is not the same
> experiment, it does not start from the same weights, and it is not measuring the same
> thing.
>
> **2. v1 and v2 build the same network.** See
> [§9 Deviations from the reference](#9-deviations-from-the-reference) and
> [§3.3](#33-why-there-is-one-class-and-not-two) — this is a fact about the reference
> implementation, not a shortcut taken here.

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Problem ColBERT Solves](#2-the-problem-colbert-solves)
3. [How ColBERT Works: Core Concepts](#3-how-colbert-works-core-concepts)
4. [Architecture Deep Dive](#4-architecture-deep-dive)
5. [Quick Start Guide](#5-quick-start-guide)
6. [Component Reference](#6-component-reference)
7. [Configuration & Model Variants](#7-configuration--model-variants)
8. [Comprehensive Usage Examples](#8-comprehensive-usage-examples)
9. [Deviations from the reference](#9-deviations-from-the-reference)
10. [Training](#10-training)
11. [Serialization & Deployment](#11-serialization--deployment)
12. [Testing & Validation](#12-testing--validation)
13. [Troubleshooting & FAQs](#13-troubleshooting--faqs)
14. [Technical Details](#14-technical-details)
15. [Citation](#15-citation)

---

## 1. Overview

### What is ColBERT?

ColBERT is a retrieval model that keeps **one vector per token** instead of one vector per
passage, and scores a (query, document) pair by letting each query term take its single best
match in the document:

```
S(q, d) = sum_i  max_j  E_q[i] · E_d[j]^T
```

The two texts are encoded **independently**, so every document embedding can be computed once,
offline, and stored. The only thing that happens at query time is that cheap similarity between
two already-computed matrices.

### Key Innovations of this Implementation

| | |
|---|---|
| **One shared encoder** | The query path and the document path traverse the *same* `ColBERTProjection` instance — the same object, not two layers with the same config. A test asserts the identity with `is` |
| **Finite sentinel masking** | Masked document positions are filled with a finite large-negative constant before the max-reduce, never `-inf`, so an all-masked document scores to a finite number rather than propagating `NaN` into a softmax |
| **`mixed_float16`-safe** | `MaxSimScorer` promotes its masking and both reductions to `float32`; the projection normalizes through a float32 reduction. Both are measured fixes, not precaution — see [§14.3](#143-two-measured-half-precision-defects) |
| **Fixed output structure** | `call()` always returns the same three keys, whether or not the optional masks were passed, because an input-dependent output *structure* breaks `.predict()` |
| **Honest provenance** | The variant table carries two labelled provenance classes and never blends them; every divergence from the reference is named in [§9](#9-deviations-from-the-reference) |

### Why ColBERT Matters

Dense single-vector retrieval averages a whole passage into one embedding. A query term
matching one rare word in a long document is diluted by every other word around it.
Cross-encoders fix that by feeding query and document through a transformer *together* — and
become unusable at scale, because nothing can be precomputed and every pair costs a full
forward pass. Late interaction is the factorization that resolves the tension: term-level
evidence survives to scoring time, but the expensive interaction between the two texts never
happens inside the network.

---

## 2. The Problem ColBERT Solves

### The cost of a single vector

Consider the query *"nikola tesla wardenclyffe tower"* against a 200-word biography that
mentions Wardenclyffe once. Mean-pooled into 768 numbers, that one mention is roughly 0.5% of
the representation. The signal is not lost so much as averaged away, and no amount of
downstream re-ranking can recover what the encoder already discarded.

### The cost of full interaction

A cross-encoder scores that pair perfectly, because "Wardenclyffe" in the query attends
directly to "Wardenclyffe" in the document. It also means a 10-million-document corpus needs
10 million transformer forward passes per query. That is why cross-encoders are re-rankers over
a shortlist, never first-stage retrievers.

### The late-interaction resolution

ColBERT moves the interaction *out* of the network and *into* the score. The encoder never sees
both texts. What survives to scoring time is a matrix of per-token vectors — so
"Wardenclyffe" in the query can still find "Wardenclyffe" in the document, at the cost of one
matrix product, and the document side of that product was computed months ago.

---

## 3. How ColBERT Works: Core Concepts

### 3.1 The high-level architecture

```
query text  ──▶ ColBERTTokenizer ──▶ [CLS] [Q] tokens [SEP] [MASK] ... [MASK]
                                         │
                                    BERT encoder  (shared weights)
                                         │
                                 ColBERTProjection  (shared instance)
                                    Dense(128, bias=False)
                                       × mask
                                       L2 normalize
                                         │
                                         ▼
document text ─▶ ColBERTTokenizer ─▶ [CLS] [D] tokens [SEP] [PAD] ...  ──▶ (same path)
                                         │
                                         ▼
                                    MaxSimScorer
                          (Q × D) → sentinel-mask → max over D → sum over Q
                                         │
                                         ▼
                                    score, shape (batch,)
```

### 3.2 The three asymmetries

Queries and documents traverse **identical weights**. They differ only in preparation, and
each difference is a rule, not a convention:

| | Query | Document |
|---|---|---|
| Marker token | `[Q]` at position 1 | `[D]` at position 1 |
| Padding | overwritten with `[MASK]`, and those slots **participate** in scoring (query augmentation) | ordinary `[PAD]`, masked out |
| Punctuation skiplist | **never applied** — the reference passes an empty skiplist on this side | applied; a punctuation position is zeroed exactly like padding |

Get either direction wrong and the model still runs, still serializes and still trains. Query
augmentation is the v1 paper's headline mechanism, and applying the document skiplist to a
query silently deletes real query terms; both directions are asserted by dedicated tests.

### 3.3 Why there is one class and not two

**ColBERT v2 did not change the network.** The official
[`stanford-futuredata/ColBERT`](https://github.com/stanford-futuredata/ColBERT) repository
ships a single
[`colbert/modeling/colbert.py`](https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/modeling/colbert.py)
for both papers. There is no v1-only code path in it. v1 behaviour is v2's code with
`use_ib_negatives=False`, `nway=2`, no distillation scores and no residual compression applied
at indexing time. ColBERTv2's own architecture section restates v1's encoder and the identical
MaxSim formula.

What v2 changed is:

1. **The supervision** — cross-encoder KL distillation over `nway` (typically 64) candidates
   with in-batch negatives, replacing v1's pairwise softmax cross-entropy over a triple.
2. **The index** — an index-time residual codec that compresses stored embeddings to
   1 or 2 bits per dimension plus a centroid id. It appears in no forward pass and in no loss.

Accordingly this package exports **one** `ColBERT` class and **two** genuine factories,
`create_colbert_v1` and `create_colbert_v2`. They are not aliases — each carries the recipe it
is documented to pair with, and each is a real, tested, exported entry point — but they build
the same architecture, and a test asserts their weight signatures are identical. The v1/v2
distinction lives in the trainers and in `ResidualCompressionCodec`.

---

## 4. Architecture Deep Dive

### 4.1 `ColBERTProjection`

`Dense(dim, use_bias=False)` with **no activation**, then a mask multiply, then L2 normalize —
in that order. The order is load-bearing: masking after normalization would leave a padded
position with a unit-norm embedding pointing in a random direction, which the max-reduce could
then select. Masking first makes the position exactly zero, whose inner product with every
query term is exactly zero.

The normalization runs through a module-private `_safe_l2_normalize` which reduces in `float32`
and floors the squared norm, because `keras.ops.normalize` returns `NaN` on an all-zero row
under `mixed_float16` — and after the mask multiply, all-zero rows are the *ordinary* case,
not an edge case.

### 4.2 `MaxSimScorer`

Builds the **full dense** `(batch, query_len, doc_len)` score matrix, writes a finite negative
sentinel over masked document positions, takes the max over the document axis and sums over the
query axis. Never index exclusion — a variable-length gather would break the static shapes the
rest of the graph depends on, and the sentinel is what makes an all-masked document reduce to a
deterministic finite constant instead of `-inf`.

### 4.3 `ColBERTTokenizer`

A plain Python class, deliberately **not** a `keras.Layer`: everything it does is host-side
`str` → `int` work that runs once, outside the compute graph. It wraps
`dl_techniques.utils.tokenizer.TiktokenPreprocessor` and owns the `[Q]`/`[D]` marker
allocation, query `[MASK]` augmentation and the document punctuation skiplist.

### 4.4 `ResidualCompressionCodec`

v2, **index-time only**. k-means centroids over a sample of the index; each vector is stored as
its nearest centroid's id plus its residual quantized to `nbits ∈ {1, 2}` per dimension.
`decode()` re-L2-normalizes the reconstruction — a detail present in the reference *code* and
absent from the paper. A structural test asserts that no symbol from `compression.py` is
reachable from `ColBERT.call` or from either loss.

---

## 5. Quick Start Guide

### Your first ColBERT score (30 seconds)

```python
from dl_techniques.models.language.colbert import ColBERTTokenizer, create_colbert_v1

tokenizer = ColBERTTokenizer(query_maxlen=8, doc_maxlen=16)

model = create_colbert_v1(
    "tiny",
    vocab_size=100277,        # cl100k_base
    query_maxlen=8,
    doc_maxlen=16,
    max_position_embeddings=64,
)

queries = tokenizer.tokenize_queries(["what is late interaction"])
docs = tokenizer.tokenize_documents(["Late interaction, explained: MaxSim."])

outputs = model({
    "query_input_ids": queries["input_ids"],
    "query_attention_mask": queries["attention_mask"],
    "doc_input_ids": docs["input_ids"],
    "doc_attention_mask": docs["attention_mask"],
    "doc_skiplist_mask": docs["skiplist_mask"],
})

print(sorted(outputs))
print(outputs["score"].shape, outputs["query_embeddings"].shape, outputs["doc_embeddings"].shape)
```

```
['doc_embeddings', 'query_embeddings', 'score']
(1,) (1, 8, 128) (1, 16, 128)
```

*(MEASURED — this snippet was executed. The score itself is a random-init number, `5.5697765`
on one seed; it means nothing, which is the whole point of the box at the top of this file.)*

The output key set is **fixed**. Omit `doc_skiplist_mask` and you still get all three keys —
the optional inputs are resolved to concrete tensors before the return, because an output
structure that varies with the inputs breaks `.predict()`.

---

## 6. Component Reference

### 6.1 `ColBERT` (Model Class)

| Member | |
|---|---|
| `ColBERT(vocab_size=..., hidden_size=..., dim=128, query_maxlen=32, doc_maxlen=220, ...)` | Direct construction |
| `ColBERT.from_variant(variant, pretrained=False, **overrides)` | Named variant. `pretrained=True` raises `NotImplementedError`; a **string** path loads a local `.keras` checkpoint |
| `ColBERT.MODEL_VARIANTS` | `"tiny"`, `"small"`, `"base"`, `"large"` — see [§7](#7-configuration--model-variants) |
| `model.encode_query(inputs, training=None)` | `(batch, query_len, dim)` embeddings. No skiplist on this side |
| `model.encode_document(inputs, training=None)` | `(batch, doc_len, dim)` embeddings. Accepts an optional `skiplist_mask` |
| `model.score(q_emb, d_emb, doc_mask=None, query_mask=None)` | `(batch,)` MaxSim scores over already-computed embeddings |
| `model(inputs)` | Both towers plus the score, as `{"score", "query_embeddings", "doc_embeddings"}` |
| `model.encoder` / `model.projection` / `model.scorer` | The three owned sub-components |

### 6.2 Factory Functions

| Function | Default variant | Pairs with |
|---|---|---|
| `create_colbert(variant="base", ...)` | `"base"` | version-neutral |
| `create_colbert_v1(variant="base", ...)` | `"base"` | `ColBERTPairwiseSoftmaxLoss`, `train_colbert_v1.py` |
| `create_colbert_v2(variant="base", ...)` | `"base"` | `ColBERTDistillationLoss`, `train_colbert_v2.py`, `ResidualCompressionCodec` |

All three delegate to `ColBERT.from_variant` with no logic of their own.

### 6.3 `ColBERTTokenizer`

| Member | |
|---|---|
| `tokenize_queries(texts)` | `{"input_ids", "attention_mask"}`, padded to `query_maxlen` with `[MASK]` |
| `tokenize_documents(texts)` | `{"input_ids", "attention_mask", "skiplist_mask"}`, padded to `doc_maxlen` |
| `punctuation_token_ids` | The skiplist as a `frozenset` (64 ids on `cl100k_base`; see [§9](#9-deviations-from-the-reference)) |
| `query_marker_token_id` / `doc_marker_token_id` | Derived from the live `n_vocab`, not hardcoded |
| `get_config()` / `from_config()` | Round trips every constructor argument |

### 6.4 `ResidualCompressionCodec`

| Member | |
|---|---|
| `fit(vectors)` | Learns k-means centroids and bucket boundaries. Returns `self` |
| `encode(vectors)` | `(codes: int32 (n,), packed: uint8 (n, bytes_per_vector))` |
| `decode(codes, packed)` | `(n, dim)` reconstruction, re-L2-normalized |
| `reconstruction_error(vectors)` | Mean round-trip L2 error |
| `bytes_per_vector`, `num_levels`, `is_fitted` | Properties |
| `save(path)` / `ResidualCompressionCodec.load(path)` | Codebook persistence |

### 6.5 Losses (in `dl_techniques.losses`)

| Class | Recipe |
|---|---|
| `ColBERTPairwiseSoftmaxLoss(nway=2)` | v1. `logsumexp_k s[b,k] - s[b,0]`. **The positive is positional — `y_true` is not consulted.** Order your tuples positive-first |
| `ColBERTDistillationLoss(nway=64, distillation_alpha=1.0)` | v2. `KL(log_softmax(alpha * teacher) ‖ log_softmax(student))` with `log_target=True` semantics and `batchmean` reduction |

---

## 7. Configuration & Model Variants

### Two provenance classes, deliberately not blended

The variant table mixes numbers from **two different sources**, and they are labelled
separately rather than presented as one citation:

**Class A — the ColBERT-side numbers.** `dim=128`, `query_maxlen=32`, `doc_maxlen=220` are read
from the official reference's
[`colbert/infra/config/settings.py`](https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/infra/config/settings.py)
(`dim: int = 128`, `query_maxlen: int = 32`, `doc_maxlen: int = 220`). They are identical in
every row because ColBERT does not scale them with the backbone; the reference ships one value
for all sizes.

**Not from any ColBERT source — this repository's own backbone ladder.** `hidden_size`,
`num_layers`, `num_heads` and `intermediate_size` are copied from
`dl_techniques.models.language.bert.model.BERT.MODEL_VARIANTS`, which tracks the Turc et al.
2019 released BERT checkpoints. **Published ColBERT uses `bert-base` only.** `tiny`, `small`
and `large` are this library's capacity ladder, offered for cheap tests and for scaling, and no
ColBERT paper reports them.

| Variant | Backbone (this repo's ladder) | ColBERT-side (Class A) | Note |
|---|---|---|---|
| `tiny` | 256 hidden / 4 layers / 4 heads / 1024 FFN | `dim=128`, `query_maxlen=32`, `doc_maxlen=220` | tests and edge; not a published size |
| `small` | 512 / 6 / 8 / 2048 | same | tight budgets; not a published size |
| `base` | 768 / 12 / 12 / 3072 | same | **the published configuration's backbone size** |
| `large` | 1024 / 24 / 16 / 4096 | same | no ColBERT paper reports this size |

### Customizing

```python
from dl_techniques.models.language.colbert import create_colbert_v2

model = create_colbert_v2(
    "base",
    vocab_size=100277,          # match your tokenizer's encoding
    dim=96,                     # narrower retrieval embeddings
    doc_maxlen=180,             # shorter passages
    mask_value=-1e4,            # the MaxSim sentinel
)
```

`max_position_embeddings` must be at least `max(query_maxlen, doc_maxlen)`; the constructor
raises `ValueError` naming the offending value otherwise.

---

## 8. Comprehensive Usage Examples

Every snippet below was **executed** before being pasted here; the outputs are the real ones.

### Example 1: The query/document asymmetry, visible

```python
from dl_techniques.models.language.colbert import ColBERTTokenizer

tok = ColBERTTokenizer(query_maxlen=8, doc_maxlen=16)
q = tok.tokenize_queries(["late interaction"])
d = tok.tokenize_documents(["Late interaction, explained."])

print("query keys :", sorted(q))
print("doc keys   :", sorted(d))
print("[Q] id      =", tok.query_marker_token_id, " [D] id =", tok.doc_marker_token_id)
print("query ids   =", q["input_ids"][0].tolist())
print("query mask  =", q["attention_mask"][0].tolist())
print("|skiplist|  =", len(tok.punctuation_token_ids))
print("doc skiplist=", d["skiplist_mask"][0].tolist())
```

```
query keys : ['attention_mask', 'input_ids']
doc keys   : ['attention_mask', 'input_ids', 'skiplist_mask']
[Q] id      = 100261  [D] id = 100262
query ids   = [100257, 100261, 5185, 16628, 100258, 100259, 100259, 100259]
query mask  = [1, 1, 1, 1, 1, 1, 1, 1]
|skiplist|  = 64
doc skiplist= [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

Read the query ids: `[CLS]=100257`, `[Q]=100261`, two content tokens, `[SEP]=100258`, then
three `[MASK]=100259` — that is query augmentation, and the mask is all ones, so those slots
participate in scoring. The document's `skiplist_mask` has two zeros: the comma and the period.

### Example 2: Offline indexing — encode once, score later

```python
from dl_techniques.models.language.colbert import ColBERTTokenizer, create_colbert_v2
import numpy as np

tok = ColBERTTokenizer(query_maxlen=8, doc_maxlen=16)
q = tok.tokenize_queries(["late interaction"])
d = tok.tokenize_documents(["Late interaction, explained."])

model = create_colbert_v2("tiny", vocab_size=100277, query_maxlen=8, doc_maxlen=16,
                          max_position_embeddings=64)

# --- indexing time: documents only, done once, stored ---
doc_emb = model.encode_document({
    "input_ids": d["input_ids"],
    "attention_mask": d["attention_mask"],
    "skiplist_mask": d["skiplist_mask"],
})

# --- query time: encode the query, score against the stored matrices ---
qry_emb = model.encode_query({
    "input_ids": q["input_ids"],
    "attention_mask": q["attention_mask"],
})
participation = d["attention_mask"] * d["skiplist_mask"]
score = model.score(qry_emb, doc_emb, doc_mask=participation, query_mask=q["attention_mask"])

print("doc_emb", doc_emb.shape, "qry_emb", qry_emb.shape, "score", score.shape)
print("masked positions are exactly zero:",
      float(np.abs(np.asarray(doc_emb)[0][np.asarray(participation)[0] == 0]).max()))
```

```
doc_emb (1, 16, 128) qry_emb (1, 8, 128) score (1,)
masked positions are exactly zero: 0.0
```

The `0.0` is the mask-before-normalize invariant, observed rather than asserted: a filtered
position has an all-zero embedding, so its inner product with every query term is exactly zero
*before* the sentinel is even applied.

### Example 3: v1 and v2 really are the same network

```python
from dl_techniques.models.language.colbert import create_colbert_v1, create_colbert_v2

kwargs = dict(vocab_size=512, query_maxlen=8, doc_maxlen=16, max_position_embeddings=64)
v1 = create_colbert_v1("tiny", **kwargs)
v2 = create_colbert_v2("tiny", **kwargs)
for m in (v1, v2):
    m.build({"query_input_ids": (None, 8), "doc_input_ids": (None, 16)})

# `v.path` is prefixed with the model's auto-numbered name ("col_bert", "col_bert_1"),
# so compare the paths BELOW that prefix.
signature = lambda m: sorted(
    (v.path.split("/", 1)[1], tuple(v.shape)) for v in m.weights
)
print("same weight signature:", signature(v1) == signature(v2))
print("params:", v1.count_params(), v2.count_params())
```

```
same weight signature: True
params: 3340288 3340288
```

### Example 4: The error contracts

```python
from dl_techniques.models.language.colbert import ColBERT

ColBERT.from_variant("enormous")
# ValueError: Unknown variant 'enormous'. Available variants: ['large', 'base', 'small', 'tiny']

ColBERT.from_variant("tiny", pretrained=True)
# NotImplementedError: No pretrained ColBERT weights are distributed with dl_techniques
# (requested variant 'tiny'), and no pretrained weights exist for the BERT backbone either. ...
```

### Example 5: Both losses

```python
import keras
from dl_techniques.losses import ColBERTPairwiseSoftmaxLoss, ColBERTDistillationLoss

v1_loss = ColBERTPairwiseSoftmaxLoss(nway=2)
scores = keras.ops.convert_to_tensor([3.0, 1.0, 0.5, 2.5])   # two <q, d+, d-> triples
print("v1 loss:", float(v1_loss(keras.ops.zeros_like(scores), scores)))

v2_loss = ColBERTDistillationLoss(nway=3, distillation_alpha=1.0)
teacher = keras.ops.convert_to_tensor([[2.0, 0.0, 1.0]])
student = keras.ops.convert_to_tensor([[3.0, 1.0, 0.0]])
print("v2 loss:", float(v2_loss(teacher, student)))
```

```
v1 loss: 1.1269280910491943
v2 loss: 0.25169697403907776
```

`ColBERTPairwiseSoftmaxLoss` **ignores `y_true` entirely** — the positive is index 0 of each
`nway` group, exactly as `labels = zeros` means in the reference. Pass
`keras.ops.zeros_like(scores)` as a placeholder; Keras' `Loss.__call__` wrapper requires a real
tensor there.

### Example 6: v2 index-time compression

```python
import numpy as np
from dl_techniques.models.language.colbert import ResidualCompressionCodec

rng = np.random.default_rng(0)
embeddings = rng.normal(size=(2048, 128))
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

codec = ResidualCompressionCodec(dim=128, nbits=2, num_centroids=256).fit(embeddings)
codes, packed = codec.encode(embeddings[:4])
restored = codec.decode(codes, packed)

print("codes", codes.shape, codes.dtype, "packed", packed.shape, packed.dtype)
print("bytes/vector:", codec.bytes_per_vector,
      " unit norm:", np.linalg.norm(restored, axis=1).round(6).tolist())
print("err nbits=2:", round(codec.reconstruction_error(embeddings[:256]), 6))

codec1 = ResidualCompressionCodec(dim=128, nbits=1, num_centroids=256).fit(embeddings)
print("err nbits=1:", round(codec1.reconstruction_error(embeddings[:256]), 6))
```

```
codes (4,) int32 packed (4, 32) uint8
bytes/vector: 32  unit norm: [1.0, 1.0, 1.0, 1.0]
err nbits=2: 0.404551
err nbits=1: 0.654056
```

`nbits=3` raises `ValueError`. Note that these errors are large because the fixture is
*isotropic Gaussian noise* on the unit sphere, which has no cluster structure for k-means to
find — real encoder output compresses far better. The property that holds regardless, and the
one the test pins, is `err(nbits=2) < err(nbits=1)`.

---

## 9. Deviations from the reference

Every item here is a place where this implementation knowingly differs from
`stanford-futuredata/ColBERT`. None of them is an accident, and none is hidden in a code
comment only.

**How much of each is test-pinned, verified 2026-08-25 by running every node id below:**

| § | The guard that would go red | Pinned? |
|---|---|---|
| 9.1 | `test_tokenization.py::test_markers_are_derived_from_the_live_vocabulary`, `::test_a_marker_colliding_with_a_special_id_raises` | yes |
| 9.2 | `test_tokenization.py::test_the_skiplist_covers_every_punctuation_symbol_in_both_spacings` | yes |
| 9.3 | `test_tokenization.py::test_the_default_attends_to_mask_tokens`, `::test_the_attend_to_mask_tokens_policy_is_pinned`, `::test_the_flag_decides_whether_augmented_slots_reach_maxsim` | yes |
| 9.4 | `test_model.py::test_the_backbone_runs_the_tanh_gelu_approximation` | yes, **since 2026-08-25** |
| 9.5 | `test_components.py::test_both_layers_run_under_mixed_float16_with_finite_outputs` (the `float32` return-dtype assertion) | yes, **since 2026-08-25** |
| 9.6 | `test_components.py::test_a_fully_masked_projection_row_is_exactly_zero` plus the `mixed_float16` finiteness arm above | yes |
| 9.7 | `test_compression.py::test_two_bits_reconstruct_strictly_better_than_one_bit` pins the only claim §9.7 makes (`err(nbits=2) < err(nbits=1)`). Its **provenance** half — "derived here, not transcribed from the reference" — is a statement about how the code was written and **no test can pin it** | **half** |
| 9.8 | `test_model.py::test_every_variant_row_carries_the_reference_colbert_defaults` | yes |
| 9.9 | `test_model.py::test_from_variant_refuses_pretrained_true` | yes |

So: eight of the nine would be caught by a named assertion if they silently drifted back, and
§9.7's provenance sentence is documentation-only by construction. §9.4 and §9.5 were
documentation-only until 2026-08-25 — an adversarial review found that the preamble here
claimed a completeness it did not have, which is why the table exists instead of a sentence.

This list has been wrong once. Iteration 1 shipped a **tenth**, unlisted divergence — the
document punctuation skiplist was being fed to the BERT backbone as its attention mask,
where the reference passes the plain padding mask and applies the skiplist only to the
projected embeddings. That is now **fixed**, not documented: the two masks are separate
tensors and the document path matches the reference exactly (§14.5). It is named here
because the claim "this list is complete" is only worth as much as its track record.

### 9.1 The `[Q]` / `[D]` markers come from free Tiktoken slots, not `[unused0]` / `[unused1]`

**Reference**: marks queries and documents with `[Q]` / `[D]`, implemented as BERT WordPiece's
`[unused0]` / `[unused1]` slots.

**Here**: there is **no WordPiece tokenizer anywhere in this library** — the NLP stack is
Tiktoken `cl100k_base`. `ColBERTTokenizer` therefore allocates the two markers from free slots
at the top of that vocabulary, following the existing convention in
`dl_techniques/utils/tokenizer.py` (which allocates its five specials downward from
`n_vocab`). The ids are **derived from the live `n_vocab`** (`n_vocab - 16`, `n_vocab - 15`;
100261 and 100262 on today's `cl100k_base`) rather than hardcoded, and the constructor raises
`ValueError` if either collides with a reserved special or falls outside the vocabulary. A
hardcoded literal would silently land on a real content token if the encoding revision ever
shifted `n_vocab` — perfectly shaped tensors, wrong retrieval forever.

**Cost**: token-id parity with published ColBERT checkpoints is **permanently forfeited**.

### 9.2 The punctuation skiplist covers both bare and space-prefixed forms — 64 ids, not 32

**Reference**: builds the skiplist as `{encode(symbol)[0] for symbol in string.punctuation}` —
32 ids.

**Here**: for each `string.punctuation` symbol, the skiplist takes the first id of **both**
`encode(symbol)` and `encode(" " + symbol)` — 64 ids on `cl100k_base`.

**Why**: because the reference's rule is **inert under BPE**, measured. On `cl100k_base` the
bare `","` is id 11, but ordinary prose (`"late interaction, explained."`) tokenizes the same
comma as `" ,"`, id 1174 — a different token. All 32 bare forms are ids 0..93, single-byte
tokens that essentially never appear in prose, so a literal port builds a 32-id skiplist that
matches almost nothing real: `skiplist_mask` would be all ones for realistic documents and the
punctuation filter would never fire. The bare and space-prefixed id sets are disjoint
(32 + 32 = 64) and every space-prefixed form is exactly one token, so the widening adds no
ambiguity and swallows no content token.

**Cost**: a skiplist literally larger than the reference's, so a token-for-token comparison
against published ColBERT behaviour will differ. Id-for-id parity was already forfeited by
§9.1, so there was nothing left to protect by transcribing a rule that does not work.

### 9.3 `attend_to_mask_tokens` defaults to `True` here, not `False`

**Reference**: `QuerySettings.attend_to_mask_tokens` defaults to **False**, zeroing the
*transformer attention* mask at the augmented `[MASK]` slots.

**Here**: the default is **True**.

**Why**: the reference carries **two** masks over a query — the transformer attention mask
(governed by that flag) and a *participation* mask multiplied onto the projected embeddings
before L2-normalize, computed as `x != pad_token_id` **after** the pads have already been
overwritten with `[MASK]`, hence all-ones at every position regardless of the flag. Augmented
`[MASK]` embeddings **always** participate in MaxSim in the reference; that is the entire query
augmentation mechanism. This tokenizer emits **one** `attention_mask`, which a consumer will
naturally use for both purposes — and `ColBERT` does exactly that. Copying the reference
default would multiply every augmented `[MASK]` embedding by zero, killing the paper's headline
mechanism while every shape, dtype and round-trip test stayed green.

**Cost**: under the default, the `[MASK]` positions are additionally visible as attention keys
to the real query terms — relative to the reference, which hides them.

**`attend_to_mask_tokens=False` is not an escape hatch back to the reference.** It does more
than reproduce the reference's transformer-side masking: because the query path derives its
participation mask from the same emitted tensor, setting it to `False` also deletes the
augmented `[MASK]` slots from MaxSim entirely — the very outcome the `True` default exists to
prevent. In the reference *both* settings leave all positions participating. MEASURED
2026-08-25, `ColBERTTokenizer(query_maxlen=8, attend_to_mask_tokens=...)` on
`"late interaction"` (5 content slots, 3 augmented), per-position L2 norms after
`encode_query`:

| `attend_to_mask_tokens` | emitted `attention_mask` | per-position L2 norm |
|---|---|---|
| `True` (default) | `[1, 1, 1, 1, 1, 1, 1, 1]` | `[1, 1, 1, 1, 1, 1, 1, 1]` |
| `False` | `[1, 1, 1, 1, 1, 0, 0, 0]` | `[1, 1, 1, 1, 1, 0, 0, 0]` |

So the honest statement of the divergence is: **the query path here carries one mask where the
reference carries two**, and the flag moves both roles at once. There is currently no way to
get the reference's exact combination (attention 0 at the augmented slots, participation 1) out
of this tokenizer; doing so would mean emitting a second, query-side participation mask. The
document path does *not* have this problem — its two masks were split in §14.5. Both flag
values are pinned by test at the tokenizer's emitted mask
(`test_the_attend_to_mask_tokens_policy_is_pinned`) **and** downstream at the `[MASK]` rows' L2
norms (`test_the_flag_decides_whether_augmented_slots_reach_maxsim`).

### 9.4 The BERT backbone runs `gelu_tanh`, not exact GELU

This library's `BERT` defaults to the tanh approximation of GELU rather than the exact `erf`
form. The default is kept rather than overridden, since bit-for-bit reproduction of published
ColBERT numbers is already impossible for the reasons above. Pinned at
`ColBERT(...).encoder.hidden_act` **and** at the backbone's serialized config, so a reloaded
model running a different non-linearity is caught too.

### 9.5 `MaxSimScorer` returns `float32` under `mixed_float16`

Under a `mixed_float16` policy the scorer promotes its masking and both reductions to `float32`
and returns `float32`, giving it an output-dtype asymmetry with the projection beside it. This
is a **measured** fix, not a precaution — see [§14.3](#143-two-measured-half-precision-defects).
The *finiteness* half was pinned from the start; the *return dtype* — the deviation itself —
was not asserted anywhere until 2026-08-25, since a `float16` return is finite too.

### 9.6 The projection normalizes through a float32 reduction

`keras.ops.normalize` guards with `max(norm, backend.epsilon())`, and `backend.epsilon()`
(1e-7) underflows at half precision — so the guard is silently absent in exactly the dtype
where it is needed. `ColBERTProjection` uses a private `_safe_l2_normalize` instead. Also a
measured fix; see [§14.3](#143-two-measured-half-precision-defects).

### 9.7 The codec's bucket derivation is this implementation's own, not a transcription

The reference computes `bucket_cutoffs` / `bucket_weights` in
`colbert/indexing/collection_indexer.py`, which was not consulted when this codec was written.
`ResidualCompressionCodec` therefore uses an **equiprobable-bin quantizer** of its own
derivation: cutoffs at the `j/L` quantiles of the pooled residual distribution, reconstruction
values at the `(j+0.5)/L` quantiles (each cell's conditional median). **No claim of numerical
parity with the official index format is made**, and none should be inferred. The property that
is claimed — and proved — is that the 2-bit partition strictly refines the 1-bit one, so
`err(nbits=2) < err(nbits=1)` follows from the refinement rather than from luck.

### 9.8 `doc_maxlen` is 220

220 is the **current official default** in the reference's `DocSettings`. The 180 sometimes
quoted for ColBERT was not found in any consulted source and is not used here.

### 9.9 No pretrained weights, for either model or backbone

Restated because it is the deviation that matters most: `ColBERT.from_variant(pretrained=True)`
raises `NotImplementedError`, and `BERT.from_variant(pretrained=True)` raises underneath it.
**Any number this package's trainers produce is a wiring result, never a retrieval-quality
claim.** Retrieval benchmarking (MS MARCO nDCG / MRR, BEIR) is not merely out of scope here —
it is structurally impossible, because there is neither a pretrained backbone nor an IR
dataset in this repository.

---

## 10. Training

Two trainers live under `src/train/language/colbert/`:

| Entry point | Recipe | Loss |
|---|---|---|
| `train_colbert_v1.py` | pairwise/listwise softmax CE over `<q, d+, d-...>` tuples, positive first | `ColBERTPairwiseSoftmaxLoss` |
| `train_colbert_v2.py` | cross-encoder KL distillation over `nway` tuples | `ColBERTDistillationLoss` |

**`train_colbert_v2.py` does NOT implement in-batch negatives.** §3 and the model docstring
describe the ColBERTv2 *paper*, whose v2 supervision is distillation **plus** in-batch negatives;
this repository implements only the distillation half. Every negative any ColBERT trainer here
sees is an explicit member of its own `<query, positive, negatives...>` tuple — no other
example's documents enter a group's loss. Measured 2026-08-25: `grep -rni 'in.\?batch'
src/train/language/colbert/` returns **0**. The row above is silent about it, which is why this
sentence exists: the gap has to be stated, not inferred from an absence.

```bash
MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m train.language.colbert.train_colbert_v1 --help
```

Both use stock `fit()` with no custom `train_step`; extra signals ride in as inputs. Outputs
land under the repo-root `results/` directory.

**They train on synthetic triples.** No MS MARCO and no IR dataset of any kind exists in this
repository. Combined with the absence of a pretrained backbone, this is why the caveat at the
top of this file is stated as strongly as it is: the trainers demonstrate that the wiring is
correct, and they demonstrate nothing else.

---

## 11. Serialization & Deployment

```python
import keras, numpy as np, os, tempfile
from dl_techniques.models.language.colbert import create_colbert

model = create_colbert("tiny", vocab_size=512, query_maxlen=8, doc_maxlen=16,
                       max_position_embeddings=64)
batch = {
    "query_input_ids": keras.ops.convert_to_tensor(np.zeros((2, 8), "int32") + 3),
    "doc_input_ids": keras.ops.convert_to_tensor(np.zeros((2, 16), "int32") + 5),
}
before = model(batch, training=False)["score"]

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "colbert.keras")
    model.save(path)
    restored = keras.models.load_model(path)      # no custom_objects needed

after = restored(batch, training=False)["score"]
print("max |delta| after round trip:",
      float(np.max(np.abs(np.asarray(before) - np.asarray(after)))))
```

```
max |delta| after round trip: 0.0
```

Every component is registered through `register_dl_technique`
(`dl_techniques.utils.keras_registration`), so `custom_objects` is never required. The
package strings are **not** uniform here, and that is deliberate: `ColBERT` uses its
defining module, `dl_techniques.models.colbert.model>ColBERT`, while `ColBERTProjection` and
`MaxSimScorer` (`components.py`) keep the coarse pre-existing `dl_techniques>ClassName`
string the 2026-08-29 migration left untouched, because re-keying an already-namespaced,
already-unique key would be a checkpoint-affecting change bought for nothing
(repo-root `MIGRATIONS.md`). `build()` materializes the whole sub-layer tree, which
is what makes the reload restore real values rather than fresh random kernels.

The codec is **not** a Keras object and does not travel inside the `.keras` file. Persist it
separately with `codec.save(path)` / `ResidualCompressionCodec.load(path)`.

---

## 12. Testing & Validation

```bash
CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_models/test_colbert/ -q
CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_losses/test_colbert_loss.py -q
```

The suite is guard-heavy by design. There are four test modules, not one file per guard;
each row below names the module and the function inside it, so every path here can be run
verbatim with `pytest <file>::<function>`. Beyond construction, forward pass and
serialization, it pins:

| Guard (`tests/test_models/test_colbert/`) | Claim |
|---|---|
| `test_components.py::test_the_maxsim_matches_the_reference_derived_oracle` | MaxSim agrees with a numpy oracle **derived from the reference formula**, not transcribed from `components.py` |
| `test_components.py::test_a_padded_doc_position_with_a_huge_embedding_cannot_win_the_max` | a masked position with a deliberately huge embedding cannot be selected |
| `test_components.py::test_an_all_masked_document_yields_a_finite_score` | an all-masked document scores finite, in `float32` and `mixed_float16` |
| `test_components.py::test_the_mask_is_applied_before_the_normalize` | the mask multiply precedes L2-normalize, so a filtered row is exactly zero and not renormalized back to unit norm |
| `test_tokenization.py::test_every_query_slot_after_sep_is_mask_and_never_pad` | augmentation applies to queries — every post-`[SEP]` slot is `[MASK]`, never `[PAD]` |
| `test_tokenization.py::test_documents_are_pad_padded_and_never_mask_augmented` | the other direction: documents are never mask-augmented |
| `test_tokenization.py::test_the_punctuation_skiplist_is_documents_only` | the skiplist applies to documents only — both directions |
| `test_tokenization.py::test_the_attend_to_mask_tokens_policy_is_pinned` | the flag's effect on the **emitted** attention mask, both values |
| `test_tokenization.py::test_the_flag_decides_whether_augmented_slots_reach_maxsim` | the flag's **downstream** effect: the augmented rows' post-`encode_query` L2 norms, both values (see §9.3) |
| `test_model.py::test_two_keras_round_trips_preserve_every_weight_value_and_the_output` | values survive save → load → save → load (**twice**: a save-side check cannot see a load-side loss) |
| `test_model.py::test_the_explicit_build_matches_the_lazy_build` | explicit and lazy build produce identical weight signatures, with `count_params() > 0` |
| `test_model.py::test_the_query_and_document_share_one_projection` | the two towers traverse the *same* projection object, asserted with `is` |
| `test_model.py::test_a_skiplisted_document_position_is_zeroed` | a skiplisted document position projects to exactly zero and moves the score |
| `test_model.py::test_a_skiplisted_position_cannot_win_the_max_even_when_it_is_the_best_match` | the adversarial arm: a perfect match planted at a skiplisted position still cannot win |
| `test_model.py::test_a_kept_position_is_untouched_by_a_skiplist_elsewhere` | the skiplist does **not** reach the backbone's attention mask — a kept position's embedding is bit-identical whether or not another position is skiplisted (the reference ordering — see §14.5) |
| `test_model.py::test_a_padded_document_position_cannot_influence_a_real_one` | the other half of the same split: the padding mask **does** reach the backbone — marking trailing positions as padding moves the real positions' embeddings (see §14.5) |
| `test_model.py::test_the_backbone_runs_the_tanh_gelu_approximation` | the backbone's activation is `gelu_tanh` at the live attribute and in its serialized config (§9.4) |
| `test_compression.py::test_two_bits_reconstruct_strictly_better_than_one_bit` | `err(nbits=2) < err(nbits=1)` |
| `test_compression.py::test_decoded_vectors_are_unit_norm` | decode returns unit-norm vectors |
| `test_compression.py::test_no_codec_symbol_is_reachable_from_the_model_or_the_losses` | no codec symbol is reachable from `ColBERT.call` or either loss — the codec is index-time only |

Every guard above was proven RED by injection before being committed. The per-module
injection ledgers — each naming the exact assertion the injection fired — are in the
docstring at the top of each of the four files; injections were applied from a `cp` backup
and restored with `diff -q`, never `git stash`. One test **not** in the table,
`test_model.py::test_the_v1_and_v2_factories_build_the_same_architecture`, cannot currently
fail: `create_colbert_v1` and `create_colbert_v2` have identical bodies (§3.3), so it is a
future-regression tripwire rather than a measurement, and its docstring says so.

---

## 13. Troubleshooting & FAQs

**`ValueError: query_maxlen must be at least 4`** — a query must hold `[CLS] [Q] <token> [SEP]`.
The document minimum is the same shape with `[D]`.

**`ValueError: ... max_position_embeddings`** — `max(query_maxlen, doc_maxlen)` exceeds the
backbone's position table. Raise `max_position_embeddings`, or shorten the sequences.

**`ColBERT.call expects a mapping ...`** — `call()` takes a dict with `query_input_ids` and
`doc_input_ids`, not positional tensors. Use `encode_query` / `encode_document` / `score` if you
want the towers separately.

**My scores are all identical / meaningless.** They are. The weights are random — see the box
at the top of this file. Train the model, and even then read §9.9 before comparing to anything
published.

**Why is `y_true` ignored by the v1 loss?** Because the positive is *positional* (index 0 of
each `nway` group), which is what `labels = zeros` means in the reference. Order your candidate
tuples positive-first or you will train on the wrong target while the loss looks healthy.

**Can I use `keras.losses.KLDivergence` for the v2 objective?** No. It implements
`sum p * log(p/q)` over probabilities and clips its inputs into `[epsilon, 1]`; handing it
log-probabilities returns a small, plausible, wrong number without raising. `ColBERTDistillationLoss`
implements the `log_target=True` kernel directly.

**Does `create_colbert_v2` give me a different architecture?** No — see
[§3.3](#33-why-there-is-one-class-and-not-two). It gives you the recipe and the codec.

---

## 14. Technical Details

### 14.1 Why the mask multiply precedes the normalize

L2 normalization maps any non-zero vector to the unit sphere. Normalizing first would give a
padded position a unit-norm embedding pointing somewhere arbitrary, and the max-reduce over the
document axis could then select it — a padding token winning a query term's best match. Masking
first makes the position exactly zero and keeps it zero.

### 14.2 Why the sentinel is finite

An all-masked document (entirely padding, or entirely punctuation) reduces over a row that
contains nothing but the sentinel. With `-inf` that row's max is `-inf`, and the sum, and then
any downstream softmax, is `NaN`. A finite large-negative constant makes the score a
deterministic `query_len * mask_value` instead. This repository has fixed the `-inf` version of
this bug ten times in `layers/attention/common.py`; it is not a hypothetical.

### 14.3 Two measured half-precision defects

Both were found by execution during implementation, not anticipated:

1. **`keras.ops.normalize` returns `NaN` on a zero row under `mixed_float16`.** Its guard is
   `max(norm, backend.epsilon())`, and `backend.epsilon()` = 1e-7 underflows in binary16. Since
   the mask multiply runs first, zero rows are the ordinary case, and one `NaN` row poisons
   every score in its batch. Fixed by reducing in `float32` and flooring the squared norm.
2. **The MaxSim sum overflows binary16 at ColBERT's own `query_maxlen=32`.** A fully-masked
   document sums to `32 * -1e4 = -3.2e5`; binary16's finite maximum is 65504. Clamping the
   sentinel to `-65504` was tried first and did **not** fix it — two clamped terms already
   overflow. Fixed by promoting the masking and both reductions to `float32`.

### 14.4 Why the output structure is fixed

`.predict()` concatenates per-batch outputs and cannot align a slot that exists in one batch
and not another. An output structure that depends on which optional inputs were supplied
therefore works under `model(inputs)` and breaks under `.predict()` — which is exactly why that
defect ships unnoticed. All optional inputs are resolved to concrete tensors before the return,
so the returned dict is unconditional.

### 14.5 The document path carries TWO masks, and they must stay separate

`ColBERT._encode` takes a padding `attention_mask` and a `participation_mask`, and they are
distinct tensors on the document path:

* the **padding mask** — and nothing else — is what `self.encoder` receives as its
  `attention_mask`, so a punctuation token stays fully visible as attention *context* to its
  neighbours;
* the **participation mask** (`attention_mask * skiplist_mask`) drives the projection's mask
  multiply and the MaxSim candidate set, so a punctuation token contributes exactly zero to
  the score.

This is the reference's ordering: `doc()` passes the plain padding mask to `self.bert` and
multiplies the skiplist onto the *projected, pre-normalization* embeddings.

Iteration 1 originally collapsed the two into one tensor and passed it to both, which fed the
skiplist to the backbone. The reproducible fact is a **property, not a sample**: under the
collapsed ordering a kept position's embedding moves when a *different* position is
skiplisted; under the split ordering it does not move at all. MEASURED over **40 seeded random
inits** (`keras.utils.set_random_seed(0..39)`, 2-layer `tiny`, 10 document positions, skiplist
zeroing positions 3 and 7), comparing the **kept**, non-punctuation positions' embeddings with
and without the skiplist:

| Ordering | kept-position `max abs(delta)`, 40 seeds |
|---|---|
| collapsed (one mask for both roles) | min `0.00035694`, median `0.00094578`, max `0.00241351` — **never zero** |
| split (current, matches the reference) | `0.0` exactly at **all 40 seeds** |

The magnitude is a property of one initialization, not of the code — quoting a single draw
(iteration 1 quoted `0.0010412931`, an adversarial reviewer measured `0.0026180223` and then a
40-seed range of `0.00033`–`0.00329` on its own protocol) tells a reader nothing reproducible,
which is why the range and the exact `0.0` carry the claim here. The figures above are this
package's own re-derivation and differ slightly from the reviewer's; the seeding protocol, not
the quantity, is what differs. The effect grows with depth and with a trained backbone.

Neither of the two skiplist guards that existed at the time could see it — both assert only
that the *filtered* position is zero, and both pass under either ordering, which is exactly why
the collapse shipped. `test_model.py::test_a_kept_position_is_untouched_by_a_skiplist_elsewhere`
pins the axis they are blind to. Do not re-collapse the masks; the anchor at the split site says
the same thing.

**The other direction is now guarded too.** Until 2026-08-25 nothing asserted that the padding
`attention_mask` reached the backbone *at all*: replacing it with `ones_like(attention_mask)`
— a model attending to padding as if it were content — left the whole colbert directory green.
`test_model.py::test_a_padded_document_position_cannot_influence_a_real_one` closes that half:
marking the trailing positions as padding must MOVE the real prefix positions' contextual
embeddings. Measured over the same 40 seeds, that delta is min `0.00030217`, median
`0.00094898`, max `0.00223164` and never once exactly `0.0`; under the `ones_like` injection it
is exactly `0.0`, which is what the assertion reads.

Note that the *query* path still uses one tensor for both roles — that is §9.3's residual
divergence, and it is a property of the tokenizer emitting a single mask, not of `_encode`.

---

## 15. Citation

```bibtex
@inproceedings{khattab2020colbert,
  title     = {ColBERT: Efficient and Effective Passage Search via Contextualized
               Late Interaction over BERT},
  author    = {Khattab, Omar and Zaharia, Matei},
  booktitle = {SIGIR},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.12832}
}

@inproceedings{santhanam2022colbertv2,
  title     = {ColBERTv2: Effective and Efficient Retrieval via Lightweight
               Late Interaction},
  author    = {Santhanam, Keshav and Khattab, Omar and Saad-Falcon, Jon and
               Potts, Christopher and Zaharia, Matei},
  booktitle = {NAACL},
  year      = {2022},
  url       = {https://arxiv.org/abs/2112.01488}
}

@article{devlin2018bert,
  title   = {BERT: Pre-training of Deep Bidirectional Transformers for Language
             Understanding},
  author  = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal = {arXiv preprint arXiv:1810.04805},
  year    = {2018},
  url     = {https://arxiv.org/abs/1810.04805}
}
```

Reference implementation: <https://github.com/stanford-futuredata/ColBERT>

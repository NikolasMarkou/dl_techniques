# ColBERT: Efficient and Effective Passage Search via Late Interaction

A Keras 3 implementation of **ColBERT v1** (Khattab and Zaharia, 2020) and **ColBERT v2**
(Santhanam et al., 2021): a BERT backbone, a bias-free 128-dimensional projection, L2
normalization, and MaxSim late interaction, packaged as one shared encoder with two
training recipes and one v2-only index-time codec.

---

> ### Read this first
>
> **No pretrained weights exist**, for ColBERT or for the BERT backbone.
> `ColBERT.from_variant(pretrained=True)` raises `NotImplementedError`, and so does
> `BERT.from_variant(pretrained=True)` underneath it; a **string** path to a local `.keras`
> checkpoint is the only working route. Every number this package's trainers produce is a
> wiring result, never a retrieval-quality claim, so do not compare any output of this
> package to a published MS MARCO or BEIR figure.
>
> **v1 and v2 build the same network.** A fact about the reference implementation, not a
> shortcut taken here. See § 3.3 and § 9.

---

## 1. Overview

ColBERT keeps **one vector per token** instead of one per passage, and scores a
(query, document) pair by letting each query term take its single best match:

```
S(q, d) = sum_i  max_j  E_q[i] . E_d[j]^T
```

The two texts are encoded **independently**, so every document embedding is computed once,
offline, and stored. Query time is just that similarity between two ready matrices.

Four properties, each pinned by a test: both paths traverse the *same* `ColBERTProjection`
instance (asserted with `is`); masked document positions get a **finite** large-negative
sentinel, never `-inf`; the scorer and projection are `mixed_float16`-safe by measured fix
(§ 14.1); and `call()` always returns the same three keys.

---

## 2. The Problem ColBERT Solves

**The cost of a single vector.** Take the query *"nikola tesla wardenclyffe tower"* against a
200-word biography mentioning Wardenclyffe once. Mean-pooled into 768 numbers, that mention is
roughly 0.5% of the representation, and no re-ranking recovers what the encoder averaged away.

**The cost of full interaction.** A cross-encoder scores that pair perfectly, because the
query term attends directly to the document term, but it needs one transformer forward pass
per pair: 10 million per query on a 10-million-document corpus. Hence re-rankers over a
shortlist, never first-stage retrievers.

**Late interaction** moves the interaction out of the network and into the score. What
survives to scoring time is a matrix of per-token vectors, so the query term still finds its
document term, for one matrix product whose document side was computed months ago.

---

## 3. How ColBERT Works: Core Concepts

### 3.1 The high-level architecture

```
query text    -> ColBERTTokenizer -> [CLS] [Q] tokens [SEP] [MASK] ... [MASK]
document text -> ColBERTTokenizer -> [CLS] [D] tokens [SEP] [PAD] ...
     -> BERT encoder (shared weights) -> ColBERTProjection (shared INSTANCE):
        Dense(128, bias=False) -> x mask -> L2 normalize
     -> MaxSimScorer: (Q x D) -> sentinel-mask -> max over D -> sum over Q
     -> score, shape (batch,)
```

### 3.2 The three asymmetries

Queries and documents traverse **identical weights**. They differ only in preparation, and
each difference is a rule, not a convention:

| | Query | Document |
|---|---|---|
| Marker token | `[Q]` at position 1 | `[D]` at position 1 |
| Padding | overwritten with `[MASK]`, and those slots **participate** in scoring (query augmentation) | ordinary `[PAD]`, masked out |
| Punctuation skiplist | **never applied**; the reference passes an empty skiplist here | applied; a punctuation position is zeroed exactly like padding |

Get either direction wrong and the model still runs, serializes and trains. Query
augmentation is the v1 paper's headline mechanism, and applying the document skiplist to a
query silently deletes real query terms. Both directions are asserted by tests.

### 3.3 Why there is one class and not two

**ColBERT v2 did not change the network.** The official
[`stanford-futuredata/ColBERT`](https://github.com/stanford-futuredata/ColBERT) repository
ships a single `colbert/modeling/colbert.py` for both papers, with no v1-only code path: v1
behaviour is v2's code with `use_ib_negatives=False`, `nway=2`, no distillation scores and no
residual compression at indexing time.

What v2 changed is the **supervision** (cross-encoder KL distillation over `nway` candidates
with in-batch negatives, replacing v1's pairwise softmax cross-entropy over a triple) and the
**index** (a residual codec, in no forward pass and no loss). So this package exports **one**
`ColBERT` class and two genuine factories that build the same architecture; a test asserts
their weight signatures are identical.

---

## 4. Architecture Deep Dive

### 4.1 `ColBERTProjection`

`Dense(dim, use_bias=False)` with **no activation**, then a mask multiply, then L2
normalize, in that order. The order is load-bearing: L2 normalization maps any non-zero
vector to the unit sphere, so masking afterwards would leave a padded position with a
unit-norm embedding pointing somewhere arbitrary, which the max-reduce could then select.
Masking first makes the position exactly zero.

Normalization runs through a module-private `_safe_l2_normalize` which reduces in `float32`
and floors the squared norm, because `keras.ops.normalize` returns `NaN` on an all-zero row
under `mixed_float16`, and after the mask multiply all-zero rows are the *ordinary* case.

### 4.2 `MaxSimScorer`

Builds the **full dense** `(batch, query_len, doc_len)` score matrix, writes a finite
negative sentinel over masked document positions, takes the max over the document axis and
sums over the query axis. Never index exclusion: a variable-length gather would break the
static shapes the rest of the graph depends on. The sentinel is finite because an all-masked
document reduces over a row containing nothing else, and with `-inf` that row's max, the
sum, and any downstream softmax are `NaN`; a finite constant makes the score a deterministic
`query_len * mask_value`.

### 4.3 `ColBERTTokenizer`

A plain Python class, deliberately **not** a `keras.Layer`: everything it does is host-side
`str` to `int` work that runs once, outside the compute graph. It wraps `TiktokenPreprocessor`
and owns the `[Q]`/`[D]` markers, query `[MASK]` augmentation and the punctuation skiplist.

### 4.4 `ResidualCompressionCodec`

v2, **index-time only**. k-means centroids over a sample of the index; each vector is stored
as its nearest centroid's id plus its residual quantized to `nbits` in `{1, 2}` per
dimension. `decode()` re-L2-normalizes, a detail present in the reference *code* and absent
from the paper. A structural test asserts no symbol from `compression.py` is reachable from
`ColBERT.call` or either loss.

---

## 5. Quick Start Guide

```python
from dl_techniques.models.language.colbert import ColBERTTokenizer, create_colbert_v1

tokenizer = ColBERTTokenizer(query_maxlen=8, doc_maxlen=16)
model = create_colbert_v1("tiny", vocab_size=100277,          # cl100k_base
                          query_maxlen=8, doc_maxlen=16,
                          max_position_embeddings=64)

queries = tokenizer.tokenize_queries(["what is late interaction"])
docs = tokenizer.tokenize_documents(["Late interaction, explained: MaxSim."])

outputs = model({"query_input_ids": queries["input_ids"],
                 "query_attention_mask": queries["attention_mask"],
                 "doc_input_ids": docs["input_ids"],
                 "doc_attention_mask": docs["attention_mask"],
                 "doc_skiplist_mask": docs["skiplist_mask"]})

print(sorted(outputs))          # ['doc_embeddings', 'query_embeddings', 'score']
print(outputs["score"].shape, outputs["query_embeddings"].shape,
      outputs["doc_embeddings"].shape)      # (1,) (1, 8, 128) (1, 16, 128)
```

The score is a random-init number and means nothing, which is the point of the box at the
top of this file. The output key set is **fixed**: omit `doc_skiplist_mask` and you still get
all three keys. That matters because `.predict()` concatenates per-batch outputs and cannot
align a slot present in one batch and absent from another.

---

## 6. Component Reference

### 6.1 `ColBERT`

| Member | |
|---|---|
| `ColBERT(vocab_size=..., dim=128, query_maxlen=32, doc_maxlen=220, ...)` | Direct construction |
| `ColBERT.from_variant(variant, pretrained=False, **overrides)` | `pretrained=True` raises `NotImplementedError`; a **string** path loads a local `.keras` checkpoint |
| `ColBERT.MODEL_VARIANTS` | `"large"`, `"base"`, `"small"`, `"tiny"` (§ 7) |
| `model.encode_query(inputs)` / `model.encode_document(inputs)` | `(batch, len, dim)`. Only the document side takes a `skiplist_mask` |
| `model.score(q_emb, d_emb, doc_mask=None, query_mask=None)` | `(batch,)` over stored embeddings |
| `model(inputs)` | `{"score", "query_embeddings", "doc_embeddings"}` |
| `model.encoder` / `.projection` / `.scorer` | The three owned sub-components |

### 6.2 Factories

`create_colbert`, `create_colbert_v1` and `create_colbert_v2` all take `(variant="base",
...)` and delegate to `ColBERT.from_variant`. v1 pairs with `ColBERTPairwiseSoftmaxLoss` and
`train_colbert_v1.py`; v2 with `ColBERTDistillationLoss`, `train_colbert_v2.py` and
`ResidualCompressionCodec`.

### 6.3 `ColBERTTokenizer`

`tokenize_queries(texts)` returns `{"input_ids", "attention_mask"}` padded to `query_maxlen`
with `[MASK]`; `tokenize_documents(texts)` returns those plus `"skiplist_mask"`, padded to
`doc_maxlen`. `punctuation_token_ids` is the skiplist as a `frozenset` (64 ids on
`cl100k_base`; § 9.2), and `query_marker_token_id` / `doc_marker_token_id` are derived from
the live `n_vocab`. `get_config()` / `from_config()` round trip every constructor argument.

### 6.4 `ResidualCompressionCodec`

`fit(vectors)` returns `self`; `encode(vectors)` returns
`(codes: int32 (n,), packed: uint8 (n, bytes_per_vector))`; `decode(codes, packed)` returns an
`(n, dim)` re-L2-normalized reconstruction; `reconstruction_error(vectors)` is the mean
round-trip L2 error, strictly lower at `nbits=2` than at `nbits=1` (`nbits=3` raises). Persist
with `save(path)` / `ResidualCompressionCodec.load(path)`.

### 6.5 Losses (in `dl_techniques.losses`)

| Class | Recipe |
|---|---|
| `ColBERTPairwiseSoftmaxLoss(nway=2)` | v1. `logsumexp_k s[b,k] - s[b,0]`. **The positive is positional; `y_true` is not consulted.** Order tuples positive-first |
| `ColBERTDistillationLoss(nway=64, distillation_alpha=1.0)` | v2. `KL(log_softmax(alpha * teacher) \|\| log_softmax(student))`, `log_target=True` semantics, `batchmean` reduction |

---

## 7. Configuration & Model Variants

The variant table mixes **two sources**, labelled separately rather than presented as one
citation. `dim=128`, `query_maxlen=32` and `doc_maxlen=220` come from the reference's
`colbert/infra/config/settings.py` and are the same in every row, since ColBERT does not
scale them with the backbone. The backbone ladder is **this repository's**, copied from
`BERT.MODEL_VARIANTS`; **published ColBERT uses `bert-base` only**.

| Variant | Backbone: hidden / layers / heads / FFN | Note |
|---|---|---|
| `tiny` | 256 / 4 / 4 / 1024 | tests and edge; not a published size |
| `small` | 512 / 6 / 8 / 2048 | tight budgets; not a published size |
| `base` | 768 / 12 / 12 / 3072 | **the published configuration's backbone size** |
| `large` | 1024 / 24 / 16 / 4096 | no ColBERT paper reports this size |

Overridable through any factory: `vocab_size` (match your tokenizer's encoding), `dim`,
`doc_maxlen`, `query_maxlen`, `mask_value` (the MaxSim sentinel).
`max_position_embeddings` must be at least `max(query_maxlen, doc_maxlen)`, or the
constructor raises `ValueError`.

---

## 8. Comprehensive Usage Examples

Every snippet below was executed; the outputs are the real ones.

### Example 1: the query/document asymmetry, visible

```python
from dl_techniques.models.language.colbert import ColBERTTokenizer

tok = ColBERTTokenizer(query_maxlen=8, doc_maxlen=16)
q = tok.tokenize_queries(["late interaction"])
d = tok.tokenize_documents(["Late interaction, explained."])

print(tok.query_marker_token_id, tok.doc_marker_token_id)
# 100261 100262
print(q["input_ids"][0].tolist(), q["attention_mask"][0].tolist())
# [100257, 100261, 5185, 16628, 100258, 100259, 100259, 100259] [1,1,1,1,1,1,1,1]
print(len(tok.punctuation_token_ids), d["skiplist_mask"][0].tolist())
# 64 [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

`[CLS]=100257`, `[Q]=100261`, two content tokens, `[SEP]=100258`, then three
`[MASK]=100259`: that is query augmentation, and the mask is all ones, so those slots
participate in scoring. The document's `skiplist_mask` zeros are the comma and the period.

### Example 2: offline indexing, encode once and score later

```python
import numpy as np
from dl_techniques.models.language.colbert import create_colbert_v2

model = create_colbert_v2("tiny", vocab_size=100277, query_maxlen=8, doc_maxlen=16,
                          max_position_embeddings=64)

# indexing time: documents only, done once, stored
doc_emb = model.encode_document({"input_ids": d["input_ids"],
                                 "attention_mask": d["attention_mask"],
                                 "skiplist_mask": d["skiplist_mask"]})
# query time: encode the query, score against the stored matrices
qry_emb = model.encode_query({"input_ids": q["input_ids"],
                              "attention_mask": q["attention_mask"]})
participation = d["attention_mask"] * d["skiplist_mask"]
score = model.score(qry_emb, doc_emb, doc_mask=participation,
                    query_mask=q["attention_mask"])
print(doc_emb.shape, qry_emb.shape, score.shape)  # (1, 16, 128) (1, 8, 128) (1,)
print(float(np.abs(np.asarray(doc_emb)[0][np.asarray(participation)[0] == 0]).max()))
# 0.0 -- the mask-before-normalize invariant of section 4.1, observed not asserted
```

---

## 9. Deviations from the reference

Every item here is a place where this implementation knowingly differs from
`stanford-futuredata/ColBERT`. Eight of the nine are pinned by a named test; § 9.7's
provenance sentence is documentation-only by construction.

### 9.1 The `[Q]` / `[D]` markers come from free Tiktoken slots

The reference implements the markers as BERT WordPiece's `[unused0]` / `[unused1]` slots.
There is no WordPiece tokenizer in this library; the NLP stack is Tiktoken `cl100k_base`.
`ColBERTTokenizer` allocates the markers from free slots at the top of that vocabulary,
derived from the live `n_vocab` (`n_vocab - 16`, `n_vocab - 15`; 100261 and 100262 today)
rather than hardcoded, and raises `ValueError` on a collision with a reserved special. A
hardcoded literal would silently land on a real content token if the encoding revision ever
shifted `n_vocab`: perfectly shaped tensors, wrong retrieval forever. **Cost**: token-id
parity with published ColBERT checkpoints is permanently forfeited.

### 9.2 The punctuation skiplist covers bare and space-prefixed forms: 64 ids, not 32

The reference builds it as `{encode(symbol)[0] for symbol in string.punctuation}`, 32 ids.
Here it takes the first id of **both** `encode(symbol)` and `encode(" " + symbol)`.

**Why**: the reference's rule is inert under BPE. On `cl100k_base` the bare `","` is id 11,
but ordinary prose (`"late interaction, explained."`) tokenizes the same comma as `" ,"`,
id 1174. All 32 bare forms are ids 0..93, single-byte tokens that essentially never appear
in prose, so a literal port builds a skiplist matching almost nothing real. The two sets are
disjoint and every space-prefixed form is exactly one token, so the widening adds no
ambiguity and swallows no content token. **Cost**: a skiplist larger than the reference's;
id parity was already forfeited by § 9.1.

### 9.3 `attend_to_mask_tokens` defaults to `True` here, not `False`

The reference's `QuerySettings.attend_to_mask_tokens` defaults to `False`, zeroing the
*transformer attention* mask at the augmented `[MASK]` slots. Here the default is `True`.

**Why**: the reference carries **two** masks over a query. The attention mask is governed by
that flag; a separate *participation* mask is multiplied onto the projected embeddings before
L2-normalize and is computed as `x != pad_token_id` **after** the pads have been overwritten
with `[MASK]`, hence all-ones regardless of the flag. Augmented `[MASK]` embeddings always
participate in MaxSim in the reference; that is the entire query augmentation mechanism. This
tokenizer emits **one** `attention_mask`, used for both purposes, so copying the reference
default would multiply every augmented `[MASK]` embedding by zero, killing the paper's
headline mechanism while every shape, dtype and round-trip test stayed green.

**Cost**: under the default the `[MASK]` positions are additionally visible as attention keys
to the real query terms, relative to the reference, which hides them.

**`attend_to_mask_tokens=False` is not an escape hatch back to the reference**, because the
query path derives its participation mask from the same emitted tensor: `False` also deletes
the augmented `[MASK]` slots from MaxSim entirely, the very outcome the `True` default exists
to prevent. Measured on `"late interaction"` at `query_maxlen=8` (5 content, 3 augmented),
`False` gives an emitted mask and post-`encode_query` L2 norms of both `[1,1,1,1,1,0,0,0]`,
against all-ones under the default. The document path splits the two roles (§ 14.2).

### 9.4 The BERT backbone runs `gelu_tanh`, not exact GELU

This library's `BERT` defaults to the tanh approximation of GELU rather than the exact `erf`
form, and the default is kept, since bit-for-bit reproduction of published ColBERT numbers is
already impossible for the reasons above. Pinned at `ColBERT(...).encoder.hidden_act` **and**
at the backbone's serialized config, so a reloaded model running a different non-linearity is
caught too.

### 9.5 `MaxSimScorer` returns `float32` under `mixed_float16`

Under a `mixed_float16` policy the scorer promotes its masking and both reductions to
`float32` and returns `float32`, an output-dtype asymmetry with the projection beside it. This
is a measured fix, not a precaution (§ 14.1). The *finiteness* half was pinned from the start;
the *return dtype*, the deviation itself, needs its own assertion, since a `float16` return is
finite too.

### 9.6 The projection normalizes through a float32 reduction

`keras.ops.normalize` guards with `max(norm, backend.epsilon())`, and `backend.epsilon()`
(1e-7) underflows at half precision, so the guard is silently absent in exactly the dtype
where it is needed; `ColBERTProjection` uses a private `_safe_l2_normalize` instead.

### 9.7 The codec's bucket derivation is this implementation's own

The reference computes `bucket_cutoffs` / `bucket_weights` in
`colbert/indexing/collection_indexer.py`, which was not consulted when this codec was
written. `ResidualCompressionCodec` uses an **equiprobable-bin quantizer** of its own
derivation: cutoffs at the `j/L` quantiles of the pooled residual distribution,
reconstruction values at the `(j+0.5)/L` quantiles. **No claim of numerical parity with the
official index format is made.** What is claimed, and proved, is that the 2-bit partition
strictly refines the 1-bit one, so `err(nbits=2) < err(nbits=1)` follows.

### 9.8 `doc_maxlen` is 220

220 is the current official default in the reference's `DocSettings`; the 180 sometimes quoted
for ColBERT was not found in any consulted source and is not used here.

### 9.9 No pretrained weights, for either model or backbone

Restated because it matters most: retrieval benchmarking (MS MARCO nDCG / MRR, BEIR) is
structurally impossible here, since there is neither a pretrained backbone nor an IR dataset
in this repository.

---

## 10. Training

Two trainers live under `src/train/language/colbert/`: `train_colbert_v1.py` (pairwise or
listwise softmax CE over `<q, d+, d-...>` tuples, positive first,
`ColBERTPairwiseSoftmaxLoss`) and `train_colbert_v2.py` (cross-encoder KL distillation over
`nway` tuples, `ColBERTDistillationLoss`). Both use stock `fit()` with no custom `train_step`.

```bash
MPLBACKEND=Agg .venv/bin/python -m train.language.colbert.train_colbert_v1 --help
```

**`train_colbert_v2.py` does NOT implement in-batch negatives.** The ColBERTv2 *paper*'s
supervision is distillation **plus** in-batch negatives; only the distillation half is
implemented here, and every negative a trainer sees is an explicit member of its own tuple.
They also train on synthetic triples: no IR dataset exists in this repository.

---

## 11. Serialization & Deployment

`model.save("colbert.keras")` then `keras.models.load_model("colbert.keras")`, with no
`custom_objects` needed. A round trip is value-exact: `max |delta|` between the two models' scores on one batch is
`0.0`. Every component is registered through `register_dl_technique`
(`dl_techniques.models.colbert.model>ColBERT`,
`dl_techniques.models.colbert.components>{ColBERTProjection,MaxSimScorer}`), and `build()`
materializes the whole sub-layer tree, which is what makes the reload restore real values
rather than fresh random kernels. The codec is **not** a Keras object and does not travel
inside the `.keras` file; persist it separately with `codec.save(path)` /
`ResidualCompressionCodec.load(path)`.

---

## 12. Testing & Validation

```bash
CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/test_models/test_colbert/ tests/test_losses/test_colbert_loss.py -q
```

Every guard was proven RED by injection before being committed. Beyond construction, forward
pass and serialization the suite pins: MaxSim against a numpy oracle derived from the
reference *formula*, not transcribed from `components.py`; that a masked position with a huge
embedding cannot win the max and an all-masked document scores finite in `float32` and
`mixed_float16`; that the mask multiply precedes L2-normalize; that augmentation is
queries-only and the skiplist documents-only; both values of `attend_to_mask_tokens` (§ 9.3);
that values survive save, load, save, load twice, because a save-side check cannot see a
load-side loss; that both towers share one projection object; and both halves of § 14.2.

---

## 13. Troubleshooting

- **`ValueError: query_maxlen must be at least 4`** - a query must hold
  `[CLS] [Q] <token> [SEP]`; the document minimum is the same shape with `[D]`.
- **`ValueError: ... max_position_embeddings`** - `max(query_maxlen, doc_maxlen)` exceeds
  the backbone's position table. Raise it, or shorten the sequences.
- **`ColBERT.call expects a mapping ...`** - `call()` takes a dict, not positional tensors.
  Use `encode_query` / `encode_document` / `score` for the towers separately.
- **`y_true` is ignored by the v1 loss** because the positive is *positional*. Order your
  tuples positive-first or you train on the wrong target while the loss looks healthy.
- **Do not use `keras.losses.KLDivergence` for the v2 objective.** It works over
  probabilities and clips into `[epsilon, 1]`; log-probabilities give a wrong number, quietly.

---

## 14. Technical Details

### 14.1 Two measured half-precision defects

Both were found by execution during implementation, not anticipated:

1. **`keras.ops.normalize` returns `NaN` on a zero row under `mixed_float16`.** Its guard is
   `max(norm, backend.epsilon())`, and `backend.epsilon()` = 1e-7 underflows in binary16.
   Since the mask multiply runs first, zero rows are the ordinary case, and one `NaN` row
   poisons every score in its batch. Fixed by reducing in `float32`.
2. **The MaxSim sum overflows binary16 at ColBERT's own `query_maxlen=32`.** A fully-masked
   document sums to `32 * -1e4 = -3.2e5`; binary16's maximum is 65504. Clamping the sentinel
   to `-65504` did **not** fix it (two clamped terms already overflow); the fix is to promote
   the masking and both reductions to `float32`.

### 14.2 The document path carries TWO masks, and they must stay separate

`ColBERT._encode` takes a padding `attention_mask` and a `participation_mask`, distinct
tensors on the document path. The **padding mask**, and nothing else, reaches `self.encoder`,
so a punctuation token stays visible as attention *context* to its neighbours; the
**participation mask** (`attention_mask * skiplist_mask`) drives the projection's mask multiply
and the MaxSim candidate set, so it contributes exactly zero to the score. This is the
reference's ordering.

Collapsing the two into one tensor feeds the skiplist to the backbone, and the difference is
a **property, not a sample**: collapsed, a kept position's embedding moves when a *different*
position is skiplisted; split, it does not move at all. Over 40 seeded random inits the
kept-position `max abs(delta)` is never zero collapsed (median around 9e-4) and exactly `0.0`
at all 40 seeds split. Do not re-collapse. The *query* path still uses one tensor for both
roles: § 9.3's residual divergence, a property of the tokenizer, not of `_encode`.

---

## 15. References

- Khattab and Zaharia, 2020. *ColBERT: Efficient and Effective Passage Search via
  Contextualized Late Interaction over BERT.* SIGIR. https://arxiv.org/abs/2004.12832
- Santhanam et al., 2022. *ColBERTv2: Effective and Efficient Retrieval via Lightweight
  Late Interaction.* NAACL. https://arxiv.org/abs/2112.01488
- Devlin et al., 2018. *BERT: Pre-training of Deep Bidirectional Transformers.*
  https://arxiv.org/abs/1810.04805
- Reference implementation: <https://github.com/stanford-futuredata/ColBERT>

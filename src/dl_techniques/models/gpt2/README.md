# GPT-2: Generative Pre-trained Transformer 2

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

A Keras 3 implementation of the **GPT-2** decoder-only language model, based on
*Language Models are Unsupervised Multitask Learners* (Radford et al., 2019).

The class is a deliberately **thin wrapper**: the whole transformer stack —
token and positional embeddings, embedding norm and dropout, causal
self-attention blocks, and the final normalization — lives in the library's
shared `TextDecoder` layer, which is itself built out of the library's
factories. `GPT2` adds the language-modeling head on top and nothing else.

> ⚠️ **No public pretrained weights are distributed by this library.** Calls
> like `GPT2.from_variant("small", pretrained=True)` raise
> `NotImplementedError` — deliberately, so that a caller who asks for trained
> weights never silently receives an untrained model. To load weights, pass a
> local file path (`pretrained="path/to/checkpoint.keras"`). To get a
> random-initialized model, omit `pretrained` (or pass `pretrained=False`).
> Train your own with `src/train/gpt2/pretrain.py` (see §11).

---

## Table of Contents

1. [Overview: What GPT-2 is and Why It Matters](#1-overview-what-gpt-2-is-and-why-it-matters)
2. [The Problem GPT-2 Solves](#2-the-problem-gpt-2-solves)
3. [How GPT-2 Works: Core Concepts](#3-how-gpt-2-works-core-concepts)
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

## 1. Overview: What GPT-2 is and Why It Matters

### What is GPT-2?

GPT-2 is a **decoder-only autoregressive transformer**. It reads a sequence of
token IDs and, at every position, predicts the next token — using only the
tokens to its left. Trained at scale on unlabeled text, that single objective
produces a model that can continue prose, answer questions, and be fine-tuned
onto downstream tasks without any architectural change.

### What this implementation is

1. **A thin, correct wrapper.** `GPT2` builds one sub-layer: `TextDecoder`
   (`dl_techniques/layers/transformers/text_decoder.py`). When
   `tie_word_embeddings=False` it also builds a single untied `Dense`
   projection. That is the entire model surface — see §15 for why nothing else
   is hand-rolled here.
2. **Reuse all the way down.** `TextDecoder` composes `TransformerLayer`,
   `create_normalization_layer`, and the shared causal/padding mask builders in
   `dl_techniques.utils.masking`. GPT-2 gets the library's tested components for
   free, and improvements to them land here without a code change.
3. **Keras 3 native and fully serializable.** The class is registered with
   `@keras.saving.register_keras_serializable()`, `get_config()` round-trips
   all twelve constructor arguments, and a `.keras` save/load needs no
   `custom_objects` (§12).
4. **Configurable attention and FFN.** `attention_type` and `ffn_type` are
   passed straight through to `TextDecoder`'s factories, so the classic recipe
   is a default, not a hard-coding.

### Why it matters here

This package is the reference decoder-only LM of the repo. Its sibling
`models/wave_field/` deliberately mirrors its public surface so that both slot
into the same training pipeline, and `src/train/gpt2/` is the CLM pre-training
scaffold that `src/train/wave_field/` reuses through the shared
`ClmPretrainConfig`.

---

## 2. The Problem GPT-2 Solves

### Modeling text without labels

```
┌─────────────────────────────────────────────────────────────┐
│  The pre-training problem                                   │
│                                                             │
│  1. Labeled NLP data is scarce and task-specific.           │
│  2. Raw text is effectively unlimited.                      │
│  3. Next-token prediction turns raw text into supervision:  │
│     every position is its own training example, and the     │
│     label is simply the token that came next.               │
└─────────────────────────────────────────────────────────────┘
```

### Why the mask matters

A model that can see the future while predicting it learns nothing. GPT-2's
answer is a **causal mask**: position `i` may attend to positions `<= i` only.

```
┌─────────────────────────────────────────────────────────────┐
│  Attention visibility (4 tokens)                            │
│                                                             │
│         t0    t1    t2    t3                                │
│   t0 [  ✓     ·     ·     ·  ]                              │
│   t1 [  ✓     ✓     ·     ·  ]     ✓ = may attend           │
│   t2 [  ✓     ✓     ✓     ·  ]     · = masked out           │
│   t3 [  ✓     ✓     ✓     ✓  ]                              │
└─────────────────────────────────────────────────────────────┘
```

In this implementation the mask is **constructed and applied inside
`TextDecoder`** (via `dl_techniques.utils.masking`), not in `gpt2.py`. Any
padding mask you supply is combined with it, never substituted for it.

---

## 3. How GPT-2 Works: Core Concepts

### High-level architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        GPT-2 (this package)                      │
│                                                                  │
│  Input IDs (B, N) ──►┌──────────────────────────────┐            │
│                      │ TextDecoder                  │            │
│                      │  ├ Token + Positional Embed  │            │
│                      │  ├ Embed Norm → Dropout      │            │
│                      │  ├ TransformerLayer × depth  │  pre-norm  │
│                      │  │   (causal attn + FFN)     │            │
│                      │  └ Final LayerNorm           │            │
│                      └──────────────┬───────────────┘            │
│                                     │ (B, N, D)                  │
│                      ┌──────────────▼───────────────┐            │
│                      │ LM head                      │            │
│                      │  tied   → hidden @ E.T       │            │
│                      │  untied → Dense(V, no bias)  │            │
│                      └──────────────┬───────────────┘            │
│                                     ▼                            │
│           {"logits": (B, N, V), "last_hidden_state": (B, N, D)}  │
└──────────────────────────────────────────────────────────────────┘
```

### The data flow

```
STEP 1: INPUT
─────────────
Text ─► tokenizer ─► input_ids (B, N)  [+ optional attention_mask (B, N)]
   The model accepts either a bare tensor of IDs or a dict
   {"input_ids": ..., "attention_mask": ...}. A dict without
   'input_ids' raises ValueError.

STEP 2: DECODE (inside TextDecoder)
───────────────────────────────────
input_ids
   ├─► token embedding + learned positional embedding
   ├─► embedding LayerNorm → dropout
   ├─► depth × [ LayerNorm → causal self-attention → residual
   │             LayerNorm → FFN                   → residual ]   (pre-norm)
   └─► final LayerNorm ─► hidden_states (B, N, D)

STEP 3: PROJECT TO VOCABULARY
─────────────────────────────
hidden_states
   ├─ tie_word_embeddings=True  ─► logits = hidden @ Eᵀ   (no new parameters)
   └─ tie_word_embeddings=False ─► logits = Dense(V, use_bias=False)(hidden)

STEP 4: TRAIN
─────────────
loss = MaskedCausalLMLoss(logits[:, :-1], input_ids[:, 1:])
   -- the shift is done by the caller / data pipeline, not by the model.
```

### Differences from BERT

| | BERT (`models/bert/`) | GPT-2 (here) |
|---|---|---|
| Attention | bidirectional | causal |
| Norm position | post-norm | pre-norm |
| Objective | masked LM | next-token prediction |
| Segment (token type) embeddings | yes | no |
| Output head | task heads | vocabulary projection, tied by default |

---

## 4. Architecture Deep Dive

### 4.1 `TextDecoder` — everything except the head

`GPT2._build_architecture` constructs exactly one `TextDecoder` named
`"decoder"`, configured with `embedding_type="learned"`,
`positional_type="learned"`, `normalization_type="layer_norm"`, and
`normalization_position="pre"`. `attention_type`, `ffn_type`, both dropout
rates, `initializer_range` and `layer_norm_eps` are forwarded from the GPT-2
constructor. Weight paths therefore all live under `decoder/...`.

### 4.2 The LM head

- **Tied (default).** `self.lm_head is None` and `call()` computes
  `ops.matmul(hidden, ops.transpose(self.decoder.word_embeddings.embeddings))`.
  This is the original GPT-2 recipe and saves a `vocab_size × embed_dim`
  parameter block.
- **Untied.** A `keras.layers.Dense(vocab_size, use_bias=False)` named
  `lm_head`, initialized with `TruncatedNormal(stddev=initializer_range)`.
  Untying is the modern preference at multi-billion-parameter scale; the
  kernel is `(D, V)` while the embedding table is `(V, D)`, i.e. genuinely
  separate variables, not a transposed view.

This `Dense` is the **only** layer `gpt2.py` builds directly, and it is
deliberate — see §15, "The factory-adoption audit".

### 4.3 Validation

`_validate_config` (a `@staticmethod`) raises `ValueError` for a non-positive
`vocab_size`, `embed_dim`, `depth` or `num_heads`, for
`embed_dim % num_heads != 0`, and for a dropout rate outside `[0, 1]`. All of
this happens in `__init__`, before any layer is constructed.

---

## 5. Quick Start Guide

### Installation

The model ships with the library; no extra dependency is required beyond the
repo's own environment (`.venv`, Keras 3 / TensorFlow 2.18).

### Your first GPT-2 model

```python
import numpy as np
from dl_techniques.models.gpt2 import create_gpt2

# A small, fast-to-build configuration. Swap "tiny" for "small"/"medium"/
# "large"/"xl" once you have the compute for them.
model = create_gpt2("tiny", vocab_size=512)

input_ids = np.random.randint(0, 512, size=(2, 64)).astype("int32")
outputs = model(input_ids)

print(outputs["logits"].shape)             # (2, 64, 512)
print(outputs["last_hidden_state"].shape)  # (2, 64, 256)
```

---

## 6. Component Reference

### 6.1 `GPT2` (model class)

The `keras.Model` subclass. Returns a **dict** with `logits` and
`last_hidden_state` — never a bare tensor — so that a dict-keyed
`compile(loss={"logits": ...})` works unchanged.

```python
from dl_techniques.models.gpt2 import GPT2

model = GPT2.from_variant("small")                       # named variant
model = GPT2.from_variant("tiny", dropout_rate=0.1)      # variant + override
model = GPT2(vocab_size=50257, embed_dim=512, depth=6, num_heads=8)
```

Key methods: `from_variant` (classmethod), `call`, `compute_output_shape`,
`get_config`, and the static `_download_weights` (always raises — see the
banner at the top).

### 6.2 `create_gpt2(...)` (module-level factory)

```python
from dl_techniques.models.gpt2 import create_gpt2

model = create_gpt2("small")                 # variant defaults to "small"
model = create_gpt2("tiny", vocab_size=200)  # vocab override
```

A thin wrapper over `GPT2.from_variant` that mirrors `create_bert` /
`create_resnet`. `vocab_size` is injected into `kwargs` only when it is not
`None`; everything else is forwarded verbatim.

### 6.3 Public API

`dl_techniques.models.gpt2` exports exactly `GPT2` and `create_gpt2` via
`__all__`. That surface is pinned by a test — see §13.

---

## 7. Configuration & Model Variants

`GPT2.MODEL_VARIANTS` (each entry also carries a human-readable
`"description"`, which `from_variant` pops before construction):

| Variant | `embed_dim` | `depth` | `num_heads` | `max_seq_len` |
|:---:|:---:|:---:|:---:|:---:|
| `tiny` | 256 | 4 | 4 | 512 |
| `small` | 768 | 12 | 12 | 1024 |
| `medium` | 1024 | 24 | 16 | 1024 |
| `large` | 1280 | 36 | 20 | 1024 |
| `xl` | 1600 | 48 | 25 | 1024 |

Note the variants set architecture only — `vocab_size` is **not** part of a
variant and always falls back to the class default `DEFAULT_VOCAB_SIZE`
(Tiktoken `cl100k_base`) unless you override it.

Constructor arguments not covered by a variant:

| Argument | Default | Notes |
|---|---|---|
| `vocab_size` | `100277` | Tiktoken `cl100k_base`. Must match your tokenizer. |
| `dropout_rate` | `0.0` | Embedding and residual paths. |
| `attention_dropout_rate` | `0.0` | Attention weights. |
| `initializer_range` | `0.02` | `TruncatedNormal` stddev. |
| `layer_norm_eps` | `1e-5` | |
| `attention_type` | `"multi_head"` | Forwarded to `TextDecoder`'s attention factory. |
| `ffn_type` | `"mlp"` | Forwarded to `TextDecoder`'s FFN factory. |
| `tie_word_embeddings` | `True` | See §4.2. |

To enumerate the variants exactly as the code has them, read them from the
class rather than trusting this table:

```python
from dl_techniques.models.gpt2 import GPT2
for name, cfg in GPT2.MODEL_VARIANTS.items():
    print(name, cfg)
```

---

## 8. Comprehensive Usage Examples

### Example 1: dict input with a padding mask

```python
import numpy as np
from dl_techniques.models.gpt2 import GPT2

model = GPT2.from_variant("tiny", vocab_size=512, dropout_rate=0.1)

# Dict input: 'input_ids' is required, 'attention_mask' is optional
# (1 = attend, 0 = padding).
batch = {
    "input_ids": np.random.randint(0, 512, size=(2, 16)).astype("int32"),
    "attention_mask": np.concatenate(
        [np.ones((2, 12), "int32"), np.zeros((2, 4), "int32")], axis=1
    ),
}
outputs = model(batch, training=False)
print(outputs["logits"].shape)  # (2, 16, 512)

# A missing 'input_ids' key is a hard error, not a silent default.
try:
    model({"attention_mask": batch["attention_mask"]})
except ValueError as e:
    print(f"ValueError: {e}")
```

### Example 2: tied vs untied LM head

```python
import numpy as np
from dl_techniques.models.gpt2 import GPT2

tied = GPT2(vocab_size=512, embed_dim=64, depth=2, num_heads=4, max_seq_len=32)
untied = GPT2(
    vocab_size=512, embed_dim=64, depth=2, num_heads=4, max_seq_len=32,
    tie_word_embeddings=False,
)

print(tied.lm_head)          # None -- logits = hidden @ E.T
print(untied.lm_head.name)   # 'lm_head' -- an independent Dense(vocab, no bias)

ids = np.random.randint(0, 512, size=(1, 8)).astype("int32")
print(tied(ids)["logits"].shape, untied(ids)["logits"].shape)
```

### Example 3: the weights contract, in full

```python
from dl_techniques.models.gpt2 import GPT2, create_gpt2

# pretrained=True is a hard error: this library distributes no GPT-2 weights.
for call in (
    lambda: GPT2.from_variant("tiny", pretrained=True),
    lambda: create_gpt2("tiny", pretrained=True),
):
    try:
        call()
    except NotImplementedError as e:
        print(f"NotImplementedError: {e}")

# A path that does not exist is also a hard error.
try:
    GPT2.from_variant("tiny", pretrained="/no/such/checkpoint.keras")
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
```

An existing path is loaded with `skip_mismatch=True` after a dummy forward
pass builds the model, and the load goes through
`utils.weight_transfer.load_weights_or_raise`, which adds a **third** hard error:
it counts variables whose value actually changed and raises when that count is
**zero** — a checkpoint whose names or shapes do not match this model would
otherwise restore nothing and return normally under `skip_mismatch=True`.

Its limit, stated so it is not mistaken for a total guarantee: the guard fires only
on a *completely* empty load. A **partial** restore (say 9 of 30 variables) logs a
warning and returns normally, and `skip_mismatch=True` — hardcoded here — is
precisely the configuration in which a partial restore is possible. Check the
logged restored-variable count against what you expect.

---

## 9. Advanced Usage Patterns

### Pattern 1: train on your own data

The model outputs a dict, so the loss is keyed on `"logits"`. Causal LM targets
are the inputs shifted by one — the **caller** does the shift; the model never
does it for you.

```python
import keras
import numpy as np
from dl_techniques.models.gpt2 import GPT2
from dl_techniques.losses import MaskedCausalLMLoss

model = GPT2(vocab_size=512, embed_dim=64, depth=2, num_heads=4, max_seq_len=32)

# The model returns a dict, so the loss is keyed on the "logits" output.
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=3e-4, clipnorm=1.0),
    loss={"logits": MaskedCausalLMLoss()},
)

# Causal LM targets are the inputs shifted by one position.
ids = np.random.randint(0, 512, size=(8, 33)).astype("int32")
history = model.fit(
    {"input_ids": ids[:, :-1]},
    {"logits": ids[:, 1:]},
    batch_size=4, epochs=1, verbose=0,
)
print(sorted(history.history.keys()))
```

### Pattern 2: GPT-2 as a feature extractor

`last_hidden_state` is the pre-projection representation. Take it and ignore
`logits` when you want features rather than a distribution over the
vocabulary — nothing needs to be disabled, the tied head costs no parameters.

### Pattern 3: text generation

`GPT2` has **no** `generate()` method — sampling is not part of the model. The
repo's sampling loop lives in `src/train/common/generation_probe.py`
(`GenerationProbeCallback`), which the pre-training script wires in as a
callback; `src/dl_techniques/models/power_sampling/` provides
inference-time sampling for any causal LM.

---

## 10. Performance Optimization

### Mixed precision

```python
import keras
import numpy as np
from dl_techniques.models.gpt2 import GPT2

keras.mixed_precision.set_global_policy("mixed_float16")
model = GPT2.from_variant("tiny", vocab_size=512)
out = model(np.random.randint(0, 512, (2, 32)).astype("int32"))
print(out["logits"].dtype)  # float16
keras.mixed_precision.set_global_policy("float32")
```

### Sequence length

Attention cost is quadratic in `max_seq_len`. `max_seq_len` also fixes the
positional table's size, so it is an architectural choice, not a runtime knob —
a model built at 1024 cannot be fed 2048 positions.

### Other levers

- Reduce `batch_size` before reducing `depth`: the activations dominate.
- `tie_word_embeddings=True` (the default) removes a `vocab × embed_dim`
  parameter block, which matters most at large `vocab_size`.
- The trainers pass `clipnorm=1.0` to `AdamW`; keep it when you write your own
  loop.

---

## 11. Training and Best Practices

Three runnable pipelines ship with the repo, under `src/train/gpt2/`:

| Script | Purpose |
|---|---|
| `pretrain.py` | Causal-LM pre-training (TFDS or HuggingFace text sources). |
| `pretrain_so.py` | Same, plus soft-orthonormal regularization on the weight matrices. |
| `finetune.py` | Domain fine-tuning of an existing checkpoint. |

```bash
# Inspect the full flag surface (this is the source of truth, not this table)
MPLBACKEND=Agg .venv/bin/python -m train.gpt2.pretrain --help
MPLBACKEND=Agg .venv/bin/python -m train.gpt2.pretrain_so --help
MPLBACKEND=Agg .venv/bin/python -m train.gpt2.finetune --help

# A short pre-training run on GPU 1
MPLBACKEND=Agg .venv/bin/python -m train.gpt2.pretrain \
    --gpu 1 --variant tiny --dataset-source tfds \
    --dataset-name imdb_reviews --max-samples 64 \
    --epochs 1 --batch-size 2 --max-seq-length 32
```

Notes that matter in practice:

- **The tokenizer defines `vocab_size`.** A mismatch is not an error, it is
  silently wrong output. The class default matches Tiktoken `cl100k_base`; the
  pre-training script chooses its own tokenizer and sizes the model to it.
- **Runs write to repo-root `results/`**, never `src/results/`. Every trainer
  in this package emits `config.json` and `training_history.json` alongside its
  checkpoints.
- **Warmup + AdamW with `clipnorm=1.0`** is the shipped recipe; see
  `train/common/nlp.py` for the schedule builder the scripts use.
- Fine-tuning starts from a local `.keras` checkpoint you produced — there is
  no public one (see the banner).

---

## 12. Serialization & Deployment

Every class involved is registered for Keras serialization, so a `.keras`
round trip needs no `custom_objects`:

```python
import keras
import numpy as np
from dl_techniques.models.gpt2 import GPT2

model = GPT2(vocab_size=512, embed_dim=64, depth=2, num_heads=4, max_seq_len=32)
ids = np.random.randint(0, 512, size=(1, 8)).astype("int32")
before = model(ids)["logits"]

model.save("gpt2_demo.keras")
# No custom_objects needed: every class is registered for serialization.
restored = keras.models.load_model("gpt2_demo.keras")
after = restored(ids)["logits"]

print(float(np.max(np.abs(np.array(before) - np.array(after)))))  # 0.0
```

`get_config()` stores all twelve constructor arguments; `from_config` is
Keras's default, which resolves to `cls(**config)` for this subclassed model.

---

## 13. Testing & Validation

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/test_models/test_gpt2/ -q
```

The suite lives in `tests/test_models/test_gpt2/` (`test_gpt2.py` +
`test_round_trip.py`) and covers initialization and validation errors, forward
shapes for tensor and dict inputs, weight tying **and** the untied head,
causal masking (a future token must not move an earlier position's logits),
`get_config` / `.keras` round trips, every named variant, gradient flow, the
`pretrained=True` error contract, and the exact `__all__` surface.

Trainer-side contract tests for this package live in
`tests/test_train/test_gpt2/`:

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/test_train/test_gpt2/ -q
```

Quote *passed with collected* when you report a result — a suite that collects
almost nothing also reports green.

---

## 14. Troubleshooting & FAQs

**`NotImplementedError: Pretrained GPT-2 weights are not distributed...`**
Working as designed. Pass a local checkpoint path or `pretrained=False`. The
error replaced an older warn-and-return-random-weights path; see the DECISION
comment on `_download_weights` in `gpt2.py`.

**`ValueError: embed_dim (X) must be divisible by num_heads (Y)`**
Raised at construction by `_validate_config`. Pick head counts that divide the
hidden size.

**`ValueError: Dictionary input must contain 'input_ids' key`**
Dict inputs are keyed exactly `input_ids` (required) and `attention_mask`
(optional).

**Out-of-memory.** Reduce `batch_size` first, then `max_seq_len` (quadratic),
then the variant. Mixed precision (§10) is usually the cheapest win.

**Generated text is nonsense.** Check that the tokenizer used at inference is
the one used at training, and that `vocab_size` matches it exactly.

### FAQ

**Q: Why is there no `generate()` on the model?**
A: Sampling is a decoding policy, not architecture. The repo keeps it in
`train/common/generation_probe.py` and `models/power_sampling/` so that a
change of sampler does not touch the model class.

**Q: Should I tie or untie the embeddings?**
A: Tied matches the original GPT-2 and is the default. Untied is the modern
choice at very large scale. Both are tested; switch with one flag.

**Q: Where is the causal mask?**
A: Inside `TextDecoder`, built by `dl_techniques.utils.masking` and combined
with your padding mask. `gpt2.py` contains no masking code at all.

---

## 15. Technical Details

### The factory-adoption audit (measured 2026-08-13)

The repo's layer-reuse policy (`models/CLAUDE.md`) says to reach for a factory
before hand-rolling a layer. This package was audited against it and found to
have **zero adoptable sites**:

- Everything except the head is delegated to `TextDecoder`, which already uses
  `TransformerLayer`, `create_normalization_layer` and the shared mask
  builders. GPT-2 reaches the factories **through** it, at the right depth.
- The one directly-constructed layer is the untied
  `Dense(vocab_size, use_bias=False)` LM head. No factory covers a bare
  vocabulary projection — it is not an attention, FFN, normalization,
  embedding or activation primitive — so there is nothing to adopt.

So `grep -rn "create_.*_layer" src/dl_techniques/models/gpt2/ --include="*.py"`
returns 0 — and that is **not** evidence of a problem, it measures the wrong
thing. The question is whether the model reaches the factories at some level,
and it does, through `TextDecoder`. (Keep the `--include="*.py"`: without it
this very section is counted as a hit.)

### Pre-norm

Each block normalizes *before* its sub-layer and adds the residual after
(`normalization_position="pre"`). This is what GPT-2 switched to, and it is why
a final `LayerNorm` is needed after the last block — without it the residual
stream would leave the stack unnormalized.

### Weight tying and the transpose

With tying, logits are `hidden @ Eᵀ` where `E` is the `(V, D)` token embedding
table — one variable serving two roles. With `tie_word_embeddings=False` the
head kernel is `(D, V)`: a *different* variable of transposed shape, which is
why an accidental alias between the two is shape-illegal at any
`vocab_size != embed_dim`.

### Output dict

`call()` always returns `{"logits", "last_hidden_state"}`. Keeping
`last_hidden_state` costs nothing (it is the tensor the head consumes) and
makes the model usable as an encoder without a second forward pass.

---

## 16. Citation

```bibtex
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and
          Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI Technical Report},
  year={2019}
}
```

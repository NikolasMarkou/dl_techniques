# WaveFieldLLM: a decoder-only LM built on a damped wave field

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

`WaveFieldLLM` is a GPT-2-shaped decoder-only language model in which
dot-product self-attention is replaced by **`WaveFieldAttention`** — an
FFT-based token mixer that scatters token content onto a 1-D field, convolves
it with a per-head damped wave kernel, and gathers it back. Everything else
(token + learned positional embeddings, pre-norm blocks, weight-tied LM head,
dict output) mirrors `models/gpt2/` so the two share a training pipeline.

> ⚠️ **No public pretrained weights are distributed by this library.** Calls
> like `WaveFieldLLM.from_variant("small", pretrained=True)` raise
> `NotImplementedError` — deliberately, so that a caller who asks for trained
> weights never silently receives an untrained model. To load weights, pass a
> local file path (`pretrained="path/to/checkpoint.keras"`). To get a
> random-initialized model, omit `pretrained` (or pass `pretrained=False`).

> 🚨 **Causality is MEASURED, not guaranteed. Read §3 before you decode
> autoregressively.** This module builds **no causal mask**. Whatever
> token-level causality the stack has comes from the wave kernel alone, and it
> depends on the exact `(field_size, max_seq_len)` pair — it is **not** monotone
> in their ratio, and some configurations leak future information into earlier
> positions. The shipped default measured clean; anything else must be
> re-measured (§3 gives the script).

---

## Table of Contents

1. [Overview: What WaveFieldLLM is and Why It Matters](#1-overview-what-wavefieldllm-is-and-why-it-matters)
2. [The Problem WaveFieldLLM Explores](#2-the-problem-wavefieldllm-explores)
3. [How WaveFieldLLM Works: Core Concepts (incl. causality)](#3-how-wavefieldllm-works-core-concepts-incl-causality)
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

## 1. Overview: What WaveFieldLLM is and Why It Matters

### What it is

A research architecture. It keeps the GPT-2 skeleton and swaps the mixer:

- **Same as GPT-2**: token embeddings, learned positional embeddings, embedding
  LayerNorm + dropout, `depth` pre-norm blocks with residual connections, final
  LayerNorm, weight-tied LM head, `{"logits", "last_hidden_state"}` output.
- **Different**: the block's first sub-layer is `WaveFieldAttention`
  (`dl_techniques/layers/attention/wave_field_attention.py`) instead of
  multi-head attention, and there is one extra hyperparameter, `field_size`.

### What it is not

It is **not** a drop-in causal LM with GPT-2's guarantees. The mixer's
causality is an empirical property of the configuration (§3). Treat this
package as an experiment platform, not as a safe autoregressive decoder.

### Why the architecture is interesting

Softmax attention costs O(N²) in sequence length and computes an explicit
pairwise interaction matrix. The wave field instead pushes every token's
content onto a shared 1-D grid, does **one FFT convolution per head** over that
grid, and reads back. Interaction becomes a physical propagation on the field
rather than an all-pairs product — a genuinely different inductive bias, with a
cost that grows with the *field* rather than with N².

---

## 2. The Problem WaveFieldLLM Explores

```
┌─────────────────────────────────────────────────────────────┐
│  Dot-product attention                                      │
│    - builds an N x N score matrix                           │
│    - every pair interacts explicitly                        │
│    - causality is a MASK: exact, by construction            │
│                                                             │
│  Wave field mixing                                          │
│    - deposits tokens on a shared 1-D grid of `field_size`   │
│    - one damped-wave FFT convolution per head               │
│    - influence decays with distance (alpha) and oscillates  │
│      (omega, phi) -- a learned propagation, not a lookup    │
│    - causality is a CONSEQUENCE of a left-aligned kernel,   │
│      and only on the GRID -- see section 3                  │
└─────────────────────────────────────────────────────────────┘
```

The question the architecture asks: can a learned propagation kernel over a
shared field substitute for explicit pairwise attention in language modeling?
This package exists to make that question runnable, measurable, and trainable.

---

## 3. How WaveFieldLLM Works: Core Concepts (incl. causality)

### The mixing step

```
tokens (B, N, D)
   │  scatter: each token deposits V * ||K|| onto grid cells,
   │           bilinearly split across TWO neighbouring cells
   ▼
field  (B, G, H)          G = field_size, H = num_heads
   │  convolve: per-head damped wave kernel
   │            k_h(t) = exp(-alpha_h * t) * cos(omega_h * t + phi_h),
   │            applied by FFT, LEFT-ALIGNED (t >= 0 only)
   │  mix: a learned coupling matrix mixes heads at each grid position
   ▼
field' (B, G, H)
   │  gather: each token reads back from its own grid position
   ▼
tokens (B, N, D)
```

### Causality — the single most important thing to know

**There is no causal mask anywhere in this package.** The only mask the block
forwards to attention is your optional `(B, N)` padding mask.

The wave kernel is causal **on the field grid**: a cell is only influenced by
cells at or before it. But the scatter/gather is *bilinear* and spans two grid
cells, so a later token can deposit into a cell that an earlier token gathers
from. Whether that happens is a property of the exact
`(field_size, max_seq_len)` pair — through the field stride
`(field_size - 1) / (max_seq_len - 1)` — and the ratio `field_size / max_seq_len`
is only a lossy summary of it.

Measured end-to-end on this model (`max_seq_len=32`, `embed_dim=64`, `depth=2`,
seeded, random init; one token substituted, worst absolute logit change over
**all** earlier positions and **all** perturbed positions, against logits of
magnitude ~1.1):

| ratio | `field_size` | stride | worst leak (CPU / GPU) | verdict |
|---:|---:|---:|---|---|
| 0.50 | 16 | 0.4839 | 5.46e-04 / 5.50e-04 | **LEAKS** |
| 0.75 | 24 | 0.7419 | 3.77e-04 / 4.05e-04 | **LEAKS** |
| 1.00 | 32 | 1.0000 | 6.71e-08 / below 1e-5 | clean |
| 1.50 | 48 | 1.5161 | 4.96e-05 / 1.24e-04 | **LEAKS** |
| 2.00 | 64 | 2.0323 | 5.96e-08 / below 1e-5 | clean ← **default** |
| 4.00 | 128 | 4.0968 | 8.94e-08 / below 1e-5 | clean |

Read this table carefully:

- **It is not monotone.** Ratio 1.50 leaks while ratio 1.00 does not. There is
  no "bigger field is safer" rule, and no ratio threshold is offered as a
  sufficient condition.
- **The default measured clean.** `field_size` defaults to `2 * max_seq_len`;
  its stride `(2M - 1) / (M - 1) > 2` keeps consecutive tokens more than one
  grid cell apart. That is *evidence*, not a proof.
- **Exact numbers are device-dependent** (CPU and GPU differ by up to ~2.5× on
  the leaky rows), so the pin in the test suite is an order-of-magnitude bound,
  not a number.
- **"clean" does not mean exactly zero.** The clean residue is float32 noise at
  logits of magnitude ~1.1, and it is *process-history dependent*: the same
  config and seed measured `0.0` in one process ordering and `1.565e-07` in
  another; ratio 2.00 gives `0.0` at seeds 1234/7 but `6.109e-07` at seed 99;
  values up to `1.185e-06` have been seen on GPU with no code change. Rely only
  on the order of magnitude — clean stays **below 1e-5**, leaky stays above it,
  and the measurements above separate the two by ~40×. The probe below will
  print small non-zero numbers on clean rows on some runs, and that is expected.
- **Change either value and you must re-measure**, with the probe below.

### Measuring your own configuration

Perturb **every** position, not just the last one: whether token `j` leaks into
token `i` depends on the fractional part of `j × stride`, so a last-token-only
probe is blind (it reported *clean* at a configuration the all-positions probe
measures at 1.96e-04).

```python
"""Measure the future-token leak of a (field_size, max_seq_len) pair.

Substitute one token at position j and look at how much the logits at every
EARLIER position move. A strictly causal model moves them by 0.
"""
import keras
import numpy as np
from dl_techniques.models.wave_field import WaveFieldLLM

MAX_SEQ_LEN, VOCAB = 32, 256


def worst_leak(model):
    ids = (np.arange(1, MAX_SEQ_LEN + 1, dtype=np.int32)[None, :]) % VOCAB
    base = keras.ops.convert_to_numpy(model(ids, training=False)["logits"])[0]
    worst = 0.0
    for j in range(1, MAX_SEQ_LEN):          # perturb EVERY position, not just the last
        ids2 = ids.copy()
        ids2[0, j] = (ids2[0, j] + 137) % VOCAB
        out = keras.ops.convert_to_numpy(model(ids2, training=False)["logits"])[0]
        worst = max(worst, float(np.abs(base[:j] - out[:j]).max()))
    return worst


for field_size in (16, 32, 48, 64):
    keras.utils.set_random_seed(1234)
    model = WaveFieldLLM(
        vocab_size=VOCAB, embed_dim=64, depth=2, num_heads=4,
        max_seq_len=MAX_SEQ_LEN, field_size=field_size,
    )
    ratio = field_size / MAX_SEQ_LEN
    print(f"field_size={field_size:4d}  ratio={ratio:4.2f}  worst leak={worst_leak(model):.3e}")
```

One observed run (single RTX 4070). The `0.000e+00` rows below are *one
process's* float32 residue, not a property: expect the leaky rows to move by up
to ~2.5×, and expect the clean rows to print anything below `1e-5` (they have
been observed up to `1.185e-06` on this GPU). Judge by the order of magnitude:

```
field_size=  16  ratio=0.50  worst leak=5.420e-04
field_size=  32  ratio=1.00  worst leak=0.000e+00
field_size=  48  ratio=1.50  worst leak=1.209e-04
field_size=  64  ratio=2.00  worst leak=0.000e+00
```

---

## 4. Architecture Deep Dive

```
Input IDs (B, N)
     │
     ▼
Token Embedding ──► PositionalEmbedding (learned, slice + broadcast-add)
     │
     ▼
Embed LayerNorm ──► Embed Dropout          (order: add → norm → dropout)
     │
     ▼
WaveFieldDecoderBlock × depth
   ├─ attn_norm → WaveFieldAttention → residual     (no causal mask)
   └─ ffn_norm  → Dense(4D, gelu) → Dense(D) → Dropout → residual
     │
     ▼
Final LayerNorm
     │
     ▼
LM head: tied → hidden @ Eᵀ | untied → Dense(vocab, no bias)
     │
     ▼
{"logits": (B, N, V), "last_hidden_state": (B, N, D)}
```

### 4.1 `WaveFieldDecoderBlock`

A pre-norm block with two external residuals (`x = inputs + h`, then
`return x + h`) — the transform is never allowed to replace the stream. It is
assembled locally rather than reusing `TransformerLayer` for one measured
reason: `WaveFieldAttention` consumes a `(B, N)` mask, while `TransformerLayer`
and `TextDecoder` build and forward a `(B, N, N)` mask. That is a contract
mismatch, not a style preference.

Its `build()` explicitly builds every sub-layer. This is required, not
defensive: `WaveFieldAttention` creates variables through `IdentityPlusNoise`
at build time, and under Keras 3's symbolic tracing of a `keras.Model`
subclass a nested build can be skipped, leaving `add_weight` without a backing
variable (`'NoneType' object has no attribute 'assign'`). The same reasoning
drives the eager builds at the end of `WaveFieldLLM._build_architecture`.

### 4.2 The embedding pipeline

The positional path goes through the shared factory,
`create_embedding_layer('positional_learned', ..., dropout_rate=0.0)`, which
owns the slice-and-add. Its own dropout is **deliberately disabled** and
`embed_dropout` is kept as a separate layer applied *after* `embed_norm`,
because this model's order is add → norm → dropout while the layer's internal
order would be add → dropout → norm. Under an identical dropout mask those two
orders differ by max |Δ| 0.395 on unit-variance activations at
`dropout_rate=0.1` — about 38% of signal RMS. Folding them together is a
behaviour change, not a cleanup. (The weight variable is
`position_embeddings/pos_embedding`, shape `(1, max_seq_len, embed_dim)`.)

### 4.3 The FFN

`Dense(4D, gelu) → Dense(D) → Dropout`, with dropout on the block **output**,
before the residual add. It is hand-rolled on purpose — see §15.

### 4.4 Validation

`_validate_config` raises `ValueError` for non-positive `vocab_size`,
`embed_dim`, `depth`, `num_heads` or `max_seq_len`, for
`embed_dim % num_heads != 0`, for `field_size <= 1`, and for a dropout rate
outside `[0, 1]` — all at construction time.

---

## 5. Quick Start Guide

```python
import numpy as np
from dl_techniques.models.wave_field import create_wave_field_llm

model = create_wave_field_llm("tiny", vocab_size=512)
outputs = model(np.random.randint(0, 512, size=(2, 64)).astype("int32"))

print(outputs["logits"].shape)             # (2, 64, 512)
print(outputs["last_hidden_state"].shape)  # (2, 64, 256)
print(model.max_seq_len, model.field_size)  # 512 1024
```

Note the printed `field_size`: it is `2 * max_seq_len`, the configuration that
measured clean in §3.

---

## 6. Component Reference

### 6.1 `WaveFieldLLM` (model class)

`keras.Model` subclass. Accepts a tensor of IDs or a dict
(`input_ids` required, `attention_mask` optional), returns
`{"logits", "last_hidden_state"}`. Key methods: `from_variant` (classmethod),
`call`, `compute_output_shape`, `get_config`, and the static
`_download_weights` (always raises).

### 6.2 `WaveFieldDecoderBlock` (layer)

Exported so a single block can be reused outside the model — it operates on
`(B, N, D)` activations and takes the same `attention_mask` keyword.

### 6.3 `create_wave_field_llm(...)` (module-level factory)

A thin wrapper over `WaveFieldLLM.from_variant`, mirroring `create_bert` /
`create_gpt2`. `vocab_size` is injected only when not `None`.

### 6.4 Public API

`dl_techniques.models.wave_field` exports exactly `WaveFieldLLM`,
`WaveFieldDecoderBlock` and `create_wave_field_llm` via `__all__`.

---

## 7. Configuration & Model Variants

`WaveFieldLLM.MODEL_VARIANTS` (each entry also carries a `"description"`,
popped by `from_variant`):

| Variant | `embed_dim` | `depth` | `num_heads` | `max_seq_len` | `field_size` |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `tiny` | 256 | 4 | 4 | 512 | 1024 |
| `small` | 768 | 12 | 12 | 1024 | 2048 |
| `medium` | 1024 | 24 | 16 | 1024 | 2048 |
| `large` | 1280 | 36 | 20 | 1024 | 2048 |
| `xl` | 1600 | 48 | 25 | 1024 | 2048 |

Every variant ships `field_size = 2 * max_seq_len`. **Overriding one without
the other changes the stride and therefore the causality behaviour** (§3).

Other constructor arguments:

| Argument | Default | Notes |
|---|---|---|
| `vocab_size` | `50261` | Tiktoken `gpt2` (50257) + 4 special tokens. |
| `field_size` | `None` → `2 * max_seq_len` | Wave field grid resolution. |
| `dropout_rate` | `0.0` | Embedding and FFN paths. |
| `attention_dropout_rate` | `0.0` | Attention output. |
| `initializer_range` | `0.02` | `TruncatedNormal` stddev. |
| `layer_norm_eps` | `1e-5` | |
| `tie_word_embeddings` | `True` | Untied builds a `Dense(vocab, use_bias=False)`. |

Read the variants from the class rather than trusting this table:

```python
from dl_techniques.models.wave_field import WaveFieldLLM
for name, cfg in WaveFieldLLM.MODEL_VARIANTS.items():
    print(name, cfg)
```

---

## 8. Comprehensive Usage Examples

### Example 1: dict input, a standalone block, and construction errors

```python
import numpy as np
from dl_techniques.models.wave_field import WaveFieldLLM, WaveFieldDecoderBlock

model = WaveFieldLLM(
    vocab_size=256, embed_dim=64, depth=2, num_heads=4, max_seq_len=32,
)

# Dict input. The padding mask is (B, N) -- NOT (B, N, N): that shape is the
# reason this model ships its own decoder block instead of TransformerLayer.
batch = {
    "input_ids": np.random.randint(0, 256, size=(2, 16)).astype("int32"),
    "attention_mask": np.concatenate(
        [np.ones((2, 12), "int32"), np.zeros((2, 4), "int32")], axis=1
    ),
}
print(model(batch, training=False)["logits"].shape)  # (2, 16, 256)

# A single block is usable standalone on (B, N, D) activations.
block = WaveFieldDecoderBlock(
    embed_dim=64, num_heads=4, max_seq_len=32, field_size=64,
)
h = np.random.normal(size=(2, 16, 64)).astype("float32")
print(block(h).shape)  # (2, 16, 64)

# Configuration errors are raised at construction, not at the first forward pass.
try:
    WaveFieldLLM(vocab_size=256, embed_dim=64, num_heads=5, max_seq_len=32)
except ValueError as e:
    print(f"ValueError: {e}")
```

### Example 2: the weights contract

```python
from dl_techniques.models.wave_field import WaveFieldLLM, create_wave_field_llm

for call in (
    lambda: WaveFieldLLM.from_variant("tiny", pretrained=True),
    lambda: create_wave_field_llm("tiny", pretrained=True),
):
    try:
        call()
    except NotImplementedError as e:
        print(f"NotImplementedError: {e}")
```

A string path is loaded with `skip_mismatch=True`; a nonexistent one raises
`FileNotFoundError`.

---

## 9. Advanced Usage Patterns

### Pattern 1: train on your own data

```python
import keras
import numpy as np
from dl_techniques.models.wave_field import WaveFieldLLM
from dl_techniques.losses import MaskedCausalLMLoss

model = WaveFieldLLM(
    vocab_size=256, embed_dim=64, depth=2, num_heads=4, max_seq_len=32,
)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=3e-4, clipnorm=1.0),
    loss={"logits": MaskedCausalLMLoss()},
)

ids = np.random.randint(0, 256, size=(8, 33)).astype("int32")
history = model.fit(
    {"input_ids": ids[:, :-1]}, {"logits": ids[:, 1:]},
    batch_size=4, epochs=1, verbose=0,
)
print(sorted(history.history.keys()))
```

The one-position shift is the caller's job, exactly as for GPT-2.

### Pattern 2: sweeping `field_size`

`field_size` is the architecture's distinctive knob: it sets the grid the wave
propagates on. Sweep it — but pair every sweep with the leak probe from §3, and
record both numbers. A configuration that trains better while leaking the
future is not a better language model.

### Pattern 3: memory-augmented variant

`src/dl_techniques/models/memory_bank/` layers a dual-tap long-term / working
memory bank on top of `WaveFieldLLM` (`wave_field_memory_llm.py`). It is a
separate package with its own tests, and it currently ships **no trainer** —
`src/train/wave_field/train_memory.py` was deleted on 2026-08-13. The model
package itself was not deleted.

---

## 10. Performance Optimization

### Cost model

Per block, the mixer's work is dominated by the FFT convolution over the field
(`field_size` cells × `num_heads`), plus the scatter and gather, which are
linear in `N`. Doubling `field_size` doubles the field work; it does not
square anything. That is the architecture's selling point — and the reason
`field_size` cannot simply be minimized: it is also the causality knob (§3).

### Mixed precision

Measured to run under `mixed_float16` on this repo's GPUs, both forward and a
`fit()` step:

```python
import keras
import numpy as np
from dl_techniques.models.wave_field import WaveFieldLLM

keras.mixed_precision.set_global_policy("mixed_float16")
model = WaveFieldLLM(vocab_size=256, embed_dim=64, depth=2, num_heads=4, max_seq_len=32)
out = model(np.random.randint(0, 256, (2, 16)).astype("int32"), training=False)
print(out["logits"].dtype)
keras.mixed_precision.set_global_policy("float32")
```

The leak measurements in §3 were taken in float32. Reduced precision changes
the numbers; if causality matters to your run, measure it in the precision you
will train in.

---

## 11. Training and Best Practices

One runnable pipeline ships with the repo: `src/train/wave_field/pretrain.py`.
It mirrors `train/gpt2/pretrain.py` — same shared `ClmPretrainConfig`, same
data sources, same callbacks — plus a `--field-size` flag.

```bash
# Full flag surface (the source of truth, not this file)
MPLBACKEND=Agg .venv/bin/python -m train.wave_field.pretrain --help

# TFDS smoke run on GPU 1
MPLBACKEND=Agg .venv/bin/python -m train.wave_field.pretrain \
    --gpu 1 --variant tiny --dataset-source tfds \
    --dataset-name imdb_reviews --max-samples 64 \
    --epochs 1 --batch-size 2 --max-seq-length 32
```

Practical notes:

- **`--max-seq-length` moves the stride.** The trainer sizes the model to the
  sequence length you ask for; `field_size` follows the variant unless you set
  `--field-size`. Re-run the §3 probe for the pair you actually train.
- **The tokenizer defines `vocab_size`** — the class default matches Tiktoken
  `gpt2` plus 4 special tokens, and the trainer passes its own value through.
- **Runs write to repo-root `results/`**, with `config.json` and
  `training_history.json` written next to the checkpoints.
- **Do not use this model for production autoregressive generation** without
  first measuring the leak for your configuration.

---

## 12. Serialization & Deployment

```python
import keras
import numpy as np
from dl_techniques.models.wave_field import WaveFieldLLM

model = WaveFieldLLM(
    vocab_size=256, embed_dim=64, depth=2, num_heads=4, max_seq_len=32,
)
ids = np.random.randint(0, 256, size=(1, 8)).astype("int32")
before = model(ids, training=False)["logits"]

model.save("wave_field_demo.keras")
restored = keras.models.load_model("wave_field_demo.keras")
after = restored(ids, training=False)["logits"]

print(float(np.max(np.abs(np.array(before) - np.array(after)))))  # 0.0
```

Both `WaveFieldLLM` and `WaveFieldDecoderBlock` are registered for Keras
serialization, so no `custom_objects` is needed. `get_config()` stores all
eleven constructor arguments (including `field_size`, resolved to its concrete
value, never left as `None`).

---

## 13. Testing & Validation

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/test_models/test_wave_field/ -q
```

The suite covers construction validation, forward shapes for tensor and dict
inputs at full and partial sequence length, the **causality ratio sweep** (the
pin for the §3 table — clean ratios must stay below a bound, leaky ratios must
stay *above* one, so that a silent change in either direction fails), weight
tying, `get_config` / `.keras` round trips, every named variant, gradient flow,
CLM loss finiteness, the `pretrained=True` error contract, and the package's
public API surface.

Trainer-side tests:

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/test_train/test_wave_field/ -q
```

Report *passed with collected* — a suite that collects almost nothing also
reports green.

---

## 14. Troubleshooting & FAQs

**"Is this model causal?"**
Only as measured, for the exact `(field_size, max_seq_len)` you built. See §3
and run the probe. Nothing in this package enforces causality.

**`NotImplementedError: Pretrained WaveFieldLLM weights are not distributed...`**
Working as designed. Pass a local checkpoint path or `pretrained=False`. This
replaced an older path that logged a warning and returned a random-init model
to a caller who had asked for pretrained weights.

**`ValueError: field_size must be > 1`** — raised at construction. So are the
divisibility and dropout-range errors.

**`'NoneType' object has no attribute 'assign'`** — the Keras 3 lazy-build trap
this package's explicit `build()` calls exist to prevent. If you refactor the
block or the model's architecture builder, keep the explicit sub-layer builds.

**A checkpoint written before 2026-08-13 will not restore its positional
table.** The positional path moved to the shared factory: the variable is now
`position_embeddings/pos_embedding` with shape `(1, M, D)` instead of
`position_embeddings/embeddings` with shape `(M, D)`. The layer name and every
other weight path are unchanged. No checkpoint existed in `results/` when the
change landed.

**Q: Why does this model not use `TransformerLayer` like GPT-2 does?**
A: The mask contract. `WaveFieldAttention` takes `(B, N)`;
`TransformerLayer` / `TextDecoder` build `(B, N, N)`. Migrating would mean
redesigning the mask contract, which is a different change with a different
risk profile.

---

## 15. Technical Details

### Why the FFN and the norms are hand-rolled (measured, 2026-08-13)

The repo's layer-reuse policy says to prefer a factory. Three candidate
adoptions in this file were measured and **refused**:

1. **FFN → `create_ffn_layer('mlp')`.** `MLPBlock.call` applies dropout
   *between* `fc1`+activation and `fc2`; this block applies it *after* the
   output projection. At `dropout_rate=0.1`, under an identical Bernoulli mask,
   the two orders differ by max |Δ| 0.3953 (~38% of signal RMS). That is a
   behaviour change at any non-zero dropout, not a refactor. (`MLPBlock` also
   names its sub-layers `fc1`/`fc2` rather than `ffn_dense_1`/`ffn_dense_2`, so
   a `by_name` weight load would not bind.)
2. **The four `LayerNormalization` sites → `create_normalization_layer`.** The
   factory's `'layer_norm'` key resolves to `keras.layers.LayerNormalization`
   itself. Routing through it constructs the identical class, saves zero lines
   (the explicit `epsilon` override is still required, since the factory's
   default differs), and fixes no defect.
3. **The embedding tables and the untied `lm_head`.** No factory covers a bare
   word-embedding table or a bare vocabulary projection; the sibling
   `TextDecoder` hand-rolls the same `keras.layers.Embedding`, so this is the
   repo's own pattern rather than a local deviation.

What *was* adopted: the positional path (§4.2), which the sibling
`TextDecoder` already routes through the same factory key.

### Weight-path layout

Block level: `attn_norm`, `attention`, `ffn_norm`, `ffn_dense_1`,
`ffn_dense_2`, `ffn_dropout`. Model level: `token_embeddings`,
`position_embeddings`, `embed_norm`, `embed_dropout`, `block_{i}`,
`final_norm`, and `lm_head` when untied.

### Relationship to `WaveFieldAttention`

The layer is the subject; this package is its language-model harness. The
layer's own docstring (`layers/attention/wave_field_attention.py`, "Causality"
section) is the authority on what the mixer does and does not promise, and it
is deliberately more conservative than any model-level claim: *"Do NOT rely on
this layer for autoregressive decoding."* The measured table in §3 is this
model's end-to-end confirmation of that warning, not a rebuttal of it.

---

## 16. Citation

The architecture shell follows GPT-2:

```bibtex
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and
          Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI Technical Report},
  year={2019}
}
```

The wave-field mixer has no external paper: it is an internal architecture,
defined by `src/dl_techniques/layers/attention/wave_field_attention.py`. Cite
that module (and this package) directly rather than attributing it to a
publication.

# WaveFieldLLM: a decoder-only LM built on a damped wave field

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

`WaveFieldLLM` is a GPT-2-shaped decoder-only language model in which dot-product self-attention
is replaced by **`WaveFieldAttention`**, an FFT-based token mixer that scatters token content onto
a 1-D field, convolves it with a per-head damped wave kernel, and gathers it back. Everything else
mirrors `models/language/gpt2/`, so the two share a training pipeline.

> **No pretrained weights are distributed by this library.**
> `WaveFieldLLM.from_variant("small", pretrained=True)` raises `NotImplementedError`, so a caller
> asking for trained weights never silently receives an untrained model. Pass a local file path
> (`pretrained="path/to/checkpoint.keras"`), or omit `pretrained` for random init.

> **Causality is MEASURED, not guaranteed. Read §3 before you decode autoregressively.** This
> module builds **no causal mask**. Whatever token-level causality the stack has comes from the
> wave kernel alone and depends on the exact `(field_size, max_seq_len)` pair. It is **not**
> monotone in their ratio, and some configurations leak the future into earlier positions. The
> shipped default measured clean; anything else must be re-measured (§3 gives the script).

---

## 1. Overview: What WaveFieldLLM is and Why It Matters

A research architecture. It keeps the GPT-2 skeleton and swaps the mixer:

- **Same as GPT-2**: token embeddings, learned positional embeddings, embedding LayerNorm +
  dropout, `depth` pre-norm blocks with residuals, final LayerNorm, weight-tied LM head,
  `{"logits", "last_hidden_state"}` output.
- **Different**: the block's first sub-layer is `WaveFieldAttention`
  (`dl_techniques/layers/attention/wave_field_attention.py`) instead of multi-head attention, and
  there is one extra hyperparameter, `field_size`.

It is **not** a drop-in causal LM with GPT-2's guarantees; the mixer's causality is an empirical
property of the configuration (§3). Treat this as an experiment platform, not a safe
autoregressive decoder.

Why it is interesting: softmax attention costs O(N^2) and builds an explicit pairwise interaction
matrix. The wave field instead pushes every token onto a shared 1-D grid, does one FFT convolution
per head, and reads back, so interaction is a propagation on the field rather than an all-pairs
product, at a cost that grows with the *field* rather than with N^2.

---

## 2. The Problem WaveFieldLLM Explores

| | Dot-product attention | Wave field mixing |
|:---|:---|:---|
| Interaction | an N x N score matrix; every pair interacts explicitly | tokens deposit on a shared 1-D grid of `field_size`; one damped-wave FFT convolution per head |
| Distance | uniform lookup, mask-limited | influence decays with distance (`alpha`) and oscillates (`omega`, `phi`): a learned propagation |
| Causality | a MASK: exact, by construction | a CONSEQUENCE of a left-aligned kernel, and only on the GRID (§3) |

The question: can a learned propagation kernel over a shared field replace explicit pairwise
attention in language modeling? This package makes it runnable and measurable.

---

## 3. How WaveFieldLLM Works: Core Concepts (incl. causality)

```
tokens (B, N, D)
   | scatter: each token deposits V * ||K|| onto grid cells, bilinearly
   |          split across TWO neighbouring cells
   v
field  (B, G, H)   G = field_size, H = num_heads
   | convolve: per-head damped wave kernel, applied by FFT, LEFT-ALIGNED (t >= 0)
   |           k_h(t) = exp(-alpha_h * t) * cos(omega_h * t + phi_h)
   | mix: a learned coupling matrix mixes heads at each grid position
   v
field' (B, G, H)
   | gather: each token reads back from its own grid position
   v
tokens (B, N, D)
```

### Causality, the single most important thing to know

**There is no causal mask anywhere in this package.** The only mask the block forwards to
attention is your optional `(B, N)` padding mask.

The wave kernel is causal **on the field grid**: a cell is only influenced by cells at or before
it. But the scatter/gather is *bilinear* and spans two grid cells, so a later token can deposit
into a cell an earlier token gathers from. Whether it does is a property of the exact
`(field_size, max_seq_len)` pair, through the stride `(field_size - 1) / (max_seq_len - 1)`; the
ratio `field_size / max_seq_len` is only a lossy summary of it.

Measured end to end (`max_seq_len=32`, `embed_dim=64`, `depth=2`, seeded, random init; one token
substituted, worst absolute logit change over **all** earlier positions and **all** perturbed
positions, against logits of magnitude ~1.1):

| ratio | `field_size` | stride | worst leak (CPU / GPU) | verdict |
|---:|---:|---:|---|---|
| 0.50 | 16 | 0.4839 | 5.46e-04 / 5.50e-04 | **LEAKS** |
| 0.75 | 24 | 0.7419 | 3.77e-04 / 4.05e-04 | **LEAKS** |
| 1.00 | 32 | 1.0000 | 6.71e-08 / below 1e-5 | clean |
| 1.50 | 48 | 1.5161 | 4.96e-05 / 1.24e-04 | **LEAKS** |
| 2.00 | 64 | 2.0323 | 5.96e-08 / below 1e-5 | clean, the **default** |
| 4.00 | 128 | 4.0968 | 8.94e-08 / below 1e-5 | clean |

Read the table carefully:

- **It is not monotone.** Ratio 1.50 leaks while ratio 1.00 does not. There is no "bigger field is
  safer" rule, and no ratio threshold is a sufficient condition.
- **The default measured clean.** `field_size` defaults to `2 * max_seq_len`; its stride
  `(2M - 1) / (M - 1) > 2` keeps consecutive tokens more than one grid cell apart. Evidence, not
  a proof.
- **"clean" is not exactly zero.** The residue is float32 noise and is process-history dependent:
  the same config and seed has measured `0.0` in one ordering and `1.565e-07` in another, up to
  `1.185e-06` on GPU. Judge by order of magnitude: clean stays **below 1e-5**, leaky above it.
- **Change either value and re-measure**, with the probe below.

### Measuring your own configuration

Perturb **every** position, not just the last: whether token `j` leaks into token `i` depends on
the fractional part of `j * stride`, so a last-token-only probe is blind (it reported *clean* at a
configuration the all-positions probe measures at 1.96e-04).

```python
import keras
import numpy as np
from dl_techniques.models.language.wave_field import WaveFieldLLM

MAX_SEQ_LEN, VOCAB = 32, 256


def worst_leak(model):
    """Substitute one token at position j; see how far EARLIER logits move."""
    ids = (np.arange(1, MAX_SEQ_LEN + 1, dtype=np.int32)[None, :]) % VOCAB
    base = keras.ops.convert_to_numpy(model(ids, training=False)["logits"])[0]
    worst = 0.0
    for j in range(1, MAX_SEQ_LEN):     # perturb EVERY position, not just the last
        ids2 = ids.copy()
        ids2[0, j] = (ids2[0, j] + 137) % VOCAB
        out = keras.ops.convert_to_numpy(model(ids2, training=False)["logits"])[0]
        worst = max(worst, float(np.abs(base[:j] - out[:j]).max()))
    return worst


for field_size in (16, 32, 48, 64):
    keras.utils.set_random_seed(1234)
    model = WaveFieldLLM(vocab_size=VOCAB, embed_dim=64, depth=2, num_heads=4,
                         max_seq_len=MAX_SEQ_LEN, field_size=field_size)
    print(f"field_size={field_size:4d}  ratio={field_size / MAX_SEQ_LEN:4.2f}  "
          f"worst leak={worst_leak(model):.3e}")
```

One observed CPU run printed `5.485e-04`, `1.192e-07`, `4.976e-05`, `5.960e-08` for field sizes
16, 32, 48, 64: the same clean/leaky split as the table. Expect leaky rows to move by up to ~2.5x
and clean rows to print anything below `1e-5`.

---

## 4. Architecture Deep Dive

```
Input IDs (B, N)
  -> Token Embedding --> PositionalEmbedding (learned, slice + broadcast-add)
  -> Embed LayerNorm --> Embed Dropout       (order: add -> norm -> dropout)
  -> WaveFieldDecoderBlock x depth
       |- attn_norm -> WaveFieldAttention -> residual     (no causal mask)
       |- ffn_norm  -> Dense(4D, gelu) -> Dense(D) -> Dropout -> residual
  -> Final LayerNorm
  -> LM head: tied -> hidden @ E^T | untied -> Dense(vocab, no bias)
  -> {"logits": (B, N, V), "last_hidden_state": (B, N, D)}
```

### 4.1 `WaveFieldDecoderBlock`

A pre-norm block with two external residuals (`x = inputs + h`, then `return x + h`), so the
transform never replaces the stream. It is assembled locally rather than reusing
`TransformerLayer` for one measured reason: `WaveFieldAttention` consumes a `(B, N)` mask, while
`TransformerLayer` and `TextDecoder` build and forward a `(B, N, N)` mask. A contract mismatch,
not a style preference.

Its `build()` explicitly builds every sub-layer, and that is required rather than defensive:
`WaveFieldAttention` creates variables through `IdentityPlusNoise` at build time, and under Keras
3's symbolic tracing of a `keras.Model` subclass a nested build can be skipped, leaving
`add_weight` without a backing variable (`'NoneType' object has no attribute 'assign'`). The eager
builds at the end of `WaveFieldLLM._build_architecture` exist for the same reason.

### 4.2 The embedding pipeline

The positional path goes through `create_embedding_layer('positional_learned', ...,
dropout_rate=0.0)`, which owns the slice-and-add. Its own dropout is **deliberately disabled** and
`embed_dropout` is a separate layer applied *after* `embed_norm`: this model's order is add, norm,
dropout, while the layer's internal order would be add, dropout, norm. Under an identical mask
those orders differ by max |delta| 0.395 on unit-variance activations at `dropout_rate=0.1`, about
38% of signal RMS, so folding them is a behaviour change, not a cleanup. The weight variable is
`position_embeddings/pos_embedding`, shape `(1, max_seq_len, embed_dim)`.

### 4.3 The FFN and validation

`Dense(4D, gelu) -> Dense(D) -> Dropout`, with dropout on the block **output**, before the
residual add; hand-rolled on purpose (§15). `_validate_config` raises `ValueError` at construction
for non-positive `vocab_size`, `embed_dim`, `depth`, `num_heads` or `max_seq_len`, for
`embed_dim % num_heads != 0`, for `field_size <= 1`, and for a dropout rate outside `[0, 1]`.

---

## 5. Quick Start Guide

```python
import numpy as np
from dl_techniques.models.language.wave_field import create_wave_field_llm

model = create_wave_field_llm("tiny", vocab_size=512)
outputs = model(np.random.randint(0, 512, size=(2, 64)).astype("int32"))

print(outputs["logits"].shape)              # (2, 64, 512)
print(outputs["last_hidden_state"].shape)   # (2, 64, 256)
print(model.max_seq_len, model.field_size)  # 512 1024
```

Note the printed `field_size`: `2 * max_seq_len`, the configuration that measured clean in §3.

---

## 6. Component Reference

| Name | Kind | Notes |
|:---|:---|:---|
| `WaveFieldLLM` | `keras.Model` | Accepts a tensor of IDs or a dict (`input_ids` required, `attention_mask` optional); returns `{"logits", "last_hidden_state"}`. Methods: `from_variant`, `call`, `compute_output_shape`, `get_config`, and the static `_download_weights` (always raises). |
| `WaveFieldDecoderBlock` | `keras.layers.Layer` | A single block, reusable outside the model on `(B, N, D)` activations, with the same `attention_mask` keyword. |
| `create_wave_field_llm` | factory | Thin wrapper over `WaveFieldLLM.from_variant`, mirroring `create_bert` / `create_gpt2`. `vocab_size` is injected only when not `None`. |

`dl_techniques.models.language.wave_field` exports exactly those three names via `__all__`.

---

## 7. Configuration & Model Variants

`WaveFieldLLM.MODEL_VARIANTS` (each entry also carries a `"description"`, popped by
`from_variant`):

| Variant | `embed_dim` | `depth` | `num_heads` | `max_seq_len` | `field_size` |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `tiny` | 256 | 4 | 4 | 512 | 1024 |
| `small` | 768 | 12 | 12 | 1024 | 2048 |
| `medium` | 1024 | 24 | 16 | 1024 | 2048 |
| `large` | 1280 | 36 | 20 | 1024 | 2048 |
| `xl` | 1600 | 48 | 25 | 1024 | 2048 |

Every variant ships `field_size = 2 * max_seq_len`. **Overriding one without the other changes
the stride and therefore the causality behaviour** (§3). Other constructor arguments:

| Argument | Default | Notes |
|---|---|---|
| `vocab_size` | `50261` | Tiktoken `gpt2` (50257) + 4 special tokens. |
| `field_size` | `None` -> `2 * max_seq_len` | Wave field grid resolution. |
| `dropout_rate` | `0.0` | Embedding and FFN paths. |
| `attention_dropout_rate` | `0.0` | Attention output. |
| `initializer_range` | `0.02` | `TruncatedNormal` stddev. |
| `layer_norm_eps` | `1e-5` | |
| `tie_word_embeddings` | `True` | Untied builds a `Dense(vocab, use_bias=False)`. |

Re-derive the table with
`for name, cfg in WaveFieldLLM.MODEL_VARIANTS.items(): print(name, cfg)` rather than trusting it.

---

## 8. Comprehensive Usage Examples

### Example 1: dict input, a standalone block, and construction errors

```python
import numpy as np
from dl_techniques.models.language.wave_field import WaveFieldLLM, WaveFieldDecoderBlock

model = WaveFieldLLM(vocab_size=256, embed_dim=64, depth=2, num_heads=4, max_seq_len=32)

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
block = WaveFieldDecoderBlock(embed_dim=64, num_heads=4, max_seq_len=32, field_size=64)
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
from dl_techniques.models.language.wave_field import WaveFieldLLM

try:
    WaveFieldLLM.from_variant("tiny", pretrained=True)   # so does create_wave_field_llm
except NotImplementedError as e:
    print(f"NotImplementedError: {e}")
```

A string path is loaded with `skip_mismatch=True`; a nonexistent one raises `FileNotFoundError`.
The load goes through `utils.weight_transfer.load_weights_or_raise`, which counts variables whose
value actually changed and raises when that count is **zero**, so a checkpoint whose names or
shapes do not match cannot restore nothing and return normally. Its limit: that guard fires only
on a *completely* empty load. A **partial** restore (say 9 of 30 variables) logs a warning and
returns normally, which under `skip_mismatch=True` is possible. Check the logged
restored-variable count.

---

## 9. Advanced Usage Patterns

### Pattern 1: train on your own data

```python
import keras
import numpy as np
from dl_techniques.models.language.wave_field import WaveFieldLLM
from dl_techniques.losses import MaskedCausalLMLoss

model = WaveFieldLLM(vocab_size=256, embed_dim=64, depth=2, num_heads=4, max_seq_len=32)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=3e-4, clipnorm=1.0),
    loss={"logits": MaskedCausalLMLoss()},
)

ids = np.random.randint(0, 256, size=(8, 33)).astype("int32")
history = model.fit(
    {"input_ids": ids[:, :-1]}, {"logits": ids[:, 1:]},
    batch_size=4, epochs=1, verbose=0,
)
print(sorted(history.history.keys()))  # ['loss']
```

The one-position shift is the caller's job, exactly as for GPT-2.

### Pattern 2: sweeping `field_size`

`field_size` sets the grid the wave propagates on. Sweep it, but pair every sweep with the leak
probe from §3 and record both numbers: a configuration that trains better while leaking the
future is not a better language model.

---

## 10. Performance Optimization

Per block, the mixer's work is dominated by the FFT convolution over the field (`field_size`
cells x `num_heads`), plus a scatter and gather linear in `N`. Doubling `field_size` doubles the
field work; it does not square anything. That is the selling point, and also why `field_size`
cannot simply be minimized: it is the causality knob (§3).

Mixed precision runs, both forward and a `fit()` step:

```python
import keras
import numpy as np
from dl_techniques.models.language.wave_field import WaveFieldLLM

keras.mixed_precision.set_global_policy("mixed_float16")
model = WaveFieldLLM(vocab_size=256, embed_dim=64, depth=2, num_heads=4, max_seq_len=32)
out = model(np.random.randint(0, 256, (2, 16)).astype("int32"), training=False)
print(out["logits"].dtype)  # float16
keras.mixed_precision.set_global_policy("float32")
```

The §3 leak measurements were taken in float32. Reduced precision changes the numbers; if
causality matters, measure it in the precision you will train in.

---

## 11. Training and Best Practices

One runnable pipeline ships with the repo: `src/train/wave_field/pretrain.py`. It mirrors
`train/gpt2/pretrain.py` (same `ClmPretrainConfig`, data sources and callbacks) plus a
`--field-size` flag. `--help` is the source of truth for its flags, not this file.

```bash
MPLBACKEND=Agg .venv/bin/python -m train.wave_field.pretrain \
    --gpu 1 --variant tiny --dataset-source tfds \
    --dataset-name imdb_reviews --max-samples 64 \
    --epochs 1 --batch-size 2 --max-seq-length 32
```

- **`--max-seq-length` moves the stride.** The trainer sizes the model to the length you ask for;
  `field_size` follows the variant unless you set `--field-size`. Re-run the §3 probe.
- **The tokenizer defines `vocab_size`.** The class default matches Tiktoken `gpt2` plus 4 special
  tokens; the trainer passes its own value through.
- **Runs write to repo-root `results/`**, with `config.json` and `training_history.json`.
- **Do not generate autoregressively** without first measuring the leak for your configuration.

---

## 12. Serialization & Deployment

```python
import keras
import numpy as np
from dl_techniques.models.language.wave_field import WaveFieldLLM

model = WaveFieldLLM(vocab_size=256, embed_dim=64, depth=2, num_heads=4, max_seq_len=32)
ids = np.random.randint(0, 256, size=(1, 8)).astype("int32")
before = model(ids, training=False)["logits"]

model.save("wave_field_demo.keras")
restored = keras.models.load_model("wave_field_demo.keras")  # no custom_objects needed
after = restored(ids, training=False)["logits"]
print(float(np.max(np.abs(np.array(before) - np.array(after)))))  # 0.0
```

`get_config()` stores all eleven constructor arguments, including `field_size` resolved to its
concrete value, never left as `None`.

---

## 13. Testing & Validation

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/test_models/test_wave_field/ tests/test_train/test_wave_field/ -q
```

The model suite covers construction validation, forward shapes, the **causality ratio sweep**
(the pin for the §3 table: clean ratios must stay below a bound and leaky ratios *above* one, so a
silent change in either direction fails), weight tying, `get_config` / `.keras` round trips, every
named variant, gradient flow, CLM loss finiteness, the `pretrained=True` contract, and the public
API surface. Report *passed with collected*: a suite that collects nothing also reports green.

---

## 14. Troubleshooting & FAQs

- **"Is this model causal?"** Only as measured, for the exact `(field_size, max_seq_len)` you
  built. Run the §3 probe. Nothing in this package enforces causality.
- **`NotImplementedError: Pretrained WaveFieldLLM weights are not distributed...`** As designed.
  Pass a local checkpoint path or `pretrained=False`.
- **`ValueError: field_size must be > 1`** and the divisibility / dropout-range errors are raised
  at construction.
- **`'NoneType' object has no attribute 'assign'`** is the Keras 3 lazy-build trap the explicit
  `build()` calls prevent. If you refactor the block or the architecture builder, keep them.
- **Why not `TransformerLayer`, like GPT-2 uses?** The mask contract: `WaveFieldAttention` takes
  `(B, N)`, `TransformerLayer` / `TextDecoder` build `(B, N, N)`.

---

## 15. Technical Details

### Why the FFN and the norms are hand-rolled

The repo's layer-reuse policy prefers a factory. Three candidate adoptions here were measured and
**refused**:

1. **FFN to `create_ffn_layer('mlp')`.** `MLPBlock.call` applies dropout *between* `fc1` +
   activation and `fc2`; this block applies it *after* the output projection. Under an identical
   Bernoulli mask at `dropout_rate=0.1` the two orders differ by max |delta| 0.3953 (~38% of
   signal RMS): a behaviour change at any non-zero dropout. `MLPBlock` also names its sub-layers
   `fc1`/`fc2`, so a `by_name` weight load would not bind.
2. **The four `LayerNormalization` sites to `create_normalization_layer`.** The factory's
   `'layer_norm'` key resolves to `keras.layers.LayerNormalization` itself, so routing through it
   constructs the identical class, still needs the explicit `epsilon`, and fixes no defect.
3. **The embedding tables and the untied `lm_head`.** No factory covers a bare word-embedding
   table or a bare vocabulary projection, and `TextDecoder` hand-rolls the same
   `keras.layers.Embedding`. What *was* adopted: the positional path (§4.2), which `TextDecoder`
   already routes through the same factory key.

### Weight-path layout

Block: `attn_norm`, `attention`, `ffn_norm`, `ffn_dense_1`, `ffn_dense_2`, `ffn_dropout`. Model:
`token_embeddings`, `position_embeddings`, `embed_norm`, `embed_dropout`, `block_{i}`,
`final_norm`, and `lm_head` when untied.

### Relationship to `WaveFieldAttention`

The layer is the subject; this package is its language-model harness. The layer's own docstring
(`layers/attention/wave_field_attention.py`, "Causality") is the authority on what the mixer
promises, and is more conservative than any model-level claim: *"Do NOT rely on this layer for
autoregressive decoding."* The §3 table confirms that warning, it does not rebut it.

---

## 16. Citation

The architecture shell follows GPT-2:

```bibtex
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  journal={OpenAI Technical Report},
  year={2019}
}
```

The wave-field mixer has no external paper. It is an internal architecture defined by
`src/dl_techniques/layers/attention/wave_field_attention.py`; cite that module directly.

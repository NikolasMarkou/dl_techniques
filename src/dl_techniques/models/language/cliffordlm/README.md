# CliffordNetLM: a causal language model built on Clifford geometric products

[![Keras 3](https://img.shields.io/badge/Keras-3.x-red.svg)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://www.tensorflow.org/)

`CliffordNetLM` is an autoregressive language model with no attention matrix and no feed-forward
network. Each layer is a `CausalCliffordNetBlock`
([`layers/geometric/clifford_block.py`](../../../layers/geometric/clifford_block.py)): a
pointwise detail stream and a locally aggregated context stream interact through the Clifford
geometric product (inner plus wedge part), and that one operation does both the token mixing and
the channel mixing. Cost is linear in sequence length and there is no KV cache.

> **No pretrained weights are distributed by this library, and there is no `pretrained`
> argument.** Any `pretrained` keyword, for example
> `CliffordNetLM.from_variant("nano", vocab_size=512, pretrained="weights.keras")`, raises
> `ValueError` ("Unrecognized keyword arguments passed to CliffordNetLM"), because the keyword
> falls through to `keras.Model`. It does not raise `NotImplementedError`, and it never returns a
> model; a boolean and a path fail the same way. A trained model comes from `keras.models.load_model` on a `.keras` file you produced
> (section 7).

---

## 1. Overview

The model is the CliffordNet vision backbone read along a sequence axis instead of a spatial one.

- **Causality comes from padding, not from a mask.** The two depthwise convolutions inside every
  block are `(1, K)` and `padding="valid"`, preceded by an explicit left-only zero pad. Position
  `i` therefore cannot read position `i + 1`. There is no causal mask anywhere in the model.
- **The global-context branch is a cumulative mean.** When it is enabled, the
  global-average-pool of the vision block (which would see the whole sequence) is replaced by a
  running mean over positions `0..i`.
- **Input is token ids** of shape `(B, L)`; **output is a dict**
  `{"logits": (B, L, vocab_size)}`, not a bare tensor.
- **Embeddings are tied by default** (`tie_word_embeddings=True`): the head multiplies by the
  transposed token-embedding matrix and adds a separate learned `output_bias` of shape
  `(vocab_size,)`.
- **No `generate()`.** Sampling is outside the class (section 6).
- **No sequence-length guard.** `max_seq_length` sizes the position table and nothing refuses a
  longer input (section 8).

---

## 2. Architecture

```text
input_ids (B, L)
     |
     v
token Embedding(vocab, D)  +  position Embedding(max_seq_length, D)
     |
     v
LayerNormalization (embed_norm)  ->  Dropout (embed_dropout)
     |  (B, L, D)
     v
+---------------------------------------------------------------+
| for i in 0 .. depth-1:                                        |
|     x = x + StochasticDepth_i( CausalCliffordNetBlock_i(x) )  |
|                                                               |
|   CausalCliffordNetBlock (transform-only, shape (B, L, D)):   |
|     x_norm = zero_centered_rms_norm(x)        (per position)  |
|     z_det  = Dense(x_norm)                    (pointwise)     |
|     z_ctx  = act(norm(dwconv(dwconv(x_norm))))  left-padded   |
|              (1, K) depthwise convs, K = 3 -> 5 positions     |
|              ending at the current one                        |
|     z_ctx  = z_ctx - z_det           (ctx_mode "diff")        |
|     G      = sparse rolling geometric product(z_det, z_ctx)   |
|              over `shifts`, inner + wedge ("full")            |
|     G     += global branch (cumulative mean), if enabled      |
|     out    = gamma * (SiLU(x_norm) + sigmoid(gate) * G)       |
+---------------------------------------------------------------+
     |  (B, L, D)
     v
LayerNormalization (head_norm)  ->  Dropout (head_dropout, only if rate > 0)
     |
     v
tied:    logits = h @ token_embedding^T + output_bias
untied:  logits = Dense(vocab, name="output_proj")(h)
     |
     v
{"logits": (B, L, vocab_size)}
```

Points that are easy to get wrong:

- **The residual and the drop-path are outside the block.** The block returns only the
  gamma-scaled update; the model adds it to the stream and applies `StochasticDepth` with a
  linear schedule up to `stochastic_depth_rate`. `gamma` (LayerScale) starts at
  `layer_scale_init` (default `1e-5`), so every block starts close to the identity.
- **The context kernel size is not a model argument.** `context_kernel_size` is the block's
  default of 3, so one block reads the 5 positions ending at the current one.
- **The block norm must be per-position.** The default is `zero_centered_rms_norm`.
  `normalization_type` with a sequence-reducing norm (for example `"batch_norm"`) raises
  `ValueError` at construction, because it would leak future positions past the left padding.
- **The global branch is fixed.** When enabled it uses `shifts=[1, 2]`, `cli_mode="full"` and the
  differential context, whatever the block-level `shifts` and `cli_mode` are.
- **`global_context_period`** turns the global branch on in every `n`-th block (1-indexed:
  blocks `n, 2n, ...`), independent of `use_global_context`. It accepts `None`, `-1` (both mean
  disabled, and `-1` is normalised to `None`) or an `int >= 1`; anything else raises
  `ValueError`.

Tied logits are computed in float32 by `utils/tied_embeddings.py::tied_embedding_logits`, also
under a mixed-precision policy:

```python
import keras
import numpy as np
from dl_techniques.models.language.cliffordlm import CliffordNetLM

keras.mixed_precision.set_global_policy("mixed_float16")
try:
    model = CliffordNetLM(vocab_size=64, max_seq_length=8, channels=16, depth=2)
    out = model(np.zeros((1, 8), dtype="int32"), training=False)
    print(keras.backend.standardize_dtype(out["logits"].dtype))  # float32
finally:
    keras.mixed_precision.set_global_policy("float32")
```

---

## 3. Variants

`CliffordNetLM.MODEL_VARIANTS`, with parameter counts measured by building each variant with
`vocab_size=50261`, `max_seq_length=512` and tied embeddings (the counts include the position
table, the embeddings and `output_bias`):

| Variant | `channels` | `depth` | `shifts` | `stochastic_depth_rate` | Parameters |
|:---:|:---:|:---:|:---|:---:|---:|
| `nano` | 128 | 12 | `[1, 2]` | 0.05 | 7,945,941 |
| `mini` | 192 | 12 | `[1, 2, 4]` | 0.10 | 13,810,709 |
| `base` | 384 | 18 | `[1, 2, 4, 8, 16]` | 0.15 | 54,143,189 |
| `large` | 512 | 20 | `[1, 2, 4, 8, 16]` | 0.20 | 94,338,645 |
| `xl` | 768 | 28 | `[1, 2, 4, 8, 16]` | 0.25 | 254,022,485 |

Every variant uses `cli_mode="full"`, `ctx_mode="diff"`, `use_global_context=False`,
`layer_scale_init=1e-5` and a `TruncatedNormal(stddev=0.02)` kernel initializer. Re-derive the
table with:

```python
from dl_techniques.models.language.cliffordlm import CliffordNetLM

for name, cfg in CliffordNetLM.MODEL_VARIANTS.items():
    print(name, {k: v for k, v in cfg.items() if k != "kernel_initializer"})
```

The variant table defines no quality, speed or memory figure; the parameter counts above are
the only measured numbers it carries.

---

## 4. Quick start

```python
import numpy as np
from dl_techniques.models.language.cliffordlm import CliffordNetLM

model = CliffordNetLM(vocab_size=256, max_seq_length=32, channels=32, depth=2)
ids = np.random.randint(0, 256, size=(2, 16)).astype("int32")
outputs = model(ids, training=False)

print(sorted(outputs))                # ['logits']
print(outputs["logits"].shape)        # (2, 16, 256)
print(model.count_params())
```

A named variant, at its real size (there is no `create_*` factory; `from_variant` is the
constructor for variants, and keyword arguments override the variant's defaults):

```python
from dl_techniques.models.language.cliffordlm import CliffordNetLM

model = CliffordNetLM.from_variant(
    "nano", vocab_size=512, max_seq_length=64, depth=2, use_global_context=True,
)
print(model.channels, model.depth, model.use_global_context)  # 128 2 True
```

The weights contract, stated by running it:

```python
from dl_techniques.models.language.cliffordlm import CliffordNetLM

try:
    CliffordNetLM.from_variant("nano", vocab_size=512, pretrained="weights.keras")
except ValueError as e:
    print(f"ValueError: {e}")
```

---

## 5. Constructor reference

`CliffordNetLM(vocab_size, max_seq_length=512, channels=128, depth=12, ...)`. Exported from
`dl_techniques.models.language.cliffordlm` (`__all__ = ["CliffordNetLM"]`).

| Argument | Default | Notes |
|:---|:---|:---|
| `vocab_size` | required | Includes special tokens. The repo's tiktoken `gpt2` setup uses 50261 (50257 + CLS, SEP, PAD, MASK). |
| `max_seq_length` | `512` | Rows of the learned position table. Not enforced (section 8). |
| `channels` | `128` | Feature width `D`, constant through the stack. |
| `depth` | `12` | Number of `CausalCliffordNetBlock` layers. |
| `shifts` | `None` -> `[1, 2]` | Channel-roll offsets of the sparse geometric product. |
| `cli_mode` | `"full"` | `"inner"`, `"wedge"` or `"full"`. |
| `ctx_mode` | `"diff"` | `"diff"` or `"abs"`. Affects the inner term only. |
| `use_global_context` | `False` | Cumulative-mean global branch in every block. |
| `global_context_period` | `None` | `None` / `-1` disabled; `int >= 1` turns the branch on in every `n`-th block. |
| `layer_scale_init` | `1e-5` | Initial LayerScale `gamma`. |
| `stochastic_depth_rate` | `0.1` | Maximum drop-path rate, linear schedule over depth. |
| `dropout_rate` | `0.0` | Embedding dropout and, if above 0, a pre-output dropout. |
| `tie_word_embeddings` | `True` | Untied builds `Dense(vocab, name="output_proj")`. |
| `use_bias` | `True` | Dense biases, and the tied `output_bias`. |
| `kernel_initializer`, `bias_initializer` | `"glorot_uniform"`, `"zeros"` | Variants override the kernel one with `TruncatedNormal(0.02)`. |
| `kernel_regularizer`, `bias_regularizer` | `None` | |
| `normalization_type` | `"zero_centered_rms_norm"` | Per-position norms only. |
| `normalization_kwargs` | `None` | Forwarded to the block norm. |

Methods: `from_variant(variant, vocab_size, max_seq_length=512, **overrides)`, `call`,
`compute_output_shape`, `get_config`, `from_config`, `build`. `build` materialises every
sub-layer by tracing `call` (`utils.model_build.materialize_sublayers`), so a model restored
from a `.keras` file has somewhere to load its weights.

The model takes only `input_ids`. It has no `attention_mask` input and no task head
(`include_top`); a padded batch is handled by the loss, not by the model.

`global_context_period` and the norm check, by running them:

```python
from dl_techniques.models.language.cliffordlm import CliffordNetLM

m = CliffordNetLM(vocab_size=64, max_seq_length=8, channels=16, depth=4,
                  global_context_period=-1)
print(m.global_context_period)                   # None
m = CliffordNetLM(vocab_size=64, max_seq_length=8, channels=16, depth=4,
                  global_context_period=2)
print([b.use_global_context for b in m.clifford_blocks])  # [False, True, False, True]

for bad in ({"global_context_period": 0}, {"normalization_type": "batch_norm"}):
    try:
        CliffordNetLM(vocab_size=64, max_seq_length=8, channels=16, depth=2, **bad)
    except ValueError as e:
        print("ValueError:", str(e)[:70])
```

---

## 6. Training and sampling

### Training

The pipeline was `src/train/cliffordnet/train_cliffordnet_nlp.py` (removed from the tree; see its
git history). It trained on English Wikipedia or a TFDS text dataset with the tiktoken GPT-2
tokenizer, compiles `AdamW` with a warmup plus cosine schedule, and keys the loss on the output
dict: `loss={"logits": MaskedCausalLMLoss()}`.

```bash
MPLBACKEND=Agg .venv/bin/python -m train.cliffordnet.train_cliffordnet_nlp \
    --variant nano --epochs 3   # at a commit before its removal
```

The same keyed compile on a tiny model, one step on
random ids, showing the input/target shift that is the caller's job:

```python
import keras
import numpy as np
from dl_techniques.losses import MaskedCausalLMLoss
from dl_techniques.models.language.cliffordlm import CliffordNetLM

model = CliffordNetLM(vocab_size=256, max_seq_length=32, channels=32, depth=2)
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=3e-4, clipnorm=1.0),
    loss={"logits": MaskedCausalLMLoss()},
)
ids = np.random.randint(1, 256, size=(8, 33)).astype("int32")
history = model.fit(ids[:, :-1], {"logits": ids[:, 1:]}, batch_size=4, epochs=1, verbose=0)
print(sorted(history.history))  # ['loss']
```

### Sampling

The class has no `generate()`. Two routes exist:

- `src/train/cliffordnet/infer_cliffordnet_nlp.py` (removed from the tree; see its git history)
  loaded a `.keras` checkpoint and runs nucleus sampling, MCMC power sampling or max-swap
  (`--method standard|power|max_swap`, `--compare` for all three).
- [`models/common/power_sampling`](../../common/power_sampling/) holds the sampler itself
  (`PowerSampler`, `make_logits_fn`), model-agnostic; its README covers the algorithm and the
  papers it follows.

The smallest possible decode, a greedy loop over the `logits` key (a demonstration of the call
contract, not a sampler to rely on):

```python
import numpy as np
from dl_techniques.models.language.cliffordlm import CliffordNetLM

model = CliffordNetLM(vocab_size=64, max_seq_length=16, channels=16, depth=2)
ids = [1, 2, 3]
for _ in range(6):                                   # stay within max_seq_length
    logits = model(np.array([ids], dtype="int32"), training=False)["logits"]
    ids.append(int(np.argmax(np.asarray(logits)[0, -1])))
print(len(ids))  # 9
```

Because the model is causal, a decoder that appends one token and re-runs the whole prefix
reads the same logits at every earlier position that it read before (section 7); there is no
cache, so each step recomputes the prefix.

---

## 7. Serialization and causality

### Round trip

```python
import os
import tempfile
import keras
import numpy as np
from dl_techniques.models.language.cliffordlm import CliffordNetLM

model = CliffordNetLM(vocab_size=64, max_seq_length=16, channels=16, depth=2)
ids = np.random.randint(0, 64, size=(1, 8)).astype("int32")
before = np.asarray(model(ids, training=False)["logits"])

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "cliffordlm.keras")
    model.save(path)
    restored = keras.models.load_model(path)          # no custom_objects needed
after = np.asarray(restored(ids, training=False)["logits"])
print(float(np.max(np.abs(before - after))))          # 0.0
print(keras.saving.get_registered_name(CliffordNetLM))
```

The registry key is `dl_techniques.models.cliffordlm.model` (the `models/` family directory is
stripped, the repo convention), so `get_registered_name` prints
`dl_techniques.models.cliffordlm.model>CliffordNetLM`. The default `Custom>CliffordNetLM` alias
also resolves. `get_config()` stores every constructor argument, with `global_context_period`
already normalised (`-1` is stored as `None`).

Layer and weight names: `token_embedding`, `position_embedding`, `embed_norm`, `embed_dropout`,
`clifford_block_{i}`, `clifford_drop_path_{i}`, `head_norm`, `head_dropout` (only when
`dropout_rate > 0`), `output_proj` when untied, and a model-level `output_bias` weight when tied
and `use_bias=True`.

### Causality, measured

Perturb the token at position `t` and compare the logits. The probe below changes every position
`t >= 1` in turn and takes the largest absolute change over all positions before `t`:

```python
import keras
import numpy as np
from dl_techniques.models.language.cliffordlm import CliffordNetLM

L, V = 16, 64


def worst_earlier_change(model):
    ids = np.random.default_rng(0).integers(0, V, size=(2, L)).astype("int32")
    base = np.asarray(model(ids, training=False)["logits"])
    worst = 0.0
    for t in range(1, L):
        ids2 = ids.copy()
        ids2[:, t] = (ids2[:, t] + 7) % V
        out = np.asarray(model(ids2, training=False)["logits"])
        worst = max(worst, float(np.abs(base[:, :t] - out[:, :t]).max()))
    return worst


configs = {
    "default": {},
    "use_global_context": dict(use_global_context=True),
    "global_context_period=2": dict(global_context_period=2),
    "cli_mode=wedge": dict(cli_mode="wedge"),
    "untied": dict(tie_word_embeddings=False),
}
for name, kw in configs.items():
    keras.utils.set_random_seed(0)
    model = CliffordNetLM(vocab_size=V, max_seq_length=L, channels=24, depth=3,
                          layer_scale_init=1.0, **kw)
    print(f"{name:26s} worst change at earlier positions: {worst_earlier_change(model):.1e}")
```

Measured on GPU and on CPU (`channels=24`, `depth=3`, `L=16`, float32, random init): the worst
change at an earlier position was exactly `0.0` for the five configurations above and also for
`cli_mode="inner"`, `ctx_mode="abs"` and `shifts=[1, 2, 4]` with the global branch, each at both
`layer_scale_init=1e-5` and `1.0`, and for the `nano` variant at `layer_scale_init=1.0` (with and
without the global branch, `max_seq_length=16`). The same perturbation always moved logits at positions
`t` and later (smallest change over those positions, `0.36` to `1.7`), so the probe is not blind.
As a control, the same probe on a non-causal `CliffordNetBlock(input_mode="sequence")` moved
earlier positions by `0.71`, and `CausalCliffordNetBlock` by `0.0`. These are measurements of
the configurations tried, not a proof for every configuration; re-run the probe for yours.

---

## 8. Known gaps

- **No length guard.** Nothing checks `L <= max_seq_length`. Measured with
  `max_seq_length=16` and a 24-token input:
  - on a **GPU** the call succeeds and returns finite logits of shape `(1, 24, 64)`. The
    position table is read out of range, and the gather returns all-zero rows for positions
    `>= 16` (measured: the absolute maximum of the position embedding at those rows is `0.0`,
    against `0.05` at in-range rows). Positions `0..15` come out identical to the 16-token call.
    The call does not warn.
  - on **CPU** the call raises `InvalidArgumentError` from `Embedding`.

  Keep `L <= max_seq_length` yourself. Other models in `language/` refuse an over-long input
  with `SequenceLengthError` (`utils/sequence_length_guard.py`); this one does not.
- **No padding mask.** `call` takes `input_ids` only. The causal convolutions mean right
  padding cannot change the logits at earlier, real positions, but the pad positions still
  produce logits, and the model neither pools nor masks them. Mask them in the loss
  (`MaskedCausalLMLoss`) or in the decoder.
- **No `generate()`, no cache.** A decode step recomputes the whole prefix.
- **No task head and no `pretrained` argument.** `include_top` and `task_config` do not exist
  here, unlike the 15 language classes that own a task head (`language/AGENTS.md`). `pretrained`
  is not accepted (section 4).
- **No `create_*` factory.** `from_variant` is the constructor for named variants. The package
  holds the model and nothing else.
- **No quality measurement is recorded.** No perplexity or benchmark result for this model
  exists in the repository, so none is stated here. The numbers in this file are parameter
  counts and the causality probe.
- **The context kernel size is fixed** at the block default of 3 and is not a model argument.

---

## 9. Tests

```bash
CUDA_VISIBLE_DEVICES=1 MPLBACKEND=Agg .venv/bin/python -m pytest \
    tests/test_models/test_cliffordlm/ -q
```

[`tests/test_models/test_cliffordlm/`](../../../../../tests/test_models/test_cliffordlm/) holds
`test_cliffordnet_lm.py` (construction, forward shape, causality with and without the global
branch, `get_config`, variants, gradient flow, norm passthrough, `global_context_period`, float32
tied logits under `mixed_float16`, the registry-key `.keras` round trip) and
`test_lazy_build_contract.py` (every sub-layer is built after `build`, so a reload finds its
weights). The block itself is covered by `tests/test_layers/test_geometric/`. Report passed
counts together with collected counts: a suite that collects nothing also reports green.

---

## 10. References

- Ji, Z., 2026. *CliffordNet: All You Need is Geometric Algebra.*
  ([arXiv:2601.06793](https://arxiv.org/abs/2601.06793)): the block, the geometric product and
  the gated geometric residual.
- Brandstetter, J. et al., 2023. *Clifford Neural Layers for PDE Modeling.*
  ([arXiv:2209.04934](https://arxiv.org/abs/2209.04934))
- Press, O. and Wolf, L., 2017. *Using the Output Embedding to Improve Language Models.*
  ([arXiv:1608.05859](https://arxiv.org/abs/1608.05859)): weight tying of the head.
- Touvron, H. et al., 2021. *Going Deeper with Image Transformers (CaiT).*
  ([arXiv:2103.17239](https://arxiv.org/abs/2103.17239)): LayerScale.
- Huang, G. et al., 2016. *Deep Networks with Stochastic Depth.*
  ([arXiv:1603.09382](https://arxiv.org/abs/1603.09382)): the drop-path schedule.

Block internals, the paper's deviations that the block follows on purpose, and the behaviour of
padded batches are documented in the module docstring of
[`clifford_block.py`](../../../layers/geometric/clifford_block.py).

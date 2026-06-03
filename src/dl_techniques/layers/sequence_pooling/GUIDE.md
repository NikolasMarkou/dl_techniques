# Sequence Pooling Implementation Guide

This guide establishes the architectural standards, coding conventions, and best
practices for implementing new sequence-pooling layers within the
`dl_techniques.layers.sequence_pooling` package.

**Goal:** Ensure every pooling mechanism — positional, statistical, learnable, or
a composite of them — shares a consistent `(batch, seq_len, dim) -> (batch, dim)`
contract, is reachable through the factory, and serializes round-trip safely.

---

## 1. Core Design Philosophy

### The "Factory First" Approach

The package exposes all pooling layers through a single registry-driven factory
(`factory.py`). Dispatch is **pure registry lookup — no `if/elif` ladder**:
`create_sequence_pooling_layer` validates the config, looks the class up in
`SEQUENCE_POOLING_REGISTRY`, merges defaults with user kwargs, filters to valid
parameter names, and instantiates. Adding a type is a registry edit, not a code
branch.

```python
# create_sequence_pooling_layer (factory.py) — the dispatch is data, not control flow
info = SEQUENCE_POOLING_REGISTRY[pooling_type]
pool_class = info['class']
params = info['optional_params'].copy()
params.update(kwargs)
return pool_class(**final_params)
```

### Keras 3 Compliance

- **Separation of concerns:** create sub-layers in `__init__`, build them
  explicitly in `build()`, execute in `call()`, and round-trip every constructor
  argument through `get_config()`. `AttentionPooling`, `WeightedPooling`, and
  `SequencePooling` all follow this `__init__`/`build`/`call`/`get_config` split,
  and every leaf layer also implements `compute_output_shape`.
- **Backend agnostic:** use `keras.ops` for all tensor manipulation (the existing
  layers use `ops.softmax`, `ops.einsum`, `ops.take_along_axis`, etc.). Never call
  `tf.*` or `torch.*` directly.
- **Bare serialization decorator (CRITICAL):** every layer is decorated with
  **bare** `@keras.saving.register_keras_serializable()`. Keras derives the
  registered key as `Custom>ClassName` from the default `package="Custom"`, which
  is **independent of `__module__`**. That is why this package could be migrated
  out of the old flat `layers/sequence_pooling.py` without breaking a single saved
  `.keras` file.

  **Do NOT add `package=` or `name=` to the decorator.** Doing so changes the key
  (e.g. `Custom>SequencePooling` → `dl_techniques>SequencePooling`) and breaks
  every existing save. Keep all three decorators bare, forever.

---

## 2. API Standards & Naming Conventions

Pooling layers consume `(batch, seq_len, dim)` and emit `(batch, dim)` (the
`sequence` facade may emit a multiple of `dim` for combined or concatenated
strategies, or `(batch, seq_len, dim)` / `(batch, seq_len * dim)` for the `none` /
`flatten` strategies).

Use the canonical parameter names already established across the package:

| Parameter concept | Standard name | Notes |
| :--- | :--- | :--- |
| Attention hidden width | `hidden_dim` | `AttentionPooling`; the facade exposes it as `attention_hidden_dim`. |
| Head count | `num_heads` | `AttentionPooling`; facade exposes `attention_num_heads`. |
| Dropout rate | `dropout_rate` | `0.0` disables dropout (no `Dropout` sub-layer is created). |
| Softmax temperature | `temperature` | Positive float; scores are divided by it before softmax. |
| Bias toggle | `use_bias` | Standard Keras meaning. |
| Kernel initializer | `kernel_initializer` | Passed through `initializers.get(...)`. |
| Kernel regularizer | `kernel_regularizer` | `None` means no regularizer. |
| Max positions | `max_seq_len` | `WeightedPooling`; facade exposes `weighted_max_seq_len`. |

**Masking convention:** `call(inputs, mask=None, training=None)`. `mask` is a
boolean/0-1 tensor of shape `(batch, seq_len)`. Statistical strategies apply the
mask numerically (additive `-1e9` for `max`, multiplicative for `mean`/`sum`),
positional strategies (`cls`, `first`, `last`, `middle`, `none`, `flatten`) are
mask-aware only where meaningful (`last` selects the last unmasked position).

---

## 3. Shared Components & Infrastructure

The `sequence` facade (`SequencePooling`) does **not** re-implement learnable
pooling. It **composes** the two leaf poolers: for the `attention` /
`multi_head_attention` strategies it instantiates `AttentionPooling`, and for the
`weighted` strategy it instantiates `WeightedPooling`, storing them in a
`self.learnable_components` dict created in `__init__` and built explicitly in
`build()`.

```python
# SequencePooling.__init__ — reuse leaf layers, do not duplicate their math
from .attention_pooling import AttentionPooling
from .weighted_pooling import WeightedPooling

if strat in ['attention', 'multi_head_attention']:
    num_heads = self.attention_num_heads if strat == 'multi_head_attention' else 1
    self.learnable_components[strat] = AttentionPooling(
        hidden_dim=self.attention_hidden_dim,
        num_heads=num_heads,
        ...
    )
elif strat == 'weighted':
    self.learnable_components[strat] = WeightedPooling(
        max_seq_len=self.weighted_max_seq_len, ...
    )
```

When you need learnable content-aware or position-weighted pooling inside another
layer, reuse `AttentionPooling` / `WeightedPooling` the same way rather than
re-deriving the softmax-weighted sum. The import graph is intentionally acyclic:
leaves import nothing intra-package; the facade and factory import the leaves.

---

## 4. Implementation Patterns

### Pattern A: The Wrapper / Facade (`SequencePooling`)

A facade owns dispatch and composition, not novel math. `SequencePooling`:

1. Normalises `strategy` to a list in `__init__` and creates any required leaf
   poolers into `self.learnable_components`.
2. In `build()`, explicitly builds each learnable component and, only when there
   are multiple strategies **and** `aggregation_method == 'weighted_sum'`, creates
   the `aggregation_weights` variable.
3. In `call()`, runs each strategy via `_apply_single_strategy`, then combines the
   outputs per `aggregation_method` (`concat` / `add` / `multiply` / `weighted_sum`).
4. `compute_output_shape` mirrors the dispatch so shape inference stays exact
   (e.g. `mean_max` → `2 * dim`, `mean_max_min` → `3 * dim`, `flatten` →
   `seq_len * dim`).

Add a new **strategy** to the facade by extending the `PoolingStrategy` literal,
adding a branch in `_apply_single_strategy`, and updating `compute_output_shape`
if the strategy changes the output width.

### Pattern B: The Standalone Leaf Layer (`AttentionPooling` / `WeightedPooling`)

A leaf layer implements one pooling mechanism from scratch with `keras.ops`:

```python
import keras
from keras import ops, layers, initializers, regularizers


@keras.saving.register_keras_serializable()   # BARE — never add package=/name=
class MyPooling(keras.layers.Layer):
    def __init__(self, hidden_dim: int = 256, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        # create sub-layers here (unbuilt)
        self.transform = layers.Dense(hidden_dim, activation='tanh', name='transform')

    def build(self, input_shape):
        super().build(input_shape)
        self.transform.build(input_shape)                 # build sub-layers explicitly
        self.context = self.add_weight(name='context', shape=(self.hidden_dim,),
                                       initializer='glorot_uniform', trainable=True)

    def call(self, inputs, mask=None, training=None):
        h = self.transform(inputs)
        scores = ops.einsum('bsh,h->bs', h, self.context) / self.temperature
        if mask is not None:
            scores = scores + (1.0 - ops.cast(mask, scores.dtype)) * (-1e9)
        weights = ops.softmax(scores, axis=1)
        return ops.einsum('bs,bsd->bd', weights, inputs)  # (batch, dim)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({'hidden_dim': self.hidden_dim, 'temperature': self.temperature})
        return config
```

---

## 5. Factory Integration Checklist

When adding a new pooling **type** to the package, update `factory.py` and
`__init__.py`:

1. **Define the class.** New file (e.g. `my_pooling.py`) with the bare
   `@keras.saving.register_keras_serializable()` decorator, the
   `__init__`/`build`/`call`/`compute_output_shape`/`get_config` methods, and
   Google-style docstrings.
2. **Add the type literal.** Extend
   `SequencePoolingType = Literal['attention', 'weighted', 'sequence', ...]`.
3. **Add the registry entry** to `SEQUENCE_POOLING_REGISTRY` with:
   - `class`: the class object,
   - `required_params`: list of mandatory `__init__` args (often `[]`),
   - `optional_params`: dict of optional args **with defaults that exactly match
     the `__init__` signature**,
   - `description`: a technical one-paragraph summary,
   - `use_case`: when to reach for it.
4. **Re-export from `__init__.py`** in the appropriate group (factory surface /
   layer classes / type aliases) and add the **string** name to `__all__`.
5. **Add a test** in `tests/test_layers/test_sequence_pooling.py` covering
   construction, forward pass, and a `get_config` / save-load round trip.

If the new type also needs to be reachable as a `SequencePooling` strategy, extend
the `PoolingStrategy` literal and `_apply_single_strategy` (Pattern A) as well.

---

## 6. Common Pitfalls to Avoid

1. **Adding `package=` / `name=` to the decorator.** This changes the registered
   key from `Custom>ClassName` and breaks every existing `.keras` save. Keep all
   decorators **bare**.
2. **Object-based `__all__`.** Use **string** names in `__all__`
   (`"SequencePooling"`, not `SequencePooling`). Object-based `__all__` breaks
   `from ... import *` and confuses linters (a known bug in `ffn/__init__.py`).
3. **Forgetting to re-export a type alias.** `PoolingStrategy` (and
   `AggregationMethod`) must stay exported from `__init__.py` — both transformer
   encoders do `from ..sequence_pooling import SequencePooling, PoolingStrategy`,
   and dropping the re-export raises `ImportError` at their import time.
4. **Missing `compute_output_shape`.** Every layer here implements it; for the
   facade it must mirror the strategy/aggregation dispatch so shape inference is
   exact (combined strategies multiply the width).
5. **Mutating the registry returned by `get_sequence_pooling_info()`.** It returns
   shallow copies of each entry; do not rely on mutating it to change global
   behaviour, and do not mutate `SEQUENCE_POOLING_REGISTRY` directly at runtime.
6. **Registry defaults drifting from `__init__`.** A misspelled key in
   `optional_params` is silently filtered out by `valid_param_names`, dropping the
   user's value to the class default. Cross-read the registry against the live
   `__init__` signature whenever either changes.

---

## 7. Migration of Legacy Layers

This package was migrated from the single flat file
`src/dl_techniques/layers/sequence_pooling.py` (three classes plus two type
aliases) into a one-class-per-file package: `attention_pooling.py`,
`weighted_pooling.py`, `sequence_pooling.py` (the facade, keeping the original
filename), plus `factory.py` and `__init__.py`.

Key invariant preserved across the migration: **serialization keys did not
change.** Because every decorator is bare, Keras derives `Custom>ClassName`
independently of `__module__`, so moving the classes into new modules left every
existing `.keras` save loadable. The two transformer encoders and the existing
test continued to import the symbols unchanged via the `__init__.py` re-export.

When migrating any further legacy pooling code into this package:
1. Keep the decorator **bare** — never let the move change the registered key.
2. Convert RST/Sphinx docstrings to Google-style (`Args:`/`Returns:`/`Raises:`).
3. Copy the forward-pass math **verbatim** — migration is structural, not a
   numerics rewrite.
4. Re-export the moved symbols from `__init__.py` with **string** `__all__`
   entries so existing callers keep resolving.
